"""Script for running inference and sampling.

Sample command:
> python scripts/run_inference.py

"""

import os
import sys
import time
import tree
import logging
import traceback
import shutil
from functools import partial
from datetime import datetime
from contextlib import nullcontext
from typing import Dict,Optional
import dataclasses

import numpy as np
import hydra
import torch
import pandas as pd
import GPUtil
import multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
import esm

from experiments import utils as eu
from analysis import metrics
from analysis import plotting
from data import data_transforms
from data import interpolant
from data import utils as du
from data import protein
from data.se3_diffuser import SE3Diffuser
from model.fold_module import FoldModule

from ProteinMPNN import protein_mpnn_pyrosetta

logging.basicConfig(level=logging.WARNING,
    # define this things every time set basic config
    # define this things every time set basic config
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ])

def create_pad_feats(pad_amt):
    return {        
        'res_mask': torch.ones(pad_amt),
        'fixed_mask': torch.zeros(pad_amt),
        'rigids_impute': torch.zeros((pad_amt, 4, 4)),
        'torsion_impute': torch.zeros((pad_amt, 7, 2)),
    }

def mpnn_design(args_to_keep, mpnn_conf, mpnn_model, full_atom_dir, prefix='',mode='monomer'):
    if full_atom_dir is not None:
        os.makedirs(full_atom_dir, exist_ok=True)
    init_feats = args_to_keep['init_feats']
    sample_output = args_to_keep['sample_output']
    diffusion_file_path = args_to_keep['diff_path']
    diff_df = args_to_keep['dataframe']
    sequences = []
    full_atom_prots = []
    struct2seq_results = []
    step_time = time.time()
    mpnn_results = protein_mpnn_pyrosetta.mpnn_design(
        config= mpnn_conf,
        protein_path=args_to_keep['diff_path'],
        model= mpnn_model,
        mode = mode
    )
    for i, mpnn_result in enumerate (mpnn_results):
        struct2seq_result = {}
        if full_atom_dir is not None and 'pose' in mpnn_result:
            full_atom_prots.append(protein.from_pdb_string(mpnn_results[i]["pose"]))
            full_atom_path = os.path.join(full_atom_dir, prefix+f'fa_{i}.pdb')
            open(full_atom_path,'w').write(protein.to_pdb(full_atom_prots[i]))
            struct2seq_result['full_atom_path'] = full_atom_path
        elif full_atom_dir is not None and 'pose' not in mpnn_result:
            full_atom_prots.append(protein.Protein(
                aatype=du.seq_to_aatype(mpnn_result['sequence']),
                atom_positions=sample_output["final_atom_positions"],
                atom_mask=sample_output["final_atom_mask"],
                residue_index=init_feats["residue_index"][0],
                chain_index=init_feats["chain_index"][0],
                b_factors=np.tile(1 -init_feats['fixed_mask'][0,..., None], 37) * 100
            ))
            full_atom_path = os.path.join(full_atom_dir, prefix+f'fa_{i}.pdb')
            open(full_atom_path,'w').write(protein.to_pdb(full_atom_prots[i]))
            struct2seq_result['full_atom_path'] = full_atom_path
        struct2seq_result['sequence'] = mpnn_result['sequence']
        struct2seq_results.append(struct2seq_result)
    struct2seq_results = {key: list(map(lambda d: d[key], struct2seq_results)) for key in struct2seq_results[0]}
    struct2seq_df = pd.DataFrame(struct2seq_results)
    struct2seq_df['diff_path'] = diffusion_file_path
    struct2seq_df = struct2seq_df.merge(diff_df,on='diff_path',how='left')
    logging.info(f'generate {len(struct2seq_results["sequence"])} sequences for scaffold {diffusion_file_path} in {time.time()-step_time:.2f}s')
    logging.info(struct2seq_df)
    args_to_keep.update({
        'dataframe' : struct2seq_df,
        'sequence' : struct2seq_results['sequence']
    })
    return args_to_keep

class Sampler:

    def __init__(
            self,
            conf: DictConfig,
            conf_overrides: Dict=None
        ):
        """Initialize sampler.

        Args:
            conf: inference config.
            gpu_id: GPU device ID.
            conf_overrides: Dict of fields to override with new values.
        """
        self._log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Prepare configs.
        self._conf = conf
        self._infer_conf = conf.inference
        self._diff_conf = self._infer_conf.diffusion
        self._sample_conf = self._infer_conf.diffusion.samples
        self._interpolant = interpolant.Interpolant(self._infer_conf.interpolant)
        self._rng = np.random.default_rng(self._infer_conf.seed)

        # Set model hub directory for ESMFold.
        torch.hub.set_dir(self._infer_conf.pt_hub_dir)

        # Set-up accelerator
        if torch.cuda.is_available():
            if self._infer_conf.gpu_id is None:
                available_gpus = ''.join(
                    [str(x) for x in GPUtil.getAvailable(
                        order='memory', limit = 8)])
                self.device = f'cuda:{available_gpus[0]}'
            else:
                self.device = f'cuda:{self._infer_conf.gpu_id}'
        else:
            self.device = 'cpu'
        self._log.info(f'Using device: {self.device}')

        # Set-up directories
        self._weights_path = self._infer_conf.weights_path
        output_dir =self._infer_conf.output_dir
        if self._infer_conf.name is None:
            dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        else:
            dt_string = self._infer_conf.name
        self._output_dir = os.path.abspath(os.path.join(output_dir, dt_string))
        os.makedirs(self._output_dir, exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')

        config_path = os.path.join(self._output_dir, 'inference_conf.yaml')

        # Load models and experiment
        self._fold_module = FoldModule(
            diffuser=SE3Diffuser(self._conf.diffuser),
            device=self.device
        )
        
        self._fold_module.init_strcture_model_from_conf(self._infer_conf.weights_path,self._conf.model)
        
        if self._infer_conf.mpnn.enable:
            self._log.info('Loading ProteinMPNN model')
            self._fold_module.init_mpnn_model_from_conf(self._infer_conf.mpnn)
        elif self._infer_conf.chroma_design.enable:
            self._log.info('Loading ChromaDesign model')
            self._fold_module.init_chroma_model_from_conf(self._infer_conf.chroma_design.weight_path)
        
        if self._infer_conf.esmfold.enable:
            self._log.info('Loading ESMFold model')
            self._fold_module.init_predict_model('esmfold')
        
        self._log.info(f'Saving inference config to {config_path}')

    def run_sampling(self):
        """Sets up inference run.

        All outputs are written to 
            {output_dir}/{date_time}
        where {output_dir} is created at initialization.
        """
        pdb_feats = None
        if self._sample_conf.ref_pdb is not None:
            pdb_feats = dataclasses.asdict(protein.from_pdb_string(open(self._sample_conf.ref_pdb).read()))
        if self._sample_conf.contigs is not None:
            contigs = [str(self._sample_conf.contigs)] * self._sample_conf.sample_nums
        elif self._sample_conf.samples_lengths is None:
            contigs = []
            for length in range(
                self._sample_conf.min_length,
                self._sample_conf.max_length+1,
                self._sample_conf.length_step
            ):
                contigs.extend([f'{str(length)}-{str(length)}'] * self._sample_conf.samples_per_length)
        else:
            contigs = []
            for length in self._sample_conf.samples_lengths:
                contigs.extend([f'{str(length)}-{str(length)}'] * self._sample_conf.samples_per_length)

        if self._infer_conf.mpnn.enable:
            sequence_conf = self._infer_conf.mpnn
        elif self._infer_conf.chroma_design.enable:
            sequence_conf = self._infer_conf.chroma_design
        else:
            sequence_conf = None

        sample_dir = os.path.join(self._output_dir, 'diffusion')
        full_atom_dir = os.path.join(self._output_dir, 'allatom')
        predict_dir = os.path.join(self._output_dir, 'predict')
        traj_dir = os.path.join(self._output_dir, 'trajactory')
        movie_dir = os.path.join(self._output_dir, 'movie')
        tmp_dir = os.path.join(self._output_dir, 'backup')
        process_list = []
        
        with (mp.Pool(processes=mp.cpu_count() if mp.cpu_count() < self._infer_conf.mpnn.cpus else self._infer_conf.mpnn.cpus) if not self._infer_conf.single_process else nullcontext() ) as pool:
            os.makedirs(tmp_dir,exist_ok=True)
            if self._infer_conf.diffusion.option.save_trajactory or self._infer_conf.diffusion.option.plot.switch_on:
                os.makedirs(traj_dir,exist_ok=True)
            if self._infer_conf.diffusion.option.plot.switch_on:
                os.makedirs(movie_dir,exist_ok=True)
            os.makedirs(sample_dir,exist_ok=True)
                
            for sample_i,contig in enumerate(contigs):
                timestap = datetime.now().strftime("%d%m%Y%H%M%S%f")
                
                if self._sample_conf.contigs is not None:
                    sample_prefix = self._sample_conf.prefix + f'{str(sample_i)}_' + (f'{timestap}_' if self._infer_conf.timestap else '')
                else:
                    length_prefix = contig.split('-')[0]
                    sample_prefix = f'{length_prefix}_{str(sample_i)}_' + (f'{timestap}_' if self._infer_conf.timestap else '')
                    
                init_feat = self._fold_module.init_feat(
                    contigs=contig,
                    ref_feats = pdb_feats,
                    hotspot = self._sample_conf.hotspot,
                )

                args_to_keep = self._fold_module.inference_fn(
                    init_feats = init_feat,
                    diffusion_dir = sample_dir,
                    diff_conf = self._diff_conf,
                    prefix = sample_prefix,
                    traj_dir = traj_dir if (self._infer_conf.diffusion.option.save_trajactory or self._infer_conf.diffusion.option.plot.switch_on) else None,
                    specified_step='diffusion',
                )
                result_df = args_to_keep['dataframe']
                if self._infer_conf.diffusion.option.plot.switch_on:
                    sample_output = args_to_keep['sample_output']
                    # traj is flipped auto in model
                    if not self._infer_conf.diffusion.option.plot.flip:
                        flip = lambda x: np.flip(np.stack(x), (0,))
                        sample_output['rigid_traj'] = flip(sample_output['rigid_traj'])
                        sample_output['rigid_0_traj'] = flip(sample_output['rigid_0_traj'])
                    plotting.write_traj(sample_output['rigid_traj'],os.path.join(movie_dir,sample_prefix+"rigid_movie.gif"))
                    plotting.write_traj(sample_output['rigid_0_traj'],os.path.join(movie_dir,sample_prefix+"rigid_0_movie.gif"))
                    
                if self._infer_conf.diffusion.option.self_consistency:
                    # Run ProteinMPNN
                    if self._infer_conf.single_process:
                        if self._fold_module.sequence_model is not None:
                            args_to_keep = self._fold_module.inference_fn(
                                sequence_conf = sequence_conf,
                                full_atom_dir = full_atom_dir,
                                prefix = sample_prefix,
                                specified_step='struct2seq',
                                args_to_keep = args_to_keep
                            )
                            result_df = args_to_keep['dataframe']
                        if self._fold_module.predict_model is not None:
                            result_df = self._fold_module.inference_fn(
                                predict_dir=predict_dir,
                                specified_step='predict',
                                args_to_keep = args_to_keep,
                                prefix = sample_prefix,
                            )
                        self._log.info(result_df)
                        result_df.to_csv(os.path.join(self._output_dir,"self_consistency.csv"), mode='a', header=False if os.path.exists(os.path.join(self._output_dir,"self_consistency.csv")) else True , index=False)
                    else:
                        def run_esm_fold_callback(args_to_keep):
                            # this is so important!! without this block subprocees will die silently without any output
                            try:
                                result_df = self._fold_module.inference_fn(
                                    predict_dir=predict_dir,
                                    specified_step='predict',
                                    args_to_keep = args_to_keep
                                )
                                result_df.to_csv(os.path.join(self._output_dir,"self_consistency.csv"), mode='a', header=False if os.path.exists(os.path.join(self._output_dir,"self_consistency.csv")) else True , index=False)
                            except:
                                raise Exception(traceback.format_exc())
                        def error_handler(error):
                            print(error)
                            sys.stdout.flush()
                        esmfold_result = pool.apply_async(mpnn_design,args=(args_to_keep,self._infer_conf.mpnn,self._fold_module.sequence_model,full_atom_dir,sample_prefix,self._infer_conf.mode.type),callback=run_esm_fold_callback,error_callback=error_handler)
                        process_list.append(esmfold_result)

            if not self._infer_conf.single_process:
                for process in process_list:
                    process.wait()

            shutil.rmtree(tmp_dir)

@hydra.main(version_base=None, config_path="../config", config_name="inference")
def run(conf: DictConfig) -> None:
    logging.getLogger(__name__).setLevel(logging.INFO)
    torch.multiprocessing.set_start_method('spawn')
    logging.disable(logging.DEBUG)
    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    sampler = Sampler(conf)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')
if __name__ == '__main__':
    run()
