import logging
import GPUtil
import tree
import copy
import os
import dataclasses
import time
from typing import Dict,Optional,Union
from functools import partial
import re

import torch
import numpy as np
import pandas as pd
import esm
from omegaconf import OmegaConf

from model.score_network import ScoreNetwork
from ProteinMPNN.protein_mpnn_utils import ProteinMPNN
from ProteinMPNN import protein_mpnn_pyrosetta

from chroma import api
# api.register_key('f231b55b2e214a1585a4a17f54dbba4e')
from chroma.models.graph_design import GraphDesign

from openfold.utils import rigid_utils as ru

from data import utils as du
from data import data_transforms
from data import all_atom
from data import protein
from data import residue_constants
from data.se3_diffuser import SE3Diffuser
from analysis import metrics
from experiments import utils as eu 


logging.basicConfig(level=logging.WARNING,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ])

class FoldModule(torch.nn.Module):
    def __init__(
        self,
        structure_model : ScoreNetwork = None,
        sequence_model : Union[ProteinMPNN , GraphDesign] = None, 
        predict_model : str = None,
        diffuser : SE3Diffuser = None,
        device = None,
        crystal_design = False,
    ):
        """
        Args:
            predict_model (str): 'esmfold' or None
            ...
        """
        super(FoldModule, self).__init__()
        self.structure_model = structure_model
        self.sequence_model = sequence_model
        self.diffuser = diffuser
        self.crystal_design = crystal_design
        
        if device is not None:
            self.device = device 
        elif structure_model is not None:
            self.device = next(structure_model.parameters()).device
        else:
            available_gpus = ''.join([str(x) for x in GPUtil.getAvailable(order='memory', limit = 8)])
            self.device = f'cuda:{available_gpus[0]}'
            
        self.predict_model = predict_model
        
        self._log = logging.getLogger(__name__)

    def init_strcture_model_from_conf(self, ckpt_path, model_conf = None):
        """Loads in model checkpoint."""
        self._log.info(f'Loading weights from {ckpt_path}')

        # Read checkpoint and create experiment.
        weights_pkl = du.read_pkl(
            ckpt_path, use_torch=True,
            map_location=self.device)

        # Merge base experiment config with checkpoint config.
        if model_conf is not None:
            model_conf = OmegaConf.merge(
                model_conf, weights_pkl['conf'].model)
        else:
            model_conf = weights_pkl['conf'].model

        self.structure_model = ScoreNetwork(model_conf)
        
        # Remove module prefix if it exists.
        # if 'ema' not in weights_pkl or weights_pkl['ema'] is None:
        model_weights = weights_pkl['model']
        # else:
            # model_weights = weights_pkl['ema']
            # logging.info('Using EMA weights')
        model_weights = {
            k.replace('module.', ''):v for k,v in model_weights.items()}
        missing_keys,unexpected_keys = self.structure_model.load_state_dict(model_weights,strict=False)
        if len(unexpected_keys)>0:
            self._log.warning(f"unexpected {len(unexpected_keys)} keys in checkpoint: {unexpected_keys[:10]}...")
        if len(missing_keys)>0:
            self._log.warning(f"missing {len(missing_keys)} keys in checkpoint: {missing_keys[:10]}...")
        self.structure_model = self.structure_model.to(self.device)
        self.structure_model.eval()
        del weights_pkl
        
    def init_mpnn_model_from_conf(self, model_conf):
        self.sequence_model = protein_mpnn_pyrosetta.model_init(model_conf)
    
    def init_chroma_model_from_conf(self,weight_path):
        params = torch.load(weight_path, map_location="cpu")
        self.sequence_model = GraphDesign(**params["init_kwargs"]).to(self.device).eval()
        missing_keys, unexpected_keys = self.sequence_model.load_state_dict(
            params["model_state_dict"], strict=False
        )
        self.sequence_model.set_gradient_checkpointing(False)
        if len(unexpected_keys) > 0:
            raise Exception(
                f"Error loading model from checkpoint file: {weight_path} contains {len(unexpected_keys)} unexpected keys: {unexpected_keys}"
            )
        
    def init_predict_model(self,model_name):
        if model_name == 'esmfold':
            self.predict_model = esm.pretrained.esmfold_v1().eval()
            self.predict_model = self.predict_model.to(self.device)
        else:
            raise ValueError(f'Unknown predict model {model_name}')
        
    def squence_design(self,config,scaffold_path):
        if isinstance(self.sequence_model,ProteinMPNN):
            result = protein_mpnn_pyrosetta.mpnn_design(
                        config= config,
                        protein_path=scaffold_path,
                        model= self.sequence_model,
                        mode = config.mode.type
                    )
        elif isinstance(self.sequence_model,GraphDesign):
            result = []
            prot_feat = du.parse_pdb_feats("",scaffold_path)
            # prot_feat = {k:torch.tensor(v,device=self.device).unsqueeze(0).expand(config.num_seqs, *v.shape) for k,v in prot_feat.items()}
            prot_feat = {k:torch.tensor(v,device=self.device).unsqueeze(0) for k,v in prot_feat.items()}
            for _ in range(config.num_seqs):
                with torch.no_grad():
                    X_sample, S_sample, _, scores = self.sequence_model.sample(
                        # input coordinate is N,CA,C,O from atom37
                        X = prot_feat['atom_positions'][:,:,[0,1,2,4]].to(torch.float32),
                        C = prot_feat['chain_index'].long(),
                        S = torch.zeros_like(prot_feat['chain_index']).long(),
                        t=config.t,
                        temperature_S=config.temperature,
                        ban_S=["C"],
                        return_scores=True
                    )
                    final_atom_positions,final_atom_mask = data_transforms.atom14_chroma_to_atom37(X_sample,S_sample)
                    chroma_aatype_to_standard_aatype = torch.tensor([residue_constants.restype_order[aa] for i,aa in enumerate(residue_constants.restypes_chroma)],dtype=torch.int64,device=self.device)
                    standard_aatype = chroma_aatype_to_standard_aatype[S_sample]    
                    result.append({
                        'sequence' : du.aatype_to_seq(standard_aatype[0]),
                        'score' : scores["neglogp_S"][0],
                        'protein' : protein.Protein(
                            aatype=standard_aatype[0].cpu().numpy(),
                            atom_positions=final_atom_positions[0].cpu().numpy(),
                            atom_mask=final_atom_mask[0].cpu().numpy(),
                            residue_index=prot_feat['residue_index'][0].cpu().numpy(),
                            chain_index=prot_feat["chain_index"][0].cpu().numpy(),
                            b_factors=final_atom_mask[0].cpu().numpy() * 100
                        )
                    })
        else:
            raise ValueError(f"sequence model type {type(self.sequence_model)} is currently not supported")
        return result

    def inference_fn(
        self,
        init_feats : Dict = None,
        diffusion_dir : str = None,
        diff_conf :  OmegaConf = None,
        sequence_conf : OmegaConf = None,
        full_atom_dir : str = None,
        predict_dir : str = None,
        traj_dir : bool = None,
        prefix : str = "",
        specified_step : str = None, # one of diffusion,struct2seq,predict
        args_to_keep : Dict = {}
    ) -> pd.DataFrame :
        ## scaffold generation ####
        if specified_step == 'diffusion' or specified_step is None:
            step_time = time.time()
            os.makedirs(diffusion_dir, exist_ok=True)
            sample_output = self.diffusion_sample(
                init_feats = init_feats,
                min_t=diff_conf.min_t,
                num_steps=diff_conf.num_t,
                noise_scale=diff_conf.noise_scale,
                esm_rate=diff_conf.rate_t_esm_condition,
                generate_seq_nums=diff_conf.rate_t_esm_seq_nums,
                temperature = diff_conf.temperature,
            )
            init_feats = {k: du.move_to_np(v) for k, v in init_feats.items()}
            diff_prot = protein.Protein(
                aatype=sample_output["pred_aatype"][0] if "pred_aatype" in sample_output else init_feats["aatype"][0],
                atom_positions=sample_output["final_atom_positions"],
                atom_mask=sample_output["final_atom_mask"],
                residue_index=init_feats["residue_index"][0],
                chain_index=init_feats["chain_index"][0],
                b_factors=np.tile(1 -init_feats['fixed_mask'][0,..., None], 37) * 100
            )
            diff_path = os.path.join(diffusion_dir, prefix+'diff.pdb')
            open(diff_path,'w').write(protein.to_pdb(diff_prot))
            
            if traj_dir is not None:
                with open(os.path.join(traj_dir, prefix+'diff.pdb'),'w') as f:
                    for t,atom_positions in  enumerate(sample_output['prot_traj']):
                        prot = protein.Protein(
                            aatype=init_feats["aatype"][0],
                            atom_positions=atom_positions,
                            atom_mask=sample_output["final_atom_mask"],
                            residue_index=init_feats["residue_index"][0],
                            chain_index=init_feats["chain_index"][0],
                            b_factors=np.tile(1 -init_feats['fixed_mask'][0,..., None], 37) * 100
                        )
                        pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                        f.write(pdb_prot)
                    f.write('END')
                    
            diffusion_result = {
                'diff_path' : diff_path,
                **metrics.calc_mdtraj_metrics(diff_path),
                "diff_time" : time.time()-step_time,
            }
            self._log.info(f'generate scaffold {diff_path} in {time.time()-step_time:.2f}s')
            diff_df = pd.DataFrame({k:[v] for k,v in diffusion_result.items()})
            args_to_keep.update({
                'init_feats' : init_feats,
                'sample_output' : sample_output,
                'diff_path' : diff_path,
                'diff_prot' : diff_prot,
                'dataframe' : diff_df
            })
            if specified_step == 'diffusion':
                return args_to_keep
            if self.sequence_model is None and 'pred_aatype' not in sample_output :
                return diff_df
        
        #### structure to sequence ####
        if specified_step == 'struct2seq' or specified_step is None:
            step_time = time.time()
            full_atom_prots = []
            struct2seq_results = []
            init_feats = args_to_keep['init_feats']
            sample_output = args_to_keep['sample_output']
            diffusion_file_path = args_to_keep['diff_path']
            diff_df = args_to_keep['dataframe']
            if full_atom_dir is not None:
                os.makedirs(full_atom_dir, exist_ok=True)

            if self.sequence_model is None and 'pred_aatype' in sample_output:
                # end to end
                for i in range (sample_output['pred_aatype'].shape[0]):
                    struct2seq_result = {}
                    sequence = du.aatype_to_seq(sample_output['pred_aatype'][i])
                    struct2seq_result['sequence'] = sequence
                    full_atom_prots.append(protein.Protein(
                        aatype=sample_output["pred_aatype"][i],
                        atom_positions=sample_output["final_atom_positions"],
                        atom_mask=sample_output["final_atom_mask"],
                        residue_index=init_feats["residue_index"][0],
                        chain_index=init_feats["chain_index"][0],
                        b_factors=np.tile(1 -init_feats['fixed_mask'][0,..., None], 37) * 100
                    ))
                    if full_atom_dir is not None:
                        full_atom_path = os.path.join(full_atom_dir, prefix+f'fa_{i}.pdb')
                        open(full_atom_path,'w').write(protein.to_pdb(full_atom_prots[i]))
                        struct2seq_result['full_atom_path'] = full_atom_path
                    struct2seq_results.append(struct2seq_result)

            elif self.sequence_model is not None and sequence_conf is not None:
                # with mpnn model
                mpnn_results = self.squence_design(sequence_conf, diffusion_file_path)
                for i, mpnn_result in enumerate (mpnn_results):
                    struct2seq_result = {}
                    struct2seq_result['sequence'] = mpnn_result['sequence']
                    if full_atom_dir is not None and 'protein' in mpnn_result:
                        full_atom_prots.append(mpnn_results[i]["protein"])
                        full_atom_path = os.path.join(full_atom_dir, prefix+f'fa_{i}.pdb')
                        open(full_atom_path,'w').write(protein.to_pdb(full_atom_prots[i]))
                        struct2seq_result['full_atom_path'] = full_atom_path
                    elif full_atom_dir is not None and 'protein' not in mpnn_result:
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
                    # add chain seperator to sequence
                    change_positions = np.where(init_feats["chain_index"][0][:-1] != init_feats["chain_index"][0][1:])[0] + 1
                    sequence_list = list(struct2seq_result['sequence'])
                    for pos in reversed(change_positions):
                        sequence_list.insert(pos, ':')
                    struct2seq_result['sequence'] = ''.join(sequence_list)
                    struct2seq_results.append(struct2seq_result)
            struct2seq_results = {key: list(map(lambda d: d[key], struct2seq_results)) for key in struct2seq_results[0]}
            struct2seq_df = pd.DataFrame(struct2seq_results)
            struct2seq_df['diff_path'] = diffusion_file_path
            struct2seq_df = struct2seq_df.merge(diff_df,on='diff_path',how='left')
            self._log.info(f'generate {len(struct2seq_results["sequence"])} sequences for scaffold {diffusion_file_path} in {time.time()-step_time:.2f}s')
            args_to_keep.update({
                'dataframe' : struct2seq_df,
                'sequence' : struct2seq_results['sequence']
            })
        if specified_step == 'struct2seq':
            return args_to_keep
        if self.predict_model is None:
            return struct2seq_df
              
        #### sequence to structure
        if specified_step == 'predict' or specified_step is None:
            struct2seq_df = args_to_keep['dataframe']
            step_time = time.time()
            self_consistency_results = []
            if predict_dir is not None:
                os.makedirs(predict_dir, exist_ok=True)
            for i ,sequence in enumerate(args_to_keep['sequence']):
                predict_prot = self.run_folding(sequence)
                self_consistency_result = self.self_consistency_eval(args_to_keep['diff_prot'],predict_prot)
                self_consistency_results.append(self_consistency_result)
                if predict_dir is not None:
                    predict_path = os.path.join(predict_dir, prefix+f'pred_{i}.pdb')
                    open(predict_path,'w').write(protein.to_pdb(predict_prot))
                    self_consistency_result["predict_path"] = predict_path
            self_consistency_results = {key: list(map(lambda d: d[key], self_consistency_results)) for key in self_consistency_results[0]}
            predict_df = pd.DataFrame(self_consistency_results)
            predict_df = pd.concat([predict_df, struct2seq_df], axis=1)
            self._log.info(f'generate {len(args_to_keep["sequence"])} predictions for scaffold {args_to_keep["diff_path"]} in {time.time()-step_time:.2f}s')
        
        return predict_df

    def diffusion_sample(
        self, 
        init_feats, 
        num_steps : int, 
        min_t : float = 0.01, 
        generate_seq_nums : int = 8, 
        temperature : float = 0.1,
        noise_scale : float = 0.1,
        self_condition : bool = True,
        esm_rate: float = 0.0,
    ):
        """Sample based on length.
        Args:
            sample_length: length to sample

        Returns:
            Sample outputs.
        """
        if self.structure_model._model_conf.embed.self_condition.struct2seq.enable:
            self.structure_model.embedding_layer.struct2seq_embedder.temperature = temperature
            self.structure_model.embedding_layer.struct2seq_embedder.seq_nums = generate_seq_nums
        
        # Process motif features.
        sample_feats = init_feats
        
        # Run inference
        device = sample_feats['rigids_t'].device
        if sample_feats['rigids_t'].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones(
                (sample_feats['rigids_t'].shape[0],)).to(device)
        reverse_steps = np.linspace(min_t, 1.0, num_steps)[::-1]
        num_t_esm = int(esm_rate * num_steps)
        reverse_steps_esm = reverse_steps[np.linspace(0, num_steps-1, num_t_esm, dtype=int)]
        dt = 1/num_steps
        all_msas = []
        all_seqs = []
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats['rigids_t']))]
        all_rigids_0 = []
        all_bb_prots = []
        all_bb_mask = []
        all_bb_0_pred = []
        with torch.no_grad():
            model_out = None
            diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
            for t in reverse_steps:
                
                model_out = None if not self_condition else model_out
                
                if t > min_t:
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                model_out = self.structure_model(sample_feats,self_condition=model_out,struct2seq=True if t in reverse_steps_esm else False)
                
                if t > min_t:
                    rot_score = self.diffuser.calc_rot_score(
                        ru.Rigid.from_tensor_7(sample_feats['rigids_t']).get_rots(),
                        ru.Rotation(model_out['pred_rotmats']),
                        sample_feats['t']
                    )
                    trans_score = self.diffuser.calc_trans_score(
                        ru.Rigid.from_tensor_7(sample_feats['rigids_t']).get_trans(),
                        model_out['pred_trans'],
                        sample_feats['t'],
                        use_torch = True,
                    )
                    rigid_pred = model_out['rigids']
                    rigids_t = self.diffuser.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
                        rot_score=du.move_to_np(rot_score),
                        trans_score=du.move_to_np(trans_score),
                        diffuse_mask=du.move_to_np(diffuse_mask),
                        t=t,
                        dt=dt,
                        noise_scale=noise_scale,
                    ) 
                else:
                    rigids_t = model_out['rigids']

                sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
                res_mask = sample_feats['res_mask'][0].detach().cpu()
                
                all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))
                all_rigids_0.append(du.move_to_np(rigid_pred.to_tensor_7()))
                atom37_0 = all_atom.to_atom37(model_out['pred_trans'], model_out['pred_rotmats'])[0]
                atom37_0 = du.adjust_oxygen_pos(atom37_0, res_mask)
                all_bb_0_pred.append(du.move_to_np(atom37_0))
                if 'pred_aatype' in model_out:
                    # currently we only take the first sequence no matter how many sequence we generate
                    # there might be a better way to merge all generated seqs to final seq like mutilple template embedding.
                    all_seqs.append(du.move_to_np(model_out['pred_aatype'][0,0]))
                else:
                    all_seqs.append(None)
                    
                atom37_t = all_atom.to_atom37(rigids_t._trans,rigids_t._rots.get_rot_mats())[0]
                atom37_t = du.adjust_oxygen_pos(atom37_t, res_mask)
                all_bb_prots.append(du.move_to_np(atom37_t))
                all_bb_mask.append(du.move_to_np(res_mask))
                
                all_msas.append( du.move_to_np(model_out['pred_aatype']) if 'pred_aatype' in model_out else None)
                    
            # update final features
            sample_out = {
                "final_atom_positions" : du.move_to_np(atom37_0[None]),
                "final_atom_mask" : du.move_to_np(init_feats["atom37_atom_exists"]),
                "fixed_mask" : du.move_to_np(init_feats["fixed_mask"]),
            }
            if 'pred_aatype' in model_out:
                sample_out['pred_aatype'] = du.move_to_np(model_out['pred_aatype'])
        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_msas = flip(all_msas)[:,None]
        all_seqs = flip(all_seqs)[:,None]
        all_bb_prots = flip(all_bb_prots)[:,None]
        all_bb_mask = flip(all_bb_mask)[:,None]
        all_rigids = flip(all_rigids)
        all_rigids_0 = flip(all_rigids_0)
        all_bb_0_pred = flip(all_bb_0_pred)[:,None]
        traj_out = {
            'seq_traj': all_seqs,
            'prot_traj': all_bb_prots,
            'mask_traj': all_bb_mask,    
            # aux traj
            'rigid_traj' : all_rigids,
            'rigid_0_traj' : all_rigids_0,
            'prot_0_traj' : all_bb_0_pred,
        }
        traj_out = tree.map_structure(lambda x: x[:,0], traj_out)
        sample_out = tree.map_structure(lambda x: x[0], sample_out)
        sample_out.update(traj_out)
        return sample_out
    
    def run_folding(self, sequence):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self.predict_model.infer(sequence)
        
        output['final_atom_positions'] = data_transforms.atom14_to_atom37(output["positions"][-1], output)
        output = {k: v.to("cpu").numpy()[0] for k, v in output.items() if k in ["aatype", "plddt","final_atom_positions","atom37_atom_exists", "residue_index", "chain_index"]}
        residue_mask = output["atom37_atom_exists"][:,1] == 1
        output = {key: value[residue_mask] for key, value in output.items()}
        pred_prot = protein.Protein(
            aatype=output["aatype"],
            atom_positions=output['final_atom_positions'],
            atom_mask=output["atom37_atom_exists"],
            residue_index=output["residue_index"] + 1,
            b_factors=output["plddt"],
            chain_index=output["chain_index"] if "chain_index" in output else None,
        )
        return pred_prot
    
    def _set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats
    
    def _calc_trans_0(self, trans_score, trans_t, t):
        beta_t = self.diffuser._se3_diffuser._r3_diffuser.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_score * cond_var + trans_t) / torch.exp(-1/2*beta_t)
    
    def self_consistency_eval(self,reference_model,predict_model,motif_mask:Optional[np.ndarray]=None):
        """Run self-consistency on design proteins against reference protein.
        
        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file
            motif_mask: Optional mask of which residues are the motif.

        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf
            Writes results in decoy_pdb_dir/sc_results.csv
        """
        eval_result = {}
        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            eval_result['motif_rmsd'] = None 
        reference_feat = du.parse_protein_feats(reference_model)
        predict_feat = du.parse_protein_feats(predict_model)
            
        # Calculate scTM of ESMFold outputs with reference protein
        _, tm_score = metrics.calc_tm_score(
            reference_feat['bb_positions'], predict_feat['bb_positions'])
        rmsd = metrics.calc_aligned_rmsd(
            reference_feat['bb_positions'], predict_feat['bb_positions'])
        if motif_mask is not None:
            sample_motif = reference_feat['bb_positions'][motif_mask]
            of_motif = predict_feat['bb_positions'][motif_mask]
            motif_rmsd = metrics.calc_aligned_rmsd(
                sample_motif, of_motif)
            eval_result['motif_rmsd'] = motif_rmsd
        eval_result['plddt'] = predict_feat['b_factors'].mean()
        eval_result['rmsd'] = rmsd
        eval_result['tm_score'] = tm_score

        return eval_result

    def init_feat(
            self,
            contigs,
            ref_feats = None,
            hotspot = None,
        ):
        
        init_feat = {
            'aatype' : np.zeros(0,dtype=np.int64),
            'res_mask': np.zeros(0,dtype=bool),
            'residue_index': np.zeros(0,dtype=np.int64),
            'chain_index' : np.zeros(0,dtype=np.int64),
            'fixed_mask': np.zeros(0,dtype=bool),
            'all_atom_positions': np.zeros((0,37,3),dtype=np.float32),
            'all_atom_mask': np.zeros((0,37),dtype=bool),
            'rigids_t':  np.zeros((0,7),dtype=np.float32),
            'hotspot': np.zeros(0,dtype=np.int64),
        }

        if hotspot is not None:
            hotspot = [re.match(r'([A-Za-z]*)(\d+)', hotspot_res).groups() for hotspot_res in hotspot]
            hotspot = [(hotspot_res[0],int(hotspot_res[1])) for hotspot_res in hotspot]

        cur_index = 0
        cur_chain_index = 0
        for contig in contigs.split('/'):
            if contig == '':
                cur_index = 0
                cur_chain_index += 1
                continue
            chain, start, end = re.match(r'([A-Za-z]*)(\d+)-(\d+)', contig).groups()
            start,end = int(start),int(end)
            if chain != '':
                sele_idx = np.argwhere(ref_feats['chain_index']==chain)[:,0]
                sele_idx = [idx for idx in sele_idx if ref_feats['residue_index'][idx] >= int(start) and ref_feats['residue_index'][idx] <= int(end)]
                seg_length = len(sele_idx)
                init_feat['aatype'] = np.concatenate([init_feat['aatype'],ref_feats['aatype'][sele_idx]])
                init_feat['fixed_mask'] = np.concatenate([init_feat['fixed_mask'],np.ones(len(sele_idx),dtype=bool)])
                init_feat['all_atom_positions'] = np.concatenate([init_feat['all_atom_positions'],ref_feats['atom_positions'][sele_idx]])
                init_feat['all_atom_mask'] = np.concatenate([init_feat['all_atom_mask'],ref_feats['atom_mask'][sele_idx]])
                hotspot_feat = np.zeros(seg_length,dtype=np.int64)
                if hotspot is not None:
                    hotspot_idx = [idx for idx in sele_idx if (ref_feats['chain_index'][idx],ref_feats['residue_index'][idx]) in hotspot]
                    hotspot_feat[hotspot_idx] = 1
                init_feat['hotspot'] = np.concatenate([init_feat['hotspot'],hotspot_feat])
                
            else:
                seg_length = np.random.randint( start, end+1 )
                init_feat['aatype'] = np.concatenate([init_feat['aatype'],np.full((seg_length),residue_constants.resname_to_idx['ALA'],dtype=np.int64)])
                init_feat['fixed_mask'] = np.concatenate([init_feat['fixed_mask'],np.zeros(seg_length,dtype=bool)])
                init_feat['all_atom_positions'] = np.concatenate([init_feat['all_atom_positions'],np.zeros((seg_length,37,3),dtype=np.float32)])
                init_feat['all_atom_mask'] = np.concatenate([init_feat['all_atom_mask'],np.zeros((seg_length,37),dtype=bool)])
                init_feat['hotspot'] = np.concatenate([init_feat['hotspot'],np.zeros(seg_length,dtype=np.int64)])
            
            init_feat['residue_index'] = np.concatenate([init_feat['residue_index'],np.arange(cur_index,cur_index+seg_length)])
            init_feat['res_mask'] = np.concatenate([init_feat['res_mask'],np.ones(seg_length,dtype=bool)])
            init_feat['chain_index'] = np.concatenate([init_feat['chain_index'],[cur_chain_index]*seg_length])
            cur_index += seg_length

        if np.sum(init_feat['fixed_mask']) > 0:
            init_feat['all_atom_positions'] = init_feat['all_atom_positions'] - np.mean(init_feat['all_atom_positions'][init_feat['fixed_mask']][:,1],axis=0,keepdims=True)

        protein_length = init_feat['aatype'].shape[0]

        init_feat = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feat)
        
        # add additional features for finetuning models
        init_feat.update({
            'ss': torch.nn.functional.one_hot(torch.full((protein_length,),3).long(),4).float(),
            'adjacency': torch.nn.functional.one_hot(torch.full((protein_length,protein_length),2).long(),3).float(),
            'hotspot': torch.nn.functional.one_hot(init_feat['hotspot'],2).float(),
        })

        data_transforms.atom37_to_frames(init_feat)
        data_transforms.make_atom14_masks(init_feat)
        # data_transforms.make_atom14_positions(init_feat)
        data_transforms.atom37_to_torsion_angles(init_feat)
        init_feat['rigids_t'] = ru.Rigid.from_tensor_4x4(init_feat['rigidgroups_gt_frames'])[:, 0].to_tensor_7()

        sample_rigids_t = self.diffuser.sample_ref(
            n_samples=protein_length,
            as_tensor_7=True,
        )['rigids_t']
        init_feat['rigids_t'][~init_feat['fixed_mask']] = sample_rigids_t[~init_feat['fixed_mask']]
        init_feat['fixed_mask'] = init_feat['fixed_mask'].float()
        init_feat = tree.map_structure(
            lambda x: x[None].to(self.device), init_feat)
        return init_feat
