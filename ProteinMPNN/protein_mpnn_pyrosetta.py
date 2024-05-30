import argparse
import os.path
import  os, sys
import logging
import dataclasses
import time
import hydra
from omegaconf import DictConfig

import pandas as pd
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import os.path
import torch.multiprocessing as mp

from experiments import utils as eu
from data import mmcif_parsing,parsers
from data import utils as du
from data import residue_constants
from data.protein import Protein,to_pdb,from_pdb_string

from ProteinMPNN.protein_mpnn_utils import  _scores, _S_to_seq, tied_featurize,parse_pdb,model_init
from ProteinMPNN.protein_mpnn_utils import StructureDatasetPDB

def mpnn(batch, model, ca_only,temperature,chain_id_dict=None):
    # global settings
    omit_AAs_list = 'CX'
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    bias_AAs_np = np.zeros(len(alphabet))
    device = next(model.parameters()).device
    with torch.no_grad():
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch, device, chain_id_dict, None, None, None, None,ca_only=ca_only)
        pssm_log_odds_mask = (pssm_log_odds_all > 0.0).float() #1.0 for true, 0.0 for false
        randn_2 = torch.randn(chain_M.shape, device=X.device)
        sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temperature, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=False, pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=False, bias_by_res=bias_by_res_all)
    
        S_sample = sample_dict["S"] 
    
        log_probs =  model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])

        mask_for_loss = mask*chain_M*chain_M_pos
        scores = _scores(S_sample, log_probs, mask_for_loss)
        scores = scores.cpu().data.numpy()
        seqs = [_S_to_seq(S_sample[i], chain_M[i]).strip() for i in range(len(batch))]

    return seqs,scores

def mpnn_design( config, protein_path ,fixed_chains=[],  model=None ,mode = 'monomer'):

    if model is None:
        model = model_init(config)
    if config.pyrosetta:
        if "pyrosetta" not in sys.modules or "pyrosetta_utils" not in sys.modules:
            try:
                global pyrosetta
                global pyrosetta_utils
                import pyrosetta
                from ProteinMPNN import pyrosetta_utils
                logging.getLogger('pyrosetta').setLevel(logging.WARNING)
                pyrosetta.init( "-beta_nov16 -in:file:silent_struct_type binary -output_pose_energies_table false" +
                    " -holes::dalphaball /storage/caolongxingLab/wangchentong/software/rosetta_master/source/external/DAlpahBall/DAlphaBall.gcc" +
                    " -use_terminal_residues true -mute basic.io.database core protocols all" +
                    " -dunbrack_prob_buried 0.8 -dunbrack_prob_nonburied 0.8" +
                    " -dunbrack_prob_buried_semi 0.8 -dunbrack_prob_nonburied_semi 0.8" )
                logging.getLogger("pyrosetta").setLevel(logging.WARNING)
                xml = f"{os.path.dirname(os.path.abspath(__file__))}/helper_scripts/xml_scripts/design.xml"
                objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_file( xml )  
            except:
                logging.info("Pyrosetta is not installed in current environment, rosetta optimization is not available")
        config.pyrosetta = True if "pyrosetta" in sys.modules else False
    num_seqs = config.num_seqs
    sample_nums = config.best_selection.sample_nums
    batch_size = min(sample_nums,config.batch_size) if config.pyrosetta else min(sample_nums*num_seqs,config.batch_size)
    batch_nums = sample_nums//batch_size if config.pyrosetta else sample_nums*num_seqs//batch_size
    
    # define designable chain in different mode
    if config.mode.type == "binder":
        fixed_chains = fixed_chains + [config.mode.binder.target_chain]
        thread_chain = config.mode.binder.binder_chain
    elif config.mode.type == "monomer":
        thread_chain = None
    else:
        thread_chain = None
        
    if "pyrosetta" in sys.modules:
        pyrosetta_movers = pyrosetta_utils.get_pyrosetta_movers(xml_objs=objs,mode=mode)
    else:
        pyrosetta_movers = None

    mpnn_result = []
    #### MPNN-Pyrosetta sequence design ####
    if config.pyrosetta:
        if pyrosetta_movers is None:
            raise ValueError("pyrosetta_movers can't be none in pyrosetta mode")
        # Pyrosetta do not allow direct load pose from cif file.
        if protein_path.endswith('.cif'):
            struct_chains = {
                chain.id: chain for chain in _mmcif_parsing.parse(
                file_id="", mmcif_string=open(protein_path).read()).mmcif_object.structure.get_chains()
            }
            chains_dict = [dataclasses.asdict(parsers.process_chain(chain, chain_id)) for chain_id, chain in struct_chains.items()]
            protein_dict = du.concat_np_features(chains_dict, False)
            protein = Protein(**protein_dict)
            pdb_string = to_pdb(protein)
            pose = pyrosetta.rosetta.core.pose.Pose()
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose,pdb_string)
        elif protein_path.endswith('.pdb'):
            pose = pyrosetta.pose_from_file(protein_path)
        else:
            raise ValueError(f'{protein_path} file type is not supported')
        for index in range(1,config.num_seqs+1):
            score_traj,sequnce_traj,protein_traj = [],[],[]
            mpnn_pose = pyrosetta.rosetta.core.pose.deep_copy(pose)
            for cycle in range(config.cycle):
                pdb_dict_list,chain_id_dict = pyrosetta_utils.parse_pose(mpnn_pose,fixed_chains, ca_only=config.ca_only)
                batch = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=20000)[0]
                seqs,scores = [],[]
                for _ in range(batch_nums):
                    batch_clones = [copy.deepcopy(batch) for i in range(batch_size)]
                    batch_seqs,batch_scores = mpnn(batch=batch_clones,model=model,ca_only=config.ca_only,temperature=config.temperature, chain_id_dict=chain_id_dict)
                    seqs.extend(batch_seqs)
                    scores.extend(batch_scores)
                min_idx = np.argmin(scores,axis=0)
                mpnn_seq = seqs[min_idx]
                mpnn_score = scores[min_idx]
                if config.cycle>1 or config.dump:
                    mpnn_pose = pyrosetta_utils.thread_mpnn_seq(mpnn_pose, mpnn_seq, chain = thread_chain )
                    if cycle<config.cycle-1:
                        pyrosetta_movers["relax_mover"].apply(mpnn_pose)
                    else:
                        # only pack sidechain on the last cycle
                        pyrosetta_movers["pack_mover"].apply(mpnn_pose)
                    # Pyrosetta Pose cant not serialization without build from source code
                    protein_traj.append(from_pdb_string(pyrosetta_utils.pose_to_string(mpnn_pose)))
                score_traj.append(mpnn_score)
                sequnce_traj.append(mpnn_seq)
            mpnn_dict = {"index":index,"sequence":sequnce_traj[-1],"score" : score_traj[-1],"sequnce_traj":sequnce_traj,"score_traj":score_traj,"scaffold":protein_path,}
            mpnn_dict["metrics"] = {
                metric_name:metric.score(mpnn_pose) for metric_name,metric in pyrosetta_movers["metrics"].items()
            }
            if config.cycle>1 or config.dump:
                mpnn_dict.update({"protein":protein_traj[-1],"protein_traj":protein_traj})
            mpnn_result.append(mpnn_dict)
    #### MPNN sequence design ####
    else:
        pdb_dict_list,chain_id_dict = parse_pdb(protein_path,fixed_chains, ca_only=config.ca_only)
        batch = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=20000)[0]
        all_mpnn_seqs,all_mpnn_scores = [],[] 
        for _ in range(batch_nums):
            batch_clones = [copy.deepcopy(batch) for i in range(batch_size)]
            seqs,scores = mpnn(batch=batch_clones,model=model,ca_only=config.ca_only,temperature=config.temperature,chain_id_dict=chain_id_dict)
            all_mpnn_seqs.extend(seqs)
            all_mpnn_scores.extend(scores)
        for index in range(config.num_seqs):
            mpnn_seuqneces = all_mpnn_seqs[index*sample_nums:(index+1)*sample_nums]
            mpnn_scores = all_mpnn_scores[index*sample_nums:(index+1)*sample_nums]
            min_idx = np.argmin(mpnn_scores,axis=0)
            mpnn_dict = {"index":index,"sequence":mpnn_seuqneces[min_idx],"score" : mpnn_scores[min_idx],"sequnce_traj":[mpnn_seuqneces[min_idx]],"score_traj":[mpnn_scores[min_idx]],"header":f'T={config.temperature}, sample={index}, score={mpnn_scores[min_idx]:.2f},scaffold={protein_path},'}
            mpnn_result.append(mpnn_dict)

    return mpnn_result

@hydra.main(version_base=None, config_path=f'{os.path.dirname(__file__)}/../config', config_name="inference")
def run(conf: DictConfig) -> None:
    from model.fold_module import FoldModule
    
    torch.multiprocessing.set_start_method('spawn')
    logging.getLogger(__name__).setLevel(logging.INFO)
    conf.inference.output_dir = os.path.abspath(conf.inference.output_dir)
    conf.inference.summary_csv = os.path.abspath(conf.inference.summary_csv)
    cache_path = os.path.realpath(f'{os.path.dirname(__file__)}/../')
    conf = eu.replace_path_with_cwd(conf,path=cache_path)
    conf.inference.mpnn.cuda = False if not torch.cuda.is_available() else conf.inference.mpnn.cuda
    print(conf.inference.mpnn)
    fold_module = FoldModule(device=torch.device('cpu' if not conf.inference.mpnn.cuda else 'cuda'))
 
    if conf.inference.mpnn.enable:
        logging.info('Loading ProteinMPNN model')
        fold_module.init_mpnn_model_from_conf(conf.inference.mpnn)
    elif conf.inference.chroma_design.enable:
        logging.info('Loading ChromaDesign model')
        fold_module.init_chroma_model_from_conf(conf.inference.chroma_design.weight_path)
    if conf.inference.esmfold.enable:
        logging.info('Loading ESMFold model')
        fold_module.init_predict_model('esmfold')
    
    output_dir = conf.inference.output_dir
    
    if conf.inference.mpnn.enable and conf.inference.mpnn.dump:
        all_atom_output_dir = os.path.join(output_dir,'mpnn')
    elif conf.inference.chroma_design.enable and conf.inference.chroma_design.dump:
        all_atom_output_dir = os.path.join(output_dir,'chroma')
    else:
        all_atom_output_dir = None
        
    if conf.inference.mpnn.enable:
        sequence_conf = conf.inference.mpnn
    elif conf.inference.chroma_design.enable:
        sequence_conf = conf.inference.chroma_design
    else:
        sequence_conf = None
    
    if conf.inference.esmfold.enable:
        predict_dir = os.path.join(output_dir, 'esmfold')
    else:
        predict_dir = None
    
    pdb_list = []
    if 'pdb' not in conf and 'pdb_list' not in conf:
        raise ValueError('Must specify either pdb or pdb_list with +pdb= or +pdb_list=')
    elif 'pdb' in conf:
        pdb_list.append(conf.pdb)
    else:
        pdb_list = open(conf.pdb_list).read().strip().splitlines()
    for pdb in pdb_list:
        prefix = os.path.splitext(os.path.basename(pdb))[0]
        diff_prot = from_pdb_string(open(pdb).read())
        diff_feat = dataclasses.asdict(diff_prot)
        args_to_keep = {
            "init_feats" : {
                "residue_index" : diff_feat["residue_index"][None],
                "chain_index" : diff_feat["chain_index"][None],
                "fixed_mask" : np.zeros_like(diff_feat["residue_index"][None]),
            },
            "sample_output" : {
                "final_atom_positions" : diff_feat["atom_positions"],
                "final_atom_mask" : diff_feat["atom_mask"],      
            },
            "diff_path" : pdb,
            "diff_prot" : diff_prot,
            "dataframe" : pd.DataFrame({
                'diff_path' : [pdb]
            })
        }
        start_time = time.time()
        if sequence_conf is not None:
            args_to_keep = fold_module.inference_fn(
                sequence_conf = sequence_conf,
                specified_step='struct2seq',
                args_to_keep = args_to_keep,
                full_atom_dir=all_atom_output_dir,
                prefix=prefix
            )
        else:
            sequence= "".join([residue_constants.restypes_with_x[aa] for aa in diff_feat['aatype']])
            change_positions = np.where(diff_feat["chain_index"][:-1] != diff_feat["chain_index"][1:])[0] + 1
            sequence_list = list(sequence)
            for pos in reversed(change_positions):
                sequence_list.insert(pos, ':')
            sequence = ''.join(sequence_list)
            args_to_keep['dataframe']['sequence'] = sequence
            args_to_keep['sequence'] = [sequence]
        
        if conf.inference.esmfold.enable:
            result_df = fold_module.inference_fn(
                predict_dir=predict_dir,
                specified_step='predict',
                args_to_keep = args_to_keep,
                prefix = prefix,
            )
        else:
            result_df = args_to_keep["dataframe"]
        with open(os.path.join("./",'mpnn_sequence.fasta'),'a+') as f:
            for i, mpnn_result in result_df.iterrows():
                f.write(f'>{os.path.splitext(os.path.basename(mpnn_result["diff_path"]))[0]}_mpnn_{i}\n {mpnn_result["sequence"]}\n')
        
        result_df.to_csv(os.path.join(output_dir,"self_consistency.csv"), mode='a', header=False if os.path.exists(os.path.join(output_dir,"self_consistency.csv")) else True , index=False)
                  
        eval_time = time.time() - start_time
        print(f'File {pdb}, Length : {len(result_df.iloc[0]["sequence"])}, num sequence : {len(result_df)}, Finished in {eval_time:.2f}s')

if __name__ == '__main__':
    run()