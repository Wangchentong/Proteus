import pickle
import os
from typing import Any
import numpy as np
from typing import List, Dict, Any
import collections
from omegaconf import OmegaConf
import dataclasses
import functools
import tree
from data import chemical
from data import residue_constants
from data import protein
from data import so3_utils
from openfold.utils import rigid_utils
from scipy.spatial.transform import Rotation
from Bio import PDB
from Bio.PDB.Chain import Chain
import string
import io
import gzip
from torch.utils import data
from torch_scatter import scatter_add, scatter
import torch
from enum import Enum
import random
import functools as fn
import logging
Protein = protein.Protein

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

# Global map from chain characters to integers.
ALPHANUMERIC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789' + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

protein_keys = ["atom_positions", "aatype", "atom_mask", "residue_index", "chain_index", "b_factors"]

# NOTE: How to give definition to a feature neet to process
#### Definition ####
# CHAIN_FEATS : all none-template and none-msa feature that need to be processed, else will be droped
# TEMPLATE_FEATURES : all extra template feature that need to be processed
# MSA_FEATURE : all extra msa feature that need to be processed

#### Property ####
# UNPADDED_FEATS : all feature that need not padding, eg. some constant scalar like diffuse time T
# PAIR_FEATS : all feature that need to be processed as residue pair, eg. pair distance
# RIGID_FEATS : all feature that need to be processed as rigid, eg. rigid groups

CHAIN_FEATS = [
    # origin
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors',
    # rename origin feature
    'all_atom_positions','atom37_atom_exists',
    # for torsion loss
    'chi_angles_sin_cos','chi_mask','torsion_angles_mask',"torsion_angles_sin_cos"
    # for structure prediction
    'residx_atom14_to_atom37',
    'res_mask','atom37_pos','atom37_mask','atom14_pos',
    # for sidechain atom rename and fape loss of sidechain(which use atom14 rather than atom37)
    'atom14_gt_exists','atom14_gt_position','atom14_alt_gt_exists','atom14_alt_gt_positions',
    'atom14_atom_exists','atom14_atom_is_ambiguous',"rigidgroups_gt_exists","rigidgroups_gt_frames","rigidgroups_alt_gt_frames",
    # for complex rel pos encoding
    'chain_index',
    # additional info for finetuning task
    'fixed_mask', 'ss',
]
TEMPLATE_FEATURES = [
    "template_mask","template_aatype","template_pseudo_beta","template_pseudo_beta_mask",
    "template_all_atom_mask","template_all_atom_positions","template_torsion_angles_mask",
    "template_torsion_angles_sin_cos","template_alt_torsion_angles_sin_cos","template_sum_probs",
]
MSA_FEATURES = [
    "msa", "msa_mask", "deletion_matrix","msa_t","msa_0","bert_mask",
]

UNPADDED_FEATS = [
    't', 'rot_score_scaling', 'trans_score_scaling', 't_seq', 't_struct', "template_sum_probs","template_mask",
]
UNPADDED_FEATS.extend(["self_condition_"+i for i in UNPADDED_FEATS])
RIGID_FEATS = [
    'rigids_0', 'rigids_t',"rigids_all_0"
]
RIGID_FEATS.extend(["self_condition_"+i for i in RIGID_FEATS])
PAIR_FEATS = [
    # TRrosetta-like pair loss feature
    'dist6d','omega6d','theta6d','phi6d','mask6d','adjacency_matrix',
    # additional info for finetuning task
    'adjacency',
]
MSA_PAD_TOKEN = 21

move_to_np = lambda x: x.cpu().detach().numpy() if isinstance(x,torch.Tensor) else x
aatype_to_seq = lambda aatype: ''.join([
        residue_constants.restypes_with_x[x] for x in aatype])
seq_to_aatype = lambda seq: np.array([
        residue_constants.restype_order_with_x[x] for x in seq],np.int64)
chroma_aatype_to_seq = lambda aatype: ''.join([
        residue_constants.restypes_chroma[x] for x in aatype]) 
chroma_aatype_to_standard_aatype = lambda aatype: np.array([
        residue_constants.chroma_restype_order_to_standard_restype_order[x] for x in aatype],np.int64)
standard_aatype_to_chroma_aatype = lambda aatype: np.array([
        residue_constants.restype_order_to_chroma_order[x] for x in aatype],np.int64)

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]

class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def write_pkl(
        save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

@fn.lru_cache(maxsize=100)
def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, 'rb') as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, 'rb') as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(f'Failed to read {read_path}. First error: {e}\n Second error: {e2}')
            raise(e)

def compare_conf(conf1, conf2):
    return OmegaConf.to_yaml(conf1) == OmegaConf.to_yaml(conf2)

def parse_pdb(filename):
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines)

def parse_pdb_lines(lines):

    # indices of residues observed in the structure
    idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
    seq = []
    for l in lines:
        if l[:4] != "ATOM":
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
        seq.append(residue_constants.restype_3to1[aa])
        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(chemical.aa2long[chemical.aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    return xyz, mask, np.array(idx_s), ''.join(seq)

def chain_str_to_int(chain_str: str):
    chain_int = 0
    for i, chain_char in enumerate(chain_str):
        chain_int = CHAIN_TO_INT[chain_char] + chain_int * len(ALPHANUMERIC)
    return chain_int

def chain_int_to_str(chain_int: int):
    chain_str = ""
    while chain_int >0:
        chain_str = chain_str + INT_TO_CHAIN[chain_int % len(ALPHANUMERIC)]
        chain_int = chain_int // len(ALPHANUMERIC)
    
    return chain_str

def parse_protein_feats(
        protein : Protein
    ):
    assemble_feat_dict = dataclasses.asdict(protein)
    assemble_feat_dict = {x: assemble_feat_dict[x] for x in CHAIN_FEATS if x in assemble_feat_dict}
    return parse_chain_feats(assemble_feat_dict, scale_factor=1)

def parse_pdb_feats(
        pdb_name: str,
        pdb_path: str,
        scale_factor=1.,
        chain_ids=None,
    ):
    """
    Args:
        pdb_name: name of PDB to parse.
        pdb_path: path to PDB file to read.
        scale_factor: factor to scale atom positions.
        mean_center: whether to mean center atom positions.
    Returns:
        Dict with CHAIN_FEATS features extracted from PDB with specified
        preprocessing.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_path)
    struct_chains = {
        chain.id: chain
        for chain in structure.get_chains()}
    
    if isinstance(chain_ids, str):
        struct_chains = {chain_ids: struct_chains[chain_ids]}
    elif isinstance(chain_ids, list):
        struct_chains = {
            chain_id: struct_chains[chain_id] for chain_id in chain_ids
        }
    else:
        struct_chains = struct_chains

    assemble_chains_dict = [
        dataclasses.asdict(process_chain(struct_chains[chain_id], chain_id)) for chain_id in struct_chains
    ]
    assemble_feat_dict = concat_np_features(assemble_chains_dict, False)
    assemble_feat_dict = {x: assemble_feat_dict[x] for x in CHAIN_FEATS if x in assemble_feat_dict}

    return parse_chain_feats(
            assemble_feat_dict, scale_factor=scale_factor)

def create_msa_feat(batch):
    """Create and concatenate MSA features."""
    msa_1hot = np.eye(23)[batch['msa']]
    deletion_matrix = batch['deletion_matrix']
    has_deletion = np.clip(deletion_matrix, 0., 1.)[..., None]
    deletion_value = (np.arctan(deletion_matrix / 3.) * (2. / np.pi))[..., None]

    msa_feat = [
        msa_1hot,
        has_deletion,
        deletion_value,
    ]
    return np.concatenate(msa_feat, axis=-1)

def rigid_frames_from_atom_14(atom_14):
    n_atoms = atom_14[:, 0]
    ca_atoms = atom_14[:, 1]
    c_atoms = atom_14[:, 2]
    return rigid_utils.Rigid.from_3_points(
        n_atoms, ca_atoms, c_atoms
    )

def compose_rotvec(r1, r2):
    """Compose two rotation euler vectors."""
    R1 = rotvec_to_matrix(r1)
    R2 = rotvec_to_matrix(r2)
    cR = np.einsum('...ij,...jk->...ik', R1, R2)
    return matrix_to_rotvec(cR)

def rotvec_to_matrix(rotvec):
    return Rotation.from_rotvec(rotvec).as_matrix()

def matrix_to_rotvec(mat):
    return Rotation.from_matrix(mat).as_rotvec()

def rotvec_to_quat(rotvec):
    return Rotation.from_rotvec(rotvec).as_quat()

def pad_feats(raw_feats, max_len, use_torch=False,max_templates=None, max_msas=None):
    padded_feats = {
        feat_name: pad(feat, max_len, use_torch=use_torch)
        for feat_name, feat in raw_feats.items()
        if feat_name not in UNPADDED_FEATS + RIGID_FEATS + TEMPLATE_FEATURES + MSA_FEATURES
    }

    # TEMPLATE_FEATURES (PADDED and UNPADDED)
    for feat_name in TEMPLATE_FEATURES:

        if feat_name in raw_feats:
            if max_templates:
                padded_feat = pad(raw_feats[feat_name], max_templates, pad_idx=0,use_torch=use_torch)
            else:
                padded_feat = raw_feats[feat_name]

            if feat_name not in UNPADDED_FEATS:
                padded_feat = pad(padded_feat, max_len, pad_idx=1,use_torch=use_torch)

            padded_feats[feat_name] = padded_feat
        # TEMPLATE_FEATURES (PADDED and UNPADDED)
        
    for feat_name in MSA_FEATURES:

        if feat_name in raw_feats:
            if max_msas is not None:
                padded_feat = pad(raw_feats[feat_name], max_msas, pad_idx=0,use_torch=use_torch)
            else:
                padded_feat = raw_feats[feat_name]

            if feat_name not in UNPADDED_FEATS:
                padded_feat = pad(padded_feat, max_len, pad_idx=1,use_torch=use_torch)

            if feat_name=='msa':
                padded_feat[raw_feats[feat_name].shape[0]:] = MSA_PAD_TOKEN
                padded_feat[:,raw_feats[feat_name].shape[1]:] = MSA_PAD_TOKEN
            
            padded_feats[feat_name] = padded_feat    
        
    # UNPADDED_FEATS(NO TEMPLATE_FEATURES)
    for feat_name in UNPADDED_FEATS:
        if feat_name in raw_feats:
            if feat_name not in TEMPLATE_FEATURES:
                    padded_feats[feat_name] = raw_feats[feat_name]
    for feat_name in RIGID_FEATS:
        if feat_name in raw_feats:

            padded_feats[feat_name] = pad_rigid(raw_feats[feat_name], max_len)

    # pad from padded feature, mean it's a second pad for pair
    for feat_name in PAIR_FEATS:
        if feat_name in padded_feats:
            padded_feats[feat_name] = pad(padded_feats[feat_name], max_len, pad_idx=1,use_torch=use_torch)
    return padded_feats

def pad_rigid(rigid: torch.tensor, max_len: int):
    num_rigids = rigid.shape[0]
    pad_amt = max_len - num_rigids
    if pad_amt<=0:
        return rigid
    shape = (pad_amt,) + tuple(rigid.shape[1:-1])
    pad_rigid = rigid_utils.Rigid.identity(
        shape, dtype=rigid.dtype, device=rigid.device, requires_grad=False)
    return torch.cat([rigid, pad_rigid.to_tensor_7()], dim=0)

def pad(x, max_len: int, pad_idx=0, use_torch=False, reverse=False):
    """Right pads dimension of numpy array.

    Args:
        x: numpy like array to pad.
        max_len: desired length after padding
        pad_idx: dimension to pad.
        use_torch: use torch padding method instead of numpy.

    Returns:
        x with its pad_idx dimension padded to max_len
    """
    # Pad only the residue dimension.
    seq_len = x.shape[pad_idx]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        return x
    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)
    if use_torch:
        pad_widths.reverse()
        return torch.nn.functional.pad(x, [i for pad_width in pad_widths for i in pad_width])
    return np.pad(x, pad_widths)

def crop(x: torch.Tensor, crop_len: int, crop_start : int, crop_idx :int = 0):
    slices = [slice(0,length) for length in x.shape]
    crop_start = max(min(crop_start,slices[crop_idx].stop-crop_len),0)
    crop_end = min(slices[crop_idx].stop,crop_start+crop_len)
    slices[crop_idx] = slice(crop_start,crop_end)
    x = x[slices]
    return x


def crop_feats(
    raw_feats, crop_size, max_templates=None,subsample_templates=False
):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    # We want each ensemble to be cropped the same way
    device = raw_feats["aatype"].device

    seq_length = raw_feats["aatype"].shape[0]

    if "template_mask" in raw_feats:
        num_templates = sum(raw_feats["template_mask"] == 1)
    else:
        num_templates = 0

    # No need to subsample templates if there aren't any
    subsample_templates = subsample_templates and num_templates

    num_res_crop_size = min(int(seq_length), crop_size)

    def _randint(lower, upper):
        return int(torch.randint(
                lower,
                upper + 1,
                (1,),
                device=device,
        )[0])

    if subsample_templates:
        templates_crop_start = _randint(0, num_templates)
        templates_select_indices = torch.randperm(
            num_templates, device=device
        )
        num_templates_crop_size = min(
            num_templates - templates_crop_start, max_templates
        )
        templates_select_indices = templates_select_indices[templates_crop_start:templates_crop_start+num_templates_crop_size]
    else:
        num_templates = min(num_templates,max_templates) if max_templates else num_templates
        templates_select_indices = list(range(num_templates))
    n = seq_length - num_res_crop_size
    x = _randint(0, n)
    right_anchor = n - x

    num_res_crop_start = _randint(0, right_anchor)

    cropped_feats = {
        feat_name: crop(feat, crop_len = num_res_crop_size, crop_start = num_res_crop_start,crop_idx = 0)
        for feat_name, feat in raw_feats.items()
        if feat_name not in UNPADDED_FEATS + TEMPLATE_FEATURES
    }
    # TEMPLATE_FEATURES (PADDED and UNPADDED)
    for feat_name in TEMPLATE_FEATURES:
        if feat_name in raw_feats:
            template_feat = raw_feats[feat_name]
            if subsample_templates:
                template_feat = template_feat[templates_select_indices]
            if feat_name not in UNPADDED_FEATS:
                template_feat = crop(template_feat, crop_len = num_res_crop_size, crop_start = num_res_crop_start,crop_idx = 1)
            
            cropped_feats[feat_name] = template_feat

    # UNPADDED_FEATS(NO TEMPLATE_FEATURES)
    for feat_name in UNPADDED_FEATS:
        if feat_name not in TEMPLATE_FEATURES:
            if feat_name in raw_feats:
                cropped_feats[feat_name] = raw_feats[feat_name]

    for feat_name in PAIR_FEATS:
        if feat_name in cropped_feats:
            cropped_feats[feat_name] = crop(cropped_feats[feat_name], crop_len = num_res_crop_size, crop_start = num_res_crop_start,crop_idx = 1)

    
    return cropped_feats



# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions
def parse_a3m(filename):

    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    #print(filename)

    if filename.split('.')[-1] == 'gz':
        fp = gzip.open(filename, 'rt')
    else:
        fp = open(filename, 'r')

    # read file line by line
    for line in fp:

        # skip labels
        if line[0] == '>':
            continue

        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c=='-' else 1 for c in line])
        i = np.zeros((L))

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a==1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos,num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)
        if len(msa) == 10000:
            break

    # convert letters into numbers
    alphabet = list("ARNDCQEGHILKMFPSTWYV-")
    encoding = np.array(alphabet, dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for letter, enc in zip(alphabet, encoding):
        res_cat = residue_constants.restype_order_with_x.get(
            letter, residue_constants.restype_num)
        msa[msa == enc] = res_cat

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)

    return msa,ins

def write_checkpoint(
        ckpt_path: str,
        model,
        conf,
        optimizer,
        epoch,
        step,
        ema = None,
        dataset = None,
        logger=None,
        use_torch=True,
        overwrite=True,
    ):
    """Serialize experiment state and stats to a pickle file.

    Args:
        ckpt_path: Path to save checkpoint.
        conf: Experiment configuration.
        optimizer: Optimizer state dict.
        epoch: Training epoch at time of checkpoint.
        step: Training steps at time of checkpoint.
        exp_state: Experiment state to be written to pickle.
        preds: Model predictions to be written as part of checkpoint.
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    if overwrite:
        for fname in os.listdir(ckpt_dir):
            if '.pkl' in fname or '.pth' in fname or '.csv' in fname:
                os.remove(os.path.join(ckpt_dir, fname))
    if logger is not None:
        logger.info(f'Serializing experiment state to {ckpt_path}')
    else:
        print(f'Serializing experiment state to {ckpt_path}')
    write_pkl(
        ckpt_path,
        {
            'model': model,
            'ema': ema,
            'conf': conf,
            'optimizer': optimizer,
            'epoch': epoch,
            'step': step,
            'dataset_csvs' : dataset.dataset_csvs if dataset is not None else None,
        },
        use_torch=use_torch)

def concat_np_features(
        np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool, axis=0):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=axis)
    return combined_dict

def length_batching(
        np_dicts: List[Dict[str, np.ndarray]],
        max_squared_res: int,
    ):
    get_len = lambda x: x['res_mask'].shape[0]
    dicts_by_length = [(get_len(x), x) for x in np_dicts]
    random.shuffle(dicts_by_length)
    length_sorted = sorted(dicts_by_length, key=lambda x: x[0], reverse=True)
    max_len = length_sorted[0][0]
    max_batch_examples = int(max_squared_res // max_len**2)
    # pad_example = lambda x: pad_feats(x, max_len)
    # padded_batch = [
    #     pad_example(x) for (_, x) in length_sorted[:max_batch_examples]]

    return torch.utils.data.default_collate(np_dicts[:max_batch_examples])

def crop_and_pad(raw_prots,data_conf):
    processed_prots = []
    if data_conf.crop:
        fix_size = min(max([prot["aatype"].shape[0] for prot in raw_prots]),data_conf.crop_size)
    else:
        fix_size = max([prot["aatype"].shape[0] for prot in raw_prots])
    for prot in raw_prots:
        processed_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), prot)
        # print("init feature : ",{k:v.shape for k,v in init_feats.items()})
        # padded_feats = pad_feats(init_feats, fix_size,use_torch=True)
        # print("after padding : ",{k:v.shape for k,v in padded_feats.items()})
        processed_feats = crop_feats(processed_feats, 
            crop_size =  data_conf.crop_size if data_conf.crop else fix_size,
            max_templates = data_conf.max_templates,
            subsample_templates = data_conf.subsample_templates)
        processed_feats = pad_feats(processed_feats, fix_size,use_torch=True,max_templates=data_conf.max_templates)
        # print("after cropping : ",{k:v.shape for k,v in cropped_feats.items()})
        processed_prots.append(processed_feats)
    return processed_prots


def create_data_loader(
        dataset: data.Dataset,
        data_conf,
        shuffle,
        sampler=None,
        num_workers=0,
        drop_last=False,
        prefetch_factor=2
        ):
    """Creates a data loader with jax compatible data structures."""
    collate_fn_list = []

    _crop_and_pad = functools.partial(crop_and_pad,data_conf=data_conf)
    _flatten = lambda x: [item for sublist in x for item in sublist]

    # if dataset.is_training:
    #     collate_fn_list.append(_flatten)
    #     collate_fn_list.append(_crop_and_pad)
    # else:
    #     # (feature,pdb_name) will be returned
    collate_fn_list.append(lambda x : [(i,j) for i,j in zip(_crop_and_pad(raw_prots=[ prot[0] for prot in _flatten(x)]),[ prot[1] for prot in _flatten(x)])])
    collate_fn_list.append(torch.utils.data._utils.collate.default_collate)

    collate_fn = lambda x : functools.reduce(lambda x,fn: fn(x),collate_fn_list,x)
    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers == 0 else prefetch_factor
    
    return data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        # Need fork https://github.com/facebookresearch/hydra/issues/964
        multiprocessing_context='fork' if num_workers != 0 else None,
        # set deterministic for worker
        worker_init_fn=worker_init_function
        )

def parse_chain_feats(chain_feats, scale_factor=1.):

    if chain_feats["chain_index"].dtype.kind in {'U', 'S'}:
        chain_feats["chain_index"] = np.vectorize(chain_str_to_int)(chain_feats["chain_index"])
   
    ca_idx = residue_constants.atom_order['CA']
    chain_feats['bb_mask'] = chain_feats['atom_mask'][:, ca_idx]
    bb_pos = chain_feats['atom_positions'][:, ca_idx]
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-5)
    centered_pos = chain_feats['atom_positions'] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    chain_feats['atom_positions'] = scaled_pos * chain_feats['atom_mask'][..., None]
    chain_feats['bb_positions'] = chain_feats['atom_positions'][:, ca_idx]
    chain_feats['b_factors'] = chain_feats['b_factors'][:, ca_idx]
    return chain_feats

def rigid_frames_from_all_atom(all_atom_pos):
    rigid_atom_pos = []
    for atom in ['C', 'CA', 'N']:
        atom_idx = residue_constants.atom_order[atom]
        atom_pos = all_atom_pos[..., atom_idx, :]
        rigid_atom_pos.append(atom_pos)
    return rigid_utils.Rigid.from_3_points(*rigid_atom_pos)

def pad_pdb_feats(raw_feats, max_len):
    padded_feats = {
        feat_name: pad(feat, max_len)
        for feat_name, feat in raw_feats.items() if feat_name not in UNPADDED_FEATS
    }
    for feat_name in UNPADDED_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = raw_feats[feat_name]
    return padded_feats

def binder_task_preprocess(feats,target_chain, crop_length, hotspot_nums):
    '''
        this function prepare all needed feature for binder design task:
        1. select neighoured continous target segment if need to crop
        2. sample hotspot residues
        3. recenter all coordinates by target  
    '''
    prot_length = feats['aatype'].shape[0]
    target_mask = (feats["chain_index"]==target_chain)
    feats["fixed_mask"] = target_mask
    cb_coordinates = generate_Cbeta(feats["all_atom_positions"][...,0,:],feats["all_atom_positions"][...,1,:],feats["all_atom_positions"][...,2,:],use_torch=True)
    cb_distance = torch.linalg.norm(cb_coordinates[:,None]-cb_coordinates[None],dim=-1)
    cb_distance[target_mask[None] | ~target_mask[:,None]] = torch.iinfo(torch.int).max
    target_contact_residue = (cb_distance<20).sum(axis=1).float()
    target_contact_residue[~feats['res_mask']] = 0
    hotspot_residue = (cb_distance<5).any(axis=1)
    hotspot_residue[~feats['res_mask']] = 0
    
    # only keep continuos reigion(at least 8 residues)
    for i in range(prot_length):
        end_idx = i
        while end_idx<prot_length and target_contact_residue[end_idx] >0:
            end_idx+=1
        if end_idx-i < 8:
            target_contact_residue[i:end_idx] = 0 
        else:
            target_contact_residue[i:end_idx] = target_contact_residue[i:end_idx].mean()
    
    if crop_length < prot_length:
        target_keep_length = crop_length - sum(~target_mask)
        if target_keep_length < sum(target_contact_residue >0):
            target_keep_index = torch.argsort(target_contact_residue,descending=True)[:target_keep_length]
        else:
            target_keep_index = torch.where(target_contact_residue >0)[0]
        keep_index = torch.cat((target_keep_index,torch.where(~target_mask)[0]))
        mask = torch.zeros(prot_length, dtype=torch.bool)
        mask[keep_index] = 1
    else:
        mask = torch.ones(prot_length, dtype=torch.bool)
    
    hotspot_residue = hotspot_residue[mask]
    hotspot_index = torch.where(hotspot_residue)[0]
    hotspot_index = hotspot_index[torch.randperm(hotspot_index.size(0))[:hotspot_nums]]
    hotspot_mask = torch.zeros(prot_length, dtype=torch.bool)
    hotspot_mask[hotspot_index] = 1
        
    for k,v in feats.items():
        feats[k] = v[mask]
    feats['hotspot'] = torch.nn.functional.one_hot(hotspot_mask.long(),2).float()
    
    target_mask = (feats["chain_index"]==target_chain)
    center_of_target = feats["all_atom_positions"][target_mask][:,1,:].mean(dim=0)
    feats["all_atom_positions"] = feats["all_atom_positions"] - center_of_target
    
    return feats 
    

def process_chain(chain: Chain, chain_id: str) -> Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.

    Forked from alphafold.common.protein.from_pdb_string

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.

    Took out lines 110-112 since that would mess up CDR numbering.

    Args:
        chain: Instance of Biopython's chain class.

    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num)
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.
            res_b_factors[residue_constants.atom_order[atom.name]
                          ] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors))

def rigid_transform_3D(A, B, verbose=False):
    # Transforms A to look like B
    # https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflection_detected = False
    if np.linalg.det(R) < 0:
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
        reflection_detected = True

    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t

    return optimal_A.T, R, t, reflection_detected

def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram

def quat_to_rotvec(quat, eps=1e-6):
    # w > 0 to ensure 0 <= angle <= pi
    flip = (quat[..., :1] < 0).float()
    quat = (-1 * quat) * flip + (1 - flip) * quat

    angle = 2 * torch.atan2(
        torch.linalg.norm(quat[..., 1:], dim=-1),
        quat[..., 0]
    )

    angle2 = angle * angle
    small_angle_scales = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_angle_scales = angle / torch.sin(angle / 2 + eps)

    small_angles = (angle <= 1e-3).float()
    rot_vec_scale = small_angle_scales * small_angles + (1 - small_angles) * large_angle_scales
    rot_vec = rot_vec_scale[..., None] * quat[..., 1:]
    return rot_vec

def quat_to_rotmat(quat, eps=1e-6):
    rot_vec = quat_to_rotvec(quat, eps)
    return so3_utils.Exp(rot_vec)

def generate_Cbeta(N,Ca,C,use_torch=False):
    # recreate Cb given N,Ca,C
    b = Ca - N 
    c = C - Ca
    if use_torch:
        a = torch.cross(b, c, dim=-1)
    else:
        a = np.cross(b, c, axis=-1)
    Cb = -0.57910144*a + 0.5689693*b - 0.5441217*c + Ca

    return Cb

def construct_block_adj_matrix(sstruct, xyz, cutoff=6, include_loops=False):

    L = xyz.shape[0]

    # three anchor atoms
    N = xyz[:, 0]
    Ca = xyz[:, 1]
    C = xyz[:, 2]

    Cb = generate_Cbeta(N, Ca, C)

    dist = np.linalg.norm(Cb[:, None] - Cb[None, :], axis=-1)  # [L,L]
    dist[np.isnan(dist)] = 999.9

    np.fill_diagonal(dist, 999.9)

    segments = []

    begin = -1
    end = -1

    for i in range(sstruct.shape[0]):
        # Starting edge case
        if i == 0:
            begin = 0
            continue

        if not sstruct[i] == sstruct[i - 1]:
            end = i
            segments.append((sstruct[i - 1], begin, end))

            begin = i

    # Ending edge case: last segment is length one
    if not end == sstruct.shape[0]:
        segments.append((sstruct[-1], begin, sstruct.shape[0]))

    block_adj = np.zeros_like(dist)
    for i in range(len(segments)):
        curr_segment = segments[i]

        if curr_segment[0] == 2 and not include_loops:
            continue

        begin_i = curr_segment[1]
        end_i = curr_segment[2]
        for j in range(i + 1, len(segments)):
            j_segment = segments[j]

            if j_segment[0] == 2 and not include_loops:
                continue

            begin_j = j_segment[1]
            end_j = j_segment[2]

            if np.any(dist[begin_i:end_i, begin_j:end_j] < cutoff):
                # Matrix is symmetic
                block_adj[begin_i:end_i, begin_j:end_j] = np.ones((end_i - begin_i, end_j - begin_j))
                block_adj[begin_j:end_j, begin_i:end_i] = np.ones((end_j - begin_j, end_i - begin_i))
    return block_adj

def save_fasta(
        pred_seqs,
        seq_names,
        file_path,
    ):
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w') as f:
        for x,y in zip(seq_names, pred_seqs):
            f.write(f'>{x}\n{y}\n')

# set seed at main fucntion and each dataloader worker
def seed_everything(seed = None) -> int:

    log = logging.getLogger(__name__)

    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    if seed is None:
        seed = random.randint(min_seed_value, max_seed_value)
        logging.info(f"No seed found, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        logging.warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = random.randint(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    log.info(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed

def worker_init_function(worker_id: int) -> None:  # pragma: no cover
    """The worker_init_fn that Lightning automatically adds to your dataloader if you previously set the seed with
    ``seed_everything(seed, workers=True)``.

    See also the PyTorch documentation on
    `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
    """
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    global_rank = 0
    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    logging.debug(
        f"Initializing random number generators of process {global_rank} worker {worker_id} with base seed {base_seed}"
    )
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)

def center_zero(pos: torch.Tensor, batch_indexes: torch.LongTensor) -> tuple[torch.Tensor,torch.Tensor]:
    """
    Move the molecule center to zero for sparse position tensors.

    Args:
        pos: [N, 3] batch positions of atoms in the molecule in sparse batch format.
        batch_indexes: [N] batch index for each atom in sparse batch format.

    Returns:
        pos: [N, 3] zero-centered batch positions of atoms in the molecule in sparse batch format.
    """
    assert len(pos.shape) == 2 and pos.shape[-1] == 3, "pos must have shape [N, 3]"

    means = scatter(pos, batch_indexes, dim=0, reduce="mean")
    return pos - means[batch_indexes], means[batch_indexes]

@torch.no_grad()
def align_structures(
    batch_positions: torch.Tensor,
    batch_indices: torch.Tensor,
    reference_positions: torch.Tensor,
    broadcast_reference: bool = False,
):
    """
    Align structures in a ChemGraph batch to a reference, e.g. for RMSD computation. This uses the
    sparse formulation of pytorch geometric. If the ChemGraph is composed of a single system, then
    the reference can be given as a single structure and broadcasted. Returns the structure
    coordinates shifted to the geometric center and the batch structures rotated to match the
    reference structures. Uses the Kabsch algorithm (see e.g. [kabsch_align1]_). No permutation of
    atoms is carried out.

    Args:
        batch_positions (Tensor): Batch of structures (e.g. from ChemGraph) which should be aligned
          to a reference.
        batch_indices (Tensor): Index tensor mapping each node / atom in batch to the respective
          system (e.g. batch attribute of ChemGraph batch).
        reference_positions (Tensor): Reference structure. Can either be a batch of structures or a
          single structure. In the second case, broadcasting is possible if the input batch is
          composed exclusively of this structure.
        broadcast_reference (bool, optional): If reference batch contains only a single structure,
          broadcast this structure to match the ChemGraph batch. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing the centered positions of batch
          structures rotated into the reference and the centered reference batch.

    References
    ----------
    .. [kabsch_align1] Lawrence, Bernal, Witzgall:
       A purely algebraic justification of the Kabsch-Umeyama algorithm.
       Journal of research of the National Institute of Standards and Technology, 124, 1. 2019.
    """
    # Minimize || Q @ R.T - P ||, which is the same as || Q - P @ R ||
    # batch_positions     -> P [BN x 3]
    # reference_positions -> Q [B / BN x 3]

    if batch_positions.shape[0] != reference_positions.shape[0]:
        if broadcast_reference:
            # Get number of systems in batch and broadcast reference structure.
            # This assumes, all systems in the current batch correspond to the reference system.
            # Typically always the case during evaluation.
            num_molecules = int(torch.max(batch_indices) + 1)
            reference_positions = reference_positions.repeat(num_molecules, 1)
        else:
            raise ValueError("Mismatch in batch dimensions.")

    # Center structures at origin (takes care of translation alignment)
    batch_positions, batch_center = center_zero(batch_positions, batch_indices)
    reference_positions, reference_center = center_zero(reference_positions, batch_indices)

    # Compute covariance matrix for optimal rotation (Q.T @ P) -> [B x 3 x 3].
    cov = scatter_add(
        batch_positions[:, None, :] * reference_positions[:, :, None], batch_indices, dim=0
    )

    # Perform singular value decomposition. (all [B x 3 x 3])
    u, _, v_t = torch.linalg.svd(cov)
    # Convenience transposes.
    u_t = u.transpose(1, 2)
    v = v_t.transpose(1, 2)

    # Compute rotation matrix correction for ensuring right-handed coordinate system
    # For comparison with other sources: det(AB) = det(A)*det(B) and det(A) = det(A.T)
    sign_correction = torch.sign(torch.linalg.det(torch.bmm(v, u_t)))
    # Correct transpose of U: diag(1, 1, sign_correction) @ U.T
    u_t[:, 2, :] = u_t[:, 2, :] * sign_correction[:, None]

    # Compute optimal rotation matrix (R = V @ diag(1, 1, sign_correction) @ U.T).
    rotation_matrices = torch.bmm(v, u_t)

    # Rotate batch positions P to optimal alignment with Q (P @ R)
    batch_positions_rotated = torch.bmm(
        batch_positions[:, None, :],
        rotation_matrices[batch_indices],
    ).squeeze(1)

    return batch_positions_rotated + batch_center, reference_positions + reference_center, rotation_matrices

def batch_align_structures(pos_1, pos_2, mask=None):
    is_numpy = isinstance(pos_1, np.ndarray)
    if is_numpy:
        pos_1 = torch.from_numpy(pos_1)
        pos_2 = torch.from_numpy(pos_2)
        if mask is not None:
            mask = torch.from_numpy(mask)
        else:
            mask = torch.ones(pos_1.shape[:-1], dtype=torch.bool)
    
    if pos_1.shape != pos_2.shape:
        raise ValueError('pos_1 and pos_2 must have the same shape.')
    if pos_1.ndim != 3:
        raise ValueError(f'Expected inputs to have shape [B, N, 3]')
    num_batch = pos_1.shape[0]
    device = pos_1.device
    batch_indices = (
        torch.ones(*pos_1.shape[:2], device=device, dtype=torch.int64) 
        * torch.arange(num_batch, device=device)[:, None]
    )
    flat_pos_1 = pos_1.reshape(-1, 3)
    flat_pos_2 = pos_2.reshape(-1, 3)
    flat_batch_indices = batch_indices.reshape(-1)
    if mask is None:
        # do not change the center of mass
        aligned_pos_1, _, align_rots = align_structures(
            flat_pos_1, flat_batch_indices, flat_pos_2)
    else:
        flat_mask = mask.reshape(-1).bool()
        _, _, align_rots = align_structures(
            flat_pos_1[flat_mask],
            flat_batch_indices[flat_mask],
            flat_pos_2[flat_mask]
        )
        aligned_pos_1 = torch.bmm(
            pos_1,
            align_rots
        )
    aligned_pos_1 = aligned_pos_1.reshape(num_batch, -1, 3)
    
    if is_numpy:
        aligned_pos_1 = aligned_pos_1.numpy()
        pos_2 = pos_2.numpy()
        align_rots = align_rots.numpy()
    return aligned_pos_1, pos_2, align_rots

def adjust_oxygen_pos(
    atom_37: torch.Tensor, pos_is_known = None
) -> torch.Tensor:
    """
    Imputes the position of the oxygen atom on the backbone by using adjacent frame information.
    Specifically, we say that the oxygen atom is in the plane created by the Calpha and C from the
    current frame and the nitrogen of the next frame. The oxygen is then placed c_o_bond_length Angstrom
    away from the C in the current frame in the direction away from the Ca-C-N triangle.

    For cases where the next frame is not available, for example we are at the C-terminus or the
    next frame is not available in the data then we place the oxygen in the same plane as the
    N-Ca-C of the current frame and pointing in the same direction as the average of the
    Ca->C and Ca->N vectors.

    Args:
        atom_37 (torch.Tensor): (N, 37, 3) tensor of positions of the backbone atoms in atom_37 ordering
                                which is ['N', 'CA', 'C', 'CB', 'O', ...]
        pos_is_known (torch.Tensor): (N,) mask for known residues.
    """

    is_batched = len(atom_37.shape) == 4  # Check if input is batched
    if not is_batched:
        atom_37 = atom_37[None, ...]
        pos_is_known = pos_is_known[None, ...]
    assert atom_37.shape[-2:] == (37, 3)
    B,N = atom_37.shape[:2]

    # Get vectors to Carbonly from Carbon alpha and N of next residue. (N-1, 3)
    # Note that the (N,) ordering is from N-terminal to C-terminal.

    # Calpha to carbonyl both in the current frame.
    calpha_to_carbonyl: torch.Tensor = (atom_37[:,:-1, 2, :] - atom_37[:,:-1, 1, :]) / (
        torch.norm(atom_37[:,:-1, 2, :] - atom_37[:,:-1, 1, :], keepdim=True, dim=1) + 1e-7
    )
    # For masked positions, they are all 0 and so we add 1e-7 to avoid division by 0.
    # The positions are in Angstroms and so are on the order ~1 so 1e-7 is an insignificant change.

    # Nitrogen of the next frame to carbonyl of the current frame.
    nitrogen_to_carbonyl: torch.Tensor = (atom_37[:,:-1, 2, :] - atom_37[:,1:, 0, :]) / (
        torch.norm(atom_37[:,:-1, 2, :] - atom_37[:,1:, 0, :], keepdim=True, dim=1) + 1e-7
    )

    carbonyl_to_oxygen: torch.Tensor = calpha_to_carbonyl + nitrogen_to_carbonyl  # (N-1, 3)
    carbonyl_to_oxygen = carbonyl_to_oxygen / (
        torch.norm(carbonyl_to_oxygen, dim=2, keepdim=True) + 1e-7
    )

    atom_37[:,:-1, 4, :] = atom_37[:,:-1, 2, :] + carbonyl_to_oxygen * 1.23

    # Now we deal with frames for which there is no next frame available.

    # Calpha to carbonyl both in the current frame. (N, 3)
    calpha_to_carbonyl_term: torch.Tensor = (atom_37[:,:, 2, :] - atom_37[:,:, 1, :]) / (
        torch.norm(atom_37[:,:, 2, :] - atom_37[:,:, 1, :], keepdim=True, dim=2) + 1e-7
    )
    # Calpha to nitrogen both in the current frame. (N, 3)
    calpha_to_nitrogen_term: torch.Tensor = (atom_37[:,:, 0, :] - atom_37[:,:, 1, :]) / (
        torch.norm(atom_37[:,:, 0, :] - atom_37[:,:, 1, :], keepdim=True, dim=1) + 1e-7
    )
    carbonyl_to_oxygen_term: torch.Tensor = (
        calpha_to_carbonyl_term + calpha_to_nitrogen_term
    )  # (N, 3)
    carbonyl_to_oxygen_term = carbonyl_to_oxygen_term / (
        torch.norm(carbonyl_to_oxygen_term, dim=2, keepdim=True) + 1e-7
    )

    # Create a mask that is 1 when the next residue is not available either
    # due to this frame being the C-terminus or the next residue is not
    # known due to pos_is_known being false.

    if pos_is_known is None:
        pos_is_known = torch.ones((B,N), dtype=torch.int64, device=atom_37.device)

    next_res_gone: torch.Tensor = ~pos_is_known.bool()  # (N,)
    next_res_gone = torch.cat(
        [next_res_gone, torch.ones((B,1), device=pos_is_known.device).bool()], dim=1
    )  # (N+1, )
    next_res_gone = next_res_gone[:,1:]  # (N,)

    atom_37[:,:, 4, :][next_res_gone==1] = (
        atom_37[:,:, 2, :][next_res_gone==1]
        + carbonyl_to_oxygen_term[next_res_gone==1] * 1.23
    )
    if not is_batched:
        atom_37 = atom_37[0]
    return atom_37



def create_rigid(rots, trans):
    rots = rigid_utils.Rotation(rot_mats=rots)
    return rigid_utils.Rigid(rots=rots, trans=trans)
