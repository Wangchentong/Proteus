""" Metrics. """
import mdtraj as md
import numpy as np
import torch
from analysis import utils as au
from data import utils as du
from openfold.np import residue_constants
from openfold.np.relax import amber_minimize
import tree
from tmtools import tm_align
from openfold.utils.tensor_utils import (
    permute_final_dims,
)
CA_IDX = residue_constants.atom_order['CA']

INTER_VIOLATION_METRICS = [
    'bonds_c_n_loss_mean',
    'angles_ca_c_n_loss_mean',
    'clashes_mean_loss',
]

SHAPE_METRICS = [
    'coil_percent',
    'helix_percent',
    'strand_percent',
    'radius_of_gyration'
]

CA_VIOLATION_METRICS = [
    'ca_ca_bond_dev',
    'ca_ca_valid_percent',
    'ca_steric_clash_percent',
    'num_ca_steric_clashes',
]

EVAL_METRICS = [
    'tm_score', 
    'lddt',
    "rmsd"
]

ALL_METRICS = (
    INTER_VIOLATION_METRICS
    + SHAPE_METRICS
    + CA_VIOLATION_METRICS
    + EVAL_METRICS
)

def calc_tm_score(pos_1, pos_2,):
    seq_1 = 'A' * pos_1.shape[0]
    seq_2 = 'A' * pos_2.shape[0]
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2 

def calc_perplexity(pred, labels, mask):
    one_hot_labels = np.eye(pred.shape[-1])[labels]
    true_probs = np.sum(pred * one_hot_labels, axis=-1)
    ce = -np.log(true_probs + 1e-5)
    per_res_perplexity = np.exp(ce)
    return np.sum(per_res_perplexity * mask) / np.sum(mask)

def calc_mdtraj_metrics(pdb_path):
    traj = md.load(pdb_path)
    pdb_ss = md.compute_dssp(traj, simplified=True)
    pdb_coil_percent = np.mean(pdb_ss == 'C')
    pdb_helix_percent = np.mean(pdb_ss == 'H')
    pdb_strand_percent = np.mean(pdb_ss == 'E')
    pdb_ss_percent = pdb_helix_percent + pdb_strand_percent 
    pdb_rg = md.compute_rg(traj)[0]
    return {
        'non_coil_percent': pdb_ss_percent,
        'coil_percent': pdb_coil_percent,
        'helix_percent': pdb_helix_percent,
        'strand_percent': pdb_strand_percent,
        'radius_of_gyration': pdb_rg,
    }

def calc_aligned_rmsd(pos_1, pos_2):
    aligned_pos_1 = du.rigid_transform_3D(pos_1, pos_2)[0]
    return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))

def violation_metric(
        atom37_pos,
        atom37_mask,
        diffuse_mask,
    ):
    metrics_dict= {}
    atom37_diffuse_mask = diffuse_mask[..., None] * atom37_mask
    prot = au.create_full_prot(atom37_pos, atom37_diffuse_mask)
    violation_metrics = amber_minimize.get_violation_metrics(prot)
    struct_violations = violation_metrics['structural_violations']
    inter_violations = struct_violations['between_residues']
    for k in INTER_VIOLATION_METRICS: 
        metrics_dict[k] = float(inter_violations[k])
    return metrics_dict

def align_metric(
        gt_atom37_pos,
        atom37_pos,
        atom37_mask,
        diffuse_mask,
        metrics = ["tm_score","lddt","rmsd"]
    ):
    metrics_dict = {}
    res_mask = np.any(atom37_mask, axis=-1)
    bb_diffuse_mask = (diffuse_mask * res_mask).astype(bool)
    unpad_gt_scaffold_pos = gt_atom37_pos[..., CA_IDX, :][bb_diffuse_mask]
    unpad_pred_scaffold_pos = atom37_pos[..., CA_IDX, :][bb_diffuse_mask]
    if "tm_score" in metrics:
        _, tm_score = calc_tm_score(
            unpad_pred_scaffold_pos, unpad_gt_scaffold_pos)
        metrics_dict["tm_score"] = tm_score
    if "lddt" in metrics:
        metrics_dict["lddt"] = lddt_ca(
            all_atom_pred_pos = atom37_pos,
            all_atom_positions = gt_atom37_pos,
            all_atom_mask=atom37_mask
        )
    if "rmsd" in metrics:
        metrics_dict["rmsd"] = calc_aligned_rmsd(
            unpad_pred_scaffold_pos, unpad_gt_scaffold_pos)

    return metrics_dict

def lddt_ca(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = False,
) -> torch.Tensor:
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    return float(lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    ))

def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score
def ca_ca_distance(ca_pos, tol=0.1):
    ca_bond_dists = np.linalg.norm(
        ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
    ca_ca_dev = np.mean(np.abs(ca_bond_dists - residue_constants.ca_ca))
    ca_ca_valid = np.mean(ca_bond_dists < (residue_constants.ca_ca + tol))
    return ca_ca_dev, ca_ca_valid

def ca_ca_clashes(ca_pos, tol=1.5):
    ca_ca_dists2d = np.linalg.norm(
        ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=0) > 0)]
    clashes = inter_dists < tol
    return np.sum(clashes), np.mean(clashes) 

def protein_metrics(
        *,
        pdb_path,
        atom37_pos,
        gt_atom37_pos,
        atom_37_mask,
        diffuse_mask,
        dssp = True,
        violation = True,
        align_metrics = ["tm_score"],
    ):

    metrics_dict = {}
    if dssp:
        metrics_dict.update(calc_mdtraj_metrics(pdb_path))
    metrics_dict.update(
        align_metric(
            gt_atom37_pos = gt_atom37_pos,
            atom37_pos = atom37_pos,
            atom37_mask = atom_37_mask,
            diffuse_mask = diffuse_mask,
            metrics= align_metrics
        )
    )
    if violation:
        metrics_dict.update(
            violation_metric(atom37_pos,atom_37_mask,diffuse_mask)
        )
    return metrics_dict 
