"""Utilities for calculating all atom representations."""
import torch
from data import residue_constants
from openfold.utils import rigid_utils as ru
from data import data_transforms
from openfold.utils import feats
from data import utils as du


Rigid = ru.Rigid
Rotation = ru.Rotation

# Residue Constants from OpenFold/AlphaFold2.
IDEALIZED_POS37 = torch.tensor(residue_constants.restype_atom37_rigid_group_positions)
IDEALIZED_POS37_MASK = torch.any(IDEALIZED_POS37, axis=-1)
IDEALIZED_POS = torch.tensor(residue_constants.restype_atom14_rigid_group_positions)
DEFAULT_FRAMES = torch.tensor(residue_constants.restype_rigid_group_default_frame)
ATOM_MASK = torch.tensor(residue_constants.restype_atom14_mask)
GROUP_IDX = torch.tensor(residue_constants.restype_atom14_to_rigid_group)

def to_atom37(trans, rots):
    num_batch, num_res, _ = trans.shape
    final_atom37 = compute_backbone(
        du.create_rigid(rots, trans),
        torch.zeros(num_batch, num_res, 2, device=trans.device)
    )[0]
    return final_atom37

def torsion_angles_to_frames(
        r: Rigid,
        alpha: torch.Tensor,
        aatype: torch.Tensor,
    ):
    """Conversion method of torsion angles to frames provided the backbone.
    
    Args:
        r: Backbone rigid groups.
        alpha: Torsion angles.
        aatype: residue types.
    
    Returns:
        All 8 frames corresponding to each torsion frame.

    """
    # [*, N, 8, 4, 4]
    default_4x4 = DEFAULT_FRAMES[aatype, ...].to(r.device)

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat(
        [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
    )

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def prot_to_torsion_angles(aatype, atom37, atom37_mask):
    """Calculate torsion angle features from protein features."""
    prot_feats = {
        'aatype': aatype,
        'all_atom_positions': atom37,
        'all_atom_mask': atom37_mask,
    }
    torsion_angles_feats = data_transforms.atom37_to_torsion_angles(prot_feats)
    torsion_angles = torsion_angles_feats['torsion_angles_sin_cos']
    torsion_mask = torsion_angles_feats['torsion_angles_mask']
    return torsion_angles, torsion_mask 


def frames_to_atom14_pos(
        r: Rigid,
        aatype: torch.Tensor,
    ):
    """Convert frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:

    """

    # [*, N, 14]
    group_mask = GROUP_IDX[aatype, ...]

    # [*, N, 14, 8]
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=DEFAULT_FRAMES.shape[-3],
    ).to(r.device)

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 14, 1]
    frame_atom_mask = ATOM_MASK[aatype, ...].unsqueeze(-1).to(r.device)

    # [*, N, 14, 3]
    frame_null_pos = IDEALIZED_POS[aatype, ...].to(r.device)
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions


def compute_backbone(bb_rigids, psi_torsions):
    torsion_angles = torch.tile(
        psi_torsions[..., None, :],
        tuple([1 for _ in range(len(bb_rigids.shape))]) + (7, 1)
    ).to(bb_rigids.device)
    aatype = torch.zeros(bb_rigids.shape).long()
    # aatype = torch.zeros(bb_rigids.shape).long().to(bb_rigids.device)
    all_frames = feats.torsion_angles_to_frames(
        bb_rigids,
        torsion_angles,
        aatype,
        DEFAULT_FRAMES.to(bb_rigids.device))
    atom14_pos = frames_to_atom14_pos(
        all_frames,
        aatype)
    atom37_bb_pos = torch.zeros(bb_rigids.shape + (37, 3),device=bb_rigids.device)
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
    atom37_bb_pos[..., :3, :] = atom14_pos[..., :3, :]
    atom37_bb_pos[..., 3, :] = atom14_pos[..., 4, :] 
    atom37_bb_pos[..., 4, :] = atom14_pos[..., 3, :]
    atom37_mask = torch.any(atom37_bb_pos, axis=-1)
    return atom37_bb_pos, atom37_mask, aatype, atom14_pos

def compute_sidechain(bb_rigids, psi_torsions):
    torsion_angles = torch.tile(
        psi_torsions[..., None, :],
        tuple([1 for _ in range(len(bb_rigids.shape))]) + (7, 1)
    ).to(bb_rigids.device)
    aatype = torch.zeros(bb_rigids.shape).long()
    # aatype = torch.zeros(bb_rigids.shape).long().to(bb_rigids.device)
    all_frames = feats.torsion_angles_to_frames(
        bb_rigids,
        torsion_angles,
        aatype,
        DEFAULT_FRAMES.to(bb_rigids.device))
    atom14_pos = frames_to_atom14_pos(
        all_frames,
        aatype)
    atom37_bb_pos = torch.zeros(bb_rigids.shape + (37, 3))
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
    atom37_bb_pos[..., :3, :] = atom14_pos[..., :3, :]
    atom37_bb_pos[..., 3, :] = atom14_pos[..., 4, :] 
    atom37_bb_pos[..., 4, :] = atom14_pos[..., 3, :]
    atom37_mask = torch.any(atom37_bb_pos, axis=-1)
    return atom37_bb_pos, atom37_mask, aatype, atom14_pos


def calculate_neighbor_angles(R_ac, R_ab):
    """Calculate angles between atoms c <- a -> b.

    Parameters
    ----------
        R_ac: Tensor, shape = (N,3)
            Vector from atom a to c.
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.

    Returns
    -------
        angle_cab: Tensor, shape = (N,)
            Angle between atoms c <- a -> b.
    """
    # cos(alpha) = (u * v) / (|u|*|v|)
    x = torch.sum(R_ac * R_ab, dim=1)  # shape = (N,)
    # sin(alpha) = |u x v| / (|u|*|v|)
    y = torch.cross(R_ac, R_ab).norm(dim=-1)  # shape = (N,)
    # avoid that for y == (0,0,0) the gradient wrt. y becomes NaN
    y = torch.max(y, torch.tensor(1e-9))  
    angle = torch.atan2(y, x)
    return angle


def vector_projection(R_ab, P_n):
    """
    Project the vector R_ab onto a plane with normal vector P_n.

    Parameters
    ----------
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.
        P_n: Tensor, shape = (N,3)
            Normal vector of a plane onto which to project R_ab.

    Returns
    -------
        R_ab_proj: Tensor, shape = (N,3)
            Projected vector (orthogonal to P_n).
    """
    a_x_b = torch.sum(R_ab * P_n, dim=-1)
    b_x_b = torch.sum(P_n * P_n, dim=-1)
    return R_ab - (a_x_b / b_x_b)[:, None] * P_n




def transrot_to_atom37(transrot_traj, res_mask):
    atom37_traj = []
    res_mask = res_mask.detach().cpu()
    num_batch = res_mask.shape[0]
    for trans, rots in transrot_traj:
        rigids = du.create_rigid(rots, trans)
        atom37 = compute_backbone(
            rigids,
            torch.zeros(
                trans.shape[0],
                trans.shape[1],
                2,
                device=trans.device
            )
        )[0]
        atom37 = atom37.detach().cpu()
        batch_atom37 = []
        for i in range(num_batch):
            batch_atom37.append(
                du.adjust_oxygen_pos(atom37[i], res_mask[i])
            )
        atom37_traj.append(torch.stack(batch_atom37))
    return atom37_traj
