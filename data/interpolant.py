import torch
from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
from data import all_atom
import copy
from scipy.optimize import linear_sum_assignment
import numpy as np
from openfold.utils import rigid_utils


def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self.igso3 = so3_utils.SampleIGSO3(1000, torch.linspace(0.1, 1.5, 1000), cache_dir='.cache')

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
       t = torch.rand(num_batch, device=self._device)
       return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask):
        """
        trans_t = (1 - t) * noise + t * trans_1
        """
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]
    
    def _batch_ot(self, trans_0, trans_1, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        ) 
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)
        
        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.move_to_np(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = t

        # Apply corruptions
        trans_t = self._corrupt_trans(trans_1, t, res_mask)
        noisy_batch['trans_t'] = trans_t

        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)
        noisy_batch['rotmats_t'] = rotmats_t
        return noisy_batch
    
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        trans_vf = (trans_1 - trans_t) / t
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / t
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)
    
    def sample(
            self,
            batch,
            model,
        ):
        device = next(model.parameters()).device
        res_mask = batch['res_mask']
        num_batch, num_res = res_mask.shape

        trans_0 = _centered_gaussian(
            num_batch, num_res, device) * du.NM_TO_ANG_SCALE
        rotmats_0 = _uniform_so3(num_batch, num_res, device)
        

        # Set-up time
        ts = np.linspace(self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)[::-1]
        d_t = ts[0] - ts[1]

        prot_traj = [(trans_0, rotmats_0)]
        seq_traj = [None]
        clean_traj = []
        model_out = None
        for t in ts:
            
            if not self._cfg.self_condition:
                model_out = None
            
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            batch['rigids_t'] = rigid_utils.Rigid(rigid_utils.Rotation(rotmats_t_1),trans_t_1).to_tensor_7()
            t = torch.ones((num_batch), device=device) * t
            batch['t'] = t
            
            with torch.no_grad():
                model_out = model(batch,self_condition=model_out)
                
            if t > self._cfg.min_t:
                # Run model.

                # Process model output.
                pred_trans_1 = model_out['pred_trans']
                pred_rotmats_1 = model_out['pred_rotmats']
                clean_traj.append(
                    (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
                )

                # Take reverse step
                trans_t_2 = self._trans_euler_step(
                    d_t, t, pred_trans_1, trans_t_1)
                rotmats_t_2 = self._rots_euler_step(
                    d_t, t, pred_rotmats_1, rotmats_t_1)
                prot_traj.append((trans_t_2, rotmats_t_2))
            else:
                pred_trans_1 = model_out['pred_trans']
                pred_rotmats_1 = model_out['pred_rotmats']
                clean_traj.append(
                    (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
                )
                prot_traj.append((pred_trans_1, pred_rotmats_1))
                
            if 'pred_aatype' in model_out:
                seq_traj.append(du.move_to_np(model_out['pred_aatype']))
            else:
                seq_traj.append(None) 
        
        # Convert trajectories to atom37.
        flip = lambda x: np.flip(np.stack(x), (0,))
        atom37_traj = flip([du.move_to_np(prot_coords) for prot_coords in all_atom.transrot_to_atom37(prot_traj, res_mask)])
        clean_atom37_traj = flip([du.move_to_np(prot_coords) for prot_coords in all_atom.transrot_to_atom37(clean_traj, res_mask)])
        return {
            'prot_traj': atom37_traj,
            'mask_traj': np.any(atom37_traj,axis=-1),
            'prot_0_traj':clean_atom37_traj,
            'seq_traj': flip(seq_traj)[:,None],
        }
