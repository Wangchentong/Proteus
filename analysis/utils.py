import numpy as np
import os
import re
from data import protein
from data import residue_constants
from scipy.spatial.transform import Rotation
from openfold.utils import rigid_utils
from typing import List

CA_IDX = residue_constants.atom_order['CA']
Rigid = rigid_utils.Rigid

def write_msa_to_a3m(
        file_path: str,
        msa: np.ndarray,
    ):
    file_path = file_path.replace('.pdb', '') + '.a3m'
    msa = np.clip(msa, 0, 20).astype(int) 
    with open(file_path, 'w') as f:
        if msa.ndim == 3:
            msa = msa[0]
        if msa.ndim == 2:
            for i, seq in enumerate(msa):
                f.write(f'>seq_{i}\n')
                f.write(''.join([residue_constants.restypes_with_x[x] for x in seq]))
                f.write('\n')
        else:
            raise ValueError(f'Invalid msa shape {msa.shape}')
        f.write('END')
    return file_path

def rigids_to_se3_vec(frame, scale_factor=1.0):
    trans = frame[:, 4:] * scale_factor
    rotvec = Rotation.from_quat(frame[:, :4]).as_rotvec()
    se3_vec = np.concatenate([rotvec, trans], axis=-1)
    return se3_vec
