import sys
import io as sysio

import numpy as np
import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta import *

    
def parse_extra_res_fa_param(param_fn):
    atmName_to_atmType = {}
    with open(param_fn) as fp:
        for line in fp:
            x = line.strip().split()
            if line.startswith('NAME'):
                lig_name = x[1]
            elif line.startswith('ATOM'):
                atmName_to_atmType[x[1]] = x[2]
    ligName_to_atms = {lig_name:atmName_to_atmType}
    return ligName_to_atms

def extract_coords_from_pose(pose, chain, atoms=['N','CA','C'], lig_to_atms = {}):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

  alpha_1 = list("ARNDCQEGHILKMFPSTWYV-atcgdryuJ")
  states = len(alpha_1)
  alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP',
             ' DA', ' DT', ' DC', ' DG',  '  A',  '  U',  '  C',  '  G',
             'LIG']
  
  aa_3_N = {a:n for n,a in enumerate(alpha_3)}
  aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
  
  lig_names = list(lig_to_atms.keys())   

  def N_to_AA(x):
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x)
    if x.ndim == 1: x = x[None]
    return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

  def rosetta_xyz_to_numpy(x):
    return np.array([x.x, x.y, x.z])

  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6

  # the pdb info struct
  info = pose.pdb_info()

  for resi in range(1, pose.size()+1):
    # for each residue
    ch = info.chain(resi)

    if ch == chain:
        residue = pose.residue(resi)

        resn = resi
        resa,resn = "",int(resn)-1

        if resn < min_resn: 
            min_resn = resn
        if resn > max_resn: 
            max_resn = resn

        xyz[resn] = {}
        xyz[resn][resa] = {}

        seq[resn] = {}
        seq[resn][resa] = residue.name3()

        # for each heavy atom
        for iatm in range(1, residue.nheavyatoms()+1):
            atom_name = residue.atom_name(iatm).strip()
            atom_xyz = rosetta_xyz_to_numpy( residue.xyz(iatm) )

            xyz[resn][resa][atom_name] = atom_xyz

  # convert to numpy arrays, fill in missing values
  seq_,xyz_,atype_ = [],[],[]
  is_lig_chain = False
  resn_to_ligName = {}
  for resn in range(min_resn,max_resn+1):
    if resn in seq:
      for k in sorted(seq[resn]): 
        res_name3 = seq[resn][k]
        if res_name3 in lig_names:
            resn_to_ligName[resn] = res_name3
            is_lig_chain = True
            seq_.append(aa_3_N['LIG']) ###GRL:hard-coding 29 ok?
        else:
            seq_.append(aa_3_N.get(res_name3,20))
    else: seq_.append(20)

    if is_lig_chain:
      #Get new atoms list just for the ligand as defined in the params file
      atoms = list(lig_to_atms[resn_to_ligName[resn]].keys())

    if resn in xyz:
      for k in sorted(xyz[resn]):
        for atom in atoms:
          if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
          else: xyz_.append(np.full(3,np.nan))
    else:
      for atom in atoms: xyz_.append(np.full(3,np.nan))

    if is_lig_chain:
        lig_atm_d = lig_to_atms[resn_to_ligName[resn]]
        for atom in atoms:
            atype_.append(lig_atm_d[atom])
    
  return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_)), np.array(atype_)

def parse_pose(pose, fixed_chain_list = [], ca_only=False):
  pdb_dict_list = []
  exist_chains = list(set([pose.pdb_info().chain(residue_id) for residue_id in range(1, pose.total_residue() + 1) ]))
  my_dict = {}
  s = 0
  concat_seq = ''
  for letter in exist_chains:
      xyz, seq, _ = extract_coords_from_pose(pose, atoms=['N','CA','C','O'] , chain=letter)
      if type(xyz) != str:
          concat_seq += seq[0]
          my_dict['seq_chain_'+letter]=seq[0]
          coords_dict_chain = {}
          if ca_only:
            coords_dict_chain['CA_chain_'+letter]=xyz[:,1,:].tolist()
          else:
            coords_dict_chain['N_chain_'+letter]=xyz[:,0,:].tolist()
            coords_dict_chain['CA_chain_'+letter]=xyz[:,1,:].tolist()
            coords_dict_chain['C_chain_'+letter]=xyz[:,2,:].tolist()
            coords_dict_chain['O_chain_'+letter]=xyz[:,3,:].tolist()
          my_dict['coords_chain_'+letter]=coords_dict_chain
          s += 1
  my_dict['name']=pose.pdb_info().name()
  my_dict['num_of_chains'] = s
  my_dict['seq'] = concat_seq
  pdb_dict_list.append(my_dict)

  all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
  designed_chain_list = [str(item) for item in all_chain_list if str(item) not in fixed_chain_list]
  chain_id_dict = {}
  chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)
  
  return pdb_dict_list,chain_id_dict

def parse_pose_with_ligand(pose, name, lig_params=[], fixed_chain_list = []):

  # ligand atoms definition
  ref_atype_to_element = {'CNH2': 'C', 'COO': 'C', 'CH0': 'C', 'CH1': 'C', 'CH2': 'C', 'CH3': 'C', 'aroC': 'C', 'Ntrp': 'N', 'Nhis': 'N', 'NtrR': 'N', 'NH2O': 'N', 'Nlys': 'N', 'Narg': 'N', 'Npro': 'N', 'OH': 'O', 'OW': 'O', 'ONH2': 'O', 'OOC': 'O', 'Oaro': 'O', 'Oet2': 'O', 'Oet3': 'O', 'S': 'S', 'SH1': 'S', 'Nbb': 'N', 'CAbb': 'C', 'CObb': 'C', 'OCbb': 'O', 'Phos': 'P', 'Pbb': 'P', 'Hpol': 'H', 'HS': 'H', 'Hapo': 'H', 'Haro': 'H', 'HNbb': 'H', 'Hwat': 'H', 'Owat': 'O', 'Opoint': 'O', 'HOH': 'O', 'Bsp2': 'B', 'F': 'F', 'Cl': 'CL', 'Br': 'BR', 'I': 'I', 'Zn2p': 'ZN', 'Co2p': 'CO', 'Cu2p': 'CU', 'Fe2p': 'FE', 'Fe3p': 'FE', 'Mg2p': 'MG', 'Ca2p': 'CA', 'Pha': 'P', 'OPha': 'O', 'OHha': 'O', 'Hha': 'H', 'CO3': 'C', 'OC3': 'O', 'Si': 'Si', 'OSi': 'O', 'Oice': 'O', 'Hice': 'H', 'Na1p': 'NA', 'K1p': 'K', 'He': 'HE', 'Li': 'LI', 'Be': 'BE', 'Ne': 'NE', 'Al': 'AL', 'Ar': 'AR', 'Sc': 'SC', 'Ti': 'TI', 'V': 'V', 'Cr': 'CR', 'Mn': 'MN', 'Ni': 'NI', 'Ga': 'GA', 'Ge': 'GE', 'As': 'AS', 'Se': 'SE', 'Kr': 'KR', 'Rb': 'RB', 'Sr': 'SR', 'Y': 'Y', 'Zr': 'ZR', 'Nb': 'NB', 'Mo': 'MO', 'Tc': 'TC', 'Ru': 'RU', 'Rh': 'RH', 'Pd': 'PD', 'Ag': 'AG', 'Cd': 'CD', 'In': 'IN', 'Sn': 'SN', 'Sb': 'SB', 'Te': 'TE', 'Xe': 'XE', 'Cs': 'CS', 'Ba': 'BA', 'La': 'LA', 'Ce': 'CE', 'Pr': 'PR', 'Nd': 'ND', 'Pm': 'PM', 'Sm': 'SM', 'Eu': 'EU', 'Gd': 'GD', 'Tb': 'TB', 'Dy': 'DY', 'Ho': 'HO', 'Er': 'ER', 'Tm': 'TM', 'Yb': 'YB', 'Lu': 'LU', 'Hf': 'HF', 'Ta': 'TA', 'W': 'W', 'Re': 'RE', 'Os': 'OS', 'Ir': 'IR', 'Pt': 'PT', 'Au': 'AU', 'Hg': 'HG', 'Tl': 'TL', 'Pb': 'PB', 'Bi': 'BI', 'Po': 'PO', 'At': 'AT', 'Rn': 'RN', 'Fr': 'FR', 'Ra': 'RA', 'Ac': 'AC', 'Th': 'TH', 'Pa': 'PA', 'U': 'U', 'Np': 'NP', 'Pu': 'PU', 'Am': 'AM', 'Cm': 'CM', 'Bk': 'BK', 'Cf': 'CF', 'Es': 'ES', 'Fm': 'FM', 'Md': 'MD', 'No': 'NO', 'Lr': 'LR', 'SUCK': 'Z', 'REPL': 'Z', 'REPLS': 'Z', 'HREPS': 'Z', 'VIRT': 'X', 'MPct': 'X', 'MPnm': 'X', 'MPdp': 'X', 'MPtk': 'X'}
  chem_elements = ['C','N','O','P','S','AC','AG','AL','AM','AR','AS','AT','AU','B','BA','BE','BI','BK','BR','CA','CD','CE','CF','CL','CM','CO','CR','CS','CU','DY','ER','ES','EU','F','FE','FM','FR','GA','GD','GE','H','HE','HF','HG','HO','I','IN','IR','K','KR','LA','LI','LR','LU','MD','MG','MN','MO','NA','NB','ND','NE','NI','NO','NP','OS','PA','PB','PD','PM','PO','PR','PT','PU','RA','RB','RE','RH','RN','RU','SB','SC','SE','SM','SN','SR','Si','TA','TB','TC','TE','TH','TI','TL','TM','U','V','W','X','XE','Y','YB','Z','ZN','ZR']
  ref_atypes_dict = dict(zip(chem_elements, range(len(chem_elements))))
  dna_rna_dict = {
    "a" : ['P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N7', 'C8','N9', "", ""],
    "t" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "C5", "C6", "C7", "", "", ""],
    "c" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6", "", "", ""],
    "g" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", ""],
    "d" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", ""],
    "r" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", "", ""],
    "y" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6", "", ""],
    "u" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"],
    "X" : 22*[""]}
  dna_rna_atom_types = np.array(["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", "O4", "O2", "N4", "C7", ""])
  idxAA_22_to_27 = np.zeros((9, 22), np.int32)
  atoms = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
        'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
        'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
        'CZ3', 'NZ']  # These are the 36 atom types mentioned in Justas's script
  all_atom_types = atoms + list(dna_rna_atom_types)

  for i, AA in enumerate(dna_rna_dict.keys()):
      for j, atom in enumerate(dna_rna_dict[AA]):
          idxAA_22_to_27[i,j] = int(np.argwhere(atom==dna_rna_atom_types)[0][0])
  c = 0
  dna_list = 'atcg'
  rna_list = 'dryu'
  protein_list = 'ARNDCQEGHILKMFPSTWYVX'
  protein_list_check = 'ARNDCQEGHILKMFPSTWYV'
  k_DNA = 10
  ligand_dumm_list = 'J'

  my_dict = {}
  s = 0
  concat_seq = ''
  concat_seq_DNA = ''
  chain_list = []
  Cb_list = []
  P_list = []
  dna_atom_list = []
  dna_atom_mask_list = []
  #
  ligand_atom_list = []
  ligand_atype_list = []
  ligand_total_length = 0
  pdb_dict_list = []

  lig_to_atms = {}
  for lig_param in lig_params:
      lig_to_atms.update(parse_extra_res_fa_param(lig_param))

  exist_chains = list(set([pose.pdb_info().chain(residue_id) for residue_id in range(1, pose.total_residue() + 1) ]))
  for letter in exist_chains:
    xyz, seq, atype = extract_coords_from_pose(pose, atoms=all_atom_types , chain=letter,lig_to_atms=lig_to_atms)
    protein_seq_flag = any([(item in seq[0]) for item in protein_list_check])
    dna_seq_flag = any([(item in seq[0]) for item in dna_list])
    rna_seq_flag = any([(item in seq[0]) for item in rna_list])
    lig_seq_flag = any([(item in seq[0]) for item in ligand_dumm_list])
    if protein_seq_flag: xyz, seq, atype = extract_coords_from_pose(pose, chain=letter, atoms = atoms)
    elif (dna_seq_flag or rna_seq_flag): xyz, seq, atype = extract_coords_from_pose(pose, chain=letter, atoms = list(dna_rna_atom_types))
    elif (lig_seq_flag): xyz,seq, atype = extract_coords_from_pose(pose, chain=letter, atoms=[], lig_to_atms=lig_to_atms)
    
    if protein_seq_flag:
      my_dict['seq_chain_'+letter]=seq[0]
      concat_seq += seq[0]
      chain_list.append(letter)
      all_atoms = np.array(xyz) #[L, 14, 3] # deleted res index on xyz--I think this was useful when there were batches of structures at once?
      b = all_atoms[:,1] - all_atoms[:,0]
      c = all_atoms[:,2] - all_atoms[:,1]
      a = np.cross(b, c, -1)
      Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + all_atoms[:,1] #virtual
      Cb_list.append(Cb)
      coords_dict_chain = {}
      coords_dict_chain['all_atoms_chain_'+letter]=xyz.tolist()
      my_dict['coords_chain_'+letter]=coords_dict_chain
    elif dna_seq_flag or rna_seq_flag: # This section is important for moving from 22-atom representation to the 27-atom representation...unless it's already in 27 format??
      all_atoms = np.array(xyz)
      P_list.append(all_atoms[:,0])
      all_atoms_ones = np.ones((all_atoms.shape[0], 22)) # I believe all_atoms.shape[0] is supposed to be the length of the sequence
      seq_ = "".join(list(np.array(list(seq))[0,]))
      concat_seq_DNA += seq_
      all_atoms27_mask = np.zeros((len(seq_), 27))
      idx = np.array([idxAA_22_to_27[np.argwhere(AA==np.array(list(dna_rna_dict.keys())))[0][0]] for AA in seq_])
      np.put_along_axis(all_atoms27_mask, idx, all_atoms_ones, 1) 
      dna_atom_list.append(all_atoms) # was all_atoms27, but all_atoms is already in that format!!
      dna_atom_mask_list.append(all_atoms27_mask)
    elif lig_seq_flag:
        temp_atype = -np.ones(len(atype))
        for k_, ros_type in enumerate(atype):
            if ros_type in list(ref_atype_to_element):
                temp_atype[k_] = ref_atypes_dict[ref_atype_to_element[ros_type]]
            else:
                temp_atype[k_] = ref_atypes_dict['X']
        all_atoms = np.array(xyz)
        ligand_atype = np.array(temp_atype,dtype=int)
        if (1-np.isnan(all_atoms)).sum() != 0:
            tmp_idx = np.argwhere(1-np.isnan(all_atoms[0,].mean(-1))==1.0)[-1][0] + 1
            ligand_atom_list.append(all_atoms[:,:tmp_idx,:])
            ligand_atype_list.append(ligand_atype[:tmp_idx])
            ligand_total_length += tmp_idx

    if len(P_list) > 0:
        Cb_stack = np.concatenate(Cb_list, 0) #[L, 3]
        P_stack = np.concatenate(P_list, 0) #[K, 3]
        dna_atom_stack = np.concatenate(dna_atom_list, 0)
        dna_atom_mask_stack = np.concatenate(dna_atom_mask_list, 0)
        
        D = np.sqrt(((Cb_stack[:,None,:]-P_stack[None,:,:])**2).sum(-1) + 1e-7)
        idx_dna = np.argsort(D,-1)[:,:k_DNA] #top 10 neighbors per residue
        dna_atom_selected = dna_atom_stack[idx_dna]
        dna_atom_mask_selected = dna_atom_mask_stack[idx_dna]
        my_dict['dna_context'] = dna_atom_selected[:,:,:-1,:].tolist()
        my_dict['dna_context_mask'] = dna_atom_mask_selected[:,:,:-1].tolist()
    else:
        my_dict['dna_context'] = 'no_DNA'
        my_dict['dna_context_mask'] = 'no_DNA'
    if ligand_atom_list:
        ligand_atom_stack = np.concatenate(ligand_atom_list, 0)
        ligand_atype_stack = np.concatenate(ligand_atype_list, 0)
        my_dict['ligand_context'] = ligand_atom_stack.tolist()
        my_dict['ligand_atype'] = ligand_atype_stack.tolist()
    else:
        my_dict['ligand_context'] = 'no_ligand'
        my_dict['ligand_atype'] = 'no_ligand'
    my_dict['ligand_length'] = int(ligand_total_length)
    #
    my_dict['name']=name
    my_dict['num_of_chains'] = s
    my_dict['seq'] = concat_seq
    pdb_dict_list.append(my_dict)
    all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
    designed_chain_list = [str(item) for item in all_chain_list if str(item) not in fixed_chain_list]
    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)
  return pdb_dict_list,chain_id_dict


def thread_mpnn_seq( pose, seq , chain = None, fixed_idxs = []):
    rsd_set = pose.residue_type_set_for_pose()

    aa1to3=dict({'A':'ALA', 'C':'CYS', 'D':'ASP', 'E':'GLU', 'F':'PHE', 'G':'GLY',
        'H':'HIS', 'I':'ILE', 'K':'LYS', 'L':'LEU', 'M':'MET', 'N':'ASN', 'P':'PRO',
        'Q':'GLN', 'R':'ARG', 'S':'SER', 'T':'THR', 'V':'VAL', 'W':'TRP', 'Y':'TYR'})

    if chain is not None:
      start_res_idx = min([residue_id for residue_id in range(1, pose.total_residue() + 1) if pose.pdb_info().chain(residue_id) == chain  ])
    else:
      start_res_idx = 1
      
    for resi, mut_to in enumerate( seq ):
      if resi not in fixed_idxs:
        resi += start_res_idx # 1 indexing
        name3 = aa1to3[ mut_to ]
        new_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( name3 ) )
        pose.replace_residue( resi, new_res, True )
    
    return pose

def mutate_residue( pose, resi, aa = 'C' ):
    rsd_set = pose.residue_type_set_for_pose()

    aa1to3=dict({'A':'ALA', 'C':'CYS', 'D':'ASP', 'E':'GLU', 'F':'PHE', 'G':'GLY',
        'H':'HIS', 'I':'ILE', 'K':'LYS', 'L':'LEU', 'M':'MET', 'N':'ASN', 'P':'PRO',
        'Q':'GLN', 'R':'ARG', 'S':'SER', 'T':'THR', 'V':'VAL', 'W':'TRP', 'Y':'TYR'})

    new_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( aa1to3[aa] ) )
    pose.replace_residue( resi, new_res, True )
    
    return pose

def pose_to_string(pose):
  output = pyrosetta.rosetta.std.ostringstream()
  pose.dump_pdb(output)
  pose_str = output.str()
  return pose_str

class sap_filter(object):
  def __init__(self,chain = None) -> None:
    
    if chain is None:
      self.chain_selector = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
    else:
      self.chain_selector = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(chain)
  
  def score(self,pose):
    sap = pyrosetta.rosetta.core.pack.guidance_scoreterms.sap.calculate_sap(
      pose,
      self.chain_selector,
      self.chain_selector,
      self.chain_selector,
    )
    return sap

def get_pyrosetta_movers(xml_objs,mode="monomer"):
  pyrosetta_movers = {}
  if mode == "monomer":
      pyrosetta_movers["pack_mover"] = xml_objs.get_mover( 'pack_monomer' )
      pyrosetta_movers["relax_mover"] = xml_objs.get_mover( 'relax_monomer' )
      pyrosetta_movers["fadesign_mover"] = xml_objs.get_mover( 'fastdes_monomer' )
      pyrosetta_movers["metrics"] = {
        "score_per_res" : xml_objs.get_filter('score_per_res'),
        "vbuns" : xml_objs.get_filter('vbuns'),
        "sap_score" : sap_filter(),
      }
  elif mode == "binder":
      pyrosetta_movers["pack_mover"] = xml_objs.get_mover( 'pack_binder' )
      pyrosetta_movers["relax_mover"] = xml_objs.get_mover( 'relax_binder_cart' )
      pyrosetta_movers["fadesign_mover"] = xml_objs.get_mover( 'fastdes_binder' )
      pyrosetta_movers["metrics"] = {
        # Monomer score term
        "score_per_res" : xml_objs.get_filter('score_per_res'),
        "vbuns" : xml_objs.get_filter('vbuns'),
        "sap_score" : sap_filter(chain="A"),
        # Binding score term
        "ddg" : xml_objs.get_filter('ddg'),
        "interface_buried_sasa" : xml_objs.get_filter('interface_buried_sasa'),
        "contact_molecular_surface" : xml_objs.get_filter('contact_molecular_surface'),
        "interface_sc" : xml_objs.get_filter('interface_sc'),
      }
  else:
    raise ValueError(f"mode : {mode} is currently not supported by pyrosetta-mpnn protocol")
  return pyrosetta_movers