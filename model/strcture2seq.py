import functools as fn
import numpy as np
import torch
from torch import nn
from data import residue_constants
from ProteinMPNN.protein_mpnn_utils import ProteinMPNN,_scores, _S_to_seq, tied_featurize
import esm
from esm import Alphabet

class MPNN_ESM(nn.Module):
    def __init__(self,c_s,c_z,checkpoint_path,ca_only=True,temperature=0.1,seq_nums=4,esm_name='esm2_t33_650M_UR50D'):
        super(MPNN_ESM, self).__init__()
        self.ca_only = ca_only 
        self.temperature = temperature
        self.seq_nums = seq_nums
        checkpoint = torch.load(checkpoint_path)
        self.mpnn_model = ProteinMPNN(ca_only=ca_only, num_letters=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0, k_neighbors=checkpoint['num_edges'])
        self.mpnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.esm, self.esm_dict = esm.pretrained.load_model_and_alphabet_hub(esm_name)
        self.esm.requires_grad_(False)
        self.esm.half()
        self.mpnn_model.requires_grad_(False)
        self.register_buffer("mpnn_to_esm", MPNN_ESM._mpnn_to_esm(self.esm_dict))
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm.num_layers + 1))
        self.esm_p_combine = nn.Parameter(torch.zeros(self.esm.num_layers))
        self.esm_s_mlp = nn.Sequential(
            nn.LayerNorm(self.esm.embed_dim),
            nn.Linear(self.esm.embed_dim, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )
        self.esm_p_mlp = nn.Linear(self.esm.attention_heads * self.esm.num_layers, c_z)
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(MPNN_ESM, self).state_dict(destination, prefix, keep_vars)
        state_dict = {key: state_dict[key] for key in state_dict.keys() if 'mpnn_model' not in key and 'esm' not in key}
        return state_dict

    @staticmethod
    def _mpnn_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [
            d.get_idx(v) for v in 'ACDEFGHIKLMNPQRSTVWYX'
        ]
        return torch.tensor(esm_reorder)
    @staticmethod
    def _mpnn_to_af():
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        af_reorder = [
            residue_constants.restype_order_with_x[v] for v in 'ACDEFGHIKLMNPQRSTVWYX'
        ]
        return torch.tensor(af_reorder)
    
    def _mpnn_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.mpnn_to_esm[aa]
    
    def _compute_language_model_representations(
        self, esmaa: torch.Tensor
    ) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=True,
        )
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        esm_s = esm_s[:, 1:-1]  # B x L x Layers x C
        esm_p = res["attentions"][...,1:-1,1:-1].view(batch_size, -1, esm_s.shape[1],esm_s.shape[1])  # B x (L x H) x T x T
        esm_p = esm_p.permute(0, 2, 3, 1)  # B x L x L x (Layer x Heads)
        return esm_s,esm_p
    
    def forward(self,batch):
        device = next(self.parameters()).device
        parsed_batch = []
        for i in range(batch['chain_index'].shape[0]):
            parsed_dict = {}
            if len(torch.unique(batch['chain_index'][i])) > 1 :
                raise ValueError('Only single chain proteins are supported')
            if sum(batch['final_atom_mask'][:,-1,1]== 0) > 0 :
                raise ValueError('ProteinMPNN-ESM model does not support strcture padding now')
                
            for chain_index in torch.unique(batch['chain_index'][i]):
                coords_dict_chain = {}
                chain_name = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[chain_index]
                mask = (batch['chain_index'][i] == chain_index)
                parsed_dict['seq_chain_'+chain_name] = "".join([residue_constants.restypes_with_x[aa] for aa in batch['aatype'][i][batch['chain_index'][i]==chain_index].tolist()])
                if self.ca_only:
                    coords_dict_chain['CA_chain_'+chain_name] = batch['final_atom_positions'][i,mask,1,:].tolist()
                else:
                    coords_dict_chain['N_chain_'+chain_name] = batch['final_atom_positions'][i,mask,0,:].tolist()
                    coords_dict_chain['CA_chain_'+chain_name] = batch['final_atom_positions'][i,mask,1,:].tolist()
                    coords_dict_chain['C_chain_'+chain_name] = batch['final_atom_positions'][i,mask,2,:].tolist()
                    coords_dict_chain['O_chain_'+chain_name] = batch['final_atom_positions'][i,mask,4,:].tolist()
                parsed_dict['coords_chain_'+chain_name]=coords_dict_chain
            parsed_dict['name'] = ''
            parsed_dict['seq'] = "".join([residue_constants.restypes_with_x[aa] for aa in batch['aatype'][i].tolist()])
            parsed_batch.append(parsed_dict)
        
        omit_AAs_list = 'CX'
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
        bias_AAs_np = np.zeros(len(alphabet))
        esm_s_concat,esm_p_concat = [],[]
        with torch.no_grad():
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(parsed_batch, device, None, None, None, None, None,ca_only=self.ca_only)
            pssm_log_odds_mask = (pssm_log_odds_all > 0.0).float()
            for i in range(self.seq_nums):
                randn_2 = torch.randn(chain_M.shape, device=device)
                sample_dict = self.mpnn_model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=self.temperature, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=False, pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=False, bias_by_res=bias_by_res_all)
            
                S_sample = sample_dict["S"]
                esmaa = self._mpnn_idx_to_esm_idx(S_sample, chain_M)
                esm_s,esm_p = self._compute_language_model_representations(esmaa)
                esm_s_concat.append(esm_s.to(torch.float32).detach())
                esm_p_concat.append(esm_p.to(torch.float32).detach())
        esm_s,esm_p = torch.stack(esm_s_concat,1),torch.stack(esm_p_concat,1)
        esm_s = (self.esm_s_combine.softmax(0) @ esm_s)
        esm_s = self.esm_s_mlp(esm_s)
        esm_p = self.esm_p_mlp(esm_p)
        
        return esm_s,esm_p