from typing import Optional, List, Tuple
import math
import functools as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.utils.checkpoint import checkpoint

from openfold.model.primitives import (
    Linear, 
    LayerNorm,
    _attention,
    _tied_attention
)
from openfold.utils.rigid_utils import Rigid
from openfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)
try:
    from flash_attn.bert_padding import unpad_input, pad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
except:
    pass

from model.utils import RotaryEmbedding
from model.utils import get_timestep_embedding

@torch.jit.ignore
def _flash_attn(q, k, v, kv_mask):
    batch_dims = q.shape[:-3]
    no_heads, n, c = q.shape[-3:]
    dtype = q.dtype

    q = q.half()
    k = k.half()
    v = v.half()
    kv_mask = kv_mask.half()

    # [*, B, N, H, C]
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)

    # [B_flat, N, H, C]
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])

    # Flattened batch size
    batch_size = q.shape[0]
    
    # [B_flat * N, H, C]
    q = q.reshape(-1, *q.shape[-2:])
    
    q_max_s = n
    q_cu_seqlens = torch.arange(
        0, (batch_size + 1) * n, step=n, dtype=torch.int32, device=q.device
    )

    # [B_flat, N, 2, H, C]
    kv = torch.stack([k, v], dim=-3) 
    kv_shape = kv.shape
    
    # [B_flat, N, 2 * H * C]
    kv = kv.reshape(*kv.shape[:-3], -1) 
    
    kv_unpad, _, kv_cu_seqlens, kv_max_s = unpad_input(kv, kv_mask)
    kv_unpad = kv_unpad.reshape(-1, *kv_shape[-3:])
   
    out = flash_attn_varlen_kvpacked_func(
        q,
        kv_unpad,
        q_cu_seqlens,
        kv_cu_seqlens,
        q_max_s,
        kv_max_s,
        dropout_p = 0.,
        softmax_scale = 1., # q has been scaled already
    )
  
    # [*, B, N, H, C]
    out = out.reshape(*batch_dims, n, no_heads, c) 

    out = out.to(dtype=dtype)

    return out

class MSATransition(nn.Module):
    """
    Feed-forward network applied to MSA activations after attention.

    Implements Algorithm 9
    """
    def __init__(self, c_m, n):
        """
        Args:
            c_m:
                MSA channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension
        """
        super(MSATransition, self).__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m, init="final")

    def _transition(self, m, mask):
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA activation
            mask:
                [*, N_seq, N_res] MSA mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA activation update
        """
        # DISCREPANCY: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        m = self._transition(m, mask)

        return m

class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, init="final"
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, init="gating"
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self,
        q_x: torch.Tensor, 
        kv_x: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self,
        o: torch.Tensor, 
        q_x: torch.Tensor
    ) -> torch.Tensor:
        if(self.linear_g is not None):
            g = self.sigmoid(self.linear_g(q_x))
        
            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        use_flash: bool = False,
        flash_mask: Optional[torch.Tensor] = None,
        rotary_embedder: bool = None,
        tied = False,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
        Returns
            [*, Q, C_q] attention update
        """
        if(use_flash and biases is not None):
            raise ValueError(
                "use_flash is incompatible with the bias option. For masking, "
                "use flash_mask instead"
            )
        
        if(biases is None):
            biases = []
        
        # [*, H, Q/K, C_hidden]
        q, k, v = self._prep_qkv(q_x, kv_x)
        if rotary_embedder is not None:
            q,k = rotary_embedder(q,k)
        # [*, Q, H, C_hidden]
        if use_flash:
            o = _flash_attn(q, k, v, flash_mask)
        elif tied:
            o = _tied_attention(q, k, v, biases)
        else:
            o = _attention(q, k, v, biases)

        o = self._wrap_up(o, q_x)

        return o

class MSAAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        column_wise : bool = False,
        msa_transformer_style : bool = True ,
        inf=1e9,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            column_wise:
                row wise attention or column wise attention ? 
            msa_transformer_style:
                sum attention logits of each rows or not ?
            inf:
                A large number to be used in computing the attention mask
        """
        super(MSAAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.column_wise = column_wise
        self.msa_transformer_style = msa_transformer_style
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_in)
        
        self.mha = Attention(
            self.c_in, 
            self.c_in, 
            self.c_in, 
            self.c_hidden, 
            self.no_heads,
        )
        self.rotary_embedder = RotaryEmbedding(self.c_hidden)

    def forward(self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        biases: Optional[torch.Tensor] = [],
        use_flash: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
            bias:
                [*, N_res, N_res] Pair bias 
                
        """
        # Try row-wise axial attention in MSA transformer, it average attention logits of each rows, so only call softmax one time
        # Test if evoformer style axial attention is obviously slower than transformer style axial attention.
        if self.column_wise:
            m = m.transpose(-2, -3)
            if mask is not None:
                mask = mask.transpose(-1, -2)
        m = self.layer_norm_m(m)

        if not self.msa_transformer_style:
            m = self.mha(
                q_x=m, 
                kv_x=m, 
                biases=None,
                use_flash=use_flash,
                flash_mask=mask,
                rotary_embedder = self.rotary_embedder,
            )
        else:
            q,k,v = self.mha._prep_qkv(m, m)
            q,k = self.rotary_embedder(q,k)
            # [*, H, Q, C_hidden]
            q = permute_final_dims(q, (1, 0, 2))
            k = permute_final_dims(k, (1, 2, 0))
            v = permute_final_dims(v, (1, 0, 2))
            # [B, R, H, Q, K]
            m = torch.matmul(q, k)
            shape = m.shape
            # [B, H, Q, K]
            m = torch.sum(m,-4) / torch.sqrt(torch.any(mask, -1).sum(-1))[:,None,None,None]
            for bias in biases:
                m = m + bias
            # [B, 1, H, Q, K]
            m = torch.nn.functional.softmax(m, -1)[...,None,:,:,:].expand(*shape)
            # [*, H, Q, C_hidden]
            m = torch.matmul(m, v)
            # [*, Q, H, C_hidden]
            m = m.transpose(-2, -3)
            m = flatten_final_dims(m, 2)

        if self.column_wise:
            m = m.transpose(-2, -3)
            if mask is not None:
                mask = mask.transpose(-1, -2)

        return m

class MSATransformerLayer(nn.Module):
    """Implements an Axial MSA Transformer block."""

    def __init__(
        self,
        c_s,
        c_m,
        c_z,
        c_gate_s,
        c_hidden,
        no_heads,
        dropout,
        transition_n,
        inf,
        use_flash,
        column_attention,
    ) -> None:
        super(MSATransformerLayer, self).__init__()
        self.use_flash = use_flash
        self.inf = inf
        # bias embedding
        self.proj_left = Linear(c_s, c_gate_s)
        self.proj_right = Linear(c_s, c_gate_s)
        
        self.to_gate = Linear(c_gate_s*c_gate_s, c_z,init="gating")
        self.to_bias = Linear(c_z, no_heads, bias=False, init="normal")
        
        self.row_attention = MSAAttention(
            c_m,
            c_hidden,
            no_heads,
            inf=inf,
            msa_transformer_style = True,
        )
        if column_attention:
            self.column_attention = MSAAttention(
                c_m,
                c_hidden,
                no_heads,
                inf=inf,
                column_wise=True,
                msa_transformer_style = False
            )
        else:
            self.column_attention = None

        self.feed_forward_layer = MSATransition(c_m,n=transition_n)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        msa_feature: torch.Tensor,
        node_feature: torch.Tensor = None,
        pair_feature: torch.Tensor = None,
        node_mask : torch.Tensor = None,
        msa_mask: torch.Tensor = None,
    ):
        batch_size,num_res = msa_feature.shape[0],msa_feature.shape[2]
        biases = []
        
        if node_mask is not None:
            # [B, 1, 1, N]
            node_mask = (self.inf * (node_mask - 1))[:,None,None,:]
            biases.append(node_mask)
        
        if node_feature is not None and pair_feature is not None:
            # gate pair bias with sequence embedding
            left = self.proj_left(node_feature)
            right = self.proj_right(node_feature)
            gate = einsum('bli,bmj->blmij', left, right).reshape(batch_size,num_res,num_res,-1)
            gate = torch.sigmoid(self.to_gate(gate))
            bias = pair_feature * gate
            # pair bias shape : [B,N,N,H]
            bias = self.to_bias(bias)
            # [B, H, I, J]
            bias = permute_final_dims( bias, ( 2, 0, 1) )
            biases.append(bias)
        
        msa_feature = msa_feature + self.dropout_layer(self.row_attention(msa_feature, msa_mask, use_flash=self.use_flash, biases=biases))
        if self.column_attention:
            msa_feature = msa_feature + self.dropout_layer(self.column_attention(msa_feature, msa_mask, use_flash=self.use_flash))
        msa_feature = msa_feature + self.dropout_layer(self.feed_forward_layer(msa_feature))
        
        return msa_feature

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm
    class MSALayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    from torch.nn import LayerNorm as MSALayerNorm

class MSACrossAttention(nn.Module):
    def __init__(
        self,
        c_s,
        c_msa,
        c_hidden,
        no_heads,
        transition_n,
        dropout,
        inf=1e9,
        use_flash = False,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            column_wise:
                row wise attention or column wise attention ? 
            msa_transformer_style:
                sum attention logits of each rows or not ?
            inf:
                A large number to be used in computing the attention mask
        """
        super(MSACrossAttention, self).__init__()

        self.c_msa = c_msa
        self.c_s = c_s
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_msa)
        self.layer_norm_s = LayerNorm(self.c_s)
        
        self.mha = Attention(
            self.c_s,
            self.c_msa, 
            self.c_msa, 
            self.c_hidden, 
            self.no_heads,
            gating=True
        )
        self.feed_forward_layer = MSATransition(c_msa,n=transition_n)
        self.dropout_layer = nn.Dropout(dropout)
        self.use_flash = use_flash

    def forward(self, 
        msa_feature: torch.Tensor, 
        node_feature: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            msa_feature:
                [*, N_seq, N_res, C_m] MSA embedding
            node_feature:
                [*, C_s] node embedding
            mask:
                [*, N_seq, N_res] MSA mask
                
        """
        biases = []
        
        node_feature = node_feature[...,:,None,:]
        s = self.layer_norm_s(node_feature)
        
        msa_feature = msa_feature.transpose(-2, -3)
        msa = self.layer_norm_m(msa_feature)
        
        if msa_mask is not None:
            # [B, R, N]
            mask = (self.inf * (msa_mask - 1))
            # [B, N, 1(H), 1(Q), R(K)]
            mask = mask.transpose(-1,-2)[:,:,None,None,:]
            biases.append(mask)
        
        s = self.dropout_layer(self.feed_forward_layer(self.mha(
                    q_x=s, 
                    kv_x=msa, 
                    biases=biases,
                )))
        
        s = s.squeeze(-2)

        return s

class MSATransformer(nn.Module):
    """Implements an Axial MSA Transformer block."""

    def __init__(
        self,
        c_s,
        c_msa,
        c_z,
        c_gate_s,
        c_hidden_msa_att,
        no_heads_msa,
        dropout,
        num_blocks,
        n_tokens,
        transition_n = 4,
        inf=1e9,
        column_attention : bool = False,
        use_flash: bool = False,
        use_ckpt: bool = False,
        **kwargs,
    ) -> None:
        
        super(MSATransformer, self).__init__()
        
        self.cross_attention_layers = nn.ModuleList([
            MSACrossAttention(
                c_s=c_s,
                c_msa=c_msa,
                c_hidden=c_hidden_msa_att,
                no_heads=no_heads_msa,
                transition_n=transition_n,
                dropout=dropout,
                inf = inf ) for _ in range(num_blocks)
        ])
        
        self.token_embedding = nn.Embedding(
            n_tokens, c_msa
        )
        
        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=c_msa
        )
        
        self.msa_transformer_layers = nn.ModuleList([
            MSATransformerLayer(
                c_s=c_s,
                c_m=c_msa,
                c_z=c_z,
                c_gate_s=c_gate_s,
                c_hidden=c_hidden_msa_att,
                no_heads=no_heads_msa,
                dropout=dropout,
                transition_n=transition_n,
                inf=inf,
                use_flash=use_flash,
                column_attention=column_attention) for _ in range(num_blocks)
        ])
        self.use_ckpt = use_ckpt
        self.num_blocks = num_blocks
        
        self.gelu = nn.GELU()
        
        self.s_layer_norm = LayerNorm(c_s)
        self.dense_layer_norm = MSALayerNorm(c_msa)
        self.output_layer_norm = MSALayerNorm(c_msa)
        
        self.dense_layer = Linear(c_msa, c_msa)
        self.output_bias = nn.Parameter(torch.zeros(n_tokens))

    def layer_with_ipa_embedding(self, t, block_i: int, msa_feature:torch.Tensor,node_feature:torch.Tensor = None,pair_feature:torch.Tensor=None, msa_mask=None, node_mask=None):
        
        node_feature_detached = node_feature.detach()
        pair_feature_detached = pair_feature.detach()
        output = {}
          
        if block_i == 0:
            num_row,num_res = msa_feature.shape[-2:]
            msa_feature = self.token_embedding(msa_feature)
            # timestep embedding
            msa_t_embed = torch.tile(self.timestep_embedder(t)[:, None, None, :], (1, num_row, num_res, 1))
            msa_feature = msa_feature + msa_t_embed
            # embed query sequence
            query_embedding = torch.zeros(msa_feature.shape,device=msa_feature.device)
            query_embedding[...,0,:,0] = 1
            msa_feature = msa_feature + query_embedding

        x = msa_feature

        # if self.use_ckpt:
        #     x = checkpoint(self.msa_transformer_layers[block_i], x, msa_mask)
        # else:
        x = self.msa_transformer_layers[block_i](
            msa_feature=x,
            node_feature=node_feature_detached,
            pair_feature=pair_feature_detached,
            node_mask=node_mask,
            msa_mask = msa_mask
        )          
            
        output['msa_feature'] = x
            
        if node_feature is not None:
            output['node_feature'] = node_feature + self.cross_attention_layers[block_i](x,node_feature,msa_mask)
            
        if block_i == self.num_blocks - 1:
            x = self.dense_layer_norm(x)
            x = self.dense_layer(x)
            x = self.gelu(x)
            x = self.output_layer_norm(x)
            msa_logits = F.linear(x, self.token_embedding.weight) + self.output_bias
            output['msa_logits'] = msa_logits
            
        return output


