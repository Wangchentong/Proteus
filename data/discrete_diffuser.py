import sys
import data.utils as du
from data import residue_constants
import numpy as np
import torch
import functools as fn
from operator import add
from sklearn.preprocessing import normalize

def loadBlosum62(path, softmax = False,double_stochastic = False, max_clip = 1000, pad_tokens = 0 ):
    residue_idx_dic = residue_constants.restype_order_with_x
    content = open(path, "r").read().splitlines()
    content = [line for line in content if not line.startswith(";")]
    labels = content[0].split()
    content = content[1:]
    blosum_matrix = np.full((len(residue_idx_dic), len(residue_idx_dic)), np.nan)
    
    for line in content:
        line = line.strip()
        linelist = line.split()
        if not len(linelist) == len(labels) + 1:
            print(len(linelist), len(labels))
            # Check if line has as may entries as labels
            raise EOFError("Blosum file is missing values.")

        for index, lab in enumerate(labels, start=1):
            if lab in residue_idx_dic and linelist[0] in residue_idx_dic:
                blosum_matrix[residue_idx_dic[linelist[0]],residue_idx_dic[lab]] = min(float(linelist[index]),max_clip)
                blosum_matrix[residue_idx_dic[lab],residue_idx_dic[linelist[0]]] = min(float(linelist[index]),max_clip)
                
    # Check quadratic
    if (blosum_matrix==np.nan).any():
        raise EOFError("Blosum file is not fully loaded")
    
    if softmax:
        blosum_matrix = np.exp(blosum_matrix)/np.sum(np.exp(blosum_matrix),axis=0)
    
    if double_stochastic:
        blosum_matrix = normalize(blosum_matrix, axis=1, norm='l1')
        while not np.isclose(np.min(np.sum(blosum_matrix, axis=0)), 1): # only checking that one value converges to 1 (prob best to do all 4 min/max)
            blosum_matrix = normalize(blosum_matrix, axis=0, norm='l1')
            blosum_matrix = normalize(blosum_matrix, axis=1, norm='l1')
    if pad_tokens:
        blosum_matrix_pad = np.zeros((len(residue_idx_dic)+pad_tokens, len(residue_idx_dic)+pad_tokens))
        blosum_matrix_pad[:-pad_tokens,:-pad_tokens] = blosum_matrix
        blosum_matrix_pad[[i for i in range(len(residue_idx_dic),len(residue_idx_dic)+pad_tokens)],[i for i in range(len(residue_idx_dic),len(residue_idx_dic)+pad_tokens)]] = 1
        return blosum_matrix_pad
    else:
        return blosum_matrix
    
def get_rate_matrix(rate):
    rate = rate - np.diag(np.diag(rate))
    rate = rate - np.diag(np.sum(rate, axis=1))
    eigvals, eigvecs = np.linalg.eigh(rate)
    return rate,eigvals,eigvecs

def usvt(eigvecs, inv_eigvecs, diag_embed):
    ns = eigvecs.shape[0]
    u = np.reshape(eigvecs, (ns, ns))
    vt = np.reshape(inv_eigvecs, (ns, ns))
    transitions = np.matmul(u, np.matmul(diag_embed, vt))
    transitions = transitions / np.sum(transitions, axis=-1, keepdims=True)
    return transitions

def stable_distribution(M):
    A = np.vstack((M.T - np.eye(M.shape[0]), np.ones((1, M.shape[0]))))

    # Right-hand side of the equation is a zero vector followed by 1
    b = np.zeros(M.shape[0] + 1)
    b[-1] = 1

    # Solve the linear system
    stable_distribution = np.linalg.lstsq(A, b, rcond=None)[0]

    # Normalize to obtain a probability distribution
    stable_distribution /= np.sum(stable_distribution)
    
    return stable_distribution
def shaped_categorical(probs, epsilon=1e-10):
    ds = probs.shape
    num_classes = ds[-1]
    distribution = torch.distributions.categorical.Categorical(
        torch.reshape(probs + epsilon, [-1, num_classes])
    )
    counts = distribution.sample()
    return torch.reshape(counts, ds[:-1])

class DiscreteDiffuser(object):
    """Generic forward model."""
    def __init__(self, conf,prior_distributions = None):
        self._conf = conf
        self.transition_matrix = loadBlosum62(self._conf.blosum_matrix_path,softmax=True, double_stochastic=True,max_clip=5,pad_tokens=2)
        self.total_transition = self._conf.total_transition
        self.replace_fraction = self._conf.replace_fraction
        self.num_states = self.transition_matrix.shape[0]
        self.rate_matrix, self.eigvals, self.eigvecs = get_rate_matrix(self.transition_matrix)
        
        if prior_distributions is not None:
            self.prior_distributions = prior_distributions
        else:
            self.prior_distributions = np.ones([self.num_states])/self.num_states

    def transition(self, t):
        # translate time (0,1) to (0,total_transition) with a exponential schedule
        # t = np.exp(np.log(self.total_transition + 1)*t) -1
        t = [(self.total_transition + 1)**t_ - 1 for t_ in t]
        diag_embed = np.array([np.diag(np.exp(self.eigvals * t_)) for t_ in t])
        transitions = usvt(self.eigvecs, self.eigvecs.T, diag_embed)
        # prevent unstablized sign in near t=0
        transitions = np.clip(transitions,a_min=0,a_max=1)
        return transitions
    

    def sample_from_prior(self,shape):
        xt = np.random.choice(self.num_states,p=self.prior_distributions, size=shape)
        return xt

    def sample_xt(self, x0, t):
        x0 = np.eye(self.num_states)[x0]
        qt = self.transition(t)
        qt0 = np.einsum("brik, bkm -> brim",x0,qt)
        qt0 = qt0.reshape((-1,self.num_states))
        xt = np.array([np.random.choice(len(p), p=p) for p in qt0])
        xt = xt.reshape(x0.shape[:-1])
        return xt

    def make_masked_msa(self, msa, msa_mask=None):
        """Create data for BERT on raw MSA."""
        if isinstance(msa, np.ndarray):
            msa = torch.tensor(msa, dtype=torch.int64)
        msa = torch.clip(msa,max=21)
        if msa_mask is None:
            msa_mask = (msa != 21) * (msa != 20)
        
        # Add a random amino acid uniformly.
        random_aa = torch.tensor([0.05] * 20 + [0.0, 0.0], dtype=torch.float32)

        categorical_probs = (
            self._conf.uniform_prob * random_aa
            + self._conf.same_prob * torch.nn.functional.one_hot(msa, num_classes=self.num_states-1)
        )

        # Put all remaining probability on [MASK] which is a new column
        pad_shapes = list(
            fn.reduce(add, [(0, 0) for _ in range(len(categorical_probs.shape))])
        )
        pad_shapes[1] = 1
        mask_prob = (
            1.0  - self._conf.same_prob - self._conf.uniform_prob
        )
        assert mask_prob >= 0.0
        categorical_probs = torch.nn.functional.pad(
            categorical_probs, pad_shapes, value=mask_prob
        )

        sh = msa.shape
        mask_position = ((torch.rand(sh) < self.replace_fraction) * msa_mask).to(torch.bool)
        bert_msa = shaped_categorical(categorical_probs)
        bert_msa = torch.where(mask_position, bert_msa, msa)


        return {
            'msa' : bert_msa,
            'bert_mask' : mask_position.to(torch.float32)
        }

    def masked_msa_loss(self,logits, msa_t, t, bert_mask, eps=1e-8, **kwargs):
        """
        Question:
            Predict x0 or xt, it seems almost the same to me.

        Args:
            logits: [*, N_seq, N_res, self.num_states] predicted residue distribution at t=0
            true_msa: [*, N_seq, N_res] true MSA
            bert_mask: [*, N_seq, N_res] MSA mask
        Returns:
            Masked MSA loss
        """
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        prob_x0 = torch.nn.functional.softmax(logits, dim=-1)
        qt = torch.tensor(self.transition(t),device=prob_x0.device,dtype=prob_x0.dtype)
        prob_xt = torch.einsum("brik, bkm -> brim",prob_x0,qt)
        log_prob_xt = torch.log(prob_xt + eps)
        errors = -log_prob_xt * torch.nn.functional.one_hot(msa_t, num_classes=self.num_states).float()
        loss = errors * bert_mask[...,None]
        loss = torch.sum(loss, dim=-1)
        denom = eps + torch.sum( bert_mask, dim=(-1, -2))
        loss = loss / denom[..., None, None]
        loss = torch.sum(loss, dim=(-1,-2))

        return loss
    
    def reverse(self, logits:torch.Tensor, t:list, temperature:float=1.0):
        prob_x0 = torch.nn.functional.softmax(logits/temperature, dim=-1)
        qt = torch.tensor(self.transition(t),device=prob_x0.device,dtype=prob_x0.dtype)
        prob_xt = prob_x0 @ qt
        xt = torch.multinomial(prob_xt.reshape(-1,self.num_states),1).reshape(prob_x0.shape[:-1])
        return xt