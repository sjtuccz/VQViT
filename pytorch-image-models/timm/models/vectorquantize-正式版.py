import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from einops import rearrange, repeat, reduce, pack, unpack

from timm.layers import trunc_normal_

class FSQ(nn.Module):
    """
    vanilla FSQ 
    @Article{Mentzer2023,
    author  = {Mentzer, Fabian and Minnen, David and Agustsson, Eirikur and Tschannen, Michael},
    journal = {arXiv preprint arXiv:2309.15505},
    title   = {Finite scalar quantization: Vq-vae made simple},
    year    = {2023},
    file    = {:FINITE SCALAR QUANTIZATION.pdf:PDF},
    groups  = {VQ},
    }

    """
    def __init__(self, n_e, channels_in, channels_dim=3, index=0, levels=[5,5,5]):
        super().__init__()
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        assert len(levels) == channels_dim
        self.codebook_dim = len(levels)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32))
        self.codebook_size = self._levels.prod().item()
        
        basis = torch.cumprod(
                torch.tensor([1] + levels[:-1], dtype=torch.int32), 
                dim=0
            )
        self.register_buffer("_basis", basis)

        self.is_pre_cal = False
    def reparameterize(self):
        print('using FSQ reparameterize')
        self.is_pre_cal = True
        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
        # implicit_codebook = torch.tensor(implicit_codebook).to(self.expand.weight.device)
        expand_dict = self.expand(implicit_codebook)
        del self.expand
        return expand_dict
    
    def forward(self, z):
        '''
        z: (b, h , channels_in)
        '''
        z = self.compress(z) # (b, h , dim)
        codes = self.quantize(z)
        # for checking
        quantization_error = torch.mean((z-codes.detach())**2)
        if random.random() < 0.0005:
            indices = self.codes_to_indices(codes)
            unique_index=torch.unique(indices)
            num_feature = z.shape[0] * z.shape[1]
            print(f"ActivatedCode:{unique_index.shape[0]}, NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}")

        if not self.training and self.is_pre_cal:
            indices = self.codes_to_indices(codes)
            return indices
        else:
            z_q = self.expand(codes)
            return z_q , torch.tensor(0.0).cuda()
    
    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes
    
    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def round_ste(self, z: torch.Tensor) ->  torch.Tensor:
        """Round with straight through gradients."""
        zhat = z.round()
        return z + (zhat - z).detach()
        # return rotate_to(z, zhat) 

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = ((self._levels - 1) * (1 + eps) / 2)
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0).to(z.device)
        shift = torch.atanh(offset / half_l)
        z_bound = torch.tanh(z + shift) * half_l - offset
        # print(f"   z_bound: {z_bound.shape}")
        return z_bound
    
    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = self.round_ste(self.bound(z)).to(z.device)
        half_width = (self._levels // 2).to(z.device)# Renormalize to [-1, 1].
        return quantized / half_width
    
class FSQ_trainableT(nn.Module):
    '''
    Based on vanilla FSQ, the backpropagation of quantization_error 
    and the dynamic trainable quantization range have been added.
    '''
    def __init__(self, n_e, channels_in, channels_dim, index=0, levels=[15,15,15], T=0, T_min=0.1):
        super().__init__()
        print(f"Using FSQ_trainableT, T init= {T}")

        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        # self.compress = nn.Linear(channels_in, channels_dim, bias=False)
        # self.expand = nn.Linear(channels_dim, channels_in, bias=False)
        assert len(levels) == channels_dim
        self.codebook_dim = len(levels)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32))
        self.register_buffer("codebook_size", torch.tensor(self._levels.prod(), dtype=torch.int32))
        self.register_buffer("T_min", torch.tensor(T_min, dtype=torch.float))
        self.T_raw = nn.Parameter(torch.tensor([T for _ in range(self.codebook_dim)], dtype=torch.float32))

        basis = torch.cumprod(
                torch.tensor([1] + levels[:-1], dtype=torch.int32), 
                dim=0
            )
        self.register_buffer("_basis", basis)

        self.is_pre_cal = False

    def get_T(self):
        T_softplus =  F.softplus(self.T_raw) 
        return T_softplus
    
    def reparameterize(self):
        print('using FSQ_trainableT reparameterize')
        self.is_pre_cal = True
        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size).to(self.codebook_size.device))
        expand_dict = self.expand(implicit_codebook)
        del self.expand
        self.register_buffer("T_softplus", self.get_T())
        del self.T_raw
        return expand_dict
    
    def forward(self, z):
        '''
        z: (b, h , channels_in)
        '''
        rand = random.random()
        input = z
        z = self.compress(z) # (b, h , dim)
        codes = self.quantize(z) # range (-T,T)
        quantization_error = torch.mean((z-codes.detach())**2)
        if rand < 0.0005:
            # analyze_tensor(input, name="input")
            # analyze_tensor(z, name="compress")
            # analyze_tensor(codes, name="codes")
            indices = self.codes_to_indices(codes)
            unique_index=torch.unique(indices)
            num_feature = z.shape[0] * z.shape[1]
            # print(f"NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}")
            print(f"ActivatedCode:{unique_index.shape[0]}, NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}")
        
        if not self.training and self.is_pre_cal:
            indices = self.codes_to_indices(codes)
            return indices
        else:
            z_q = self.expand(codes)
            return z_q , quantization_error
            # return z_q , torch.tensor(0.0).cuda()
    
    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        if not self.training and self.is_pre_cal:
            return (zhat - half_width) / half_width *self.T_softplus
        else:
            return (zhat - half_width) / half_width *self.get_T()

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes
    
    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        if not self.training and self.is_pre_cal:
            return (zhat_normalized / self.T_softplus * half_width) + half_width
        return (zhat_normalized / self.get_T() * half_width) + half_width
    
    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def round_ste(self, z: torch.Tensor) ->  torch.Tensor:
        """Round with straight through gradients."""
        zhat = z.round()
        return z + (zhat - z).detach()
        # return zhat
        # return rotate_to(z, zhat) 

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = ((self._levels - 1) * (1 + eps) / 2)
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0).to(z.device)
        shift = torch.atanh(offset / half_l)
        if not self.training and self.is_pre_cal:
            z_bound = torch.tanh(z/self.T_softplus + shift) * half_l - offset
        else:
            z_bound = torch.tanh(z/self.get_T() + shift) * half_l - offset
        return z_bound
    
    def quantize(self, z):
        # print("z shape :",z.shape)
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = self.round_ste(self.bound(z)).to(z.device)
        half_width = (self._levels // 2).to(z.device)# Renormalize to [-T, T].
        if not self.training and self.is_pre_cal:
            return quantized / half_width * self.T_softplus
        return quantized / half_width * self.get_T()
        
    
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, channels_in, channels_dim, index=0):
        super().__init__()
        self.embedding = nn.Embedding(n_e, channels_dim)
        trunc_normal_(self.embedding.weight,mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        self.codebook_size = n_e
        self.codebook_dim = channels_dim
        self.is_pre_cal = False
    def reparameterize(self):
        print('using VQ reparameterize')
        self.is_pre_cal = True
        dict_embeding = self.embedding.weight.data.clone().detach()
        expand_dict = self.expand(dict_embeding)
        del self.expand
        return expand_dict
    def forward(self, z):
        input = z
        z = self.compress(z)
        z_flattened = z.view(-1, z.shape[-1]) # bhw,c
        num_feature = z_flattened.shape[0]
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        indices = torch.argmin(d, dim=1)
        if not self.training and self.is_pre_cal:
            return indices
        unique_index=torch.unique(indices)
        z_q = self.embedding(indices)
        if self.training:
            loss_dict = torch.mean((z_q.detach()-z_flattened)**2) + 2.0* torch.mean((z_q - z_flattened.detach()) ** 2)
            z_q = z_q.detach()+ (z_flattened - z_flattened.detach())
            z_q = z_q.view(z.shape)
            z_q = self.expand(z_q)
            if random.random() < 0.0008:
                print(f"activated vectors:{unique_index.shape[0]} , num feature: {num_feature}, num dict: {self.codebook_size}")
            return z_q, loss_dict 
        else:
            z_q = z_q.view(z.shape)
            z_q = self.expand(z_q)
            if random.random() < 0.001:
                print(f"activated vectors:{unique_index.shape[0]} , num feature: {num_feature}, num dict: {self.codebook_size}")
            return z_q , torch.tensor(0.0).cuda()

################# replace STE with rotation trick  #################
def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d
def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one
def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)
def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = l2norm(u + q, dim = 1).detach()

    return (
        e -
        2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) +
        2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())
    )
def safe_div(num, den, eps = 1e-6):
    return num / den.clamp(min = eps)
def rotate_to(src, tgt):
    # rotation trick STE (https://arxiv.org/abs/2410.06424) to get gradients through VQ layer.
    src, inverse = pack_one(src, '* d')
    tgt, _ = pack_one(tgt, '* d')

    norm_src = src.norm(dim = -1, keepdim = True)
    norm_tgt = tgt.norm(dim = -1, keepdim = True)

    rotated_tgt = efficient_rotation_trick_transform(
        safe_div(src, norm_src),
        safe_div(tgt, norm_tgt),
        src
    ).squeeze()

    rotated = rotated_tgt * safe_div(norm_tgt, norm_src).detach()

    return inverse(rotated)

def orthogonal_loss_fn(t, unique_index=None):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n, d = t.shape[:2]
    if unique_index:
        mask = ~torch.notisin(torch.arange(n).to(t.device), unique_index)
        t[mask]=0
    normed_codes = l2norm(t)
    cosine_sim = torch.einsum('i d, j d -> i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (n ** 2) - (1 / n)

