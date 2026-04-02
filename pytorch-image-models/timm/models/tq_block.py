import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from einops import rearrange, pack, unpack

from timm.layers import trunc_normal_


def choose_tq(tq_type, dic_n, dim, dic_dim, tq_level=[3,3,3,3], tq_Tinit=1, input_format='NLC'):
    if tq_type == 'TQ' or tq_type == 'tq':
        return TQ_Qscale_deQscale(channels_in=dim, channels_dim=dic_dim, levels=tq_level, T=tq_Tinit, input_format=input_format)
    else:
        raise RuntimeError('tq type not implemented')
    
VALID_INPUT_FORMATS = {"NLC", "NCHW", "NHWC"}

class TQ_Qscale_deQscale(nn.Module):
    '''
    Based on vanilla TQ, quantization scaling factor & dequantization scaling factor have been added.
    '''
    def __init__(self, channels_in, channels_dim, levels=[15,15,15], T=1, input_format='NLC'):
        super().__init__()
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        assert len(levels) == channels_dim, "ensure len(levels) == channels_dim"
        assert all(element % 2 == 1 for element in levels), f"levels must be odd numbers, but: {levels}"
        assert input_format in VALID_INPUT_FORMATS, \
        f'input_format must be {list(VALID_INPUT_FORMATS)}, but: {input_format}'
        self.codebook_dim = len(levels)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32))
        levels_tensor = torch.tensor(levels, dtype=torch.float32)
        half_l = (levels_tensor // 2).float()
        self.register_buffer('half_l', half_l)

        self.register_buffer("codebook_size", self._levels.prod().clone().detach().to(torch.int32))        # self.T_raw = nn.Parameter(torch.tensor(T, dtype=torch.float32))
        self.T_raw = nn.Parameter(torch.tensor([T for _ in range(self.codebook_dim)], dtype=torch.float32)) 
        self.anti_q = nn.Parameter(torch.tensor([T for _ in range(self.codebook_dim)], dtype=torch.float32)) 
        basis = torch.cumprod(
                torch.tensor([1] + levels[:-1], dtype=torch.int32), 
                dim=0
            )
        self.register_buffer("_basis", basis)
        self.token_wise_rep = False

        self.codebook_meter = CodebookMeter(codebook_size=self.codebook_size.item())
        self.input_format = input_format  # 'nhc' or 'nch' or None
        self.fold_dim = None  # to be set if needed
    
    def reparameterize(self):
        print('using TQ reparameterize')
        scale_factor = self.half_l / self.anti_q
        _inv_scale_factor = self.anti_q / self.half_l 
        self.register_buffer('scale_factor', scale_factor)
        self.register_buffer('_inv_scale_factor', _inv_scale_factor)
        _inv_T_raw = 1.0 / self.T_raw
        self.register_buffer('_inv_T_raw', _inv_T_raw)
        self.token_wise_rep = True
        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size, device=self.codebook_size.device)).to(self.codebook_size.device)
        expand_dict = self.expand(implicit_codebook)
        del self.expand
        return expand_dict

    def forward(self, z):
        '''
        z: (b, h , channels_in)
        '''
        input = z
        if self.input_format == 'NCHW':
            z = z.flatten(2).transpose(1, 2) #->(N, L, C)
        elif self.input_format == 'NHWC':
            z = z.flatten(1, 2) # (N, H, W, C) -> (N, L, C)
        z = self.compress(z) # (b, h , dim)
        if self.token_wise_rep:
            indices = self.quantize2index(z)
            self.codebook_meter.update(indices)
            return indices
        else:
            codes = self.quantize(z)
            z_q = self.expand(codes)
            if self.input_format == 'NLC':
                return z_q
            elif self.input_format == 'NCHW':
                z_q = z_q.transpose(1, 2).view(input.shape)
                return z_q
            elif self.input_format == 'NHWC':
                z_q = z_q.view(input.shape)
                return z_q
    

    def bound(self, z):
        if self.token_wise_rep:
            return z.mul_(self._inv_T_raw).tanh_().mul_(self.half_l)
        return torch.tanh(z/self.T_raw) * self.half_l
    
    def round_ste(self, z: torch.Tensor) ->  torch.Tensor:
        """Round with straight through gradients."""
        if self.token_wise_rep:
            return z.round()
        return z + (z.round() - z).detach() 
    
    


    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1').contiguous()
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def _scale_and_shift_inverse(self, zhat):
        return (zhat - self.half_l) / self.half_l * self.anti_q

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes
    

    def quantize(self, z):
        # print("z shape :",z.shape)
        """ Quantizes z, returns quantized zhat, same shape as z. """
        if self.token_wise_rep:
            return z.mul_(self._inv_T_raw).tanh_().mul_(self.half_l).round().mul_(self._inv_scale_factor)
        # return self.round_ste((2*torch.sigmoid(1.6*z)-1) * self.half_l)/ self.half_l
        # return self.round_ste((2*torch.sigmoid(1.6*z/self.T_raw)-1) * self.half_l)/ self.half_l *self.anti_q
        return self.round_ste(torch.tanh(z/self.T_raw) * self.half_l)/ self.half_l *self.anti_q

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        if self.token_wise_rep:
            return zhat.mul_(self.scale_factor).add_(self.half_l).mul_(self._basis).sum(dim=-1).to(torch.int32)
        else:
            return ((zhat * self.half_l / self.anti_q) + self.half_l).mul_(self._basis).sum(dim=-1).to(torch.int32)
        
    def quantize2index(self, z):
        return z.mul_(self._inv_T_raw).tanh_().mul_(self.half_l).round().add_(self.half_l).mul_(self._basis).sum(dim=-1).to(torch.int32)

class CodebookMeter:
    """Computes Codebook utilization using PyTorch tensors."""
    
    def __init__(self, codebook_size):
        self.codebook_size = codebook_size
        # self.device = device
        self.reset()

    def reset(self):
        self.register_mask = torch.zeros(self.codebook_size, dtype=torch.bool)
        self.avg = 0.0
        self.sum = 0
        self.count = 0

    def update(self, indices: torch.Tensor):
        # indices = indices.to(self.device)
        # zero_based_indices = indices - 1
        unique_indices = torch.unique(indices)
        self.register_mask[unique_indices] = True
        
        current_utilization = torch.sum(self.register_mask).item() / self.codebook_size
        
        # 更新运行平均值 (可选，用于追踪历史平均利用率)
        self.sum += current_utilization
        self.count += 1
        self.avg = self.sum / self.count if self.count > 0 else 0

    @property
    def utilization(self):
        """返回当前累计的码本利用率"""
        return torch.sum(self.register_mask).item() / self.codebook_size

