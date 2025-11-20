import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from einops import rearrange, repeat, reduce, pack, unpack

from timm.layers import trunc_normal_


def choose_vq(vq_type, dic_n, dim, dic_dim, fsq_level=[3,3,3,3], fsq_Tinit=1, input_format='NLC'):
    if vq_type == 'vq':
            return VectorQuantizer(n_e=dic_n, channels_in=dim, channels_dim=dic_dim)
    elif vq_type == 'fsq_qde':
        return FSQ_Qscale_deQscale_equal(channels_in=dim, channels_dim=dic_dim, levels=fsq_level, T=fsq_Tinit)
    elif vq_type == 'fsq':
        return FSQ(channels_in=dim, channels_dim=dic_dim, levels=fsq_level)
    elif vq_type == 'fsq_q':
        return FSQ_Qscale(channels_in=dim, channels_dim=dic_dim, levels=fsq_level, T=fsq_Tinit)
    # elif vq_type == 'tfsqs':
    #     return FSQ_trainableT_scale(channels_in=dim, channels_dim=dic_dim, levels=fsq_level, T=fsq_Tinit)
    elif vq_type == 'fsq_qd':
        return FSQ_Qscale_deQscale(channels_in=dim, channels_dim=dic_dim, levels=fsq_level, T=fsq_Tinit, input_format=input_format)
    elif vq_type == 'bottleneck': # for ablation study
        return Bottleneck(channels_in=dim, channels_dim=dic_dim)
    else:
        raise RuntimeError('vq type not implemented')

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
    def __init__(self,channels_in, channels_dim=3, levels=[5,5,5]):
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
        self.token_wise_rep = False
        self.codebook_meter = CodebookMeter(codebook_size=self.codebook_size)
    def reparameterize(self):
        print('using FSQ reparameterize')
        self.token_wise_rep = True
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
        if random.random() < 0.005:
            indices = self.codes_to_indices(codes)
            unique_index=torch.unique(indices)
            num_feature = z.shape[0] * z.shape[1]
            print(f"ActivatedCode:{unique_index.shape[0]}, NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}")

        if not self.training and self.token_wise_rep:
            indices = self.codes_to_indices(codes)
            self.codebook_meter.update(indices)
            return indices
        else:
            z_q = self.expand(codes)
            # return z_q , quantization_error
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
class Bottleneck(nn.Module):
    """
    only Linear Bottleneck w/o VQ, for ablation study
    """
    def __init__(self,channels_in, channels_dim=3):
        super().__init__()
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)

    def forward(self, z):
        '''
        z: (b, h , channels_in)
        '''
        z = self.compress(z) # (b, h , dim)
        z_q = self.expand(z)
        return z_q , torch.tensor(0.0).cuda()

import numpy as np
def analyze_tensor(x, bins=10, name="tensor"):
    """
    分析张量的统计信息，并打印结果。
    
    Args:
        x (torch.Tensor): 输入张量。
        bins (int): 直方图的分桶数量（用于统计值分布）。
        name (str): 张量名称（用于打印标识）。
    """
    if not isinstance(x, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor.")
    
    # 最大值、最小值、均值、标准差
    x_np = x.detach().cpu().numpy()  # 转为NumPy数组（脱离计算图）
    max_val = np.max(x_np)
    min_val = np.min(x_np)
    mean_val = np.mean(x_np)
    std_val = np.std(x_np)
    
    # 直方图统计（分桶区间）
    hist, bin_edges = np.histogram(x_np, bins=bins)
    bin_ranges = [f"[{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f})" for i in range(bins)]
    
    # 打印结果
    print(f"\n===== Tensor Analysis: {name} =====")
    print(f"Shape: {x.shape}")
    print(f"Max: {max_val:.6f}, Min: {min_val:.6f}, Mean: {mean_val:.6f}, Std: {std_val:.6f}")
    print("\nValue Distribution (Histogram):")
    for i in range(bins):
        print(f"Bin {i+1}: {bin_ranges[i]} -> {hist[i]} values ({hist[i] / x.numel() * 100:.2f}%)")
    
    # 可选：返回统计结果（如需进一步处理）
    stats = {
        "max": max_val,
        "min": min_val,
        "mean": mean_val,
        "std": std_val,
        "histogram": (hist, bin_edges)
    }
    return stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


#################################### rotation trick utils ####################################
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

class FSQ_Qscale_deQscale_equal(nn.Module):
    '''
    Based on vanilla FSQ, dynamic trainable quantization scale have been added.
    '''
    def __init__(self, channels_in, channels_dim, levels=[15,15,15], T=0):
        super().__init__()
        print(f"Using FSQ_trainableT, T init= {T}")
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        assert len(levels) == channels_dim
        self.codebook_dim = len(levels)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32))
        self.register_buffer("codebook_size", torch.tensor(self._levels.prod(), dtype=torch.int32))
        # self.T_raw = nn.Parameter(torch.tensor(T, dtype=torch.float32))
        self.T_raw = nn.Parameter(torch.tensor([T for _ in range(self.codebook_dim)], dtype=torch.float32)) 
        # self.anti_q = nn.Parameter(torch.tensor([T for _ in range(self.codebook_dim)], dtype=torch.float32)) 
        basis = torch.cumprod(
                torch.tensor([1] + levels[:-1], dtype=torch.int32), 
                dim=0
            )
        self.register_buffer("_basis", basis)
        self.token_wise_rep = False

        self.codebook_meter = CodebookMeter(codebook_size=self.codebook_size.item())

    def grad_scale(self, x, scale=1):
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad
    def get_scale(self):
        return self.grad_scale(self.T_raw, scale=1)
        # return self.grad_scale(self.T_raw, scale=10)
    
    def reparameterize(self):
        print('using FSQ_trainableT reparameterize')
        self.token_wise_rep = True
        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size).to(self.codebook_size.device))
        expand_dict = self.expand(implicit_codebook)
        del self.expand
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
            # self.codebook_meter.update(indices)
            unique_index=torch.unique(indices)
            num_feature = z.shape[0] * z.shape[1]
            # print(f"NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}")
            # print(f"ActivatedCode:{unique_index.shape[0]}, NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}, T={self.get_scale().data} AQ={self.anti_q.data}")
            print(f"ActivatedCode:{unique_index.shape[0]}, NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}, T={self.get_scale().data}")
        
        if not self.training and self.token_wise_rep:
            indices = self.codes_to_indices(codes)
            self.codebook_meter.update(indices)
            return indices
        else:
            z_q = self.expand(codes)
            # return z_q , quantization_error
            return z_q , torch.tensor(0.0).cuda()
    
    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width * self.get_scale()

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes
    
    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width / self.get_scale()) + half_width
    
    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def round_ste(self, z: torch.Tensor) ->  torch.Tensor:
        """Round with straight through gradients."""
        zhat = z.round()
        return z + (zhat - z).detach() # STE

    def round_rotation(self, z: torch.Tensor) ->  torch.Tensor:
        """Round with straight through gradients."""
        zhat = z.round()
        return rotate_to(z, zhat)  # rotation trick

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = ((self._levels - 1) * (1 + eps) / 2)
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0).to(z.device)
        shift = torch.atanh(offset / half_l)
        z_bound = torch.tanh(z/self.get_scale() + shift) * half_l - offset
        return z_bound
    
    def quantize(self, z):
        # print("z shape :",z.shape)
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = self.round_ste(self.bound(z)).to(z.device)
        # quantized = self.round_rotation(self.bound(z)).to(z.device)
        half_width = (self._levels // 2).to(z.device)# Renormalize to [-T, T].
        # return quantized / half_width *self.anti_q
        return quantized / half_width *self.get_scale()
        # return quantized / half_width
class FSQ_Qscale_deQscale(nn.Module):
    '''
    Based on vanilla FSQ, quantization scaling factor & dequantization scaling factor have been added.
    '''
    def __init__(self, channels_in, channels_dim, levels=[15,15,15], T=1, input_format='NLC'):
        super().__init__()
        print(f"Using FSQ_trainableT, T init= {T}")
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        assert len(levels) == channels_dim
        self.codebook_dim = len(levels)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32))
        self.register_buffer("codebook_size", torch.tensor(self._levels.prod(), dtype=torch.int32))
        # self.T_raw = nn.Parameter(torch.tensor(T, dtype=torch.float32))
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
    def grad_scale(self, x, scale=1):
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad
    def get_scale(self):
        return self.grad_scale(self.T_raw, scale=1)
        # return self.grad_scale(self.T_raw, scale=10)
    
    def reparameterize(self):
        print('using FSQ_trainableT reparameterize')
        self.token_wise_rep = True
        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size).to(self.codebook_size.device))
        expand_dict = self.expand(implicit_codebook)
        del self.expand
        return expand_dict
    
    def forward(self, z):
        '''
        z: (b, h , channels_in)
        '''
        rand = random.random()
        input = z
        if self.input_format == 'NCHW':
            z = z.flatten(2).transpose(1, 2).contiguous() #->(N, L, C)
        elif self.input_format == 'NHWC':
            z = z.flatten(1, 2)  # (N, H, W, C) -> (N, L, C)
        # print("z shape :",z.shape)
        z = self.compress(z) # (b, h , dim)
        codes = self.quantize(z) # range (-T,T)
        quantization_error = torch.mean((z-codes.detach())**2)
        if rand < 0.0005:
            # analyze_tensor(input, name="input")
            # analyze_tensor(z, name="compress")
            # analyze_tensor(codes, name="codes")
            indices = self.codes_to_indices(codes)
            # self.codebook_meter.update(indices)
            unique_index=torch.unique(indices)
            num_feature = z.shape[0] * z.shape[1]
            # print(f"NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}")
            print(f"ActivatedCode:{unique_index.shape[0]}, NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}, T={self.get_scale().data} AQ={self.anti_q.data}")
        
        if not self.training and self.token_wise_rep:
            indices = self.codes_to_indices(codes)
            self.codebook_meter.update(indices)
            return indices
        else:
            z_q = self.expand(codes)
            if self.input_format == 'NCHW':
                z_q = z_q.transpose(1, 2).view(input.shape).contiguous()
            elif self.input_format == 'NHWC':
                z_q = z_q.view(input.shape).contiguous()
            # return z_q , quantization_error
            return z_q , torch.tensor(0.0).cuda()
    
    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width * self.anti_q

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes
    
    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width / self.anti_q) + half_width
    
    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def round_ste(self, z: torch.Tensor) ->  torch.Tensor:
        """Round with straight through gradients."""
        zhat = z.round()
        return z + (zhat - z).detach() # STE

    def round_rotation(self, z: torch.Tensor) ->  torch.Tensor:
        """Round with straight through gradients."""
        zhat = z.round()
        return rotate_to(z, zhat)  # rotation trick

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = ((self._levels - 1) * (1 + eps) / 2)
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0).to(z.device)
        shift = torch.atanh(offset / half_l)
        z_bound = torch.tanh(z/self.get_scale() + shift) * half_l - offset
        return z_bound
    
    def quantize(self, z):
        # print("z shape :",z.shape)
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = self.round_ste(self.bound(z)).to(z.device)
        # quantized = self.round_rotation(self.bound(z)).to(z.device)
        half_width = (self._levels // 2).to(z.device)# Renormalize to [-T, T].
        return quantized / half_width *self.anti_q
class FSQ_Qscale(nn.Module):
    '''
    Based on vanilla FSQ, quantization scaling factor & dequantization scaling factor have been added.
    '''
    def __init__(self, channels_in, channels_dim, levels=[15,15,15], T=0):
        super().__init__()
        print(f"Using FSQ_trainableT, T init= {T}")
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        assert len(levels) == channels_dim
        self.codebook_dim = len(levels)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32))
        self.register_buffer("codebook_size", torch.tensor(self._levels.prod(), dtype=torch.int32))
        # self.T_raw = nn.Parameter(torch.tensor(T, dtype=torch.float32))
        self.T_raw = nn.Parameter(torch.tensor([T for _ in range(self.codebook_dim)], dtype=torch.float32)) 
        basis = torch.cumprod(
                torch.tensor([1] + levels[:-1], dtype=torch.int32), 
                dim=0
            )
        self.register_buffer("_basis", basis)
        self.token_wise_rep = False

        self.codebook_meter = CodebookMeter(codebook_size=self.codebook_size.item())

    def grad_scale(self, x, scale=1):
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad
    def get_scale(self):
        return self.grad_scale(self.T_raw, scale=1)
        # return self.grad_scale(self.T_raw, scale=10)
    
    def reparameterize(self):
        print('using FSQ_trainableT reparameterize')
        self.token_wise_rep = True
        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size).to(self.codebook_size.device))
        expand_dict = self.expand(implicit_codebook)
        del self.expand
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
            # self.codebook_meter.update(indices)
            unique_index=torch.unique(indices)
            num_feature = z.shape[0] * z.shape[1]
            # print(f"NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}")
            print(f"ActivatedCode:{unique_index.shape[0]}, NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}, T={self.get_scale().data}")
        
        if not self.training and self.token_wise_rep:
            indices = self.codes_to_indices(codes)
            self.codebook_meter.update(indices)
            return indices
        else:
            z_q = self.expand(codes)
            # return z_q , quantization_error
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
        return (zhat_normalized * half_width ) + half_width
    
    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def round_ste(self, z: torch.Tensor) ->  torch.Tensor:
        """Round with straight through gradients."""
        zhat = z.round()
        return z + (zhat - z).detach() # STE

    def round_rotation(self, z: torch.Tensor) ->  torch.Tensor:
        """Round with straight through gradients."""
        zhat = z.round()
        return rotate_to(z, zhat)  # rotation trick

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = ((self._levels - 1) * (1 + eps) / 2)
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0).to(z.device)
        shift = torch.atanh(offset / half_l)
        z_bound = torch.tanh(z/self.get_scale() + shift) * half_l - offset
        return z_bound
    
    def quantize(self, z):
        # print("z shape :",z.shape)
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = self.round_ste(self.bound(z)).to(z.device)
        # quantized = self.round_rotation(self.bound(z)).to(z.device)
        half_width = (self._levels // 2).to(z.device)# Renormalize to [-T, T].
        return quantized / half_width




class FSQ_trainableT_scale(nn.Module):
    def __init__(self, channels_in, channels_dim, levels=[15,15,15], T=0):
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
        self.T_raw = nn.Parameter(torch.tensor([T for _ in range(self.codebook_dim)], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.ones(channels_dim, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(channels_dim, dtype=torch.float32))

        basis = torch.cumprod(
                torch.tensor([1] + levels[:-1], dtype=torch.int32), 
                dim=0
            )
        self.register_buffer("_basis", basis)

        self.token_wise_rep = False
    def get_scale(self):
        return self.T_raw
    def reparameterize(self):
        print('using FSQ_trainableT reparameterize')
        self.token_wise_rep = True
        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size).to(self.codebook_size.device))
        expand_dict = self.expand(self.alpha*implicit_codebook+self.beta)
        del self.expand
        self.register_buffer("T_softplus", self.get_scale())
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
            print(f"ActivatedCode:{unique_index.shape[0]}, NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}, T={self.get_scale().data}, alpha={self.alpha.data}, beta={self.beta.data}")
        
        if not self.training and self.token_wise_rep:
            indices = self.codes_to_indices(codes)
            return indices
        else:
            z_q = self.expand(codes)
            # return z_q , quantization_error
            return z_q , torch.tensor(0.0).cuda()
    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        if not self.training and self.token_wise_rep:
            return (zhat - half_width) / half_width *self.alpha+self.beta
        else:
            return (zhat - half_width) / half_width *self.alpha+self.beta

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes
    
    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        if not self.training and self.token_wise_rep:
            return ((zhat_normalized-self.beta)/self.alpha / half_width) + half_width
        return ((zhat_normalized-self.beta)/self.alpha / half_width) + half_width
    
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
        if not self.training and self.token_wise_rep:
            z_bound = torch.tanh(z/self.T_softplus + shift) * half_l - offset
        else:
            z_bound = torch.tanh(z/self.get_scale() + shift) * half_l - offset
        return z_bound
    
    def quantize(self, z):
        # print("z shape :",z.shape)
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = self.round_ste(self.bound(z)).to(z.device)
        half_width = (self._levels // 2).to(z.device)# Renormalize to [-T, T].
        if not self.training and self.token_wise_rep:
            return quantized / half_width * self.alpha+self.beta
        return quantized / half_width * self.alpha+self.beta

if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    levels_=[19,19,19]   #codebook num =  降低37.9% error
    levels_=[17,17,17]   #codebook num =  降低37.9% error
    levels_=[15]   #codebook num =  降低37.9% error
    step = 0.02
    std_range = torch.arange(0.1, 1.7, step)  # x轴：std从0.1到1.6
    t_range = torch.arange(0.1, 3.1, step)    # y轴：T从0.1到3.0

    # 存储误差矩阵（行为T，列为std）
    error_matrix = torch.zeros(len(t_range), len(std_range))


    for i, t in enumerate(t_range):
        for j, std in enumerate(std_range):
            input = 0.001 + std * torch.randn(1, 100000, 1)
            input = torch.clamp(input, min=-10, max=10)
            vq = FSQ_T(n_e=0, channels_in=5, channels_dim=1, levels=levels_, T=t.item())
            output1 = vq.quantize(input)
            error = torch.abs(output1 - input).sum()
            error_matrix[i, j] = error

    # 转换为NumPy数组
    std_values = std_range.numpy()
    t_values = t_range.numpy()
    error_values = error_matrix.numpy()

    max_error_per_std = np.max(error_values, axis=0)
    error_ratio_matrix = error_values / max_error_per_std
    # 找到每个std对应的最小error的T
    min_t_indices = np.argmin(error_ratio_matrix, axis=0)  # 沿T轴找最小值索引
    min_t_values = t_values[min_t_indices]           # 对应的T值

    # 绘制热力图
    plt.figure(figsize=(10, 6))
    from matplotlib.colors import PowerNorm

    heatmap = plt.imshow(
        error_ratio_matrix,
        cmap='viridis',
        norm=PowerNorm(gamma=0.42),  # 中间值颜色拉伸
        # cmap='viridis',
        extent=[std_values.min(), std_values.max(), t_values.min(), t_values.max()],
        aspect='auto',
        origin='lower'
    )
    plt.colorbar(heatmap, label='Error Ratio (error / max_error_for_std)')
    plt.xlabel('Noise Std (std)')
    plt.ylabel('Temperature (T)')
    plt.title('Quantization Error vs. Std and T')

    # 绘制最小误差折线（红色虚线）
    plt.plot(
        std_values, 
        min_t_values, 
        'r', 
        linewidth=2, 
        markersize=8,
        label='Optimal T for min error'
    )
    plt.legend()

    # 保存图像（支持PNG/PDF/SVG等格式）
    output_path = "./VQViT/FSQfig/quantization_error_heatmap.png"  # 修改为你的保存路径
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # dpi控制分辨率
    print(f"图像已保存至: {output_path}")
    # from sklearn.decomposition import PCA
    # std = 1.5
    # input = 0.001 + std * torch.randn(512, 50, 384)
    # input = input.cpu()
    # samples = rearrange(input, 'b h d ->(b h) d') 
    # x_np = samples.numpy() if isinstance(samples, torch.Tensor) else samples
    # codebook_dim = 4
    # pca = PCA(n_components=codebook_dim)
    # x_reduced = pca.fit_transform(x_np)
    # centroids = torch.from_numpy(x_reduced)
    # output = rearrange(centroids, '(b h) d -> b h d', b=input.shape[0], h=input.shape[1])
    # print(output.shape)
    # print(f'mean: {output.mean()}, std: {output.std()}')
