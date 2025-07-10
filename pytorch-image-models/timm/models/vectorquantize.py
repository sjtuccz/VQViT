import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from einops import rearrange, repeat, reduce, pack, unpack

from timm.layers import trunc_normal_
class TokenToImageToToken(nn.Module):
    def __init__(self, token_dim=768, image_size=224, patch_size=16, out_channels=2):
        super().__init__()
        self.token_dim = token_dim
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2  # N = H*W / (P*P)
        # 3x3 卷积降维至 2 通道
        self.conv = nn.Conv2d(
            in_channels=token_dim,
            out_channels=out_channels,  # 降维至 2 通道
            kernel_size=3,
            stride=1,
            padding=1,  # 保持空间尺寸不变
        )
        self.proj = nn.Linear(token_dim, out_channels)

    def forward(self, tokens):
        """
        输入: tokens (B, N+1, D) 或 (B, N, D)
        输出: new_tokens (B, N, 2) 或 (B, N+1, 2)
        """
        B, num_tokens, D = tokens.shape
        # 可选：去掉 [CLS] token（假设它是第 0 个 token）
        if num_tokens == self.num_patches + 1:
            cls_token = tokens[:, 0, :]  # (B, D)
            tokens = tokens[:, 1:, :]    # (B, N, D)
        else:
            cls_token = None
        # 1. Token → 图像 (B, N, D) → (B, D, H, W)
        H = W = int(self.num_patches**0.5)  # 假设 N = H*W
        x = tokens.transpose(1, 2)  # (B, D, N)
        x = x.view(B, D, H, W)      # (B, D, H, W)
        # 2. 3x3 卷积降维 (B, D, H, W) → (B, 2, H, W)
        x = self.conv(x)  # (B, 2, H, W)
        # 3. 图像 → Token (B, 2, H, W) → (B, N, 2)
        x = x.flatten(2).transpose(1, 2)  # (B, N, 2)
        if cls_token is not None and self.proj:
            cls_token = cls_token.unsqueeze(1)  # (B, 1, D)
            # 注意：如果原始 token_dim (D) 和降维后的 2 不匹配，需调整 cls_token
            # 例如用线性层投影 cls_token 到 2 维：
            cls_token = self.proj(cls_token)  # (B, 1, 2)
            x = torch.cat([cls_token, x], dim=1)     # (B, N+1, 2)
        return x

def orthogonal_loss_fn(t, unique_index=None):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n, d = t.shape[:2]
    if unique_index:
        mask = ~torch.notisin(torch.arange(n).to(t.device), unique_index)
        t[mask]=0
    normed_codes = l2norm(t)
    cosine_sim = torch.einsum('i d, j d -> i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (n ** 2) - (1 / n)
# self.vq = FSQ(dic_n, dim, dic_dim, index)
class FSQ(nn.Module):
    def __init__(self, n_e, channels_in, channels_dim, index=0, levels=[8,7,7,7]):
        super().__init__()
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        assert len(levels) == channels_dim
        self.codebook_dim = len(levels)
        # self._levels = torch.tensor(levels, dtype=torch.int32)
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
        # input = z
        z = self.compress(z) # (b, h , dim)
        # print(f"z shape: {z.shape}")
        codes = self.quantize(z)
        quantization_error = torch.mean((z-codes.detach())**2)
        if random.random() < 0.0003:
            # quantization_error = (z - codes).abs().mean()
            indices = self.codes_to_indices(codes)
            unique_index=torch.unique(indices)
            num_feature = z.shape[0] * z.shape[1]
            print(f"ActivatedCode:{unique_index.shape[0]}, NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}")
        if not self.training and self.is_pre_cal:
            indices = self.codes_to_indices(codes)
            return indices
        else:
            z_q = self.expand(codes)
            # return z_q , torch.tensor(0.0).cuda()
            return z_q , quantization_error
    
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
        half_l = ((self._levels - 1) * (1 + eps) / 2).to(z.device)
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0).to(z.device)
        shift = (offset / half_l).atanh()
        z_bound = (z + shift).tanh() * half_l - offset
        # print(f"   z_bound: {z_bound.shape}")
        return z_bound
    
    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = self.round_ste(self.bound(z)).to(z.device)
        half_width = (self._levels // 2).to(z.device)# Renormalize to [-1, 1].
        return quantized / half_width
    

class FSQ_T(nn.Module):
    def __init__(self, n_e, channels_in, channels_dim, index=0, levels=[15,15,15], T=1):
        super().__init__()
        print(f"Using FSQ_T,T= {T}")

        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        assert len(levels) == channels_dim
        self.codebook_dim = len(levels)
        # self._levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32))
        # self.codebook_size = self._levels.prod().item()
        self.register_buffer("codebook_size", self._levels.prod())
        self.register_buffer("T", torch.tensor(T, dtype=torch.int32))
        basis = torch.cumprod(
                torch.tensor([1] + levels[:-1], dtype=torch.int32), 
                dim=0
            )
        self.register_buffer("_basis", basis)

        self.is_pre_cal = False
    def reparameterize(self):
        print('using FSQ reparameterize')
        self.is_pre_cal = True
        print(f"codebook_size: {self.codebook_size}")
        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
        print(f"implicit_codebook: {implicit_codebook.shape}")
        # implicit_codebook = torch.tensor(implicit_codebook).to(self.expand.weight.device)
        expand_dict = self.expand(implicit_codebook)
        del self.expand
        return expand_dict
    
    def forward(self, z):
        '''
        z: (b, h , channels_in)
        '''
        input = z
        z = self.compress(z) # (b, h , dim)
        # print(f"z shape: {z.shape}")
        codes = self.quantize(z) # 输出范围(-T,T)
        quantization_error = torch.mean((z-codes.detach())**2)
        if random.random() < 0.0003:
            
            indices = self.codes_to_indices(codes)
            unique_index=torch.unique(indices)
            num_feature = z.shape[0] * z.shape[1]
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
        return (zhat - half_width) / half_width *self.T

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes
    
    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized / self.T * half_width) + half_width
    
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
        z_bound = torch.tanh(z/self.T + shift) * half_l - offset
        return z_bound
    
    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = self.round_ste(self.bound(z)).to(z.device)
        half_width = (self._levels // 2).to(z.device)# Renormalize to [-T, T].
        return quantized / half_width * self.T
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

class FSQ_GumbelSoftmax(nn.Module):
    def __init__(self, n_e, channels_in, channels_dim, levels=[15, 15, 15], tau=1.0, hard=True):
        super().__init__()
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        
        self.levels = torch.tensor(levels, dtype=torch.int32)
        self.codebook_size = self.levels.prod()
        self.codebook_dim = len(levels)
        self.n_e = n_e
        
        # 初始化所有维度的候选值（并行存储）
        # codebook: (codebook_dim, max_level), 不足的用0填充（通过mask处理）
        max_level = max(levels)
        self.codebook = nn.Parameter(
            torch.stack([
                F.pad(torch.linspace(-1.0, 1.0, level), (0, max_level - level), value=0)
                for level in levels
            ])
        )  # shape: (codebook_dim, max_level)
        
        # 创建mask，标记有效候选位置
        self.register_buffer("codebook_mask", 
            torch.stack([
                F.pad(torch.ones(level), (0, max_level - level), value=0)
                for level in levels
            ]).bool()
        )  # shape: (codebook_dim, max_level)
        
        # Gumbel-Softmax 参数
        self.tau = tau
        self.hard = hard
        
        # 预计算基础权重（用于索引计算）
        basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.int32), dim=0)
        self.register_buffer("_basis", basis)

    def forward(self, z):
        z = self.compress(z)  # (batch_size, seq_len, channels_dim)
        codes, indices = self.quantize(z)
        quantization_error = torch.mean((z - codes)**2)
        z_q = self.expand(codes)
        return z_q, quantization_error

    def quantize(self, z):
        """
        向量化实现 Gumbel-Softmax 量化
        z: (batch_size, seq_len, channels_dim)
        """
        batch_size, seq_len, _ = z.shape
        max_level = self.codebook.size(1)
        
        # 计算所有维度的距离矩阵
        # z_expand: (batch_size, seq_len, channels_dim, 1)
        # codebook: (1, 1, channels_dim, max_level)
        distances = torch.abs(
            z.unsqueeze(-1) - self.codebook.unsqueeze(0).unsqueeze(0)
        )  # shape: (batch_size, seq_len, channels_dim, max_level)
        
        # 应用mask，无效位置填充大数
        distances = torch.where(
            self.codebook_mask.unsqueeze(0).unsqueeze(0),
            distances,
            torch.tensor(float('inf'), device=z.device)
        )
        
        # Gumbel-Softmax 采样
        logits = -distances
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10))
        noisy_logits = (logits + gumbel_noise) / self.tau
        probs = F.softmax(noisy_logits, dim=-1)  # (batch_size, seq_len, channels_dim, max_level)
        
        # 采样结果
        if self.hard and not self.training:
            # 硬性采样（推理）
            indices = torch.argmax(probs, dim=-1)  # (batch_size, seq_len, channels_dim)
            codes = torch.gather(
                self.codebook.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1),
                dim=-1,
                index=indices.unsqueeze(-1)
            ).squeeze(-1)  # (batch_size, seq_len, channels_dim)
        else:
            # 软性采样（训练）
            if self.hard:
                # 直通梯度技巧
                indices = torch.argmax(probs, dim=-1)
                one_hot = F.one_hot(indices, num_classes=max_level).float()
                codes = torch.einsum('b s d l, d l -> b s d', one_hot, self.codebook)
                probs = (one_hot - probs).detach() + probs
            else:
                # 加权平均
                codes = torch.einsum('b s d l, d l -> b s d', probs, self.codebook)
                indices = torch.argmax(probs, dim=-1)
        
        return codes, indices

    def codes_to_indices(self, codes):
        """向量化索引查找"""
        distances = torch.abs(
            codes.unsqueeze(-1) - self.codebook.unsqueeze(0).unsqueeze(0)
        )  # (batch_size, seq_len, channels_dim, max_level)
        distances = torch.where(
            self.codebook_mask.unsqueeze(0).unsqueeze(0),
            distances,
            torch.tensor(float('inf'), device=codes.device)
        )
        indices = torch.argmin(distances, dim=-1)
        return indices

    def indices_to_codes(self, indices):
        """向量化候选值查找"""
        return torch.gather(
            self.codebook.unsqueeze(0).unsqueeze(0).expand(indices.size(0), indices.size(1), -1, -1),
            dim=-1,
            index=indices.unsqueeze(-1)
        ).squeeze(-1)

class FSQ_trainableT(nn.Module):
    def __init__(self, n_e, channels_in, channels_dim, index=0, levels=[15,15,15], T=0, T_min=0.1, T_max=5.0):
        super().__init__()
        print(f"Using FSQ_trainableT, T init= {T}, T_max= {T_max}")

        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        assert len(levels) == channels_dim
        self.codebook_dim = len(levels)
        # self._levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32))
        self.register_buffer("codebook_size", torch.tensor(self._levels.prod(), dtype=torch.int32))
        # self.codebook_size = self._levels.prod().item()
        self.register_buffer("T_min", torch.tensor(T_min, dtype=torch.float))
        self.register_buffer("T_max", torch.tensor(T_max, dtype=torch.float))
        # self.T_raw = nn.Parameter(torch.tensor(T, dtype=torch.float32))
        self.T_raw = nn.Parameter(torch.tensor([T for _ in range(self.codebook_dim)], dtype=torch.float32))


        basis = torch.cumprod(
                torch.tensor([1] + levels[:-1], dtype=torch.int32), 
                dim=0
            )
        self.register_buffer("_basis", basis)

        self.is_pre_cal = False

    def get_T(self):
        # 后续可以改进该部分代码，在推理阶段直接获取T_sigmoid，不要重复计算。
        # Sigmoid映射到[T_min, T_max]
        # T_sigmoid =  self.T_max * torch.sigmoid(self.T_raw)
        # T_sigmoid = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self.T_raw)
        T_sigmoid =  F.softplus(self.T_raw) 
        if random.random() < 0.0001:
            print(f"FSQ T_sigmoid: {T_sigmoid}")
        return T_sigmoid
    
    def reparameterize(self):
        print('using FSQ_trainableT reparameterize')
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
        rand = random.random()


        input = z
        z = self.compress(z) # (b, h , dim)
        # print(f"z shape: {z.shape}")
        codes = self.quantize(z) # 输出范围(-T,T)
        # quantization_error = torch.mean((z-codes)**2)
        quantization_error = torch.mean((z-codes.detach())**2)
        # codes  = codes.detach()+z-z.detach()
        # if quantization_error>1.0:
        if rand < 0.0005:
            # analyze_tensor(input, name="input")
            # analyze_tensor(z, name="compress")
            # analyze_tensor(codes, name="codes")
            indices = self.codes_to_indices(codes)
            unique_index=torch.unique(indices)
            num_feature = z.shape[0] * z.shape[1]
            print(f"ActivatedCode:{unique_index.shape[0]}, NumFeature: {num_feature}, CodebookSize: {self.codebook_size}, QuantiError: {quantization_error}")
        # quantization_error = (z - codes).abs().mean()
        
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
        T_sigmoid = self.get_T()
        return (zhat - half_width) / half_width *T_sigmoid

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes
    
    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        T_sigmoid = self.get_T()
        return (zhat_normalized / T_sigmoid * half_width) + half_width
    
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
        T_sigmoid = self.get_T()
        z_bound = torch.tanh(z/T_sigmoid + shift) * half_l - offset
        return z_bound
    
    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = self.round_ste(self.bound(z)).to(z.device)
        half_width = (self._levels // 2).to(z.device)# Renormalize to [-T, T].
        T_sigmoid = self.get_T()
        return quantized / half_width * T_sigmoid
    
class FSQ_AdaptiveQuant(nn.Module):
    def __init__(self, n_e, channels_in, channels_dim, levels=[15,15,15], bandwidth=0.5):
        super().__init__()
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        self.levels = torch.tensor(levels, dtype=torch.int32)
        self.bandwidth = bandwidth  # KDE带宽参数
        self.codebook_dim = len(levels)
        
        # 初始化动态量化参数
        self.quant_centers = nn.ParameterList([
            nn.Parameter(torch.linspace(-1, 1, levels[i])) 
            for i in range(self.codebook_dim)
        ])

    def kde_estimate(self, z):
        """核密度估计：计算每个维度的数据密度分布"""
        # z: (batch, seq_len, dim)
        z = z.reshape(-1, self.codebook_dim)  # 展平为 (N, dim)
        kde_weights = []
        for d in range(self.codebook_dim):
            samples = z[:, d].unsqueeze(0)  # (1, N)
            centers = self.quant_centers[d].unsqueeze(1)  # (levels[d], 1)
            # 用高斯核计算密度权重
            weights = torch.exp(-0.5 * ((centers - samples) / self.bandwidth) **2)  # (levels[d], N)
            weights = weights.mean(dim=1)  # (levels[d],)
            kde_weights.append(weights)
        return kde_weights  # List[tensor(levels[d])]

    def adaptive_quantize(self, z, kde_weights):
        """非均匀量化：根据密度调整量化间隔"""
        quantized = []
        for d in range(self.codebook_dim):
            # 动态调整量化中心（密度高的区域中心更密集）
            centers = self.quant_centers[d]  # (levels[d],)
            weights = kde_weights[d] + 1e-6  # 避免除零
            # 密度加权后的量化步长
            scaled_centers = centers * (1.0 / weights.sum()) * weights
            # 最近邻量化（STE）
            dist = torch.abs(z[..., d].unsqueeze(-1) - scaled_centers.unsqueeze(0))  # (..., levels[d])
            _, idx = torch.min(dist, dim=-1)
            q = scaled_centers[idx]
            quantized.append(q)
        return torch.stack(quantized, dim=-1)  # (..., dim)

    def forward(self, z):
        z_compressed = self.compress(z)  # (b, h, dim)
        kde_weights = self.kde_estimate(z_compressed)  # 密度估计
        codes = self.adaptive_quantize(z_compressed, kde_weights)
        codes= codes.reshape(z_compressed.shape)  # 自适应量化
        # codes = codes.detach() + z_compressed - z_compressed.detach()  # STE
        z_q = self.expand(codes)
        return z_q, torch.tensor(0.0).to(z.device)
class FSQ_AdaptiveQuant_todo_fix(nn.Module):
    def __init__(self, n_e, channels_in, channels_dim, levels=[15,15,15], bandwidth=0.5):
        super().__init__()
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        self.levels = torch.tensor(levels, dtype=torch.int32)
        self.bandwidth = bandwidth
        self.codebook_dim = len(levels)
        
        # 初始化动态量化参数（填充到最大level长度以便向量化）
        self.max_level = max(levels)
        padded_centers = []
        for i in range(self.codebook_dim):
            centers = torch.linspace(-1, 1, levels[i])
            # 用NaN填充不足部分（后续计算会忽略）
            padded = F.pad(centers, (0, self.max_level - levels[i]), value=float('nan'))
            padded_centers.append(padded)
        self.quant_centers = nn.Parameter(torch.stack(padded_centers))  # (codebook_dim, max_level)

    def kde_estimate(self, z):
        """完全向量化的核密度估计"""
        # z: (batch, seq_len, dim) -> (N, dim)
        z_flat = z.reshape(-1, self.codebook_dim)
        
        # 计算高斯核密度
        centers = self.quant_centers.unsqueeze(0)  # (1, dim, max_level)
        samples = z_flat.unsqueeze(-1)           # (N, dim, 1)
        weights = torch.exp(-0.5 * ((centers - samples) / self.bandwidth) **2)
        
        # 创建掩码忽略NaN填充部分
        mask = ~torch.isnan(self.quant_centers).unsqueeze(0)
        weights = weights * mask.float()
        
        # 按有效中心点数量归一化
        kde_weights = weights.sum(dim=0) / mask.sum(dim=0).float()  # (dim, max_level)
        return kde_weights

    def adaptive_quantize(self, z, kde_weights):
        """完全向量化的自适应量化"""
        # z: (b, s, dim)
        # kde_weights: (dim, max_level)
        
        # 动态调整中心点
        valid_mask = ~torch.isnan(self.quant_centers)
        weights_sum = kde_weights.sum(dim=1, keepdim=True)
        scaled_centers = torch.where(
            valid_mask,
            self.quant_centers * (1.0 / weights_sum) * kde_weights,
            torch.zeros_like(self.quant_centers)
        )
        
        # 并行计算最近邻
        dist = torch.abs(z.unsqueeze(-1) - scaled_centers.unsqueeze(0).unsqueeze(0))  # (b, s, dim, max_level)
        _, idx = torch.min(dist, dim=-1)  # (b, s, dim)
        
        # 收集量化值
        expanded_centers = scaled_centers.expand(z.size(0), z.size(1), *scaled_centers.shape)
        quantized = torch.gather(
            expanded_centers,  # (b, s, dim, max_level)
            dim=-1,
            index=idx.unsqueeze(-1)  # (b, s, dim, 1)
        ).squeeze(-1)
        
        return quantized.detach()+z-z.detach()  # (b, s, dim)

    def forward(self, z):
        z_compressed = self.compress(z)  # (b, h, dim)
        kde_weights = self.kde_estimate(z_compressed)  # (dim, max_level)
        codes = self.adaptive_quantize(z_compressed, kde_weights)
        z_q = self.expand(codes)
        return z_q, torch.tensor(0.0).to(z.device)
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
class VectorQuantizer_Sim(nn.Module):
    # Sim_vq
    def __init__(self, n_e, channels_in, channels_dim, index=0):
        super().__init__()
        self.embedding = nn.Embedding(n_e, channels_dim)
        trunc_normal_(self.embedding.weight,mean=0.0, std=1.0, a=-3.0, b=3.0)
        # self.compress = TokenToImageToToken(token_dim=384, image_size=224, patch_size=32, out_channels=2)
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        self.codebook_size = n_e
        self.is_pre_cal = False
        self.code_transform = nn.Linear(channels_dim, channels_dim)
    def reparameterize(self):
        print('using VQ reparameterize')
        self.is_pre_cal = True
        dict_embeding = self.code_transform(self.embedding.weight.data)
        expand_dict = self.expand(dict_embeding)
        del self.code_transform
        del self.expand
        return expand_dict
    def forward(self, z):
        input = z

        # embedding_data = z.detach().cpu().numpy()
        # mean = embedding_data.mean()
        # std = embedding_data.std()
        # min_val = embedding_data.min()
        # max_val = embedding_data.max()
        # print(f"   Mean: {mean}, Std: {std}, Min: {min_val}, Max: {max_val}")
        implicit_codebook = self.code_transform(self.embedding.weight.data) #更不更新字典
        z = self.compress(z)
        z_flattened = z.view(-1, z.shape[-1]) # bhw,c
        num_feature = z_flattened.shape[0]
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(implicit_codebook ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(implicit_codebook, 'n d -> d n'))
        # d = torch.cdist(z_flattened, self.embedding.weight, p=2)**2
#  F.mse_loss
# loss = F.mse_loss(z_q.detach(),z_flattened) + 0.5* F.mse_loss(z_q,z_flattened.detach())
# loss = F.mse_loss(z_q,z_flattened)
        indices = torch.argmin(d, dim=1)
        if not self.training and self.is_pre_cal:
            return indices
        unique_index=torch.unique(indices)
        z_q = implicit_codebook[indices]
        if self.training:
            # loss_dict = torch.mean((z_q - z_flattened) ** 2)
            # loss_dict = 0.1*torch.mean((z_q.detach()-z_flattened)**2) + 1.0* \
            # torch.mean((z_q - z_flattened.detach()) ** 2)
            loss_dict = torch.mean((z_q.detach()-z_flattened)**2) + 2.0* torch.mean((z_q - z_flattened.detach()) ** 2)
            # loss_dict = 2.0* torch.mean((z_q - z_flattened.detach()) ** 2)
            # z_q = z_q+ (z_flattened - z_flattened.data)
            z_q = z_q.detach()+ (z_flattened - z_flattened.detach())
            # z_q = rotate_to(z_flattened, z_q)
            z_q = z_q.view(z.shape)
            z_q = self.expand(z_q)
            # loss_dict = torch.mean((z_q - input) ** 2)
            # indices2 = torch.argmin(d, dim=0)
            # loss_dict2 = torch.mean((z_flattened[indices2, :].detach() - self.embedding.weight) ** 2)
            if random.random() < 0.0008:
                print(f"activated vectors:{unique_index.shape[0]} , num feature: {num_feature}, num dict: {self.codebook_size}")
            # loss_dict3=torch.mean((z_q-input)**2)
            # loss_dict3 = F.mse_loss(z_q, input)
            # loss = F.mse_loss(z_q.detach(),z_flattened) + 0.5* F.mse_loss(z_q,z_flattened.detach())
            # return z_q, loss_dict
            return z_q, loss_dict #+ orthogonal_loss_fn(self.embedding.weight)
            # return z_q, loss_dict + 0.5*loss_dict2
        else:
            z_q = z_q.view(z.shape)
            z_q = self.expand(z_q)
            if random.random() < 0.001:
                print(f"activated vectors:{unique_index.shape[0]} , num feature: {num_feature}, num dict: {self.codebook_size}")
            return z_q , torch.tensor(0.0).cuda()

class VectorQuantizer_LinearRebuild(nn.Module):
    def __init__(self, n_e, channels_in, channels_dim, index=0):
        super().__init__()
        self.embedding = nn.Embedding(n_e, channels_dim)
        trunc_normal_(self.embedding.weight,mean=0.0, std=1.0, a=-3.0, b=3.0)
        # self.compress = TokenToImageToToken(token_dim=384, image_size=224, patch_size=32, out_channels=2)
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        self.codebook_size = n_e
        self.is_pre_cal = False
    def reparameterize(self):
        print('using VQ reparameterize')
        self.is_pre_cal = True
        dict_embeding = self.embedding.weight.data.clone().detach()
        expand_dict = self.expand(dict_embeding)
        # del self.embedding
        del self.expand
        return expand_dict
    def forward(self, z):
        input = z

        # embedding_data = z.detach().cpu().numpy()
        # mean = embedding_data.mean()
        # std = embedding_data.std()
        # min_val = embedding_data.min()
        # max_val = embedding_data.max()
        # print(f"   Mean: {mean}, Std: {std}, Min: {min_val}, Max: {max_val}")
        
        z = self.compress(z)
        z_flattened = z.view(-1, z.shape[-1]) # bhw,c
        num_feature = z_flattened.shape[0]
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        # d = torch.cdist(z_flattened, self.embedding.weight, p=2)**2
#  F.mse_loss
# loss = F.mse_loss(z_q.detach(),z_flattened) + 0.5* F.mse_loss(z_q,z_flattened.detach())
# loss = F.mse_loss(z_q,z_flattened)
        indices = torch.argmin(d, dim=1)
        if not self.training and self.is_pre_cal:
            return indices
        unique_index=torch.unique(indices)
        z_q_embedding = self.embedding(indices)
        if self.training:
            # loss_dict = torch.mean((z_q - z_flattened) ** 2)
            # loss_dict = 0.1*torch.mean((z_q.detach()-z_flattened)**2) + 1.0* \
            # torch.mean((z_q - z_flattened.detach()) ** 2)
            # loss_dict = torch.mean((z_q.detach()-z_flattened)**2) + 2.0* torch.mean((z_q - z_flattened.detach()) ** 2)
            # loss_dict = 2.0* torch.mean((z_q - z_flattened.detach()) ** 2)
            # z_q = z_q+ (z_flattened - z_flattened.data)
            # z_q = rotate_to(z_flattened, z_q)
            z_q = z_q_embedding.view(z.shape)
            z_q = self.expand(z_q)
            # loss_dict = torch.mean((z_q - input) ** 2)
            loss_dict = torch.mean((z_q - input.detach()) ** 2)
            z_q_embedding = z_q_embedding.detach()+ (z_flattened - z_flattened.detach())
            # loss_dict = torch.mean((z_q - input) ** 2)
            # indices2 = torch.argmin(d, dim=0)
            # loss_dict2 = torch.mean((z_flattened[indices2, :].detach() - self.embedding.weight) ** 2)
            if random.random() < 0.0008:
                print(f"activated vectors:{unique_index.shape[0]} , num feature: {num_feature}, num dict: {self.codebook_size}")
            # loss_dict3=torch.mean((z_q-input)**2)
            # loss_dict3 = F.mse_loss(z_q, input)
            # loss = F.mse_loss(z_q.detach(),z_flattened) + 0.5* F.mse_loss(z_q,z_flattened.detach())
            # return z_q, loss_dict
            return z_q, loss_dict #+ orthogonal_loss_fn(self.embedding.weight)
            # return z_q, loss_dict + 0.5*loss_dict2
        else:
            z_q = z_q_embedding.view(z.shape)
            z_q = self.expand(z_q)
            if random.random() < 0.001:
                print(f"activated vectors:{unique_index.shape[0]} , num feature: {num_feature}, num dict: {self.codebook_size}")
            return z_q , torch.tensor(0.0).cuda()


class VectorQuantizer_CosSim(nn.Module):
    def __init__(self, n_e, channels_in, channels_dim, index=0):
        super().__init__()
        self.embedding = nn.Embedding(n_e, channels_dim)
        trunc_normal_(self.embedding.weight,mean=0.0, std=1.0, a=-3.0, b=3.0)
        # self.compress = TokenToImageToToken(token_dim=384, image_size=224, patch_size=32, out_channels=2)
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        self.codebook_size = n_e
        self.is_pre_cal = False
    def reparameterize(self):
        print('using VQ reparameterize')
        self.is_pre_cal = True
        dict_embeding = self.embedding.weight.data.clone().detach()
        expand_dict = self.expand(dict_embeding)
        # del self.embedding
        del self.expand
        return expand_dict
    def forward(self, z):
        input = z

        # embedding_data = z.detach().cpu().numpy()
        # mean = embedding_data.mean()
        # std = embedding_data.std()
        # min_val = embedding_data.min()
        # max_val = embedding_data.max()
        # print(f"   Mean: {mean}, Std: {std}, Min: {min_val}, Max: {max_val}")
        
        z = self.compress(z)
        z_flattened = z.view(-1, z.shape[-1]) # bhw,c
        num_feature = z_flattened.shape[0]

        # 计算余弦相似度
        cos_sim = torch.matmul(z_flattened,self.embedding.weight.transpose(0, 1))                   # [batch_size, num_embeddings]
        indices = torch.argmax(cos_sim, dim=1) 
        if not self.training and self.is_pre_cal:
            return indices
        unique_index=torch.unique(indices)
        z_q = self.embedding(indices)
        if self.training:
            loss_dict = (1 - F.cosine_similarity(z_q.detach(), z_flattened, dim=-1)).mean() + \
           2.0 * (1 - F.cosine_similarity(z_q, z_flattened.detach(), dim=-1)).mean()
            # loss_dict = torch.mean((z_q.detach()-z_flattened)**2) + 0.1* torch.mean((z_q - z_flattened.detach()) ** 2)
            # z_q = z_q+ (z_flattened - z_flattened.data)
            z_q = z_q.detach()+ (z_flattened - z_flattened.detach())
            # z_q = rotate_to(z_flattened, z_q)
            z_q = z_q.view(z.shape)
            z_q = self.expand(z_q)
            # loss_dict = torch.mean((z_q - input) ** 2)
            # indices2 = torch.argmin(cos_sim, dim=0)
            # loss_dict2 = (1 - F.cosine_similarity(z_flattened[indices2, :], self.embedding.weight, dim=-1)).mean() 
            if random.random() < 0.0002:
                print(f"activated vectors:{unique_index.shape[0]} , num feature: {num_feature}, num dict: {self.codebook_size}")
            # loss_dict3=torch.mean((z_q-input)**2)
            # loss_dict3 = F.mse_loss(z_q, input)
            # loss = F.mse_loss(z_q.detach(),z_flattened) + 0.5* F.mse_loss(z_q,z_flattened.detach())
            # return z_q, loss_dict
            return z_q, loss_dict
            # return z_q, loss_dict + 0.5*loss_dict2
        else:
            z_q = z_q.view(z.shape)
            z_q = self.expand(z_q)
            if random.random() < 0.001:
                print(f"activated vectors:{unique_index.shape[0]} , num feature: {num_feature}, num dict: {self.codebook_size}")
            return z_q , torch.tensor(0.0).cuda()
class VectorQuantizer_noLinear(nn.Module):
    def __init__(self, n_e, channels_in, channels_dim, index=0):
        super().__init__()
        self.embedding = nn.Embedding(n_e, channels_in)
        trunc_normal_(self.embedding.weight,mean=0.0, std=1.0, a=-1.5, b=1.5)
        # self.compress = TokenToImageToToken(token_dim=384, image_size=224, patch_size=32, out_channels=2)
        self.codebook_size = n_e
        self.is_pre_cal = False
    def reparameterize(self):
        print('using VQ reparameterize')
        self.is_pre_cal = True
        dict_embeding = self.embedding.weight.data.clone().detach()
        # del self.embedding
        return dict_embeding
    def forward(self, z):
        input = z
        
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
            # loss_dict = torch.mean((z_q.detach()-z_flattened)**2) + 2.0* \
            # torch.mean((z_q - z_flattened.detach()) ** 2)
            loss_dict = 2.0* torch.mean((z_q - z_flattened.detach()) ** 2)
#------------------------------------------ lossdict2 with mask------------------------------------------------
            # indices2 = torch.argmin(d, dim=0)
            # loss_per_element = (z_flattened[indices2, :].detach() - self.embedding.weight) ** 2
            # mask = torch.isin(torch.arange(loss_per_element.size(0)).to(loss_per_element.device), unique_index)
            # loss_per_element[mask] = 0
            # loss_dict2=torch.mean(loss_per_element)
#------------------------------------------ lossdict2 with mask------------------------------------------------

            # z_q = z_q.data+ (z_flattened - z_flattened.data)
            z_q = rotate_to(z_flattened, z_q)
            z_q = z_q.view(z.shape)
            # loss_dict = torch.mean((z_q - input) ** 2)
            # indices2 = torch.argmin(d, dim=0)
            # loss_dict2 = torch.mean((z_flattened[indices2, :] - self.embedding.weight) ** 2)
            if random.random() < 0.0002:
                print(f"activated vectors:{unique_index.shape[0]} , num feature: {num_feature}, num dict: {self.codebook_size}")
            # loss_dict3=torch.mean((z_q-input)**2)
            # loss_dict3 = F.mse_loss(z_q, input)
            # loss = F.mse_loss(z_q.detach(),z_flattened) + 0.5* F.mse_loss(z_q,z_flattened.detach())
            # return z_q, loss_dict
            # return z_q, loss_dict + 0.5*loss_dict2
            return z_q, loss_dict
        else:
            z_q = z_q.view(z.shape)
            if random.random() < 0.001:
                print(f"activated vectors:{unique_index.shape[0]} , num feature: {num_feature}, num dict: {self.codebook_size}")
            return z_q , torch.tensor(0.0).cuda()
class VectorQuantizer_LossMask(nn.Module):
    def __init__(self, n_e, channels_in, channels_dim, index=0):
        super().__init__()
        self.embedding = nn.Embedding(n_e, channels_dim)
        trunc_normal_(self.embedding.weight,mean=0.0, std=1.0, a=-3.0, b=3.0)
        # self.compress = TokenToImageToToken(token_dim=384, image_size=224, patch_size=32, out_channels=2)
        self.compress = nn.Linear(channels_in, channels_dim)
        self.expand = nn.Linear(channels_dim, channels_in)
        self.codebook_size = n_e
        self.is_pre_cal = False
    def reparameterize(self):
        print('using VQ reparameterize')
        self.is_pre_cal = True
        dict_embeding = self.embedding.weight.data.clone().detach()
        expand_dict = self.expand(dict_embeding)
        # del self.embedding
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
#  F.mse_loss
# loss = F.mse_loss(z_q.detach(),z_flattened) + 0.5* F.mse_loss(z_q,z_flattened.detach())
# loss = F.mse_loss(z_q,z_flattened)
        indices = torch.argmin(d, dim=1)
        if not self.training and self.is_pre_cal:
            return indices
        unique_index=torch.unique(indices)
        z_q = self.embedding(indices)
        if self.training:
            loss_dict = torch.mean((z_q.detach()-z_flattened)**2) + 2.0* \
            torch.mean((z_q - z_flattened.detach()) ** 2)
            indices2 = torch.argmin(d, dim=0)
            loss_per_element = (z_flattened[indices2, :].detach() - self.embedding.weight) ** 2

            mask = torch.isin(torch.arange(loss_per_element.size(0)).to(loss_per_element.device), unique_index)
            loss_per_element[mask] = 0

            loss_dict2=torch.mean(loss_per_element)
            z_q = z_q.data+ (z_flattened - z_flattened.data)
            # z_q = rotate_to(z_flattened, z_q)
            z_q = z_q.view(z.shape)
            z_q = self.expand(z_q)
            if random.random() < 0.0002:
                print(f"activated vectors:{unique_index.shape[0]} , num feature: {num_feature}, num dict: {self.codebook_size}")
            return z_q, loss_dict+0.2*loss_dict2
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
    output_path = "/home/mulan/ccz/VQViT/FSQfig/quantization_error_heatmap.png"  # 修改为你的保存路径
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
