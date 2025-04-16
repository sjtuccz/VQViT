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
        if random.random() < 0.0003:
            quantization_error = (z - codes).abs().mean()
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
    

class VectorQuantizer(nn.Module):
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
        z_q = self.embedding(indices)
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
