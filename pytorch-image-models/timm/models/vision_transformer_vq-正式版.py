"""
Vector Quantized Vision Transformer (VQ-ViT) in PyTorch

class MSA_VQFFN_Block : vanilla MSA + VQ-FFN
class VQMSA_FFN_Block : VQ-MSA + vanilla FFN

VQ-ViT: The framework strictly follows the ViT architecture. (b) Two successive VQ
    Transformer blocks, where we replace FFN with VQ-FFN and MSA with VQ-MSA alternatively.

vanilla ViT :  12x[ MSA + FFN ]
VQ-ViT: [ MSA + VQ-FFN ] + [ VQ-MSA + FFN ] ... A total of 12 blocks

"""
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
import time
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, Mlp3Linear,\
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from ._builder import build_model_with_cfg
from ._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

from timm.models.vectorquantize import  VectorQuantizer, FSQ,  FSQ_trainableT, FSQ_T

__all__ = ['vqVisionTransformer']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)
import random
import numpy as np
class Attention(nn.Module):
    ''' vanilla Attention'''
    fused_attn: Final[bool]
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        # print(f'Attention q mean:{q.mean().item()}, std: {q.std().item()}, k mean:{k.mean().item()} , std: {k.std().item()}, qk std:{(q @ k.transpose(-2, -1)).std().item()}, scale:{self.scale}   ')
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class vq_Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,

            vq_type='tfsq',
            fsq_level = [3,3,3,3],
            dic_n=1000, dic_dim=4, index=0, fsq_Tinit=-1
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.is_pre_cal = False
        if vq_type == 'vq':
            self.vq = VectorQuantizer(dic_n, dim, dic_dim, index)
        elif vq_type == 'tfsq':
            self.vq = FSQ_trainableT(dic_n, dim, dic_dim, index, levels=fsq_level, T=fsq_Tinit)
        elif vq_type == 'fsq':
            self.vq = FSQ_T(dic_n, dim, dic_dim, index, levels=fsq_level, T=1)
        else:
            raise NotImplementedError
    def reparameterize(self, vq_embedding):
        '''
        vq_embedding: codebook->norm
        '''
        print('using Attention reparameterize')
        self.is_pre_cal = True
        self.qkv_dict = nn.Embedding(vq_embedding.shape[0], 3*self.dim)
        vq_embedding = vq_embedding.unsqueeze(0)
        B, N, C = vq_embedding.shape
        qkv = self.qkv(vq_embedding).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        q_flat = q.permute(0, 2, 1, 3).reshape(-1, self.dim)  # shape: [B*N*num_heads, dim]
        k_flat = k.permute(0, 2, 1, 3).reshape(-1, self.dim)
        v_flat = v.permute(0, 2, 1, 3).reshape(-1, self.dim)
        self.qkv_dict.weight.data.copy_(
            torch.cat([q_flat, k_flat, v_flat], dim=1).reshape(-1, 3 * self.dim)
        )
        del self.qkv
        del self.q_norm, self.k_norm, self.scale

        self.proj_codebook = nn.Embedding(self.vq.codebook_size, self.dim)

        vq_dict = self.vq.reparameterize()
        vq_dict = self.proj(vq_dict)
        self.proj_codebook.weight.data.copy_(vq_dict)
        del self.proj

    def forward(self, x, shape=None):
        if self.is_pre_cal:
            B, N, C = shape
            
            embedding_index_map =  x
            qkv = self.qkv_dict(embedding_index_map).reshape(B, N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        else:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            # print(f'VQ Attention q mean:{q.mean().item()}, std: {q.std().item()}, k mean:{k.mean().item()} , std: {k.std().item()}, qk std:{(q @ k.transpose(-2, -1)).std().item()}, scale:{self.scale}   ')
            q = q * self.scale
            # print(f'q.shape before rep: {q.shape}')
        

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        if self.is_pre_cal:
            loss_dict = torch.tensor(0.0).cuda()
            embedding_index =  self.vq(x)
            x = self.proj_codebook(embedding_index)
        else:
            x, loss_dict = self.vq(x)
            x = self.proj(x)
        x = self.proj_drop(x)
        return x, loss_dict


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
class MSA_VQFFN_Block(nn.Module):
    '''
    vanilla MSA + VQ-FFN
    '''

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            dic_n=1024, dic_dim=8,
            index=0,
            vq_type='tfsq',
            fsq_level = [3,3,3,3],
            fsq_Tinit=-1
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if vq_type == 'vq':
            print(f'using vq, codebook: {dic_n, dic_dim}')
            self.vq = VectorQuantizer(dic_n, dim, dic_dim, index)
        elif vq_type == 'tfsq' or vq_type == 'fsq':
            print(f'using FSQ_trainableT, levels={fsq_level}')      
            self.vq = FSQ_trainableT(dic_n, dim, dic_dim, index, levels=fsq_level, T=fsq_Tinit)
        else:
            raise NotImplementedError
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.is_pre_cal = False
        self.dict_n = dic_n
        self.dim = dim
    def reparameterize(self):
        print('using Block reparameterize')
        self.is_pre_cal = True
        self.pre_cal_dict = nn.Embedding(self.vq.codebook_size, self.dim)
        vq_dict = self.vq.reparameterize()
        x=self.norm2(vq_dict)
        x=self.mlp(x)
        x = self.ls2(x)
        self.pre_cal_dict.weight.data.copy_(x)
        del self.norm2
        del self.mlp
        del self.ls2
    def forward(self, x):
        input0 = x
        x = self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = input0 + x
        input = x
        loss_dict=torch.tensor(0.0).cuda()
        if not self.training and self.is_pre_cal:
            embedding_index =  self.vq(x)
            z_q = self.pre_cal_dict(embedding_index)
            z_q = z_q.view(input.shape)
            return z_q+input, loss_dict
        else:
            feat0 = x 
            x, loss_dict = self.vq(x)
            x = self.norm2(x)
            x = self.mlp(x)
            x = self.ls2(x)
            x = self.drop_path2(x)
            x = x + input
            feat = x 
            return x, loss_dict, (feat0, feat)

class VQMSA_FFN_Block(nn.Module):
    """
    VQ-MSA + vanilla FFN
    """
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            dic_n=1024, dic_dim=8,
            index=0,
            vq_type='tfsq',
            fsq_level = [3,3,3,3],
            fsq_Tinit=-1
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = vq_Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            vq_type=vq_type,
            fsq_level = fsq_level,
            dic_n=1000, dic_dim=len(fsq_level), index=index,
            fsq_Tinit = fsq_Tinit
            # dic_n=dic_n, dic_dim=dic_dim, index=index
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.is_pre_cal = False
        self.dict_n = dic_n
        self.dim = dim
        if vq_type == 'vq':
            self.vq = VectorQuantizer(dic_n, dim, dic_dim, index)
        elif vq_type == 'tfsq' or vq_type == 'fsq':
            print(f'using FSQ_trainableT in attn,  levels={fsq_level}')
            self.vq = FSQ_trainableT(dic_n, dim, dic_dim, index, levels=fsq_level, T=fsq_Tinit)
        else:
            raise NotImplementedError
    def reparameterize(self):
        print('using Block reparameterize')
        self.is_pre_cal = True
        vq_dict = self.vq.reparameterize()
        vq_dict = self.norm1(vq_dict)
        self.attn.reparameterize(vq_dict)
        del self.norm1

    def forward(self, x):
        '''
        plug vq Block before first norm 
        '''
        input0 = x
        if self.is_pre_cal:
            shape = x.shape
            qkv_vq_loss=torch.tensor(0.0).cuda()
            embedding_index =  self.vq(x)
            x, vq_proj_loss = self.attn(embedding_index, shape)
        else:
            x, qkv_vq_loss = self.vq(x)  #         plug vq Block before first norm 
            x, vq_proj_loss = self.attn(self.norm1(x))

        feat_attn = x
        x = self.drop_path1(self.ls1(x))
        x = input0 + x
        input = x
        
        feat0 = x # Distillation
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.ls2(x)
        x = self.drop_path2(x)
        x = x + input
        feat = x # Distillation

        return x, 0.5*qkv_vq_loss+0.5*vq_proj_loss, (feat0,feat)


class VQMSA_VQFFN_Block(nn.Module):
    '''
    VQ-MSA + VQ-FFN (for test)
    '''
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            dic_n=1024, dic_dim=8,
            index=0,
            vq_type='tfsq',
            fsq_level = [3,3,3,3],
            fsq_Tinit=-1
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = vq_Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            # 暂时固定值
            vq_type=vq_type,
            fsq_level = fsq_level,
            dic_n=1000, dic_dim=len(fsq_level), index=index,
            fsq_Tinit = fsq_Tinit
            
            # dic_n=dic_n, dic_dim=dic_dim, index=index
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.is_pre_cal = False
        self.dict_n = dic_n
        self.dim = dim
        if vq_type == 'vq':
            self.vq = VectorQuantizer(dic_n, dim, dic_dim, index)
            self.ffn_vq = VectorQuantizer(dic_n, dim, dic_dim, index)
        elif vq_type == 'tfsq' or vq_type == 'fsq':
            print(f'using FSQ_trainableT in attn,  levels={fsq_level}')
            self.vq = FSQ_trainableT(dic_n, dim, dic_dim, index, levels=fsq_level, T=fsq_Tinit)
            self.ffn_vq = FSQ_trainableT(dic_n, dim, dic_dim, index, levels=fsq_level, T=fsq_Tinit)
        else:
            raise NotImplementedError
    def reparameterize(self):
        print('using Block reparameterize')
        self.is_pre_cal = True
        vq_dict = self.vq.reparameterize()
        vq_dict = self.norm1(vq_dict)
        # print(f'vq_dict.shape: {vq_dict.shape}')
        self.attn.reparameterize(vq_dict)
        del self.norm1

        self.ffn_dict = nn.Embedding(self.ffn_vq.codebook_size, self.dim)
        vq_ffn_dict = self.ffn_vq.reparameterize()
        x=self.norm2(vq_ffn_dict)
        x=self.mlp(x)
        x = self.ls2(x)
        self.ffn_dict.weight.data.copy_(x)
        del self.norm2
        del self.mlp
        del self.ls2

    def forward(self, x):
        input0 = x
        if self.is_pre_cal:
            shape = x.shape
            qkv_vq_loss=ffn_vq_loss=torch.tensor(0.0).cuda()
            embedding_index =  self.vq(x)
            x, vq_proj_loss = self.attn(embedding_index, shape)
            feat_attn = x
            x = self.drop_path1(self.ls1(x))
            # feat = x # 蒸馏位置1
            x = input0 + x
            input = x
            
            feat0 = x # 蒸馏位置2
            ffn_embedding_index =  self.ffn_vq(x)
            z_q = self.ffn_dict(ffn_embedding_index)
            z_q = z_q.view(input.shape)
            return z_q+input, 0.333*(qkv_vq_loss+vq_proj_loss+ffn_vq_loss)
        else:
            x, qkv_vq_loss = self.vq(x)
            x, vq_proj_loss = self.attn(self.norm1(x))
            feat_attn = x
            x = self.drop_path1(self.ls1(x))
            # feat = x # 蒸馏位置1
            x = input0 + x
            input = x
            
            feat0 = x # 蒸馏位置2
            x, ffn_vq_loss = self.ffn_vq(x)
            x = self.norm2(x)
            x = self.mlp(x)
            # feat = x # z蒸馏位置3
            x = self.ls2(x)
            x = self.drop_path2(x)
            x = x + input
            feat = x # 蒸馏位置4
            # return x, loss_dict, feat0
            # return x, qkv_vq_loss, (feat_attn,feat)
            # return x, 0.5*qkv_vq_loss+0.5*vq_proj_loss, feat0
            return x, 0.333*(qkv_vq_loss+vq_proj_loss+ffn_vq_loss), (feat0,feat)

class Block(nn.Module):
    """
    Transformer block of vanilla ViT
    """
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, is_feat=False, init_codebook_feat = False):
        input0 = x
        x=self.norm1(x)
        if init_codebook_feat:
            x, attn_proj_init_feat = self.attn(x, init_codebook_feat=init_codebook_feat)
        else:
            x=self.attn(x)
        feat_attn = x
        x = self.drop_path1(self.ls1(x))
        x = input0 + x
        feat0 = x 
        input = x
        init_feat = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.ls2(x)
        x = self.drop_path2(x)
        x = x + input
        feat = x 
        return x, torch.tensor(0.0).cuda(), (feat0,feat)
class vqVisionTransformer(nn.Module):
    """ Vector Quantized Vision Transformer
    VQ-ViT: (a) The framework strictly follows the ViT architecture. (b) Two successive VQ
    Transformer blocks, where we replace FFN with VQ-FFN and MSA with VQ-MSA alternatively.
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = None,
            mlp_layer: Type[nn.Module] = Mlp,
            dic_n=2048, # Parameters of traditional VQ 
            dic_dim=4, # the same as length of fsq level
            vq_type='tfsq', fsq_level=[3,3,3,3], fsq_Tinit=-1
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
            dic_n: Parameters of traditional VQ.
            vq_type='tfsq': choose vector quantization method: vq or tfsq
            fsq_level: Parameters of FSQ
            fsq_Tinit: Parameters of FSQ, For the initialization of the fsq range
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        block_list = list()
       
        # block_list=[VQMSA_VQFFN_Block(
        #         dim=embed_dim,
        #         num_heads=num_heads,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         qk_norm=qk_norm,
        #         init_values=init_values,
        #         proj_drop=proj_drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[i],
        #         norm_layer=norm_layer,
        #         act_layer=act_layer,
        #         mlp_layer=mlp_layer,
        #         dic_n=dic_n, dic_dim=dic_dim,
        #         index=i,
        #         vq_type=vq_type,
        #         fsq_level = fsq_level,
        #         fsq_Tinit=fsq_Tinit
        #     ) for i in range(depth)]

        for i in range(depth):
            if i%2==0:
                block_list.append(MSA_VQFFN_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                dic_n=dic_n, dic_dim=dic_dim,
                index=i,
                vq_type=vq_type,
                fsq_level = fsq_level,
                fsq_Tinit=fsq_Tinit
            ) )
            else:
                block_list.append(VQMSA_FFN_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                dic_n=dic_n, dic_dim=dic_dim,
                index=i,
                vq_type=vq_type,
                fsq_level = fsq_level,
                fsq_Tinit=fsq_Tinit
            ))
        self.blocks = nn.Sequential(*block_list)

        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

        self.is_pre_cal = False

    def reparameterize(self):
        print('using Model reparameterize')
        self.is_pre_cal = True
        for block in self.blocks:
            block.reparameterize()
    def use_shared_codebook(self):
        print('using shared codebook')
        first_vq, *rest_vq = self.blocks
        codebook = first_vq.vq.embedding
        for vq in rest_vq:
            vq.vq.embedding = codebook  
        
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map ' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ):
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, is_feat=False):
        feat=list()
        # x = self.forward_features(x,is_feat)
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        loss_dict = list()
        index_record = list()
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            # print(f' Block index: {i}')
            if isinstance(x, tuple) and len(x) > 1:
                if not self.training and self.is_pre_cal:
                    index_record.append(x[1])
                else:
                    loss_dict.append(x[1])
                if is_feat:
                    feat.append(x[2])
                x= x[0]
            elif isinstance(x, tuple):
                x= x[0]
        x = self.norm(x)
        x = self.forward_head(x)
        if is_feat and loss_dict:
            return x, torch.stack(loss_dict).sum(), feat
        elif loss_dict:
            return x, torch.stack(loss_dict).sum()
        elif index_record:
            return x, index_record
        else:
            return x


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def resize_pos_embed(
        posemb,
        posemb_new,
        num_prefix_tokens=1,
        gs_new=(),
        interpolation='bicubic',
        antialias=False,
):
    """ Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of resample_abs_pos_embed

    Adapted from:
        https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(f'Resized position embedding: {posemb.shape} ({[gs_old, gs_old]}) to {posemb_new.shape} ({gs_new}).')
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, antialias=antialias, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


@torch.no_grad()
def _load_weights(model: vqVisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False
    big_vision = False
    if not prefix:
        if 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'
        elif 'params/embedding/kernel' in w:
            prefix = 'params/'
            big_vision = True
        elif 'params/img/embedding/kernel' in w:
            prefix = 'params/img/'
            big_vision = True

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            model.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )

    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
    else:
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        old_shape = pos_embed_w.shape
        num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if (isinstance(model.head, nn.Linear) and
            f'{prefix}head/bias' in w and
            model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]):
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    if model.attn_pool is not None:
        block_prefix = f'{prefix}MAPHead_0/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
        model.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
        model.attn_pool.kv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
        model.attn_pool.kv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
        model.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
        model.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
        model.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        model.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        model.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        model.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        for r in range(2):
            getattr(model.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
            getattr(model.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias']))


def _convert_openai_clip(state_dict, model, prefix='visual.'):
    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'), ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'), ('ln_pre', 'norm_pre'), ('ln_post', 'norm'), ('ln_', 'norm'),
        ('in_proj_', 'qkv.'), ('out_proj', 'proj'), ('mlp.c_fc', 'mlp.fc1'), ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
            if v.shape[1] != model.pos_embed.shape[1]:
                # To resize pos embedding when using model at different size from pretrained weights
                v = resize_pos_embed(
                    v,
                    model.pos_embed,
                    0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                    model.patch_embed.grid_size
                )
        out_dict[k] = v
    return out_dict


def _convert_dinov2(state_dict, model):
    import re
    out_dict = {}
    state_dict.pop("mask_token", None)
    if 'register_tokens' in state_dict:
        # convert dinov2 w/ registers to no_embed_class timm model (neither cls or reg tokens overlap pos embed)
        out_dict['reg_token'] = state_dict.pop('register_tokens')
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        out_dict['pos_embed'] = state_dict.pop('pos_embed')[:, 1:]
    for k, v in state_dict.items():
        if re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn(
        state_dict,
        model,
        adapt_layer_scale=False,
        interpolation='bicubic',
        antialias=True,
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''

    if 'visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model)
    elif 'module.visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model, prefix='module.visual.')

    if "mask_token" in state_dict:
        state_dict = _convert_dinov2(state_dict, model)

    if "encoder" in state_dict:
        state_dict = state_dict['encoder']
        prefix = 'module.'

    if 'visual.trunk.pos_embed' in state_dict:
        # convert an OpenCLIP model with timm vision encoder
        # FIXME remap final nn.Linear if it exists outside of the timm .trunk (ie in visual.head.proj)
        prefix = 'visual.trunk.'

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # All parameters are the same as those of vanilla ViT.
    'vqvit_tiny_patch16_224': _cfg(),
    'vqvit_small_patch16_224': _cfg(),
    'vqvit_small_patch32_224': _cfg(),
    'vqvit_base_patch16_224': _cfg(),
    'vqvit_base_patch32_224': _cfg(),
    'vqvit_large_patch16_224': _cfg(),
    'vqvit_large_patch32_224': _cfg(),
}
default_cfgs = generate_default_cfgs(default_cfgs)


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = True
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        vqVisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )
    
# All parameters are the same as those of vanilla ViT.
@register_model
def vqvit_tiny_patch16_224(pretrained=False, **kwargs) -> vqVisionTransformer:
    """ 
    VQViT is adapted from
    ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vqvit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vqvit_tiny_patch16_384(pretrained=False, **kwargs) -> vqVisionTransformer:
    """ 
    VQViT is adapted from
    ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vqvit_tiny_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vqvit_small_patch32_224(pretrained=False, **kwargs) -> vqVisionTransformer:
    """ 
    VQViT is adapted from
    ViT-Small (ViT-S/32)
    """
    model_args = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6)#, mlp_layer=Mlp3Linear)
    model = _create_vision_transformer('vqvit_small_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vqvit_small_patch32_384(pretrained=False, **kwargs) -> vqVisionTransformer:
    """ 
    VQViT is adapted from
    ViT-Small (ViT-S/32) at 384x384.
    """
    model_args = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, img_size=384)
    model = _create_vision_transformer('vqvit_small_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vqvit_small_patch16_224(pretrained=False, **kwargs) -> vqVisionTransformer:
    """ 
    VQViT is adapted from
    ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vqvit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vqvit_small_patch16_384(pretrained=False, **kwargs) -> vqVisionTransformer:
    """ 
    VQViT is adapted from
    ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vqvit_small_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vqvit_small_patch8_224(pretrained=False, **kwargs) -> vqVisionTransformer:
    """ 
    VQViT is adapted from
    ViT-Small (ViT-S/8)
    """
    model_args = dict(patch_size=8, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vqvit_small_patch8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vqvit_base_patch32_224(pretrained=False, **kwargs) -> vqVisionTransformer:
    """ 
    VQViT is adapted from
    ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vqvit_base_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vqvit_base_patch32_384(pretrained=False, **kwargs) -> vqVisionTransformer:
    """ 
    VQViT is adapted from
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vqvit_base_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vqvit_base_patch16_224(pretrained=False, **kwargs) -> vqVisionTransformer:
    """ 
    VQViT is adapted from
    ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vqvit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vqvit_large_patch32_224(pretrained=False, **kwargs) -> vqVisionTransformer:
    """ 
    VQViT is adapted from
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_args = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vqvit_large_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vqvit_large_patch16_224(pretrained=False, **kwargs) -> vqVisionTransformer:
    """
    VQViT is adapted from
     ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vqvit_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
if __name__ == '__main__':
    import torch.autograd.profiler as profiler
    #python -m timm.models.poolformer.py 
    model = vqvit_small_patch16_224(pretrained=False, num_classes=1000).cuda().eval()
    model.reparameterize()
    input = torch.randn(1024, 3, 224, 224).cuda()
    # output = model(input)
    # print(output.shape)
    # print(model)
    with profiler.profile(use_cuda=True, record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            output = model(input)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))