
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


from timm.models.vectorquantize import VectorQuantizer_LossMask, VectorQuantizer_noLinear, VectorQuantizer_CosSim, VectorQuantizer, VectorQuantizer_LinearRebuild, VectorQuantizer_Sim, TokenToImageToToken, FSQ, FSQ_T,FSQ_trainableT, FSQ_AdaptiveQuant,FSQ_GumbelSoftmax
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
class vq_attn_Attention(nn.Module):
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

            vq_type='vq',
            fsq_level = [7,7,7,7],
            dic_n=1000, dic_dim=4, index=0,fsq_Tmax = 10, fsq_Tinit=-1
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
            # self.vq = FSQ_AdaptiveQuant(dic_n, dim, dic_dim, levels=fsq_level)
            self.vq = FSQ_trainableT(dic_n, dim, dic_dim, index, levels=fsq_level, T=fsq_Tinit, T_max=fsq_Tmax)
            # self.vq = FSQ_GumbelSoftmax(dic_n, dim, dic_dim, levels=fsq_level)
        elif vq_type == 'fsq':
            self.vq = FSQ_T(dic_n, dim, dic_dim, index, levels=fsq_level, T=1)
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

class VQMSA(nn.Module):

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
            vq_type='vq',
            fsq_level = [7,7,7,7],
            fsq_Tmax = 10,
            fsq_Tinit=-1
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = vq_attn_Attention(
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
            fsq_Tmax = fsq_Tmax,
            fsq_Tinit = fsq_Tinit
            
            # dic_n=dic_n, dic_dim=dic_dim, index=index
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        

        self.is_pre_cal = False
        self.dict_n = dic_n
        self.dim = dim
        if vq_type == 'vq':
            self.vq = VectorQuantizer(dic_n, dim, dic_dim, index)
        elif vq_type == 'tfsq':
            print(f'using FSQ_trainableT in attn,  levels={fsq_level}')
            self.vq = FSQ_trainableT(dic_n, dim, dic_dim, index, levels=fsq_level, T=fsq_Tinit,T_max=fsq_Tmax)
        elif vq_type == 'fsq':
            self.vq = FSQ_T(dic_n, dim, dic_dim, index, levels=fsq_level, T=1)
    def reparameterize(self):
        print('using Block reparameterize')
        self.is_pre_cal = True
        vq_dict = self.vq.reparameterize()
        vq_dict = self.norm1(vq_dict)
        self.attn.reparameterize(vq_dict)
        del self.norm1

    def forward(self, x):
        input0 = x
        if self.is_pre_cal:
            shape = x.shape
            qkv_vq_loss=torch.tensor(0.0).cuda()
            embedding_index =  self.vq(x)
            x, vq_proj_loss = self.attn(embedding_index, shape)
        else:
            x, qkv_vq_loss = self.vq(x)
            x, vq_proj_loss = self.attn(self.norm1(x))
        x = self.drop_path1(self.ls1(x))
        # feat = x # 蒸馏位置1
        x = input0 + x

        return x, 0.5*qkv_vq_loss+0.5*vq_proj_loss
    
def format_param_count(param_count, decimal_places=3):
    if param_count >= 1e9:
        return f"{round(param_count / 1e9, decimal_places)}G" 
    elif param_count >= 1e6:
        return f"{round(param_count / 1e6, decimal_places)}M"  
    else:
        return f"{round(param_count / 1e3, decimal_places)}K" 


class Attention(nn.Module):
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

    def forward(self, x, init_codebook_feat=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale  #
            attn = q @ k.transpose(-2, -1) # 1,6,197,64
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        init_feat = x
        x = self.proj(x)
        x = self.proj_drop(x)
        if init_codebook_feat:
            return x, init_feat
        else:
            return x
class QKVMatDot(nn.Module):
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

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x, init_codebook_feat=False):
        print(x.shape)
        B, N, C = x.shape
        qkv = x.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        print(q.shape,k.shape,v.shape)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, 384)
        return x





class MSA(nn.Module):

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
            norm_layer=nn.LayerNorm,
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

    def forward(self, x, is_feat=False, init_codebook_feat = False):
        input0 = x
        x=self.norm1(x)
        if init_codebook_feat:
            x, attn_proj_init_feat = self.attn(x, init_codebook_feat=init_codebook_feat)
        else:
            x=self.attn(x)
        x = self.drop_path1(self.ls1(x))
        x = input0 + x
        return x


def cal_qkvMatDot_FLOPs(batch=1,head_num=6,seq_len=197,dim=384,block_num=12,isvq=False):
    
    '''
    x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
    or

    q = q * self.scale
    attn = q @ k.transpose(-2, -1) # b h n d @ b h d n
    attn = attn.softmax(dim=-1)
    x = attn @ v   # b h n n @ b h n d

    This code cannot be automatically calculated for FLOPs by these packages: ptflops calflops ptflops  fvcore thop
    
    vit-s-16: q,k,v.shape=(1,6,197,64) (b,h_num,seq_len,head_dim) (b,h,n,d)

    '''
    b=batch
    h = head_num
    n = seq_len
    d=dim//head_num
    print(b,h,n,d)
    if isvq:
        FLOPs= block_num*((2*d-1)*n*n*b*h + 3*b*h*n*n-1 + (2*n-1)*n*d*b*h)
    else:
        FLOPs= block_num*(b*h*n*d + (2*d-1)*n*n*b*h + 3*b*h*n*n-1 + (2*n-1)*n*d*b*h)
    return format_param_count(FLOPs)

class FFN(nn.Module):

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            proj_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()

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
        input = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.ls2(x)
        x = self.drop_path2(x)
        x = x + input
        return x
class vq_ffn_Block(nn.Module):

    def __init__(
            self,
            dim=384,
            mlp_ratio=4.,
            proj_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            dic_n=1024, dic_dim=4,
            index=0,
            vq_type='tfsq',
            fsq_level = [3,3,3,3],
            fsq_Tmax = 10,
            fsq_Tinit=-1
    ):
        super().__init__()
        if vq_type == 'vq':
            print(f'using vq, codebook: {dic_n, dic_dim}')
            self.vq = VectorQuantizer(dic_n, dim, dic_dim, index)
        elif vq_type == 'tfsq':
            print(f'using FSQ_trainableT, levels={fsq_level}')
            self.vq = FSQ_trainableT(dic_n, dim, dic_dim, index, levels=fsq_level, T=fsq_Tinit,T_max=fsq_Tmax )
        elif vq_type == 'fsq':
            self.vq = FSQ_T(dic_n, dim, dic_dim, index, levels=fsq_level, T=1)
        
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
if __name__ == '__main__':
    from thop import profile, clever_format
    from fvcore.nn import FlopCountAnalysis
    from ptflops import get_model_complexity_info

    # vqmsa = VQMSA(dim=384,
    #         num_heads=6,
    #         qkv_bias=False,
    #         qk_norm=False,
    #         attn_drop=0.,
    #         proj_drop=0.,
    #         norm_layer=nn.LayerNorm,
    #         vq_type='tfsq',
    #         fsq_level = [3,3,3,3],
    #         dic_n=1000, dic_dim=4, index=0,fsq_Tmax = 10, fsq_Tinit=-1)
    # msa = MSA(dim=384,
    #         num_heads=6,
    #         qkv_bias=False,
    #         qk_norm=False,
    #         attn_drop=0.,
    #         proj_drop=0.,
    #         )
    # model=vqmsa
    # model=msa
    # model.reparameterize()

    input_shape_3=(1,197,384)




    model = nn.Linear(384,4)
    # model.reparameterize()
    model = model.cuda().eval()
    param_count = sum([m.numel() for m in model.parameters()])
    param_count = format_param_count(param_count)
    input=torch.randn(input_shape_3).cuda()
    # output=model(input)
    # if isinstance(output, tuple):
    #     output = output[0]
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops} ,  params: {param_count}")

    softmaxqkv = cal_qkvMatDot_FLOPs(batch=1,head_num = 6, seq_len = 50,dim = 384,block_num=12,isvq=True)
    print(f"qkv FLOPs: {softmaxqkv} ")

    # input = torch.randn(input_shape_3).cuda()  # 输入张量形状需匹配模型
    # flops = FlopCountAnalysis(model, input)
    # print(f"FlopCountAnalysis FLOPs: {flops.total()} ")


    # macs, params = get_model_complexity_info(model, (197,384*3), as_strings=True)
    # print(f"ptflops FLOPs: {macs}")

    # from calflops import calculate_flops
    # flops, macs, _ = calculate_flops(
    # model=model,
    # input_shape=input_shape_3,
    # output_as_string=True
    # )
    # print(f"calflops FLOPs: {flops}")

