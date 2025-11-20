# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Vector Quantized PoolFormer implementation
"""
import os
import copy
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.layers.helpers import to_2tuple
from timm.models.vectorquantize import choose_vq

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'poolformer_s': _cfg(crop_pct=0.9),
    'poolformer_m': _cfg(crop_pct=0.95),
}
from einops import rearrange, repeat, reduce, pack, unpack
import random
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            feat1 = x
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
            feat2 = x
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            feat1 = x
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            feat2 = x
        # print(f'use layer scale: {self.use_layer_scale}, x shape: {x.shape}')
        return x, (feat1, feat2)

class vq_PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5, 
                 
                vq_type='fsq_qd',fsq_level = [3,3,3,3],
                dic_n=None, dic_dim=4, fsq_Tinit=1):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.vq = choose_vq(vq_type=vq_type, dic_n=dic_n, dim=dim, dic_dim=dic_dim, fsq_level=fsq_level, fsq_Tinit=fsq_Tinit, input_format='NCHW')
        self.token_wise_rep = False
        self.dim = dim
    def reparameterize(self):
        ''' 
        reparameterize the vq dict and calculate the rep_codebook for inference, 
        the case where the codebook is not a square matrix has also been taken into consideration. 
        '''
        print('using Block reparameterize')
        self.token_wise_rep = True
        self.rep_codebook = nn.Embedding(self.vq.codebook_size, self.dim)
        # print(self.dim)
        fixed_codebook = self.vq.reparameterize() # (codebook size, dim)
        N, D = fixed_codebook.shape[0], fixed_codebook.shape[1]
        fixed_codebook_transposed = fixed_codebook.transpose(0, 1).contiguous() # N, D -> D, N
        # handle the case where N is not a perfect square number for Conv
        h = int(torch.sqrt(torch.tensor(N)).ceil().item())
        w = (N + h - 1) // h
        if h * w > N:
            pad_size = h * w - N
            x_padded = torch.cat([fixed_codebook_transposed, torch.zeros(D, pad_size, device=fixed_codebook_transposed.device)], dim=1)
        else:
            x_padded = fixed_codebook_transposed
        fixed_codebook_rep = x_padded.reshape(1, D,h,w) # (1, D, h, w)
        x = self.mlp(fixed_codebook_rep)
        if self.use_layer_scale:
            x = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)* x
            del self.layer_scale_2
        x = x.reshape(D, -1) # (D, h*w)
        if h * w > N:
            x = x[:, :N] # (D, N)
        x = x.transpose(0, 1).contiguous() # (N, D)
        self.rep_codebook.weight.data.copy_(x)
        del self.mlp

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            feat1 = x # for distillation
            x = self.norm2(x)
            loss_dict=torch.tensor(0.0).cuda()
            if not self.training and self.token_wise_rep:
                embedding_index =  self.vq(x)
                z_q = self.rep_codebook(embedding_index)
                x = z_q.transpose(1, 2).reshape(feat1.shape).contiguous()
                return x+feat1, loss_dict
            x , loss_dict= self.vq(x)
            x = self.mlp(x)
            x = self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)* x)
            x = x + feat1

            return x, loss_dict, (feat1, x)# for distillation
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            feat1 = x # for distillation
            x = self.norm2(x)
            loss_dict=torch.tensor(0.0).cuda()
            if not self.training and self.token_wise_rep:
                embedding_index =  self.vq(x)
                z_q = self.rep_codebook(embedding_index)
                x = z_q.transpose(1, 2).reshape(feat1.shape)
                return x+feat1, loss_dict
            x , loss_dict= self.vq(x)
            
            x = self.mlp(x)
            x = self.drop_path(x)
            x = x + feat1

            return x, loss_dict, (feat1, x)# for distillation

class MultiOutputSequential(nn.Sequential):
    '''A sequential container for modules with multiple outputs.'''    
    def forward(self, x):
        quantize_loss_list = []
        for module in self:
                x = module(x)  # 解包元组输入
                if isinstance(x, tuple):
                    if len(x) > 2:
                        quantize_loss_list.append(x[1])
                    x = x[0]
        return x, quantize_loss_list  
def basic_blocks(dim, index, layers, 
                 pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop_rate=.0, drop_path_rate=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 
                 vq_type='fsq_qd',fsq_level = [3,3,3,3],
                dic_n=None, dic_dim=4, fsq_Tinit=1):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks 
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        if block_idx%2==0:
            blocks.append(PoolFormerBlock(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio, 
                act_layer=act_layer, norm_layer=norm_layer, 
                drop=drop_rate, drop_path=block_dpr, 
                use_layer_scale=use_layer_scale, 
                layer_scale_init_value=layer_scale_init_value, 
                ))
        else:
            blocks.append(vq_PoolFormerBlock(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio, 
                act_layer=act_layer, norm_layer=norm_layer, 
                drop=drop_rate, drop_path=block_dpr, 
                use_layer_scale=use_layer_scale, 
                layer_scale_init_value=layer_scale_init_value, 

                vq_type=vq_type, fsq_level = fsq_level,
                dic_n=dic_n, dic_dim=dic_dim, fsq_Tinit=fsq_Tinit
                ))
    blocks = MultiOutputSequential(*blocks)
    # blocks = nn.Sequential(*blocks)

    return blocks


class vqPoolFormer(nn.Module):
    """
    PoolFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and 
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad: 
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained: 
        for mmdetection and mmsegmentation to load pretrained weights
    """
    def __init__(self, layers, embed_dims=None, 
                 mlp_ratios=None, downsamples=None, 
                 pool_size=3, 
                 norm_layer=GroupNorm, act_layer=nn.GELU, 
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2, 
                 down_patch_size=3, down_stride=2, down_pad=1, 
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5, 
                 fork_feat=False,
                 init_cfg=None, 
                 pretrained=None, 

                 vq_type='fsq_qd',fsq_level = [3,3,3,3],
                dic_n=None, dic_dim=4, fsq_Tinit=1,
                 **kwargs):

        super().__init__()
        self.token_wise_rep = False
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans=3, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, 
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value,
                                 
                                 vq_type=vq_type,fsq_level = fsq_level,
                                dic_n=dic_n, dic_dim=dic_dim, fsq_Tinit=fsq_Tinit
                                 )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride, 
                        padding=down_pad, 
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model 
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading 
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            
            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)
    def reparameterize(self):
        self.token_wise_rep = True
        for module in self.network:  # 遍历ModuleList中的每个模块
            print(f"Found module: {module.__class__.__name__.lower()}, {isinstance(module, nn.Sequential)}")
            if isinstance(module, nn.Sequential):
                for name, submodule in module.named_children():
                    print(f"    Found sub module: {submodule.__class__.__name__.lower()}")
                    if submodule.__class__.__name__.lower().startswith('vq'):
                        print(f"        Found VQ module in Sequential: {submodule.__class__.__name__.lower()}")
                        submodule.reparameterize()
            else:
                # 情况2：模块是直接的非Sequential（如Conv2d）
                if module.__class__.__name__.lower().startswith('vq'):
                    # print(f"Found standalone VQ module: {module.__class__.__name__.lower()}")
                    submodule.reparameterize()
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x, is_feat=False):
        # input embedding
        x = self.patch_embed(x)
        # through backbone
        quantize_loss_list = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if isinstance(x, tuple):
                quantize_loss_list.extend(x[1])
                x = x[0]
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        if quantize_loss_list:
            return cls_out, torch.stack(quantize_loss_list).mean()
        return cls_out
    @torch.jit.ignore
    def print_codebook_utilization(self):
        sum_util = 0.0
        average_util = 0.0
        count = 0
        found_modules = []  # 用于记录找到的模块及其路径
        # 递归遍历所有子模块
        for module_name, module in self.named_modules():
            # 检查模块是否是 VQ 实例（根据您的类名调整条件）
            # 例如，如果您的VQ类名为 VectorQuantizer：
            if hasattr(module, 'codebook_meter'): # 或者使用 isinstance(module, VectorQuantizer)
                utilization = module.codebook_meter.utilization # 注意：这里是属性，不是方法调用
                print(f'VQ Module [{module_name}] Codebook Utilization: {utilization * 100:.2f}%')
                found_modules.append(module_name)
                sum_util += utilization
                count += 1
        if count > 0:
            average_util = sum_util / count
            print(f'Average VQ Codebook Utilization (across {count} modules): {average_util*100:.2f}%')
        else:
            print('No VQ modules with codebook meters found.')
        return average_util

model_urls = {
    "poolformer_s12": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar",
    "poolformer_s24": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar",
    "poolformer_s36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar",
    "poolformer_m36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m36.pth.tar",
    "poolformer_m48": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m48.pth.tar",
}


@register_model
def vqpoolformer_s12(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = vqPoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s12']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def vqpoolformer_s24(pretrained=False, **kwargs):
    """
    PoolFormer-S24 model, Params: 21M
    """
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = vqPoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s24']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def vqpoolformer_s36(pretrained=False, **kwargs):
    """
    PoolFormer-S36 model, Params: 31M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = vqPoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s36']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def vqpoolformer_m36(pretrained=False, **kwargs):
    """
    PoolFormer-M36 model, Params: 56M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = vqPoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_m']
    if pretrained:
        url = model_urls['poolformer_m36']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def vqpoolformer_m48(pretrained=False, **kwargs):
    """
    PoolFormer-M48 model, Params: 73M
    """
    layers = [8, 8, 24, 8]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = vqPoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_m']
    if pretrained:
        url = model_urls['poolformer_m48']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


if has_mmseg and has_mmdet:
    """
    The following models are for dense prediction based on 
    mmdetection and mmsegmentation
    """
    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_s12_feat(PoolFormer):
        """
        PoolFormer-S12 model, Params: 12M
        """
        def __init__(self, **kwargs):
            layers = [2, 2, 6, 2]
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims, 
                mlp_ratios=mlp_ratios, downsamples=downsamples, 
                fork_feat=True,
                **kwargs)

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_s24_feat(PoolFormer):
        """
        PoolFormer-S24 model, Params: 21M
        """
        def __init__(self, **kwargs):
            layers = [4, 4, 12, 4]
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims, 
                mlp_ratios=mlp_ratios, downsamples=downsamples, 
                fork_feat=True,
                **kwargs)

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_s36_feat(PoolFormer):
        """
        PoolFormer-S36 model, Params: 31M
        """
        def __init__(self, **kwargs):
            layers = [6, 6, 18, 6]
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims, 
                mlp_ratios=mlp_ratios, downsamples=downsamples, 
                layer_scale_init_value=1e-6, 
                fork_feat=True,
                **kwargs)

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_m36_feat(PoolFormer):
        """
        PoolFormer-S36 model, Params: 56M
        """
        def __init__(self, **kwargs):
            layers = [6, 6, 18, 6]
            embed_dims = [96, 192, 384, 768]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims, 
                mlp_ratios=mlp_ratios, downsamples=downsamples, 
                layer_scale_init_value=1e-6, 
                fork_feat=True,
                **kwargs)

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_m48_feat(PoolFormer):
        """
        PoolFormer-M48 model, Params: 73M
        """
        def __init__(self, **kwargs):
            layers = [8, 8, 24, 8]
            embed_dims = [96, 192, 384, 768]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims, 
                mlp_ratios=mlp_ratios, downsamples=downsamples, 
                layer_scale_init_value=1e-6, 
                fork_feat=True,
                **kwargs)


if __name__ == '__main__':
    #python -m timm.models.poolformer.py 
    model = vqpoolformer_s12(pretrained=False, num_classes=1000)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    if isinstance(output, tuple):
        print(output[0].shape, output[1])
    else:
        print(output.shape)
    # print(model)