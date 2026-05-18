""" TQ_ConvNeXt: Doing...
"""
# TQ_ConvNeXt
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the MIT license

# TQ_ConvNeXt-V2
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree (Attribution-NonCommercial 4.0 International (CC BY-NC 4.0))
# No code was used directly from TQ_ConvNeXt-V2, however the weights are CC BY-NC 4.0 so beware if using commercially.

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import trunc_normal_, AvgPool2dSame, DropPath, Mlp, GlobalResponseNormMlp, \
    LayerNorm2d, LayerNorm, create_conv2d, get_act_layer, make_divisible, to_ntuple
from timm.layers import NormMlpClassifierHead, ClassifierHead
from ._builder import build_model_with_cfg
from ._manipulate import named_apply, checkpoint_seq
from ._registry import generate_default_cfgs, register_model, register_model_deprecations
from timm.models.tq_block import choose_tq
__all__ = ['TQ_ConvNeXt']  # model_registry will add each entrypoint fn to this


class Downsample(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()

        if in_chs != out_chs:
            self.conv = create_conv2d(in_chs, out_chs, 1, stride=1)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class ConvNeXtBlock(nn.Module):
    """ TQ_ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            kernel_size: int = 7,
            stride: int = 1,
            dilation: Union[int, Tuple[int, int]] = (1, 1),
            mlp_ratio: float = 4,
            conv_mlp: bool = False,
            conv_bias: bool = True,
            use_grn: bool = False,
            ls_init_value: Optional[float] = 1e-6,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Optional[Callable] = None,
            drop_path: float = 0.,
    ):
        """

        Args:
            in_chs: Block input channels.
            out_chs: Block output channels (same as in_chs if None).
            kernel_size: Depthwise convolution kernel size.
            stride: Stride of depthwise convolution.
            dilation: Tuple specifying input and output dilation of block.
            mlp_ratio: MLP expansion ratio.
            conv_mlp: Use 1x1 convolutions for MLP and a NCHW compatible norm layer if True.
            conv_bias: Apply bias for all convolution (linear) layers.
            use_grn: Use GlobalResponseNorm in MLP (from TQ_ConvNeXt-V2)
            ls_init_value: Layer-scale init values, layer-scale applied if not None.
            act_layer: Activation layer.
            norm_layer: Normalization layer (defaults to LN if not specified).
            drop_path: Stochastic depth probability.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            norm_layer = LayerNorm2d if conv_mlp else LayerNorm
        mlp_layer = partial(GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp)
        self.use_conv_mlp = conv_mlp
        self.conv_dw = create_conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation[0],
            depthwise=True,
            bias=conv_bias,
        )
        self.norm = norm_layer(out_chs)
        self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs)) if ls_init_value is not None else None
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(in_chs, out_chs, stride=stride, dilation=dilation[0])
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)
        return x
class TQ_ConvNeXtBlock(nn.Module):
    """ TQ_ConvNeXt TQ Block, only tq MLP
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            kernel_size: int = 7,
            stride: int = 1,
            dilation: Union[int, Tuple[int, int]] = (1, 1),
            mlp_ratio: float = 4,
            conv_mlp: bool = False,
            conv_bias: bool = True,
            use_grn: bool = False,
            ls_init_value: Optional[float] = 1e-6,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Optional[Callable] = None,
            drop_path: float = 0.,

            tq_type='TQ',tq_level = [5,5,5,5],
            dic_n=None, dic_dim=4, tq_Tinit=1,
    ):
        """

        Args:
            in_chs: Block input channels.
            out_chs: Block output channels (same as in_chs if None).
            kernel_size: Depthwise convolution kernel size.
            stride: Stride of depthwise convolution.
            dilation: Tuple specifying input and output dilation of block.
            mlp_ratio: MLP expansion ratio.
            conv_mlp: Use 1x1 convolutions for MLP and a NCHW compatible norm layer if True.
            conv_bias: Apply bias for all convolution (linear) layers.
            use_grn: Use GlobalResponseNorm in MLP (from TQ_ConvNeXt-V2)
            ls_init_value: Layer-scale init values, layer-scale applied if not None.
            act_layer: Activation layer.
            norm_layer: Normalization layer (defaults to LN if not specified).
            drop_path: Stochastic depth probability.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            norm_layer = LayerNorm2d if conv_mlp else LayerNorm
        assert not use_grn, 'GRN not supported in TQ block, use regular TQ_ConvNeXtBlock for that'
        mlp_layer = partial(GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp)
        self.use_conv_mlp = conv_mlp
        self.conv_dw = create_conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation[0],
            depthwise=True,
            bias=conv_bias,
        )
        self.norm = norm_layer(out_chs)
        self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs)) if ls_init_value is not None else None
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(in_chs, out_chs, stride=stride, dilation=dilation[0])
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.tq = choose_tq(tq_type=tq_type, dic_n=dic_n, dim=out_chs, dic_dim=dic_dim, tq_level=tq_level, tq_Tinit=tq_Tinit, input_format='NCHW' if conv_mlp else'NHWC')
        self.token_wise_rep = False
        self.dim = out_chs
        self.register_buffer("rep_codebook", torch.tensor(0))

    def reparameterize(self):
        ''' 
        reparameterize the tq dict and calculate the rep_codebook for inference, 
        the case where the codebook is not a square matrix has also been taken into consideration. 
        '''
        print('using TQ-ConvnextBlock reparameterize')
        self.token_wise_rep = True
        # self.rep_codebook = nn.Embedding(self.tq.codebook_size, self.dim)
        fixed_codebook = self.tq.reparameterize() # (codebook size, dim)
        if self.use_conv_mlp: # mlp是卷积 nchw 格式
            HW, C = fixed_codebook.shape[0], fixed_codebook.shape[1]
            fixed_codebook_transposed = fixed_codebook.transpose(0, 1).contiguous() # (C, HW)
            h = int(torch.sqrt(torch.tensor(HW)).ceil().item())
            w = (HW + h - 1) // h
            if h * w > HW:
                pad_size = h * w - HW
                x_padded = torch.cat([fixed_codebook_transposed, torch.zeros(C, pad_size, device=fixed_codebook_transposed.device)], dim=1)
            else:
                x_padded = fixed_codebook_transposed
            fixed_codebook_rep = x_padded.reshape(1, C,h,w) # (1, C, h, w)
            x = self.mlp(fixed_codebook_rep)
            if self.gamma is not None:
                x = x.mul(self.gamma.reshape(1, -1, 1, 1))
            x = x.reshape(C, -1) # (C, HW)
            if h * w > HW:
                x = x[:, :HW].contiguous() # (C, HW)
            x = x.transpose(0, 1).contiguous() # (HW, C)
        else: # mlp是线性层，(N, HW, C)格式
            x = self.mlp(fixed_codebook) # (HW, C)
            if self.gamma is not None:
                x = x.mul(self.gamma.reshape(1, -1)) # (HW, C)
        self.rep_codebook=x.data.contiguous()
        del self.mlp
        del self.gamma


    def forward(self, x):
        shortcut = x
        x_conv_dw = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x_conv_dw)
            if self.token_wise_rep:
                embedding_index =  self.tq(x)
                z_q = self.rep_codebook[embedding_index] # (N, H*W, C)
                x = z_q.transpose(1, 2).reshape(x_conv_dw.shape) # (N, H*W, C) -> (N, C, H, W)
            else:
                x = self.tq(x)
                x = self.mlp(x)
        else:
            x_nhwc = x_conv_dw.permute(0, 2, 3, 1).contiguous() # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x_nhwc)
            if self.token_wise_rep:
                embedding_index =  self.tq(x)
                z_q = self.rep_codebook[embedding_index] # (N, H*W, C)
                x = z_q.reshape(x_nhwc.shape)# (N, H*W, C) -> (N, H, W, C)
            else:
                x = self.tq(x)
                x = self.mlp(x)
            x = x.permute(0, 3, 1, 2).contiguous() # (N, H, W, C) -> (N, C, H, W)


        if not self.token_wise_rep and self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)
        return x


class TQ_ConvNeXtStage(nn.Module):

    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=7,
            stride=2,
            all_stage_depth=(3, 3, 9, 3),
            stage_index=0,
            dilation=(1, 1),
            drop_path_rates=None,
            ls_init_value=1.0,
            conv_mlp=False,
            conv_bias=True,
            use_grn=False,
            act_layer='gelu',
            norm_layer=None,
            norm_layer_cl=None
    ):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = 'same' if dilation[1] > 1 else 0  # same padding needed if dilation used
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                create_conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=ds_ks,
                    stride=stride,
                    dilation=dilation[0],
                    padding=pad,
                    bias=conv_bias,
                ),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()
        depth = all_stage_depth[stage_index]
        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            sum_depth_index = sum(all_stage_depth[:stage_index])+i
            if sum_depth_index%2==0:
                stage_blocks.append(TQ_ConvNeXtBlock(
                    in_chs=in_chs,
                    out_chs=out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation[1],
                    drop_path=drop_path_rates[i],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    conv_bias=conv_bias,
                    use_grn=use_grn,
                    act_layer=act_layer,
                    norm_layer=norm_layer if conv_mlp else norm_layer_cl,
                ))
            else:
                stage_blocks.append(ConvNeXtBlock(
                    in_chs=in_chs,
                    out_chs=out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation[1],
                    drop_path=drop_path_rates[i],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    conv_bias=conv_bias,
                    use_grn=use_grn,
                    act_layer=act_layer,
                    norm_layer=norm_layer if conv_mlp else norm_layer_cl,
                ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)
        self.token_wise_rep = False
    def reparameterize(self):
        self.token_wise_rep = True
        for block in self.blocks:
            if hasattr(block, 'reparameterize'):
                block.reparameterize()
    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class TQ_ConvNeXt(nn.Module):
    r""" TQ_ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    """

    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            output_stride: int = 32,
            depths: Tuple[int, ...] = (3, 3, 9, 3),
            dims: Tuple[int, ...] = (96, 192, 384, 768),
            kernel_sizes: Union[int, Tuple[int, ...]] = 7,
            ls_init_value: Optional[float] = 1e-6,
            stem_type: str = 'patch',
            patch_size: int = 4,
            head_init_scale: float = 1.,
            head_norm_first: bool = False,
            head_hidden_size: Optional[int] = None,
            conv_mlp: bool = False,
            conv_bias: bool = True,
            use_grn: bool = False,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Optional[Union[str, Callable]] = None,
            norm_eps: Optional[float] = None,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,

            dic_dim=4,
            tq_type='TQ',
            tq_level=[5,5,5,5],
    ):
        """
        Args:
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            global_pool: Global pooling type.
            output_stride: Output stride of network, one of (8, 16, 32).
            depths: Number of blocks at each stage.
            dims: Feature dimension at each stage.
            kernel_sizes: Depthwise convolution kernel-sizes for each stage.
            ls_init_value: Init value for Layer Scale, disabled if None.
            stem_type: Type of stem.
            patch_size: Stem patch size for patch stem.
            head_init_scale: Init scaling value for classifier weights and biases.
            head_norm_first: Apply normalization before global pool + head.
            head_hidden_size: Size of MLP hidden layer in head if not None and head_norm_first == False.
            conv_mlp: Use 1x1 conv in MLP, improves speed for small networks w/ chan last.
            conv_bias: Use bias layers w/ all convolutions.
            use_grn: Use Global Response Norm (ConvNeXt-V2) in MLP.
            act_layer: Activation layer type.
            norm_layer: Normalization layer type.
            drop_rate: Head pre-classifier dropout rate.
            drop_path_rate: Stochastic depth drop rate.
        """
        super().__init__()
        assert output_stride in (8, 16, 32)
        kernel_sizes = to_ntuple(4)(kernel_sizes)
        if norm_layer is None:
            norm_layer = LayerNorm2d
            norm_layer_cl = norm_layer if conv_mlp else LayerNorm
            if norm_eps is not None:
                norm_layer = partial(norm_layer, eps=norm_eps)
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)
        else:
            assert conv_mlp,\
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            norm_layer_cl = norm_layer
            if norm_eps is not None:
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        assert stem_type in ('patch', 'overlap', 'overlap_tiered')
        if stem_type == 'patch':
            # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=conv_bias),
                norm_layer(dims[0]),
            )
            stem_stride = patch_size
        else:
            mid_chs = make_divisible(dims[0] // 2) if 'tiered' in stem_type else dims[0]
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, mid_chs, kernel_size=3, stride=2, padding=1, bias=conv_bias),
                nn.Conv2d(mid_chs, dims[0], kernel_size=3, stride=2, padding=1, bias=conv_bias),
                norm_layer(dims[0]),
            )
            stem_stride = 4

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        curr_stride = stem_stride
        dilation = 1
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(TQ_ConvNeXtStage(
                prev_chs,
                out_chs,
                kernel_size=kernel_sizes[i],
                stride=stride,
                dilation=(first_dilation, dilation),
                all_stage_depth=depths,
                stage_index=i,
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                use_grn=use_grn,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl,
            ))
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs

        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default TQ_ConvNeXt ordering (pretrained FB weights)
        if head_norm_first:
            assert not head_hidden_size
            self.norm_pre = norm_layer(self.num_features)
            self.head = ClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
            )
        else:
            self.norm_pre = nn.Identity()
            self.head = NormMlpClassifierHead(
                self.num_features,
                num_classes,
                hidden_size=head_hidden_size,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
                norm_layer=norm_layer,
                act_layer='gelu',
            )
        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)
        self.token_wise_rep = False
    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm_pre', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x, is_feat=False):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    def reparameterize(self):
        self.token_wise_rep = True
        for stage in self.stages: 
            print(f"Found module: {stage.__class__.__name__.lower()}, {isinstance(stage, nn.Sequential)}")
            if isinstance(stage, nn.Sequential):
                for name, block in stage.named_children():
                    if hasattr(block, 'reparameterize'):
                        print(f"        Found TQ module in Sequential: {block.__class__.__name__.lower()}")
                        block.reparameterize()
            else:
                if hasattr(stage, 'reparameterize'):
                    stage.reparameterize()

    def print_codebook_utilization(self):
        sum_util = 0.0
        average_util = 0.0
        count = 0
        found_modules = []  # 用于记录找到的模块及其路径
        # 递归遍历所有子模块
        for module_name, module in self.named_modules():
            # 检查模块是否是 TQ 实例（根据您的类名调整条件）
            # 例如，如果您的TQ类名为 VectorQuantizer：
            if hasattr(module, 'codebook_meter'): # 或者使用 isinstance(module, VectorQuantizer)
                utilization = module.codebook_meter.utilization # 注意：这里是属性，不是方法调用
                print(f'TQ  Module [{module_name}] Codebook Utilization: {utilization * 100:.2f}%')
                found_modules.append(module_name)
                sum_util += utilization
                count += 1
        if count > 0:
            average_util = sum_util / count
            print(f'Average TQ Codebook Utilization (across {count} modules): {average_util*100:.2f}%')
        else:
            print('No TQ modules with codebook meters found.')
        return average_util


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.zeros_(module.bias)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']

    out_dict = {}
    if 'visual.trunk.stem.0.weight' in state_dict:
        out_dict = {k.replace('visual.trunk.', ''): v for k, v in state_dict.items() if k.startswith('visual.trunk.')}
        if 'visual.head.proj.weight' in state_dict:
            out_dict['head.fc.weight'] = state_dict['visual.head.proj.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.proj.weight'].shape[0])
        elif 'visual.head.mlp.fc1.weight' in state_dict:
            out_dict['head.pre_logits.fc.weight'] = state_dict['visual.head.mlp.fc1.weight']
            out_dict['head.pre_logits.fc.bias'] = state_dict['visual.head.mlp.fc1.bias']
            out_dict['head.fc.weight'] = state_dict['visual.head.mlp.fc2.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.mlp.fc2.weight'].shape[0])
        return out_dict

    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        if 'grn' in k:
            k = k.replace('grn.beta', 'mlp.grn.bias')
            k = k.replace('grn.gamma', 'mlp.grn.weight')
            v = v.reshape(v.shape[-1])
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v

    return out_dict


def _create_tq_convnext(variant, pretrained=False, **kwargs):
    if kwargs.get('pretrained_cfg', '') == 'fcmae':
        # NOTE fcmae pretrained weights have no classifier or final norm-layer (`head.norm`)
        # This is workaround loading with num_classes=0 w/o removing norm-layer.
        kwargs.setdefault('pretrained_strict', False)

    model = build_model_with_cfg(
        TQ_ConvNeXt, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }

default_cfgs = generate_default_cfgs({
    # inputsize 3 224 224, crop-pct 0.95
    'tq_convnext_tiny': _cfg(),
    'tq_convnext_small': _cfg(),
    'tq_convnext_base': _cfg(),
    'tq_convnext_large': _cfg(),
})

@register_model
def tq_convnext_tiny(pretrained=False, **kwargs) -> TQ_ConvNeXt:
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
    model = _create_tq_convnext('tq_convnext_tiny', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def tq_convnext_small(pretrained=False, **kwargs) -> TQ_ConvNeXt:
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
    model = _create_tq_convnext('tq_convnext_small', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def tq_convnext_base(pretrained=False, **kwargs) -> TQ_ConvNeXt:
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    model = _create_tq_convnext('tq_convnext_base', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def tq_convnext_large(pretrained=False, **kwargs) -> TQ_ConvNeXt:
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
    model = _create_tq_convnext('tq_convnext_large', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
