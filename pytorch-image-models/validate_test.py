#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import csv
import glob
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600' 
import torch
import torch.nn as nn
import torch.nn.parallel

from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.layers import apply_test_time_pool, set_fast_norm
from timm.models import create_model, load_checkpoint, is_model, list_models
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser, \
    decay_batch_step, check_batch_size_retry, ParseKwargs, reparameterize_model
from timm.data.constants import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD, TINY_IMAGENET_MEAN, TINY_IMAGENET_STD,IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=0.875, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--amp-impl', default='native', type=str,
                    help='AMP impl to use, "native" or "apex" (default: native)')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
parser.add_argument('--reparam', default=False, action='store_true',
                    help='token-reparameterize model')
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)


scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")

parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--results-format', default='csv', type=str,
                    help='Format for results file one of (csv, json) (default: csv).')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--retry', default=False, action='store_true',
                    help='Enable batch size decay & retry for single model validation')

# for vqvit pre_calculate
parser.add_argument('--repara', action='store_true', default=False,
                    help='vqvit pre calculate .')



def add_attention_qkvMatDot_FLOPs(flops, model_name)-> int:
    if 'vit_tiny_patch16_224' in model_name:
        return flops + cal_qkvMatDot_FLOPs(batch=1,head_num=3,seq_len=197,dim=192,block_num=12)
    elif 'vit_small_patch16_224' in model_name:
        return flops + cal_qkvMatDot_FLOPs(batch=1,head_num=6,seq_len=197,dim=384,block_num=12)
    elif 'vit_base_patch16_224' in model_name:
        return flops + cal_qkvMatDot_FLOPs(batch=1,head_num=12,seq_len=197,dim=768,block_num=12)
    elif 'vit_small_patch32_224' in model_name:
        return flops + cal_qkvMatDot_FLOPs(batch=1,head_num=6,seq_len=50,dim=384,block_num=12)
    else:
        print("Model attention qkvMatDot FLOPs not added!")
        return flops
    
    

def cal_qkvMatDot_FLOPs(batch=1,head_num=6,seq_len=197,dim=384,block_num=12):
    
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

    This code cannot be automatically calculated for FLOPs by these packages: ptflops calflops ptflops fvcore thop
    
    (vq)vit-t-16: batch=1,head_num=3,seq_len=197,dim=192,block_num=12 ,result: (180M)180.23M
    (vq)vit-s-16: batch=1,head_num=6,seq_len=197,dim=384,block_num=12  ,result: (360.005M)360.46M
    (vq)vit-s-32: batch=1,head_num=6,seq_len=50,dim=384,block_num=12    ,result: (23.11M)23.22M
    (vq)vit-s-32(384): batch=1,head_num=6,seq_len= 145,dim=384,block_num=12, result: (194.95MB)195.28M
    (vq)vit-b-16: batch=1,head_num=12,seq_len=197,dim=768,block_num=12  ,result: (0.72G)0.72G
    '''
    b=batch
    h = head_num
    n = seq_len
    d=dim//head_num
    # if isvq:
    #     FLOPs= 0.5*block_num*((2*d-1)*n*n*b*h + 3*b*h*n*n-1 + (2*n-1)*n*d*b*h) #MACs -> FLOPs
    # else:
    FLOPs= 0.5*block_num*(b*h*n*d + (2*d-1)*n*n*b*h + 3*b*h*n*n-1 + (2*n-1)*n*d*b*h)
    return FLOPs

def format_param_count(param_count, decimal_places=2):
    if param_count >= 1e9:
        return f"{round(param_count / 1e9, decimal_places)}B" 
    elif param_count >= 1e6:
        return f"{round(param_count / 1e6, decimal_places)}M"  
    else:
        return f"{round(param_count / 1e3, decimal_places)}K" 
def validate(args):
    # might as well try to validate something
    # args.pretrained = args.pretrained or not args.checkpoint
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_autocast = suppress
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            assert args.amp_dtype == 'float16'
            use_amp = 'apex'
            _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            assert args.amp_dtype in ('float16', 'bfloat16')
            use_amp = 'native'
            amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
            _logger.info('Validating in mixed precision with native PyTorch AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)

    if args.fast_norm:
        set_fast_norm()

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=in_chans,
        global_pool=args.gp,
        scriptable=args.torchscript,
        **args.model_kwargs,
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True,
    )
    
    # if args.pre_calculate:
        # model.pre_calculate()
    
    
    # _logger.info('Model %s created, param count: %d' % (args.model, param_count))
    
    if use_amp == 'apex':
        model = amp.initialize(model, opt_level='O1')

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    crop_pct = data_config['crop_pct']
    
    input=torch.randn(1,data_config['input_size'][0],data_config['input_size'][1],data_config['input_size'][2]).to(device)
    if args.reparam:
        model.eval()
        model = model.to(device)
        output_before_reparam=model(input)
        if isinstance(output_before_reparam, tuple):
            output_before_reparam = output_before_reparam[0]
        model = model.to('cpu')
        model.reparameterize()
    for name, param in model.named_parameters():
        num_params = param.numel()
        print(f"{name:35} | Shape: {str(list(param.shape)):20} | Params: {num_params:10,}")


    from thop import profile, clever_format
    model.eval()
    model = model.to(device)
    output=model(input)
    if isinstance(output, tuple):
        output = output[0]
    if args.reparam:
        is_same = torch.allclose(output_before_reparam, output, atol=1e-3)
        print("output is consistent Before and after token wise reparameterization:" ,is_same)
    param_count = sum([m.numel() for m in model.parameters()])
    flops, params = profile(model, inputs=(input,))
    if 'vit' in args.model:
        flops = add_attention_qkvMatDot_FLOPs(flops, args.model)
    if 'swin' in args.model:
        if hasattr(model, 'flops'):
            flops = model.flops()
        else:
            raise NotImplementedError("Please implement flops calculation function for swin model")
    flops, params = clever_format([flops, params], "%.3f")

    if 'vq' in args.model:
        average_util = model.print_codebook_utilization()
    results = OrderedDict(
        model=args.model,
        param_count=format_param_count(param_count),
        FLOPs=flops,
        img_size=data_config['input_size'],
        crop_pct=crop_pct,
        interpolation=data_config['interpolation'],
        params_thop=params,
        codebook_utilization = f"{average_util:.2%}" if 'vq' in args.model else 'N/A'
    )

    return results


def _try_run(args, initial_batch_size):
    batch_size = initial_batch_size
    results = OrderedDict()
    error_str = 'Unknown'
    while batch_size:
        args.batch_size = batch_size * args.num_gpu  # multiply by num-gpu for DataParallel case
        try:
            if torch.cuda.is_available() and 'cuda' in args.device:
                torch.cuda.empty_cache()
            results = validate(args)
            return results
        except RuntimeError as e:
            error_str = str(e)
            _logger.error(f'"{error_str}" while running validation.')
            if not check_batch_size_retry(error_str):
                break
        batch_size = decay_batch_step(batch_size)
        _logger.warning(f'Reducing batch size to {batch_size} for retry.')
    results['error'] = error_str
    _logger.error(f'{args.model} failed to validate ({error_str}).')
    return results


_NON_IN1K_FILTERS = ['*_in21k', '*_in22k', '*in12k', '*_dino', '*fcmae', '*seer']


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(
                pretrained=True,
                exclude_filters=_NON_IN1K_FILTERS,
            )
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(
                args.model,
                pretrained=True,
            )
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            initial_batch_size = args.batch_size
            for m, c in model_cfgs:
                args.model = m
                args.checkpoint = c
                r = _try_run(args, initial_batch_size)
                if 'error' in r:
                    continue
                if args.checkpoint:
                    r['checkpoint'] = args.checkpoint
                results.append(r)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
    else:
        if args.retry:
            results = _try_run(args, args.batch_size)
        else:
            results = validate(args)

    if args.results_file:
        write_results(args.results_file, results, format=args.results_format)

    # output results in JSON to stdout w/ delimiter for runner script
    print(f'--result\n{json.dumps(results, indent=4)}')


def write_results(results_file, results, format='csv'):
    with open(results_file, mode='w') as cf:
        if format == 'json':
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()



if __name__ == '__main__':
    main()
    # batch=1,head_num=6,seq_len= 145,dim=384,block_num=12
    # result = cal_qkvMatDot_FLOPs(batch=1,head_num=6,seq_len=145,dim=384,block_num=12, isvq=True)
    # print(result)  
