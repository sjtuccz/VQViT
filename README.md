# Supplementary materials

Note: The Reproducibility Checklist was not attached at the end of the main text. It has been added in this supplementary material.

**VQViT-main.zip contains all the code for the paper "VQ-ViT: Accelerating Vision Transformer via Token-wise Reparametrization".**


## Training
Environments:

 `
 pytorch==3.8 torch==2.4.1 torchvision==0.19.1
 `
 
### ImageNet-1k:
Please note that for large models such as vqvit_base or above, when training on ImageNet-1k, please enable Cutmix\Mixup and droppath.

```sh
CUDA_VISIBLE_DEVICES=2 python trainvq.py -j 12 --vqtype tfsq --dict-dim 4 --fsq-level 3 3 3 3 --FLfn cos --Disfn DKD --klloss-weight 4.0 --featureloss-weight 0.0 --model vqvit_small_patch16_224 --teacher-model vit_small_patch16_224 --output ./output/in1k --dataset imagenet1k --data-dir /path/imagenet1k --initial-checkpoint ./output/in1k/vit_small_patch16_224/model_best.pth.tar --input-size 3 224 224  --sched cosine  --min-lr 1e-5 --warmup-lr 1e-4 --epochs 360 --warmup-epochs 5 --drop 0.0 --amp --cooldown-epochs 10 --featureloss-reduction sum --dictloss-weight 1.0 --clip-grad 600.0 --T 1.0 --scale 0.7 1.0  --mixup 0.0 --cutmix 0.0 --smoothing 0.0 --drop-path 0.0 -b 1024 --grad-accum-steps 4  --lr 4e-4 --opt adamw --weight-decay 0.01 --model-kwargs fsq_Tinit=-1
```

### CIFAR:
```sh
CUDA_VISIBLE_DEVICES=0 python trainvq.py -j 28 --vqtype tfsq --dict-dim 3 --fsq-level 3 3 3 --FLfn cos --Disfn DKD --klloss-weight 4.0 --featureloss-weight 1.0 --model vqvit_small_patch16_224 --teacher-model vit_small_patch16_224 --output /path/VQViT/output/in1k --dataset imagenet1k --data-dir /path/imagenet1k --initial-checkpoint /path/vit_small_patch16_224/model_best.pth.tar --input-size 3 224 224  --sched cosine  --min-lr 1e-6 --warmup-lr 1e-5 --epochs 200 --warmup-epochs 5 --drop 0.0 --amp --cooldown-epochs 10 --featureloss-reduction sum --dictloss-weight 1.0 --clip-grad 600.0 --T 1.0 --scale 0.7 1.0  --mixup 0.0 --cutmix 0.0 --smoothing 0.0 --drop-path 0.0 -b 128 --grad-accum-steps 1  --lr 1.5e-4 --opt adamw --weight-decay 0.04 --model-kwargs fsq_Tmax=3  fsq_Tinit=-1 
```

## Test
Please note that the *--reparam* option will perform token reparameterization on VQ-ViT, corresponding to the "inference phase" architecture in the text.

### ImageNet-1k:
```sh
CUDA_VISIBLE_DEVICES=3 python validate.py --model vqvit_small_patch16_224 --dataset imagenet1k --data-dir /path/imagenet1k --checkpoint /path/VQViT/vqvit_small_patch32_224/model_best.pth.tar --model-kwargs vq_type='tfsq' dic_dim=4 fsq_level=[3,3,3,3] --reparam
```
then, you will get:
```sh
"dataset": "imagenet1k",
"checkpoint": "/path/in1k/vqvit_small_patch16_224-77.69/model_best.pth.tar",
"model": "vqvit_small_patch16_224",
"top1": 77.69,
"param_count": "12.36M",
"FLOPs": "2.159G", #The FLOPs result does not include the matrix multiplication of attention. Please manually add it. For details, please refer to the function cal_qkvMatDot_FLOPs at line 189 of validate.py.
"average_batchtime": "0.101s,    9.95/s"
```

### CIFAR:
```sh
CUDA_VISIBLE_DEVICES=3 python validate.py --model vqvit_small_patch32_224 --dataset torch/cifar10 --data-dir /path/cifar10 --checkpoint /path/cifar10/in1k_pre_c10-vqvit_small_patch32_224-97.96/model_best.pth.tar --model-kwargs vq_type='tfsq' dic_dim=3 fsq_level=[3,3,3] --reparam
```
then, you will get:
```sh
"dataset": "torch/cifar10",
"checkpoint": "/path/VQViT/output/cifar10/in1k_pre_c10-vqvit_small_patch32_224-97.96/model_best.pth.tar",
"model": "vqvit_small_patch32_224",
"top1": 97.81,
"param_count": "12.18M",
"FLOPs": "590.683M", #The FLOPs result does not include the matrix multiplication of attention. Please manually add it. For details, please refer to the function cal_qkvMatDot_FLOPs at line 189 of validate.py.
"average_batchtime": "0.043s,   23.14/s" 
```

