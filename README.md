# Supplementary materials

VQ-ViT: Accelerating Vision Transformer via Token-wise Reparametrization

Note: The Reproducibility Checklist was not attached at the end of the main text. It has been added in this supplementary material.

## Code Overview

 We trained VQ-ViT using the `timm` framework. Inside `pytorch-image-models`, we have made the following modifications. (Though one could look at the diff, we think it is convenient to summarize them here.)

- added `timm/models/vision_transformer_vq.py`: VQ-ViT model
- added `trainvq.py`: training script for VQ-ViT
- added `distill.py` : distillation method for VQ-ViT, used in `trainvq.py`
- modified `validate.py`: for the validation of token-wise reparameterized VQVIT
- modified `timm/models/_factory.py`: *create_teacher_model* function is added, used in `trainvq.py`

- modified `timm/models/vision_transformer.py`: Modify to output intermediate layer features for the distillation of VQViT
- modified `timm/optim/optim_factory.py`: The weight decay  of the nn.embedding is set to 0.
- modified `timm/models/__init__.py`: Add VQ-ViT model
- modified `timm/optim/__init__.py`: *param_group_fn_with_weight_decay_vq* function .etc is added , [Non-essential utilities] Used for debugging/analysis, safe to ignore.
- modified `timm/data/constants.py`: Mean and variance of the dataset
- modified `timm/utils/summary.py`[Non-essential utilities]
- modified `train.py`[Non-essential utilities]


## Training
 We recommend a more up-to-date training environment. Experience has proven that newer versions have faster training speeds: 
 The training environment we are using:

 `
 pytorch==3.8 torch==2.4.1 torchvision==0.19.1
 `

Other environments:

 `
 Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-139-generic x86_64), 
  GPU: Nvidia A100, 
  CPU:  Intel(R) Xeon(R) Gold 6346 CPU @ 3.10GHz, 
 NVIDIA-SMI Driver Version: 535.183.01
 `

 You could train a VQViT as follows (Detailed parameters):

On ImageNet-1k:
Please note that for large models such as vqvit_base or above, when training on ImageNet-1k, please enable Cutmix\Mixup and droppath.

```sh
CUDA_VISIBLE_DEVICES=2 python trainvq.py -j 12 --vqtype tfsq --dict-dim 4 --fsq-level 3 3 3 3 --FLfn cos --Disfn DKD --klloss-weight 4.0 --featureloss-weight 0.0 --model vqvit_small_patch16_224 --teacher-model vit_small_patch16_224 --output ./output/in1k --dataset imagenet1k --data-dir /path/imagenet1k --initial-checkpoint ./output/in1k/vit_small_patch16_224/model_best.pth.tar --input-size 3 224 224  --sched cosine  --min-lr 1e-5 --warmup-lr 1e-4 --epochs 360 --warmup-epochs 5 --drop 0.0 --amp --cooldown-epochs 10 --featureloss-reduction sum --dictloss-weight 1.0 --clip-grad 600.0 --T 1.0 --scale 0.7 1.0  --mixup 0.0 --cutmix 0.0 --smoothing 0.0 --drop-path 0.0 -b 1024 --grad-accum-steps 4  --lr 4e-4 --opt adamw --weight-decay 0.01 --model-kwargs fsq_Tinit=-1
```
On CIFAR:
```sh
CUDA_VISIBLE_DEVICES=0 python trainvq.py -j 28 --vqtype tfsq --dict-dim 3 --fsq-level 3 3 3 --FLfn cos --Disfn DKD --klloss-weight 4.0 --featureloss-weight 1.0 --model vqvit_small_patch16_224 --teacher-model vit_small_patch16_224 --output /path/VQViT/output/in1k --dataset imagenet1k --data-dir /path/imagenet1k --initial-checkpoint /path/vit_small_patch16_224/model_best.pth.tar --input-size 3 224 224  --sched cosine  --min-lr 1e-6 --warmup-lr 1e-5 --epochs 200 --warmup-epochs 5 --drop 0.0 --amp --cooldown-epochs 10 --featureloss-reduction sum --dictloss-weight 1.0 --clip-grad 600.0 --T 1.0 --scale 0.7 1.0  --mixup 0.0 --cutmix 0.0 --smoothing 0.0 --drop-path 0.0 -b 128 --grad-accum-steps 1  --lr 1.5e-4 --opt adamw --weight-decay 0.04 --model-kwargs fsq_Tmax=3  fsq_Tinit=-1 
```


 You could validate a VQViT as follows:

 Please note that the *--reparam* option will perform token reparameterization on VQ-ViT, corresponding to the "inference phase" architecture in the text.

On ImageNet-1k:
```sh
CUDA_VISIBLE_DEVICES=3 python validate.py --model vqvit_small_patch16_224 --dataset imagenet1k --data-dir /path/imagenet1k --checkpoint /path/VQViT/vqvit_small_patch32_224/model_best.pth.tar --model-kwargs vq_type='tfsq' dic_dim=4 fsq_level=[3,3,3,3] --reparam
```
then, you will get:
```sh
"dataset": "imagenet1k",
    "checkpoint": "/path/VQViT/output/cifar10/in1k_pre_c10-vqvit_small_patch32_224-97.96/model_best.pth.tar",
    "model": "vqvit_small_patch16_224",
    "top1": 77.69,
    "param_count": "12.36M",
    "FLOPs": "2.159G", #The FLOPs result does not include the matrix multiplication of attention. Please manually add it. For details, please refer to the function cal_qkvMatDot_FLOPs at line 189 of validate.py.
    "average_batchtime": "0.101s,    9.95/s"
```

On CIFAR:
```sh
CUDA_VISIBLE_DEVICES=3 python validate.py --model vqvit_small_patch32_224 --dataset torch/cifar10 --data-dir /path/cifar10 --checkpoint /path/VQViT/output/cifar10/in1k_pre_c10-vqvit_small_patch32_224-97.96/model_best.pth.tar --model-kwargs vq_type='tfsq' dic_dim=3 fsq_level=[3,3,3] --reparam
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


## Credits

The code is heavily adapted from timm (huggingface/pytorch-image-models)
