# VQViT
under revision.......


_**Accompanying paper: 

## Code Overview

The most important code is in `astroformer.py`. We trained Astroformers using the `timm` framework, which we copied from [here](https://github.com/huggingface/pytorch-image-models).

Inside `pytorch-image-models`, we have made the following modifications. (Though one could look at the diff, we think it is convenient to summarize them here.)

- added `timm/models/vision_transformer_vq.py`
- added `trainvq.py`
- modified `timm/models/vision_transformer.py`
- modified `timm/models/optim_factory.py`
- modified `timm/models/__init__.py`
- modified `timm/optim/__init__.py`
- modified `timm/data/constants.py`
- modified `timm/models/_factory.py`
- modified `timm/utils/summary.py`
- modified `train.py`
- modified `validate.py`

## Training
 We recommend a more up-to-date training environment. Experience has proven that newer versions have faster training speeds: 
 
 `
 pytorch==3.8 torch==2.4.1 torchvision==0.19.1
 `

You could train a ViT as follows:

```sh
CUDA_VISIBLE_DEVICES=1 python train.py -j 16 --model vit_tiny_patch16_224 --output /output/cifar10 --dataset torch/cifar10 --weight-decay .3 --lr 4e-4 --input-size 3 224 224 --batch-size 128 --grad-accum-steps 1 --opt adamw --sched cosine --min-lr 1e-7 --warmup-lr 1e-5 --epochs 300 --warmup-epochs 10 --mixup 1.0 --cutmix 1.0  --smoothing 0.1 --drop 0.0 --amp  --scale 0.75 1.0 
```

 You could train a VQViT as follows:

```sh
CUDA_VISIBLE_DEVICES=0 trainvq.py --model vqvit_tiny_patch16_224 --teacher-model vit_tiny_patch16_224 --output /output/cifar100 --dataset torch/cifar100 --initial-checkpoint --input-size 3 224 224 --batch-size 128 --opt adamw --sched cosine --lr 4e-4 --min-lr 1e-7 --warmup-lr 1e-5 --epochs 500 --warmup-epochs 10 --drop 0.0 --amp --cooldown-epochs 10 --featureloss-reduction mean --featureloss-weight 2.0 --dictloss-weight 1.0 --clip-grad 300.0 --T 1.0 --scale 0.75 1.0  --mixup 1.0 --cutmix 1.0 --smoothing 0.0 --drop-path 0.0 --weight-decay 5e-3
```
or 2 GPUs (Using multiple GPUs may result in a slight decrease in accuracy compared to using a single one) :
```sh
CUDA_VISIBLE_DEVICES=0,1 python  -m torch.distributed.run --nproc_per_node=2 --master_port=29501 trainvq.py --dataset torch/cifar100 --model vqvit_tiny_patch16_224 --input-size 3 224 224 --batch-size 128 --opt adamw --sched cosine --lr 4e-4 --min-lr 1e-7 --warmup-lr 1e-5 --epochs 500 --warmup-epochs 10 --drop 0.0 --amp --output /output/CIFAR100 --teacher-model vit_tiny_patch16_224 --featureloss-reduction mean --featureloss-weight 2.0 --dictloss-weight 1.0 --clip-grad 300.0 --T 1.0 --scale 0.75 1.0  --mixup 1.0 --cutmix 1.0 --smoothing 0.0 --drop-path 0.0 --weight-decay 5e-3 --initial-checkpoint /your_vit_checkpoint_path/model_best.pth.tar --dict-num 256 --dict-dim 2
```

You could use the same script with the other VQViT models : `--model` `vqvit_small_patch16_224`, `vqvit_base_patch16_224`, `vqvit_small_patch32_224` and modify the corresponding teacher models：`--teacher-model` `vit_small_patch16_224`, `vit_base_patch16_224`, `vit_small_patch32_224` .When selecting a dataset, specify the dataset location in the command or script.  `--dataset` `torch/cifar100`,`tiny-imagenet`

## Main Results

### Imagenet-1K

| Model Name   | Top-1 Accuracy | FLOPs | Params |
|--------------|----------------|-------|--------|

### CIFAR-100

| Model Name   | Top-1 Accuracy | FLOPs | Params |
|--------------|----------------|-------|--------|


### CIFAR-10

| Model Name   | Top-1 Accuracy | FLOPs | Params |
|--------------|----------------|-------|--------|

### Tiny Imagenet

| Model Name   | Top-1 Accuracy | FLOPs | Params |
|--------------|----------------|-------|--------|

## Citation

If you use this work, please cite the following paper:

BibTeX:

```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={}
}
```

MLA:

```
```


## Credits

The code is heavily adapted from [timm](https://github.com/huggingface/pytorch-image-models).
