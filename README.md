## Spatio-Temporal Masked Autoencoders: A PyTorch Implementation

This is a PyTorch/GPU implementation of the paper.

## Dataset

+ First, you need to go to the [official website](https://ultrasuite.github.io/data/uxtd/) to download the UXTD data set and process it into a classified data set.

## Getting Started

### pretrain

+ To pre-train ViT-Base with **distributed training**, run the following on 1 nodes with 1 GPUs each, you can modify the params depend on your own server:

```
PORT=${PORT:-29501}

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python -m torch.distributed.run --nproc_per_node 1 --master_port=$PORT main_pretrain.py \
    --accum_iter 4 \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 20 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path xxxxx \
    --output_dir outputs
```

### finetune

- You need to modify the `finetune` and `data_path`.

```
PORT=${PORT:-29502}

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node 1 --master_port=$PORT main_finetune.py \
    --batch_size 256 \
    --finetune xxxx \
    --model vit_base_patch16 \
    --epochs 100 \
    --data_path xxxx \
    --output_dir outputs \
    --nb_classes 5 \
    --blr 3e-4 --layer_decay 0.65 --mixup 0.5 --cutmix 0.5 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25
```


### eval

+ you can eval the classification result by the following orders.

```
PORT=${PORT:-29503}

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node 1 --master_port=$PORT main_finetunev2.py \
    --batch_size 512 \
    --finetune xxxx \
    --model vit_base_patch16 \
    --data_path xxxx \
    --num_workers 16 \
    --nb_classes 5 \
    --eval
```
