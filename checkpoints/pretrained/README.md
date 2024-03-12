# NAYER

This repo implements the Unleash Data Generation for Efficient and Effective Data-Free Knowledge Distillation

## Quick Start

### 1. Prepare the files

To reproduce our results, please download pre-trained teacher models from [Dropbox-Models (266 MB)](https://www.dropbox.com/sh/w8xehuk7debnka3/AABhoazFReE_5mMeyvb4iUWoa?dl=0) and extract them as `checkpoints/pretrained`.
Instead, you can train a model from scratch as follows.
```bash
python train_scratch.py --model wrn40_2 --dataset cifar10 --batch-size 256 --lr 0.1 --epoch 200 --gpu 0
```
   
### 2. Reproduce our results
* To get similar results of our method on CIFAR datasets, run the script in `scripts/'. (A sample is shown below) 
  Synthesized images and logs will be saved in `checkpoints/nayer`.
    ```bash
    # g-steps is the number of iterations in synthesizing
    python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 120 \
    --dataset cifar100 --method nayer --lr_g 4e-3 --teacher wrn40_2 --student wrn16_2 --save_dir run/c100w402w162-nayer \
    --adv 1.33 --bn 10.0 --oh 0.5 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 \
    --log_tag c100w402w162-nayer-ep120
    ```
