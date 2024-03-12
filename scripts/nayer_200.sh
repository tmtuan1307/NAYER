#!/bin/bash
# modify --dataset (and nothing else) to apply to CIFAR10

### CIFAR100
# c100r34r18
python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 220 \
--dataset cifar100 --method nayer --lr_g 4e-3 --teacher resnet34 --student resnet18 --save_dir run/c100r34r18-nayer \
--adv 1.33 --bn 10.0 --oh 0.5 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c100r34r18-nayer-ep220

# c100w402w161
python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 220 \
--dataset cifar100 --method nayer --lr_g 4e-3 --teacher wrn40_2 --student wrn16_1 --save_dir run/c100w402w161-nayer \
--adv 1.33 --bn 10.0 --oh 0.5 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c100w402w161-nayer-ep220

# c100w402w162
python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 220 \
--dataset cifar100 --method nayer --lr_g 4e-3 --teacher wrn40_2 --student wrn16_2 --save_dir run/c100w402w162-nayer \
--adv 1.33 --bn 10.0 --oh 0.5 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c100w402w162-nayer-ep220

# c100w402w401
python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 220 \
--dataset cifar100 --method nayer --lr_g 4e-3 --teacher wrn40_2 --student wrn40_1 --save_dir run/c100w402w401-nayer \
--adv 1.33 --bn 10.0 --oh 0.5 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c100w402w401-nayer-ep220

# c100vgg11r18
python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 220 \
--dataset cifar100 --method nayer --lr_g 4e-3 --teacher vgg11 --student resnet18 --save_dir run/c100vgg11r18-nayer \
--adv 1.33 --bn 10.0 --oh 0.5 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c100vgg11r18-nayer-ep220


### CIFAR10
# c10r34r18
python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 220 \
--dataset cifar10 --method nayer --lr_g 4e-3 --teacher resnet34 --student resnet18 --save_dir run/c10r34r18-nayer \
--adv 1.33 --bn 10.0 --oh 0.5 --g_steps 30 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c10r34r18-nayer

# c10w402w161
python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 220 \
--dataset cifar10 --method nayer --lr_g 4e-3 --teacher wrn40_2 --student wrn16_1 --save_dir run/c10w402w161-nayer \
--adv 1.33 --bn 10.0 --oh 0.5 --g_steps 30 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c10w402w161-nayer

# c10w402w162
python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 220 \
--dataset cifar10 --method nayer --lr_g 4e-3 --teacher wrn40_2 --student wrn16_2 --save_dir run/c10w402w162-nayer \
--adv 1.33 --bn 10.0 --oh 0.5 --g_steps 30 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c10w402w162-nayer

# c10w402w401
python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 220 \
--dataset cifar10 --method nayer --lr_g 4e-3 --teacher wrn40_2 --student wrn40_1 --save_dir run/c10w402w401-nayer \
--adv 1.33 --bn 10.0 --oh 0.5 --g_steps 30 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c10w402w401-nayer

# c10vgg11r18
python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 220 \
--dataset cifar10 --method nayer --lr_g 4e-3 --teacher vgg11 --student resnet18 --save_dir run/c10vgg11r18-nayer \
--adv 1.33 --bn 10.0 --oh 0.5 --g_steps 30 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c10vgg11r18-nayer

### TinyImageNet
# tir34r18
python datafree_kd.py --batch_size 256 --synthesis_batch_size 200 --lr 0.1 --gpu 0 --warmup 20 --epochs 220 \
--dataset tiny_imagenet --method nayer --lr_g 4e-3 --teacher resnet34 --student resnet18 --save_dir run/tir34r18-nayer \
--adv 1.33 --bn 10.0 --oh 0.5 --g_steps 60 --g_life 10 --g_loops 4 --gwp_loops 20 --kd_steps 1000 --ep_steps 1000 \
--log_tag tir34r18-nayer-ep220