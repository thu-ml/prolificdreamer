#!/bin/bash

gpu=$1
prompt=$2

echo "CUDA:$gpu, Prompt: $prompt"

filename=$(echo "$prompt" | sed 's/ /-/g')
n_particles=1


CUDA_VISIBLE_DEVICES=$gpu python main.py --text "$prompt" --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles $n_particles --h 512  --w 512 --t5_iters 5000 --workspace exp-nerf-stage1/

recent_ckpt=$(find exp-nerf-stage1 -type d -name "*$filename*" -exec ls -d {}/checkpoints \; | head -n 1)

CUDA_VISIBLE_DEVICES=$gpu python main.py --text "$prompt" --iters 15000 --scale 100 --dmtet --mesh_idx 0  --init_ckpt $recent_ckpt/best_df_ep0250.pth --normal True --sds True --density_thresh 0.1 --lambda_normal 5000 --workspace exp-dmtet-stage2/

recent_ckpt=$(find exp-dmtet-stage2 -type d -name "*$filename*" -exec ls -d {}/checkpoints \; | head -n 1)

CUDA_VISIBLE_DEVICES=$gpu python main.py --text "$prompt" --iters 30000 --scale 7.5 --dmtet --mesh_idx 0  --init_ckpt $recent_ckpt/best_df_ep0150.pth --density_thresh 0.1 --finetune True --workspace exp-dmtet-stage3/
