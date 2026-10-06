#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/Coding3/celeba-dit-lab

mkdir -p outputs/flow_dit_cfg

PYTHONPATH=. /home/kuroko/.conda/envs/ai/bin/python train.py \
  --output-dir outputs/flow_dit_cfg \
  --cached-latents cached_data/celeba_train_latents_clustered.pt \
  --batch-size 32 \
  --epochs 120 \
  --lr 2e-4 \
  --num-classes 64 \
  --class-dropout 0.15 \
  --cfg-scale 3.0 \
  --log-every 200 \
  --sample-every 1000 \
  --checkpoint-every 10 \
  --sample-steps 50 \
  --sample-count 6 \
  --amp > outputs/flow_dit_cfg/train.log 2>&1
