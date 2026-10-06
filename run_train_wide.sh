#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/Coding3/celeba-dit-lab

mkdir -p outputs/flow_dit_wide_177m

# Model scaled 3.06x:
# Width: hidden_size=896 (up from 512)
# Depth: 12 blocks
# Heads: 14 heads (head_dim=64)
# Total params: 176.9M (3.06x baseline 57.8M)

PYTHONPATH=. /home/kuroko/.conda/envs/ai/bin/python train.py \
  --output-dir outputs/flow_dit_wide_177m \
  --cached-latents cached_data/celeba_train_latents_clustered.pt \
  --batch-size 32 \
  --epochs 120 \
  --hidden-size 896 \
  --depth 12 \
  --num-heads 14 \
  --lr 2e-4 \
  --num-classes 64 \
  --class-dropout 0.15 \
  --cfg-scale 4.0 \
  --log-every 200 \
  --sample-every 1000 \
  --checkpoint-every 10 \
  --sample-steps 50 \
  --sample-count 6 \
  --amp > outputs/flow_dit_wide_177m/train.log 2>&1
