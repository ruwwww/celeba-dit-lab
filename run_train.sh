#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/Coding3/celeba-dit-lab

mkdir -p outputs/flow_dit

PYTHONPATH=. /home/kuroko/.conda/envs/ai/bin/python train.py \
  --output-dir outputs/flow_dit \
  --batch-size 32 \
  --epochs 30 \
  --lr 2e-4 \
  --workers 0 \
  --log-every 50 \
  --sample-every 500 \
  --checkpoint-every 1 \
  --sample-steps 30 \
  --sample-count 6 \
  --amp > outputs/flow_dit/train.log 2>&1
