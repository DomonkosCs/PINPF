#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python train.py \
  --seed 42 \
  --n-batch 1000 \
  --results-dir "results/run_tdoa/ckp" \
  --epochs 6000 \
  --lr 0.005 \
  --n-particles 250 \
  --d-lambda 0.01 \
  --checkpoint-freq 100 \
  --weight-decay 0.0001 \
  --log-dir "results/run_tdoa/log" \
  --layers 6 \
  --hidden-dim 64 \
  --mini-batch-size 64 \
  --grad-clip 1.0 \
  --features "grad_log_p,log_h,grad_log_h" \
  --data-path "data/dataset_tdoa.pt" \
  --config-type tdoa \
  --use-exact-divergence \
  --feature-clamp 100000.0 \
  --particle-bounds 50.0 \
  --log-prob-floor -300.0
