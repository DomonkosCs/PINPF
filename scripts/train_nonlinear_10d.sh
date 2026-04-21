#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python train.py \
  --seed 2468 \
  --n-batch 200 \
  --results-dir "results/run_nonlinear_10d/ckp" \
  --epochs 5000 \
  --lr 0.008 \
  --n-particles 500 \
  --d-lambda 0.01 \
  --checkpoint-freq 100 \
  --weight-decay 0.0001 \
  --log-dir "results/run_nonlinear_10d/log" \
  --layers 6 \
  --hidden-dim 64 \
  --mini-batch-size 128 \
  --grad-clip 5.0 \
  --features "grad_log_p,log_h,grad_log_h" \
  --data-path "data/nonlinear_10d/dataset.pt" \
  --config-type nonlinear \
  --feature-clamp 100000.0 \
  --particle-bounds 50.0 \
  --log-prob-floor -300.0
