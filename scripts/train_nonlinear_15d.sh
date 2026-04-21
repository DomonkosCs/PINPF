#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python train.py \
  --seed 1234 \
  --n-batch 1000 \
  --results-dir "results/run_nonlinear_15d/ckp" \
  --epochs 5000 \
  --lr 0.006 \
  --n-particles 1000 \
  --d-lambda 0.01 \
  --checkpoint-freq 50 \
  --weight-decay 0.0004 \
  --log-dir "results/run_nonlinear_15d/log" \
  --layers 8 \
  --hidden-dim 128 \
  --mini-batch-size 128 \
  --grad-clip 5.0 \
  --features "grad_log_p,log_h,grad_log_h" \
  --data-path "data/nonlinear_15d/dataset.pt" \
  --config-type nonlinear \
  --feature-clamp 100000.0 \
  --particle-bounds 50.0 \
  --log-prob-floor -300.0
