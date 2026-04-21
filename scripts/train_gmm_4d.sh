#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python train.py \
  --seed 43 \
  --n-batch 1000 \
  --results-dir "results/run_4d_gmm/ckp" \
  --epochs 6000 \
  --lr 0.004 \
  --n-particles 500 \
  --d-lambda 0.01 \
  --checkpoint-freq 100 \
  --weight-decay 0.0001 \
  --log-dir "results/run_4d_gmm/log" \
  --layers 6 \
  --hidden-dim 64 \
  --mini-batch-size 64 \
  --grad-clip 1.0 \
  --features "grad_log_p,log_h,grad_log_h" \
  --data-path "data/dataset_gmm_4d.pt" \
  --config-type gmm \
  --use-exact-divergence \
  --feature-clamp 100000.0 \
  --particle-bounds 50.0 \
  --log-prob-floor -300.0
