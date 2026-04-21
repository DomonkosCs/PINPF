#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

python3 evaluation/2d_tdoa/table/create_comparison.py \
    --out-dir                "$SCRIPT_DIR" \
    --data-path              data/dataset_tdoa.pt \
    --model-path             checkpoints/tdoa/model_epoch_best.pth \
    --train-config           checkpoints/tdoa/train_config.json \
    --mode                   test \
    --d-lambda               1.0 \
    --n-particles            1000 \
    --seed                   42 \
    --n-particles-svgd       1000 \
    --svgd-iter              200 \
    --svgd-lr                0.2 \
    --n-particles-annealed   1000 \
    --annealed-steps         10 \
    --annealed-mcmc-per-step 3 \
    --annealed-step-size     0.1
