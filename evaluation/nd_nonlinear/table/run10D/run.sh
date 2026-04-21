#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

python3 evaluation/nd_nonlinear/table/create_comparison.py \
    --out-dir                "$SCRIPT_DIR" \
    --data-path              data/nonlinear_10d/dataset.pt \
    --model-path             checkpoints/nonlinear_10d/model_epoch_best.pth \
    --train-config           checkpoints/nonlinear_10d/train_config.json \
    --mode                   test \
    --d-lambda               0.5 \
    --n-particles            2000 \
    --gt-path                data/nonlinear_10d/gt_test.pt \
    --gt-samples             10000 \
    --seed                   42 \
    --n-particles-svgd       500 \
    --svgd-iter              50 \
    --svgd-lr                0.2 \
    --n-particles-annealed   1000 \
    --annealed-steps         10 \
    --annealed-mcmc-per-step 2 \
    --annealed-step-size     0.1
