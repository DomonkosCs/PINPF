#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

python3 evaluation/4d_gmm/table/create_comparison.py \
    --out-dir                "$SCRIPT_DIR" \
    --data-path              data/dataset_gmm_4d.pt \
    --model-path             checkpoints/gmm_4d/model_epoch_best.pth \
    --train-config           checkpoints/gmm_4d/train_config.json \
    --nsf-path               checkpoints/neural_spline_flow/flow_best.pt \
    --mode                   test \
    --d-lambda               0.5 \
    --n-particles            1500 \
    --n-particles-nsf        1500 \
    --n-particles-svgd       500 \
    --gt-samples             10000 \
    --seed                   42 \
    --svgd-iter              500 \
    --svgd-lr                0.2 \
    --n-particles-annealed   1000 \
    --annealed-steps         10 \
    --annealed-mcmc-per-step 5 \
    --annealed-step-size     0.1
