# PINPF — Physics-Informed Neural Particle Flow

This repository accompanies our paper, published in *Knowledge-Based Systems* (Elsevier):
> [Physics-informed neural particle flow for the Bayesian update step](https://www.sciencedirect.com/science/article/pii/S0950705126009354)

The full text is freely available under the journal's open access policy. This README covers installation and reproducing the main comparison tables.

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.x are required. A CUDA-capable GPU is strongly recommended for training.

---

## Repository layout

```
data/              # datasets (included)
checkpoints/       # pre-trained models (included)
evaluation/        # per-experiment table generation scripts
  2d_tdoa/
  4d_gmm/
  nd_nonlinear/
flow/              # dynamics, integrators, FPE loss
models/            # neural network architecture
problems/          # prior/measurement models and dataset configs
training/          # training loop
scripts/           # training shell scripts
train.py           # training entry point
```

---

## Reproducing the comparison tables

Pre-trained checkpoints are provided in `checkpoints/`. Run the evaluation scripts from the repository root:

```bash
bash evaluation/4d_gmm/table/run.sh
bash evaluation/2d_tdoa/table/run.sh
bash evaluation/nd_nonlinear/table/run15D/run.sh
bash evaluation/nd_nonlinear/table/run10D/run.sh
```

Each script writes `comparison.pt`, `comparison.txt`, `comparison.tex`, and `comparison.csv` to its own directory.

---

## Training from scratch

Training scripts with the exact hyperparameters used in the paper are in `scripts/`:

```bash
bash scripts/train_gmm_4d.sh
bash scripts/train_tdoa.sh
bash scripts/train_nonlinear_10d.sh
bash scripts/train_nonlinear_15d.sh
```

Each run saves checkpoints and a `config.json` to `results/<run_name>/`.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{csuzdi2026physics,
title = {Physics-informed neural particle flow for the Bayesian update step},
journal = {Knowledge-Based Systems},
volume = {346},
pages = {116209},
year = {2026},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2026.116209},
url = {https://www.sciencedirect.com/science/article/pii/S0950705126009354},
author = {Domonkos Csuzdi and Tamás Bécsi and Olivér Törő},
keywords = {Particle flow, Physics-informed learning, Bayesian update, Amortized inference, Log-homotopy},
}
```
