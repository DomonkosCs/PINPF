# PINPF — Physics-Informed Neural Particle Flow

Code for the paper **"Physics-informed neural particle flow for the Bayesian update step"**.

For theoretical background, derivations, and experimental details see the paper (TBA). This README covers installation and reproducing the main comparison tables.

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
