"""
Full baseline comparison for the TDOA model on the test split.

Evaluates PINPF, Incompressible Flow, Local Gaussian Exact Flow,
Mean Gaussian Exact Flow, SVGD, and Annealed MCMC against precomputed
ground truth posterior samples. Prints LaTeX and markdown tables.
"""

import csv
import json
import os
import sys
import time
import argparse

import torch
from torch import nn
import numpy as np

# Project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from problems.configs import ConfigTDOA
from models import NeuralFlowModel, load_neural_flow_model
from flow import (
    NeuralFlow,
    IncompressibleFlow,
    LocalGaussianExactFlow,
    MeanGaussianExactFlow,
    create_euler_adaptive,
)
from utils import (
    sliced_wasserstein_distance,
    energy_distance,
    generate_flow_samples,
    parse_features,
)
from svgd import run_svgd
from amcmc import run_annealed_mcmc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Helpers ─────────────────────────────────────────────────────────────────


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def _timed(fn):
    """Run fn(), return (result, elapsed)."""
    _sync()
    t0 = time.time()
    result = fn()
    _sync()
    return result, time.time() - t0


def _make_task(config, i):
    """Create per-task sub-models and return GT samples."""
    prior_sub = config.prior_model_const(
        mu_prior=config.prior_model.mu_prior[i].unsqueeze(0).to(DEVICE),
        var_prior=config.prior_model.var_prior[i].unsqueeze(0).to(DEVICE),
    )
    meas_sub = config.meas_model_const(
        sigma=config.meas_model.sigma[i].unsqueeze(0).to(DEVICE),
    )
    z_task = config.zs[i].unsqueeze(0).to(DEVICE)
    gt_i = config.true_samples[i].to(DEVICE)  # (N_gt, D)
    return prior_sub, meas_sub, z_task, gt_i


# ── Output ──────────────────────────────────────────────────────────────────


def save_summary(results_dict, n_tasks, out_dir):
    """Save combined summary (.txt, .tex, .pt, .csv) after all methods are done."""

    # ── Combined .pt ──
    pt_path = os.path.join(out_dir, "comparison.pt")
    torch.save(
        {
            name: {
                "ed": torch.tensor(res["ed"], dtype=torch.float32),
                "swd": torch.tensor(res["swd"], dtype=torch.float32),
                "time": res["time"],
            }
            for name, res in results_dict.items()
        },
        pt_path,
    )
    print(f"Saved {pt_path}")

    # ── Plain text ──
    txt_path = os.path.join(out_dir, "comparison.txt")
    with open(txt_path, "w") as f:
        header = f"{'Method':<35} {'ED':>18} {'SWD':>18} {'Time/task':>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for name, res in results_dict.items():
            ed_arr = np.array(res["ed"])
            swd_arr = np.array(res["swd"])
            t = res["time"]
            f.write(
                f"{name:<35} "
                f"{ed_arr.mean():>6.4f}+/-{ed_arr.std():<7.4f} "
                f"{swd_arr.mean():>6.4f}+/-{swd_arr.std():<7.4f} "
                f"{t / n_tasks:>8.4f}s\n"
            )
    print(f"Saved {txt_path}")

    # ── LaTeX ──
    tex_path = os.path.join(out_dir, "comparison.tex")
    with open(tex_path, "w") as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"    \centering" + "\n")
        f.write(
            f"    \\caption{{Performance summary over {n_tasks} test tasks "
            r"(mean $\pm$ std).}" + "\n"
        )
        f.write(r"    \label{tab:tdoa_comparison}" + "\n")
        f.write(r"    \begin{tabular}{@{}lccc@{}}" + "\n")
        f.write(r"        \toprule" + "\n")
        f.write(r"        Method & ED & SWD & Time [s] \\" + "\n")
        f.write(r"        \midrule" + "\n")
        for name, res in results_dict.items():
            ed_arr = np.array(res["ed"])
            swd_arr = np.array(res["swd"])
            t = res["time"]
            f.write(
                f"        {name:<35} & "
                f"${ed_arr.mean():.4f} \\pm {ed_arr.std():.4f}$ & "
                f"${swd_arr.mean():.4f} \\pm {swd_arr.std():.4f}$ & "
                f"{t / n_tasks:.4f} \\\\\n"
            )
        f.write(r"        \bottomrule" + "\n")
        f.write(r"    \end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"Saved {tex_path}")

    # ── CSV (per-task) ──
    csv_path = os.path.join(out_dir, "comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Sample_Index", "ED", "SWD"])
        for name, res in results_dict.items():
            for idx, (ed_v, swd_v) in enumerate(zip(res["ed"], res["swd"])):
                writer.writerow([name, idx, ed_v, swd_v])
    print(f"Saved {csv_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Full baseline comparison for TDOA model on test split."
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to best PINPF checkpoint.",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        required=True,
        help="Path to training config JSON (provides layers, hidden_dim, features).",
    )
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--d-lambda", type=float)
    parser.add_argument("--n-particles", type=int)
    parser.add_argument("--n-particles-svgd", type=int)
    parser.add_argument("--n-particles-annealed", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svgd-iter", type=int)
    parser.add_argument("--svgd-lr", type=float)
    parser.add_argument("--annealed-steps", type=int)
    parser.add_argument("--annealed-mcmc-per-step", type=int)
    parser.add_argument("--annealed-step-size", type=float)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write output files. Defaults to the script's own directory.",
    )
    args = parser.parse_args()

    # Read model architecture from train config
    with open(args.train_config) as f:
        train_cfg = json.load(f)
    args.layers = train_cfg["layers"]
    args.hidden_dim = train_cfg["hidden_dim"]
    args.features = train_cfg.get("features", "grad_log_p,log_h,grad_log_h")

    extra_features = parse_features(args.features)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else SCRIPT_DIR
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    print(f"Device: {DEVICE}")
    print(
        f"Model: layers={args.layers}, hidden_dim={args.hidden_dim}, "
        f"features={args.features}"
    )

    config = ConfigTDOA(
        device=DEVICE,
        data_path=args.data_path,
        mode=args.mode,
        max_samples=args.max_samples,
    )
    n_tasks = config.n_batch
    print(f"Split: {args.mode}, tasks: {n_tasks}, dim: {config.state_dim}")

    # Load PINPF model
    print(f"Loading model from {args.model_path}...")
    model = NeuralFlowModel(
        activation=nn.SiLU(),
        layers=args.layers,
        neurons_per_layer=args.hidden_dim,
        state_dim=config.state_dim,
        meas_dim=config.meas_dim,
        extra_features=extra_features,
    )
    model = load_neural_flow_model(model, args.model_path, device=DEVICE)

    # Save eval config
    eval_config = {
        "data_path": args.data_path,
        "model_path": args.model_path,
        "train_config": args.train_config,
        "mode": args.mode,
        "layers": args.layers,
        "hidden_dim": args.hidden_dim,
        "features": args.features,
        "d_lambda": args.d_lambda,
        "n_particles": args.n_particles,
        "seed": args.seed,
        "n_tasks": n_tasks,
        "baselines": {
            "svgd": {
                "n_particles": args.n_particles_svgd,
                "n_iter": args.svgd_iter,
                "lr": args.svgd_lr,
            },
            "annealed_mcmc": {
                "n_particles": args.n_particles_annealed,
                "n_steps": args.annealed_steps,
                "n_mcmc_per_step": args.annealed_mcmc_per_step,
                "step_size": args.annealed_step_size,
            },
        },
    }
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(eval_config, f, indent=2)
    print(f"Saved {cfg_path}")

    algos = [
        "PINPF",
        "Incompressible Flow",
        "Local Gaussian exact flow",
        "Mean Gaussian exact flow",
        "SVGD",
        "Annealed MCMC",
    ]
    metrics = {a: {"ed": [], "swd": [], "time": []} for a in algos}

    integrator = create_euler_adaptive(DELTA_L=args.d_lambda, verbose=False)

    for i in range(n_tasks):
        print(f"\nTask {i + 1}/{n_tasks}")
        prior_sub, meas_sub, z_task, gt_i = _make_task(config, i)

        def _metrics(x_pred):
            pred = x_pred.cpu()
            gt = gt_i.cpu()
            n_min = min(gt.shape[0], pred.shape[0])
            swd = sliced_wasserstein_distance(gt[:n_min], pred[:n_min]).item()
            ed = energy_distance(gt, pred).item()
            return ed, swd

        x0 = prior_sub.sample(args.n_particles).to(DEVICE)

        # ── PINPF ──
        ode_func = NeuralFlow(
            model=model,
            prior_model=prior_sub,
            meas_model=meas_sub,
            z=z_task,
            extra_features=extra_features,
        )

        def _run_pinpf():
            with torch.no_grad():
                x_out, _, _ = generate_flow_samples(integrator, ode_func, x0)
            return x_out[0]

        x_pinpf, t = _timed(_run_pinpf)
        ed, swd = _metrics(x_pinpf)
        metrics["PINPF"]["ed"].append(ed)
        metrics["PINPF"]["swd"].append(swd)
        metrics["PINPF"]["time"].append(t)
        print(f"  PINPF:   ED={ed:.4f}, SWD={swd:.4f}, t={t:.4f}s")

        # ── Incompressible Flow ──
        ode_inc = IncompressibleFlow(
            meas_model=meas_sub,
            prior_model=prior_sub,
            z=z_task,
        )
        x0_inc = prior_sub.sample(args.n_particles).to(DEVICE)

        def _run_inc():
            with torch.no_grad():
                traj = integrator(ode_inc, x0_inc).detach()
            return traj[-1][0]

        x_inc, t = _timed(_run_inc)
        ed, swd = _metrics(x_inc)
        metrics["Incompressible Flow"]["ed"].append(ed)
        metrics["Incompressible Flow"]["swd"].append(swd)
        metrics["Incompressible Flow"]["time"].append(t)
        print(f"  Incomp:  ED={ed:.4f}, SWD={swd:.4f}, t={t:.4f}s")

        # ── Local Gaussian Exact Flow ──
        flow_loc = LocalGaussianExactFlow(
            meas_model=meas_sub,
            prior_model=prior_sub,
            z=z_task,
        )
        x0_loc = prior_sub.sample(args.n_particles).to(DEVICE)

        def _run_loc():
            traj = integrator(flow_loc, x0_loc).detach()
            return traj[-1][0]

        x_loc, t = _timed(_run_loc)
        ed, swd = _metrics(x_loc)
        metrics["Local Gaussian exact flow"]["ed"].append(ed)
        metrics["Local Gaussian exact flow"]["swd"].append(swd)
        metrics["Local Gaussian exact flow"]["time"].append(t)
        print(f"  LocGaus: ED={ed:.4f}, SWD={swd:.4f}, t={t:.4f}s")

        # ── Mean Gaussian Exact Flow ──
        flow_mg = MeanGaussianExactFlow(
            meas_model=meas_sub,
            prior_model=prior_sub,
            z=z_task,
        )
        x0_mg = prior_sub.sample(args.n_particles).to(DEVICE)

        def _run_mg():
            traj = integrator(flow_mg, x0_mg).detach()
            return traj[-1][0]

        x_mg, t = _timed(_run_mg)
        ed, swd = _metrics(x_mg)
        metrics["Mean Gaussian exact flow"]["ed"].append(ed)
        metrics["Mean Gaussian exact flow"]["swd"].append(swd)
        metrics["Mean Gaussian exact flow"]["time"].append(t)
        print(f"  MnGaus:  ED={ed:.4f}, SWD={swd:.4f}, t={t:.4f}s")

        # ── SVGD ──
        x0_svgd = prior_sub.sample(args.n_particles_svgd).squeeze(0)

        def _run_svgd():
            x_out, _ = run_svgd(
                prior=lambda x: prior_sub.log_prob(x),
                meas=lambda x, z: meas_sub.log_prob(x, z),
                z=z_task,
                x0=x0_svgd,
                n_iter=args.svgd_iter,
                lr=args.svgd_lr,
            )
            return x_out

        x_svgd, t = _timed(_run_svgd)
        ed, swd = _metrics(x_svgd)
        metrics["SVGD"]["ed"].append(ed)
        metrics["SVGD"]["swd"].append(swd)
        metrics["SVGD"]["time"].append(t)
        print(f"  SVGD:    ED={ed:.4f}, SWD={swd:.4f}, t={t:.4f}s")

        # ── Annealed MCMC ──
        x0_ann = prior_sub.sample(args.n_particles_annealed).squeeze(0)

        def _run_ann():
            return run_annealed_mcmc(
                prior=lambda x: prior_sub.log_prob(x),
                meas=lambda x, z: meas_sub.log_prob(x, z),
                z=z_task,
                x0=x0_ann,
                n_steps=args.annealed_steps,
                n_mcmc_per_step=args.annealed_mcmc_per_step,
                step_size=args.annealed_step_size,
                use_nuts=True,
            )

        x_ann, t = _timed(_run_ann)
        ed, swd = _metrics(x_ann)
        metrics["Annealed MCMC"]["ed"].append(ed)
        metrics["Annealed MCMC"]["swd"].append(swd)
        metrics["Annealed MCMC"]["time"].append(t)
        print(f"  AMCMC:   ED={ed:.4f}, SWD={swd:.4f}, t={t:.4f}s")

    # Convert to results dict format expected by save_summary
    results = {}
    for name in algos:
        m = metrics[name]
        results[name] = {
            "ed": m["ed"],
            "swd": m["swd"],
            "time": sum(m["time"]),
        }

    # ── Summary ──
    print("\n" + "=" * 80)
    print(f"{'Method':<35} {'ED':>18} {'SWD':>18} {'Time/task':>10}")
    print("-" * 80)
    for name, res in results.items():
        ed_arr = np.array(res["ed"])
        swd_arr = np.array(res["swd"])
        t = res["time"]
        print(
            f"{name:<35} "
            f"{ed_arr.mean():>6.4f}+/-{ed_arr.std():<7.4f} "
            f"{swd_arr.mean():>6.4f}+/-{swd_arr.std():<7.4f} "
            f"{t / n_tasks:>8.4f}s"
        )

    save_summary(results, n_tasks, out_dir)


if __name__ == "__main__":
    main()
