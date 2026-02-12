import torch
import numpy as np
import pandas as pd
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dhflows import (
    IncompressibleFlow,
    LocalGaussianExactFlow,
    MeanGaussianExactFlow,
    NeuralFlow,
)
from models import NeuralFlowModel
from solvers import create_euler_adaptive
from utils import energy_distance, sliced_wasserstein_distance
from config_tdoa import ConfigTDOA
from svgd import run_svgd
from amcmc import run_annealed_mcmc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


def load_neural_flow_model(model_path):
    model = NeuralFlowModel(layers=6, neurons_per_layer=64)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def generate_flow_samples(integrator, ode_func, x0):
    trajectory = integrator(ode_func, x0).detach()
    final_particles = trajectory[-1].clone().detach()
    return final_particles


def evaluate_tdoa_comparison():
    print("Loading dataset...")
    data_path = "data/dataset_tdoa.pt"

    config_test = ConfigTDOA(
        device=device,
        mode="test",
        max_samples=100,
        data_path=data_path,
    )

    N_test = config_test.get_dataset_size()
    print(f"Number of test cases: {N_test}")

    # Load Neural Flow Model
    neural_model_path = "trainings/tdoa/ckp/model_epoch_best.pth"
    neural_model = load_neural_flow_model(neural_model_path)

    # Metrics storage
    results = {
        "Neural Flow": {"ED": [], "SWD": [], "Time": []},
        "Incompressible Flow": {"ED": [], "SWD": [], "Time": []},
        "Local Gaussian Exact Flow": {"ED": [], "SWD": [], "Time": []},
        "Mean Gaussian Exact Flow": {"ED": [], "SWD": [], "Time": []},
        "SVGD": {"ED": [], "SWD": [], "Time": []},
        "Annealed MCMC": {"ED": [], "SWD": [], "Time": []},
    }

    # Integrator settings
    integrator = create_euler_adaptive(DELTA_L=1.0)

    # Loop over test cases
    batch_size = 1

    for i, (sub_prior, sub_meas, z_mb, gt_samples_batch) in enumerate(
        config_test.iter_minibatches(batch_size, device, shuffle=False)
    ):
        print(f"Processing batch {i+1}...")

        # Ensure z_obs is [B, 1] for broadcasting
        if z_mb.dim() == 1:
            z_mb = z_mb.unsqueeze(-1)

        current_batch_size = z_mb.shape[0]
        n_particles = 1000  # gt_samples_batch.shape[1]

        prior_model = sub_prior
        meas_model = sub_meas

        x_gt_grid = gt_samples_batch

        if x_gt_grid.shape[1] > n_particles:
            x_gt_grid_subsampled = x_gt_grid[:, :n_particles, :]

        x0 = prior_model.sample(n_particles)

        # Define flows
        flows = {}
        if neural_model is not None:
            flows["Neural Flow"] = NeuralFlow(
                neural_model, prior_model, meas_model, z_mb
            )

        flows["Incompressible Flow"] = IncompressibleFlow(meas_model, prior_model, z_mb)
        flows["Local Gaussian Exact Flow"] = LocalGaussianExactFlow(
            meas_model, prior_model, z_mb
        )
        flows["Mean Gaussian Exact Flow"] = MeanGaussianExactFlow(
            meas_model, prior_model, z_mb
        )

        # Evaluate each flow
        for name, flow in flows.items():
            try:
                # Run flow
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()

                final_particles = generate_flow_samples(integrator, flow, x0)

                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()
                elapsed = t1 - t0

                # Compute metrics for each sample in the batch
                x_pred = final_particles[0].unsqueeze(0)  # [1, N, D]
                x_gt_subsampled = x_gt_grid_subsampled[0].unsqueeze(0)  # [1, N, D]
                x_gt = x_gt_grid[0].unsqueeze(0)  # [1, N, D]

                ed = energy_distance(x_gt, x_pred).item()
                swd = sliced_wasserstein_distance(x_gt_subsampled, x_pred).item()

                results[name]["ED"].append(ed)
                results[name]["SWD"].append(swd)
                results[name]["Time"].append(
                    elapsed
                )  # Time is per batch (which is 1 here)
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                # Append NaNs or handle error
                for j in range(current_batch_size):
                    results[name]["ED"].append(np.nan)
                    results[name]["SWD"].append(np.nan)
                    results[name]["Time"].append(np.nan)

        # SVGD
        try:
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            # SVGD doesn't support batching in this implementation, but batch_size is 1
            x0_single = x0[0]
            z_single = z_mb[0]

            x_svgd, _ = run_svgd(
                prior_model,
                meas_model,
                z_single,
                x0_single,
                n_iter=200,
                lr=0.2,
            )
            # x_svgd is [N, D]

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            elapsed = t1 - t0

            x_pred = x_svgd.unsqueeze(0)  # [1, N, D]
            x_gt = x_gt_grid[0].unsqueeze(0)
            x_gt_subsampled = x_gt_grid_subsampled[0].unsqueeze(0)  # [1, N, D]

            ed = energy_distance(x_gt, x_pred).item()
            swd = sliced_wasserstein_distance(x_gt_subsampled, x_pred).item()

            results["SVGD"]["ED"].append(ed)
            results["SVGD"]["SWD"].append(swd)
            results["SVGD"]["Time"].append(elapsed)

        except Exception as e:
            print(f"Error evaluating SVGD: {e}")
            results["SVGD"]["ED"].append(np.nan)
            results["SVGD"]["SWD"].append(np.nan)
            results["SVGD"]["Time"].append(np.nan)

        # Annealed MCMC
        try:
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            x0_single = x0[0]
            z_single = z_mb[0]

            x_amcmc = run_annealed_mcmc(
                prior_model,
                meas_model,
                z_single,
                x0_single,
                n_steps=10,
                n_mcmc_per_step=3,
                use_nuts=True,
            )
            # x_amcmc is [N, D]

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            elapsed = t1 - t0

            x_pred = x_amcmc.unsqueeze(0)
            x_gt = x_gt_grid[0].unsqueeze(0)
            x_gt_subsampled = x_gt_grid_subsampled[0].unsqueeze(0)  # [1, N, D]

            ed = energy_distance(x_gt, x_pred).item()
            swd = sliced_wasserstein_distance(x_gt_subsampled, x_pred).item()

            results["Annealed MCMC"]["ED"].append(ed)
            results["Annealed MCMC"]["SWD"].append(swd)
            results["Annealed MCMC"]["Time"].append(elapsed)

        except Exception as e:
            print(f"Error evaluating Annealed MCMC: {e}")
            results["Annealed MCMC"]["ED"].append(np.nan)
            results["Annealed MCMC"]["SWD"].append(np.nan)
            results["Annealed MCMC"]["Time"].append(np.nan)

    # Save full results
    full_results_data = []
    for name, metrics in results.items():
        n_samples = len(metrics["ED"])
        for k in range(n_samples):
            full_results_data.append(
                {
                    "Method": name,
                    "Sample_Index": k,
                    "ED": metrics["ED"][k],
                    "SWD": metrics["SWD"][k],
                    "Time": metrics["Time"][k],
                }
            )

    # Save full results dictionary to pt file
    pt_output_path = os.path.join(
        os.path.dirname(__file__), "tdoa_comparison_results.pt"
    )
    torch.save(results, pt_output_path)
    print(f"Full results saved to {pt_output_path}")

    # Create DataFrame
    df_data = []
    for name, metrics in results.items():
        if not metrics["ED"]:
            continue

        ed_mean = np.nanmean(metrics["ED"])
        ed_std = np.nanstd(metrics["ED"])
        swd_mean = np.nanmean(metrics["SWD"])
        swd_std = np.nanstd(metrics["SWD"])
        time_mean = np.nanmean(metrics["Time"])
        time_std = np.nanstd(metrics["Time"])

        df_data.append(
            {
                "Method": name,
                "ED Mean": ed_mean,
                "ED Std": ed_std,
                "SWD Mean": swd_mean,
                "SWD Std": swd_std,
                "Time Mean": time_mean,
                "Time Std": time_std,
            }
        )

    df = pd.DataFrame(df_data)
    print("\nEvaluation Results (2D TDOA):")
    print(df.to_string(index=False))

    # Print LaTeX Table
    print("\n" + r"\begin{table}[t]")
    print(r"    \centering")
    print(
        f"    \\caption{{Performance summary over {N_test} samples from the test dataset (mean values).}}"
    )
    print(r"    \label{tab:performance_summary}")
    print(
        r"    % \resizebox{\columnwidth}{!}{...} fits the table to the exact width of the column"
    )
    print(r"        \begin{tabular}{@{}lccc@{}}")
    print(r"            \toprule")
    print(r"            Method & ED & SWD & Time [s] \\")
    print(r"            \midrule")

    for index, row in df.iterrows():
        method = row["Method"]
        ed_mean = row["ED Mean"]
        swd_mean = row["SWD Mean"]
        time_mean = row["Time Mean"]
        print(
            f"            {method:<25} & {ed_mean:.4f} & {swd_mean:.4f} & {time_mean:.4f} \\\\"
        )

    print(r"            \bottomrule")
    print(r"        \end{tabular}%")
    print(r"\end{table}")


if __name__ == "__main__":
    evaluate_tdoa_comparison()
