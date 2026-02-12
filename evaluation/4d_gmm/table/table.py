import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from utils import energy_distance, sliced_wasserstein_distance


def summarize_4d_results(results_dir, n_batch: int = 100):
    timings_path = os.path.join(results_dir, "timings.pt")
    if not os.path.exists(timings_path):
        raise FileNotFoundError(f"timings.pt not found in {results_dir}.")

    timings = torch.load(timings_path)

    ed_pinpf = []
    ed_incomp = []
    ed_nsf = []
    ed_svgd = []
    ed_annealed = []
    swd_pinpf = []
    swd_incomp = []
    swd_nsf = []
    swd_svgd = []
    swd_annealed = []

    for batch in range(n_batch):
        analytic_path = os.path.join(results_dir, f"batch_{batch:03d}_analytic.pt")
        pinpf_path = os.path.join(results_dir, f"batch_{batch:03d}_pinpf.pt")
        incomp_path = os.path.join(results_dir, f"batch_{batch:03d}_incomp.pt")
        nsf_path = os.path.join(results_dir, f"batch_{batch:03d}_nsf.pt")
        svgd_path = os.path.join(results_dir, f"batch_{batch:03d}_svgd.pt")
        annealed_path = os.path.join(results_dir, f"batch_{batch:03d}_annealed.pt")

        if not (
            os.path.exists(analytic_path)
            and os.path.exists(pinpf_path)
            and os.path.exists(svgd_path)
            and os.path.exists(annealed_path)
        ):
            # Stop at first missing batch; assumes results are contiguous from 0.
            n_batch = batch
            break

        x_ref = torch.load(analytic_path)  # [N_ref, D]
        x_pinpf = torch.load(pinpf_path)  # [N_pinpf, D]
        x_svgd = torch.load(svgd_path)  # [N_svgd, D]
        x_annealed = torch.load(annealed_path)  # [N_ann, D]

        # Load Incomp / NSF if available (or handle missing)
        # Assuming simulate runs generated them if requested.
        # But if user didn't run them, this might fail or we should conditionally add.
        # For this request, assuming we want to display them if they exist.
        has_incomp = os.path.exists(incomp_path)
        has_nsf = os.path.exists(nsf_path)

        if has_incomp:
            x_incomp = torch.load(incomp_path)
        else:
            x_incomp = None

        if has_nsf:
            x_nsf = torch.load(nsf_path)
        else:
            x_nsf = None

        # Match sample counts where appropriate for SWD
        n_pinpf = min(x_ref.shape[0], x_pinpf.shape[0])
        n_svgd = min(x_ref.shape[0], x_svgd.shape[0])
        n_ann = min(x_ref.shape[0], x_annealed.shape[0])

        ed_pinpf.append(energy_distance(x_ref, x_pinpf).item())
        ed_svgd.append(energy_distance(x_ref, x_svgd).item())
        ed_annealed.append(energy_distance(x_ref, x_annealed).item())

        if has_incomp:
            ed_incomp.append(energy_distance(x_ref, x_incomp).item())
            n_incomp = min(x_ref.shape[0], x_incomp.shape[0])
            swd_incomp.append(
                sliced_wasserstein_distance(
                    x_ref[:n_incomp], x_incomp[:n_incomp]
                ).item()
            )

        if has_nsf:
            ed_nsf.append(energy_distance(x_ref, x_nsf).item())
            n_nsf = min(x_ref.shape[0], x_nsf.shape[0])
            swd_nsf.append(
                sliced_wasserstein_distance(x_ref[:n_nsf], x_nsf[:n_nsf]).item()
            )

        swd_pinpf.append(
            sliced_wasserstein_distance(x_ref[:n_pinpf], x_pinpf[:n_pinpf]).item()
        )
        swd_svgd.append(
            sliced_wasserstein_distance(x_ref[:n_svgd], x_svgd[:n_svgd]).item()
        )
        swd_annealed.append(
            sliced_wasserstein_distance(x_ref[:n_ann], x_annealed[:n_ann]).item()
        )

    ed_pinpf = torch.tensor(ed_pinpf)
    ed_svgd = torch.tensor(ed_svgd)
    ed_annealed = torch.tensor(ed_annealed)
    swd_pinpf = torch.tensor(swd_pinpf)
    swd_svgd = torch.tensor(swd_svgd)
    swd_annealed = torch.tensor(swd_annealed)

    # Optional tensors
    ed_incomp = torch.tensor(ed_incomp) if ed_incomp else torch.tensor([])
    swd_incomp = torch.tensor(swd_incomp) if swd_incomp else torch.tensor([])
    ed_nsf = torch.tensor(ed_nsf) if ed_nsf else torch.tensor([])
    swd_nsf = torch.tensor(swd_nsf) if swd_nsf else torch.tensor([])

    time_pinpf = timings["time_pinpf"][: len(ed_pinpf)]
    time_svgd = timings["time_svgd"][: len(ed_svgd)]
    time_annealed = timings["time_annealed"][: len(ed_annealed)]

    time_incomp = (
        timings["time_incomp"][: len(ed_incomp)]
        if "time_incomp" in timings and len(ed_incomp) > 0
        else torch.tensor([])
    )
    time_nsf = (
        timings["time_nsf"][: len(ed_nsf)]
        if "time_nsf" in timings and len(ed_nsf) > 0
        else torch.tensor([])
    )

    def compute_mean(x: torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        return x.mean().item()

    metrics = {
        "PINPF": {
            "ED": compute_mean(ed_pinpf),
            "SWD": compute_mean(swd_pinpf),
            "Time": compute_mean(time_pinpf),
        },
        "Incompressible": {
            "ED": compute_mean(ed_incomp),
            "SWD": compute_mean(swd_incomp),
            "Time": compute_mean(time_incomp),
        },
        "NSF": {
            "ED": compute_mean(ed_nsf),
            "SWD": compute_mean(swd_nsf),
            "Time": compute_mean(time_nsf),
        },
        "SVGD": {
            "ED": compute_mean(ed_svgd),
            "SWD": compute_mean(swd_svgd),
            "Time": compute_mean(time_svgd),
        },
        "Annealed MCMC": {
            "ED": compute_mean(ed_annealed),
            "SWD": compute_mean(swd_annealed),
            "Time": compute_mean(time_annealed),
        },
    }

    # Print table
    print("\nPerformance summary over", len(ed_pinpf), "batches")
    header = f"{'Method':<15} | {'ED (mean)':<25} | {'SWD (mean)':<25} | {'Time [s] (mean)':<25}"
    print(header)
    print("-" * len(header))

    for method, vals in metrics.items():
        ed_mean = vals["ED"]
        swd_mean = vals["SWD"]
        t_mean = vals["Time"]
        print(
            f"{method:<15} | " f"{ed_mean:.4f} | " f"{swd_mean:.4f} | " f"{t_mean:.4f}"
        )

    print("\n" + r"\begin{table}[t]")
    print(r"    \centering")
    print(
        f"    \\caption{{Performance summary over {len(ed_pinpf)} samples from the test dataset (mean values).}}"
    )
    print(r"    \label{tab:performance_summary}")
    print(r"        \begin{tabular}{@{}lccc@{}}")
    print(r"            \toprule")
    print(r"            Method & ED & SWD & GPU Time [s] \\")
    print(r"            \midrule")

    m_pinpf = metrics["PINPF"]
    print(
        f"            Ours (PINPF)          & {m_pinpf['ED']:.4f} & {m_pinpf['SWD']:.4f} & {m_pinpf['Time']:.4f} \\\\"
    )

    m_incomp = metrics["Incompressible"]
    if not torch.isnan(torch.tensor(m_incomp["ED"])):
        print(
            f"            Incompressible & {m_incomp['ED']:.4f} & {m_incomp['SWD']:.4f} & {m_incomp['Time']:.4f} \\\\"
        )

    m_nsf = metrics["NSF"]
    if not torch.isnan(torch.tensor(m_nsf["ED"])):
        print(
            f"            NSF             & {m_nsf['ED']:.4f} & {m_nsf['SWD']:.4f} & {m_nsf['Time']:.4f} \\\\"
        )

    m_svgd = metrics["SVGD"]
    print(
        f"            SVGD          & {m_svgd['ED']:.4f} & {m_svgd['SWD']:.4f} & {m_svgd['Time']:.4f} \\\\"
    )

    m_ann = metrics["Annealed MCMC"]
    print(
        f"            Annealed MCMC & {m_ann['ED']:.4f} & {m_ann['SWD']:.4f} & {m_ann['Time']:.4f} \\\\"
    )

    print(r"            \bottomrule")
    print(r"        \end{tabular}%")
    print(r"\end{table}")


if __name__ == "__main__":
    summarize_4d_results(results_dir="evaluation/4d_gmm/table/result_latest")
