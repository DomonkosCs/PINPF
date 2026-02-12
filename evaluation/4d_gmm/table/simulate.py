import os
import sys
import time
import torch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from config_gmm import ConfigGMM

from models import NeuralFlowModel, load_neural_flow_model
from dhflows import NeuralFlow, IncompressibleFlow
from solvers import create_euler_adaptive, create_euler

from nsf_gmm import build_conditional_nsf, make_conditioning
from svgd import run_svgd
from amcmc import run_annealed_mcmc
from utils import generate_flow_samples
from gmm_analytic_posterior import AnalyticGMMPosterior


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Auto-detect device


def eval_4d_sampling(
    data_path: str,
    model_path: str,
    nsf_path: str = None,
    n_total_samples: int = 100,
    algos=None,
    n_particles_flow: int = 1500,
    n_particles_nsf: int = 1500,
    n_particles_incomp: int = 1500,
    n_particles_svgd: int = 1500,
    n_particles_annealed: int = 1500,
    n_svgd_iter: int = 500,
    svgd_lr: float = 0.2,
    n_anneal_steps: int = 10,
    n_mcmc_per_step: int = 5,
    annealed_step_size: float = 0.1,
    use_nuts: bool = True,
    out_dir: str = "paper/4d_gmm/results_4d_sampling",
    omit_grad: bool = False,
):
    if algos is None:
        algos = ["pinpf", "svgd", "amcmc", "incomp", "nsf"]

    torch.manual_seed(42)
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset config
    config = ConfigGMM(
        device=DEVICE,
        mode="test",
        data_path=data_path,
        max_samples=n_total_samples,
    )

    if "pinpf" in algos:
        model = NeuralFlowModel(
            activation=nn.SiLU(),
            layers=6,
            neurons_per_layer=64,
            state_dim=config.state_dim,
            meas_dim=config.meas_dim,
            omit_grad_features=omit_grad,
        )
        model = load_neural_flow_model(model, model_path, device=DEVICE)
        model.eval()

    nsf_model = None
    if "nsf" in algos:
        if nsf_path is None:
            raise ValueError("nsf_path must be provided if 'nsf' is in algos")
        print(f"Loading NSF model from {nsf_path}...")

        state_dim = config.state_dim
        K = config.gmm_weights.shape[1]
        context_dim = (K * state_dim) + state_dim + state_dim + K + (K * state_dim)

        checkpoint = torch.load(nsf_path, map_location=DEVICE, weights_only=False)

        nsf_model = build_conditional_nsf(
            state_dim=state_dim,
            context_dim=context_dim,
            n_blocks=20,
            hidden_features=128,
            num_bins=32,
            dropout=0.1,
        ).to(DEVICE)

        if "flow_state" in checkpoint:
            nsf_model.load_state_dict(checkpoint["flow_state"])
        elif "model_state_dict" in checkpoint:
            nsf_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            nsf_model.load_state_dict(checkpoint)
        nsf_model.eval()

    # Analytic posterior for all batches
    analytic_posterior = AnalyticGMMPosterior(
        config.prior_model, config.meas_model, config.zs
    )

    # Integrator for PINPF / Incompressible
    integrator = create_euler_adaptive(DELTA_L=0.5, verbose=False)
    integrator_incomp = create_euler(dlamb=0.01)

    timings = []

    for i in range(config.n_batch):
        batch = i
        print(f"Sample {i+1}/{config.n_batch}")

        z_obs_batch = config.zs[i].unsqueeze(0)  # [1, Meas_Dim]

        prior_sub = config.prior_model_const(
            mu_prior=config.prior_model.mu_prior[i].unsqueeze(0),
            var_prior=config.prior_model.var_prior[i].unsqueeze(0),
        )
        meas_sub = config.meas_model_const(
            weights=config.gmm_weights[i].unsqueeze(0),
            sigmas=config.gmm_sigmas[i].unsqueeze(0),
        )

        if "pinpf" in algos:
            x0_flow = prior_sub.sample(n_particles_flow)  # [1, N, D]

            ode_func_single = NeuralFlow(
                model=model,
                prior_model=prior_sub,
                meas_model=meas_sub,
                z=z_obs_batch,
                omit_grads=omit_grad,
            )

            if torch.cuda.is_available() and DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                x_pinpf_out, _, _ = generate_flow_samples(
                    integrator, ode_func_single, x0_flow
                )
            if torch.cuda.is_available() and DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            x_pinpf_final = x_pinpf_out[0]  # [N, D]

            # Save PINPF samples
            torch.save(
                x_pinpf_final.cpu(),
                os.path.join(out_dir, f"batch_{batch:03d}_pinpf.pt"),
            )

        # ---------------------- Incompressible Flow ----------------------
        if "incomp" in algos:
            x0_incomp = prior_sub.sample(n_particles_incomp)  # [1, N, D]

            flow_incomp = IncompressibleFlow(
                meas_model=meas_sub,
                prior_model=prior_sub,
                z=z_obs_batch,  # [1, MeasDim]
            )

            if torch.cuda.is_available() and DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_inc0 = time.time()
            # Gradient tracking integration
            try:
                x_incomp_out = integrator_incomp(flow_incomp, x0_incomp)
                x_incomp_final = x_incomp_out[-1].detach().squeeze(0)  # [N, D]
            except Exception as e:
                print(f"Incompressible Flow failed for batch {batch}: {e}")
                x_incomp_final = x0_incomp.squeeze(0)  # Fallback = prior

            if torch.cuda.is_available() and DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_inc1 = time.time()

            torch.save(
                x_incomp_final.cpu(),
                os.path.join(out_dir, f"batch_{batch:03d}_incomp.pt"),
            )

        # ---------------------- Neural Spline Flow (NSF) ----------------------
        if "nsf" in algos and nsf_model is not None:
            # Prepare context
            # mu_prior: [1, D], var_prior: [1, D]
            # z_obs_batch: [1, K*D]
            # gmm_weights: [1, K]
            # gmm_sigmas: [1, K*D]

            # Extract raw tensors from config for this batch
            b_mu = config.prior_model.mu_prior[batch : batch + 1]
            b_var = config.prior_model.var_prior[batch : batch + 1]
            b_z = config.zs[batch : batch + 1]
            b_w = config.gmm_weights[batch : batch + 1]
            b_sig = config.gmm_sigmas[batch : batch + 1]

            context = make_conditioning(b_mu, b_var, b_z, b_w, b_sig)  # [1, C]

            if torch.cuda.is_available() and DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_nsf0 = time.time()

            with torch.no_grad():
                x_nsf_final = nsf_model.sample(n_particles_nsf, context=context)
                x_nsf_final = x_nsf_final.squeeze(0)  # [N, D]

            if torch.cuda.is_available() and DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_nsf1 = time.time()

            torch.save(
                x_nsf_final.cpu(),
                os.path.join(out_dir, f"batch_{batch:03d}_nsf.pt"),
            )

        # ---------------------- SVGD ----------------------
        if "svgd" in algos:
            x0_svgd = prior_sub.sample(n_particles_svgd).squeeze(0)  # [N, D]

            if torch.cuda.is_available() and DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.time()
            x_svgd, _ = run_svgd(
                prior=lambda x: prior_sub.log_prob(x),
                meas=lambda x, z: meas_sub.log_prob(x, z),
                z=z_obs_batch.squeeze(0),
                x0=x0_svgd,
                n_iter=n_svgd_iter,
                lr=svgd_lr,
            )
            if torch.cuda.is_available() and DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.time()

            torch.save(
                x_svgd.cpu(),
                os.path.join(out_dir, f"batch_{batch:03d}_svgd.pt"),
            )

        # ---------------------- Annealed MCMC ----------------------
        if "amcmc" in algos:
            x0_annealed = prior_sub.sample(n_particles_annealed).squeeze(0)  # [N, D]

            if torch.cuda.is_available() and DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t4 = time.time()
            x_annealed = run_annealed_mcmc(
                prior=lambda x: prior_sub.log_prob(x),
                meas=lambda x, z: meas_sub.log_prob(x, z),
                z=z_obs_batch.squeeze(0),
                x0=x0_annealed,
                n_steps=n_anneal_steps,
                n_mcmc_per_step=n_mcmc_per_step,
                step_size=annealed_step_size,
                use_nuts=use_nuts,
            )
            if torch.cuda.is_available() and DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t5 = time.time()

            torch.save(
                x_annealed.cpu(),
                os.path.join(out_dir, f"batch_{batch:03d}_annealed.pt"),
            )

        # ---------------------- Analytic samples (optional) ----------------------
        x_analytic = (
            analytic_posterior.sample(10000)[batch].squeeze(0).cpu()
        )  # [N_analytic, D]
        torch.save(
            x_analytic,
            os.path.join(out_dir, f"batch_{batch:03d}_analytic.pt"),
        )

        entry = {"batch": batch}
        if "pinpf" in algos:
            entry["time_pinpf"] = t1 - t0
        if "incomp" in algos:
            entry["time_incomp"] = t_inc1 - t_inc0
        if "nsf" in algos:
            entry["time_nsf"] = t_nsf1 - t_nsf0
        if "svgd" in algos:
            entry["time_svgd"] = t3 - t2
        if "amcmc" in algos:
            entry["time_annealed"] = t5 - t4
        timings.append(entry)

    # Save timings (only for selected algorithms)
    timings_tensor = {
        "batch": torch.tensor([t["batch"] for t in timings], dtype=torch.long),
    }
    if "pinpf" in algos:
        timings_tensor["time_pinpf"] = torch.tensor([t["time_pinpf"] for t in timings])
    if "incomp" in algos:
        timings_tensor["time_incomp"] = torch.tensor(
            [t["time_incomp"] for t in timings]
        )
    if "nsf" in algos:
        timings_tensor["time_nsf"] = torch.tensor([t["time_nsf"] for t in timings])
    if "svgd" in algos:
        timings_tensor["time_svgd"] = torch.tensor([t["time_svgd"] for t in timings])
    if "amcmc" in algos:
        timings_tensor["time_annealed"] = torch.tensor(
            [t["time_annealed"] for t in timings]
        )

    torch.save(timings_tensor, os.path.join(out_dir, "timings.pt"))


if __name__ == "__main__":
    data_path = "data/dataset_gmm_4d.pt"
    # PINPF model
    model_path = "trainings/gmm_4d/ckp/model_epoch_best.pth"
    # NSF model
    nsf_path = "trainings/neural_spline_flow/flow_best.pt"

    eval_4d_sampling(
        data_path=data_path,
        model_path=model_path,
        nsf_path=nsf_path,
        n_total_samples=100,
        algos=["pinpf", "svgd", "amcmc", "incomp", "nsf"],
        omit_grad=False,
    )
