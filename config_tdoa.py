import torch
import os
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from meas_models import TDOAMeasurementModel
from prior_models import DiagonalGaussianPrior


class ConfigTDOA:
    def __init__(
        self, device, mode="train", data_path="data/dataset_tdoa.pt", max_samples=None
    ):
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. Run generate_data.py first."
            )

        print(f"Loading {mode} data from {data_path}...")
        full_data = torch.load(data_path, map_location="cpu")
        data = full_data[mode]

        if max_samples is not None and max_samples < len(data["z_obs"]):
            print(
                f"Limiting dataset to {max_samples} samples (original: {len(data['z_obs'])})"
            )
            for k in data:
                data[k] = data[k][:max_samples]

        self.state_dim = 2
        self.meas_dim = 1

        self.mu_prior = data["mu_prior"].to(device)
        self.var_prior = data["var_prior"].to(device)
        self.sigma_meas = data["sigma_meas"].to(device)
        self.zs = data["z_obs"].to(device)
        self.x_true = data["x_true"].to(device)
        self.true_samples = data["posterior_samples"].to(device)

        self.n_batch = self.zs.shape[0]
        self.device = device

        self.meas_model_const = TDOAMeasurementModel
        self.prior_model_const = DiagonalGaussianPrior

        self.prior_model = self.prior_model_const(
            mu_prior=self.mu_prior, var_prior=self.var_prior
        )
        self.meas_model = self.meas_model_const(sigma=self.sigma_meas)

    def iter_minibatches(
        self, mini_batch_size: int, device: torch.device, shuffle: bool = True
    ):
        n_tasks = self.n_batch
        mini_batch_size = max(1, min(mini_batch_size, n_tasks))
        perm = torch.randperm(n_tasks) if shuffle else torch.arange(n_tasks)

        for start in range(0, n_tasks, mini_batch_size):
            end = min(start + mini_batch_size, n_tasks)
            idx = perm[start:end]

            prior_sub = self.prior_model_const(
                mu_prior=self.prior_model.mu_prior[idx].to(device),
                var_prior=self.prior_model.var_prior[idx].to(device),
            )
            meas_sub = self.meas_model_const(
                sigma=self.meas_model.sigma[idx].to(device)
            )
            z_mb = self.zs[idx].to(device)
            true_samples_mb = self.true_samples[idx].to(device)

            yield prior_sub, meas_sub, z_mb, true_samples_mb

    def get_dataset_size(self):
        return self.n_batch


# --- Pyro Model Definition ---
def vectorized_tdoa_model(mu_prior, std_prior, sigma_meas, z_obs=None):
    """
    Defines a batch of independent TDOA inference problems.
    inputs are shaped [Batch_Size, ...]
    """
    batch_size = mu_prior.shape[0]

    meas_model_helper = TDOAMeasurementModel(
        sigma=torch.tensor(1.0, device=mu_prior.device)
    )

    with pyro.plate("data", batch_size):
        x = pyro.sample("x", dist.Normal(mu_prior, std_prior).to_event(1))

        pred_z = meas_model_helper.eval(x)

        pyro.sample("obs", dist.Normal(pred_z, sigma_meas), obs=z_obs)


def run_pyro_batch(
    z_obs_b,
    sigma_meas_b,
    mu_prior_b,
    var_prior_b,
    num_samples=5000,
    warmup_steps=1000,
    device="cpu",
):
    """
    Runs NUTS sampler on a batch of tasks.
    """
    # Convert variance to std for Pyro
    std_prior_b = torch.sqrt(var_prior_b)

    # Move to device
    z_obs_b = z_obs_b.to(device)
    sigma_meas_b = sigma_meas_b.to(device)
    mu_prior_b = mu_prior_b.to(device)
    std_prior_b = std_prior_b.to(device)

    kernel = NUTS(vectorized_tdoa_model, jit_compile=False)

    mcmc = MCMC(
        kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=1,
    )

    # Run MCMC
    mcmc.run(mu_prior_b, std_prior_b, sigma_meas_b, z_obs_b)

    # Get samples
    # Shape returned by Pyro: [num_samples, batch_size, state_dim]
    samples = mcmc.get_samples()["x"]

    # Transpose to [batch_size, num_samples, state_dim] for storage
    return samples.permute(1, 0, 2).cpu()


def generate_dataset(
    device,
    n_train=1000,
    n_val=100,
    n_test=100,
    seed=42,
    save_path="data/dataset_tdoa.pt",
):
    print(f"Generating dataset with Seed: {seed}")
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    total_samples = n_train + n_val + n_test
    state_dim = 2

    # --- 1. Generate Problem Parameters ---

    # A. True States (Target) ~ centered at 4.0
    target_mean = torch.tensor([4.0, 4.0], device=device)
    target_std = torch.tensor([1.5, 1.5], device=device)
    x_true = target_mean + target_std * torch.randn(
        total_samples, state_dim, device=device
    )

    # B. Measurement Noise (Sigma)
    sigma_meas = torch.rand(total_samples, device=device) * 0.5 + 0.4

    # C. Observations (z)
    meas_model = TDOAMeasurementModel(sigma=sigma_meas)
    z_clean = meas_model.eval(x_true)
    noise = torch.randn(total_samples, device=device) * sigma_meas
    z_obs = z_clean + noise

    # D. Priors (Badly initialized)
    # Mean offset from true
    mu_prior_std = torch.tensor([4.0, 5.0], device=device)
    mu_offset = mu_prior_std * torch.randn(total_samples, state_dim, device=device)
    mu_prior = x_true + mu_offset

    # Variance random
    var_prior_mean = torch.tensor([5.0, 5.0], device=device)
    var_prior_std = torch.tensor([1.0, 1.0], device=device)
    var_prior = var_prior_mean + var_prior_std * torch.randn(
        total_samples, state_dim, device=device
    )
    var_prior = torch.abs(var_prior)

    # --- 2. Generate Ground Truth via Pyro MCMC ---
    print(f"Starting Pyro NUTS Sampling for {total_samples} tasks...")

    # Run Pyro on full dataset
    posterior_samples = run_pyro_batch(
        z_obs, sigma_meas, mu_prior, var_prior, device=device
    )

    # --- 3. Split and Save ---
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, total_samples)

    full_data = {
        "x_true": x_true,
        "z_obs": z_obs,
        "sigma_meas": sigma_meas,
        "mu_prior": mu_prior,
        "var_prior": var_prior,
        "posterior_samples": posterior_samples,
    }

    dataset = {
        "train": {k: v[idx_train] for k, v in full_data.items()},
        "val": {k: v[idx_val] for k, v in full_data.items()},
        "test": {k: v[idx_test] for k, v in full_data.items()},
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dataset, save_path)
    print(f"Dataset saved to {save_path}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")


if __name__ == "__main__":
    generate_dataset(device="cuda" if torch.cuda.is_available() else "cpu")
