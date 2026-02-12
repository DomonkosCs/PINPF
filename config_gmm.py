import torch
import os
import pyro
from meas_models import GaussianMixtureMeasurementModel
from prior_models import DiagonalGaussianPrior


class ConfigGMM:
    def __init__(
        self,
        device,
        data_path,
        mode,
        max_samples=None,
    ):
        full_data = torch.load(data_path, map_location="cpu")
        data = full_data[mode]

        if max_samples is not None and max_samples < len(data["z_obs"]):
            print(
                f"Limiting dataset to {max_samples} samples (original: {len(data['z_obs'])})"
            )
            for k in data:
                data[k] = data[k][:max_samples]

        self.state_dim = data["mu_prior"].shape[-1]

        # Flatten measurement dimension for the network input
        self.meas_dim = data["z_obs"][0].numel()

        self.mu_prior = data["mu_prior"].to(device)
        self.var_prior = data["var_prior"].to(device)
        self.zs = data["z_obs"].to(device)

        # GMM parameters
        self.gmm_weights = data["gmm_weights"].to(device)
        self.gmm_means = data["gmm_means"].to(device)
        self.gmm_sigmas = data["gmm_sigmas"].to(device)

        self.n_batch = self.zs.shape[0]
        self.device = device

        self.meas_model_const = GaussianMixtureMeasurementModel
        self.prior_model_const = DiagonalGaussianPrior

        self.prior_model = self.prior_model_const(
            mu_prior=self.mu_prior, var_prior=self.var_prior
        )
        self.meas_model = self.meas_model_const(
            weights=self.gmm_weights, sigmas=self.gmm_sigmas
        )

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
                weights=self.gmm_weights[idx].to(device),
                sigmas=self.gmm_sigmas[idx].to(device),
            )
            z_mb = self.zs[idx].to(device)
            if hasattr(self, "true_samples"):
                true_samples_mb = self.true_samples[idx].to(device)
            else:
                true_samples_mb = None

            yield prior_sub, meas_sub, z_mb, true_samples_mb

    def get_dataset_size(self):
        return self.n_batch


def generate_dataset(
    n_train,
    n_val,
    n_test,
    save_path,
    dim,
    n_modes,
    seed=42,
):
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    gen_device = "cpu"
    total_samples = n_train + n_val + n_test

    # equal weights for all modes
    weights = torch.ones(total_samples, n_modes, device=gen_device) / n_modes

    # Generate random means from Uniform(-3, 3)
    # Shape: (total_samples, n_modes, dim)
    means = torch.rand(total_samples, n_modes, dim, device=gen_device) * 6.0 - 3.0

    # Generate random sigmas from Uniform(0.3, 0.7)
    # Shape: (total_samples, n_modes, dim)
    sigmas = torch.rand(total_samples, n_modes, dim, device=gen_device) * 0.4 + 0.3

    z_obs = means

    # Centered at 0, varying diagonal sigma
    mu_prior = torch.zeros(total_samples, dim, device=gen_device)
    # Random variance between 1.0 and 10.0
    var_prior = torch.rand(total_samples, dim, device=gen_device) * 9.0 + 1.0

    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, total_samples)

    full_data = {
        "z_obs": z_obs,
        "mu_prior": mu_prior,
        "var_prior": var_prior,
        "gmm_weights": weights,
        "gmm_means": means,
        "gmm_sigmas": sigmas,
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4, help="State dimension")
    parser.add_argument("--n-modes", type=int, default=3, help="Likelihood GMM modes")
    parser.add_argument("--n-train", type=int, default=2000, help="Training samples")
    parser.add_argument("--n-val", type=int, default=200, help="Validation samples")
    parser.add_argument("--n-test", type=int, default=200, help="Test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Save path (default: data/dataset_gmm_{dim}d.pt)",
    )
    args = parser.parse_args()

    save_path = args.save_path or f"data/dataset_gmm_{args.dim}d.pt"

    generate_dataset(
        save_path=save_path,
        dim=args.dim,
        n_modes=args.n_modes,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
    )
