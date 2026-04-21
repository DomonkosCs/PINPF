"""
Config classes for different problem types.
"""

import os
import torch
from problems.meas_models import (
    NonlinearGaussianMeasurementModel,
    TDOAMeasurementModel,
    GaussianMixtureMeasurementModel,
)
from problems.prior_models import DiagonalGaussianPrior


class ConfigNonlinear:
    def __init__(self, device, data_path, mode, max_samples=None):
        full_data = torch.load(data_path, map_location="cpu")
        data = full_data[mode]

        if max_samples is not None and max_samples < len(data["z_obs"]):
            print(
                f"Limiting dataset to {max_samples} samples (original: {len(data['z_obs'])})"
            )
            for k in data:
                data[k] = data[k][:max_samples]

        self.state_dim = data["mu_prior"].shape[-1]
        self.meas_dim = data["z_obs"].shape[-1]

        self.mu_prior = data["mu_prior"].to(device)
        self.var_prior = data["var_prior"].to(device)
        self.zs = data["z_obs"].to(device)
        self.alpha = data["alpha"].to(device)
        self.sigma = data["sigma"].to(device)

        self.n_batch = self.zs.shape[0]
        self.device = device

        self.meas_model_const = NonlinearGaussianMeasurementModel
        self.prior_model_const = DiagonalGaussianPrior

        self.prior_model = self.prior_model_const(
            mu_prior=self.mu_prior, var_prior=self.var_prior
        )
        self.meas_model = self.meas_model_const(alpha=self.alpha, sigma=self.sigma)

    def iter_minibatches(self, mini_batch_size, device, shuffle=True):
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
                alpha=self.alpha[idx].to(device),
                sigma=self.sigma[idx].to(device),
            )
            z_mb = self.zs[idx].to(device)
            yield prior_sub, meas_sub, z_mb, None

    def get_dataset_size(self):
        return self.n_batch


class ConfigTDOA:
    def __init__(
        self, device, mode="train", data_path="data/dataset_tdoa.pt", max_samples=None
    ):
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. Generate data first."
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
