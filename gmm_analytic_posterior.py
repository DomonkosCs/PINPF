import torch
import torch.distributions as dist


class AnalyticGMMPosterior:
    def __init__(self, prior_model, meas_model, z_obs):
        """
        Computes the analytic posterior for a Gaussian Prior and GMM Likelihood.

        Args:
            prior_model: DiagonalGaussianPrior
            meas_model: GaussianMixtureMeasurementModel
            z_obs: Tensor of shape (Batch, K, Dim) or (Batch, Dim) if K=1
                   These are the centers of the likelihood components.
        """
        self.prior_model = prior_model
        self.meas_model = meas_model
        self.z_obs = z_obs

        self.mu_0 = prior_model.mu_prior  # (B, D)
        self.var_0 = prior_model.var_prior  # (B, D)

        if z_obs.dim() == 2:
            # (B, D) -> (B, 1, D)
            self.z_k = z_obs.unsqueeze(1)
        else:
            # (B, K, D)
            self.z_k = z_obs

        self.K = self.z_k.shape[1]
        self.B = self.z_k.shape[0]
        self.D = self.z_k.shape[2]

        self.w_k = meas_model.weights  # (K,) or (B, K)
        if self.w_k.dim() == 1:
            self.w_k = self.w_k.unsqueeze(0).expand(self.B, -1)  # (B, K)

        self.sigmas_k = meas_model.sigmas
        if self.sigmas_k.dim() == 2:
            # (K, D) -> (B, K, D)
            self.sigmas_k = self.sigmas_k.unsqueeze(0).expand(self.B, -1, -1)

        self.var_k = self.sigmas_k**2  # (B, K, D)

        self.compute_posterior_params()

    def compute_posterior_params(self):
        # Expand prior params to (B, K, D)
        mu_0_exp = self.mu_0.unsqueeze(1).expand(-1, self.K, -1)
        var_0_exp = self.var_0.unsqueeze(1).expand(-1, self.K, -1)

        self.var_post = (var_0_exp * self.var_k) / (var_0_exp + self.var_k)  # (B, K, D)
        self.sigma_post = torch.sqrt(self.var_post)

        term1 = mu_0_exp / var_0_exp
        term2 = self.z_k / self.var_k
        self.mu_post = self.var_post * (term1 + term2)  # (B, K, D)

        var_scale = var_0_exp + self.var_k  # (B, K, D)

        diff = self.z_k - mu_0_exp
        log_gauss = -0.5 * torch.sum((diff**2) / var_scale, dim=-1)  # (B, K)
        log_gauss = log_gauss - 0.5 * torch.sum(torch.log(var_scale), dim=-1)

        log_w_prime = torch.log(self.w_k + 1e-10) + log_gauss  # (B, K)

        log_norm = torch.logsumexp(log_w_prime, dim=-1, keepdim=True)
        self.log_w_post = log_w_prime - log_norm
        self.w_post = torch.exp(self.log_w_post)  # (B, K)

    def sample(self, n_samples):
        cat_dist = dist.Categorical(probs=self.w_post)
        comp_indices = cat_dist.sample((n_samples,)).T  # (B, n_samples)

        batch_indices = (
            torch.arange(self.B, device=self.mu_post.device)
            .unsqueeze(1)
            .expand(-1, n_samples)
        )

        mu_selected = self.mu_post[batch_indices, comp_indices, :]  # (B, n_samples, D)
        sigma_selected = self.sigma_post[
            batch_indices, comp_indices, :
        ]  # (B, n_samples, D)

        eps = torch.randn_like(mu_selected)
        samples = mu_selected + sigma_selected * eps

        return samples
