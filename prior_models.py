import torch


class DiagonalGaussianPrior:
    """Diag Gaussian prior N(mu, diag(var))"""

    def __init__(self, mu_prior, var_prior):
        self.name = "isotropic_gaussian"
        self.mu_prior = mu_prior  # (n_batch, D)
        self.var_prior = var_prior  # (n_batch, D) treated as diagonal entries
        self.n_batch = mu_prior.shape[0]
        self.dim = mu_prior.shape[1]

    def sample(self, n_samples, batch_idx=None):
        device = self.mu_prior.device
        dtype = self.mu_prior.dtype

        if batch_idx is not None:
            x0 = torch.randn(
                self.n_batch, n_samples, self.dim, device=device, dtype=dtype
            ) * torch.sqrt(self.var_prior[0]).unsqueeze(0) + self.mu_prior[0].unsqueeze(
                0
            )
            return x0

        x0 = torch.randn(
            self.n_batch, n_samples, self.dim, device=device, dtype=dtype
        ) * torch.sqrt(self.var_prior).unsqueeze(1) + self.mu_prior.unsqueeze(1)
        return x0

    def log_prob(self, x, batch_idx=None):
        if batch_idx is not None:
            return -0.5 * torch.sum(
                ((x - self.mu_prior[batch_idx].unsqueeze(0)) ** 2)
                / self.var_prior[batch_idx].unsqueeze(0),
                dim=-1,
            )
        return -0.5 * torch.sum(
            ((x - self.mu_prior.unsqueeze(1)) ** 2) / self.var_prior.unsqueeze(1),
            dim=-1,
        )
