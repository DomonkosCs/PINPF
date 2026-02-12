import torch


class TDOAMeasurementModel:
    """A class for the TDOA measurement model."""

    SENSOR_1 = torch.tensor([-3.0, 0.0])
    SENSOR_2 = torch.tensor([3.0, 0.0])

    def __init__(self, sigma=0.5):
        self.name = "tdoa"
        self.sigma = torch.atleast_1d(sigma)

    def eval(self, x):
        """Compute the noiseless TDOA measurement."""
        s1 = self.SENSOR_1.to(device=x.device, dtype=x.dtype)
        s2 = self.SENSOR_2.to(device=x.device, dtype=x.dtype)
        dist1 = torch.norm(x - s1, dim=-1)
        dist2 = torch.norm(x - s2, dim=-1)
        ddist = dist1 - dist2
        return ddist

    def sample(self, x):
        """Generate a noisy measurement sample z = h(x) + n, n ~ N(0, sigma^2)"""
        clean_measurement = self.eval(x)
        noise = torch.randn_like(clean_measurement) * self.sigma.to(
            device=x.device
        ).unsqueeze(-1)
        return clean_measurement + noise

    def log_prob(self, x, z, batch_idx=None):
        """Log probability of the measurement model h(x) = p(z|x)"""
        if batch_idx is not None:
            return (
                -0.5
                * (
                    (z - self.eval(x))
                    / self.sigma[batch_idx].to(device=x.device).unsqueeze(-1)
                )
                ** 2
            )
        return (
            -0.5
            * ((z - self.eval(x)) / self.sigma.to(device=x.device).unsqueeze(-1)) ** 2
        )


class GaussianMixtureMeasurementModel:
    """
    Measurement model where the likelihood is a Gaussian Mixture centered at K measurements.
    z should be of shape (..., K, D), representing K measurements.
    Likelihood propto sum(w_k * N(x; z_k, Sigma_k))
    """

    def __init__(self, weights, sigmas):
        """
        Args:
            weights: (K,) mixture weights (sum to 1)
            sigmas: (K, D) standard deviations of the noise components (diagonal cov)
        """
        self.name = "gmm_noise"
        self.weights = torch.as_tensor(weights)
        self.sigmas = torch.as_tensor(sigmas)

        # Normalize weights
        self.weights = self.weights / self.weights.sum()

    def eval(self, x):
        """
        Not used for this model.
        """
        return x

    def sample(self, x):
        """
        Not implemented.
        """
        raise NotImplementedError()

    def log_prob(self, x, z, batch_idx=None):
        """
        Log probability of z given x (interpreted as likelihood of x given z).
        p(z|x) propto sum(w_k * N(x; z_k, Sigma_k))

        x: (..., D)
        z: (..., K, D)
        """
        # x: (..., D) -> (..., 1, D)
        x_expanded = x.unsqueeze(-2)

        if z.shape[0] != 1:
            means = z.unsqueeze(-3)
            sigmas = self.sigmas.unsqueeze(-3).to(x.device)
            weights = self.weights.unsqueeze(-2).to(x.device)
        else:
            means = z
            sigmas = self.sigmas.to(x.device)
            weights = self.weights.to(x.device)

        if batch_idx is not None:
            # If batch_idx is provided, we slice sigmas
            # batch_idx can be an int or a tensor of indices
            sigmas = sigmas[batch_idx]

        if sigmas.dim() == 2 and x.dim() > 1:
            sigmas = sigmas.unsqueeze(0)

        residual = x_expanded - means

        term1 = -0.5 * (residual / sigmas) ** 2
        term1 = torch.sum(term1, dim=-1)
        term2 = -torch.sum(torch.log(sigmas), dim=-1)  # (..., K)
        term3 = (
            -0.5 * x.shape[-1] * torch.log(torch.tensor(2 * torch.pi, device=x.device))
        )

        log_prob_k = term1 + term2 + term3

        log_weights = torch.log(weights)

        log_weighted = log_weights + log_prob_k

        return torch.logsumexp(log_weighted, dim=-1)
