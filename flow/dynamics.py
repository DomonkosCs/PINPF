import torch
import torch.nn as nn
from utils import create_features


class NeuralFlow(nn.Module):
    def __init__(self, model, prior_model, meas_model, z, extra_features=None):
        super().__init__()
        # Load to CPU to be safe, caller should move model/tensors to device as needed
        self.prior_model = prior_model
        self.meas_model = meas_model
        self.z = z
        self.model = model
        self.extra_features = extra_features

    def forward(self, lam, x):
        with torch.enable_grad():
            x_clone = x.clone().detach().requires_grad_(True)
            z_b = self.z
            f_net_input, _, _ = create_features(
                x_clone,
                lam,
                self.prior_model,
                self.meas_model,
                z_b,
                extra_features=self.extra_features,
            )
            f = self.model.f_net(f_net_input)
        return f, None


class IncompressibleFlow(nn.Module):

    def __init__(self, meas_model, prior_model, z):
        super().__init__()
        self.meas_model = meas_model
        self.prior_model = prior_model
        self.z = z

    def forward(self, lam, x):
        with torch.enable_grad():
            # By convention, solvers expect the function f(t, y) -> f(lam, x)

            # Ensure x requires gradients to compute the score
            x_clone = x.clone().detach().requires_grad_(True)

            # Homotopy log probability: log g(x) + lambda*log h(x)
            # Ensure lam is a tensor for computation if it's a float
            if isinstance(lam, float):
                lam = torch.tensor(lam, device=x.device)

            log_p = self.prior_model.log_prob(x_clone) + lam * self.meas_model.log_prob(
                x_clone, self.z
            )

            # The score: grad log p(x, lambda)
            # create_graph=True is needed if we were to differentiate through this again, but here we just need the value.
            # However, if x_clone doesn't actually participate in the graph (e.g. log_prob implementation is detached), this fails.
            # Assuming log_prob is differentiable.
            score = torch.autograd.grad(log_p.sum(), x_clone, create_graph=False)[0]

            # Squared norm of the score
            score_norm_sq = torch.sum(score**2, dim=-1, keepdim=True)

            # log h(x)
            # Unsqueeze to make dimensions compatible for broadcasting: (N,) -> (N, 1)
            log_h_val = self.meas_model.log_prob(x, self.z).unsqueeze(-1)

            # Add epsilon for numerical stability to avoid division by zero
            epsilon = 1e-9

            # The flow vector f(x, lambda) from Eq. (14)
            f = -log_h_val * score / (score_norm_sq + epsilon)

        return f, None


class LocalGaussianExactFlow(nn.Module):
    def __init__(self, meas_model, prior_model, z):
        super().__init__()

        self.meas_model = meas_model
        self.prior_model = prior_model
        self.z = z

        self.V_prior = torch.diag_embed(prior_model.var_prior)
        self.mu_prior = prior_model.mu_prior
        sigma = meas_model.sigma
        if sigma.dim() == 1:
            eye = torch.eye(z.shape[-1], device=z.device)
            self.R_meas = sigma[:, None, None] ** 2 * eye  # (B, m, m)
        else:
            self.R_meas = torch.diag_embed(sigma**2)  # (B, m, m)

        self.R_meas_inv = torch.linalg.inv(self.R_meas)
        self.V_prior_inv = torch.linalg.inv(self.V_prior)

    def forward(self, lam, x):
        B, N, D = x.shape
        device = x.device
        I = torch.eye(D, device=device)

        with torch.enable_grad():
            x_clone = x.clone().detach().requires_grad_(True)
            h = self.meas_model.eval(x_clone)  # (B, N) or (B, N, m)
            if h.dim() == 2:
                h = h.unsqueeze(-1)  # (B, N, 1)
            m = h.shape[-1]
            H_rows = []
            for k in range(m):
                g = torch.autograd.grad(
                    h[..., k].sum(),
                    x_clone,
                    retain_graph=(k < m - 1),
                    create_graph=False,
                )[
                    0
                ]  # (B, N, D)
                H_rows.append(g)
            H = torch.stack(H_rows, dim=-2)  # (B, N, m, D)

        P = self.V_prior.to(device).unsqueeze(1)  # (B, 1, D, D)

        P_H_T = P @ H.transpose(-1, -2)  # (B, N, D, m)

        H_P_H_T = H @ P_H_T  # (B, N, m, m)

        A_inv_term = torch.linalg.inv(
            lam * H_P_H_T + self.R_meas.to(device).unsqueeze(1)
        )  # (B, N, m, m)

        A = -0.5 * P_H_T @ A_inv_term @ H  # (B, N, D, D)

        innovation = self.z.to(device).unsqueeze(1) - h  # (B, N, m)
        innovation = innovation.unsqueeze(-1)  # (B, N, m, 1)

        z_corrected = innovation + H @ x_clone.unsqueeze(-1)  # (B, N, m, 1)

        P_H_T_R_inv = P_H_T @ self.R_meas_inv.to(device).unsqueeze(1)  # (B, N, D, m)

        term_b1_inner = P_H_T_R_inv @ z_corrected  # (B, N, D, 1)
        term_b1 = (I + lam * A) @ term_b1_inner  # (B, N, D, 1)

        term_b2 = A @ self.mu_prior.to(device).unsqueeze(1).unsqueeze(
            -1
        )  # (B, N, D, 1)

        b = (I + 2 * lam * A) @ (term_b1 + term_b2)  # (B, N, D, 1)

        f = A @ x_clone.unsqueeze(-1) + b  # (B, N, D, 1)

        return f.squeeze(-1), None


class MeanGaussianExactFlow(nn.Module):
    def __init__(self, meas_model, prior_model, z):
        super().__init__()

        self.meas_model = meas_model
        self.prior_model = prior_model
        self.z = z

        try:
            self.V_prior = prior_model.cov_prior
        except:
            self.V_prior = torch.diag_embed(prior_model.var_prior)
        self.mu_prior = prior_model.mu_prior

        sigma = meas_model.sigma
        if sigma.dim() == 1:
            eye = torch.eye(z.shape[-1], device=z.device)
            self.R_meas = sigma[:, None, None] ** 2 * eye  # (B, m, m)
        else:
            self.R_meas = torch.diag_embed(sigma**2)  # (B, m, m)
        self.R_meas_inv = torch.linalg.inv(self.R_meas)
        try:
            self.H = meas_model.H
        except AttributeError:
            self.H = None

        self.V_prior_inv = torch.linalg.inv(self.V_prior)

    def forward(self, lam, x):
        B, N, D = x.shape
        device = x.device
        I = torch.eye(D, device=device)

        x_mean = torch.mean(x, dim=1, keepdim=True)  # (B, 1, D)
        with torch.enable_grad():
            x_mean_grad = x_mean.clone().detach().requires_grad_(True)
            h_mean = self.meas_model.eval(x_mean_grad)  # (B, 1, m) or (B, 1)
            if h_mean.dim() == 2:
                h_mean = h_mean.unsqueeze(1)  # (B, 1, m)
            m = h_mean.shape[-1]
            if self.H is not None:
                H = (
                    self.H.to(device=x.device, dtype=x.dtype)
                    .unsqueeze(0)
                    .expand(B, -1, -1)
                )  # (B, m, D)
            else:
                H_rows = []
                for k in range(m):
                    g = torch.autograd.grad(
                        h_mean[..., k].sum(),
                        x_mean_grad,
                        retain_graph=(k < m - 1),
                        create_graph=False,
                    )[
                        0
                    ]  # (B, 1, D)
                    H_rows.append(g.squeeze(1))  # (B, D)
                H = torch.stack(H_rows, dim=-2)  # (B, m, D)

        P = self.V_prior.to(device)  # (B, D, D)
        P_H_T = P @ H.transpose(-1, -2)  # (B, D, m)
        H_P_H_T = H @ P_H_T  # (B, m, m)

        A_inv_term = torch.linalg.inv(
            lam * H_P_H_T + self.R_meas.to(device)
        )  # (B, m, m)

        A = -0.5 * P_H_T @ A_inv_term @ H  # (B, D, D)

        innovation = self.z.to(device).unsqueeze(1) - h_mean  # (B, 1, m)

        z_corrected = innovation + (H @ x_mean_grad.transpose(-1, -2)).transpose(
            -1, -2
        )  # (B, 1, m)

        P_H_T_R_inv = P_H_T @ self.R_meas_inv.to(device)  # (B, D, m)

        term_b1_inner = P_H_T_R_inv @ z_corrected.transpose(-1, -2)  # (B, D, 1)
        term_b1 = (I + lam * A) @ term_b1_inner  # (B, D, 1)

        term_b2 = A @ self.mu_prior.to(device).unsqueeze(-1)  # (B, D, 1)

        b = (I + 2 * lam * A) @ (term_b1 + term_b2)  # (B, D, 1)

        f = (A @ x.transpose(-1, -2) + b).transpose(-1, -2)  # (B, N, D)

        return f, None
