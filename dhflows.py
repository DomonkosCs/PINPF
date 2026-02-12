import torch
import torch.nn as nn
from utils import create_features


class NeuralFlow(nn.Module):
    def __init__(self, model, prior_model, meas_model, z, omit_grads: bool = False):
        super().__init__()
        self.prior_model = prior_model
        self.meas_model = meas_model
        self.z = z
        self.model = model
        self.omit_grads = omit_grads

    def forward(self, lam, x):
        with torch.enable_grad():
            x_clone = x.clone().detach().requires_grad_(True)
            f_net_input, _, _ = create_features(
                x_clone,
                lam,
                self.prior_model,
                self.meas_model,
                self.z,
                omit_grads=self.omit_grads,
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

            # Ensure x requires gradients to compute the score
            x_clone = x.clone().detach().requires_grad_(True)

            # Homotopy log probability: log g(x) + λ*log h(x)
            # Ensure lam is a tensor for computation if it's a float
            if isinstance(lam, float):
                lam = torch.tensor(lam, device=x.device)

            log_p = self.prior_model.log_prob(x_clone) + lam * self.meas_model.log_prob(
                x_clone, self.z
            )

            score = torch.autograd.grad(log_p.sum(), x_clone, create_graph=False)[0]

            # Squared norm of the score
            score_norm_sq = torch.sum(score**2, dim=-1, keepdim=True)

            # log h(x)
            # Unsqueeze to make dimensions compatible for broadcasting: (N,) -> (N, 1)
            log_h_val = self.meas_model.log_prob(x, self.z).unsqueeze(-1)

            # Add epsilon for numerical stability to avoid division by zero
            epsilon = 1e-9

            # The flow vector f(x,λ) from Eq. (14)
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
        eye = torch.eye(z.shape[-1], device=z.device)  # [md,md]
        self.R_meas = meas_model.sigma[:, None, None] ** 2 * eye  # [b,md,md]

        self.R_meas_inv = torch.linalg.inv(self.R_meas)
        self.V_prior_inv = torch.linalg.inv(self.V_prior)

    def forward(self, lam, x):
        # B: batch_size, N: n_particles, D: dimension of state (2)
        B, N, D = x.shape
        device = x.device
        I = torch.eye(D, device=device)

        x_clone = x.clone().detach().requires_grad_(True)
        h = self.meas_model.eval(x_clone)  # (B, N)

        grad_h = torch.autograd.grad(h.sum(), x_clone, create_graph=False)[0]
        H = grad_h.unsqueeze(-2)  # (B,N,1,D)

        P = self.V_prior.to(device).unsqueeze(1)  # (B, 1, D, D)

        P_H_T = P @ H.transpose(-1, -2)  # (B, N, D, 1)

        H_P_H_T = H @ P_H_T  # (B, N, 1, 1)

        A_inv_term = torch.linalg.inv(
            lam * H_P_H_T + self.R_meas.to(device).unsqueeze(1)
        )  # (B, N, 1, 1)

        A = -0.5 * P_H_T @ A_inv_term @ H  # (B, N, D, D)

        innovation = self.z.to(device) - h  # (B, N)
        innovation = innovation.unsqueeze(-1).unsqueeze(-1)  # (B,N,1,1)

        z_corrected = innovation + H @ x_clone.unsqueeze(-1)  #  (B, N, 1, 1)

        P_H_T_R_inv = P_H_T @ self.R_meas_inv.to(device).unsqueeze(-1)  #  (B, N, D, 1)

        term_b1_inner = P_H_T_R_inv @ z_corrected
        term_b1 = (I + lam * A) @ term_b1_inner

        term_b2 = A @ self.mu_prior.to(device).unsqueeze(1).unsqueeze(-1)  # (B,N,D,1)

        b = (I + 2 * lam * A) @ (term_b1 + term_b2)

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

        eye = torch.eye(z.shape[-1], device=z.device)  # [md,md]
        self.R_meas = meas_model.sigma[:, None, None] ** 2 * eye  # [b,md,md]
        self.R_meas_inv = torch.linalg.inv(self.R_meas)
        try:
            self.H = meas_model.H
        except AttributeError:
            self.H = None

        self.V_prior_inv = torch.linalg.inv(self.V_prior)

    def forward(self, lam, x):
        # B: batch_size, N: n_particles, D: dimension of state (2)
        B, N, D = x.shape
        device = x.device
        I = torch.eye(D, device=device)

        x_mean = torch.mean(x, dim=1, keepdim=True)  # (B,1, D)
        x_mean_grad = x_mean.clone().detach().requires_grad_(True)
        h_mean = self.meas_model.eval(x_mean_grad)  # (B, meas_dim)
        if self.H is not None:
            H = (
                self.H.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)
            )  # (B, meas_dim, state_dim)
        else:
            H = torch.autograd.grad(h_mean.sum(), x_mean_grad, create_graph=False)[0]

        P = self.V_prior.to(device)  # (B, state_sim, state_dim)
        P_H_T = P @ H.transpose(-1, -2)  # (B, state_dim, meas_dim)
        H_P_H_T = H @ P_H_T  # (B, meas_dim, meas_dim)

        A_inv_term = torch.linalg.inv(lam * H_P_H_T + self.R_meas.to(device))

        A = -0.5 * P_H_T @ A_inv_term @ H  # (B, state_dim, state_dim)

        innovation = self.z.to(device) - h_mean  # (B, 1, meas_dim)
        innovation = torch.atleast_3d(innovation)

        z_corrected = innovation + (H @ x_mean_grad.transpose(-1, -2)).transpose(
            -1, -2
        )  #  (B, 1, meas_dim)

        P_H_T_R_inv = P_H_T @ self.R_meas_inv.to(device)  #  (B, state_dim, meas_dim)

        term_b1_inner = P_H_T_R_inv @ z_corrected.transpose(-1, -2)  # (B, state_dim, 1)
        term_b1 = (I + lam * A) @ term_b1_inner  # (B, D, 1)

        term_b2 = A @ self.mu_prior.to(device).unsqueeze(-1)  # (B, D, 1)

        b = (I + 2 * lam * A) @ (term_b1 + term_b2)  # (B, D, 1)

        f = (A @ x.transpose(-1, -2) + b).transpose(-1, -2)  # (B, N, D)

        return f, None
