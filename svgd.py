import torch
import torch.autograd as autograd
import torch.optim as optim


class SVGD:
    """
    Implementation of Stein Variational Gradient Descent.
    Reference: https://github.com/dilinwang820/Stein-Variational-Gradient-Descent
    """

    def __init__(self, prior_model, meas_model, z_obs):
        """
        Args:
            prior_model: Object with .log_prob(x)
            meas_model: Object with .log_prob(x, z)
            z_obs: The specific observation for this inference task (shape: [Meas_Dim])
        """
        self.prior_model = prior_model
        self.meas_model = meas_model
        self.z_obs = z_obs

    def _rbf_kernel(self, X):
        """
        Computes the RBF kernel matrix and its gradients.
        """
        N, _ = X.shape

        diff = X.unsqueeze(1) - X.unsqueeze(0)  # [N, N, D]

        dist_sq = torch.sum(diff**2, dim=-1)

        # Median heuristic for bandwidth h
        median = torch.median(dist_sq.view(-1))
        h = median / (torch.log(torch.tensor(N, dtype=X.dtype, device=X.device)) + 1e-6)

        h = torch.max(h, torch.tensor(1e-3, device=X.device))

        k_xy = torch.exp(-dist_sq / h)  # [N, N]

        grad_k = -k_xy.unsqueeze(-1) * diff * (2 / h)  # [N, N, D]

        return k_xy, grad_k

    def get_phi(self, X):
        """
        Calculates the SVGD update direction (phi).
        """
        N = X.shape[0]

        X_detach = X.clone().detach().requires_grad_(True)

        z_input = self.z_obs

        log_prior = self.prior_model.log_prob(X_detach)
        log_like = self.meas_model.log_prob(X_detach, z_input)
        log_prob = log_prior + log_like

        score = autograd.grad(log_prob.sum(), X_detach)[0]  # [N, D]

        k_xy, dx_k_xy = self._rbf_kernel(X.detach())

        term1 = torch.matmul(k_xy, score)
        term2 = torch.sum(dx_k_xy, dim=0)

        phi = (term1 + term2) / N

        return phi


def run_svgd(prior, meas, z, x0, n_iter=100, lr=0.1):

    def _as_log_prob_obj(obj):
        if hasattr(obj, "log_prob"):
            return obj

        if callable(obj):

            class _Wrapper:
                def __init__(self, func):
                    self.func = func

                def log_prob(self, *args, **kwargs):
                    return self.func(*args, **kwargs)

            return _Wrapper(obj)

        raise TypeError(
            "prior and meas must be either callables or objects exposing a log_prob method"
        )

    prior_obj = _as_log_prob_obj(prior)
    meas_obj = _as_log_prob_obj(meas)

    svgd = SVGD(prior_obj, meas_obj, z)

    x = x0.clone().detach().requires_grad_(True)

    optimizer = optim.Adagrad([x], lr=lr)

    traj = [x.clone().detach()]

    for _ in range(n_iter):
        optimizer.zero_grad()

        phi = svgd.get_phi(x)
        x.grad = -phi

        optimizer.step()

        traj.append(x.clone().detach())

    return x.detach(), torch.stack(traj)
