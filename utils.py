import torch

# Valid optional features that can be included in the NN input beyond [x, lam, z].
VALID_OPTIONAL_FEATURES = frozenset({"grad_log_p", "log_h", "grad_log_h", "grad_log_g"})
# Default: original feature set (backward-compatible).
DEFAULT_FEATURES = frozenset({"grad_log_p", "log_h", "grad_log_h"})


def parse_features(features_str: str) -> frozenset:
    """Parse a comma-separated feature string into a frozenset."""
    if not features_str or features_str.strip().lower() == "none":
        return frozenset()
    names = frozenset(s.strip() for s in features_str.split(",") if s.strip())
    unknown = names - VALID_OPTIONAL_FEATURES
    if unknown:
        raise ValueError(
            f"Unknown optional features: {unknown}. "
            f"Valid options: {sorted(VALID_OPTIONAL_FEATURES)}"
        )
    return names


def compute_input_dim(state_dim: int, meas_dim: int, extra_features: frozenset) -> int:
    """Compute NN input dimension for a given feature set.

    Base: state_dim (x) + 1 (lam) + meas_dim (z).
    Optional: grad_log_p (+state_dim), log_h (+1), grad_log_h (+state_dim), grad_log_g (+state_dim).
    """
    dim = state_dim + 1 + meas_dim
    if "grad_log_p" in extra_features:
        dim += state_dim
    if "log_h" in extra_features:
        dim += 1
    if "grad_log_h" in extra_features:
        dim += state_dim
    if "grad_log_g" in extra_features:
        dim += state_dim
    return dim


def divergence_batched(y, x):
    """
    y.shape = [B, N, D], x.shape = [B, N, D]
    returns divergence per item: [B, N]
    """
    B, N, D = y.shape
    div = 0.0
    for i in range(D):
        grad_i = torch.autograd.grad(
            y[..., i].sum(),  # sum over B and N to get a scalar
            x,
            create_graph=True,
        )[
            0
        ]  # [B, N, D]
        div = div + grad_i[..., i]  # accumulate diagonal term -> [B, N]
    return div


def divergence_hutchinson(y, x, e=None):
    """
    Hutchinson trace estimator for divergence of y w.r.t. x.
    Uses a single Rademacher noise probe (FFJORD pattern).
    y.shape = [B, N, D], x.shape = [B, N, D]
    Returns: [B, N]
    """
    if e is None:
        e = torch.randint(0, 2, y.shape, device=y.device, dtype=y.dtype) * 2 - 1
    e_dydx = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=e,
        create_graph=True,
    )[0]
    return (e_dydx * e).sum(dim=-1)


def energy_distance(x, y):
    """
    Computes the Squared Energy Distance between two sets of samples x and y.
    D^2(x, y) = 2 * E||x - y|| - E||x - x'|| - E||y - y'||

    Args:
        x: Ground truth samples [Batch, N_x, Dim]
        y: Predicted samples [Batch, N_y, Dim]

    Returns:
        dist: Scalar Tensor (Average Squared Energy Distance over batch)
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)

    # x: [B, Nx, D], y: [B, Ny, D]
    # We use torch.cdist (p=2) for efficient Euclidean distance calculation

    # 1. Term XY: Mean distance between samples in X and Y
    # dist_xy shape: [B, Nx, Ny]
    dist_xy = torch.cdist(x, y, p=2)
    term_xy = torch.mean(dist_xy, dim=(1, 2))  # [B]

    # 2. Term XX: Mean distance between samples in X (Ground Truth)
    # dist_xx shape: [B, Nx, Nx]
    dist_xx = torch.cdist(x, x, p=2)
    term_xx = torch.mean(dist_xx, dim=(1, 2))  # [B]

    # 3. Term YY: Mean distance between samples in Y (Predicted)
    # dist_yy shape: [B, Ny, Ny]
    dist_yy = torch.cdist(y, y, p=2)
    term_yy = torch.mean(dist_yy, dim=(1, 2))  # [B]

    # D^2 = 2 * E[||x-y||] - E[||x-x'||] - E[||y-y'||]
    sq_energy_dist = 2 * term_xy - term_xx - term_yy

    # Clamp to 0 to avoid -0.0000 due to float precision errors on identical sets
    sq_energy_dist = torch.clamp(sq_energy_dist, min=0.0)

    return torch.mean(sq_energy_dist)


def sliced_wasserstein_distance(x, y, n_projections=1000, p=2):
    """
    Compute the Sliced Wasserstein Distance between two sets of samples x and y.
    Args:
        x: Tensor of shape (n_samples, n_features) or (batch, n_samples, n_features)
        y: Tensor of shape (n_samples, n_features) or (batch, n_samples, n_features)
        n_projections: Number of random projections to use
        p: The order of the Wasserstein distance
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)

    # x, y: [B, N, D]
    B, N, D = x.shape

    # Generate random projection directions: [D, n_projections]
    direction = torch.randn(D, n_projections, device=x.device)
    direction = direction / torch.norm(direction, dim=0, keepdim=True)

    # Project the samples: [B, N, D] @ [D, n_proj] -> [B, N, n_proj]
    x_proj = x @ direction
    y_proj = y @ direction

    # Sort the projected samples along the N dimension
    x_proj_sorted, _ = torch.sort(x_proj, dim=1)
    y_proj_sorted, _ = torch.sort(y_proj, dim=1)

    # Compute the p-Wasserstein distance for each projection and batch
    # |x - y|^p -> mean over N -> mean over projections
    dist = torch.abs(x_proj_sorted - y_proj_sorted) ** p
    swd_batch = torch.mean(dist, dim=(1, 2))  # [B]

    swd = torch.mean(swd_batch ** (1 / p))
    return swd


def generate_flow_samples(integrator, ode_func, x0, return_curl=False):
    """
    Generate samples by integrating the ODE.
    Returns:
        final_particles: Tensor of shape (batch, n_particles, dim)
        trajectory: Tensor of shape (n_steps, batch, n_particles, dim)
    """
    if return_curl:
        trajectory, curl = integrator(ode_func, x0, return_curl)
        trajectory = trajectory.detach()
    else:
        trajectory = integrator(ode_func, x0).detach()
        curl = None
    final_particles = trajectory[-1].clone().detach()
    return final_particles, trajectory, curl


def create_features(
    x,
    t,
    sub_prior,
    sub_meas,
    z_b,
    extra_features: frozenset = None,
    grad_clip=None,
    log_prob_floor: float = -100.0,
    grad_clamp: float = 1e4,
):
    if extra_features is None:
        extra_features = DEFAULT_FEATURES

    assert x.dtype == torch.float32, f"Expected float32 input, got {x.dtype}"

    if isinstance(t, torch.Tensor):
        lam = t.to(x.device)
    else:
        lam = torch.tensor(t, device=x.device, dtype=torch.float32)

    # Handle measurement shape for broadcasting
    if z_b.ndim == 1:
        z_b = z_b.unsqueeze(1)
    elif z_b.ndim == 2 and z_b.shape[1] > 1:
        z_b = z_b.unsqueeze(1)

    log_h_val = sub_meas.log_prob(x, z_b)

    # Clamp log probabilities to prevent -inf from dominating
    log_h_val_clamped = torch.clamp(log_h_val, min=log_prob_floor)
    log_prior = sub_prior.log_prob(x)
    log_prior_clamped = torch.clamp(log_prior, min=log_prob_floor)

    log_p = log_prior_clamped + lam * log_h_val_clamped

    grad_log_p = torch.autograd.grad(log_p.sum(), x, create_graph=True)[0]
    grad_log_h = torch.autograd.grad(log_h_val.sum(), x, create_graph=True)[0]

    # Clamp gradients to prevent extreme values from -inf log probs
    grad_log_p = torch.clamp(grad_log_p, min=-grad_clamp, max=grad_clamp)
    grad_log_h = torch.clamp(grad_log_h, min=-grad_clamp, max=grad_clamp)

    # Compute prior gradient only if needed as NN input
    if "grad_log_g" in extra_features:
        grad_log_g = torch.autograd.grad(log_prior.sum(), x, create_graph=True)[0]
        grad_log_g = torch.clamp(grad_log_g, min=-grad_clamp, max=grad_clamp)

    lam_feat = lam.view(1, 1, 1).expand(x.size(0), x.size(1), 1)

    z_b_flat = z_b.view(z_b.size(0), -1)
    z_b_flat_expanded = z_b_flat.unsqueeze(1)
    z_feat = z_b_flat_expanded.repeat(1, x.size(1), 1)

    # Build NN input: base [x, lam, z] + optional features
    parts = [x, lam_feat, z_feat]
    if "grad_log_p" in extra_features:
        parts.append(grad_log_p)
    if "log_h" in extra_features:
        parts.append(log_h_val.unsqueeze(-1))
    if "grad_log_h" in extra_features:
        parts.append(grad_log_h)
    if "grad_log_g" in extra_features:
        parts.append(grad_log_g)
    f_net_input = torch.cat(parts, dim=-1)

    return f_net_input, grad_log_p, log_h_val
