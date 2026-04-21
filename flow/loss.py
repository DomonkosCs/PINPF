import torch


def fpe_loss(
    f: torch.Tensor,
    div_f: torch.Tensor,
    grad_log_p: torch.Tensor,
    log_h: torch.Tensor,
) -> torch.Tensor:
    """
    Fokker-Planck equation residual loss (pure function, no side effects).

    The FPE residual measures how well the velocity field f satisfies:
        d/dlam log rho = -div(f) - <grad_log_p, f>

    Args:
        f: velocity field output [B, N, D]
        div_f: divergence of f w.r.t. x [B, N]
        grad_log_p: score of homotopy density [B, N, D]
        log_h: log-likelihood values [B, N]

    Returns:
        Scalar mean squared residual.
    """
    grad_log_p_dot_f = torch.einsum("bnd,bnd->bn", grad_log_p, f)
    rhs = -div_f - grad_log_p_dot_f
    lhs = log_h - log_h.mean(dim=-1, keepdim=True)
    return ((lhs - rhs) ** 2).mean()
