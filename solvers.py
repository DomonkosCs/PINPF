import torch
import numpy as np
import time


def solve_euler(ode_func, x0, dt):
    """
    Solves ODE/SDE using fixed-step Euler method.
    Automatically detects if the function represents an ODE or SDE.

    Args:
        ode_func (nn.Module): The dynamics function that returns (f, B).
        x0 (torch.Tensor): Initial particle positions.
        dt (float): Fixed step size.

    Returns:
        torch.Tensor: Trajectory of particle positions.
    """
    num_steps = int(1.0 / dt)

    # Store the full history of particle positions
    trajectory = [x0.clone()]

    x_current = x0.clone()

    for i in range(num_steps):
        lam = torch.tensor([i * dt], device=x0.device, dtype=x0.dtype)

        # Get both drift and possibly diffusion from our function
        result = ode_func(lam, x_current)
        f, B = result

        # Regular Euler step for ODE
        if B is None:
            x_current += f * dt
        # Euler-Maruyama step for SDE
        else:
            w = torch.randn_like(x_current)  # Random noise
            diffusion_step = B @ w.unsqueeze(-1) * np.sqrt(dt)
            x_current += f * dt + diffusion_step.squeeze(-1)

        trajectory.append(x_current.clone())

    return torch.stack(trajectory)


def solve_euler_adaptive(ode_func, x0, DELTA_L, min_dt=1e-3, verbose=False):
    """
    Solves ODE/SDE using adaptive-step Euler method.
    Automatically detects if the function represents an ODE or SDE.

    Args:
        ode_func (nn.Module): The dynamics function that returns (f, B).
        x0 (torch.Tensor): Initial particle positions.
        DELTA_L (float): The adaptive step-size parameter.
        min_dt (float): Minimum step size to avoid infinite loops.

    Returns:
        torch.Tensor: Trajectory of particle positions.
    """
    trajectory = [x0.clone()]

    x_current = x0.clone()
    lam = torch.tensor(0.0, device=x0.device, dtype=x0.dtype)

    while lam < 1.0:
        # Get both drift and possibly diffusion from our function
        result = ode_func(lam, x_current)

        f, B = result

        # Calculate adaptive step size based on drift magnitude
        max_f_norm = torch.norm(f, dim=-1).max()
        dt = DELTA_L / max_f_norm

        # Ensure minimum step size to prevent infinite loops
        dt = max(dt.item(), min_dt)

        # Clamp to ensure we don't overshoot the end
        dt = min(dt, 1.0 - lam)

        # Regular Euler step for ODE
        if B is None:
            x_current += f * dt
        # Euler-Maruyama step for SDE
        else:
            w = torch.randn_like(x_current)  # Random noise
            diffusion_step = B @ w.unsqueeze(-1) * np.sqrt(dt)
            x_current += f * dt + diffusion_step.squeeze(-1)

        lam += dt
        trajectory.append(x_current.clone())

    if verbose:
        print(f"nfe: {len(trajectory) - 1}")  # Subtract 1 for the initial state
    return torch.stack(trajectory)


def create_euler(dlamb=0.01):
    """
    Create a fixed-step Euler integrator with a specified step size.

    Args:
        dlamb (float): Step size for the Euler method.

    Returns:
        function: A function that takes an ODE/SDE function and initial state.
    """

    def euler_integrator(ode_func, x0):
        return solve_euler(ode_func, x0, dlamb)

    return euler_integrator


def create_euler_adaptive(DELTA_L, min_dt=1e-3, verbose=False):
    """
    Create an adaptive-step Euler integrator with a specified step size parameter.

    Args:
        DELTA_L (float): Adaptive step-size parameter.
        min_dt (float): Minimum step size.

    Returns:
        function: A function that takes an ODE/SDE function and initial state.
    """

    def euler_adaptive_integrator(ode_func, x0):
        return solve_euler_adaptive(ode_func, x0, DELTA_L, min_dt, verbose)

    return euler_adaptive_integrator
