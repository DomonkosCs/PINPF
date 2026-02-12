import torch
from pyro.infer.mcmc import MCMC, HMC, NUTS


class AnnealedPotential:
    def __init__(self, prior_log_prob, likelihood_log_prob, z_obs):
        self.prior_log_prob = prior_log_prob
        self.likelihood_log_prob = likelihood_log_prob
        self.z_obs = z_obs
        self.beta = 0.0

    def __call__(self, params):
        x = params["x"]
        log_prior = self.prior_log_prob(x)
        log_like = self.likelihood_log_prob(x, self.z_obs)
        return -(log_prior + self.beta * log_like).sum()


def run_annealed_mcmc(
    prior, meas, z, x0, n_steps=50, n_mcmc_per_step=1, step_size=0.1, use_nuts=False
):
    """
    Runs Annealed MCMC (SMC sampler-like) to transition from prior to posterior.

    Args:
        prior: Object with .log_prob(x)
        meas: Object with .log_prob(x, z)
        z: Observation
        x0: Initial particles sampled from prior [N, D]
        n_steps: Number of annealing steps (betas)
        n_mcmc_per_step: Number of MCMC steps per annealing step
        step_size: Step size for HMC
        use_nuts: Whether to use NUTS (if True) or HMC (if False)

    Returns:
        final_particles: [N, D]
    """

    # Allow passing in plain callables
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

    potential = AnnealedPotential(prior_obj.log_prob, meas_obj.log_prob, z)

    current_samples = x0.clone()

    betas = torch.linspace(0, 1, n_steps + 1)

    print(f"Starting Annealed MCMC with {n_steps} steps...")

    for i, beta in enumerate(betas):
        potential.beta = beta

        if use_nuts:
            kernel = NUTS(
                potential_fn=potential,
                step_size=step_size,
                adapt_step_size=False,
                jit_compile=False,
            )
        else:
            kernel = HMC(
                potential_fn=potential,
                step_size=step_size,
                num_steps=5,
                jit_compile=False,
            )

        mcmc = MCMC(
            kernel,
            num_samples=n_mcmc_per_step,
            warmup_steps=0,
            initial_params={"x": current_samples},
            num_chains=1,
            disable_progbar=True,
        )

        mcmc.run()
        samples = mcmc.get_samples()["x"]  # [n_mcmc_per_step, N, D]
        current_samples = samples[-1]

        if i % 10 == 0:
            print(f"Step {i}/{n_steps}, beta={beta:.4f}")

    return current_samples
