import os
import argparse
import sys

import torch
from torch import optim
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    RandomPermutation,
    PiecewiseRationalQuadraticCouplingTransform,
)
from nflows.nn.nets import ResidualNet
from nflows.utils import create_alternating_binary_mask
from utils import sliced_wasserstein_distance, energy_distance


def load_dataset(path: str, split: str = "train", device="cpu"):
    data = torch.load(path, map_location="cpu", weights_only=False)[split]

    mu_prior = data["mu_prior"].to(device)  # [N, state_dim]
    var_prior = data["var_prior"].to(device)  # [N, state_dim]

    z_obs = data["z_obs"].to(device)  # [N, K, state_dim]
    gmm_weights = data["gmm_weights"].to(device)  # [N, K]
    gmm_sigmas = data["gmm_sigmas"].to(device)  # [N, K, state_dim]

    posterior_samples = data["posterior_samples"].to(
        device
    )  # [N, num_mcmc_samples, state_dim]

    return mu_prior, var_prior, z_obs, gmm_weights, gmm_sigmas, posterior_samples


# create conditioning vector per task: [z_obs, mu_prior(flat), var_prior(flat), gmm_weights, gmm_sigmas]
def make_conditioning(mu_prior, var_prior, z_obs, gmm_weights, gmm_sigmas):
    # sizes:
    # mu_prior: [N, D]
    # var_prior: [N, D]
    # z_obs: [N, K, D] -> flatten to [N, K*D]
    # gmm_weights: [N, K] -> [N, K]
    # gmm_sigmas: [N, K, D] -> flatten to [N, K*D]

    N = mu_prior.shape[0]

    if z_obs.dim() == 3:
        z_obs_flat = z_obs.reshape(N, -1)
    else:
        z_obs_flat = z_obs

    if gmm_sigmas.dim() == 3:
        gmm_sigmas_flat = gmm_sigmas.reshape(N, -1)
    else:
        gmm_sigmas_flat = gmm_sigmas

    # Concatenate
    cond = torch.cat(
        [z_obs_flat, mu_prior, var_prior, gmm_weights, gmm_sigmas_flat], dim=-1
    )  # [N, context_dim]
    return cond


def build_conditional_nsf(
    state_dim: int,
    context_dim: int,
    n_blocks: int = 4,
    hidden_features: int = 64,
    num_bins: int = 8,
    tail_bound: float = 6.0,
    dropout: float = 0.0,
):
    transforms = []

    for i in range(n_blocks):
        transforms.append(RandomPermutation(features=state_dim))

        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=create_alternating_binary_mask(
                    features=state_dim, even=(i % 2 == 0)
                ),
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    context_features=context_dim,
                    num_blocks=2,
                    dropout_probability=dropout,
                    use_batch_norm=True,
                ),
                tails="linear",
                tail_bound=tail_bound,
                num_bins=num_bins,
                apply_unconditional_transform=False,
            )
        )

    transform = CompositeTransform(transforms)
    base_dist = StandardNormal([state_dim])

    return Flow(transform, base_dist)


def evaluate(
    flow,
    mu_prior,
    var_prior,
    z_obs,
    gmm_weights,
    gmm_sigmas,
    posterior_samples,
    n_flow_particles=500,
    n_gt_particles=10000,
    task_batch_size=32,
    device="cpu",
):
    flow.eval()
    n_tasks = mu_prior.shape[0]
    n_gt_total = posterior_samples.shape[1]

    # Check if we can satisfy the particle requirement
    if n_gt_total < n_gt_particles:
        n_gt_particles = n_gt_total

    total_swd = 0.0
    total_ed = 0.0

    with torch.no_grad():
        for i in range(0, n_tasks, task_batch_size):
            end = min(i + task_batch_size, n_tasks)
            B = end - i

            # Prepare batch data
            mu = mu_prior[i:end].to(device)
            var = var_prior[i:end].to(device)
            z = z_obs[i:end].to(device)
            w = gmm_weights[i:end].to(device)
            sig = gmm_sigmas[i:end].to(device)

            # Select GT samples (subsample)
            # We take random indices
            idx = torch.randperm(n_gt_total)[:n_gt_particles]
            gt = posterior_samples[i:end][:, idx, :].to(device)

            # Make context
            context = make_conditioning(mu, var, z, w, sig)

            samples = flow.sample(n_flow_particles, context=context)

            if n_flow_particles != n_gt_particles:
                n_eval = min(n_flow_particles, n_gt_particles)
                gt_eval = gt[:, :n_eval, :]
                samples_eval = samples[:, :n_eval, :]
            else:
                gt_eval = gt
                samples_eval = samples

            # Compute metrics
            swd = sliced_wasserstein_distance(gt_eval, samples_eval)
            ed = energy_distance(gt_eval, samples_eval)

            total_swd += swd.item() * B
            total_ed += ed.item() * B

    avg_swd = total_swd / n_tasks
    avg_ed = total_ed / n_tasks
    return avg_swd, avg_ed


def train(args):
    device = torch.device(
        args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    )
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading data from {args.data_path}")
    # Load Train
    (
        mu_prior,
        var_prior,
        z_obs,
        gmm_weights,
        gmm_sigmas,
        posterior_samples,
    ) = load_dataset(args.data_path, split="train", device="cpu")
    # Load Validation
    val_data = load_dataset(args.data_path, split="val", device="cpu")
    # Load Test
    test_data = load_dataset(args.data_path, split="test", device="cpu")

    state_dim = posterior_samples.shape[-1]

    train_context = make_conditioning(
        mu_prior, var_prior, z_obs, gmm_weights, gmm_sigmas
    )  # [N_tasks, context_dim]
    context_dim = train_context.shape[-1]

    print(f"State dim: {state_dim}, Context dim: {context_dim}")
    print(
        f"Training Tasks: {mu_prior.shape[0]}, GT Samples per task: {posterior_samples.shape[1]}"
    )

    n_train_tasks = mu_prior.shape[0]
    n_gt_samples = posterior_samples.shape[1]

    flow = build_conditional_nsf(
        state_dim=state_dim,
        context_dim=context_dim,
        n_blocks=args.n_blocks,
        hidden_features=args.hidden,
        num_bins=args.bins,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(flow.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.eval_only:
        print("Evaluation Mode Enabled")
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            ckpt_path = os.path.join(args.out_dir, "flow_best.pt")

        print(f"Loading checkpoint from: {ckpt_path}")
        best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        flow.load_state_dict(best_ckpt["flow_state"])

        print("Evaluating on Test Set...")
        test_swd, test_ed = evaluate(
            flow,
            *test_data,
            n_flow_particles=args.n_flow_particles_eval,
            n_gt_particles=args.n_gt_particles_eval,
            task_batch_size=args.task_batch_size,
            device=device,
        )
        print(f"Test SWD: {test_swd:.4f}, Test ED: {test_ed:.4f}")
        return

    best_loss = float("inf")

    # Task Indices for shuffling
    task_indices = torch.arange(n_train_tasks)

    for epoch in range(1, args.epochs + 1):
        flow.train()
        total_loss = 0.0

        # Shuffle tasks
        shuffled_indices = task_indices[torch.randperm(n_train_tasks)]

        # Iterate over task batches
        num_batches = int(np.ceil(n_train_tasks / args.task_batch_size))
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}")

        for b_idx in pbar:
            start_idx = b_idx * args.task_batch_size
            end_idx = min(start_idx + args.task_batch_size, n_train_tasks)
            current_batch_indices = shuffled_indices[start_idx:end_idx]
            current_batch_size = len(current_batch_indices)

            batch_context = train_context[current_batch_indices].to(device)

            batch_samples_all = posterior_samples[current_batch_indices]

            if n_gt_samples > args.n_flow_particles_train:
                sample_idx = torch.randperm(n_gt_samples)[: args.n_flow_particles_train]
                batch_samples = batch_samples_all[:, sample_idx, :].to(device)
            else:
                batch_samples = batch_samples_all.to(device)

            x_flat = batch_samples.reshape(-1, state_dim)

            c_expanded = batch_context.unsqueeze(1).expand(
                -1, batch_samples.shape[1], -1
            )
            c_flat = c_expanded.reshape(-1, context_dim)

            log_prob = flow.log_prob(x_flat, context=c_flat)  # [B * N_sub]
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * current_batch_size  # Weighted by num tasks
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        avg_loss = total_loss / n_train_tasks
        print(f"Epoch {epoch} average NLL: {avg_loss:.6f}")

        # Validation
        print("Evaluating on Validation Set...")
        val_swd, val_ed = evaluate(
            flow,
            *val_data,
            n_flow_particles=args.n_flow_particles_eval,
            n_gt_particles=args.n_gt_particles_eval,
            task_batch_size=args.task_batch_size,
            device=device,
        )
        print(f"Epoch {epoch} Val SWD: {val_swd:.4f}, Val ED: {val_ed:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"flow_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "flow_state": flow.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_swd": val_swd,
                "val_ed": val_ed,
            },
            ckpt_path,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "flow_state": flow.state_dict(),
                    "context_dim": context_dim,
                    "state_dim": state_dim,
                    "args": args,
                    "val_swd": val_swd,
                    "val_ed": val_ed,
                },
                os.path.join(args.out_dir, "flow_best.pt"),
            )
            print("Saved best model.")

    print("Training finished. Best NLL:", best_loss)

    # Test Evaluation
    print("Evaluating on Test Set with Best Model...")
    # Load best model
    best_ckpt = torch.load(
        os.path.join(args.out_dir, "flow_best.pt"),
        map_location=device,
        weights_only=False,
    )
    flow.load_state_dict(best_ckpt["flow_state"])

    test_swd, test_ed = evaluate(
        flow,
        *test_data,
        n_flow_particles=args.n_flow_particles_eval,
        n_gt_particles=args.n_gt_particles_eval,
        task_batch_size=args.task_batch_size,
        device=device,
    )
    print(f"Test SWD: {test_swd:.4f}, Test ED: {test_ed:.4f}")

    # Save results to text file
    with open(os.path.join(args.out_dir, "results.txt"), "w") as f:
        f.write(f"Best NLL: {best_loss}\n")
        f.write(f"Test SWD: {test_swd}\n")
        f.write(f"Test ED: {test_ed}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/dataset_gmm_4d.pt",
    )
    parser.add_argument("--out-dir", type=str, default="models/flow_nsf_task")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--task-batch-size", type=int, default=64, help="Number of tasks per batch"
    )

    parser.add_argument(
        "--n-flow-particles-train",
        type=int,
        default=500,
        help="Number of GT samples to use for training gradient step",
    )

    parser.add_argument(
        "--n-flow-particles-eval",
        type=int,
        default=1500,
        help="Number of particles to sample from Flow for eval",
    )

    parser.add_argument(
        "--n-gt-particles-eval",
        type=int,
        default=10000,
        help="Number of GT particles to use for eval comparison",
    )

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-blocks", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--eval-only", action="store_true", help="Only evaluate checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to evaluate",
    )
    args = parser.parse_args()

    train(args)
