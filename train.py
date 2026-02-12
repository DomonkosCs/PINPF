import os
import time
import argparse
import json
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from config_tdoa import ConfigTDOA
from config_gmm import ConfigGMM
from utils import divergence_batched, create_features
from models import NeuralFlowModel

try:
    _has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
except Exception:
    _has_mps = False

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if _has_mps else "cpu")
)
print(f"Using device: {DEVICE}")


def save_model(model, results_dir, filename):
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def advance_particles(
    model,
    prior_model,
    meas_model,
    z_mb,
    n_particles,
    d_lambda,
    device,
    omit_grad_features,
    feature_clamp,
    loss_clamp,
    particle_bounds,
    log_prob_floor,
):
    x_prior = prior_model.sample(n_particles).to(device)
    x_current = x_prior.clone()

    num_steps = int(1.0 / d_lambda)

    x_current.requires_grad_(True)
    for i in range(num_steps):
        lam_val = i * d_lambda

        feat, grad_log_p, log_h_val = create_features(
            x_current.float(),
            lam_val,
            prior_model,
            meas_model,
            z_mb,
            omit_grads=omit_grad_features,
        )

        # Clamp features to prevent extreme values
        feat = torch.clamp(feat, min=-feature_clamp, max=feature_clamp)
        grad_log_p = torch.clamp(grad_log_p, min=-feature_clamp, max=feature_clamp)
        log_h_val = torch.clamp(log_h_val, min=log_prob_floor, max=feature_clamp)

        # Model forward pass
        f = model.forward(feat)

        div_f = divergence_batched(f, x_current)

        # LOSS CALCULATION
        grad_log_p_f = torch.einsum("bnd,bnd->bn", grad_log_p, f)
        rhs = -div_f - grad_log_p_f
        lhs = log_h_val - torch.mean(log_h_val, dim=-1, keepdim=True)
        diff_sq = (lhs - rhs) ** 2
        diff_sq = torch.clamp(diff_sq, min=-loss_clamp, max=loss_clamp)
        fpe_loss = diff_sq.mean()

        # Update particles
        # We are not training end-to-end through time, but treating each step as a local regression.
        x_current = (x_current + f.float() * d_lambda).detach()

        # Clamp particle positions to reasonable bounds
        x_current = torch.clamp(x_current, min=-particle_bounds, max=particle_bounds)
        x_current.requires_grad_(True)

        yield fpe_loss, x_current


def train_model(
    results_dir,
    config,
    n_particles,
    d_lambda,
    num_epochs,
    checkpoint_freq,
    lr,
    lr_gamma,
    scheduler_freq,
    layers,
    hidden_dim,
    mini_batch_size,
    log_dir=None,
    grad_clip=1.0,
    load_model_path=None,
    omit_grad_features: bool = False,
    feature_clamp: float = 1e5,
    loss_clamp: float = 1e8,
    particle_bounds: float = 50.0,
    log_prob_floor: float = -300.0,
):
    model = NeuralFlowModel(
        state_dim=config.state_dim,
        meas_dim=config.meas_dim,
        layers=layers,
        neurons_per_layer=hidden_dim,
        omit_grad_features=omit_grad_features,
    )
    model.to(DEVICE)

    if load_model_path is not None and os.path.isfile(load_model_path):
        try:
            state_dict = torch.load(load_model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"Loaded model weights from {load_model_path} for fine-tuning.")
        except Exception as e:
            print(f"Failed to load model from {load_model_path}: {e}")
            print("Continuing with randomly initialized weights.")

    print(f"Starting training on {DEVICE}...")

    # TensorBoard setup
    if log_dir is None:
        run_name = f"pinpf-{time.strftime('%Y%m%d-%H%M%S')}"
        log_dir = os.path.join("runs", run_name)

    writer = SummaryWriter(log_dir=log_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

    progress_bar = tqdm(
        range(0, num_epochs),
        desc="Training PINPF",
        initial=0,
        total=num_epochs,
    )

    full_batch_size = config.get_dataset_size()
    mini_batch_size = min(mini_batch_size, full_batch_size)

    start_time = time.time()

    for epoch in progress_bar:
        model.train()
        epoch_fpe_sum = 0.0
        num_minibatches = 0

        for prior_model, meas_model, z_mb, _ in config.iter_minibatches(
            mini_batch_size, DEVICE, shuffle=True
        ):
            optimizer.zero_grad(set_to_none=True)

            num_steps = int(1.0 / d_lambda)

            step_iterator = advance_particles(
                model,
                prior_model,
                meas_model,
                z_mb,
                n_particles,
                d_lambda,
                DEVICE,
                omit_grad_features=omit_grad_features,
                feature_clamp=feature_clamp,
                loss_clamp=loss_clamp,
                particle_bounds=particle_bounds,
                log_prob_floor=log_prob_floor,
            )

            batch_fpe_sum = 0.0

            for step_loss, _ in step_iterator:
                loss = step_loss / num_steps
                loss.backward()
                batch_fpe_sum += step_loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_fpe_sum += batch_fpe_sum / num_steps
            num_minibatches += 1

        if (epoch + 1) % scheduler_freq == 0 and scheduler.get_last_lr()[0] > 1e-6:
            scheduler.step()

        avg_fpe = epoch_fpe_sum / max(1, num_minibatches)

        # Logging
        writer.add_scalar("loss/fpe", avg_fpe, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        elapsed = time.time() - start_time
        eta = elapsed / (epoch + 1) * (num_epochs - (epoch + 1))

        progress_bar.set_postfix(
            {
                "FPE": f"{avg_fpe:.2e}",
                "ETA": f"{eta:.0f}s",
            }
        )

        if (epoch + 1) % checkpoint_freq == 0:
            save_model(model, results_dir, f"model_epoch_{epoch+1}.pth")

    # Save final model
    save_model(model, results_dir, "model_epoch_final.pth")

    writer.close()
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training PINPF")
    parser.add_argument(
        "--config-type",
        type=str,
        required=True,
        choices=["tdoa", "gmm"],
        help="Configuration type: tdoa, gmm",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-batch", type=int, default=5)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument(
        "--d-lambda", type=float, default=0.01, help="Homotopy step size"
    )
    parser.add_argument("--checkpoint-freq", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.9, help="LR Gamma")
    parser.add_argument("--scheduler-freq", type=int, default=100)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--load-model-path", type=str, default=None)
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the dataset file based on the configuration type",
    )
    parser.add_argument(
        "--omit-grad-features",
        action="store_true",
        help="Omit grad_log_p and grad_log_h from NN features",
    )
    parser.add_argument(
        "--feature-clamp",
        type=float,
        default=1e5,
        help="Clamp value for features to prevent extreme values",
    )
    parser.add_argument(
        "--loss-clamp",
        type=float,
        default=1e8,
        help="Clamp value for loss components to prevent explosion",
    )
    parser.add_argument(
        "--particle-bounds",
        type=float,
        default=50.0,
        help="Clamp particle positions to [-bounds, bounds] in each dimension",
    )
    parser.add_argument(
        "--log-prob-floor",
        type=float,
        default=-300.0,
        help="Minimum log probability before a particle is considered 'lost'",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = f"{args.results_dir}_{timestamp}"
    print(f"Results will be saved to: {results_dir}")

    os.makedirs(results_dir, exist_ok=True)
    config_path = os.path.join(results_dir, "config.json")
    config_dict = vars(args)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    # Select config based on --config-type
    if args.config_type == "gmm":
        ConfigClass = ConfigGMM
    elif args.config_type == "tdoa":
        ConfigClass = ConfigTDOA
    else:
        raise ValueError(f"Unsupported config type: {args.config_type}")

    config = ConfigClass(
        device=DEVICE,
        mode="train",
        max_samples=args.n_batch,
        data_path=args.data_path,
    )

    _ = train_model(
        results_dir=results_dir,
        config=config,
        n_particles=args.n_particles,
        d_lambda=args.d_lambda,
        num_epochs=args.epochs,
        checkpoint_freq=args.checkpoint_freq,
        layers=args.layers,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        lr_gamma=args.lr_gamma,
        scheduler_freq=args.scheduler_freq,
        log_dir=args.log_dir,
        mini_batch_size=args.mini_batch_size,
        grad_clip=args.grad_clip,
        load_model_path=args.load_model_path,
        omit_grad_features=args.omit_grad_features,
        feature_clamp=args.feature_clamp,
        loss_clamp=args.loss_clamp,
        particle_bounds=args.particle_bounds,
        log_prob_floor=args.log_prob_floor,
    )
    print("Training complete.")
