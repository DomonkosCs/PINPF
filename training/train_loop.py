"""Core training loop with AdamW + linear warmup + cosine annealing."""

import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import NeuralFlowModel
from utils import (
    divergence_batched,
    divergence_hutchinson,
    create_features,
)
from flow.loss import fpe_loss

try:
    _has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
except Exception:
    _has_mps = False
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if _has_mps else "cpu")
)


def save_model(model, results_dir, filename):
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def check_tensor_valid(tensor):
    if tensor is None:
        return True
    return torch.isfinite(tensor).all()


def advance_particles(
    model,
    sub_prior,
    sub_meas,
    z_mb,
    n_particles,
    d_lambda,
    device,
    extra_features=None,
    feature_clamp: float = 1e4,
    particle_bounds: float = 50.0,
    log_prob_floor: float = -100.0,
    use_hutchinson: bool = True,
):
    x_prior = sub_prior.sample(n_particles).to(device)
    x_current = x_prior.clone()

    num_steps = int(1.0 / d_lambda)

    x_current.requires_grad_(True)
    for i in range(num_steps):
        lam_val = i * d_lambda

        try:
            feat, grad_log_p, log_h_val = create_features(
                x_current.float(),
                lam_val,
                sub_prior,
                sub_meas,
                z_mb,
                extra_features=extra_features,
            )

            if not check_tensor_valid(feat) or not check_tensor_valid(grad_log_p):
                yield None, 0.0, x_current.detach()
                return

            log_h_val = torch.clamp(log_h_val, min=log_prob_floor, max=feature_clamp)

            f = model.forward(feat)

            if not check_tensor_valid(f):
                yield None, 0.0, x_current.detach()
                return

            if use_hutchinson:
                div_f = divergence_hutchinson(f, x_current)
            else:
                div_f = divergence_batched(f, x_current)

            if not check_tensor_valid(div_f):
                yield None, 0.0, x_current.detach()
                return

            fpe_loss_val = fpe_loss(f, div_f, grad_log_p, log_h_val)
            step_loss = fpe_loss_val

            x_current = (x_current + f.float() * d_lambda).detach()
            x_current = torch.clamp(
                x_current, min=-particle_bounds, max=particle_bounds
            )

            x_current.requires_grad_(True)

            yield step_loss, fpe_loss_val.detach(), x_current

        except RuntimeError as e:
            print(f"RuntimeError at step {i}: {e}")
            yield None, 0.0, x_current.detach()
            return


def train_model(
    results_dir,
    config,
    n_particles,
    d_lambda,
    num_epochs,
    checkpoint_freq,
    lr,
    layers,
    hidden_dim,
    mini_batch_size,
    weight_decay=1e-4,
    log_dir=None,
    grad_clip=1.0,
    extra_features=None,
    use_hutchinson: bool = True,
    feature_clamp: float = 1e5,
    particle_bounds: float = 50.0,
    log_prob_floor: float = -300.0,
):
    activation = nn.SiLU()
    model = NeuralFlowModel(
        state_dim=config.state_dim,
        meas_dim=config.meas_dim,
        layers=layers,
        neurons_per_layer=hidden_dim,
        activation=activation,
        extra_features=extra_features,
    )
    model.to(DEVICE)
    print(f"Starting training on {DEVICE}...")

    if log_dir is None:
        run_name = f"pinpf-{time.strftime('%Y%m%d-%H%M%S')}"
        log_dir = os.path.join("runs", run_name)

    writer = SummaryWriter(log_dir=log_dir)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = max(1, int(0.05 * num_epochs))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    n_particles_train = n_particles

    progress_bar = tqdm(
        range(0, num_epochs),
        desc="Training pinpf",
        initial=0,
        total=num_epochs,
    )

    full_batch_size = config.get_dataset_size()
    mini_batch_size = min(mini_batch_size, full_batch_size)

    start_time = time.time()

    for epoch in progress_bar:
        model.train()
        epoch_loss_sum = 0.0
        epoch_fpe_sum = 0.0
        num_minibatches = 0
        nan_inf_count = 0

        optimizer.zero_grad(set_to_none=True)

        for i, (sub_prior, sub_meas, z_mb, _) in enumerate(
            config.iter_minibatches(mini_batch_size, DEVICE, shuffle=True)
        ):
            num_steps = int(1.0 / d_lambda)

            step_iterator = advance_particles(
                model,
                sub_prior,
                sub_meas,
                z_mb,
                n_particles_train,
                d_lambda,
                DEVICE,
                extra_features=extra_features,
                feature_clamp=feature_clamp,
                particle_bounds=particle_bounds,
                log_prob_floor=log_prob_floor,
                use_hutchinson=use_hutchinson,
            )

            batch_loss_sum = 0.0
            batch_fpe_sum = 0.0
            batch_had_nan = False

            for step_loss, fpe_item, _ in step_iterator:
                if step_loss is None:
                    print(f"Early termination signaled at batch {i}. Cleaning up.")
                    batch_had_nan = True
                    nan_inf_count += 1
                    break

                loss = step_loss / num_steps

                if not torch.isfinite(loss):
                    print(f"NaN/Inf detected in loss at batch {i}. Cleaning up.")
                    batch_had_nan = True
                    nan_inf_count += 1
                    break

                loss.backward()

                batch_loss_sum += float(loss.detach().cpu())
                batch_fpe_sum += float(fpe_item.item())

            if batch_had_nan:
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            valid_gradients = True
            for param in model.parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        valid_gradients = False
                        break

            if not valid_gradients:
                print(f"NaN/Inf in gradients at batch {i}. Skipping update.")
                optimizer.zero_grad(set_to_none=True)
                nan_inf_count += 1
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss_sum += batch_loss_sum
            epoch_fpe_sum += batch_fpe_sum / num_steps
            num_minibatches += 1

        scheduler.step()

        avg_epoch_loss = epoch_loss_sum / max(1, num_minibatches)
        avg_fpe = epoch_fpe_sum / max(1, num_minibatches)

        writer.add_scalar("loss/total", avg_epoch_loss, epoch)
        writer.add_scalar("loss/fpe", avg_fpe, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("train/nan_inf_count", nan_inf_count, epoch)

        if nan_inf_count > num_minibatches * 0.1:
            print(
                f"Warning: High NaN/Inf rate in epoch {epoch}: {nan_inf_count}/{num_minibatches} batches"
            )
            if nan_inf_count > num_minibatches * 0.5:
                print("Critical instability detected. Reducing learning rate by 50%.")
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.5

        elapsed = time.time() - start_time
        eta = elapsed / (epoch + 1) * (num_epochs - (epoch + 1))

        progress_bar.set_postfix(
            {
                "L": f"{avg_epoch_loss:.2e}",
                "FPE": f"{avg_fpe:.2e}",
                "ETA": f"{eta:.0f}s",
            }
        )

        if epoch == 0:
            os.makedirs(results_dir, exist_ok=True)

        if (epoch + 1) % checkpoint_freq == 0:
            save_model(model, results_dir, f"model_epoch_{epoch+1}.pth")

    save_model(model, results_dir, "model_epoch_final.pth")

    writer.close()
    return model
