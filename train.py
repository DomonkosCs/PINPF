#!/usr/bin/env python
"""pinpf training entry point."""

import os
import time
import json
import torch

from training.config import parse_args, create_config
from training.train_loop import train_model, DEVICE


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = f"{args.results_dir}_{timestamp}"
    print(f"Results will be saved to: {results_dir}")

    os.makedirs(results_dir, exist_ok=True)

    config_dict = vars(args).copy()
    config_dict.pop("extra_features", None)
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    config = create_config(args, DEVICE)

    train_model(
        results_dir=results_dir,
        config=config,
        n_particles=args.n_particles,
        d_lambda=args.d_lambda,
        num_epochs=args.epochs,
        checkpoint_freq=args.checkpoint_freq,
        layers=args.layers,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_dir=args.log_dir,
        mini_batch_size=args.mini_batch_size,
        grad_clip=args.grad_clip,
        extra_features=args.extra_features,
        use_hutchinson=not args.use_exact_divergence,
        feature_clamp=args.feature_clamp,
        particle_bounds=args.particle_bounds,
        log_prob_floor=args.log_prob_floor,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
