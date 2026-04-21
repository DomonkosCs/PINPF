"""Config dispatch and argument parsing for training."""

import argparse
from problems import ConfigTDOA, ConfigGMM, ConfigNonlinear


def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="pinpf for Bayesian inference")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--n-batch", type=int, default=100, help="Batch size")
    parser.add_argument(
        "--results-dir", type=str, default="results_tmp/results1", help="Results dir"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--n-particles", type=int, default=250, help="Particles per epoch"
    )
    parser.add_argument("--d-lambda", type=float, default=0.01, help="Step size")
    parser.add_argument(
        "--checkpoint-freq", type=int, default=100, help="Checkpoint freq"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay"
    )
    parser.add_argument("--log-dir", type=str, default=None, help="Log dir")
    parser.add_argument("--layers", type=int, default=6, help="Layers")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dim")
    parser.add_argument(
        "--mini-batch-size", type=int, default=64, help="Mini batch size"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="grad_log_p,log_h,grad_log_h",
        help=(
            "Comma-separated optional NN input features beyond [x, lam, z]. "
            "Valid: grad_log_p, log_h, grad_log_h, grad_log_g. "
            "Use 'none' for base features only."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/dataset_gmm_4d.pt",
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--config-type",
        type=str,
        default="gmm",
        choices=["tdoa", "gmm", "nonlinear"],
        help="Config type: tdoa, gmm, or nonlinear",
    )
    parser.add_argument(
        "--use-exact-divergence",
        action="store_true",
        help="Use exact divergence instead of Hutchinson (slower, for validation)",
    )
    parser.add_argument(
        "--feature-clamp",
        type=float,
        default=1e5,
        help="Clamp value for features to prevent extreme values",
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

    from utils import parse_features

    args.extra_features = parse_features(args.features)

    return args


def create_config(args, device):
    """Create training config object based on --config-type.

    Returns:
        Training config object.
    """
    if args.config_type == "gmm":
        ConfigClass = ConfigGMM
    elif args.config_type == "nonlinear":
        ConfigClass = ConfigNonlinear
    else:  # tdoa (default)
        ConfigClass = ConfigTDOA

    config = ConfigClass(
        device=device,
        mode="train",
        max_samples=args.n_batch,
        data_path=args.data_path,
    )

    return config
