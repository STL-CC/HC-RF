"""Release argument center for HC-RF.

This module keeps only production-relevant arguments and supports dynamic
model-specific argument injection based on `--model`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import add_model_specific_args
from utils import to_serializable


def _build_base_parser() -> argparse.ArgumentParser:
    """Build parser with global/common arguments only."""
    parser = argparse.ArgumentParser(
        description="HC-RF: Longitudinal MRI prediction with conditional rectified flow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    exp_group = parser.add_argument_group("Experiment")
    exp_group.add_argument("--model", type=str, default="hc_rf", help="Model name")
    exp_group.add_argument("--exp_name", type=str, default="hcrf_release", help="Experiment tag")
    exp_group.add_argument("--log_root", type=str, default="logs", help="Log root directory")
    exp_group.add_argument("--data_path", type=str, required=True, help="Path to HDF5 dataset")
    exp_group.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")
    exp_group.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    exp_group.add_argument("--max_samples", type=int, default=-1, help="Limit total samples, -1 means all")
    exp_group.add_argument("--random_seed", type=int, default=42, help="Global random seed")

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument("--device", type=str, default="cuda", help="Compute device")
    runtime_group.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    runtime_group.add_argument("--batch_size", type=int, default=16, help="Train batch size")
    runtime_group.add_argument("--eval_batch_size", type=int, default=4, help="Eval batch size")
    runtime_group.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    runtime_group.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    runtime_group.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    runtime_group.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use torch.amp")

    optim_group = parser.add_argument_group("Optimization")
    optim_group.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    optim_group.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    optim_group.add_argument("--weight_decay", type=float, default=0.0)
    optim_group.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping, <=0 disables")
    optim_group.add_argument(
        "--lr_policy",
        type=str,
        default="cosine_warmup",
        choices=["none", "step", "cosine", "cosine_warmup", "plateau"],
    )
    optim_group.add_argument("--warmup_epochs", type=int, default=10)
    optim_group.add_argument("--step_size", type=int, default=10)
    optim_group.add_argument("--gamma", type=float, default=0.5)
    optim_group.add_argument("--eta_min", type=float, default=1e-6)
    optim_group.add_argument("--patience", type=int, default=5)

    log_group = parser.add_argument_group("Evaluation & Checkpoint")
    log_group.add_argument("--eval_every", type=int, default=1, help="Evaluate every N epochs")
    log_group.add_argument("--save_every", type=int, default=None, help="Save periodic ckpt every N epochs")
    log_group.add_argument("--best_metric", type=str, default="loss", choices=["loss", "ssim"])
    log_group.add_argument("--save_best", action=argparse.BooleanOptionalAction, default=True)
    log_group.add_argument("--resume_from", type=str, default=None, help="Resume from log folder")
    log_group.add_argument("--load_from", type=str, default=None, help="Load best model from log folder")
    log_group.add_argument("--ssim_data_range", type=float, default=1.0)

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    data_group.add_argument("--cf_input_channels", type=int, default=3)

    return parser


def get_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    """Return parser with global + model-specific arguments.

    The parser is built in two phases so model-specific args are attached
    according to the user-selected `--model`.
    """
    arg_list = list(argv) if argv is not None else list(sys.argv[1:])
    model_name = "hc_rf"
    for i, token in enumerate(arg_list):
        if token.startswith("--model="):
            model_name = token.split("=", 1)[1].strip() or model_name
            break
        if token == "--model" and i + 1 < len(arg_list):
            model_name = arg_list[i + 1].strip() or model_name
            break

    parser = _build_base_parser()
    add_model_specific_args(parser, model_name)
    return parser


def save_args(args: argparse.Namespace, path: str) -> None:
    """Persist argument namespace to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: to_serializable(v) for k, v in vars(args).items()}, f, ensure_ascii=False, indent=2)


def load_args(path: str) -> argparse.Namespace:
    """Load argument namespace from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return argparse.Namespace(**payload)
