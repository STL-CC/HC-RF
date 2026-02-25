"""Main entry for HC-RF release training.

Responsibilities:
- Parse args and initialize environment
- Run train/val/test loop and checkpoint lifecycle
- Export config.json, result.json and final_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from typing import Dict, Optional
import torch
from torch.utils.tensorboard import SummaryWriter

from args import get_parser
from dataloader import get_loaders
from models import get_model_class
from utils import (
    AverageMeter,
    count_parameters,
    format_parameters,
    get_scheduler,
    load_json,
    make_logger,
    save_json,
    setup_seed,
    summarize_metrics,
    to_serializable,
)


def _resolve_log_dir(path_or_dir: str) -> str:
    if os.path.isfile(path_or_dir):
        parent = os.path.dirname(path_or_dir)
        if os.path.basename(parent) == "ckpts":
            return os.path.dirname(parent)
        return parent
    if os.path.basename(path_or_dir) == "ckpts":
        return os.path.dirname(path_or_dir)
    return path_or_dir


def _resolve_best_ckpt(log_dir: str) -> Optional[str]:
    path = os.path.join(log_dir, "ckpts", "best_model.pth")
    return path if os.path.exists(path) else None


def _resolve_latest_ckpt(log_dir: str) -> Optional[str]:
    path = os.path.join(log_dir, "ckpts", "latest_model.pth")
    return path if os.path.exists(path) else None


def _merge_resume_args(parser: argparse.ArgumentParser, cli_args: argparse.Namespace) -> argparse.Namespace:
    """Resume behavior: default <- config.json <- explicit CLI overrides."""
    defaults = vars(parser.parse_args(["--data_path", cli_args.data_path]))
    resume_log_dir = _resolve_log_dir(cli_args.resume_from)
    config_path = os.path.join(resume_log_dir, "config.json")
    config = load_json(config_path) if os.path.exists(config_path) else {}

    merged = dict(defaults)
    merged.update(config)

    for key, value in vars(cli_args).items():
        if value is not None and value != defaults.get(key):
            merged[key] = value

    merged["resume_from"] = cli_args.resume_from
    merged["load_from"] = None
    return argparse.Namespace(**merged)


def _create_optimizer(model, args):
    params = model.get_param_groups(args) if hasattr(model, "get_param_groups") else [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr, "weight_decay": args.weight_decay}
    ]
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def _save_checkpoint(
    ckpt_path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metrics: Dict,
    args,
) -> None:
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metrics": best_metrics,
        "args": {k: to_serializable(v) for k, v in vars(args).items()},
    }
    torch.save(payload, ckpt_path)


def _load_checkpoint(ckpt_path: str, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=False)
    if optimizer is not None and checkpoint.get("optim_state") is not None:
        optimizer.load_state_dict(checkpoint["optim_state"])
    if scheduler is not None and checkpoint.get("sched_state") is not None:
        scheduler.load_state_dict(checkpoint["sched_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint


def _is_better(args, current: Dict[str, float], best: Dict[str, float]) -> bool:
    metric = args.best_metric
    if metric == "ssim":
        return current.get("ssim", -float("inf")) > best.get("ssim", -float("inf"))
    return current.get("loss", float("inf")) < best.get("loss", float("inf"))


def run_experiment(args, split_bundle: Dict, root_log_dir: str, logger) -> Dict:
    split_name = split_bundle["split_name"]
    split_dir = os.path.join(root_log_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    ckpt_dir = os.path.join(split_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(split_dir, "tensorboard"))
    save_json(os.path.join(split_dir, "config.json"), {k: to_serializable(v) for k, v in vars(args).items()})

    model_cls = get_model_class(args.model)
    model = model_cls(args).to(args.device)
    optimizer = _create_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda") if args.amp and args.device.type == "cuda" else None

    logger.info("[%s] Model=%s Params=%s", split_name, args.model, format_parameters(count_parameters(model)))

    start_epoch = 0
    best_metrics = {"loss": float("inf"), "ssim": -float("inf"), "best_epoch": -1}

    if args.load_from:
        load_dir = _resolve_log_dir(args.load_from)
        best_ckpt = _resolve_best_ckpt(load_dir)
        if best_ckpt is not None:
            _load_checkpoint(best_ckpt, model)
            logger.info("[%s] Loaded weights from %s", split_name, best_ckpt)

    if args.resume_from:
        resume_dir = _resolve_log_dir(args.resume_from)
        latest_ckpt = _resolve_latest_ckpt(resume_dir)
        if latest_ckpt is not None:
            ckpt = _load_checkpoint(latest_ckpt, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_metrics = ckpt.get("best_metrics", best_metrics)
            logger.info("[%s] Resumed from %s at epoch=%d", split_name, latest_ckpt, start_epoch)

    train_loader = split_bundle["train_loader"]
    val_loader = split_bundle["val_loader"]
    test_loader = split_bundle["test_loader"]

    if val_loader is not None and hasattr(model, "set_vis_batch"):
        try:
            model.set_vis_batch(next(iter(val_loader)))
        except StopIteration:
            pass

    train_time_meter = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_metrics = model.train_one_epoch(
            train_loader,
            optimizer,
            scheduler,
            args.device,
            epoch,
            args,
            scaler=scaler,
        )

        should_eval = (val_loader is not None) and (args.eval_every > 0) and (((epoch + 1) % args.eval_every == 0) or (epoch == args.epochs - 1))
        val_metrics = None
        if should_eval:
            val_metrics = model.evaluate_one_epoch(val_loader, args.device, args, compute_ssim=True)

        if scheduler is not None:
            if args.lr_policy == "plateau" and val_metrics is not None:
                scheduler.step(val_metrics["loss"].avg)
            elif args.lr_policy != "none":
                scheduler.step()

        epoch_seconds = time.time() - epoch_start
        train_time_meter.update(epoch_seconds)

        current_lr = optimizer.param_groups[0]["lr"]
        model.log_one_epoch(
            logger=logger,
            writer=writer,
            epoch=epoch,
            args=args,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            time_tracker={"epoch": train_time_meter},
            current_lr=current_lr,
        )

        current_metrics = {
            "loss": val_metrics["loss"].avg if val_metrics is not None else train_metrics["loss"].avg,
            "ssim": val_metrics["ssim"].avg if (val_metrics is not None and "ssim" in val_metrics) else train_metrics.get("ssim", AverageMeter()).avg,
        }
        is_best = _is_better(args, current_metrics, best_metrics)
        if is_best:
            best_metrics.update(current_metrics)
            best_metrics["best_epoch"] = epoch + 1
            if args.save_best:
                _save_checkpoint(
                    os.path.join(ckpt_dir, "best_model.pth"),
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_metrics,
                    args,
                )

        _save_checkpoint(
            os.path.join(ckpt_dir, "latest_model.pth"),
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_metrics,
            args,
        )

        if args.save_every is not None and args.save_every > 0 and ((epoch + 1) % args.save_every == 0):
            _save_checkpoint(
                os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metrics,
                args,
            )

    best_ckpt = os.path.join(ckpt_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        _load_checkpoint(best_ckpt, model)

    eval_loader = test_loader if test_loader is not None else val_loader
    if eval_loader is None:
        per_sample = {"loss": [], "ssim": [], "l1": [], "l2": []}
    else:
        per_sample = model.evaluate_full(eval_loader, args.device, args)

    result = {
        "split": split_name,
        "best_epoch": int(best_metrics.get("best_epoch", -1)),
        "best_metrics": best_metrics,
        "avg_epoch_seconds": float(train_time_meter.avg),
        "metrics": summarize_metrics(per_sample),
        "num_eval_samples": len(eval_loader.dataset) if eval_loader is not None else 0,
        "split_stats": split_bundle["stats"],
    }

    save_json(os.path.join(split_dir, "result.json"), result)
    writer.close()
    return result


def main() -> None:
    parser = get_parser()
    cli_args = parser.parse_args()

    if cli_args.resume_from and cli_args.load_from:
        raise ValueError("--resume_from and --load_from cannot be used together")

    args = _merge_resume_args(parser, cli_args) if cli_args.resume_from else cli_args

    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    setup_seed(args.random_seed)

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    logger, root_log_dir = make_logger(args.log_root, args.exp_name, args.model)
    logger.info("Experiment started: model=%s", args.model)
    logger.info("Device=%s", args.device)

    save_json(os.path.join(root_log_dir, "config.json"), {k: to_serializable(v) for k, v in vars(args).items()})

    split_bundle = get_loaders(args, logger)

    start = time.time()
    logger.info("Running split: %s", split_bundle["split_name"])
    result = run_experiment(args, split_bundle, root_log_dir, logger)

    final_results = {
        "num_splits": 1,
        "metrics": result["metrics"],
        "best_epoch": result["best_epoch"],
        "avg_epoch_seconds": result["avg_epoch_seconds"],
        "split": result,
    }
    save_json(os.path.join(root_log_dir, "final_results.json"), final_results)

    logger.info("Training done in %s", timedelta(seconds=int(time.time() - start)))
    logger.info("Saved final summary: %s", os.path.join(root_log_dir, "final_results.json"))


if __name__ == "__main__":
    main()
