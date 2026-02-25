"""Utilities for reproducible training and experiment management."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Dict, Iterable

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_seed(seed: int = 42) -> None:
    """Set random seed for Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Track running average for scalar metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


def make_logger(
    log_root: str,
    exp_name: str,
    model_name: str,
    log_dir: str | None = None,
    append: bool = False,
) -> tuple[logging.Logger, str]:
    """Create logger and output directory.

    Directory format: `YYMMDD-HH:MM:SS-ModelName-ExpName`.
    """
    if log_dir is None:
        timestamp = datetime.now().strftime("%y%m%d-%H:%M:%S")
        folder = f"{timestamp}-{model_name}-{exp_name}"
        log_dir = os.path.join(log_root, folder)
    os.makedirs(log_dir, exist_ok=True)

    logger_name = f"{model_name}:{exp_name}:{os.path.basename(log_dir)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"), mode="a" if append else "w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_dir


def get_scheduler(optimizer, args):
    """Build scheduler from release configs."""
    if args.lr_policy == "none":
        return None
    if args.lr_policy == "step":
        return lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.lr_policy == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.eta_min)
    if args.lr_policy == "plateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.gamma, patience=args.patience)
    if args.lr_policy == "cosine_warmup":
        warmup_epochs = max(0, int(args.warmup_epochs))
        decay_epochs = max(1, int(args.epochs) - warmup_epochs)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return 1.0
            progress = (epoch - warmup_epochs) / decay_epochs
            min_ratio = args.eta_min / max(args.lr, 1e-12)
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unsupported lr policy: {args.lr_policy}")


def compute_ssim_2d(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute SSIM for 2D images using a simple implementation.
    
    Args:
        pred: Predicted images [B, C, H, W]
        target: Target images [B, C, H, W]
        data_range: Data range for SSIM calculation
        
    Returns:
        SSIM values tensor [B]
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    kernel_size = 11
    sigma = 1.5
    channels = pred.size(1)
    device = pred.device
    dtype = pred.dtype
    
    # Reuse cached Gaussian kernels to avoid rebuilding every call.
    cache_key = (kernel_size, channels, device, dtype)
    if not hasattr(compute_ssim_2d, '_kernel_cache'):
        compute_ssim_2d._kernel_cache = {}
    
    if cache_key not in compute_ssim_2d._kernel_cache:
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        window = window.to(device, dtype=dtype)
        window = window.expand(channels, 1, -1, -1).contiguous()
        compute_ssim_2d._kernel_cache[cache_key] = window
    
    window = compute_ssim_2d._kernel_cache[cache_key]
    
    # Compute means
    mu1 = F.conv2d(pred, window, padding=kernel_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=kernel_size // 2, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(pred ** 2, window, padding=kernel_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=kernel_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=kernel_size // 2, groups=channels) - mu1_mu2
    
    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Average over spatial dimensions and channels
    ssim_vals = ssim_map.flatten(1).mean(dim=1)
    
    return ssim_vals


def compute_batch_metrics_2d(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    data_range: float = 1.0,
    per_sample: bool = False
) -> Dict[str, Iterable[float]]:
    """Compute SSIM, L1, L2 metrics for a 2D batch.
    
    Args:
        pred: Predicted images [B, C, H, W] or [B, C, D, H, W] for 2.5D
        target: Target images (same shape as pred)
        data_range: Data range for SSIM calculation
        per_sample: Whether to return per-sample metrics
        
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # Handle 2.5D case (B, C, D, H, W) -> treat as batch of 2D
        if pred.dim() == 5:
            B, C, D, H, W = pred.shape
            # Take the middle slice for SSIM
            mid_d = D // 2
            pred_2d = pred[:, :, mid_d, :, :]
            target_2d = target[:, :, mid_d, :, :]
            ssim_vals = compute_ssim_2d(pred_2d, target_2d, data_range)
        else:
            ssim_vals = compute_ssim_2d(pred, target, data_range)
        
        l1_vals = F.l1_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
        l2_vals = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
        loss_vals = l2_vals

    if per_sample:
        return {
            "ssim": ssim_vals.detach().cpu().tolist(),
            "l1": l1_vals.detach().cpu().tolist(),
            "l2": l2_vals.detach().cpu().tolist(),
            "loss": loss_vals.detach().cpu().tolist(),
        }

    return {
        "ssim": float(ssim_vals.mean().item()),
        "l1": float(l1_vals.mean().item()),
        "l2": float(l2_vals.mean().item()),
        "loss": float(loss_vals.mean().item()),
    }


def save_json(path: str, payload: Dict) -> None:
    """Save dictionary to json file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    """Load dictionary from json file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_metrics(metrics: Dict[str, Iterable[float]]) -> Dict:
    """Summarize metric list values with mean/std."""
    summary = {}
    for key, values in metrics.items():
        values_np = np.array(values, dtype=np.float32)
        summary[key] = {
            "mean": float(values_np.mean()) if values_np.size else None,
            "std": float(values_np.std()) if values_np.size else None,
        }
    return summary


def make_slice_figure(
    x: torch.Tensor, 
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
    num_samples: int = 3,
    title: str = ""
) -> plt.Figure:
    """Visualize model input/target/prediction triplets."""
    batch_size = x.size(0)
    num_samples = min(num_samples, batch_size)
    
    # Randomly select samples
    if num_samples < batch_size:
        indices = torch.randperm(batch_size)[:num_samples]
    else:
        indices = torch.arange(batch_size)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    # Handle single sample case
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        x_np = x[idx].detach().cpu().squeeze().numpy()
        y_true_np = y_true[idx].detach().cpu().squeeze().numpy()
        y_pred_np = y_pred[idx].detach().cpu().squeeze().numpy()

        # Handle 2.5D case
        if x_np.ndim == 3:
            mid = x_np.shape[0] // 2
            x_np = x_np[mid]
            y_true_np = y_true_np[mid] if y_true_np.ndim == 3 else y_true_np
            y_pred_np = y_pred_np[mid] if y_pred_np.ndim == 3 else y_pred_np

        axes[i, 0].imshow(x_np, cmap="gray")
        axes[i, 0].set_title(f"Input (T1) - Sample {i+1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(y_true_np, cmap="gray")
        axes[i, 1].set_title(f"Target (T2) - Sample {i+1}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(y_pred_np, cmap="gray")
        axes[i, 2].set_title(f"Prediction - Sample {i+1}")
        axes[i, 2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    return fig


def make_slice_figure_with_roi(
    x: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    roi_mask_input: torch.Tensor,
    roi_mask_target: torch.Tensor,
    num_samples: int = 3,
    title: str = "",
    alpha: float = 0.35,
) -> plt.Figure:
    """Visualize slices with ROI mask overlays.

    Args:
        x: Input slices [B, C, H, W]
        y_true: Target slices [B, C, H, W]
        y_pred: Predicted slices [B, C, H, W]
        roi_mask_input: ROI masks for input [B, 2, H, W]
        roi_mask_target: ROI masks for target [B, 2, H, W]
        num_samples: Number of samples to visualize
        title: Figure title
        alpha: Overlay transparency

    Returns:
        Matplotlib figure with overlays
    """
    batch_size = x.size(0)
    num_samples = min(num_samples, batch_size)

    if num_samples < batch_size:
        indices = torch.randperm(batch_size)[:num_samples]
    else:
        indices = torch.arange(batch_size)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        x_np = x[idx].detach().cpu().squeeze().numpy()
        y_true_np = y_true[idx].detach().cpu().squeeze().numpy()
        y_pred_np = y_pred[idx].detach().cpu().squeeze().numpy()
        roi_in = roi_mask_input[idx].detach().cpu().numpy()
        roi_tg = roi_mask_target[idx].detach().cpu().numpy()

        if x_np.ndim == 3:
            mid = x_np.shape[0] // 2
            x_np = x_np[mid]
            y_true_np = y_true_np[mid] if y_true_np.ndim == 3 else y_true_np
            y_pred_np = y_pred_np[mid] if y_pred_np.ndim == 3 else y_pred_np

        axes[i, 0].imshow(x_np, cmap="gray")
        axes[i, 0].imshow(roi_in[0], cmap="Reds", alpha=alpha)
        axes[i, 0].imshow(roi_in[1], cmap="Blues", alpha=alpha)
        axes[i, 0].set_title(f"Input (T1) + ROI - Sample {i+1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(y_true_np, cmap="gray")
        axes[i, 1].imshow(roi_tg[0], cmap="Reds", alpha=alpha)
        axes[i, 1].imshow(roi_tg[1], cmap="Blues", alpha=alpha)
        axes[i, 1].set_title(f"Target (T2) + ROI - Sample {i+1}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(y_pred_np, cmap="gray")
        axes[i, 2].imshow(roi_tg[0], cmap="Reds", alpha=alpha)
        axes[i, 2].imshow(roi_tg[1], cmap="Blues", alpha=alpha)
        axes[i, 2].set_title(f"Prediction + ROI - Sample {i+1}")
        axes[i, 2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    return fig


def to_serializable(obj):
    """Convert object to JSON-serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable representation
    """
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_parameters(num_params: int) -> str:
    """Format parameter count with appropriate suffix.
    
    Args:
        num_params: Number of parameters
        
    Returns:
        Formatted string
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)
