"""Base model interface for 2D/2.5D models."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import nn
from tqdm import tqdm

from utils import AverageMeter, compute_batch_metrics_2d


class BaseModel(nn.Module):
    """Abstract base class for models with training helpers.
    
    All models should inherit from this class and implement:
    - forward(): Model forward pass
    - predict(): Generate predictions for evaluation
    
    Optional overrides:
    - train_one_epoch(): Custom training logic
    - evaluate_one_epoch(): Custom evaluation logic
    - log_one_epoch(): Custom logging logic
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to parser.
        
        Args:
            parser: Argument parser
            
        Returns:
            Modified parser
        """
        return parser

    def build_criterion(self, args):
        """Build loss criterion.
        
        Args:
            args: Configuration arguments
            
        Returns:
            Loss function
        """
        return nn.MSELoss()

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate predictions (for evaluation).
        
        Should be overridden for generative models (diffusion, flow, etc.)
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Predicted tensor
        """
        return self(x)

    def train_one_epoch(
        self,
        loader,
        optimizer,
        scheduler,
        device,
        epoch: int,
        args,
        scaler=None,
    ) -> Dict[str, AverageMeter]:
        """Train for one epoch.
        
        Args:
            loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (can be None)
            device: Training device
            epoch: Current epoch number
            args: Configuration arguments
            scaler: GradScaler for AMP (can be None)
            
        Returns:
            Dictionary of metric AverageMeters
        """
        self.train()
        meters = {name: AverageMeter() for name in ["loss", "ssim", "l1", "l2"]}

        progress = tqdm(loader, desc=f"Train {epoch + 1}", leave=True, dynamic_ncols=True)
        for batch in progress:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            if scaler is not None:
                with torch.autocast(device_type=device.type, enabled=True):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, targets)
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if hasattr(args, 'grad_clip') and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                if hasattr(args, 'grad_clip') and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), args.grad_clip)
                
                optimizer.step()

            # Compute metrics
            with torch.no_grad():
                metrics = compute_batch_metrics_2d(outputs, targets, args.ssim_data_range)
            
            meters["loss"].update(loss.item(), inputs.size(0))
            for key in ["ssim", "l1", "l2"]:
                meters[key].update(metrics[key], inputs.size(0))

            progress.set_postfix(
                loss=f"{meters['loss'].avg:.4e}",
                ssim=f"{meters['ssim'].avg:.3f}",
            )

        return meters

    @torch.no_grad()
    @torch.no_grad()
    def evaluate_one_epoch(
        self, 
        loader, 
        device, 
        args,
        compute_ssim: bool = True
    ) -> Dict[str, AverageMeter]:
        """Evaluate for one epoch.
        
        Args:
            loader: Evaluation data loader
            device: Device
            args: Configuration arguments
            compute_ssim: Whether to compute SSIM (can be slow)
            
        Returns:
            Dictionary of metric AverageMeters
        """
        self.eval()
        torch.cuda.empty_cache()  # Clear cache before evaluation
        meters = {name: AverageMeter() for name in ["loss", "ssim", "l1", "l2"]}

        progress = tqdm(loader, desc="Eval", leave=True, dynamic_ncols=True)
        for batch_idx, batch in enumerate(progress):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            
            # Use predict() for generative models
            outputs = self.predict(inputs, args=args)
            
            # Periodic cache clearing to prevent OOM
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            loss = self.criterion(outputs, targets)
            
            if compute_ssim:
                metrics = compute_batch_metrics_2d(outputs, targets, args.ssim_data_range)
            else:
                metrics = {
                    "ssim": 0.0,
                    "l1": float(torch.abs(outputs - targets).mean().item()),
                    "l2": float(((outputs - targets) ** 2).mean().item()),
                }

            meters["loss"].update(loss.item(), inputs.size(0))
            for key in ["ssim", "l1", "l2"]:
                meters[key].update(metrics[key], inputs.size(0))

            progress.set_postfix(
                loss=f"{meters['loss'].avg:.4e}",
                ssim=f"{meters['ssim'].avg:.3f}",
            )

        return meters

    @torch.no_grad()
    def evaluate_full(
        self, 
        loader, 
        device, 
        args
    ) -> Dict[str, Iterable[float]]:
        """Full evaluation returning per-sample metrics.
        
        Args:
            loader: Evaluation data loader
            device: Device
            args: Configuration arguments
            
        Returns:
            Dictionary of per-sample metric lists
        """
        self.eval()
        metrics_list: Dict[str, list] = {"loss": [], "ssim": [], "l1": [], "l2": []}

        for batch in tqdm(loader, desc="Full Eval", leave=True):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            
            outputs = self.predict(inputs, args=args)
            loss = self.criterion(outputs, targets)
            
            metrics = compute_batch_metrics_2d(
                outputs, targets, args.ssim_data_range, per_sample=True
            )

            metrics_list["loss"].extend([float(loss.item())] * inputs.size(0))
            metrics_list["ssim"].extend(metrics["ssim"])
            metrics_list["l1"].extend(metrics["l1"])
            metrics_list["l2"].extend(metrics["l2"])
            
            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return metrics_list

    def log_one_epoch(
        self,
        logger,
        writer,
        epoch: int,
        args,
        train_metrics: Dict[str, AverageMeter],
        val_metrics: Optional[Dict[str, AverageMeter]],
        time_tracker: Dict,
        current_lr: float,
    ) -> None:
        """Log metrics for one epoch.
        
        Args:
            logger: Logger instance
            writer: TensorBoard writer
            epoch: Current epoch
            args: Configuration arguments
            train_metrics: Training metrics
            val_metrics: Validation metrics (can be None)
            time_tracker: Time tracking dictionary
            current_lr: Current learning rate
        """
        # Get memory usage
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            mem_str = f" | Peak Mem: {peak_mem:.2f}G"
        else:
            mem_str = ""

        if val_metrics is None:
            logger.info(
                "Epoch %4d/%4d | Time: %.2fs | LR: %.3e | "
                "Train Loss: %.6e | SSIM: %.2f%% | L1: %.6e | L2: %.6e%s",
                epoch + 1,
                args.epochs,
                time_tracker["epoch"].val,
                current_lr,
                train_metrics["loss"].avg,
                train_metrics["ssim"].avg * 100.0,
                train_metrics["l1"].avg,
                train_metrics["l2"].avg,
                mem_str,
            )
        else:
            logger.info(
                "Epoch %4d/%4d | Time: %.2fs | LR: %.3e | "
                "Loss: %.6e (train) / %.6e (val) | "
                "SSIM: %.2f%% / %.2f%% | L1: %.6e / %.6e | L2: %.6e / %.6e%s",
                epoch + 1,
                args.epochs,
                time_tracker["epoch"].val,
                current_lr,
                train_metrics["loss"].avg,
                val_metrics["loss"].avg,
                train_metrics["ssim"].avg * 100.0,
                val_metrics["ssim"].avg * 100.0,
                train_metrics["l1"].avg,
                val_metrics["l1"].avg,
                train_metrics["l2"].avg,
                val_metrics["l2"].avg,
                mem_str,
            )

        # TensorBoard logging
        if writer is None:
            return

        # Log scalars
        if val_metrics is None:
            writer.add_scalars("loss", {"train": train_metrics["loss"].avg}, epoch)
            writer.add_scalars("ssim", {"train": train_metrics["ssim"].avg}, epoch)
            writer.add_scalars("l1", {"train": train_metrics["l1"].avg}, epoch)
            writer.add_scalars("l2", {"train": train_metrics["l2"].avg}, epoch)
        else:
            writer.add_scalars(
                "loss",
                {"train": train_metrics["loss"].avg, "val": val_metrics["loss"].avg},
                epoch,
            )
            writer.add_scalars(
                "ssim",
                {"train": train_metrics["ssim"].avg, "val": val_metrics["ssim"].avg},
                epoch,
            )
            writer.add_scalars(
                "l1",
                {"train": train_metrics["l1"].avg, "val": val_metrics["l1"].avg},
                epoch,
            )
            writer.add_scalars(
                "l2",
                {"train": train_metrics["l2"].avg, "val": val_metrics["l2"].avg},
                epoch,
            )
        
        writer.add_scalar("learning_rate", current_lr, epoch)

    def set_vis_batch(self, batch: Dict) -> None:
        """Set visualization batch for logging.
        
        Args:
            batch: Batch dictionary with 'input' and 'target'
        """
        self.vis_batch = batch

    def get_vis_figure(self, epoch: int, device) -> Optional["matplotlib.figure.Figure"]:
        """Generate visualization figure.
        
        Args:
            epoch: Current epoch
            device: Device
            
        Returns:
            Matplotlib figure or None
        """
        if not hasattr(self, 'vis_batch') or self.vis_batch is None:
            return None
        
        from utils import make_slice_figure
        
        self.eval()
        with torch.no_grad():
            x = self.vis_batch["input"].to(device)
            y_true = self.vis_batch["target"].to(device)
            y_pred = self.predict(x).clamp(0.0, 1.0)
        
        return make_slice_figure(
            x, y_true, y_pred,
            num_samples=3,
            title=f"Epoch {epoch + 1}"
        )
