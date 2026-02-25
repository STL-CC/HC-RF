"""Conditional Rectified Flow model with patient meta-vector conditioning.

Adds History Encoder (HE) to produce patient-level vector c, which is injected
as context tokens into DiT attention. This release keeps ROI-guided dynamic
attention, ROI-focused flow loss, and ROI area regression in the default path.
"""

from __future__ import annotations

from typing import Dict, Optional
import math
import os

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from modules.dit import DiT2DConditionedWithContext
from modules.seg_heads import build_seg_head
from utils import AverageMeter, compute_batch_metrics_2d, make_slice_figure_with_roi
from .base import BaseModel


# ======================= History Encoder (from HETest) =======================

class CosineTimeEncoding(nn.Module):
    def __init__(self, out_dim: int, hidden_dim: int = 128, num_freqs: int = 8):
        super().__init__()
        self.num_freqs = num_freqs
        self.proj = nn.Sequential(
            nn.Linear(num_freqs * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, t: torch.Tensor):
        squeeze = False
        if t.dim() == 1:
            t = t.unsqueeze(1)
            squeeze = True
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        freqs = torch.arange(1, self.num_freqs + 1, device=t.device, dtype=t.dtype).view(1, 1, -1)
        angles = 2 * math.pi * t * freqs
        feats = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        out = self.proj(feats)
        return out.squeeze(1) if squeeze else out


class ScalarEncoder(nn.Module):
    def __init__(self, hidden_dim: int, score_dim: int = 5):
        super().__init__()
        self.score_dim = score_dim
        score_emb_dim = max(8, hidden_dim // 8)
        self.score_proj = nn.ModuleList([nn.Linear(1, score_emb_dim) for _ in range(score_dim)])
        self.missing_emb = nn.Parameter(torch.zeros(score_dim, score_emb_dim))
        self.age_roi_proj = nn.Linear(3, score_emb_dim)
        self.fusion = nn.Sequential(
            nn.Linear(score_emb_dim * (score_dim + 1), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, score_vals, score_missing, visit_ages, visit_rois):
        score_feats = []
        for i in range(self.score_dim):
            v = score_vals[..., i:i + 1]
            m = score_missing[..., i:i + 1]
            emb = self.score_proj[i](v) + m * self.missing_emb[i]
            score_feats.append(emb)
        score_feats = torch.cat(score_feats, dim=-1)
        age_roi = torch.cat([visit_ages, visit_rois], dim=-1)
        age_roi = self.age_roi_proj(age_roi)
        x = torch.cat([score_feats, age_roi], dim=-1)
        x = self.fusion(x)
        return x


class VisitEncoder(nn.Module):
    def __init__(self, hidden_dim: int, time_dim: int):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.GroupNorm(4, 8),
            nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.scalar_encoder = ScalarEncoder(hidden_dim=hidden_dim)
        self.time_encoder = CosineTimeEncoding(out_dim=time_dim, hidden_dim=hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(32 + hidden_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, image, score_vals, score_missing, visit_ages, visit_rois, visit_times):
        img_feat = self.image_encoder(image)
        scalar_feat = self.scalar_encoder(score_vals, score_missing, visit_ages, visit_rois)
        scalar_feat = scalar_feat.squeeze(1)
        time_feat = self.time_encoder(visit_times)
        x = torch.cat([img_feat, scalar_feat, time_feat], dim=-1)
        return self.fusion(x)


class HistorySequenceEncoder(nn.Module):
    """Encode history sequence and output persona token."""

    def __init__(self, hidden_dim: int, time_dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.visit_encoder = VisitEncoder(hidden_dim=hidden_dim, time_dim=time_dim)
        self.time_pos_encoding = CosineTimeEncoding(out_dim=hidden_dim, hidden_dim=hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.future_query_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, history_images, score_vals, score_missing, visit_ages, visit_rois, visit_times, time_gap, history_mask=None):
        B, T = history_images.shape[:2]
        flat_images = history_images.view(B * T, *history_images.shape[2:])
        flat_scores = score_vals.view(B * T, 1, -1)
        flat_missing = score_missing.view(B * T, 1, -1)
        flat_ages = visit_ages.view(B * T, 1, -1)
        flat_rois = visit_rois.view(B * T, 1, -1)
        flat_times = visit_times.reshape(B * T)

        flat_feats = self.visit_encoder(
            flat_images,
            flat_scores,
            flat_missing,
            flat_ages,
            flat_rois,
            flat_times,
        )
        visit_feats = flat_feats.view(B, T, -1)
        time_emb = self.time_pos_encoding(visit_times)
        visit_feats = visit_feats + time_emb

        src_key_padding_mask = (history_mask == 0) if history_mask is not None else None

        last_visit_time = visit_times[:, -1].unsqueeze(1)
        future_time = last_visit_time + time_gap.unsqueeze(1)
        future_pos_emb = self.time_pos_encoding(future_time)
        future_token = self.future_query_token.expand(B, -1, -1) + future_pos_emb
        full_input = torch.cat([visit_feats, future_token], dim=1)

        if src_key_padding_mask is not None:
            future_mask = torch.zeros(B, 1, dtype=torch.bool, device=history_images.device)
            src_key_padding_mask = torch.cat([src_key_padding_mask, future_mask], dim=1)

        output = self.transformer(full_input, src_key_padding_mask=src_key_padding_mask)
        persona = self.output_proj(output[:, -1])
        return persona, output[:, :-1]


class StaticEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_genders: int = 3, num_apoe: int = 4):
        super().__init__()
        self.gender_embedding = nn.Embedding(num_genders, hidden_dim // 2)
        self.apoe_embedding = nn.Embedding(num_apoe, hidden_dim // 2)
        self.age_proj = nn.Linear(1, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, static_cats, static_age):
        gender_feat = self.gender_embedding(static_cats[:, 0])
        apoe_feat = self.apoe_embedding(static_cats[:, 1])
        static_feat = torch.cat([gender_feat, apoe_feat], dim=-1)
        age_feat = self.age_proj(static_age)
        return self.fusion(torch.cat([static_feat, age_feat], dim=-1))


class HistoryEncoder(nn.Module):
    """History Encoder producing fused patient vector c."""

    def __init__(
        self,
        hidden_dim: int,
        time_dim: int,
        fusion_dim: int,
        num_genders: int,
        num_apoe: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.sequence_encoder = HistorySequenceEncoder(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.static_encoder = StaticEncoder(hidden_dim=hidden_dim, num_genders=num_genders, num_apoe=num_apoe)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, fusion_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(self, batch):
        persona, visit_feats = self.sequence_encoder(
            history_images=batch["history_images"],
            score_vals=batch["score_vals"],
            score_missing=batch["score_missing"],
            visit_ages=batch["visit_ages"],
            visit_rois=batch["visit_rois"],
            visit_times=batch["visit_times"],
            time_gap=batch["time_gap"].squeeze(-1),
            history_mask=batch["history_mask"],
        )
        static_feat = self.static_encoder(batch["static_cats"], batch["static_age"])
        mask = batch["history_mask"].unsqueeze(-1)
        visit_feat = (visit_feats * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        fused = self.fusion(torch.cat([static_feat, visit_feat, persona], dim=-1))
        return fused, {"visit_feat": visit_feat, "persona": persona, "static_feat": static_feat}


class ROIRegressionHead(nn.Module):
    def __init__(self, fusion_dim: int, pred_hidden: int, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, pred_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(pred_hidden, out_dim),
        )

    def forward(self, fused):
        return self.net(fused)


class ROIGuidedAttentionBias(nn.Module):
    def __init__(
        self,
        c_dim: int,
        num_roi_types: int = 2,
        diag_base_values: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.num_roi_types = num_roi_types
        if diag_base_values is None:
            diag_base_values = [5.0, 1.5]
        if len(diag_base_values) != num_roi_types:
            raise ValueError(
                "diag_base_values length must match num_roi_types "
                f"({len(diag_base_values)} vs {num_roi_types})."
            )

        init_matrix = torch.zeros(num_roi_types, num_roi_types)
        for i, val in enumerate(diag_base_values):
            init_matrix[i, i] = val
        self.raw_base_bias = nn.Parameter(init_matrix)

        self.dynamic_net = nn.Sequential(
            nn.Linear(c_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, num_roi_types * num_roi_types),
        )
        nn.init.zeros_(self.dynamic_net[-1].weight)
        nn.init.zeros_(self.dynamic_net[-1].bias)

    def forward(
        self,
        roi_masks: torch.Tensor,
        fused: torch.Tensor,
        img_size: int,
        patch_size: int,
        context_tokens: int,
    ) -> torch.Tensor:
        if roi_masks.dim() == 3:
            roi_masks = roi_masks.unsqueeze(1)

        if roi_masks.dim() != 4:
            raise ValueError("roi_masks must have shape [B, C, H, W] or [B, H, W].")

        if roi_masks.size(1) == 1 and self.num_roi_types > 1:
            roi_masks = roi_masks.repeat(1, self.num_roi_types, 1, 1)
        elif roi_masks.size(1) != self.num_roi_types:
            raise ValueError(
                "roi_masks channel count must match num_roi_types "
                f"({roi_masks.size(1)} vs {self.num_roi_types})."
            )

        batch_size = roi_masks.size(0)
        h_patches = img_size // patch_size
        w_patches = img_size // patch_size
        patch_masks = F.adaptive_max_pool2d(roi_masks.float(), (h_patches, w_patches))
        patch_masks = (patch_masks.view(batch_size, self.num_roi_types, -1) > 0.5).float()

        base_bias = F.softplus(self.raw_base_bias)
        raw_scale = self.dynamic_net(fused).view(batch_size, self.num_roi_types, self.num_roi_types)
        dynamic_scale = torch.tanh(raw_scale)
        total_weight = base_bias * (1.0 + dynamic_scale)

        attn_bias = torch.einsum("bui,buv,bvj->bij", patch_masks, total_weight, patch_masks)
        total_tokens = patch_masks.size(2) + context_tokens
        bias = torch.zeros(batch_size, total_tokens, total_tokens, device=roi_masks.device)
        bias[:, :patch_masks.size(2), :patch_masks.size(2)] = attn_bias
        return bias


# ======================= Conditional Rectified Flow =======================

class HCRF2D(BaseModel):
    """Rectified Flow with History Encoder conditioning and ROI-guided options."""

    @staticmethod
    def add_args(parser):
        flow_group = parser.add_argument_group("Conditional Rectified Flow 2D")

        flow_group.add_argument(
            "--rf_backbone", type=str, default="dit_b",
            choices=["dit", "dit_s", "dit_b", "dit_l", "dit_xl"],
            help="DiT backbone size"
        )
        flow_group.add_argument("--rf_dropout", type=float, default=0.0, help="Dropout rate")

        flow_group.add_argument("--rf_dit_patch_size", type=int, default=16)
        flow_group.add_argument("--rf_dit_hidden_size", type=int, default=384)
        flow_group.add_argument("--rf_dit_depth", type=int, default=12)
        flow_group.add_argument("--rf_dit_num_heads", type=int, default=6)

        flow_group.add_argument("--rf_sampling_steps", type=int, default=100)
        flow_group.add_argument("--rf_use_ode_solver", type=str, default="heun",
                               choices=["euler", "heun", "midpoint", "dopri5"])
        flow_group.add_argument("--rf_use_reflow", action="store_true")
        flow_group.add_argument("--rf_reflow_steps", type=int, default=5)
        flow_group.add_argument("--rf_reflow_weight", type=float, default=0.1)
        flow_group.add_argument("--rf_reflow_ratio", type=float, default=0.2)
        flow_group.add_argument("--rf_noise_schedule", type=str, default="logit_normal",
                               choices=["uniform", "logit_normal", "cosine"])
        flow_group.add_argument("--rf_logit_mean", type=float, default=0.0)
        flow_group.add_argument("--rf_logit_std", type=float, default=1.0)

        flow_group.add_argument("--cf_context_tokens", type=int, default=1, help="Number of context tokens for c")

        flow_group.add_argument("--he_time_dim", type=int, default=32)
        flow_group.add_argument("--he_hidden_dim", type=int, default=64)
        flow_group.add_argument("--he_fusion_dim", type=int, default=128)
        flow_group.add_argument("--he_pred_hidden", type=int, default=128)
        flow_group.add_argument("--he_num_heads", type=int, default=4)
        flow_group.add_argument("--he_num_layers", type=int, default=2)
        flow_group.add_argument("--he_max_visits", type=int, default=9)
        flow_group.add_argument("--he_lr", type=float, default=0.0, help="History encoder learning rate (0 means frozen)")
        flow_group.add_argument("--cf_aux_weight", type=float, default=0.1, help="Auxiliary loss weight")

        flow_group.add_argument(
            "--cf_dynamic_roi_diag_base_values",
            type=float,
            nargs="+",
            default=[5.0, 1.5],
            help="Diagonal base bias values for hierarchical ROI attention",
        )
        flow_group.add_argument("--cf_roi_num_types", type=int, default=2, help="Number of ROI types for dynamic attention")
        flow_group.add_argument(
            "--cf_roi_focus_weight",
            type=float,
            default=5.0,
            help="ROI weight multiplier for velocity loss (1.0 disables weighting)",
        )
        flow_group.add_argument("--cf_roi_overlay_alpha", type=float, default=0.35,
                               help="ROI overlay alpha for visualization")

        # Reflow-stage extra losses
        flow_group.add_argument("--cf_reflow_seg_weight", type=float, default=1.0,
                               help="Weight for ROI mask segmentation loss (dice + BCE)")
        flow_group.add_argument("--cf_reflow_seg_dice_weight", type=float, default=1.0,
                               help="Dice weight inside ROI segmentation loss")
        flow_group.add_argument("--cf_reflow_seg_bce_weight", type=float, default=1.0,
                               help="BCE weight inside ROI segmentation loss")
        flow_group.add_argument("--cf_reflow_roi_lpips_weight", type=float, default=0.5,
                               help="Weight for LPIPS on merged ROI mask")
        flow_group.add_argument(
            "--cf_seg_head_weights",
            type=str,
            default="./seg_head_weights.pt",
            help="Optional ROI seg head weights path (missing file is ignored by release code)",
        )
        flow_group.add_argument(
            "--cf_seg_head_mobilenetv4_name",
            type=str,
            default="mobilenetv4_conv_medium",
            help="timm model name for MobileNetV4",
        )
        flow_group.add_argument(
            "--cf_seg_head_lr",
            type=float,
            default=None,
            help="Seg head learning rate override (None=use base lr, 0=freeze)",
        )
        return parser

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vis_batch = None

        input_channels = getattr(args, "cf_input_channels", 3)
        in_channels = 1
        out_channels = 1
        cond_channels = input_channels

        self.out_channels = out_channels

        backbone = getattr(args, "rf_backbone", "dit_b")
        img_size = getattr(args, "img_size", [224, 224])[0]
        context_tokens = getattr(args, "cf_context_tokens", 1)

        if backbone in ["dit", "dit_s"]:
            hidden_size, depth, num_heads = 384, 12, 6
        elif backbone == "dit_b":
            hidden_size, depth, num_heads = 768, 12, 12
        elif backbone == "dit_l":
            hidden_size, depth, num_heads = 1024, 24, 16
        elif backbone == "dit_xl":
            hidden_size, depth, num_heads = 1152, 28, 16
        else:
            hidden_size = getattr(args, "rf_dit_hidden_size", 384)
            depth = getattr(args, "rf_dit_depth", 12)
            num_heads = getattr(args, "rf_dit_num_heads", 6)

        self.velocity_net = DiT2DConditionedWithContext(
            img_size=img_size,
            patch_size=getattr(args, "rf_dit_patch_size", 16),
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=cond_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            dropout=getattr(args, "rf_dropout", 0.0),
            context_tokens=context_tokens,
        )
        self.c_to_spatial = None
        self.token_dim = hidden_size

        # Load pretrained DiT weights if available
        # Release version defaults to no pretrained backbone.
        self._load_pretrained(args, backbone, in_channels)

        # History Encoder
        num_genders = getattr(args, "he_num_genders", 3)
        num_apoe = getattr(args, "he_num_apoe", 4)
        self.history_encoder = HistoryEncoder(
            hidden_dim=getattr(args, "he_hidden_dim", 64),
            time_dim=getattr(args, "he_time_dim", 32),
            fusion_dim=getattr(args, "he_fusion_dim", 128),
            num_genders=num_genders,
            num_apoe=num_apoe,
            num_heads=getattr(args, "he_num_heads", 4),
            num_layers=getattr(args, "he_num_layers", 2),
        )

        self.c_projector = nn.Sequential(
            nn.Linear(getattr(args, "he_fusion_dim", 128), self.token_dim),
            nn.SiLU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.context_tokens = context_tokens
        if self.context_tokens > 1:
            self.c_token_expander = nn.Linear(self.token_dim, self.token_dim * self.context_tokens)
        else:
            self.c_token_expander = None

        self.aux_head = ROIRegressionHead(
            fusion_dim=getattr(args, "he_fusion_dim", 128),
            pred_hidden=getattr(args, "he_pred_hidden", 128),
        )

        self._apply_he_freeze(args)

        self.criterion = nn.MSELoss()
        self.sampling_steps = getattr(args, "rf_sampling_steps", 100)
        self.use_reflow = getattr(args, "rf_use_reflow", False)
        self.reflow_steps = getattr(args, "rf_reflow_steps", 10)
        self.reflow_ratio = getattr(args, "rf_reflow_ratio", 0.5)
        self.reflow_weight = getattr(args, "rf_reflow_weight", 0.1)
        self.noise_schedule = getattr(args, "rf_noise_schedule", "uniform")
        self.logit_mean = getattr(args, "rf_logit_mean", 0.0)
        self.logit_std = getattr(args, "rf_logit_std", 1.0)

        self.dynamic_roi_attention = ROIGuidedAttentionBias(
            c_dim=getattr(args, "he_fusion_dim", 128),
            num_roi_types=getattr(args, "cf_roi_num_types", 2),
            diag_base_values=getattr(args, "cf_dynamic_roi_diag_base_values", [5.0, 1.5]),
        )
        self.roi_focus_weight = getattr(args, "cf_roi_focus_weight", 5.0)
        self.aux_weight = getattr(args, "cf_aux_weight", 0.1)

        # Reflow-stage extra losses
        self.reflow_seg_weight = getattr(args, "cf_reflow_seg_weight", 0.0)
        self.reflow_seg_dice_weight = getattr(args, "cf_reflow_seg_dice_weight", 1.0)
        self.reflow_seg_bce_weight = getattr(args, "cf_reflow_seg_bce_weight", 1.0)
        self.reflow_roi_lpips_weight = getattr(args, "cf_reflow_roi_lpips_weight", 0.0)
        self.roi_seg_head = build_seg_head(
            in_channels=self.out_channels,
            out_channels=2,
            mobilenet_v4_name=getattr(args, "cf_seg_head_mobilenetv4_name", None),
        )
        seg_weights = getattr(args, "cf_seg_head_weights", None)
        if seg_weights:
            if os.path.exists(seg_weights):
                state = torch.load(seg_weights, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.roi_seg_head.load_state_dict(state, strict=False)
        if getattr(args, "cf_seg_head_lr", None) == 0:
            for p in self.roi_seg_head.parameters():
                p.requires_grad = False
        self._lpips_net = None

    def _load_pretrained(self, args, backbone: str, in_channels: int) -> None:
        # Publication release: pretrained backbone loading is disabled by default.
        _ = (args, backbone, in_channels)
        return

    def _apply_he_freeze(self, args) -> None:
        he_lr = getattr(args, "he_lr", 0.0)
        freeze = he_lr <= 0
        if freeze:
            for p in self.history_encoder.parameters():
                p.requires_grad = False

    def get_param_groups(self, args):
        base_params = []
        he_params = []
        seg_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("history_encoder."):
                he_params.append(p)
            elif name.startswith("roi_seg_head."):
                seg_params.append(p)
            else:
                base_params.append(p)
        groups = [{"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay}]
        if he_params:
            groups.append({"params": he_params, "lr": getattr(args, "he_lr", 0.0), "weight_decay": args.weight_decay})
        if seg_params:
            seg_lr = getattr(args, "cf_seg_head_lr", None)
            if seg_lr is None:
                seg_lr = args.lr
            groups.append({"params": seg_params, "lr": seg_lr, "weight_decay": args.weight_decay})
        return groups

    def _sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.noise_schedule == "uniform":
            return torch.rand(batch_size, device=device)
        if self.noise_schedule == "logit_normal":
            u = torch.randn(batch_size, device=device) * self.logit_std + self.logit_mean
            return torch.sigmoid(u)
        if self.noise_schedule == "cosine":
            u = torch.rand(batch_size, device=device)
            return 1.0 - torch.cos(u * math.pi / 2)
        return torch.rand(batch_size, device=device)

    def _prepare_he_batch(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        keys = [
            "history_images",
            "score_vals",
            "score_missing",
            "visit_ages",
            "visit_rois",
            "visit_times",
            "history_mask",
            "static_cats",
            "static_age",
            "time_gap",
        ]
        return {k: batch[k].to(device, non_blocking=True) for k in keys}

    def _build_context_tokens(self, batch: Dict[str, torch.Tensor]):
        device = next(self.history_encoder.parameters()).device
        he_batch = self._prepare_he_batch(batch, device)
        fused, _ = self.history_encoder(he_batch)
        c = self.c_projector(fused)
        if self.context_tokens > 1 and self.c_token_expander is not None:
            expanded = self.c_token_expander(c)
            c_tokens = expanded.view(c.size(0), self.context_tokens, self.token_dim)
            return c_tokens, fused
        return c, fused

    def _build_roi_attention_bias(
        self,
        roi_mask: Optional[torch.Tensor],
        img_size: int,
        patch_size: int,
        context_tokens: int,
        fused: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if roi_mask is None or fused is None:
            return None
        roi_mask_dyn = roi_mask
        if roi_mask_dyn.dim() == 3:
            roi_mask_dyn = roi_mask_dyn.unsqueeze(1)
        if roi_mask_dyn.dim() == 4 and roi_mask_dyn.size(1) != self.dynamic_roi_attention.num_roi_types:
            roi_mask_dyn = roi_mask_dyn.repeat(1, self.dynamic_roi_attention.num_roi_types, 1, 1)
        return self.dynamic_roi_attention(
            roi_mask_dyn,
            fused,
            img_size=img_size,
            patch_size=patch_size,
            context_tokens=context_tokens,
        )

    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        context_tokens: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.velocity_net(
            x_t,
            cond,
            t,
            context_tokens=context_tokens,
            attn_bias=attn_bias,
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cond is None:
            cond = x
        if t is None:
            t = torch.zeros(x.size(0), device=x.device)
        return self._predict_velocity(x, cond, t)

    def predict(self, cond: torch.Tensor, steps: Optional[int] = None, args=None, **kwargs) -> torch.Tensor:
        if steps is None:
            steps = getattr(args, "flow_steps_eval", self.sampling_steps)
        return self.sample(cond, steps=steps, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        steps: int = 100,
        return_trajectory: bool = False,
        solver: Optional[str] = None,
        init_noise: Optional[torch.Tensor] = None,
        context_tokens: Optional[torch.Tensor] = None,
        roi_mask: Optional[torch.Tensor] = None,
        fused: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = cond.device
        batch_size = cond.size(0)
        if solver is None:
            solver = getattr(self.args, "rf_use_ode_solver", "heun")

        if init_noise is not None:
            x = init_noise.clone()
        else:
            x = torch.randn(
                cond.size(0),
                self.out_channels,
                cond.size(2),
                cond.size(3),
                device=cond.device,
            )

        attn_bias = self._build_roi_attention_bias(
            roi_mask,
            img_size=getattr(self.args, "img_size", [224, 224])[0],
            patch_size=getattr(self.args, "rf_dit_patch_size", 16),
            context_tokens=self.context_tokens,
            fused=fused,
        )

        if solver == "dopri5":
            try:
                from torchdiffeq import odeint
            except Exception as e:
                raise ValueError("dopri5 requires torchdiffeq. Install with: pip install torchdiffeq") from e

            t = torch.linspace(0.0, 1.0, steps + 1, device=device)

            def f(t_scalar, x_state):
                t_batch = torch.full((batch_size,), float(t_scalar), device=device)
                return self._predict_velocity(x_state, cond, t_batch, context_tokens, attn_bias)

            traj = odeint(f, x, t, method="dopri5")
            x = traj[-1]
            x = torch.clamp(x, 0.0, 1.0)
            if return_trajectory:
                return [t_i.clone() for t_i in traj]
            return x

        if return_trajectory:
            trajectory = [x.clone()]

        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((batch_size,), i / steps, device=device)
            if solver == "euler":
                v = self._predict_velocity(x, cond, t, context_tokens, attn_bias)
                x = x + v * dt
            elif solver == "heun":
                v1 = self._predict_velocity(x, cond, t, context_tokens, attn_bias)
                x_temp = x + v1 * dt
                t_next = torch.full((batch_size,), (i + 1) / steps, device=device)
                v2 = self._predict_velocity(x_temp, cond, t_next, context_tokens, attn_bias)
                x = x + (v1 + v2) * 0.5 * dt
            elif solver == "midpoint":
                v1 = self._predict_velocity(x, cond, t, context_tokens, attn_bias)
                t_mid = torch.full((batch_size,), (i + 0.5) / steps, device=device)
                x_mid = x + v1 * (dt * 0.5)
                v_mid = self._predict_velocity(x_mid, cond, t_mid, context_tokens, attn_bias)
                x = x + v_mid * dt
            else:
                raise ValueError(f"Unknown solver: {solver}")
            x = torch.clamp(x, -3.0, 3.0)
            if return_trajectory:
                trajectory.append(x.clone())

        x = torch.clamp(x, 0.0, 1.0)
        if return_trajectory:
            return trajectory
        return x

    def _roi_weighted_loss(self, v_pred: torch.Tensor, v_target: torch.Tensor, roi_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if roi_mask is None or self.roi_focus_weight == 1.0:
            return self.criterion(v_pred, v_target)
        if roi_mask.dim() == 3:
            roi_mask = roi_mask.unsqueeze(1)
        weight = torch.ones_like(roi_mask)
        weight = weight + (roi_mask > 0).float() * (self.roi_focus_weight - 1.0)
        diff = (v_pred - v_target) ** 2
        return (diff * weight).mean()

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pred = pred.flatten(2)
        target = target.flatten(2)
        intersection = (pred * target).sum(dim=2)
        denom = pred.sum(dim=2) + target.sum(dim=2)
        dice = (2.0 * intersection + eps) / (denom + eps)
        return 1.0 - dice.mean()

    def _get_lpips_net(self, device: torch.device) -> nn.Module:
        if self._lpips_net is None:
            try:
                import lpips
            except Exception as exc:
                raise RuntimeError("lpips is required for ROI LPIPS loss. Install: pip install lpips") from exc
            self._lpips_net = lpips.LPIPS(net="alex")
            self._lpips_net.eval()
        return self._lpips_net.to(device)

    def _prepare_mri_channel(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pred.size(1) > 1:
            pred_mri = pred.mean(dim=1, keepdim=True)
        else:
            pred_mri = pred
        if target.size(1) > 1:
            target_mri = target.mean(dim=1, keepdim=True)
        else:
            target_mri = target
        return pred_mri.float(), target_mri.float()

    def _compute_reflow_aux_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi_target: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        pred_mri, target_mri = self._prepare_mri_channel(pred, target)

        if roi_target is not None and (self.reflow_seg_weight > 0 or self.reflow_roi_lpips_weight > 0):
            roi_target = torch.nan_to_num(roi_target, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
            roi_logits = self.roi_seg_head(pred_mri)
            roi_probs = torch.sigmoid(roi_logits)
            if self.reflow_seg_weight > 0:
                bce = F.binary_cross_entropy_with_logits(roi_logits, roi_target)
                dice = self._dice_loss(roi_probs, roi_target)
                losses["roi_seg"] = (
                    self.reflow_seg_dice_weight * dice + self.reflow_seg_bce_weight * bce
                ) * self.reflow_seg_weight
            if self.reflow_roi_lpips_weight > 0:
                roi_mask = torch.clamp(roi_target.sum(dim=1, keepdim=True), 0.0, 1.0)
                lpips_net = self._get_lpips_net(pred.device).float()
                with torch.autocast(device_type=pred.device.type, enabled=False):
                    pred_norm = pred_mri.float().repeat(1, 3, 1, 1) * 2.0 - 1.0
                    target_norm = target_mri.float().repeat(1, 3, 1, 1) * 2.0 - 1.0
                    masked_pred = pred_norm * roi_mask
                    masked_target = target_norm * roi_mask
                    lpips_val = lpips_net(masked_pred, masked_target).mean()
                losses["roi_lpips"] = lpips_val * self.reflow_roi_lpips_weight

        return losses

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
        self.train()
        meters = {name: AverageMeter() for name in [
            "loss",
            "flow_loss",
            "reflow_loss",
            "reflow_aux_loss",
            "aux_loss",
            "roi_seg_loss",
            "roi_lpips_loss",
            "ssim",
            "l1",
            "l2",
        ]}

        skipped_batches = 0
        total_batches = 0
        progress = tqdm(loader, desc=f"Train {epoch + 1}", leave=True, dynamic_ncols=True)
        for batch in progress:
            total_batches += 1
            cond = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            roi_mask = batch.get("roi_mask_focus")
            roi_mask = roi_mask.to(device, non_blocking=True) if roi_mask is not None else None

            batch_size = cond.size(0)
            optimizer.zero_grad(set_to_none=True)

            z_0 = torch.randn_like(target)
            t = self._sample_timestep(batch_size, device)
            t_expand = t.view(-1, 1, 1, 1)
            x_t = (1.0 - t_expand) * z_0 + t_expand * target
            v_target = target - z_0

            context_tokens, fused = self._build_context_tokens(batch)
            context_tokens = context_tokens.to(device)

            attn_bias = self._build_roi_attention_bias(
                roi_mask,
                img_size=getattr(args, "img_size", [224, 224])[0],
                patch_size=getattr(args, "rf_dit_patch_size", 16),
                context_tokens=self.context_tokens,
                fused=fused,
            )

            if scaler is not None:
                with torch.autocast(device_type=device.type, enabled=True):
                    v_pred = self._predict_velocity(x_t, cond, t, context_tokens, attn_bias)
                    flow_loss = self._roi_weighted_loss(v_pred, v_target, roi_mask)
                    loss = flow_loss

                    aux_pred = self.aux_head(fused)
                    aux_target = batch["target_roi"].to(device)
                    aux_loss = self.criterion(aux_pred, aux_target)
                    loss = loss + self.aux_weight * aux_loss

                    apply_reflow = self.use_reflow and self.reflow_weight > 0
                    if apply_reflow and torch.rand(1).item() < self.reflow_ratio:
                        with torch.no_grad():
                            z_0_hat = torch.randn_like(target)
                            x_1_hat = self.sample(
                                cond,
                                steps=self.reflow_steps,
                                init_noise=z_0_hat,
                                context_tokens=context_tokens,
                                roi_mask=roi_mask,
                                fused=fused,
                            )
                        t_hat = self._sample_timestep(batch_size, device)
                        t_hat_expand = t_hat.view(-1, 1, 1, 1)
                        x_t_hat = (1.0 - t_hat_expand) * z_0_hat + t_hat_expand * x_1_hat
                        v_target_hat = x_1_hat - z_0_hat
                        v_pred_hat = self._predict_velocity(x_t_hat, cond, t_hat, context_tokens, attn_bias)
                        reflow_loss = self._roi_weighted_loss(v_pred_hat, v_target_hat, roi_mask)
                        roi_target = batch.get("roi_mask_target")
                        if roi_target is not None:
                            roi_target = roi_target.to(device, non_blocking=True)
                        extra_losses = self._compute_reflow_aux_losses(
                            x_1_hat,
                            target,
                            roi_target,
                        )
                        reflow_aux = sum(extra_losses.values()) if extra_losses else torch.tensor(0.0, device=device)
                        loss = loss + self.reflow_weight * reflow_loss + reflow_aux
                    else:
                        reflow_loss = torch.tensor(0.0, device=device)
                        reflow_aux = torch.tensor(0.0, device=device)
                        extra_losses = {}

                if not torch.isfinite(loss).all():
                    skipped_batches += 1
                    if skipped_batches <= 5:
                        progress.write("[WARN] Non-finite loss detected; skipping batch update.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                if hasattr(args, "grad_clip") and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                v_pred = self._predict_velocity(x_t, cond, t, context_tokens, attn_bias)
                flow_loss = self._roi_weighted_loss(v_pred, v_target, roi_mask)
                loss = flow_loss

                aux_pred = self.aux_head(fused)
                aux_target = batch["target_roi"].to(device)
                aux_loss = self.criterion(aux_pred, aux_target)
                loss = loss + self.aux_weight * aux_loss

                apply_reflow = self.use_reflow and self.reflow_weight > 0
                if apply_reflow and torch.rand(1).item() < self.reflow_ratio:
                    with torch.no_grad():
                        z_0_hat = torch.randn_like(target)
                        x_1_hat = self.sample(
                            cond,
                            steps=self.reflow_steps,
                            init_noise=z_0_hat,
                            context_tokens=context_tokens,
                            roi_mask=roi_mask,
                            fused=fused,
                        )
                    t_hat = self._sample_timestep(batch_size, device)
                    t_hat_expand = t_hat.view(-1, 1, 1, 1)
                    x_t_hat = (1.0 - t_hat_expand) * z_0_hat + t_hat_expand * x_1_hat
                    v_target_hat = x_1_hat - z_0_hat
                    v_pred_hat = self._predict_velocity(x_t_hat, cond, t_hat, context_tokens, attn_bias)
                    reflow_loss = self._roi_weighted_loss(v_pred_hat, v_target_hat, roi_mask)
                    roi_target = batch.get("roi_mask_target")
                    if roi_target is not None:
                        roi_target = roi_target.to(device, non_blocking=True)
                    extra_losses = self._compute_reflow_aux_losses(
                        x_1_hat,
                        target,
                        roi_target,
                    )
                    reflow_aux = sum(extra_losses.values()) if extra_losses else torch.tensor(0.0, device=device)
                    loss = loss + self.reflow_weight * reflow_loss + reflow_aux
                else:
                    reflow_loss = torch.tensor(0.0, device=device)
                    reflow_aux = torch.tensor(0.0, device=device)
                    extra_losses = {}

                if not torch.isfinite(loss).all():
                    skipped_batches += 1
                    if skipped_batches <= 5:
                        progress.write("[WARN] Non-finite loss detected; skipping batch update.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                if hasattr(args, "grad_clip") and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), args.grad_clip)
                optimizer.step()

            meters["loss"].update(loss.item(), batch_size)
            meters["flow_loss"].update(flow_loss.item(), batch_size)
            meters["reflow_loss"].update(reflow_loss.item(), batch_size)
            meters["reflow_aux_loss"].update(reflow_aux.item(), batch_size)
            meters["aux_loss"].update(aux_loss.item(), batch_size)
            if "roi_seg" in extra_losses:
                meters["roi_seg_loss"].update(extra_losses["roi_seg"].item(), batch_size)
            if "roi_lpips" in extra_losses:
                meters["roi_lpips_loss"].update(extra_losses["roi_lpips"].item(), batch_size)
            progress.set_postfix(loss=f"{meters['loss'].avg:.4e}", flow=f"{meters['flow_loss'].avg:.4e}")

        if skipped_batches == total_batches and total_batches > 0:
            raise RuntimeError("All batches skipped due to non-finite loss. Aborting epoch.")

        return meters

    @torch.no_grad()
    def evaluate_one_epoch(self, loader, device, args, compute_ssim: bool = True) -> Dict[str, AverageMeter]:
        self.eval()
        torch.cuda.empty_cache()
        meters = {name: AverageMeter() for name in ["loss", "ssim", "l1", "l2", "aux_loss"]}
        steps = getattr(args, "flow_steps_eval", self.sampling_steps)

        progress = tqdm(loader, desc="Eval", leave=True, dynamic_ncols=True)
        for batch_idx, batch in enumerate(progress):
            cond = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            roi_mask = batch.get("roi_mask_focus")
            roi_mask = roi_mask.to(device, non_blocking=True) if roi_mask is not None else None

            context_tokens, fused = self._build_context_tokens(batch)
            context_tokens = context_tokens.to(device)

            pred = self.sample(
                cond,
                steps=steps,
                context_tokens=context_tokens,
                roi_mask=roi_mask,
                fused=fused,
            )

            if batch_idx % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            loss = self.criterion(pred, target)
            aux_pred = self.aux_head(fused)
            aux_target = batch["target_roi"].to(device)
            aux_loss = self.criterion(aux_pred, aux_target)
            if compute_ssim:
                metrics = compute_batch_metrics_2d(pred, target, args.ssim_data_range)
            else:
                with torch.no_grad():
                    l1_val = float(torch.abs(pred - target).mean().item())
                    l2_val = float(((pred - target) ** 2).mean().item())
                metrics = {"ssim": 0.0, "l1": l1_val, "l2": l2_val}

            meters["loss"].update(loss.item(), cond.size(0))
            meters["aux_loss"].update(aux_loss.item(), cond.size(0))
            for key in ["ssim", "l1", "l2"]:
                meters[key].update(metrics[key], cond.size(0))

            ssim_str = f"{meters['ssim'].avg:.3f}" if compute_ssim else "-"
            progress.set_postfix(
                loss=f"{meters['loss'].avg:.4e}",
                ssim=ssim_str,
                aux=f"{meters['aux_loss'].avg:.4e}",
            )

        return meters

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
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            mem_str = f" | Peak Mem: {peak_mem:.2f}G"
        else:
            mem_str = ""

        if val_metrics is None:
            logger.info(
                "Epoch %4d/%4d | Time: %.2fs | LR: %.3e | "
                "Train Loss: %.6e | Flow Loss: %.6e | Reflow: %.6e | Aux: %.6e%s",
                epoch + 1,
                args.epochs,
                time_tracker["epoch"].val,
                current_lr,
                train_metrics["loss"].avg,
                train_metrics["flow_loss"].avg,
                train_metrics.get("reflow_loss", AverageMeter()).avg,
                train_metrics.get("aux_loss", AverageMeter()).avg,
                mem_str,
            )
        else:
            ssim_val = val_metrics["ssim"].avg
            ssim_str = f"{ssim_val * 100.0:.2f}%" if ssim_val > 0 else "-"
            logger.info(
                "Epoch %4d/%4d | Time: %.2fs | LR: %.3e | "
                "Loss: %.6e (train) / %.6e (val) | "
                "SSIM: %s | L1: %.6e | L2: %.6e%s",
                epoch + 1,
                args.epochs,
                time_tracker["epoch"].val,
                current_lr,
                train_metrics["loss"].avg,
                val_metrics["loss"].avg,
                ssim_str,
                val_metrics["l1"].avg,
                val_metrics["l2"].avg,
                mem_str,
            )

        if writer is None:
            return

        writer.add_scalar("train/loss", train_metrics["loss"].avg, epoch)
        writer.add_scalar("train/flow_loss", train_metrics["flow_loss"].avg, epoch)
        if "reflow_loss" in train_metrics:
            writer.add_scalar("train/reflow_loss", train_metrics["reflow_loss"].avg, epoch)
        if "aux_loss" in train_metrics:
            writer.add_scalar("train/aux_loss", train_metrics["aux_loss"].avg, epoch)

        if val_metrics is not None:
            writer.add_scalar("val/loss", val_metrics["loss"].avg, epoch)
            if val_metrics["ssim"].avg > 0:
                writer.add_scalar("val/ssim", val_metrics["ssim"].avg, epoch)
            writer.add_scalar("val/l1", val_metrics["l1"].avg, epoch)
            writer.add_scalar("val/l2", val_metrics["l2"].avg, epoch)
            if "aux_loss" in val_metrics:
                writer.add_scalar("val/aux_loss", val_metrics["aux_loss"].avg, epoch)
            writer.add_scalars(
                "loss_comparison",
                {"train": train_metrics["loss"].avg, "val": val_metrics["loss"].avg},
                epoch,
            )

        writer.add_scalar("learning_rate", current_lr, epoch)

        if hasattr(self, "vis_batch") and self.vis_batch is not None:
            device = next(self.parameters()).device
            fig = self.get_vis_figure(epoch, device)
            if fig is not None:
                writer.add_figure("flow/prediction", fig, epoch)

    def get_vis_figure(self, epoch: int, device: torch.device):
        if not hasattr(self, "vis_batch") or self.vis_batch is None:
            return None

        self.eval()
        with torch.no_grad():
            cond = self.vis_batch["input"].to(device)
            target = self.vis_batch["target"].to(device)
            roi_in = self.vis_batch.get("roi_mask_input")
            roi_tg = self.vis_batch.get("roi_mask_target")
            roi_in = roi_in.to(device) if roi_in is not None else None
            roi_tg = roi_tg.to(device) if roi_tg is not None else None

            context_tokens, fused = self._build_context_tokens(self.vis_batch)
            context_tokens = context_tokens.to(device)

            pred = self.sample(
                cond,
                steps=self.sampling_steps,
                context_tokens=context_tokens,
                roi_mask=roi_tg,
                fused=fused,
            )

            x_base = self.vis_batch.get("input_mri", cond).to(device)
            y_base = self.vis_batch.get("target_mri", target).to(device)

        if roi_in is None or roi_tg is None:
            return None
        return make_slice_figure_with_roi(
            x_base,
            y_base,
            pred,
            roi_in,
            roi_tg,
            num_samples=3,
            title=f"Epoch {epoch + 1}",
            alpha=getattr(self.args, "cf_roi_overlay_alpha", 0.35),
        )


HCRFModel = HCRF2D
