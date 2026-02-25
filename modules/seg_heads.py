"""Segmentation head implementation for HC-RF release.

This release keeps only the MobileNetV4 segmentation head used in the paper.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _LRASPPHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.aspp_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.aspp_conv(x)
        gate = self.image_pool(x)
        return self.project(feat * gate)


class MobileNetV4SegHead(nn.Module):
    """MobileNetV4 backbone with a lightweight LR-ASPP decoder."""

    def __init__(
        self,
        model_name: str,
        in_channels: int = 1,
        out_channels: int = 2,
        pretrained: bool = False,
        low_level_index: int = 1,
        high_level_index: int = -1,
    ) -> None:
        super().__init__()
        try:
            import timm
        except Exception as exc:
            raise RuntimeError("timm is required for MobileNetV4 segmentation head") from exc

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(low_level_index, high_level_index),
            in_chans=in_channels,
        )
        low_level_channels = self.backbone.feature_info.channels()[0]
        high_level_channels = self.backbone.feature_info.channels()[-1]

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
        )
        self.high_level_head = _LRASPPHead(high_level_channels, 128)
        self.fuse = nn.Sequential(
            nn.Conv2d(128 + 24, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        feats = self.backbone(x)
        low_level, high_level = feats[0], feats[-1]

        high = self.high_level_head(high_level)
        high = F.interpolate(high, size=low_level.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_level_proj(low_level)
        fused = torch.cat([high, low], dim=1)
        out = self.fuse(fused)
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


def build_seg_head(
    in_channels: int = 1,
    out_channels: int = 2,
    mobilenet_v4_name: str | None = None,
) -> nn.Module:
    if not mobilenet_v4_name:
        raise ValueError("mobilenet_v4_name must be provided")
    return MobileNetV4SegHead(
        model_name=mobilenet_v4_name,
        in_channels=in_channels,
        out_channels=out_channels,
        pretrained=False,
    )
