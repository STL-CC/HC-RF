"""Model registry for release version.

Only the HC-RF model is retained for the final paper release.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import BaseModel
from .hc_rf import HCRF2D


MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "hc_rf": HCRF2D,
    "conditional_rectified_flow": HCRF2D,
}


def get_model_class(name: str) -> Type[BaseModel]:
    """Return model class by model name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


def add_model_specific_args(parser, model_name: str):
    """Dynamically inject model-specific CLI arguments."""
    model_cls = get_model_class(model_name)
    return model_cls.add_args(parser)


def add_model_args(parser, model_name: str):
    """Backward-compatible wrapper."""
    return add_model_specific_args(parser, model_name)
