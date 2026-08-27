"""Construction and checkpoint loading for supported tactile backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtla.tac_encoder.config import SUPPORTED_MODEL_IDS

from .anytouch1 import AnyTouch1Backbone
from .anytouch2 import AnyTouch2Backbone
from .base import FeatureBackbone
from .sparsh_vjepa import SparshVJEPABackbone


MODEL_REGISTRY = {
    "anytouch1": AnyTouch1Backbone,
    "anytouch2": AnyTouch2Backbone,
    "sparsh_vjepa": SparshVJEPABackbone,
}


def build_backbone(model_id: str, **kwargs: Any) -> FeatureBackbone:
    try:
        model_class = MODEL_REGISTRY[model_id]
    except KeyError as error:
        raise ValueError(f"Unknown model_id {model_id!r}; expected one of {SUPPORTED_MODEL_IDS}") from error
    return model_class(**kwargs)


def load_backbone_checkpoint(model: FeatureBackbone, path: str | Path) -> dict[str, Any]:
    if hasattr(model, "load_pretrained"):
        report = model.load_pretrained(path)
        if isinstance(report, tuple):
            missing, unexpected = report
            return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}
        return report
    raise TypeError(f"{type(model).__name__} does not support external checkpoints")
