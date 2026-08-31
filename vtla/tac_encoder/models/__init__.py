"""Compatibility exports for the pre-package tactile encoder API."""

from ..common.backbone import (
    EncodedFeatures,
    FeatureTokens,
    ReconstructionBackbone,
    ReconstructionOutput,
)
from ..registry import build_backbone, load_backbone_checkpoint

__all__ = [
    "EncodedFeatures",
    "FeatureTokens",
    "ReconstructionBackbone",
    "ReconstructionOutput",
    "build_backbone",
    "load_backbone_checkpoint",
]
