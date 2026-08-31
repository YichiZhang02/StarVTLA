"""Compatibility wrapper for :mod:`vtla.tac_encoder.registry`."""

from ..registry import MODEL_REGISTRY, build_backbone, load_backbone_checkpoint

__all__ = ["MODEL_REGISTRY", "build_backbone", "load_backbone_checkpoint"]
