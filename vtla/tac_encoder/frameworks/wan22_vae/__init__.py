"""Wan2.2 tactile VAE package."""

from .config import checkpoint_kwargs
from .model import Wan22VAEBackbone


CHECKPOINT_PREFIXES = Wan22VAEBackbone.checkpoint_prefixes

__all__ = ["CHECKPOINT_PREFIXES", "Wan22VAEBackbone", "checkpoint_kwargs"]
