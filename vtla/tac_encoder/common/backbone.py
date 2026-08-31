"""Common model protocol and structured feature outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ReconstructionOutput:
    loss: Tensor
    reconstruction: Tensor
    mask: Tensor


@dataclass
class FeatureTokens:
    tokens: Tensor
    sensor_ids: Tensor
    time_ids: Tensor
    token_mask: Tensor


@dataclass
class EncodedFeatures:
    """Full features before downstream pooling.

    ``spatial_grid`` is ``[B,S,T',H',W',D]``. ``global_tokens`` is
    ``[B,S,G,D]`` and ``global_time_ids`` identifies its temporal unit; ``-1``
    denotes a window-global token.
    """

    global_tokens: Tensor
    spatial_grid: Tensor
    global_time_ids: Tensor
    valid: Tensor | None = None
    interleave_global: bool = False


class FeatureBackbone(nn.Module, ABC):
    model_id: str
    feature_dim: int
    patch_size: int
    tubelet_size: int
    num_frames: int

    @abstractmethod
    def encode_features(self, images: Tensor) -> EncodedFeatures:
        raise NotImplementedError

    def extract_pooled_features(self, images: Tensor, pool_size: int = 3) -> FeatureTokens:
        from .pooling import pool_encoded_features

        return pool_encoded_features(self.encode_features(images), pool_size=pool_size)

    @abstractmethod
    def pooled_tokens_per_sensor(self, pool_size: int) -> int:
        """Return the fixed downstream token count produced for one sensor."""
        raise NotImplementedError


class ReconstructionBackbone(FeatureBackbone, ABC):
    @abstractmethod
    def forward_reconstruction(self, images: Tensor, mask_ratio: float) -> ReconstructionOutput:
        raise NotImplementedError

    def forward(self, images: Tensor, mask_ratio: float) -> ReconstructionOutput:
        """DDP-compatible alias for the reconstruction training path."""
        return self.forward_reconstruction(images, mask_ratio)

    @abstractmethod
    def patchify(self, images: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def unpatchify(self, patches: Tensor) -> Tensor:
        raise NotImplementedError
