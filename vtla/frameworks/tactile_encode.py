#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared pooled tactile-backbone token builder used by all VTLA policies.

Only checkpoints produced by ``vtla.tac_encoder.train`` are supported. Each tactile
sensor is encoded independently inside one batched backbone call, then its spatial
features are reduced with fixed ``3x3`` adaptive average pooling.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vtla.tac_encoder.inference import TactileBackboneFeatureExtractor


class TactileEncoder(nn.Module):
    """Unified tactile backbone plus a projection into the policy token space."""

    def __init__(self, config, output_dim: int):
        super().__init__()
        self.tactile_keys = list(config.tactile_encoder_keys())
        if not self.tactile_keys:
            raise ValueError(
                "TactileEncoder requires tactile_mode='encode' with non-empty tactile_keys."
            )
        self.num_frames = int(getattr(config, "tactile_num_frames", 1))
        self.pool_size = int(getattr(config, "tactile_pool_size", 3))
        self.extractor = TactileBackboneFeatureExtractor.from_pretrained(
            config.tactile_encoder_path,
            freeze=config.freeze_tactile_encoder,
            pool_size=self.pool_size,
        )
        self.image_size = self.extractor.image_size
        if self.num_frames != self.extractor.num_frames:
            raise ValueError(
                f"Tactile checkpoint requires tactile_num_frames={self.extractor.num_frames}, "
                f"got {self.num_frames}."
            )
        if self.num_frames <= 1:
            raise ValueError("Unified tactile checkpoints require tactile_num_frames > 1.")
        self.output_dim = int(output_dim)
        self.proj = nn.Linear(self.extractor.feature_dim, self.output_dim)
        if self.extractor.compute_dtype is not None:
            self.proj.to(dtype=self.extractor.compute_dtype)

    @property
    def feature_dim(self) -> int:
        return self.extractor.feature_dim

    @property
    def num_tokens(self) -> int:
        """Total pooled tokens for all configured sensors and the full window."""
        return len(self.tactile_keys) * self.extractor.tokens_per_sensor

    @property
    def total_tokens(self) -> int:
        return self.num_tokens

    def _missing_keys(self, batch: dict[str, Tensor]) -> list[str]:
        return [k for k in self.tactile_keys if k not in batch]

    def forward_flat(self, batch: dict[str, Tensor]) -> Tensor:
        """Return ``[B, all_sensor_window_tokens, output_dim]``."""
        return self.forward(batch)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        missing = self._missing_keys(batch)
        if missing:
            raise ValueError(
                f"tactile_mode='encode' expected tactile keys missing from the batch: {missing}. "
                f"Batch keys: {list(batch.keys())}"
            )

        device = self.proj.weight.device
        imgs = []
        for key in self.tactile_keys:
            img = batch[key]
            if img.dim() != 5:
                raise ValueError(
                    "Unified tactile checkpoints expect each tactile key as [B,T,C,H,W], "
                    f"got {tuple(img.shape)} for {key!r}."
                )
            if img.shape[2] != 3:
                raise ValueError(
                    f"Unified tactile checkpoints require 3 channels, got {img.shape[2]} "
                    f"for {key!r}."
                )
            if img.device != device:
                img = img.to(device, non_blocking=True)
            if img.dtype == torch.uint8:
                img = img.float().div_(255.0)
            elif not img.is_floating_point():
                raise TypeError(
                    f"Tactile input {key!r} must be uint8 or floating point, got {img.dtype}."
                )
            if tuple(img.shape[-2:]) != (self.image_size, self.image_size):
                batch_size, frames = img.shape[:2]
                img = F.interpolate(
                    img.flatten(0, 1).float(),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).unflatten(0, (batch_size, frames))
            imgs.append(img)

        sample = imgs[0]
        stacked = torch.stack(imgs, dim=1)
        return self.proj(self.extractor(stacked).tokens)
