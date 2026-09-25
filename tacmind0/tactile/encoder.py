"""Offline reconstruction of the v6 encoder; no world-model runtime required."""

from pathlib import Path

import torch
from torch import nn
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from .tactile_sequence import sequence_patch_coordinates, wrap_vit_tactile_sequence


class _Backbone(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.model = Dinov2WithRegistersModel(Dinov2WithRegistersConfig(**config))


def build_encoder(backbone_config: dict) -> nn.Module:
    if (
        backbone_config.get("patch_size") != 14
        or backbone_config.get("hidden_size") != 384
    ):
        raise ValueError("v6 requires patch_size=14 and hidden_size=384")
    return wrap_vit_tactile_sequence(
        _Backbone(backbone_config),
        frame_size=42,
        patch_size=14,
        history_size=8,
        time_stride=5,
        canonicalize_right=True,
        causal_attention=True,
    )


class FrozenTactileEncoder(nn.Module):
    def __init__(self, backbone_config: dict, trainable: bool = False) -> None:
        super().__init__()
        self.encoder = build_encoder(backbone_config)
        self.trainable = trainable
        self.requires_grad_(trainable)
        self.train(trainable)

    def set_trainable(self, trainable: bool) -> None:
        self.trainable = bool(trainable)
        self.requires_grad_(self.trainable)
        self.train(self.trainable)

    def rebuild_position_buffers(self) -> None:
        """Restore deterministic buffers omitted by HF meta-device checkpoint loading."""
        encoder = self.encoder
        device = next(self.parameters()).device
        coordinates, finger_ids, slot_ids = sequence_patch_coordinates(
            frame_size=42,
            patch_size=14,
            history_size=8,
            time_stride=5,
            canonicalize_right=True,
        )
        prefix = encoder.prefix_length
        coordinates = torch.cat(
            (torch.zeros((prefix, 3), dtype=torch.long), coordinates[1:])
        )
        mask = torch.zeros(len(coordinates), len(coordinates), dtype=torch.bool)
        mask[:prefix, :] = True
        mask[prefix:, prefix:] = slot_ids.unsqueeze(0) <= slot_ids.unsqueeze(1)
        encoder.coordinates = coordinates.to(device)
        encoder.finger_ids = finger_ids.to(device)
        encoder.slot_ids = slot_ids.to(device)
        encoder.causal_mask = mask.to(device)
        for attention in encoder._attention_layers():
            attention_device = next(attention.parameters()).device
            attention.coordinates = coordinates.to(attention_device)
            attention.causal_mask = mask.to(attention_device)

    def train(self, mode: bool = True) -> "FrozenTactileEncoder":
        super().train(mode if self.trainable else False)
        return self

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 6 or tuple(pixels.shape[1:]) != (2, 8, 3, 42, 42):
            raise ValueError("tactile_pixel_values must be (B,2,8,3,42,42)")
        if not torch.is_floating_point(pixels) or not torch.isfinite(pixels).all():
            raise ValueError(
                "tactile_pixel_values must be finite floating-point BGR [0,1]"
            )
        if pixels.min() < 0 or pixels.max() > 1:
            raise ValueError("tactile_pixel_values must be in [0,1]")
        parameter = next(self.parameters())
        pixels = pixels.to(device=parameter.device, dtype=parameter.dtype)
        return self.encoder(pixels).last_hidden_state[:, 0]

    def load_world_model(self, path: str | Path) -> dict:
        state = torch.load(path, map_location="cpu", weights_only=True)
        prefix = "jepa.encoder."
        selected = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not selected:
            raise ValueError("Checkpoint has no jepa.encoder parameters")
        self.encoder.load_state_dict(selected, strict=True)
        self.set_trainable(self.trainable)
        return {
            "loaded_keys": sorted(selected),
            "excluded_keys": sorted(key for key in state if not key.startswith(prefix)),
        }


class TactileFiLM(nn.Module):
    def __init__(self, vision_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(384, 384), nn.SiLU(), nn.Linear(384, 2 * vision_dim)
        )
        self.reset_identity()

    def reset_identity(self) -> None:
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, vision: torch.Tensor, tactile: torch.Tensor, images_per_sample: int = 2
    ) -> torch.Tensor:
        if vision.shape[0] != tactile.shape[0] * images_per_sample:
            raise ValueError(
                "Current images must contain exactly two ordered views per sample"
            )
        gamma, beta = self.net(tactile.to(self.net[0].weight.dtype)).chunk(2, dim=-1)
        gamma = (
            gamma.repeat_interleave(images_per_sample, 0).unsqueeze(1).to(vision.dtype)
        )
        beta = (
            beta.repeat_interleave(images_per_sample, 0).unsqueeze(1).to(vision.dtype)
        )
        return (1 + gamma) * vision + beta
