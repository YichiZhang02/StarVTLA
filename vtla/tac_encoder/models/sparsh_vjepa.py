"""Encoder-only Sparsh V-JEPA adapter for downstream feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from torch import Tensor, nn

from .base import EncodedFeatures, FeatureBackbone
from .checkpoint import load_filtered_state, read_state_dict


class _PatchEmbed3D(nn.Module):
    def __init__(self, num_frames: int, tubelet_size: int, image_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.num_patches = (num_frames // tubelet_size) * (image_size // patch_size) ** 2
        self.proj = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, video: Tensor) -> Tensor:
        return self.proj(video).flatten(2).transpose(1, 2)


class _SinusoidalPositionEmbedding(nn.Module):
    """Sparsh's fixed 3D sinusoidal embedding with explicit grid interpolation."""

    def __init__(
        self,
        embed_dim: int,
        source_grid: tuple[int, int, int],
        target_grid: tuple[int, int, int],
    ) -> None:
        super().__init__()
        bands = math.ceil(embed_dim / (2 * len(source_grid)))
        frequencies = torch.linspace(0, 1.0, steps=bands + 1)[:-1]
        self.register_buffer(
            "frequency_bands",
            torch.stack([10000.0**-frequencies for _ in source_grid]),
        )
        self.embed_dim = int(embed_dim)
        self.source_grid = tuple(int(value) for value in source_grid)
        self.target_grid = tuple(int(value) for value in target_grid)

    def forward(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        axes = [torch.arange(size, device=device, dtype=torch.float32) for size in self.source_grid]
        coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        features = coordinates[..., None] * self.frequency_bands.to(device=device)
        position = torch.cat([features.sin(), features.cos()], dim=-1)
        position = position.flatten(-2, -1)[..., : self.embed_dim]
        position = position.reshape(1, *self.source_grid, self.embed_dim).permute(0, 4, 1, 2, 3)
        if self.source_grid != self.target_grid:
            position = F.interpolate(
                position,
                size=self.target_grid,
                mode="trilinear",
                align_corners=False,
            )
        return position.permute(0, 2, 3, 4, 1).reshape(1, -1, self.embed_dim).to(dtype=dtype)


class _SparshEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_frames: int,
        tubelet_size: int,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        position_source_grid: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed3D(num_frames, tubelet_size, image_size, patch_size, embed_dim)
        target_grid = (num_frames // tubelet_size, image_size // patch_size, image_size // patch_size)
        self.pos_embed = _SinusoidalPositionEmbedding(
            embed_dim,
            source_grid=position_source_grid,
            target_grid=target_grid,
        )
        self.register_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    qkv_bias=True,
                    init_values=1.0,
                    norm_layer=lambda dim, **kwargs: nn.LayerNorm(dim, eps=1e-6, **kwargs),
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        nn.init.normal_(self.register_tokens, std=1e-6)

    def forward_tokens(self, tokens: Tensor) -> Tensor:
        registers = self.register_tokens.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([registers, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)[:, 1:]

    def add_position_embedding(self, tokens: Tensor) -> Tensor:
        return tokens + self.pos_embed(device=tokens.device, dtype=tokens.dtype)


class SparshVJEPABackbone(FeatureBackbone):
    model_id = "sparsh_vjepa"

    def __init__(
        self,
        *,
        num_frames: int = 4,
        image_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        pretrained_path: str = "",
        checkpoint_source_grid: tuple[int, int, int] = (2, 20, 15),
    ) -> None:
        super().__init__()
        if num_frames % tubelet_size or image_size % patch_size:
            raise ValueError("num_frames/image_size must be divisible by tubelet/patch size")
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.feature_dim = int(embed_dim)
        self.temporal_units = self.num_frames // self.tubelet_size
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.temporal_units * self.grid_size**2
        self.encoder = _SparshEncoder(
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            position_source_grid=checkpoint_source_grid,
        )
        self.load_report = None
        if pretrained_path:
            self.load_report = self.load_pretrained(pretrained_path, checkpoint_source_grid)

    def _flatten(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        if images.ndim != 6:
            raise ValueError(f"expected [B,S,T,C,H,W], got {tuple(images.shape)}")
        b, sensors, frames, channels, height, width = images.shape
        if (frames, channels, height, width) != (
            self.num_frames,
            3,
            self.image_size,
            self.image_size,
        ):
            raise ValueError("input shape does not match Sparsh configuration")
        return (
            images.reshape(b * sensors, frames, channels, height, width).permute(0, 2, 1, 3, 4),
            (b, sensors),
        )

    def encode_features(self, images: Tensor) -> EncodedFeatures:
        video, (b, sensors) = self._flatten(images)
        tokens = self.encoder.forward_tokens(
            self.encoder.add_position_embedding(self.encoder.patch_embed(video))
        )
        spatial = tokens.reshape(
            b,
            sensors,
            self.temporal_units,
            self.grid_size,
            self.grid_size,
            self.feature_dim,
        )
        global_tokens = tokens.mean(dim=1).reshape(b, sensors, 1, self.feature_dim)
        return EncodedFeatures(
            global_tokens=global_tokens,
            spatial_grid=spatial,
            global_time_ids=torch.full((1,), -1, device=images.device, dtype=torch.long),
        )

    def load_pretrained(
        self,
        path: str | Path,
        source_grid: tuple[int, int, int] = (2, 20, 15),
    ) -> dict[str, Any]:
        raw = {
            key.removeprefix("module."): value for key, value in read_state_dict(path).items()
        }
        state = {}
        prefix_groups = (
            ("model.target_encoder.", "target_encoder."),
            ("model.context_encoder.", "context_encoder."),
            ("encoder.",),
        )
        prefixes = next(
            (group for group in prefix_groups if any(key.startswith(group) for key in raw)),
            (),
        )
        for key, value in raw.items():
            prefix = next((candidate for candidate in prefixes if key.startswith(candidate)), None)
            if prefix is not None:
                state["encoder." + key[len(prefix) :]] = value
        if not state:
            encoder_roots = ("blocks.", "norm.", "patch_embed.", "pos_embed.", "register_tokens")
            if any(key.startswith(encoder_roots) for key in raw):
                state = {"encoder." + key: value for key, value in raw.items()}
            else:
                state = raw
        report = load_filtered_state(self, state, source=str(path))
        required_missing = [key for key in report["missing_keys"] if key.startswith("encoder.")]
        required_mismatch = [key for key in report["shape_mismatch"] if key.startswith("encoder.")]
        if required_missing or required_mismatch:
            raise ValueError(
                "Sparsh encoder checkpoint is incompatible: "
                f"missing={required_missing[:8]}, shape_mismatch={required_mismatch[:8]}"
            )
        return report
