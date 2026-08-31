"""Shared temporal patchification and pixel reconstruction decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from timm.models.vision_transformer import Block
from torch import Tensor, nn


def random_masking(tokens: Tensor, mask_ratio: float) -> tuple[Tensor, Tensor, Tensor]:
    if not 0 <= mask_ratio < 1:
        raise ValueError("mask_ratio must be in [0, 1)")
    batch, length, dim = tokens.shape
    keep = max(1, int(length * (1 - mask_ratio)))
    noise = torch.rand(batch, length, device=tokens.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :keep]
    visible = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, dim))
    mask = torch.ones(batch, length, device=tokens.device)
    mask[:, :keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return visible, mask, ids_restore


def patchify_video(images: Tensor, patch_size: int, tubelet_size: int) -> Tensor:
    """Patchify ``[B,C,T,H,W]`` into time-major tube tokens."""
    if images.ndim != 5:
        raise ValueError(f"expected [B,C,T,H,W], got {tuple(images.shape)}")
    _, _, frames, height, width = images.shape
    if frames % tubelet_size or height % patch_size or width % patch_size:
        raise ValueError("video dimensions must be divisible by tubelet/patch size")
    return rearrange(
        images,
        "b c (t u) (h p) (w q) -> b (t h w) (u p q c)",
        u=tubelet_size,
        p=patch_size,
        q=patch_size,
    )


def unpatchify_video(
    patches: Tensor,
    *,
    patch_size: int,
    tubelet_size: int,
    temporal_units: int,
    grid_height: int,
    grid_width: int,
) -> Tensor:
    expected = temporal_units * grid_height * grid_width
    if patches.shape[1] != expected:
        raise ValueError(f"expected {expected} patches, got {patches.shape[1]}")
    return rearrange(
        patches,
        "b (t h w) (u p q c) -> b c (t u) (h p) (w q)",
        t=temporal_units,
        h=grid_height,
        w=grid_width,
        u=tubelet_size,
        p=patch_size,
        q=patch_size,
        c=3,
    )


class TemporalPixelDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        num_patches: int,
        prediction_dim: int,
        *,
        decoder_dim: int = 256,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
    ) -> None:
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_dim,
                    num_heads=decoder_heads,
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=lambda dim, **kwargs: nn.LayerNorm(dim, eps=1e-6, **kwargs),
                )
                for _ in range(decoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(decoder_dim, eps=1e-6)
        self.pred = nn.Linear(decoder_dim, prediction_dim)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

    def forward(self, visible: Tensor, ids_restore: Tensor) -> Tensor:
        visible = self.decoder_embed(visible)
        missing = ids_restore.shape[1] - visible.shape[1]
        mask_tokens = self.mask_token.expand(visible.shape[0], missing, -1)
        restored = torch.cat([visible, mask_tokens], dim=1)
        restored = torch.gather(
            restored, 1, ids_restore.unsqueeze(-1).expand(-1, -1, restored.shape[-1])
        )
        restored = restored + self.decoder_pos_embed
        for block in self.blocks:
            restored = block(restored)
        return self.pred(self.norm(restored))
