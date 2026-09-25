"""Last-slot mosaic readout for visuo-JEPA (WM-05).

Vendor ``jepa.JEPA.encode`` takes ``last_hidden_state[:, 0]`` (the ViT CLS)
as the tactile latent. One CLS over the whole 4x4 mosaic mixes empty history
cells with current L7|R7. This wrap keeps the full mosaic image and replaces
only that CLS with a pool of the current left/right gel patches.

``cls`` (default) leaves the encoder unchanged. ``last_slot_pool`` means L7
and R7 patch tokens. ``last_slot_lr`` concatenates the two finger means then
a Linear(2D, D) that starts as 0.5/0.5 (same as pool) so L/R can split later.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

HISTORY_FRAMES = 8
from tacmind0.tactile.mosaic_posemb import (
    _encoder_image_patch_size,
    _vit_hidden_size,
    mosaic_patch_ids,
)

READOUT_MODES = ("cls", "last_slot_pool", "last_slot_lr")
READOUT_ALIASES = {
    "false": "cls",
    "off": "cls",
    "none": "cls",
    "0": "cls",
    "cls": "cls",
    "token": "cls",
    "pool": "last_slot_pool",
    "last_slot_pool": "last_slot_pool",
    "last_pool": "last_slot_pool",
    "mean": "last_slot_pool",
    "lr": "last_slot_lr",
    "last_slot_lr": "last_slot_lr",
    "current_lr": "last_slot_lr",
    "concat": "last_slot_lr",
}


def parse_mosaic_readout(value: bool | str | None) -> str:
    if value is None or value is False:
        return "cls"
    if value is True:
        return "last_slot_pool"
    key = str(value).strip().lower()
    if key not in READOUT_ALIASES:
        raise ValueError(
            f"mosaic_readout must be cls, last_slot_pool, or last_slot_lr, got {value!r}"
        )
    return READOUT_ALIASES[key]


def last_slot_patch_masks(
    img_size: int,
    patch_size: int,
    *,
    last_slot: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean masks over row-major ViT patches for current L7 and R7."""
    finger, slot = mosaic_patch_ids(int(img_size), int(patch_size))
    slot_id = HISTORY_FRAMES - 1 if last_slot is None else int(last_slot)
    left = (finger == 0) & (slot == slot_id)
    right = (finger == 1) & (slot == slot_id)
    if int(left.sum()) == 0 or int(right.sum()) == 0:
        raise ValueError(
            f"last-slot masks empty for img_size={img_size} patch_size={patch_size} "
            f"slot={slot_id}"
        )
    return left.contiguous(), right.contiguous()


def pool_last_slot(
    tokens: torch.Tensor,
    left_mask: torch.Tensor,
    right_mask: torch.Tensor,
    *,
    fuse: nn.Module | None = None,
) -> torch.Tensor:
    """Pool current L/R patch tokens. ``tokens`` is ``(B, 1+N, D)`` with CLS first."""
    if tokens.ndim != 3 or tokens.size(1) < 2:
        raise ValueError(f"expected (B, 1+N, D) tokens, got {tuple(tokens.shape)}")
    patches = tokens[:, 1:]
    n_ids = int(left_mask.numel())
    if patches.size(1) != n_ids:
        raise ValueError(
            f"expected {n_ids} mosaic patches, got {int(patches.size(1))}"
        )
    mask_device = patches.device
    left = patches[:, left_mask.to(device=mask_device)]
    right = patches[:, right_mask.to(device=mask_device)]
    left_mean = left.mean(dim=1)
    right_mean = right.mean(dim=1)
    if fuse is None:
        return 0.5 * (left_mean + right_mean)
    return fuse(torch.cat([left_mean, right_mean], dim=-1))


def _mean_fuse_linear(dim: int) -> nn.Linear:
    """``Linear(2D, D)`` that starts as the mean of the two finger vectors."""
    fuse = nn.Linear(int(dim) * 2, int(dim))
    nn.init.zeros_(fuse.weight)
    nn.init.zeros_(fuse.bias)
    eye = torch.eye(int(dim), dtype=fuse.weight.dtype)
    fuse.weight.data[:, : int(dim)] = 0.5 * eye
    fuse.weight.data[:, int(dim) :] = 0.5 * eye
    return fuse


def _replace_cls_token(output: Any, readout: torch.Tensor) -> Any:
    hidden = output.last_hidden_state
    if hidden.ndim != 3 or hidden.size(1) < 2:
        raise ValueError(
            f"readout expects last_hidden_state (B, 1+N, D), got {tuple(hidden.shape)}"
        )
    if readout.shape != (hidden.size(0), hidden.size(-1)):
        raise ValueError(
            f"readout {tuple(readout.shape)} incompatible with "
            f"last_hidden_state {tuple(hidden.shape)}"
        )
    rewritten = hidden.clone()
    rewritten[:, 0] = readout.to(dtype=hidden.dtype)
    try:
        output.last_hidden_state = rewritten
        return output
    except (AttributeError, TypeError):
        return type("Out", (), {"last_hidden_state": rewritten})()


class MosaicReadoutEncoder(nn.Module):
    """``jepa.encoder`` wrap: ViT still sees 4x4; CLS token is last-slot L/R."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        img_size: int,
        patch_size: int,
        mode: str = "last_slot_pool",
    ) -> None:
        super().__init__()
        chosen = parse_mosaic_readout(mode)
        if chosen == "cls":
            raise ValueError("MosaicReadoutEncoder needs last_slot_pool or last_slot_lr")
        img_size, patch_size = _encoder_image_patch_size(encoder, img_size, patch_size)
        left, right = last_slot_patch_masks(img_size, patch_size)
        self.encoder = encoder
        self.mode = chosen
        self.register_buffer("left_mask", left, persistent=False)
        self.register_buffer("right_mask", right, persistent=False)
        hidden = _vit_hidden_size(encoder)
        self.fuse = _mean_fuse_linear(hidden) if chosen == "last_slot_lr" else None

    @property
    def config(self) -> Any:
        return self.encoder.config

    def forward(self, pixel_values: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        output = self.encoder(pixel_values, *args, **kwargs)
        readout = pool_last_slot(
            output.last_hidden_state,
            self.left_mask,
            self.right_mask,
            fuse=self.fuse,
        )
        return _replace_cls_token(output, readout)


def wrap_vit_mosaic_readout(
    encoder: nn.Module,
    *,
    img_size: int,
    patch_size: int,
    mode: str | bool = "cls",
    enabled: bool = True,
) -> nn.Module:
    """Replace CLS with last-slot L/R pooling when ``mode`` is not ``cls``."""
    chosen = parse_mosaic_readout(mode)
    if not enabled or chosen == "cls" or isinstance(encoder, MosaicReadoutEncoder):
        return encoder
    return MosaicReadoutEncoder(
        encoder,
        img_size=int(img_size),
        patch_size=int(patch_size),
        mode=chosen,
    )
