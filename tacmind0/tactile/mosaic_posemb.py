"""Mosaic positional encodings for 4x4 GelSight mosaics.

ViT 2D position treats the raster as a photo: L1 is next to R1, while the
next left-finger time L2 wraps to the next row.

Additive wraps (CLS unchanged):

- ``finger_slot`` (WM-01): finger ``{L,R}`` plus slot ``{0..7}``. Too
  factorized; visuo-JEPA did not prefer it.
- ``cell`` (WM-08): one embedding per 4x4 cell (16 ids). Same idea, one table.

The ``factorized_rope`` mode is a separate architecture ablation. It removes
the ViT's global raster position table, adds only a learnable left/right
finger embedding, and applies three-axis RoPE to every tactile self-attention
Q/K using ``(slot, cell_local_y, cell_local_x)`` coordinates.  The
``spiral_rope`` mode keeps the temporal axis but replaces axial spatial RoPE
with multi-directional projections inside each tactile cell.  The
``canonical_spiral_rope`` mode additionally mirrors the right finger's local
X coordinate into a shared tactile frame.

Default ``model.mosaic_posemb=false``. See ``docs/lewm_wm_experiments.md``.

The ``mosaic_cell_mask`` wrapper keeps the full RGB+marker Mosaic as input,
then randomly retains a fixed number of complete 4x4 Mosaic cells after
PatchEmbed.  This changes the transformer sequence length without changing
the marker representation or the image-level patch projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

GRID_COLS = GRID_ROWS = 4
HISTORY_FRAMES = 8
SIDE_COLS = 2

N_FINGERS = 2
N_CELLS = GRID_ROWS * GRID_COLS
SPIRAL_DIRECTIONS = 8
POSEMB_MODES = (
    "off",
    "finger_slot",
    "cell",
    "factorized_rope",
    "spiral_rope",
    "canonical_spiral_rope",
)
POSEMB_ALIASES = {
    "false": "off",
    "off": "off",
    "none": "off",
    "0": "off",
    "true": "finger_slot",
    "finger_slot": "finger_slot",
    "on": "finger_slot",
    "cell": "cell",
    "cells": "cell",
    "factorized_rope": "factorized_rope",
    "3axis_rope": "factorized_rope",
    "spiral_rope": "spiral_rope",
    "spatiotemporal_spiral": "spiral_rope",
    "canonical_spiral_rope": "canonical_spiral_rope",
    "canonical_spiral": "canonical_spiral_rope",
}


def parse_mosaic_posemb(value: bool | int | str | None) -> str:
    if value is None or value is False:
        return "off"
    if value is True:
        return "finger_slot"
    if isinstance(value, int) and not isinstance(value, bool):
        if value == 0:
            return "off"
        if value == 1:
            return "finger_slot"
        raise ValueError(f"mosaic_posemb int must be 0 or 1, got {value!r}")
    key = str(value).strip().lower()
    if key not in POSEMB_ALIASES:
        raise ValueError(
            "mosaic_posemb must be false, finger_slot, cell, factorized_rope, "
            "spiral_rope, or canonical_spiral_rope, "
            f"got {value!r}"
        )
    return POSEMB_ALIASES[key]


def parse_mosaic_cell_mask_ratio(value: float | int | str | None) -> float:
    """Parse a train-time Mosaic cell drop ratio in ``[0, 1)``.

    ``0`` disables the wrapper.  The sampler always keeps at least one cell;
    for the intended ``0.5`` setting it keeps exactly 8 of the 16 cells.
    """
    if value is None or value is False:
        return 0.0
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"mosaic_cell_mask_ratio must be a number in [0, 1), got {value!r}"
        ) from exc
    if not 0.0 <= ratio < 1.0:
        raise ValueError(
            f"mosaic_cell_mask_ratio must be in [0, 1), got {value!r}"
        )
    return ratio


def mosaic_patch_ids(
    img_size: int,
    patch_size: int,
    *,
    grid_cols: int = GRID_COLS,
    grid_rows: int = GRID_ROWS,
    side_cols: int = SIDE_COLS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-major ViT patch ``(finger, slot)`` ids for a square mosaic.

    Layout (2 columns per finger)::

        L0 L1 | R0 R1
        L2 L3 | R2 R3
        L4 L5 | R4 R5
        L6 L7 | R6 R7
    """
    size = int(img_size)
    patch = int(patch_size)
    if size <= 0 or patch <= 0:
        raise ValueError(f"img_size and patch_size must be positive, got {size} {patch}")
    if size % patch != 0:
        raise ValueError(f"img_size {size} must be divisible by patch_size {patch}")
    grid = size // patch
    if grid % int(grid_cols) != 0 or grid % int(grid_rows) != 0:
        raise ValueError(
            f"mosaic {grid_rows}x{grid_cols} must divide the patch grid {grid}"
        )
    cell_h = grid // int(grid_rows)
    cell_w = grid // int(grid_cols)
    rows = torch.arange(grid)
    cols = torch.arange(grid)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    cell_row = torch.div(rr, cell_h, rounding_mode="floor")
    cell_col = torch.div(cc, cell_w, rounding_mode="floor")
    finger = (cell_col >= int(side_cols)).to(torch.long)
    local_col = cell_col % int(side_cols)
    slot = cell_row * int(side_cols) + local_col
    if int(slot.max()) >= HISTORY_FRAMES or int(finger.max()) >= N_FINGERS:
        raise ValueError(f"slot/finger overflow: slot={int(slot.max())} finger={int(finger.max())}")
    return finger.reshape(-1).contiguous(), slot.reshape(-1).contiguous()


def mosaic_cell_ids(
    img_size: int,
    patch_size: int,
    *,
    grid_cols: int = GRID_COLS,
    grid_rows: int = GRID_ROWS,
    side_cols: int = SIDE_COLS,
) -> torch.Tensor:
    """Row-major 4x4 cell id for each ViT patch (0 = L0, 15 = R7)."""
    finger, slot = mosaic_patch_ids(
        img_size,
        patch_size,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        side_cols=side_cols,
    )
    row = torch.div(slot, int(side_cols), rounding_mode="floor")
    local_col = slot % int(side_cols)
    col = finger * int(side_cols) + local_col
    return (row * int(grid_cols) + col).contiguous()


def mosaic_patch_coordinates(
    img_size: int,
    patch_size: int,
    *,
    grid_cols: int = GRID_COLS,
    grid_rows: int = GRID_ROWS,
    side_cols: int = SIDE_COLS,
) -> torch.Tensor:
    """Return ``(CLS, patches)`` coordinates ``(slot, local_y, local_x)``.

    The two fingers share the same cell-local spatial frame.  Thus L4 and R4
    have the same temporal coordinate and the same local ``(y, x)`` grid;
    finger identity is carried only by the additive finger embedding.
    """
    size = int(img_size)
    patch = int(patch_size)
    if size <= 0 or patch <= 0 or size % patch != 0:
        raise ValueError(f"img_size and patch_size must define a grid, got {size} {patch}")
    grid = size // patch
    if grid % int(grid_cols) != 0 or grid % int(grid_rows) != 0:
        raise ValueError(
            f"mosaic {grid_rows}x{grid_cols} must divide the patch grid {grid}"
        )
    cell_h = grid // int(grid_rows)
    cell_w = grid // int(grid_cols)
    rows = torch.arange(grid)
    cols = torch.arange(grid)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    cell_row = torch.div(rr, cell_h, rounding_mode="floor")
    cell_col = torch.div(cc, cell_w, rounding_mode="floor")
    local_y = rr % cell_h
    local_x = cc % cell_w
    slot = cell_row * int(side_cols) + (cell_col % int(side_cols))
    if int(slot.max()) >= HISTORY_FRAMES:
        raise ValueError(f"slot overflow: {int(slot.max())}")
    patches = torch.stack((slot, local_y, local_x), dim=-1).reshape(-1, 3)
    cls = torch.zeros((1, 3), dtype=torch.long)
    return torch.cat((cls, patches.to(dtype=torch.long)), dim=0).contiguous()


def _rotary_axis_dims(head_dim: int) -> tuple[int, int, int]:
    """Split an even head dimension into three even rotary subdimensions."""
    usable_pairs = max(int(head_dim), 0) // 2
    base, remainder = divmod(usable_pairs, 3)
    return tuple(2 * (base + int(axis < remainder)) for axis in range(3))


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    even = values[..., 0::2]
    odd = values[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_three_axis_rope(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    theta: float = 10_000.0,
) -> torch.Tensor:
    """Apply factorized RoPE to ``(B, heads, tokens, head_dim)`` values."""
    if values.ndim != 4:
        raise ValueError(f"expected attention values (B,H,N,D), got {tuple(values.shape)}")
    coords = torch.as_tensor(coordinates, device=values.device)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != values.shape[2]:
        raise ValueError(
            f"expected {values.shape[2]} three-axis coordinates, got {tuple(coords.shape)}"
        )
    axis_dims = _rotary_axis_dims(int(values.shape[-1]))
    out = values.clone()
    offset = 0
    for axis, width in enumerate(axis_dims):
        if width == 0:
            continue
        inv_freq = 1.0 / (
            float(theta)
            ** (torch.arange(0, width, 2, device=values.device, dtype=torch.float32) / width)
        )
        angles = coords[:, axis].to(torch.float32).unsqueeze(-1) * inv_freq
        cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype=values.dtype)
        sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype=values.dtype)
        part = values[..., offset : offset + width]
        out[..., offset : offset + width] = part * cos + _rotate_half(part) * sin
        offset += width
    return out


def apply_three_axis_rope_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    theta: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q and K with the same factorized ``(t, y, x)`` coordinates."""
    return (
        apply_three_axis_rope(query, coordinates, theta=theta),
        apply_three_axis_rope(key, coordinates, theta=theta),
    )


def _apply_one_axis_rope(
    values: torch.Tensor,
    positions: torch.Tensor,
    *,
    theta: float,
) -> torch.Tensor:
    """Rotate all pair channels in ``values`` using one scalar coordinate."""
    width = int(values.shape[-1])
    if width == 0:
        return values
    if width % 2:
        raise ValueError(f"RoPE width must be even, got {width}")
    inv_freq = 1.0 / (
        float(theta)
        ** (torch.arange(0, width, 2, device=values.device, dtype=torch.float32) / width)
    )
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype=values.dtype)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype=values.dtype)
    return values * cos + _rotate_half(values) * sin


def _spatiotemporal_rope_dims(head_dim: int) -> tuple[int, int]:
    """Split head channels into temporal and multi-directional spatial pairs."""
    pairs = max(int(head_dim), 0) // 2
    temporal_pairs = (pairs + 2) // 3
    temporal_width = 2 * temporal_pairs
    return temporal_width, 2 * (pairs - temporal_pairs)


def apply_spatiotemporal_spiral_rope(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    theta: float = 10_000.0,
    directions: int = SPIRAL_DIRECTIONS,
) -> torch.Tensor:
    """Apply temporal RoPE plus 2-D multi-directional Spiral RoPE.

    ``coordinates`` is ``(tokens, 3)`` with ``(time, local_y, local_x)``.
    The temporal channels use the first third of each attention head.  The
    remaining pairs cycle through ``directions`` projections
    ``x*cos(phi) + y*sin(phi)`` with ``phi=k*pi/directions``.  This keeps the
    total rotary budget fixed while adding diagonal/oblique spatial relations.
    """
    if values.ndim != 4:
        raise ValueError(f"expected attention values (B,H,N,D), got {tuple(values.shape)}")
    coords = torch.as_tensor(coordinates, device=values.device)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != values.shape[2]:
        raise ValueError(
            f"expected {values.shape[2]} (time,y,x) coordinates, got {tuple(coords.shape)}"
        )
    if int(directions) <= 0:
        raise ValueError(f"directions must be positive, got {directions}")

    temporal_width, spatial_width = _spatiotemporal_rope_dims(int(values.shape[-1]))
    out = values.clone()
    if temporal_width:
        out[..., :temporal_width] = _apply_one_axis_rope(
            values[..., :temporal_width], coords[:, 0], theta=theta
        )
    if spatial_width:
        spatial_pairs = spatial_width // 2
        pair_ids = torch.arange(spatial_pairs, device=values.device)
        direction_ids = pair_ids % int(directions)
        phi = direction_ids.to(torch.float32) * (torch.pi / float(directions))
        y = coords[:, 1].to(torch.float32).unsqueeze(-1)
        x = coords[:, 2].to(torch.float32).unsqueeze(-1)
        projections = x * phi.cos() + y * phi.sin()
        frequency_ids = torch.div(pair_ids, int(directions), rounding_mode="floor")
        frequency_count = max((spatial_pairs + int(directions) - 1) // int(directions), 1)
        inv_freq = 1.0 / (
            float(theta)
            ** (frequency_ids.to(torch.float32) / float(frequency_count))
        )
        angles = projections * inv_freq.unsqueeze(0)
        cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype=values.dtype)
        sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype=values.dtype)
        spatial = values[..., temporal_width:]
        out[..., temporal_width:] = spatial * cos + _rotate_half(spatial) * sin
    return out


def apply_spatiotemporal_spiral_rope_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    theta: float = 10_000.0,
    directions: int = SPIRAL_DIRECTIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same temporal + Spiral RoPE coordinates to Q and K."""
    return (
        apply_spatiotemporal_spiral_rope(
            query, coordinates, theta=theta, directions=directions
        ),
        apply_spatiotemporal_spiral_rope(
            key, coordinates, theta=theta, directions=directions
        ),
    )


def _add_patch_embeddings(
    tokens: torch.Tensor, extra: torch.Tensor, n_ids: int, num_register_tokens: int = 0
) -> torch.Tensor:
    prefix = 1 + max(int(num_register_tokens), 0)
    if tokens.ndim != 3 or tokens.size(1) < 2:
        raise ValueError(f"expected (B, 1+N, D) tokens, got {tuple(tokens.shape)}")
    cls, patches = tokens[:, :prefix], tokens[:, prefix:]
    if patches.size(1) != n_ids:
        raise ValueError(
            f"expected {n_ids} mosaic patches, got {patches.size(1)} "
            "(img_size/patch_size must match the 4x4 mosaic)"
        )
    return torch.cat([cls, patches + extra.to(dtype=patches.dtype)], dim=1)


def apply_mosaic_posemb(
    tokens: torch.Tensor,
    finger_emb: nn.Embedding,
    slot_emb: nn.Embedding,
    finger_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    num_register_tokens: int = 0,
) -> torch.Tensor:
    """Add finger+slot embeddings to patch tokens. ``tokens`` is ``(B, 1+N, D)`` with CLS first."""
    ids_device = tokens.device
    extra = finger_emb(finger_ids.to(ids_device)) + slot_emb(slot_ids.to(ids_device))
    return _add_patch_embeddings(
        tokens, extra, int(finger_ids.numel()), num_register_tokens
    )


def apply_cell_posemb(
    tokens: torch.Tensor,
    cell_emb: nn.Embedding,
    cell_ids: torch.Tensor,
    num_register_tokens: int = 0,
) -> torch.Tensor:
    """Add one 4x4-cell embedding to each patch token. CLS unchanged."""
    extra = cell_emb(cell_ids.to(tokens.device))
    return _add_patch_embeddings(tokens, extra, int(cell_ids.numel()), num_register_tokens)


class MosaicViTEmbeddings(nn.Module):
    """Wrap HuggingFace ``ViTEmbeddings`` and add mosaic cell or finger/slot ids.

    ``ViTModel.forward`` reads ``self.embeddings.patch_embeddings`` for dtype
    casting, so unknown attributes delegate to the inner embeddings module.
    """

    def __init__(
        self,
        embeddings: nn.Module,
        *,
        hidden_size: int,
        patch_size: int,
        num_register_tokens: int = 0,
        mode: str = "finger_slot",
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.patch_size = int(patch_size)
        self.num_register_tokens = max(int(num_register_tokens), 0)
        self.mode = parse_mosaic_posemb(mode)
        if self.mode == "off":
            raise ValueError("MosaicViTEmbeddings needs finger_slot or cell")
        dim = int(hidden_size)
        if self.mode == "cell":
            self.cell_emb = nn.Embedding(N_CELLS, dim)
            nn.init.zeros_(self.cell_emb.weight)
        else:
            self.finger_emb = nn.Embedding(N_FINGERS, dim)
            self.slot_emb = nn.Embedding(HISTORY_FRAMES, dim)
            nn.init.zeros_(self.finger_emb.weight)
            nn.init.zeros_(self.slot_emb.weight)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self._modules.get("embeddings")
            if inner is None:
                raise
            return getattr(inner, name)

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        hidden = self.embeddings(pixel_values, **kwargs)
        height, width = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
        if height != width:
            raise ValueError(f"mosaic posemb expects a square image, got {height}x{width}")
        patch = getattr(self.embeddings, "patch_size", self.patch_size)
        if isinstance(patch, (tuple, list)):
            patch = int(patch[0])
        if self.mode == "cell":
            cell_ids = mosaic_cell_ids(height, int(patch))
            return apply_cell_posemb(
                hidden, self.cell_emb, cell_ids, self.num_register_tokens
            )
        finger_ids, slot_ids = mosaic_patch_ids(height, int(patch))
        return apply_mosaic_posemb(
            hidden,
            self.finger_emb,
            self.slot_emb,
            finger_ids,
            slot_ids,
            self.num_register_tokens,
        )


class MosaicCellMaskEmbeddings(nn.Module):
    """HF ViT embeddings with per-sample random complete-cell token pruning.

    The input is still the complete 4x4 RGB+marker Mosaic.  Masking happens
    after the wrapped embedding module has produced patch tokens, so every
    flattened JEPA frame samples its own cell subset while the prefix tokens
    (CLS and optional registers) remain intact.
    """

    def __init__(
        self,
        embeddings: nn.Module,
        *,
        img_size: int,
        patch_size: int,
        num_register_tokens: int = 0,
        ratio: float = 0.5,
        mask_eval: bool = False,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.num_register_tokens = max(int(num_register_tokens), 0)
        self.mask_ratio = parse_mosaic_cell_mask_ratio(ratio)
        self.mask_eval = bool(mask_eval)
        self.num_cells = N_CELLS
        self.register_buffer(
            "patch_cell_ids",
            mosaic_cell_ids(self.img_size, self.patch_size),
            persistent=False,
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self._modules.get("embeddings")
            if inner is None:
                raise
            return getattr(inner, name)

    def _selected_patch_indices(
        self,
        batch_size: int,
        patch_count: int,
        cell_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        drop = min(round(self.num_cells * self.mask_ratio), self.num_cells - 1)
        keep_cells = self.num_cells - drop
        sampled = torch.rand(
            batch_size, self.num_cells, device=device
        ).topk(keep_cells, dim=1, largest=False).indices.sort(dim=1).values
        selected_cells = torch.zeros(
            batch_size, self.num_cells, dtype=torch.bool, device=device
        )
        selected_cells.scatter_(1, sampled, True)
        keep_patches = selected_cells[:, cell_ids.to(device=device)]
        positions = torch.arange(patch_count, device=device).expand(batch_size, -1)
        selected = positions.masked_fill(~keep_patches, patch_count)
        selected = selected.sort(dim=1).values[:, : int(keep_patches.sum(dim=1)[0])]
        return selected, sampled

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        hidden = self.embeddings(pixel_values, **kwargs)
        if not (self.training or self.mask_eval) or self.mask_ratio == 0.0:
            return hidden
        if pixel_values.ndim != 4:
            raise ValueError(
                f"mosaic cell masking expects NCHW pixels, got {tuple(pixel_values.shape)}"
            )
        height, width = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
        if height != width:
            raise ValueError(
                f"mosaic cell masking expects a square image, got {height}x{width}"
            )
        patch = getattr(self.embeddings, "patch_size", self.patch_size)
        if isinstance(patch, (tuple, list)):
            patch = int(patch[0])
        cell_ids = mosaic_cell_ids(height, int(patch)).to(device=hidden.device)
        prefix = 1 + self.num_register_tokens
        patch_count = int(cell_ids.numel())
        if hidden.ndim != 3 or hidden.size(1) != prefix + patch_count:
            raise ValueError(
                "mosaic cell masking expected prefix+4x4 patch tokens, got "
                f"hidden={tuple(hidden.shape)} prefix={prefix} patches={patch_count}"
            )
        selected, sampled = self._selected_patch_indices(
            int(hidden.size(0)), patch_count, cell_ids, hidden.device
        )
        prefix_tokens = hidden[:, :prefix]
        patch_tokens = hidden[:, prefix:]
        selected_tokens = patch_tokens.gather(
            1, selected.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        )
        # Useful for smoke tests and diagnostics; it is intentionally not a
        # parameter or checkpoint state because masks are sampled per forward.
        self.last_keep_cells = sampled
        return torch.cat((prefix_tokens, selected_tokens), dim=1)


class MosaicCellMaskEncoder(nn.Module):
    """Wrap a ViT and prune random complete Mosaic cells after PatchEmbed."""

    def __init__(
        self,
        vit: nn.Module,
        *,
        img_size: int,
        patch_size: int,
        ratio: float = 0.5,
        mask_eval: bool = False,
    ) -> None:
        super().__init__()
        embedding_owner, embeddings = _resolve_vit_embeddings(vit)
        hidden = _vit_hidden_size(embedding_owner)
        resolved_img, resolved_patch = _encoder_image_patch_size(
            embedding_owner, img_size, patch_size
        )
        config = _encoder_config(embedding_owner)
        num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)
        embedding_owner.embeddings = MosaicCellMaskEmbeddings(
            embeddings,
            img_size=resolved_img,
            patch_size=resolved_patch,
            num_register_tokens=num_register_tokens,
            ratio=ratio,
            mask_eval=mask_eval,
        )
        self.vit = vit
        self.mask_ratio = parse_mosaic_cell_mask_ratio(ratio)
        self.mask_eval = bool(mask_eval)

    @property
    def config(self) -> Any:
        return _encoder_config(self.vit)

    def forward(self, pixel_values: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.vit(pixel_values, *args, **kwargs)


def _disable_absolute_position_embedding(embeddings: nn.Module) -> None:
    """Replace the HF ViT raster table with a non-persistent zero buffer."""
    position = getattr(embeddings, "position_embeddings", None)
    if position is None:
        raise TypeError("factorized_rope requires HuggingFace ViT position_embeddings")
    if isinstance(position, nn.Parameter):
        del embeddings.position_embeddings
        embeddings.register_buffer(
            "position_embeddings", torch.zeros_like(position), persistent=False
        )
    else:
        position.zero_()
        if torch.is_tensor(position):
            position.requires_grad_(False)


class MosaicFactorizedRoPEEmbeddings(nn.Module):
    """HF ViT embeddings with local tactile coordinates and finger identity."""

    def __init__(
        self,
        embeddings: nn.Module,
        *,
        hidden_size: int,
        patch_size: int,
        img_size: int,
        num_register_tokens: int = 0,
        finger_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.patch_size = int(patch_size)
        self.img_size = int(img_size)
        self.num_register_tokens = max(int(num_register_tokens), 0)
        reference = next(iter(embeddings.parameters()), None)
        device = reference.device if reference is not None else None
        dtype = reference.dtype if reference is not None else None
        self.finger_emb = nn.Embedding(N_FINGERS, int(hidden_size)).to(
            device=device, dtype=dtype
        )
        nn.init.normal_(self.finger_emb.weight, std=0.02)
        self.finger_gate = nn.Parameter(
            torch.tensor(float(finger_gate_init), device=device, dtype=dtype)
        )
        _disable_absolute_position_embedding(self.embeddings)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self._modules.get("embeddings")
            if inner is None:
                raise
            return getattr(inner, name)

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        hidden = self.embeddings(pixel_values, **kwargs)
        height, width = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
        if height != width:
            raise ValueError(f"factorized_rope expects a square image, got {height}x{width}")
        patch = getattr(self.embeddings, "patch_size", self.patch_size)
        if isinstance(patch, (tuple, list)):
            patch = int(patch[0])
        finger_ids, _slot_ids = mosaic_patch_ids(height, int(patch))
        extra = self.finger_gate * self.finger_emb(finger_ids.to(hidden.device))
        return _add_patch_embeddings(
            hidden, extra, int(finger_ids.numel()), self.num_register_tokens
        )


class MosaicRoPESelfAttention(nn.Module):
    """HF ViT self-attention with tactile RoPE on Q and K."""

    def __init__(
        self,
        attention: nn.Module,
        coordinates: torch.Tensor,
        *,
        theta: float = 10_000.0,
        rope_kind: str = "three_axis",
        spiral_directions: int = SPIRAL_DIRECTIONS,
    ) -> None:
        super().__init__()
        for name in ("query", "key", "value"):
            module = getattr(attention, name, None)
            if module is None:
                raise TypeError(f"factorized_rope requires HF ViT attention.{name}")
            setattr(self, name, module)
        self.num_attention_heads = int(attention.num_attention_heads)
        self.attention_head_size = int(attention.attention_head_size)
        self.all_head_size = int(getattr(attention, "all_head_size", self.num_attention_heads * self.attention_head_size))
        self.dropout_prob = float(getattr(attention, "dropout_prob", 0.0))
        self.scaling = float(getattr(attention, "scaling", self.attention_head_size**-0.5))
        self.is_causal = False
        self.rope_theta = float(theta)
        if rope_kind not in {"three_axis", "spatiotemporal_spiral"}:
            raise ValueError(f"unknown tactile RoPE kind: {rope_kind}")
        self.rope_kind = rope_kind
        self.spiral_directions = int(spiral_directions)
        self.register_buffer("coordinates", torch.as_tensor(coordinates).clone(), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del kwargs
        batch_size = hidden_states.shape[0]
        new_shape = (
            batch_size,
            -1,
            self.num_attention_heads,
            self.attention_head_size,
        )
        query = self.query(hidden_states).view(*new_shape).transpose(1, 2)
        key = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        if self.rope_kind == "three_axis":
            query, key = apply_three_axis_rope_qk(
                query, key, self.coordinates, theta=self.rope_theta
            )
        else:
            query, key = apply_spatiotemporal_spiral_rope_qk(
                query,
                key,
                self.coordinates,
                theta=self.rope_theta,
                directions=self.spiral_directions,
            )
        dropout = self.dropout_prob if self.training else 0.0
        if head_mask is None:
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout,
                is_causal=False,
                scale=self.scaling,
            )
        else:
            scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            probs = torch.nn.functional.dropout(probs, p=dropout, training=self.training)
            probs = probs * head_mask
            context = torch.matmul(probs, value)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.all_head_size
        )
        return context, None


class MosaicFactorizedRoPEEncoder(nn.Module):
    """Replace HF ViT raster PE and patch attention with tactile RoPE."""

    def __init__(
        self,
        vit: nn.Module,
        *,
        img_size: int,
        patch_size: int,
        rope_theta: float = 10_000.0,
        finger_gate_init: float = 0.1,
        rope_kind: str = "three_axis",
        canonicalize_right: bool = False,
    ) -> None:
        super().__init__()
        embedding_owner, embeddings = _resolve_vit_embeddings(vit)
        hidden = _vit_hidden_size(embedding_owner)
        img_size, patch_size = _encoder_image_patch_size(
            embedding_owner, img_size, patch_size
        )
        config = _encoder_config(embedding_owner)
        num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)
        patch_coordinates = mosaic_patch_coordinates(img_size, patch_size)[1:]
        finger_ids, _slot_ids = mosaic_patch_ids(img_size, patch_size)
        if canonicalize_right:
            patch_grid = img_size // patch_size
            cell_width = patch_grid // GRID_COLS
            patch_coordinates = patch_coordinates.clone()
            right = finger_ids.to(torch.bool)
            patch_coordinates[:, 2] = torch.where(
                right,
                int(cell_width - 1) - patch_coordinates[:, 2],
                patch_coordinates[:, 2],
            )
        coordinates = torch.cat(
            (
                torch.zeros((1 + num_register_tokens, 3), dtype=torch.long),
                patch_coordinates,
            ),
            dim=0,
        ).contiguous()
        embedding_owner.embeddings = MosaicFactorizedRoPEEmbeddings(
            embeddings,
            hidden_size=hidden,
            patch_size=patch_size,
            img_size=img_size,
            num_register_tokens=num_register_tokens,
            finger_gate_init=finger_gate_init,
        )
        # DINOv2-with-registers keeps the HF ViT in ``.model``.  Use the
        # embedding owner resolved above so both a bare ViT and that wrapper
        # patch the same attention implementation.
        layers = getattr(getattr(embedding_owner, "encoder", None), "layer", None)
        if layers is None or not len(layers):
            raise TypeError("factorized_rope requires HF ViT encoder.layer")
        for layer in layers:
            outer_attention = getattr(layer, "attention", None)
            attention = getattr(outer_attention, "attention", None)
            if attention is None:
                raise TypeError("factorized_rope requires HF ViT layer attention.attention")
            outer_attention.attention = MosaicRoPESelfAttention(
                attention,
                coordinates,
                theta=rope_theta,
                rope_kind=rope_kind,
            )
        self.vit = vit
        self.rope_theta = float(rope_theta)
        self.rope_kind = rope_kind
        self.canonicalize_right = bool(canonicalize_right)
        self.register_buffer("coordinates", coordinates, persistent=False)

    @property
    def config(self) -> Any:
        return _encoder_config(self.vit)

    def forward(self, pixel_values: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.vit(pixel_values, *args, **kwargs)


@dataclass
class MosaicStreamState:
    """Per-layer KV cache for incremental tactile Mosaic inference."""

    patch_keys: list[torch.Tensor]
    patch_values: list[torch.Tensor]
    cls_state: torch.Tensor
    next_slot: int


class MosaicCausalRoPESelfAttention(nn.Module):
    """Tactile RoPE attention with slot-causal, same-slot bidirectional access."""

    def __init__(
        self,
        attention: nn.Module,
        coordinates: torch.Tensor,
        causal_mask: torch.Tensor,
        *,
        theta: float = 10_000.0,
        prefix_length: int = 1,
        rope_kind: str = "three_axis",
        spiral_directions: int = SPIRAL_DIRECTIONS,
    ) -> None:
        super().__init__()
        for name in ("query", "key", "value"):
            module = getattr(attention, name, None)
            if module is None:
                raise TypeError(f"causal tactile attention requires attention.{name}")
            setattr(self, name, module)
        self.num_attention_heads = int(attention.num_attention_heads)
        self.attention_head_size = int(attention.attention_head_size)
        self.all_head_size = int(
            getattr(
                attention,
                "all_head_size",
                self.num_attention_heads * self.attention_head_size,
            )
        )
        self.dropout_prob = float(getattr(attention, "dropout_prob", 0.0))
        self.scaling = float(
            getattr(attention, "scaling", self.attention_head_size**-0.5)
        )
        self.rope_theta = float(theta)
        self.prefix_length = max(int(prefix_length), 1)
        if rope_kind not in {"three_axis", "spatiotemporal_spiral"}:
            raise ValueError(f"unknown tactile RoPE kind: {rope_kind}")
        if int(spiral_directions) <= 0:
            raise ValueError(f"spiral_directions must be positive, got {spiral_directions}")
        self.rope_kind = rope_kind
        self.spiral_directions = int(spiral_directions)
        self.register_buffer(
            "coordinates", torch.as_tensor(coordinates).clone(), persistent=False
        )
        self.register_buffer(
            "causal_mask", torch.as_tensor(causal_mask).clone(), persistent=False
        )
        self.capture_stream_state = False
        self.last_patch_keys: torch.Tensor | None = None
        self.last_patch_values: torch.Tensor | None = None

    def _project(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch_size = hidden_states.shape[0]
        new_shape = (
            batch_size,
            -1,
            self.num_attention_heads,
            self.attention_head_size,
        )
        query = self.query(hidden_states).view(*new_shape).transpose(1, 2)
        key = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        return query, key, value

    def _rotate(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rotate_tensor(query, coordinates), self._rotate_tensor(key, coordinates)

    def _rotate_tensor(
        self, values: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        if self.rope_kind == "three_axis":
            return apply_three_axis_rope(values, coordinates, theta=self.rope_theta)
        return apply_spatiotemporal_spiral_rope(
            values,
            coordinates,
            theta=self.rope_theta,
            directions=self.spiral_directions,
        )

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dropout = self.dropout_prob if self.training else 0.0
        if head_mask is None:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=dropout,
                is_causal=False,
                scale=self.scaling,
            )
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probs = F.dropout(probs, p=dropout, training=self.training)
        probs = probs * head_mask
        return torch.matmul(probs, value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del kwargs
        query, key, value = self._project(hidden_states)
        coordinates = self.coordinates[: hidden_states.size(1)].to(hidden_states.device)
        query, key = self._rotate(query, key, coordinates)
        mask = self.causal_mask[
            : hidden_states.size(1), : hidden_states.size(1)
        ].to(hidden_states.device)
        context = self._attend(query, key, value, mask=mask, head_mask=head_mask)
        if self.capture_stream_state:
            self.last_patch_keys = key[:, :, self.prefix_length :].detach()
            self.last_patch_values = value[:, :, self.prefix_length :].detach()
        context = context.transpose(1, 2).contiguous().view(
            hidden_states.shape[0], -1, self.all_head_size
        )
        return context, None

    def forward_incremental(
        self,
        query_states: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
        current_states: torch.Tensor,
        query_coordinates: torch.Tensor,
        current_coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attend to cached historical patches plus the current L/R patches.

        ``past_keys``/``past_values`` are already projected and RoPE-rotated.
        Only ``current_states`` is projected in this call, so a stream step
        never re-encodes the historical tokens.
        """
        query, _unused_key, _unused_value = self._project(query_states)
        _unused_prefix_query, prefix_key, prefix_value = self._project(
            query_states[:, : self.prefix_length]
        )
        _unused_query, current_key, current_value = self._project(current_states)
        query = self._rotate_tensor(query, query_coordinates.to(query.device))
        current_key = self._rotate_tensor(
            current_key, current_coordinates.to(current_key.device)
        )
        prefix_coordinates = query_coordinates[: self.prefix_length].to(query.device)
        prefix_key = self._rotate_tensor(prefix_key, prefix_coordinates)
        key = torch.cat((prefix_key, past_keys, current_key), dim=2)
        value = torch.cat((prefix_value, past_values, current_value), dim=2)
        mask = torch.zeros(
            (query.size(2), key.size(2)), device=query.device, dtype=torch.bool
        )
        mask[: self.prefix_length, :] = True
        mask[self.prefix_length :, self.prefix_length :] = True
        context = self._attend(query, key, value, mask=mask)
        context = context.transpose(1, 2).contiguous().view(
            query_states.shape[0], -1, self.all_head_size
        )
        return context, current_key, current_value


class MosaicCausalEncoder(nn.Module):
    """Causal tactile ViT with an incremental current-left/right KV path.

    Full training uses a slot-causal attention mask.  ``stream_init`` accepts
    one full 4x4 Mosaic window; subsequent ``stream_step`` calls accept only
    the current left/right cells shaped ``(B, 2, 3, cell_h, cell_w)`` or a
    full Mosaic from which those cells are cropped.  Each transformer layer
    reuses cached historical K/V and computes Q/K/V only for the new 32 patch
    tokens plus the CLS readout.
    """

    def __init__(
        self,
        vit: nn.Module,
        *,
        img_size: int,
        patch_size: int,
        rope_theta: float = 10_000.0,
        rope_kind: str = "three_axis",
        canonicalize_right: bool = False,
        spiral_directions: int = SPIRAL_DIRECTIONS,
    ) -> None:
        super().__init__()
        if rope_kind not in {"three_axis", "spatiotemporal_spiral"}:
            raise ValueError(f"unknown tactile RoPE kind: {rope_kind}")
        if int(spiral_directions) <= 0:
            raise ValueError(f"spiral_directions must be positive, got {spiral_directions}")
        embedding_owner, embeddings = _resolve_vit_embeddings(vit)
        hidden = _vit_hidden_size(embedding_owner)
        resolved_img, resolved_patch = _encoder_image_patch_size(
            embedding_owner, img_size, patch_size
        )
        config = _encoder_config(embedding_owner)
        num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)
        prefix_length = 1 + num_register_tokens
        patch_coordinates = mosaic_patch_coordinates(resolved_img, resolved_patch)[1:]
        finger_ids, _slot_ids = mosaic_patch_ids(resolved_img, resolved_patch)
        if canonicalize_right:
            patch_grid = resolved_img // resolved_patch
            cell_width = patch_grid // GRID_COLS
            patch_coordinates = patch_coordinates.clone()
            patch_coordinates[:, 2] = torch.where(
                finger_ids.to(torch.bool),
                int(cell_width - 1) - patch_coordinates[:, 2],
                patch_coordinates[:, 2],
            )
        coordinates = torch.cat(
            (torch.zeros((prefix_length, 3), dtype=torch.long), patch_coordinates), dim=0
        ).contiguous()
        patch_slots = patch_coordinates[:, 0]
        causal_mask = torch.zeros(
            coordinates.size(0), coordinates.size(0), dtype=torch.bool
        )
        causal_mask[:prefix_length, :] = True
        causal_mask[prefix_length:, prefix_length:] = (
            patch_slots.unsqueeze(0) <= patch_slots.unsqueeze(1)
        )
        embedding_owner.embeddings = MosaicFactorizedRoPEEmbeddings(
            embeddings,
            hidden_size=hidden,
            patch_size=resolved_patch,
            img_size=resolved_img,
            num_register_tokens=num_register_tokens,
        )
        layers = getattr(getattr(embedding_owner, "encoder", None), "layer", None)
        if layers is None or not len(layers):
            raise TypeError("causal tactile attention requires HF ViT encoder.layer")
        for layer in layers:
            outer_attention = getattr(layer, "attention", None)
            attention = getattr(outer_attention, "attention", None)
            if attention is None:
                raise TypeError("causal tactile attention requires layer attention.attention")
            outer_attention.attention = MosaicCausalRoPESelfAttention(
                attention,
                coordinates,
                causal_mask,
                theta=rope_theta,
                prefix_length=prefix_length,
                rope_kind=rope_kind,
                spiral_directions=spiral_directions,
            )
        self.vit = vit
        # Keep only a boolean here. Registering the same ViT a second time as
        # ``self._embedding_owner`` would duplicate checkpoint state keys.
        self._owner_is_inner = embedding_owner is not vit
        self.img_size = resolved_img
        self.patch_size = resolved_patch
        self.cell_size = resolved_img // GRID_COLS
        self.prefix_length = prefix_length
        self.rope_theta = float(rope_theta)
        self.rope_kind = rope_kind
        self.canonicalize_right = bool(canonicalize_right)
        self.spiral_directions = int(spiral_directions)
        self.register_buffer("coordinates", coordinates, persistent=False)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    @property
    def config(self) -> Any:
        return _encoder_config(self.vit)

    def forward(self, pixel_values: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.vit(pixel_values, *args, **kwargs)

    @property
    def _embedding_owner(self) -> nn.Module:
        return getattr(self.vit, "model") if self._owner_is_inner else self.vit

    def _attention_layers(self) -> list[MosaicCausalRoPESelfAttention]:
        return [
            layer.attention.attention
            for layer in self._embedding_owner.encoder.layer
        ]

    def stream_init(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, MosaicStreamState]:
        """Encode one full history Mosaic and return its CLS plus KV state."""
        if pixel_values.ndim != 4:
            raise ValueError("stream_init expects full Mosaic pixels shaped (B, C, H, W)")
        embedding_output = self._embedding_owner.embeddings(
            pixel_values, interpolate_pos_encoding=True
        )
        for attention in self._attention_layers():
            attention.last_patch_keys = None
            attention.last_patch_values = None
            attention.capture_stream_state = True
        try:
            encoder_output = self._embedding_owner.encoder(embedding_output)
        finally:
            for attention in self._attention_layers():
                attention.capture_stream_state = False
        sequence = getattr(encoder_output, "last_hidden_state", None)
        if sequence is None:
            sequence = encoder_output[0]
        output_hidden = self._embedding_owner.layernorm(sequence)
        keys = [attention.last_patch_keys for attention in self._attention_layers()]
        values = [attention.last_patch_values for attention in self._attention_layers()]
        if any(key is None for key in keys) or any(value is None for value in values):
            raise RuntimeError("causal tactile stream initialization did not capture KV")
        cls = output_hidden[:, 0]
        state = MosaicStreamState(
            patch_keys=[key for key in keys if key is not None],
            patch_values=[value for value in values if value is not None],
            # The stream starts a fresh CLS query for every update.  Keeping
            # the embedding-stage CLS seed is equivalent to the full ViT's
            # first-layer input; keeping the previous final CLS would apply
            # layer norms and MLPs twice on the next update.
            cls_state=embedding_output[:, : self.prefix_length].detach(),
            next_slot=HISTORY_FRAMES,
        )
        return cls, state

    def _current_cells(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim == 5:
            if pixels.size(1) != 2:
                raise ValueError("stream_step cells must contain exactly left and right images")
            return pixels
        if pixels.ndim != 4:
            raise ValueError("stream_step expects (B,2,C,cell,cell) or full Mosaic pixels")
        if pixels.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"full Mosaic must be {self.img_size}x{self.img_size}, got {tuple(pixels.shape[-2:])}"
            )
        start = self.img_size - self.cell_size
        left = pixels[:, :, start : start + self.cell_size, self.cell_size : 2 * self.cell_size]
        right = pixels[:, :, start : start + self.cell_size, 3 * self.cell_size :]
        return torch.stack((left, right), dim=1)

    def _embed_current_cells(self, cells: torch.Tensor, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embedding_owner.embeddings
        if not isinstance(embeddings, MosaicFactorizedRoPEEmbeddings):
            raise TypeError("causal stream expects MosaicFactorizedRoPEEmbeddings")
        batch, fingers = int(cells.size(0)), int(cells.size(1))
        if fingers != 2:
            raise ValueError("causal stream expects left and right cells")
        flat = cells.reshape(batch * fingers, *cells.shape[2:])
        patch_tokens = embeddings.embeddings.patch_embeddings(
            flat, interpolate_pos_encoding=True
        )
        patch_tokens = patch_tokens.reshape(batch, fingers * patch_tokens.size(1), -1)
        finger_ids = torch.arange(2, device=patch_tokens.device).repeat_interleave(
            patch_tokens.size(1) // 2
        )
        patch_tokens = patch_tokens + embeddings.finger_gate * embeddings.finger_emb(
            finger_ids
        ).unsqueeze(0).to(dtype=patch_tokens.dtype)
        cell_grid = self.cell_size // self.patch_size
        rows = torch.arange(cell_grid, device=patch_tokens.device)
        cols = torch.arange(cell_grid, device=patch_tokens.device)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        local = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
        patch_coords = torch.cat(
            (
                torch.cat(
                    (
                        torch.full((local.size(0), 1), int(slot), device=local.device),
                        local,
                    ),
                    dim=1,
                ),
                torch.cat(
                    (
                        torch.full((local.size(0), 1), int(slot), device=local.device),
                        local,
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        ).to(dtype=torch.long)
        if self.canonicalize_right:
            local_width = cell_grid - 1
            patch_coords[:, 2] = torch.where(
                torch.arange(2, device=patch_coords.device)
                .repeat_interleave(local.size(0))
                .to(torch.bool),
                local_width - patch_coords[:, 2],
                patch_coords[:, 2],
            )
        return patch_tokens, patch_coords

    @torch.no_grad()
    def stream_step(
        self, pixels: torch.Tensor, state: MosaicStreamState
    ) -> tuple[torch.Tensor, MosaicStreamState]:
        """Compute only current L/R cells using the cached historical K/V."""
        cells = self._current_cells(pixels)
        current, patch_coords = self._embed_current_cells(cells, state.next_slot)
        prefix = state.cls_state
        new_keys: list[torch.Tensor] = []
        new_values: list[torch.Tensor] = []
        for index, layer in enumerate(self._embedding_owner.encoder.layer):
            attention = layer.attention.attention
            if not isinstance(attention, MosaicCausalRoPESelfAttention):
                raise TypeError("causal stream found an unexpected attention wrapper")
            query_input = torch.cat((prefix, current), dim=1)
            query_norm = layer.layernorm_before(query_input)
            current_norm = query_norm[:, self.prefix_length :]
            query_coords = torch.cat(
                (
                    torch.zeros(
                        (self.prefix_length, 3),
                        device=current.device,
                        dtype=torch.long,
                    ),
                    patch_coords,
                ),
                dim=0,
            )
            context, current_key, current_value = attention.forward_incremental(
                query_norm,
                state.patch_keys[index],
                state.patch_values[index],
                current_norm,
                query_coords,
                patch_coords,
            )
            hidden = query_input + context
            layer_output = layer.output(
                layer.intermediate(layer.layernorm_after(hidden)), hidden
            )
            prefix, current = (
                layer_output[:, : self.prefix_length],
                layer_output[:, self.prefix_length :],
            )
            new_keys.append(torch.cat((state.patch_keys[index], current_key.detach()), dim=2))
            new_values.append(torch.cat((state.patch_values[index], current_value.detach()), dim=2))
        cls = self._embedding_owner.layernorm(prefix)[:, 0]
        next_state = MosaicStreamState(
            patch_keys=new_keys,
            patch_values=new_values,
            cls_state=state.cls_state,
            next_slot=state.next_slot + 1,
        )
        return cls, next_state


class MosaicPosembEncoder(nn.Module):
    """``jepa.encoder`` wrapper: same ViT forward, mosaic ids on patch tokens."""

    def __init__(
        self,
        vit: nn.Module,
        *,
        img_size: int,
        patch_size: int,
        mode: str = "finger_slot",
    ) -> None:
        super().__init__()
        embedding_owner, embeddings = _resolve_vit_embeddings(vit)
        hidden = _vit_hidden_size(embedding_owner)
        img_size, patch_size = _encoder_image_patch_size(
            embedding_owner, img_size, patch_size
        )
        mosaic_patch_ids(img_size, patch_size)
        config = _encoder_config(embedding_owner)
        num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)
        embedding_owner.embeddings = MosaicViTEmbeddings(
            embeddings,
            hidden_size=hidden,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            mode=mode,
        )
        self.vit = vit
        self.mode = parse_mosaic_posemb(mode)

    @property
    def config(self) -> Any:
        return _encoder_config(self.vit)

    def forward(self, pixel_values: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        # Vendor JEPA passes interpolate_pos_encoding=True. Do not inject a
        # different default than HuggingFace ViTModel (None).
        return self.vit(pixel_values, *args, **kwargs)


def wrap_vit_mosaic_posemb(
    encoder: nn.Module,
    *,
    img_size: int,
    patch_size: int,
    mode: str | bool = "finger_slot",
    enabled: bool = True,
) -> nn.Module:
    """Replace a HuggingFace ViT with a mosaic positional wrapper when enabled."""
    chosen = parse_mosaic_posemb(mode)
    if not enabled or chosen == "off" or isinstance(encoder, MosaicPosembEncoder):
        return encoder
    if chosen in {"factorized_rope", "spiral_rope", "canonical_spiral_rope"}:
        if isinstance(encoder, MosaicFactorizedRoPEEncoder):
            return encoder
        return MosaicFactorizedRoPEEncoder(
            encoder,
            img_size=int(img_size),
            patch_size=int(patch_size),
            rope_kind=(
                "three_axis" if chosen == "factorized_rope" else "spatiotemporal_spiral"
            ),
            canonicalize_right=chosen == "canonical_spiral_rope",
        )
    return MosaicPosembEncoder(
        encoder,
        img_size=int(img_size),
        patch_size=int(patch_size),
        mode=chosen,
    )


def wrap_vit_mosaic_cell_mask(
    encoder: nn.Module,
    *,
    img_size: int,
    patch_size: int,
    ratio: float | int | str = 0.5,
    mask_eval: bool = False,
    enabled: bool = True,
) -> nn.Module:
    """Prune random complete Mosaic cells after the ViT PatchEmbed stage."""
    chosen = parse_mosaic_cell_mask_ratio(ratio)
    if not enabled or chosen == 0.0 or isinstance(encoder, MosaicCellMaskEncoder):
        return encoder
    return MosaicCellMaskEncoder(
        encoder,
        img_size=int(img_size),
        patch_size=int(patch_size),
        ratio=chosen,
        mask_eval=bool(mask_eval),
    )


def wrap_vit_mosaic_causal(
    encoder: nn.Module,
    *,
    img_size: int,
    patch_size: int,
    rope_theta: float = 10_000.0,
    rope_kind: str = "three_axis",
    canonicalize_right: bool = False,
    spiral_directions: int = SPIRAL_DIRECTIONS,
    enabled: bool = True,
) -> nn.Module:
    """Use slot-causal tactile attention with a streamable KV-cache path.

    ``rope_kind`` is applied inside the causal wrapper so a positional design
    and causal attention are one coherent encoder, rather than two wrappers
    competing to replace the HuggingFace attention modules.
    """
    if not enabled or isinstance(encoder, MosaicCausalEncoder):
        return encoder
    return MosaicCausalEncoder(
        encoder,
        img_size=int(img_size),
        patch_size=int(patch_size),
        rope_theta=float(rope_theta),
        rope_kind=rope_kind,
        canonicalize_right=bool(canonicalize_right),
        spiral_directions=int(spiral_directions),
    )


def _encoder_image_patch_size(
    encoder: nn.Module, img_size: int, patch_size: int
) -> tuple[int, int]:
    """Resolve a patch size and a valid mosaic input size.

    Some pretrained ViTs advertise a native size such as 518 (37x37
    patches) but are evaluated here at 224 (16x16 patches).  Prefer the
    configured patch size, while using the requested runtime image size when
    the native size cannot form the required 4x4 mosaic grid.
    """
    config = _encoder_config(encoder)
    image = getattr(config, "image_size", None) if config is not None else None
    patch = getattr(config, "patch_size", None) if config is not None else None
    if isinstance(image, (tuple, list)):
        image = image[0]
    if isinstance(patch, (tuple, list)):
        patch = patch[0]
    resolved_patch = int(patch if patch is not None else patch_size)
    resolved_image = int(image if image is not None else img_size)
    if (
        resolved_image <= 0
        or resolved_image % resolved_patch != 0
        or (resolved_image // resolved_patch) % GRID_ROWS != 0
        or (resolved_image // resolved_patch) % GRID_COLS != 0
    ):
        resolved_image = int(img_size)
    return resolved_image, resolved_patch


def _vit_hidden_size(vit: nn.Module) -> int:
    config = _encoder_config(vit)
    hidden = getattr(config, "hidden_size", None) if config is not None else None
    if hidden is None:
        raise TypeError("mosaic_posemb needs encoder.config.hidden_size")
    return int(hidden)


def _encoder_config(encoder: nn.Module) -> Any:
    """Return config from a ViT or a thin wrapper exposing ``.model``."""
    config = getattr(encoder, "config", None)
    if config is not None:
        return config
    inner = getattr(encoder, "model", None)
    return getattr(inner, "config", None)


def _resolve_vit_embeddings(encoder: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Find the embedding owner for a ViT or DINO wrapper.

    ``DinoV2WithRegistersEncoder`` keeps the HuggingFace model in ``.model``
    and removes register tokens after its forward pass. Decorating that inner
    model preserves the wrapper's token contract while adding mosaic ids only
    to patch tokens.
    """
    embeddings = getattr(encoder, "embeddings", None)
    if embeddings is not None:
        return encoder, embeddings
    inner = getattr(encoder, "model", None)
    embeddings = getattr(inner, "embeddings", None)
    if embeddings is not None:
        return inner, embeddings
    raise TypeError(
        "mosaic_posemb requires a ViT with .embeddings or .model.embeddings"
    )


def cell_index(
    finger: int,
    slot: int,
    *,
    grid_cols: int = GRID_COLS,
    side_cols: int = SIDE_COLS,
) -> int:
    """Raster index of one mosaic cell (row-major 4x4)."""
    row, local_col = divmod(int(slot), int(side_cols))
    col = int(finger) * int(side_cols) + local_col
    return row * int(grid_cols) + col


def run_wrap_readout_ablation(
    *,
    mosaic_posemb: bool,
    seed: int = 0,
    steps: int = 200,
    batch_size: int = 64,
    dim: int = 32,
    lr: float = 5e-3,
) -> dict[str, float]:
    """CPU proxy: query L2 identity after shuffling the 4x4 raster each batch.

    Keys are frozen 2D raster embeddings plus optional finger/slot ids. The
    query is L2's finger/slot (treatment) or a free vector (2D-only). Lower
    MSE with ``mosaic_posemb=True`` means those ids retrieve the cell. Not a
    JEPA score.
    """
    torch.manual_seed(int(seed))
    n_cells = GRID_ROWS * GRID_COLS
    pos_2d = nn.Embedding(n_cells, dim)
    nn.init.normal_(pos_2d.weight, std=0.02)
    pos_2d.weight.requires_grad_(False)
    finger_emb = nn.Embedding(N_FINGERS, dim)
    slot_emb = nn.Embedding(HISTORY_FRAMES, dim)
    free_query = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(free_query, std=0.02)
    if mosaic_posemb:
        nn.init.normal_(finger_emb.weight, std=0.02)
        nn.init.normal_(slot_emb.weight, std=0.02)
        params: list[nn.Parameter] = list(finger_emb.parameters()) + list(
            slot_emb.parameters()
        )
    else:
        nn.init.zeros_(finger_emb.weight)
        nn.init.zeros_(slot_emb.weight)
        finger_emb.weight.requires_grad_(False)
        slot_emb.weight.requires_grad_(False)
        params = [free_query]
    optim = torch.optim.Adam(params, lr=float(lr))
    finger_ids, slot_ids = mosaic_patch_ids(GRID_COLS, patch_size=1)
    target_cell = cell_index(0, 2)
    scale = dim**-0.5
    last = 0.0

    def _batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        content = torch.randn(n, n_cells, dim)
        perm = torch.stack([torch.randperm(n_cells) for _ in range(n)])
        raster_pos = torch.arange(n_cells).unsqueeze(0).expand(n, -1)
        placed = content.gather(1, perm.unsqueeze(-1).expand(-1, -1, dim))
        keys = pos_2d(raster_pos)
        keys = keys + finger_emb(finger_ids[perm]) + slot_emb(slot_ids[perm])
        if mosaic_posemb:
            query = finger_emb.weight[0] + slot_emb.weight[2]
        else:
            query = free_query
        scores = torch.matmul(keys, query) * scale
        attn = torch.softmax(scores, dim=-1)
        pred = torch.matmul(attn.unsqueeze(1), placed).squeeze(1)
        return pred, content[:, target_cell]

    for _ in range(int(steps)):
        pred, target = _batch(int(batch_size))
        loss = torch.nn.functional.mse_loss(pred, target)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        last = float(loss.detach())
    with torch.no_grad():
        pred, target = _batch(256)
        eval_loss = float(torch.nn.functional.mse_loss(pred, target))
    return {
        "mosaic_posemb": float(bool(mosaic_posemb)),
        "train_mse": last,
        "eval_mse": eval_loss,
        "seed": float(seed),
        "steps": float(steps),
    }
