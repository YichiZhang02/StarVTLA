"""Canonical Flux tactile encodings shared by acquisition and recording."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

PHYSICAL_SCALE = 1000.0
DEFORM_OFFSET = 30000.0

def encode_tactile_u16(depth: NDArray, deform: NDArray) -> NDArray[np.uint16]:
    """Pack Flux depth/deformation fields using ``tactile_u16_fixed_v1``."""
    depth_arr = np.asarray(depth, dtype=np.float32)
    deform_arr = np.asarray(deform, dtype=np.float32)
    if depth_arr.ndim != 2 or deform_arr.shape != (*depth_arr.shape, 2):
        raise ValueError(
            f"Flux tactile shape mismatch: depth={depth_arr.shape}, deform={deform_arr.shape}"
        )
    if not np.isfinite(depth_arr).all() or not np.isfinite(deform_arr).all():
        raise ValueError("Flux tactile frame contains NaN or Inf")

    packed = np.empty((*depth_arr.shape, 3), dtype=np.uint16)
    packed[..., 0] = np.clip(
        np.rint(depth_arr * PHYSICAL_SCALE), 0, np.iinfo(np.uint16).max
    ).astype(np.uint16)
    packed[..., 1:] = np.clip(
        np.rint(deform_arr * PHYSICAL_SCALE + DEFORM_OFFSET),
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    return np.ascontiguousarray(packed)


def decode_tactile_u16(packed: NDArray) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Decode ``tactile_u16_fixed_v1`` into Flux physical fields."""
    packed_arr = np.asarray(packed)
    if packed_arr.dtype != np.uint16 or packed_arr.ndim != 3 or packed_arr.shape[-1] != 3:
        raise ValueError(
            f"Expected HWC uint16 tactile frame, got shape={packed_arr.shape} dtype={packed_arr.dtype}"
        )
    value = packed_arr.astype(np.float32)
    depth = value[..., 0] / PHYSICAL_SCALE
    deform = (value[..., 1:] - DEFORM_OFFSET) / PHYSICAL_SCALE
    return depth, deform
