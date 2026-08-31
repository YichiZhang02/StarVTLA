"""Checkpoint namespace handling and explicit 3D position interpolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def read_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.is_dir():
        candidates = [path / name for name in ("model.safetensors", "pytorch_model.bin", "checkpoint.pth")]
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state: Any = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "module"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} does not contain a state dict")
    return {str(key): value for key, value in state.items() if isinstance(value, torch.Tensor)}


def interpolate_video_position_embedding(
    position: torch.Tensor,
    *,
    source_grid: tuple[int, int, int],
    target_grid: tuple[int, int, int],
    has_cls: bool,
) -> torch.Tensor:
    """Trilinearly interpolate a flattened ``[T,H,W]`` learned embedding."""
    squeeze = False
    if position.ndim == 2:
        position = position.unsqueeze(0)
        squeeze = True
    if position.ndim != 3 or position.shape[0] != 1:
        raise ValueError(f"unsupported position embedding shape {tuple(position.shape)}")
    prefix = position[:, :1] if has_cls else position[:, :0]
    spatial = position[:, 1:] if has_cls else position
    expected = source_grid[0] * source_grid[1] * source_grid[2]
    if spatial.shape[1] != expected:
        raise ValueError(
            f"source_grid {source_grid} has {expected} positions, checkpoint contains {spatial.shape[1]}"
        )
    spatial = spatial.reshape(1, *source_grid, spatial.shape[-1]).permute(0, 4, 1, 2, 3)
    spatial = F.interpolate(spatial.float(), size=target_grid, mode="trilinear", align_corners=False)
    spatial = spatial.permute(0, 2, 3, 4, 1).reshape(1, -1, position.shape[-1]).to(position.dtype)
    result = torch.cat([prefix, spatial], dim=1)
    return result.squeeze(0) if squeeze else result


def load_filtered_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    source: str,
) -> dict[str, Any]:
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    shape_mismatch = {
        key: [list(value.shape), list(model_state[key].shape)]
        for key, value in state.items()
        if key in model_state and model_state[key].shape != value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    report = {
        "source": source,
        "loaded_tensors": len(compatible),
        "missing_keys": list(missing),
        "unexpected_keys": [key for key in state if key not in model_state],
        "shape_mismatch": shape_mismatch,
    }
    if not compatible:
        raise ValueError(f"Checkpoint {source} did not contain any compatible model tensors")
    return report
