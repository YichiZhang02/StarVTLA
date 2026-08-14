from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .core.action_dit import ActionDiT


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).float()
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Resize every mismatched axis using FastWAM's sequential 1D interpolation."""
    if tuple(src.shape) == target_shape:
        return src

    output = src.float()
    while output.ndim < len(target_shape):
        output = output.unsqueeze(0)
    while output.ndim > len(target_shape):
        if output.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank: source={tuple(src.shape)}, target={target_shape}."
            )
        output = output.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        if output.shape[dim] == new_size:
            continue
        permutation = [index for index in range(output.ndim) if index != dim] + [dim]
        inverse = [0] * output.ndim
        for index, original_dim in enumerate(permutation):
            inverse[original_dim] = index
        output = output.permute(*permutation).contiguous()
        output = _interpolate_last_dim(output, new_size)
        output = output.permute(*inverse).contiguous()

    if tuple(output.shape) != target_shape:
        raise ValueError(
            f"Interpolation produced {tuple(output.shape)} from {tuple(src.shape)}; "
            f"expected {target_shape}."
        )
    return output.to(dtype=src.dtype)


def build_action_dit_backbone_payload(
    video_expert: torch.nn.Module,
    action_expert: ActionDiT,
    *,
    apply_alpha_scaling: bool = True,
    source: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Interpolate a Wan2.2 Video DiT into the narrower FastWAM ActionDiT."""
    action_state = action_expert.state_dict()
    video_state = video_expert.state_dict()
    backbone_keys = ActionDiT.backbone_key_set(action_state)

    backbone_state_dict: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    for key in sorted(backbone_keys):
        if key not in video_state:
            raise ValueError(f"ActionDiT backbone key {key!r} is absent from the Wan2.2 Video DiT.")
        src = video_state[key]
        target = action_state[key]
        if tuple(src.shape) == tuple(target.shape):
            value = src
            copied += 1
        else:
            value = resize_tensor_to_shape(src, tuple(target.shape))
            if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
                alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                value = value.float() * alpha
            interpolated += 1
        backbone_state_dict[key] = value.detach().to(
            device="cpu", dtype=target.dtype
        ).contiguous()

    first_block = action_expert.blocks[0]
    payload = {
        "format_version": 1,
        "source": dict(source or {}),
        "policy": {
            "skip_prefixes": list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": {
            "hidden_dim": int(action_expert.hidden_dim),
            "ffn_dim": int(action_expert.ffn_dim),
            "num_layers": len(action_expert.blocks),
            "num_heads": int(action_expert.num_heads),
            "attn_head_dim": int(action_expert.attn_head_dim),
            "text_dim": int(action_expert.text_dim),
            "freq_dim": int(action_expert.freq_dim),
            "eps": float(first_block.norm3.eps),
        },
    }
    stats = {
        "copied": copied,
        "interpolated": interpolated,
        "skipped": len(action_state) - len(backbone_keys),
    }
    return payload, stats
