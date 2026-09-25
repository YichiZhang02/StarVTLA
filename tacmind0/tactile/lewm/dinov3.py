"""Frozen DINOv3 loader for visuo JEPA context (not TurboVLA training)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


DEFAULT_DINOV3_HUB = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def resolve_dinov3_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    text = str(path).strip()
    return text or None


def is_local_dinov3_snapshot(path: str | Path) -> bool:
    return Path(path).expanduser().exists()


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def cls_hidden_size(module: nn.Module, fallback: int | None = None) -> int:
    config = getattr(module, "config", None)
    for key in ("hidden_size", "embed_dim", "width"):
        value = getattr(config, key, None) if config is not None else None
        if value is not None:
            return int(value)
    if fallback is not None:
        return int(fallback)
    raise ValueError("cannot infer DINOv3 CLS size; pass embed_dim fallback or a stub with config.hidden_size")


def extract_cls(output: Any) -> torch.Tensor:
    """Return (N, D) CLS from a HuggingFace or DINOv3 forward output."""
    if hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state
        return hidden[:, 0]
    token = getattr(output, "x_norm_clstoken", None)
    if token is not None:
        return token
    if isinstance(output, dict):
        if "last_hidden_state" in output:
            return output["last_hidden_state"][:, 0]
        if "x_norm_clstoken" in output:
            return output["x_norm_clstoken"]
    if torch.is_tensor(output):
        if output.ndim == 3:
            return output[:, 0]
        if output.ndim == 2:
            return output
    raise TypeError(f"unsupported DINOv3 output type {type(output)!r}")


def _resolve_pretrained_source(resolved: str, *, allow_hf_download: bool) -> str:
    """Return a local snapshot directory or hub id for ``from_pretrained``."""
    if is_local_dinov3_snapshot(resolved):
        return resolved
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        if not allow_hf_download:
            raise FileNotFoundError(
                f"DINOv3 path {resolved!r} does not exist and huggingface_hub "
                "is not installed."
            ) from exc
        return resolved
    try:
        return snapshot_download(resolved, local_files_only=True)
    except Exception:
        if not allow_hf_download:
            raise FileNotFoundError(
                f"DINOv3 path {resolved!r} is not a local snapshot and is not "
                "in the HuggingFace cache. Set DINOV3_PATH to a local dir, or "
                f"allow_hf_download=true (hub id {DEFAULT_DINOV3_HUB})."
            ) from None
        import os

        kwargs: dict[str, Any] = {"local_files_only": False}
        endpoint = os.environ.get("HF_ENDPOINT", "").strip()
        if endpoint:
            kwargs["endpoint"] = endpoint
        return snapshot_download(resolved, **kwargs)


def load_frozen_dinov3(
    path: str | Path,
    *,
    allow_hf_download: bool = False,
) -> nn.Module:
    """Load a local snapshot or HuggingFace hub DINOv3/ViT and freeze it."""
    resolved = resolve_dinov3_path(path)
    if not resolved:
        raise ValueError("dinov3_path is required to load a frozen vision encoder")
    source = _resolve_pretrained_source(resolved, allow_hf_download=allow_hf_download)
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError("transformers is required to load DINOv3") from exc
    local_only = is_local_dinov3_snapshot(source)
    kwargs: dict[str, Any] = {"local_files_only": local_only}
    try:
        model = AutoModel.from_pretrained(source, trust_remote_code=True, **kwargs)
    except TypeError:
        model = AutoModel.from_pretrained(source, **kwargs)
    return freeze_module(model)


def encode_cls(module: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    """Encode ``(N, 3, H, W)`` RGB ImageNet tensors to CLS ``(N, D)``."""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"expected (N, 3, H, W), got {tuple(frames.shape)}")
    param = next(module.parameters(), None)
    if param is not None:
        frames = frames.to(device=param.device, dtype=param.dtype)
    attempts = (
        lambda: module(pixel_values=frames, interpolate_pos_encoding=True),
        lambda: module(frames, interpolate_pos_encoding=True),
        lambda: module(pixel_values=frames),
        lambda: module(frames),
    )
    last_error: BaseException | None = None
    for attempt in attempts:
        try:
            return extract_cls(attempt())
        except TypeError as exc:
            last_error = exc
    raise TypeError(
        f"DINOv3 encoder rejected input shape {tuple(frames.shape)}"
    ) from last_error
