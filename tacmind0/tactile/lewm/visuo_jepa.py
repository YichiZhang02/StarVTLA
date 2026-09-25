"""Vendor JEPA with a frozen DINOv3 visual context.

Tactile mosaic CLS stays ``info['emb']`` (JEPA target / SIGReg). DINOv3 CLS
is either added inside ``predict`` (``vision_context=residual``) or
concatenated onto the action encoder / AdaLN (``vision_context=action``).
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from tacmind0.tactile.lewm.dinov3 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    cls_hidden_size,
    encode_cls,
    extract_cls,
    freeze_module,
    load_frozen_dinov3,
    resolve_dinov3_path,
)

DEFAULT_EMBED_DIM = 192
VISION_CENTER_MOMENTUM = 0.1
# WM-28 checkpoints predate these buffers. Safe to leave at init when drop=0.
LEGACY_OPTIONAL_STATE_KEYS = frozenset(
    {
        "vision_mean",
        "missing_tokens.vision_null",
        "missing_tokens.action_null",
    }
)


def load_compatible_state_dict(module: nn.Module, state_dict: dict[str, Any]) -> list[str]:
    """Load a DualEncoderJEPA checkpoint; allow buffers added after WM-28."""
    incompatible = module.load_state_dict(state_dict, strict=False)
    extra_missing = [
        key
        for key in incompatible.missing_keys
        if key not in LEGACY_OPTIONAL_STATE_KEYS
    ]
    if extra_missing:
        raise RuntimeError(
            "checkpoint is missing unexpected keys: " + ", ".join(extra_missing)
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint has unexpected keys: "
            + ", ".join(incompatible.unexpected_keys)
        )
    return list(incompatible.missing_keys)


def _as_batched_image(image: Any) -> tuple[torch.Tensor, bool]:
    tensor = torch.as_tensor(image)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0), True
    if tensor.ndim == 4:
        return tensor, False
    raise ValueError(
        f"expected (H,W,3) or (T,H,W,3)/(T,3,H,W), got {tuple(tensor.shape)}"
    )


def _nchw_unit_interval(frames: torch.Tensor, *, kind: str) -> torch.Tensor:
    tensor = torch.as_tensor(frames)
    if tensor.ndim != 4:
        raise ValueError(f"expected 4D {kind} frames, got {tuple(tensor.shape)}")
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    if tensor.shape[1] != 3:
        raise ValueError(f"expected 3-channel {kind}, got {tuple(tensor.shape)}")
    as_uint8 = tensor.dtype == torch.uint8
    out = tensor.float()
    if as_uint8 or float(out.max()) > 1.5:
        out = out.div(255.0)
    return out


def bgr_to_rgb_imagenet(frames: torch.Tensor) -> torch.Tensor:
    """Convert BGR uint8 HWC or float CHW into RGB ImageNet NCHW.

    Accepts ``(N, H, W, 3)``, ``(N, 3, H, W)``, or the same with a leading
    batch/time dim flattened by the caller.
    """
    rgb = _nchw_unit_interval(frames, kind="vision")[:, [2, 1, 0], ...]
    mean = rgb.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = rgb.new_tensor(IMAGENET_STD)[:, None, None]
    return (rgb - mean) / std


def resize_nchw(frames: torch.Tensor, img_size: int) -> torch.Tensor:
    size = int(img_size)
    if frames.shape[-2] == size and frames.shape[-1] == size:
        return frames
    return F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)


MOSAIC_GRID = 4
MOSAIC_LAYOUTS = ("full", "current_lr", "last_row")
VISION_VIEWS = ("agentview", "wrist")
VISION_VIEW_ALIASES = {
    "agentview": ("agentview",),
    "wrist": ("wrist",),
    "agentview_wrist": ("agentview", "wrist"),
    "agentview+wrist": ("agentview", "wrist"),
}
VIEW_TO_BATCH_KEY = {"agentview": "vision", "wrist": "wrist"}
VISION_FUSES = ("mean", "concat")
VISION_HISTORIES = ("all", "current")
VISION_PRECISIONS = ("bf16", "fp32")
VISION_PRECISION_ALIASES = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "16": "bf16",
    "fp32": "fp32",
    "float32": "fp32",
    "32": "fp32",
}
VISION_COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")
TACTILE_STREAM_COMPILE_MODES = ("default",)
VISION_HISTORY_ALIASES = {
    "all": "all",
    "full": "all",
    "window": "all",
    "current": "current",
    "last": "current",
    "current_only": "current",
}


def set_tactile_encoder_trainable(jepa: nn.Module, trainable: bool = True) -> None:
    """Set the JEPA tactile encoder's gradient and mode explicitly.

    ``jepa.encoder`` is the trainable tactile mosaic ViT.  The camera DINO
    encoder is a separate ``vision_encoder`` and is intentionally not touched
    here.  Making this contract explicit prevents a future wrapper or
    checkpoint-loading path from silently freezing tactile parameters.
    """
    encoder = getattr(jepa, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise TypeError("JEPA must expose a torch.nn.Module encoder")
    enabled = bool(trainable)
    # The vendor HuggingFace encoder has its own freeze_backbone switch and
    # uses it to wrap forward() in torch.no_grad().  Changing only
    # Parameter.requires_grad is therefore insufficient, especially after
    # the mosaic positional/readout wrappers have been added.
    for module in encoder.modules():
        if hasattr(module, "freeze_backbone"):
            module.freeze_backbone = not enabled
    for parameter in encoder.parameters():
        parameter.requires_grad_(enabled)
    if enabled:
        encoder.train()
    else:
        encoder.eval()
FORCE_CONTEXTS = ("off", "residual", "action")
FORCE_CONTEXT_ALIASES = {
    "false": "off",
    "off": "off",
    "none": "off",
    "0": "off",
    "residual": "residual",
    "predict": "residual",
    "context": "residual",
    "action": "action",
    "act": "action",
    "concat_action": "action",
}
FORCE_REPRS = ("raw", "magdir")
FORCE_REPR_ALIASES = {
    "false": "raw",
    "off": "raw",
    "none": "raw",
    "0": "raw",
    "raw": "raw",
    "wrench": "raw",
    "6d": "raw",
    "magdir": "magdir",
    "mag_dir": "magdir",
    "logdir": "magdir",
    "unit": "magdir",
}
FORCE_MAGDIR_DIM = 8
DEFAULT_PROPRIO_DIM = 8
VISION_CONTEXTS = ("off", "residual", "action")
VISION_CONTEXT_ALIASES = {
    "false": "off",
    "off": "off",
    "none": "off",
    "0": "off",
    "residual": "residual",
    "predict": "residual",
    "add": "residual",
    "true": "residual",
    "action": "action",
    "act": "action",
    "concat_action": "action",
    "adaln": "action",
}
PROPRIO_CONTEXTS = ("off", "action")
PROPRIO_CONTEXT_ALIASES = {
    "false": "off",
    "off": "off",
    "none": "off",
    "0": "off",
    "action": "action",
    "act": "action",
    "concat": "action",
    "concat_action": "action",
}
MISSING_VISION = ("skip", "null")
MISSING_VISION_ALIASES = {
    "skip": "skip",
    "zero": "skip",
    "off": "skip",
    "none": "skip",
    "residual_off": "skip",
    "null": "null",
    "token": "null",
    "learned": "null",
}
MISSING_ACTION = ("zero", "skip_pred", "null")
MISSING_ACTION_ALIASES = {
    "zero": "zero",
    "zeros": "zero",
    "fill": "zero",
    "skip_pred": "skip_pred",
    "skip": "skip_pred",
    "skip_loss": "skip_pred",
    "null": "null",
    "token": "null",
    "learned": "null",
}


def parse_mosaic_layout(value: str | None) -> str:
    layout = "full" if value is None else str(value).strip().lower()
    if layout not in MOSAIC_LAYOUTS:
        raise ValueError(
            f"mosaic_layout must be one of {MOSAIC_LAYOUTS}, got {value!r}"
        )
    return layout


def parse_vision_views(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return ("agentview",)
    if isinstance(value, str):
        key = value.strip().lower().replace(" ", "")
        if key in VISION_VIEW_ALIASES:
            return VISION_VIEW_ALIASES[key]
        parts = tuple(part for part in key.replace("+", ",").split(",") if part)
    else:
        parts = tuple(str(part).strip().lower() for part in value if str(part).strip())
    if not parts:
        raise ValueError("vision_views must name at least one camera")
    unknown = [part for part in parts if part not in VISION_VIEWS]
    if unknown:
        raise ValueError(
            f"vision_views must be agentview, wrist, or agentview_wrist, got {value!r}"
        )
    seen: list[str] = []
    for part in parts:
        if part not in seen:
            seen.append(part)
    return tuple(seen)


def parse_vision_fuse(value: str | None) -> str:
    fuse = "mean" if value is None else str(value).strip().lower()
    if fuse not in VISION_FUSES:
        raise ValueError(f"vision_fuse must be one of {VISION_FUSES}, got {value!r}")
    return fuse


def parse_vision_context(value: bool | str | None) -> str:
    """``residual`` adds DINO CLS in ``predict``; ``action`` concats onto AdaLN."""
    if value is None or value is False:
        return "off"
    if value is True:
        return "residual"
    key = str(value).strip().lower().replace("-", "_")
    if key not in VISION_CONTEXT_ALIASES:
        raise ValueError(
            f"vision_context must be one of {VISION_CONTEXTS}, got {value!r}"
        )
    return VISION_CONTEXT_ALIASES[key]


def resolve_vision_context(
    vision_context: bool | str | None,
    vision_residual: bool = True,
) -> str:
    """Hydra may pass both the bool alias and the site name."""
    if vision_context is None:
        return "residual" if bool(vision_residual) else "off"
    return parse_vision_context(vision_context)


def parse_drop_p(value: bool | str | float | int | None, *, name: str) -> float:
    """Bernoulli probability that a whole window drops that modality."""
    if value is None or value is False:
        return 0.0
    if value is True:
        return 1.0
    try:
        prob = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}") from exc
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}")
    return prob


def parse_missing_vision(value: bool | str | None) -> str:
    """How dropped cameras enter residual/AdaLN: skip-add, or a learned null."""
    if value is None or value is False:
        return "skip"
    if value is True:
        return "null"
    key = str(value).strip().lower().replace("-", "_")
    if key not in MISSING_VISION_ALIASES:
        raise ValueError(
            f"missing_vision must be one of {MISSING_VISION}, got {value!r}"
        )
    return MISSING_VISION_ALIASES[key]


def parse_missing_action(value: bool | str | None) -> str:
    """How dropped actions enter AdaLN / pred_loss."""
    if value is None or value is False:
        return "skip_pred"
    if value is True:
        return "null"
    key = str(value).strip().lower().replace("-", "_")
    if key not in MISSING_ACTION_ALIASES:
        raise ValueError(
            f"missing_action must be one of {MISSING_ACTION}, got {value!r}"
        )
    return MISSING_ACTION_ALIASES[key]


def parse_vision_history(value: bool | str | None) -> str:
    """``all`` encodes every camera in the JEPA window; ``current`` only the last context step."""
    if value is None or value is False:
        return "all"
    if value is True:
        return "current"
    key = str(value).strip().lower().replace("-", "_")
    if key not in VISION_HISTORY_ALIASES:
        raise ValueError(
            f"vision_history must be one of {VISION_HISTORIES}, got {value!r}"
        )
    return VISION_HISTORY_ALIASES[key]


def parse_vision_precision(value: str | None) -> str:
    """Choose the frozen camera encoder's CUDA autocast precision."""
    key = "bf16" if value is None else str(value).strip().lower().replace("-", "")
    if key not in VISION_PRECISION_ALIASES:
        raise ValueError(
            f"vision_precision must be one of {VISION_PRECISIONS}, got {value!r}"
        )
    return VISION_PRECISION_ALIASES[key]


def parse_vision_compile_mode(value: str | None) -> str:
    """Validate the optional torch.compile mode for frozen camera DINO."""
    key = "reduce-overhead" if value is None else str(value).strip().lower()
    if key not in VISION_COMPILE_MODES:
        raise ValueError(
            f"vision_compile_mode must be one of {VISION_COMPILE_MODES}, got {value!r}"
        )
    return key


def parse_tactile_stream_compile_mode(value: str | None) -> str:
    """Validate the safe compiled tactile streaming mode."""
    key = "default" if value is None else str(value).strip().lower()
    if key not in TACTILE_STREAM_COMPILE_MODES:
        raise ValueError(
            "tactile_stream_compile_mode must be one of "
            f"{TACTILE_STREAM_COMPILE_MODES}, got {value!r}"
        )
    return key


def vision_keys_for_views(views: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(VIEW_TO_BATCH_KEY[view] for view in views)


def parse_force_context(value: bool | str | None) -> str:
    if value is None or value is False:
        return "off"
    if value is True:
        return "residual"
    key = str(value).strip().lower()
    if key not in FORCE_CONTEXT_ALIASES:
        raise ValueError(
            f"force_context must be off, residual, or action, got {value!r}"
        )
    return FORCE_CONTEXT_ALIASES[key]


def force_bt6(force: Any) -> torch.Tensor:
    """Return rest-subtracted wrench as ``(B, T, 6)``."""
    tensor = torch.as_tensor(force).float()
    if tensor.ndim == 2 and tensor.size(-1) == 6:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 3 or tensor.size(-1) != 6:
        raise ValueError(f"expected force (B, T, 6), got {tuple(tensor.shape)}")
    return tensor


def parse_force_repr(value: bool | str | None) -> str:
    if value is None or value is False:
        return "raw"
    if value is True:
        return "magdir"
    key = str(value).strip().lower()
    if key not in FORCE_REPR_ALIASES:
        raise ValueError(f"force_repr must be raw or magdir, got {value!r}")
    return FORCE_REPR_ALIASES[key]


def force_feature_dim(force_repr: str | bool | None = "raw") -> int:
    if parse_force_repr(force_repr) == "magdir":
        return FORCE_MAGDIR_DIM
    return 6


def force_magdir_bt8(force: Any, *, mag_eps: float = 1e-6) -> torch.Tensor:
    """Per-finger ``log1p(||F||)`` plus unit vector: ``(B, T, 8)``.

    Applied to the incoming 6-D wrench (dataset z-score of the rest residual).
    Zero-magnitude fingers keep a zero unit vector.
    """
    wrench = force_bt6(force)
    parts: list[torch.Tensor] = []
    for start in (0, 3):
        vec = wrench[..., start : start + 3]
        mag = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        scale = mag.clamp_min(float(mag_eps))
        unit = torch.where(mag > float(mag_eps), vec / scale, torch.zeros_like(vec))
        parts.append(torch.cat([torch.log1p(mag), unit], dim=-1))
    return torch.cat(parts, dim=-1)


def parse_proprio_context(value: bool | str | None) -> str:
    if value is None or value is False:
        return "off"
    if value is True:
        return "action"
    key = str(value).strip().lower()
    if key not in PROPRIO_CONTEXT_ALIASES:
        raise ValueError(
            f"proprio_context must be off or action, got {value!r}"
        )
    return PROPRIO_CONTEXT_ALIASES[key]


def proprio_btd(proprio: Any) -> torch.Tensor:
    """Return proprioception as ``(B, T, D)``."""
    tensor = torch.as_tensor(proprio).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 3 or tensor.size(-1) < 1:
        raise ValueError(f"expected proprio (B, T, D), got {tuple(tensor.shape)}")
    return tensor


def concat_action_condition(
    action: Any, extra: Any, *, name: str
) -> torch.Tensor:
    """Concatenate a per-step condition onto the last action dim."""
    action_t = torch.as_tensor(action).float()
    extra_t = torch.as_tensor(extra).float()
    extra_t = extra_t.to(device=action_t.device, dtype=action_t.dtype)
    if action_t.ndim == 2:
        action_t = action_t.unsqueeze(1)
    if extra_t.ndim == 2:
        extra_t = extra_t.unsqueeze(1)
    if action_t.size(1) == 1 and extra_t.size(1) > 1:
        action_t = action_t.expand(extra_t.size(0), extra_t.size(1), action_t.size(-1))
    elif extra_t.size(1) == 1 and action_t.size(1) > 1:
        extra_t = extra_t.expand(action_t.size(0), action_t.size(1), extra_t.size(-1))
    if action_t.shape[:2] != extra_t.shape[:2]:
        raise ValueError(
            f"action {tuple(action_t.shape)} incompatible with "
            f"{name} {tuple(extra_t.shape)}"
        )
    return torch.cat([action_t, extra_t], dim=-1)


def parse_vision_force_gate(value: bool | str | None) -> bool:
    if value is None or value is False:
        return False
    if value is True:
        return True
    key = str(value).strip().lower()
    if key in ("true", "on", "gate", "contact", "1"):
        return True
    if key in ("false", "off", "none", "0"):
        return False
    raise ValueError(f"vision_force_gate must be true or false, got {value!r}")


def parse_on_off(value: bool | str | int | None, *, name: str) -> bool:
    if value is None or value is False:
        return False
    if value is True:
        return True
    if isinstance(value, bool):
        return bool(value)
    key = str(value).strip().lower()
    if key in ("true", "on", "1"):
        return True
    if key in ("false", "off", "none", "0"):
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def pred_infonce(
    pred_emb: Any,
    tgt_emb: Any,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """In-batch InfoNCE of pred vs stop-grad target, scaled by ``log(N)``.

    Each ``(batch, time)`` step is one query. The matching target step is
    the positive; other steps in the batch (other time, other episode) are
    negatives. Dividing by ``log(N)`` keeps a random match near 1 so λ
    stays comparable to ``force_pred_weight``.
    """
    pred = torch.as_tensor(pred_emb)
    tgt = torch.as_tensor(tgt_emb)
    if pred.shape != tgt.shape or pred.ndim != 3:
        raise ValueError(
            f"pred_infonce expects matching (B, T, D), got "
            f"{tuple(pred.shape)} vs {tuple(tgt.shape)}"
        )
    pred_n = F.normalize(pred.reshape(-1, pred.size(-1)).float(), dim=-1)
    tgt_n = F.normalize(tgt.detach().reshape(-1, tgt.size(-1)).float(), dim=-1)
    n_items = int(pred_n.size(0))
    logits = pred_n @ tgt_n.transpose(0, 1) / max(float(temperature), 1e-6)
    labels = torch.arange(n_items, device=pred_n.device)
    nce = F.cross_entropy(logits, labels)
    return nce / math.log(max(n_items, 2))


def pred_action_hinge(
    pred_true: Any,
    pred_shuffled: Any,
    target: Any,
    *,
    margin: float = 0.05,
    floor: float = 1e-8,
) -> torch.Tensor:
    """Angular hinge that forces the predictor to *use* the action.

    The evaluation metric ``delta_shuffle = MSE(shuffled) - MSE(true)`` is
    near zero whenever the predictor bypasses the action input.  A first
    MSE-ratio formulation (``relu(log1p(m) - log(MSE_shuf / MSE_true))``)
    proved degenerate (2026-09-15 A/B): the ratio is invariant to a global
    output scale, so the cheapest satisfying solution co-adapts the
    encoder to shrink the target scale and the predictor to amplify its
    output (pred/target std ratio ~6x, encoder eff-rank collapse,
    ``delta_wrong_future`` ~ 0, ``delta_zero_action`` < 0) — an amplifier,
    not a world model.

    This version is scale-free by construction and lives in direction
    space, matching how the eval metric is consumed (relative, per
    window):

    ``loss = relu(cos(pred_shuffled, target) - cos(pred_true, target) +
    margin)``

    The true-action prediction must align with the target at least
    ``margin`` cosine better than the shuffled-action prediction.
    Amplitude games cannot help: only moving the prediction *direction*
    does.  ``margin`` is in cosine units; the healthy TacBench-only
    baseline separates shuffled vs true predictions by a large angular
    gap, while a bypassed action path gives ~0 gap.  All inputs are
    ``(B, T, D)``; the margin must be in ``[0, 2)``.
    """
    pred_true = torch.as_tensor(pred_true)
    pred_shuffled = torch.as_tensor(pred_shuffled)
    target = torch.as_tensor(target)
    if pred_true.shape != pred_shuffled.shape or pred_true.shape != target.shape:
        raise ValueError(
            f"pred_action_hinge expects matching (B, T, D), got "
            f"{tuple(pred_true.shape)} / {tuple(pred_shuffled.shape)} / "
            f"{tuple(target.shape)}"
        )
    if float(margin) < 0.0 or float(margin) >= 2.0:
        raise ValueError(f"margin must be in [0, 2), got {margin}")
    if pred_true.numel() == 0:
        return pred_true.new_zeros(())
    dim = pred_true.size(-1)
    cos = lambda a, b: F.cosine_similarity(
        a.reshape(-1, dim), b.reshape(-1, dim), dim=-1
    ).reshape(a.shape[:-1])
    gap = cos(pred_shuffled, target) - cos(pred_true, target)
    return F.relu(gap.mean() + float(margin))


def align_cells_to_pred(
    cells: Any, pred_emb: torch.Tensor, n_preds: int
) -> torch.Tensor:
    """Drop the first ``n_preds`` mosaic-cell frames to match ``pred_emb``."""
    tensor = torch.as_tensor(cells)
    tensor = tensor.to(device=pred_emb.device, dtype=pred_emb.dtype)
    if tensor.size(0) != pred_emb.size(0):
        raise ValueError(
            f"cells batch {int(tensor.size(0))} != pred {int(pred_emb.size(0))}"
        )
    start = max(int(n_preds), 0)
    if tensor.size(1) == pred_emb.size(1) + start:
        return tensor[:, start:]
    if tensor.size(1) == pred_emb.size(1):
        return tensor
    raise ValueError(
        f"cells length {int(tensor.size(1))} != pred steps "
        f"{int(pred_emb.size(1))} (n_preds={n_preds})"
    )


def parse_force_pred(value: bool | str | None) -> bool:
    if value is None or value is False:
        return False
    if value is True:
        return True
    key = str(value).strip().lower()
    if key in ("true", "on", "pred", "output", "1"):
        return True
    if key in ("false", "off", "none", "0"):
        return False
    raise ValueError(f"force_pred must be true or false, got {value!r}")


def align_force_to_pred(
    force: Any, pred_emb: torch.Tensor, n_preds: int
) -> torch.Tensor:
    """Match JEPA pred steps: drop the first ``n_preds`` force frames."""
    wrench = force_bt6(force).to(device=pred_emb.device, dtype=pred_emb.dtype)
    start = max(int(n_preds), 0)
    if wrench.size(0) != pred_emb.size(0):
        raise ValueError(
            f"force batch {int(wrench.size(0))} != pred {int(pred_emb.size(0))}"
        )
    if wrench.size(1) == pred_emb.size(1) + start:
        return wrench[:, start:]
    if wrench.size(1) == pred_emb.size(1):
        return wrench
    raise ValueError(
        f"force length {int(wrench.size(1))} != pred steps "
        f"{int(pred_emb.size(1))} (n_preds={n_preds})"
    )


def crop_mosaic_nchw(frames: torch.Tensor, layout: str) -> torch.Tensor:
    """Keep the full 4x4, the last row, or current L7|R7, then fill ``img_size``.

    Cells are resized independently so bilinear does not mix L/R seams.
    ``current_lr`` is L7 (row 3, col 1) beside R7 (row 3, col 3).
    ``last_row`` is L6 L7 | R6 R7.
    """
    chosen = parse_mosaic_layout(layout)
    if chosen == "full":
        return frames
    height, width = int(frames.shape[-2]), int(frames.shape[-1])
    if height % MOSAIC_GRID != 0 or width % MOSAIC_GRID != 0:
        raise ValueError(
            f"mosaic crop expects a {MOSAIC_GRID}x{MOSAIC_GRID} grid, got {height}x{width}"
        )
    cell_h, cell_w = height // MOSAIC_GRID, width // MOSAIC_GRID
    cols = (1, 3) if chosen == "current_lr" else (0, 1, 2, 3)
    cells = [
        frames[..., 3 * cell_h : 4 * cell_h, col * cell_w : (col + 1) * cell_w]
        for col in cols
    ]
    out_w = width // len(cols)
    resized = [
        F.interpolate(cell, size=(height, out_w), mode="bilinear", align_corners=False)
        for cell in cells
    ]
    return torch.cat(resized, dim=-1)


CURRENT_RECON_CELL = 16


def current_lr_cells_nchw(
    frames: torch.Tensor, layout: str = "full"
) -> torch.Tensor:
    """Return current L7 and R7 cells as ``(..., 2, C, cell_h, cell_w)``.

    ``layout`` must match ``MosaicBgrToFloat``: after ``current_lr`` / ``last_row``
    crop the tensor is no longer a 4x4, so slicing the old L7|R7 window is wrong.
    """
    chosen = parse_mosaic_layout(layout)
    tensor = torch.as_tensor(frames)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim not in (4, 5):
        raise ValueError(
            f"expected mosaic (B, C, H, W) or (B, T, C, H, W), got {tuple(tensor.shape)}"
        )
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    if chosen == "current_lr":
        if width % 2 != 0:
            raise ValueError(f"current_lr mosaic width must be even, got {width}")
        mid = width // 2
        left = tensor[..., :, :mid]
        right = tensor[..., :, mid:]
        return torch.stack((left, right), dim=-4)
    if chosen == "last_row":
        if width % MOSAIC_GRID != 0:
            raise ValueError(
                f"last_row mosaic width must divide {MOSAIC_GRID}, got {width}"
            )
        col_w = width // MOSAIC_GRID
        left = tensor[..., :, 1 * col_w : 2 * col_w]
        right = tensor[..., :, 3 * col_w : 4 * col_w]
        return torch.stack((left, right), dim=-4)
    if height % MOSAIC_GRID != 0 or width % MOSAIC_GRID != 0:
        raise ValueError(
            f"current cells expect a {MOSAIC_GRID}x{MOSAIC_GRID} grid, got {height}x{width}"
        )
    cell_h, cell_w = height // MOSAIC_GRID, width // MOSAIC_GRID
    left = tensor[..., 3 * cell_h : 4 * cell_h, 1 * cell_w : 2 * cell_w]
    right = tensor[..., 3 * cell_h : 4 * cell_h, 3 * cell_w : 4 * cell_w]
    return torch.stack((left, right), dim=-4)


def current_pad_temporal_energy(
    pixels: torch.Tensor, layout: str = "full"
) -> torch.Tensor:
    """Per-step L2 of current L7|R7 change. Shape ``(B, T)``.

    Empty gel barely moves; marker/contact motion is large. This is a soft
    tactile weight, not a binary contact mask.
    """
    tensor = torch.as_tensor(pixels).float()
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(1)
    cells = current_lr_cells_nchw(tensor, layout=layout)
    flat = cells.flatten(start_dim=2)
    time = int(flat.size(1))
    energy = flat.new_zeros(flat.size(0), time)
    if time < 2:
        return energy
    delta = (flat[:, 1:] - flat[:, :-1]).pow(2).mean(dim=-1)
    energy[:, 1:] = delta
    energy[:, 0] = delta[:, 0]
    return energy


def _gel_cell_head(embed_dim: int) -> nn.Sequential:
    cell = CURRENT_RECON_CELL
    return nn.Sequential(
        nn.Linear(int(embed_dim), 256),
        nn.GELU(),
        nn.Linear(256, 2 * 3 * cell * cell),
    )


def _decode_gel_cells(head: nn.Module, emb: Any) -> torch.Tensor:
    tensor = torch.as_tensor(emb)
    if tensor.ndim != 3:
        raise ValueError(f"expected emb (B, T, D), got {tuple(tensor.shape)}")
    cell = CURRENT_RECON_CELL
    flat = torch.sigmoid(head(tensor))
    return flat.reshape(tensor.size(0), tensor.size(1), 2, 3, cell, cell)


def current_cell_targets(
    pixels: torch.Tensor,
    size: int = CURRENT_RECON_CELL,
    layout: str = "full",
) -> torch.Tensor:
    """Downsampled current L7|R7 in ``[0, 1]``, shape ``(B, T, 2, 3, S, S)``."""
    tensor = torch.as_tensor(pixels).float()
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(1)
    cells = current_lr_cells_nchw(tensor, layout=layout)
    batch, time, sides, channels, height, width = cells.shape
    flat = cells.reshape(batch * time * sides, channels, height, width)
    small = F.interpolate(flat, size=(int(size), int(size)), mode="bilinear", align_corners=False)
    return small.reshape(batch, time, sides, channels, int(size), int(size))


def resize_mosaic_nchw(frames: torch.Tensor, img_size: int) -> torch.Tensor:
    """Resize a square 4x4 mosaic per cell so bilinear does not mix seams."""
    size = int(img_size)
    height, width = int(frames.shape[-2]), int(frames.shape[-1])
    if height == size and width == size:
        return frames
    if height != width:
        raise ValueError(f"mosaic resize expects square frames, got {height}x{width}")
    if size % MOSAIC_GRID != 0:
        raise ValueError(f"img_size must be divisible by {MOSAIC_GRID}, got {size}")
    if height % MOSAIC_GRID != 0:
        raise ValueError(f"mosaic size must be divisible by {MOSAIC_GRID}, got {height}")
    src = height // MOSAIC_GRID
    dst = size // MOSAIC_GRID
    batch, channels = int(frames.shape[0]), int(frames.shape[1])
    cells = frames.reshape(batch, channels, MOSAIC_GRID, src, MOSAIC_GRID, src)
    cells = cells.permute(0, 2, 4, 1, 3, 5).contiguous()
    cells = cells.reshape(batch * MOSAIC_GRID * MOSAIC_GRID, channels, src, src)
    resized = F.interpolate(cells, size=(dst, dst), mode="bilinear", align_corners=False)
    resized = resized.reshape(batch, MOSAIC_GRID, MOSAIC_GRID, channels, dst, dst)
    return (
        resized.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(batch, channels, size, size)
    )


def bgr_to_nchw_float(frames: torch.Tensor) -> torch.Tensor:
    """Convert BGR uint8 HWC or float CHW into BGR NCHW in ``[0, 1]``.

    Marker mosaics are not ImageNet photos: no RGB swap and no ImageNet mean/std.
    """
    return _nchw_unit_interval(frames, kind="mosaic")


class MosaicBgrToFloat:
    """Picklable mosaic transform: BGR ``[0, 1]`` NCHW, no ImageNet stats.

    HWC is OpenCV BGR. CHW float is already NCHW and is only rescaled/resized.
    """

    def __init__(self, img_size: int = 224, layout: str = "full") -> None:
        self.img_size = int(img_size)
        self.layout = parse_mosaic_layout(layout)

    def __call__(self, image: Any) -> torch.Tensor:
        tensor, squeeze = _as_batched_image(image)
        nchw = resize_mosaic_nchw(bgr_to_nchw_float(tensor), self.img_size)
        nchw = crop_mosaic_nchw(nchw, self.layout)
        return nchw.squeeze(0) if squeeze else nchw


class VisionBgrToRgbImagenet:
    """Picklable per-sample transform for vendor-style HDF5 vision columns.

    HWC uint8 is OpenCV BGR. SWM's HDF5 loader permutes HWC→CHW *before*
    this transform, still BGR uint8; that must be converted too. Float CHW
    is already RGB ImageNet and is only resized.
    """

    def __init__(self, img_size: int = 224) -> None:
        self.img_size = int(img_size)

    def __call__(self, image: Any) -> torch.Tensor:
        tensor, squeeze = _as_batched_image(image)
        if tensor.shape[-1] == 3:
            nchw = bgr_to_rgb_imagenet(tensor)
        elif tensor.shape[1] == 3:
            if tensor.dtype == torch.uint8:
                nchw = bgr_to_rgb_imagenet(tensor)
            else:
                nchw = tensor.float()
        else:
            raise ValueError(f"unsupported vision shape {tuple(tensor.shape)}")
        nchw = resize_nchw(nchw, self.img_size)
        return nchw.squeeze(0) if squeeze else nchw


class MissingTokens(nn.Module):
    """Learned nulls for dropped vision residual / action AdaLN (WM-32/33)."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        width = int(embed_dim)
        self.vision_null = nn.Parameter(torch.zeros(1, 1, width))
        self.action_null = nn.Parameter(torch.zeros(1, 1, width))


def _time_batch(vision: torch.Tensor, img_size: int) -> tuple[torch.Tensor, int, int]:
    tensor = torch.as_tensor(vision)
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError(f"expected vision (B, T, ...), got {tuple(tensor.shape)}")
    batch, time = int(tensor.shape[0]), int(tensor.shape[1])
    flat = tensor.reshape(batch * time, *tensor.shape[2:])
    return VisionBgrToRgbImagenet(img_size)(flat), batch, time


class DualEncoderJEPA(nn.Module):
    """Wrap vendor ``jepa.JEPA``: tactile ViT CLS is ``emb``; DINO is predict-only.

    ``mosaic_posemb`` is ``false`` (default), ``finger_slot`` (WM-01),
    ``cell`` (WM-08: one 4x4-cell table), ``factorized_rope`` (three-axis
    temporal/local RoPE), ``spiral_rope`` (temporal + multi-directional
    local RoPE), or ``canonical_spiral_rope`` (right-finger local-X
    canonicalization). ``vision_context`` is
    ``residual`` (default: add frozen DINO CLS in ``predict``),
    ``action`` (WM-31: concat the projected CLS onto the action encoder /
    AdaLN, do not add to tactile CLS), or ``off``. ``vision_residual``
    is kept as a bool alias of ``residual``. ``vision_center`` subtracts
    the batch (train) or EMA (eval) mean of the 192-d vision CLS before
    residual/action use. WM-31 on the mix combo: center is a no-op;
    AdaLN is a window-only nick; both together hurt. Keep residual /
    no center unless a new arm says otherwise. ``vision_history`` is
    ``all`` (default: every camera in the loaded window) or ``current``
    (WM-30: DINO only the last context step; older slots get a zero
    residual). ``vision_views`` selects
    agentview, wrist, or both. Two-view fusion is ``vision_fuse=mean``
    (default) or ``concat`` (cat projected CLS, then Linear back to 192-d).
    The 4x4 mosaic stays full. ``mosaic_rest_sub`` is ``off`` (default),
    ``subtract``, ``divide``, ``log``, ``smooth_div``, or ``highpass``.
    Applied in the trainer dataset wrapper, not here. ``force_context`` is
    ``off`` (default), ``residual`` (WM-15: Linear 6-d wrench added in
    ``predict``, like the camera residual), or ``action`` (WM-16: concat
    wrench onto the action encoder input). Force never enters the tactile
    CLS target or SIGReg. ``force_pred`` (WM-19/20) decodes next-step 6-D
    wrench from ``pred_emb`` only; it is an auxiliary loss, not a new JEPA
    target. ``current_recon`` (WM-18) is a tiny MLP that
    decodes current L7|R7 from tactile CLS through a sigmoid onto
    ``[0, 1]``; it is an auxiliary loss only.
    ``next_recon`` (WM-26) is the same head on ``pred_emb`` targeting
    **next** L7|R7. ``pred_stopgrad`` (WM-24) detaches the JEPA target.
    ``pred_nce`` (WM-25) is in-batch InfoNCE of pred vs stop-grad target.
    ``mosaic_readout`` is ``cls`` (default), ``last_slot_pool`` (WM-05:
    mean of current L7|R7 patch tokens), or ``last_slot_lr`` (concat then
    Linear, starts as the same mean). The 4x4 image stays full.
    ``force_repr`` is ``raw`` (default 6-D) or ``magdir`` (8-D log-magnitude
    plus unit direction per finger). ``vision_force_gate`` multiplies the
    camera residual by ``exp(-||F||)`` so contact down-weights vision.
    ``proprio_context`` is ``off`` (default) or ``action`` (WM-23: concat
    8-D joint/gripper state onto the action encoder, same site as WM-16).
    Force and proprio never enter the tactile CLS target or SIGReg.
    ``mosaic_cell_mask_ratio`` (default 0) keeps the complete RGB+marker
    Mosaic input but randomly retains complete 4x4 cells after PatchEmbed.
    The mask is sampled independently for every flattened frame in the one
    shared JEPA encoder call, so both context and target frames are masked;
    there are not two separate encoders.  With ratio 0.5, 8 of 16 cells are
    kept and the usual 257-token ViT sequence becomes 129 tokens during
    training.  Evaluation remains unmasked unless ``mosaic_cell_mask_eval``
    is enabled.
    ``mosaic_causal_attention`` replaces raster self-attention with a
    slot-causal mask: a patch can attend to its own slot and earlier slots,
    while every left/right patch pair in the same slot remains bidirectional.
    It uses local ``(time, y, x)`` RoPE plus a learnable finger embedding and
    removes the global raster position table.  The optional
    ``mosaic_causal_streaming`` flag exposes ``encode_tactile_stream`` for
    inference: after one full-window bootstrap, each update embeds only the
    current L/R cells and reuses per-layer historical K/V.
    ``drop_vision_p`` / ``drop_action_p`` (WM-32/33) drop a whole window
    with that Bernoulli probability during training (eval stays complete
    unless ``drop_eval``). Dropped vision uses ``missing_vision=skip``
    (no residual add; same as zeros) or ``null`` (learned token).
    Dropped action uses ``zero`` (AdaLN sees 0), ``skip_pred`` (no
    pred_loss on that window), or ``null`` (learned act_emb token).
    On the mix combo: vision ``skip`` is the missing-camera method
    (nick vs WM-28, not a new base); vision ``null`` lost. Do not
    drop action on the complete mix. YAML ``drop_*=0``.
    See ``docs/lewm_wm_experiments.md``.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        predictor: nn.Module | None = None,
        action_encoder: nn.Module | None = None,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
        *,
        vision_encoder: nn.Module | None = None,
        dinov3_path: str | None = None,
        embed_dim: int = DEFAULT_EMBED_DIM,
        img_size: int = 224,
        allow_hf_download: bool = False,
        jepa: nn.Module | None = None,
        mosaic_posemb: bool | str = False,
        mosaic_cell_mask_ratio: float | int | str | None = 0.0,
        mosaic_cell_mask_eval: bool | str = False,
        mosaic_causal_attention: bool | str = False,
        mosaic_causal_streaming: bool | str = False,
        tactile_layout: str = "mosaic",
        tactile_history_size: int = 8,
        tactile_history_stride: int = 5,
        tactile_frame_size: int = 42,
        tactile_patch_size: int = 14,
        tactile_rope: str = "canonical_spiral",
        tactile_causal_attention: bool | str = False,
        tactile_streaming: bool | str = False,
        tactile_stream_compile: bool | str = True,
        tactile_stream_compile_mode: str = "default",
        vision_residual: bool = True,
        vision_context: bool | str | None = None,
        vision_center: bool | str = False,
        vision_history: bool | str = "all",
        vision_precision: str | None = "bf16",
        vision_compile: bool | str = False,
        vision_compile_mode: str = "reduce-overhead",
        tactile_encoder_trainable: bool = True,
        history_size: int = 3,
        patch_size: int = 14,
        vision_views: str | list[str] | tuple[str, ...] = "agentview",
        mosaic_layout: str = "full",
        vision_fuse: str = "mean",
        mosaic_rest_sub: bool | str = False,
        force_context: bool | str = False,
        current_recon: bool | str = False,
        current_recon_weight: float = 0.1,
        next_recon: bool | str = False,
        next_recon_weight: float = 0.1,
        pred_stopgrad: bool | str = False,
        pred_nce: bool | str = False,
        pred_nce_weight: float = 0.05,
        pred_nce_temp: float = 0.1,
        # Train-time delta_shuffle hinge: penalize a predictor that ignores
        # the action input. Loss is zero once MSE(shuffled-action pred) is
        # at least ``1 + margin`` times worse than MSE(true pred).
        pred_action_hinge: bool | str = False,
        pred_action_hinge_weight: float = 1.0,
        pred_action_hinge_margin: float = 0.1,
        pred_action_hinge_warmup: int = 200,
        force_pred: bool | str = False,
        force_pred_weight: float = 0.05,
        mosaic_readout: bool | str = "cls",
        force_repr: bool | str = "raw",
        vision_force_gate: bool | str = False,
        proprio_context: bool | str = False,
        drop_vision_p: bool | str | float | int | None = 0.0,
        drop_action_p: bool | str | float | int | None = 0.0,
        missing_vision: bool | str | None = "skip",
        missing_action: bool | str | None = "skip_pred",
        drop_eval: bool | str = False,
    ) -> None:
        super().__init__()
        if jepa is None:
            if encoder is None or predictor is None or action_encoder is None:
                raise ValueError(
                    "encoder, predictor, and action_encoder are required without jepa="
                )
            from .jepa import JEPA

            jepa = JEPA(
                encoder,
                predictor,
                action_encoder,
                projector=projector,
                pred_proj=pred_proj,
            )
        from tacmind0.tactile.mosaic_posemb import (
            parse_mosaic_cell_mask_ratio,
            parse_mosaic_posemb,
            wrap_vit_mosaic_causal,
            wrap_vit_mosaic_cell_mask,
            wrap_vit_mosaic_posemb,
        )
        from tacmind0.tactile.lewm.mosaic_readout import (
            parse_mosaic_readout,
            wrap_vit_mosaic_readout,
        )
        from tacmind0.tactile.lewm.mosaic_rest import parse_mosaic_rest_sub
        from tacmind0.tactile.tactile_sequence import (
            parse_tactile_layout,
            parse_tactile_rope,
            wrap_vit_tactile_sequence,
        )

        posemb_mode = parse_mosaic_posemb(mosaic_posemb)
        causal_attention = parse_on_off(
            mosaic_causal_attention, name="mosaic_causal_attention"
        )
        causal_streaming = parse_on_off(
            mosaic_causal_streaming, name="mosaic_causal_streaming"
        )
        tactile_mode = parse_tactile_layout(tactile_layout)
        tactile_rope_mode = parse_tactile_rope(tactile_rope)
        tactile_causal = parse_on_off(
            tactile_causal_attention, name="tactile_causal_attention"
        )
        tactile_stream = parse_on_off(tactile_streaming, name="tactile_streaming")
        tactile_compile = parse_on_off(
            tactile_stream_compile, name="tactile_stream_compile"
        )
        tactile_compile_mode = parse_tactile_stream_compile_mode(
            tactile_stream_compile_mode
        )
        readout_mode = parse_mosaic_readout(mosaic_readout)
        # Compilation is meaningful only for the inference-only stream path;
        # keep it a harmless no-op for legacy/full-window configurations.
        tactile_compile = tactile_compile and tactile_stream
        if tactile_stream and not tactile_causal:
            raise ValueError("tactile_streaming requires tactile_causal_attention=true")
        if tactile_mode == "sequence_3x3":
            if posemb_mode != "off" or causal_attention or causal_streaming:
                raise ValueError(
                    "sequence_3x3 uses tactile_rope/tactile_causal_attention; "
                    "mosaic_posemb and mosaic causal flags must be off"
                )
            if readout_mode != "cls":
                raise ValueError("sequence_3x3 currently requires mosaic_readout=cls")
            jepa.encoder = wrap_vit_tactile_sequence(
                jepa.encoder,
                frame_size=int(tactile_frame_size),
                patch_size=int(tactile_patch_size),
                history_size=int(tactile_history_size),
                time_stride=int(tactile_history_stride),
                canonicalize_right=tactile_rope_mode == "canonical_spiral",
                causal_attention=tactile_causal,
            )
        if causal_streaming and not causal_attention:
            raise ValueError(
                "mosaic_causal_streaming requires mosaic_causal_attention=true"
            )
        causal_posemb_modes = {
            "off",
            "factorized_rope",
            "spiral_rope",
            "canonical_spiral_rope",
        }
        if causal_attention and posemb_mode not in causal_posemb_modes:
            raise ValueError(
                "mosaic_causal_attention supports mosaic_posemb=false, "
                "factorized_rope, spiral_rope, or canonical_spiral_rope; "
                f"got {posemb_mode!r}"
            )
        if posemb_mode != "off" and not causal_attention:
            jepa.encoder = wrap_vit_mosaic_posemb(
                jepa.encoder,
                img_size=int(img_size),
                patch_size=int(patch_size),
                mode=posemb_mode,
            )
        if readout_mode != "cls":
            jepa.encoder = wrap_vit_mosaic_readout(
                jepa.encoder,
                img_size=int(img_size),
                patch_size=int(patch_size),
                mode=readout_mode,
            )
        mask_ratio = parse_mosaic_cell_mask_ratio(mosaic_cell_mask_ratio)
        mask_eval = parse_on_off(
            mosaic_cell_mask_eval, name="mosaic_cell_mask_eval"
        )
        if mask_ratio > 0.0 and posemb_mode != "off":
            raise ValueError(
                "mosaic_cell_mask_ratio currently requires mosaic_posemb=false"
            )
        if mask_ratio > 0.0 and readout_mode != "cls":
            raise ValueError(
                "mosaic_cell_mask_ratio currently requires mosaic_readout=cls"
            )
        if causal_attention and mask_ratio > 0.0:
            raise ValueError(
                "mosaic_causal_attention and mosaic_cell_mask_ratio are separate "
                "OL-0 ablations; enable only one"
            )
        if mask_ratio > 0.0:
            jepa.encoder = wrap_vit_mosaic_cell_mask(
                jepa.encoder,
                img_size=int(img_size),
                patch_size=int(patch_size),
                ratio=mask_ratio,
                mask_eval=mask_eval,
            )
        if causal_attention:
            if readout_mode != "cls":
                raise ValueError(
                    "mosaic_causal_attention currently requires mosaic_readout=cls"
                )
            jepa.encoder = wrap_vit_mosaic_causal(
                jepa.encoder,
                img_size=int(img_size),
                patch_size=int(patch_size),
                rope_kind=(
                    "three_axis"
                    if posemb_mode in {"off", "factorized_rope"}
                    else "spatiotemporal_spiral"
                ),
                canonicalize_right=posemb_mode == "canonical_spiral_rope",
            )
        self.jepa = jepa
        self.tactile_encoder_trainable = bool(tactile_encoder_trainable)
        set_tactile_encoder_trainable(
            self.jepa, trainable=self.tactile_encoder_trainable
        )
        freeze_unused = getattr(
            self.jepa.encoder, "freeze_unused_embedding_parameters", None
        )
        if callable(freeze_unused):
            freeze_unused()
        self.mosaic_posemb = posemb_mode
        self.mosaic_cell_mask_ratio = mask_ratio
        self.mosaic_cell_mask_eval = mask_eval
        self.mosaic_causal_attention = causal_attention
        self.mosaic_causal_streaming = causal_streaming
        self.tactile_layout = tactile_mode
        self.tactile_history_size = int(tactile_history_size)
        self.tactile_history_stride = int(tactile_history_stride)
        self.tactile_frame_size = int(tactile_frame_size)
        self.tactile_patch_size = int(tactile_patch_size)
        self.tactile_rope = tactile_rope_mode
        self.tactile_causal_attention = tactile_causal
        self.tactile_streaming = tactile_stream
        self.tactile_stream_compile = tactile_compile
        self.tactile_stream_compile_mode = tactile_compile_mode
        self.mosaic_readout = readout_mode
        self.vision_context = resolve_vision_context(vision_context, vision_residual)
        self.vision_residual = self.vision_context == "residual"
        self.vision_center = parse_on_off(vision_center, name="vision_center")
        self.vision_history = parse_vision_history(vision_history)
        self.vision_precision = parse_vision_precision(vision_precision)
        self.vision_compile = parse_on_off(vision_compile, name="vision_compile")
        self.vision_compile_mode = parse_vision_compile_mode(vision_compile_mode)
        self.history_size = max(int(history_size), 1)
        self.vision_views = parse_vision_views(vision_views)
        self.mosaic_layout = parse_mosaic_layout(mosaic_layout)
        self.vision_fuse = parse_vision_fuse(vision_fuse)
        self.mosaic_rest_sub = parse_mosaic_rest_sub(mosaic_rest_sub)
        self.force_context = parse_force_context(force_context)
        self.force_repr = parse_force_repr(force_repr)
        self.vision_force_gate = parse_vision_force_gate(vision_force_gate)
        self.proprio_context = parse_proprio_context(proprio_context)
        self.drop_vision_p = parse_drop_p(drop_vision_p, name="drop_vision_p")
        self.drop_action_p = parse_drop_p(drop_action_p, name="drop_action_p")
        self.missing_vision = parse_missing_vision(missing_vision)
        self.missing_action = parse_missing_action(missing_action)
        self.drop_eval = parse_on_off(drop_eval, name="drop_eval")
        self.current_recon = parse_on_off(current_recon, name="current_recon")
        self.current_recon_weight = float(current_recon_weight)
        self.next_recon = parse_on_off(next_recon, name="next_recon")
        self.next_recon_weight = float(next_recon_weight)
        self.pred_stopgrad = parse_on_off(pred_stopgrad, name="pred_stopgrad")
        self.pred_nce = parse_on_off(pred_nce, name="pred_nce")
        self.pred_nce_weight = float(pred_nce_weight)
        self.pred_nce_temp = float(pred_nce_temp)
        self.pred_action_hinge = parse_on_off(
            pred_action_hinge, name="pred_action_hinge"
        )
        self.pred_action_hinge_weight = float(pred_action_hinge_weight)
        self.pred_action_hinge_margin = float(pred_action_hinge_margin)
        self.pred_action_hinge_warmup = int(pred_action_hinge_warmup)
        self.force_pred = parse_force_pred(force_pred)
        self.force_pred_weight = float(force_pred_weight)
        self.embed_dim = int(embed_dim)
        self.img_size = int(img_size)
        self.register_buffer("vision_mean", torch.zeros(1, 1, self.embed_dim))
        if vision_encoder is None:
            path = resolve_dinov3_path(dinov3_path) or resolve_dinov3_path(
                os.environ.get("DINOV3_PATH")
            )
            if not path:
                raise ValueError(
                    "DualEncoderJEPA needs a frozen DINOv3: set DINOV3_PATH to a "
                    "local HuggingFace directory, pass dinov3_path=, or pass "
                    "vision_encoder="
                )
            vision_encoder = load_frozen_dinov3(
                path, allow_hf_download=bool(allow_hf_download)
            )
        else:
            vision_encoder = freeze_module(vision_encoder)
        self.vision_encoder = vision_encoder
        self._compiled_vision_forward: Any | None = None
        self._vision_compile_failed = False
        self._vision_compile_logger = logging.getLogger(__name__)
        dino_dim = cls_hidden_size(self.vision_encoder)
        self.vision_proj = nn.Linear(dino_dim, self.embed_dim)
        n_views = len(self.vision_views)
        if self.vision_fuse == "concat" and n_views > 1:
            self.vision_fuse_proj = nn.Linear(self.embed_dim * n_views, self.embed_dim)
        else:
            self.vision_fuse_proj = nn.Identity()
        if not self._uses_vision():
            freeze_module(self.vision_proj)
            freeze_module(self.vision_fuse_proj)
        self.force_proj = nn.Linear(force_feature_dim(self.force_repr), self.embed_dim)
        nn.init.zeros_(self.force_proj.weight)
        nn.init.zeros_(self.force_proj.bias)
        if self.force_context != "residual":
            freeze_module(self.force_proj)
        self.current_cell_head = _gel_cell_head(self.embed_dim)
        if not self.current_recon:
            freeze_module(self.current_cell_head)
        self.next_cell_head = _gel_cell_head(self.embed_dim)
        if not self.next_recon:
            freeze_module(self.next_cell_head)
        self.force_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 6),
        )
        nn.init.zeros_(self.force_head[-1].weight)
        nn.init.zeros_(self.force_head[-1].bias)
        if not self.force_pred:
            freeze_module(self.force_head)
        self.missing_tokens = MissingTokens(self.embed_dim)
        if self.missing_vision != "null":
            self.missing_tokens.vision_null.requires_grad_(False)
        if self.missing_action != "null":
            self.missing_tokens.action_null.requires_grad_(False)
        self._vision_emb: torch.Tensor | None = None
        self._force_emb: torch.Tensor | None = None
        self._force_gate: torch.Tensor | None = None
        self._predict_time_start: int | None = None
        self._available_vision: torch.Tensor | None = None
        self._available_action: torch.Tensor | None = None

    def _action_encoder_in_dim(self) -> int | None:
        patch = getattr(self.jepa.action_encoder, "patch_embed", None)
        channels = getattr(patch, "in_channels", None)
        return int(channels) if channels is not None else None

    @property
    def encoder(self) -> nn.Module:
        return self.jepa.encoder

    @property
    def predictor(self) -> nn.Module:
        return self.jepa.predictor

    @property
    def action_encoder(self) -> nn.Module:
        return self.jepa.action_encoder

    @property
    def projector(self) -> nn.Module:
        return self.jepa.projector

    @property
    def pred_proj(self) -> nn.Module:
        return self.jepa.pred_proj

    @torch.no_grad()
    def encode_tactile_stream(
        self,
        pixels: torch.Tensor,
        state: Any | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Encode tactile history incrementally with the causal Mosaic cache.

        The sequence path bootstraps ``(B, 2, 8, 3, 42, 42)`` and then accepts
        only current ``(B, 2, 3, 42, 42)`` L/R frames.  The legacy Mosaic
        path retains its original full-Mosaic/current-cell contract.
        This inference-only helper intentionally bypasses camera/action
        conditioning; callers can feed the returned tactile embedding into
        the policy-specific conditioning path.
        """
        from tacmind0.tactile.tactile_sequence import TactileSequenceEncoder

        encoder = self.jepa.encoder
        if isinstance(encoder, TactileSequenceEncoder):
            if not self.tactile_streaming:
                raise RuntimeError("tactile_streaming=false; enable it for stream inference")
            if state is None:
                cls, next_state = encoder.stream_init(pixels)
            else:
                if self.tactile_stream_compile:
                    cls, next_state = encoder.stream_step_compiled(
                        pixels,
                        state,
                        mode=self.tactile_stream_compile_mode,
                    )
                else:
                    cls, next_state = encoder.stream_step(pixels, state)
            return self.jepa.projector(cls), next_state

        from tacmind0.tactile.mosaic_posemb import MosaicCausalEncoder

        if not isinstance(encoder, MosaicCausalEncoder):
            raise RuntimeError(
                "encode_tactile_stream requires mosaic_causal_attention=true"
            )
        if state is None:
            cls, next_state = encoder.stream_init(pixels)
        else:
            cls, next_state = encoder.stream_step(pixels, state)
        return self.jepa.projector(cls), next_state

    def train(self, mode: bool = True) -> DualEncoderJEPA:
        super().train(mode)
        self.vision_encoder.eval()
        if not self.tactile_encoder_trainable:
            self.jepa.encoder.eval()
        if not self._uses_vision():
            self.vision_proj.eval()
            self.vision_fuse_proj.eval()
        if self.force_context != "residual":
            self.force_proj.eval()
        if not self.current_recon:
            self.current_cell_head.eval()
        if not self.next_recon:
            self.next_cell_head.eval()
        if not self.force_pred:
            self.force_head.eval()
        if self.missing_vision != "null" and self.missing_action != "null":
            self.missing_tokens.eval()
        return self

    def _uses_vision(self) -> bool:
        return self.vision_context in ("residual", "action")

    def _needs_force(self) -> bool:
        return self.force_context != "off" or self.vision_force_gate

    def _needs_action_extras(self) -> bool:
        return (
            self._needs_force()
            or self.proprio_context == "action"
            or self.vision_context == "action"
        )

    def _center_vision(self, vis: torch.Tensor) -> torch.Tensor:
        if not self.vision_center:
            return vis
        batch_mean = vis.mean(dim=(0, 1), keepdim=True)
        if self.training:
            updated = (1.0 - VISION_CENTER_MOMENTUM) * self.vision_mean + (
                VISION_CENTER_MOMENTUM * batch_mean.detach()
            )
            self.vision_mean.copy_(updated.reshape_as(self.vision_mean))
            return vis - batch_mean
        mean = self.vision_mean.to(device=vis.device, dtype=vis.dtype)
        return vis - mean

    def _pad_action_to_encoder(self, act: torch.Tensor) -> torch.Tensor:
        if int(act.size(-1)) == 0:
            return act
        needed = self._action_encoder_in_dim()
        extra = (needed - int(act.size(-1))) if needed is not None else 0
        if extra > 0:
            return torch.cat([act, act.new_zeros(*act.shape[:-1], extra)], dim=-1)
        return act

    def _action_extras_bt(self, info: dict[str, Any]) -> torch.Tensor | None:
        """Observed-window extras concatenated onto action (force, proprio, vision)."""
        extras: list[torch.Tensor] = []
        if self.force_context == "action":
            if "force" not in info:
                raise KeyError(
                    "force_context / vision_force_gate requires batch['force']"
                )
            extras.append(self._force_features(force_bt6(info["force"])))
        if self.proprio_context == "action":
            if "proprio" not in info:
                raise KeyError("proprio_context requires batch['proprio']")
            extras.append(proprio_btd(info["proprio"]))
        if self.vision_context == "action":
            if self._vision_emb is None:
                raise RuntimeError(
                    "encode() must run before action extras so vision context exists"
                )
            extras.append(self._vision_emb)
        if not extras:
            return None
        packed = extras[0]
        for piece in extras[1:]:
            packed = concat_action_condition(packed, piece, name="action_extra")
        return packed if packed.ndim == 3 else packed.unsqueeze(1)

    def _force_features(self, wrench: torch.Tensor) -> torch.Tensor:
        if self.force_repr == "magdir":
            return force_magdir_bt8(wrench)
        return wrench

    def _drop_active(self) -> bool:
        return bool(self.training or self.drop_eval)

    def _sample_availability(
        self, batch: int, prob: float, *, like: torch.Tensor
    ) -> torch.Tensor:
        device = like.device
        dtype = like.dtype if like.is_floating_point() else torch.float32
        if not self._drop_active() or float(prob) <= 0.0:
            return torch.ones(batch, 1, 1, device=device, dtype=dtype)
        if float(prob) >= 1.0:
            return torch.zeros(batch, 1, 1, device=device, dtype=dtype)
        keep = torch.rand(batch, 1, 1, device=device) >= float(prob)
        return keep.to(dtype=dtype)

    def _broadcast_avail(
        self, avail: torch.Tensor | None, series: torch.Tensor
    ) -> torch.Tensor | None:
        if avail is None:
            return None
        avail = avail.to(device=series.device, dtype=series.dtype)
        if avail.size(0) != series.size(0):
            if series.size(0) % avail.size(0) != 0:
                raise ValueError(
                    f"availability {tuple(avail.shape)} incompatible with "
                    f"{tuple(series.shape)}"
                )
            avail = avail.repeat_interleave(series.size(0) // avail.size(0), dim=0)
        return avail

    def _apply_vision_missing(self, vis: torch.Tensor) -> torch.Tensor:
        avail = self._broadcast_avail(self._available_vision, vis)
        if avail is None:
            return vis
        if self.missing_vision == "null":
            null = self.missing_tokens.vision_null.to(device=vis.device, dtype=vis.dtype)
            return avail * vis + (1.0 - avail) * null
        return vis * avail

    def _apply_action_null(self, act_emb: torch.Tensor) -> torch.Tensor:
        if self.missing_action != "null":
            return act_emb
        avail = self._broadcast_avail(self._available_action, act_emb)
        if avail is None:
            return act_emb
        null = self.missing_tokens.action_null.to(
            device=act_emb.device, dtype=act_emb.dtype
        )
        return avail * act_emb + (1.0 - avail) * null

    def encode_with_forced_drop(
        self,
        info: dict[str, Any],
        *,
        vision_p: float | None = None,
        action_p: float | None = None,
    ) -> dict[str, Any]:
        """Eval helper: encode a clone with drop_eval and optional p=1 overrides."""
        old = (self.drop_eval, self.drop_vision_p, self.drop_action_p)
        self.drop_eval = True
        if vision_p is not None:
            self.drop_vision_p = parse_drop_p(vision_p, name="drop_vision_p")
        if action_p is not None:
            self.drop_action_p = parse_drop_p(action_p, name="drop_action_p")
        try:
            cloned = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in info.items()
            }
            return self.encode(cloned)
        finally:
            self.drop_eval, self.drop_vision_p, self.drop_action_p = old

    def _per_sample_flag(
        self,
        flag: Any,
        batch: int,
        *,
        name: str,
    ) -> torch.Tensor:
        """Reduce windowed meta columns to one flag per sample.

        SWM may load ``has_*`` / ``source_id`` as ``(B,)`` or ``(B, T)`` when
        ``num_steps>1``.  Flags are constant inside a clip, so keep the last
        step of the window.
        """
        tensor = torch.as_tensor(flag)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1).expand(batch)
        elif tensor.ndim >= 2:
            tensor = tensor.reshape(int(tensor.size(0)), -1)[:, -1]
        else:
            tensor = tensor.reshape(-1)
            if int(tensor.numel()) == batch:
                pass
            elif int(tensor.numel()) % batch == 0:
                tensor = tensor.view(batch, -1)[:, -1]
            else:
                raise ValueError(
                    f"{name} length {int(tensor.numel())} incompatible with "
                    f"batch {batch}"
                )
        if int(tensor.numel()) != batch:
            raise ValueError(
                f"{name} length {int(tensor.numel())} != batch {batch}"
            )
        return tensor

    def _gate_availability(
        self,
        avail: torch.Tensor,
        flag: Any | None,
        *,
        name: str,
    ) -> torch.Tensor:
        """AND random drop masks with optional per-sample ``has_*`` flags."""
        if flag is None:
            return avail
        # ``_sample_availability`` returns ``(B, 1, 1)``.  Do not use
        # ``avail.numel()`` as batch: ``(B,1,1)*(B,)`` broadcasts to
        # ``(B,1,B)`` under PyTorch trailing-align rules.
        batch = int(avail.size(0))
        tensor = self._per_sample_flag(flag, batch, name=name).to(
            device=avail.device, dtype=avail.dtype
        )
        mask = (tensor > 0.5).to(dtype=avail.dtype)
        while mask.ndim < avail.ndim:
            mask = mask.unsqueeze(-1)
        return avail * mask

    def encode(self, info: dict[str, Any]) -> dict[str, Any]:
        info = dict(info)
        if "pixels" not in info and "tactile" in info:
            info["pixels"] = info["tactile"]
        pixels = info["pixels"]
        batch = int(pixels.size(0))
        like = info["action"] if "action" in info else pixels
        self._available_vision = self._sample_availability(
            batch, self.drop_vision_p, like=like
        )
        self._available_action = self._sample_availability(
            batch, self.drop_action_p, like=like
        )
        self._available_vision = self._gate_availability(
            self._available_vision, info.get("has_vision"), name="has_vision"
        )
        self._available_action = self._gate_availability(
            self._available_action, info.get("has_action"), name="has_action"
        )
        action_encoder = self.jepa.action_encoder
        set_source = getattr(action_encoder, "set_source_id", None)
        clear_source = getattr(action_encoder, "clear_source_id", None)
        if callable(set_source) and "source_id" in info:
            set_source(self._per_sample_flag(info["source_id"], batch, name="source_id"))
        wrench = None
        try:
            if self.missing_action == "zero":
                if "action" not in info:
                    raise KeyError("missing_action=zero requires batch['action']")
                avail = self._broadcast_avail(self._available_action, info["action"])
                assert avail is not None
                info["action"] = info["action"] * avail
            if self._uses_vision():
                self._vision_emb = self._center_vision(self._encode_vision(info))
            else:
                self._vision_emb = None
            if self._needs_force():
                if "force" not in info:
                    raise KeyError(
                        "force_context / vision_force_gate requires batch['force']"
                    )
                wrench = force_bt6(info["force"])
                if self.force_context == "action":
                    info["action"] = concat_action_condition(
                        info["action"],
                        self._force_features(wrench),
                        name="force",
                    )
            if self.proprio_context == "action":
                if "proprio" not in info:
                    raise KeyError("proprio_context requires batch['proprio']")
                info["action"] = concat_action_condition(
                    info["action"],
                    proprio_btd(info["proprio"]),
                    name="proprio",
                )
            if self.vision_context == "action":
                assert self._vision_emb is not None
                info["action"] = concat_action_condition(
                    info["action"],
                    self._apply_vision_missing(self._vision_emb),
                    name="vision",
                )
            info = self.jepa.encode(info)
            info["act_emb"] = self._apply_action_null(info["act_emb"])
            info["available_vision"] = self._available_vision.reshape(batch)
            info["available_action"] = self._available_action.reshape(batch)
            if wrench is not None and self.vision_force_gate:
                mag = 0.5 * (
                    torch.linalg.vector_norm(wrench[..., :3], dim=-1)
                    + torch.linalg.vector_norm(wrench[..., 3:], dim=-1)
                )
                self._force_gate = torch.exp(-mag).unsqueeze(-1).to(
                    device=info["emb"].device, dtype=info["emb"].dtype
                )
            else:
                self._force_gate = None
            if self.force_context == "residual":
                assert wrench is not None
                feat = self._force_features(wrench).to(
                    device=info["emb"].device, dtype=info["emb"].dtype
                )
                self._force_emb = self.force_proj(feat)
            else:
                self._force_emb = None
            return info
        finally:
            if callable(clear_source):
                clear_source()

    def decode_current_cells(self, emb: torch.Tensor) -> torch.Tensor:
        """Map tactile CLS ``(B, T, D)`` to current L7|R7 ``(B, T, 2, 3, S, S)``."""
        if not self.current_recon:
            raise RuntimeError("decode_current_cells requires current_recon=true")
        return _decode_gel_cells(self.current_cell_head, emb)

    def decode_next_cells(self, emb: torch.Tensor) -> torch.Tensor:
        """Map predicted CLS ``(B, T, D)`` to next L7|R7 ``(B, T, 2, 3, S, S)``."""
        if not self.next_recon:
            raise RuntimeError("decode_next_cells requires next_recon=true")
        return _decode_gel_cells(self.next_cell_head, emb)

    def decode_force(self, emb: torch.Tensor) -> torch.Tensor:
        """Map latent CLS ``(B, T, D)`` to a 6-D wrench ``(B, T, 6)``."""
        if not self.force_pred:
            raise RuntimeError("decode_force requires force_pred=true")
        tensor = torch.as_tensor(emb)
        if tensor.ndim != 3:
            raise ValueError(f"expected emb (B, T, D), got {tuple(tensor.shape)}")
        return self.force_head(tensor)

    def _vision_current_index(self, time: int) -> int | None:
        if self.vision_history == "all":
            return None
        if time < 1:
            raise ValueError(f"vision time must be >= 1, got {time}")
        return min(self.history_size, time) - 1

    def _encode_vision(self, info: dict[str, Any]) -> torch.Tensor:
        flats: list[torch.Tensor] = []
        batch = window = encoded_time = None
        current_index: int | None = None
        compact_current = False
        for key in vision_keys_for_views(self.vision_views):
            if key not in info:
                views = ",".join(self.vision_views)
                raise KeyError(
                    f"DualEncoderJEPA vision_views={views} requires batch[{key!r}]"
                )
            tensor = torch.as_tensor(info[key])
            if tensor.ndim == 4:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 5:
                raise ValueError(f"expected vision (B, T, ...), got {tuple(tensor.shape)}")
            this_batch, this_window = int(tensor.shape[0]), int(tensor.shape[1])
            if batch is None:
                batch = this_batch
                if self.vision_history == "current" and this_window == 1:
                    pixels = torch.as_tensor(info.get("pixels"))
                    if pixels.ndim < 2:
                        raise ValueError(
                            "compact current-frame vision requires batch['pixels'] "
                            "with a temporal dimension"
                        )
                    window = int(pixels.shape[1])
                    compact_current = True
                else:
                    window = this_window
                current_index = self._vision_current_index(window)
            elif compact_current:
                if (this_batch, this_window) != (batch, 1):
                    raise ValueError(
                        f"vision view {key} compact shape {(this_batch, this_window)} "
                        f"!= {(batch, 1)}"
                    )
            elif (this_batch, this_window) != (batch, window):
                raise ValueError(
                    f"vision view {key} shape {(this_batch, this_window)} "
                    f"!= {(batch, window)}"
                )
            if current_index is not None and not compact_current:
                tensor = tensor[:, current_index : current_index + 1]
            nchw, this_batch, this_time = _time_batch(tensor, self.img_size)
            if encoded_time is None:
                encoded_time = this_time
            elif (this_batch, this_time) != (batch, encoded_time):
                raise ValueError(
                    f"vision view {key} encoded shape {(this_batch, this_time)} "
                    f"!= {(batch, encoded_time)}"
                )
            flats.append(nchw)
        stacked = torch.cat(flats, dim=0)
        device = stacked.device
        use_bf16 = (
            device.type == "cuda"
            and self.vision_precision == "bf16"
            and torch.cuda.is_bf16_supported()
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            if self.vision_compile:
                cls = self._encode_compiled_vision(stacked)
            else:
                cls = encode_cls(self.vision_encoder, stacked)
        projection_dtype = self.vision_proj.weight.dtype
        projected = self.vision_proj(cls.detach().to(dtype=projection_dtype))
        n_views = len(flats)
        shaped = projected.reshape(n_views, batch, encoded_time, self.embed_dim)
        if self.vision_fuse == "concat" and n_views > 1:
            fused = shaped.permute(1, 2, 0, 3).reshape(
                batch, encoded_time, n_views * self.embed_dim
            )
            fused = self.vision_fuse_proj(fused)
        else:
            fused = shaped.mean(dim=0)
        if current_index is None:
            return fused
        full = fused.new_zeros(batch, window, fused.size(-1))
        full[:, current_index] = fused[:, 0]
        return full

    def _encode_compiled_vision(self, stacked: torch.Tensor) -> torch.Tensor:
        """Run frozen camera DINO through an optional non-stateful compiler."""
        compiler = getattr(torch, "compile", None)
        if compiler is None:
            raise RuntimeError("vision_compile=true requires torch.compile")
        if self._compiled_vision_forward is None and not self._vision_compile_failed:
            self._compiled_vision_forward = compiler(
                self.vision_encoder.forward,
                mode=self.vision_compile_mode,
                fullgraph=False,
                dynamic=False,
            )
        try:
            output = self._compiled_vision_forward(
                pixel_values=stacked, interpolate_pos_encoding=True
            )
            return extract_cls(output)
        except Exception:
            if self._vision_compile_failed:
                raise
            self._vision_compile_failed = True
            self._compiled_vision_forward = None
            self._vision_compile_logger.warning(
                "torch.compile camera DINO failed; falling back to eager forward",
                exc_info=True,
            )
            return encode_cls(self.vision_encoder, stacked)

    def _vision_context(self, emb: torch.Tensor) -> torch.Tensor:
        if self._vision_emb is None:
            raise RuntimeError("encode() must run before predict() so vision context exists")
        vis = self._vision_emb
        if vis.size(0) != emb.size(0):
            if emb.size(0) % vis.size(0) != 0:
                raise ValueError(
                    f"vision context {tuple(vis.shape)} incompatible with emb {tuple(emb.shape)}"
                )
            vis = vis.repeat_interleave(emb.size(0) // vis.size(0), dim=0)
        time = int(emb.size(1))
        start = 0 if self._predict_time_start is None else int(self._predict_time_start)
        if start < 0:
            raise ValueError(f"predict time start must be >= 0, got {start}")
        context = vis.new_zeros(vis.size(0), time, vis.size(-1))
        observed = int(vis.size(1))
        lo = min(start, observed)
        hi = min(start + time, observed)
        if hi > lo:
            context[:, lo - start : hi - start] = vis[:, lo:hi]
        if context.size(-1) != emb.size(-1):
            raise ValueError(
                f"vision context {tuple(context.shape)} incompatible with emb {tuple(emb.shape)}"
            )
        return context.to(device=emb.device, dtype=emb.dtype)

    def _force_context(self, emb: torch.Tensor) -> torch.Tensor:
        if self._force_emb is None:
            raise RuntimeError("encode() must run before predict() so force context exists")
        wrench = self._force_emb
        if wrench.size(0) != emb.size(0):
            if emb.size(0) % wrench.size(0) != 0:
                raise ValueError(
                    f"force context {tuple(wrench.shape)} incompatible with emb {tuple(emb.shape)}"
                )
            wrench = wrench.repeat_interleave(emb.size(0) // wrench.size(0), dim=0)
        time = int(emb.size(1))
        start = 0 if self._predict_time_start is None else int(self._predict_time_start)
        context = wrench.new_zeros(wrench.size(0), time, wrench.size(-1))
        observed = int(wrench.size(1))
        lo = min(start, observed)
        hi = min(start + time, observed)
        if hi > lo:
            context[:, lo - start : hi - start] = wrench[:, lo:hi]
        return context.to(device=emb.device, dtype=emb.dtype)

    def _align_series(self, series: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if series.size(0) != emb.size(0):
            if emb.size(0) % series.size(0) != 0:
                raise ValueError(
                    f"series {tuple(series.shape)} incompatible with emb {tuple(emb.shape)}"
                )
            series = series.repeat_interleave(emb.size(0) // series.size(0), dim=0)
        time = int(emb.size(1))
        start = 0 if self._predict_time_start is None else int(self._predict_time_start)
        context = series.new_zeros(series.size(0), time, series.size(-1))
        observed = int(series.size(1))
        lo = min(start, observed)
        hi = min(start + time, observed)
        if hi > lo:
            context[:, lo - start : hi - start] = series[:, lo:hi]
        return context.to(device=emb.device, dtype=emb.dtype)

    def _gate_context(self, emb: torch.Tensor) -> torch.Tensor:
        if self._force_gate is None:
            raise RuntimeError("encode() must run before predict() so force gate exists")
        return self._align_series(self._force_gate, emb)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        fused = emb
        if self.vision_residual:
            vis = self._apply_vision_missing(self._vision_context(emb))
            if self.vision_force_gate:
                vis = vis * self._gate_context(emb)
            fused = fused + vis
        if self.force_context == "residual":
            fused = fused + self._force_context(emb)
        return self.jepa.predict(fused, act_emb)

    def rollout(
        self,
        info: dict[str, Any],
        action_sequence: torch.Tensor,
        history_size: int = 3,
    ) -> dict[str, Any]:
        """Vendor JEPA rollout with observed-frame DINO (zeros beyond T_obs).

        Force / proprio / action-site vision concatenated onto the action
        encoder follow the same rule: keep the observed-window values, zero
        the extras after ``T_obs``.
        """
        if "pixels" not in info:
            raise KeyError("pixels not in info_dict")
        observed = int(info["pixels"].size(2))
        batch, samples, horizon = (int(dim) for dim in action_sequence.shape[:3])
        act_0, act_future = torch.split(action_sequence, [observed, horizon - observed], dim=2)
        info["action"] = act_0
        init = {
            key: value[:, 0] for key, value in info.items() if torch.is_tensor(value)
        }
        init = self.encode(init)
        extras = self._action_extras_bt(init)
        window = max(int(history_size), 1)
        emb = init["emb"].unsqueeze(1).expand(batch, samples, *init["emb"].shape[1:])
        info["emb"] = emb
        emb = emb.reshape(batch * samples, *emb.shape[2:]).clone()
        act = act_0.reshape(batch * samples, *act_0.shape[2:])
        future = act_future.reshape(batch * samples, *act_future.shape[2:])
        if extras is not None:
            extra = extras.unsqueeze(1).expand(
                batch, samples, extras.size(1), extras.size(-1)
            )
            extra = extra.reshape(batch * samples, extras.size(1), extras.size(-1))
            if extra.size(1) != act.size(1):
                raise ValueError(
                    f"action extras {tuple(extra.shape)} incompatible with "
                    f"observed action {tuple(act.shape)}"
                )
            act = torch.cat([act, extra], dim=-1)
            future = torch.cat(
                [future, future.new_zeros(*future.shape[:-1], extra.size(-1))],
                dim=-1,
            )
        n_steps = horizon - observed
        try:
            for step in range(n_steps):
                emb = self._rollout_append(emb, act, window)
                act = torch.cat([act, future[:, step : step + 1]], dim=1)
            emb = self._rollout_append(emb, act, window)
        finally:
            self._predict_time_start = None
        info["predicted_emb"] = emb.reshape(batch, samples, *emb.shape[1:])
        return info

    def _rollout_append(
        self, emb: torch.Tensor, act: torch.Tensor, window: int
    ) -> torch.Tensor:
        if self._needs_action_extras():
            act = self._pad_action_to_encoder(act)
        act_emb = self.action_encoder(act)
        length = int(emb.size(1))
        take = min(window, length)
        self._predict_time_start = length - take
        pred = self.predict(emb[:, -take:], act_emb[:, -take:])[:, -1:]
        return torch.cat([emb, pred], dim=1)

    def criterion(self, info_dict: dict[str, Any]) -> torch.Tensor:
        from .jepa import JEPA

        return JEPA.criterion(self, info_dict)

    def get_cost(self, info_dict: dict[str, Any], action_candidates: torch.Tensor) -> torch.Tensor:
        from .jepa import JEPA

        return JEPA.get_cost(self, info_dict, action_candidates)
