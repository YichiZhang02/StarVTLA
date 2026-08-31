"""Checkpoint-backed pooled feature extraction for downstream policies."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .common.backbone import FeatureTokens
from .registry import build_backbone, get_encoder_spec


def _checkpoint_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        path = next(
            (candidate for candidate in (path / "best.pth", path / "last.pth") if candidate.is_file()),
            path,
        )
    return path


class TactileBackboneFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        pool_size: int = 3,
        freeze: bool = False,
        architecture_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.architecture_config = dict(architecture_config or {})
        self.pool_size = int(pool_size)
        self.feature_dim = int(backbone.feature_dim)
        self.image_size = int(backbone.image_size)
        self.compute_dtype = None
        self.model_id = str(backbone.model_id)
        self.num_frames = int(backbone.num_frames)
        self.tokens_per_sensor = backbone.pooled_tokens_per_sensor(self.pool_size)
        self.freeze = bool(freeze)
        if self.freeze:
            self.backbone.requires_grad_(False)

    @staticmethod
    def _resolve_architecture_config(model_id: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Return the complete, path-free constructor config needed by a policy checkpoint."""
        try:
            backbone_class = get_encoder_spec(model_id).backbone_class
        except ValueError as error:
            raise ValueError(f"Unknown tactile backbone model_id: {model_id!r}") from error
        signature = inspect.signature(backbone_class)
        resolved: dict[str, Any] = {"model_id": model_id}
        for name, parameter in signature.parameters.items():
            if name in {"pretrained_path", "checkpoint_source_grid", "include_decoder"}:
                continue
            if name in kwargs:
                resolved[name] = kwargs[name]
            elif parameter.default is not inspect.Parameter.empty:
                resolved[name] = parameter.default
        return resolved

    @classmethod
    def from_config(
        cls,
        architecture_config: dict[str, Any],
        *,
        pool_size: int = 3,
        freeze: bool = False,
    ) -> "TactileBackboneFeatureExtractor":
        """Build an uninitialized extractor whose weights will come from the policy checkpoint."""
        architecture_config = dict(architecture_config)
        try:
            model_id = str(architecture_config.pop("model_id"))
        except KeyError as error:
            raise ValueError("tactile_encoder_config is missing model_id") from error
        architecture_config.pop("pretrained_path", None)
        backbone_class = get_encoder_spec(model_id).backbone_class
        build_kwargs = dict(architecture_config)
        if "include_decoder" in inspect.signature(backbone_class).parameters:
            build_kwargs["include_decoder"] = False
        backbone = build_backbone(model_id, pretrained_path="", **build_kwargs)
        if hasattr(backbone, "discard_training_modules"):
            backbone.discard_training_modules()
        return cls(
            backbone,
            pool_size=pool_size,
            freeze=freeze,
            architecture_config={"model_id": model_id, **architecture_config},
        )

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        pool_size: int = 3,
        freeze: bool = False,
    ) -> "TactileBackboneFeatureExtractor":
        checkpoint_path = _checkpoint_file(path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        if not isinstance(checkpoint, dict) or "model_id" not in checkpoint:
            raise ValueError(f"Not a unified tactile backbone checkpoint: {checkpoint_path}")
        args = checkpoint.get("args", {})
        spec = get_encoder_spec(checkpoint["model_id"])
        kwargs = spec.checkpoint_kwargs(args)
        architecture_config = cls._resolve_architecture_config(checkpoint["model_id"], kwargs)
        if "include_decoder" in inspect.signature(spec.backbone_class).parameters:
            kwargs["include_decoder"] = False
        backbone = build_backbone(checkpoint["model_id"], **kwargs)
        if checkpoint.get("format_version") == 2:
            encoder_state = checkpoint.get("encoder")
            if not isinstance(encoder_state, dict):
                raise ValueError(f"Checkpoint has no encoder state: {checkpoint_path}")
        else:
            legacy_state = checkpoint.get("model")
            if not isinstance(legacy_state, dict):
                raise ValueError(f"Not a unified tactile backbone checkpoint: {checkpoint_path}")
            prefixes = spec.checkpoint_prefixes
            encoder_state = {
                key: value for key, value in legacy_state.items() if key.startswith(prefixes)
            }
        current = backbone.state_dict()
        mismatched = {
            key: (
                tuple(value.shape),
                None if key not in current else tuple(current[key].shape),
            )
            for key, value in encoder_state.items()
            if key not in current or current[key].shape != value.shape
        }
        if mismatched:
            raise ValueError(f"Encoder checkpoint shape mismatch: {list(mismatched.items())[:5]}")
        missing_expected = [
            key
            for key in current
            if key.startswith(spec.checkpoint_prefixes) and key not in encoder_state
        ]
        if missing_expected:
            raise ValueError(f"Encoder checkpoint is incomplete: {missing_expected[:8]}")
        backbone.load_state_dict(encoder_state, strict=False)
        if hasattr(backbone, "discard_training_modules"):
            backbone.discard_training_modules()
        return cls(
            backbone,
            pool_size=pool_size,
            freeze=freeze,
            architecture_config=architecture_config,
        )

    def forward(self, images: Tensor) -> FeatureTokens:
        if images.ndim == 5:
            images = images.unsqueeze(1)
        if images.ndim != 6:
            raise ValueError(f"expected [B,S,T,C,H,W] or [B,T,C,H,W], got {tuple(images.shape)}")
        if images.shape[2] != self.num_frames:
            raise ValueError(f"checkpoint requires T={self.num_frames}, got T={images.shape[2]}")
        if self.freeze:
            with torch.no_grad():
                return self.backbone.extract_pooled_features(images, self.pool_size)
        return self.backbone.extract_pooled_features(images, self.pool_size)
