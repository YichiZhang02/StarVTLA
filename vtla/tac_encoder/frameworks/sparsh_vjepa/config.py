"""Sparsh V-JEPA checkpoint reconstruction configuration."""

from collections.abc import Mapping
from typing import Any


def checkpoint_kwargs(args: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_frames": int(args.get("num_frames", 4)),
        "image_size": int(args.get("image_size", 224)),
        "pretrained_path": "",
    }
    for target, source in (
        ("embed_dim", "encoder_dim"),
        ("depth", "encoder_depth"),
        ("num_heads", "encoder_heads"),
    ):
        value = args.get(source)
        if value is not None:
            kwargs[target] = value
    return kwargs
