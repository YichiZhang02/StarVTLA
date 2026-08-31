"""AnyTouch1 checkpoint reconstruction configuration."""

from collections.abc import Mapping
from typing import Any


def checkpoint_kwargs(args: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "num_frames": int(args.get("num_frames", 4)),
        "image_size": int(args.get("image_size", 224)),
        "arch": args.get("anytouch1_arch", "vit_l"),
        "mask_ratio": float(args.get("mask_ratio", 0.75)),
        "pretrained_path": "",
    }
