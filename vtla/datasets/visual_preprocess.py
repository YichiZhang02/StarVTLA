"""Serializable visual preprocessing contract shared by datasets and checkpoints."""

from __future__ import annotations

from typing import Any


VISUAL_PREPROCESS_VERSION = 1
TACTILE_UINT8_ENCODING = "tactile_u8_linear_v1"


def make_visual_preprocess(
    *,
    size: int,
    wrist_undistort: bool,
    tactile_encoding: str | None,
) -> dict[str, Any]:
    if size <= 0:
        raise ValueError(f"Visual resize size must be positive, got {size}")
    return {
        "version": VISUAL_PREPROCESS_VERSION,
        "wrist_undistort": bool(wrist_undistort),
        "wrist_crop": None,
        "rotation_degrees": 0,
        "resize": {
            "height": int(size),
            "width": int(size),
            "mode": "stretch",
            "interpolation": "lanczos",
        },
        "tactile_encoding": tactile_encoding,
    }


def validate_visual_preprocess(value: dict[str, Any] | None) -> None:
    if value is None:
        return
    if value.get("version") != VISUAL_PREPROCESS_VERSION:
        raise ValueError(
            f"Unsupported visual_preprocess version {value.get('version')!r}; "
            f"expected {VISUAL_PREPROCESS_VERSION}"
        )
    if not isinstance(value.get("wrist_undistort"), bool):
        raise ValueError("visual_preprocess.wrist_undistort must be boolean")
    crop = value.get("wrist_crop")
    if crop is not None and (not isinstance(crop, int) or crop <= 0):
        raise ValueError("visual_preprocess.wrist_crop must be null or a positive integer")
    if value.get("rotation_degrees") not in {0, 90, 180, 270}:
        raise ValueError("visual_preprocess.rotation_degrees must be one of 0/90/180/270")
    resize = value.get("resize")
    if not isinstance(resize, dict):
        raise ValueError("visual_preprocess.resize must be an object")
    if resize.get("mode") != "stretch":
        raise ValueError("visual_preprocess.resize.mode must be 'stretch'")
    if resize.get("interpolation") != "lanczos":
        raise ValueError("visual_preprocess.resize.interpolation must be 'lanczos'")
    for axis in ("height", "width"):
        size = resize.get(axis)
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"visual_preprocess.resize.{axis} must be a positive integer")
    tactile_encoding = value.get("tactile_encoding")
    if tactile_encoding not in {None, TACTILE_UINT8_ENCODING}:
        raise ValueError(
            "visual_preprocess.tactile_encoding must be null or "
            f"{TACTILE_UINT8_ENCODING!r}"
        )
