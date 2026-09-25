"""v6 prepared-uint8 contract: BGR -> AREA 56 -> bilinear 42 -> round -> /255."""

import json
from collections import deque
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F

HISTORY = 8
STRIDE = 5


def history_indices(frame_index: int) -> list[int]:
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, Integral)
        or frame_index < 0
    ):
        raise ValueError("frame_index must be a non-negative integer")
    return [max(0, frame_index - 35 + 5 * i) for i in range(8)]


def validate_marker_frame(frame: Any) -> np.ndarray:
    value = np.asarray(frame)
    if (
        value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[-1] != 3
        or min(value.shape[:2]) == 0
    ):
        raise ValueError("Marker frames must be nonempty HWC uint8 with three channels")
    return value


def _prepare_marker_frame(frame: Any, *, color: str) -> np.ndarray:
    value = validate_marker_frame(frame)
    if color == "rgb":
        value = value[..., ::-1]
    # HDF5-derived training videos may already contain the exact v6 42x42
    # prepared marker frame. Keep it unchanged instead of blurring it through
    # the native-frame AREA56 -> bilinear42 path a second time.
    if value.shape[:2] == (42, 42):
        return np.ascontiguousarray(value)
    return cv2.resize(value, (56, 56), interpolation=cv2.INTER_AREA)


def prepare_frame_pair(
    left: Any, right: Any, *, color: str = "rgb"
) -> torch.Tensor:
    """Prepare one current left/right pair with the same v6 pixel contract."""
    if color not in ("rgb", "bgr"):
        raise ValueError("color must be rgb or bgr")
    cells = [
        _prepare_marker_frame(frame, color=color)
        for frame in (left, right)
    ]
    pixels = torch.from_numpy(np.stack(cells)).permute(0, 3, 1, 2).float()
    pixels = F.interpolate(
        pixels, size=(42, 42), mode="bilinear", align_corners=False
    )
    return pixels.clamp(0, 255).round() / 255.0


def prepare_history(
    left: Sequence[Any], right: Sequence[Any], *, color: str = "rgb"
) -> torch.Tensor:
    if len(left) != HISTORY or len(right) != HISTORY:
        raise ValueError(
            "Exactly eight oldest-to-current marker frames per finger are required"
        )
    if color not in ("rgb", "bgr"):
        raise ValueError("color must be rgb or bgr")
    cells = [
        _prepare_marker_frame(frame, color=color)
        for frames in (left, right)
        for frame in frames
    ]
    pixels = torch.from_numpy(np.stack(cells)).permute(0, 3, 1, 2).float()
    pixels = F.interpolate(
        pixels, size=(42, 42), mode="bilinear", align_corners=False
    )
    return pixels.clamp(0, 255).round().reshape(2, 8, 3, 42, 42) / 255.0


class LoadTactileHistory:
    def __init__(self, image_dir: str) -> None:
        self.image_dir = image_dir

    def __call__(self, data: dict) -> dict:
        from tacmind0.data.transforms import LoadImages

        t = data["meta_data"]["frame_index"]
        rows = [json.loads(data["raw_lines"][i]) for i in history_indices(t)]
        loader = LoadImages(
            ["marker_left", "marker_right"], self.image_dir, require_rgb=True
        )
        decoded = [loader(row)["images"] for row in rows]
        frames = [[pair[side] for pair in decoded] for side in (0, 1)]
        data["tactile_pixel_values"] = prepare_history(*frames).unsqueeze(0)
        return data


class MarkerHistory:
    """One instance per simulation lane; append every control tick, not policy request."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.frames = deque(maxlen=36)
        self.frame_index = -1

    def append(self, left: np.ndarray, right: np.ndarray, *, frame_index: int) -> None:
        history_indices(frame_index)
        if frame_index != self.frame_index + 1:
            raise ValueError(
                "Marker history requires every consecutive control tick; reset at episode start"
            )
        left, right = validate_marker_frame(left), validate_marker_frame(right)
        self.frames.append((left.copy(), right.copy()))
        self.frame_index = frame_index

    def selected(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not self.frames:
            raise ValueError("Empty marker history")
        start = max(0, self.frame_index - 35)
        indices = [i - start for i in history_indices(self.frame_index)]
        return tuple([[self.frames[i][side] for i in indices] for side in (0, 1)])

    def tensor(self) -> torch.Tensor:
        return prepare_history(*self.selected())
