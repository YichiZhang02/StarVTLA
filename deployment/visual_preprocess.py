"""Apply the checkpoint-owned rotation and resize contract to online images."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from vtla.datasets.visual_preprocess import validate_visual_preprocess


def apply_visual_preprocess(
    observation: dict[str, Any], contract: dict[str, Any] | None
) -> dict[str, Any]:
    if contract is None:
        raise ValueError(
            "Checkpoint is missing visual_preprocess; reprocess the dataset and retrain the policy."
        )
    validate_visual_preprocess(contract)
    resize = contract["resize"]
    target = (int(resize["width"]), int(resize["height"]))
    rotation = int(contract["rotation_degrees"])
    result = observation.copy()
    for key, value in observation.items():
        if not isinstance(value, np.ndarray) or value.ndim != 3:
            continue
        if "image" not in key and "cam" not in key and "tactile" not in key:
            continue
        image = value
        if rotation:
            image = np.rot90(image, k=rotation // 90).copy()
        if (image.shape[1], image.shape[0]) != target:
            image = cv2.resize(image, target, interpolation=cv2.INTER_LANCZOS4)
        result[key] = image
    return result
