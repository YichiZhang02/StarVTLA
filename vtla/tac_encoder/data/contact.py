"""Contact detection for ``tactile_u8_linear_v1`` frames.

Scores are computed on source-resolution uint8 frames before resize. Hysteresis
and smoothing are applied independently for every episode and sensor.
"""

from __future__ import annotations

import math

import numpy as np


CONTACT_METHOD = "neutral_top1_residual_hysteresis_v1"


def neutral_residual_topk_score(
    frames: np.ndarray,
    top_fraction: float = 0.01,
) -> np.ndarray:
    """Return the mean of the largest residual pixels for each HWC uint8 frame.

    ``frames`` may have any leading dimensions and must end in ``[H, W, 3]``.
    The three channels represent depth, deformation-x, and deformation-y.
    """
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        raise TypeError(f"contact scorer requires uint8 input, got {frames.dtype}")
    if frames.ndim < 3 or frames.shape[-1] != 3:
        raise ValueError(f"expected [..., H, W, 3], got {frames.shape}")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")

    depth = frames[..., 0].astype(np.float32)
    deform_x = frames[..., 1].astype(np.float32) - 128.0
    deform_y = frames[..., 2].astype(np.float32) - 128.0
    residual = np.sqrt(depth * depth + (deform_x * deform_x + deform_y * deform_y) / 2.0)
    leading_shape = residual.shape[:-2]
    flat = residual.reshape(*leading_shape, -1)
    count = max(1, int(math.ceil(flat.shape[-1] * top_fraction)))
    top = np.partition(flat, flat.shape[-1] - count, axis=-1)[..., -count:]
    return top.mean(axis=-1, dtype=np.float32).astype(np.float32, copy=False)


def centered_moving_average(scores: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average with edge padding on one episode only."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError(f"expected one-dimensional scores, got {scores.shape}")
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if scores.size == 0 or window == 1:
        return scores.copy()
    radius = window // 2
    padded = np.pad(scores, (radius, radius), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def hysteresis_debounce(
    scores: np.ndarray,
    enter_threshold: float = 8.0,
    exit_threshold: float = 6.0,
    debounce_frames: int = 3,
) -> np.ndarray:
    """Apply non-retroactive hysteresis/debounce to one episode's scores."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError(f"expected one-dimensional scores, got {scores.shape}")
    if exit_threshold >= enter_threshold:
        raise ValueError("exit_threshold must be smaller than enter_threshold")
    if debounce_frames <= 0:
        raise ValueError("debounce_frames must be positive")

    result = np.zeros(scores.shape, dtype=np.bool_)
    in_contact = False
    enter_count = 0
    exit_count = 0
    for index, score in enumerate(scores):
        if in_contact:
            if score <= exit_threshold:
                exit_count += 1
                if exit_count >= debounce_frames:
                    in_contact = False
                    exit_count = 0
            else:
                exit_count = 0
        else:
            if score >= enter_threshold:
                enter_count += 1
                if enter_count >= debounce_frames:
                    in_contact = True
                    enter_count = 0
            else:
                enter_count = 0
        result[index] = in_contact
    return result


def compute_contact_mask(
    raw_scores: np.ndarray,
    episode_index: np.ndarray,
    *,
    smoothing_frames: int = 5,
    enter_threshold: float = 8.0,
    exit_threshold: float = 6.0,
    debounce_frames: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth and threshold ``[F,S]`` raw scores without crossing episodes."""
    raw_scores = np.asarray(raw_scores, dtype=np.float32)
    episode_index = np.asarray(episode_index)
    if raw_scores.ndim != 2:
        raise ValueError(f"raw_scores must have shape [F,S], got {raw_scores.shape}")
    if episode_index.shape != (raw_scores.shape[0],):
        raise ValueError("episode_index length does not match raw_scores")

    smoothed = np.empty_like(raw_scores)
    contact = np.zeros_like(raw_scores, dtype=np.bool_)
    if raw_scores.shape[0] == 0:
        return smoothed, contact
    boundaries = np.flatnonzero(np.r_[True, episode_index[1:] != episode_index[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        for sensor in range(raw_scores.shape[1]):
            values = centered_moving_average(raw_scores[start:stop, sensor], smoothing_frames)
            smoothed[start:stop, sensor] = values
            contact[start:stop, sensor] = hysteresis_debounce(
                values,
                enter_threshold=enter_threshold,
                exit_threshold=exit_threshold,
                debounce_frames=debounce_frames,
            )
    return smoothed, contact
