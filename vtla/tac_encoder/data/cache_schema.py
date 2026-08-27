"""Versioned on-disk schema for tactile backbone caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CACHE_VERSION = "tactile_backbone_npy_v2_compact"
SPLIT_NAME_TO_ID = {"train": 0, "val": 1, "test": 2}

REQUIRED_ARRAYS = {
    "frames.npy",
    "valid.npy",
    "timestamps.npy",
    "frame_index.npy",
    "episode_index.npy",
    "sensor_names.npy",
    "contact_scores.npy",
    "contact_mask.npy",
    "windows.npy",
    "window_anchor.npy",
    "split.npy",
    "cache_version.npy",
    "image_size.npy",
    "num_frames.npy",
    "frame_stride.npy",
    "fps.npy",
    "source_signature.npy",
    "contact_method.npy",
    "contact_enter_threshold.npy",
    "contact_exit_threshold.npy",
    "contact_debounce_frames.npy",
    "contact_smoothing_frames.npy",
    "contact_top_fraction.npy",
    "anchor_contact_policy.npy",
}


def stable_hash(data: object) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_signature(path: str | Path) -> str:
    """Hash immutable cache metadata and array shape/dtype contracts."""
    cache = validate_cache(path)
    arrays = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in cache.arrays.items()
    }
    metadata = {
        "arrays": arrays,
        "cache_version": cache.scalar("cache_version"),
        "source_signature": cache.scalar("source_signature"),
        "image_size": cache.scalar("image_size"),
        "num_frames": cache.scalar("num_frames"),
        "frame_stride": cache.scalar("frame_stride"),
        "contact_method": cache.scalar("contact_method"),
        "contact_enter_threshold": cache.scalar("contact_enter_threshold"),
        "contact_exit_threshold": cache.scalar("contact_exit_threshold"),
        "contact_debounce_frames": cache.scalar("contact_debounce_frames"),
        "contact_smoothing_frames": cache.scalar("contact_smoothing_frames"),
        "contact_top_fraction": cache.scalar("contact_top_fraction"),
        "anchor_contact_policy": cache.scalar("anchor_contact_policy"),
    }
    return stable_hash(metadata)


@dataclass
class TactileCache:
    root: Path
    arrays: dict[str, np.ndarray]

    def scalar(self, name: str):
        value = self.arrays[f"{name}.npy"]
        return value.item()

    def close(self) -> None:
        for array in self.arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


def validate_cache(path: str | Path, mmap_mode: str = "r") -> TactileCache:
    root = Path(path)
    missing = sorted(name for name in REQUIRED_ARRAYS if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"Incomplete tactile cache {root}: missing {missing}")
    arrays = {name: np.load(root / name, mmap_mode=mmap_mode, allow_pickle=False) for name in REQUIRED_ARRAYS}
    cache = TactileCache(root=root, arrays=arrays)
    if cache.scalar("cache_version") != CACHE_VERSION:
        raise ValueError(
            f"Unsupported tactile cache version {cache.scalar('cache_version')!r}; expected {CACHE_VERSION!r}"
        )

    frames = arrays["frames.npy"]
    valid = arrays["valid.npy"]
    timestamps = arrays["timestamps.npy"]
    frame_index = arrays["frame_index.npy"]
    episode_index = arrays["episode_index.npy"]
    sensor_names = arrays["sensor_names.npy"]
    scores = arrays["contact_scores.npy"]
    mask = arrays["contact_mask.npy"]
    windows = arrays["windows.npy"]
    anchors = arrays["window_anchor.npy"]
    split = arrays["split.npy"]
    frame_count, sensor_count = frames.shape[:2] if frames.ndim == 5 else (-1, -1)
    image_size = int(cache.scalar("image_size"))
    num_frames = int(cache.scalar("num_frames"))
    expected = (frame_count, sensor_count, image_size, image_size, 3)
    if frames.dtype != np.uint8 or frames.shape != expected:
        raise ValueError(f"frames.npy must be uint8 {expected}, got {frames.dtype} {frames.shape}")
    if valid.shape != (frame_count, sensor_count) or valid.dtype != np.bool_:
        raise ValueError("valid.npy has an invalid shape or dtype")
    if scores.shape != valid.shape or scores.dtype != np.float32:
        raise ValueError("contact_scores.npy has an invalid shape or dtype")
    if mask.shape != valid.shape or mask.dtype != np.bool_:
        raise ValueError("contact_mask.npy has an invalid shape or dtype")
    for name, array, dtype in (
        ("timestamps", timestamps, np.float64),
        ("frame_index", frame_index, np.int64),
        ("episode_index", episode_index, np.int64),
    ):
        if array.shape != (frame_count,):
            raise ValueError(f"{name}.npy has an invalid shape")
        if array.dtype != dtype:
            raise ValueError(f"{name}.npy must have dtype {dtype}, got {array.dtype}")
    if sensor_names.shape != (sensor_count,) or sensor_names.dtype.kind not in {"U", "S"}:
        raise ValueError("sensor_names.npy has an invalid shape or dtype")
    if windows.ndim != 2 or windows.shape[1] != num_frames or windows.dtype != np.int64:
        raise ValueError("windows.npy has an invalid shape or dtype")
    if anchors.shape != (len(windows),) or anchors.dtype != np.int64:
        raise ValueError("window_anchor.npy has an invalid shape or dtype")
    if split.shape != (len(windows),) or split.dtype != np.uint8:
        raise ValueError("split.npy has an invalid shape or dtype")
    if np.any(split > max(SPLIT_NAME_TO_ID.values())):
        raise ValueError("split.npy contains an unknown split ID")
    if not np.isfinite(scores).all():
        raise ValueError("contact_scores.npy contains non-finite values")
    if len(windows):
        if int(windows.min()) < 0 or int(windows.max()) >= frame_count:
            raise ValueError("windows.npy contains out-of-range frame rows")
        if not np.array_equal(windows[:, -1], anchors):
            raise ValueError("window anchors must equal the last window row")
        episode_rows = episode_index[windows]
        if np.any(episode_rows != episode_rows[:, :1]):
            raise ValueError("windows.npy contains a cross-episode window")
        expected_steps = int(cache.scalar("frame_stride"))
        if np.any(np.diff(frame_index[windows], axis=1) != expected_steps):
            raise ValueError("windows.npy does not follow frame_stride")
        if not np.all(valid[windows]):
            raise ValueError("windows.npy contains invalid sensor frames")
        fps = float(cache.scalar("fps"))
        expected_seconds = expected_steps / fps
        tolerance = max(1e-6, 0.51 / fps)
        if not np.allclose(
            np.diff(timestamps[windows], axis=1), expected_seconds, atol=tolerance, rtol=0
        ):
            raise ValueError("windows.npy does not follow the expected timestamp stride")
        anchor_contact = mask[anchors]
        policy = str(cache.scalar("anchor_contact_policy"))
        keep = anchor_contact.any(axis=1) if policy == "any" else anchor_contact.all(axis=1)
        if not np.all(keep):
            raise ValueError("windows.npy contains an anchor that violates the contact policy")
    return cache
