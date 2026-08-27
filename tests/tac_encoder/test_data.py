from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vtla.tac_encoder.data.cache_schema import CACHE_VERSION, validate_cache
from vtla.tac_encoder.data.contact import compute_contact_mask, neutral_residual_topk_score
from vtla.tac_encoder.data.npy_tactile_dataset import (
    WeightedMixtureSampler,
    build_training_dataset,
    resolve_tactile_dataset,
)
from vtla.tac_encoder.process_backbone_data import _shrink_npy_first_axis, build_windows


def _scalar(root: Path, name: str, value) -> None:
    np.save(root / f"{name}.npy", np.asarray(value))


def _write_cache(
    root: Path,
    *,
    windows: int,
    sensor_names=("finger0",),
    episode=0,
    image_size: int = 8,
) -> None:
    root.mkdir(parents=True)
    frame_count = windows + 6
    sensor_count = len(sensor_names)
    np.save(
        root / "frames.npy",
        np.zeros((frame_count, sensor_count, image_size, image_size, 3), dtype=np.uint8),
    )
    np.save(root / "valid.npy", np.ones((frame_count, sensor_count), dtype=np.bool_))
    np.save(root / "timestamps.npy", np.arange(frame_count, dtype=np.float64) / 30)
    np.save(root / "frame_index.npy", np.arange(frame_count, dtype=np.int64))
    np.save(root / "episode_index.npy", np.full(frame_count, episode, dtype=np.int64))
    np.save(root / "sensor_names.npy", np.asarray(sensor_names))
    np.save(root / "contact_scores.npy", np.ones((frame_count, sensor_count), dtype=np.float32) * 9)
    np.save(root / "contact_mask.npy", np.ones((frame_count, sensor_count), dtype=np.bool_))
    rows = np.asarray([[i, i + 2, i + 4, i + 6] for i in range(windows)], dtype=np.int64)
    np.save(root / "windows.npy", rows)
    np.save(root / "window_anchor.npy", rows[:, -1])
    np.save(root / "split.npy", np.zeros(windows, dtype=np.uint8))
    _scalar(root, "cache_version", CACHE_VERSION)
    _scalar(root, "image_size", image_size)
    _scalar(root, "num_frames", 4)
    _scalar(root, "frame_stride", 2)
    _scalar(root, "fps", 30.0)
    _scalar(root, "source_signature", "test-source")
    _scalar(root, "contact_method", "neutral_top1_residual_hysteresis_v1")
    _scalar(root, "contact_enter_threshold", 8.0)
    _scalar(root, "contact_exit_threshold", 6.0)
    _scalar(root, "contact_debounce_frames", 3)
    _scalar(root, "contact_smoothing_frames", 5)
    _scalar(root, "contact_top_fraction", 0.01)
    _scalar(root, "anchor_contact_policy", "any")


def test_contact_score_and_nonretroactive_debounce() -> None:
    frames = np.zeros((2, 10, 10, 3), dtype=np.uint8)
    frames[..., 1:] = 128
    frames[1, 0, 0, 0] = 10
    scores = neutral_residual_topk_score(frames, top_fraction=0.01)
    np.testing.assert_allclose(scores, [0, 10])

    raw = np.asarray([[0], [9], [9], [9], [7], [5], [5], [5]], dtype=np.float32)
    smoothed, contact = compute_contact_mask(
        raw,
        np.zeros(len(raw), dtype=np.int64),
        smoothing_frames=1,
        enter_threshold=8,
        exit_threshold=6,
        debounce_frames=3,
    )
    np.testing.assert_array_equal(smoothed[:, 0], raw[:, 0])
    np.testing.assert_array_equal(contact[:, 0], [False, False, False, True, True, True, True, False])


def test_shrink_npy_first_axis_preserves_compact_rows(tmp_path: Path) -> None:
    path = tmp_path / "frames.npy"
    frames = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.uint8,
        shape=(10, 2, 3, 3, 3),
        version=(2, 0),
    )
    expected = np.arange(4 * 2 * 3 * 3 * 3, dtype=np.uint8).reshape(4, 2, 3, 3, 3)
    frames[:4] = expected
    frames.flush()
    del frames

    original_size = path.stat().st_size
    _shrink_npy_first_axis(path, 4)
    compact = np.load(path, mmap_mode="r", allow_pickle=False)

    assert compact.shape == expected.shape
    assert path.stat().st_size < original_size
    np.testing.assert_array_equal(compact, expected)


def test_window_generation_respects_episode_stride_and_anchor_contact() -> None:
    frame_index = np.r_[np.arange(9), np.arange(9)]
    episodes = np.r_[np.zeros(9), np.ones(9)].astype(np.int64)
    timestamps = np.tile(np.arange(9) / 30, 2)
    valid = np.ones((18, 1), dtype=np.bool_)
    contact = np.zeros((18, 1), dtype=np.bool_)
    contact[[6, 8, 15], 0] = True
    windows, anchors = build_windows(
        frame_index,
        episodes,
        timestamps,
        valid,
        contact,
        num_frames=4,
        frame_stride=2,
        fps=30,
        anchor_contact_policy="any",
    )
    np.testing.assert_array_equal(windows, [[0, 2, 4, 6], [2, 4, 6, 8], [9, 11, 13, 15]])
    np.testing.assert_array_equal(anchors, [6, 8, 15])


def test_mmap_dataset_and_weighted_mixture_sampler(tmp_path: Path) -> None:
    catalog = tmp_path / "data"
    cache_root = tmp_path / "cache"
    for dataset_id, count in (("small", 2), ("large", 20)):
        (catalog / dataset_id / "meta").mkdir(parents=True)
        (catalog / dataset_id / "meta" / "info.json").write_text("{}")
        _write_cache(cache_root / dataset_id, windows=count)
    registry = tmp_path / "mixtures.yaml"
    registry.write_text(
        "version: 1\nmixtures:\n  pair:\n    datasets:\n"
        "      - dataset_id: small\n        weight: 3\n        root: " + str(catalog / "small") + "\n"
        "      - dataset_id: large\n        weight: 1\n        root: " + str(catalog / "large") + "\n"
    )
    resolved = resolve_tactile_dataset(
        "pair", cache_root=cache_root, dataset_catalog_root=catalog, mixture_config=registry
    )
    dataset = build_training_dataset(resolved, split="train")
    assert isinstance(dataset.datasets[0].cache.arrays["frames.npy"], np.memmap)
    assert dataset[0]["images"].shape == (1, 4, 3, 8, 8)

    sampler = WeightedMixtureSampler(dataset, num_samples=4000, seed=7)
    sampled = list(sampler)
    small_count = sum(index < dataset.offsets[1] for index in sampled)
    assert 0.72 < small_count / len(sampled) < 0.78
    validate_cache(cache_root / "small").close()


def test_default_cache_location_is_inside_each_dataset(tmp_path: Path) -> None:
    catalog = tmp_path / "data"
    source = catalog / "tiny"
    (source / "meta").mkdir(parents=True)
    (source / "meta" / "info.json").write_text("{}")
    _write_cache(source / "tactile_backbone_cache", windows=1)

    resolved = resolve_tactile_dataset(
        "tiny",
        dataset_catalog_root=catalog,
        mixture_config=tmp_path / "missing-mixtures.yaml",
    )

    assert Path(resolved.members[0].cache_root) == source / "tactile_backbone_cache"
