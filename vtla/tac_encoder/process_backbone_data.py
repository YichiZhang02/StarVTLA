"""Build versioned tactile reconstruction caches from processed uint8 datasets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.dataset as pa_dataset
from tqdm import tqdm

from vtla.datasets.video_utils import decode_tactile_video_frames_pyav

from .config import TactileDataConfig
from .data.cache_schema import CACHE_VERSION, stable_hash, validate_cache
from .data.contact import CONTACT_METHOD, compute_contact_mask, neutral_residual_topk_score
from .data.npy_tactile_dataset import IN_PLACE_CACHE_DIR, resolve_tactile_dataset


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _tactile_keys(info: dict[str, Any], requested: list[str] | None) -> list[str]:
    features = info.get("features", {})
    keys = requested or [
        key
        for key, feature in features.items()
        if feature.get("tactile_encoding") == "tactile_u8_linear_v1"
    ]
    if not keys:
        raise ValueError("No tactile_u8_linear_v1 features were found")
    for key in keys:
        feature = features.get(key)
        if feature is None:
            raise KeyError(f"Unknown tactile feature {key!r}")
        errors = []
        if feature.get("dtype") != "video":
            errors.append(f"dtype={feature.get('dtype')!r}")
        if feature.get("tactile_encoding") != "tactile_u8_linear_v1":
            errors.append(f"tactile_encoding={feature.get('tactile_encoding')!r}")
        if feature.get("storage_dtype") != "uint8":
            errors.append(f"storage_dtype={feature.get('storage_dtype')!r}")
        shape = feature.get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3:
            errors.append(f"shape={shape!r}")
        if errors:
            raise ValueError(f"Feature {key!r} is not processed uint8 tactile data: {', '.join(errors)}")
    return keys


def _read_columns(root: Path) -> dict[str, np.ndarray]:
    table = pa_dataset.dataset(str(root / "data"), format="parquet").to_table(
        columns=["index", "frame_index", "episode_index", "timestamp"]
    )
    result = {name: table[name].combine_chunks().to_numpy() for name in table.column_names}
    order = np.argsort(result["index"])
    return {name: np.asarray(values[order]) for name, values in result.items()}


def _read_episodes(root: Path) -> list[dict[str, Any]]:
    table = pa_dataset.dataset(str(root / "meta" / "episodes"), format="parquet").to_table()
    episodes = sorted(table.to_pylist(), key=lambda item: int(item["episode_index"]))
    return episodes


def _video_path(root: Path, info: dict, feature_key: str, episode: dict) -> Path:
    feature = info["features"][feature_key]
    template = feature.get("video_path") or info.get("video_path")
    if not template:
        raise ValueError(f"No video_path template for feature {feature_key!r}")
    prefix = f"videos/{feature_key}"
    return root / template.format(
        video_key=feature_key,
        episode_index=int(episode["episode_index"]),
        chunk_index=int(episode[f"{prefix}/chunk_index"]),
        file_index=int(episode[f"{prefix}/file_index"]),
    )


def _resize_uint8_hwc(frames: np.ndarray, image_size: int) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"decoder returned invalid tactile frames: {frames.dtype} {frames.shape}")
    if frames.shape[1:3] == (image_size, image_size):
        return np.ascontiguousarray(frames)
    output = np.empty((len(frames), image_size, image_size, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        output[index] = cv2.resize(
            frame,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
    return output


def _shrink_npy_first_axis(path: Path, new_length: int) -> None:
    """Rewrite a v2 NPY header and truncate an append-style first axis."""
    with path.open("r+b") as handle:
        version = np.lib.format.read_magic(handle)
        if version != (2, 0):
            raise ValueError(f"Expected NPY v2.0 for compacting, got {version}")
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        data_offset = handle.tell()
        if fortran_order or not shape:
            raise ValueError("Only C-order arrays with a leading axis can be compacted")
        if new_length < 0 or new_length > shape[0]:
            raise ValueError(f"Invalid compacted length {new_length} for shape {shape}")
        compact_shape = (new_length, *shape[1:])
        header = {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": compact_shape,
        }
        handle.seek(0)
        np.lib.format.write_array_header_2_0(handle, header)
        if handle.tell() != data_offset:
            raise RuntimeError("NPY header size changed while compacting the leading axis")
        row_bytes = int(dtype.itemsize * np.prod(shape[1:], dtype=np.int64))
        handle.truncate(data_offset + new_length * row_bytes)


@dataclass(frozen=True)
class _EpisodeResult:
    episode_number: int
    source_frames: int
    windows: np.ndarray
    anchors: np.ndarray
    split: np.ndarray


def assign_episode_splits(
    episode_ids: np.ndarray,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[int, int]:
    unique = np.unique(np.asarray(episode_ids, dtype=np.int64))
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")
    shuffled = unique.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    available = max(0, len(shuffled) - 1)
    n_test = min(available, int(round(len(shuffled) * test_ratio)))
    if test_ratio > 0 and available > 0:
        n_test = max(1, n_test)
    available -= n_test
    n_val = min(available, int(round(len(shuffled) * val_ratio)))
    if val_ratio > 0 and available > 0:
        n_val = max(1, n_val)
    mapping = {int(episode): 0 for episode in shuffled}
    for episode in shuffled[:n_test]:
        mapping[int(episode)] = 2
    for episode in shuffled[n_test : n_test + n_val]:
        mapping[int(episode)] = 1
    return mapping


def build_windows(
    frame_index: np.ndarray,
    episode_index: np.ndarray,
    timestamps: np.ndarray,
    valid: np.ndarray,
    contact_mask: np.ndarray,
    *,
    num_frames: int,
    frame_stride: int,
    fps: float,
    anchor_contact_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    anchors = []
    offsets = np.arange(-(num_frames - 1) * frame_stride, 1, frame_stride, dtype=np.int64)
    boundaries = np.flatnonzero(np.r_[True, episode_index[1:] != episode_index[:-1], True])
    tolerance = max(1e-6, 0.51 / float(fps))
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        by_frame = {int(frame_index[row]): row for row in range(start, stop)}
        for anchor in range(start, stop):
            desired = frame_index[anchor] + offsets
            if any(int(value) not in by_frame for value in desired):
                continue
            rows = np.asarray([by_frame[int(value)] for value in desired], dtype=np.int64)
            if not np.all(valid[rows]):
                continue
            anchor_contact = contact_mask[anchor]
            keep = bool(anchor_contact.any()) if anchor_contact_policy == "any" else bool(anchor_contact.all())
            if not keep:
                continue
            expected_delta = np.diff(desired).astype(np.float64) / float(fps)
            if not np.allclose(np.diff(timestamps[rows]), expected_delta, atol=tolerance, rtol=0):
                continue
            windows.append(rows)
            anchors.append(anchor)
    return (
        np.asarray(windows, dtype=np.int64).reshape(-1, num_frames),
        np.asarray(anchors, dtype=np.int64),
    )


def _save_scalar(root: Path, name: str, value: Any) -> None:
    np.save(root / f"{name}.npy", np.asarray(value))


def _source_signature(
    root: Path,
    info: dict,
    config: TactileDataConfig,
    sensor_names: list[str],
    *,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    tolerance_s: float,
) -> str:
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    data_files = sorted((root / "data").rglob("*.parquet"))
    video_files = sorted(
        path
        for sensor_name in sensor_names
        for path in (root / "videos" / sensor_name).rglob("*.mp4")
    )
    payload = {
        "info": info,
        "config": config.to_dict(),
        "split": {"val_ratio": val_ratio, "test_ratio": test_ratio, "seed": split_seed},
        "tolerance_s": tolerance_s,
        "sensor_names": sensor_names,
        "source_files": [
            [str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns]
            for path in episode_files + data_files + video_files
        ],
    }
    return stable_hash(payload)


def build_member_cache(
    dataset_id: str,
    source_root: Path,
    destination: Path,
    config: TactileDataConfig,
    *,
    requested_tactile_keys: list[str] | None,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    tolerance_s: float,
    num_workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    config.validate()
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    info = _load_json(source_root / "meta" / "info.json")
    sensor_names = _tactile_keys(info, requested_tactile_keys)
    signature = _source_signature(
        source_root,
        info,
        config,
        sensor_names,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        tolerance_s=tolerance_s,
    )
    if destination.exists() and not overwrite:
        try:
            cache = validate_cache(destination)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(
                f"Existing cache {destination} is incomplete or uses an older schema; "
                "use --overwrite to rebuild it"
            ) from error
        if cache.scalar("source_signature") != signature:
            raise ValueError(
                f"Existing cache {destination} does not match the source/config; use --overwrite explicitly"
            )
        report = {
            "dataset_id": dataset_id,
            "cache": str(destination),
            "status": "reused",
            "frames": int(cache.arrays["frames.npy"].shape[0]),
            "source_frames": int(info.get("total_frames", cache.arrays["frames.npy"].shape[0])),
            "windows": int(cache.arrays["windows.npy"].shape[0]),
        }
        cache.close()
        return report

    columns = _read_columns(source_root)
    episodes = _read_episodes(source_root)
    total_frames = len(columns["index"])
    if total_frames == 0 or not episodes:
        raise ValueError(f"Processed dataset {dataset_id!r} contains no episodes or frames")
    if total_frames != int(info.get("total_frames", total_frames)):
        raise ValueError("info.json total_frames does not match parquet metadata")
    if not np.array_equal(columns["index"], np.arange(total_frames)):
        raise ValueError("Processed dataset index must be contiguous from zero")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.tmp-", dir=destination.parent))
    try:
        sensor_count = len(sensor_names)
        frames_out = np.lib.format.open_memmap(
            temporary / "frames.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(total_frames, sensor_count, config.image_size, config.image_size, 3),
            version=(2, 0),
        )
        cached_valid = np.ones((total_frames, sensor_count), dtype=np.bool_)
        cached_timestamps = np.empty(total_frames, dtype=np.float64)
        cached_frame_index = np.empty(total_frames, dtype=np.int64)
        cached_episode_index = np.empty(total_frames, dtype=np.int64)
        cached_contact_scores = np.empty((total_frames, sensor_count), dtype=np.float32)
        cached_contact_mask = np.empty((total_frames, sensor_count), dtype=np.bool_)
        full_contact_scores = np.empty((total_frames, sensor_count), dtype=np.float32)
        full_contact_mask = np.empty((total_frames, sensor_count), dtype=np.bool_)
        split_map = assign_episode_splits(
            columns["episode_index"], val_ratio=val_ratio, test_ratio=test_ratio, seed=split_seed
        )

        commit_condition = threading.Condition()
        next_episode_to_commit = 0
        next_cache_row = 0
        aborted = False
        worker_count = min(num_workers, len(episodes))
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 1
        decoder_threads = max(1, available_cpus // worker_count)
        cv2.setNumThreads(decoder_threads)

        def process_episode(episode_number: int, episode: dict[str, Any]) -> _EpisodeResult:
            nonlocal aborted, next_cache_row, next_episode_to_commit
            try:
                start = int(episode["dataset_from_index"])
                stop = int(episode["dataset_to_index"])
                length = stop - start
                if length != int(episode["length"]):
                    raise ValueError(f"Episode {episode['episode_index']} has inconsistent bounds")
                episode_timestamps = columns["timestamp"][start:stop].astype(np.float64)
                episode_frames = []
                raw_scores = np.empty((length, sensor_count), dtype=np.float32)
                for sensor_index, sensor_name in enumerate(sensor_names):
                    from_timestamp = float(episode[f"videos/{sensor_name}/from_timestamp"])
                    query_timestamps = (from_timestamp + episode_timestamps).tolist()
                    decoded = decode_tactile_video_frames_pyav(
                        _video_path(source_root, info, sensor_name, episode),
                        query_timestamps,
                        tolerance_s,
                        return_uint8=True,
                        return_numpy_hwc=True,
                        decoder_threads=decoder_threads,
                    )
                    if len(decoded) != length:
                        raise RuntimeError(
                            f"Decoded {len(decoded)} frames for episode {episode['episode_index']}, "
                            f"expected {length}"
                        )
                    episode_frames.append(decoded)
                    raw_scores[:, sensor_index] = neutral_residual_topk_score(
                        decoded, config.contact_top_fraction
                    )

                episode_ids = columns["episode_index"][start:stop].astype(np.int64)
                contact_scores, contact_mask = compute_contact_mask(
                    raw_scores,
                    episode_ids,
                    smoothing_frames=config.contact_smoothing_frames,
                    enter_threshold=config.contact_enter_threshold,
                    exit_threshold=config.contact_exit_threshold,
                    debounce_frames=config.contact_debounce_frames,
                )
                full_contact_scores[start:stop] = contact_scores
                full_contact_mask[start:stop] = contact_mask
                local_windows, local_anchors = build_windows(
                    columns["frame_index"][start:stop],
                    episode_ids,
                    episode_timestamps,
                    np.ones((length, sensor_count), dtype=np.bool_),
                    contact_mask,
                    num_frames=config.num_frames,
                    frame_stride=config.frame_stride,
                    fps=float(info["fps"]),
                    anchor_contact_policy=config.anchor_contact_policy,
                )
                retained = (
                    np.unique(local_windows.reshape(-1))
                    if len(local_windows)
                    else np.empty(0, dtype=np.int64)
                )
                compact_frames = np.empty(
                    (len(retained), sensor_count, config.image_size, config.image_size, 3),
                    dtype=np.uint8,
                )
                for sensor_index, decoded in enumerate(episode_frames):
                    compact_frames[:, sensor_index] = _resize_uint8_hwc(
                        decoded[retained], config.image_size
                    )

                with commit_condition:
                    while episode_number != next_episode_to_commit and not aborted:
                        commit_condition.wait()
                    if aborted:
                        raise RuntimeError("Episode processing was cancelled after another worker failed")
                    cache_start = next_cache_row
                    cache_stop = cache_start + len(retained)
                    frames_out[cache_start:cache_stop] = compact_frames
                    cached_timestamps[cache_start:cache_stop] = episode_timestamps[retained]
                    cached_frame_index[cache_start:cache_stop] = columns["frame_index"][start:stop][retained]
                    cached_episode_index[cache_start:cache_stop] = episode_ids[retained]
                    cached_contact_scores[cache_start:cache_stop] = contact_scores[retained]
                    cached_contact_mask[cache_start:cache_stop] = contact_mask[retained]

                    source_to_cache = np.full(length, -1, dtype=np.int64)
                    source_to_cache[retained] = np.arange(cache_start, cache_stop, dtype=np.int64)
                    windows = source_to_cache[local_windows]
                    anchors = source_to_cache[local_anchors]
                    split = np.full(
                        len(windows),
                        split_map[int(episode["episode_index"])],
                        dtype=np.uint8,
                    )
                    next_cache_row = cache_stop
                    next_episode_to_commit += 1
                    commit_condition.notify_all()

                return _EpisodeResult(
                    episode_number=episode_number,
                    source_frames=length,
                    windows=windows,
                    anchors=anchors,
                    split=split,
                )
            except BaseException:
                with commit_condition:
                    aborted = True
                    commit_condition.notify_all()
                raise

        episode_results: list[_EpisodeResult | None] = [None] * len(episodes)
        with tqdm(
            total=total_frames,
            desc=f"[{dataset_id}] decode/select",
            unit="frame",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as progress:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(process_episode, episode_number, episode): episode_number
                    for episode_number, episode in enumerate(episodes)
                }
                for future in as_completed(futures):
                    result = future.result()
                    episode_results[result.episode_number] = result
                    progress.update(result.source_frames)
                    progress.set_postfix(
                        episodes=f"{sum(item is not None for item in episode_results)}/{len(episodes)}",
                        cached=next_cache_row,
                    )
        frames_out.flush()
        del frames_out
        cached_frame_count = next_cache_row
        _shrink_npy_first_axis(temporary / "frames.npy", cached_frame_count)
        completed_results = [result for result in episode_results if result is not None]
        windows = np.concatenate(
            [result.windows for result in completed_results], axis=0
        ).reshape(-1, config.num_frames)
        anchors = np.concatenate([result.anchors for result in completed_results])
        split = np.concatenate([result.split for result in completed_results])

        np.save(temporary / "valid.npy", cached_valid[:cached_frame_count])
        np.save(temporary / "timestamps.npy", cached_timestamps[:cached_frame_count])
        np.save(temporary / "frame_index.npy", cached_frame_index[:cached_frame_count])
        np.save(temporary / "episode_index.npy", cached_episode_index[:cached_frame_count])
        np.save(temporary / "sensor_names.npy", np.asarray(sensor_names, dtype=np.str_))
        np.save(temporary / "contact_scores.npy", cached_contact_scores[:cached_frame_count])
        np.save(temporary / "contact_mask.npy", cached_contact_mask[:cached_frame_count])
        np.save(temporary / "windows.npy", windows)
        np.save(temporary / "window_anchor.npy", anchors)
        np.save(temporary / "split.npy", split)
        _save_scalar(temporary, "cache_version", CACHE_VERSION)
        _save_scalar(temporary, "image_size", config.image_size)
        _save_scalar(temporary, "num_frames", config.num_frames)
        _save_scalar(temporary, "frame_stride", config.frame_stride)
        _save_scalar(temporary, "fps", float(info["fps"]))
        _save_scalar(temporary, "source_signature", signature)
        _save_scalar(temporary, "contact_method", CONTACT_METHOD)
        _save_scalar(temporary, "contact_enter_threshold", config.contact_enter_threshold)
        _save_scalar(temporary, "contact_exit_threshold", config.contact_exit_threshold)
        _save_scalar(temporary, "contact_debounce_frames", config.contact_debounce_frames)
        _save_scalar(temporary, "contact_smoothing_frames", config.contact_smoothing_frames)
        _save_scalar(temporary, "contact_top_fraction", config.contact_top_fraction)
        _save_scalar(temporary, "anchor_contact_policy", config.anchor_contact_policy)
        report = {
            "dataset_id": dataset_id,
            "source_root": str(source_root),
            "cache": str(destination),
            "status": "built",
            "frames": cached_frame_count,
            "source_frames": total_frames,
            "cached_frame_fraction": cached_frame_count / total_frames,
            "episodes": len(episodes),
            "sensors": sensor_names,
            "contact_frames_per_sensor": full_contact_mask.sum(axis=0).astype(int).tolist(),
            "windows": len(windows),
            "windows_by_split": {
                name: int(np.sum(split == split_id))
                for name, split_id in (("train", 0), ("val", 1), ("test", 2))
            },
            "score_quantiles": {
                str(q): np.quantile(full_contact_scores, q, axis=0).astype(float).tolist()
                for q in (0.01, 0.1, 0.5, 0.9, 0.99)
            },
            "num_workers": worker_count,
            "decoder_threads_per_worker": decoder_threads,
            "resize_interpolation": "opencv_inter_area",
            "config": config.to_dict(),
            "source_signature": signature,
        }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_cache(temporary).close()
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_member_metadata(resolved, requested_tactile_keys: list[str] | None) -> None:
    reference = None
    for member in resolved.members:
        info = _load_json(Path(member.root) / "meta" / "info.json")
        keys = _tactile_keys(info, requested_tactile_keys)
        current = (info.get("fps"), info.get("robot_type"), info.get("features"), keys)
        if reference is None:
            reference = current
        elif current != reference:
            raise ValueError(
                f"Mixture member {member.dataset_id!r} has incompatible FPS, robot type, feature schema, "
                "or tactile sensor ordering"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--dataset_catalog_root", type=Path, default=Path("playground/data"))
    parser.add_argument("--mixture_config", type=Path, default=Path("configs/data_mixtures.yaml"))
    parser.add_argument(
        "--cache_root",
        type=Path,
        help=(
            "Optional centralized cache root. By default each processed dataset writes "
            f"<dataset_root>/{IN_PLACE_CACHE_DIR}."
        ),
    )
    parser.add_argument("--tactile_keys", nargs="+")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--contact_enter_threshold", type=float, default=8.0)
    parser.add_argument("--contact_exit_threshold", type=float, default=6.0)
    parser.add_argument("--contact_debounce_frames", type=int, default=3)
    parser.add_argument("--contact_smoothing_frames", type=int, default=5)
    parser.add_argument("--contact_top_fraction", type=float, default=0.01)
    parser.add_argument("--anchor_contact_policy", choices=["any", "all"], default="any")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--tolerance_s", type=float, default=0.1)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Parallel episode decode workers (default: 4).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TactileDataConfig(
        image_size=args.image_size,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        contact_enter_threshold=args.contact_enter_threshold,
        contact_exit_threshold=args.contact_exit_threshold,
        contact_debounce_frames=args.contact_debounce_frames,
        contact_smoothing_frames=args.contact_smoothing_frames,
        contact_top_fraction=args.contact_top_fraction,
        anchor_contact_policy=args.anchor_contact_policy,
    )
    resolved = resolve_tactile_dataset(
        args.dataset_id,
        cache_root=args.cache_root,
        dataset_catalog_root=args.dataset_catalog_root,
        mixture_config=args.mixture_config,
        require_caches=False,
    )
    _validate_member_metadata(resolved, args.tactile_keys)
    reports = []
    for member in resolved.members:
        destination = (
            Path(member.root) / IN_PLACE_CACHE_DIR
            if args.cache_root is None
            else args.cache_root / member.dataset_id
        )
        reports.append(
            build_member_cache(
                member.dataset_id,
                Path(member.root),
                destination,
                config,
                requested_tactile_keys=args.tactile_keys,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                split_seed=args.split_seed,
                tolerance_s=args.tolerance_s,
                num_workers=args.num_workers,
                overwrite=args.overwrite,
            )
        )
    print(json.dumps({"dataset_id": args.dataset_id, "members": reports}, indent=2))


if __name__ == "__main__":
    main()
