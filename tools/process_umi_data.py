#!/usr/bin/env python3
"""Import a unified-format UMI v2.5 dataset into the VTLA EE training contract.

The source is never modified. For every episode, the importer aligns data and selected RGB streams
to their common shortest length, canonicalizes camera keys and metadata, writes the task text, then
uses ``convert_umi_to_eepose.py`` to materialize the rot6d/quaternion EE columns and statistics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.convert_umi_to_eepose import (  # noqa: E402
    NEW_FEATURES,
    feature_stats,
    pose_indices,
    sorted_data_files,
)


CAMERA_KEY_MAP = {
    "observation.images.cam_left_undist": "observation.images.left_cam_wrist",
    "observation.images.cam_right_undist": "observation.images.right_cam_wrist",
    "observation.images.ego_right_undist": "observation.images.cam_top",
}
VISUAL_DTYPES = frozenset({"video", "image", "tactile"})
NO_TACTILE_VISUAL_KEYS = frozenset(CAMERA_KEY_MAP.values())
MISSING_VALUE = 9930.0
DATASET_INFO_FIELDS = {
    "codebase_version",
    "fps",
    "features",
    "total_episodes",
    "total_frames",
    "total_tasks",
    "chunks_size",
    "data_files_size_in_mb",
    "video_files_size_in_mb",
    "data_path",
    "video_path",
    "robot_type",
    "splits",
    "ee_num_arms",
    "ee_arm_sides",
    "undistort",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--task", required=True, help="Task instruction stored in tasks.parquet")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--action-gap", type=int, default=6)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--tactile-mode",
        choices=("none",),
        default="none",
        help="Tactile import mode. Only 'none' is currently supported; tactile media and metadata are omitted.",
    )
    for side in ("left", "right"):
        parser.add_argument(
            f"--{side}-gripper-open",
            type=float,
            help=f"Raw {side} gripper value for fully open; omit with --{side}-gripper-closed to use the data minimum.",
        )
        parser.add_argument(
            f"--{side}-gripper-closed",
            type=float,
            help=f"Raw {side} gripper value for fully closed; omit with --{side}-gripper-open to use 0.",
        )
    return parser.parse_args()


def _load_info(root: Path) -> dict:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing UMI metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_camera_key(value: str) -> str:
    for source, target in CAMERA_KEY_MAP.items():
        value = value.replace(source, target)
    return value


def _copy_non_video(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True)
    for entry in sorted(src.iterdir()):
        if entry.name in {"videos", "frames_cache"}:
            continue
        target = dst / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)


def _video_hw(feature: dict) -> tuple[int, int]:
    shape = feature.get("shape") or []
    names = [str(name).lower() for name in feature.get("names") or []]
    if len(shape) != 3 or set(names) != {"height", "width", "channels"}:
        raise ValueError(f"Cannot resolve video axes from shape={shape}, names={names}")
    return int(shape[names.index("height")]), int(shape[names.index("width")])


def _resize_intrinsics(feature: dict, source_h: int, source_w: int, size: int) -> None:
    intrinsics = feature.get("intrinsics") or {}
    if not intrinsics:
        return
    source = next(iter(intrinsics.values()))
    sx, sy = size / source_w, size / source_h
    feature["intrinsics"] = {
        f"{size}x{size}": {
            "fx": float(source["fx"]) * sx,
            "fy": float(source["fy"]) * sy,
            "ppx": float(source["ppx"]) * sx,
            "ppy": float(source["ppy"]) * sy,
        }
    }


def rewrite_info(
    root: Path,
    source_info: dict,
    size: int,
    total_frames: int | None = None,
    tactile_mode: str = "none",
) -> dict:
    if tactile_mode != "none":
        raise ValueError(f"Unsupported UMI tactile mode: {tactile_mode}")
    info = json.loads(json.dumps(source_info))
    features = info.get("features", {})
    missing = sorted(set(CAMERA_KEY_MAP) - set(features))
    if missing:
        raise ValueError(f"UMI dataset is missing required camera features: {missing}")

    rewritten = {}
    for key, feature in features.items():
        target = CAMERA_KEY_MAP.get(key, key)
        # A no-tactile conversion emits only the three canonical RGB streams. Source-only RGB
        # and tactile declarations must not survive in metadata because their media is not copied.
        if feature.get("dtype") in VISUAL_DTYPES and target not in NO_TACTILE_VISUAL_KEYS:
            continue
        if target in rewritten:
            raise ValueError(f"Camera rename collision at {target}")
        if key in CAMERA_KEY_MAP:
            source_h, source_w = _video_hw(feature)
            _resize_intrinsics(feature, source_h, source_w, size)
            feature["shape"] = [size, size, 3]
            feature["names"] = ["height", "width", "channels"]
            video_info = feature.setdefault("info", {})
            video_info.update(
                {
                    "video.height": size,
                    "video.width": size,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.fps": float(info["fps"]),
                    "video.channels": 3,
                }
            )
        rewritten[target] = feature
    info["features"] = rewritten
    info["robot_type"] = "umi"
    info["ee_num_arms"] = 2
    info["ee_arm_sides"] = ["right", "left"]
    info["total_tasks"] = 1
    if total_frames is not None:
        info["total_frames"] = total_frames
    # Existing inference uses this marker to enable online wrist undistortion.
    info["undistort"] = {"source_preprocessed": True}
    info = {key: value for key, value in info.items() if key in DATASET_INFO_FIELDS}
    (root / "meta" / "info.json").write_text(
        json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    return info


def rewrite_global_stats(
    root: Path, source_info: dict | None = None, tactile_mode: str = "none"
) -> None:
    if tactile_mode != "none":
        raise ValueError(f"Unsupported UMI tactile mode: {tactile_mode}")
    path = root / "meta" / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    rewritten = {_replace_camera_key(key): value for key, value in stats.items()}
    if source_info is not None:
        omitted_visual = {
            _replace_camera_key(key)
            for key, feature in source_info.get("features", {}).items()
            if feature.get("dtype") in VISUAL_DTYPES
            and _replace_camera_key(key) not in NO_TACTILE_VISUAL_KEYS
        }
        rewritten = {key: value for key, value in rewritten.items() if key not in omitted_visual}
    tables = [pq.read_table(data_path) for data_path in sorted_data_files(root)]
    if tables:
        data = pa.concat_tables(tables)
        for name in data.column_names:
            if name not in rewritten:
                continue
            values = data.column(name).to_pylist()
            if not values:
                continue
            array = np.stack(values) if isinstance(values[0], (list, tuple)) else np.asarray(values)[:, None]
            if not np.issubdtype(array.dtype, np.number):
                continue
            rewritten[name] = {
                stat: (
                    np.asarray(value, dtype=np.int64).tolist()
                    if stat == "count"
                    else np.asarray(value).tolist()
                )
                for stat, value in feature_stats(array).items()
            }
    path.write_text(json.dumps(rewritten, indent=4, ensure_ascii=False), encoding="utf-8")


def rewrite_data_task_indices(
    root: Path, target_lengths: dict[int, int] | None = None
) -> dict[int, int]:
    lengths: dict[int, int] = {}
    next_index = 0
    for path in sorted_data_files(root):
        table = pq.read_table(path)
        if target_lengths is not None:
            episodes = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
            frame_indices = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
            keep = np.asarray(
                [frame < target_lengths[int(episode)] for episode, frame in zip(episodes, frame_indices, strict=True)],
                dtype=np.bool_,
            )
            table = table.filter(pa.array(keep))
        episodes = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        unique, counts = np.unique(episodes, return_counts=True)
        for episode, count in zip(unique, counts, strict=True):
            lengths[int(episode)] = lengths.get(int(episode), 0) + int(count)
        if "index" in table.column_names:
            indices = pa.array(np.arange(next_index, next_index + len(table)), type=table.schema.field("index").type)
            table = table.set_column(table.column_names.index("index"), "index", indices)
            next_index += len(table)
        zeros = pa.array(np.zeros(len(table), dtype=np.int64), type=pa.int64())
        index = table.column_names.index("task_index")
        table = table.set_column(index, "task_index", zeros)
        pq.write_table(table, path)
    expected = list(range(len(lengths)))
    if sorted(lengths) != expected:
        raise ValueError(f"episode_index must be contiguous 0..N, got {sorted(lengths)[:10]}")
    if target_lengths is not None and lengths != target_lengths:
        raise ValueError(f"Trimmed data lengths differ from common target: {lengths} != {target_lengths}")
    return lengths


def rewrite_episode_metadata(
    root: Path,
    task: str,
    fps: int,
    lengths: dict[int, int],
    tactile_mode: str = "none",
) -> None:
    if tactile_mode != "none":
        raise ValueError(f"Unsupported UMI tactile mode: {tactile_mode}")
    episode_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise FileNotFoundError("UMI dataset is missing meta/episodes parquet files")
    starts: dict[int, int] = {}
    cursor = 0
    for episode in sorted(lengths):
        starts[episode] = cursor
        cursor += lengths[episode]
    for path in episode_files:
        table = pq.read_table(path)
        table = table.rename_columns([_replace_camera_key(name) for name in table.column_names])
        stale_columns = [name for name in table.column_names if name.startswith("stats/")]
        for name in table.column_names:
            if not name.startswith("videos/"):
                continue
            video_key = name.split("/", 2)[1]
            if video_key not in NO_TACTILE_VISUAL_KEYS:
                stale_columns.append(name)
        if stale_columns:
            table = table.drop(stale_columns)
        episodes = [int(value) for value in table.column("episode_index").to_pylist()]
        tasks = pa.array([[task] for _ in episodes], type=pa.list_(pa.string()))
        table = table.set_column(table.column_names.index("tasks"), "tasks", tasks)
        for name, values in {
            "length": [lengths[episode] for episode in episodes],
            "dataset_from_index": [starts[episode] for episode in episodes],
            "dataset_to_index": [starts[episode] + lengths[episode] for episode in episodes],
        }.items():
            if name not in table.column_names:
                raise ValueError(f"Episode metadata is missing {name}")
            column = pa.array(values, type=table.schema.field(name).type)
            table = table.set_column(table.column_names.index(name), name, column)
        for camera in CAMERA_KEY_MAP.values():
            from_name = f"videos/{camera}/from_timestamp"
            to_name = f"videos/{camera}/to_timestamp"
            if from_name not in table.column_names or to_name not in table.column_names:
                raise ValueError(f"Episode metadata is missing video timestamps for {camera}")
            starts = np.asarray(table.column(from_name).to_pylist(), dtype=np.float64)
            stops = pa.array(
                [start + lengths[episode] / fps for start, episode in zip(starts, episodes, strict=True)],
                type=pa.float64(),
            )
            table = table.set_column(table.column_names.index(to_name), to_name, stops)
        pq.write_table(table, path)


def rewrite_tasks(root: Path, task: str) -> None:
    frame = pd.DataFrame({"task_index": [0]}, index=pd.Index([task]))
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=True), root / "meta" / "tasks.parquet")


def _episode_video_lengths(root: Path, old_key: str, lengths: dict[int, int]) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        table = pq.read_table(path)
        episodes = [int(value) for value in table.column("episode_index").to_pylist()]
        chunks = table.column(f"videos/{old_key}/chunk_index").to_pylist()
        files = table.column(f"videos/{old_key}/file_index").to_pylist()
        for episode, chunk, file_index in zip(episodes, chunks, files, strict=True):
            location = (int(chunk), int(file_index))
            expected = lengths[episode]
            if location in result and result[location] != expected:
                raise ValueError(f"Multiple episodes share video file {old_key} {location}")
            result[location] = expected
    return result


def _probe_frames(path: Path) -> int:
    output = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=nb_frames", "-of", "csv=p=0", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return int(output)


def resolve_common_episode_lengths(root: Path, source_info: dict) -> dict[int, int]:
    """Return the shortest aligned length across data and the selected RGB streams."""
    data_lengths: dict[int, int] = {}
    for path in sorted_data_files(root):
        episodes = np.asarray(
            pq.read_table(path, columns=["episode_index"]).column("episode_index").to_pylist(),
            dtype=np.int64,
        )
        unique, counts = np.unique(episodes, return_counts=True)
        for episode, count in zip(unique, counts, strict=True):
            data_lengths[int(episode)] = data_lengths.get(int(episode), 0) + int(count)

    episode_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise FileNotFoundError("UMI dataset is missing meta/episodes parquet files")
    target_lengths = dict(data_lengths)
    shortest_source = {episode: "data" for episode in target_lengths}
    frame_cache: dict[Path, int] = {}
    seen_episodes: set[int] = set()
    fps = int(source_info["fps"])
    for path in episode_files:
        table = pq.read_table(path)
        episodes = [int(value) for value in table.column("episode_index").to_pylist()]
        seen_episodes.update(episodes)
        for camera in CAMERA_KEY_MAP:
            chunks = table.column(f"videos/{camera}/chunk_index").to_pylist()
            files = table.column(f"videos/{camera}/file_index").to_pylist()
            from_timestamps = table.column(f"videos/{camera}/from_timestamp").to_pylist()
            for episode, chunk, file_index, from_timestamp in zip(
                episodes, chunks, files, from_timestamps, strict=True
            ):
                video_path = root / source_info["video_path"].format(
                    video_key=camera,
                    chunk_index=int(chunk),
                    file_index=int(file_index),
                )
                if not video_path.is_file():
                    raise FileNotFoundError(video_path)
                if video_path not in frame_cache:
                    frame_cache[video_path] = _probe_frames(video_path)
                start_frame = round(float(from_timestamp) * fps)
                available = frame_cache[video_path] - start_frame
                if available <= 0:
                    raise ValueError(
                        f"Episode {episode} has no frames available in {video_path} "
                        f"after start frame {start_frame}."
                    )
                if available < target_lengths[episode]:
                    target_lengths[episode] = available
                    shortest_source[episode] = camera

    if seen_episodes != set(data_lengths):
        raise ValueError(
            f"Episode metadata/data mismatch: metadata={sorted(seen_episodes)}, "
            f"data={sorted(data_lengths)}"
        )
    trimmed = 0
    for episode in sorted(target_lengths):
        removed = data_lengths[episode] - target_lengths[episode]
        if removed:
            trimmed += removed
            print(
                f"[align] episode {episode}: {data_lengths[episode]} -> "
                f"{target_lengths[episode]} frames (shortest={shortest_source[episode]})"
            )
    print(
        f"[align] common-shortest policy: {len(target_lengths)} episodes, "
        f"trimmed {trimmed} data frame(s)"
    )
    return target_lengths


def _encode_video(job: tuple[Path, Path, int, int, int, int]) -> tuple[Path, int]:
    source, target, frames, size, fps, crf = job
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", "0:v:0",
            "-frames:v", str(frames), "-vf", f"scale={size}:{size}:flags=lanczos",
            "-c:v", "libx264", "-crf", str(crf), "-g", "12", "-pix_fmt", "yuv420p",
            "-an", "-vsync", "0", str(target),
        ],
        check=True,
    )
    actual = _probe_frames(target)
    if actual != frames:
        raise RuntimeError(f"{target}: encoded {actual} frames, expected {frames}")
    return target, actual


def process_videos(
    src: Path, dst: Path, source_info: dict, lengths: dict[int, int], size: int, jobs: int, crf: int
) -> None:
    fps = int(source_info["fps"])
    work = []
    for old_key, new_key in CAMERA_KEY_MAP.items():
        locations = _episode_video_lengths(src, old_key, lengths)
        for (chunk, file_index), expected in locations.items():
            source = src / source_info["video_path"].format(
                video_key=old_key, chunk_index=chunk, file_index=file_index
            )
            target = dst / source_info["video_path"].format(
                video_key=new_key, chunk_index=chunk, file_index=file_index
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            work.append((source, target, expected, size, fps, crf))

    print(f"[videos] encoding {len(work)} streams with {jobs} workers")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_encode_video, job) for job in work]
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(work):
                print(f"         {completed}/{len(work)}")


def resolve_gripper_calibration(
    root: Path, info: dict, explicit: dict[str, tuple[float | None, float | None]]
) -> dict[str, tuple[float, float]]:
    state_idx = pose_indices(info["features"]["observation.state"]["names"])
    values = {"left": [], "right": []}
    for path in sorted_data_files(root):
        table = pq.read_table(path, columns=["observation.state"])
        vectors = np.stack(table.column("observation.state").to_pylist()).astype(np.float64)
        for side in values:
            values[side].append(vectors[:, state_idx[side]["grip"]])

    calibration = {}
    for side, chunks in values.items():
        raw = np.concatenate(chunks)
        if not np.isfinite(raw).all() or np.any(np.isclose(raw, MISSING_VALUE)):
            raise ValueError(f"{side} gripper contains missing or non-finite values")
        open_value, closed_value = explicit[side]
        if (open_value is None) != (closed_value is None):
            raise ValueError(f"{side} gripper requires both --{side}-gripper-open and --{side}-gripper-closed")
        if open_value is None:
            # UMI v2.5 encodes a more open gripper as a lower (more negative) angle.
            # The closed command is defined as zero by the UMI control contract; it does not
            # need to appear in every demonstration, so do not infer it from the observed max.
            open_value, closed_value = float(raw.min()), 0.0
            print(
                f"[gripper] {side}: auto calibration open={open_value:.9f}, "
                f"closed={closed_value:.9f}"
            )
        calibration[side] = (float(open_value), float(closed_value))
    return calibration


def run_converter(root: Path, args: argparse.Namespace, calibration: dict[str, tuple[float, float]]) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("convert_umi_to_eepose.py")),
        "--root", str(root),
        "--horizon", str(args.horizon),
        "--action-gap", str(args.action_gap),
    ]
    for side in ("left", "right"):
        open_value, closed_value = calibration[side]
        command += [
            f"--{side}-gripper-open", str(open_value),
            f"--{side}-gripper-closed", str(closed_value),
        ]
    subprocess.run(command, check=True)


def validate_output(root: Path, lengths: dict[int, int]) -> None:
    info = _load_info(root)
    if info.get("robot_type") != "umi":
        raise RuntimeError("Processed dataset robot_type is not 'umi'")
    missing = sorted(set(NEW_FEATURES) - set(info.get("features", {})))
    if missing:
        raise RuntimeError(f"Processed dataset is missing EE features: {missing}")
    visual_keys = {
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") in VISUAL_DTYPES
    }
    if visual_keys != NO_TACTILE_VISUAL_KEYS:
        raise RuntimeError(
            "No-tactile output must contain exactly the emitted RGB streams; "
            f"got {sorted(visual_keys)}"
        )
    for path in sorted_data_files(root):
        table = pq.read_table(path, columns=["action_absolute_ee"])
        actions = np.stack(table.column("action_absolute_ee").to_pylist())
        grips = actions[:, [9, 19]]
        if grips.min() < -1e-6 or grips.max() > 1 + 1e-6:
            raise RuntimeError(f"Normalized gripper values outside [0, 1]: {grips.min()}..{grips.max()}")
    for key in CAMERA_KEY_MAP.values():
        locations = _episode_video_lengths(root, key, lengths)
        for (chunk, file_index), expected in locations.items():
            path = root / info["video_path"].format(
                video_key=key,
                chunk_index=chunk,
                file_index=file_index,
            )
            if _probe_frames(path) != expected:
                raise RuntimeError(f"Video/data frame mismatch: {path}")


def main() -> int:
    args = parse_args()
    args.task = args.task.strip()
    if not args.task or args.task.lower() == "unknown task":
        raise SystemExit("--task must be a real task instruction, not 'Unknown task'")
    if args.size <= 0 or args.horizon <= 0 or args.action_gap < 0 or args.jobs <= 0:
        raise SystemExit("size/horizon/jobs must be positive and action-gap must be non-negative")
    if not args.src.is_dir():
        raise SystemExit(f"Source dataset does not exist: {args.src}")
    partial = args.dst.with_name(f".{args.dst.name}.partial")
    if args.dst.exists() or partial.exists():
        raise SystemExit(f"Destination or partial output already exists: {args.dst}, {partial}")

    source_info = _load_info(args.src)
    target_lengths = resolve_common_episode_lengths(args.src, source_info)
    print(f"[copy] {args.src} -> {partial}")
    _copy_non_video(args.src, partial)
    try:
        lengths = rewrite_data_task_indices(partial, target_lengths=target_lengths)
        calibration = resolve_gripper_calibration(
            partial,
            source_info,
            {
                "left": (args.left_gripper_open, args.left_gripper_closed),
                "right": (args.right_gripper_open, args.right_gripper_closed),
            },
        )
        rewrite_tasks(partial, args.task)
        rewrite_episode_metadata(
            partial,
            args.task,
            int(source_info["fps"]),
            lengths,
            tactile_mode=args.tactile_mode,
        )
        rewrite_global_stats(
            partial, source_info=source_info, tactile_mode=args.tactile_mode
        )
        rewrite_info(
            partial,
            source_info,
            args.size,
            total_frames=sum(lengths.values()),
            tactile_mode=args.tactile_mode,
        )
        process_videos(args.src, partial, source_info, lengths, args.size, args.jobs, args.crf)
        run_converter(partial, args, calibration)
        manifest = {
            "source": str(args.src.resolve()),
            "robot_type": "umi",
            "camera_key_map": CAMERA_KEY_MAP,
            "image_size": [args.size, args.size],
            "task": args.task,
            "horizon": args.horizon,
            "action_gap": args.action_gap,
            "tactile_mode": args.tactile_mode,
            "alignment": {
                "policy": "common_shortest_per_episode",
                "total_frames": sum(lengths.values()),
                "episode_lengths": lengths,
            },
            "gripper_calibration": {
                side: {
                    "open": values[0],
                    "closed": values[1],
                    "normalized_open": 1,
                    "normalized_closed": 0,
                }
                for side, values in calibration.items()
            },
        }
        (partial / "meta" / "umi_processing.json").write_text(
            json.dumps(manifest, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        validate_output(partial, lengths)
        partial.rename(args.dst)
    except Exception:
        print(f"Processing failed; partial output retained for inspection: {partial}", file=sys.stderr)
        raise

    print(f"Done: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
