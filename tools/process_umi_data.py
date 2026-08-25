#!/usr/bin/env python3
"""Import a unified-format UMI v2.5 dataset into the VTLA EE training contract.

The source is never modified. The importer copies non-video data, canonicalizes camera keys and
metadata, trims each video to its episode length while resizing it, writes the task text, then uses
``convert_umi_to_eepose.py`` to materialize the rot6d/quaternion EE columns and statistics.
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

from tools.convert_umi_to_eepose import NEW_FEATURES, pose_indices, sorted_data_files  # noqa: E402


CAMERA_KEY_MAP = {
    "observation.images.cam_left_undist": "observation.images.left_cam_wrist",
    "observation.images.cam_right_undist": "observation.images.right_cam_wrist",
    "observation.images.ego_right_undist": "observation.images.cam_top",
}
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
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-gripper-open", type=float)
        parser.add_argument(f"--{side}-gripper-closed", type=float)
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


def rewrite_info(root: Path, source_info: dict, size: int) -> dict:
    info = json.loads(json.dumps(source_info))
    features = info.get("features", {})
    missing = sorted(set(CAMERA_KEY_MAP) - set(features))
    if missing:
        raise ValueError(f"UMI dataset is missing required camera features: {missing}")

    rewritten = {}
    for key, feature in features.items():
        target = CAMERA_KEY_MAP.get(key, key)
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
    # Existing inference uses this marker to enable online wrist undistortion.
    info["undistort"] = {"source_preprocessed": True}
    info = {key: value for key, value in info.items() if key in DATASET_INFO_FIELDS}
    (root / "meta" / "info.json").write_text(
        json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    return info


def rewrite_global_stats(root: Path) -> None:
    path = root / "meta" / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    rewritten = {_replace_camera_key(key): value for key, value in stats.items()}
    path.write_text(json.dumps(rewritten, indent=4, ensure_ascii=False), encoding="utf-8")


def rewrite_data_task_indices(root: Path) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for path in sorted_data_files(root):
        table = pq.read_table(path)
        episodes = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        unique, counts = np.unique(episodes, return_counts=True)
        for episode, count in zip(unique, counts, strict=True):
            lengths[int(episode)] = lengths.get(int(episode), 0) + int(count)
        zeros = pa.array(np.zeros(len(table), dtype=np.int64), type=pa.int64())
        index = table.column_names.index("task_index")
        table = table.set_column(index, "task_index", zeros)
        pq.write_table(table, path)
    expected = list(range(len(lengths)))
    if sorted(lengths) != expected:
        raise ValueError(f"episode_index must be contiguous 0..N, got {sorted(lengths)[:10]}")
    return lengths


def rewrite_episode_metadata(root: Path, task: str, fps: int, lengths: dict[int, int]) -> None:
    episode_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise FileNotFoundError("UMI dataset is missing meta/episodes parquet files")
    for path in episode_files:
        table = pq.read_table(path)
        table = table.rename_columns([_replace_camera_key(name) for name in table.column_names])
        episodes = [int(value) for value in table.column("episode_index").to_pylist()]
        tasks = pa.array([[task] for _ in episodes], type=pa.list_(pa.string()))
        table = table.set_column(table.column_names.index("tasks"), "tasks", tasks)
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
            open_value, closed_value = float(raw.min()), float(raw.max())
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
    calibration = resolve_gripper_calibration(
        args.src,
        source_info,
        {
            "left": (args.left_gripper_open, args.left_gripper_closed),
            "right": (args.right_gripper_open, args.right_gripper_closed),
        },
    )
    print(f"[copy] {args.src} -> {partial}")
    _copy_non_video(args.src, partial)
    try:
        lengths = rewrite_data_task_indices(partial)
        rewrite_tasks(partial, args.task)
        rewrite_episode_metadata(partial, args.task, int(source_info["fps"]), lengths)
        rewrite_global_stats(partial)
        rewrite_info(partial, source_info, args.size)
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
