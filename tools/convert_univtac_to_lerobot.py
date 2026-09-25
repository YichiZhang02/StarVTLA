#!/usr/bin/env python3
"""Convert UniVTAC Isaac 4.5 HDF5 demonstrations to local LeRobot v3 datasets.

The source records observations every two 120 Hz simulator steps and has no
commanded-action column. As in UniVTAC's own HDF5 loader, action[t] is the
measured joint configuration at the next recorded frame. The last frame of
each source episode has no target and is omitted.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vtla.datasets.io_utils import write_info  # noqa: E402
from vtla.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from vtla.datasets.visual_preprocess import make_visual_preprocess  # noqa: E402

LOGGER = logging.getLogger("convert_univtac")
SOURCE_PATHS = {
    "observation.images.cam_top": "observation/head/rgb",
    "observation.images.cam_wrist": "observation/wrist/rgb",
    "observation.images.left_cam_finger0": "tactile/left_gsmini/rgb_marker",
    "observation.images.right_cam_finger0": "tactile/right_gsmini/rgb_marker",
}
JOINT_NAMES = [*(f"panda_joint{i}.pos" for i in range(1, 8)), "gripper.pos"]
SOURCE_FPS = 120
DEFAULT_SAVE_EVERY = 2


def source_files(task_root: Path, limit: int | None) -> list[Path]:
    paths = sorted(task_root.glob("hdf5/*.hdf5"), key=lambda p: int(p.stem))
    if not paths:
        raise FileNotFoundError(f"No HDF5 episodes in {task_root / 'hdf5'}")
    if limit is not None:
        paths = paths[:limit]
    return paths


def read_instruction(repo: Path, task: str) -> str:
    path = repo / "instructions" / f"{task}.json"
    instructions = json.loads(path.read_text(encoding="utf-8"))["seen"]
    if len(instructions) != 1 or not isinstance(instructions[0], str):
        raise ValueError(f"Expected one seen instruction in {path}")
    return instructions[0]


def decode_rgb(value: bytes, size: int) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV failed to decode UniVTAC image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (size, size):
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return rgb


def canonical_qpos(raw: np.ndarray) -> np.ndarray:
    """Use UniVTAC's baseline convention: first seven arm joints, two fingers."""
    if raw.shape[-1] != 9:
        raise ValueError(f"Expected 9 UniVTAC joints, got shape {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite joint position in source HDF5")
    fingers = raw[..., 7:9]
    if np.max(np.abs(fingers[..., 0] - fingers[..., 1])) > 0.002:
        raise ValueError("Finger positions differ by more than 2 mm; cannot reduce to qpos8")
    result = np.concatenate((raw[..., :7], fingers.mean(axis=-1, keepdims=True)), axis=-1)
    return result.astype(np.float32, copy=False)


def validate_source(f: h5py.File, path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps = np.asarray(f["step"][:], dtype=np.int64)
    joints = np.asarray(f["embodiment/joint"][:], dtype=np.float32)
    if len(steps) < 2 or joints.shape != (len(steps), 9):
        raise ValueError(f"Invalid step/joint shape in {path}: {steps.shape}, {joints.shape}")
    for source_key in SOURCE_PATHS.values():
        if source_key not in f or len(f[source_key]) != len(steps):
            raise ValueError(f"Missing or misaligned {source_key} in {path}")
    return steps, canonical_qpos(joints)


def contiguous_segments(steps: np.ndarray, save_every: int) -> list[tuple[int, int]]:
    boundaries = [0, *(np.flatnonzero(np.diff(steps) != save_every) + 1).tolist(), len(steps)]
    segments = [(start, end) for start, end in zip(boundaries, boundaries[1:]) if end - start >= 2]
    if not segments:
        raise ValueError("Source episode has no contiguous observation pairs")
    return segments


def features(size: int) -> dict[str, dict]:
    image = {"dtype": "video", "shape": (size, size, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES},
        "observation.sim_step": {"dtype": "int64", "shape": (1,), "names": ["sim_step"]},
        **{key: dict(image) for key in SOURCE_PATHS},
    }


def write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def convert_task(args: argparse.Namespace, task: str) -> None:
    task_root = args.source / task
    paths = source_files(task_root, args.max_episodes)
    instruction = read_instruction(args.univtac_repo, task)
    output = args.output_base / f"univtac_isaac45_{task}_lerobot"
    repo_id = f"local/{output.name}"
    fps = SOURCE_FPS // args.save_every
    if SOURCE_FPS % args.save_every:
        raise ValueError("save_every must divide 120 Hz")
    manifest_path = output / "univtac_conversion.json"
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"{output} exists; pass --resume to append missing episodes")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_root"] != str(args.source.resolve()) or manifest["image_size"] != args.image_size:
            raise ValueError(f"Conversion settings differ from {manifest_path}")
        completed = set(manifest["source_episodes"])
        dataset = LeRobotDataset.resume(
            repo_id, root=output, vcodec=args.vcodec, streaming_encoding=True,
        )
        if dataset.meta.total_episodes != len(completed):
            raise ValueError(
                f"Episode count in {output} differs from conversion manifest; "
                "inspect the interrupted episode before resuming"
            )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output,
            fps=fps,
            features=features(args.image_size),
            robot_type="franka_panda",
            use_videos=True,
            vcodec=args.vcodec,
            metadata_buffer_size=1,
            streaming_encoding=True,
        )
        dataset.meta.info.visual_preprocess = make_visual_preprocess(
            size=args.image_size, wrist_undistort=False, tactile_encoding=None
        )
        write_info(dataset.meta.info, dataset.meta.root)
        completed = set()
        manifest = {
            "source_root": str(args.source.resolve()),
            "task": task,
            "source_episodes": [],
            "source_format": "UniVTAC isaac45 HDF5",
            "target_format": "LeRobot v3",
            "robot_type": "franka_panda",
            "fps": fps,
            "image_size": args.image_size,
            "instruction": instruction,
            "action_semantics": "next recorded measured qpos, not original commanded qpos",
            "state_semantics": "first seven HDF5 joint values plus mean of two finger values",
            "omitted_source_fields": ["actor", "atom", "embodiment/ee", "tactile/*/depth", "tactile/*/marker", "tactile/*/pose", "tactile/*/rgb"],
        }
        write_manifest(manifest_path, manifest)
    if dataset.meta.fps != fps or dataset.meta.info.robot_type != "franka_panda":
        raise ValueError("Existing dataset metadata is incompatible with source")

    try:
        for index, path in enumerate(paths, 1):
            if path.stem in completed:
                continue
            LOGGER.info("[%s %d/%d] %s", task, index, len(paths), path)
            with h5py.File(path, "r") as f:
                steps, qpos = validate_source(f, path)
                segments = contiguous_segments(steps, args.save_every)
                if len(segments) > 1:
                    LOGGER.warning("Split %s into %d episodes at discontinuous sim steps", path, len(segments))
                encoded_images = {key: f[source_key][:-1] for key, source_key in SOURCE_PATHS.items()}
                for segment_index, (start, end) in enumerate(segments):
                    source_id = path.stem if len(segments) == 1 else f"{path.stem}#part{segment_index}"
                    if source_id in completed:
                        continue
                    for t in range(start, end - 1):
                        frame = {
                            "observation.state": qpos[t].copy(),
                            "action": qpos[t + 1].copy(),
                            "observation.sim_step": np.array([steps[t]], dtype=np.int64),
                            "task": instruction,
                        }
                        for feature_key, image_sequence in encoded_images.items():
                            frame[feature_key] = decode_rgb(image_sequence[t], args.image_size)
                        dataset.add_frame(frame)
                    dataset.save_episode(parallel_encoding=False)
                    manifest["source_episodes"].append(source_id)
                    write_manifest(manifest_path, manifest)
    finally:
        dataset.finalize()
    LOGGER.info("Converted %s: %d episodes, %d frames -> %s", task, dataset.meta.total_episodes, dataset.meta.total_frames, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "playground/data/UniVTAC/isaac45")
    parser.add_argument("--univtac-repo", type=Path, default=ROOT / "playground/simulations/UniVTAC")
    parser.add_argument("--output-base", type=Path, default=ROOT / "playground/data")
    parser.add_argument("--tasks", nargs="+", help="Task names; default: every task under source")
    parser.add_argument("--max-episodes", type=int, help="Limit each task, useful for smoke tests")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.image_size <= 0 or args.save_every <= 0 or (args.max_episodes is not None and args.max_episodes <= 0):
        parser.error("image-size, save-every and max-episodes must be positive")
    args.source = args.source.resolve()
    args.univtac_repo = args.univtac_repo.resolve()
    args.output_base = args.output_base.resolve()
    if args.tasks is None:
        args.tasks = sorted(p.name for p in args.source.iterdir() if (p / "hdf5").is_dir())
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    for task in args.tasks:
        convert_task(args, task)


if __name__ == "__main__":
    main()
