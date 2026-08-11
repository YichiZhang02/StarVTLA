#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline: add EE-pose columns to a UMI-style LeRobot v3.0 dataset (in place).

Unlike ``convert_joints_to_eepose.py`` (which runs Realman forward kinematics on joint angles),
a UMI dataset already stores the end-effector POSE directly, so this script skips FK entirely and
just converts the stored quaternion to the rot6d layout the VLA infra expects, and also stores the
pose with quaternion rotation directly.

Input layout (per ``meta/info.json`` of this dataset, LEFT arm first then RIGHT):
    observation.state : 28-dim, per arm [gripper_Position_Rad, x, y, z, Quat_X, Quat_Y, Quat_Z,
                        Quat_W, Acc_X, Acc_Y, Acc_Z, Gyro_X, Gyro_Y, Gyro_Z]  (IMU is DROPPED)
    action            : 16-dim, per arm [gripper_Position_Rad, x, y, z, Quat_X, Quat_Y, Quat_Z, Quat_W]

ADDS four columns to the SAME dataset (the original joint-less columns are left untouched):

    observation.state_episode_ee : 20-dim, EE pose of the STATE relative to each episode's FIRST
                                   frame (T0^{-1}·Tt), expressed in the first-frame frame. rot6d.
    action_episode_ee            : 20-dim, EE pose of the ACTION (the real teleop command) relative
                                   to the SAME T0 as the state (so T0 cancels at train time). rot6d.
    observation.state_episode_quat : 16-dim, same as state_episode_ee but rotation as quat [x,y,z,w]
                                     (per arm: xyz(3) + quat(4) + gripper(1) = 8).
    action_episode_quat            : 16-dim, same as action_episode_ee but quat rotation.

Output uses per arm ``[xyz(3), rot6d(6), gripper(1)]`` for rot6d (20 = 2 * 10) and
``[xyz(3), quat_xyzw(4), gripper(1)]`` for quat (16 = 2 * 8), ordered RIGHT arm first then LEFT.
Gripper is kept absolute. This matches ``convert_joints_to_eepose.py`` byte-for-byte, so the
existing ``state_mode='episode_ee'`` / ``action_mode='relative_ee'`` path consumes it unchanged.

At train time the action chunk (future action_episode_ee values, delta_indices 1..chunk) is
relativized against the current state_episode_ee anchor S_t, giving ``S_t^{-1} · A_{t+k}`` (T0
cancels): the commanded future pose relative to the current observed pose. See
``vtla/engine/utils/ee_transforms.py``.

Updates ``meta/info.json`` (features), ``meta/stats.json`` (global), and ``meta/episodes/*.parquet``
(per-episode stats) so the dataset loads with the new features.

Usage:
    python tools/convert_umi_to_eepose.py --root playground/data/<dataset>
    python tools/convert_umi_to_eepose.py --src <src> --dst <dst>   # copy first
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.spatial.transform import Rotation as R

# Allow running as a standalone script: put the repo root on sys.path so ``vtla`` is importable
# regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vtla.engine.utils.ee_transforms import ee_to_relative  # noqa: E402

PER_ARM_DIM = 10
PER_ARM_DIM_QUAT = 8
EE_DIM = 20       # rot6d: 2 arms * 10
EE_DIM_QUAT = 16  # quat:  2 arms * 8
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
NEW_FEATURES = (
    "observation.state_episode_ee",
    "action_episode_ee",
    "observation.state_episode_quat",
    "action_episode_quat",
)


# ----------------------------------------------------------------------------
# Layout helpers
def build_names() -> list[str]:
    """20-dim output feature names, RIGHT arm first then LEFT."""
    names: list[str] = []
    for side in ("right", "left"):
        names += [f"{side}_ee_x", f"{side}_ee_y", f"{side}_ee_z"]
        names += [f"{side}_ee_rot6d_{i}" for i in range(6)]
        names += [f"{side}_gripper"]
    return names


def build_names_quat() -> list[str]:
    """16-dim quat output feature names, RIGHT arm first then LEFT."""
    names: list[str] = []
    for side in ("right", "left"):
        names += [f"{side}_ee_x", f"{side}_ee_y", f"{side}_ee_z"]
        names += [f"{side}_ee_qx", f"{side}_ee_qy", f"{side}_ee_qz", f"{side}_ee_qw"]
        names += [f"{side}_gripper"]
    return names


def pose_indices(names: list[str]) -> dict:
    """Derive per-arm pose/gripper column indices from the input feature names.

    Returns ``{side: {"grip": int, "pos": [3], "quat": [4]}}`` for side in (left, right). IMU
    columns (Acc_*/Gyro_*) are simply ignored. Quaternion order is normalised to (x, y, z, w).
    """
    out: dict[str, dict] = {
        s: {"grip": None, "pos": [None, None, None], "quat": [None, None, None, None]}
        for s in ("left", "right")
    }
    quat_axis = {"x": 0, "y": 1, "z": 2, "w": 3}
    for i, n in enumerate(names):
        low = n.lower()
        side = "left" if low.startswith("left") else "right" if low.startswith("right") else None
        if side is None:
            continue
        rest = low[len(side) + 1:]  # strip "left_"/"right_"
        if "position_rad" in rest or rest == "gripper":
            out[side]["grip"] = i
        elif rest in ("x", "y", "z"):
            out[side]["pos"]["xyz".index(rest)] = i
        elif rest.startswith("quat_"):
            out[side]["quat"][quat_axis[rest.split("_")[1]]] = i
    for side in ("left", "right"):
        d = out[side]
        if d["grip"] is None or None in d["pos"] or None in d["quat"]:
            raise ValueError(f"Could not locate {side} pose/gripper columns in names={names}")
    return out


def split_arm_pose(vec: np.ndarray, idx: dict, side: str):
    """Extract (pos xyz(3,), quat xyzw(4,), gripper float) for one arm from a frame vector."""
    vec = np.asarray(vec, dtype=np.float64)
    d = idx[side]
    pos = vec[d["pos"]]
    quat = vec[d["quat"]]  # (x, y, z, w)
    grip = float(vec[d["grip"]])
    return pos, quat, grip


# ----------------------------------------------------------------------------
# Pose -> rot6d (no FK; the dataset already stores the pose)
def quat_to_mat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Unit-normalise (defensively) then convert (x, y, z, w) quaternion to a rotation matrix."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("zero-norm quaternion")
    return R.from_quat(q / n).as_matrix()  # scipy uses (x, y, z, w)


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    return np.concatenate([mat[:, 0], mat[:, 1]]).astype(np.float64)


def mat_to_quat_np(mat: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion [x, y, z, w]."""
    q = R.from_matrix(mat).as_quat()  # scipy returns (x, y, z, w)
    return q.astype(np.float64)


def relative_arm_ee(pos, mat, grip, p0, R0) -> np.ndarray:
    """Pose relative to first frame (T0^{-1}·Tt): pos=R0^T(pt-p0), rot6d(R0^T·Rt), grip absolute."""
    R0t = R0.T
    p_rel = R0t @ (pos - p0)
    R_rel = R0t @ mat
    return np.concatenate([p_rel, mat_to_rot6d(R_rel), [grip]]).astype(np.float64)


def relative_arm_ee_quat(pos, mat, grip, p0, R0) -> np.ndarray:
    """Like relative_arm_ee but stores rotation as quaternion [x,y,z,w]. Returns 8-dim.

    The relative transform (T0^{-1}·Tt) is computed in rotation-matrix space, then the resulting
    rotation is stored as a quaternion.
    """
    R0t = R0.T
    p_rel = R0t @ (pos - p0)
    R_rel = R0t @ mat
    return np.concatenate([p_rel, mat_to_quat_np(R_rel), [grip]]).astype(np.float64)


def to_episode_ee(vec: np.ndarray, idx: dict, baseline) -> np.ndarray:
    """Frame pose vector -> 20-dim relative-first-frame EE (RIGHT then LEFT), using ``baseline`` T0."""
    (Rp0, RR0), (Lp0, LR0) = baseline
    out = []
    for side, (p0, R0) in (("right", (Rp0, RR0)), ("left", (Lp0, LR0))):
        pos, quat, grip = split_arm_pose(vec, idx, side)
        out.append(relative_arm_ee(pos, quat_to_mat(quat), grip, p0, R0))
    return np.concatenate(out).astype(np.float32)


def to_episode_quat_umi(vec: np.ndarray, idx: dict, baseline) -> np.ndarray:
    """Frame pose vector -> 16-dim relative-first-frame EE with quat rotation (RIGHT then LEFT).

    The dataset stores a quaternion directly, but the relative transform (T0^{-1}·Tt) must still be
    computed in rotation-matrix space, so the stored quat is converted to a matrix, relativized, and
    the result stored back as a quaternion.
    """
    (Rp0, RR0), (Lp0, LR0) = baseline
    out = []
    for side, (p0, R0) in (("right", (Rp0, RR0)), ("left", (Lp0, LR0))):
        pos, quat, grip = split_arm_pose(vec, idx, side)
        out.append(relative_arm_ee_quat(pos, quat_to_mat(quat), grip, p0, R0))
    return np.concatenate(out).astype(np.float32)


# ----------------------------------------------------------------------------
# Dataset I/O (LeRobot v3.0)
def sorted_data_files(root: Path) -> list[Path]:
    files = glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)

    def key(f: str):
        m = re.search(r"chunk-(\d+)/file-(\d+)", f)
        return (int(m.group(1)), int(m.group(2)))

    return [Path(f) for f in sorted(files, key=key)]


# ----------------------------------------------------------------------------
# meta/episodes rebuild (defensive: some UMI dumps ship data+videos but no episodes metadata)
def _video_keys(info: dict) -> list[str]:
    """Feature keys whose dtype is 'video' (the camera/tactile streams)."""
    return [k for k, f in info["features"].items() if f.get("dtype") == "video"]


def _video_file_index(root: Path, video_key: str) -> list[Path]:
    """Sorted list of a video key's on-disk files (chunk/file order)."""
    files = glob.glob(str(root / "videos" / video_key / "**" / "*.*"), recursive=True)
    vids = [f for f in files if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))]

    def key(f: str):
        m = re.search(r"chunk-(\d+)/file-(\d+)", f)
        return (int(m.group(1)), int(m.group(2)))

    return [Path(f) for f in sorted(vids, key=key)]


def _video_nb_frames(path: Path) -> int:
    """Container-metadata frame count for a video (fast; no full decode)."""
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not out or not out.isdigit():
        raise RuntimeError(f"ffprobe could not read nb_frames from {path} (got {out!r})")
    return int(out)


def _episode_lengths(data_files: list[Path]) -> tuple[list[int], list[int]]:
    """Per-episode frame counts and cumulative start frames (episode order = index order)."""
    ep_len: dict[int, int] = {}
    for f in data_files:
        df = pq.read_table(f, columns=["episode_index"]).to_pandas()
        for ep, cnt in df["episode_index"].value_counts().items():
            ep_len[int(ep)] = ep_len.get(int(ep), 0) + int(cnt)
    eps = sorted(ep_len)
    if eps != list(range(len(eps))):
        raise RuntimeError(f"episode_index not contiguous 0..N: got {eps[:5]}...")
    lengths = [ep_len[e] for e in eps]
    starts = np.cumsum([0, *lengths]).tolist()  # len = N+1
    return lengths, starts


def _map_episodes_to_video_files(
    video_files: list[Path], starts: list[int], total_frames: int, video_key: str, fps: int,
) -> list[dict]:
    """For each episode, find which video file holds it and its in-file start timestamp.

    LeRobot v3.0 invariant: an episode never spans two video files, so every file boundary
    lands on an episode start. Returns per-episode
    ``{"file_index", "from_frame_in_file"}`` (chunk assumed 0, matching the flat UMI layout).
    """
    nb = [_video_nb_frames(p) for p in video_files]
    if sum(nb) != total_frames:
        raise RuntimeError(
            f"{video_key}: sum(video frames)={sum(nb)} != data frames={total_frames}; "
            "cannot rebuild episodes safely."
        )
    file_starts = np.cumsum([0, *nb]).tolist()  # global start frame of each video file
    n_ep = len(starts) - 1
    out = []
    for e in range(n_ep):
        gstart = starts[e]
        # locate the file whose [file_start, file_start+nb) contains this episode's start frame
        fi = int(np.searchsorted(file_starts, gstart, side="right") - 1)
        if not (0 <= fi < len(video_files)):
            raise RuntimeError(f"{video_key}: episode {e} start {gstart} outside video files")
        # boundary must align with an episode start (else episode spans two files -> unsupported)
        out.append({"file_index": fi, "from_frame_in_file": gstart - file_starts[fi]})
    return out


def _episode_tasks(root: Path) -> list[list[str]]:
    """Per-episode task-name lists, read from each episode's task_index via tasks.parquet."""
    tasks_df = pq.read_table(root / "meta" / "tasks.parquet").to_pandas()
    # tasks.parquet is indexed by task string with a task_index column; invert it.
    idx_to_task = {int(r["task_index"]): str(name) for name, r in tasks_df.iterrows()}
    data_files = sorted_data_files(root)
    ep_task: dict[int, int] = {}
    for f in data_files:
        df = pq.read_table(f, columns=["episode_index", "task_index"]).to_pandas()
        for ep, ti in df.groupby("episode_index")["task_index"].first().items():
            ep_task.setdefault(int(ep), int(ti))
    return [[idx_to_task.get(ep_task[e], "Unknown task")] for e in sorted(ep_task)]


def rebuild_episodes(root: Path) -> None:
    """Rebuild ``meta/episodes/chunk-000/file-000.parquet`` from data parquet + video frame counts.

    Defensive: only called when the episodes metadata is missing/empty. Reconstructs the exact
    schema the LeRobot v3.0 loader consumes (episode_index, tasks, length, data indices,
    dataset_from/to_index, and per-video chunk/file/from_timestamp/to_timestamp). Stats columns are
    intentionally omitted here — the main conversion pass appends them afterwards, and the loader
    drops all ``stats/*`` columns on read anyway.
    """
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = int(info.get("fps", 30))
    data_files = sorted_data_files(root)
    lengths, starts = _episode_lengths(data_files)
    total_frames = starts[-1]
    n_ep = len(lengths)
    tasks = _episode_tasks(root)

    # data file index per episode (UMI dumps are typically a single flat data file, but handle many)
    data_loc: dict[int, tuple[int, int]] = {}
    for f in data_files:
        m = re.search(r"chunk-(\d+)/file-(\d+)", str(f))
        ci, fi = int(m.group(1)), int(m.group(2))
        for ep in pq.read_table(f, columns=["episode_index"]).to_pandas()["episode_index"].unique():
            data_loc.setdefault(int(ep), (ci, fi))

    vkeys = _video_keys(info)
    vid_maps = {}
    for vk in vkeys:
        vfiles = _video_file_index(root, vk)
        if not vfiles:
            raise RuntimeError(f"video key {vk} has no files on disk; cannot rebuild episodes")
        vid_maps[vk] = _map_episodes_to_video_files(vfiles, starts, total_frames, vk, fps)

    rows: dict[str, list] = {
        "episode_index": [], "tasks": [], "length": [],
        "data/chunk_index": [], "data/file_index": [],
        "dataset_from_index": [], "dataset_to_index": [],
        "meta/episodes/chunk_index": [], "meta/episodes/file_index": [],
    }
    for vk in vkeys:
        for suf in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            rows[f"videos/{vk}/{suf}"] = []

    for e in range(n_ep):
        ci, fi = data_loc[e]
        rows["episode_index"].append(e)
        rows["tasks"].append(tasks[e])
        rows["length"].append(lengths[e])
        rows["data/chunk_index"].append(ci)
        rows["data/file_index"].append(fi)
        rows["dataset_from_index"].append(starts[e])
        rows["dataset_to_index"].append(starts[e + 1])
        rows["meta/episodes/chunk_index"].append(0)
        rows["meta/episodes/file_index"].append(0)
        for vk in vkeys:
            m = vid_maps[vk][e]
            frm = m["from_frame_in_file"] / fps
            rows[f"videos/{vk}/chunk_index"].append(0)
            rows[f"videos/{vk}/file_index"].append(m["file_index"])
            rows[f"videos/{vk}/from_timestamp"].append(frm)
            rows[f"videos/{vk}/to_timestamp"].append(frm + lengths[e] / fps)

    out_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(rows), out_path)
    print(f"      rebuilt {n_ep} episodes -> {out_path.relative_to(root)}")


def _episodes_present(root: Path) -> bool:
    """True if meta/episodes has at least one non-empty parquet file."""
    files = glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True)
    for f in files:
        try:
            if pq.read_metadata(f).num_rows > 0:
                return True
        except Exception:
            continue
    return False


def compute_baselines(data_files: list[Path], st_idx: dict) -> dict[int, tuple]:
    """episode_index -> ((R_p0,R_R0),(L_p0,L_R0)) from each episode's frame_index==0 STATE pose."""
    baselines: dict[int, tuple] = {}
    for f in data_files:
        df = pq.read_table(
            f, columns=["episode_index", "frame_index", "observation.state"]
        ).to_pandas()
        first = df[df["frame_index"] == 0]
        for _, row in first.iterrows():
            ep = int(row["episode_index"])
            if ep in baselines:
                continue
            rp, rq, _ = split_arm_pose(row["observation.state"], st_idx, "right")
            lp, lq, _ = split_arm_pose(row["observation.state"], st_idx, "left")
            baselines[ep] = ((rp, quat_to_mat(rq)), (lp, quat_to_mat(lq)))
    return baselines


def compute_relative_ee_stats(per_ep: dict, horizon: int, n_arms: int) -> dict:
    """Stats of the RELATIVE action ``S_t^{-1}·A_{t+k}`` over all valid (t, k) within episodes.

    This is what action_mode='relative_ee' feeds the model: the current STATE pose S_t is the
    anchor and the future ACTION pose A_{t+k} is the target. ``k`` ranges 1..horizon (chunk starts
    at t+1); chunk_size must be <= horizon at train time (else re-run with a larger --horizon).
    """
    rels = []
    for d in per_ep.values():
        S = torch.from_numpy(np.stack(d["s"]).astype(np.float32))  # (L, EE_DIM) state
        A = torch.from_numpy(np.stack(d["a"]).astype(np.float32))  # (L, EE_DIM) action
        L = S.shape[0]
        for k in range(1, horizon + 1):
            if L - k <= 0:
                break
            rels.append(ee_to_relative(S[: L - k], A[k:], n_arms=n_arms).numpy())
    return feature_stats(np.concatenate(rels))


def compute_relative_quat_stats(per_ep: dict, horizon: int, n_arms: int) -> dict:
    """Stats of the RELATIVE quat action ``S_t^{-1}·A_{t+k}`` (quat format, 16-dim for 2 arms)."""
    rels = []
    for d in per_ep.values():
        S = torch.from_numpy(np.stack(d["s_quat"]).astype(np.float32))  # (L, EE_DIM_QUAT) state
        A = torch.from_numpy(np.stack(d["a_quat"]).astype(np.float32))  # (L, EE_DIM_QUAT) action
        L = S.shape[0]
        for k in range(1, horizon + 1):
            if L - k <= 0:
                break
            rels.append(ee_to_relative(S[: L - k], A[k:], n_arms=n_arms, rot_mode="quat").numpy())
    return feature_stats(np.concatenate(rels))


def feature_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "count": np.array([arr.shape[0]], dtype=np.int64),
        "q01": np.quantile(arr, 0.01, axis=0),
        "q10": np.quantile(arr, 0.10, axis=0),
        "q50": np.quantile(arr, 0.50, axis=0),
        "q90": np.quantile(arr, 0.90, axis=0),
        "q99": np.quantile(arr, 0.99, axis=0),
    }


def _fsl_f32(arr2d: np.ndarray) -> pa.Array:
    """(N, EE_DIM) float32 -> pyarrow fixed_size_list<float>[EE_DIM] (matches existing action column)."""
    flat = pa.array(np.ascontiguousarray(arr2d, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, EE_DIM)


def _fsl_f32_quat(arr2d: np.ndarray) -> pa.Array:
    """(N, EE_DIM_QUAT) float32 -> pyarrow fixed_size_list<float>[EE_DIM_QUAT]."""
    flat = pa.array(np.ascontiguousarray(arr2d, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, EE_DIM_QUAT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, help="Dataset dir to modify in place")
    ap.add_argument("--src", type=Path, help="Source dataset (used with --dst to copy first)")
    ap.add_argument("--dst", type=Path, help="Destination dataset (copy of --src, then modify)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--horizon", type=int, default=32,
                    help="Max action chunk horizon for action_relative_ee stats; train chunk_size must be <= this.")
    ap.add_argument("--frame-eps", type=float, default=0.05,
                    help="Max |action_pos - state_pos| at episode start to assert same world frame (metres).")
    args = ap.parse_args()

    if args.src and args.dst:
        if args.dst.exists():
            if args.overwrite:
                shutil.rmtree(args.dst)
            else:
                raise SystemExit(f"dst exists (use --overwrite): {args.dst}")
        print(f"[copy] {args.src} -> {args.dst}")
        shutil.copytree(args.src, args.dst)
        root = args.dst
    elif args.root:
        root = args.root
    else:
        raise SystemExit("provide --root, or --src and --dst")

    info = json.loads((root / "meta" / "info.json").read_text())
    st_idx = pose_indices(info["features"]["observation.state"]["names"])
    ac_idx = pose_indices(info["features"]["action"]["names"])
    out_names = build_names()
    n_arms = EE_DIM // PER_ARM_DIM

    # Defensive: some UMI dumps ship data + videos but no meta/episodes metadata, which the LeRobot
    # v3.0 loader requires (load_episodes -> FileNotFoundError otherwise). Rebuild it from the data
    # parquet + video frame counts before touching stats. No-op when it already exists.
    if _episodes_present(root):
        print("[0/4] meta/episodes present — skipping rebuild")
    else:
        print("[0/4] meta/episodes missing — rebuilding from data + video frame counts")
        rebuild_episodes(root)

    data_files = sorted_data_files(root)
    print(f"[1/4] baselines from {len(data_files)} data files")
    baselines = compute_baselines(data_files, st_idx)
    print(f"      {len(baselines)} episode baselines")

    # accumulate global + per-episode stats
    all_state, all_action = [], []
    all_state_eq, all_action_eq = [], []
    per_ep: dict[int, dict[str, list]] = {}

    print("[2/4] converting data parquet (adding columns)")
    for f in data_files:
        tab = pq.read_table(f)
        df = tab.to_pandas()
        ep_col = df["episode_index"].to_numpy()
        state_col = df["observation.state"].to_numpy()
        action_col = df["action"].to_numpy()
        st_ee = np.zeros((len(df), EE_DIM), dtype=np.float32)
        ac_ee = np.zeros((len(df), EE_DIM), dtype=np.float32)
        st_eq = np.zeros((len(df), EE_DIM_QUAT), dtype=np.float32)
        ac_eq = np.zeros((len(df), EE_DIM_QUAT), dtype=np.float32)
        for i in range(len(df)):
            ep = int(ep_col[i])
            base = baselines[ep]
            # State and ACTION share the state's T0 baseline so that S_t^{-1}·A_{t+k} cancels T0.
            st_ee[i] = to_episode_ee(state_col[i], st_idx, base)
            ac_ee[i] = to_episode_ee(action_col[i], ac_idx, base)
            # quat variants (same relative transform, rotation stored as quaternion)
            st_eq[i] = to_episode_quat_umi(state_col[i], st_idx, base)
            ac_eq[i] = to_episode_quat_umi(action_col[i], ac_idx, base)
            per_ep.setdefault(ep, {"s": [], "a": [], "s_quat": [], "a_quat": []})
            per_ep[ep]["s"].append(st_ee[i])
            per_ep[ep]["a"].append(ac_ee[i])
            per_ep[ep]["s_quat"].append(st_eq[i])
            per_ep[ep]["a_quat"].append(ac_eq[i])
        all_state.append(st_ee)
        all_action.append(ac_ee)
        all_state_eq.append(st_eq)
        all_action_eq.append(ac_eq)

        # sanity: state & action must be in the same world frame (their T0 is shared).
        first_rows = np.where(df["frame_index"].to_numpy() == 0)[0]
        for r in first_rows:
            sp, _, _ = split_arm_pose(state_col[r], st_idx, "right")
            apos, _, _ = split_arm_pose(action_col[r], ac_idx, "right")
            d = float(np.linalg.norm(sp - apos))
            if d > args.frame_eps:
                raise SystemExit(
                    f"episode {int(ep_col[r])}: right-arm |action_pos - state_pos|={d:.3f}m "
                    f"> --frame-eps={args.frame_eps}; action/state may not share a world frame."
                )

        # drop pre-existing new columns (idempotent re-run), then append fresh
        for col in NEW_FEATURES:
            if col in tab.column_names:
                tab = tab.drop([col])
        tab = tab.append_column("observation.state_episode_ee",   _fsl_f32(st_ee))
        tab = tab.append_column("action_episode_ee",               _fsl_f32(ac_ee))
        tab = tab.append_column("observation.state_episode_quat",  _fsl_f32_quat(st_eq))
        tab = tab.append_column("action_episode_quat",             _fsl_f32_quat(ac_eq))
        pq.write_table(tab, f)
        print(f"      {f.relative_to(root)}  ({len(df)} frames)")

    # ---- meta/info.json ----
    print("[3/4] meta/info.json + meta/stats.json")
    out_names_quat = build_names_quat()
    template = dict(info["features"]["action"])
    for feat in NEW_FEATURES:
        if "quat" in feat:
            info["features"][feat] = {**template, "shape": [EE_DIM_QUAT], "names": list(out_names_quat)}
        else:
            info["features"][feat] = {**template, "shape": [EE_DIM], "names": list(out_names)}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False))

    # ---- meta/stats.json (global) ----
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    rel_stats      = compute_relative_ee_stats(per_ep, horizon=args.horizon, n_arms=n_arms)
    rel_quat_stats = compute_relative_quat_stats(per_ep, horizon=args.horizon, n_arms=EE_DIM_QUAT // PER_ARM_DIM_QUAT)
    stat_sources = (
        ("observation.state_episode_ee",   feature_stats(np.concatenate(all_state))),
        ("action_episode_ee",               feature_stats(np.concatenate(all_action))),
        # action_relative_ee: the relativized target the model trains on (St^-1·A_{t+k}).
        ("action_relative_ee",              rel_stats),
        # quat variants
        ("observation.state_episode_quat",  feature_stats(np.concatenate(all_state_eq))),
        ("action_episode_quat",             feature_stats(np.concatenate(all_action_eq))),
        ("action_relative_quat",            rel_quat_stats),
    )
    for feat, st in stat_sources:
        stats[feat] = {k: (v.astype(np.int64).tolist() if k == "count" else v.astype(np.float32).tolist())
                       for k, v in st.items()}
    stats_path.write_text(json.dumps(stats, indent=4, ensure_ascii=False))
    print(f"      action_relative_ee stats over horizon={args.horizon} "
          f"(q01..q99 range example dim0: {rel_stats['q01'][0]:.4f}..{rel_stats['q99'][0]:.4f})")

    # ---- meta/episodes/*.parquet (per-episode stats) ----
    print("[4/4] meta/episodes per-episode stats")
    ep_stats = {ep: {
        "observation.state_episode_ee":  feature_stats(np.stack(d["s"])),
        "action_episode_ee":              feature_stats(np.stack(d["a"])),
        "observation.state_episode_quat": feature_stats(np.stack(d["s_quat"])),
        "action_episode_quat":            feature_stats(np.stack(d["a_quat"])),
    } for ep, d in per_ep.items()}
    ep_files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    for ef in ep_files:
        tab = pq.read_table(ef)
        eps = [int(e) for e in tab.column("episode_index").to_pylist()]
        for feat in NEW_FEATURES:
            for stat in STAT_KEYS:
                col = f"stats/{feat}/{stat}"
                if col in tab.column_names:
                    tab = tab.drop([col])
                vals = [ep_stats[ep][feat][stat].tolist() for ep in eps]
                typ = pa.list_(pa.int64()) if stat == "count" else pa.list_(pa.float64())
                tab = tab.append_column(col, pa.array(vals, type=typ))
        pq.write_table(tab, ef)

    print(f"\nDone ✅  added rot6d and quat EE columns to {root}")
    print(f"  rot6d features: observation.state_episode_ee, action_episode_ee  ({EE_DIM}-dim)")
    print(f"  quat  features: observation.state_episode_quat, action_episode_quat  ({EE_DIM_QUAT}-dim)")
    print(f"  layout: {out_names}")


if __name__ == "__main__":
    main()
