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

"""Generate TCP rot6d absolute/episode pose columns and zero-centered relative action statistics.

Joint input uses FK followed by the robot tool calibration; UMI input is already TCP.
Raw source vectors are preserved. Quaternion input conversion is only a source/SDK boundary.
Use --src/--dst to copy first, or tools/migrate_tcp_dataset.py for staged migration.
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

from vtla.engine.utils.ee_transforms import encode_relative_tcp  # noqa: E402
from vtla.datasets.tcp_contract import build_tcp_contract, clean_legacy_ee_metadata, drop_legacy_ee_columns

PER_ARM_DIM = 10
EE_DIM = 20       # rot6d: 2 arms * 10
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
NEW_FEATURES = (
    "observation.state_episode_ee",
    "action_episode_ee",
    "observation.state_absolute_ee",
    "action_absolute_ee",
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
            if low in ("gripper_left", "left_gripper"):
                out["left"]["grip"] = i
            elif low in ("gripper_right", "right_gripper"):
                out["right"]["grip"] = i
            continue
        rest = low[len(side) + 1:]  # strip "left_"/"right_"
        if "position_rad" in rest or rest == "gripper":
            out[side]["grip"] = i
        elif rest in ("x", "y", "z"):
            out[side]["pos"]["xyz".index(rest)] = i
        elif rest.startswith("quat_") and rest.split("_")[1] in quat_axis:
            out[side]["quat"][quat_axis[rest.split("_")[1]]] = i
        elif rest in ("qx", "qy", "qz", "qw"):
            out[side]["quat"][quat_axis[rest[1]]] = i
    for side in ("left", "right"):
        d = out[side]
        if d["grip"] is None or None in d["pos"] or None in d["quat"]:
            raise ValueError(f"Could not locate {side} pose/gripper columns in names={names}")
    return out


def set_gripper_calibration(idx: dict, calibration: dict[str, tuple[float, float]]) -> None:
    """Attach per-side ``(open, closed)`` raw values used to produce [0, 1] gripper targets."""
    for side, (open_value, closed_value) in calibration.items():
        if np.isclose(open_value, closed_value):
            raise ValueError(f"{side} gripper open and closed calibration values must differ")
        idx[side]["grip_open"] = float(open_value)
        idx[side]["grip_closed"] = float(closed_value)


def split_arm_pose(vec: np.ndarray, idx: dict, side: str):
    """Extract (pos xyz(3,), quat xyzw(4,), gripper float) for one arm from a frame vector."""
    vec = np.asarray(vec, dtype=np.float64)
    d = idx[side]
    pos = vec[d["pos"]]
    quat = vec[d["quat"]]  # (x, y, z, w)
    grip = float(vec[d["grip"]])
    if "grip_open" in d:
        # Robot grippers use 1=open and 0=closed. Clipping handles small sensor overshoot.
        grip = float(np.clip((grip - d["grip_closed"]) / (d["grip_open"] - d["grip_closed"]), 0, 1))
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


def relative_arm_ee(pos, mat, grip, p0, R0) -> np.ndarray:
    """Pose relative to first frame (T0^{-1}·Tt): pos=R0^T(pt-p0), rot6d(R0^T·Rt), grip absolute."""
    R0t = R0.T
    p_rel = R0t @ (pos - p0)
    R_rel = R0t @ mat
    return np.concatenate([p_rel, mat_to_rot6d(R_rel), [grip]]).astype(np.float64)


def to_episode_ee(vec: np.ndarray, idx: dict, baseline) -> np.ndarray:
    """Frame pose vector -> 20-dim relative-first-frame EE (RIGHT then LEFT), using ``baseline`` T0."""
    (Rp0, RR0), (Lp0, LR0) = baseline
    out = []
    for side, (p0, R0) in (("right", (Rp0, RR0)), ("left", (Lp0, LR0))):
        pos, quat, grip = split_arm_pose(vec, idx, side)
        out.append(relative_arm_ee(pos, quat_to_mat(quat), grip, p0, R0))
    return np.concatenate(out).astype(np.float32)


def to_absolute_ee_umi(vec: np.ndarray, idx: dict) -> np.ndarray:
    """Stored world pose -> canonical base-frame rot6d layout."""
    out = []
    for side in ("right", "left"):
        pos, quat, grip = split_arm_pose(vec, idx, side)
        out.append(np.concatenate([pos, mat_to_rot6d(quat_to_mat(quat)), [grip]]))
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


def compute_relative_ee_stats(
    per_ep: dict, horizon: int, n_arms: int, action_gap: int = 0
) -> dict:
    """Statistics of TCP-local zero-centered actions over valid, unpadded target offsets."""
    rels = []
    for d in per_ep.values():
        S = torch.from_numpy(np.stack(d["s_abs"]).astype(np.float32))  # (L, EE_DIM) state
        A = torch.from_numpy(np.stack(d["a_abs"]).astype(np.float32))  # (L, EE_DIM) action
        L = S.shape[0]
        for k in range(action_gap, action_gap + horizon):
            if L - k <= 0:
                break
            rels.append(encode_relative_tcp(S[: L - k], A[k:], n_arms=n_arms).numpy())
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, help="Dataset dir to modify in place")
    ap.add_argument("--src", type=Path, help="Source dataset (used with --dst to copy first)")
    ap.add_argument("--dst", type=Path, help="Destination dataset (copy of --src, then modify)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preserve-ee-grippers", action="store_true",
                    help="During migration, retain calibrated grippers from existing absolute EE columns.")
    ap.add_argument("--horizon", type=int, default=32,
                    help="Number of action steps used for relative-action statistics.")
    ap.add_argument("--action-gap", type=int, default=0,
                    help="First GT action offset; stats cover gap .. gap+horizon-1.")
    ap.add_argument("--frame-eps", type=float, default=0.05,
                    help="Max |action_pos - state_pos| at episode start to assert same world frame (metres).")
    for side in ("left", "right"):
        ap.add_argument(f"--{side}-gripper-open", type=float)
        ap.add_argument(f"--{side}-gripper-closed", type=float)
    args = ap.parse_args()

    if args.horizon <= 0:
        ap.error("--horizon must be positive")
    if args.action_gap < 0:
        ap.error("--action-gap must be non-negative")

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
    calibration_values = [
        args.left_gripper_open,
        args.left_gripper_closed,
        args.right_gripper_open,
        args.right_gripper_closed,
    ]
    if any(value is not None for value in calibration_values):
        if any(value is None for value in calibration_values):
            ap.error("all four left/right gripper open/closed values must be provided together")
        calibration = {
            "left": (args.left_gripper_open, args.left_gripper_closed),
            "right": (args.right_gripper_open, args.right_gripper_closed),
        }
        set_gripper_calibration(st_idx, calibration)
        set_gripper_calibration(ac_idx, calibration)
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
    all_state, all_action, all_state_abs, all_action_abs = [], [], [], []
    per_ep: dict[int, dict[str, list]] = {}

    print("[2/4] converting data parquet (adding columns)")
    for f in data_files:
        tab = drop_legacy_ee_columns(pq.read_table(f))
        df = tab.to_pandas()
        ep_col = df["episode_index"].to_numpy()
        state_col = df["observation.state"].to_numpy()
        action_col = df["action"].to_numpy()
        st_ee = np.zeros((len(df), EE_DIM), dtype=np.float32)
        ac_ee = np.zeros((len(df), EE_DIM), dtype=np.float32)
        st_abs = np.zeros((len(df), EE_DIM), dtype=np.float32)
        ac_abs = np.zeros((len(df), EE_DIM), dtype=np.float32)
        for i in range(len(df)):
            ep = int(ep_col[i])
            base = baselines[ep]
            # State and ACTION share the state's T0 baseline so that S_t^{-1}·A_{t+k} cancels T0.
            st_ee[i] = to_episode_ee(state_col[i], st_idx, base)
            ac_ee[i] = to_episode_ee(action_col[i], ac_idx, base)
            st_abs[i] = to_absolute_ee_umi(state_col[i], st_idx)
            ac_abs[i] = to_absolute_ee_umi(action_col[i], ac_idx)
            if args.preserve_ee_grippers:
                for source, outputs in (("observation.state_absolute_ee", (st_ee, st_abs)),
                                        ("action_absolute_ee", (ac_ee, ac_abs))):
                    if source in df:
                        original = np.asarray(df[source].iloc[i])
                        if original.shape != outputs[0][i].shape:
                            raise ValueError(f"Cannot preserve grippers: invalid {source} shape {original.shape}")
                        for output in outputs:
                            output[i, 9::10] = original[9::10]
            per_ep.setdefault(ep, {"s": [], "a": [],
                                   "s_abs": [], "a_abs": []})
            per_ep[ep]["s"].append(st_ee[i])
            per_ep[ep]["a"].append(ac_ee[i])
            per_ep[ep]["s_abs"].append(st_abs[i])
            per_ep[ep]["a_abs"].append(ac_abs[i])
        all_state.append(st_ee)
        all_action.append(ac_ee)
        all_state_abs.append(st_abs)
        all_action_abs.append(ac_abs)

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
        tab = tab.append_column("observation.state_absolute_ee",  _fsl_f32(st_abs))
        tab = tab.append_column("action_absolute_ee",              _fsl_f32(ac_abs))
        pq.write_table(tab, f)
        print(f"      {f.relative_to(root)}  ({len(df)} frames)")

    clean_legacy_ee_metadata(root, info)
    info["tcp_contract"] = build_tcp_contract("umi", args.horizon, args.action_gap)

    # ---- meta/info.json ----
    print("[3/4] meta/info.json + meta/stats.json")
    template = dict(info["features"]["action"])
    for feat in NEW_FEATURES:
        info["features"][feat] = {**template, "shape": [EE_DIM], "names": list(out_names)}
    info["robot_type"] = "umi"
    info["ee_num_arms"] = n_arms
    info["ee_arm_sides"] = ["right", "left"]
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False))

    # ---- meta/stats.json (global) ----
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    rel_stats = compute_relative_ee_stats(
        per_ep, horizon=args.horizon, n_arms=n_arms, action_gap=args.action_gap
    )
    stat_sources = (
        ("observation.state_episode_ee",   feature_stats(np.concatenate(all_state))),
        ("action_episode_ee",               feature_stats(np.concatenate(all_action))),
        ("observation.state_absolute_ee",  feature_stats(np.concatenate(all_state_abs))),
        ("action_absolute_ee",              feature_stats(np.concatenate(all_action_abs))),
        # action_relative_ee: the relativized target the model trains on (St^-1·A_{t+k}).
        ("action_relative_ee",              rel_stats),
    )
    for feat, st in stat_sources:
        stats[feat] = {k: (v.astype(np.int64).tolist() if k == "count" else v.astype(np.float32).tolist())
                       for k, v in st.items()}
    stats_path.write_text(json.dumps(stats, indent=4, ensure_ascii=False))
    print(f"      action_relative_ee stats over gap={args.action_gap}, horizon={args.horizon} "
          f"(q01..q99 range example dim0: {rel_stats['q01'][0]:.4f}..{rel_stats['q99'][0]:.4f})")

    # ---- meta/episodes/*.parquet (per-episode stats) ----
    print("[4/4] meta/episodes per-episode stats")
    ep_stats = {ep: {
        "observation.state_episode_ee":  feature_stats(np.stack(d["s"])),
        "action_episode_ee":              feature_stats(np.stack(d["a"])),
        "observation.state_absolute_ee": feature_stats(np.stack(d["s_abs"])),
        "action_absolute_ee":             feature_stats(np.stack(d["a_abs"])),
        "action_relative_ee": compute_relative_ee_stats({ep: d}, args.horizon, n_arms, args.action_gap)
        if len(d["s_abs"]) > args.action_gap else None,
    } for ep, d in per_ep.items()}
    ep_files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    for ef in ep_files:
        tab = drop_legacy_ee_columns(pq.read_table(ef))
        eps = [int(e) for e in tab.column("episode_index").to_pylist()]
        for feat in (*NEW_FEATURES, "action_relative_ee"):
            for stat in STAT_KEYS:
                col = f"stats/{feat}/{stat}"
                if col in tab.column_names:
                    tab = tab.drop([col])
                vals = [ep_stats[ep][feat][stat].tolist() if ep_stats[ep][feat] is not None else None for ep in eps]
                typ = pa.list_(pa.int64()) if stat == "count" else pa.list_(pa.float64())
                tab = tab.append_column(col, pa.array(vals, type=typ))
        pq.write_table(tab, ef)

    print(f"\nDone ✅  added TCP rot6d columns to {root}")
    print(f"  rot6d features: observation.state_episode_ee, action_episode_ee  ({EE_DIM}-dim)")
    print(f"  layout: {out_names}")


if __name__ == "__main__":
    main()
