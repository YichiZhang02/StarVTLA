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

"""Offline: add EE-pose columns to a joint LeRobot v3.0 dataset (in place).

Reads the joint ``observation.state`` names, automatically detects one or two complete Realman
arms, and via forward kinematics ADDS eight columns to the SAME dataset. Joint columns are left
untouched, so joint-mode training is unaffected. EE dimensions are 10/8 per arm for rot6d/quat.
The RealMan B/ISF FK variant is selected strictly from ``meta/info.json`` robot_type.

    observation.state_episode_ee : 10 dims/arm, EE pose of the STATE joints relative to each
                                   episode's FIRST frame (T0^{-1}·Tt), expressed in that frame.
                                   Rotation uses rot6d (first two rotation-matrix columns).
    action_episode_ee            : FK of the real joint ACTION command, expressed relative to the
                                   state's episode-first pose T0.
    observation.state_absolute_ee: 10 dims/arm, EE pose in the robot base frame (Tt,
                                   NO T0 subtraction) — keeps absolute workspace position. rot6d.
    action_absolute_ee           : FK of the real joint ACTION command in the robot base frame.

    observation.state_episode_quat : 8 dims/arm, same as state_episode_ee but rotation as quaternion
                                     [x, y, z, w].
    action_episode_quat            : IDENTICAL to observation.state_episode_quat.
    observation.state_absolute_quat: same as state_absolute_ee but quaternion rotation.
    action_absolute_quat           : IDENTICAL to observation.state_absolute_quat.

``action_relative_ee`` / ``action_relative_quat`` stats (the relativized targets the model trains on)
are anchor-independent (T0 cancels in St^-1·S_{t+k}), so they are computed once and reused by both
episode and absolute state modes.

rot6d columns use per arm ``[xyz(3), rot6d(6), gripper(1)]``, ordered RIGHT arm first then LEFT
when both exist. quat columns use per arm ``[xyz(3), quat_xyzw(4), gripper(1)]``.
Gripper is kept absolute in both representations.

At train time, relative EE targets are computed from the current observed pose and future commanded
action: ``T_state_t^{-1} · T_action_{t+k}``. See ``vtla/engine/utils/ee_transforms.py``.

Updates ``meta/info.json`` (features), ``meta/stats.json`` (global), and ``meta/episodes/*.parquet``
(per-episode stats) so the dataset loads with the new features.

Usage:
    python tools/convert_joints_to_eepose.py --root playground/data/<dataset>
    python tools/convert_joints_to_eepose.py --src <src> --dst <dst>   # copy first
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.spatial.transform import Rotation as R

# Allow running as a standalone script (python tools/convert_joints_to_eepose.py): put the repo
# root on sys.path so ``vtla`` is importable regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deployment.robots import RobotConfig  # noqa: E402
from vtla.engine.utils.ee_transforms import ee_to_relative  # noqa: E402
from vtla.engine.utils.ee_kinematics import make_realman_algo  # noqa: E402

PER_ARM_DIM = 10
PER_ARM_DIM_QUAT = 8
EE_DIM = 20       # rot6d: 2 arms * 10
EE_DIM_QUAT = 16  # quat:  2 arms * 8
DOF = 7
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
NEW_FEATURES = (
    "observation.state_episode_joint",
    "observation.state_episode_ee",
    "action_episode_ee",
    "observation.state_absolute_ee",
    "action_absolute_ee",
    "observation.state_episode_quat",
    "action_episode_quat",
    "observation.state_absolute_quat",
    "action_absolute_quat",
)


# ----------------------------------------------------------------------------
# Layout helpers
def build_names(sides: tuple[str, ...] = ("right", "left")) -> list[str]:
    """Rot6d output names in canonical arm order (right before left when both exist)."""
    names: list[str] = []
    for side in sides:
        names += [f"{side}_ee_x", f"{side}_ee_y", f"{side}_ee_z"]
        names += [f"{side}_ee_rot6d_{i}" for i in range(6)]
        names += [f"{side}_gripper"]
    return names


def build_names_quat(sides: tuple[str, ...] = ("right", "left")) -> list[str]:
    """Quaternion output names in canonical arm order."""
    names: list[str] = []
    for side in sides:
        names += [f"{side}_ee_x", f"{side}_ee_y", f"{side}_ee_z"]
        names += [f"{side}_ee_qx", f"{side}_ee_qy", f"{side}_ee_qz", f"{side}_ee_qw"]
        names += [f"{side}_gripper"]
    return names


def joint_indices(names: list[str]) -> dict:
    """Derive one or two complete arm layouts from input feature names."""
    idx = {"left_joints": [], "right_joints": [], "left_grip": None, "right_grip": None}
    for i, n in enumerate(names):
        low = n.lower()
        side = "left" if low.startswith("left") else "right" if low.startswith("right") else None
        if side is None:
            continue
        if "gripper" in low:
            idx[f"{side}_grip"] = i
        elif "joint" in low:
            idx[f"{side}_joints"].append(i)
    sides: list[str] = []
    for side in ("right", "left"):
        joints = idx[f"{side}_joints"]
        grip = idx[f"{side}_grip"]
        if not joints and grip is None:
            continue
        if len(joints) != DOF:
            raise ValueError(
                f"Expected {DOF} {side}_joints, found {len(joints)} in names={names}"
            )
        if grip is None:
            raise ValueError(f"Missing {side} gripper index in names={names}")
        sides.append(side)
    if not sides:
        raise ValueError(f"No complete left/right arm found in names={names}")
    idx["sides"] = tuple(sides)
    return idx


def split_arms(vec: np.ndarray, jidx: dict):
    """Joint vector -> per-arm ``(joints, gripper)`` tuples in canonical arm order."""
    vec = np.asarray(vec, dtype=np.float64)
    return tuple(
        (vec[jidx[f"{side}_joints"]], float(vec[jidx[f"{side}_grip"]]))
        for side in jidx["sides"]
    )


# ----------------------------------------------------------------------------
# Kinematics
def fk(algo: Any, joints_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-arm FK. 7 joint radians -> (pos xyz(3,), rotation matrix(3,3))."""
    joints_deg = np.degrees(joints_rad).tolist()
    pose = algo.rm_algo_forward_kinematics(joints_deg, flag=0)  # [x,y,z, qw,qx,qy,qz]
    pos = np.array(pose[:3], dtype=np.float64)
    qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
    mat = R.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy uses (x,y,z,w)
    return pos, mat


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
    """Like relative_arm_ee but stores rotation as quaternion [x,y,z,w]. Returns 8-dim."""
    R0t = R0.T
    p_rel = R0t @ (pos - p0)
    R_rel = R0t @ mat
    return np.concatenate([p_rel, mat_to_quat_np(R_rel), [grip]]).astype(np.float64)


def absolute_arm_ee_quat(pos, mat, grip) -> np.ndarray:
    """Like absolute_arm_ee but stores rotation as quaternion [x,y,z,w]. Returns 8-dim."""
    return np.concatenate([pos, mat_to_quat_np(mat), [grip]]).astype(np.float64)


def fk_both(algo: Any, joint_vector: np.ndarray, jidx: dict):
    """Run FK for every arm present in ``jidx`` (one or two arms)."""
    return tuple((fk(algo, joints), grip) for joints, grip in split_arms(joint_vector, jidx))


def to_episode_ee(algo: Any, joint_vector: np.ndarray, jidx: dict, baseline) -> np.ndarray:
    """Joints -> per-arm relative-first-frame rot6d EE using ``baseline`` T0."""
    arms = fk_both(algo, joint_vector, jidx)
    return np.concatenate([
        relative_arm_ee(pos, mat, grip, pos0, mat0)
        for ((pos, mat), grip), (pos0, mat0) in zip(arms, baseline, strict=True)
    ]).astype(np.float32)


def absolute_arm_ee(pos, mat, grip) -> np.ndarray:
    """Pose in the robot base frame (Tt, no T0): pos/rot6d absolute, gripper absolute."""
    return np.concatenate([pos, mat_to_rot6d(mat), [grip]]).astype(np.float64)


def to_absolute_ee(algo: Any, joint_vector: np.ndarray, jidx: dict) -> np.ndarray:
    """Joints -> per-arm base-frame rot6d EE, without an episode baseline."""
    return np.concatenate([
        absolute_arm_ee(pos, mat, grip)
        for (pos, mat), grip in fk_both(algo, joint_vector, jidx)
    ]).astype(np.float32)


def to_episode_quat(algo: Any, joint_vector: np.ndarray, jidx: dict, baseline) -> np.ndarray:
    """Joints -> per-arm relative-first-frame quaternion EE."""
    arms = fk_both(algo, joint_vector, jidx)
    return np.concatenate([
        relative_arm_ee_quat(pos, mat, grip, pos0, mat0)
        for ((pos, mat), grip), (pos0, mat0) in zip(arms, baseline, strict=True)
    ]).astype(np.float32)


def to_absolute_quat(algo: Any, joint_vector: np.ndarray, jidx: dict) -> np.ndarray:
    """Joints -> per-arm base-frame quaternion EE."""
    return np.concatenate([
        absolute_arm_ee_quat(pos, mat, grip)
        for (pos, mat), grip in fk_both(algo, joint_vector, jidx)
    ]).astype(np.float32)


# ----------------------------------------------------------------------------
# Dataset I/O (LeRobot v3.0)
def sorted_data_files(root: Path) -> list[Path]:
    files = glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)

    def key(f: str):
        m = re.search(r"chunk-(\d+)/file-(\d+)", f)
        return (int(m.group(1)), int(m.group(2)))

    return [Path(f) for f in sorted(files, key=key)]


def compute_baselines(algo: Any, data_files: list[Path], jidx: dict) -> dict[int, tuple]:
    """Map episode index to each present arm's first-frame ``(position, rotation)``."""
    baselines: dict[int, tuple] = {}
    for f in data_files:
        df = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pandas()
        first = df[df["frame_index"] == 0]
        for _, row in first.iterrows():
            ep = int(row["episode_index"])
            if ep in baselines:
                continue
            arms = fk_both(algo, row["observation.state"], jidx)
            baselines[ep] = tuple(pose for pose, _grip in arms)
    return baselines


def compute_joint_baselines(data_files: list[Path]) -> dict[int, np.ndarray]:
    """Map episode index to its first raw joint observation."""
    baselines: dict[int, np.ndarray] = {}
    for f in data_files:
        df = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pandas()
        for _, row in df[df["frame_index"] == 0].iterrows():
            baselines.setdefault(int(row["episode_index"]), np.asarray(row["observation.state"], dtype=np.float32))
    return baselines


def compute_relative_ee_stats(per_ep: dict, horizon: int, n_arms: int) -> dict:
    """Stats of the RELATIVE action ``S_t^{-1}·S_{t+k}`` over all valid (t, k) within episodes.

    This is what action_mode='relative_ee' feeds the model (the per-frame stored episode_ee is
    absolute-in-episode, but training relativizes it). Stored under ``action_relative_ee`` and used
    for action normalization. ``k`` ranges 1..horizon (chunk starts at t+1); chunk_size must be
    <= horizon at train time (otherwise re-run with a larger --horizon).
    """
    rels = []
    for d in per_ep.values():
        S = torch.from_numpy(np.stack(d["s_abs"]).astype(np.float32))
        A = torch.from_numpy(np.stack(d["a_abs"]).astype(np.float32))
        L = S.shape[0]
        for k in range(1, horizon + 1):
            if L - k <= 0:
                break
            rels.append(ee_to_relative(S[: L - k], A[k:], n_arms=n_arms).numpy())
    return feature_stats(np.concatenate(rels))


def compute_relative_quat_stats(per_ep: dict, horizon: int, n_arms: int) -> dict:
    """Stats of the RELATIVE quat action ``S_t^{-1}·S_{t+k}`` (quat format, 16-dim for 2 arms)."""
    rels = []
    for d in per_ep.values():
        S = torch.from_numpy(np.stack(d["s_abs_quat"]).astype(np.float32))
        A = torch.from_numpy(np.stack(d["a_abs_quat"]).astype(np.float32))
        L = S.shape[0]
        for k in range(1, horizon + 1):
            if L - k <= 0:
                break
            rels.append(ee_to_relative(S[: L - k], A[k:], n_arms=n_arms, rot_mode="quat").numpy())
    return feature_stats(np.concatenate(rels))


def compute_relative_joint_stats(per_ep: dict, horizon: int, relative_mask: np.ndarray) -> dict:
    """Stats for future absolute joint commands relative to the current observed joints."""
    rels = []
    for d in per_ep.values():
        state = np.stack(d["joint_state"]).astype(np.float32)
        action = np.stack(d["joint_action"]).astype(np.float32)
        for k in range(1, horizon + 1):
            if len(state) - k <= 0:
                break
            rel = action[k:].copy()
            rel[:, relative_mask] -= state[:-k, relative_mask]
            rels.append(rel)
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


def _fsl_f32(arr2d: np.ndarray, dim: int) -> pa.Array:
    """Convert a 2D float32 array to a PyArrow fixed-size-list column."""
    flat = pa.array(np.ascontiguousarray(arr2d, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, help="Dataset dir to modify in place")
    ap.add_argument("--src", type=Path, help="Source dataset (used with --dst to copy first)")
    ap.add_argument("--dst", type=Path, help="Destination dataset (copy of --src, then modify)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--horizon", type=int, default=32,
                    help="Max action chunk horizon for action_relative_ee stats; train chunk_size must be <= this.")
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
    robot_type = info.get("robot_type")
    supported_robot_types = RobotConfig.get_kinematics_robot_types()
    if robot_type not in supported_robot_types:
        raise SystemExit(
            f"Unsupported or missing robot_type={robot_type!r} in meta/info.json; "
            f"expected one of {supported_robot_types}."
        )
    in_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    if list(action_names) != list(in_names):
        raise SystemExit(
            "Joint conversion requires action and observation.state to use the same ordered joint layout."
        )
    jidx = joint_indices(in_names)
    sides = jidx["sides"]
    try:
        robot_config_cls = RobotConfig.validate_kinematics_sides(robot_type, sides)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    n_arms = len(sides)
    ee_dim = n_arms * PER_ARM_DIM
    ee_dim_quat = n_arms * PER_ARM_DIM_QUAT
    out_names = build_names(sides)

    force_type = robot_config_cls.kinematics_force_type
    assert force_type is not None
    algo = make_realman_algo(force_type)
    print(
        f"[kinematics] robot_type={robot_type}, "
        f"force_type={force_type.upper()}"
    )
    data_files = sorted_data_files(root)
    print(f"[1/4] baselines from {len(data_files)} data files")
    baselines = compute_baselines(algo, data_files, jidx)
    joint_baselines = compute_joint_baselines(data_files)
    print(f"      {len(baselines)} episode baselines")

    # accumulate global + per-episode stats
    all_state_joint_episode = []
    all_state, all_action = [], []
    all_state_abs, all_action_abs = [], []
    all_state_eq, all_action_eq = [], []
    all_state_aq, all_action_aq = [], []
    per_ep: dict[int, dict[str, list]] = {}

    print("[2/4] converting data parquet (adding columns)")
    for f in data_files:
        tab = pq.read_table(f)
        df = tab.to_pandas()
        ep_col = df["episode_index"].to_numpy()
        state_col = df["observation.state"].to_numpy()
        action_col = df["action"].to_numpy()
        st_joint_episode = np.zeros((len(df), len(in_names)), dtype=np.float32)
        st_ee = np.zeros((len(df), ee_dim), dtype=np.float32)
        ac_ee = np.zeros((len(df), ee_dim), dtype=np.float32)
        st_abs = np.zeros((len(df), ee_dim), dtype=np.float32)
        ac_abs = np.zeros((len(df), ee_dim), dtype=np.float32)
        st_eq = np.zeros((len(df), ee_dim_quat), dtype=np.float32)
        ac_eq = np.zeros((len(df), ee_dim_quat), dtype=np.float32)
        st_aq = np.zeros((len(df), ee_dim_quat), dtype=np.float32)
        ac_aq = np.zeros((len(df), ee_dim_quat), dtype=np.float32)
        for i in range(len(df)):
            ep = int(ep_col[i])
            base = baselines[ep]
            st_joint_episode[i] = np.asarray(state_col[i], dtype=np.float32)
            joint_mask = np.array(["gripper" not in str(name).lower() for name in in_names])
            st_joint_episode[i, joint_mask] -= joint_baselines[ep][joint_mask]
            st_ee[i] = to_episode_ee(algo, state_col[i], jidx, base)
            ac_ee[i] = to_episode_ee(algo, action_col[i], jidx, base)
            st_abs[i] = to_absolute_ee(algo, state_col[i], jidx)
            ac_abs[i] = to_absolute_ee(algo, action_col[i], jidx)
            # quat variants
            st_eq[i] = to_episode_quat(algo, state_col[i], jidx, base)
            ac_eq[i] = to_episode_quat(algo, action_col[i], jidx, base)
            st_aq[i] = to_absolute_quat(algo, state_col[i], jidx)
            ac_aq[i] = to_absolute_quat(algo, action_col[i], jidx)
            per_ep.setdefault(ep, {"s": [], "a": [], "s_abs": [], "a_abs": [],
                                   "s_quat": [], "a_quat": [], "s_abs_quat": [], "a_abs_quat": [],
                                   "joint_episode": [], "joint_state": [], "joint_action": []})
            per_ep[ep]["joint_episode"].append(st_joint_episode[i])
            per_ep[ep]["joint_state"].append(np.asarray(state_col[i], dtype=np.float32))
            per_ep[ep]["joint_action"].append(np.asarray(action_col[i], dtype=np.float32))
            per_ep[ep]["s"].append(st_ee[i])
            per_ep[ep]["a"].append(ac_ee[i])
            per_ep[ep]["s_abs"].append(st_abs[i])
            per_ep[ep]["a_abs"].append(ac_abs[i])
            per_ep[ep]["s_quat"].append(st_eq[i])
            per_ep[ep]["a_quat"].append(ac_eq[i])
            per_ep[ep]["s_abs_quat"].append(st_aq[i])
            per_ep[ep]["a_abs_quat"].append(ac_aq[i])
        all_state_joint_episode.append(st_joint_episode)
        all_state.append(st_ee)
        all_action.append(ac_ee)
        all_state_abs.append(st_abs)
        all_action_abs.append(ac_abs)
        all_state_eq.append(st_eq)
        all_action_eq.append(ac_eq)
        all_state_aq.append(st_aq)
        all_action_aq.append(ac_aq)

        # drop pre-existing new columns (idempotent re-run), then append fresh
        for col in NEW_FEATURES:
            if col in tab.column_names:
                tab = tab.drop([col])
        tab = tab.append_column("observation.state_episode_joint", _fsl_f32(st_joint_episode, len(in_names)))
        tab = tab.append_column("observation.state_episode_ee",   _fsl_f32(st_ee, ee_dim))
        tab = tab.append_column("action_episode_ee",               _fsl_f32(ac_ee, ee_dim))
        tab = tab.append_column("observation.state_absolute_ee",   _fsl_f32(st_abs, ee_dim))
        tab = tab.append_column("action_absolute_ee",              _fsl_f32(ac_abs, ee_dim))
        tab = tab.append_column("observation.state_episode_quat",  _fsl_f32(st_eq, ee_dim_quat))
        tab = tab.append_column("action_episode_quat",             _fsl_f32(ac_eq, ee_dim_quat))
        tab = tab.append_column("observation.state_absolute_quat", _fsl_f32(st_aq, ee_dim_quat))
        tab = tab.append_column("action_absolute_quat",            _fsl_f32(ac_aq, ee_dim_quat))
        pq.write_table(tab, f)
        print(f"      {f.relative_to(root)}  ({len(df)} frames)")

    # ---- meta/info.json ----
    print("[3/4] meta/info.json + meta/stats.json")
    out_names_quat = build_names_quat(sides)
    template = dict(info["features"]["action"])
    for feat in NEW_FEATURES:
        if feat == "observation.state_episode_joint":
            info["features"][feat] = {**template, "shape": [len(in_names)], "names": list(in_names)}
        elif "quat" in feat:
            info["features"][feat] = {**template, "shape": [ee_dim_quat], "names": list(out_names_quat)}
        else:
            info["features"][feat] = {**template, "shape": [ee_dim], "names": list(out_names)}
    info["ee_num_arms"] = n_arms
    info["ee_arm_sides"] = list(sides)
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False))

    # ---- meta/stats.json (global) ----
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    # action_relative_ee: stats of the relativized action the model actually trains on.
    rel_stats = compute_relative_ee_stats(per_ep, horizon=args.horizon, n_arms=n_arms)
    rel_quat_stats = compute_relative_quat_stats(per_ep, horizon=args.horizon, n_arms=n_arms)
    relative_joint_stats = compute_relative_joint_stats(per_ep, args.horizon, joint_mask)
    stat_sources = (
        ("observation.state_episode_joint", feature_stats(np.concatenate(all_state_joint_episode))),
        ("observation.state_episode_ee",   feature_stats(np.concatenate(all_state))),
        ("action_episode_ee",               feature_stats(np.concatenate(all_action))),
        ("observation.state_absolute_ee",   feature_stats(np.concatenate(all_state_abs))),
        ("action_absolute_ee",              feature_stats(np.concatenate(all_action_abs))),
        # action_relative_ee is anchor-independent (St^-1·S_{t+k} cancels T0).
        ("action_relative_ee",              rel_stats),
        ("action_relative_joint",           relative_joint_stats),
        # quat variants
        ("observation.state_episode_quat",  feature_stats(np.concatenate(all_state_eq))),
        ("action_episode_quat",             feature_stats(np.concatenate(all_action_eq))),
        ("observation.state_absolute_quat", feature_stats(np.concatenate(all_state_aq))),
        ("action_absolute_quat",            feature_stats(np.concatenate(all_action_aq))),
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
        "observation.state_episode_joint": feature_stats(np.stack(d["joint_episode"])),
        "observation.state_episode_ee":   feature_stats(np.stack(d["s"])),
        "action_episode_ee":               feature_stats(np.stack(d["a"])),
        "observation.state_absolute_ee":   feature_stats(np.stack(d["s_abs"])),
        "action_absolute_ee":              feature_stats(np.stack(d["a_abs"])),
        "observation.state_episode_quat":  feature_stats(np.stack(d["s_quat"])),
        "action_episode_quat":             feature_stats(np.stack(d["a_quat"])),
        "observation.state_absolute_quat": feature_stats(np.stack(d["s_abs_quat"])),
        "action_absolute_quat":            feature_stats(np.stack(d["a_abs_quat"])),
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
    print(f"  rot6d features: observation.state_episode_ee, action_episode_ee, "
          f"observation.state_absolute_ee, action_absolute_ee  ({ee_dim}-dim)")
    print(f"  quat  features: observation.state_episode_quat, action_episode_quat, "
          f"observation.state_absolute_quat, action_absolute_quat  ({ee_dim_quat}-dim)")
    print(f"  arms: {list(sides)}")
    print(f"  layout: {out_names}")


if __name__ == "__main__":
    main()
