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

Reads the joint ``observation.state`` / ``action`` (16-dim, dual-arm) and, via Realman forward
kinematics, ADDS eight columns to the SAME dataset (joint columns are left untouched, so joint-mode
training is unaffected):

    observation.state_episode_ee : 20-dim, EE pose of the STATE joints relative to each episode's
                                   FIRST frame (T0^{-1}·Tt), expressed in the first-frame frame.
                                   Rotation as rot6d (first two columns of the rotation matrix).
    action_episode_ee            : 20-dim, IDENTICAL to observation.state_episode_ee (the state's own
                                   trajectory). Kept as a separate column so it can carry the action
                                   horizon (delta_indices) independently of the state window.
    observation.state_absolute_ee: 20-dim, EE pose of the STATE joints in the robot base frame (Tt,
                                   NO T0 subtraction) — keeps absolute workspace position. rot6d.
    action_absolute_ee           : 20-dim, IDENTICAL to observation.state_absolute_ee (separate column
                                   only so it can carry the action horizon independently). rot6d.

    observation.state_episode_quat : 16-dim, same as state_episode_ee but rotation as quaternion
                                     [x, y, z, w] (per arm: xyz(3) + quat(4) + gripper(1) = 8).
    action_episode_quat            : 16-dim, IDENTICAL to observation.state_episode_quat.
    observation.state_absolute_quat: 16-dim, same as state_absolute_ee but quat rotation.
    action_absolute_quat           : 16-dim, IDENTICAL to observation.state_absolute_quat.

``action_relative_ee`` / ``action_relative_quat`` stats (the relativized targets the model trains on)
are anchor-independent (T0 cancels in St^-1·S_{t+k}), so they are computed once and reused by both
episode and absolute state modes.

rot6d columns use per arm ``[xyz(3), rot6d(6), gripper(1)]``, ordered RIGHT arm first then LEFT
(20 = 2 * 10). quat columns use per arm ``[xyz(3), quat_xyzw(4), gripper(1)]`` (16 = 2 * 8).
Gripper is kept absolute in both representations.

The action is the STATE's own future trajectory: at train time the action chunk (future
state_episode_ee values) is relativized against the current state_episode_ee anchor S_t, giving
``S_t^{-1} · S_{t+k}`` = ``T_state_t^{-1} · T_state_{t+k}`` (T0 cancels): the future observed pose
relative to the current observed pose. See ``vtla/engine/utils/ee_transforms.py``.

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

from vtla.engine.utils.ee_transforms import ee_to_relative  # noqa: E402

# Realman SDK (vendored under deployment/sdk); FK is offline, no arm connection needed.
_SDK = _REPO_ROOT / "deployment" / "sdk"
if str(_SDK) not in sys.path:
    sys.path.insert(0, str(_SDK))

from Robotic_Arm.rm_ctypes_wrap import rm_force_type_e, rm_robot_arm_model_e  # noqa: E402
from Robotic_Arm.rm_robot_interface import Algo  # noqa: E402

PER_ARM_DIM = 10
PER_ARM_DIM_QUAT = 8
EE_DIM = 20       # rot6d: 2 arms * 10
EE_DIM_QUAT = 16  # quat:  2 arms * 8
DOF = 7
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
NEW_FEATURES = (
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


def joint_indices(names: list[str]) -> dict:
    """Derive per-arm joint/gripper indices from the input feature names (robust to ordering)."""
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
    for k in ("left_joints", "right_joints"):
        if len(idx[k]) != DOF:
            raise ValueError(f"Expected {DOF} {k}, found {len(idx[k])} in names={names}")
    if idx["left_grip"] is None or idx["right_grip"] is None:
        raise ValueError(f"Missing gripper index in names={names}")
    return idx


def split_arms(vec: np.ndarray, jidx: dict):
    """16-dim joint vector -> (right_joints, right_grip, left_joints, left_grip)."""
    vec = np.asarray(vec, dtype=np.float64)
    return (
        vec[jidx["right_joints"]],
        float(vec[jidx["right_grip"]]),
        vec[jidx["left_joints"]],
        float(vec[jidx["left_grip"]]),
    )


# ----------------------------------------------------------------------------
# Kinematics
def fk(algo: Algo, joints_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def fk_both(algo: Algo, vec16: np.ndarray, jidx: dict):
    rj, rg, lj, lg = split_arms(vec16, jidx)
    return (fk(algo, rj), rg), (fk(algo, lj), lg)


def to_episode_ee(algo: Algo, vec16: np.ndarray, jidx: dict, baseline) -> np.ndarray:
    """16-dim joints -> 20-dim relative-first-frame EE (right then left), using ``baseline`` T0."""
    ((rp, rm), rg), ((lp, lm), lg) = fk_both(algo, vec16, jidx)
    (Rp0, RR0), (Lp0, LR0) = baseline
    return np.concatenate(
        [relative_arm_ee(rp, rm, rg, Rp0, RR0), relative_arm_ee(lp, lm, lg, Lp0, LR0)]
    ).astype(np.float32)


def absolute_arm_ee(pos, mat, grip) -> np.ndarray:
    """Pose in the robot base frame (Tt, no T0): pos/rot6d absolute, gripper absolute."""
    return np.concatenate([pos, mat_to_rot6d(mat), [grip]]).astype(np.float64)


def to_absolute_ee(algo: Algo, vec16: np.ndarray, jidx: dict) -> np.ndarray:
    """16-dim joints -> 20-dim base-frame EE (right then left), NO episode baseline (one step less)."""
    ((rp, rm), rg), ((lp, lm), lg) = fk_both(algo, vec16, jidx)
    return np.concatenate(
        [absolute_arm_ee(rp, rm, rg), absolute_arm_ee(lp, lm, lg)]
    ).astype(np.float32)


def to_episode_quat(algo: Algo, vec16: np.ndarray, jidx: dict, baseline) -> np.ndarray:
    """16-dim joints -> 16-dim relative-first-frame EE with quat rotation (right then left)."""
    ((rp, rm), rg), ((lp, lm), lg) = fk_both(algo, vec16, jidx)
    (Rp0, RR0), (Lp0, LR0) = baseline
    return np.concatenate(
        [relative_arm_ee_quat(rp, rm, rg, Rp0, RR0), relative_arm_ee_quat(lp, lm, lg, Lp0, LR0)]
    ).astype(np.float32)


def to_absolute_quat(algo: Algo, vec16: np.ndarray, jidx: dict) -> np.ndarray:
    """16-dim joints -> 16-dim base-frame EE with quat rotation (right then left)."""
    ((rp, rm), rg), ((lp, lm), lg) = fk_both(algo, vec16, jidx)
    return np.concatenate(
        [absolute_arm_ee_quat(rp, rm, rg), absolute_arm_ee_quat(lp, lm, lg)]
    ).astype(np.float32)


# ----------------------------------------------------------------------------
# Dataset I/O (LeRobot v3.0)
def sorted_data_files(root: Path) -> list[Path]:
    files = glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)

    def key(f: str):
        m = re.search(r"chunk-(\d+)/file-(\d+)", f)
        return (int(m.group(1)), int(m.group(2)))

    return [Path(f) for f in sorted(files, key=key)]


def compute_baselines(algo: Algo, data_files: list[Path], jidx: dict) -> dict[int, tuple]:
    """episode_index -> ((R_p0,R_R0),(L_p0,L_R0)) from each episode's frame_index==0 STATE."""
    baselines: dict[int, tuple] = {}
    for f in data_files:
        df = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pandas()
        first = df[df["frame_index"] == 0]
        for _, row in first.iterrows():
            ep = int(row["episode_index"])
            if ep in baselines:
                continue
            ((rp, rm), _), ((lp, lm), _) = fk_both(algo, row["observation.state"], jidx)
            baselines[ep] = ((rp, rm), (lp, lm))
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
        S = torch.from_numpy(np.stack(d["s"]).astype(np.float32))  # (L, EE_DIM)
        L = S.shape[0]
        for k in range(1, horizon + 1):
            if L - k <= 0:
                break
            rels.append(ee_to_relative(S[: L - k], S[k:], n_arms=n_arms).numpy())
    return feature_stats(np.concatenate(rels))


def compute_relative_quat_stats(per_ep: dict, horizon: int, n_arms: int) -> dict:
    """Stats of the RELATIVE quat action ``S_t^{-1}·S_{t+k}`` (quat format, 16-dim for 2 arms)."""
    rels = []
    for d in per_ep.values():
        S = torch.from_numpy(np.stack(d["s_quat"]).astype(np.float32))  # (L, EE_DIM_QUAT)
        L = S.shape[0]
        for k in range(1, horizon + 1):
            if L - k <= 0:
                break
            rels.append(ee_to_relative(S[: L - k], S[k:], n_arms=n_arms, rot_mode="quat").numpy())
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
    in_names = info["features"]["observation.state"]["names"]
    jidx = joint_indices(in_names)
    out_names = build_names()

    algo = Algo(rm_robot_arm_model_e.RM_MODEL_RM_75_E, rm_force_type_e.RM_MODEL_RM_B_E)
    data_files = sorted_data_files(root)
    print(f"[1/4] baselines from {len(data_files)} data files")
    baselines = compute_baselines(algo, data_files, jidx)
    print(f"      {len(baselines)} episode baselines")

    # accumulate global + per-episode stats
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
        st_ee = np.zeros((len(df), EE_DIM), dtype=np.float32)
        ac_ee = np.zeros((len(df), EE_DIM), dtype=np.float32)
        st_abs = np.zeros((len(df), EE_DIM), dtype=np.float32)
        ac_abs = np.zeros((len(df), EE_DIM), dtype=np.float32)
        st_eq = np.zeros((len(df), EE_DIM_QUAT), dtype=np.float32)
        ac_eq = np.zeros((len(df), EE_DIM_QUAT), dtype=np.float32)
        st_aq = np.zeros((len(df), EE_DIM_QUAT), dtype=np.float32)
        ac_aq = np.zeros((len(df), EE_DIM_QUAT), dtype=np.float32)
        for i in range(len(df)):
            ep = int(ep_col[i])
            base = baselines[ep]
            st_ee[i] = to_episode_ee(algo, state_col[i], jidx, base)
            # action = the state's OWN future trajectory (relativized to current obs at train time),
            # so action_episode_ee is identical per-frame to state_episode_ee (separate column only
            # so it can carry the action horizon independently).
            ac_ee[i] = st_ee[i]
            # absolute_ee: base-frame FK with NO T0 subtraction. The relative training target
            # St^-1·S_{t+k} is anchor-independent, so action_absolute_ee mirrors state_absolute_ee.
            st_abs[i] = to_absolute_ee(algo, state_col[i], jidx)
            ac_abs[i] = st_abs[i]
            # quat variants
            st_eq[i] = to_episode_quat(algo, state_col[i], jidx, base)
            ac_eq[i] = st_eq[i]
            st_aq[i] = to_absolute_quat(algo, state_col[i], jidx)
            ac_aq[i] = st_aq[i]
            per_ep.setdefault(ep, {"s": [], "a": [], "s_abs": [], "a_abs": [],
                                   "s_quat": [], "a_quat": [], "s_abs_quat": [], "a_abs_quat": []})
            per_ep[ep]["s"].append(st_ee[i])
            per_ep[ep]["a"].append(ac_ee[i])
            per_ep[ep]["s_abs"].append(st_abs[i])
            per_ep[ep]["a_abs"].append(ac_abs[i])
            per_ep[ep]["s_quat"].append(st_eq[i])
            per_ep[ep]["a_quat"].append(ac_eq[i])
            per_ep[ep]["s_abs_quat"].append(st_aq[i])
            per_ep[ep]["a_abs_quat"].append(ac_aq[i])
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
        tab = tab.append_column("observation.state_episode_ee",   _fsl_f32(st_ee))
        tab = tab.append_column("action_episode_ee",               _fsl_f32(ac_ee))
        tab = tab.append_column("observation.state_absolute_ee",   _fsl_f32(st_abs))
        tab = tab.append_column("action_absolute_ee",              _fsl_f32(ac_abs))
        tab = tab.append_column("observation.state_episode_quat",  _fsl_f32_quat(st_eq))
        tab = tab.append_column("action_episode_quat",             _fsl_f32_quat(ac_eq))
        tab = tab.append_column("observation.state_absolute_quat", _fsl_f32_quat(st_aq))
        tab = tab.append_column("action_absolute_quat",            _fsl_f32_quat(ac_aq))
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
    # action_relative_ee: stats of the relativized action the model actually trains on.
    rel_stats      = compute_relative_ee_stats(per_ep, horizon=args.horizon, n_arms=EE_DIM // PER_ARM_DIM)
    rel_quat_stats = compute_relative_quat_stats(per_ep, horizon=args.horizon, n_arms=EE_DIM_QUAT // PER_ARM_DIM_QUAT)
    stat_sources = (
        ("observation.state_episode_ee",   feature_stats(np.concatenate(all_state))),
        ("action_episode_ee",               feature_stats(np.concatenate(all_action))),
        ("observation.state_absolute_ee",   feature_stats(np.concatenate(all_state_abs))),
        ("action_absolute_ee",              feature_stats(np.concatenate(all_action_abs))),
        # action_relative_ee is anchor-independent (St^-1·S_{t+k} cancels T0).
        ("action_relative_ee",              rel_stats),
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
          f"observation.state_absolute_ee, action_absolute_ee  ({EE_DIM}-dim)")
    print(f"  quat  features: observation.state_episode_quat, action_episode_quat, "
          f"observation.state_absolute_quat, action_absolute_quat  ({EE_DIM_QUAT}-dim)")
    print(f"  layout: {out_names}")


if __name__ == "__main__":
    main()
