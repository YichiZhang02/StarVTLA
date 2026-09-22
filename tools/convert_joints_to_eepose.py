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
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

# Allow running as a standalone script (python tools/convert_joints_to_eepose.py): put the repo
# root on sys.path so ``vtla`` is importable regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deployment.robots import RobotConfig  # noqa: E402
from vtla.engine.utils.ee_transforms import encode_relative_tcp  # noqa: E402
from vtla.datasets.tcp_contract import build_tcp_contract, clean_legacy_ee_metadata, drop_legacy_ee_columns
from vtla.engine.utils.ee_kinematics import (make_realman_algo, joint_indices, fk_both, to_episode_ee, to_absolute_ee)  # noqa: E402

PER_ARM_DIM = 10
EE_DIM = 20       # rot6d: 2 arms * 10
DOF = 7
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
NEW_FEATURES = (
    "observation.state_episode_joint",
    "observation.state_episode_ee",
    "action_episode_ee",
    "observation.state_absolute_ee",
    "action_absolute_ee",
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


# ----------------------------------------------------------------------------
# Kinematics


# ----------------------------------------------------------------------------
# Dataset I/O (LeRobot v3.0)
def sorted_data_files(root: Path) -> list[Path]:
    files = glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)

    def key(f: str):
        m = re.search(r"chunk-(\d+)/file-(\d+)", f)
        return (int(m.group(1)), int(m.group(2)))

    return [Path(f) for f in sorted(files, key=key)]


def compute_baselines(algo: Any, data_files: list[Path], jidx: dict, calibration) -> dict[int, tuple]:
    """Map episode index to each present arm's first-frame ``(position, rotation)``."""
    baselines: dict[int, tuple] = {}
    for f in data_files:
        df = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pandas()
        first = df[df["frame_index"] == 0]
        for _, row in first.iterrows():
            ep = int(row["episode_index"])
            if ep in baselines:
                continue
            arms = fk_both(algo, row["observation.state"], jidx, calibration)
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


def compute_relative_ee_stats(
    per_ep: dict, horizon: int, n_arms: int, action_gap: int = 0
) -> dict:
    """Statistics of TCP-local zero-centered actions over valid, unpadded target offsets."""
    rels = []
    for d in per_ep.values():
        S = torch.from_numpy(np.stack(d["s_abs"]).astype(np.float32))
        A = torch.from_numpy(np.stack(d["a_abs"]).astype(np.float32))
        L = S.shape[0]
        for k in range(action_gap, action_gap + horizon):
            if L - k <= 0:
                break
            rels.append(encode_relative_tcp(S[: L - k], A[k:], n_arms=n_arms).numpy())
    return feature_stats(np.concatenate(rels))


def compute_relative_joint_stats(
    per_ep: dict, horizon: int, relative_mask: np.ndarray, action_gap: int = 0
) -> dict:
    """Stats for future absolute joint commands relative to the current observed joints."""
    rels = []
    for d in per_ep.values():
        state = np.stack(d["joint_state"]).astype(np.float32)
        action = np.stack(d["joint_action"]).astype(np.float32)
        for k in range(action_gap, action_gap + horizon):
            valid = len(state) - k
            if valid <= 0:
                break
            rel = action[k : k + valid].copy()
            rel[:, relative_mask] -= state[:valid, relative_mask]
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
    ap.add_argument("--preserve-ee-grippers", action="store_true",
                    help="During migration, retain calibrated grippers from existing absolute EE columns.")
    ap.add_argument("--horizon", type=int, default=32,
                    help="Number of action steps used for relative-action statistics.")
    ap.add_argument("--action-gap", type=int, default=0,
                    help="First GT action offset; stats cover gap .. gap+horizon-1.")
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
    out_names = build_names(sides)

    force_type = robot_config_cls.kinematics_force_type
    assert force_type is not None
    calibration = {side: RobotConfig.get_flange_tcp_calibration(robot_type, side) for side in sides}
    algo = make_realman_algo(force_type)
    print(
        f"[kinematics] robot_type={robot_type}, "
        f"force_type={force_type.upper()}"
    )
    data_files = sorted_data_files(root)
    print(f"[1/4] baselines from {len(data_files)} data files")
    baselines = compute_baselines(algo, data_files, jidx, calibration)
    joint_baselines = compute_joint_baselines(data_files)
    print(f"      {len(baselines)} episode baselines")

    # accumulate global + per-episode stats
    all_state_joint_episode = []
    all_state, all_action = [], []
    all_state_abs, all_action_abs = [], []
    per_ep: dict[int, dict[str, list]] = {}

    print("[2/4] converting data parquet (adding columns)")
    for f in data_files:
        tab = drop_legacy_ee_columns(pq.read_table(f))
        df = tab.to_pandas()
        ep_col = df["episode_index"].to_numpy()
        state_col = df["observation.state"].to_numpy()
        action_col = df["action"].to_numpy()
        st_joint_episode = np.zeros((len(df), len(in_names)), dtype=np.float32)
        st_ee = np.zeros((len(df), ee_dim), dtype=np.float32)
        ac_ee = np.zeros((len(df), ee_dim), dtype=np.float32)
        st_abs = np.zeros((len(df), ee_dim), dtype=np.float32)
        ac_abs = np.zeros((len(df), ee_dim), dtype=np.float32)
        for i in range(len(df)):
            ep = int(ep_col[i])
            base = baselines[ep]
            st_joint_episode[i] = np.asarray(state_col[i], dtype=np.float32)
            joint_mask = np.array(["gripper" not in str(name).lower() for name in in_names])
            st_joint_episode[i, joint_mask] -= joint_baselines[ep][joint_mask]
            st_ee[i] = to_episode_ee(algo, state_col[i], jidx, base, flange_tcp_calibration=calibration)
            ac_ee[i] = to_episode_ee(algo, action_col[i], jidx, base, flange_tcp_calibration=calibration)
            st_abs[i] = to_absolute_ee(algo, state_col[i], jidx, flange_tcp_calibration=calibration)
            ac_abs[i] = to_absolute_ee(algo, action_col[i], jidx, flange_tcp_calibration=calibration)
            if args.preserve_ee_grippers:
                for source, outputs in (("observation.state_absolute_ee", (st_ee, st_abs)),
                                        ("action_absolute_ee", (ac_ee, ac_abs))):
                    if source in df:
                        original = np.asarray(df[source].iloc[i])
                        if original.shape != outputs[0][i].shape:
                            raise ValueError(f"Cannot preserve grippers: invalid {source} shape {original.shape}")
                        for output in outputs:
                            output[i, 9::10] = original[9::10]
            per_ep.setdefault(ep, {"s": [], "a": [], "s_abs": [], "a_abs": [],

                                   "joint_episode": [], "joint_state": [], "joint_action": []})
            per_ep[ep]["joint_episode"].append(st_joint_episode[i])
            per_ep[ep]["joint_state"].append(np.asarray(state_col[i], dtype=np.float32))
            per_ep[ep]["joint_action"].append(np.asarray(action_col[i], dtype=np.float32))
            per_ep[ep]["s"].append(st_ee[i])
            per_ep[ep]["a"].append(ac_ee[i])
            per_ep[ep]["s_abs"].append(st_abs[i])
            per_ep[ep]["a_abs"].append(ac_abs[i])
        all_state_joint_episode.append(st_joint_episode)
        all_state.append(st_ee)
        all_action.append(ac_ee)
        all_state_abs.append(st_abs)
        all_action_abs.append(ac_abs)

        # drop pre-existing new columns (idempotent re-run), then append fresh
        for col in NEW_FEATURES:
            if col in tab.column_names:
                tab = tab.drop([col])
        tab = tab.append_column("observation.state_episode_joint", _fsl_f32(st_joint_episode, len(in_names)))
        tab = tab.append_column("observation.state_episode_ee",   _fsl_f32(st_ee, ee_dim))
        tab = tab.append_column("action_episode_ee",               _fsl_f32(ac_ee, ee_dim))
        tab = tab.append_column("observation.state_absolute_ee",   _fsl_f32(st_abs, ee_dim))
        tab = tab.append_column("action_absolute_ee",              _fsl_f32(ac_abs, ee_dim))
        pq.write_table(tab, f)
        print(f"      {f.relative_to(root)}  ({len(df)} frames)")

    clean_legacy_ee_metadata(root, info)
    info["tcp_contract"] = build_tcp_contract(info["robot_type"], args.horizon, args.action_gap)

    # ---- meta/info.json ----
    print("[3/4] meta/info.json + meta/stats.json")
    template = dict(info["features"]["action"])
    for feat in NEW_FEATURES:
        if feat == "observation.state_episode_joint":
            info["features"][feat] = {**template, "shape": [len(in_names)], "names": list(in_names)}
        else:
            info["features"][feat] = {**template, "shape": [ee_dim], "names": list(out_names)}
    info["ee_num_arms"] = n_arms
    info["ee_arm_sides"] = list(sides)
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False))

    # ---- meta/stats.json (global) ----
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    # action_relative_ee: stats of the relativized action the model actually trains on.
    rel_stats = compute_relative_ee_stats(
        per_ep, horizon=args.horizon, n_arms=n_arms, action_gap=args.action_gap
    )
    relative_joint_stats = compute_relative_joint_stats(
        per_ep, args.horizon, joint_mask, action_gap=args.action_gap
    )
    stat_sources = (
        ("observation.state_episode_joint", feature_stats(np.concatenate(all_state_joint_episode))),
        ("observation.state_episode_ee",   feature_stats(np.concatenate(all_state))),
        ("action_episode_ee",               feature_stats(np.concatenate(all_action))),
        ("observation.state_absolute_ee",   feature_stats(np.concatenate(all_state_abs))),
        ("action_absolute_ee",              feature_stats(np.concatenate(all_action_abs))),
        # action_relative_ee is anchor-independent (St^-1·S_{t+k} cancels T0).
        ("action_relative_ee",              rel_stats),
        ("action_relative_joint",           relative_joint_stats),
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
        "observation.state_episode_joint": feature_stats(np.stack(d["joint_episode"])),
        "observation.state_episode_ee":   feature_stats(np.stack(d["s"])),
        "action_episode_ee":               feature_stats(np.stack(d["a"])),
        "observation.state_absolute_ee":   feature_stats(np.stack(d["s_abs"])),
        "action_absolute_ee":              feature_stats(np.stack(d["a_abs"])),
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
    print(f"  rot6d features: observation.state_episode_ee, action_episode_ee, "
          f"observation.state_absolute_ee, action_absolute_ee  ({ee_dim}-dim)")
    print(f"  arms: {list(sides)}")
    print(f"  layout: {out_names}")


if __name__ == "__main__":
    main()
