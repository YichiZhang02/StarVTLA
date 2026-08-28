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

"""Offline: compute and report the mean of a LeRobot dataset's ``observation.state``.

By default it averages the FIRST frame of every episode across episodes — i.e. the mean starting
pose of the robot for diagnostics. Use ``--frames all`` to
average over every frame instead (that global mean already lives in ``meta/stats.json``; this tool
just recomputes it grouped nicely by joint name).

For a dual-arm joint dataset the state is 16-dim
(``left_main_joint1..7``, ``left_main_gripper``, ``right_main_joint1..7``, ``right_main_gripper``).
The joint means are emitted as JSON and the gripper means are reported on the side. Deployment
home is always captured from the connected robot and cannot be overridden by this output.

Usage:
    python tools/compute_dataset_mean_state.py --root playground/data/<dataset>
    python tools/compute_dataset_mean_state.py --root <dataset> --frames all
    python tools/compute_dataset_mean_state.py --root <dataset> --state-key observation.state
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

def _load_state_names(root: Path, state_key: str) -> list[str] | None:
    """State component names from meta/info.json (for pretty printing + home_joints keys)."""
    info = json.loads((root / "meta" / "info.json").read_text())
    feat = info.get("features", {}).get(state_key)
    if feat is None:
        raise SystemExit(f"错误: {state_key!r} 不在 {root}/meta/info.json 的 features 里")
    return feat.get("names")


def _first_frame_states(root: Path, state_key: str) -> np.ndarray:
    """Stack the FIRST frame (min frame_index) of every episode -> (n_episodes, state_dim)."""
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"错误: {root}/data 下没有 parquet")
    # episode_index -> (best_frame_index, state_vector); keep the smallest frame_index seen.
    best: dict[int, tuple[int, np.ndarray]] = {}
    for f in files:
        df = pq.read_table(f, columns=["episode_index", "frame_index", state_key]).to_pandas()
        for ep, fi, st in zip(df["episode_index"], df["frame_index"], df[state_key]):
            ep, fi = int(ep), int(fi)
            if ep not in best or fi < best[ep][0]:
                best[ep] = (fi, np.asarray(st, dtype=np.float64))
    return np.stack([best[ep][1] for ep in sorted(best)])


def _all_frame_states(root: Path, state_key: str) -> np.ndarray:
    """Stack EVERY frame's state -> (n_frames, state_dim)."""
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"错误: {root}/data 下没有 parquet")
    chunks = []
    for f in files:
        df = pq.read_table(f, columns=[state_key]).to_pandas()
        chunks.append(np.stack([np.asarray(v, dtype=np.float64) for v in df[state_key]]))
    return np.concatenate(chunks, axis=0)


def main():
    ap = argparse.ArgumentParser(description="计算并输出数据集 state 均值")
    ap.add_argument("--root", required=True, help="数据集目录 (含 meta/ 与 data/)")
    ap.add_argument("--state-key", default="observation.state", help="要求均值的 state 列名")
    ap.add_argument("--frames", choices=["first", "all"], default="first",
                    help="first=每 episode 首帧再跨 episode 平均 (home 位姿); all=全帧全局平均")
    args = ap.parse_args()

    root = Path(args.root)
    if not (root / "meta" / "info.json").is_file():
        raise SystemExit(f"错误: 不是有效数据集 (缺 meta/info.json): {root}")

    names = _load_state_names(root, args.state_key)
    states = (_first_frame_states(root, args.state_key) if args.frames == "first"
              else _all_frame_states(root, args.state_key))

    mean = states.mean(axis=0)
    std = states.std(axis=0)
    lo, hi = states.min(axis=0), states.max(axis=0)
    n, dim = states.shape
    if names is None:
        names = [f"dim{i}" for i in range(dim)]

    src = "每 episode 首帧" if args.frames == "first" else "全部帧"
    print(f"# 数据集: {root}", file=sys.stderr)
    print(f"# state 列: {args.state_key}  ({dim} 维)  样本: {n} 个 {src}", file=sys.stderr)
    print(f"# {'name':<20} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}", file=sys.stderr)
    for nm, m, s, a, b in zip(names, mean, std, lo, hi):
        print(f"  {nm:<20} {m:>10.6f} {s:>10.6f} {a:>10.6f} {b:>10.6f}", file=sys.stderr)

    # Joint JSON excludes grippers so it remains easy to inspect or compare.
    joints = {nm: round(float(m), 6) for nm, m in zip(names, mean) if "joint" in nm}
    grippers = {nm: round(float(m), 6) for nm, m in zip(names, mean) if "gripper" in nm}

    if joints:
        print(json.dumps(joints, separators=(", ", ": ")))
    else:
        # Non-joint state (such as EE): print the complete mean vector.
        print(json.dumps([round(float(m), 6) for m in mean]))

    if grippers:
        gvals = list(grippers.values())
        gmean = round(float(np.mean(gvals)), 6)
        print(f"# gripper 均值: {grippers}  (左右平均={gmean}) -> --robot.home_gripper={gmean}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
