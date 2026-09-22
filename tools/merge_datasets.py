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

"""Offline: merge several LeRobot v3.0 datasets into ONE training dataset.

``aggregate_datasets`` (the merge engine) requires EVERY source dataset to expose the *exact same*
feature set — same keys, same shapes. Our pen-in-case captures don't satisfy that: the 260701
``notac`` run has no finger (tactile) cameras, while 260702 carries four of them. Merging them
directly raises ``ValueError: Same features is expected``.

So this tool first intersects the feature sets across all sources, drops any key a source has but
the intersection doesn't (e.g. the finger cams that only 260702 owns), then merges the aligned
copies. Result: a single dataset whose feature set is exactly what every source shares — for the
pen-in-case pair that means wrist+top cams + state/action + EE columns, i.e. a ``notac`` dataset
you train with ``tactile_mode=none``.

Non-destructive: the source datasets are never modified. Aligned per-source copies land in a temp
dir (removed on success unless ``--keep-intermediate``); the merged dataset is written to ``--out``.

Usage:
    python tools/merge_datasets.py \
        --roots playground/data/A playground/data/B \
        --out   playground/data/A_B_merged

    # then train on the single merged dataset (no training-code change needed):
    bash train.sh A_B_merged pi05 4 16 20000 false none episode_rot6d relative_rot6d
"""

import argparse
import logging
import shutil
from pathlib import Path

from vtla.datasets.dataset_tools import merge_datasets, remove_feature
from vtla.datasets.lerobot_dataset import LeRobotDataset

# Bookkeeping columns every LeRobot dataset carries; never candidates for removal.
_REQUIRED = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _feature_signature(features: dict) -> dict[str, tuple]:
    """Map feature name -> (dtype, shape-as-tuple) so we can compare across datasets robustly."""
    sig = {}
    for name, info in features.items():
        shape = tuple(info.get("shape", ()))
        sig[name] = (info.get("dtype"), shape)
    return sig


def _common_features(datasets: list[LeRobotDataset]) -> set[str]:
    """Keys present in ALL datasets with identical (dtype, shape). These survive the merge."""
    sigs = [_feature_signature(ds.meta.features) for ds in datasets]
    common = set(sigs[0])
    for sig in sigs[1:]:
        common &= set(sig)
    # Keep only keys whose (dtype, shape) agree everywhere (a mismatch can't be aggregated).
    agreed = set()
    for name in common:
        vals = {sig[name] for sig in sigs}
        if len(vals) == 1:
            agreed.add(name)
        else:
            logging.warning(
                "Feature %r has differing dtype/shape across datasets (%s); dropping from merge.",
                name,
                vals,
            )
    return agreed | _REQUIRED  # required cols are always kept even if create() adds them implicitly


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Two or more source dataset roots (e.g. playground/data/foo playground/data/bar).",
    )
    ap.add_argument("--out", required=True, help="Output root for the merged dataset.")
    ap.add_argument(
        "--repo-id",
        default=None,
        help="repo_id for the merged dataset (default: basename of --out).",
    )
    ap.add_argument(
        "--tmp-dir",
        default=None,
        help="Where to stage feature-aligned per-source copies (default: <out>_align_tmp).",
    )
    ap.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the aligned per-source copies instead of deleting them after a successful merge.",
    )
    ap.add_argument(
        "--video-files-size-in-mb",
        type=int,
        default=10,
        help=(
            "Max size (MB) per merged video file. Small values keep many small per-episode mp4s instead "
            "of a few giant ones; large mp4s make random-access frame decoding much slower at train time "
            "(pyav re-parses the whole file index on every sample). Default 10 restores a small-file layout "
            "close to the un-merged source datasets. Pass 0 to use aggregate's default (large files)."
        ),
    )
    ap.add_argument(
        "--data-files-size-in-mb",
        type=int,
        default=0,
        help="Max size (MB) per merged parquet data file. 0 (default) uses aggregate's default.",
    )
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    if len(roots) < 2:
        ap.error("--roots needs at least two datasets to merge.")
    for r in roots:
        if not (r / "meta" / "info.json").is_file():
            ap.error(f"Not a LeRobot dataset (no meta/info.json): {r}")

    out = Path(args.out)
    if out.exists():
        ap.error(f"--out already exists, refusing to overwrite: {out}")
    repo_id = args.repo_id or out.name
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else out.parent / f"{out.name}_align_tmp"

    # Load sources. repo_id is just a label here; data is read from the local root.
    logging.info("Loading %d source datasets", len(roots))
    sources = [LeRobotDataset(repo_id=r.name, root=r) for r in roots]

    common = _common_features(sources)
    logging.info("Common feature set (%d keys): %s", len(common), sorted(common))

    # Align each source to the common feature set (drop extras like finger cams on 260702).
    aligned: list[LeRobotDataset] = []
    created_tmp: list[Path] = []
    for src, root in zip(sources, roots, strict=True):
        extras = [k for k in src.meta.features if k not in common]
        if not extras:
            logging.info("[%s] already matches common features; using as-is.", root.name)
            aligned.append(src)
            continue
        logging.info("[%s] removing %d extra feature(s): %s", root.name, len(extras), extras)
        aligned_root = tmp_dir / root.name
        aligned_root.parent.mkdir(parents=True, exist_ok=True)
        aligned_ds = remove_feature(
            src,
            feature_names=extras,
            output_dir=aligned_root,
            repo_id=f"{root.name}_aligned",
        )
        aligned.append(aligned_ds)
        created_tmp.append(aligned_root)

    logging.info("Merging %d datasets -> %s", len(aligned), out)
    merged = merge_datasets(
        aligned,
        output_repo_id=repo_id,
        output_dir=out,
        video_files_size_in_mb=args.video_files_size_in_mb or None,
        data_files_size_in_mb=args.data_files_size_in_mb or None,
    )

    logging.info(
        "Merged dataset ready: %s | episodes=%d frames=%d",
        out,
        merged.meta.total_episodes,
        merged.meta.total_frames,
    )

    if created_tmp and not args.keep_intermediate:
        logging.info("Removing intermediate aligned copies under %s", tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    elif created_tmp:
        logging.info("Keeping intermediate aligned copies under %s", tmp_dir)

    print("=" * 67)
    print(f"完成 ✅  合并数据集: {out}")
    print(f"  episodes={merged.meta.total_episodes}  frames={merged.meta.total_frames}")
    print(f"  features={sorted(merged.meta.features)}")
    print(f"  训练示例: bash train.sh {repo_id} pi05 4 16 20000 false none episode_rot6d relative_rot6d")
    print("=" * 67)


if __name__ == "__main__":
    main()
