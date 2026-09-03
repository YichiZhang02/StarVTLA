#!/usr/bin/env python3
"""Resolve an ordinary or named mixture ID for the shell training entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vtla.datasets.feature_schema import mixture_feature_schema_diff
from vtla.datasets.mixture_registry import load_mixture_definitions, resolve_member_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    parser.add_argument("--catalog-root", type=Path, default=Path("playground/data"))
    parser.add_argument("--mixture-config", type=Path, default=Path("configs/data_mixtures.yaml"))
    return parser.parse_args()


def load_features(root: Path) -> tuple[dict, dict]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    with info_path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    return info, info.get("features", {})


def main() -> None:
    args = parse_args()
    definitions = load_mixture_definitions(args.mixture_config)
    definition = definitions.get(args.dataset_id)
    physical_root = args.catalog_root / args.dataset_id
    if definition is None:
        roots = [physical_root]
        kind = "dataset"
        normalized_weights = [1.0]
    else:
        if physical_root.is_dir():
            raise ValueError(
                f"Dataset ID {args.dataset_id!r} is both a mixture and a directory: {physical_root}"
            )
        roots = [resolve_member_root(definition, member, args.catalog_root) for member in definition.members]
        if any(root is None for root in roots):
            raise ValueError(
                f"Mixture {args.dataset_id!r} has members without local roots; train.sh requires local datasets."
            )
        kind = "mixture"
        normalized_weights = list(definition.normalized_weights)

    infos_and_features = [load_features(Path(root)) for root in roots]
    reference_info, reference_features = infos_and_features[0]
    for root, (info, features) in zip(roots[1:], infos_and_features[1:], strict=True):
        feature_differences = mixture_feature_schema_diff(reference_features, features)
        if feature_differences:
            details = "\n  - ".join(feature_differences)
            raise ValueError(
                f"Mixture member feature schema differs from the first member: {root}\n"
                f"  - {details}"
            )
        if info.get("fps") != reference_info.get("fps"):
            raise ValueError(f"Mixture member FPS differs from the first member: {root}")
        if info.get("robot_type") != reference_info.get("robot_type"):
            raise ValueError(f"Mixture member robot_type differs from the first member: {root}")

    videos = [(key, value) for key, value in reference_features.items() if value.get("dtype") == "video"]
    is_tactile = lambda key, value: bool(value.get("tactile_encoding")) or "finger" in key.lower()  # noqa: E731
    tactile = [key for key, value in videos if is_tactile(key, value)]
    wrist = [key for key, value in videos if not is_tactile(key, value) and "wrist" in key.lower()]
    top = [key for key, value in videos if not is_tactile(key, value) and key not in wrist]

    print(kind)
    print("|".join(str(Path(root)) for root in roots))
    print("[" + ",".join(top) + "]")
    print("[" + ",".join(wrist) + "]")
    print("[" + ",".join(tactile) + "]")
    print(",".join(f"{weight:.8g}" for weight in normalized_weights))


if __name__ == "__main__":
    main()
