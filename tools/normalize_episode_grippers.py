#!/usr/bin/env python3

"""Normalize both grippers jointly within every episode, in place.

For each episode, this script pools the calibrated left/right gripper values from
both the canonical observation and action EE features. One shared affine transform
is then applied to both sides so the pooled minimum and maximum become ``--min``
and ``--max`` exactly. All gripper-bearing observation/action vector features are
updated; an unsided placeholder such as ``gripper=9930`` is intentionally ignored.

The source dataset is overwritten without a backup. Parquet files are first written
to sibling temporary files and only replaced after transformation, validation, and
global stats generation all succeed. Image/video stats are preserved because their
values do not live in the data Parquet files.

Example:
    python tools/normalize_episode_grippers.py \
        --dataset-id my_dataset --min 0.5 --max 1.0

Inspect without writing:
    python tools/normalize_episode_grippers.py \
        --dataset-id my_dataset --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


BASIC_STAT_KEYS = ("min", "max", "mean", "std", "count")
QUANTILE_LEVELS = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}
DEFAULT_OBSERVATION_SOURCE = "observation.state_episode_ee"
DEFAULT_ACTION_SOURCE = "action_episode_ee"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "playground" / "data"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument(
        "--dataset-id",
        help="dataset directory name under --data-root (normally playground/data)",
    )
    dataset.add_argument("--root", type=Path, help="explicit LeRobot dataset root")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"parent directory used with --dataset-id (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument("--min", dest="output_min", type=float, default=0.5)
    parser.add_argument("--max", dest="output_max", type=float, default=1.0)
    parser.add_argument(
        "--observation-source",
        default=DEFAULT_OBSERVATION_SOURCE,
        help="calibrated observation feature used to derive episode bounds",
    )
    parser.add_argument(
        "--action-source",
        default=DEFAULT_ACTION_SOURCE,
        help="calibrated action feature used to derive episode bounds",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="minimum allowed pooled gripper span within an episode",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        help="action horizon for action_relative stats; defaults to umi_processing.json",
    )
    parser.add_argument(
        "--action-gap",
        type=int,
        help="action gap for action_relative stats; defaults to umi_processing.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def data_files(root: Path) -> list[Path]:
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no data parquet files found under {root / 'data'}")
    return paths


def side_gripper_indices(names: list[str]) -> dict[str, int]:
    """Return left/right gripper indices, ignoring unsided placeholder fields."""
    result: dict[str, int] = {}
    for index, name in enumerate(names):
        low = str(name).lower()
        if "gripper" not in low:
            continue
        if low.startswith("left_") or low.endswith("_left"):
            side = "left"
        elif low.startswith("right_") or low.endswith("_right"):
            side = "right"
        else:
            continue
        if side in result:
            raise ValueError(f"multiple {side} gripper dimensions found in names={names}")
        result[side] = index
    return result


def gripper_features(info: dict, parquet_columns: set[str]) -> dict[str, dict[str, int]]:
    found: dict[str, dict[str, int]] = {}
    for feature, spec in info.get("features", {}).items():
        if feature not in parquet_columns or not isinstance(spec, dict):
            continue
        names = spec.get("names")
        if not isinstance(names, list):
            continue
        indices = side_gripper_indices(names)
        if indices:
            if set(indices) != {"left", "right"}:
                raise ValueError(f"feature {feature!r} has only one sided gripper: {indices}")
            if not (feature == "action" or feature.startswith(("action_", "observation."))):
                raise ValueError(f"cannot classify gripper feature as observation/action: {feature}")
            found[feature] = indices
    return found


def vector_values(table: pa.Table, feature: str) -> np.ndarray:
    values = np.asarray(table.column(feature).to_pylist(), dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{feature} must be a dense vector feature, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{feature} contains missing or non-finite values")
    return values


def source_grippers(
    table: pa.Table,
    observation_source: str,
    action_source: str,
    source_indices: dict[str, dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    outputs = []
    for feature in (observation_source, action_source):
        values = vector_values(table, feature)
        indices = source_indices[feature]
        outputs.append(values[:, [indices["left"], indices["right"]]])
    return outputs[0], outputs[1]


def compute_episode_bounds(
    paths: list[Path],
    observation_source: str,
    action_source: str,
    source_indices: dict[str, dict[str, int]],
    epsilon: float,
) -> dict[int, tuple[float, float]]:
    bounds: dict[int, list[float]] = {}
    for path in paths:
        table = pq.read_table(path, columns=["episode_index", observation_source, action_source])
        episodes = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        observation, action = source_grippers(
            table, observation_source, action_source, source_indices
        )
        for episode in np.unique(episodes):
            mask = episodes == episode
            pooled = np.concatenate((observation[mask].reshape(-1), action[mask].reshape(-1)))
            low, high = float(pooled.min()), float(pooled.max())
            current = bounds.setdefault(int(episode), [low, high])
            current[0] = min(current[0], low)
            current[1] = max(current[1], high)

    result = {episode: (values[0], values[1]) for episode, values in bounds.items()}
    degenerate = {
        episode: (low, high)
        for episode, (low, high) in result.items()
        if high - low <= epsilon
    }
    if degenerate:
        preview = list(sorted(degenerate.items()))[:10]
        raise ValueError(
            f"{len(degenerate)} episodes have pooled gripper span <= {epsilon}: {preview}"
        )
    return result


def normalize_rows(
    values: np.ndarray,
    episodes: np.ndarray,
    bounds: dict[int, tuple[float, float]],
    output_min: float,
    output_max: float,
) -> np.ndarray:
    normalized = np.empty_like(values, dtype=np.float64)
    scale = output_max - output_min
    for episode in np.unique(episodes):
        mask = episodes == episode
        low, high = bounds[int(episode)]
        normalized[mask] = output_min + scale * (values[mask] - low) / (high - low)
    return np.clip(normalized, output_min, output_max)


def replace_vector_column(table: pa.Table, feature: str, values: np.ndarray) -> pa.Table:
    column_index = table.column_names.index(feature)
    arrow_type = table.schema.field(feature).type
    replacement = pa.array(values.tolist(), type=arrow_type)
    return table.set_column(column_index, feature, replacement)


def transform_table(
    table: pa.Table,
    targets: dict[str, dict[str, int]],
    source_indices: dict[str, dict[str, int]],
    observation_source: str,
    action_source: str,
    bounds: dict[int, tuple[float, float]],
    output_min: float,
    output_max: float,
) -> pa.Table:
    episodes = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    observation, action = source_grippers(
        table, observation_source, action_source, source_indices
    )
    normalized_observation = normalize_rows(
        observation, episodes, bounds, output_min, output_max
    )
    normalized_action = normalize_rows(action, episodes, bounds, output_min, output_max)

    for feature, indices in targets.items():
        values = vector_values(table, feature)
        normalized = (
            normalized_action
            if feature == "action" or feature.startswith("action_")
            else normalized_observation
        )
        values[:, indices["left"]] = normalized[:, 0]
        values[:, indices["right"]] = normalized[:, 1]
        table = replace_vector_column(table, feature, values)
    return table


def feature_stats(values: np.ndarray, keys: tuple[str, ...]) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"cannot compute stats for array with shape {values.shape}")

    available = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([values.shape[0]], dtype=np.int64),
    }
    for key, quantile in QUANTILE_LEVELS.items():
        if key in keys:
            available[key] = np.quantile(values, quantile, axis=0)
    return {
        key: available[key].astype(np.int64 if key == "count" else np.float32).tolist()
        for key in keys
        if key in available
    }


def column_values(paths: list[Path], column: str) -> np.ndarray:
    chunks = []
    for path in paths:
        data = pq.read_table(path, columns=[column]).column(column).to_pylist()
        chunks.append(np.asarray(data, dtype=np.float64))
    return np.concatenate(chunks, axis=0)


def resolve_temporal_config(
    root: Path, horizon_arg: int | None, action_gap_arg: int | None
) -> tuple[int, int]:
    manifest_path = root / "meta" / "umi_processing.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    horizon = horizon_arg if horizon_arg is not None else manifest.get("horizon")
    action_gap = action_gap_arg if action_gap_arg is not None else manifest.get("action_gap")
    if horizon is None or action_gap is None:
        raise ValueError(
            "action_relative stats require --horizon and --action-gap when they are not "
            "recorded in meta/umi_processing.json"
        )
    if int(horizon) <= 0 or int(action_gap) < 0:
        raise ValueError(f"invalid horizon/action_gap: {horizon}/{action_gap}")
    return int(horizon), int(action_gap)


def relative_stats(
    paths: list[Path],
    state_feature: str,
    action_feature: str,
    horizon: int,
    action_gap: int,
    rot_mode: str,
    keys: tuple[str, ...],
) -> dict[str, list]:
    import torch

    from vtla.engine.utils.ee_transforms import ee_to_relative

    per_episode: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for path in paths:
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", state_feature, action_feature]
        )
        episodes = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        frames = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
        states = vector_values(table, state_feature).astype(np.float32)
        actions = vector_values(table, action_feature).astype(np.float32)
        for row, episode in enumerate(episodes):
            per_episode[int(episode)].append((int(frames[row]), states[row], actions[row]))

    relative = []
    arm_width = 10 if rot_mode == "rot6d" else 8
    for episode in sorted(per_episode):
        rows = sorted(per_episode[episode], key=lambda item: item[0])
        states = torch.from_numpy(np.stack([row[1] for row in rows]))
        actions = torch.from_numpy(np.stack([row[2] for row in rows]))
        n_arms = states.shape[1] // arm_width
        length = states.shape[0]
        for offset in range(action_gap, action_gap + horizon):
            if length - offset <= 0:
                break
            relative.append(
                ee_to_relative(
                    states[: length - offset],
                    actions[offset:],
                    n_arms=n_arms,
                    rot_mode=rot_mode,
                ).numpy()
            )
    if not relative:
        raise ValueError("no valid state/action pairs available for action_relative stats")
    return feature_stats(np.concatenate(relative, axis=0), keys)


def rebuild_stats(
    root: Path,
    paths: list[Path],
    old_stats: dict,
    horizon_arg: int | None,
    action_gap_arg: int | None,
) -> dict:
    rebuilt = dict(old_stats)
    schema = pq.read_schema(paths[0])
    parquet_columns = set(schema.names)
    for column in schema.names:
        keys = tuple(old_stats.get(column, {}).keys()) or BASIC_STAT_KEYS
        try:
            values = column_values(paths, column)
        except (TypeError, ValueError):
            continue
        rebuilt[column] = feature_stats(values, keys)

    relative_specs = (
        (
            "action_relative_ee",
            "observation.state_episode_ee",
            "action_episode_ee",
            "rot6d",
        ),
        (
            "action_relative_quat",
            "observation.state_episode_quat",
            "action_episode_quat",
            "quat",
        ),
    )
    requested = [spec for spec in relative_specs if spec[0] in old_stats]
    if requested:
        horizon, action_gap = resolve_temporal_config(root, horizon_arg, action_gap_arg)
        for output, state, action, mode in requested:
            if state not in parquet_columns or action not in parquet_columns:
                raise ValueError(f"cannot rebuild {output}: missing {state} or {action}")
            keys = tuple(old_stats[output].keys()) or tuple(BASIC_STAT_KEYS) + tuple(QUANTILE_LEVELS)
            rebuilt[output] = relative_stats(
                paths, state, action, horizon, action_gap, mode, keys
            )
    return rebuilt


def validate_transformed(
    paths: list[Path],
    observation_source: str,
    action_source: str,
    source_indices: dict[str, dict[str, int]],
    output_min: float,
    output_max: float,
) -> None:
    bounds = compute_episode_bounds(
        paths, observation_source, action_source, source_indices, epsilon=0.0
    )
    tolerance = 2e-6
    failures = {
        episode: (low, high)
        for episode, (low, high) in bounds.items()
        if not (
            np.isclose(low, output_min, atol=tolerance, rtol=0)
            and np.isclose(high, output_max, atol=tolerance, rtol=0)
        )
    }
    if failures:
        raise ValueError(f"post-write episode range validation failed: {list(failures.items())[:10]}")


def main() -> None:
    args = parse_args()
    if args.dataset_id is not None:
        dataset_id = Path(args.dataset_id)
        if dataset_id.name != args.dataset_id or args.dataset_id in (".", ".."):
            raise ValueError("--dataset-id must be one directory name, not a path")
        root = (args.data_root / dataset_id).resolve()
    else:
        root = args.root.resolve()
    if not np.isfinite([args.output_min, args.output_max, args.epsilon]).all():
        raise ValueError("--min, --max, and --epsilon must be finite")
    if args.output_min >= args.output_max:
        raise ValueError(f"--min must be less than --max, got {args.output_min} >= {args.output_max}")
    if args.epsilon < 0:
        raise ValueError("--epsilon must be non-negative")

    info = load_json(root / "meta" / "info.json")
    stats_path = root / "meta" / "stats.json"
    old_stats = load_json(stats_path)
    paths = data_files(root)
    first_schema = pq.read_schema(paths[0])
    targets = gripper_features(info, set(first_schema.names))
    for source in (args.observation_source, args.action_source):
        if source not in targets:
            raise ValueError(
                f"source feature {source!r} is missing or has no left/right gripper names"
            )
    source_indices = {
        args.observation_source: targets[args.observation_source],
        args.action_source: targets[args.action_source],
    }

    bounds = compute_episode_bounds(
        paths,
        args.observation_source,
        args.action_source,
        source_indices,
        args.epsilon,
    )
    spans = np.asarray([high - low for low, high in bounds.values()])
    print(f"dataset: {root}")
    print(f"episodes: {len(bounds)}")
    print(f"target pooled range per episode: [{args.output_min}, {args.output_max}]")
    print(
        "source pooled span min/median/max: "
        f"{spans.min():.9g} / {np.median(spans):.9g} / {spans.max():.9g}"
    )
    print("gripper features: " + ", ".join(targets))
    if args.dry_run:
        print("dry run complete; no files changed")
        return

    temp_paths: dict[Path, Path] = {}
    stats_temp = stats_path.with_name(f".{stats_path.name}.normalize-grippers.tmp")
    committed = False
    try:
        if stats_temp.exists():
            raise FileExistsError(f"stale temporary file exists: {stats_temp}")
        for path in paths:
            temp = path.with_name(f".{path.name}.normalize-grippers.tmp")
            if temp.exists():
                raise FileExistsError(f"stale temporary file exists: {temp}")
            table = pq.read_table(path)
            transformed = transform_table(
                table,
                targets,
                source_indices,
                args.observation_source,
                args.action_source,
                bounds,
                args.output_min,
                args.output_max,
            )
            pq.write_table(transformed, temp, compression="snappy")
            temp_paths[path] = temp
            print(f"prepared {path.relative_to(root)} ({len(table)} rows)")

        transformed_paths = [temp_paths[path] for path in paths]
        validate_transformed(
            transformed_paths,
            args.observation_source,
            args.action_source,
            source_indices,
            args.output_min,
            args.output_max,
        )
        new_stats = rebuild_stats(
            root, transformed_paths, old_stats, args.horizon, args.action_gap
        )
        stats_temp.write_text(
            json.dumps(new_stats, indent=4, ensure_ascii=False), encoding="utf-8"
        )

        for path in paths:
            os.replace(temp_paths[path], path)
        os.replace(stats_temp, stats_path)
        committed = True
    finally:
        if not committed:
            for temp in temp_paths.values():
                temp.unlink(missing_ok=True)
            stats_temp.unlink(missing_ok=True)

    print(f"updated {len(paths)} parquet file(s) and {stats_path.relative_to(root)}")
    print("done")


if __name__ == "__main__":
    main()
