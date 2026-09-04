"""Map-style virtual dataset mixtures backed by existing LeRobot datasets."""

from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .feature_schema import mixture_feature_schema_diff
from .lerobot_dataset import LeRobotDataset
from .mixture_registry import MixtureDefinition
from .visual_preprocess import validate_visual_preprocess

logger = logging.getLogger(__name__)


def validate_mixture_metadata(datasets: list[LeRobotDataset]) -> None:
    if not datasets:
        raise ValueError("A dataset mixture must contain at least one dataset.")
    reference = datasets[0].meta
    reference_visual_preprocess = getattr(reference, "visual_preprocess", None)
    if not reference.robot_type:
        raise ValueError(f"Mixture reference dataset {datasets[0].repo_id!r} has no robot_type")
    if reference_visual_preprocess is None:
        raise ValueError(
            f"Mixture reference dataset {datasets[0].repo_id!r} has no visual_preprocess contract"
        )
    validate_visual_preprocess(reference_visual_preprocess)
    errors = []
    for dataset in datasets[1:]:
        meta = dataset.meta
        if meta.fps != reference.fps:
            errors.append(f"{dataset.repo_id}: fps expected {reference.fps}, got {meta.fps}")
        if meta.robot_type != reference.robot_type:
            errors.append(
                f"{dataset.repo_id}: robot_type expected {reference.robot_type!r}, got {meta.robot_type!r}"
            )
        visual_preprocess = getattr(meta, "visual_preprocess", None)
        try:
            validate_visual_preprocess(visual_preprocess)
        except ValueError as exc:
            errors.append(f"{dataset.repo_id}: {exc}")
        if visual_preprocess != reference_visual_preprocess:
            errors.append(
                f"{dataset.repo_id}: visual_preprocess expected "
                f"{reference_visual_preprocess!r}, got {visual_preprocess!r}"
            )
        errors.extend(
            f"{dataset.repo_id}: {difference}"
            for difference in mixture_feature_schema_diff(reference.features, meta.features)
        )
    if errors:
        details = "\n  - ".join(errors)
        raise ValueError(
            "Dataset mixture members must have identical FPS, robot_type, visual preprocessing, "
            "and feature schemas:\n"
            f"  - {details}"
        )


def aggregate_weighted_stats(
    stats_list: list[dict[str, dict[str, np.ndarray]] | None],
    weights: tuple[float, ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate dataset-level statistics according to mixture sampling weights."""
    data_keys = {key for stats in stats_list if stats is not None for key in stats}
    aggregated: dict[str, dict[str, np.ndarray]] = {}
    for feature_key in data_keys:
        present = [
            (stats[feature_key], weight)
            for stats, weight in zip(stats_list, weights, strict=True)
            if stats and feature_key in stats
        ]
        if len(present) != len(stats_list):
            logger.warning(
                "Mixture statistic %s is missing from %d member(s); aggregating the available subset.",
                feature_key,
                len(stats_list) - len(present),
            )
        feature_weights = np.asarray([weight for _, weight in present], dtype=np.float64)
        feature_weights /= feature_weights.sum()
        feature_stats = [stats for stats, _ in present]
        common_keys = set.intersection(*(set(stats) for stats in feature_stats))
        if not {"mean", "std"}.issubset(common_keys):
            logger.warning("Skipping incomplete mixture statistics for feature %s", feature_key)
            continue

        means = np.stack([np.asarray(stats["mean"]) for stats in feature_stats])
        variances = np.stack([np.asarray(stats["std"]) ** 2 for stats in feature_stats])
        broadcast_weights = feature_weights.reshape((len(feature_weights),) + (1,) * (means.ndim - 1))
        mean = (means * broadcast_weights).sum(axis=0)
        variance = ((variances + (means - mean) ** 2) * broadcast_weights).sum(axis=0)
        result = {"mean": mean, "std": np.sqrt(variance)}
        if "min" in common_keys:
            result["min"] = np.min(np.stack([np.asarray(stats["min"]) for stats in feature_stats]), axis=0)
        if "max" in common_keys:
            result["max"] = np.max(np.stack([np.asarray(stats["max"]) for stats in feature_stats]), axis=0)
        if "count" in common_keys:
            result["count"] = np.sum(
                np.stack([np.asarray(stats["count"]) for stats in feature_stats]), axis=0
            )
        for stat_key in sorted(key for key in common_keys if key.startswith("q") and key[1:].isdigit()):
            values = np.stack([np.asarray(stats[stat_key]) for stats in feature_stats])
            value_weights = feature_weights.reshape((len(feature_weights),) + (1,) * (values.ndim - 1))
            result[stat_key] = (values * value_weights).sum(axis=0)
        aggregated[feature_key] = result
    return aggregated


@dataclass
class MixtureMetadata:
    features: dict[str, dict]
    stats: dict[str, dict[str, np.ndarray]]
    tasks: pd.DataFrame
    fps: int
    robot_type: str | None
    visual_preprocess: dict | None

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature["dtype"] in ("video", "image")]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature["dtype"] == "video"]


class MixtureLeRobotDataset(torch.utils.data.Dataset):
    """A weighted virtual mixture with concatenated map-style index space."""

    def __init__(self, datasets: list[LeRobotDataset], definition: MixtureDefinition):
        super().__init__()
        if len(datasets) != len(definition.members):
            raise ValueError("Mixture definition and loaded dataset counts do not match.")
        validate_mixture_metadata(datasets)
        self.repo_id = definition.dataset_id
        self.repo_ids = [member.dataset_id for member in definition.members]
        self.definition = definition
        self._datasets = datasets
        self.weights = definition.normalized_weights
        self.roots = [Path(dataset.root) for dataset in datasets]
        self.root = None
        self.episodes = None

        self._offsets = []
        running = 0
        for dataset in datasets:
            self._offsets.append(running)
            running += len(dataset)
        self._num_frames = running

        task_names = []
        for dataset in datasets:
            task_names.extend(str(task) for task in dataset.meta.tasks.index)
        unique_tasks = list(dict.fromkeys(task_names))
        tasks = pd.DataFrame(index=pd.Index(unique_tasks, name="task"))
        reference = datasets[0].meta
        self.meta = MixtureMetadata(
            features=reference.features,
            stats=aggregate_weighted_stats([dataset.meta.stats for dataset in datasets], self.weights),
            tasks=tasks,
            fps=reference.fps,
            robot_type=reference.robot_type,
            visual_preprocess=reference.visual_preprocess,
        )

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_episodes(self) -> int:
        return sum(dataset.num_episodes for dataset in self._datasets)

    @property
    def text_embedding_cache_dirs(self) -> list[Path]:
        return [root / "text_embeddings" / "wan22" for root in self.roots]

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for mixture of length {len(self)}.")
        dataset_idx = bisect_right(self._offsets, idx) - 1
        item = self._datasets[dataset_idx][idx - self._offsets[dataset_idx]]
        item["dataset_index"] = torch.tensor(dataset_idx, dtype=torch.long)
        return item

    def __repr__(self) -> str:
        members = ", ".join(
            f"{repo_id}={weight:.4f}" for repo_id, weight in zip(self.repo_ids, self.weights, strict=True)
        )
        return (
            f"{self.__class__.__name__}(id={self.repo_id!r}, members=[{members}], "
            f"frames={self.num_frames}, episodes={self.num_episodes})"
        )
