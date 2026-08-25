#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import hashlib
import logging
from pathlib import Path
from pprint import pformat

import torch

from vtla.engine.configs import PreTrainedConfig
from vtla.engine.configs.train import TrainPipelineConfig
from vtla.engine.transforms import ImageTransforms
from vtla.engine.utils.constants import ACTION, IMAGENET_STATS, OBS_PREFIX, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .mixture_registry import resolve_member_root, resolve_mixture
from .multi_dataset import MixtureLeRobotDataset


def _metadata_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    metadata_paths = [root / "meta" / "info.json", root / "meta" / "stats.json", root / "meta" / "tasks.parquet"]
    metadata_paths.extend(sorted((root / "meta" / "episodes").glob("**/*.parquet")))
    for path in metadata_paths:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def resolve_delta_timestamps(cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    # Tactile keys get their OWN temporal window (tactile_num_frames / tactile_frame_offset),
    # independent of the shared observation window. Resolved once here so the branch below can
    # override observation.* tactile keys with tactile-specific delta indices.
    tactile_delta = None
    tactile_keys: set[str] = set()
    if getattr(cfg, "tactile_windowed", None) is not None and cfg.tactile_windowed():
        tactile_keys = set(cfg.tactile_windowed_keys())
        tactile_delta = [i / ds_meta.fps for i in cfg.tactile_delta_indices()]

    windowed_observation_keys = None
    if hasattr(cfg, "windowed_observation_keys"):
        windowed_observation_keys = set(cfg.windowed_observation_keys())
    windowed_action_keys = None
    if hasattr(cfg, "windowed_action_keys"):
        windowed_action_keys = set(cfg.windowed_action_keys())

    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        # All EE action columns must be chunked over the same horizon as `action`.
        # Covers: action_episode_ee, action_absolute_ee (rot6d) and
        #         action_episode_quat, action_absolute_quat (quat).
        _ee_action_keys = {
            ACTION + "_episode_ee", ACTION + "_absolute_ee",
            ACTION + "_episode_quat", ACTION + "_absolute_quat",
        }
        if (
            key in ({ACTION} | _ee_action_keys)
            and cfg.action_delta_indices is not None
            and (windowed_action_keys is None or key in windowed_action_keys)
        ):
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX):
            # Tactile keys override the shared observation window with their own tactile window.
            if key in tactile_keys and tactile_delta is not None:
                delta_timestamps[key] = list(tactile_delta)
            elif cfg.observation_delta_indices is not None and (
                windowed_observation_keys is None or key in windowed_observation_keys
            ):
                delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MixtureLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Returns:
        LeRobotDataset | MixtureLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    mixture = resolve_mixture(
        cfg.dataset.repo_id,
        registry_path=cfg.dataset.mixture_config,
        resolved=cfg.dataset.resolved_mixture,
    )

    if mixture is None:
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta)
        # Decode only the cameras the policy actually consumes (skips e.g. finger cams when
        # tactile_mode='none'); avoids wasting data-loader time on unused video streams.
        use_video_keys = None
        if hasattr(cfg.trainable_config, "decoded_video_keys"):
            keys = cfg.trainable_config.decoded_video_keys()
            if keys:
                use_video_keys = keys
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                tolerance_s=cfg.tolerance_s,
                use_video_keys=use_video_keys,
            )
        else:
            from .streaming_dataset import StreamingLeRobotDataset

            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
            )
    else:
        if cfg.dataset.streaming:
            raise ValueError("Named dataset mixtures do not support dataset.streaming=true.")
        if cfg.dataset.episodes is not None:
            raise ValueError(
                "Use per-member episodes in the mixture registry instead of dataset.episodes for a mixture."
            )
        collision_paths = []
        if cfg.dataset.root is not None and Path(cfg.dataset.root).is_dir():
            collision_paths.append(Path(cfg.dataset.root))
        if cfg.dataset.catalog_root is not None:
            catalog_candidate = Path(cfg.dataset.catalog_root) / cfg.dataset.repo_id
            if catalog_candidate.is_dir() and catalog_candidate not in collision_paths:
                collision_paths.append(catalog_candidate)
        if collision_paths:
            raise ValueError(
                f"Dataset ID {cfg.dataset.repo_id!r} is both a named mixture and an existing dataset "
                f"directory: {collision_paths}. Rename one of them."
            )

        member_datasets = []
        for member in mixture.members:
            member_root = resolve_member_root(mixture, member, cfg.dataset.catalog_root)
            member_revision = member.revision or cfg.dataset.revision
            member_meta = LeRobotDatasetMetadata(
                member.dataset_id,
                root=member_root,
                revision=member_revision,
            )
            if member.episodes is not None and any(
                episode >= member_meta.total_episodes for episode in member.episodes
            ):
                raise ValueError(
                    f"Mixture {mixture.dataset_id!r} member {member.dataset_id!r} selects an episode "
                    f"outside [0, {member_meta.total_episodes})."
                )
            delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, member_meta)
            use_video_keys = None
            if hasattr(cfg.trainable_config, "decoded_video_keys"):
                keys = cfg.trainable_config.decoded_video_keys()
                if keys:
                    use_video_keys = keys
            member_datasets.append(
                LeRobotDataset(
                    member.dataset_id,
                    root=member_root,
                    episodes=member.episodes,
                    delta_timestamps=delta_timestamps,
                    image_transforms=image_transforms,
                    revision=member_revision,
                    video_backend=cfg.dataset.video_backend,
                    return_uint8=True,
                    tolerance_s=cfg.tolerance_s,
                    use_video_keys=use_video_keys,
                )
            )
        dataset = MixtureLeRobotDataset(member_datasets, mixture)
        resolved_mixture = mixture.to_dict()
        resolved_mixture["metadata"] = [
            {
                "dataset_id": child.repo_id,
                "fingerprint": _metadata_fingerprint(Path(child.root)),
                "num_frames": child.num_frames,
                "num_episodes": child.num_episodes,
                "root": str(child.root),
            }
            for child in member_datasets
        ]
        saved_metadata = (cfg.dataset.resolved_mixture or {}).get("metadata")
        if saved_metadata is not None:
            saved_fingerprints = {
                item["dataset_id"]: item["fingerprint"] for item in saved_metadata
            }
            current_fingerprints = {
                item["dataset_id"]: item["fingerprint"] for item in resolved_mixture["metadata"]
            }
            if saved_fingerprints != current_fingerprints:
                raise ValueError(
                    f"Dataset metadata changed since mixture {mixture.dataset_id!r} was resolved. "
                    f"Saved fingerprints: {saved_fingerprints}; current: {current_fingerprints}."
                )
        cfg.dataset.resolved_mixture = resolved_mixture
        logging.info("Resolved dataset mixture: %s", dataset)

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            # Some converted tactile/video streams intentionally have no pixel statistics in
            # stats.json. ImageNet normalization is authoritative for every visual input, so
            # materialize the feature entry instead of assuming the offline stats file has one.
            dataset.meta.stats.setdefault(key, {})
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
