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
import logging
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                ep_length = end_index - start_index
                if drop_n_first_frames + drop_n_last_frames >= ep_length:
                    logger.warning(
                        "Episode %d has %d frames but drop_n_first_frames=%d and "
                        "drop_n_last_frames=%d removes all frames. Skipping.",
                        episode_idx,
                        ep_length,
                        drop_n_first_frames,
                        drop_n_last_frames,
                    )
                    continue
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        if not indices:
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


class MixtureSampler(Sampler[int]):
    """Sample a dataset by mixture weight, then a valid frame uniformly within it."""

    def __init__(
        self,
        dataset,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        if drop_n_first_frames < 0 or drop_n_last_frames < 0:
            raise ValueError("drop_n_first_frames and drop_n_last_frames must be non-negative.")
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.valid_indices = [
            self._valid_child_indices(child, drop_n_first_frames, drop_n_last_frames)
            for child in dataset._datasets
        ]
        empty = [dataset.repo_ids[i] for i, indices in enumerate(self.valid_indices) if not indices]
        if empty:
            raise ValueError(f"No valid frames remain for mixture members: {empty}")
        self.num_samples = num_samples or sum(len(indices) for indices in self.valid_indices)
        self.weights = torch.tensor(dataset.weights, dtype=torch.double)

    @staticmethod
    def _valid_child_indices(dataset, drop_first: int, drop_last: int) -> list[int]:
        if drop_first == 0 and drop_last == 0:
            return list(range(len(dataset)))
        selected_episodes = (
            sorted(dataset.episodes)
            if dataset.episodes is not None
            else list(range(dataset.meta.total_episodes))
        )
        indices = []
        local_start = 0
        for episode_idx in selected_episodes:
            episode = dataset.meta.episodes[episode_idx]
            episode_length = int(episode["dataset_to_index"] - episode["dataset_from_index"])
            if drop_first + drop_last >= episode_length:
                logger.warning(
                    "Dataset %s episode %d has no frames after dropping %d first and %d last frames.",
                    dataset.repo_id,
                    episode_idx,
                    drop_first,
                    drop_last,
                )
            else:
                indices.extend(range(local_start + drop_first, local_start + episode_length - drop_last))
            local_start += episode_length
        return indices

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        dataset_choices = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=generator
        ).tolist()
        offsets = self.dataset._offsets
        for dataset_idx in dataset_choices:
            candidates = self.valid_indices[dataset_idx]
            local_pos = torch.randint(len(candidates), (1,), generator=generator).item()
            yield offsets[dataset_idx] + candidates[local_pos]

    def __len__(self) -> int:
        return self.num_samples
