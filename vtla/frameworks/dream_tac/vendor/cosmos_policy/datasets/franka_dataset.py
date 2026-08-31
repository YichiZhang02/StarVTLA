# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Franka/tactile robot tasks dataloader (single-arm).

Loads preprocessed Franka data (cam_front + cam_high, optionally tactile_rectify_left + tactile_rectify_right).
qpos = state (6 or 7 dim). action = 7 dim (pose + gripper).
Maps to LIBERO-style: cam_front = 主视角 (primary), cam_high = 腕部 (wrist).
When tactile paths exist in HDF5, adds two tactile views; sequence becomes 12 slots (state_t=12).
Set ``use_tactile=False`` to keep the 8-slot LIBERO layout even if HDF5 lists tactile videos
(e.g. baseline training on tactile-preprocessed data).
Inherits from LIBERODataset for single-arm pipeline compatibility.
"""

import os
import pickle
from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm

from cosmos_policy.datasets.dataset_common import (
    build_rollout_step_index_mapping,
    calculate_epoch_structure,
    compute_monte_carlo_returns,
    load_or_compute_dataset_statistics,
    load_or_compute_post_normalization_statistics,
)
from cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    get_hdf5_files,
    rescale_data,
)
from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.datasets.aloha_dataset import load_video_as_images
from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.utils.utils import duplicate_array
from cosmos_policy.utils.tactile_self_attn_gate import mean_abs_diff_uint8_pair, scalar_gate_from_raw

import torch


class FrankaDataset(LIBERODataset):
    """
    Dataset for Franka/tactile preprocessed data (cam_front=主视角, cam_high=腕部).
    Optional: tactile_rectify_left, tactile_rectify_right (same processing as primary/wrist).
    When tactile is active: video has 12 slots (adds current/future tactile left & right); else 8 slots.

    Args:
        use_tactile: If ``None`` (default), load and use tactile when HDF5 contains tactile video paths.
            If ``False``, never load tactile videos and always use the 8-slot layout (for no-tactile baselines).
        use_tactile_image_aug: If ``False``, skip ``apply_image_aug`` on tactile blocks only (resize still applied).
    """

    def __init__(
        self,
        data_dir: str,
        is_train: bool = True,
        chunk_size: int = 8,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images=False,
        normalize_actions=True,
        normalize_proprio=True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        use_wrist_images: bool = True,
        use_third_person_images: bool = True,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 0.5,
        success_rollout_sampling_prob: float = 0.5,
        treat_success_rollouts_as_demos: bool = False,
        return_value_function_returns: bool = True,
        gamma: float = 0.99,
        use_tactile: Optional[bool] = None,
        use_tactile_image_aug: bool = True,
    ):
        # Same attributes as LIBERODataset
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.rollout_data_dir = rollout_data_dir
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.success_rollout_sampling_prob = success_rollout_sampling_prob
        self.treat_success_rollouts_as_demos = treat_success_rollouts_as_demos
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self._use_tactile_mode = use_tactile
        self.use_tactile_image_aug = use_tactile_image_aug

        assert self.use_wrist_images or self.use_third_person_images, (
            "Must use at least one of wrist images or third-person images!"
        )

        hdf5_files = get_hdf5_files(data_dir, is_train=is_train)
        if os.environ.get("DEBUGGING", "False").lower() == "true":
            hdf5_files = hdf5_files[:1]

        rollout_hdf5_files = []
        if self.rollout_data_dir and os.path.exists(self.rollout_data_dir):
            rollout_hdf5_files = get_hdf5_files(self.rollout_data_dir)

        self.data = {}
        self.rollout_episode_metadata = {}
        self.num_episodes = 0
        self.num_steps = 0
        self.rollout_num_episodes = 0
        self.rollout_num_steps = 0
        self.unique_commands = set()
        self._suite_to_step_indices = {}
        self.use_tactile = False

        def _read_path(ds):
            val = ds[()]
            if isinstance(val, bytes):
                return val.decode("utf-8")
            return str(val)

        for file in tqdm(hdf5_files, desc="Loading Franka episodes"):
            with h5py.File(file, "r") as f:
                obs_group = f["observations"]
                if "video_paths" not in obs_group or "cam_front" not in obs_group["video_paths"] or "cam_high" not in obs_group["video_paths"]:
                    raise ValueError(f"Franka HDF5 must have observations/video_paths/cam_front and cam_high: {file}")

                # Proprio and action
                proprio = np.array(f["observations/qpos"][:], dtype=np.float32)
                actions = np.array(f["action"][:], dtype=np.float32)

                file_dir = os.path.dirname(file)
                cam_front_path = os.path.join(file_dir, _read_path(obs_group["video_paths"]["cam_front"]))
                cam_high_path = os.path.join(file_dir, _read_path(obs_group["video_paths"]["cam_high"]))

                # cam_front = 主视角 (primary), cam_high = 腕部 (wrist)
                images = load_video_as_images(cam_front_path, resize_size=self.final_image_size)
                wrist_images = load_video_as_images(cam_high_path, resize_size=self.final_image_size)
                num_steps = min(len(images), len(wrist_images), proprio.shape[0], actions.shape[0])
                images = images[:num_steps]
                wrist_images = wrist_images[:num_steps]
                proprio = proprio[:num_steps]
                actions = actions[:num_steps]

                # Optional: tactile_rectify_left, tactile_rectify_right (same processing as cam_front/cam_high)
                tactile_left_images = None
                tactile_right_images = None
                vp = obs_group["video_paths"]
                if (
                    self._use_tactile_mode is not False
                    and "tactile_rectify_left" in vp
                    and "tactile_rectify_right" in vp
                ):
                    tl_path = os.path.join(file_dir, _read_path(vp["tactile_rectify_left"]))
                    tr_path = os.path.join(file_dir, _read_path(vp["tactile_rectify_right"]))
                    tl_imgs = load_video_as_images(tl_path, resize_size=self.final_image_size)
                    tr_imgs = load_video_as_images(tr_path, resize_size=self.final_image_size)
                    t_len = min(len(tl_imgs), len(tr_imgs), num_steps)
                    tactile_left_images = np.array(tl_imgs[:t_len], dtype=np.uint8)
                    tactile_right_images = np.array(tr_imgs[:t_len], dtype=np.uint8)
                    if t_len < num_steps:
                        pad_left = np.zeros((num_steps - t_len, self.final_image_size, self.final_image_size, 3), dtype=np.uint8)
                        tactile_left_images = np.concatenate([tactile_left_images, pad_left], axis=0)
                        tactile_right_images = np.concatenate([tactile_right_images, pad_left], axis=0)
                    self.use_tactile = True
                if tactile_left_images is None:
                    # Placeholder so every episode has same keys; not used when use_tactile is False
                    tactile_left_images = np.zeros((num_steps, self.final_image_size, self.final_image_size, 3), dtype=np.uint8)
                    tactile_right_images = np.zeros((num_steps, self.final_image_size, self.final_image_size, 3), dtype=np.uint8)

                command = f.attrs.get("task_name", "unknown")
                if isinstance(command, bytes):
                    command = command.decode("utf-8")
                self.unique_commands.add(command)

                returns = (
                    compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)
                    if self.return_value_function_returns
                    else None
                )

                self.data[self.num_episodes] = dict(
                    images=images,
                    wrist_images=wrist_images,
                    tactile_left_images=tactile_left_images,
                    tactile_right_images=tactile_right_images,
                    proprio=proprio,
                    actions=actions,
                    command=command,
                    num_steps=num_steps,
                    suite="franka",
                    returns=returns.copy() if self.return_value_function_returns else None,
                )
                self.num_episodes += 1
                self.num_steps += num_steps

        self._build_step_index_mapping()
        self.chunk_size = chunk_size

        if t5_text_embeddings_path != "":
            with open(t5_text_embeddings_path, "rb") as file:
                self.t5_text_embeddings = pickle.load(file)
        else:
            self.t5_text_embeddings = {}

        # Use Franka-specific stats file so we don't load 14-dim ALOHA stats; Franka uses 7-dim actions.
        self.dataset_stats = load_or_compute_dataset_statistics(
            data_dir=self.data_dir,
            data=self.data,
            calculate_dataset_statistics_func=calculate_dataset_statistics,
            stats_filename="dataset_statistics_franka.json",
        )

        if self.normalize_actions or self.normalize_proprio:
            if self.normalize_actions:
                self.data = rescale_data(self.data, self.dataset_stats, "actions")
            if self.normalize_proprio:
                self.data = rescale_data(self.data, self.dataset_stats, "proprio")
            self.dataset_stats_post_norm = load_or_compute_post_normalization_statistics(
                data_dir=self.data_dir,
                data=self.data,
                calculate_dataset_statistics_func=calculate_dataset_statistics,
                stats_filename="dataset_statistics_post_norm_franka.json",
            )

        if len(rollout_hdf5_files) > 0:
            # LIBERO-style rollout loading not implemented for Franka; keep empty
            pass

        self._build_rollout_step_index_mapping()
        self._calculate_epoch_structure()

    def __getitem__(self, idx):
        """Override to insert tactile_rectify_left/right into video when use_tactile (12 slots)."""
        sample = super().__getitem__(idx)
        if not self.use_tactile:
            return sample

        global_step_idx = idx % self.num_steps
        episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
        episode_data = self.data[episode_idx]
        num_steps = episode_data["num_steps"]
        future_frame_idx = min(relative_step_idx + self.chunk_size, num_steps - 1)

        # Build tactile frames: 4 slots (current left, current right, future left, future right), each 4 dup -> 16 frames
        def make_tactile_block(step_idx):
            tl = episode_data["tactile_left_images"][step_idx]  # (H, W, 3)
            tr = episode_data["tactile_right_images"][step_idx]
            # duplicate_array expects (H,W,C), returns (4,H,W,C) for num_duplicates_per_image=4
            tl4 = duplicate_array(tl, total_num_copies=self.num_duplicates_per_image)
            tr4 = duplicate_array(tr, total_num_copies=self.num_duplicates_per_image)
            block = np.concatenate([tl4, tr4], axis=0)  # (8, H, W, 3)
            return block

        tactile_cur = make_tactile_block(relative_step_idx)   # (8, H, W, 3)
        tactile_fut = make_tactile_block(future_frame_idx)   # (8, H, W, 3)
        tactile_use_aug = self.use_image_aug and self.use_tactile_image_aug
        tactile_cur = preprocess_image(
            tactile_cur,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=tactile_use_aug,
            stronger_image_aug=self.use_stronger_image_aug if tactile_use_aug else False,
        )  # (C, 8, H, W)
        tactile_fut = preprocess_image(
            tactile_fut,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=tactile_use_aug,
            stronger_image_aug=self.use_stronger_image_aug if tactile_use_aug else False,
        )  # (C, 8, H, W)

        # Parent video: (C, 29, H, W) = 1 (blank) + 7*4 (proprio, wrist, primary, action, future x3). Insert 8 frames after primary (13), 8 after future primary (29).
        video = sample["video"]  # (C, T, H, W)
        new_video = torch.cat(
            [
                video[:, :13],   # blank, proprio, wrist, primary
                tactile_cur,     # 8 frames (tactile_left 4, tactile_right 4)
                video[:, 13:29], # action, future proprio, wrist, primary
                tactile_fut,     # 8 frames (future tactile)
            ],
            dim=1,
        )  # (C, 45, H, W) = 1 + 11*4 for state_t=12
        sample["video"] = new_video

        # Shift latent indices that come after the two inserted blocks (each 2 slots = 8 frames)
        sample["action_latent_idx"] = sample["action_latent_idx"] + 2
        if sample.get("future_proprio_latent_idx", -1) >= 0:
            sample["future_proprio_latent_idx"] = sample["future_proprio_latent_idx"] + 2
        if sample.get("future_wrist_image_latent_idx", -1) >= 0:
            sample["future_wrist_image_latent_idx"] = sample["future_wrist_image_latent_idx"] + 2
        if sample.get("future_image_latent_idx", -1) >= 0:
            sample["future_image_latent_idx"] = sample["future_image_latent_idx"] + 2

        # Tactile latent indices for loss logging (slots 10 and 11 in the 12-slot sequence)
        sample["future_tactile_left_latent_idx"] = torch.tensor(10, dtype=torch.long)
        sample["future_tactile_right_latent_idx"] = torch.tensor(11, dtype=torch.long)

        # Per-step scalar gate for tactile self-attn bias (vs previous timestep in episode)
        tl = episode_data["tactile_left_images"]
        tr = episode_data["tactile_right_images"]
        if relative_step_idx > 0:
            raw = mean_abs_diff_uint8_pair(
                tl[relative_step_idx],
                tr[relative_step_idx],
                tl[relative_step_idx - 1],
                tr[relative_step_idx - 1],
            )
        else:
            raw = 0.0
        sample["tactile_self_attn_gate"] = torch.tensor(scalar_gate_from_raw(raw), dtype=torch.float32)

        return sample
