#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any

from vtla.engine.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from vtla.engine.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from vtla.engine.utils.constants import ACTION, OBS_STATE

from ..sensor_routing import SensorRoutingMixin

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(SensorRoutingMixin, PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 32  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 16  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32
    # state_mode comes from SensorRoutingMixin (none | joint | ee).

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Deprecated compatibility fields. New configs use action_mode='relative_joint'.
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: dict[str, Any] | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    # wrist_only / top_camera_keys / wrist_camera_keys / tactile_keys / tactile_mode /
    # tactile_encoder_type come from SensorRoutingMixin.

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    # Source for the PaliGemma tokenizer. None -> use the HF hub name "google/paligemma-3b-pt-224"
    # (unchanged default). Only the EE rebuild path in train.py sets this to the local tokenizer dir
    # under pretrained_path so the rebuilt-from-scratch processor stays offline-usable.
    paligemma_tokenizer_path: str | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.action_start_offset < 0 or self.action_start_offset + self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"action_start_offset ({self.action_start_offset}) + n_action_steps ({self.n_action_steps}) "
                f"must be in [0, chunk_size={self.chunk_size}]"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        # Shared enum / reserved-mode validation (tactile_mode, state_mode, encoder).
        self.validate_sensor_modes()


    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}


        # Route cameras (wrist_only) + tactile (as_image) + state via the shared mixin.
        empty_keys = tuple(self.add_empty_cameras(self.empty_cameras, self.image_resolution))
        self.prune_unselected_visual_features(extra_keep=empty_keys)
        # PI05 pads state to max_state_dim and materialises it when missing (joint mode).
        self.apply_state_mode(padded_state_dim=self.max_state_dim)
        # Select joint vs EE action as the canonical `action` output feature.
        self.apply_action_mode()
        self.validate_routed_keys()

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        # relative_ee predicts the FUTURE trajectory relative to the current observation, so the
        # action chunk starts at t+1 (k=0 would be the current pose => identity, wasted). Joint mode
        # keeps the openpi convention of starting at the current frame (t).
        if self.action_reference == "relative":
            return list(range(1, self.chunk_size + 1))
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
