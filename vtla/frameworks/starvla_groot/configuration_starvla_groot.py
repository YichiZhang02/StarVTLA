#!/usr/bin/env python

# Copyright 2025 starVLA community & The HuggingFace Inc. team. All rights reserved.
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
"""Config for the QwenGR00T framework, ported from starVLA into vtla.

A Qwen-VL backbone (prefix encoder) combined with the GR00T flow-matching DiT
action head. Registered under the policy type ``"starvla_groot"``.
"""

from dataclasses import dataclass, field

from vtla.engine.configs import NormalizationMode, PreTrainedConfig
from vtla.engine.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from vtla.engine.utils.constants import ACTION

from ..sensor_routing import SensorRoutingMixin

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("starvla_groot")
@dataclass
class StarvlaGrootConfig(SensorRoutingMixin, PreTrainedConfig):
    # === VLM backbone (Qwen2.5-VL / Qwen3-VL / Qwen3.5) ===
    base_vlm: str = "./playground/pretrained_models/Qwen3.5-0.8B"
    attn_implementation: str = "sdpa"  # "flash_attention_2" | "eager" | "sdpa"
    dtype: str = "bfloat16"  # backbone load dtype: "bfloat16" | "float32"

    n_obs_steps: int = 1
    chunk_size: int = 32  # action_horizon: number of action steps predicted by the head
    n_action_steps: int = 16  # number of action steps executed before re-planning

    # === Action head (GR00T flow-matching / DiT) ===
    action_model_type: str = "DiT-B"  # "DiT-B" | "DiT-L"
    action_head_hidden_size: int = 1024  # MLP width for state_encoder / action_decoder
    num_inference_timesteps: int = 4  # denoising Euler steps at inference
    repeated_diffusion_steps: int = 8  # inference-time ensemble size: number of noise
    # initializations denoised in parallel at inference and averaged into one action chunk.
    # Training forwards a single noise sample per element (use a larger batch_size to reduce
    # gradient variance instead of replicating the action head, which avoids the memory blow-up).
    num_target_vision_tokens: int = 32  # learnable planning query tokens
    add_pos_embed: bool = True
    max_seq_len: int = 1024

    # Flow-matching noise schedule (Beta distribution)
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # DiT transformer sub-config (cross_attention_dim is aligned to the VLM hidden at runtime)
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": 16,
            "output_dim": 1024,
            "positional_embeddings": None,
        }
    )

    # Optional action/state dims. When None they are inferred from dataset features.
    action_dim: int | None = None
    state_dim: int | None = None

    # === Multi-view / tactile / state routing ===
    # wrist_only / top_camera_keys / wrist_camera_keys / tactile_mode /
    # tactile_keys / tactile_encoder_type / state_mode come from SensorRoutingMixin.
    empty_cameras: int = 0
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,  # images handled by the Qwen processor
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False
    device: str | None = None

    # Finetuning settings
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False  # freeze the whole VLM, train only the action head

    # Optimizer / scheduler settings
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.action_start_offset < 0 or self.action_start_offset + self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"action_start_offset ({self.action_start_offset}) + n_action_steps ({self.n_action_steps}) "
                f"must be in [0, chunk_size={self.chunk_size}]"
            )
        if self.action_model_type not in ["DiT-B", "DiT-L"]:
            raise ValueError(f"Invalid action_model_type: {self.action_model_type}")
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        # Shared enum / reserved-mode validation (tactile_mode, state_mode, encoder).
        self.validate_sensor_modes()

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}

        empty_keys = tuple(self.add_empty_cameras(self.empty_cameras, self.image_resolution))
        self.prune_unselected_visual_features(extra_keep=empty_keys)
        self.apply_state_mode()  # real state dim is read from features by the policy
        self.apply_action_mode()
        self.validate_routed_keys()

        if ACTION not in self.output_features:
            raise ValueError("StarvlaGroot requires an 'action' output feature.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
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
        return list(range(self.action_gap, self.action_gap + self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
