from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vtla.engine.configs import FeatureType, NormalizationMode, PreTrainedConfig
from vtla.engine.optim import AdamWConfig, DiffuserSchedulerConfig
from vtla.engine.utils.constants import ACTION, OBS_STATE

from ..sensor_routing import (
    ACTION_ABSOLUTE_EE,
    ACTION_ABSOLUTE_QUAT,
    ACTION_EPISODE_EE,
    ACTION_EPISODE_QUAT,
    OBS_STATE_ABSOLUTE_EE,
    OBS_STATE_ABSOLUTE_QUAT,
    OBS_STATE_EPISODE_EE,
    OBS_STATE_EPISODE_JOINT,
    OBS_STATE_EPISODE_QUAT,
    SensorRoutingMixin,
)


def _default_video_dit_config() -> dict[str, Any]:
    return {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1e-6,
        "seperated_timestep": True,
        "require_clip_embedding": False,
        "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": True,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": 16,
        "action_group_causal_mask_mode": "group_diagonal",
    }


def _default_action_dit_config() -> dict[str, Any]:
    return {
        "action_dim": 16,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1e-6,
        "use_gradient_checkpointing": True,
    }


@PreTrainedConfig.register_subclass("fastwam")
@dataclass
class FastWAMConfig(SensorRoutingMixin, PreTrainedConfig):
    """VTLA configuration for the base Wan2.2 FastWAM policy."""

    n_obs_steps: int = 33
    chunk_size: int = 32
    n_action_steps: int = 16
    video_frame_stride: int = 4
    # Each camera is resized independently, then cameras are concatenated horizontally.
    camera_image_size: tuple[int, int] = (224, 224)
    # Derived from camera_image_size and camera count. Kept for old checkpoint compatibility.
    video_size: tuple[int, int] | None = None
    # Legacy checkpoints store the resolved RGB keys here. New runs derive them from
    # SensorRoutingMixin's wrist/top routing fields during feature validation.
    camera_keys: list[str] = field(default_factory=list)
    num_cameras: int = 0

    context_len: int = 128
    text_dim: int = 4096
    prompt_template: str = "A video recorded from a robot's point of view executing the following instruction: {task}"
    text_embedding_cache_dir: Path | None = None
    # Inference loads T5 for arbitrary prompts; vtla.train always forces this off.
    load_text_encoder: bool = True

    model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    video_dit_pretrained_path: Path | None = Path(
        "playground/pretrained_models/Wan2.2-TI2V-5B"
    )
    vae_pretrained_path: Path | None = Path(
        "playground/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    )
    text_encoder_pretrained_path: Path | None = Path(
        "playground/pretrained_models/Wan2.2-TI2V-5B/"
        "models_t5_umt5-xxl-enc-bf16.pth"
    )
    tokenizer_pretrained_path: Path | None = Path(
        "playground/pretrained_models/Wan2.2-TI2V-5B/google/umt5-xxl"
    )
    action_dit_pretrained_path: Path | None = Path(
        "playground/pretrained_models/Wan2.2-TI2V-5B/interpolated_dit/"
        "InterpolatedDiT_from_official_Wan2.2_alphascale_1024hdim.pt"
    )
    skip_dit_load_from_pretrain: bool = False

    dtype: str = "bfloat16"
    mot_checkpoint_mixed_attn: bool = True
    video_dit_config: dict[str, Any] = field(default_factory=_default_video_dit_config)
    action_dit_config: dict[str, Any] = field(default_factory=_default_action_dit_config)

    video_train_shift: float = 5.0
    video_infer_shift: float = 5.0
    video_num_train_timesteps: int = 1000
    action_train_shift: float = 5.0
    action_infer_shift: float = 5.0
    action_num_train_timesteps: int = 1000
    num_inference_steps: int = 10
    loss_lambda_video: float = 1.0
    loss_lambda_action: float = 1.0

    # FastWAM-only training visualization. train.sh exposes only `enabled`;
    # the remaining values are stable policy defaults saved in the config.
    visualization_enabled: bool = False
    visualization_freq: int = 1000
    visualization_num_samples: int = 4
    visualization_num_inference_steps: int = 10
    visualization_seed: int = 42
    visualization_include_vae_reconstruction: bool = True

    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_obs_steps < 2 or (self.n_obs_steps - 1) % self.video_frame_stride != 0:
            raise ValueError("FastWAM requires `(n_obs_steps - 1)` to be divisible by video_frame_stride.")
        if self.chunk_size % ((self.n_obs_steps - 1) // self.video_frame_stride) != 0:
            raise ValueError("FastWAM action horizon must be divisible by the number of video transitions.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size.")
        if self.action_start_offset < 0 or self.action_start_offset + self.n_action_steps > self.chunk_size:
            raise ValueError("action_start_offset + n_action_steps must fit inside chunk_size.")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("FastWAM dtype must be 'bfloat16' or 'float32'.")
        self.validate_sensor_modes()
        if self.tactile_mode == "as_image" and self.tactile_insert_location != "encoder":
            raise ValueError(
                "FastWAM tactile_mode='as_image' uses Wan VAE history tokens and therefore "
                "requires tactile_insert_location='encoder'."
            )
        self.camera_image_size = tuple(int(size) for size in self.camera_image_size)
        if len(self.camera_image_size) != 2 or any(size <= 0 for size in self.camera_image_size):
            raise ValueError("FastWAM camera_image_size must contain two positive dimensions.")
        if self.camera_image_size[0] % 16 or self.camera_image_size[1] % 16:
            raise ValueError("FastWAM per-camera image dimensions must be multiples of 16.")
        expected_frames = (self.n_obs_steps - 1) // self.video_frame_stride + 1
        if expected_frames % 4 != 1:
            raise ValueError("Subsampled FastWAM video length must satisfy T % 4 == 1.")
        if self.visualization_freq <= 0:
            raise ValueError("FastWAM visualization_freq must be positive.")
        if self.visualization_num_samples <= 0:
            raise ValueError("FastWAM visualization_num_samples must be positive.")
        if self.visualization_num_inference_steps <= 0:
            raise ValueError("FastWAM visualization_num_inference_steps must be positive.")

    def validate_features(self) -> None:
        self.input_features = self.input_features or {}
        self.output_features = self.output_features or {}
        self.camera_keys = self.resolved_rgb_camera_keys()
        if not self.camera_keys:
            raise ValueError("FastWAM requires at least one camera.")
        for key in self.camera_keys:
            self.require_visual_feature(key, "camera")
        if self.tactile_mode in ("as_image", "encode"):
            if not self.tactile_keys:
                raise ValueError(f"FastWAM tactile_mode={self.tactile_mode!r} requires tactile_keys.")
            for key in self.tactile_keys:
                self.require_visual_feature(key, "tactile")

        keep_visual = set(self.camera_keys) | set(self.tactile_windowed_keys())
        for key in list(self.input_features):
            feature = self.input_features[key]
            if feature.type is FeatureType.VISUAL and key not in keep_visual:
                self.input_features.pop(key)
        self.apply_state_mode()
        self.apply_action_mode()

        self.num_cameras = len(self.camera_keys)
        self.video_size = (
            self.camera_image_size[0],
            self.camera_image_size[1] * self.num_cameras,
        )
        if self.state_mode != "none" and self.robot_state_feature is None:
            raise ValueError(f"FastWAM requires {OBS_STATE}.")
        if self.action_feature is None:
            raise ValueError(f"FastWAM requires {ACTION}.")
        action_dim = int(self.action_feature.shape[0])
        self.input_features = {
            key: feature
            for key, feature in self.input_features.items()
            if key in {*self.camera_keys, *self.tactile_windowed_keys(), OBS_STATE}
        }
        self.output_features = {ACTION: self.action_feature}
        self.video_dit_config = {**self.video_dit_config, "action_dim": action_dim}
        self.action_dit_config = {**self.action_dit_config, "action_dim": action_dim}

    def resolved_rgb_camera_keys(self) -> list[str]:
        if self.camera_keys:
            return self._dedupe(list(self.camera_keys))
        return self.selected_camera_keys()

    def decoded_video_keys(self) -> list[str]:
        return self._dedupe(self.resolved_rgb_camera_keys() + self.tactile_windowed_keys())

    def windowed_observation_keys(self) -> list[str]:
        state_key = {
            "absolute_joint": OBS_STATE,
            "episode_joint": OBS_STATE_EPISODE_JOINT,
            "episode_rot6d": OBS_STATE_EPISODE_EE,
            "absolute_rot6d": OBS_STATE_ABSOLUTE_EE,
            "episode_quat": OBS_STATE_EPISODE_QUAT,
            "absolute_quat": OBS_STATE_ABSOLUTE_QUAT,
        }.get(self.state_mode)
        keys = self.resolved_rgb_camera_keys()
        if state_key is not None:
            keys.append(state_key)
        return self._dedupe(keys)

    def windowed_action_keys(self) -> list[str]:
        if self.action_representation == "joint":
            return [ACTION]
        if self.action_representation == "rot6d":
            return [ACTION_ABSOLUTE_EE]
        return [ACTION_ABSOLUTE_QUAT]

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(self.n_obs_steps))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_gap, self.action_gap + self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )
