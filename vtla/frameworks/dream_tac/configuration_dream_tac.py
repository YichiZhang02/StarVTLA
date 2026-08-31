from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vtla.engine.configs import FeatureType, NormalizationMode, PreTrainedConfig
from vtla.engine.optim import AdamWConfig, DiffuserSchedulerConfig
from vtla.engine.utils.constants import ACTION, OBS_STATE
from vtla.frameworks.sensor_routing import (
    ACTION_ABSOLUTE_EE,
    ACTION_ABSOLUTE_QUAT,
    OBS_STATE_ABSOLUTE_EE,
    OBS_STATE_ABSOLUTE_QUAT,
    OBS_STATE_EPISODE_EE,
    OBS_STATE_EPISODE_JOINT,
    OBS_STATE_EPISODE_QUAT,
    SensorRoutingMixin,
)

from .slot_layout import DreamTacSlotLayout, compile_slot_layout


@PreTrainedConfig.register_subclass("dream_tac")
@dataclass
class DreamTacConfig(SensorRoutingMixin, PreTrainedConfig):
    """StarVTLA adapter config for the Dream-Tac Cosmos policy."""

    chunk_size: int = 20
    n_action_steps: int = 8
    future_prediction_offset: int | None = None
    image_size: int = 224
    temporal_compression_factor: int = 4

    context_len: int = 512
    text_dim: int = 1024
    prompt_template: str = "{task}"
    text_embedding_cache_dir: Path | None = None
    text_embedding_cache_dirs: list[Path] = field(default_factory=list)

    pretrained_path: Path | None = Path(
        "playground/pretrained_models/Cosmos-Predict2-2B-Video2World"
    )
    slot_layout_version: int = 1
    slot_layout_fingerprint: str = ""
    dtype: str = "bfloat16"
    num_inference_steps: int = 5
    inference_seed: int = 1

    action_loss_weight: float = 2.0
    rgb_loss_weight: float = 1.0
    tactile_loss_weight: float = 1.0
    state_loss_weight: float = 0.25

    visualization_enabled: bool = False
    visualization_freq: int = 1000
    visualization_num_samples: int = 4
    visualization_num_inference_steps: int = 5
    visualization_seed: int = 42
    visualization_include_vae_reconstruction: bool = True

    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 2000

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pretrained_path is None or not str(self.pretrained_path).strip():
            raise ValueError("Dream-Tac pretrained_path must name Cosmos or Dream-Tac weights.")
        if self.tactile_mode == "encode":
            raise ValueError("Dream-Tac does not support tactile_mode='encode'; use 'none' or 'as_image'.")
        self.validate_sensor_modes()
        if self.future_prediction_offset is None:
            self.future_prediction_offset = self.action_gap + self.chunk_size
        if self.chunk_size != 20:
            raise ValueError("Dream-Tac preserves the original fixed chunk_size=20 behavior.")
        if self.action_gap < 0:
            raise ValueError("Dream-Tac action_gap must be non-negative.")
        if self.future_prediction_offset != self.action_gap + self.chunk_size:
            raise ValueError(
                "Dream-Tac future_prediction_offset must equal action_gap + chunk_size."
            )
        if self.n_action_steps <= 0 or self.action_start_offset + self.n_action_steps > self.chunk_size:
            raise ValueError("Dream-Tac action execution window must fit inside chunk_size.")
        if self.image_size <= 0 or self.image_size % 8:
            raise ValueError("Dream-Tac image_size must be a positive multiple of 8.")
        if self.temporal_compression_factor != 4:
            raise ValueError("Dream-Tac requires temporal_compression_factor=4.")
        if self.tactile_mode == "as_image" and (
            self.tactile_num_frames != 1 or self.tactile_frame_offset != 1
        ):
            raise ValueError(
                "Dream-Tac owns its [previous, current, future] tactile sampling; "
                "tactile_num_frames and tactile_frame_offset must remain 1."
            )
        if self.state_representation not in {"none", "joint", "rot6d", "quat"}:
            raise ValueError("Dream-Tac requires no state or a joint, rot6d, or quat state representation.")
        if self.action_representation not in {"joint", "rot6d", "quat"}:
            raise ValueError("Dream-Tac requires a joint, rot6d, or quat action representation.")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("Dream-Tac dtype must be 'bfloat16' or 'float32'.")
        if any(weight < 0 for weight in self.loss_weights().values()):
            raise ValueError("Dream-Tac loss weights must be non-negative.")
        if not any(self.loss_weights().values()):
            raise ValueError("At least one Dream-Tac loss weight must be positive.")
        if self.visualization_freq <= 0 or self.visualization_num_samples <= 0:
            raise ValueError("Dream-Tac visualization frequency and sample count must be positive.")
        layout = self.slot_layout()
        if self.slot_layout_fingerprint and self.slot_layout_fingerprint != layout.fingerprint:
            raise ValueError(
                "Dream-Tac slot layout differs from the checkpoint contract: "
                f"saved={self.slot_layout_fingerprint}, resolved={layout.fingerprint}."
            )
        self.slot_layout_fingerprint = layout.fingerprint

    def validate_checkpoint_layout(self, checkpoint_config: "DreamTacConfig") -> None:
        """Reject sensor-layout changes when fine-tuning an existing checkpoint."""
        saved = checkpoint_config.slot_layout_fingerprint
        resolved = self.slot_layout().fingerprint
        if saved != resolved:
            raise ValueError(
                "Dream-Tac slot layout differs from the pretrained checkpoint: "
                f"saved={saved}, resolved={resolved}. Start a new training run from Cosmos pretrained_path "
                "to use a different sensor layout."
            )

    def loss_weights(self) -> dict[str, float]:
        return {
            "action": float(self.action_loss_weight),
            "rgb": float(self.rgb_loss_weight),
            "tactile": float(self.tactile_loss_weight),
            "state": float(self.state_loss_weight),
        }

    def slot_layout(self) -> DreamTacSlotLayout:
        return compile_slot_layout(
            wrist_only=self.wrist_only,
            wrist_camera_keys=list(self.wrist_camera_keys),
            top_camera_keys=list(self.top_camera_keys),
            tactile_mode=self.tactile_mode,
            tactile_keys=list(self.tactile_keys),
            state_mode=self.state_mode,
            temporal_compression_factor=self.temporal_compression_factor,
            version=self.slot_layout_version,
        )

    def resolved_rgb_keys(self) -> list[str]:
        return list(self.slot_layout().rgb_keys)

    def resolved_tactile_keys(self) -> list[str]:
        return list(self.slot_layout().tactile_keys)

    def decoded_video_keys(self) -> list[str]:
        layout = self.slot_layout()
        return [*layout.rgb_keys, *layout.tactile_keys]

    def tactile_windowed(self) -> bool:
        return self.tactile_mode == "as_image"

    def tactile_windowed_keys(self) -> list[str]:
        return self.resolved_tactile_keys()

    def tactile_delta_indices(self) -> list[int]:
        if self.tactile_mode != "as_image":
            return []
        return [-1, 0, int(self.future_prediction_offset)]

    def windowed_observation_keys(self) -> list[str]:
        state_key = {
            "absolute_joint": OBS_STATE,
            "episode_joint": OBS_STATE_EPISODE_JOINT,
            "episode_rot6d": OBS_STATE_EPISODE_EE,
            "absolute_rot6d": OBS_STATE_ABSOLUTE_EE,
            "episode_quat": OBS_STATE_EPISODE_QUAT,
            "absolute_quat": OBS_STATE_ABSOLUTE_QUAT,
        }.get(self.state_mode)
        keys = self.resolved_rgb_keys()
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
        return [0, int(self.future_prediction_offset)]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_gap, self.action_gap + self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        self.input_features = self.input_features or {}
        self.output_features = self.output_features or {}
        required_visuals = self.decoded_video_keys()
        for key in required_visuals:
            self.require_visual_feature(key, "Dream-Tac input")
        for key in list(self.input_features):
            feature = self.input_features[key]
            if feature.type is FeatureType.VISUAL and key not in required_visuals:
                self.input_features.pop(key)
        self.apply_state_mode()
        self.apply_action_mode()
        if self.state_mode != "none" and self.robot_state_feature is None:
            raise ValueError(f"Dream-Tac state_mode={self.state_mode!r} requires proprioception.")
        if self.action_feature is None:
            raise ValueError(f"Dream-Tac requires {ACTION}.")
        self.input_features = {
            key: feature
            for key, feature in self.input_features.items()
            if key in {*required_visuals, OBS_STATE}
        }
        self.output_features = {ACTION: self.action_feature}

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
