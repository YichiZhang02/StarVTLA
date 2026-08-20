#!/usr/bin/env python

"""Configuration for the independently registered DINO-aligned StarVLA-GR00T."""

from dataclasses import dataclass, field

from vtla.engine.configs import NormalizationMode, PreTrainedConfig
from vtla.engine.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from vtla.engine.utils.constants import ACTION

from ..sensor_routing import SensorRoutingMixin


@PreTrainedConfig.register_subclass("starvla_groot_dinoalign")
@dataclass
class StarvlaGrootDinoAlignConfig(SensorRoutingMixin, PreTrainedConfig):
    """Qwen3.5-GR00T policy trained against a frozen DINOv3 visual teacher."""

    # Qwen VLM backbone.
    base_vlm: str = "./playground/pretrained_models/Qwen3.5-0.8B"
    attn_implementation: str = "sdpa"
    dtype: str = "bfloat16"

    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 16

    # GR00T flow-matching action head.
    action_model_type: str = "DiT-B"
    action_head_hidden_size: int = 1024
    num_inference_timesteps: int = 4
    repeated_diffusion_steps: int = 8
    num_target_vision_tokens: int = 32
    add_pos_embed: bool = True
    max_seq_len: int = 1024

    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

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

    action_dim: int | None = None
    state_dim: int | None = None

    # Multi-view / tactile / state routing.
    empty_cameras: int = 0
    image_resolution: tuple[int, int] = (224, 224)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    gradient_checkpointing: bool = False
    device: str | None = None
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # Frozen DINOv3 ViT-B/16 teacher. Loading is controlled by a runtime-only
    # constructor argument supplied by the policy factory, not by saved config.
    dinov3_model_name: str = "vit_base_patch16_dinov3"
    dinov3_checkpoint: str | None = (
        "./playground/pretrained_models/vit_base_patch16_dinov3.lvd1689m"
    )
    dinov3_input_size: int = 256
    dinov3_patch_size: int = 16
    dinov3_hidden_size: int = 768
    dinov3_teacher_dtype: str = "bfloat16"

    # DINO alignment objective.
    dino_alignment_weight: float = 0.1
    dino_global_loss_weight: float = 0.2
    dino_alignment_warmup_steps: int = 1_000

    # Geometry-preserving, on-GPU student-view illumination augmentation.
    dino_light_augmentation: bool = True
    dino_light_augmentation_probability: float = 0.8
    dino_brightness_range: tuple[float, float] = (0.6, 1.4)
    dino_contrast_range: tuple[float, float] = (0.7, 1.3)
    dino_gamma_range: tuple[float, float] = (0.7, 1.5)
    dino_shadow_probability: float = 0.5
    dino_shadow_strength_range: tuple[float, float] = (0.2, 0.6)

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
        if self.action_model_type not in ("DiT-B", "DiT-L"):
            raise ValueError(f"Invalid action_model_type: {self.action_model_type}")
        if self.dtype not in ("bfloat16", "float32"):
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.dinov3_teacher_dtype not in ("bfloat16", "float32"):
            raise ValueError(f"Invalid dinov3_teacher_dtype: {self.dinov3_teacher_dtype}")
        if self.image_resolution != (224, 224):
            raise ValueError(
                "starvla_groot_dinoalign currently requires image_resolution=(224, 224); "
                "Qwen smart-resizes it to 256x256 for exact DINO patch alignment."
            )
        if self.dinov3_input_size != 256 or self.dinov3_patch_size != 16:
            raise ValueError("DINO alignment currently requires a 256 input and a ViT /16 teacher.")
        if not 0.0 <= self.dino_global_loss_weight <= 1.0:
            raise ValueError("dino_global_loss_weight must be in [0, 1].")
        if self.dino_alignment_weight < 0.0:
            raise ValueError("dino_alignment_weight must be non-negative.")
        if self.dino_alignment_warmup_steps < 0:
            raise ValueError("dino_alignment_warmup_steps must be non-negative.")
        if not 0.0 <= self.dino_light_augmentation_probability <= 1.0:
            raise ValueError("dino_light_augmentation_probability must be in [0, 1].")
        if not 0.0 <= self.dino_shadow_probability <= 1.0:
            raise ValueError("dino_shadow_probability must be in [0, 1].")
        for name in (
            "dino_brightness_range",
            "dino_contrast_range",
            "dino_gamma_range",
            "dino_shadow_strength_range",
        ):
            low, high = getattr(self, name)
            if low < 0.0 or low > high:
                raise ValueError(f"{name} must be a non-negative (min, max) pair, got {(low, high)}")
        if self.freeze_vision_encoder and self.dino_alignment_weight > 0:
            raise ValueError("DINO alignment requires freeze_vision_encoder=false.")
        if self.train_expert_only and self.dino_alignment_weight > 0:
            raise ValueError("DINO alignment requires train_expert_only=false.")
        self.validate_sensor_modes()

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}

        empty_keys = tuple(self.add_empty_cameras(self.empty_cameras, self.image_resolution))
        self.prune_unselected_visual_features(extra_keep=empty_keys)
        self.apply_state_mode()
        self.apply_action_mode()
        self.validate_routed_keys()
        if ACTION not in self.output_features:
            raise ValueError("StarvlaGrootDinoAlign requires an 'action' output feature.")

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
        if self.action_reference == "relative":
            return list(range(1, self.chunk_size + 1))
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
