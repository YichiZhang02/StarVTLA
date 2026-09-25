from dataclasses import dataclass, field
from pathlib import Path

from vtla.engine.configs import NormalizationMode, PreTrainedConfig
from vtla.engine.utils.constants import ACTION
from vtla.frameworks.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("tacmind0")
@dataclass
class TacMind0Config(PI05Config):
    """Infra action/state routing with TacMind0's fixed tactile sampling contract."""

    tactile_mode: str = "as_image"
    tactile_num_frames: int = 8
    tactile_frame_offset: int = 5
    chunk_size: int = 32
    n_action_steps: int = 16
    max_action_dim: int = 32
    freeze_vlm_embedding: bool = True
    vlm_gradient_checkpointing: bool = True
    ae_gradient_checkpointing: bool = True
    optimizer_weight_decay: float = 1e-10
    base_model_path: Path | None = Path("playground/pretrained_models/tacmind0/base_policy")
    tactile_weights_path: Path | None = Path(
        "playground/pretrained_models/tacmind0/tac_lewm_finetuned/checkpoint-10000/encoder.pt"
    )
    tactile_backbone_config_path: Path | None = Path(
        "playground/pretrained_models/tacmind0/tac_lewm_finetuned/checkpoint-10000/backbone_config.json"
    )
    native_config: dict = field(default_factory=dict)
    num_inference_steps: int = 10
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.tactile_mode != "as_image":
            raise ValueError("TacMind0 requires tactile_mode=as_image.")
        if (self.tactile_num_frames, self.tactile_frame_offset) != (8, 5):
            raise ValueError("TacMind0 requires eight tactile frames with stride five.")
        if len(self.tactile_keys) != 2:
            raise ValueError("TacMind0 requires exactly two tactile keys, ordered left then right.")
        if not 1 <= len(self.selected_camera_keys()) <= 2:
            raise ValueError("TacMind0 requires one or two selected RGB cameras.")
        if self.max_action_dim != 32:
            raise ValueError("TacMind0 native action projection requires max_action_dim=32.")
        if self.chunk_size <= 0 or self.n_action_steps <= 0:
            raise ValueError("TacMind0 action chunk and execution size must be positive.")

    @property
    def drop_n_last_frames(self) -> int:
        return self.action_gap

    def validate_features(self):
        super().validate_features()
        if ACTION not in self.output_features:
            raise ValueError("TacMind0 requires actions.")
        if self.output_features[ACTION].shape[0] > 32:
            raise ValueError("TacMind0 action width exceeds 32.")

    def validate_checkpoint_layout(self, saved: "TacMind0Config") -> None:
        fields = ("action_mode", "state_mode", "chunk_size", "tactile_keys", "tactile_num_frames",
                  "tactile_frame_offset", "wrist_only", "top_camera_keys", "wrist_camera_keys")
        changed = [key for key in fields if getattr(self, key) != getattr(saved, key)]
        if changed:
            raise ValueError(f"TacMind0 checkpoint layout mismatch: {changed}")
