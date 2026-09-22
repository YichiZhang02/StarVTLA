from dataclasses import dataclass, field
from pathlib import Path

from vtla.engine.configs import PreTrainedConfig
from vtla.engine.utils.constants import ACTION
from vtla.frameworks.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("n0_vtla")
@dataclass
class N0VTLAConfig(PI05Config):
    """N0-VTLA with StarVTLA action semantics and a dedicated tactile path.

    `base_model_path` imports original N0-VTLA weights once; `pretrained_path`
    is reserved for complete StarVTLA policy checkpoints, including processors.
    """

    tactile_mode: str = "as_image"
    chunk_size: int = 50
    n_action_steps: int = 50
    base_model_path: Path | None = None
    allow_random_init: bool = False
    n_latent: int = 5
    tactile_pool_grid: int = 3
    predictor_n_layers: int = 2
    predictor_n_heads: int = 8
    predictor_arch: str = "tactile_kv"
    z_gate_zero_init: bool = True
    g_to_expert: bool = False
    tactile_image_size: int = 224
    dinov2_config: dict = field(default_factory=dict)
    # Explicitly opt into new action projections when importing a different width.
    reinitialize_action_projections: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.tactile_mode != "as_image":
            raise ValueError("N0-VTLA only supports tactile_mode='as_image'.")
        if self.paligemma_variant != "gemma_2b":
            raise ValueError("N0-VTLA requires the native gemma_2b PaliGemma prefix.")
        if not self.selected_camera_keys():
            raise ValueError("N0-VTLA requires at least one RGB camera.")
        if not self.tactile_keys or len(set(self.tactile_keys)) != len(self.tactile_keys):
            raise ValueError("N0-VTLA requires nonempty, unique tactile_keys.")
        if set(self.tactile_keys) & set(self.selected_camera_keys()):
            raise ValueError("RGB and tactile keys must not overlap.")
        if self.tactile_num_frames != 1 or self.tactile_frame_offset != 1:
            raise ValueError("N0-VTLA owns [episode baseline, current] sampling; tactile history knobs must be 1.")
        if self.predictor_arch not in {"joint_kv", "tactile_kv"}:
            raise ValueError("predictor_arch must be joint_kv or tactile_kv.")
        if min(self.n_latent, self.predictor_n_layers, self.predictor_n_heads, self.tactile_pool_grid,
               self.chunk_size, self.n_action_steps, self.num_inference_steps) <= 0:
            raise ValueError("N0-VTLA token, layer, chunk and inference counts must be positive.")
        patch = self.dinov2_config.get("patch_size", 14)
        if self.tactile_image_size <= 0 or self.tactile_image_size % patch:
            raise ValueError("tactile_image_size must be a positive multiple of the DINOv2 patch size.")
        if self.rtc_config is not None or self.compile_model:
            raise ValueError("N0-VTLA currently requires rtc_config=None and compile_model=False.")
        if self.action_gap < 0:
            raise ValueError("action_gap must be non-negative.")

    def tactile_windowed(self) -> bool:
        return True

    def tactile_windowed_keys(self) -> list[str]:
        return list(self.tactile_keys)

    def tactile_delta_indices(self) -> list[int]:
        # The reader replaces the first index with the exact episode start.
        return [0, 0]

    def episode_start_image_keys(self) -> list[str]:
        return list(self.tactile_keys)

    @property
    def drop_n_last_frames(self) -> int:
        # A positive action_gap otherwise creates all-padding target chunks at
        # episode ends. Both indexed and mixture training samplers honor this.
        return self.action_gap

    def validate_features(self):
        self.input_features = self.input_features or {}
        self.output_features = self.output_features or {}
        for key in self.image_keys():
            self.require_visual_feature(key, "N0-VTLA input")
        self.prune_unselected_visual_features()
        self.apply_state_mode()
        self.apply_action_mode()
        if self.action_feature is None:
            raise ValueError("N0-VTLA requires dataset action features.")
        if self.action_feature.shape[0] > self.max_action_dim:
            raise ValueError("Action feature exceeds max_action_dim; explicitly expand the model projections.")
        if self.action_representation in {"rot6d"}:
            width = 10
            if self.action_feature.shape != (width * self.ee_num_arms,):
                raise ValueError("EEF action width must match action_representation and ee_num_arms.")
        if self.action_mode == "relative_joint" and self.relative_exclude_joints:
            if not self.action_feature_names or len(self.action_feature_names) != self.action_feature.shape[0]:
                raise ValueError("relative_joint needs complete action feature names to preserve absolute grippers.")
        if self.state_mode != "none":
            if self.robot_state_feature is None or self.robot_state_feature.shape[0] > self.max_state_dim:
                raise ValueError("Missing state or state feature exceeds max_state_dim.")
        self.output_features = {ACTION: self.action_feature}

    def validate_checkpoint_layout(self, saved):
        fields = ("action_mode", "state_mode", "ee_num_arms", "max_action_dim", "max_state_dim",
                  "paligemma_variant", "action_expert_variant", "predictor_arch", "n_latent",
                  "predictor_n_layers", "predictor_n_heads", "z_gate_zero_init", "g_to_expert",
                  "tactile_pool_grid", "tactile_image_size", "dinov2_config", "chunk_size")
        changed = [name for name in fields if getattr(self, name) != getattr(saved, name)]
        if self.image_keys() != saved.image_keys():
            changed.append("sensor order")
        if changed:
            raise ValueError(f"N0-VTLA checkpoint contract mismatch: {changed}. Use base_model_path for adaptation.")
