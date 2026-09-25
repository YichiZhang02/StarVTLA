"""TacDream entry point: python -m dmtac05.tacdream --help."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from tacmind0.constants.robot import ActionMode, RobotStateDesc, RobotType
from tacmind0.data.augmentations import (
    NoAugmentationPipeline,
    TrainingTransformPipeline,
)
from tacmind0.data.collator import TrainingCollator
from tacmind0.data.transforms import (
    ChatTokenization,
    LoadImages,
    Normalize,
    PadAction,
    Pipeline,
    PixelTransform,
)
from tacmind0.exp.dmtac05_exp import (
    DMTac05DataConfig,
    DMTac05Exp,
    DMTac05InferenceConfig,
    DMTac05ModelConfig,
    DMTac05TrainerConfig,
)
from tacmind0.model.dmtac05.dmtac05_arch import (
    DMTac05Config,
    DMTac05ForConditionalGeneration,
)
from tacmind0.tac_frs.serving import TacFRSServingMixin
from tacmind0.tac_frs.stream import StreamSessions
from tacmind0.tactile.dataset import TacDreamDataset
from tacmind0.tactile.encoder import FrozenTactileEncoder, TactileFiLM
from tacmind0.tactile.preprocessing import LoadTactileHistory
from tacmind0.tactile.provenance import sha256, validate_initialization

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / "playground" / "pretrained_models" / "tacmind0"
CONTRACT = {
    "version": 1,
    "source": "marker",
    "color": "bgr",
    "history_size": 8,
    "history_stride": 5,
    "frame_size": 42,
    "patch_size": 14,
    "resize": "area56_bilinear42_round_uint8",
    "readout": "cls",
    "streaming": False,
    "vision_views": ["agentview", "wrist"],
    "film": {"hidden_dim": 384, "activation": "silu", "identity_init": True},
    "visual_augmentation": {
        "agentview": "none",
        "wrist": "none",
        "probability": 0.5,
    },
    "action_mode": "relative",
    "output_action_dim": 8,
    "state_desc": ["joint"] * 7 + ["gripper"],
    "add_state": True,
    "n_bins": 256,
    "image_prompts": ["Agent view", "Wrist"],
}


def validate_visual_augmentation(value: dict) -> None:
    if not isinstance(value, dict):
        raise ValueError("TacDream visual_augmentation must be an object")
    for view in ("agentview", "wrist"):
        if value.get(view) not in ("none", "light"):
            raise ValueError(f"TacDream {view} augmentation must be 'none' or 'light'")
    probability = value.get("probability")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not np.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("TacDream augmentation probability must be in [0, 1]")


def finetuned_encoder_provenance(
    weights: str | Path, *, required: bool, backbone: str | Path | None = None
) -> dict | None:
    path = Path(weights)
    sidecar = path.parent / "encoder_provenance.json"
    if not sidecar.is_file():
        if required:
            raise ValueError(
                "Fine-tune Tac-LeWM first; encoder_provenance.json is required"
            )
        return None
    provenance = json.loads(sidecar.read_text())
    validate_initialization(provenance)
    if sha256(path) != provenance.get("encoder_sha256"):
        raise ValueError("Fine-tuned encoder checksum does not match provenance")
    if backbone is not None and sha256(Path(backbone)) != provenance.get(
        "backbone_sha256"
    ):
        raise ValueError(
            "Fine-tuned encoder backbone checksum does not match provenance"
        )
    return provenance


def validate_contract(contract: dict) -> None:
    """Reject checkpoint settings that the fixed v1 encoder cannot implement."""
    fixed = (
        "version",
        "source",
        "color",
        "history_size",
        "history_stride",
        "frame_size",
        "patch_size",
        "resize",
        "readout",
        "streaming",
        "vision_views",
        "film",
        "output_action_dim",
    )
    for key in fixed:
        if contract.get(key) != CONTRACT[key]:
            raise ValueError(f"Unsupported TacDream {key}: {contract.get(key)!r}")
    descriptors = contract.get("state_desc")
    if not isinstance(descriptors, list) or len(descriptors) != 8:
        raise ValueError("TacDream state_desc must describe exactly eight dimensions")
    for value in descriptors:
        RobotStateDesc(value)
    ActionMode(contract.get("action_mode"))
    if not isinstance(contract.get("add_state"), bool):
        raise ValueError("TacDream add_state must be boolean")
    bins = contract.get("n_bins")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("TacDream n_bins must be an integer >= 2")
    prompts = contract.get("image_prompts")
    if (
        not isinstance(prompts, list)
        or len(prompts) != 2
        or not all(isinstance(value, str) and value.strip() for value in prompts)
    ):
        raise ValueError("TacDream requires two nonempty image prompts")
    validate_visual_augmentation(
        contract.get("visual_augmentation", CONTRACT["visual_augmentation"])
    )


def strict_policy_load(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    overrides: dict | None = None,
) -> DMTac05ForConditionalGeneration:
    config = DMTac05Config.from_pretrained(path, local_files_only=True)
    if config.tactile_config is not None:
        validate_contract(config.tactile_config)
    for name, value in (overrides or {}).items():
        setattr(config, name, value)
    model, info = DMTac05ForConditionalGeneration.from_pretrained(
        path,
        config=config,
        dtype=dtype,
        local_files_only=True,
        output_loading_info=True,
    )
    failures = {
        key: info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if info.get(key)
    }
    if failures:
        raise RuntimeError(f"Strict policy loading failed: {failures}")
    model._tacdream_strict_checkpoint = str(Path(path).resolve())
    return model


@dataclass
class TacDreamModelConfig(DMTac05ModelConfig):
    model_name_or_path: str = str(RESOURCE / "base_policy")
    chunk_size: int = 32
    tactile_weights: str = str(
        RESOURCE / "tac_lewm_finetuned" / "checkpoint-10000" / "encoder.pt"
    )
    backbone_config: str = str(
        RESOURCE / "tac_lewm_finetuned" / "checkpoint-10000" / "backbone_config.json"
    )
    require_finetuned_tactile: bool = True
    vision_attn_implementation: str = "sdpa"
    llm_attn_implementation: str = "sdpa"
    action_attn_implementation: str = "sdpa"
    liger_kernel: bool = False

    def _config_overrides(self):
        config = DMTac05Config.from_pretrained(
            self.model_name_or_path, local_files_only=True
        )
        if config.tactile_config is not None:
            validate_contract(config.tactile_config)
            self.chunk_size = config.chunk_size
        return {**super()._config_overrides(), "use_suffix_graph": False}

    def _load_base_checkpoint_model(self):
        saved = DMTac05Config.from_pretrained(
            self.model_name_or_path, local_files_only=True
        )
        provenance = None
        if saved.tactile_config is None:
            provenance = finetuned_encoder_provenance(
                self.tactile_weights,
                required=self.require_finetuned_tactile,
                backbone=self.backbone_config,
            )
        elif self.require_finetuned_tactile:
            provenance = saved.tactile_config.get("encoder_initialization", {})
            validate_initialization(provenance)
        model = strict_policy_load(
            self.model_name_or_path,
            dtype=self._torch_dtype(),
            overrides=self._config_overrides(),
        )
        if model.config.tactile_config is None:
            backbone = json.loads(Path(self.backbone_config).read_text())
            encoder = FrozenTactileEncoder(backbone)
            report = encoder.load_world_model(self.tactile_weights)
            model.config.tactile_config = {**deepcopy(CONTRACT), "backbone": backbone}
            if provenance is not None:
                model.config.tactile_config["encoder_initialization"] = provenance
            model.model.config.tactile_config = model.config.tactile_config
            model.tactile_encoder = encoder.to(dtype=self._torch_dtype())
            model.tactile_film = TactileFiLM(
                model.config.vlm_config.text_config.hidden_size
            )
            model.tactile_film.to(dtype=self._torch_dtype())
            model.tactile_load_report = report
        return model

    def build_model(self, use_lora=False):
        if (
            use_lora
            or (Path(self.model_name_or_path) / "adapter_config.json").is_file()
        ):
            raise ValueError("TacDream v1 requires full-policy training, not LoRA")
        return super().build_model(use_lora=False)


@dataclass
class TacDreamDataConfig(DMTac05DataConfig):
    dataset_name: str = "tacdream"
    jsonl_dir: str = str(ROOT / "data" / "episodes")
    image_dir: str = str(ROOT / "data")
    norm_file: str = str(RESOURCE / "base_policy" / "norm_stats.json")
    action_mode: ActionMode = ActionMode.RELATIVE
    add_state: bool = True
    state_desc: list[str] = field(default_factory=lambda: list(CONTRACT["state_desc"]))
    image_prompts: list[str] = field(
        default_factory=lambda: list(CONTRACT["image_prompts"])
    )
    agentview_augmentation: Literal["none", "light"] = "none"
    wrist_augmentation: Literal["none", "light"] = "none"
    augmentation_probability: float = 0.5

    def visual_augmentation_contract(self) -> dict:
        value = {
            "agentview": self.agentview_augmentation,
            "wrist": self.wrist_augmentation,
            "probability": self.augmentation_probability,
        }
        validate_visual_augmentation(value)
        return value

    def _dataset_info(self):
        return {
            "jsonl_dir": self.jsonl_dir,
            "image_dir": self.image_dir,
            "image_keys": ["images_1", "images_2"],
            "image_prompts": list(self.image_prompts),
            "robot_type": RobotType.FRANKA,
            "state_desc": [RobotStateDesc(value) for value in self.state_desc],
        }

    def norm_stats_path(self, action_horizon):
        return Path(self.norm_file)

    def build_norm_stats_dataset(self, action_horizon):
        info = self._dataset_info()
        return TacDreamDataset(
            self.jsonl_dir,
            Pipeline([self._action_transform(action_horizon)]),
            self.dataset_name,
            self._dataset_meta(info),
        )

    def build_dataset(self, processor, action_horizon, tokenizer_max_length=1024):
        if self.is_history:
            raise ValueError("TacDream v1 training uses two current visual views")
        info = self._dataset_info()
        augmentation = self.visual_augmentation_contract()
        transforms = Pipeline(
            [
                LoadTactileHistory(self.image_dir),
                self._action_transform(action_horizon),
                LoadImages(info["image_keys"], self.image_dir),
                PixelTransform(
                    image_pipelines=[
                        (
                            TrainingTransformPipeline(p=augmentation["probability"])
                            if mode == "light"
                            else NoAugmentationPipeline()
                        )
                        for mode in (
                            augmentation["agentview"],
                            augmentation["wrist"],
                        )
                    ]
                ),
                Normalize(
                    str(self.norm_stats_path(action_horizon)),
                    ["state", "action"],
                    use_quantiles=True,
                ),
                ChatTokenization(
                    processor,
                    n_bins=self.n_bins,
                    max_length=tokenizer_max_length,
                    image_prompts=info["image_prompts"],
                    add_state=self.add_state,
                ),
                PadAction(32),
            ]
        )
        return (
            TacDreamDataset(
                self.jsonl_dir, transforms, self.dataset_name, self._dataset_meta(info)
            ),
            TrainingCollator(processor.tokenizer.pad_token_id, tokenizer_max_length),
        )


@dataclass
class TacDreamInferenceConfig(TacFRSServingMixin, DMTac05InferenceConfig):
    tac_frs_sessions: StreamSessions = field(
        default_factory=StreamSessions, init=False, repr=False
    )
    output_action_dim: int = 8
    image_prompts: list[str] = field(default_factory=lambda: ["Agent view", "Wrist"])
    port: int = 7897

    def _parse_state(self, states) -> np.ndarray:
        state = super()._parse_state(states)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(
                "TacDream observation.state must contain eight finite values"
            )
        return state

    def _resolve_state_desc(self, robot_type: str | None):
        if robot_type is not None and robot_type != RobotType.FRANKA.value:
            raise ValueError(
                "TacDream v1 requires the checkpoint's Franka robot contract"
            )
        return super()._resolve_state_desc(robot_type)


@dataclass
class TacDreamTrainerConfig(DMTac05TrainerConfig):
    output_dir: str = str(ROOT / "playground" / "results" / "models" / "tacmind0" / "stage1")
    fsdp1: bool = False
    per_device_train_batch_size: int = 1


@dataclass
class TacDreamExp(DMTac05Exp):
    model_config: TacDreamModelConfig = field(default_factory=TacDreamModelConfig)
    data_config: TacDreamDataConfig = field(default_factory=TacDreamDataConfig)
    inference_config: TacDreamInferenceConfig = field(
        default_factory=TacDreamInferenceConfig
    )
    trainer_config: TacDreamTrainerConfig = field(default_factory=TacDreamTrainerConfig)

    def _restore_checkpoint_contract(self, path: str | Path) -> None:
        config = DMTac05Config.from_pretrained(path, local_files_only=True)
        if config.tactile_config is None:
            return
        contract = config.tactile_config
        validate_contract(contract)
        norm = Path(path) / "norm_stats.json"
        if not norm.is_file():
            raise FileNotFoundError("TacDream checkpoint must contain norm_stats.json")
        self.model_config.chunk_size = config.chunk_size
        self.data_config.action_mode = ActionMode(contract["action_mode"])
        self.data_config.norm_file = str(norm)
        self.data_config.add_state = contract["add_state"]
        self.data_config.n_bins = contract["n_bins"]
        self.data_config.state_desc = list(contract["state_desc"])
        self.data_config.image_prompts = list(contract["image_prompts"])
        augmentation = contract.get(
            "visual_augmentation", CONTRACT["visual_augmentation"]
        )
        validate_visual_augmentation(augmentation)
        self.data_config.agentview_augmentation = augmentation["agentview"]
        self.data_config.wrist_augmentation = augmentation["wrist"]
        self.data_config.augmentation_probability = augmentation["probability"]
        self.inference_config.output_action_dim = contract["output_action_dim"]
        self.inference_config.image_prompts = list(contract["image_prompts"])

    def _prepare_training_contract(self) -> None:
        from transformers.trainer_utils import get_last_checkpoint

        output = Path(self.trainer_config.output_dir)
        latest = get_last_checkpoint(str(output)) if output.is_dir() else None
        if latest is not None:
            # The parent Trainer resumes this same checkpoint. Build the model and
            # dataset from its contract before reading data or loading any weights.
            self.model_config.model_name_or_path = latest
        self._restore_checkpoint_contract(self.model_config.model_name_or_path)
        contract = {
            **deepcopy(CONTRACT),
            "action_mode": self.data_config.action_mode.value,
            "add_state": self.data_config.add_state,
            "n_bins": self.data_config.n_bins,
            "state_desc": list(self.data_config.state_desc),
            "image_prompts": list(self.data_config.image_prompts),
            "visual_augmentation": self.data_config.visual_augmentation_contract(),
        }
        validate_contract(contract)

    def _initialize_train(self):
        self._prepare_training_contract()
        super()._initialize_train()
        self.model.config.tactile_config.update(
            {
                "action_mode": self.data_config.action_mode.value,
                "add_state": self.data_config.add_state,
                "n_bins": self.data_config.n_bins,
                "state_desc": list(self.data_config.state_desc),
                "image_prompts": list(self.data_config.image_prompts),
                "visual_augmentation": self.data_config.visual_augmentation_contract(),
            }
        )

    def _initialize_inference_runtime(self):
        if self.inference_config.backend != "default":
            raise ValueError("TacDream requires the default PyTorch backend")
        self._restore_checkpoint_contract(self.model_config.model_name_or_path)
        super()._initialize_inference_runtime()
        path = Path(self.model_config.model_name_or_path)
        if (path / "tac_frs.json").is_file():
            from tacmind0.tac_frs.model import TacFRS

            self.inference_config.tac_frs_runtime = TacFRS.load(
                path, self.inference_config.device
            )
            self.inference_config.tac_frs_sessions = StreamSessions()


def main() -> None:
    exp = tyro.cli(TacDreamExp)
    if exp.task == "train":
        exp.train()
    elif exp.task == "inference":
        exp.inference()
    else:
        raise ValueError(exp.task)


if __name__ == "__main__":
    main()
