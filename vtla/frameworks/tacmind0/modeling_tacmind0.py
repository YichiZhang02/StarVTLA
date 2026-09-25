"""TacMind0 native Gemma3/Tac-LeWM model behind the StarVTLA policy API."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor

from tacmind0.model.dmtac05.dmtac05_arch import DMTac05Config, DMTac05ForConditionalGeneration
from tacmind0.tactile.encoder import FrozenTactileEncoder, TactileFiLM
from tacmind0.tactile.preprocessing import prepare_history
from tacmind0.tactile.provenance import sha256, validate_initialization
from vtla.engine.configs import PreTrainedConfig
from vtla.engine.utils.constants import ACTION, OBS_STATE
from vtla.frameworks.pretrained import PreTrainedPolicy

from .configuration_tacmind0 import TacMind0Config


def _rgb_uint8(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB image, got {tuple(image.shape)}")
    image = image.detach().to("cpu")
    if image.dtype == torch.uint8:
        return image.permute(1, 2, 0).contiguous().numpy()
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("TacMind0 images must be uint8 or finite [0,1] floats")
    return (image.permute(1, 2, 0).float().clamp(0, 1) * 255).round().byte().numpy()


def _validate_tactile_weights(weights_path: Path, backbone_path: Path) -> dict:
    sidecar = weights_path.with_name("encoder_provenance.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"TacMind0 tactile encoder provenance missing: {sidecar}")
    provenance = json.loads(sidecar.read_text())
    validate_initialization(provenance)
    if sha256(weights_path) != provenance.get("encoder_sha256"):
        raise ValueError("TacMind0 tactile encoder checksum does not match provenance")
    if sha256(backbone_path) != provenance.get("backbone_sha256"):
        raise ValueError("TacMind0 tactile backbone checksum does not match provenance")
    return provenance


class TacMind0Policy(PreTrainedPolicy):
    config_class = TacMind0Config
    name = "tacmind0"

    def __init__(self, config: TacMind0Config, *, core_model: nn.Module | None = None,
                 processor=None, processor_path: str | Path | None = None,
                 initialize_base: bool = True, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.action_dim = int(config.action_feature.shape[0])
        if core_model is not None:
            self.model = core_model
        elif initialize_base:
            base = Path(config.base_model_path)
            if not base.is_dir():
                raise FileNotFoundError(f"TacMind0 base policy missing: {base}")
            native_config = DMTac05Config.from_pretrained(base, local_files_only=True)
            if native_config.tactile_config is not None:
                raise ValueError("Use a StarVTLA tacmind0 checkpoint for a tactile-initialized model")
            native_config.chunk_size = config.chunk_size
            self.model, info = DMTac05ForConditionalGeneration.from_pretrained(
                base, config=native_config,
                dtype=torch.bfloat16 if config.dtype == "bfloat16" else torch.float32,
                local_files_only=True, output_loading_info=True,
            )
            failures = {key: info[key] for key in
                        ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
            if failures:
                raise RuntimeError(f"TacMind0 base weight mismatch: {failures}")
            backbone_path = Path(config.tactile_backbone_config_path)
            tactile_weights_path = Path(config.tactile_weights_path)
            provenance = _validate_tactile_weights(tactile_weights_path, backbone_path)
            backbone = json.loads(backbone_path.read_text())
            encoder = FrozenTactileEncoder(backbone, trainable=True)
            encoder.load_world_model(tactile_weights_path)
            encoder.set_trainable(True)
            self.model.tactile_encoder = encoder.to(dtype=next(self.model.parameters()).dtype)
            self.model.tactile_film = TactileFiLM(
                self.model.config.vlm_config.text_config.hidden_size
            ).to(dtype=next(self.model.parameters()).dtype)
            self.model.config.tactile_config = {
                "backbone": backbone, "source": "marker", "color": "bgr",
                "history_size": 8, "history_stride": 5,
                "encoder_initialization": provenance,
            }
            self.model.model.config.tactile_config = self.model.config.tactile_config
            config.native_config = self.model.config.to_dict()
        else:
            if not config.native_config:
                raise ValueError("TacMind0 checkpoint is missing native_config")
            self.model = DMTac05ForConditionalGeneration(DMTac05Config(**config.native_config))
        # The imported native class defaults to a frozen tactile encoder. Its train()
        # override and forward path are explicitly made trainable in this vendored copy.
        self.model.tactile_encoder.set_trainable(True)
        if hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing(
                vlm_gradient_checkpointing=config.vlm_gradient_checkpointing,
                ae_gradient_checkpointing=config.ae_gradient_checkpointing,
            )
        if config.freeze_vlm_embedding and hasattr(self.model, "freeze_vlm_embedding"):
            self.model.freeze_vlm_embedding()
        source = processor_path or config.base_model_path
        self.processor = processor or AutoProcessor.from_pretrained(source, local_files_only=True)
        self.reset()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, strict=True, **kwargs):
        saved = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        if not isinstance(saved, TacMind0Config):
            raise ValueError("Expected a StarVTLA tacmind0 checkpoint")
        if config is not None:
            config.validate_checkpoint_layout(saved)
            config.native_config = saved.native_config
        processor_path = Path(pretrained_name_or_path) / "tacmind0_processor"
        if not processor_path.is_dir():
            raise FileNotFoundError(f"TacMind0 checkpoint processor missing: {processor_path}")
        return super().from_pretrained(
            pretrained_name_or_path, config=config or saved, strict=strict,
            initialize_base=False, processor_path=processor_path, **kwargs,
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory / "tacmind0_processor")

    def get_optim_params(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def reset(self):
        self._action_queue = deque()

    def _model_inputs(self, batch: dict) -> dict[str, torch.Tensor]:
        camera_keys = self.config.selected_camera_keys()
        tactile_keys = self.config.tactile_keys
        batch_size = batch[camera_keys[0]].shape[0]
        device = next(self.model.parameters()).device
        tokens = []
        histories = []
        tasks = batch.get("task", self.config.single_task)
        if isinstance(tasks, str):
            tasks = [tasks] * batch_size
        if tasks is None or len(tasks) != batch_size:
            raise ValueError("TacMind0 requires one task string per observation")
        states = batch.get(OBS_STATE)
        for sample in range(batch_size):
            views = [Image.fromarray(_rgb_uint8(batch[key][sample])) for key in camera_keys]
            if len(views) == 1:
                views.append(views[0])
            user_content = [{"type": "text", "text": f"Robot: {self.config.robot_type or 'unknown'}\nOverall speed: 0.5\nTask: {tasks[sample]}.\n"}]
            for index, view in enumerate(views):
                user_content.append({"type": "text", "text": f"View {index + 1} image: "})
                user_content.append({"type": "image", "image": view})
            if self.config.state_mode != "none":
                if states is None:
                    raise ValueError("TacMind0 state_mode requires observation.state")
                state = states[sample].detach().float().cpu().numpy()
                bins = np.floor((np.clip(state, -1, 1) + 1) * 0.5 * 255).astype(int)
                user_content.append({"type": "text", "text": "States: " + " ".join(map(str, bins))})
            messages = [{"role": "user", "content": user_content}]
            tokens.append(self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ))
            pair = []
            for key in tactile_keys:
                frames = batch[key][sample]
                if frames.ndim != 4 or frames.shape[0] != 8:
                    raise ValueError(f"TacMind0 tactile key {key} must be [B,8,3,H,W]")
                pair.append([_rgb_uint8(frame) for frame in frames])
            histories.append(prepare_history(*pair))
        pad_id = self.processor.tokenizer.pad_token_id
        width = max(entry["input_ids"].shape[1] for entry in tokens)
        result = {}
        for key, fill in (("input_ids", pad_id), ("attention_mask", 0), ("token_type_ids", 0)):
            result[key] = torch.cat([
                F.pad(entry[key], (0, width - entry[key].shape[1]), value=fill)
                for entry in tokens
            ], dim=0).to(device)
        result["pixel_values"] = torch.cat([entry["pixel_values"] for entry in tokens]).to(device)
        result["tactile_pixel_values"] = torch.stack(histories).to(device)
        return result

    def forward(self, batch, reduction="mean"):
        if reduction != "mean":
            raise ValueError("TacMind0 currently supports reduction='mean' only")
        action = batch[ACTION]
        if action.ndim != 3 or action.shape[1:] != (self.config.chunk_size, self.action_dim):
            raise ValueError("TacMind0 action shape does not match chunk_size/action feature")
        valid = torch.ones_like(action, dtype=torch.bool)
        if "action_is_pad" in batch:
            valid &= ~batch["action_is_pad"].bool().unsqueeze(-1)
        if "action_mask" in batch:
            valid &= batch["action_mask"].bool()
        if not valid.any(dim=(1, 2)).all():
            raise ValueError("TacMind0 training sample has no valid actions")
        inputs = self._model_inputs(batch)
        outputs = self.model(
            **inputs,
            action=F.pad(action, (0, 32 - self.action_dim)).to(inputs["pixel_values"].device),
            action_mask=F.pad(valid, (0, 32 - self.action_dim)),
        )
        return outputs.loss, {"action_loss": outputs.loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        self.eval()
        inputs = self._model_inputs(batch)
        return self.model.inference_action(
            **inputs, diffusion_steps=self.config.num_inference_steps,
        )[..., :self.action_dim]

    @torch.no_grad()
    def select_action(self, batch, **kwargs):
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)
            start = self.config.action_start_offset
            self._action_queue.extend(chunk[:, start:start + self.config.n_action_steps].transpose(0, 1))
        return self._action_queue.popleft()
