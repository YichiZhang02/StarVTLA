#!/usr/bin/env python

"""Independent StarVLA-GR00T policy with training-time DINOv3 alignment."""

from collections import deque
from contextlib import nullcontext

import torch
from torch import Tensor

from vtla.engine.utils.constants import ACTION, OBS_STATE
from vtla.engine.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from .action_head.flow_matching_head import (
    ActionHeadConfig,
    FlowmatchingActionHead,
)
from ..tactile_encode import TactileEncoder
from ..utils import expand_tactile_as_image_window
from .configuration_starvla_groot_dinoalign import StarvlaGrootDinoAlignConfig
from .dinov3_alignment import DinoAlignmentHead, DinoV3Teacher, IlluminationAugment
from .qwen_vl_interface import DinoAlignQwenVLInterface


class StarvlaGrootDinoAlignPolicy(PreTrainedPolicy):
    """Qwen3.5-GR00T whose visual tokens are distilled from frozen DINOv3."""

    config_class = StarvlaGrootDinoAlignConfig
    name = "starvla_groot_dinoalign"

    def __init__(
        self,
        config: StarvlaGrootDinoAlignConfig,
        load_dino_teacher: bool = False,
        **kwargs,
    ):
        require_package("transformers", extra="starvla_groot_dinoalign")
        require_package("timm", extra="starvla_groot_dinoalign")
        super().__init__(config)
        config.validate_features()
        self.config = config

        action_dim = config.action_dim or int(config.output_features[ACTION].shape[0])
        if config.state_mode == "none":
            state_dim = 0
        elif config.state_dim is not None:
            state_dim = int(config.state_dim)
        elif OBS_STATE in config.input_features:
            state_dim = int(config.input_features[OBS_STATE].shape[0])
        else:
            state_dim = 0
        self.action_dim = action_dim
        self.state_dim = state_dim

        load_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        self.qwen_vl = DinoAlignQwenVLInterface(
            base_vlm=config.base_vlm,
            attn_implementation=config.attn_implementation,
            dtype=load_dtype,
            image_resolution=config.image_resolution,
        )

        diffusion_model_cfg = dict(config.diffusion_model_cfg)
        diffusion_model_cfg["cross_attention_dim"] = self.qwen_vl.hidden_size
        self.action_head = FlowmatchingActionHead(
            ActionHeadConfig(
                action_model_type=config.action_model_type,
                hidden_size=config.action_head_hidden_size,
                action_dim=action_dim,
                state_dim=state_dim,
                action_horizon=config.chunk_size,
                num_inference_timesteps=config.num_inference_timesteps,
                num_target_vision_tokens=config.num_target_vision_tokens,
                add_pos_embed=config.add_pos_embed,
                max_seq_len=config.max_seq_len,
                noise_beta_alpha=config.noise_beta_alpha,
                noise_beta_beta=config.noise_beta_beta,
                noise_s=config.noise_s,
                num_timestep_buckets=config.num_timestep_buckets,
                diffusion_model_cfg=diffusion_model_cfg,
            )
        )

        self.tactile_encoder = None
        if config.tactile_mode == "encode":
            self.tactile_encoder = TactileEncoder(config, self.qwen_vl.hidden_size)
            if (
                config.tactile_insert_location == "encoder"
                and not self.qwen_vl.supports_prefix_injection()
            ):
                raise RuntimeError(
                    "Encoder-side tactile injection requires the Qwen3.5 prefix API."
                )

        qwen_visual_size = int(self.qwen_vl.model.config.vision_config.out_hidden_size)
        self.dino_alignment_head = DinoAlignmentHead(
            qwen_hidden_size=qwen_visual_size,
            dino_hidden_size=config.dinov3_hidden_size,
        )
        self.illumination_augment = IlluminationAugment(
            probability=config.dino_light_augmentation_probability,
            brightness_range=config.dino_brightness_range,
            contrast_range=config.dino_contrast_range,
            gamma_range=config.dino_gamma_range,
            shadow_probability=config.dino_shadow_probability,
            shadow_strength_range=config.dino_shadow_strength_range,
        )
        self.register_buffer("dino_alignment_step", torch.zeros((), dtype=torch.long))

        # The teacher is deliberately not registered as a child module. It is frozen,
        # excluded from the optimizer/checkpoint, and absent at inference.
        object.__setattr__(self, "_dino_teacher", None)

        self._set_requires_grad()
        self.to(config.device)
        if load_dino_teacher and config.dino_alignment_weight > 0.0:
            self._load_dino_teacher()
        self.reset()

    @property
    def dino_teacher(self) -> DinoV3Teacher | None:
        return object.__getattribute__(self, "_dino_teacher")

    def _apply(self, fn, recurse: bool = True):
        # The teacher is intentionally unregistered, so mirror later policy.to(...)
        # moves explicitly while still excluding it from state_dict/optimizer.
        super()._apply(fn, recurse=recurse)
        teacher = object.__getattribute__(self, "__dict__").get("_dino_teacher")
        if teacher is not None:
            teacher._apply(fn, recurse=recurse)
        return self

    def _load_dino_teacher(self) -> None:
        if not self.config.dinov3_checkpoint:
            raise FileNotFoundError(
                "starvla_groot_dinoalign training requires a local DINOv3 ViT-B/16 "
                "checkpoint. Set --policy.dinov3_checkpoint=/path/to/checkpoint."
            )
        device = next(self.parameters()).device
        teacher_dtype = (
            torch.bfloat16
            if self.config.dinov3_teacher_dtype == "bfloat16"
            else torch.float32
        )
        teacher = DinoV3Teacher(
            model_name=self.config.dinov3_model_name,
            checkpoint=self.config.dinov3_checkpoint,
            input_size=self.config.dinov3_input_size,
            patch_size=self.config.dinov3_patch_size,
            expected_hidden_size=self.config.dinov3_hidden_size,
            device=device,
            dtype=teacher_dtype,
        )
        object.__setattr__(self, "_dino_teacher", teacher)

    def _set_requires_grad(self) -> None:
        if self.config.train_expert_only:
            self.qwen_vl.eval()
            for parameter in self.qwen_vl.parameters():
                parameter.requires_grad_(False)
        elif self.config.freeze_vision_encoder:
            visual = self._get_qwen_visual_encoder()
            if visual is not None:
                visual.eval()
                for parameter in visual.parameters():
                    parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.train_expert_only:
            self.qwen_vl.eval()
        elif self.config.freeze_vision_encoder:
            visual = self._get_qwen_visual_encoder()
            if visual is not None:
                visual.eval()
        if self.dino_teacher is not None:
            self.dino_teacher.eval()
        return self

    def _get_qwen_visual_encoder(self):
        visual = getattr(self.qwen_vl.model, "visual", None)
        if visual is not None:
            return visual
        core = getattr(self.qwen_vl.model, "model", None)
        return getattr(core, "visual", None)

    def _build_images(self, batch: dict[str, Tensor]) -> Tensor:
        import torch.nn.functional as F

        batch = expand_tactile_as_image_window(batch, self.config)
        expected_keys = self.config.image_feature_keys_expanded()
        present_keys = [key for key in expected_keys if key in batch]
        if not present_keys:
            raise ValueError(
                f"No VLM image keys present. Expected one of {expected_keys}, "
                f"got {list(batch)}"
            )

        device = next(self.parameters()).device
        target_size = (self.qwen_vl._target_h, self.qwen_vl._target_w)
        views: list[Tensor] = []
        for key in present_keys:
            camera = batch[key]
            if camera.dim() != 4:
                raise ValueError(
                    f"Expected camera '{key}' as [B, C, H, W], got {tuple(camera.shape)}"
                )
            camera = camera.to(device=device, dtype=torch.float32, non_blocking=True)
            if camera.shape[-2:] != target_size:
                camera = F.interpolate(
                    camera,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            views.append(camera)
        return torch.stack(views, dim=1)

    @staticmethod
    def _get_instructions(batch: dict[str, Tensor], batch_size: int) -> list[str]:
        tasks = batch.get("task")
        if tasks is None:
            return [""] * batch_size
        if isinstance(tasks, str):
            tasks = [tasks]
        return list(tasks)

    def _get_state(self, batch: dict[str, Tensor], device, dtype) -> Tensor | None:
        if self.config.state_mode == "none" or self.state_dim == 0 or OBS_STATE not in batch:
            return None
        state = batch[OBS_STATE].to(device=device, dtype=dtype)
        return state.unsqueeze(1) if state.dim() == 2 else state

    def _encode_prefix(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor] | None]:
        clean_images = self._build_images(batch)
        batch_size, num_views = clean_images.shape[:2]
        instructions = self._get_instructions(batch, batch_size)

        alignment_active = (
            self.training
            and self.config.dino_alignment_weight > 0.0
            and torch.is_grad_enabled()
        )
        if alignment_active and self.dino_teacher is None:
            raise RuntimeError(
                "DINO alignment is enabled but the teacher is not loaded. Construct this policy "
                "through make_policy(..., for_training=True) and set dinov3_checkpoint."
            )

        student_images = clean_images
        if alignment_active and self.config.dino_light_augmentation:
            student_images = self.illumination_augment(clean_images)

        tactile_tokens = None
        if self.tactile_encoder is not None:
            tactile_tokens = self.tactile_encoder.forward_flat(batch)
        encoder_tactile = (
            tactile_tokens
            if tactile_tokens is not None and self.config.tactile_insert_location == "encoder"
            else None
        )
        last_hidden, attention_mask, qwen_image_tokens = (
            self.qwen_vl.forward_with_image_features(
                student_images, instructions, extra_embeds=encoder_tactile
            )
        )

        if tactile_tokens is not None and encoder_tactile is None:
            tactile_tokens = tactile_tokens.to(last_hidden.device, last_hidden.dtype)
            last_hidden = torch.cat((last_hidden, tactile_tokens), dim=1)
            tactile_mask = torch.ones(
                tactile_tokens.shape[:2], dtype=torch.bool, device=attention_mask.device
            )
            attention_mask = torch.cat((attention_mask, tactile_mask), dim=1)

        alignment_losses = None
        if alignment_active:
            output_grid = (
                self.qwen_vl._grid_h // self.qwen_vl._merge_size,
                self.qwen_vl._grid_w // self.qwen_vl._merge_size,
            )
            flat_clean = clean_images.reshape(-1, *clean_images.shape[-3:])
            dino_tokens = self.dino_teacher(flat_clean, output_grid=output_grid)
            qwen_tokens = qwen_image_tokens.reshape(
                batch_size, num_views, qwen_image_tokens.shape[1], qwen_image_tokens.shape[2]
            )
            dino_tokens = dino_tokens.reshape(
                batch_size, num_views, dino_tokens.shape[1], dino_tokens.shape[2]
            )
            alignment_losses = self.dino_alignment_head(
                qwen_tokens,
                dino_tokens,
                global_loss_weight=self.config.dino_global_loss_weight,
            )
        return last_hidden, attention_mask, alignment_losses

    def _alignment_weight(self) -> float:
        warmup = self.config.dino_alignment_warmup_steps
        if warmup == 0:
            return self.config.dino_alignment_weight
        progress = min((int(self.dino_alignment_step.item()) + 1) / warmup, 1.0)
        return self.config.dino_alignment_weight * progress

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        last_hidden, attention_mask, alignment_losses = self._encode_prefix(batch)
        device_type = last_hidden.device.type
        autocast = (
            torch.autocast("cuda", dtype=torch.float32)
            if device_type == "cuda"
            else nullcontext()
        )
        with autocast:
            actions = batch[ACTION].to(last_hidden.device, last_hidden.dtype)
            actions_target = actions[:, -self.config.chunk_size :, :]
            state = self._get_state(batch, last_hidden.device, last_hidden.dtype)
            action_loss = self.action_head(
                last_hidden, actions_target, state, encoder_attention_mask=attention_mask
            )

        total_loss = action_loss
        loss_dict = {"action_loss": action_loss.mean().item()}
        if alignment_losses is not None:
            alignment_loss, patch_loss, global_loss = alignment_losses
            alignment_weight = self._alignment_weight()
            total_loss = total_loss + alignment_weight * alignment_loss
            self.dino_alignment_step.add_(1)
            loss_dict.update(
                {
                    "dino_alignment_loss": alignment_loss.mean().item(),
                    "dino_patch_loss": patch_loss.mean().item(),
                    "dino_global_loss": global_loss.mean().item(),
                    "dino_alignment_weight": alignment_weight,
                }
            )

        loss_dict["loss"] = total_loss.mean().item()
        if reduction == "none":
            return total_loss, loss_dict
        return total_loss.mean(), loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        last_hidden, attention_mask, _ = self._encode_prefix(batch)
        repetitions = max(1, int(self.config.repeated_diffusion_steps))
        autocast = (
            torch.autocast("cuda", dtype=torch.float32)
            if last_hidden.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            state = self._get_state(batch, last_hidden.device, last_hidden.dtype)
            if repetitions > 1:
                batch_size = last_hidden.shape[0]
                hidden_repeated = last_hidden.repeat(repetitions, 1, 1)
                mask_repeated = attention_mask.repeat(repetitions, 1)
                state_repeated = state.repeat(repetitions, 1, 1) if state is not None else None
                predictions = self.action_head.predict_action(
                    hidden_repeated,
                    state_repeated,
                    encoder_attention_mask=mask_repeated,
                )
                predictions = predictions.view(
                    repetitions, batch_size, *predictions.shape[1:]
                ).mean(dim=0)
            else:
                predictions = self.action_head.predict_action(
                    last_hidden, state, encoder_attention_mask=attention_mask
                )
        return predictions.float()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if not self._action_queue:
            offset = self.config.action_start_offset
            actions = self.predict_action_chunk(batch)[
                :, offset : offset + self.config.n_action_steps
            ]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()

    def _get_default_peft_targets(self) -> dict:
        return {
            "target_modules": r"(.*self_attn\.(q|v)_proj)",
            "modules_to_save": ["action_head", "dino_alignment_head"],
        }
