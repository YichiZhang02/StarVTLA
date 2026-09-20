from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from vtla.engine.configs import PreTrainedConfig
from vtla.engine.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from vtla.frameworks.pretrained import PreTrainedPolicy
from vtla.frameworks.pi05.modeling_pi05 import resize_with_pad_torch
from .configuration_n0_vtla import N0VTLAConfig
from .core import N0VTLACore


def image_to_unit(image):
    """Accept BCHW uint8 or [0,1] floats; fail instead of silently rescaling physical fields."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB image, got {tuple(image.shape)}.")
    if image.dtype == torch.uint8:
        return image.float() / 255
    image = image.float()
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("N0-VTLA images must be uint8 or finite floats in [0,1].")
    return image


class N0VTLAPolicy(PreTrainedPolicy):
    config_class = N0VTLAConfig
    name = "n0_vtla"

    def __init__(self, config: N0VTLAConfig, core_model: nn.Module | None = None,
                 initialize_base: bool = True, **kwargs):
        super().__init__(config)
        config.validate_features()
        if core_model is None and initialize_base and not config.base_model_path and not config.allow_random_init:
            raise ValueError("Set base_model_path to N0-VTLA weights, or explicitly allow_random_init=True.")
        self.model = core_model if core_model is not None else N0VTLACore(config)
        self.action_dim = int(config.action_feature.shape[0])
        if core_model is None and initialize_base and config.base_model_path:
            from .runtime import load_native_weights
            load_native_weights(self.model, config.base_model_path, config.reinitialize_action_projections)
        self.reset()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, strict=True, **kwargs):
        saved = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        if not isinstance(saved, N0VTLAConfig):
            raise ValueError("Expected a StarVTLA n0_vtla checkpoint. Import native weights with base_model_path.")
        if config is not None:
            config.validate_checkpoint_layout(saved)
            # Factory inputs still contain raw joint and derived EEF columns;
            # compare widths only after routing to the selected representation.
            config.validate_features()
            if config.action_feature is not None and saved.action_feature is not None:
                if config.action_feature.shape != saved.action_feature.shape:
                    raise ValueError("Checkpoint action feature width differs from the dataset.")
            if config.robot_state_feature is not None and saved.robot_state_feature is not None:
                if config.robot_state_feature.shape != saved.robot_state_feature.shape:
                    raise ValueError("Checkpoint state feature width differs from the dataset.")
        return super().from_pretrained(pretrained_name_or_path, config=config or saved,
                                       strict=strict, initialize_base=False, **kwargs)

    def get_optim_params(self):
        return (p for p in self.parameters() if p.requires_grad)

    def reset(self):
        self._action_queue = deque()
        self._baseline = {}

    @staticmethod
    def _view_mask(batch, key, image):
        mask = batch.get(key + "_mask")
        if mask is None:
            return torch.ones(image.shape[0], device=image.device, dtype=torch.bool)
        mask = torch.as_tensor(mask, device=image.device, dtype=torch.bool).reshape(-1)
        if mask.shape[0] != image.shape[0]:
            raise ValueError(f"Invalid per-sample mask for {key}.")
        return mask

    def _inputs(self, batch, *, training):
        images, image_masks, differences, view_masks = [], [], [], []
        for key in self.config.selected_camera_keys():
            image = image_to_unit(batch[key])
            resized = resize_with_pad_torch(image.permute(0, 2, 3, 1), *self.config.image_resolution)
            images.append(resized.permute(0, 3, 1, 2) * 2 - 1)
            image_masks.append(self._view_mask(batch, key, image))
        for key in self.config.tactile_keys:
            image = batch[key]
            if image.ndim == 5:
                if image.shape[1] != 2:
                    raise ValueError("Training tactile must contain exactly [episode baseline, current].")
                baseline, current = image[:, 0], image[:, 1]
            else:
                current = image
                baseline = batch.get(key + ".baseline")
                if baseline is None:
                    if training:
                        raise ValueError("Training requires the exact episode tactile baseline; use make_dataset().")
                    if key not in self._baseline:
                        self._baseline[key] = current.detach().clone()
                    baseline = self._baseline[key]
            if baseline.shape != current.shape:
                raise ValueError("Tactile baseline shape changed; reset policy at episode boundaries.")
            if not training:
                self._baseline[key] = baseline.detach().clone()
            current, baseline = image_to_unit(current), image_to_unit(baseline)
            # Native Observation converts images to [-1,1] BEFORE subtraction.
            diff = 2 * (current - baseline.to(current.device))
            differences.append(F.interpolate(diff, size=(self.config.tactile_image_size,) * 2,
                                             mode="bilinear", align_corners=False))
            view_masks.append(self._view_mask(batch, key, current))
        return (images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
                differences, view_masks)

    def forward(self, batch, reduction="mean", noise=None, time=None):
        actions = batch[ACTION]
        if actions.ndim != 3 or actions.shape[1:] != (self.config.chunk_size, self.action_dim):
            raise ValueError("Actions must have shape [B, chunk_size, active action dimension].")
        padded = F.pad(actions, (0, self.config.max_action_dim - self.action_dim))
        losses = self.model(*self._inputs(batch, training=True), padded, noise=noise, time=time)
        losses = losses[..., :self.action_dim]
        valid = torch.ones_like(losses, dtype=torch.bool)
        if "action_is_pad" in batch:
            valid &= ~batch["action_is_pad"].bool().unsqueeze(-1)
        if "action_mask" in batch:
            mask = batch["action_mask"].bool()
            if mask.ndim == 2:
                mask = mask[:, None, :]
            valid &= mask[..., :self.action_dim]
        count = valid.sum((1, 2))
        if (count == 0).any():
            raise ValueError("A training sample has no valid action targets.")
        per_sample = losses.masked_fill(~valid, 0).sum((1, 2)) / count
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be mean or none.")
        return (per_sample.mean() if reduction == "mean" else per_sample,
                {"action_loss": per_sample.mean().detach()})

    @torch.no_grad()
    def predict_action_chunk(self, batch, noise=None, **kwargs):
        self.eval()
        return self.model.sample_actions(*self._inputs(batch, training=False), noise=noise)[..., :self.action_dim]

    @torch.no_grad()
    def select_action(self, batch, **kwargs):
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)
            start = self.config.action_start_offset
            self._action_queue.extend(chunk[:, start:start + self.config.n_action_steps].transpose(0, 1))
        return self._action_queue.popleft()
