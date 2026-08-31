from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from vtla.engine.utils.constants import ACTION
from vtla.frameworks.pretrained import PreTrainedPolicy

from .configuration_dream_tac import DreamTacConfig
from .runtime import load_dream_tac_core


class DreamTacPolicy(PreTrainedPolicy):
    config_class = DreamTacConfig
    name = "dream_tac"

    def __init__(
        self,
        config: DreamTacConfig,
        dataset_stats: dict[str, Any] | None = None,
        core_model: nn.Module | None = None,
        **_: Any,
    ) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.dataset_stats = dataset_stats
        self.action_dim = int(config.action_feature.shape[0])
        self.layout = config.slot_layout()
        self.core = core_model or load_dream_tac_core(config)
        self._iteration = 0
        self.reset()

    def get_optim_params(self):
        return (parameter for parameter in self.core.parameters() if parameter.requires_grad)

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque()

    def _core_autocast(self, reference: Tensor):
        """Match Cosmos activations to the precision selected by its core config."""
        precision = getattr(self.core, "precision", None)
        device_type = reference.device.type
        supported = precision in {torch.bfloat16, torch.float16} and (
            device_type == "cuda" or (device_type == "cpu" and precision == torch.bfloat16)
        )
        if not supported:
            return nullcontext()
        return torch.autocast(device_type=device_type, dtype=precision)

    @staticmethod
    def _slot_loss(per_frame: Tensor, indices: Tensor | Any) -> Tensor:
        indices = torch.as_tensor(indices, device=per_frame.device).long()
        if indices.ndim == 1:
            indices = indices.unsqueeze(1)
        if indices.shape[1] == 0:
            return per_frame.new_zeros(())
        rows = torch.arange(per_frame.shape[0], device=per_frame.device).unsqueeze(1)
        return per_frame[rows, indices].mean()

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        with self._core_autocast(batch["video"]):
            output, original_loss = self.core.training_step(batch, iteration=self._iteration)
        self._iteration += 1
        per_frame = output.get("edm_loss_per_frame")
        if per_frame is None:
            raise KeyError("Dream-Tac core.training_step() did not return edm_loss_per_frame.")
        if per_frame.ndim != 2:
            raise ValueError(f"Expected Dream-Tac EDM loss [B,T], got {tuple(per_frame.shape)}.")

        action_loss = self._slot_loss(per_frame, batch["action_latent_idx"])
        state_indices = (
            batch["future_proprio_latent_idx"]
            if self.layout.future_state_index is not None
            else torch.empty(per_frame.shape[0], 0, dtype=torch.long, device=per_frame.device)
        )
        state_loss = self._slot_loss(per_frame, state_indices)
        rgb_loss = self._slot_loss(per_frame, batch["future_rgb_latent_indices"])
        tactile_loss = self._slot_loss(per_frame, batch["future_tactile_latent_indices"])
        weights = self.config.loss_weights()
        loss = (
            weights["action"] * action_loss
            + weights["rgb"] * rgb_loss
            + weights["tactile"] * tactile_loss
            + weights["state"] * state_loss
        )
        return loss, {
            "action_loss": action_loss.detach(),
            "video_loss": rgb_loss.detach(),
            "tactile_loss": tactile_loss.detach(),
            "state_loss": state_loss.detach(),
            "edm_loss": original_loss.detach(),
        }

    @staticmethod
    def extract_action_chunk(
        latent: Tensor,
        action_indices: Tensor,
        chunk_size: int = 20,
        action_dim: int | None = None,
    ) -> Tensor:
        if action_dim is None or action_dim <= 0:
            raise ValueError("Dream-Tac action_dim must be provided by the active policy feature.")
        rows = torch.arange(latent.shape[0], device=latent.device)
        action_indices = torch.as_tensor(action_indices, device=latent.device).long()
        frame = latent[rows, :, action_indices, :, :].flatten(1)
        elements = chunk_size * action_dim
        repeats = frame.shape[1] // elements
        if repeats < 1:
            raise ValueError(
                f"Dream-Tac action latent has {frame.shape[1]} values, fewer than required {elements}."
            )
        return frame[:, : repeats * elements].reshape(
            latent.shape[0], repeats, chunk_size, action_dim
        ).mean(1)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        generation_kwargs: dict[str, Any] = {
            "n_sample": int(batch["video"].shape[0]),
            "num_steps": self.config.num_inference_steps,
            "seed": self.config.inference_seed,
            "is_negative_prompt": False,
        }
        if noise is not None:
            generation_kwargs["x_sigma_max"] = noise
        with self._core_autocast(batch["video"]):
            generated = self.core.generate_samples_from_batch(dict(batch), **generation_kwargs)
        if isinstance(generated, tuple):
            generated = generated[0]
        normalized = self.extract_action_chunk(
            generated,
            torch.as_tensor(batch["action_latent_idx"], device=generated.device),
            self.config.chunk_size,
            self.action_dim,
        )
        return normalized.float()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        batch = dict(batch)
        batch.pop(ACTION, None)
        batch.pop("actions", None)
        if not self._action_queue:
            actions = self.predict_action_chunk(batch, **kwargs)
            start = self.config.action_start_offset
            stop = start + self.config.n_action_steps
            self._action_queue.extend(actions[:, start:stop].transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def generate_training_visualization(
        self,
        sample: dict[str, Any],
        output_dir: Path,
        step: int,
        sample_index: int = 0,
    ) -> dict[str, Any]:
        from .visualization import (
            decoded_to_unit_video,
            image_metrics,
            pixel_frames_for_slot,
            save_modality_comparisons,
        )

        was_training = self.training
        self.eval()
        try:
            device_sample = {
                key: value.to(self.config.device) if isinstance(value, Tensor) else value
                for key, value in sample.items()
            }
            started = perf_counter()
            with self._core_autocast(device_sample["video"]):
                generated = self.core.generate_samples_from_batch(
                    dict(device_sample),
                    n_sample=1,
                    num_steps=self.config.visualization_num_inference_steps,
                    seed=self.config.visualization_seed,
                    is_negative_prompt=False,
                )
            if isinstance(generated, tuple):
                generated = generated[0]
            inference_s = perf_counter() - started
            with self._core_autocast(generated):
                decoded = decoded_to_unit_video(self.core.decode(generated))
            ground_truth = device_sample["video"].detach().float().cpu() / 255.0
            slot_map = {
                slot.name: slot.index
                for slot in self.layout.slots
                if slot.phase == "future" and slot.modality in {"rgb", "tactile"}
            }
            predictions = {
                name: pixel_frames_for_slot(
                    decoded, slot, self.config.temporal_compression_factor
                )
                for name, slot in slot_map.items()
            }
            targets = {
                name: pixel_frames_for_slot(
                    ground_truth, slot, self.config.temporal_compression_factor
                )
                for name, slot in slot_map.items()
            }
            normalized_action = self.extract_action_chunk(
                generated,
                device_sample["action_latent_idx"],
                self.config.chunk_size,
                self.action_dim,
            ).float().cpu()
            target_action = device_sample["actions"].float().cpu()
            action_error = normalized_action - target_action
            step_dir = Path(output_dir) / "visualizations" / f"step_{int(step):06d}"
            image_paths = save_modality_comparisons(
                predictions, targets, step_dir=step_dir, sample_index=sample_index
            )
            return {
                "step": int(step),
                "sample_index": int(sample_index),
                "seed": int(self.config.visualization_seed),
                "num_inference_steps": int(self.config.visualization_num_inference_steps),
                "inference_s": float(inference_s),
                "action_mae_normalized": float(action_error.abs().mean()),
                "action_rmse_normalized": float(action_error.square().mean().sqrt()),
                "image_paths": image_paths,
                "image_path": next(iter(image_paths.values())),
                **image_metrics(predictions, targets),
            }
        finally:
            self.train(was_training)

    @torch.no_grad()
    def generate_training_visualizations(
        self, samples: list[dict[str, Any]], output_dir: Path, step: int
    ) -> dict[str, Any]:
        from .visualization import summarize_sample_metrics, write_metrics

        metrics = [
            self.generate_training_visualization(sample, output_dir, step, index)
            for index, sample in enumerate(samples)
        ]
        summary = summarize_sample_metrics(
            metrics,
            step=step,
            seed=self.config.visualization_seed,
            num_inference_steps=self.config.visualization_num_inference_steps,
        )
        # FastWAM's summarizer intentionally aggregates PSNR/SSIM and inference time only.
        for name in ("action_mae_normalized", "action_rmse_normalized"):
            summary["aggregate"][f"mean_{name}"] = sum(float(item[name]) for item in metrics) / len(metrics)
        metrics_path = Path(output_dir) / "visualizations" / f"step_{int(step):06d}" / "metrics.json"
        write_metrics(metrics_path, summary)
        summary["metrics_path"] = str(metrics_path)
        return summary

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)
        state = {
            f"core.{name}": parameter.detach().cpu().contiguous()
            for name, parameter in self.core.named_parameters()
            if parameter.requires_grad
        }
        if not state:
            raise RuntimeError("Dream-Tac has no trainable core parameters to save.")
        save_file(state, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def _load_as_safetensor(
        cls,
        model: "DreamTacPolicy",
        model_file: str,
        map_location: str,
        strict: bool,
    ) -> "DreamTacPolicy":
        state = load_file(model_file, device=map_location)
        core_state = {name.removeprefix("core."): value for name, value in state.items() if name.startswith("core.")}
        if not core_state:
            raise ValueError(f"Dream-Tac checkpoint has no core tensors: {model_file}")
        _missing, unexpected = model.core.load_state_dict(core_state, strict=False)
        trainable_names = {name for name, parameter in model.core.named_parameters() if parameter.requires_grad}
        missing_trainable = sorted(trainable_names - set(core_state))
        if strict and (missing_trainable or unexpected):
            raise RuntimeError(
                f"Dream-Tac checkpoint mismatch; missing trainable={missing_trainable}, unexpected={unexpected}."
            )
        return model
