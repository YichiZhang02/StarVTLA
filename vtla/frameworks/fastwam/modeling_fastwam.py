from __future__ import annotations

from collections import deque
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
import torch.nn.functional as F

from vtla.engine.utils.constants import ACTION

from ..pretrained import PreTrainedPolicy
from ..tactile_encode import TactileEncoder
from .configuration_fastwam import FastWAMConfig
from .core.fastwam import FastWAM


class FastWAMTactileVAEContextEncoder(nn.Module):
    """Encode tactile history as frozen Wan VAE latents projected to context tokens."""

    def __init__(self, config: FastWAMConfig, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.tactile_keys = list(config.tactile_windowed_keys())
        if not self.tactile_keys:
            raise ValueError("FastWAM tactile_mode='as_image' requires tactile_keys.")
        self.image_size = tuple(int(size) for size in config.camera_image_size)
        self.proj = nn.Linear(int(latent_dim), int(output_dim))
        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        self.proj.to(device=config.device, dtype=dtype)

    def _prepare_video(self, batch: dict[str, Tensor]) -> Tensor:
        missing = [key for key in self.tactile_keys if key not in batch]
        if missing:
            raise ValueError(f"FastWAM tactile VAE context is missing keys: {missing}.")
        cameras = []
        for key in self.tactile_keys:
            image = torch.as_tensor(batch[key])
            if image.ndim == 4:
                image = image.unsqueeze(1)
            if image.ndim != 5:
                raise ValueError(
                    "FastWAM tactile VAE input must be [B,T,C,H,W] or [B,C,H,W], "
                    f"got {tuple(image.shape)} for {key!r}."
                )
            if image.shape[2] != 3:
                raise ValueError(f"FastWAM tactile images must have 3 channels, got {image.shape[2]}.")
            original_dtype = image.dtype
            image = image.float()
            flat = F.interpolate(
                image.flatten(0, 1),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
            image = flat.unflatten(0, (image.shape[0], image.shape[1]))
            if original_dtype == torch.uint8:
                image = image / 127.5 - 1.0
            else:
                image = image * 2.0 - 1.0
            cameras.append(image)
        tactile_video = torch.cat(cameras, dim=-1)
        return tactile_video.permute(0, 2, 1, 3, 4).contiguous()

    def forward_flat(self, batch: dict[str, Tensor], fastwam: FastWAM) -> Tensor:
        tactile_video = self._prepare_video(batch).to(
            device=fastwam.device,
            dtype=fastwam.torch_dtype,
        )
        with torch.no_grad():
            latents = fastwam._encode_video_latents(tactile_video)
        tokens = latents.permute(0, 2, 3, 4, 1).flatten(1, 3)
        return self.proj(tokens.to(dtype=self.proj.weight.dtype))


class FastWAMPolicy(PreTrainedPolicy):
    config_class = FastWAMConfig
    name = "fastwam"

    def __init__(
        self,
        config: FastWAMConfig,
        dataset_stats: dict[str, Any] | None = None,
        core_model: nn.Module | None = None,
        tactile_encoder: nn.Module | None = None,
        **_: Any,
    ) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.dataset_stats = dataset_stats
        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        action_dim = int(config.action_feature.shape[0])
        state_feature = config.robot_state_feature
        proprio_dim = None if state_feature is None else int(state_feature.shape[0])

        self.fastwam = core_model or FastWAM.from_wan22_pretrained(
            device=str(config.device),
            torch_dtype=dtype,
            model_id=config.model_id,
            tokenizer_model_id=config.tokenizer_model_id,
            tokenizer_max_len=config.context_len,
            load_text_encoder=config.load_text_encoder,
            proprio_dim=proprio_dim,
            video_dit_config={**config.video_dit_config, "action_dim": action_dim},
            action_dit_config={**config.action_dit_config, "action_dim": action_dim},
            action_dit_pretrained_path=self._path_str(config.action_dit_pretrained_path),
            video_dit_pretrained_path=self._path_str(config.video_dit_pretrained_path),
            vae_pretrained_path=self._path_str(config.vae_pretrained_path),
            text_encoder_pretrained_path=self._path_str(config.text_encoder_pretrained_path),
            tokenizer_pretrained_path=self._path_str(config.tokenizer_pretrained_path),
            skip_dit_load_from_pretrain=config.skip_dit_load_from_pretrain,
            mot_checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
            video_train_shift=config.video_train_shift,
            video_infer_shift=config.video_infer_shift,
            video_num_train_timesteps=config.video_num_train_timesteps,
            action_train_shift=config.action_train_shift,
            action_infer_shift=config.action_infer_shift,
            action_num_train_timesteps=config.action_num_train_timesteps,
            loss_lambda_video=config.loss_lambda_video,
            loss_lambda_action=config.loss_lambda_action,
        )
        self.tactile_encoder = None
        if config.tactile_mode == "encode":
            self.tactile_encoder = tactile_encoder or TactileEncoder(config, config.text_dim)
        elif config.tactile_mode == "as_image":
            self.tactile_encoder = tactile_encoder or FastWAMTactileVAEContextEncoder(
                config,
                latent_dim=int(self.fastwam.vae.z_dim),
                output_dim=config.text_dim,
            )
        self._freeze_conditioning_modules()
        self.reset()

    @staticmethod
    def _path_str(path: Path | str | None) -> str | None:
        return None if path is None else str(path)

    def _freeze_conditioning_modules(self) -> None:
        for name in ("vae", "text_encoder"):
            module = getattr(self.fastwam, name, None)
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def train(self, mode: bool = True) -> "FastWAMPolicy":
        super().train(mode)
        self._freeze_conditioning_modules()
        return self

    def get_optim_params(self):
        groups = [self.fastwam.mot.parameters()]
        proprio_encoder = getattr(self.fastwam, "proprio_encoder", None)
        if proprio_encoder is not None:
            groups.append(proprio_encoder.parameters())
        if self.tactile_encoder is not None:
            groups.append(self.tactile_encoder.parameters())
        return (parameter for parameter in chain.from_iterable(groups) if parameter.requires_grad)

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque()

    def _with_text_context(self, sample: dict[str, Any]) -> dict[str, Any]:
        tasks = sample.get("task")
        online_text = (
            getattr(self.fastwam, "text_encoder", None) is not None
            and getattr(self.fastwam, "tokenizer", None) is not None
        )
        if online_text:
            if tasks is None:
                raise ValueError("FastWAM online text encoding requires task text.")
            if isinstance(tasks, str):
                tasks = [tasks]
            prompts = [self.config.prompt_template.format(task=str(task)) for task in tasks]
            context, context_mask = self.fastwam.encode_prompt(prompts)
            return {**sample, "context": context, "context_mask": context_mask}
        if "context" in sample and "context_mask" in sample:
            return sample
        raise ValueError(
            "FastWAM has neither cached context nor a loaded text encoder/tokenizer for task text."
        )

    def _with_tactile_context(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.tactile_encoder is None:
            return sample
        if self.config.tactile_mode == "as_image":
            tactile_context = self.tactile_encoder.forward_flat(sample, self.fastwam)
        else:
            tactile_context = self.tactile_encoder.forward_flat(sample)
        tactile_context_mask = torch.ones(
            tactile_context.shape[:2],
            dtype=torch.bool,
            device=tactile_context.device,
        )
        return {
            **sample,
            "tactile_context": tactile_context,
            "tactile_context_mask": tactile_context_mask,
            "tactile_insert_location": self.config.tactile_insert_location,
        }

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        sample = self._with_tactile_context(self._with_text_context(dict(batch)))
        return self.fastwam(sample)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        del noise
        sample = self._with_tactile_context(self._with_text_context(dict(batch)))
        video = sample["video"]
        proprio = sample.get("proprio")
        if video.ndim != 5:
            raise ValueError(f"FastWAM video must be [B,3,T,H,W], got {video.shape}.")
        if proprio is not None and proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)

        chunks = []
        for index in range(video.shape[0]):
            result = self.fastwam.infer_action(
                prompt=None,
                input_image=video[index : index + 1, :, 0],
                action_horizon=self.config.chunk_size,
                proprio=None if proprio is None else proprio[index : index + 1, 0],
                context=sample["context"][index : index + 1],
                context_mask=sample["context_mask"][index : index + 1],
                tactile_context=(
                    None
                    if "tactile_context" not in sample
                    else sample["tactile_context"][index : index + 1]
                ),
                tactile_context_mask=(
                    None
                    if "tactile_context_mask" not in sample
                    else sample["tactile_context_mask"][index : index + 1]
                ),
                tactile_insert_location=self.config.tactile_insert_location,
                num_inference_steps=self.config.num_inference_steps,
            )
            chunks.append(result["action"])
        return torch.stack(chunks)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        batch = dict(batch)
        batch.pop(ACTION, None)
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
            pil_frames_to_tensor,
            save_comparison_image,
            timed_inference,
            video_psnr,
            video_ssim,
        )

        was_training = self.training
        self.eval()
        try:
            device_sample = {
                key: value.to(self.config.device) if isinstance(value, torch.Tensor) else value
                for key, value in sample.items()
            }
            conditioned = self._with_tactile_context(self._with_text_context(device_sample))
            video = conditioned["video"]
            proprio = conditioned.get("proprio")
            if video.ndim != 5 or video.shape[0] != 1:
                raise ValueError(
                    "FastWAM visualization expects one processed video [1,3,T,H,W], "
                    f"got {tuple(video.shape)}."
                )

            infer_kwargs = {
                "prompt": None,
                "input_image": video[:, :, 0],
                "num_frames": int(video.shape[2]),
                "action": None,
                "action_horizon": int(self.config.chunk_size),
                "proprio": None if proprio is None else proprio[0, 0],
                "context": conditioned["context"],
                "context_mask": conditioned["context_mask"],
                "tactile_context": conditioned.get("tactile_context"),
                "tactile_context_mask": conditioned.get("tactile_context_mask"),
                "tactile_insert_location": self.config.tactile_insert_location,
                "num_inference_steps": self.config.visualization_num_inference_steps,
                "seed": self.config.visualization_seed,
                "tiled": False,
            }
            result, inference_s = timed_inference(lambda: self.fastwam.infer(**infer_kwargs))
            pred = pil_frames_to_tensor(result["video"])
            gt = ((video[0].detach().float().cpu().clamp(-1, 1) + 1.0) * 0.5).contiguous()
            if pred.shape != gt.shape:
                raise ValueError(
                    f"FastWAM visualization prediction/GT shape mismatch: {pred.shape} vs {gt.shape}."
                )
            if pred.shape[1] <= 1:
                raise ValueError("FastWAM visualization requires at least one predicted future frame.")
            pred = pred[:, 1:]
            gt = gt[:, 1:]

            streams: list[tuple[str, Tensor]] = [("generated", pred)]
            metrics: dict[str, Any] = {
                "step": int(step),
                "seed": int(self.config.visualization_seed),
                "num_inference_steps": int(self.config.visualization_num_inference_steps),
                "inference_s": float(inference_s),
                "psnr_generated_vs_gt": video_psnr(pred, gt),
                "ssim_generated_vs_gt": video_ssim(pred, gt),
            }
            if self.config.visualization_include_vae_reconstruction:
                gt_latents = self.fastwam._encode_video_latents(
                    video.to(device=self.fastwam.device, dtype=self.fastwam.torch_dtype)
                )
                vae_reconstruction = pil_frames_to_tensor(
                    self.fastwam._decode_latents(gt_latents, tiled=False)
                )
                if vae_reconstruction.shape != gt.shape:
                    # `gt` has already dropped the conditioning frame.
                    vae_reconstruction = vae_reconstruction[:, 1:]
                else:
                    # Retained for fake/test cores that may already return future frames only.
                    vae_reconstruction = vae_reconstruction
                if vae_reconstruction.shape != gt.shape:
                    raise ValueError(
                        "FastWAM VAE reconstruction/GT shape mismatch: "
                        f"{vae_reconstruction.shape} vs {gt.shape}."
                    )
                streams.append(("vae_reconstruction", vae_reconstruction))
                metrics.update(
                    {
                        "psnr_generated_vs_vae": video_psnr(pred, vae_reconstruction),
                        "ssim_generated_vs_vae": video_ssim(pred, vae_reconstruction),
                        "psnr_vae_vs_gt": video_psnr(vae_reconstruction, gt),
                        "ssim_vae_vs_gt": video_ssim(vae_reconstruction, gt),
                    }
                )
            streams.append(("ground_truth", gt))

            step_dir = Path(output_dir) / "visualizations" / f"step_{int(step):06d}"
            image_path = step_dir / f"sample_{int(sample_index):03d}.png"
            save_comparison_image(streams, image_path)
            metrics.update(
                {
                    "image_path": str(image_path),
                    "sample_index": int(sample_index),
                    "num_future_frames": int(gt.shape[1]),
                    "panels": [name for name, _ in streams],
                }
            )
            return metrics
        finally:
            self.train(was_training)

    @torch.no_grad()
    def generate_training_visualizations(
        self,
        samples: list[dict[str, Any]],
        output_dir: Path,
        step: int,
    ) -> dict[str, Any]:
        from .visualization import summarize_sample_metrics, write_metrics

        sample_metrics = [
            self.generate_training_visualization(
                sample,
                output_dir=output_dir,
                step=step,
                sample_index=sample_index,
            )
            for sample_index, sample in enumerate(samples)
        ]
        summary = summarize_sample_metrics(
            sample_metrics,
            step=step,
            seed=self.config.visualization_seed,
            num_inference_steps=self.config.visualization_num_inference_steps,
        )
        step_dir = Path(output_dir) / "visualizations" / f"step_{int(step):06d}"
        metrics_path = step_dir / "metrics.json"
        write_metrics(metrics_path, summary)
        summary["metrics_path"] = str(metrics_path)
        return summary

    def _save_pretrained(self, save_directory: Path) -> None:
        runtime_load_text_encoder = self.config.load_text_encoder
        self.config.load_text_encoder = True
        try:
            # Policy checkpoints are inference artifacts; training resume restores its
            # separate train_config and forces this flag off before model construction.
            self.config._save_pretrained(save_directory)
        finally:
            self.config.load_text_encoder = runtime_load_text_encoder
        state = {
            f"mot.{key}": value.detach().cpu().contiguous()
            for key, value in self.fastwam.mot.state_dict().items()
        }
        proprio_encoder = getattr(self.fastwam, "proprio_encoder", None)
        if proprio_encoder is not None:
            state.update(
                {
                    f"proprio_encoder.{key}": value.detach().cpu().contiguous()
                    for key, value in proprio_encoder.state_dict().items()
                }
            )
        if self.tactile_encoder is not None:
            state.update(
                {
                    f"tactile_encoder.{key}": value.detach().cpu().contiguous()
                    for key, value in self.tactile_encoder.state_dict().items()
                }
            )
        save_file(state, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def _load_as_safetensor(
        cls,
        model: "FastWAMPolicy",
        model_file: str,
        map_location: str,
        strict: bool,
    ) -> "FastWAMPolicy":
        state = load_file(model_file, device=map_location)
        mot_state = {key.removeprefix("mot."): value for key, value in state.items() if key.startswith("mot.")}
        if not mot_state:
            raise ValueError(f"FastWAM checkpoint contains no `mot.*` tensors: {model_file}")
        model.fastwam.mot.load_state_dict(mot_state, strict=strict)
        proprio_state = {
            key.removeprefix("proprio_encoder."): value
            for key, value in state.items()
            if key.startswith("proprio_encoder.")
        }
        if proprio_state and model.fastwam.proprio_encoder is not None:
            model.fastwam.proprio_encoder.load_state_dict(proprio_state, strict=strict)
        tactile_state = {
            key.removeprefix("tactile_encoder."): value
            for key, value in state.items()
            if key.startswith("tactile_encoder.")
        }
        if tactile_state and model.tactile_encoder is not None:
            model.tactile_encoder.load_state_dict(tactile_state, strict=strict)
        return model
