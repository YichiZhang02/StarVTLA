from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import torch

from vtla.frameworks.fastwam.visualization import (
    save_comparison_image,
    summarize_sample_metrics,
    video_psnr,
    video_ssim,
    write_metrics,
)


def capture_samples(batch: dict[str, Any], num_samples: int) -> list[dict[str, Any]]:
    keys = {
        "video",
        "actions",
        "action_latent_idx",
        "t5_text_embeddings",
        "t5_text_mask",
        "fps",
        "padding_mask",
        "image_size",
        "num_conditional_frames",
        "dataset_name",
        "proprio",
        "future_proprio",
        "tactile_self_attn_gate",
        "rollout_data_mask",
        "world_model_sample_mask",
        "value_function_sample_mask",
        "value_function_return",
        "value_latent_idx",
        "current_proprio_latent_idx",
        "current_wrist_image_latent_idx",
        "current_image_latent_idx",
        "future_proprio_latent_idx",
        "future_wrist_image_latent_idx",
        "future_image_latent_idx",
        "future_tactile_left_latent_idx",
        "future_tactile_right_latent_idx",
        "current_sensor_latent_indices",
        "future_rgb_latent_indices",
        "future_tactile_latent_indices",
        "tactile_latent_indices",
        "slot_layout_fingerprint",
        "dataset_index",
        "task",
    }
    batch_size = next(
        (value.shape[0] for value in batch.values() if isinstance(value, torch.Tensor) and value.ndim),
        1,
    )
    samples = []
    for index in range(min(int(num_samples), int(batch_size))):
        sample: dict[str, Any] = {}
        for key in keys & set(batch):
            value = batch[key]
            if isinstance(value, torch.Tensor):
                sample[key] = value[index : index + 1].detach().cpu().clone()
            elif isinstance(value, (list, tuple)):
                sample[key] = [value[index]]
            else:
                sample[key] = value
        samples.append(sample)
    return samples


def decoded_to_unit_video(decoded: torch.Tensor) -> torch.Tensor:
    decoded = decoded.detach().float().cpu()
    if decoded.ndim != 5 or decoded.shape[1] != 3:
        raise ValueError(f"Dream-Tac decoder must return [B,3,T,H,W], got {decoded.shape}.")
    if decoded.min() < -0.05:
        decoded = (decoded + 1.0) * 0.5
    elif decoded.max() > 1.5:
        decoded = decoded / 255.0
    return decoded.clamp(0, 1)


def pixel_frames_for_slot(
    video: torch.Tensor, slot: int, temporal_compression_factor: int = 4
) -> torch.Tensor:
    start = 1 + (int(slot) - 1) * temporal_compression_factor
    return video[0, :, start : start + temporal_compression_factor].contiguous()


def save_modality_comparisons(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    step_dir: Path,
    sample_index: int,
) -> dict[str, str]:
    paths = {}
    for modality in predictions:
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", modality)
        path = step_dir / f"sample_{sample_index:03d}_{safe_name}.png"
        save_comparison_image(
            [(f"generated_{modality}", predictions[modality]), (f"ground_truth_{modality}", targets[modality])],
            path,
        )
        paths[modality] = str(path)
    return paths


def image_metrics(
    predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> dict[str, float]:
    metrics = {}
    for name in predictions:
        metrics[f"psnr_{name}"] = video_psnr(predictions[name], targets[name])
        metrics[f"ssim_{name}"] = video_ssim(predictions[name], targets[name])
    return metrics


__all__ = [
    "capture_samples",
    "decoded_to_unit_video",
    "image_metrics",
    "pixel_frames_for_slot",
    "save_modality_comparisons",
    "summarize_sample_metrics",
    "write_metrics",
]
