from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def capture_samples(
    batch: dict[str, Any],
    tactile_keys: Sequence[str],
    num_samples: int,
) -> list[dict[str, Any]]:
    """Keep deterministic, CPU-resident processed samples for the whole run."""
    keys = {
        "video",
        "action",
        "proprio",
        "context",
        "context_mask",
        "task",
        *tactile_keys,
    }
    available = next(
        (value.shape[0] for value in batch.values() if isinstance(value, torch.Tensor) and value.ndim > 0),
        1,
    )
    count = min(int(num_samples), int(available))
    samples: list[dict[str, Any]] = []
    for index in range(count):
        sample: dict[str, Any] = {}
        for key in keys:
            if key not in batch:
                continue
            value = batch[key]
            if isinstance(value, torch.Tensor):
                sample[key] = value[index : index + 1].detach().cpu().clone()
            elif isinstance(value, (list, tuple)):
                sample[key] = [value[index]]
            else:
                sample[key] = value
        samples.append(sample)
    return samples


def pil_frames_to_tensor(frames: Sequence[Image.Image]) -> torch.Tensor:
    tensors = [
        torch.from_numpy(np.asarray(frame.convert("RGB"), dtype=np.float32).copy())
        .permute(2, 0, 1)
        .div(255.0)
        for frame in frames
    ]
    return torch.stack(tensors, dim=1)


def video_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred.float() - target.float()).pow(2).mean(dim=(0, 2, 3))
    return float((10.0 * torch.log10(1.0 / (mse + 1e-8))).mean().item())


def video_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_bt = pred.float().permute(1, 0, 2, 3).contiguous()
    target_bt = target.float().permute(1, 0, 2, 3).contiguous()
    coords = torch.arange(11, dtype=torch.float32) - 5
    gaussian = torch.exp(-(coords**2) / (2.0 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    kernel = torch.outer(gaussian, gaussian).view(1, 1, 11, 11).repeat(3, 1, 1, 1)
    mu_x = F.conv2d(pred_bt, kernel, padding=5, groups=3)
    mu_y = F.conv2d(target_bt, kernel, padding=5, groups=3)
    sigma_x = F.conv2d(pred_bt.square(), kernel, padding=5, groups=3) - mu_x.square()
    sigma_y = F.conv2d(target_bt.square(), kernel, padding=5, groups=3) - mu_y.square()
    sigma_xy = F.conv2d(pred_bt * target_bt, kernel, padding=5, groups=3) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + 0.01**2) * (2 * sigma_xy + 0.03**2)
    denominator = (mu_x.square() + mu_y.square() + 0.01**2) * (
        sigma_x + sigma_y + 0.03**2
    )
    return float((numerator / (denominator + 1e-12)).mean().item())


def _load_label_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def save_comparison_image(
    streams: Sequence[tuple[str, torch.Tensor]],
    output_path: Path,
) -> None:
    """Save future frames as a labeled panel: columns are streams, rows are timesteps."""
    if not streams:
        raise ValueError("FastWAM visualization requires at least one stream.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = [stream[1] for stream in streams]
    frame_count = tensors[0].shape[1]
    if any(tensor.shape[1] != frame_count for tensor in tensors):
        raise ValueError("FastWAM visualization streams have different frame counts.")
    if frame_count <= 0:
        raise ValueError("FastWAM visualization requires at least one future frame.")

    _, _, height, width = tensors[0].shape
    if any(tuple(tensor.shape[2:]) != (height, width) for tensor in tensors):
        raise ValueError("FastWAM visualization streams have different spatial sizes.")

    gutter_width = 72
    header_height = 48
    canvas = Image.new(
        "RGB",
        (gutter_width + len(streams) * width, header_height + frame_count * height),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    header_font = _load_label_font(20)
    time_font = _load_label_font(18)

    display_names = {
        "generated": "Generated",
        "vae_reconstruction": "VAE reconstruction",
        "ground_truth": "Ground truth",
    }
    for column, (name, tensor) in enumerate(streams):
        title = display_names.get(name, name.replace("_", " "))
        left = gutter_width + column * width
        bounds = draw.textbbox((0, 0), title, font=header_font)
        title_width = bounds[2] - bounds[0]
        draw.text(
            (left + (width - title_width) // 2, 12),
            title,
            fill=(20, 20, 20),
            font=header_font,
        )
        for frame_index in range(frame_count):
            array = (
                tensor[:, frame_index]
                .permute(1, 2, 0)
                .clamp(0, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
                .numpy()
            )
            canvas.paste(Image.fromarray(array), (left, header_height + frame_index * height))

    for frame_index in range(frame_count):
        label = f"t={frame_index + 1}"
        bounds = draw.textbbox((0, 0), label, font=time_font)
        label_width = bounds[2] - bounds[0]
        label_height = bounds[3] - bounds[1]
        top = header_height + frame_index * height
        draw.text(
            ((gutter_width - label_width) // 2, top + (height - label_height) // 2),
            label,
            fill=(20, 20, 20),
            font=time_font,
        )
        if frame_index:
            draw.line((0, top, canvas.width, top), fill=(210, 210, 210), width=1)
    for column in range(len(streams) + 1):
        left = gutter_width + column * width
        draw.line((left, 0, left, canvas.height), fill=(210, 210, 210), width=1)

    canvas.save(output_path)


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def summarize_sample_metrics(
    sample_metrics: Sequence[dict[str, Any]],
    step: int,
    seed: int,
    num_inference_steps: int,
) -> dict[str, Any]:
    if not sample_metrics:
        raise ValueError("FastWAM visualization has no sample metrics to summarize.")
    metric_names = sorted(
        {
            key
            for metrics in sample_metrics
            for key, value in metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (key == "inference_s" or key.startswith(("psnr_", "ssim_")))
        }
    )
    aggregate = {}
    for name in metric_names:
        values = [float(metrics[name]) for metrics in sample_metrics if name in metrics]
        if values and all(math.isfinite(value) for value in values):
            aggregate[f"mean_{name}"] = sum(values) / len(values)
    aggregate["total_inference_s"] = sum(
        float(metrics.get("inference_s", 0.0)) for metrics in sample_metrics
    )
    return {
        "step": int(step),
        "seed": int(seed),
        "num_inference_steps": int(num_inference_steps),
        "num_samples": len(sample_metrics),
        "aggregate": aggregate,
        "samples": list(sample_metrics),
    }


def timed_inference(callable_):
    start = perf_counter()
    result = callable_()
    return result, perf_counter() - start
