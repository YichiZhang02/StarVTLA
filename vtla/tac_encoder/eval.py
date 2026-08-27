"""Validation and reconstruction visualization shared by all backbones."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def evaluate_reconstruction(model, loader, device, mask_ratio: float, autocast_dtype=None) -> dict:
    model.eval()
    total = 0.0
    count = 0
    member_loss = defaultdict(float)
    member_count = defaultdict(int)
    device_type = device.type
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        members = batch["member_index"].to(device)
        for member in members.unique(sorted=True):
            selected = images[members == member]
            with torch.autocast(
                device_type=device_type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                output = model.forward_reconstruction(selected, mask_ratio)
            sample_count = len(selected)
            value = float(output.loss.detach())
            total += value * sample_count
            count += sample_count
            member_index = int(member.item())
            member_loss[member_index] += value * sample_count
            member_count[member_index] += sample_count
    return {
        "loss": total / max(count, 1),
        "samples": count,
        "member_loss": {
            str(member): member_loss[member] / member_count[member] for member in sorted(member_count)
        },
    }


def select_visualization_indices(dataset, per_level: int = 1) -> list[int]:
    """Select deterministic low/mid/high anchor-contact examples from mmap metadata."""
    candidates = []
    for offset, child in zip(dataset.offsets, dataset.datasets, strict=True):
        window_rows = child.indices
        anchors = child.cache.arrays["window_anchor.npy"][window_rows]
        scores = child.cache.arrays["contact_scores.npy"][anchors].max(axis=1)
        candidates.extend((float(score), offset + local) for local, score in enumerate(scores))
    if not candidates:
        return []
    values = np.asarray([item[0] for item in candidates])
    selected = []
    available = np.ones(len(candidates), dtype=np.bool_)
    for quantile in (0.1, 0.5, 0.9):
        target = float(np.quantile(values, quantile))
        order = np.argsort(np.abs(values - target))
        taken = 0
        for position in order:
            if available[position]:
                selected.append(candidates[position][1])
                available[position] = False
                taken += 1
                if taken == per_level:
                    break
    return selected


def _pixel_mask(model, mask: Tensor) -> Tensor:
    if model.tubelet_size == 1:
        patch_values = mask.unsqueeze(-1).expand(*mask.shape, model.patch_size**2 * 3)
        return model.unpatchify(patch_values)
    # [B,S,T',H',W'] -> [B,S,T,C,H,W]
    result = mask.repeat_interleave(model.tubelet_size, dim=2)
    result = result.repeat_interleave(model.patch_size, dim=3).repeat_interleave(model.patch_size, dim=4)
    return result.unsqueeze(3).expand(-1, -1, -1, 3, -1, -1)


@torch.no_grad()
def save_reconstruction_visualization(
    model,
    dataset,
    indices: Iterable[int],
    destination: str | Path,
    device: torch.device,
    mask_ratio: float,
    autocast_dtype=None,
) -> None:
    indices = list(indices)
    if not indices:
        return
    samples = [dataset[index] for index in indices]
    images = torch.stack([sample["images"] for sample in samples]).to(device)
    model.eval()
    fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(0)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            output = model.forward_reconstruction(images, mask_ratio)
    reconstruction = output.reconstruction.float()
    pixel_mask = _pixel_mask(model, output.mask).float()
    masked = images * (1 - pixel_mask)
    error = (images - reconstruction).abs().mean(dim=3, keepdim=True).expand_as(images)

    rows = images.shape[0] * images.shape[1] * images.shape[2]
    figure, axes = plt.subplots(rows, 4, figsize=(12, max(2.0, rows * 2.2)), squeeze=False)
    titles = ("original", "masked", "reconstruction", "error")
    tensors = (images, masked, reconstruction, error)
    row = 0
    for sample_index in range(images.shape[0]):
        for sensor in range(images.shape[1]):
            for time in range(images.shape[2]):
                for column, tensor in enumerate(tensors):
                    image = tensor[sample_index, sensor, time].permute(1, 2, 0).cpu().clamp(0, 1)
                    axes[row, column].imshow(image)
                    axes[row, column].axis("off")
                    if row == 0:
                        axes[row, column].set_title(titles[column])
                axes[row, 0].set_ylabel(f"n{sample_index} s{sensor} t{time}")
                row += 1
    figure.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120)
    plt.close(figure)
