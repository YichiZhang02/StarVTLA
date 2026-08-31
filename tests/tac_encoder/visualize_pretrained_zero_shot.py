"""Manual zero-shot visualization for the three original pretrained backbones.

Run from the repository root:

    CUDA_VISIBLE_DEVICES=0 python -m tests.tac_encoder.visualize_pretrained_zero_shot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from vtla.tac_encoder.data.npy_tactile_dataset import (
    build_training_dataset,
    resolve_tactile_dataset,
)
from vtla.tac_encoder.eval import select_visualization_indices
from vtla.tac_encoder.common.training import require_fully_pretrained
from vtla.tac_encoder.registry import build_backbone, get_training_recipe


PRETRAINED = {
    "anytouch1": Path("playground/pretrained_models/AnyTouch-ViT-L-16/checkpoint.pth"),
    "anytouch2": Path("playground/pretrained_models/AnyTouch2-Model/checkpoint-4frames.pth"),
    "sparsh_vjepa": Path(
        "playground/pretrained_models/Sparsh-VJEPA-Small/vjepa_vitsmall.safetensors"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="backbone_training_data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("playground/results/backbones/pretrained_zero_shot_visualization"),
    )
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--vis-per-level", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _build(model_id: str):
    kwargs = {
        "num_frames": 4,
        "image_size": 224,
        "pretrained_path": str(PRETRAINED[model_id]),
    }
    if model_id == "anytouch1":
        kwargs["arch"] = "vit_l"
    model = build_backbone(model_id, **kwargs)
    if model_id != "sparsh_vjepa":
        require_fully_pretrained(model, model.load_report)
    return model


def _latent_pca_rgb(first: torch.Tensor, second: torch.Tensor):
    height, width, feature_dim = first.shape
    joined = torch.cat([first.reshape(-1, feature_dim), second.reshape(-1, feature_dim)])
    joined = joined.float().cpu()
    joined -= joined.mean(dim=0, keepdim=True)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        _, _, basis = torch.pca_lowrank(joined, q=3, center=False)
    projected = joined @ basis
    low = torch.quantile(projected, 0.01, dim=0)
    high = torch.quantile(projected, 0.99, dim=0)
    projected = ((projected - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    split = height * width
    return (
        projected[:split].reshape(height, width, 3).numpy(),
        projected[split:].reshape(height, width, 3).numpy(),
    )


@torch.inference_mode()
def _save_sparsh_latents(model, dataset, indices, destination: Path, device) -> None:
    images = torch.stack([dataset[index]["images"] for index in indices]).to(device)
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        latents = model.encode_features(images).spatial_grid.float()

    rows = images.shape[0] * images.shape[1]
    figure, axes = plt.subplots(rows, 5, figsize=(15, max(2.4, rows * 2.6)), squeeze=False)
    titles = ("input t0", "input t1", "latent PCA t0", "latent PCA t1", "latent change")
    row = 0
    for sample in range(images.shape[0]):
        for sensor in range(images.shape[1]):
            input_t0 = images[sample, sensor, :2].mean(dim=0).permute(1, 2, 0).cpu()
            input_t1 = images[sample, sensor, 2:].mean(dim=0).permute(1, 2, 0).cpu()
            latent_t0 = latents[sample, sensor, 0]
            latent_t1 = latents[sample, sensor, 1]
            pca_t0, pca_t1 = _latent_pca_rgb(latent_t0, latent_t1)
            change = (1.0 - F.cosine_similarity(latent_t0, latent_t1, dim=-1)).cpu()
            panels = (input_t0.clamp(0, 1), input_t1.clamp(0, 1), pca_t0, pca_t1)
            for column, panel in enumerate(panels):
                axes[row, column].imshow(panel)
            axes[row, 4].imshow(
                change,
                cmap="magma",
                vmin=0,
                vmax=max(0.25, float(change.max())),
            )
            axes[row, 0].set_ylabel(f"n{sample} s{sensor}")
            for column in range(5):
                axes[row, column].axis("off")
                if row == 0:
                    axes[row, column].set_title(titles[column])
            row += 1
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    resolved = resolve_tactile_dataset(args.dataset_id)
    dataset = build_training_dataset(resolved, split="val")
    indices = select_visualization_indices(dataset, args.vis_per_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "visualization_indices.json").write_text(
        json.dumps({"indices": indices}, indent=2) + "\n",
        encoding="utf-8",
    )

    for model_id in ("anytouch1", "anytouch2"):
        model = _build(model_id).to(device)
        get_training_recipe(model_id).save_visualization(
            model,
            dataset,
            indices,
            args.output_dir / model_id / "reconstruction.png",
            device,
            SimpleNamespace(mask_ratio=args.mask_ratio),
            torch.bfloat16 if device.type == "cuda" else None,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sparsh = _build("sparsh_vjepa").to(device)
    _save_sparsh_latents(
        sparsh,
        dataset,
        indices,
        args.output_dir / "sparsh_vjepa" / "encoder_latents.png",
        device,
    )


if __name__ == "__main__":
    main()
