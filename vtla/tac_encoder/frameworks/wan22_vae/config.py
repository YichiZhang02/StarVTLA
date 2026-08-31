"""Wan2.2 VAE checkpoint reconstruction configuration."""

from collections.abc import Mapping
from typing import Any


def checkpoint_kwargs(args: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "num_frames": int(args.get("num_frames", 4)),
        "image_size": int(args.get("image_size", 224)),
        "latent_dim": int(args.get("wan22_latent_dim", 48)),
        "base_dim": int(args.get("wan22_base_dim", 160)),
        "decoder_base_dim": int(args.get("wan22_decoder_base_dim", 256)),
        "kl_weight": float(args.get("vae_kl_weight", 1e-6)),
        "pretrained_path": "",
    }
