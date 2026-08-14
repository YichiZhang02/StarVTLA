#!/usr/bin/env python
"""Generate an interpolated DiT backbone from local official Wan2.2 shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vtla.frameworks.fastwam.action_dit_preprocess import build_action_dit_backbone_payload
from vtla.frameworks.fastwam.configuration_fastwam import (
    _default_action_dit_config,
    _default_video_dit_config,
)
from vtla.frameworks.fastwam.core.action_dit import (
    OFFICIAL_WAN22_DIT_MODEL_HASH,
    OFFICIAL_WAN22_REPO_ID,
    ActionDiT,
)
from vtla.frameworks.fastwam.core.helpers.io import hash_model_file
from vtla.frameworks.fastwam.core.helpers.loader import _load_registered_model


DEFAULT_WAN_DIR = Path("playground/pretrained_models/Wan2.2-TI2V-5B")
DEFAULT_OUTPUT = DEFAULT_WAN_DIR / (
    "interpolated_dit/InterpolatedDiT_from_official_Wan2.2_alphascale_1024hdim.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wan-dir",
        type=Path,
        default=DEFAULT_WAN_DIR,
        help=(
            "Local directory containing the three official Video DiT shards "
            "and fastwam_source.json."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--no-alpha-scaling", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_official_source(wan_dir: Path) -> dict[str, str]:
    manifest_path = wan_dir / "fastwam_source.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing source manifest: {manifest_path}. "
            "The generated payload must retain the official Wan2.2 source commit."
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    repo_id = manifest.get("repo_id")
    commit = manifest.get("commit")
    if repo_id != OFFICIAL_WAN22_REPO_ID:
        raise ValueError(
            f"Expected source repo {OFFICIAL_WAN22_REPO_ID!r}, got {repo_id!r}."
        )
    if not commit:
        raise ValueError(f"Source manifest has no commit: {manifest_path}")
    return {"repo_id": repo_id, "commit": str(commit)}


def find_video_dit_shards(wan_dir: Path) -> list[Path]:
    shards = sorted(wan_dir.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if len(shards) != 3:
        raise FileNotFoundError(
            f"Expected three official Wan2.2 Video DiT shards in {wan_dir}, got {len(shards)}."
        )
    return shards


def main() -> None:
    args = parse_args()
    wan_dir = args.wan_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Interpolated DiT payload already exists: {output}; pass --overwrite."
        )

    source = load_official_source(wan_dir)
    shards = find_video_dit_shards(wan_dir)
    model_hash = hash_model_file([str(path) for path in shards])
    if model_hash != OFFICIAL_WAN22_DIT_MODEL_HASH:
        raise ValueError(
            "The local Video DiT shards do not match the official Wan2.2 model structure: "
            f"expected {OFFICIAL_WAN22_DIT_MODEL_HASH}, got {model_hash}."
        )

    dtype = torch_dtype(args.dtype)
    video_config = _default_video_dit_config()
    action_config = _default_action_dit_config()

    print(f"Loading local Wan2.2 Video DiT on {args.device} as {args.dtype}...")
    video_expert = _load_registered_model(
        [str(path) for path in shards],
        "wan_video_dit",
        torch_dtype=dtype,
        device=args.device,
        model_kwargs_override=video_config,
    )
    target_dit = ActionDiT(**action_config).to(device=args.device, dtype=dtype)
    source.update(
        {
            "video_dit_model_hash": model_hash,
            "video_dit_files": [path.name for path in shards],
        }
    )
    payload, stats = build_action_dit_backbone_payload(
        video_expert,
        target_dit,
        apply_alpha_scaling=not args.no_alpha_scaling,
        source=source,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary_output)
    temporary_output.replace(output)
    print(f"Saved interpolated DiT backbone to {output}: {stats}")


if __name__ == "__main__":
    main()
