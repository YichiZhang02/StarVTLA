from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from typing import Any

import torch
import time

from .io import ModelConfig, hash_model_file, load_state_dict
from .state_dict_converters import (
    wan_video_vae_state_dict_converter,
)
from ..wan_video_dit import WanVideoDiT
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..wan_video_vae import WanVideoVAE38
from ..logging_config import get_logger

logger = get_logger(__name__)
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"
OFFICIAL_WAN22_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"


@dataclass
class Wan22LoadedComponents:
    dit: WanVideoDiT
    vae: WanVideoVAE38
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str | list[str]
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None


WAN22_MODEL_REGISTRY = [
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": WanTextEncoder,
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": WanVideoDiT,
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth")
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": WanVideoVAE38,
        "state_dict_converter": wan_video_vae_state_dict_converter,
    },
]


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")

    validated = dict(dit_config)

    signature = inspect.signature(WanVideoDiT.__init__)
    allowed_keys = set()
    required_keys = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        allowed_keys.add(name)
        if param.default is inspect.Signature.empty:
            required_keys.add(name)

    unknown_keys = sorted(set(validated) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in `dit_config`: {unknown_keys}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )

    missing_keys = sorted(required_keys - set(validated))
    if missing_keys:
        raise ValueError(
            f"Missing required keys in `dit_config`: {missing_keys}. "
            "Please specify all required WanVideoDiT constructor args."
        )

    return validated


def _load_registered_model(
    path,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs_override: dict[str, Any] | None = None,
):
    model_hash = hash_model_file(path)

    matched_config = None
    for config in WAN22_MODEL_REGISTRY:
        if config["model_hash"] == model_hash and config["model_name"] == model_name:
            matched_config = config
            break
    if matched_config is None:
        raise ValueError(
            f"Cannot detect model type for {model_name}. File: {path}. "
            f"Model hash: {model_hash}. This standalone package follows DiffSynth hash-based loading."
        )

    model_class = matched_config["model_class"]
    model_kwargs = dict(matched_config.get("extra_kwargs", {}))
    if model_kwargs_override is not None:
        model_kwargs.update(model_kwargs_override)
    state_dict_converter = matched_config.get("state_dict_converter")

    model = model_class(**model_kwargs)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch_dtype)
    return model


def _resolve_configs(model_id: str, tokenizer_model_id: str, redirect_common_files: bool = False):
    dit_config = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    text_config = ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
    vae_config = ModelConfig(model_id=model_id, origin_file_pattern="Wan2.2_VAE.pth")
    tokenizer_config = ModelConfig(model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/")

    if redirect_common_files:
        redirect_dict = {
            "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
            "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
        }
        text_config.model_id, text_config.origin_file_pattern = redirect_dict[text_config.origin_file_pattern]
        vae_config.model_id, vae_config.origin_file_pattern = redirect_dict[vae_config.origin_file_pattern]
    return dit_config, text_config, vae_config, tokenizer_config


def _resolve_explicit_path(
    path: str | Path | list[str] | tuple[str, ...],
    component_name: str,
    directory_pattern: str | None = None,
) -> str | list[str]:
    """Resolve an explicitly configured local component without consulting download settings."""
    if isinstance(path, (list, tuple)):
        resolved = [str(Path(item).expanduser().resolve()) for item in path]
    else:
        candidate = Path(path).expanduser().resolve()
        if candidate.is_dir() and directory_pattern is not None:
            resolved = [str(item) for item in sorted(candidate.glob(directory_pattern))]
        else:
            resolved = str(candidate)

    resolved_items = resolved if isinstance(resolved, list) else [resolved]
    if not resolved_items or any(not Path(item).exists() for item in resolved_items):
        raise FileNotFoundError(
            f"Explicit path for {component_name} does not exist or matched no files: {path}"
        )
    return resolved[0] if isinstance(resolved, list) and len(resolved) == 1 else resolved


def _validate_official_snapshot_manifest(path: str | Path, model_id: str) -> None:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir() or candidate.name != "Wan2.2-TI2V-5B":
        return
    manifest_path = candidate / "fastwam_source.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Official Wan2.2 directory is missing provenance manifest {manifest_path}. "
            "Provide fastwam_source.json from the verified official snapshot."
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if model_id != OFFICIAL_WAN22_REPO_ID or manifest.get("repo_id") != OFFICIAL_WAN22_REPO_ID:
        raise ValueError(
            "FastWAM official Wan2.2 initialization requires "
            f"{OFFICIAL_WAN22_REPO_ID!r}; config={model_id!r}, manifest={manifest.get('repo_id')!r}."
        )
    if not manifest.get("commit"):
        raise ValueError(f"Official Wan2.2 provenance manifest has no commit: {manifest_path}")


def _resolve_component_path(
    explicit_path: str | Path | list[str] | tuple[str, ...] | None,
    fallback_config: ModelConfig,
    component_name: str,
    directory_pattern: str | None = None,
) -> str | list[str]:
    if explicit_path is not None:
        return _resolve_explicit_path(
            explicit_path,
            component_name=component_name,
            directory_pattern=directory_pattern,
        )
    fallback_config.download_if_necessary()
    return fallback_config.path


def load_wan22_ti2v_5b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = False,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
    dit_path: str | Path | list[str] | tuple[str, ...] | None = None,
    vae_path: str | Path | None = None,
    text_encoder_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
):
    logger.info("Loading Wan2.2-TI2V-5B components...")
    start = time.time()

    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.2-TI2V-5B loading.")
    validated_dit_config = _validate_dit_config(dit_config)
    if dit_path is not None and not isinstance(dit_path, (list, tuple)):
        _validate_official_snapshot_manifest(dit_path, model_id)

    dit_model_config, text_config, vae_config, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )

    resolved_vae_path = _resolve_component_path(
        vae_path,
        vae_config,
        component_name="Wan2.2 VAE",
        directory_pattern="*.safetensors",
    )
    resolved_text_encoder_path: str | list[str] | None = None
    resolved_tokenizer_path: str | list[str] | None = None
    if load_text_encoder:
        resolved_text_encoder_path = _resolve_component_path(
            text_encoder_path,
            text_config,
            component_name="Wan text encoder",
            directory_pattern="*.safetensors",
        )
        resolved_tokenizer_path = _resolve_component_path(
            tokenizer_path,
            tokenizer_config,
            component_name="UMT5 tokenizer",
        )

    if skip_dit_load_from_pretrain:
        logger.info(
            "Skipping pretrained video DiT load (`skip_dit_load_from_pretrain=True`); "
            "initializing video expert randomly and expecting checkpoint override."
        )
        dit: WanVideoDiT = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        resolved_dit_path = _resolve_component_path(
            dit_path,
            dit_model_config,
            component_name="Wan2.2 video DiT",
            directory_pattern="diffusion_pytorch_model*.safetensors",
        )
        dit = _load_registered_model(
            resolved_dit_path,
            "wan_video_dit",
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs_override=validated_dit_config,
        )
        dit_path = (
            [str(item) for item in resolved_dit_path]
            if isinstance(resolved_dit_path, list)
            else str(resolved_dit_path)
        )
    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        text_encoder = _load_registered_model(
            resolved_text_encoder_path,
            "wan_video_text_encoder",
            torch_dtype=torch_dtype,
            device=device,
        )
        tokenizer = HuggingfaceTokenizer(
            name=resolved_tokenizer_path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        text_encoder_path = str(resolved_text_encoder_path)
        tokenizer_path = str(resolved_tokenizer_path)
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); "
            "training must provide cached `context/context_mask`."
        )
    vae: WanVideoVAE38 = _load_registered_model(
        resolved_vae_path,
        "wan_video_vae",
        torch_dtype=torch_dtype,
        device=device,
    )
    logger.info("Finished loading Wan2.2-TI2V-5B components in %.2f seconds.", time.time() - start)
    return Wan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        vae_path=str(resolved_vae_path),
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )
