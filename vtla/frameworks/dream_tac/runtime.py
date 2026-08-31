from __future__ import annotations

import ctypes
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from .slot_layout import DreamTacSlotLayout


_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
_COSMOS_POLICY_ROOT = _VENDOR_ROOT / "cosmos_policy"
_COSMOS_CONFIG_FILE = "cosmos_policy/config/config.py"
_BASE_EXPERIMENT = "cosmos_predict2_2b_480p_libero"


def _preload_cuda_wheel_libraries() -> None:
    """Expose cuDNN wheel libraries before Transformer Engine is imported."""
    spec = importlib.util.find_spec("nvidia.cudnn")
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    if not locations:
        return
    lib_dir = Path(locations[0]) / "lib"
    for name in (
        "libcudnn.so.9",
        "libcudnn_graph.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_heuristic.so.9",
    ):
        library = lib_dir / name
        if library.is_file():
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def _cosmos_module_paths(module: Any) -> tuple[Path, ...]:
    paths = [Path(path).resolve() for path in getattr(module, "__path__", ())]
    module_file = getattr(module, "__file__", None)
    if module_file:
        paths.append(Path(module_file).resolve().parent)
    return tuple(paths)


def _is_vendored_cosmos_module(module: Any) -> bool:
    expected = _COSMOS_POLICY_ROOT.resolve()
    return expected in _cosmos_module_paths(module)


def ensure_dream_tac_importable() -> None:
    """Import StarVTLA's vendored Cosmos Policy implementation."""
    _preload_cuda_wheel_libraries()
    if not _COSMOS_POLICY_ROOT.is_dir():
        raise ImportError(
            "StarVTLA's vendored Cosmos Policy source is missing. Expected "
            f"{_COSMOS_POLICY_ROOT}."
        )
    vendor_root = str(_VENDOR_ROOT)
    if vendor_root in sys.path:
        sys.path.remove(vendor_root)
    sys.path.insert(0, vendor_root)

    existing = sys.modules.get("cosmos_policy")
    if existing is not None:
        if not _is_vendored_cosmos_module(existing):
            raise ImportError(
                "cosmos_policy was imported from outside StarVTLA before Dream-Tac: "
                f"{_cosmos_module_paths(existing)}. Restart the process so the vendored runtime can be used."
            )
        return

    module = importlib.import_module("cosmos_policy")
    if not _is_vendored_cosmos_module(module):
        raise ImportError(
            "Dream-Tac resolved an external cosmos_policy package instead of StarVTLA's vendor: "
            f"{_cosmos_module_paths(module)}."
        )


def resolve_cosmos_pretrained_path(pretrained_path: str) -> tuple[Path, Path]:
    source = Path(pretrained_path).expanduser()
    if not source.exists():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise FileNotFoundError(
                f"Dream-Tac pretrained_path does not exist locally: {pretrained_path}."
            ) from exc
        source = Path(snapshot_download(repo_id=pretrained_path))

    if source.is_file():
        checkpoint = source
        asset_root = source.parent
    else:
        candidates = [
            source / "model-480p-16fps.pt",
            source / "model.pt",
            source / "model/model.pt",
        ]
        checkpoint = next((path for path in candidates if path.is_file()), None)
        if checkpoint is None and (source / "model").is_dir():
            checkpoint = source
        if checkpoint is None:
            root_pt = sorted(source.glob("*.pt"))
            if len(root_pt) == 1:
                checkpoint = root_pt[0]
        if checkpoint is None:
            raise FileNotFoundError(
                "Dream-Tac Cosmos pretrained_path must contain model-480p-16fps.pt, model.pt, "
                f"or a model/ checkpoint directory: {source}."
            )
        asset_root = source
    return Path(checkpoint), asset_root


def resolve_cosmos_text_assets(pretrained_path: str) -> tuple[Path, Path]:
    """Resolve the T5 encoder and tokenizer bundled with Cosmos Predict2 weights."""
    _, asset_root = resolve_cosmos_pretrained_path(pretrained_path)
    encoder_path = asset_root / "text_encoder"
    tokenizer_path = asset_root / "tokenizer"

    missing = []
    if not (encoder_path / "config.json").is_file():
        missing.append(encoder_path / "config.json")
    if not any(encoder_path.glob("*.safetensors")):
        missing.append(encoder_path / "*.safetensors")
    if not (tokenizer_path / "tokenizer_config.json").is_file():
        missing.append(tokenizer_path / "tokenizer_config.json")
    if not (
        (tokenizer_path / "tokenizer.json").is_file()
        or (tokenizer_path / "spiece.model").is_file()
    ):
        missing.append(tokenizer_path / "tokenizer.json|spiece.model")
    if missing:
        details = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Dream-Tac Cosmos pretrained_path is missing bundled T5 assets: " + details
        )
    return encoder_path.resolve(), tokenizer_path.resolve()


def build_cosmos_experiment_opts(
    layout: DreamTacSlotLayout, asset_root: Path
) -> list[str]:
    tactile = ",".join(str(index) for index in layout.tactile_indices)
    opts = [
        f"model.config.state_t={layout.state_t}",
        f"model.config.min_num_conditional_frames={layout.num_conditional_frames}",
        f"model.config.max_num_conditional_frames={layout.num_conditional_frames}",
        f"model.config.tokenizer.chunk_duration={layout.pixel_frames}",
        "model.config.resize_online=false",
        f"++model.config.net.use_tactile_self_attn_bias={'true' if layout.tactile_indices else 'false'}",
        f"++model.config.net.tactile_latent_t_indices=[{tactile}]",
    ]
    vae_path = asset_root / "tokenizer/tokenizer.pth"
    if vae_path.is_file():
        opts.append(f"model.config.tokenizer.vae_pth={vae_path.resolve()}")
    return opts


def load_dream_tac_core(config: Any):
    ensure_dream_tac_importable()
    checkpoint, asset_root = resolve_cosmos_pretrained_path(str(config.pretrained_path))
    os.environ["COSMOS_POLICY_BASE_CHECKPOINT"] = str(checkpoint.resolve())
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    layout = config.slot_layout()
    model, _ = load_model_from_checkpoint(
        experiment_name=_BASE_EXPERIMENT,
        s3_checkpoint_dir=str(checkpoint.resolve()),
        config_file=_COSMOS_CONFIG_FILE,
        experiment_opts=build_cosmos_experiment_opts(layout, asset_root),
        enable_fsdp=False,
        load_ema_to_reg=True,
        instantiate_ema=False,
        to_device=str(config.device),
    )
    state_t = int(getattr(model.config, "state_t", -1))
    if state_t != layout.state_t:
        raise ValueError(f"Cosmos state_t={state_t} does not match slot layout {layout.state_t}.")
    pixel_frames = int(model.tokenizer.get_pixel_num_frames(state_t))
    if pixel_frames != layout.pixel_frames:
        raise ValueError(
            f"Cosmos tokenizer produces {pixel_frames} frames; layout requires {layout.pixel_frames}."
        )
    return model


__all__ = [
    "build_cosmos_experiment_opts",
    "ensure_dream_tac_importable",
    "load_dream_tac_core",
    "resolve_cosmos_pretrained_path",
    "resolve_cosmos_text_assets",
]
