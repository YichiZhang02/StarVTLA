"""Model builder + unified pretrained-weight loader.

A single ``pretrained_path`` drives the three init modes:
  * empty / None                       -> from_scratch
  * a HF CLIP dir (keys ``vision_model.*``)        -> from CLIP (encoder+proj, partial)
  * an AnyTouch ckpt (.pth or converted HF dir)    -> from AnyTouch (full, strict-ish)

The source namespace is auto-detected from the state-dict keys, so the caller
does not need to say which kind of checkpoint it is.
"""
import os

import torch
from transformers import CLIPVisionConfig

from .mae_model import TactileMAE
from .vit_decoder import ViTDecoderConfig

# CLIP ViT presets (match the released CLIP configs: gelu, eps 1e-5).
ARCH_PRESETS = {
    "vit_l": dict(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                  num_attention_heads=16, patch_size=14, image_size=224,
                  projection_dim=768),
    "vit_b": dict(hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                  num_attention_heads=12, patch_size=16, image_size=224,
                  projection_dim=512),
}


def build_model(arch="vit_l", mask_ratio=0.75, use_sensor_token=True,
                use_same_patchemb=True, norm_pix_loss=False, visible_loss_weight=0.0):
    if arch not in ARCH_PRESETS:
        raise ValueError(f"arch must be one of {list(ARCH_PRESETS)}, got {arch}")
    p = ARCH_PRESETS[arch]
    vision_config = CLIPVisionConfig(
        hidden_size=p["hidden_size"], intermediate_size=p["intermediate_size"],
        num_hidden_layers=p["num_hidden_layers"], num_attention_heads=p["num_attention_heads"],
        patch_size=p["patch_size"], image_size=p["image_size"], num_channels=3,
        hidden_act="gelu", layer_norm_eps=1e-5, attn_implementation="eager")
    vision_config.projection_dim = p["projection_dim"]
    vision_config._attn_implementation = "eager"

    decoder_config = ViTDecoderConfig(
        hidden_size=512, intermediate_size=2048, num_attention_heads=16,
        num_hidden_layers=8, patch_size=p["patch_size"], num_channels=3,
        layer_norm_eps=1e-12)

    model = TactileMAE(vision_config, decoder_config, mask_ratio=mask_ratio,
                       use_sensor_token=use_sensor_token,
                       use_same_patchemb=use_same_patchemb, norm_pix_loss=norm_pix_loss,
                       visible_loss_weight=visible_loss_weight)
    model.initialize_weights()
    return model


# --------------------------------------------------------------------- loading
def _read_state_dict(path):
    """Load a flat {name: tensor} dict from a file or a directory."""
    if os.path.isdir(path):
        for fname in ("model.safetensors", "pytorch_model.bin", "open_clip_pytorch_model.bin"):
            fp = os.path.join(path, fname)
            if os.path.exists(fp):
                path = fp
                break
        else:
            raise FileNotFoundError(f"No known weight file found in directory {path}")

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)

    # unwrap common containers
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return {k: v for k, v in sd.items() if hasattr(v, "shape")}


def _from_clip(clip_sd):
    """Remap a HF CLIP state-dict to our namespace (encoder + projection + video_*)."""
    new = {}
    for k, v in clip_sd.items():
        if k.startswith("vision_model.") and "position_ids" not in k:
            new[k.replace("vision_model.", "touch_model.")] = v.clone()
            if "embeddings.patch_embedding" in k:
                # 2D conv (out,in,ph,pw) -> 3D conv (out, in=3(repeat), kT=3, ph, pw)
                new["video_patch_embedding.weight"] = v.clone().unsqueeze(1).repeat(1, 3, 1, 1, 1)
            if "embeddings.position_embedding" in k:
                new["video_position_embedding.weight"] = v.clone()
        if k.startswith("visual_projection"):
            new[k.replace("visual_projection", "touch_projection")] = v.clone()
    return new


def _from_open_clip(oc_sd):
    """Remap an open_clip ViT state-dict (``visual.*``) to our namespace."""
    import re
    new = {}
    for k, v in oc_sd.items():
        if not k.startswith("visual."):
            continue
        if k == "visual.class_embedding":
            new["touch_model.embeddings.class_embedding"] = v.clone()
        elif k == "visual.positional_embedding":
            new["touch_model.embeddings.position_embedding.weight"] = v.clone()
            new["video_position_embedding.weight"] = v.clone()
        elif k == "visual.conv1.weight":
            new["touch_model.embeddings.patch_embedding.weight"] = v.clone()
            new["video_patch_embedding.weight"] = v.clone().unsqueeze(1).repeat(1, 3, 1, 1, 1)
        elif k.startswith("visual.ln_pre."):
            new["touch_model.pre_layrnorm." + k.split(".")[-1]] = v.clone()
        elif k.startswith("visual.ln_post."):
            new["touch_model.post_layernorm." + k.split(".")[-1]] = v.clone()
        elif k == "visual.proj":
            new["touch_projection.weight"] = v.t().contiguous()  # (out,in) for nn.Linear
        elif k.startswith("visual.transformer.resblocks."):
            m = re.match(r"visual\.transformer\.resblocks\.(\d+)\.(.+)", k)
            i, rest = m.group(1), m.group(2)
            base = f"touch_model.encoder.layers.{i}."
            if rest.startswith("ln_1."):
                new[base + "layer_norm1." + rest.split(".")[-1]] = v.clone()
            elif rest.startswith("ln_2."):
                new[base + "layer_norm2." + rest.split(".")[-1]] = v.clone()
            elif rest == "attn.in_proj_weight":
                d = v.shape[0] // 3
                new[base + "self_attn.q_proj.weight"] = v[:d].clone()
                new[base + "self_attn.k_proj.weight"] = v[d:2 * d].clone()
                new[base + "self_attn.v_proj.weight"] = v[2 * d:].clone()
            elif rest == "attn.in_proj_bias":
                d = v.shape[0] // 3
                new[base + "self_attn.q_proj.bias"] = v[:d].clone()
                new[base + "self_attn.k_proj.bias"] = v[d:2 * d].clone()
                new[base + "self_attn.v_proj.bias"] = v[2 * d:].clone()
            elif rest.startswith("attn.out_proj."):
                new[base + "self_attn.out_proj." + rest.split(".")[-1]] = v.clone()
            elif rest.startswith("mlp.c_fc."):
                new[base + "mlp.fc1." + rest.split(".")[-1]] = v.clone()
            elif rest.startswith("mlp.c_proj."):
                new[base + "mlp.fc2." + rest.split(".")[-1]] = v.clone()
    return new


def _strip_prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_pretrained(model, pretrained_path, verbose=True):
    """Load weights into ``model`` from any supported source. Returns (missing, unexpected)."""
    if not pretrained_path:
        if verbose:
            print("[tactile_mae] No pretrained_path -> training from scratch.")
        return [], []

    raw = _read_state_dict(pretrained_path)
    keys = list(raw.keys())

    # ---- detect source namespace ----
    if any(k.startswith("touch_mae_model.") for k in keys):
        source = "anytouch_full"          # raw AnyTouch multi-model .pth
        sd = _strip_prefix(raw, "touch_mae_model.")
    elif any(k.startswith("touch_model.") for k in keys) or "sensor_token" in raw:
        source = "anytouch"               # our namespace (converted ckpt)
        sd = raw
    elif any(k.startswith("vision_model.") for k in keys) or \
            any(k.startswith("visual_projection") for k in keys):
        source = "clip"
        sd = _from_clip(raw)
    elif any(k.startswith("visual.") for k in keys):
        source = "open_clip"
        sd = _from_open_clip(raw)
    else:
        raise ValueError(
            f"Could not recognize checkpoint format at {pretrained_path}. "
            f"First keys: {keys[:5]}")

    # keep only keys the model actually has (drops e.g. text encoder / logit_scale)
    model_keys = set(model.state_dict().keys())
    sd = {k: v for k, v in sd.items() if k in model_keys}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"[tactile_mae] Loaded pretrained ({source}) from {pretrained_path}: "
              f"{len(sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
        if source in ("clip", "open_clip"):
            print("  (CLIP load is partial: decoder / sensor_token / mask_token use init weights)")
        # surface anything unexpected that is genuinely dropped
        if unexpected:
            print(f"  unexpected (ignored): {list(unexpected)[:6]}{' ...' if len(unexpected) > 6 else ''}")
    return missing, unexpected
