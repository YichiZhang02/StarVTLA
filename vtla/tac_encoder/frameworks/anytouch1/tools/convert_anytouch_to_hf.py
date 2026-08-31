"""Convert an AnyTouch checkpoint (.pth) into a unified HF-style directory.

The released AnyTouch checkpoint is a full multi-model state-dict where the MAE
lives under the ``touch_mae_model.`` prefix. This tool extracts just the MAE
weights (our namespace) and writes ``{out_dir}/model.safetensors`` +
``{out_dir}/config.json`` so that, alongside the HF CLIP directory, every init
source can be passed through the single ``--pretrained_path`` argument.

Usage:
    python -m vtla.tac_encoder.frameworks.anytouch1.tools.convert_anytouch_to_hf \
        --src playground/pretrained_models/checkpoint.pth \
        --out playground/pretrained_models/anytouch_mae_vitl --arch vit_l
"""
import os
import json
import argparse

import torch
from safetensors.torch import save_file


def extract_mae_state_dict(src):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
        ck = ck["model"]
    keys = list(ck.keys())
    if any(k.startswith("touch_mae_model.") for k in keys):
        sd = {k[len("touch_mae_model."):]: v for k, v in ck.items()
              if k.startswith("touch_mae_model.")}
    elif any(k.startswith("touch_model.") for k in keys):
        # already a bare MAE state-dict (e.g. saved by our own train.py)
        sd = {k: v for k, v in ck.items() if hasattr(v, "shape")}
    else:
        raise ValueError(f"No MAE weights found in {src}. First keys: {keys[:5]}")
    return {k: v.contiguous().clone() for k, v in sd.items() if hasattr(v, "shape")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="AnyTouch .pth checkpoint")
    ap.add_argument("--out", required=True, help="output HF-style directory")
    ap.add_argument("--arch", default="vit_l", choices=["vit_b", "vit_l"])
    args = ap.parse_args()

    sd = extract_mae_state_dict(args.src)
    os.makedirs(args.out, exist_ok=True)
    save_file(sd, os.path.join(args.out, "model.safetensors"))
    config = {
        "model_type": "tactile_mae",
        "arch": args.arch,
        "use_sensor_token": "sensor_token" in sd,
        "source": "anytouch",
        "num_tensors": len(sd),
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {len(sd)} tensors to {args.out}/model.safetensors")
    print(f"Config: {config}")


if __name__ == "__main__":
    main()
