# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cosmos Policy inference server for Franka (single-arm).
# Inputs: cam_front (primary), cam_high (wrist), optional tactile_left/right,
#         state (proprio 6-dim pose only), instruction.
#
# Usage:
#   export FRANKA_COSMOS_CKPT=/path/to/checkpoints/iter_000003350
#   export FRANKA_DATASET_STATS_PATH=/path/to/dataset_statistics_franka.json
#   export FRANKA_T5_EMBEDDINGS_PATH=/path/to/t5_embeddings.pkl
#   export FRANKA_USE_TACTILE=1  # enable tactile (default: auto-detect from model state_t)
#   cd /path/to/cosmos-policy
#   uv run --extra cu128 --group libero --python 3.10 python -m cosmos_policy.experiments.robot.franka.franka_server

# Inference API:
#   POST /infer  Content-Type: application/json
#   {
#     "images": {
#       "cam_front": "<base64>",
#       "cam_high": "<base64>",
#       "tactile_left": "<base64>",   // optional, required when use_tactile=True
#       "tactile_right": "<base64>"   // optional, required when use_tactile=True
#     },
#     "state": [x, y, z, roll, pitch, yaw],
#     "instruction": "pick ..."
#   }

import base64
import io
import logging
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SERVICE_CONFIG = {
    "host": "0.0.0.0",
    "port": 5001,
    "debug": False,
    "threaded": True,
    "max_content_length": 16 * 1024 * 1024,
}

DATA_ROOT = "/path/to/tactile_data_hupai/hupai"
CONFIG_NAME = os.environ.get("FRANKA_COSMOS_CONFIG", "cosmos_predict2_2b_480p_franka_hupai_tactile")
CKPT_PATH = os.environ.get("FRANKA_COSMOS_CKPT", "")
CONFIG_FILE = os.environ.get("FRANKA_COSMOS_CONFIG_FILE", "cosmos_policy/config/config.py")
DATASET_STATS_PATH = os.environ.get("FRANKA_DATASET_STATS_PATH", f"{DATA_ROOT}/dataset_statistics_franka.json")
T5_EMBEDDINGS_PATH = os.environ.get("FRANKA_T5_EMBEDDINGS_PATH", f"{DATA_ROOT}/t5_embeddings.pkl")
CHUNK_SIZE = int(os.environ.get("FRANKA_CHUNK_SIZE", "20"))
NUM_DENOISING_STEPS = int(os.environ.get("FRANKA_NUM_DENOISING_STEPS", "5"))
# "1"/"true" = force on, "0"/"false" = force off, "" = auto-detect from model.config.state_t
_TACTILE_ENV = os.environ.get("FRANKA_USE_TACTILE", "").strip().lower()

app = Flask(__name__)
CORS(app)
model = None
cosmos_config = None
dataset_stats = None
cfg = None


def build_franka_cfg(use_tactile: bool = False):
    return SimpleNamespace(
        suite="franka",
        config=CONFIG_NAME,
        ckpt_path=CKPT_PATH,
        config_file=CONFIG_FILE,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        use_wrist_image=True,
        num_wrist_images=1,
        use_third_person_image=True,
        num_third_person_images=1,
        use_tactile=use_tactile,
        use_jpeg_compression=True,
        trained_with_image_aug=True,
        flip_images=False,
        use_variance_scale=False,
        chunk_size=CHUNK_SIZE,
        num_denoising_steps_action=NUM_DENOISING_STEPS,
    )


def load_model():
    global model, cosmos_config, dataset_stats, cfg
    if not CKPT_PATH or not os.path.exists(CKPT_PATH):
        logger.error("CKPT_PATH is not set or does not exist. Set FRANKA_COSMOS_CKPT.")
        return False

    # Build cfg with tactile=False first, then auto-detect after model loads
    cfg = build_franka_cfg(use_tactile=False)
    cfg.ckpt_path = CKPT_PATH
    try:
        logger.info("Loading Cosmos model...")
        model, cosmos_config = get_model(cfg)

        # Auto-detect tactile from model state_t, unless env var forces it
        if _TACTILE_ENV in ("1", "true", "yes"):
            use_tactile = True
        elif _TACTILE_ENV in ("0", "false", "no"):
            use_tactile = False
        else:
            use_tactile = getattr(model.config, "state_t", 8) == 12
        cfg.use_tactile = use_tactile
        logger.info("use_tactile=%s (model state_t=%s)", use_tactile, getattr(model.config, "state_t", "?"))

        logger.info("Loading dataset stats...")
        if not DATASET_STATS_PATH:
            logger.warning("FRANKA_DATASET_STATS_PATH not set; actions will not be unnormalized.")
            dataset_stats = {}
            cfg.unnormalize_actions = False
        else:
            dataset_stats = load_dataset_stats(DATASET_STATS_PATH)
        if T5_EMBEDDINGS_PATH and os.path.exists(T5_EMBEDDINGS_PATH):
            logger.info("Loading T5 embeddings cache from %s", T5_EMBEDDINGS_PATH)
            init_t5_text_embeddings_cache(T5_EMBEDDINGS_PATH)
        elif T5_EMBEDDINGS_PATH:
            logger.warning("T5 embeddings file not found at %s; will compute on the fly.", T5_EMBEDDINGS_PATH)
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return False


def decode_image_base64(b64_str: str) -> np.ndarray:
    """Decode base64 image to numpy (H, W, 3) uint8 RGB."""
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


@app.route("/health", methods=["GET"])
def health_check():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify({"status": "healthy", "model_loaded": True, "use_tactile": cfg.use_tactile})


@app.route("/info", methods=["GET"])
def service_info():
    return jsonify({
        "service_name": "Cosmos Policy Franka API",
        "version": "2.0.0",
        "config": CONFIG_NAME,
        "ckpt_path": CKPT_PATH,
        "chunk_size": CHUNK_SIZE,
        "use_tactile": cfg.use_tactile if cfg else False,
        "endpoints": {"/health": "GET", "/info": "GET", "/infer": "POST"},
    })


@app.route("/infer", methods=["POST"])
def infer_api():
    """Inference endpoint.

    Required JSON fields:
      images.cam_front  (base64)
      images.cam_high   (base64)
      state             (list of 6 floats: pose only)
      instruction       (string)

    Optional (when use_tactile=True):
      images.tactile_left   (base64)
      images.tactile_right  (base64)
    """
    start_time = time.time()
    if model is None or cfg is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Missing JSON body"}), 400

    for k in ("images", "state", "instruction"):
        if k not in data:
            return jsonify({"success": False, "error": f"Missing field: {k}"}), 400

    images = data["images"]
    if "cam_front" not in images or "cam_high" not in images:
        return jsonify({"success": False, "error": "images must contain cam_front and cam_high (base64)"}), 400

    # Accept both "tactile_left"/"tactile_right" and "tactile_rectify_left"/"tactile_rectify_right"
    tactile_left_key = next((k for k in ("tactile_left", "tactile_rectify_left") if k in images), None)
    tactile_right_key = next((k for k in ("tactile_right", "tactile_rectify_right") if k in images), None)

    if cfg.use_tactile and (tactile_left_key is None or tactile_right_key is None):
        return jsonify({
            "success": False,
            "error": "use_tactile is enabled but tactile images missing. "
                     "Send tactile_left/tactile_right or tactile_rectify_left/tactile_rectify_right.",
        }), 400

    try:
        cam_front = decode_image_base64(images["cam_front"])
        cam_high = decode_image_base64(images["cam_high"])
        state = np.array(data["state"], dtype=np.float32)
        if state.shape != (6,):
            return jsonify({"success": False, "error": "state must be length 6 (pose only, no gripper)"}), 400
        instruction = str(data["instruction"]).strip()

        obs = {
            "primary_image": cam_front,
            "wrist_image": cam_high,
            "proprio": state,
        }
        if cfg.use_tactile and tactile_left_key and tactile_right_key:
            obs["tactile_left_image"] = decode_image_base64(images[tactile_left_key])
            obs["tactile_right_image"] = decode_image_base64(images[tactile_right_key])

        action_return = get_action(
            cfg,
            model,
            dataset_stats,
            obs,
            instruction,
            seed=0,
            randomize_seed=False,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=True,
        )
        actions = action_return["actions"]
        elapsed = time.time() - start_time
        actions_serializable = [a.tolist() if hasattr(a, "tolist") else a for a in actions]

        response = {
            "success": True,
            "actions": actions_serializable,
            "processing_time": elapsed,
        }

        # Include predicted future images as base64 if available
        future_preds = action_return.get("future_image_predictions")
        if future_preds:
            future_images_b64 = {}
            for key, img_arr in future_preds.items():
                if img_arr is not None:
                    try:
                        img_np = np.asarray(img_arr).astype(np.uint8)
                        if img_np.ndim == 4:
                            img_np = img_np[0]
                        pil_img = Image.fromarray(img_np)
                        buf = io.BytesIO()
                        pil_img.save(buf, format="JPEG", quality=85)
                        future_images_b64[key] = base64.b64encode(buf.getvalue()).decode("ascii")
                    except Exception:
                        pass
            if future_images_b64:
                response["future_images"] = future_images_b64

        return jsonify(response)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def main():
    if not CKPT_PATH:
        logger.error("Set FRANKA_COSMOS_CKPT to the Cosmos checkpoint path.")
        sys.exit(1)
    if not load_model():
        sys.exit(1)
    logger.info(
        "Franka Cosmos API starting at http://%s:%s  (use_tactile=%s)",
        SERVICE_CONFIG["host"], SERVICE_CONFIG["port"], cfg.use_tactile,
    )
    app.run(
        host=SERVICE_CONFIG["host"],
        port=SERVICE_CONFIG["port"],
        debug=SERVICE_CONFIG["debug"],
        threaded=SERVICE_CONFIG["threaded"],
    )


if __name__ == "__main__":
    main()
