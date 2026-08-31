#!/usr/bin/env bash
# Open-loop eval for Franka cut-banana policy trained with
# experiment=cosmos_predict2_2b_480p_franka_cut_banana_20260321_no_tactile_img_aug
#
# Override any variable before sourcing or calling:
#   FRANKA_COSMOS_CKPT=.../checkpoints/iter_000005000
#   OPENLOOP_OUT=./my_out
#   EPISODE=episode_0
#
# Extra args are forwarded (e.g. --future_pred_eval, --frame_start / --frame_end):
#   ./run_openloop_cut_banana_no_tactile_img_aug.sh --future_pred_eval
#   ./run_openloop_cut_banana_no_tactile_img_aug.sh --frame_start 100 --frame_end 400
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

export FRANKA_COSMOS_CONFIG="${FRANKA_COSMOS_CONFIG:-cosmos_predict2_2b_480p_franka_cut_banana_20260321_no_tactile_img_aug}"
export FRANKA_COSMOS_CKPT="${FRANKA_COSMOS_CKPT:-/tmp/imaginaire4-output/cosmos_policy_franka_cut_banana/cosmos_v2_finetune/cosmos_predict2_2b_480p_franka_cut_banana_20260321_no_tactile_img_aug/checkpoints/iter_000007000}"
export FRANKA_DATASET_STATS_PATH="${FRANKA_DATASET_STATS_PATH:-/path/to/attention_data_cut_banana/cut_banana_20260321/dataset_statistics_franka.json}"
export FRANKA_T5_EMBEDDINGS_PATH="${FRANKA_T5_EMBEDDINGS_PATH:-/path/to/attention_data_cut_banana/cut_banana_20260321/t5_embeddings.pkl}"
export FRANKA_CHUNK_SIZE="${FRANKA_CHUNK_SIZE:-20}"

CUT_BANANA_TRAIN="${CUT_BANANA_TRAIN:-/path/to/attention_data_cut_banana/cut_banana_20260321/train}"
EPISODE="${EPISODE:-episode_0}"
OPENLOOP_OUT="${OPENLOOP_OUT:-${REPO_ROOT}/cosmos_policy/ckpt/openloop_cut_banana_no_tactile_img_aug}"

if [[ ! -e "$FRANKA_COSMOS_CKPT" ]]; then
  echo "Checkpoint not found: $FRANKA_COSMOS_CKPT"
  echo "Set FRANKA_COSMOS_CKPT to your iter_* directory."
  exit 1
fi

echo "CONFIG=$FRANKA_COSMOS_CONFIG"
echo "CKPT=$FRANKA_COSMOS_CKPT"
echo "OUT=$OPENLOOP_OUT"

exec uv run --extra cu128 --group libero --python 3.10 python -m cosmos_policy.experiments.robot.franka.run_franka_openloop \
  --hdf5 "${CUT_BANANA_TRAIN}/${EPISODE}.hdf5" \
  --cam_front "${CUT_BANANA_TRAIN}/${EPISODE}_cam_front.mp4" \
  --cam_high "${CUT_BANANA_TRAIN}/${EPISODE}_cam_high.mp4" \
  --tactile_left "${CUT_BANANA_TRAIN}/${EPISODE}_tactile_rectify_left.mp4" \
  --tactile_right "${CUT_BANANA_TRAIN}/${EPISODE}_tactile_rectify_right.mp4" \
  --out_dir "$OPENLOOP_OUT" \
  "$@"
