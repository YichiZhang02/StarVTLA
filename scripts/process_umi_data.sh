#!/bin/bash
# Unified-format UMI -> canonical VTLA EE dataset.
# Usage: TASK="Put the board eraser into the cup." bash scripts/process_umi_data.sh <dataset_id> [size] [horizon] [action_gap]
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

dataset_id=${1:?"Usage: TASK='...' bash scripts/process_umi_data.sh <dataset_id> [size] [horizon] [action_gap]"}
size=${2:-256}
horizon=${3:-32}
action_gap=${4:-6}
task=${TASK:-}
jobs=${JOBS:-12}

if [ -z "${task}" ]; then
  echo "Error: set TASK to the real language instruction for this dataset" >&2
  exit 1
fi

src="playground/data/${dataset_id}"
dst="playground/data/${dataset_id}_processed"
gripper_args=()
if [ -n "${LEFT_GRIPPER_OPEN:-}" ]; then
  gripper_args+=(
    --left-gripper-open "${LEFT_GRIPPER_OPEN}"
    --left-gripper-closed "${LEFT_GRIPPER_CLOSED:?LEFT_GRIPPER_CLOSED is required}"
  )
fi
if [ -n "${RIGHT_GRIPPER_OPEN:-}" ]; then
  gripper_args+=(
    --right-gripper-open "${RIGHT_GRIPPER_OPEN}"
    --right-gripper-closed "${RIGHT_GRIPPER_CLOSED:?RIGHT_GRIPPER_CLOSED is required}"
  )
fi

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python tools/process_umi_data.py \
  --src "${src}" \
  --dst "${dst}" \
  --task "${task}" \
  --size "${size}" \
  --horizon "${horizon}" \
  --action-gap "${action_gap}" \
  --tactile-mode none \
  --jobs "${jobs}" \
  "${gripper_args[@]}"

echo "Training dataset: ${dst}"
echo "Example: bash train.sh $(basename "${dst}") pi05 1 32 10000 true none episode_rot6d relative_rot6d ${action_gap} none"
