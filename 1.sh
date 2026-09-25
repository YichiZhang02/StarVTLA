#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python -m tools.tacmind0.verify_weights

dataset_id=rm_isf_umi_left_20260925

bash train.sh "$dataset_id" pi05 \
    4 8 40000 true \
    none absolute_rot6d relative_rot6d 6 strong

# N0_VTLA_BASE_PATH="${N0_VTLA_BASE_PATH:-playground/pretrained_models/n0-vtla-base}" \
# PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-playground/pretrained_models/pi05_base/paligemma-3b-pt-224-tokenizer}" \
TACTILE_NUM_FRAMES=1 \
TACTILE_FRAME_OFFSET=1 \
bash train.sh "$dataset_id" n0_vtla \
    4 8 40000 true \
    as_image absolute_rot6d relative_rot6d 6 strong

dataset_id=rm_isf_umi_left_20260925
bash train.sh "$dataset_id" tacmind0 \
    4 4 40000 true \
    as_image absolute_rot6d relative_rot6d 6 strong
