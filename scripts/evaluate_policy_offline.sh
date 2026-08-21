#!/bin/bash
# Offline episode evaluation: dataset observations -> predict_action_chunk -> GT comparison.
# Usage: evaluate_policy_offline.sh <dataset_id> <pretrained_id> <step|last> [episodes] [stride] [device]
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO_ROOT="$(pwd)"

# if [ "$#" -lt 3 ]; then
#   echo "用法: bash scripts/evaluate_policy_offline.sh" >&2
#   echo "  <dataset_id> <pretrained_id> <step|last> [episodes] [stride] [device]" >&2
#   exit 2
# fi

dataset_id=${1:-rm_isf_umi_left_20260820_insert_easy_precise_undist_uint8_256}
pretrained_id=${2:-20260821_rm_isf_umi_left_20260820_insert_easy_precise_undist_uint8_256_starvla_groot_wristonly_true_tactile_none_state_absolute_rot6d_action_relative_rot6d_aug_strong}
step=${3:-3000}
episodes=${4:-0-2}  # 不传默认all
stride=${5:-16}
device=${6:-cuda}

if [ "${step}" != "last" ]; then
  if [[ ! "${step}" =~ ^[0-9]+$ ]]; then
    echo "错误: step 必须是非负整数或 last: ${step}" >&2
    exit 2
  fi
  printf -v step "%06d" "$((10#${step}))"
fi

dataset_root="playground/data/${dataset_id}"
checkpoint="playground/results/models/${pretrained_id}/checkpoints/${step}/pretrained_model"

if [ ! -f "${dataset_root}/meta/info.json" ]; then
  echo "错误: 数据集不存在或缺少 meta/info.json: ${dataset_root}" >&2
  exit 1
fi
if [ ! -f "${checkpoint}/config.json" ]; then
  echo "错误: checkpoint 不存在或缺少 config.json: ${checkpoint}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python tools/evaluate_policy_offline.py \
  --dataset-root "${dataset_root}" \
  --checkpoint "${checkpoint}" \
  --episodes "${episodes}" \
  --stride "${stride}" \
  --device "${device}"
