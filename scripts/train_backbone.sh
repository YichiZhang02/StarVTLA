#!/usr/bin/env bash
set -euo pipefail

dataset_id=${1:-backbone_training_data}
model_id=${2:-anytouch1}  # anytouch1 | anytouch2 | sparsh_vjepa | wan22_vae

num_processes=${3:-4}
default_batch_size=32
if [[ "${model_id}" == "wan22_vae" ]]; then
  default_batch_size=4
fi
batch_size=${4:-${default_batch_size}}   # per process / per GPU
epochs=${5:-5}
lr=${6:-1e-5}

image_size=${7:-224}
tactile_num_frames=${8:-4}  # 每步输入的触觉帧数（含当前帧）
tactile_frame_offset=${9:-2}  # 相邻两个触觉帧的采样间隔（帧数）
resume=${10:-}  # checkpoint 路径；留空表示不恢复训练


repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/starvtla_matplotlib}"
run_name="$(date +%Y%m%d)_${dataset_id}_${model_id}_${tactile_num_frames}frames_${tactile_frame_offset}stride"
output_root="${repo_root}/playground/results/backbones"
output_dir="${output_dir:-${output_root}/${run_name}}"
cd "${repo_root}"

case "${model_id}" in
  anytouch1)
    pretrained_path="${repo_root}/playground/pretrained_models/AnyTouch-ViT-L-16/checkpoint.pth"
    ;;
  anytouch2)
    pretrained_path="${repo_root}/playground/pretrained_models/AnyTouch2-Model/checkpoint-4frames.pth"
    ;;
  sparsh_vjepa)
    # Initialize context/target from the public encoder; train the predictor from scratch.
    pretrained_path="${repo_root}/playground/pretrained_models/Sparsh-VJEPA-Small/vjepa_vitsmall.safetensors"
    ;;
  wan22_vae)
    pretrained_path="${repo_root}/playground/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    ;;
  *)
    echo "Unknown model_id: ${model_id} (anytouch1|anytouch2|sparsh_vjepa|wan22_vae)" >&2
    exit 1
    ;;
esac

if [[ ! -f "${pretrained_path}" ]]; then
  echo "Pretrained checkpoint not found: ${pretrained_path}" >&2
  exit 1
fi

common_args=(
  --dataset_id "${dataset_id}"
  --model_id "${model_id}"
  --pretrained_path "${pretrained_path}"
  --output_dir "${output_dir}"
  --batch_size "${batch_size}"
  --epochs "${epochs}"
  --lr "${lr}"
  --image_size "${image_size}"
  --num_frames "${tactile_num_frames}"
  --frame_stride "${tactile_frame_offset}"
)

if [[ -n "${resume}" ]]; then
  common_args+=(--resume "${resume}")
fi
if [[ "${model_id}" == "wan22_vae" ]]; then
  common_args+=(--vae_kl_weight "${VAE_KL_WEIGHT:-1e-6}")
fi

echo "Pretrained path: ${pretrained_path}"
echo "Output dir: ${output_dir}"

if [[ "${num_processes}" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node "${num_processes}" -m vtla.tac_encoder.train "${common_args[@]}"
else
  exec python -m vtla.tac_encoder.train "${common_args[@]}"
fi
