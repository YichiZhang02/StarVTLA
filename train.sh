#!/bin/bash
cd "$(dirname "$0")" || exit 1   # 切到脚本所在目录(仓库根), 服务器/本地通用, 无需手改
REPO_ROOT="$(pwd)"               # 自动探测 (仅用于 PYTHONPATH 等运行期, 不写进保存的 config)

# =================== 需要改动的配置 ===================
# 模型和数据集配置
dataset_id=${1:-260821_boarderaser_to_cup_trainready_rgb640x480_camlr_egor}  # 数据集名
policy_type=${2:-starvla_groot_dinoalign}          # act | diffusion | pi05 | starvla_groot | starvla_groot_dinoalign | fastwam | dream_tac

# 训练配置
num_processes=${3:-4}
batch_size=${4:-16}
steps=${5:-40_000}
save_freq=5_000
log_freq=100

# 数据配置
wrist_only=${6:-true}  # true | false
tactile_mode=${7:-none}  # none | as_image | encode
state_mode=${8:-absolute_joint}  # none | absolute_joint | episode_joint | absolute_rot6d | episode_rot6d | absolute_quat | episode_quat
action_mode=${9:-absolute_joint}  # absolute_joint | relative_joint | absolute_rot6d | relative_rot6d | absolute_quat | relative_quat
action_gap=${10:-6}  # GT action 起点相对当前观测向未来偏移的帧数

# 数据增强
augmentation_mode=${11:-strong}  # none | mild | strong
color_temp_range=${COLOR_TEMP_RANGE-'[0,0]'}  # 色温增强 [min,max] (逗号后不要空格)

# 触觉encoder配置（仅 tactile_mode=encode 时生效）
tactile_encoder_path=${TACTILE_ENCODER_PATH:-${12:-/mnt/data/xidong_data/StarVTLA/playground/results/backbones/20260826_backbone_training_data_anytouch2_4frames_2stride/best.pth}}
tactile_insert_location=${TACTILE_INSERT_LOCATION:-encoder}  # 触觉插入位置 encoder | decoder
tactile_pool_size=${TACTILE_POOL_SIZE:-3}  # 3 表示下游 AdaptiveAvgPool2d(3x3)
tactile_num_frames=${TACTILE_NUM_FRAMES:-4}  # 每步输入的触觉帧数（含当前帧）
tactile_frame_offset=${TACTILE_FRAME_OFFSET:-2}  # 相邻两个触觉帧的采样间隔（帧数）

# cpu / gpu training
policy_device=${POLICY_DEVICE:-cuda}

# WAM 训练图片可视化。其余频率/sample 数/采样步数/seed 固定在 FastWAMConfig。
visualization_enabled=${VISUALIZATION_ENABLED:-true}

# =================== 不是很需要改动的配置 ===================
# 保存的模型/日志名拼接规则
policy_suffix="wristonly_${wrist_only}_tactile_${tactile_mode}_state_${state_mode}_action_${action_mode}_gap_${action_gap}_aug_${augmentation_mode}"
# 运行名: <时间>_<数据集>_<framework>_<路由后缀>, 用于输出目录/job_name/日志名 (保持一致)
run_name="$(date +%Y%m%d)_${dataset_id}_${policy_type}_${policy_suffix}"

# 路径配置 (相对路径, 会被写进 train_config.json -> 跨机器可移植)
dataset_root=playground/data
output_root=playground/results/models
mixture_config=configs/data_mixtures.yaml

# 普通数据集和 mixture 共用同一个 dataset_id 解析入口。
if ! resolved_dataset_output=$(PYTHONPATH=${REPO_ROOT}:${PYTHONPATH} python tools/resolve_training_dataset.py \
  "${dataset_id}" --catalog-root "${dataset_root}" --mixture-config "${mixture_config}"); then
  exit 1
fi
mapfile -t resolved_dataset <<< "${resolved_dataset_output}"
if [ "${#resolved_dataset[@]}" -ne 6 ]; then
  echo "Failed to resolve training dataset: ${dataset_id}"; exit 1
fi
dataset_kind=${resolved_dataset[0]}
IFS='|' read -r -a dataset_member_roots <<< "${resolved_dataset[1]}"
top_cam=${TOP_CAM:-${resolved_dataset[2]}}
wrist_cam=${WRIST_CAM:-${resolved_dataset[3]}}
tactile_keys=${TACTILE_KEYS:-${resolved_dataset[4]}}
dataset_weights=${resolved_dataset[5]}
dataset_root_arg=
if [ "${dataset_kind}" = "dataset" ]; then
  dataset_root_arg="--dataset.root=${dataset_member_roots[0]}"
fi

output_dir=${output_root}/${run_name}
log_file="${output_dir}/${run_name}.log"
tmp_log="$(mktemp "${TMPDIR:-/tmp}/${run_name}.XXXXXX.log")"  # 训练期间先把日志写到系统临时目录(不污染 tac_infra), 跑完再搬到 output_dir 下
tmp_status="${tmp_log}.status"  # POSIX sh 没有 PIPESTATUS，管道左侧通过文件传出真实退出码

# =================== 完全不需要改动的配置 ===================
# action和state合法性检查
case "${state_mode}" in
  none|absolute_joint|episode_joint|absolute_rot6d|episode_rot6d|absolute_quat|episode_quat) ;;
  *) echo "Invalid state_mode: ${state_mode}"; exit 1 ;;
esac
case "${action_mode}" in
  absolute_joint|relative_joint|absolute_rot6d|relative_rot6d|absolute_quat|relative_quat) ;;
  *) echo "Invalid action_mode: ${action_mode}"; exit 1 ;;
esac

# 预训练模型路径和基础 VLM 配置
pretrained_path=${PRETRAINED_PATH:-}
base_vlm=${BASE_VLM:-}
case "${policy_type}" in
  pi05)          pretrained_path=${pretrained_path:-playground/pretrained_models/pi05_base} ;;
  starvla_groot|starvla_groot_dinoalign) base_vlm=${base_vlm:-playground/pretrained_models/Qwen3.5-0.8B} ;;
  dream_tac)     pretrained_path=playground/pretrained_models/Cosmos-Predict2-2B-Video2World ;;
  act|diffusion|fastwam) : ;;  # 从底座或随机初始化，不加载 VTLA policy checkpoint
  *)             echo "Unknown policy_type: ${policy_type} (expected act|diffusion|pi05|starvla_groot|starvla_groot_dinoalign|fastwam|dream_tac)"; exit 1 ;;
esac


# 额外参数自动配置
extra_args=""
case "${policy_type}" in
  pi05)
    extra_args="${extra_args} --policy.dtype=bfloat16 --policy.compile_model=false --policy.gradient_checkpointing=false"
    ;;
  starvla_groot)
    extra_args="${extra_args} --policy.dtype=bfloat16 --policy.gradient_checkpointing=false --policy.base_vlm=${base_vlm}"
    ;;
  starvla_groot_dinoalign)
    extra_args="${extra_args} --policy.dtype=bfloat16 --policy.gradient_checkpointing=false --policy.base_vlm=${base_vlm}"
    ;;
  fastwam)
    extra_args="${extra_args} --dataset.return_uint8=true --policy.dtype=bfloat16 --policy.load_text_encoder=false"
    extra_args="${extra_args} --policy.visualization_enabled=${visualization_enabled}"
    ;;
  dream_tac)
    if [ "${tactile_mode}" = "encode" ]; then
      echo "dream_tac supports tactile_mode=none or as_image, not encode"
      exit 1
    fi
    extra_args="${extra_args} --dataset.return_uint8=true --policy.dtype=bfloat16"
    extra_args="${extra_args} --policy.visualization_enabled=${visualization_enabled}"
    ;;
  act|diffusion)
    : # 这两个没有 VLM/dtype 相关字段
    ;;
  *)
    echo "Unknown policy_type: ${policy_type} (expected act|diffusion|pi05|starvla_groot|starvla_groot_dinoalign|fastwam|dream_tac)"; exit 1
    ;;
esac

if [ -n "${pretrained_path}" ]; then
  extra_args="${extra_args} --policy.pretrained_path=${pretrained_path}"
fi

# augmentation_mode: none | default | mild
if [ "${augmentation_mode}" != "none" ]; then
  extra_args="${extra_args} --dataset.image_transforms.preset=${augmentation_mode}"
fi

# 色温(白平衡)增强: 非空时注入采样范围, 覆盖偏黄/偏冷的部署环境
if [ -n "${color_temp_range}" ]; then
  extra_args="${extra_args} --dataset.image_transforms.color_temp=${color_temp_range}"
fi

# tactile_mode=encode 时追加 tactile encoder 相关参数（四个 framework 通用）
if [ "${tactile_mode}" = "encode" ]; then
  if [ -z "${tactile_encoder_path}" ]; then
    echo "tactile_mode=encode 需要提供 TACTILE_ENCODER_PATH（或第 12 个位置参数）指向 train_backbone.sh 生成的 checkpoint"; exit 1
  fi
  extra_args="${extra_args} --policy.tactile_encoder_path=${tactile_encoder_path}"
  extra_args="${extra_args} --policy.tactile_pool_size=${tactile_pool_size}"
fi

# 触觉时序窗口（encode 和 as_image 均生效；F=1 时完全向后兼容）
if [ "${tactile_mode}" != "none" ] && [ "${policy_type}" != "dream_tac" ]; then
  extra_args="${extra_args} --policy.tactile_insert_location=${tactile_insert_location}"
  extra_args="${extra_args} --policy.tactile_num_frames=${tactile_num_frames}"
  extra_args="${extra_args} --policy.tactile_frame_offset=${tactile_frame_offset}"
fi


# 用花括号组把「参数打印 + 训练」整体管道给 tee，这样日志里既有配置也有训练过程。
{
echo "Log file: $log_file"
echo "Training with dataset: $dataset_id"
echo "Dataset kind: ${dataset_kind} | Members: ${dataset_member_roots[*]} | Weights: ${dataset_weights}"
echo "Policy type: $policy_type"
echo "Pretrained path: ${pretrained_path:-<scratch>} | Base VLM: ${base_vlm:-<none>}"
echo "Steps: $steps | Batch size: $batch_size | Num processes: $num_processes"
echo "Policy device: ${policy_device}"
echo "Wrist only: $wrist_only | Tactile mode: $tactile_mode | State mode: $state_mode | Action mode: $action_mode | Augmentation mode: $augmentation_mode"
echo "Action gap: ${action_gap} frame(s)"
echo "Color temp range: ${color_temp_range:-<off>}"
echo "Top cam keys:   ${top_cam}"
echo "Wrist cam keys: ${wrist_cam}"
echo "Tactile keys:   ${tactile_keys}"
if [ "${tactile_mode}" = "encode" ]; then
  echo "Tactile encoder path: ${tactile_encoder_path} (${tactile_pool_size}x${tactile_pool_size} pooled backbone, trained jointly)"
fi
if [ "${tactile_mode}" != "none" ]; then
  echo "Tactile context: insert=${tactile_insert_location} | num_frames=${tactile_num_frames} | frame_offset=${tactile_frame_offset}"
fi
echo "Output dir: $output_dir"
echo "Extra args: ${extra_args}"

# wam相关
WM_List="fastwam dream_tac"
case " ${WM_List} " in
  *" ${policy_type} "*)
    echo "Video visualization: ${visualization_enabled}"
    for member_root in "${dataset_member_roots[@]}"; do
      world_model=wan22
      [ "${policy_type}" = "dream_tac" ] && world_model=dream_tac
      text_embedding_dir="${member_root}/text_embeddings/${world_model}"
      required_assets=("${text_embedding_dir}/manifest.json" "${text_embedding_dir}/embeddings.safetensors")
      assets_ready=true
      for required_asset in "${required_assets[@]}"; do
        [ -f "${required_asset}" ] || assets_ready=false
      done
      if [ "${assets_ready}" = "true" ]; then
        echo "Reusing precomputed text embeddings: ${text_embedding_dir}"
      else
        echo "Precomputing text embeddings for ${policy_type}: ${member_root}"
        precompute_extra_args=()
        if [ "${policy_type}" = "dream_tac" ]; then
          precompute_extra_args+=(--pretrained-path "${pretrained_path}")
        fi
        PYTHONPATH=${REPO_ROOT}:${PYTHONPATH} python tools/precompute_world_model_text_embeddings.py \
          --dataset-root "${member_root}" \
          --world-model "${world_model}" \
          "${precompute_extra_args[@]}" || exit 1
      fi
    done
    ;;
esac

PYTHONPATH=${REPO_ROOT}:${PYTHONPATH} accelerate launch \
    --num_processes=$num_processes \
    -m vtla.train \
    --dataset.repo_id=$dataset_id \
    ${dataset_root_arg} \
    --dataset.catalog_root=${dataset_root} \
    --dataset.mixture_config=${mixture_config} \
    --dataset.video_backend=pyav \
    --policy.type=${policy_type} \
    --policy.wrist_only=${wrist_only} \
    --policy.tactile_mode=${tactile_mode} \
    --policy.state_mode=${state_mode} \
    --policy.action_mode=${action_mode} \
    --policy.action_gap=${action_gap} \
    --policy.top_camera_keys="${top_cam}" \
    --policy.wrist_camera_keys="${wrist_cam}" \
    --policy.tactile_keys="${tactile_keys}" \
    --policy.device=${policy_device} \
    --policy.push_to_hub=false \
    ${extra_args} \
    --output_dir=${output_dir} \
    --job_name=${run_name} \
    --steps=${steps} \
    --save_freq=${save_freq} \
    --batch_size=${batch_size} \
    --log_freq=${log_freq} \
    --tolerance_s=0.04 \
    --wandb.enable=false
train_status=$?
printf '%s\n' "$train_status" > "$tmp_status"
exit "$train_status"
} 2>&1 | tee "$tmp_log"
if [ -r "$tmp_status" ]; then
  IFS= read -r train_status < "$tmp_status"
else
  train_status=1
fi
rm -f "$tmp_status"

# 正常跑完才把临时日志搬到 output_dir 下的最终位置; 否则直接删除临时日志
if [ "$train_status" -eq 0 ]; then
  mkdir -p "$output_dir"
  mv "$tmp_log" "$log_file"
  echo "Log saved to: $log_file"
else
  rm -f "$tmp_log"
  echo "Training failed (exit ${train_status}), temp log removed: $tmp_log"
fi
exit "$train_status"
