#!/bin/bash
cd "$(dirname "$0")/.." || exit 1   # 切到仓库根, 服务器/本地通用, 无需手改
REPO_ROOT="$(pwd)"               # 自动探测 (仅运行期用, 不写进保存的 config)

# ===================触觉 backbone（Tactile MAE）预训练启动器===================
# 复用 vtla/tac_encoder/tactile_mae，直接吃 LeRobot 数据训练 AnyTouch stage1 风格的 MAE。
# dataset_ids 第一个位置参数，可一个也可多个（多个用引号包成一串，空格分隔）
#   单个: bash train_enc.sh rm_nist_260320_strawberry
#   多个: bash train_enc.sh "rm_nist_260320_strawberry rm_nist_260520_usb"

# =================== 需要改动的配置 ===================
# 数据集 / 初始化
dataset_ids=${1:-pretrained_data}
init_mode=${2:-clip}            # scratch | clip | anytouch
arch=${3:-vit_b}                # vit_l | vit_b

# 训练配置
num_processes=${4:-4}
batch_size=${5:-128}
epochs=${6:-100}


# =================== 不是很需要改动的配置（自动推导） ===================
# 触觉相机 key（按数据集命名调整，支持多路）。与 train.sh 统一为列表写法 [key1,key2,...]。
# 留空则按数据模式自动选默认：raw frame_cache 单路 finger0，LeRobot 双臂四路 finger。
tactile_keys=${TACTILE_KEYS:-}

# 数据模式：留空自动识别 raw frame_cache vs LeRobot；RAW_FRAME_CACHE=1/0 可强制覆盖
raw_frame_cache=${RAW_FRAME_CACHE:-}

warmup_epochs=${WARMUP_EPOCHS:-1}
num_workers=${NUM_WORKERS:-12}
# 稳定性：bf16 autocast 防 fp16 溢出 NaN（lr 已通过 blr 调小，无需梯度裁剪）
amp_dtype=${AMP_DTYPE:-bfloat16}
# 默认端口（20000-39999），便于同时跑多个；可用 MASTER_PORT 固定
master_port=${MASTER_PORT:-$((20000 + $$ % 20000))}

# 触觉路由 / MAE 配置
# sensor_id: -1=agnostic（默认）| 3=gelsight 系 | 6=空闲槽位
sensor_id=${SENSOR_ID:--1}
mask_ratio=${MASK_RATIO:-0.75}
val_ratio=${VAL_RATIO:-0.05}
# 可见区重建 loss 权重 λ：loss = loss_masked + λ·loss_visible（0=标准 MAE）
visible_loss_weight=${VISIBLE_LOSS_WEIGHT:-0.1}
# 缓存按此尺寸 resize；raw 模式必须与构建 frame_cache 时一致（all_<image_size>_v1）
image_size=${IMAGE_SIZE:-224}

# 接触帧筛选：只挑接触帧训练（逐通道 std > 阈值），非接触帧随机保留 keep_ratio
# contact_filter=0 可关闭；首次运行会算并缓存各数据集 meta/contact_std.npz
contact_filter=${CONTACT_FILTER:-1}
contact_std_threshold=${CONTACT_STD_THRESHOLD:-0.5}
noncontact_keep_ratio=${NONCONTACT_KEEP_RATIO:-0.05}
# 构建 contact_std 缓存时逐 episode 顺序解码，每隔 stride 帧算一次 std（其余就近填充），加速首次缓存
contact_stride=${CONTACT_STRIDE:-1}

# 路径配置 (相对路径, 跨机器可移植)
dataset_root=playground/data
output_root=playground/results/backbones

# 输出/日志命名标签：当前时间（如 20260603_210836），可用 RUN_NAME 覆盖
run_tag=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}


# ===================数据模式自动识别（raw frame_cache vs LeRobot）===================
# pretrained_data 这类是「裸 frame_cache」：无 LeRobot meta，单路相机，需 --raw_frame_cache，
# 且跳过 stage1 warm-up / 关闭 contact_filter。下面按 meta/info.json 是否存在自动判定；
# 可用 RAW_FRAME_CACHE=1/0 强制覆盖。两类数据不能混在同一次训练里。
_has_lerobot=0; _has_raw=0
for ds in ${dataset_ids}; do
  if [ -f "${dataset_root}/${ds}/meta/info.json" ]; then _has_lerobot=1; else _has_raw=1; fi
done
if [ -z "${raw_frame_cache}" ]; then
  if [ "${_has_raw}" = "1" ] && [ "${_has_lerobot}" = "0" ]; then raw_frame_cache=1; else raw_frame_cache=0; fi
fi
if [ "${raw_frame_cache}" != "1" ] && [ "${_has_raw}" = "1" ] && [ "${_has_lerobot}" = "1" ]; then
  echo "ERROR: dataset_ids 同时含裸 frame_cache 与 LeRobot 数据集，无法同跑，请分开训练。"; exit 1
fi

# ===================初始化权重（统一 pretrained_path，按 arch 选择）===================
# 三种初始化由一个 pretrained_path 驱动，加载器自动识别来源：
#   scratch  -> 空，随机初始化
#   clip     -> HF CLIP 目录，加载 encoder+projection（有 B 和 L 两种）
#   anytouch -> AnyTouch 权重（.pth 或转好的 HF 目录），strict 加载完整 MAE（仅 L）
PM=playground/pretrained_models
pretrained_path=
case "${init_mode}" in
  scratch)
    pretrained_path=
    ;;
  clip)
    case "${arch}" in
      vit_l) pretrained_path=${PM}/CLIP-ViT-L-14-DataComp.XL-s13B-b90K ;;
      vit_b) pretrained_path=${PM}/CLIP-ViT-B-16-DataComp.XL-s13B-b90K ;;   # 需 HF 格式（bundled CLIP-B-16 仅 open_clip，需先转换）
      *)     echo "Unknown arch: ${arch} (expected vit_l|vit_b)"; exit 1 ;;
    esac
    ;;
  anytouch)
    if [ "${arch}" != "vit_l" ]; then
      echo "anytouch 仅有 ViT-L 权重，不支持 arch=${arch}（请用 scratch 或 clip）"; exit 1
    fi
    pretrained_path=${PM}/AnyTouch-ViT-L-16   # 或转换后的 ${PM}/anytouch_mae_vitl
    ;;
  *)
    echo "Unknown init_mode: ${init_mode} (expected scratch|clip|anytouch)"; exit 1
    ;;
esac

# 触觉相机 key 默认值（依赖前面识别出的 raw_frame_cache）
# 双臂数据 rm_umi_dual_pen_open 有四路 finger（left/right × finger0/1）。
# 注：本脚本底层是 argparse(nargs="+")，所以下面会把列表转成空格分隔再传 --camera_keys。
if [ -z "${tactile_keys}" ]; then
  if [ "${raw_frame_cache}" = "1" ]; then
    tactile_keys='[observation.images.cam_finger0]'
  else
    tactile_keys='[observation.images.left_cam_finger0,observation.images.left_cam_finger1,observation.images.right_cam_finger0,observation.images.right_cam_finger1]'
  fi
fi
# 列表 [a,b,c] -> 空格分隔 "a b c"
_tac_csv=${tactile_keys#[}; _tac_csv=${_tac_csv%]}
finger_cams=$(printf '%s' "${_tac_csv}" | tr ',' ' ')

# raw frame_cache 模式没有逐帧解码来源，contact_filter 不可用，强制关闭
if [ "${raw_frame_cache}" = "1" ]; then contact_filter=0; fi
contact_args=
if [ "${contact_filter}" = "1" ]; then
  contact_args="--contact_filter --contact_std_threshold ${contact_std_threshold} --noncontact_keep_ratio ${noncontact_keep_ratio} --contact_stride ${contact_stride}"
fi
# raw 模式给 train 传 --raw_frame_cache（跳过 LeRobot，直接读裸缓存）
raw_args=
if [ "${raw_frame_cache}" = "1" ]; then raw_args="--raw_frame_cache"; fi

# ===================输出路径（log 与 ckpt 同目录）===================
output_dir=${output_root}/${run_tag}_tacmae_${arch}_from_${init_mode}
log_file="${output_dir}/${run_tag}_tacmae_${arch}_${init_mode}.log"
tmp_log="$(mktemp "${TMPDIR:-/tmp}/${run_tag}_tacmae_${arch}_${init_mode}.XXXXXX.log")"  # 训练期间先把日志写到系统临时目录(不污染 tac_infra), 跑完再搬到 output_dir 下
mkdir -p "${output_dir}"

# 记录本次 run 用到的数据集（每行一个 dataset_id），便于回溯
printf '%s\n' ${dataset_ids} > "${output_dir}/datasets.txt"

# ===================启动器（单卡 python / 多卡 torchrun）===================
if [ "${num_processes}" -gt 1 ]; then
  launcher="torchrun --nproc_per_node=${num_processes} --master_port=${master_port} -m"
else
  launcher="python -m"
fi

# 与 train.sh 一致：参数打印 + 训练整体管道给 tee，日志先落临时文件，跑完再搬进 output_dir。
{
echo "Log file: $log_file"
echo "Dataset(s): ${dataset_ids}"
echo "Init mode: ${init_mode} | Arch: ${arch}"
echo "Pretrained path: ${pretrained_path:-<scratch>}"
echo "Sensor id: ${sensor_id} | Mask ratio: ${mask_ratio} | Val ratio: ${val_ratio}"
echo "Tactile keys: ${tactile_keys}  ->  --camera_keys ${finger_cams}"
echo "Data mode: $([ "${raw_frame_cache}" = "1" ] && echo "raw frame_cache (image_size=${image_size}, no LeRobot)" || echo "LeRobot")"
echo "Contact filter: ${contact_filter} (perchannel-std thr=${contact_std_threshold}, keep_ratio=${noncontact_keep_ratio})"
echo "Epochs: ${epochs} | Batch size: ${batch_size} | Num processes: ${num_processes}"
echo "Output dir: ${output_dir}"

if [ "${raw_frame_cache}" = "1" ]; then
  echo "[stage 1/2] skipped (raw frame_cache; 缓存已离线构建，无需 LeRobot warm-up)"
else
echo "[stage 1/2] Warm up dataset caches"
PYTHONPATH=${REPO_ROOT}:${PYTHONPATH} python -m vtla.tac_encoder.tactile_mae.process_data \
    --dataset_root ${dataset_root} \
    --dataset_ids ${dataset_ids} \
    --camera_keys ${finger_cams} \
    --val_ratio ${val_ratio} \
    --tolerance_s 0.1 \
    ${contact_args} \
    --num_workers ${num_workers}
fi

echo "[stage 2/2] Train tactile MAE backbone"
PYTHONPATH=${REPO_ROOT}:${PYTHONPATH} ${launcher} \
    vtla.tac_encoder.tactile_mae.train \
    --arch ${arch} \
    --pretrained_path "${pretrained_path}" \
    --dataset_root ${dataset_root} \
    --dataset_ids ${dataset_ids} \
    --camera_keys ${finger_cams} \
    --sensor_id ${sensor_id} \
    --mask_ratio ${mask_ratio} \
    --visible_loss_weight ${visible_loss_weight} \
    --use_sensor_token \
    --use_same_patchemb \
    --sensor_token_for_all --beta_start 0.0 --beta_end 0.75 \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --warmup_epochs ${warmup_epochs} \
    --weight_decay 0.1 \
    --blr 1e-5 \
    --amp_dtype ${amp_dtype} \
    --val_ratio ${val_ratio} \
    --tolerance_s 0.1 \
    --image_size ${image_size} \
    ${raw_args} \
    ${contact_args} \
    --num_workers ${num_workers} \
    --output_dir "${output_dir}"
} 2>&1 | tee "$tmp_log"
train_status=${PIPESTATUS[0]}   # 取管道第一段(训练进程)的真实退出码, tee 的退出码不可靠

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
