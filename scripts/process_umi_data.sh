#!/bin/bash
# UMI 数据集离线预处理流水线: 去畸变(undist) -> 降分辨率(resize) -> UMI 位姿转末端位姿(umi2ee)。
# 用法: bash process_umi_data.sh <dataset_id> [size] [horizon]
#   只需指定 dataset_id, 其余参数全部自动 (与 train.sh 同风格)。
# 与 process_joint_data.sh 的唯一区别: 第 3 步用 convert_umi_to_eepose (UMI 已存末端位姿, 跳过 FK;
# 并在 meta/episodes 缺失时自动重建)。
# 产物 (非破坏, 逐级生成新副本):
#   <id>            (原始, 不动)
#   -> <id>_undist        鱼眼去畸变 + 居中裁 896  (仅腕部相机)
#   -> <id>_undist_<size> 降到 size×size (默认 256)  <-- 训练用这个
#   最后在 <id>_undist_<size> 上就地 convert_umi_to_eepose (加 EE 列)。
set -e
cd "$(dirname "$0")/.." || exit 1   # 切到仓库根, 服务器/本地通用
REPO_ROOT="$(pwd)"

# =================== 配置 (只有 dataset_id 必填) ===================
dataset_id=${1:?"用法: bash process_umi_data.sh <dataset_id> [size] [horizon]"}
size=${2:-256}        # 降分辨率目标边长 (默认 256, 给 224 裁剪留余量)
horizon=${3:-32}      # action_relative_ee 统计的最大 chunk 步长; 训练 chunk_size 须 <= 该值

# 可选 env 覆盖 (一般不用动)
crop=${CROP:-896}     # 去畸变后居中裁剪边长 (须与训练/推理一致)
jobs=${JOBS:-12}      # ffmpeg 并行 worker 数 (12 是这台机器实测甜点区; NVDEC 解码空出的 CPU 给编码用)
# CAMERAS / CALIB 留空 = 用 undistort 工具内置默认 (腕部相机 + tools/calib/x5_*.json)
cameras_arg=${CAMERAS:+--cameras ${CAMERAS}}
calib_arg=${CALIB:+--calib ${CALIB}}

# =================== 路径 (逐级派生) ===================
dataset_root=playground/data
src=${dataset_root}/${dataset_id}
undist=${dataset_root}/${dataset_id}_undist
final=${dataset_root}/${dataset_id}_undist_${size}

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}

echo "==================================================================="
echo "UMI 数据预处理: ${dataset_id}"
echo "  1) undist : ${src} -> ${undist}   (crop ${crop})"
echo "  2) resize : ${undist} -> ${final}   (size ${size})"
echo "  3) umi2ee (就地, 无 FK, 加 EE 列): ${final}   (horizon ${horizon})"
echo "==================================================================="

if [ ! -d "${src}" ]; then
  echo "错误: 源数据集不存在: ${src}"; exit 1
fi

# =================== 1) 去畸变 (原生分辨率 -> 896) ===================
# 顺序很重要: 先在原生 1920×1080 上去畸变并裁 896, 再降采样。反过来会糊且 FOV 不对。
if [ -d "${undist}" ]; then
  echo "[1/3] 已存在, 跳过去畸变: ${undist}"
else
  echo "[1/3] 去畸变 -> ${undist}"
  python tools/undistort_dataset_videos.py \
    --src "${src}" \
    --dst "${undist}" \
    --crop "${crop}" \
    --jobs "${jobs}" \
    ${cameras_arg} ${calib_arg}
fi

# =================== 2) 降分辨率 (896 -> size) ===================
if [ -d "${final}" ]; then
  echo "[2/3] 已存在, 跳过降分辨率: ${final}"
else
  echo "[2/3] 降分辨率 -> ${final}"
  python tools/downscale_dataset_videos.py \
    --src "${undist}" \
    --dst "${final}" \
    --size "${size}" \
    --jobs "${jobs}" 
fi

# =================== 3) UMI 位姿 -> 末端位姿 (就地, 幂等) ===================
# UMI 数据本身已存末端位姿, 跳过 FK; 若 meta/episodes 缺失会自动从 data+视频帧数重建。
echo "[3/3] convert_umi_to_eepose (就地) -> ${final}"
python tools/convert_umi_to_eepose.py \
  --root "${final}" \
  --horizon "${horizon}"

echo "==================================================================="
echo "完成 ✅  训练数据集: ${final}"
echo "  训练示例: bash train.sh ${dataset_id}_undist_${size} pi05 1 32 10000 false none episode_ee relative_ee"
echo "==================================================================="
