#!/bin/bash
# 关节数据集离线预处理流水线: 去畸变 -> 触觉 uint8 -> 全视频 resize -> 关节转末端位姿。
# 用法: bash process_joint_data.sh <dataset_id> [size] [horizon]
#   只需指定 dataset_id, 其余参数全部自动 (与 train.sh 同风格)。
# 只有 undist 创建副本；后续均在原目录处理，成功后追加后缀并改名:
#   有触觉: <id> -> <id>_undist -> <id>_undist_uint8 -> <id>_undist_uint8_<size>
#   无触觉: <id> -> <id>_undist -> <id>_undist_<size>
# 最后在 resize 后的数据集上就地 convert_joints_to_eepose (FK 加 EE 列)。
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1   # 切到仓库根, 服务器/本地通用
REPO_ROOT="$(pwd)"

# =================== 配置 (只有 dataset_id 必填) ===================
dataset_id=${1:-rm_umi_dual_260711_pen_in_case}
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
uint8=${undist}_uint8
final_with_tactile=${uint8}_${size}
final_without_tactile=${undist}_${size}

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}

inspect_tactile_state() {
  python -c '
import json
import sys
from pathlib import Path

info = json.loads((Path(sys.argv[1]) / "meta" / "info.json").read_text())
features = info.get("features", {})
encodings = {
    feature.get("tactile_encoding")
    for feature in features.values()
    if feature.get("tactile_encoding")
}
known = {"tactile_u16_fixed_v1", "tactile_u8_linear_v1"}
unknown = sorted(encodings - known)
unmarked = [
    key for key, feature in features.items()
    if feature.get("dtype") == "video"
    and not feature.get("tactile_encoding")
    and (
        feature.get("storage_dtype") == "uint16"
        or str(feature.get("video_path", "")).endswith(".mkv")
    )
]
if unmarked:
    print("unmarked")
elif unknown:
    print("unsupported:" + ",".join(unknown))
elif encodings == {"tactile_u16_fixed_v1"}:
    print("raw")
elif encodings == {"tactile_u8_linear_v1"}:
    print("uint8")
elif encodings == known:
    print("mixed")
else:
    print("none")
' "$1"
}

echo "==================================================================="
echo "关节数据预处理: ${dataset_id}"
echo "  1) undist : ${src} -> ${undist}   (crop ${crop})"
echo "  2) tactile uint16_to_uint8 (有触觉时, 就地): ${undist} -> ${uint8}"
echo "  3) resize 所有 MP4 (就地): -> *_${size}"
echo "  4) joint2ee (就地, FK 加 EE 列, horizon ${horizon})"
echo "==================================================================="

if [ ! -d "${src}" ]; then
  echo "错误: 源数据集不存在: ${src}"; exit 1
fi

if [ -d "${final_with_tactile}" ] && [ -d "${final_without_tactile}" ]; then
  echo "错误: 同时存在两个最终目录，无法判断应继续使用哪个:"
  echo "  ${final_with_tactile}"
  echo "  ${final_without_tactile}"
  exit 1
fi
if [ ! -d "${final_with_tactile}" ] && [ ! -d "${final_without_tactile}" ] \
  && [ -d "${undist}" ] && [ -d "${uint8}" ]; then
  echo "错误: 同时存在两个中间目录，无法安全恢复:"
  echo "  ${undist}"
  echo "  ${uint8}"
  exit 1
fi
if [ -d "${final_with_tactile}" ]; then
  state=$(inspect_tactile_state "${final_with_tactile}")
  if [ "${state}" != "uint8" ]; then
    echo "错误: ${final_with_tactile} 名称表示 uint8，但元数据状态为 ${state}"; exit 1
  fi
fi
if [ -d "${final_without_tactile}" ]; then
  state=$(inspect_tactile_state "${final_without_tactile}")
  if [ "${state}" != "none" ]; then
    echo "错误: ${final_without_tactile} 应是无触觉产物，但元数据状态为 ${state}"; exit 1
  fi
fi

# =================== 1) 去畸变 (原生分辨率 -> 896) ===================
# 顺序很重要: 先在原生 1920×1080 上去畸变并裁 896, 再降采样。反过来会糊且 FOV 不对。
if [ -d "${final_with_tactile}" ]; then
  echo "[1/4] 已有完整产物，跳过去畸变: ${final_with_tactile}"
elif [ -d "${final_without_tactile}" ]; then
  echo "[1/4] 已有完整产物，跳过去畸变: ${final_without_tactile}"
elif [ -d "${uint8}" ]; then
  echo "[1/4] 已存在触觉 uint8 阶段，跳过去畸变: ${uint8}"
elif [ -d "${undist}" ]; then
  echo "[1/4] 已存在, 跳过去畸变: ${undist}"
else
  echo "[1/4] 去畸变 -> ${undist}"
  python tools/undistort_dataset_videos.py \
    --src "${src}" \
    --dst "${undist}" \
    --crop "${crop}" \
    --jobs "${jobs}" \
    ${cameras_arg} ${calib_arg} 
fi

# =================== 2) 触觉 uint16 -> uint8 MP4 (就地) ===================
if [ -d "${final_with_tactile}" ]; then
  final=${final_with_tactile}
  echo "[2/4] 已存在, 跳过触觉 uint8 转换: ${final}"
elif [ -d "${final_without_tactile}" ]; then
  final=${final_without_tactile}
  echo "[2/4] 已存在, 无触觉数据集: ${final}"
else
  if [ -d "${uint8}" ]; then
    work=${uint8}
  else
    work=${undist}
  fi
  tactile_state=$(inspect_tactile_state "${work}")
  case "${tactile_state}" in
    raw)
      if [ "${work}" = "${uint8}" ]; then
        echo "错误: ${uint8} 的元数据仍是 uint16 触觉"; exit 1
      fi
      echo "[2/4] 触觉 uint16 -> uint8 lossless RGB MP4 (就地): ${work}"
      python tools/tactile_uint16_to_uint8.py --root "${work}" --jobs "${jobs}"
      mv -- "${work}" "${uint8}"
      work=${uint8}
      ;;
    uint8)
      if [ "${work}" != "${uint8}" ]; then
        echo "[2/4] 触觉已是 uint8，补充目录后缀: ${work} -> ${uint8}"
        mv -- "${work}" "${uint8}"
        work=${uint8}
      else
        echo "[2/4] 触觉已是 uint8: ${work}"
      fi
      ;;
    none)
      echo "[2/4] 未发现触觉，跳过 uint8 转换"
      ;;
    mixed|unmarked|unsupported:*)
      echo "错误: ${work} 的触觉编码状态无法安全处理: ${tactile_state}"; exit 1
      ;;
    *)
      echo "错误: 未知触觉编码状态: ${tactile_state}"; exit 1
      ;;
  esac

  # =================== 3) 所有 MP4 resize (就地) ===================
  final=${work}_${size}
  if [ -e "${final}" ]; then
    echo "错误: resize 目标已存在: ${final}"; exit 1
  fi
  echo "[3/4] resize 所有视觉/触觉 MP4 -> ${size}x${size} (就地): ${work}"
  python tools/downscale_dataset_videos.py \
    --root "${work}" \
    --size "${size}" \
    --jobs "${jobs}"
  mv -- "${work}" "${final}"
fi

if [ -d "${final_with_tactile}" ] || [ -d "${final_without_tactile}" ]; then
  echo "[3/4] resize 产物: ${final}"
fi

# =================== 4) 关节 -> 末端位姿 (就地, 幂等) ===================
echo "[4/4] convert_joints_to_eepose (就地) -> ${final}"
python tools/convert_joints_to_eepose.py \
  --root "${final}" \
  --horizon "${horizon}"

echo "==================================================================="
echo "完成: 训练数据集: ${final}"
echo "  训练示例: bash train.sh $(basename "${final}") pi05 1 32 10000 false none episode_rot6d relative_rot6d"
echo "==================================================================="
