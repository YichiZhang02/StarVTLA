#!/bin/bash
# 计算数据集 state 均值并输出关节 JSON，供数据和硬件姿态诊断。
# 默认取每个 episode 首帧再跨 episode 平均 (= 机器人起始位姿的均值);
# --frames all 则对全部帧求全局均值 (与 meta/stats.json 里的 mean 一致, 只是按关节名好看地打印)。
# 用法: bash compute_mean_state.sh <dataset_id> [frames] [state_key]
#   dataset_id : playground/data 下的目录名 (不含路径)
#   frames     : first (默认, home 位姿) | all (全帧全局均值)
#   state_key  : 默认 observation.state (关节角); 也可传 EE 列名等
# 输出: stdout 是关节均值 JSON (仅关节, 不含 gripper);
#       stderr 是逐关节 mean/std/min/max 表 + gripper 均值 (供核对)。
set -e
cd "$(dirname "$0")/.." || exit 1   # 切到仓库根, 服务器/本地通用
REPO_ROOT="$(pwd)"

dataset_id=${1:-rm_umi_dual_260708_pen_in_case_merged_3_notac_undist_256}
frames=${2:-first}
state_key=${3:-observation.state}

dataset_root=playground/data
src=${dataset_root}/${dataset_id}

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}

if [ ! -d "${src}" ]; then
  echo "错误: 源数据集不存在: ${src}" >&2; exit 1
fi

python tools/compute_dataset_mean_state.py \
  --root "${src}" \
  --frames "${frames}" \
  --state-key "${state_key}"
