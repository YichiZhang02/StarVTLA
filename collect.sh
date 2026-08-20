#!/bin/sh
set -e
cd "$(dirname "$0")"   # 切到仓库根, 使 playground/... 相对路径生效, 服务器/本地通用

# =================== 可调参数 ===================
name=${1:-rm_tactile_demo1}                       # 数据集基础名
single_task=${2:-"Grab the object"}             # 任务文字描述 (会写入每一帧)
num_episodes=${3:-30}                             # 录制集数

# 按时间命名: local/<时间戳>_<基础名>
repo_id="local/$(date +%Y%m%d_%H%M%S)_${name}"

robot_type=realman_ugripper_left              # 双臂 (触觉随 use_tactile, 默认开)
teleop_type=left_realman_ugripper_leader
fps=30
episode_time_s=60                                # 每集最长录制秒数 (可中途按右键提前保存)
reset_time_s=15                                  # 集间复位场景秒数
# 不传 --dataset.root: collect.py 默认存到 playground/data/<repo_id 末段> (相对路径)

# =================== 启动 ===================
python -m deployment.collect \
  --robot.type=${robot_type} \
  --teleop.type=${teleop_type} \
  --dataset.repo_id=${repo_id} \
  --dataset.single_task="${single_task}" \
  --dataset.num_episodes=${num_episodes} \
  --dataset.fps=${fps} \
  --dataset.episode_time_s=${episode_time_s} \
  --dataset.reset_time_s=${reset_time_s} \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --display_data=false \
  --play_sounds=true
