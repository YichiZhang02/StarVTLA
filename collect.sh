#!/bin/sh
set -e
cd "$(dirname "$0")"   # 切到仓库根, 使 playground/... 相对路径生效, 服务器/本地通用

# =================== 可调参数 ===================
robot_type=rm_isf_umi_left                    # rm_base_umi_dual | rm_isf_umi_left

name=${1:-insert_easy_robust}                # 数据集基础名
single_task=${2:-"insert the object to the hole"}        # 任务文字描述 (会写入每一帧)
num_episodes=${3:-25}                      # 录制集数
mode=${4:-drag}                          # teleop | drag
drag_gripper_close_value=${5:-0.3}         # 0=最紧, 1=全开

# 复位选项
reset_before_episode=${6:-true}           # 与 mode 独立: true=每个 episode 前复位
home_joints=                              # 留空：连接时读取当前关节角作为本次采集的固定复位点
home_duration_s=4.0                       # 平滑复位耗时（秒）

# 按时间命名: local/<时间戳>_<基础名>
repo_id="local/${robot_type}_$(date +%Y%m%d)_${name}"


fps=30
episode_time_s=60                                # 每集最长录制秒数 (可中途按右键提前保存)
# 不传 --dataset.root: collect.py 默认存到 playground/data/<repo_id 末段> (相对路径)

# =================== 启动 ===================
set -- python -m deployment.collect \
  "--mode=${mode}" \
  "--robot.type=${robot_type}" \
  "--reset_before_episode=${reset_before_episode}" \
  "--robot.home_duration_s=${home_duration_s}"

if [ -n "${home_joints}" ]; then
  set -- "$@" "--robot.home_joints=${home_joints}"
fi

set -- "$@" \
  "--drag_gripper_close_value=${drag_gripper_close_value}" \
  "--dataset.repo_id=${repo_id}" \
  "--dataset.single_task=${single_task}" \
  "--dataset.num_episodes=${num_episodes}" \
  "--dataset.fps=${fps}" \
  "--dataset.episode_time_s=${episode_time_s}" \
  "--dataset.video=true" \
  "--dataset.push_to_hub=false" \
  "--display_data=false" \
  "--play_sounds=true"

exec "$@"
