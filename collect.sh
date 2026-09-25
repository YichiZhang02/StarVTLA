#!/bin/sh
set -e
cd "$(dirname "$0")"   # 切到仓库根, 使 playground/... 相对路径生效, 服务器/本地通用

# =================== 可调参数 ===================
robot_type=rm_isf_umi_left                    # rm_base_umi_dual | rm_isf_umi_left

dataset_source_group=${1:-Daimon/realman_single}  # source/group
name=${2:-wipe_board}
single_task=${3:-"wipe the board"}        # 任务文字描述 (会写入每一帧)
num_episodes=${4:-100}                      # 录制集数
mode=${5:-drag}                          # teleop | drag
drag_gripper_close_value=${6:-0.3}         # 0=最紧, 1=全开

# 复位选项
reset_before_episode=${7:-true}           # true=按左右键结束时先复位确认，再保存或重录
home_duration_s=2.0                       # 平滑复位耗时（秒）
home_joint_tolerance_deg=1.0              # 关节反馈到位容差（度）
home_settle_timeout_s=2.0                 # 2s 后未到位时最多继续保持等待的时间
    
# 与原采集命名规则一致：robot_type_YYYYMMDD_name。
dataset_id="${robot_type}_$(date +%Y%m%d)_${name}"
repo_id="local/${dataset_id}"


fps=30
episode_time_s=300                                # 每集最长录制秒数 (可中途按右键提前保存)
# collect.py 将 source/group 和 dataset_id 解析为 playground/data/<source>/<group>/<dataset_id>

# =================== 启动 ===================
set -- python -m deployment.collect \
  "--mode=${mode}" \
  "--robot.type=${robot_type}" \
  "--reset_before_episode=${reset_before_episode}" \
  "--robot.home_duration_s=${home_duration_s}" \
  "--robot.home_joint_tolerance_deg=${home_joint_tolerance_deg}" \
  "--robot.home_settle_timeout_s=${home_settle_timeout_s}"

set -- "$@" \
  "--drag_gripper_close_value=${drag_gripper_close_value}" \
  "--dataset_source_group=${dataset_source_group}" \
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
