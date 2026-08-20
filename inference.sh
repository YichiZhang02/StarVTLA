#!/bin/sh
set -e
cd "$(dirname "$0")"   # 切到仓库根, 使 playground/... 相对路径生效, 服务器/本地通用

# =================== 可调参数 ===================
pretrained_id=${1:-20260624_230105_rm_umi_dual_260617_pen_place_cap_notop_undist_256_diffusion_wristonly_true_tactile_none_state_episode_ee_action_relative_ee}
step=${2:-15000}

# 推理时的 action chunk 旋钮 (空=不覆盖, 用 checkpoint config.json 里的训练值):
#   n_action_steps      执行 chunk 里的前几个动作后再重规划 (chunk[:n])。chunk_size 与训练同步, 不用动。
#   action_start_offset 执行前先丢掉 chunk 的前 m 个动作 (chunk[m:]); 实际执行 chunk[m : m+n]。
# 顺序须与 README / 用法行一致: $3=n_action_steps, $4=action_start_offset。
n_action_steps=${3:-}
action_start_offset=${4:-}

# 复位选项
reset_before_episode=${6:-true}           # 与其他参数独立: true=每个 episode 前复位
home_joints=                              # 留空：连接时读取当前关节角作为本次推理的固定复位点
home_duration_s=2.0                       # 平滑复位耗时（秒）
# 显式 home 示例：
# home_joints='{"left_main_joint1": -0.109262, "left_main_joint2": 0.235679, "left_main_joint3": 0.118975, "left_main_joint4": 1.265910, "left_main_joint5": 0.034194, "left_main_joint6": 1.589552, "left_main_joint7": -0.278270, "right_main_joint1": 0.041508, "right_main_joint2": 0.100594, "right_main_joint3": 0.046601, "right_main_joint4": 1.527823, "right_main_joint5": 0.011595, "right_main_joint6": 1.477732, "right_main_joint7": 0.472311}'

# ===============================================
# step 自动补零到 6 位: 5000 -> 005000 (expr 强制十进制, 兼容已带前导零的输入, POSIX sh 可用)
step=$(printf "%06d" "$(expr "$step" + 0)")

policy_path=playground/results/models/${pretrained_id}/checkpoints/${step}/pretrained_model
echo "测试policy: ${policy_path}"

# 按时间命名: local/<时间戳>_<基础名>
name=${pretrained_id}_step_${step}    
repo_id="local/eval_$(date +%Y%m%d_%H%M%S)_${name}"
echo "录制数据集: ${repo_id}  ->  playground/eval/${repo_id##*/}"

# =================== 启动 (match_policy 自动对齐硬件 + 任务) ===================
set -- python -m deployment.inference \
  "--robot.type=realman_ugripper_dual" \
  "--policy.path=${policy_path}" \
  "--dataset.repo_id=${repo_id}" \
  "--match_policy=true" \
  "--reset_before_episode=${reset_before_episode}" \
  "--robot.home_duration_s=${home_duration_s}" \
  "--robot.home_gripper=1.0" \
  "--robot.max_ee_pos_step_m=0.1"

# 空值不产生 override：action chunk 使用 checkpoint 默认，home 使用连接时当前关节角。
[ -n "${n_action_steps}" ] && set -- "$@" "--policy.n_action_steps=${n_action_steps}"
[ -n "${action_start_offset}" ] && set -- "$@" "--policy.action_start_offset=${action_start_offset}"
[ -n "${home_joints}" ] && set -- "$@" "--robot.home_joints=${home_joints}"

exec "$@"  # ee 的 max step 初次建议 0.01，确认轨迹后再使用 0.1

# --- 手动模式示例 (关掉自动对齐时, 硬件/任务需自己给) ---
# python -m deployment.inference \
#   --robot.type=realman_ugripper_dual \
#   --policy.path=${policy_path} \
#   --dataset.repo_id=${repo_id} \
#   --match_policy=false \
#   --robot.use_tactile=false \
#   --dataset.single_task="Grasp the cap and pull it off the pen."
