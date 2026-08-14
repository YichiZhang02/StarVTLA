# Deployment

`deployment/` 负责睿尔曼 RM75b 双臂的数据采集、硬件自检和 policy 推理。所有命令默认从仓库根目录运行；根目录脚本会自动切换到正确工作目录。

## 目录

```text
deployment/
├── hardware/       # 机械臂、夹爪、相机和触觉设备封装
├── robots/         # RobotConfig 与整机组合
├── teleoperators/  # 主臂遥操作配置
├── sdk/            # 厂商 SDK，本地放置
├── tools/          # 硬件检查和辅助命令
├── collect.py      # 数据采集入口
└── inference.py    # Policy 推理入口
```

## 厂商 SDK

硬件封装通过 `hardware/_sdk_paths.py` 将本地 SDK 加入 Python import path。目录约定：

```text
deployment/sdk/
├── Robotic_Arm/          # 睿尔曼机械臂 SDK
│   └── libs/linux_x86/libapi_c.so
├── dm_lingkong_grip/     # 领控电爪客户端
├── fish_camera_client/   # 鱼眼相机 gRPC 客户端
└── dmrobotics/           # Flux 触觉传感器 SDK
```

仓库环境已包含 `Robotic_Arm`。其余 SDK 需要按上述目录名放置；目录不存在时不会在启动阶段自动下载，而会在对应硬件模块 import 时报告错误。

## Robot Config

机器人配置位于：

```text
deployment/robots/<robot_type>/config_<robot_type>.py
```

通过 `@RobotConfig.register_subclass("<robot_type>")` 注册，并使用以下参数选择：

```bash
--robot.type=<robot_type>
```

`realman_ugripper_dual` 的配置见 [config_realman_ugripper_dual.py](robots/realman_ugripper_dual/config_realman_ugripper_dual.py)。上机前重点确认：

- 左右从臂 IP 和 TCP 端口。
- 左右控制板 IP。
- 本机触觉 UDP 回传 IP。
- 左右夹爪满行程。
- 触觉图像尺寸、深度和形变编码范围。
- 启用的机械臂、顶部相机、腕部相机和触觉开关。

配置字段可以通过命令行覆盖：

```bash
--robot.use_tactile=false
--robot.left_follower_ip=192.168.1.200
```

`--robot.id` 用于区分同型号的不同机器人，并影响标定文件目录。

## 硬件自检

先运行不连接硬件的存在性检查：

```bash
python -m deployment.tools.hardware_check
```

连接相机和触觉并抓取一帧：

```bash
python -m deployment.tools.hardware_check --stage camera --show
```

主从同步会实际驱动从臂，必须确认机器人周围安全、急停可用，并显式传入确认参数：

```bash
python -m deployment.tools.hardware_check --stage teleop --confirm-move
```

## 数据采集

推荐通过根目录包装脚本启动：

```bash
bash collect.sh <name> <single_task> <num_episodes>
```

示例：

```bash
bash collect.sh rm_tactile_demo "抓笔" 30
```

脚本自动生成带时间戳的 repo ID。默认输出：

```text
playground/data/<timestamp>_<name>/
```

也可以直接调用 Python 入口：

```bash
python -m deployment.collect \
  --robot.type=realman_ugripper_dual \
  --robot.id=realman_dual \
  --teleop.type=bi_realman_ugripper_leader \
  --dataset.repo_id=local/$(date +%Y%m%d_%H%M%S)_grab_pen \
  --dataset.single_task="抓笔" \
  --dataset.num_episodes=50 \
  --dataset.fps=30 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=15 \
  --robot.use_tactile=true \
  --dataset.push_to_hub=false
```

采集过程中可以提前结束并保存当前 episode。`episode_time_s` 是单集最长时间，`reset_time_s` 是 episode 间人工复位场景的时间。

## Policy 推理

推荐入口：

```bash
bash inference.sh \
  <pretrained_id> <step> [n_action_steps] [action_start_offset]
```

`pretrained_id` 是 `playground/results/models/` 下的 run 目录名。脚本加载：

```text
playground/results/models/<pretrained_id>/checkpoints/<step_6_digits>/pretrained_model
```

示例：

```bash
bash inference.sh \
  rm_umi_dual_pen_open_diffusion_wristonly_false_tactile_none_state_joint \
  5000
```

Action chunk 覆盖参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `n_action_steps` | checkpoint 配置 | 每次规划后执行的 action 数量 |
| `action_start_offset` | checkpoint 配置 | 执行前丢弃的陈旧 action 数量 |

实际执行区间为：

```text
chunk[action_start_offset : action_start_offset + n_action_steps]
```

必须满足两者之和不大于训练时的 `chunk_size`。

```bash
# 执行前 8 个 action
bash inference.sh <id> 5000 8

# 丢弃前 4 个，再执行 8 个
bash inference.sh <id> 5000 8 4
```

直接调用并自动匹配 checkpoint 的硬件和任务配置：

```bash
python -m deployment.inference \
  --robot.type=realman_ugripper_dual \
  --policy.path=<path_to_pretrained_model> \
  --dataset.repo_id=local/eval_$(date +%Y%m%d_%H%M%S)_pen \
  --match_policy=true
```

推理录制默认保存到 `playground/eval/`。

## Home Joints

连接双臂并读取当前关节位置：

```bash
python -m deployment.tools.read_home_joints
```

只读取单臂：

```bash
python -m deployment.tools.read_home_joints --side left
python -m deployment.tools.read_home_joints --side right
```

覆盖 IP：

```bash
python -m deployment.tools.read_home_joints \
  --left-ip 192.168.1.200 \
  --right-ip 192.168.1.201
```

stdout 输出可以直接用于 `--robot.home_joints` 的参数字符串；逐关节统计写入 stderr。

离线从数据集估计 home joints 的方式见 [scripts/README.md](../scripts/README.md#state-均值)。

## 腕部鱼眼去畸变

如果训练数据经过鱼眼去畸变，推理必须执行相同几何变换。`--match_policy=true` 按以下顺序判定：

1. 读取训练数据集 `meta/info.json` 中的 `undistort` marker 和 crop。
2. 训练数据不可访问时，根据数据集名称是否包含 `undist` 兜底。
3. 两者均不满足时关闭去畸变。

在线变换位于 `hardware/wrist_cameras/undistort.py`，标定文件位于同目录的 `calib/`。

手动覆盖：

```bash
--robot.undistort_wrist=true
--robot.undistort_crop=896
```

采集阶段没有 policy 可以用于自动判断，因此默认保存原始鱼眼图像。不要在采集端丢弃原始视场；训练前使用 [离线去畸变工具](../tools/README.md#鱼眼去畸变) 生成副本。
