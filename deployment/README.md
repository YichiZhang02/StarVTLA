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

触觉 `uint16` 采集、精确字节平面和线性 `uint8` 派生格式统一遵循
[Flux 触觉数据编码与转换标准](TACTILE_ENCODING_STANDARD.md)。

配置字段可以通过命令行覆盖：

```bash
--robot.use_tactile=false
--robot.left_follower_ip=192.168.1.200
```

机器人标定文件按 `robot.type` 隔离，默认位于
`$HF_LEROBOT_CALIBRATION/robots/<robot_type>/calibration.json`。

单左臂新硬件使用 `realman_ugripper_left`：

```bash
--robot.type=realman_ugripper_left
```

该类型固定启用逻辑左臂，默认连接从臂 `192.168.1.201:8080` 和末端板
`192.168.1.10`，且默认不连接额外的顶部 USB 相机。

单左臂配置使用独立字段，不接受 dual 的 `arms`、`left_*_ip` 或 `right_*_ip` 参数：

```bash
--robot.follower_ip=192.168.1.201
--robot.board_ip=192.168.1.10
--robot.pc_host=192.168.1.102
```

安全检查：

```bash
python -m deployment.tools.hardware_check \
  --robot-type realman_ugripper_left \
  --stage existence
```

采集时使用独立的左主臂类型，使动作字段与机器人 `left_*` 特征一致：

该主臂默认串口为 `/dev/ttyRealmanUGripperLeftLeader`，通过 USB 设备序列号绑定，
不会占用旧双臂 rig 的 `/dev/ttyLeaderL`。

```bash
python -m deployment.collect \
  --robot.type=realman_ugripper_left \
  --teleop.type=left_realman_ugripper_leader \
  --dataset.repo_id=local/left_test \
  --dataset.single_task="hardware validation" \
  --dataset.num_episodes=1 \
  --dataset.push_to_hub=false
```

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

启用触觉时，触觉会和 wrist camera 一样注册为数据集 video feature。权威
`uint16` 帧使用无损 FFV1 Matroska，保存在同一个 `videos/` 层级：

```text
playground/data/<timestamp>_<name>/
├── videos/
│   ├── observation.images.left_cam_wrist/chunk-000/file-000.mp4
│   ├── observation.images.left_cam_finger0/chunk-000/file-000.mkv
│   ├── observation.images.left_cam_finger1/chunk-000/file-000.mkv
│   ├── observation.images.right_cam_finger0/chunk-000/file-000.mkv
│   └── observation.images.right_cam_finger1/chunk-000/file-000.mkv
└── meta/
    ├── info.json
    ├── tactile_encoding.json
    └── episodes/...
```

### 触觉采集编码

dmrobotics Flux SDK 的 `getDepth()` 和 `getDeformation2D()` 返回 `float32`。采集进程按
固定比例和偏置编码成 HWC `uint16`，不根据单个数据集动态计算 min/max：

```python
U16[..., 0] = clip(rint(depth    * 1000),         0, 65535)
U16[..., 1] = clip(rint(deform_x * 1000 + 30000), 0, 65535)
U16[..., 2] = clip(rint(deform_y * 1000 + 30000), 0, 65535)
```

通道索引固定为 `0=depth`、`1=deform_x`、`2=deform_y`。物理值解码为：

```python
depth    = U16[..., 0] / 1000
deform_x = (U16[..., 1] - 30000) / 1000
deform_y = (U16[..., 2] - 30000) / 1000
```

权威视频格式固定为：

```text
container: Matroska (.mkv)
codec: FFV1
pixel format: gbrp16le
decode format: rgb48le
dtype/layout: uint16 / HWC
encoding: tactile_u16_fixed_v1
```

FFV1 MKV 是后续审计、重新量化和转换的权威数据。解码为 `rgb48le` 后的三个数组通道
才是上述语义顺序；不要按播放器显示颜色推断通道。`meta/info.json` 会记录触觉 feature、
codec、pixel format、storage dtype 和视频路径，`meta/tactile_encoding.json` 记录固定比例、
偏置及通道定义；episode parquet 会像 wrist video 一样记录文件编号和时间范围。

训练前通过 `scripts/process_joint_data.sh` 将权威 MKV 转换成线性 `uint8` MP4，完整步骤、
量化公式和最终 YUV420 存储约定见 [Workflow Scripts](../scripts/README.md#触觉处理流程)。

重录时尚未提交的触觉 MKV 会一起删除。
触觉源端推送率由独立的 `--robot.tactile_max_fps` 控制，默认与采集频率一致为 30，
不会再受鱼眼 `stream_max_fps` 的限速策略影响。

也可以直接调用 Python 入口：

```bash
python -m deployment.collect \
  --robot.type=realman_ugripper_dual \
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
