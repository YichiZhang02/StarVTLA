# Deployment

`deployment/` 提供完整机器人配置、硬件封装、数据采集、硬件自检和 policy 推理。命令均从仓库根目录运行。

## 机器人注册表

目前支持两个机器人类型：

| `robot_type` | 臂布局 | `kinematics_force_type` | `teleop_type` |
| --- | --- | --- | --- |
| `rm_base_umi_dual` | `("right", "left")` | `base` | `bi_realman_ugripper_leader` |
| `rm_isf_umi_left` | `("left",)` | `isf` | `left_realman_ugripper_leader` |

注册信息由具体 RobotConfig 自己声明：

```text
deployment/robots/<robot_type>/
├── config_<robot_type>.py
└── <robot_type>.py
```

每个新机器人必须：

1. 使用 `@RobotConfig.register_subclass("<robot_type>")` 注册唯一名称。
2. 声明 `kinematics_force_type`，用于选择 RealMan FK/IK。
3. 声明 `kinematics_sides`，用于校验数据集的单臂或双臂 feature 布局。
4. 声明 `teleop_type`，并确保对应 TeleoperatorConfig 已注册。
5. 在 `deployment/robots/__init__.py` 导入配置，使注册在 CLI 解析前完成。

采集端不会维护另一份机器人名称列表。`deployment.collect` 直接查询 RobotConfig 注册表，并在连接硬件前拒绝未注册类型、缺失遥操作器或显式错配的遥操作器。`drag` 模式不创建遥操作器。

## Robot Type 数据契约

`robot_type` 是物理构型和运动学的唯一公开参数：

```text
robot.type
  -> collect 写入 meta/info.json
  -> joint-to-EE 读取并选择 B/ISF FK
  -> train 写入 checkpoint config.json
  -> inference 读取并选择 RobotConfig + B/ISF FK/IK
```

不支持旧名称兼容、默认构型或基于数据集名称的猜测。以下情况都会立即失败：

- 数据集缺少 `robot_type`。
- `robot_type` 不在 RobotConfig 运动学注册表中。
- 单臂/双臂 feature 与 `kinematics_sides` 不一致。
- checkpoint 缺少或包含未知 `robot_type`。

## 厂商 SDK

硬件模块通过 `deployment/hardware/_sdk_paths.py` 加载本地 SDK：

```text
deployment/sdk/
├── Robotic_Arm/          RealMan SDK 和 libs/linux_x86/libapi_c.so
├── dm_lingkong_grip/     领控电爪客户端
├── fish_camera_client/   鱼眼相机 gRPC 客户端
└── dmrobotics/           Flux 触觉传感器 SDK
```

SDK 不会自动下载。缺少某个 SDK 时，对应硬件模块会在导入或连接阶段报错。

## 硬件配置

双臂配置见 [config_rm_base_umi_dual.py](robots/rm_base_umi_dual/config_rm_base_umi_dual.py)，单左臂配置见 [config_rm_isf_umi_left.py](robots/rm_isf_umi_left/config_rm_isf_umi_left.py)。上机前应逐项核对：

- 从臂 IP、TCP 端口和左右臂物理对应关系。
- 末端板 IP、夹爪行程和 CAN 参数。
- 本机触觉 UDP 回传 IP 与端口。
- 腕部相机、顶部相机和触觉开关。
- `home_joints`、复位时间和 EE 单步安全限制。

配置字段可通过 CLI 覆盖。例如：

```bash
python -m deployment.collect \
  --robot.type=rm_isf_umi_left \
  --robot.follower_ip=192.168.1.201 \
  --robot.board_ip=192.168.1.10 \
  --robot.pc_host=192.168.1.102 \
  --robot.use_tactile=false \
  --mode=drag \
  --dataset.repo_id=local/hardware_test \
  --dataset.single_task="hardware validation" \
  --dataset.num_episodes=1 \
  --dataset.push_to_hub=false
```

机器人标定默认保存在 `$HF_LEROBOT_CALIBRATION/robots/<robot_type>/calibration.json`，不同构型不会共享标定文件。

## 硬件自检

存在性检查不驱动机器人：

```bash
python -m deployment.tools.hardware_check \
  --robot-type rm_isf_umi_left \
  --stage existence
```

连接相机和触觉并显示一帧：

```bash
python -m deployment.tools.hardware_check \
  --robot-type rm_isf_umi_left \
  --stage camera \
  --show
```

遥操作检查会实际驱动从臂。确认工作区清空、急停可用后再运行：

```bash
python -m deployment.tools.hardware_check \
  --robot-type rm_isf_umi_left \
  --stage teleop \
  --confirm-move
```

## 数据采集

在 [collect.sh](../collect.sh) 顶部设置 `robot_type`，然后运行：

```bash
bash collect.sh <name> <task_text> <num_episodes> [teleop|drag] \
  [drag_gripper_close_value] [reset_before_episode]
```

示例：

```bash
bash collect.sh insert_easy \
  "insert the object to the hole" 25 drag 0.3 true
```

输出目录为：

```text
playground/data/<robot_type>_<YYYYMMDD>_<name>/
```

采集引擎把 `robot.robot_type` 原样写入 `meta/info.json.robot_type`。不要在采集后通过目录名推断类型。

两种模式的区别：

| 模式 | 控制来源 | 遥操作器 |
| --- | --- | --- |
| `teleop` | RobotConfig 声明的 leader | 自动创建并校验 |
| `drag` | 人工拖动从臂 | 不连接 leader |

`reset_before_episode=true` 时，首个 episode 开始前回到固定 home。录制中按左键重录或右键保存时，会先停止拖动并复位，再清空或提交内存中的 episode；由于机器人已经在 home，下一轮开始前不会重复复位。collect 和 inference 共用这一顺序。`home_joints` 为空时，连接后读取当前关节位置并在本次运行中固定使用；上机前先确认该姿态安全。

## 触觉采集编码

Flux SDK 输出 `float32` depth 和二维 deformation。采集时使用固定比例和偏置编码为 HWC `uint16`：

```python
u16[..., 0] = clip(round(depth * 1000), 0, 65535)
u16[..., 1] = clip(round(deform_x * 1000 + 30000), 0, 65535)
u16[..., 2] = clip(round(deform_y * 1000 + 30000), 0, 65535)
```

固定通道语义为 `depth, deform_x, deform_y`。反解公式为：

```python
depth = u16[..., 0] / 1000
deform_x = (u16[..., 1] - 30000) / 1000
deform_y = (u16[..., 2] - 30000) / 1000
```

权威存储格式为：

```text
encoding:    tactile_u16_fixed_v1
container:   Matroska (.mkv)
codec:       FFV1
pixel format:gbrp16le
decode:      rgb48le, uint16 HWC
```

不要根据播放器显示颜色判断通道。`meta/info.json` 记录 feature 和视频编码，`meta/tactile_encoding.json` 记录比例、偏置和通道定义。训练前由 `scripts/process_joint_data.sh` 生成 `tactile_u8_linear_v1` 派生视频；原始 MKV 是唯一可用于数值审计和重新量化的来源。

## Policy 推理

推荐入口：

```bash
bash inference.sh <run_id> <step> [n_action_steps] [action_start_offset]
```

脚本加载：

```text
playground/results/models/<run_id>/checkpoints/<step_6_digits>/pretrained_model
```

例如 `step=3000` 会读取 `checkpoints/003000/pretrained_model`。推理录像默认写入 `playground/eval/`。

机器人身份始终归 checkpoint 所有。`deployment.inference` 会在连接硬件前：

1. 读取 checkpoint 的 `robot_type`。
2. 将启动时的 RobotConfig 替换成对应的注册类型。
3. 选择匹配的 B/ISF 在线 FK/IK。
4. 根据 action representation 选择 `joint` 或 `ee` 动作空间。
5. 在 `match_policy=true` 时同步触觉、额外相机、任务文本和腕部去畸变配置。

即使设置 `match_policy=false`，第 1 至 3 步也不会关闭。缺失 `robot_type` 的 checkpoint 不可用于当前真机链路。

Action chunk 实际执行范围为：

```text
chunk[action_start_offset : action_start_offset + n_action_steps]
```

两者之和必须不超过 checkpoint 的 `chunk_size`。

## 腕部去畸变

采集保存原始鱼眼图像；离线处理先在原始分辨率去畸变并中心裁剪，再缩放到训练尺寸。推理时 `match_policy=true` 根据训练配置启用相同变换，默认裁剪尺寸为 `896`。

标定文件位于：

```text
tools/calib/x5_left_intrinsics.json
tools/calib/x5_right_intrinsics.json
```

手动覆盖仅用于明确知道 checkpoint 图像几何契约的场景：

```bash
--robot.undistort_wrist=true
--robot.undistort_crop=896
```

## 安全

- 首次连接使用硬件自检，不直接启动 policy。
- 复位或遥操作前确认急停有效，机械臂工作区无人且无障碍物。
- EE 推理初次运行使用较小的 `max_ee_pos_step_m`，观察轨迹后再调整。
- 不要让 B checkpoint 控制 ISF 机器人，或让单臂 checkpoint 控制双臂机器人；代码会阻止这种错配，不应绕过校验。
