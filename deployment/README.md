# Deployment

`deployment/` 提供完整机器人配置、硬件封装、数据采集、硬件自检和 policy 推理。命令均从仓库根目录运行。

## 机器人注册表

目前支持以下机器人类型：

| `robot_type` | 臂布局 | `kinematics_force_type` | `teleop_type` |
| --- | --- | --- | --- |
| `rm_base_umi_dual` | `("right", "left")` | `base` | `rm_leader_dual` |
| `rm_isf_umi_left` | `("left",)` | `isf` | `rm_leader_left` |
| `rm_isf_umi_right` | `("right",)` | `isf` | `rm_leader_right` |

主臂串口使用与 rig 构型绑定的稳定 udev 名称：

| `teleop_type` | 左主臂串口 | 右主臂串口 |
| --- | --- | --- |
| `rm_leader_dual` | `/dev/ttyRealmanBaseLeaderL` | `/dev/ttyRealmanBaseLeaderR` |
| `rm_leader_left` | `/dev/ttyRealmanISFLeaderL` | - |
| `rm_leader_right` | - | `/dev/ttyRealmanISFLeaderR` |

`rm_isf_umi_right` 默认连接机械臂 `192.168.1.200:8080`、末端板/夹爪
`192.168.1.11:55551`，右主臂串口使用 `/dev/ttyRealmanISFLeaderR`。

仓库不保存与某台主机或某个 USB 设备绑定的 udev 规则。首次在一台机器上使用时，
需要根据该机器实际连接的主臂创建本地规则：

1. 每次只连接一条主臂，确认当前串口节点，并查看设备的唯一属性：

   ```bash
   ls -l /dev/serial/by-id/
   udevadm info --attribute-walk --name=/dev/ttyUSB0
   ```

   将 `/dev/ttyUSB0` 替换为实际节点，记录可稳定区分设备的 `idVendor`、
   `idProduct` 和 `serial`。如果设备没有唯一序列号，不要仅按临时的
   `/dev/ttyUSB*` 编号绑定；应结合物理 USB 端口属性区分。

/

3. 重载规则、重新触发设备并检查软链接：

   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger --subsystem-match=tty --action=add
   sudo udevadm settle
   ls -l /dev/ttyRealman*
   ```

规则只存在于本机的 `/etc/udev/rules.d/`，不会随仓库同步。更换主机、USB 转串口设备
或主臂后，应重新读取设备属性并配置，不能直接复用其他机器的序列号。

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

`robot_type` 是物理构型和运动学的唯一公开参数。普通真机数据链路为：

```text
robot.type
  -> collect 写入 meta/info.json
  -> joint-to-EE 读取并选择 B/ISF FK
  -> train 写入 checkpoint config.json
  -> inference 读取并选择 RobotConfig + B/ISF FK/IK
```

unified-format UMI 数据是唯一例外：dataset 和 checkpoint 都保存 `robot_type=umi`，训练不把它
改成具体机械臂。推理 UMI checkpoint 时必须显式传入具体 `--robot.type`，并以该 CLI 类型选择
RobotConfig 和 B/ISF FK/IK；非 UMI checkpoint 仍忽略 CLI 类型并以 checkpoint 为准。

不支持旧名称兼容或基于数据集名称的猜测。以下情况都会立即失败：

- 数据集缺少 `robot_type`。
- 非 UMI 数据集的 `robot_type` 不在 RobotConfig 运动学注册表中。
- 单臂/双臂 feature 与 `kinematics_sides` 不一致。
- checkpoint 缺少或包含未知 `robot_type`，或 UMI checkpoint 未显式提供具体 CLI 类型。

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

遥操作检查会实际驱动从臂。确认工作区清空、急停可用后再运行。hardware check
使用 `2 deg/s` 的逐关节目标角速度限制，参数未指定时也默认 `2 deg/s`：

```bash
python -m deployment.tools.hardware_check \
  --robot-type rm_isf_umi_left \
  --stage teleop \
  --confirm-move \
  --max-joint-speed-deg-s 2
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
bash inference.sh \
  <pretrained_id> <step> [robot_type] [n_action_steps] [action_start_offset] \
  [control_fps] [reset_before_episode] [single_task]
```

脚本加载：

```text
playground/results/models/<pretrained_id>/checkpoints/<step_6_digits>/pretrained_model
```

例如 `step=3000` 会读取 `checkpoints/003000/pretrained_model`。推理录像默认写入 `playground/eval/`。

`control_fps` 控制机器人 action 下发和推理录像的目标频率，必须为正整数，默认 `30 Hz`。例如下面以 `20 Hz` 下发 16 个 action，并从 chunk 的第 6 个位置开始执行：

```bash
pretrained_id=20260821_rm_isf_umi_left_20260820_insert_easy_precise_undist_uint8_256_starvla_groot_wristonly_true_tactile_none_state_absolute_rot6d_action_relative_rot6d_aug_strong
bash inference.sh "${pretrained_id}" 3000 rm_isf_umi_left 16 6 20 true \
  "Grasp the cap and pull it off the pen."
```

训练数据的时间基准是 `30 Hz`；降低下发频率会放慢真实轨迹，实际可达到的频率还受新 chunk 的同步推理耗时限制。

`single_task` 默认留空，由 `match_policy` 从 checkpoint 自动读取。多数据集混合训练的多任务模型可以显式传入任务文本；该文本会作为本次推理运行的语言语义输入，并且不会被 `match_policy` 覆盖。

普通 checkpoint 的机器人身份归 checkpoint 所有。`deployment.inference` 会在连接硬件前：

1. 读取 checkpoint 的 `robot_type`。
2. 将启动时的 RobotConfig 替换成对应的注册类型。
3. 选择匹配的 B/ISF 在线 FK/IK。
4. 根据 action representation 选择 `joint` 或 `ee` 动作空间。
5. 在 `match_policy=true` 时同步触觉、额外相机、任务文本和腕部去畸变配置。

若 checkpoint 的类型为 `umi`，第 2 至 3 步改为使用显式传入的具体 `--robot.type`；未传、仍传
`umi` 或单/双臂布局不匹配都会在连接硬件前失败。非 UMI checkpoint 即使收到不同 CLI 类型也
仍以 checkpoint 为准。该规则不受 `match_policy=false` 影响。

Action chunk 实际执行范围为：

```text
chunk[action_start_offset : action_start_offset + n_action_steps]
```

两者之和必须不超过 checkpoint 的 `chunk_size`。

`action_gap` 是训练时写入 checkpoint 的 GT 时间偏移，不是在线切片参数。若训练使用
`action_gap=6`，模型 chunk 第 0 项对应 `t+6`；推理再设置 `action_start_offset=4` 时，实际首先执行的目标对应 `t+10`。因此有效首目标偏移为：

```text
action_gap + action_start_offset
```

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
