# StarVTLA

StarVTLA 是面向视觉、触觉和机器人动作学习的训练与真机部署仓库。它包含 LeRobot 数据采集与处理、ACT/Diffusion/pi05/StarVLA-GR00T/FastWAM/Dream-Tac 策略训练、触觉 MAE 预训练，以及 RealMan 机械臂的在线推理。

## 支持的机器人

机器人身份是数据和模型契约的一部分。目前支持以下严格名称：

| `robot_type` | 构型 | RealMan FK/IK | 遥操作器 |
| --- | --- | --- | --- |
| `rm_base_umi_dual` | base/B 版双臂 + UMI 夹爪 | `RM_MODEL_RM_B_E` | `rm_leader_dual` |
| `rm_isf_umi_left` | ISF 版单左臂 + UMI 夹爪 | `RM_MODEL_RM_ISF_E` | `rm_leader_left` |
| `rm_isf_umi_right` | ISF 版单右臂 + UMI 夹爪 | `RM_MODEL_RM_ISF_E` | `rm_leader_right` |

## 目录

```text
deployment/   真机配置、硬件接口、采集和推理
scripts/      数据处理和训练工作流脚本
tools/        可独立调用的离线数据工具
vtla/         数据集、processor 和 policy 实现
playground/   本地数据、预训练权重、训练结果和评测录像
collect.sh    采集入口
train.sh      policy 训练入口
inference.sh  真机推理入口
```

所有仓库脚本都以仓库根目录为运行基准。运行资产默认位于：

```text
playground/data/<dataset_id>
playground/pretrained_models/<model>
playground/results/models/<pretrained_id>
playground/results/backbones/<run_id>
playground/eval/<eval_id>
```

## 环境

建议使用 Python 3.11 或更高版本。Python 3.10 仍可能运行，但部分依赖会发出版本警告。

```bash
conda create -n vtla python=3.11 -y
conda activate vtla

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

真机运行还需要把厂商 SDK 放到 `deployment/sdk/`；具体目录见 [Deployment](deployment/README.md#厂商-sdk)。视频处理依赖 `ffmpeg` 和 `ffprobe`。

## 硬件自检

硬件自检支持 `rm_base_umi_dual`、`rm_isf_umi_left` 和 `rm_isf_umi_right`。
先按当前 rig 设置机器人类型，后续三个阶段复用该变量：

```bash
robot_type=rm_isf_umi_left
```

第一步检查机械臂、末端板和主臂串口是否存在，不会驱动机械臂：

```bash
python -m deployment.tools.hardware_check \
  --robot-type "${robot_type}" \
  --stage existence
```

第二步连接腕部相机和触觉传感器并显示画面，不会驱动机械臂：

```bash
python -m deployment.tools.hardware_check \
  --robot-type "${robot_type}" \
  --stage camera \
  --show
```

rig 没有触觉传感器时在命令末尾添加 `--no-tactile`；需要保存检查帧时添加 `--save`。

第三步检查主从臂遥操作，会实际驱动从臂。确认主从臂初始姿态接近、工作区清空且
急停可用后再运行。hardware check 使用 `2 deg/s` 的逐关节目标角速度限制：

```bash
python -m deployment.tools.hardware_check \
  --robot-type "${robot_type}" \
  --stage teleop \
  --confirm-move \
  --duration 5 \
  --max-joint-speed-deg-s 2
```

`--max-joint-speed-deg-s` 未指定时默认为 `2 deg/s`。限速从从臂实测关节位置起步，
并对每个关节独立生效。不需要检查夹爪时添加 `--no-gripper`。更完整的参数说明见
[Deployment 硬件自检](deployment/README.md#硬件自检)。

## 完整工作流

### 1. 采集

先在 [collect.sh](collect.sh) 中设置唯一的 `robot_type`，再运行：

```bash
bash collect.sh <name> <task_text> <num_episodes> [teleop|drag]
```

采集示例：

```bash
bash collect.sh insert_easy "insert the object to the hole" 25 drag
```

`deployment.collect` 从 RobotConfig 注册表校验类型，并自动选择该机器人声明的遥操作器。采集结果的 `meta/info.json` 会记录完全相同的 `robot_type`。

### 2. 数据处理

#### 2.1 处理关节数据
```bash
bash scripts/process_joint_data.sh <dataset_id> [size] [horizon]
```

脚本保留原始数据，依次执行鱼眼去畸变、触觉 `uint16 -> uint8` 派生转换、视频缩放和 joint-to-EE。FK 类型只能来自数据集的 `robot_type`，没有命令行覆盖参数。

处理示例：

```bash
dataset_id=rm_isf_umi_left_20260820_insert_easy_precise
bash scripts/process_joint_data.sh "${dataset_id}" 224 32 6
```

最后一个参数是 relative-action 统计使用的 `action_gap`；训练 relative 模型时必须与 `train.sh` 的值一致。absolute action 不使用这组 relative 统计。

#### 2.2 处理UMI数据
```bash
bash scripts/process_umi_data.sh <dataset_id> [size] [horizon] [action_gap]
```

`size` 默认是 `224`，三路 RGB 和四路触觉都会直接拉伸到 `224x224`。源 RGB 使用
`_undist` key，已经去畸变，本流程不会再次去畸变。

处理示例：

```bash
dataset_id=260821_boarderaser_to_cup_trainready_rgb640x480_camlr_egor
TASK="Put the board eraser into the cup." \
  bash scripts/process_umi_data.sh "${dataset_id}" 224 32 6
```

输出为 `<dataset_id>_processed`。流程将三路 RGB 和四路触觉映射到标准
`cam_top`、`left/right_cam_wrist`、`left/right_cam_finger0/1` key，把触觉从
`uint16` 定点编码转换为训练使用的 `uint8`，并删除未使用的额外 RGB/video feature 及其陈旧统计。
夹爪会分别扫描 `observation.state` 和 `action`：每侧数据集最小值映射为张开 `1`，原始值
`0` 映射为闭合 `0`。数据集和后续 checkpoint 的 `robot_type` 都保持 `umi`。该流程不依赖
无效 joint 字段；推荐使用 episode EE state 和 relative EE action。

#### 2.3 处理 Backbone 数据（可选）

只有需要训练触觉 backbone 时才执行该步骤。输入应是完成上述处理、触觉图像已转换为 `uint8` 的数据集：

```bash
bash scripts/process_backbone_data.sh <dataset_id> \
  --image_size 224 --num_frames 4 --frame_stride 2
```

脚本筛选有效接触窗口，并把训练实际引用的触觉帧写入各数据集自己的 `tactile_backbone_cache/`。

### 3. 训练

#### 3.1 Backbone 训练（可选）

可以训练 Tactile Backbone：

```bash
bash scripts/train_backbone.sh \
  <dataset_id> <model_id> [num_processes] [batch_size] [epochs] \
  [lr] [image_size] [tactile_num_frames] [tactile_frame_offset] [resume]
```

训练示例：

```bash
bash scripts/train_backbone.sh <dataset_id> wan22_vae 4 4 5 1e-5 224 4 2
```

各模型目标和权重要求见 [Backbone 模型文档](#backbone模型文档)。

#### 3.2 VTLA 训练

```bash
bash train.sh \
  <dataset_id> <policy_type> <num_processes> <batch_size> <steps> \
  <wrist_only> <tactile_mode> <state_mode> <action_mode> \
  [action_gap] <augmentation_mode> [tactile_encoder_path]
```

关节动作示例：

```bash
dataset_id=rm_isf_umi_left_20260820_insert_easy_precise_processed
bash train.sh "${dataset_id}" starvla_groot 1 4 10000 \
  true none absolute_joint absolute_joint 0 none
```

相对 EE 动作示例：

```bash
dataset_id=rm_isf_umi_left_20260820_insert_easy_precise_processed
bash train.sh "${dataset_id}" starvla_groot 1 4 10000 \
  true none absolute_rot6d relative_rot6d 6 none
```

`action_gap` 默认为 `6`。当 `chunk_size=32` 时，`action_gap=6` 使用 `t+6 ... t+37` 作为 GT；relative action 的 pose anchor 仍是当前观测 `S(t)`。

训练会校验数据集的臂布局，并把 `robot_type` 写入每个 checkpoint 的 policy config。

#### 3.3 数据集 Mixture

在 `configs/data_mixtures.yaml` 中可以把已有数据集注册为一个不占额外数据存储的虚拟数据集：

```yaml
version: 1
mixtures:
  <data_all>:
    root: playground/data
    datasets:
      - dataset_id: <data1>
      - dataset_id: <data2>
      - dataset_id: <data3>
```

mixture 和普通数据集使用同一个 ID 入口，不需要特殊前缀：

```bash
bash train.sh <data_all>

bash train_backbone.sh <data_all>
```

每个成员的 `weight` 默认为 `1`，归一化后作为先选择数据集的概率；选中成员后再在它的有效 frame 中均匀采样。因此默认是数据集级等权，不受成员 frame 数量影响。成员必须具有一致的 `robot_type` 和 FPS。feature schema 对每个 feature 严格比较 key、`dtype`、`shape`、`names`、`tactile_encoding` 和 `storage_dtype`；相机 `intrinsics`、`imu_to_rgb_camera`、`extrinsics`，视频 codec/`pix_fmt`、`video_path` 和 `external_video` 允许不同。因此不同设备的相机标定和封装参数可以保留，不会阻止 mixture 训练。

### 4. 离线推理

```bash
bash scripts/evaluate_policy_offline.sh \
  <dataset_id> <pretrained_id> <step|last> [episodes] [stride] [device]
```

评估 episode 0 到 2 的示例：

```bash
dataset_id=rm_isf_umi_left_20260820_insert_easy_precise_undist_uint8_256
pretrained_id=20260821_rm_isf_umi_left_20260820_insert_easy_precise_undist_uint8_256_starvla_groot_wristonly_true_tactile_none_state_absolute_rot6d_action_relative_rot6d_aug_strong
bash scripts/evaluate_policy_offline.sh \
  "${dataset_id}" "${pretrained_id}" 3000 0-2 1 cuda
```

工具按完整 episode 调用 `predict_action_chunk()` 对比数据集 GT，同时输出 `action_mode` 和
`robot_command` 空间的曲线及误差指标。结果保存在 `<pretrained_id>/offline_eval/<step>/<dataset_id>/`。

### 5. 在线推理（真机）

```bash
bash inference.sh \
  <pretrained_id> <step> [inference_mode] [robot_type] [n_action_steps] [action_start_offset] \
  [control_fps] [reset_before_episode] [single_task]
```

推理示例：

```bash
pretrained_id=20260821_rm_isf_umi_left_20260820_insert_easy_precise_undist_uint8_256_starvla_groot_wristonly_true_tactile_none_state_absolute_rot6d_action_relative_rot6d_aug_strong
bash inference.sh "${pretrained_id}" 5000 async rm_isf_umi_left
```

`inference_mode` 可选 `sync` 或 `async`：

- `sync`：当前 action queue 执行完后读取最新观测，完成一次推理并执行新 chunk。
- `async`：观测与模型推理持续并行刷新最新 chunk；执行端保证当前 chunk 完整下发，结束后只领取最新推理结果。

`control_fps` 是机器人动作下发目标频率，默认 `30 Hz`。模型按 `30 Hz` 数据训练，降低该值会按比例放慢轨迹的真实执行速度。同步模式在 chunk 交界处受推理耗时影响；异步模式在已有最新 chunk 时不会等待推理。

`single_task` 默认为空，此时 `match_policy` 从 checkpoint 自动读取任务；多任务模型可以显式传入任务文本，控制本次推理的语言语义输入。

普通 checkpoint 的实际机器人类型始终由 checkpoint 覆盖，不能通过 `match_policy=false` 绕过。
当 checkpoint 的 `robot_type=umi` 时，必须把具体物理机器人类型作为 `inference.sh` 第 4 个参数
显式传入；此时以 CLI 为准。两种情况都会启用对应 B/ISF 构型的在线 FK/IK。
FPS、腕部去畸变及 RGB/触觉 resize 同样始终由 checkpoint 控制；旧 checkpoint 不兼容。


## VTLA模型文档

| 模型 | 文档 |
| --- | --- |
| ACT | [vtla/frameworks/act/README.md](vtla/frameworks/act/README.md) |
| Diffusion Policy | [vtla/frameworks/diffusion/README.md](vtla/frameworks/diffusion/README.md) |
| pi0.5 | [vtla/frameworks/pi05/README.md](vtla/frameworks/pi05/README.md) |
| StarVLA-GR00T | [vtla/frameworks/starvla_groot/README.md](vtla/frameworks/starvla_groot/README.md) |
| StarVLA-GR00T DINOAlign | [vtla/frameworks/starvla_groot_dinoalign/README.md](vtla/frameworks/starvla_groot_dinoalign/README.md) |
| FastWAM | [vtla/frameworks/fastwam/README.md](vtla/frameworks/fastwam/README.md) |
| Dream-Tac | [vtla/frameworks/dream_tac/README.md](vtla/frameworks/dream_tac/README.md) |

## Backbone模型文档

| 模型 | 文档 |
| --- | --- |
| AnyTouch1 | [vtla/tac_encoder/frameworks/anytouch1/README.md](vtla/tac_encoder/frameworks/anytouch1/README.md) |
| AnyTouch2 | [vtla/tac_encoder/frameworks/anytouch2/README.md](vtla/tac_encoder/frameworks/anytouch2/README.md) |
| Sparsh V-JEPA | [vtla/tac_encoder/frameworks/sparsh_vjepa/README.md](vtla/tac_encoder/frameworks/sparsh_vjepa/README.md) |
| Wan2.2 VAE | [vtla/tac_encoder/frameworks/wan22_vae/README.md](vtla/tac_encoder/frameworks/wan22_vae/README.md) |




## Git Usage
```bash
git pull --rebase origin main

git add .
git commit -m "..."
git push origin main
```
数据集、模型权重、训练结果和评测录像属于本地运行资产，不应加入提交；具体忽略范围见 [.gitignore](.gitignore)。

## TODO List
```bash
1 DAgger的采集
2 Controller实现
3 处理UMI数据
```
