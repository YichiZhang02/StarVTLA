# StarVTLA

末端位姿统一为 TCP rot6d；`relative_rot6d` 采用 TCP 局部零中心编码。旧 EE 数据/模型需迁移重训，见 [TCP 数据与动作约定](tools/TCP_ACTIONS.md)。

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
playground/data/<source>/<division>/<dataset_id>
playground/pretrained_models/<model>
playground/results/models/<pretrained_id>
playground/results/backbones/<run_id>
playground/eval/<eval_id>
```

训练数据按来源、划分和数据集 ID 放在三级目录。可选择单个数据集、一个划分或一个来源：

```bash
bash train.sh Daimon/realman_single/<dataset_id> ...
bash train.sh Daimon/realman_single/all ...
bash train.sh Daimon/all/all ...
bash scripts/train_backbone.sh Daimon/realman_single/<dataset_id> ...
```

训练用的叶子目录必须含 `meta/info.json`。`all` 会按目录名排序展开成员，
默认 `weight=1`；需要筛选成员、指定权重或 episodes 时，使用 `playground/data/data_mixtures.yaml`。
参与同一次训练的成员仍须满足现有的 feature schema、FPS、robot_type 等一致性检查。
解析出的成员列表写入训练配置，恢复训练时不会重新扫描目录。

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
bash collect.sh <source/group> <name> [task_text] [num_episodes] [teleop|drag]
```

采集示例：

```bash
bash collect.sh Daimon/realman_single insert_easy \
  "insert the object to the hole" 25 drag
```

`deployment.collect` 从 RobotConfig 注册表校验类型，并自动选择该机器人声明的遥操作器。采集结果的 `meta/info.json` 会记录完全相同的 `robot_type`。
第一个参数指定 `source/group`，第二个参数 `name` 会生成 `<robot_type>_<YYYYMMDD>_<name>` 作为 `dataset_id`，最终目录为 `playground/data/<source>/<group>/<dataset_id>/`。缺失的父目录会自动创建。采集目标必须是具体数据集，不能是注册组合名或 `all`。

### 2. 数据处理

#### 2.1 处理关节数据
```bash
bash scripts/process_joint_data.sh <dataset_id> [size] [horizon] [action_gap]
```

脚本保留原始数据，依次执行鱼眼去畸变、触觉 `uint16 -> uint8` 派生转换、视频缩放，以及 joint → FK flange → 工具标定 → TCP rot6d。FK 类型只能来自数据集的 `robot_type`，没有命令行覆盖参数。

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
无效 joint 字段；UMI pose 已经是 TCP，不再施加 flange-to-TCP 偏移。推荐使用 `episode_rot6d` state 和 `relative_rot6d` action。

#### 已处理数据的迁移

旧 EE 数据需从原始 joint/UMI 列重新生成，旧 EE checkpoint 需重训。迁移保留源目录，目标目录必须尚不存在：

```bash
python tools/migrate_tcp_dataset.py \
  --src playground/data/Daimon/realman_single/old_dataset \
  --dst playground/data/Daimon/realman_single/tcp_dataset \
  --horizon 32 --action-gap 6
```

可先追加 `--dry-run` 检查迁移条件。已迁移数据只需调整动作统计窗口时，使用
[统计量重建工具](tools/README.md#tcp-数据迁移与统计量重建)，无需重复处理视频。

#### 2.3 处理 Backbone 数据（可选）

只有需要训练触觉 backbone 时才执行该步骤。输入应是完成上述处理、触觉图像已转换为 `uint8` 的数据集：

```bash
bash scripts/process_backbone_data.sh <registered_name|source/group/dataset_id> \
  --image_size 224 --num_frames 4 --frame_stride 2
```

脚本筛选有效接触窗口，并把训练实际引用的触觉帧写入各数据集自己的 `tactile_backbone_cache/`。

### 3. 训练

#### 3.1 Backbone 训练（可选）

可以训练 Tactile Backbone：

```bash
bash scripts/train_backbone.sh \
  <registered_name|source/group/dataset_id> <model_id> [num_processes] [batch_size] [epochs] \
  [lr] [image_size] [tactile_num_frames] [tactile_frame_offset] [resume]
```

第一个位置参数是注册组合名或完整三级数据路径。

训练示例：

```bash
bash scripts/train_backbone.sh Daimon/realman_single/<dataset_id> wan22_vae 4 4 5 1e-5 224 4 2
```

各模型目标和权重要求见 [Backbone 模型文档](#backbone模型文档)。

#### 3.2 VTLA 训练

```bash
bash train.sh \
  <registered_name|source/group/dataset_id> <policy_type> <num_processes> <batch_size> <steps> \
  <wrist_only> <tactile_mode> <state_mode> <action_mode> \
  [action_gap] [augmentation_mode] [tactile_encoder_path]
```

`dataset_mixture` 是 `$1`，`policy_type` 是 `$2`，后续参数依上面的顺序排列。

TacMind0 使用 `policy_type=tacmind0`，训练入口会强制 `tactile_mode=as_image`，
并按原模型的两路触觉、8 帧、间隔 5 帧输入专用 Tac-LeWM encoder；训练时该 encoder
参与优化。数据集需有两路触觉 key，完整权重与用法见
[TacMind0 说明](vtla/frameworks/tacmind0/README.md)。

关节动作示例：

```bash
dataset_id=rm_isf_umi_left_20260918_assemble_gearS_processed
bash train.sh "Daimon/realman_single/${dataset_id}" starvla_groot 1 4 10000 \
  true none absolute_joint absolute_joint 0 none
```

相对 EE 动作示例：

```bash
dataset_id=rm_isf_umi_left_20260918_assemble_gearS_processed
bash train.sh "Daimon/realman_single/${dataset_id}" starvla_groot 1 4 10000 \
  true none absolute_rot6d relative_rot6d 6 none
```

`action_gap` 默认为 `6`。当 `chunk_size=32` 时，`action_gap=6` 使用 `t+6 ... t+37` 作为 GT；relative action 的 pose anchor 仍是当前观测 `S(t)`。

训练会校验数据集的臂布局，并把 `robot_type` 写入每个 checkpoint 的 policy config。

末端 action 只支持 `absolute_rot6d` 和 `relative_rot6d`，关节模式保持不变。
绝对位姿为基座系 TCP；相对动作在当前 TCP 系下计算，整个 chunk 共用当前观测锚点：

```text
dp = Rs.T @ (pa - ps)
dr = rot6d(Rs.T @ Ra) - [1, 0, 0, 0, 1, 0]
```

夹爪始终为绝对指令。反归一化后的 `dr=0` 表示不旋转；归一化模型输出的零值没有该保证。
`state_mode=none` 仍需 processor 保存当前 TCP 锚点。`ee_frame` 已移除，FK、工具标定和
驱动下发转换统一由 `robot_type` 选择。完整定义见 [TCP 数据与动作约定](tools/TCP_ACTIONS.md)。

#### 3.3 数据集 Mixture

也可以在 `playground/data/data_mixtures.yaml` 中跨来源组合整个划分或单个数据集：

```yaml
version: 1
mixtures:
  Mix:
    datasets:
      - {source: Daimon, group: realman_single, dataset_id: all}
      - {source: N0, group: umi, dataset_id: demo, weight: 2}
```

训练 `data_mixtures.yaml` 中定义的组合时，直接传组合名：

```bash
bash train.sh Mix starvla_groot 1 4 10000

bash scripts/process_backbone_data.sh Mix
bash scripts/train_backbone.sh Mix anytouch1
```

`Mix` 对应 YAML 中的 `mixtures.Mix`；按其中的成员、episodes 和权重训练。
普通数据集使用 `source/group/id`，最后一段可为 `all`。

VLA 训练中，每个成员的 `weight` 默认为 `1`，表示**每帧采样倍率**。先按以下概率选择数据集，再在该成员的有效帧中均匀随机抽样：

```text
p_i = weight_i × N_i / Σ(weight_j × N_j)
```

`N_i` 是成员经过 `episodes` 筛选和训练帧裁剪后的有效帧数；当前 VLA 训练入口按策略配置应用尾帧裁剪。裁剪后没有有效帧的 episode 不参与采样；成员完全没有有效帧时会报错。采样有放回，允许重复抽帧，不保证一轮遍历每一帧。

不设置权重时，所有有效帧获得相同采样机会，大数据集获得更多总采样次数。例如：

| 成员 | 有效帧数 | 默认 `weight` | 数据集采样概率 |
|---|---:|---:|---:|
| A | 10,000 | 1 | 10% |
| B | 90,000 | 1 | 90% |

需要额外强调某个来源时，可以显式设置倍率：

```yaml
datasets:
  - dataset_id: <data_a>
    weight: 2.0
  - dataset_id: <data_b>
```

此时 A 中每个有效帧被抽中的概率是 B 中每帧的两倍。若帧数仍为上表数值，来源概率为 `20,000 : 90,000`，即约 `18.2% : 81.8%`。已有非默认权重继续生效；若希望完全按有效帧数混合，应省略所有成员的权重或全部设为 `1`。

VLA 的 mean/std 聚合使用同一来源概率；成员内部仍沿用已有全量统计，不会因筛选 episodes 或裁剪帧而重新计算子集统计。启动日志打印各成员的有效帧数、配置倍率和最终概率，训练日志中的 `Mixture sample fractions` 则反映实际采样比例。

解析结果 `resolved_data_mixture.json` 和保存的训练配置会记录 `sampling_strategy=weighted_frames`、有效帧数及最终概率。加载旧 checkpoint 中没有策略标记的 resolved mixture 时，保持原来的 `p_i ∝ weight_i` 数据集级权重语义。

上述变更仅适用于 VLA。Backbone 仍按原来的成员权重选择来源，不自动乘有效帧数。

成员必须具有一致的 `robot_type`、FPS 和 `tcp_contract`；TCP relative 统计也必须存在且使用相同动作窗口。feature schema 对每个 feature 严格比较 key、`dtype`、`shape`、`names`、`tactile_encoding` 和 `storage_dtype`；相机 `intrinsics`、`imu_to_rgb_camera`、`extrinsics`，视频 codec/`pix_fmt`、`video_path` 和 `external_video` 允许不同。因此不同设备的相机标定和封装参数可以保留，不会阻止 mixture 训练。

### 4. 离线推理

```bash
bash scripts/evaluate_policy_offline.sh \
  <dataset_id> <pretrained_id> <step|last> [episodes] [stride] [device]
```

评估 episode 0 到 2 的示例（`tcp_dataset` 和 `your_retrained_tcp_run` 分别替换为迁移后的数据集和重训运行目录）：

```bash
dataset_id=tcp_dataset
pretrained_id=your_retrained_tcp_run
bash scripts/evaluate_policy_offline.sh \
  "${dataset_id}" "${pretrained_id}" 3000 0-2 1 cuda
```

工具按完整 episode 调用 `predict_action_chunk()` 对比数据集 GT，同时输出 `action_mode` 和
`robot_command` 空间的曲线及误差指标。EE 的 `robot_command` 是还原后的基座系绝对 TCP，尚未经过驱动端限幅及 TCP-to-flange 转换。结果保存在 `<pretrained_id>/offline_eval/<step>/<dataset_id>/`。

### 5. 在线推理（真机）

```bash
bash inference.sh \
  <pretrained_id> <step> [inference_mode] [robot_type] [n_action_steps] [action_start_offset] \
  [control_fps] [reset_before_episode] [single_task]
```

推理示例：

```bash
pretrained_id=your_retrained_tcp_run
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
| N0-VTLA | [vtla/frameworks/n0_vtla/README.md](vtla/frameworks/n0_vtla/README.md) |

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
