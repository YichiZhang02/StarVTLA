# StarVTLA

StarVTLA 是面向视觉、触觉和机器人动作学习的训练与真机部署仓库。它包含 LeRobot 数据采集与处理、ACT/Diffusion/pi05/StarVLA-GR00T/FastWAM 策略训练、触觉 MAE 预训练，以及 RealMan 机械臂的在线推理。

## 支持的机器人

机器人身份是数据和模型契约的一部分。目前只支持以下两个严格名称：

| `robot_type` | 构型 | RealMan FK/IK | 遥操作器 |
| --- | --- | --- | --- |
| `rm_base_umi_dual` | base/B 版双臂 + UMI 夹爪 | `RM_MODEL_RM_B_E` | `bi_realman_ugripper_leader` |
| `rm_isf_umi_left` | ISF 版单左臂 + UMI 夹爪 | `RM_MODEL_RM_ISF_E` | `left_realman_ugripper_leader` |

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

### 2. 处理关节数据

```bash
bash scripts/process_joint_data.sh <dataset_id> [size] [horizon]
```

脚本保留原始数据，依次执行鱼眼去畸变、触觉 `uint16 -> uint8` 派生转换、视频缩放和 joint-to-EE。FK 类型只能来自数据集的 `robot_type`，没有命令行覆盖参数。

处理示例：

```bash
dataset_id=rm_isf_umi_left_20260820_insert_easy_precise
bash scripts/process_joint_data.sh "${dataset_id}" 256 32
```

### 3. 训练

```bash
bash train.sh \
  <dataset_id> <policy_type> <num_processes> <batch_size> <steps> \
  <wrist_only> <tactile_mode> <state_mode> <action_mode> \
  <augmentation_mode>
```

关节动作示例：

```bash
dataset_id=rm_isf_umi_left_20260820_insert_easy_precise_undist_uint8_256
bash train.sh "${dataset_id}" starvla_groot 1 4 10000 \
  true none absolute_joint absolute_joint none
```

相对 EE 动作示例：

```bash
dataset_id=rm_isf_umi_left_20260820_insert_easy_precise_undist_uint8_256
bash train.sh "${dataset_id}" starvla_groot 1 4 10000 \
  true none absolute_rot6d relative_rot6d none
```

训练会校验数据集的臂布局，并把 `robot_type` 写入每个 checkpoint 的 policy config。

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
bash inference.sh <pretrained_id> <step> [n_action_steps] [action_start_offset]
```

推理示例：

```bash
pretrained_id=20260821_rm_isf_umi_left_20260820_insert_easy_precise_undist_uint8_256_starvla_groot_wristonly_true_tactile_none_state_absolute_rot6d_action_relative_rot6d_aug_strong
bash inference.sh "${pretrained_id}" 3000 16 6
```

`inference.sh` 中的初始 `--robot.type` 只是 Draccus 解析所需的启动配置。实际机器人类型始终由 checkpoint 覆盖，不能通过 `match_policy=false` 绕过。EE checkpoint 会自动启用与其 B/ISF 构型匹配的在线 FK/IK。

## 文档

| 内容 | 文档 |
| --- | --- |
| 本地数据、权重和输出布局 | [playground/README.md](playground/README.md) |
| 机器人、SDK、采集、推理和安全 | [deployment/README.md](deployment/README.md) |
| 数据处理和训练脚本 | [scripts/README.md](scripts/README.md) |
| 独立离线工具 | [tools/README.md](tools/README.md) |
| Policy、state/action 和触觉路由 | [vtla/README.md](vtla/README.md) |
| StarVLA-GR00T | [vtla/frameworks/starvla_groot/README.md](vtla/frameworks/starvla_groot/README.md) |
| StarVLA-GR00T DINOAlign | [vtla/frameworks/starvla_groot_dinoalign/README.md](vtla/frameworks/starvla_groot_dinoalign/README.md) |
| 触觉 MAE | [vtla/tac_encoder/tactile_mae/README.md](vtla/tac_encoder/tactile_mae/README.md) |

## Git Usage
```bash
git pull --rebase origin main

git add .
git commit -m "..."
git push origin main
```
数据集、模型权重、训练结果和评测录像属于本地运行资产，不应加入提交；具体忽略范围见 [.gitignore](.gitignore)。
