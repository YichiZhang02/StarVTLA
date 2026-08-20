# StarVTLA

VTLA 训练与部署基础设施，覆盖触觉 backbone 预训练、机器人数据处理、ACT/Diffusion/pi05/StarVLA-GR00T/DINOAlign/FastWAM 策略训练，以及真机采集和推理。

## Repository Layout

```text
.
├── vtla/          # 数据、配置、处理管线和 policy 实现
├── deployment/    # 硬件、采集和推理
├── scripts/       # 训练与数据处理工作流
├── tools/         # 可独立运行的离线工具
├── playground/    # 数据、权重、训练结果和评测录制
├── train.sh       # VTLA policy 训练入口
├── collect.sh     # 真机数据采集入口
└── inference.sh   # 真机 policy 推理入口
```

所有运行时路径都相对于仓库根目录，并统一放在 `playground/`。`scripts/*.sh` 会自动切换到仓库根后再执行。

## Quick Start

创建环境并安装依赖：

```bash
conda create -n vtla python=3.10 -y
conda activate vtla

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```




将数据和预训练权重放入 `playground/`。完整目录和文件约定见 [playground/README.md](playground/README.md)。

## Tactile Data

Flux 触觉采集首先将 SDK 的 `float32` depth/deformation 固定编码为 HWC `uint16`：depth
乘 `1000`，两路 deformation 乘 `1000` 后加 `30000`。权威数据使用无损
`FFV1/gbrp16le` MKV，与 wrist camera 一样保存在数据集 `videos/` 下。

训练前运行 `scripts/process_joint_data.sh`：它将 depth `0..1000` 和 deformation
`30000 ± 1000` 线性量化到 uint8，先生成无损 RGB 中间 MP4，再与视觉视频一起 resize，
最终保存为便于训练和播放的 H.264/YUV420 MP4。最终 uint8 是近似派生数据，不替代原始
uint16 MKV。采集格式见 [Deployment](deployment/README.md#触觉采集编码)，处理细节见
[Workflow Scripts](scripts/README.md#触觉处理流程)。


## Training
预处理数据：
```bash
# 关节数据：去畸变、触觉 uint8、缩放所有视频、生成 EE 列
bash scripts/process_joint_data.sh <dataset_id>

# UMI 数据：去畸变、缩放、生成 EE 列
bash scripts/process_umi_data.sh <dataset_id>
```

训练触觉 Backbone：

```bash
bash scripts/train_enc.sh <dataset_id> <init_mode> <model size>
```

训练 VTLA：

```bash
bash train.sh \
  <dataset_id> <poilcy_type> <num_gpu> <batch_size> <training step>
```

`train.sh` 的第 8、9 个参数分别控制 state/action 数据契约：

```text
state_mode:  none | absolute_joint | episode_joint | absolute_rot6d |
             episode_rot6d | absolute_quat | episode_quat
action_mode: absolute_joint | relative_joint | absolute_rot6d |
             relative_rot6d | absolute_quat | relative_quat
```

例如，无 proprioception 输入、预测相对当前观测的 rot6d EE 动作：

```bash
bash train.sh <dataset_id> pi05 1 32 10000 false none none relative_rot6d
```

## Collection and Inference
硬件检查：

```bash
python -m deployment.tools.hardware_check
```

采集真机数据：
```bash
bash collect.sh <name> <single_task> <num_episodes>
```

真机推理：
```bash
bash inference.sh <pretrained_id> <step>
```

## Detailed Documentation

| 任务 | 入口 | 文档 |
| --- | --- | --- |
| 准备数据和权重 | `playground/` | [Runtime layout](playground/README.md) |
| 训练 VTLA policy | `train.sh` | [VTLA training](vtla/README.md) |
| 训练触觉 backbone | `scripts/train_enc.sh` | [Workflow scripts](scripts/README.md#触觉-backbone-训练) |
| 处理关节或 UMI 数据 | `scripts/process_*_data.sh` | [Workflow scripts](scripts/README.md) |
| 运行单项离线工具 | `tools/*.py` | [Offline tools](tools/README.md) |
| 硬件检查与数据采集 | `deployment/`、`collect.sh` | [Deployment](deployment/README.md) |
| Policy 推理 | `inference.sh` | [Policy inference](deployment/README.md#policy-推理) |


## Git Workflow

```bash
git pull --rebase origin main

git add .
git commit -m "..."
git push origin main
```

运行数据、模型权重和训练结果不进入 Git，具体忽略范围见 [.gitignore](.gitignore)。

## TODO List
```bash
采集代码优化
推理速度可选择
多机和cotraining
```
