# Playground

`playground/` 是 StarVTLA 的本地运行资产目录。数据集、预训练权重、训练结果和真机评测录像都使用相对于仓库根目录的路径，便于在不同机器间迁移代码和配置。

## 目录

```text
playground/
├── data/                 LeRobot 数据集和触觉 raw frame cache
├── pretrained_models/    基础模型、teacher 和初始化权重
├── results/
│   ├── backbones/        触觉 MAE checkpoint 与日志
│   └── models/           VTLA policy checkpoint 与日志
└── eval/                 真机推理录制
```

大文件由 `.gitignore` 排除。不要把数据、模型权重、视频或训练输出提交到 Git。

## 数据集

每个本地数据集使用独立目录：

```text
playground/data/<dataset_id>/
├── data/
├── meta/
│   └── info.json
└── videos/
```

当前采集数据：

```text
rm_isf_umi_left_20260820_insert_easy_precise
rm_isf_umi_left_20260820_insert_easy_robust
```

对应的处理后训练数据：

```text
rm_isf_umi_left_20260820_insert_easy_precise_undist_uint8_256
rm_isf_umi_left_20260820_insert_easy_robust_undist_uint8_256
```

### Robot Type

LeRobot 数据集的 `meta/info.json` 必须记录物理机器人类型：

```json
{
  "robot_type": "rm_isf_umi_left"
}
```

当前只允许 `rm_base_umi_dual` 和 `rm_isf_umi_left`。该值决定 joint-to-EE 使用 B 还是 ISF FK，并在训练时写入 checkpoint。不同 `robot_type` 的数据不能合并训练；缺失或旧名称不会被自动兼容。

完整 joint 数据处理：

```bash
bash scripts/process_joint_data.sh <dataset_id> 256 32
```

处理顺序、目录后缀和触觉量化标准见 [Workflow Scripts](../scripts/README.md#关节数据处理)，单项工具见 [Offline Tools](../tools/README.md)。

### Raw Frame Cache

不含 `meta/info.json` 的目录可作为触觉 MAE 的 raw frame cache。它面向单路连续触觉图像流，不能用于 VTLA policy 训练，也不能和 LeRobot 数据集在同一个 tactile-MAE run 中混用。

```text
playground/data/pretrained_data/
```

构建和训练方式见 [Tactile MAE](../vtla/tac_encoder/tactile_mae/README.md#raw-frame-cache)。

## 预训练权重

本地目录名必须与脚本默认路径一致，或通过相应环境变量显式覆盖：

| 目录 | 用途 |
| --- | --- |
| `CLIP-ViT-L-14-DataComp.XL-s13B-b90K` | 触觉 MAE ViT-L 初始化 |
| `CLIP-ViT-B-16-DataComp.XL-s13B-b90K` | 触觉 MAE ViT-B 初始化，要求 HF 格式 |
| `AnyTouch-ViT-L-16` | AnyTouch 完整初始化和 VTLA tactile encoder |
| `pi05_base` | pi05 policy 初始化 checkpoint |
| `Qwen3.5-0.8B` | StarVLA-GR00T 基础 VLM |
| `vit_base_patch16_dinov3.lvd1689m` | DINOAlign 训练期 teacher |
| `Wan2.2-TI2V-5B` | FastWAM 的 Video DiT、VAE、T5 和 tokenizer |

下载到指定目录的通用形式：

```bash
huggingface-cli download <repo_id> \
  --local-dir playground/pretrained_models/<directory_name>
```

`pi05_base` 是完整 policy checkpoint，`Qwen3.5-0.8B` 是基础 VLM，两者不能互换。具体初始化方式见 [VTLA Training](../vtla/README.md#policy-类型)。

FastWAM 的 Wan2.2 目录还需要记录官方来源的 `fastwam_source.json`，并生成 `interpolated_dit/`。准备命令见 [FastWAM 资产](../tools/README.md#fastwam-资产)。

## 训练输出

触觉 backbone：

```text
playground/results/backbones/<timestamp>_tacmae_<arch>_from_<init_mode>/
```

VTLA policy：

```text
playground/results/models/<run_id>/
├── checkpoints/<step>/pretrained_model/
└── <run_id>.log
```

训练日志先写入系统临时目录。正常完成后才移动到 run 目录；失败或中断时临时日志会被清理。

checkpoint 的 policy `config.json` 应包含训练数据集的 `robot_type`，单任务训练还会包含 `single_task`。推理不依赖训练数据集目录来选择机器人运动学。

## 推理输出

```text
playground/eval/eval_<timestamp>_<run_id>_step_<step>/
```

评测目录是新的 LeRobot 录制数据集。实际 RobotConfig、B/ISF FK/IK 和 action space 由 checkpoint 自动匹配，说明见 [Deployment](../deployment/README.md#policy-推理)。
