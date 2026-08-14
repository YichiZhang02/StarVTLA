# Playground

`playground/` 是本仓库统一的运行时目录。数据集、预训练权重、训练结果和评测录制均放在这里，代码和配置因此可以继续使用跨机器可移植的相对路径。

所有命令默认从仓库根目录运行。

## 目录结构

```text
playground/
├── data/                 # 训练、预训练和离线处理后的数据集
├── pretrained_models/    # 基础模型与初始化权重
├── results/
│   ├── backbones/        # 触觉 MAE checkpoint 与日志
│   └── models/           # VTLA policy checkpoint 与日志
└── eval/                 # 推理阶段录制的评测数据集
```

这些目录下的大文件由 `.gitignore` 排除；仓库只保留目录骨架，不提交数据、模型或训练产物。

## 数据集

每个数据集使用独立目录：

```text
playground/data/<dataset_id>/
```

支持两类输入：

- LeRobot 数据集：包含 `meta/info.json`，可用于 VTLA 策略训练和触觉 MAE 训练。
- 裸 frame cache：不包含 LeRobot metadata，面向触觉 MAE 预训练的单路连续图像流。

示例：

```text
playground/data/
├── rm_umi_dual_pen_open/
│   └── meta/info.json
└── pretrained_data/
```

数据去畸变、降分辨率、EE 列生成和合并方式见 [scripts/README.md](../scripts/README.md) 与 [tools/README.md](../tools/README.md)。

## 预训练权重

常用目录约定如下：

| 目录 | 用途 | 来源 |
| --- | --- | --- |
| `CLIP-ViT-L-14-DataComp.XL-s13B-b90K` | 触觉 MAE 的 ViT-L 初始化 | [laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K](https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K) |
| `CLIP-ViT-B-16-DataComp.XL-s13B-b90K` | 触觉 MAE 的 ViT-B 初始化，要求 HF 格式 | [laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K](https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K) |
| `AnyTouch-ViT-L-16` | AnyTouch 初始化和 VTLA 触觉 encoder | [AnyTouch](https://github.com/GeWu-Lab/AnyTouch) |
| `pi05_base` | pi05 完整 policy checkpoint | [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) |
| `Qwen3.5-0.8B` | `starvla_groot` / `starvla_groot_dinoalign` 底座 VLM | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| `vit_base_patch16_dinov3.lvd1689m` | `starvla_groot_dinoalign` 训练期冻结 teacher | [timm/vit_base_patch16_dinov3.lvd1689m](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m) |
| `Wan2.2-TI2V-5B` | FastWAM 的 Video DiT、VAE、T5、tokenizer 与插值 DiT | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |

推荐下载形式：

```bash
huggingface-cli download <repo_id> \
  --local-dir playground/pretrained_models/<directory_name>
```

`pi05_base` 是完整 policy checkpoint；`Qwen3.5-0.8B` 是基础 VLM。两者的加载语义不同，具体训练配置见 [vtla/README.md](../vtla/README.md)。

FastWAM 使用的本地 Wan2.2 目录还需要 `fastwam_source.json`，用于记录官方 repo 和 commit。插值权重生成方法见 [tools/README.md](../tools/README.md#生成插值-dit)。

## 输出

触觉 backbone 训练输出：

```text
playground/results/backbones/<run>/
```

VTLA policy 训练输出：

```text
playground/results/models/<run>/
```

训练日志先写入系统临时目录。训练正常结束后日志才会移动到对应结果目录；失败或中断时临时日志会被删除。

推理录制默认输出：

```text
playground/eval/<repo_id>/
```
