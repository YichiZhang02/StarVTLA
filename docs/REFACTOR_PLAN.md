# Tactile Backbone 训练重构

## 目标与状态

三个 backbone 通过同一个命令行入口启动，但各自拥有独立的训练 recipe。统一入口只负责数据、DDP、AMP、日志和 checkpoint 生命周期，模型构建、训练目标、优化器、EMA、可视化及 encoder 导出均由 recipe 管理。

| `model_id` | 训练目标 | 预训练要求 | 可视化 |
| --- | --- | --- | --- |
| `anytouch1` | masked pixel reconstruction | 官方 encoder + decoder | original / masked / reconstruction / error |
| `anytouch2` | masked pixel + temporal residual reconstruction | 官方 encoder + pixel decoder + residual decoder | 像素重建图和 residual prediction / target / error |
| `sparsh_vjepa` | masked latent JEPA prediction + variance regularization | context encoder + target encoder + predictor | predicted latent / target latent / error map |

实现位置：

```text
vtla/tac_encoder/train.py                  # 统一运行时
vtla/tac_encoder/training/base.py          # recipe 协议
vtla/tac_encoder/training/registry.py      # model_id 分发
vtla/tac_encoder/training/anytouch1.py
vtla/tac_encoder/training/anytouch2.py
vtla/tac_encoder/training/sparsh_vjepa.py
```

## 不变量

1. 训练时不允许存在未被 checkpoint 覆盖的可训练参数。缺失或 shape mismatch 会在第一个训练 step 前报错。
2. AnyTouch1 的四帧都参与 masked pixel loss。
3. AnyTouch2 两个 decoder 共享同一 spatial mask，总 loss 为 masked pixel MSE 与 masked adjacent-frame residual MSE 之和。
4. V-JEPA 使用可训练 context encoder、无梯度 target encoder、可训练 predictor；target encoder 在每次 optimizer update 后使用 `0.998 -> 1.0` EMA 更新。
5. 下游 checkpoint 只读取 `encoder`，不读取 decoder 或 predictor。下游 Sparsh 模型本身也是纯 encoder，不创建随机像素 decoder。
6. pooling 只属于下游特征路径，绝不位于 encoder 与训练 decoder/predictor 之间。

## 官方权重

| `model_id` | 默认路径 | 当前状态 |
| --- | --- | --- |
| `anytouch1` | `playground/pretrained_models/AnyTouch-ViT-L-16/checkpoint.pth` | 完整 encoder + decoder |
| `anytouch2` | `playground/pretrained_models/AnyTouch2-Model/checkpoint-4frames.pth` | 完整 encoder + pixel/residual decoder |
| `sparsh_vjepa` | `playground/pretrained_models/Sparsh-VJEPA-Small/vjepa_vitsmall_full.ckpt` | 尚未提供 |

现有 `vjepa_vitsmall.safetensors` 只有 174 个 encoder tensor，不含 `context_encoder`、`target_encoder` 和 `predictor`。它可以用于下游 encoder 初始化，但不能满足“所有训练参数都来自 pretrained checkpoint”的训练要求。训练入口会明确拒绝该文件，不能用随机 predictor 绕过。

完整 V-JEPA checkpoint 支持常见 Lightning namespace：`state_dict` 外层以及 `model.` / `module.` 参数前缀。最终仍要求所有可训练 tensor 名称和 shape 完整匹配。

## 数据契约

模型输入统一为：

```text
[B, S, T, C, H, W]
```

默认值为 `T=4`、`C=3`、`H=W=224`。cache 中保存 `[N, S, H, W, C]` uint8 帧和窗口索引，dataset 在取样时转换为 `[S, T, C, H, W]` float `[0,1]`。

`frame_stride` 只控制原始视频中的物理采样间隔。cache 的 `num_frames`、`frame_stride` 和 `image_size` 必须与训练参数完全一致。

## 训练路径

```text
dataset/cache
    -> unified runtime
        -> recipe.build_model()
        -> recipe.step()
        -> optimizer update
        -> recipe.after_optimizer_step()
        -> per-update warmup/cosine scheduler
        -> recipe.save_visualization()
        -> encoder/trainer split checkpoint
```

AnyTouch1 使用参考实现的 AdamW `(0.9, 0.99)`、参数维度分组 weight decay、逐 update warmup + cosine；默认 weight decay 为 `0.1`。

AnyTouch2 使用同样的逐 update调度。像素分支重建归一化帧，residual 分支预测相邻归一化帧之差；两项 loss 都只统计 masked spatial patches。

V-JEPA 对两组 block mask 分别进行 latent L1 prediction，附加 patch variance regularization。默认 weight decay 为 `0.04` 并余弦增长到 `0.4`，默认 warmup 为 40 epochs、最终 LR 为 `1e-6`，与参考配置一致。短程微调可显式传 `--warmup_epochs` 覆盖。

## Checkpoint V2

训练 checkpoint 的关键字段为：

```text
format_version: 2
model_id
objective
encoder                 # 唯一下游加载的模型权重
trainer                 # decoder 或 context encoder + predictor
optimizer
scheduler
scaler
epoch
best_loss
args
resolved_data_mixture
resolved_signature
load_report
```

AnyTouch 的 `encoder` 不包含任何 `decoder` / `mask_token` tensor。V-JEPA 的 `encoder` 来自 EMA target encoder；context encoder和 predictor 只保存在 `trainer` 中。resume 要求 V2 格式、相同 `model_id`、相同 objective 和相同数据签名。

## 下游特征

默认 `3x3` spatial pooling 后每个传感器的 token 数：

| backbone | 每传感器 token |
| --- | --- |
| AnyTouch1 | `4 x (1 global + 9 spatial) = 40` |
| AnyTouch2 | `1 global + 2 x 9 spatial = 19` |
| Sparsh V-JEPA | `1 global + 2 x 9 spatial = 19` |

`TactileBackboneFeatureExtractor.from_pretrained()` 对 V2 checkpoint 只加载 `checkpoint["encoder"]`，并检查所有预期 encoder tensor 的名称和 shape。

## 启动方式

```bash
# 4 GPU；batch_size 是每 GPU batch
bash scripts/train_backbone.sh backbone_training_data anytouch1 4 32 5 1e-5 224 4 2
bash scripts/train_backbone.sh backbone_training_data anytouch2 4 32 5 1e-5 224 4 2
bash scripts/train_backbone.sh backbone_training_data sparsh_vjepa 4 32 5 1e-5 224 4 2
```

batch size 验收只测试每 GPU `32, 64, 128, 256`，从小到大运行，选择不会 OOM 且吞吐合理的最大值。三种模型最终都需进行 4 GPU、5 epochs 的真实训练。

V-JEPA 的 batch benchmark 和正式训练必须等完整官方训练 checkpoint 放到固定路径后进行；encoder-only 权重不会被当作可训练 checkpoint。

## 2026-08-26 四卡实测

`backbone_training_data` 为真实双传感器数据，batch size 均为每 GPU 数值：

| backbone | batch | 结果 | 每卡 peak allocated | 全局吞吐 |
| --- | ---: | --- | ---: | ---: |
| AnyTouch1 | 32 | stable | 50.81 GiB | 108.3 samples/s |
| AnyTouch1 | 64 | OOM | 75.63 GiB 后仍需 2.10 GiB | - |
| AnyTouch2 | 32 | stable | 18.27 GiB | 180.9 samples/s |
| AnyTouch2 | 64 | stable | 34.28 GiB | 230.1 samples/s |
| AnyTouch2 | 128 | stable | 66.34 GiB | 261.9 samples/s |
| AnyTouch2 | 256 | OOM | 75.81 GiB 后仍需 398 MiB | - |

最终选择：AnyTouch1 每卡 `32`，AnyTouch2 每卡 `128`。

两者均已完成 4 GPU、5 epochs 正式训练：

```text
playground/results/backbones/20260826_backbone_training_data_anytouch1_refactor_v2_4frames_2stride
playground/results/backbones/20260826_backbone_training_data_anytouch2_refactor_v2_4frames_2stride
```

AnyTouch1 最终 train/validation loss 为 `0.0009456 / 0.0009111`。AnyTouch2 最终 train/validation loss 为 `0.0116388 / 0.0081193`，validation pixel/residual MSE 分别为 `0.0054874 / 0.0026319`。

Sparsh V-JEPA 尚未进行 batch benchmark 或正式训练，唯一阻塞是完整官方训练 checkpoint 缺失。

## 验收项

- recipe registry 对三个 `model_id` 分发正确。
- AnyTouch1/2 真实 checkpoint 的所有可训练 tensor 均被加载。
- AnyTouch2 pixel decoder 和 residual decoder 都收到梯度。
- V-JEPA target encoder 无梯度，context encoder/predictor 有梯度，EMA 后 target 发生变化。
- encoder-only V-JEPA checkpoint 被明确拒绝，完整 checkpoint 可 strict 覆盖训练参数。
- V2 checkpoint 的下游加载不读取 trainer state。
- pixel 模型与 latent 模型分别生成约定的可视化。
- CPU tests、单 GPU smoke、4 GPU DDP smoke 通过。
- 每个 backbone 完成常见 batch size 实测与 5 epochs 正式训练。
