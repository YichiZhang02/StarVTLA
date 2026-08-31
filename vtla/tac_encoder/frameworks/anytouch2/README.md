# AnyTouch2

`anytouch2` 使用视频 patch encoder，并联合训练两个重建分支：原始触觉像素和相邻帧的
temporal residual。一个空间 mask 会复用于同一窗口的所有 temporal tube。

## 模型契约

| 项目 | 值 |
| --- | --- |
| 输入 | `[B, S, T, 3, H, W]`，范围 `[0,1]` |
| 默认 patch/tubelet | `16 / 2` |
| 默认特征维度 | `512` |
| 训练目标 | masked pixel MSE + masked temporal-residual MSE |
| 预训练权重 | `playground/pretrained_models/AnyTouch2-Model/checkpoint-4frames.pth` |

训练 checkpoint 的 `encoder` 保存 CLIP video encoder、projection 和 sensor token；两个
reconstruction decoder 只保存在 `trainer`，不会进入下游 policy。

```bash
bash scripts/train_backbone.sh <dataset_id> anytouch2 4 32 5 1e-5 224 4 2
```

实现入口为 [model.py](model.py)，video patch 转换位于 [patching.py](patching.py)，训练
recipe 位于 [training.py](training.py)。
