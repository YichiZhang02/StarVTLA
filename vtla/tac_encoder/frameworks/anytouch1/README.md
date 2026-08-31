# AnyTouch1

`anytouch1` 是 AnyTouch 第一阶段的触觉 MAE 适配器。每个时间帧独立经过 ViT，训练时随机
遮挡空间 patch 并重建像素；多帧只在统一输入和下游 token 排列中组成窗口。

## 模型契约

| 项目 | 值 |
| --- | --- |
| 输入 | `[B, S, T, 3, 224, 224]`，范围 `[0,1]` |
| 默认骨干 | ViT-L/16 |
| 训练目标 | masked pixel MSE |
| 下游特征 | 每帧 global token + spatial grid |
| 预训练权重 | `playground/pretrained_models/AnyTouch-ViT-L-16/checkpoint.pth` |

训练 checkpoint 的 `encoder` 保存 ViT、projection、video position embedding 和 sensor
token；MAE decoder 保存在 `trainer`，下游加载时会被删除。

```bash
bash scripts/train_backbone.sh <dataset_id> anytouch1 4 32 5 1e-5 224 4 2
```

实现入口为 [model.py](model.py)，训练 recipe 位于 [training.py](training.py)。
