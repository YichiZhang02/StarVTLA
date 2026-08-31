# Sparsh V-JEPA

`sparsh_vjepa` 在同一个触觉历史窗口内预测被 mask 的 latent target。它预测的是缺失的
spatiotemporal patch 表征，不是未来触觉帧。

## 模型契约

| 项目 | 值 |
| --- | --- |
| 输入 | `[B, S, T, 3, H, W]`，范围 `[0,1]` |
| 默认 patch/tubelet | `16 / 2` |
| 默认 encoder | ViT-S，特征维度 `384` |
| 训练目标 | masked latent prediction |
| 预训练权重 | `playground/pretrained_models/Sparsh-VJEPA-Small/vjepa_vitsmall.safetensors` |

公开 checkpoint 只有 encoder。训练初始化时，context encoder 和 EMA target encoder 都从
该权重加载，predictor 从随机权重开始；若输入完整训练 checkpoint，则三部分全部恢复。
下游 checkpoint 只使用 context encoder。

```bash
bash scripts/train_backbone.sh <dataset_id> sparsh_vjepa 4 32 5 1e-5 224 4 2
```

encoder-only 适配器位于 [model.py](model.py)，JEPA predictor、mask sampler 和 EMA 更新位于
[training.py](training.py)。
