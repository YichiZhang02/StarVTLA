# Wan2.2 VAE

`wan22_vae` 使用 Wan2.2-TI2V-5B 的 48-channel VAE 做触觉全图重建后训练。初始化权重与
Cosmos3-Edge 中的 VAE 数值相同；这里使用官方 FP32 `.pth` 以便继续训练。

## 模型契约

| 项目 | 值 |
| --- | --- |
| 输入 | `[B, S, T, 3, H, W]`，范围 `[0,1]` |
| 默认空间压缩率 | `16` |
| 默认 latent 维度 | `48` |
| 训练目标 | full-frame L1 reconstruction + weighted posterior KL |
| 默认 KL 权重 | `1e-6` |
| 预训练权重 | `playground/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` |

当前触觉数据中每个时间步是一张图像。适配器会把 `S x T` 张图像展平为独立的单帧 VAE
样本，不会把四帧误解释成 Wan 的 causal video chunk。训练时从 posterior 采样，验证和
下游编码使用 posterior mean；下游特征还会应用 Wan2.2 官方 latent mean/std 归一化。

训练 checkpoint 的 `encoder` 保存 VAE encoder 和 posterior projection，decoder 保存在
`trainer`。下游 policy 加载时删除 decoder，仅保留 `48 x H/16 x W/16` latent grid。

```bash
bash scripts/train_backbone.sh <dataset_id> wan22_vae 4 4 5 1e-5 224 4 2
```

Wan2.2 VAE 较大，示例使用较小的 per-GPU batch。该 recipe 是面向触觉重建的后训练目标，
不等同于 Wan 原始包含感知或对抗项的完整 VAE 训练配方。

KL 权重可在 shell 入口覆盖：

```bash
VAE_KL_WEIGHT=1e-5 bash scripts/train_backbone.sh <dataset_id> wan22_vae
```

实现入口为 [model.py](model.py)，训练 recipe 位于 [training.py](training.py)。
