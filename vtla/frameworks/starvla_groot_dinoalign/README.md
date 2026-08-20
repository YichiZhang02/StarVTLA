# StarVLA-GR00T DINOAlign

`starvla_groot_dinoalign` 是独立注册的 StarVLA-GR00T policy。它在训练阶段使用冻结的
DINOv3 ViT-B/16 teacher，对齐 Qwen3.5 视觉 backbone 的 post-merger token，以提高模型
对亮度、曝光和局部阴影变化的鲁棒性。

该模型拥有独立的 configuration、policy、processor、Qwen interface 和 action head，修改
它不会改变原版 `starvla_groot`。DINOv3 只参与训练，不进入部署推理图。

## 视觉对齐

每张图像的特征路径：

```text
基准图像 x ───────────────→ Frozen DINOv3 ViT-B/16
  256x256                        16x16x768 patch tokens
                                      │ 2x2 average pooling
                                      ▼
                                  8x8x768 teacher tokens

光照增强 A(x) → Qwen3.5 visual backbone
  256x256                        8x8x1024 student tokens
                                      │ LayerNorm + Linear(1024, 768)
                                      ▼
                              patch/global cosine alignment
```

Qwen 的配置输入为 `224x224`，其 smart resize 实际将图像调整为 `256x256`。因此：

| 分支 | 原始网格 | 对齐网格 | 每张图 token 数 | 维度 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5 visual backbone | `8x8` | `8x8` | 64 | 1024 |
| DINOv3 ViT-B/16 | `16x16` | `8x8` | 64 | 768 |

## 损失函数

```text
L_dino  = 0.8 * L_patch + 0.2 * L_global
L_total = L_action + lambda(step) * L_dino
```

默认配置：

| 配置 | 默认值 | 说明 |
| --- | ---: | --- |
| `dino_alignment_weight` | `0.1` | warmup 完成后的 DINO loss 权重 |
| `dino_global_loss_weight` | `0.2` | global loss 在 `L_dino` 中的比例 |
| `dino_alignment_warmup_steps` | `1000` | 对齐权重线性 warmup 步数 |

对齐权重按以下公式增长：

```text
lambda(step) = 0.1 * min((step + 1) / 1000, 1)
```

例如 step 0、99、499、999 的权重分别为 `0.0001`、`0.01`、`0.05`、`0.1`。
`dino_alignment_step` 保存在 policy checkpoint 中，恢复训练不会重新开始 warmup。

这个 warmup 与学习率 warmup 相互独立。模型保留原版 StarVLA-GR00T 的
`scheduler_warmup_steps=1000`；当总训练步数少于 `scheduler_decay_steps=30000` 时，学习率
warmup 会由 scheduler 自动缩放，而 DINO loss warmup 固定为 1000 次有效训练前向。

## Teacher 冻结和梯度

DINOv3 teacher 使用以下机制保持冻结：

- 所有 teacher 参数均设置 `requires_grad=False`。
- teacher 始终处于 `eval` 模式，调用 `train(True)` 也不会启用训练模式。
- teacher 前向使用 `torch.inference_mode()`。
- teacher token 进入对齐损失前再次调用 `detach()`。
- teacher 作为非注册模块保存，不进入 `policy.parameters()`、optimizer 或 policy
  `state_dict`。

梯度只沿 student 路径传播：

```text
L_dino → alignment projector → Qwen image tokens → Qwen visual backbone
L_dino ✕ DINO tokens / DINO backbone / teacher 输入图像
```

因此使用 DINO 对齐时必须保持：

```text
freeze_vision_encoder=false
train_expert_only=false
```

配置校验会拒绝冻结 Qwen 视觉塔同时启用 DINO loss 的组合。

## 光照增强

训练时 DINO teacher 接收基准图像，Qwen student 接收保持几何位置不变的光照增强图像。
默认增强包括：

- brightness：`[0.6, 1.4]`
- contrast：`[0.7, 1.3]`
- gamma：`[0.7, 1.5]`
- soft shadow strength：`[0.2, 0.6]`
- 总增强概率：`0.8`
- 阴影概率：`0.5`

这些变换不会 crop、平移或旋转图像，因此 Qwen 与 DINO 的 64 个空间 token 保持逐位置对应。

## 权重

默认 teacher：

```text
model:      vit_base_patch16_dinov3
checkpoint: playground/pretrained_models/vit_base_patch16_dinov3.lvd1689m
input:      256x256
hidden:     768
dtype:      bfloat16
```

checkpoint 参数既可以指向 timm 权重文件，也可以指向包含 `model.safetensors` 或
`pytorch_model.bin` 的目录。通过环境变量覆盖默认位置：

```bash
DINOV3_CHECKPOINT=/path/to/dinov3_directory_or_file \
  bash train.sh <dataset_id> starvla_groot_dinoalign
```

本地 `lvd1689m/model.safetensors` 已完成真实加载、前向和反向验证。

## 训练

训练配置会强制要求：

```text
augmentation_mode=none
COLOR_TEMP_RANGE=[0,0]
```

`TrainPipelineConfig.validate()` 会在加载模型和数据前检查这两个值。`mild`、`strong`、
非零色温范围以及显式关闭 `COLOR_TEMP_RANGE` 都会报错，确保 DINO teacher 接收未经过
dataset 级增强的基准图像。Qwen student 的增强只由 policy 内部
`dino_light_augmentation` 控制。

使用默认 DINOv3 路径：

```bash
bash train.sh <dataset_id> starvla_groot_dinoalign
```

完整示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash train.sh \
  rm_umi_dual_pen_open starvla_groot_dinoalign 4 16 30000 \
  true none absolute_joint absolute_joint none
```

常用 policy override：

```text
--policy.dino_alignment_weight=0.1
--policy.dino_global_loss_weight=0.2
--policy.dino_alignment_warmup_steps=1000
--policy.dino_light_augmentation=true
--policy.dino_light_augmentation_probability=0.8
```

## 日志

训练 tqdm 仅在 policy 返回 DINO loss 时显示：

```text
action_loss=... dino_loss=... lr=...
```

其中 tqdm 的 `dino_loss` 对应未乘 warmup 权重的 `dino_alignment_loss`。原版
`starvla_groot` 和其他 policy 不显示该字段。

W&B 每个 `log_freq` 记录：

| 日志键 | 含义 |
| --- | --- |
| `action_loss` | flow-matching action loss |
| `dino_alignment_loss` | `0.8 * patch + 0.2 * global` |
| `dino_patch_loss` | 逐空间 token cosine loss |
| `dino_global_loss` | 每视角全局平均特征 cosine loss |
| `dino_alignment_weight` | 当前 warmup 后的实际权重 |
| `loss` | action loss 与加权 DINO loss 之和 |

实际加入总损失的 DINO 项为：

```text
dino_alignment_weight * dino_alignment_loss
```

## 推理

推理时不会加载 DINOv3 teacher，也不会执行光照增强或计算对齐 loss。部署模型只保留
训练后的 Qwen3.5、alignment projector 参数和 GR00T action head；projector 不参与推理
前向，因此 DINOAlign 不增加部署计算量。

## 验证结果

- DINOv3 输出：`(batch*views, 64, 768)`。
- Qwen 输出：`(batch*views, 64, 1024)`。
- 真实 Qwen3.5 + DINOv3 联合 forward/backward 通过。
- DINOv3 85,641,216 个参数全部冻结，所有参数梯度为 `None`。
- teacher 输入梯度为 `None`，student token 和 alignment projector 梯度非零。
- DINOv3 teacher 不出现在 policy `state_dict` 中。
