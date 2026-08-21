# StarVLA-GR00T DINOAlign

`starvla_groot_dinoalign` 是独立注册的 StarVLA-GR00T 变体。训练时使用冻结的 DINOv3 ViT-B/16 teacher，对齐 Qwen3.5 视觉 backbone 的 post-merger token，以增强对亮度、曝光和局部阴影变化的鲁棒性。DINO teacher 不进入部署推理图。

共享的数据集、`robot_type`、state/action 和触觉契约见 [VTLA Training](../../README.md)。

## 对齐路径

```text
基准图像 x -> Frozen DINOv3 ViT-B/16
  256x256      16x16x768 patch tokens
                   -> 2x2 average pooling
                   -> 8x8x768 teacher tokens

光照增强 A(x) -> Qwen3.5 visual backbone
  256x256         8x8x1024 student tokens
                     -> LayerNorm + Linear(1024, 768)
                     -> patch/global cosine alignment
```

Qwen 配置输入为 `224x224`，smart resize 实际产生 `256x256`。两条路径最终都得到每张图 64 个空间对应 token：

| 分支 | 原始网格 | 对齐网格 | token 维度 |
| --- | ---: | ---: | ---: |
| Qwen3.5 student | `8x8` | `8x8` | 1024，再投影到 768 |
| DINOv3 teacher | `16x16` | `8x8` | 768 |

## Loss

```text
L_dino  = 0.8 * L_patch + 0.2 * L_global
L_total = L_action + lambda(step) * L_dino
```

默认配置：

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `dino_alignment_weight` | `0.1` | warmup 后的最大对齐权重 |
| `dino_global_loss_weight` | `0.2` | global loss 在 `L_dino` 中的比例 |
| `dino_alignment_warmup_steps` | `1000` | 对齐权重线性 warmup |

```text
lambda(step) = 0.1 * min((step + 1) / 1000, 1)
```

`dino_alignment_step` 会保存到 checkpoint，恢复训练不会重新开始 warmup。该 warmup 与学习率 scheduler 的 warmup 相互独立。

## Teacher 冻结边界

Teacher 保持冻结的机制包括：

- 所有参数 `requires_grad=False`。
- 始终处于 `eval()`，policy 切到 train mode 时也不改变。
- 前向运行在 `torch.inference_mode()`。
- teacher token 在进入 loss 前再次 `detach()`。
- teacher 不注册为 policy 子模块，不进入 optimizer 或 policy `state_dict`。

梯度只沿以下路径传播：

```text
L_dino -> alignment projector -> Qwen image tokens -> Qwen visual backbone
```

启用 DINO 对齐时必须保持：

```text
freeze_vision_encoder=false
train_expert_only=false
```

配置校验会拒绝冻结 student 视觉塔的组合。

## 光照增强

Teacher 接收基准图像，student 接收只改变光照、不改变几何位置的增强图像。默认范围：

| 变换 | 默认值 |
| --- | --- |
| brightness | `[0.6, 1.4]` |
| contrast | `[0.7, 1.3]` |
| gamma | `[0.7, 1.5]` |
| soft shadow strength | `[0.2, 0.6]` |
| 总增强概率 | `0.8` |
| 阴影概率 | `0.5` |

增强不执行 crop、平移或旋转，保证 teacher 和 student 的 64 个空间 token 逐位置对应。

Dataset 级增强必须关闭：

```text
augmentation_mode=none
COLOR_TEMP_RANGE=[0,0]
```

`TrainPipelineConfig.validate()` 会在加载数据和模型前检查这两个条件。`mild`、`strong`、非零色温或显式关闭色温配置都会失败。

## Teacher 权重

默认配置：

```text
model:      vit_base_patch16_dinov3
checkpoint: playground/pretrained_models/vit_base_patch16_dinov3.lvd1689m
input:      256x256
hidden:     768
dtype:      bfloat16
```

checkpoint 可以是 timm 权重文件，也可以是包含 `model.safetensors` 或 `pytorch_model.bin` 的目录。通过环境变量覆盖：

```bash
DINOV3_CHECKPOINT=/path/to/dinov3 \
  bash train.sh <dataset_id> starvla_groot_dinoalign
```

## 训练

最小命令：

```bash
bash train.sh <dataset_id> starvla_groot_dinoalign
```

完整示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
COLOR_TEMP_RANGE='[0,0]' \
  bash train.sh <dataset_id> starvla_groot_dinoalign 4 16 30000 \
  true none absolute_joint absolute_joint 6 none
```

常用 override：

```text
--policy.dino_alignment_weight=0.1
--policy.dino_global_loss_weight=0.2
--policy.dino_alignment_warmup_steps=1000
--policy.dino_light_augmentation=true
--policy.dino_light_augmentation_probability=0.8
```

## 日志

训练进度在存在 DINO loss 时显示 `action_loss`、`dino_loss` 和 `lr`。其中 `dino_loss` 是尚未乘当前 warmup 权重的 `dino_alignment_loss`。

W&B 记录：

| 键 | 含义 |
| --- | --- |
| `action_loss` | flow-matching action loss |
| `dino_alignment_loss` | patch/global 组合 loss |
| `dino_patch_loss` | 逐空间 token cosine loss |
| `dino_global_loss` | 每视角全局特征 cosine loss |
| `dino_alignment_weight` | 当前实际权重 |
| `loss` | action loss + 加权 DINO loss |

## 推理

```bash
bash inference.sh <run_id> <step>
```

推理不加载 DINOv3，不执行训练期光照增强，也不计算 alignment loss。部署前向使用训练后的 Qwen3.5 和 GR00T action head；alignment projector 保存在 checkpoint 中但不参与推理计算。机器人类型仍严格来自 checkpoint。
