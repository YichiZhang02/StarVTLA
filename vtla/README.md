# VTLA Policy Training

`vtla/` 提供统一的数据集、配置、预处理管线和 policy 接口。仓库根目录的 `train.sh` 是日常策略训练入口。

## Policy 类型

| 类型 | 初始化方式 | 说明 |
| --- | --- | --- |
| `act` | 随机初始化 | Transformer action policy |
| `diffusion` | 随机初始化 | Conditional diffusion policy |
| `pi05` | `pi05_base` 完整 checkpoint | VLM prefix 加 flow matching action expert |
| `starvla_groot` | `Qwen3.5-0.8B` 基础 VLM | Qwen VLM 加 GR00T action head |
| `starvla_groot_dinoalign` | `Qwen3.5-0.8B` + DINOv3 ViT-B/16 | 冻结 DINO teacher 对齐 Qwen 视觉 backbone；推理不加载 DINO |
| `fastwam` | Wan2.2 组件和插值 DiT | 联合视频与动作生成的 world-action model |

预训练权重的目录约定见 [playground/README.md](../playground/README.md#预训练权重)。

## 训练入口

```bash
bash train.sh \
  <dataset_id> <policy_type> <num_processes> <batch_size> <steps> \
  <wrist_only> <tactile_mode> <state_mode> <action_mode> \
  <augmentation_mode> [tactile_encoder_path] [tactile_insert_location]
```

当前默认值以 `train.sh` 为准：

| 位置 | 参数 | 默认值 | 说明 |
| --- | --- | --- | --- |
| 1 | `dataset_id` | `rm_umi_dual_260708_pen_in_case_notac_undist_256` | `playground/data/` 下的数据集 |
| 2 | `policy_type` | `fastwam` | `act`、`diffusion`、`pi05`、`starvla_groot`、`starvla_groot_dinoalign` 或 `fastwam` |
| 3 | `num_processes` | `4` | Accelerate 进程数 |
| 4 | `batch_size` | `16` | 每进程 batch size |
| 5 | `steps` | `10_000` | 训练步数 |
| 6 | `wrist_only` | `true` | 是否只使用 wrist RGB 相机 |
| 7 | `tactile_mode` | `none` | `none`、`as_image` 或 `encode` |
| 8 | `state_mode` | `joint` | state 表示模式 |
| 9 | `action_mode` | `joint` | action 表示模式 |
| 10 | `augmentation_mode` | `none` | `none`、`mild` 或其他已注册 preset |
| 11 | `tactile_encoder_path` | `AnyTouch-ViT-L-16` | `encode` 模式的初始化权重 |
| 12 | `tactile_insert_location` | `encoder` | `encoder` 或 `decoder` |

最小示例：

```bash
bash train.sh rm_umi_dual_pen_open diffusion 4 16 20000 false none joint joint
```

输出位于：

```text
playground/results/models/<timestamp>_<dataset>_<policy>_<routing>/
```

checkpoint 和成功完成后的日志保存在同一 run 目录。

## 相机与触觉路由

相机 key 通过环境变量覆盖，格式为 Draccus 列表：

```bash
TOP_CAM='[observation.images.cam_top]'
WRIST_CAM='[observation.images.left_cam_wrist,observation.images.right_cam_wrist]'
TACTILE_KEYS='[observation.images.left_cam_finger0,observation.images.left_cam_finger1,observation.images.right_cam_finger0,observation.images.right_cam_finger1]'
```

触觉模式：

- `none`：不向 policy 输入触觉。
- `as_image`：把触觉帧作为额外图像输入。
- `encode`：使用触觉 MAE encoder，将触觉表示作为额外 token 或 context。

`encode` 模式的 encoder 和 query token 会随 policy 一起训练，不会冻结。

常用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TACTILE_ENCODER_PATH` | `playground/pretrained_models/AnyTouch-ViT-L-16` | 触觉 encoder 初始化 |
| `TACTILE_INSERT_LOCATION` | `encoder` | 插入 policy encoder 或 decoder |
| `TACTILE_NUM_TOKENS` | `16` | 每张触觉图生成的 token 数 |
| `TACTILE_NUM_FRAMES` | `1` | 每个观测使用的触觉历史帧数 |
| `TACTILE_FRAME_OFFSET` | `1` | 相邻触觉历史帧的间隔 |
| `COLOR_TEMP_RANGE` | `[0,0]` | 色温增强采样范围；显式空值可关闭 |
| `DINOV3_CHECKPOINT` | `playground/pretrained_models/vit_base_patch16_dinov3.lvd1689m` | DINOAlign 训练期 ViT-B/16 teacher 权重文件或目录 |

`starvla_groot_dinoalign` 强制要求 `augmentation_mode=none` 且
`COLOR_TEMP_RANGE=[0,0]`；其 Qwen student 光照增强由 policy 内部单独完成。

## 触觉时序窗口

`TACTILE_NUM_FRAMES=F` 和 `TACTILE_FRAME_OFFSET=K` 对应采样索引：

```text
[-(F-1)*K, ..., -K, 0]
```

例如 `F=3, K=2` 会取 `[-4, -2, 0]`。训练时由 dataset delta timestamps 获取；推理时 preprocessing pipeline 维护滑动窗口，并在 episode reset 时清空。

| Policy | `encode` 多帧 | `as_image` 多帧 |
| --- | --- | --- |
| ACT | 展平为额外 token | 每帧作为独立相机 |
| pi05 | 展平为额外 prefix token | 每帧作为独立相机 |
| starvla_groot | 拼接到 hidden states | 每帧作为独立相机 |
| starvla_groot_dinoalign | 拼接到 hidden states | 每帧作为独立图像并参与 DINO 对齐 |
| diffusion | 加入 global conditioning | 不支持 |
| fastwam | 注入 Video DiT 或动作 DiT context | 不支持 |

示例：

```bash
TACTILE_NUM_FRAMES=3 TACTILE_FRAME_OFFSET=2 \
  bash train.sh rm_umi_dual_pen_open act 4 16 50000 false encode joint joint
```

触觉 backbone 的训练和结构见 [Tactile MAE README](tac_encoder/tactile_mae/README.md)。

## EE 模式

除关节角外，state 和 action 支持末端位姿。训练前需要先生成对应数据列，见 [tools/README.md](../tools/README.md#生成-ee-pose-列)。

旋转表示：

| 格式 | 每臂维度 | 双臂维度 | 旋转部分 |
| --- | --- | --- | --- |
| `rot6d` | 10 | 20 | 旋转矩阵前两列 |
| `quat` | 8 | 16 | `[x, y, z, w]` quaternion |

State 模式：

| `state_mode` | 坐标系 | 旋转格式 |
| --- | --- | --- |
| `joint` | 关节空间 | 原始关节表示 |
| `episode_rot6d` | episode 首帧相对 | rot6d |
| `absolute_rot6d` | 机器人 base | rot6d |
| `episode_quat` | episode 首帧相对 | quaternion |
| `absolute_quat` | 机器人 base | quaternion |

Action 模式：

| `action_mode` | 说明 |
| --- | --- |
| `joint` | 关节动作 |
| `rot6d` | 相对当前观测的 EE action，rot6d 表示 |
| `quat` | 相对当前观测的 EE action，quaternion 表示 |

`state_mode` 和 `action_mode` 的旋转格式必须一致。旧名称 `episode_ee`、`absolute_ee` 和 `relative_ee` 仍作为 rot6d 别名接受。

```bash
bash train.sh rm_umi_dual_pen_open pi05 1 32 10000 false none episode_rot6d rot6d

bash train.sh rm_umi_dual_pen_open diffusion 4 32 20000 false none episode_quat quat
```

## FastWAM

FastWAM 需要本地 Wan2.2 Video DiT、VAE、T5、tokenizer、插值 DiT 骨干，以及每个数据集的文本 embedding。

准备命令和文件布局见：

- [生成插值 DiT](../tools/README.md#生成插值-dit)
- [预计算文本 Embedding](../tools/README.md#预计算文本-embedding)
- [运行时权重目录](../playground/README.md#预训练权重)

FastWAM 训练时会自动复用或生成数据集文本缓存。`VISUALIZATION_ENABLED` 控制训练可视化，默认开启。

触觉作为 `encode` context 时可以通过 `TACTILE_INSERT_LOCATION=encoder|decoder` 选择注入 Video DiT 或动作 DiT。触觉不计入 Wan 视频相机数，也不会扩大 RGB 拼接宽度。

```bash
TACTILE_INSERT_LOCATION=decoder \
TACTILE_NUM_FRAMES=3 \
TACTILE_FRAME_OFFSET=2 \
CUDA_VISIBLE_DEVICES=0 \
  bash train.sh rm_umi_dual_pen_open fastwam 1 1 50000 true encode joint joint
```

## Policy 专项文档

- [Tactile MAE](tac_encoder/tactile_mae/README.md)
- [StarVLA-GR00T](frameworks/starvla_groot/README.md)
- [StarVLA-GR00T DINOAlign](frameworks/starvla_groot_dinoalign/README.md)
