# VTLA Training

`vtla/` 包含 LeRobot 数据集、训练配置、pre/postprocessor 和全部 policy 实现。日常训练入口是仓库根目录的 [train.sh](../train.sh)。

命名数据集 mixture 定义在 [`configs/data_mixtures.yaml`](../configs/data_mixtures.yaml)。mixture ID 与普通 dataset ID 一样直接传给 `train.sh`；数据只从各成员原目录读取，不会生成合并副本。

## Policy 类型

| `policy_type` | 初始化 | 说明 |
| --- | --- | --- |
| [`act`](frameworks/act/README.md) | 随机初始化 | Transformer action chunk policy |
| [`diffusion`](frameworks/diffusion/README.md) | 随机初始化 | Conditional diffusion policy |
| [`pi05`](frameworks/pi05/README.md) | 本地 `pi05_base` | VLM prefix + flow-matching action expert |
| [`starvla_groot`](frameworks/starvla_groot/README.md) | 本地 Qwen3.5-0.8B | Qwen vision-language backbone + GR00T action head |
| [`starvla_groot_dinoalign`](frameworks/starvla_groot_dinoalign/README.md) | Qwen3.5 + DINOv3 teacher | 训练期视觉对齐，部署不加载 teacher |
| [`fastwam`](frameworks/fastwam/README.md) | 本地 Wan2.2 组件 | 联合视频与动作生成的 world-action model |
| [`dream_tac`](frameworks/dream_tac/README.md) | 本地 Cosmos-Predict2-2B-Video2World | 联合 RGB、触觉、状态和动作生成的 world-action model |

## 训练命令

```bash
bash train.sh \
  <dataset_id> <policy_type> <num_processes> <batch_size> <steps> \
  <wrist_only> <tactile_mode> <state_mode> <action_mode> \
  [action_gap] <augmentation_mode> \
  [tactile_encoder_path] [tactile_insert_location] [tactile_pool_size]
```

| 位置 | 参数 | 脚本默认值 | 可选值或含义 |
| ---: | --- | --- | --- |
| 1 | `dataset_id` | 脚本内当前数据集 | `playground/data/` 下的目录名 |
| 2 | `policy_type` | `starvla_groot_dinoalign` | 上表所列 policy 类型 |
| 3 | `num_processes` | `4` | Accelerate 进程数 |
| 4 | `batch_size` | `4` | 每进程 batch size |
| 5 | `steps` | `40000` | 训练步数 |
| 6 | `wrist_only` | `true` | 是否只使用 wrist RGB |
| 7 | `tactile_mode` | `none` | `none`、`as_image`、`encode` |
| 8 | `state_mode` | `absolute_joint` | 见下文 |
| 9 | `action_mode` | `absolute_joint` | 见下文 |
| 10 | `action_gap` | `6` | GT action 起点相对当前观测向未来偏移的帧数 |
| 11 | `augmentation_mode` | `none` | `none`、`mild`、`strong` |
| 12 | `tactile_encoder_path` | 空 | `train_backbone.sh` 生成的 checkpoint，仅用于首次初始化 `encode` 训练 |
| 13 | `tactile_insert_location` | `encoder` | `encoder`、`decoder` |
| 14 | `tactile_pool_size` | `3` | `3` 表示下游 `AdaptiveAvgPool2d(3,3)` |

关节训练示例：

```bash
bash train.sh <dataset_id> starvla_groot 1 4 10000 \
  true none absolute_joint absolute_joint 6 none
```

输出为：

```text
playground/results/models/<date>_<dataset>_<policy>_<routing>/
├── checkpoints/<step>/pretrained_model/
└── <run_name>.log
```

## Robot Type 契约

训练不接收独立的 `robot_type` 参数。`make_policy()` 从数据集 `meta/info.json.robot_type` 读取它，校验臂 feature 布局，然后写入 policy 的 `config.json`：

```text
dataset robot_type -> policy.config.robot_type -> checkpoint config.json
```

物理机器人数据使用注册的具体类型，例如：

- `rm_base_umi_dual`：B/base 双臂。
- `rm_isf_umi_left`：ISF 单左臂。

由 `scripts/process_umi_data.sh` 导入的通用 UMI pose 数据固定使用 `robot_type=umi`。训练只允许
EE state（或 `none`）和 EE action，并从 canonical EE feature names 校验单/双臂布局；checkpoint
继续保存 `umi`，不会在训练时伪装成具体机械臂。

具体类型 checkpoint 在推理时仍是 RobotConfig 和 FK/IK 的权威来源。只有 `umi` checkpoint 是
例外：推理必须显式提供具体 `--robot.type`，运行时以 CLI 类型选择 RobotConfig 和 FK/IK，磁盘上的
checkpoint 配置保持 `umi`。

单任务数据集的任务文本也会写入 `policy.config.single_task`。多任务数据集不会选取任意一个任务，推理时需要显式指定。

## State 和 Action

`state_mode`：

| 模式 | 输入表示 | 坐标参考 |
| --- | --- | --- |
| `none` | 不输入 proprioception | 无 |
| `absolute_joint` | 原始 joint state | 绝对关节角 |
| `episode_joint` | joint state | 相对 episode 首帧 |
| `absolute_rot6d` | EE pose | robot base，rot6d |
| `episode_rot6d` | EE pose | 相对 episode 首帧，rot6d |
| `absolute_quat` | EE pose | robot base，quaternion |
| `episode_quat` | EE pose | 相对 episode 首帧，quaternion |

`action_mode`：

| 模式 | 输出表示 | 参考 |
| --- | --- | --- |
| `absolute_joint` | joint | 数据集中的绝对动作 |
| `relative_joint` | joint | 相对当前观测 |
| `absolute_rot6d` | EE rot6d | robot base |
| `relative_rot6d` | EE rot6d | 相对当前观测 EE |
| `absolute_quat` | EE quaternion | robot base |
| `relative_quat` | EE quaternion | 相对当前观测 EE |

EE 模式要求数据集先经过：

```bash
bash scripts/process_joint_data.sh <dataset_id> 256 32
# 或 unified-format UMI v2.5：
TASK="..." bash scripts/process_umi_data.sh <dataset_id> 256 32 6
```

每臂 rot6d 为 10 维 `[xyz, rot6d(6), gripper]`，quaternion 为 8 维 `[xyz, xyzw, gripper]`。`rm_base_umi_dual` 分别为 20/16 维，`rm_isf_umi_left` 分别为 10/8 维。

state 和 action 使用 EE 时应采用相同旋转表示。例如：

```bash
bash train.sh <processed_dataset_id> pi05 1 32 10000 \
  false none absolute_rot6d relative_rot6d 6 none
```

第 10 个参数是 `action_gap`。`chunk_size=32, action_gap=6` 时，GT 时间窗口为 `t+6 ... t+37`。`relative_*` action 仍以当前观测 `S(t)` 为 pose anchor；推理 postprocessor 将其恢复为可执行的绝对目标。EE action 会让部署端自动选择 `robot.action_space=ee`，joint action 则选择 `joint`。

## 相机路由

`train.sh` 默认从数据集的 video feature 自动分类：

- key 中包含 `finger` 或带 `tactile_encoding` 的 feature 作为触觉。
- 其余 key 中包含 `wrist` 的 feature 作为腕部 RGB。
- 剩余 video feature 作为顶部或环境 RGB。

需要显式覆盖时使用 Draccus 列表格式：

```bash
TOP_CAM='[observation.images.cam_top]' \
WRIST_CAM='[observation.images.left_cam_wrist]' \
TACTILE_KEYS='[observation.images.left_cam_finger0,observation.images.left_cam_finger1]' \
  bash train.sh <dataset_id> act
```

这样单左臂和双臂数据无需在文档或脚本中维护两套固定 key。

## 触觉路由

| `tactile_mode` | 行为 |
| --- | --- |
| `none` | policy 不消费触觉 feature |
| `as_image` | 每路触觉作为额外图像视角 |
| `encode` | 统一触觉 backbone 经下游 spatial pooling 后生成 token/context |

`encode` 模式的 tactile encoder 会随 policy 训练，不默认冻结。spatial pooling 仅发生在 policy 特征提取阶段，不参与 backbone 重建训练。常用变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `TACTILE_ENCODER_PATH` | 空 | `train_backbone.sh` 生成的 checkpoint |
| `TACTILE_INSERT_LOCATION` | `encoder` | 注入 encoder 或 decoder |
| `TACTILE_POOL_SIZE` | `3` | 下游 spatial pooling 边长；`3` 表示 `3x3` |
| `TACTILE_NUM_FRAMES` | `4` | 每次观测的历史帧数 |
| `TACTILE_FRAME_OFFSET` | `2` | 历史帧间隔 |

时间窗口 `F=TACTILE_NUM_FRAMES`、`K=TACTILE_FRAME_OFFSET` 对应：

```text
[-(F-1)*K, ..., -K, 0]
```

例如 `F=3, K=2` 使用 `[-4, -2, 0]`。训练通过 delta timestamps 取帧，推理 processor 维护窗口并在 episode reset 时清空。

| Policy | `encode` 多帧 | `as_image` 多帧 |
| --- | --- | --- |
| ACT | 额外 token | 独立图像视角 |
| pi05 | 额外 prefix token | 独立图像视角 |
| StarVLA-GR00T | hidden-state context | 独立图像视角 |
| DINOAlign | hidden-state context | 独立图像并参与训练期对齐 |
| Diffusion | global conditioning | 不支持 |
| FastWAM | Video DiT 或 action DiT context | Wan VAE history token，要求插入 encoder |

触觉 backbone 训练见 [Tactile Encoders](tac_encoder/README.md)。
训练初始化后，backbone 结构写入 policy 的 `config.json`，权重写入
`model.safetensors`；推理只依赖 policy checkpoint，不再读取 `tactile_encoder_path`。

## 数据增强

`augmentation_mode` 由 dataset image transform preset 处理。`COLOR_TEMP_RANGE` 可单独控制色温范围：

```bash
COLOR_TEMP_RANGE='[-0.1,0.1]' bash train.sh <dataset_id> act
```

`starvla_groot_dinoalign` 必须使用：

```text
augmentation_mode=none
COLOR_TEMP_RANGE=[0,0]
```

它的 student 光照增强在 policy 内完成，不能再叠加 dataset 级几何或颜色增强。

## FastWAM

FastWAM 需要本地 Wan2.2 Video DiT、VAE、T5、tokenizer 和插值 DiT。训练前还必须生成数据集文本缓存：

```bash
python tools/precompute_world_model_text_embeddings.py \
  --dataset-root playground/data/<dataset_id> \
  --world-model wan22 \
  --device cuda
```

缺少 `text_embeddings/wan22/manifest.json` 或 `embeddings.safetensors` 时，训练会直接报错。生成插值 DiT 和完整资产说明见 [Offline Tools](../tools/README.md#fastwam-资产)。

```bash
TACTILE_INSERT_LOCATION=decoder \
TACTILE_NUM_FRAMES=3 \
VISUALIZATION_ENABLED=true \
  bash train.sh <dataset_id> fastwam 1 1 50000 \
  true encode absolute_joint absolute_joint 6 none
```

## Dream-Tac

Dream-Tac 使用仓库内置的 Cosmos Policy core，并接入 StarVTLA 的 LeRobot 数据、动态
sensor slot、Accelerate 训练、checkpoint、文本 embedding 缓存和 visualization eval。
它支持 `tactile_mode=none|as_image`，state/action 可选 joint、rot6d 或 quaternion，单双臂
由数据 feature 和相机、触觉 key 共同确定。

```bash
VISUALIZATION_ENABLED=true bash train.sh <dataset_id> dream_tac 1 1 50000 \
  false as_image absolute_rot6d absolute_rot6d 0 none
```

环境安装、预训练资产、slot 布局、T5 预计算、loss、推理、可视化和 checkpoint 语义见
[Dream-Tac 专项文档](frameworks/dream_tac/README.md)。

## 专项文档

- [StarVLA-GR00T](frameworks/starvla_groot/README.md)
- [StarVLA-GR00T DINOAlign](frameworks/starvla_groot_dinoalign/README.md)
- [Dream-Tac](frameworks/dream_tac/README.md)
- [Tactile Encoders](tac_encoder/README.md)
- [AnyTouch1 Backbone](tac_encoder/frameworks/anytouch1/README.md)
- [AnyTouch2 Backbone](tac_encoder/frameworks/anytouch2/README.md)
- [Sparsh V-JEPA Backbone](tac_encoder/frameworks/sparsh_vjepa/README.md)
- [Wan2.2 VAE Backbone](tac_encoder/frameworks/wan22_vae/README.md)
