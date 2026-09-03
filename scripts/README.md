# Workflow Scripts

`scripts/` 提供数据处理、数据集合并、state 统计和触觉 backbone 训练的日常工作流。脚本会切换到仓库根目录，统一通过 Bash 启动：

```bash
bash scripts/<script>.sh ...
```

## 脚本索引

| 脚本 | 用途 |
| --- | --- |
| `process_joint_data.sh` | 关节数据去畸变、触觉转换、缩放和 FK-to-EE |
| `process_umi_data.sh` | 导入 unified-format UMI v2.5 数据并生成 EE 训练数据 |
| `process_backbone_data.sh` | 在 processed dataset 内生成触觉 backbone `.npy` cache |
| `train_backbone.sh` | 统一训练 AnyTouch1、AnyTouch2 或 Sparsh reconstruction backbone |
| `compute_mean_state.sh` | 统计 home joint 候选或全局 state |
| `merge_datasets.sh` | 合并脚本内配置的一组数据集 |
| `evaluate_policy_offline.sh` | 按完整 episode 离线评估 checkpoint |

## 关节数据处理

```bash
bash scripts/process_joint_data.sh <dataset_id> [size] [horizon] [action_gap]
```

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `dataset_id` | 脚本内默认值 | `playground/data/` 下的源数据集 |
| `size` | `256` | 最终视频边长 |
| `horizon` | `32` | 相对 action 统计包含的动作数量，通常等于训练 `chunk_size` |
| `action_gap` | `0` | 相对 action 统计的第一个 GT 偏移，必须与训练一致 |

环境变量：

| 变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `CROP` | `896` | 鱼眼去畸变后的中心裁剪边长 |
| `JOBS` | `12` | ffmpeg 并行 worker 数 |
| `CAMERAS` | 工具默认 | 覆盖需要去畸变的相机列表 |
| `CALIB` | 工具默认 | 覆盖标定文件 |

处理顺序固定为：

```text
源数据集
  -> 腕部 RGB 在原始分辨率去畸变并裁剪
  -> raw 触觉固定范围量化为 uint8
  -> 所有训练视频缩放到 size x size
  -> 根据 robot_type 选择 B/ISF FK 并生成 EE 列
  -> 重命名为 <id>_processed
```

源数据集不会被修改。目录产物为：

```text
<id>                         原始数据
<id>_undist                  去畸变阶段
<id>_undist_uint8            有 raw 触觉时的 uint8 中间阶段
<id>_undist_uint8_<size>     有触觉时的 resize 中间阶段
<id>_undist_<size>           无触觉时的 resize 中间阶段
<id>_processed               全部流程成功后的最终训练数据
```

脚本可从已完成的中间目录继续，但会拒绝同时存在多个矛盾阶段或元数据与目录后缀不一致的情况。

### Robot Type 前置条件

源数据集的 `meta/info.json` 必须包含严格 `robot_type`：

```json
{"robot_type": "rm_base_umi_dual"}
```

或：

```json
{"robot_type": "rm_isf_umi_left"}
```

`convert_joints_to_eepose.py` 不提供 `--robot-type` 覆盖。它同时检测 joint feature 的臂布局：

- `rm_base_umi_dual` 必须包含完整的 right 和 left 两臂，每臂 7 个关节和 1 个夹爪。
- `rm_isf_umi_left` 必须只包含完整的 left 臂，共 7 个关节和 1 个夹爪。

缺失、未知或布局不一致都会停止处理，避免生成错误 EE 数据。

当前 ISF 数据可直接运行：

```bash
bash scripts/process_joint_data.sh \
  rm_isf_umi_left_20260820_insert_easy_precise 256 32 6
```

## 触觉处理标准

只对标记为 `tactile_u16_fixed_v1` 的 feature 执行量化：

```text
FFV1/gbrp16le MKV, uint16 HWC
  -> 固定范围线性量化
  -> libx264rgb/gbrp/CRF0 MP4
  -> resize
  -> libx264/yuv420p/CRF0 MP4, tactile_u8_linear_v1
```

量化不使用 episode 或数据集统计量：

```python
# depth: 0..1000 -> 0..255
u8_depth = round(255 * clip(u16_depth, 0, 1000) / 1000)

# deformation: 29000..31000 -> 0..255，30000 -> 128
delta = clip(u16_deformation - 30000, -1000, 1000)
u8_deformation = where(
    delta < 0,
    128 - round(128 * abs(delta) / 1000),
    128 + round(127 * delta / 1000),
)
```

| 语义 | `uint16` | `uint8` |
| --- | ---: | ---: |
| depth 最小/中点/最大 | `0 / 500 / 1000` | `0 / 128 / 255` |
| deformation 下限/中心/上限 | `29000 / 30000 / 31000` | `0 / 128 / 255` |

超出范围会饱和。量化不可逆；最终 YUV420 视频还包含 RGB/YUV 转换和色度抽样，因此数值审计必须使用原始 `uint16` MKV。处理后 `meta/info.json` 和 `meta/tactile_encoding.json` 会同步更新。

## UMI Pose 数据处理

```bash
TASK="Put the board eraser into the cup." \
  bash scripts/process_umi_data.sh <dataset_id> [size] [horizon] [action_gap]
```

该流程适用于 unified-format UMI v2.5 双臂数据。它不修改源目录；全部验证成功后，输出
`playground/data/<dataset_id>_processed`：

```text
复制非视频 metadata/data
  -> 三路 RGB key 改为 cam_top + left/right_cam_wrist
  -> 四路触觉 key 改为 left/right_cam_finger0/1
  -> RGB 缩放到 size x size，所有视频裁掉超过 episode 长度的尾帧
  -> 触觉从 uint16 定点编码转换为 uint8 无损 RGB MP4
  -> 删除未使用的额外 RGB/video feature 及其陈旧统计
  -> 写入真实 task 和 robot_type=umi
  -> 按全数据最小夹爪值和原始零点完成夹爪标定
  -> 从 UMI pose 生成 absolute/episode rot6d 与 quaternion EE 列
  -> 生成 relative-action stats 和 meta/umi_processing.json
  -> 完整验证后将临时目录原子改名为 <id>_processed
```

相机映射固定为：

| UMI v2.5 key | VTLA key |
| --- | --- |
| `observation.images.ego_right_undist` | `observation.images.cam_top` |
| `observation.images.cam_left_undist` | `observation.images.left_cam_wrist` |
| `observation.images.cam_right_undist` | `observation.images.right_cam_wrist` |

触觉映射固定为：

| UMI v2.5 key | VTLA key |
| --- | --- |
| `observation.depth_deformation.tactile_left_left` | `observation.images.left_cam_finger0` |
| `observation.depth_deformation.tactile_left_right` | `observation.images.left_cam_finger1` |
| `observation.depth_deformation.tactile_right_left` | `observation.images.right_cam_finger0` |
| `observation.depth_deformation.tactile_right_right` | `observation.images.right_cam_finger1` |

源 RGB 的 `_undist` key 表示图像已经去畸变，本流程不会重复去畸变。`size` 默认 `224`，
只作用于三路 RGB。触觉保持传感器原生 `96x128`（高 x 宽），不会被该流程缩放；后续模型
预处理负责调整模型输入尺寸。触觉视频从 FFV1/`gbrp16le` uint16 MKV 按项目固定范围量化为
`tactile_u8_linear_v1`，再保存为 `libx264rgb`/`gbrp` uint8 MP4。

processed 数据只保留 `cam_top`、`left/right_cam_wrist` 和四个 `left/right_cam_finger0/1`
视觉 feature；其他源 RGB/video feature、视频引用和已失效的触觉像素统计会被删除。

`TASK` 必须是真实指令，不能是 `Unknown task`。`JOBS` 默认 `12`。夹爪会同时扫描
`observation.state` 和 `action`，默认将每侧全数据最小值（设备实际能达到的最大张开量）映射为
`1`，原始值 `0` 映射为闭合 `0`，并裁剪到 `[0,1]`。需要手动覆盖时，可成对设置
`LEFT_GRIPPER_OPEN/CLOSED`、`RIGHT_GRIPPER_OPEN/CLOSED`。

该流程不执行 RealMan FK，也不使用 UMI 中无效的 joint/finger 占位字段。UMI absolute pose
没有 robot-base 外参标定时只适合数据分析；跨平台上机训练应优先使用 episode state + relative action。

## 触觉 Backbone 训练

```bash
bash scripts/process_backbone_data.sh <dataset_id> [--num_workers 4] [--overwrite]
bash scripts/train_backbone.sh \
  <dataset_id> <model_id> [num_processes] [batch_size] [epochs] \
  [lr] [image_size] [tactile_num_frames] [tactile_frame_offset] [resume]
```

`dataset_id` 可以是普通 processed dataset，也可以是 `configs/data_mixtures.yaml` 中的 mixture。cache 固定写入每个 concrete dataset 的 `tactile_backbone_cache/`；训练阶段只读取这些 `.npy` 文件。预处理默认使用 4 个 episode worker，只 resize 和保存有效接触窗口实际引用的唯一帧；可按 CPU 和内存情况调整 `--num_workers`。

旧的 `tactile_backbone_npy_v1` 全量 cache 不会被静默复用。首次切换到紧凑 v2 cache 时使用 `--overwrite` 显式重建；之后相同配置直接运行会复用现有 cache。

`train_backbone.sh` 根据 `model_id` 自动选择固定的预训练权重：

| `model_id` | `pretrained_path` |
| --- | --- |
| `anytouch1` | `playground/pretrained_models/AnyTouch-ViT-L-16/checkpoint.pth` |
| `anytouch2` | `playground/pretrained_models/AnyTouch2-Model/checkpoint-4frames.pth` |
| `sparsh_vjepa` | `playground/pretrained_models/Sparsh-VJEPA-Small/vjepa_vitsmall.safetensors` |
| `wan22_vae` | `playground/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` |

Sparsh 公开的 `vjepa_vitsmall.safetensors` 只包含 encoder。训练会用它同时初始化
context encoder 和 target encoder，JEPA predictor 从随机权重开始训练。若提供包含
context encoder、target encoder 和 predictor 的完整 checkpoint，则三部分都会恢复。

`wan22_vae` 对窗口内每张触觉图像独立执行 posterior sampling 和全图重建，loss 为 L1
reconstruction 加 `1e-6` 权重的 KL。它不会把 `T=4` 当作 Wan causal video 序列。VAE
显存和 checkpoint 占用明显高于其他 backbone；未传第 4 个参数时，它的 per-GPU batch
默认是 `4`，其他模型仍为 `32`。可用 `VAE_KL_WEIGHT` 环境变量覆盖 KL 权重。

```bash
bash scripts/train_backbone.sh dataset_a anytouch2 4 64 5 1e-5 224 4 2
# 从 checkpoint 恢复
bash scripts/train_backbone.sh dataset_a anytouch2 4 64 5 1e-5 224 4 2 path/to/last.pth
```

| 参数 | 默认值 | 可选值 |
| --- | --- | --- |
| `dataset_id` | `backbone_training_data` | 普通 dataset 或 named mixture |
| `model_id` | `anytouch1` | `anytouch1`、`anytouch2`、`sparsh_vjepa`、`wan22_vae` |
| `num_processes` | `4` | `1` 使用 Python，多进程使用 torchrun |
| `batch_size` | `32` | 每进程 / 每 GPU batch size；`wan22_vae` 默认 `4` |
| `epochs` | `5` | epoch 数 |
| `lr` | `1e-5` | AdamW 初始学习率 |
| `image_size` | `224` | cache 和模型输入分辨率 |
| `tactile_num_frames` | `4` | 每个触觉窗口包含的帧数 |
| `tactile_frame_offset` | `2` | 相邻触觉帧在原数据中的间隔 |
| `resume` | 空 | V2 checkpoint 路径；空值表示从固定 pretrained checkpoint 开始 |

输出保存在：

```text
playground/results/backbones/<YYYYMMDD>_<dataset_id>_<model_id>/
```

模型目录、公共接口和扩展方式见 [Tactile Encoders](../vtla/tac_encoder/README.md)。
各模型的结构、目标和 checkpoint 契约见对应文档：
[AnyTouch1](../vtla/tac_encoder/frameworks/anytouch1/README.md)、
[AnyTouch2](../vtla/tac_encoder/frameworks/anytouch2/README.md)、
[Sparsh V-JEPA](../vtla/tac_encoder/frameworks/sparsh_vjepa/README.md)、
[Wan2.2 VAE](../vtla/tac_encoder/frameworks/wan22_vae/README.md)。

## State 统计

```bash
bash scripts/compute_mean_state.sh [dataset_id] [first|all] [state_key]
```

`first` 对每个 episode 的首帧求统计，适合核对起始关节分布；`all` 对全部帧统计。默认列为 `observation.state`。关节 state 的 stdout 是诊断 JSON，详细统计写入 stderr；部署 home 始终取机械臂连接时姿态。

## 合并数据集

当前 [merge_datasets.sh](merge_datasets.sh) 的输出 ID 和源数据集数组定义在脚本内。修改后运行：

```bash
bash scripts/merge_datasets.sh
```

底层工具会取所有输入数据集 dtype/shape 一致的公共 feature，再合并为单一数据集。数据集聚合还要求相同 `fps` 和完全相同的 `robot_type`；不同物理构型不能合并。

需要从命令行指定输入时直接使用：

```bash
python tools/merge_datasets.py \
  --roots playground/data/A playground/data/B \
  --out playground/data/A_B_merged \
  --repo-id A_B_merged
```

## Policy 离线评估

```bash
bash scripts/evaluate_policy_offline.sh \
  <dataset_id> <pretrained_id> <step|last> [episodes] [stride] [device]
```

示例：

```bash
bash scripts/evaluate_policy_offline.sh \
  rm_isf_umi_left_20260820_insert_easy_precise_undist_uint8_256 \
  20260821_rm_isf_umi_left_20260820_insert_easy_precise_undist_uint8_256_starvla_groot_wristonly_true_tactile_none_state_absolute_rot6d_action_relative_rot6d_aug_strong \
  3000 0-2 1 cuda
```

`episodes` 支持 `all`、`0,2,5` 或 `0-3`。输出位于对应 checkpoint 的
`<pretrained_id>/offline_eval/<step|last>/<dataset_id>/`，其中 `offline_eval` 与 `checkpoints` 同级。目录中包含
action mode 空间与机器人命令空间的完整 episode 曲线、NPZ
预测数据及 episode/全局误差指标。
