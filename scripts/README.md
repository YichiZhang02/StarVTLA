# Workflow Scripts

`scripts/` 提供数据处理、数据集合并、state 统计和触觉 backbone 训练的日常工作流。脚本会切换到仓库根目录，统一通过 Bash 启动：

```bash
bash scripts/<script>.sh ...
```

## 脚本索引

| 脚本 | 用途 |
| --- | --- |
| `process_joint_data.sh` | 关节数据去畸变、触觉转换、缩放和 FK-to-EE |
| `process_umi_data.sh` | 已有 UMI pose 数据去畸变、缩放和 EE 标准化 |
| `train_enc.sh` | 训练 AnyTouch stage-1 风格的触觉 MAE |
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
bash scripts/process_umi_data.sh <dataset_id> [size] [horizon] [action_gap]
```

该流程适用于数据中已经保存末端位姿的 UMI 数据：

```text
<id>
  -> <id>_undist
  -> <id>_undist_<size>
  -> convert_umi_to_eepose 原地生成统一 EE 列和统计
```

它不执行 RealMan FK，因此不用于当前 joint-only 真机采集数据。`CROP`、`JOBS`、`CAMERAS` 和 `CALIB` 与关节处理脚本一致。

## 触觉 Backbone 训练

```bash
bash scripts/train_enc.sh \
  <dataset_ids> <init_mode> <arch> <num_processes> <batch_size> <epochs>
```

多个数据集 ID 使用带引号的空格分隔字符串：

```bash
bash scripts/train_enc.sh \
  "dataset_a dataset_b" clip vit_b 4 128 100
```

| 参数 | 默认值 | 可选值 |
| --- | --- | --- |
| `dataset_ids` | `pretrained_data` | 一个或多个本地数据集 |
| `init_mode` | `clip` | `scratch`、`clip`、`anytouch` |
| `arch` | `vit_b` | `vit_b`、`vit_l` |
| `num_processes` | `4` | `1` 使用 Python，多进程使用 torchrun |
| `batch_size` | `128` | 每进程 batch size |
| `epochs` | `100` | epoch 数 |

`anytouch` 初始化只支持 `vit_l`。输出保存在：

```text
playground/results/backbones/<timestamp>_tacmae_<arch>_from_<init_mode>/
```

脚本通过 `meta/info.json` 区分 LeRobot 数据集和 raw frame cache；两种模式不能在同一次训练中混用。raw 模式没有逐帧 LeRobot 来源，因此自动关闭 contact filter。

常用环境变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `TACTILE_KEYS` | 自动选择 | 触觉 camera key 列表 |
| `RAW_FRAME_CACHE` | 自动识别 | `1/0` 强制选择输入模式 |
| `SENSOR_ID` | `-1` | AnyTouch sensor token |
| `MASK_RATIO` | `0.75` | MAE mask 比例 |
| `VISIBLE_LOSS_WEIGHT` | `0.1` | 可见 patch loss 权重 |
| `CONTACT_FILTER` | `1` | LeRobot 接触帧筛选 |
| `CONTACT_STD_THRESHOLD` | `0.5` | 接触判定阈值 |
| `NONCONTACT_KEEP_RATIO` | `0.05` | 非接触帧保留比例 |
| `IMAGE_SIZE` | `224` | 输入和 cache 图像尺寸 |
| `NUM_WORKERS` | `12` | 数据 worker 数 |
| `RUN_NAME` | 当前时间 | 输出目录前缀 |

模型和初始化细节见 [Tactile MAE](../vtla/tac_encoder/tactile_mae/README.md)。

## State 统计

```bash
bash scripts/compute_mean_state.sh [dataset_id] [first|all] [state_key]
```

`first` 对每个 episode 的首帧求统计，适合得到 home joint 候选；`all` 对全部帧统计。默认列为 `observation.state`。关节 state 的 stdout 可直接用作 `--robot.home_joints` 参数，详细统计写入 stderr。

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
