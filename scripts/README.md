# Workflow Scripts

`scripts/` 保存面向日常操作的 shell 工作流。每个脚本都会先切换到仓库根目录，因此可以通过相对路径访问 `tools/` 和 `playground/`。

统一使用 Bash：

```bash
bash scripts/<script>.sh ...
```

## 脚本索引

| 脚本 | 用途 |
| --- | --- |
| `train_enc.sh` | 训练触觉 MAE backbone |
| `process_joint_data.sh` | 关节数据：去畸变、触觉 uint8、全视频 resize、FK 转 EE |
| `process_umi_data.sh` | UMI 数据：去畸变、降分辨率、生成 EE 表示 |
| `compute_mean_state.sh` | 计算 state 均值并生成 home joints 参数 |
| `merge_datasets.sh` | 对齐特征并合并预先配置的数据集列表 |

## 触觉 Backbone 训练

```bash
bash scripts/train_enc.sh \
  <dataset_ids> <init_mode> <arch> <num_processes> <batch_size> <epochs>
```

示例：

```bash
bash scripts/train_enc.sh rm_umi_dual_pen_open clip vit_l 4 128 100
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `dataset_ids` | `pretrained_data` | 一个或多个数据集；多个 ID 使用带引号的空格分隔字符串 |
| `init_mode` | `clip` | `scratch`、`clip` 或 `anytouch` |
| `arch` | `vit_b` | `vit_b` 或 `vit_l` |
| `num_processes` | `4` | 大于 1 时使用 `torchrun` |
| `batch_size` | `128` | 每进程 batch size |
| `epochs` | `100` | 训练轮数 |

常用环境变量：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `TACTILE_KEYS` | 自动选择 | 触觉相机 key 列表 |
| `RAW_FRAME_CACHE` | 自动识别 | 强制选择裸 frame cache 或 LeRobot 模式 |
| `SENSOR_ID` | `-1` | AnyTouch sensor token |
| `MASK_RATIO` | `0.75` | MAE mask 比例 |
| `CONTACT_FILTER` | `1` | LeRobot 模式下启用接触帧筛选 |
| `CONTACT_STD_THRESHOLD` | `0.5` | 接触判定阈值 |
| `NONCONTACT_KEEP_RATIO` | `0.05` | 非接触帧保留比例 |
| `IMAGE_SIZE` | `224` | 输入和 frame cache 尺寸 |
| `NUM_WORKERS` | `12` | 数据 worker 数 |
| `RUN_NAME` | 当前时间 | 输出目录前缀 |

包含 `meta/info.json` 的目录按 LeRobot 数据集处理；否则按裸 frame cache 处理。两种数据不能在同一次训练中混用。

模型结构、初始化权重映射和底层训练参数见 [Tactile MAE README](../vtla/tac_encoder/tactile_mae/README.md)。

## 关节数据预处理

流程：原始视频去畸变、触觉 uint16 线性转 uint8 MP4、所有视觉/触觉 MP4 降到训练分辨率、通过 FK 生成 EE 列。视频 feature、触觉编码和单/双臂 joint 布局均从 `meta/info.json` 自动读取；无触觉数据集会自动跳过 uint8 转换。

```bash
bash scripts/process_joint_data.sh <dataset_id> [size] [horizon]
```

当前默认值：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `dataset_id` | `rm_umi_dual_260711_pen_in_case` | `playground/data/` 下的目录名 |
| `size` | `256` | 最终视频边长 |
| `horizon` | `32` | 相对 action 统计的最大步长 |

产物：

```text
<id>                       # 原始数据，不修改
<id>_undist                # 去畸变副本，后续阶段在此目录原地处理
<id>_undist_uint8          # 有触觉时的中间名；转换完成后改名
<id>_undist_uint8_<size>   # 有触觉的最终训练数据，包含 EE 列
<id>_undist_<size>         # 无触觉的最终训练数据，包含 EE 列
```

### 触觉处理流程

`process_joint_data.sh` 对含有 `tactile_u16_fixed_v1` feature 的数据集执行以下步骤：

```text
权威 uint16 FFV1/gbrp16le MKV
  -> 去畸变阶段原样复制触觉，视觉相机去畸变
  -> 固定范围线性量化为 uint8
  -> libx264rgb/gbrp/CRF0 MP4（无损 uint8 中间文件）
  -> 与视觉 MP4 一起 resize 到 size x size
  -> libx264/yuv420p/CRF0 MP4（最终训练和可视化文件）
```

第二步调用 `tools/tactile_uint16_to_uint8.py`。量化范围是项目标准的一部分，不使用
episode 或数据集统计量：

```python
# depth: U16 0..1000 -> U8 0..255
U8[..., 0] = rint(255 * clip(U16[..., 0], 0, 1000) / 1000)

# deformation: U16 30000 ± 1000 -> U8 0..255，静止中心为 128
delta = clip(U16[..., 1:]-30000, -1000, 1000)
U8[..., 1:] = where(
    delta < 0,
    128 - rint(128 * abs(delta) / 1000),
    128 + rint(127 * delta / 1000),
)
```

参考码值：

| 语义 | uint16 | uint8 |
| --- | ---: | ---: |
| depth 最小值 | 0 | 0 |
| depth 中点 | 500 | 128 |
| depth 最大值 | 1000 | 255 |
| deformation 下限 | 29000 | 0 |
| deformation 负半程 | 29500 | 64 |
| deformation 静止中心 | 30000 | 128 |
| deformation 正半程 | 30500 | 192 |
| deformation 上限 | 31000 | 255 |

超出范围的值会饱和到 `0` 或 `255`。`uint16 -> uint8` 本身是有量化和饱和的派生转换，
不能从单张 uint8 恢复原始 uint16；必须保留最初采集的数据集作为权威来源。

转换工具先用 `libx264rgb/gbrp/CRF0` 保存量化后的 RGB 语义通道，因此 resize 前的中间
MP4 可以逐像素还原量化后的 uint8。第三步调用 `tools/downscale_dataset_videos.py`，将所有
视觉和触觉 MP4 原地 resize；最终触觉使用 `libx264/yuv420p/CRF0`，可被 VSCode 等通用
播放器正常显示。由于 RGB/YUV 转换和 4:2:0 色度抽样，即使 CRF 为 0，最终解码值仍是
近似值，不再逐像素可逆。数值审计必须回到原始 uint16 MKV。

处理时同步更新 `meta/info.json` 和 `meta/tactile_encoding.json`：最终触觉 feature 为
`tactile_u8_linear_v1`、`storage_dtype=uint8`、`video.codec=h264`、
`video.pix_fmt=yuv420p`，清单记录 `encoder=libx264`、`crf=0`、`lossless=false` 和
`reversible=false`。如果输入没有触觉 feature，脚本自动跳过转换；如果输入已经是
`tactile_u8_linear_v1`，则不会重复量化。

权威采集编码见 [Deployment](../deployment/README.md#触觉采集编码)，完整固定编码标准见
[TACTILE_ENCODING_STANDARD.md](../deployment/TACTILE_ENCODING_STANDARD.md)。

## UMI 数据预处理

UMI 数据已经保存末端位姿，因此第三步不执行 FK，而是从已有 pose 生成统一的 EE 表示和统计。

```bash
bash scripts/process_umi_data.sh <dataset_id> [size] [horizon]
```

`dataset_id` 必填；`size` 默认 `256`，`horizon` 默认 `32`。

两个预处理脚本都支持：

- `CROP`：去畸变后的中心裁剪尺寸，默认 `896`。
- `JOBS`：并行 worker 数，默认 `12`。
- `CAMERAS`：覆盖处理的相机列表。
- `CALIB`：覆盖标定文件。

已存在的中间目录会被跳过。底层工具和处理顺序见 [tools/README.md](../tools/README.md)。

## State 均值

```bash
bash scripts/compute_mean_state.sh [dataset_id] [frames] [state_key]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `dataset_id` | 脚本内当前数据集 | `playground/data/` 下的目录名 |
| `frames` | `first` | `first` 取各 episode 首帧；`all` 取全部帧 |
| `state_key` | `observation.state` | 需要统计的列 |

对于关节 state，stdout 输出可以直接用于 `--robot.home_joints` 的参数字符串；统计表写到 stderr。

## 合并数据集

当前 `merge_datasets.sh` 是面向固定批次的工作脚本，`out_id` 和 `srcs` 数组直接定义在脚本中。修改这两个配置后运行：

```bash
bash scripts/merge_datasets.sh
```

脚本会取源数据集的公共 feature，删除不一致的额外相机，并输出单一训练数据集。如果输出目录已经存在，脚本会拒绝覆盖。

需要命令行参数化合并时，直接使用底层工具：

```bash
python tools/merge_datasets.py \
  --roots playground/data/A playground/data/B \
  --out playground/data/A_B_merged \
  --repo-id A_B_merged
```
