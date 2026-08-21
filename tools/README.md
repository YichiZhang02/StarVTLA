# Offline Tools

`tools/` 包含可独立调用的数据集和模型准备工具。命令均从仓库根目录运行；日常完整流程优先使用 [scripts/README.md](../scripts/README.md) 中的包装脚本。

## 工具索引

| 工具 | 用途 |
| --- | --- |
| `undistort_dataset_videos.py` | 腕部鱼眼去畸变和中心裁剪 |
| `tactile_uint16_to_uint8.py` | 权威 `uint16` 触觉转训练用 `uint8` |
| `downscale_dataset_videos.py` | 缩放视觉和训练用触觉视频 |
| `convert_joints_to_eepose.py` | 根据严格 `robot_type` 使用 FK 生成 EE 列 |
| `convert_umi_to_eepose.py` | 从已有 UMI pose 生成统一 EE 列 |
| `merge_datasets.py` | 对齐公共 feature 并合并 LeRobot 数据集 |
| `compute_dataset_mean_state.py` | 统计 state 和 home joint 候选 |
| `generate_interpolated_dit.py` | 从 Wan2.2 权重生成插值 DiT |
| `precompute_world_model_text_embeddings.py` | 预计算 FastWAM 文本 embedding |

## 推荐顺序

关节采集数据：

```text
undistort_dataset_videos.py
  -> tactile_uint16_to_uint8.py（有 raw 触觉时）
  -> downscale_dataset_videos.py
  -> convert_joints_to_eepose.py
```

UMI pose 数据将最后一步替换为 `convert_umi_to_eepose.py`。鱼眼视频必须先在原始分辨率去畸变和裁剪，再缩放；反向顺序会改变视场并损失细节。

## 鱼眼去畸变

```bash
python tools/undistort_dataset_videos.py \
  --src playground/data/<dataset_id> \
  --dst playground/data/<dataset_id>_undist \
  --crop 896 \
  --jobs 8 \
  --verify
```

默认只重编码腕部 RGB，相机以外的视频原样复制。标定文件为：

```text
tools/calib/x5_left_intrinsics.json
tools/calib/x5_right_intrinsics.json
```

使用 `--test` 可只生成快速视觉检查，不写目标数据集。`--calib`、`--cameras`、`--gop`、`--crf`、`--codec` 和 `--gpu-decode` 可覆盖默认处理参数。

## 触觉转换

原地转换：

```bash
python tools/tactile_uint16_to_uint8.py \
  --root playground/data/<dataset_id> \
  --jobs 4
```

保留源数据并创建副本：

```bash
python tools/tactile_uint16_to_uint8.py \
  --src playground/data/<dataset_id> \
  --dst playground/data/<dataset_id>_uint8
```

工具只接受项目定义的 raw 触觉编码，按固定范围生成 `tactile_u8_linear_v1`，并更新数据集元数据。量化公式和可逆性说明见 [Workflow Scripts](../scripts/README.md#触觉处理标准)。

## 视频缩放

原地模式：

```bash
python tools/downscale_dataset_videos.py \
  --root playground/data/<dataset_id> \
  --size 256 \
  --jobs 8 \
  --verify
```

副本模式：

```bash
python tools/downscale_dataset_videos.py \
  --src playground/data/<dataset_id> \
  --dst playground/data/<dataset_id>_256 \
  --size 256
```

默认 RGB 编码为 `libx264`、CRF 18、GOP 4。raw `uint16` 触觉不会被当作普通 RGB 缩放；应先按触觉标准生成训练用 `uint8`。

## Joint-to-EE

原地生成：

```bash
python tools/convert_joints_to_eepose.py \
  --root playground/data/<dataset_id> \
  --horizon 32
```

副本模式：

```bash
python tools/convert_joints_to_eepose.py \
  --src playground/data/<dataset_id> \
  --dst playground/data/<dataset_id>_ee \
  --horizon 32
```

FK 类型只从 `meta/info.json.robot_type` 读取。工具没有 `--robot-type` 参数，也不会从目录名、关节数或默认值猜测 B/ISF。它检测 feature 中完整的关节和夹爪并严格校验：

| `robot_type` | 允许的臂 | joint 输入维度 | rot6d EE | quaternion EE |
| --- | --- | ---: | ---: | ---: |
| `rm_base_umi_dual` | right + left | 16 | 20 | 16 |
| `rm_isf_umi_left` | left | 8 | 10 | 8 |

每臂布局：

```text
joint:  [joint1..joint7, gripper]                    8
rot6d:  [xyz, rotation_matrix_col0, col1, gripper] 10
quat:   [xyz, qx, qy, qz, qw, gripper]              8
```

双臂输出顺序为 right 后 left。工具保留原关节列，并增加：

| 列 | 语义 |
| --- | --- |
| `observation.state_episode_joint` | 相对 episode 首帧的 joint state |
| `observation.state_episode_ee` | 相对 episode 首帧的 rot6d EE state |
| `action_episode_ee` | episode 坐标系 rot6d action |
| `observation.state_absolute_ee` | robot base 坐标系 rot6d EE state |
| `action_absolute_ee` | robot base 坐标系 rot6d action |
| `observation.state_episode_quat` | 相对 episode 首帧的 quaternion EE state |
| `action_episode_quat` | episode 坐标系 quaternion action |
| `observation.state_absolute_quat` | robot base 坐标系 quaternion EE state |
| `action_absolute_quat` | robot base 坐标系 quaternion action |

相对 action 的统计也会写入 `meta/stats.json` 和 episode metadata。`horizon` 必须覆盖训练时可能使用的 action chunk 长度。

## UMI-to-EE

```bash
python tools/convert_umi_to_eepose.py \
  --root playground/data/<dataset_id> \
  --horizon 32
```

该工具读取已有末端位姿，不调用 RealMan FK。它生成与 joint-to-EE 相同的 rot6d/quaternion 数据契约，适用于真正的 UMI pose 数据，不用于当前 8/16 维 joint 采集数据。

## 合并数据集

```bash
python tools/merge_datasets.py \
  --roots playground/data/A playground/data/B \
  --out playground/data/A_B_merged \
  --repo-id A_B_merged
```

工具以 dtype 和 shape 为准取公共 feature，为有额外 feature 的输入创建临时对齐副本，再执行聚合。源数据不修改，已有输出不会被覆盖。聚合要求输入的 `fps` 和 `robot_type` 完全一致，因此不能混合 B/ISF 或单/双臂数据。

## State 统计

各 episode 首帧：

```bash
python tools/compute_dataset_mean_state.py \
  --root playground/data/<dataset_id> \
  --frames first
```

所有帧和指定 feature：

```bash
python tools/compute_dataset_mean_state.py \
  --root playground/data/<dataset_id> \
  --frames all \
  --state-key observation.state
```

对 joint state 使用 `first` 时，stdout 输出可作为 `--robot.home_joints`；详细均值、标准差和范围写入 stderr。

## FastWAM 资产

从本地 Wan2.2 Video DiT 生成插值骨干：

```bash
python tools/generate_interpolated_dit.py \
  --wan-dir playground/pretrained_models/Wan2.2-TI2V-5B \
  --device cuda \
  --dtype bfloat16
```

默认输出到输入目录的 `interpolated_dit/`。输入必须包含官方权重分片和记录来源的 `fastwam_source.json`；工具不下载 VAE、T5 或 tokenizer。

为数据集任务文本生成 Wan2.2 embedding：

```bash
python tools/precompute_world_model_text_embeddings.py \
  --dataset-root playground/data/<dataset_id> \
  --world-model wan22 \
  --device cuda
```

输出位于 `playground/data/<dataset_id>/text_embeddings/wan22/`。FastWAM 训练要求该目录中存在 manifest 和 safetensors 缓存。
