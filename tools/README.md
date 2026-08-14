# Offline Tools

`tools/` 保存可以单独运行的数据和模型准备工具。除特别说明外，命令均从仓库根目录执行。

## 工具索引

| 工具 | 用途 |
| --- | --- |
| `undistort_dataset_videos.py` | 腕部鱼眼视频去畸变和中心裁剪 |
| `downscale_dataset_videos.py` | 降低 RGB 视频分辨率，保留触觉无损视频 |
| `convert_joints_to_eepose.py` | 通过 FK 为关节数据集生成 EE 列 |
| `convert_umi_to_eepose.py` | 为 UMI pose 数据生成统一 EE 列 |
| `merge_datasets.py` | 对齐公共 feature 并合并 LeRobot 数据集 |
| `compute_dataset_mean_state.py` | 统计 state 均值和范围 |
| `generate_interpolated_dit.py` | 从本地 Wan2.2 Video DiT 生成插值 DiT 骨干 |
| `precompute_world_model_text_embeddings.py` | 为 FastWAM 数据集预计算文本 embedding |

## 推荐数据处理顺序

腕部相机使用鱼眼镜头时，应先在原始分辨率去畸变和裁剪，再降低分辨率：

```text
原始数据集
  -> undistort_dataset_videos.py
  -> downscale_dataset_videos.py
  -> convert_joints_to_eepose.py 或 convert_umi_to_eepose.py
```

不要先把全幅鱼眼图像缩到 `256x256` 再去畸变，否则会损失细节并改变中心裁剪视场。

日常使用可以直接运行封装后的 [scripts/process_joint_data.sh](../scripts/process_joint_data.sh) 或 [scripts/process_umi_data.sh](../scripts/process_umi_data.sh)。

## 鱼眼去畸变

该工具只重编码指定的腕部 RGB 相机。触觉 finger 视频和其他相机原样复制，同时更新目标数据集的图像 feature 信息。

快速视觉检查：

```bash
python tools/undistort_dataset_videos.py \
  --src playground/data/rm_umi_dual_pen_open \
  --test
```

完整处理：

```bash
python tools/undistort_dataset_videos.py \
  --src playground/data/rm_umi_dual_pen_open \
  --dst playground/data/rm_umi_dual_pen_open_undist \
  --crop 896 \
  --jobs 8 \
  --verify
```

默认标定文件位于：

```text
tools/calib/x5_left_intrinsics.json
tools/calib/x5_right_intrinsics.json
```

`--calib` 可以接收单个 JSON，也可以使用 `camera=path` 形式覆盖多路标定。常用参数还包括 `--cameras`、`--gop`、`--crf`、`--codec`、`--gpu-decode` 和 `--overwrite`。

工具需要 `ffmpeg`、`ffprobe` 和 OpenCV。

## 视频降分辨率

RGB 视频被重编码到指定正方形尺寸，帧数、fps 和时间戳保持不变。16-bit 无损触觉视频原样复制，避免量化损坏。

```bash
python tools/downscale_dataset_videos.py \
  --src playground/data/rm_umi_dual_pen_open_undist \
  --dst playground/data/rm_umi_dual_pen_open_undist_256 \
  --size 256 \
  --jobs 8 \
  --verify
```

默认使用 `libx264`、CRF 18 和 GOP 4。小 GOP 可以降低训练期间随机 seek 的成本。A100 等数据中心 GPU 通常没有 NVENC，保持 CPU 编码即可。

## 生成 EE Pose 列

关节数据通过睿尔曼 FK 生成 EE pose：

```bash
python tools/convert_joints_to_eepose.py \
  --root playground/data/rm_umi_dual_pen_open \
  --horizon 32
```

UMI 数据直接转换已有的末端位姿：

```bash
python tools/convert_umi_to_eepose.py \
  --root playground/data/rm_umi_dual_pen_open \
  --horizon 32
```

使用 `--root` 会原地修改数据集。需要保留源副本时使用：

```bash
python tools/convert_joints_to_eepose.py \
  --src playground/data/source \
  --dst playground/data/source_with_ee
```

两种工具都会生成 rot6d 和 quaternion 表示及其归一化统计。主要列包括：

| 列 | 维度 | 语义 |
| --- | --- | --- |
| `observation.state_episode_ee` | 20 | episode 相对 rot6d state |
| `observation.state_absolute_ee` | 20 | base 坐标系 rot6d state |
| `action_episode_ee` | 20 | episode 相对 rot6d action |
| `observation.state_episode_quat` | 16 | episode 相对 quaternion state |
| `observation.state_absolute_quat` | 16 | base 坐标系 quaternion state |
| `action_episode_quat` | 16 | episode 相对 quaternion action |
| `action_relative_ee` | 20 | rot6d 相对 action 的统计列 |
| `action_relative_quat` | 16 | quaternion 相对 action 的统计列 |

这些列对应的训练配置见 [vtla/README.md 的 EE 模式](../vtla/README.md#ee-模式)。

## 合并数据集

```bash
python tools/merge_datasets.py \
  --roots playground/data/A playground/data/B \
  --out playground/data/A_B_merged \
  --repo-id A_B_merged
```

工具先计算所有输入数据集的公共 feature，创建临时对齐副本，再合并为一个 LeRobot 数据集。默认拒绝覆盖已有输出。

固定批次工作流也可以通过 [scripts/merge_datasets.sh](../scripts/merge_datasets.sh) 执行。

## State 统计

按 episode 首帧统计 home pose 候选：

```bash
python tools/compute_dataset_mean_state.py \
  --root playground/data/<dataset_id> \
  --frames first
```

统计全部帧：

```bash
python tools/compute_dataset_mean_state.py \
  --root playground/data/<dataset_id> \
  --frames all \
  --state-key observation.state
```

## 生成插值 DiT

该工具只读取本地官方 Wan2.2 Video DiT 分片，不下载或处理 VAE、T5 和 tokenizer。

输入目录必须包含：

- 三个 `diffusion_pytorch_model-*-of-*.safetensors` 分片。
- 记录官方 repo 和 commit 的 `fastwam_source.json`。

```bash
python tools/generate_interpolated_dit.py \
  --wan-dir playground/pretrained_models/Wan2.2-TI2V-5B \
  --device cuda \
  --dtype bfloat16
```

默认输出：

```text
playground/pretrained_models/Wan2.2-TI2V-5B/interpolated_dit/
└── InterpolatedDiT_from_official_Wan2.2_alphascale_1024hdim.pt
```

已有输出需要重新生成时增加 `--overwrite`。默认启用 alpha scaling；`--no-alpha-scaling` 用于显式关闭。

## 预计算文本 Embedding

FastWAM 训练前，为数据集的任务文本生成 UMT5 embedding：

```bash
python tools/precompute_world_model_text_embeddings.py \
  --dataset-root playground/data/<dataset_id> \
  --world-model wan22 \
  --device cuda
```

输出位于：

```text
playground/data/<dataset_id>/text_embeddings/wan22/
```

`train.sh` 在 FastWAM 模式下会检查该缓存并在缺失时自动生成。任意任务文本推理仍需要加载文本 encoder。
