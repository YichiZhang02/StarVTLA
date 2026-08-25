# Tactile Encoder 重构规划

## 1. 背景与目标

当前 `vtla/tac_encoder/tactile_mae` 实现以 AnyTouch stage-1 MAE 为基础：训练时直接读取 LeRobot 数据或 frame cache，下游特征提取在 CLIP encoder 内插入 learned query tokens。此次重构的目标是建立一套统一、可扩展的触觉 backbone 训练和推理接口。

本次重构包含以下内容：

1. 下游特征提取不再使用 query-based pooling，改为固定 `AdaptiveAvgPool2d(3, 3)`。
2. 通过 `model_id` 支持 `anytouch1`、`anytouch2` 和 `sparsh_vjepa` 三种 backbone。
3. 三种模型只进行像素重建训练，共用一个 Python 训练入口和一个 Bash 启动脚本。
4. 保留当前训练中的重建评估和可视化，并扩展到多时间步、多传感器。
5. 只使用 processed dataset 中 `tactile_u8_linear_v1` 编码的 uint8 触觉数据。
6. 预处理阶段筛选接触时序窗口并保存为 `.npy`；训练阶段只读取 `.npy`，不再访问视频、parquet 或原始图片。
7. 默认输入分辨率为 `224 x 224`，默认时间窗口为 `T=4`、`frame_stride=2`。

非目标：

- 本阶段不训练策略模型，不引入对比学习、语言对齐或原始 V-JEPA loss。
- 本阶段不在 backbone 内融合不同物理传感器。
- 不要求三个 backbone 输出完全相同数量的 token，只要求统一输出协议并保留 token 的时间和传感器身份。

## 2. 核心设计原则

### 2.1 重建路径与下游特征路径分离

`3x3 pooling` 只能用于下游特征接口，不能放在 encoder 和 reconstruction decoder 之间。

```text
重建路径：
frames -> encoder full patch/tube tokens -> reconstruction decoder -> reconstructed frames

特征路径：
frames -> encoder full patch/tube tokens -> reshape spatial grid
       -> AdaptiveAvgPool2d(3, 3) -> downstream tactile tokens
```

重建需要完整的空间 token。如果先将 `16x16` 或 `14x14` patch grid 压缩到 `3x3`，decoder 将失去恢复局部细节所需的信息。

### 2.2 多传感器分别编码

数据集统一返回：

```text
[B, S, T, C, H, W]
```

其中：

- `B`：batch size；
- `S`：物理触觉传感器数量，当前支持 `1` 或 `2`；
- `T`：时间步数量，默认 `4`；
- `C,H,W`：单张触觉图，默认 `[3,224,224]`。

不要把多个传感器拼成 `224 x (224*S)`，也不要沿 channel 维拼接。像素级拼接会改变预训练模型的 patch grid 和位置编码，并人为建立跨传感器边界的空间邻接关系。

编码时将 sensor 维折叠到 batch 维：

```text
AnyTouch1:
[B,S,T,C,H,W] -> [B*S*T,C,H,W]

AnyTouch2 / Sparsh V-JEPA:
[B,S,T,C,H,W] -> [B*S,C,T,H,W]
```

encoder 输出后恢复 `B`、`S` 和时间维，加入 sensor/time embedding，最后再根据下游模型需要 flatten。跨传感器融合由策略模型或后续 fusion module 完成。

### 2.3 时间信息不在 pooling 时丢失

空间 pooling 对每个时间帧或 temporal tube 独立执行，不在时间维上做平均：

```text
[B,S,T',D,H',W']
    -> spatial pool only
[B,S,T',D,3,3]
    -> reshape
[B,S,T',9,D]
```

统一特征输出同时返回 `sensor_ids` 和 `time_ids`。这样后续模型可以区分同一时间的不同传感器，以及同一传感器的不同时间位置。

## 3. 目标目录结构

```text
vtla/tac_encoder/
├── REFACTOR_PLAN.md
├── config.py
├── train.py
├── eval.py
├── inference.py
├── process_backbone_data.py
├── data/
│   ├── __init__.py
│   ├── cache_schema.py
│   ├── contact.py
│   └── npy_tactile_dataset.py
└── models/
    ├── __init__.py
    ├── base.py
    ├── registry.py
    ├── pooling.py
    ├── decoder.py
    ├── anytouch1.py
    ├── anytouch2.py
    └── sparsh_vjepa.py

scripts/
├── process_backbone_data.sh
└── train_backbone.sh
```

旧的 `vtla/tac_encoder/tactile_mae` 在迁移期间保留，用于重建结果和 checkpoint 加载行为对齐。旧训练入口可暂时改成兼容 wrapper，待验收完成后再删除。

生产代码不直接 import `ref_repo` 中可执行的训练代码。需要的模型定义、checkpoint 转换和位置编码处理应整理到 StarVTLA 自己维护的 adapter 中。

## 4. 统一模型协议

### 4.1 基础接口

`models/base.py` 定义统一协议：

```python
class ReconstructionBackbone(nn.Module):
    model_id: str
    feature_dim: int
    patch_size: int
    tubelet_size: int

    def forward_reconstruction(
        self,
        images,       # [B, S, T, 3, 224, 224]
        mask_ratio,
    ):
        """Return loss, reconstruction and mask."""

    def encode_features(self, images):
        """Return full patch/tube features before downstream pooling."""

    def patchify(self, images):
        ...

    def unpatchify(self, patches):
        ...
```

统一 reconstruction 输出：

```text
loss:           scalar
reconstruction: [B,S,T,3,224,224]
mask:           与模型 patch/tube 布局对应的 bool/float tensor
```

统一 feature 输出使用结构化对象，而不是只返回一个无法解释的扁平 tensor：

```text
global_tokens
spatial_tokens
sensor_ids
time_ids
token_mask
```

### 4.2 Model registry

`models/registry.py` 根据 `model_id` 管理：

- adapter/build 函数；
- checkpoint loader；
- 输入 normalization；
- 图像或视频输入排列；
- patch/tubelet 配置；
- reconstruction decoder 配置；
- 位置编码插值策略；
- 可视化所需的 patchify/unpatchify adapter。

支持的 ID 固定为：

```text
anytouch1
anytouch2
sparsh_vjepa
```

Bash 不实现大段模型分支逻辑，只负责传入 `model_id`、路径和训练超参数。

## 5. 三种 backbone 的具体行为

| `model_id` | `T=4` 编码方式 | 重建 decoder | 时间编码 | `3x3` 输出/传感器 |
| --- | --- | --- | --- | --- |
| `anytouch1` | 四帧独立编码，折叠为 `B*S*T` | 尽量复用当前 MAE decoder | backbone 内无跨帧交互；输出后增加 time embedding | `4 x (1 global + 9 spatial) = 40` |
| `anytouch2` | 原生视频输入，tubelet size `2` | StarVTLA temporal MAE decoder | 原生 temporal tube/position encoding | `1 global + 2 x 9 spatial = 19` |
| `sparsh_vjepa` | 原生视频输入，tubelet size `2` | StarVTLA temporal MAE decoder | 原生 temporal tube/position encoding | `1 global + 2 x 9 spatial = 19` |

### 5.1 AnyTouch1

- 单帧 `224x224`、ViT-L/14 时 patch grid 为 `16x16`。
- 不继续使用当前“重复为三帧后走 Conv3d”的路径来表达真正的 `T=4`，避免只处理部分时间步或产生含混的 temporal semantics。
- 四帧分别执行完整的 2D MAE 编码和重建，确保所有四帧参与 loss。
- backbone 本身没有跨时间交互。下游输出通过显式 time embedding 标识时间顺序；如后续需要时序融合，应交给策略模型，而不是在本次重构中额外引入新 temporal encoder。

### 5.2 AnyTouch2

- 默认按 ViT-B/16、tubelet size `2` 处理。
- `224x224` 下空间 grid 为 `14x14`，`T=4` 得到 `T'=2` 个 temporal tubes。
- 复用 encoder 权重；如果参考发布不包含兼容的像素 decoder，则使用 StarVTLA 维护的 temporal MAE decoder。

### 5.3 Sparsh V-JEPA

- 默认按 ViT-B/16、tubelet size `2` 处理。
- `T=4`、`224x224` 时输出 `2 x 14 x 14` tube tokens。
- V-JEPA predictor 不是像素重建 decoder。本项目只复用 encoder 权重，并训练 StarVTLA temporal MAE decoder，不能将其描述为恢复原始 V-JEPA 训练目标。
- 参考 checkpoint 的预训练分辨率与 `224x224` 不一致时，加载阶段必须执行时空位置编码插值，并记录插值前后 grid。

### 5.4 `frame_stride=2` 与预训练权重

默认窗口为：

```text
T = 4
frame_stride = 2
relative frame offsets = [-6, -4, -2, 0]
```

`frame_stride` 只影响数据采样的物理时间间隔，不改变 checkpoint 参数形状，因此不会直接导致权重加载冲突。它可能与预训练数据的时间采样频率不同，所以必须写入缓存 metadata 和训练 config，保证实验可追溯。

## 6. `3x3` pooling 和 token 约定

`models/pooling.py` 提供无可学习参数的统一实现：

```python
SpatialPool3x3 = nn.AdaptiveAvgPool2d((3, 3))
```

处理顺序：

```text
patch tokens
-> 去除/分离 global、sensor 等非空间 token
-> 根据 adapter 提供的 T', H', W' reshape
-> 对每个 sensor、temporal unit 独立做 3x3 pooling
-> 得到 9 个 spatial tokens
-> 拼回 global token
```

默认 `T=4` 时：

| Backbone | `S=1` | `S=2` |
| --- | ---: | ---: |
| AnyTouch1 | 40 tokens | 80 tokens |
| AnyTouch2 | 19 tokens | 38 tokens |
| Sparsh V-JEPA | 19 tokens | 38 tokens |

AnyTouch1 的计数是每帧一个 global token；AnyTouch2/Sparsh 的计数是每个 sensor/window 一个 global token，加两个 temporal tubes 的空间 token。不要保留旧 feature extractor 的 learned query token。AnyTouch 内部原生 sensor token 如对 encoder 有用可以继续存在，但不直接作为公共下游输出。

## 7. `.npy` 数据缓存设计

### 7.1 输入约束

`process_backbone_data.py` 只接受当前 processed dataset，并强制校验触觉 feature：

```text
storage dtype:     uint8
tactile_encoding:  tactile_u8_linear_v1
channel layout:    HWC, [depth, deform_x, deform_y]
```

不满足约束时直接报错，不回退到 uint16、RGB 视频或其他 normalization 路径。

### 7.2 每个数据集的缓存文件

每个源数据集生成独立缓存：

```text
<cache_root>/<dataset_id>/
├── frames.npy           # [F,S,224,224,3], uint8
├── valid.npy            # [F,S], bool
├── timestamps.npy       # [F], float64
├── frame_index.npy      # [F], int64
├── episode_index.npy    # [F], int64
├── sensor_names.npy     # [S], unicode
├── contact_scores.npy   # [F,S], float32
├── contact_mask.npy     # [F,S], bool
├── windows.npy          # [N,T], int64 frame-row indices
├── window_anchor.npy    # [N], int64
├── split.npy            # [N], uint8; train/val/test ID
└── cache_version.npy    # scalar unicode/integer
```

训练所需的配置和 metadata 也保存为非 object dtype 的 `.npy`。预处理可以另外生成便于人工查看的 JSON 报告，但训练代码不得依赖该 JSON。

不保存完整的 `[N,S,T,H,W,C]` 窗口。滑动窗口之间会重复使用历史帧，完整物化会将图像数据最多重复约四次。`windows.npy` 只保存对 `frames.npy` 的行索引，训练时使用：

```python
frames = np.load(path, mmap_mode="r")
window = frames[windows[index]]
```

不同源数据集可以有不同的 `S` 和 sensor name；每个数据集保持独立 mmap，训练侧使用 dataset-local reader 和 `ConcatDataset` 组合。

### 7.3 时序窗口规则

默认参数：

```text
num_frames = 4
frame_stride = 2
image_size = 224
anchor_contact_policy = any
```

窗口 `[t-6,t-4,t-2,t]` 必须满足：

1. 所有帧属于同一个 episode。
2. `frame_index` 顺序严格递增且间隔精确为 `2`。
3. timestamp 间隔在数据集 FPS 对应的容差内。
4. 所有请求的传感器在对应帧有效，或由显式 `valid.npy` 标记并在训练时屏蔽。
5. anchor 帧 `t` 至少有一个传感器检测到接触。
6. 历史帧不要求有接触；必须保留接触发生前的动态信息。

默认 `anchor_contact_policy=any`：只要任一传感器在 anchor 时刻接触，就保留该时间窗口的所有传感器。可预留 `all` 和 `per_sensor` 配置，但不作为默认行为。

train/validation/test 按 episode 划分，不能按 frame 或 window 随机划分，防止相邻窗口泄漏到不同 split。

## 8. 接触检测

当前实现使用最大 per-channel spatial standard deviation 和固定阈值。该规则对 `tactile_u8_linear_v1` 的 deformation 通道可能将静止噪声误判为接触，因此新缓存需要版本化的 uint8 contact scorer。

已知 neutral value 约为：

```text
[depth, deform_x, deform_y] = [0, 128, 128]
```

建议候选能量：

```text
depth
abs(deform_x - 128)
abs(deform_y - 128)
```

然后使用高分位数或 top-k 像素进行 robust 聚合，避免单个噪声像素主导分数。具体 threshold 不在初版代码中写死未经验证的 magic number，而是：

1. 提供 `--report-only` 或 `--dry-run`。
2. 输出每个数据集、episode、sensor 的 score 分位数。
3. 输出不同候选阈值对应的接触帧数和有效窗口数。
4. 人工抽样查看低、中、高分数帧后确定默认阈值。
5. 将 scorer 版本、threshold、分位数和窗口统计写入缓存 metadata/报告。

## 9. 归一化与数据增强

缓存始终保存原始 uint8，不保存归一化后的 float tensor。训练加载顺序：

```text
uint8 mmap -> float [0,1] -> model-specific normalization -> backbone
```

初始 normalization 约定：

| Model | Normalization |
| --- | --- |
| AnyTouch1 | ImageNet mean/std，与现有 MAE 路径保持一致 |
| AnyTouch2 | mean `[0.48145466, 0.4578275, 0.40821073]`，std `[0.26862954, 0.26130258, 0.27577711]` |
| Sparsh V-JEPA | 按参考 checkpoint 配置；若参考预处理仅缩放，则保持 `[0,1]` |

processed uint8 已经表达 depth/deformation，不再执行 AnyTouch2 或 Sparsh 针对普通光学触觉 RGB 的 background subtraction。

同一个 temporal window 内的所有时间步必须使用相同几何变换参数。默认先关闭 flip 和 color jitter：deformation 通道具有方向语义，普通 RGB flip/color augmentation 可能破坏符号和物理含义。后续如增加增强，应为 uint8 语义通道设计专用变换，并同时作用于同一窗口内的所有帧。

## 10. 统一训练入口

Python 入口：

```bash
torchrun --nproc_per_node=<N> \
  -m vtla.tac_encoder.train \
  --model_id anytouch2 \
  --cache_root <cache_root> \
  --dataset_ids <dataset_a> <dataset_b> \
  --num_frames 4 \
  --frame_stride 2 \
  --image_size 224 \
  --mask_ratio 0.75 \
  --pool_size 3 \
  --output_dir <output_dir>
```

Bash 入口：

```bash
bash scripts/process_backbone_data.sh \
  <processed_dataset_ids> <cache_root>

bash scripts/train_backbone.sh \
  <model_id> <cache_root> <dataset_ids> \
  [num_processes] [batch_size] [epochs]
```

建议公共参数：

```text
--model_id {anytouch1,anytouch2,sparsh_vjepa}
--pretrained_path
--cache_root
--dataset_ids
--num_frames 4
--frame_stride 2
--image_size 224
--mask_ratio 0.75
--pool_size 3
--batch_size
--epochs
--eval_freq
--save_freq
--resume
--amp_dtype
```

训练只优化 reconstruction objective。checkpoint 至少保存：

- model/optimizer/scaler/scheduler state；
- `model_id` 和 adapter config；
- cache version、dataset IDs 和 contact scorer config；
- `num_frames`、`frame_stride`、image/patch/tubelet size；
- normalization config；
- checkpoint load 和位置编码插值报告。

## 11. 重建 loss 和 decoder

统一采用 masked pixel reconstruction：

```text
AnyTouch1 target:
[patch_h, patch_w, 3]

AnyTouch2/Sparsh target:
[tubelet=2, patch_h, patch_w, 3]
```

默认只计算 masked patches/tubes 的 MSE，`visible_loss_weight=0`。如需复现旧 AnyTouch1 实验，可显式配置 visible loss，但不能成为三个模型行为不一致的隐式默认值。

AnyTouch2 和 Sparsh 可以共用 decoder 结构，但各自的输入 projection、feature dim、patch size、position embedding 和 checkpoint namespace 由 adapter 管理。

## 12. 训练中评估与可视化

保留当前训练行为：

- 周期性计算 validation masked reconstruction MSE；
- 保存 best checkpoint；
- 每个评估周期输出 `recon_vis/recon_epochXXXX.png`；
- 固定选择 low/mid/high contact-strength 样本，使不同 epoch 可直接比较；
- 可视化列继续包含 `original | masked | reconstruction | error`。

新布局按 sensor 和 time 展开，例如：

```text
rows:    low / mid / high contact samples
groups:  sensor 0, sensor 1
within:  t-6, t-4, t-2, t
cols:    original | masked | reconstruction | error
```

可视化必须通过统一模型接口获得 reconstruction 和 mask，不能在 `eval.py` 中按 `model_id` 重复实现三套 patch 逻辑。

## 13. 预处理 CLI 规划

`process_backbone_data.py` 建议支持：

```text
--dataset_root
--dataset_ids
--cache_root
--tactile_keys
--image_size 224
--num_frames 4
--frame_stride 2
--contact_method
--contact_threshold
--anchor_contact_policy any
--val_ratio
--test_ratio
--split_seed
--num_workers
--report-only
--resume
--overwrite
```

处理阶段：

1. 读取并验证 processed dataset metadata。
2. 识别或校验 tactile keys，固定 sensor 顺序。
3. 按 episode 顺序解码 uint8 tactile frames，并 resize 到 `224x224`。
4. 写入临时 mmap `.npy`，完成后原子重命名为最终文件。
5. 计算 per-sensor contact score 和 mask。
6. 生成不跨 episode 的时序窗口索引。
7. 按 episode 生成 split。
8. 校验全部数组 shape、dtype、索引边界和窗口时间差。
9. 输出统计报告和少量 contact preview。

`--resume` 只能继续 schema、参数和源数据签名完全相同的缓存；不匹配时应拒绝继续，而不是静默混用。

## 14. 实施阶段

### Phase 1：缓存和数据契约

- 实现 cache schema、uint8 校验、contact report 和窗口生成。
- 实现 mmap-only dataset loader。
- 验证 episode、sensor、timestamp 和 split 不变量。

这是第一优先级。模型训练开始前必须先确认缓存没有跨 episode、错序或 contact 误筛选。

### Phase 2：AnyTouch1 迁移

- 将当前模型包装到统一 reconstruction 接口。
- 改为完整处理 `T=4` 的 frame-independent 路径。
- 使用新 `.npy` loader 训练。
- 对齐旧实现的单帧 reconstruction loss 和可视化。

### Phase 3：AnyTouch2

- 移植最小必要 encoder 定义。
- 实现 checkpoint loader 和 temporal MAE decoder。
- 验证 `T=4`、tubelet `2`、`14x14` token grid。

### Phase 4：Sparsh V-JEPA

- 移植 encoder adapter。
- 实现时空位置编码插值和加载报告。
- 复用 temporal reconstruction decoder 协议。

### Phase 5：统一特征提取

- 实现 `SpatialPool3x3`。
- 删除公共接口中的 learned query tokens。
- 输出结构化 sensor/time token metadata。
- 接入现有 policy tactile encoder 调用点。

### Phase 6：入口切换与清理

- 完成单卡和 DDP smoke test。
- 将旧入口改为带 deprecation warning 的 wrapper。
- 更新训练和 policy 文档。
- parity test 通过后删除不再使用的在线视频/cache 分支和 query pooling 实现。

## 15. 测试与验收标准

### 15.1 数据测试

- 只接受 `tactile_u8_linear_v1`、uint8、三通道输入。
- `windows.npy` 中所有行索引合法。
- 任一窗口都不跨 episode。
- `T=4, frame_stride=2` 精确对应 `[-6,-4,-2,0]`。
- anchor 接触策略正确，历史非接触帧得到保留。
- split 间 episode 不重叠。
- `S=1` 和 `S=2` 均可加载。
- 训练 loader 使用 `mmap_mode="r"`，测试中禁止访问 raw dataset/video/parquet。

### 15.2 模型测试

- 三个 `model_id` 都能构造、前向、反向和保存/恢复。
- reconstruction 输出均为 `[B,S,4,3,224,224]`。
- AnyTouch1 四个时间步都影响 loss 和梯度。
- AnyTouch2/Sparsh 输出 `T'=2`、空间 grid `14x14`。
- `3x3` pool 后 token 数符合 40/19 per sensor 的约定。
- sensor/time IDs 与 flatten 后 token 顺序一一对应。
- checkpoint loader 对允许的 missing/unexpected keys 有显式白名单，其他不匹配直接报错。
- Sparsh 位置编码插值后可以在 `224x224, T=4` 完成前向。

### 15.3 训练测试

- 每个模型在小缓存上完成一次单卡 overfit 测试。
- 每个模型完成 DDP one-step smoke test。
- AMP、resume 和 best-checkpoint 保存可用。
- validation loss 和 `recon_epochXXXX.png` 按配置周期生成。
- 固定 visualization indices 在不同 epoch 保持不变。

## 16. 风险与待确认项

1. **contact threshold**：必须先用真实 processed uint8 数据生成分布和 preview，再确定默认值。
2. **checkpoint 结构**：AnyTouch2 和 Sparsh 的实际发布 checkpoint 需要逐项核对 encoder key、position embedding 和 tubelet 配置。
3. **decoder 初始化**：AnyTouch2/Sparsh 的像素 decoder 是新模块，训练初期可能需要 encoder warmup/freeze 策略；该策略应作为显式配置。
4. **global token 定义**：模型没有原生 CLS 时，统一使用所有有效 patch/tube token 的 mean pooling，不能额外引入 learned query。
5. **传感器缺帧**：需要决定严格丢弃窗口还是通过 `valid/token mask` 支持缺失。初版建议严格丢弃不完整窗口，降低训练行为复杂度。
6. **增强语义**：depth/deformation 通道不是自然 RGB，任何 flip、颜色或归一化变更都需要单独验证物理语义。

## 17. 完成定义

满足以下条件后视为重构完成：

1. 一个预处理 Python 入口和一个 Bash 脚本能从 processed uint8 dataset 生成版本化 `.npy` 缓存。
2. 一个训练 Python 入口和一个 Bash 脚本能按 `model_id` 训练三种 backbone。
3. 训练过程不读取原始 LeRobot 数据、视频、parquet 或图片。
4. 三个模型都重建完整的四个时间步，并保留现有周期性可视化。
5. 下游特征接口完全移除 query-based pooling，改为逐 sensor、逐 temporal unit 的 `3x3` spatial pooling。
6. `S=1/2`、`T=4`、`frame_stride=2` 的数据与 token 语义均有自动化测试覆盖。
