# Tactile Encoders

该模块提供统一的触觉 backbone 数据处理、预训练、checkpoint 和下游特征提取接口。

## 编码器

| `model_id` | 目录 | 训练目标 |
| --- | --- | --- |
| `anytouch1` | [`frameworks/anytouch1/`](frameworks/anytouch1/README.md) | masked image reconstruction |
| `anytouch2` | [`frameworks/anytouch2/`](frameworks/anytouch2/README.md) | pixel + temporal residual reconstruction |
| `sparsh_vjepa` | [`frameworks/sparsh_vjepa/`](frameworks/sparsh_vjepa/README.md) | masked latent prediction |
| `wan22_vae` | [`frameworks/wan22_vae/`](frameworks/wan22_vae/README.md) | full-frame VAE reconstruction + KL |

Sparsh V-JEPA 支持直接使用公开的 encoder-only checkpoint：context encoder 和 target
encoder 从该权重初始化，predictor 从随机权重开始训练。

Wan2.2 VAE 将窗口中的每张触觉图像作为独立单帧样本训练，不沿时间预测未来帧。训练时
保存完整 VAE，供 policy 使用的 encoder checkpoint 则不包含 decoder。

每个编码器目录包含自己的 `config.py`、`model.py` 和 `training.py`。AnyTouch1 的
MAE 内部实现位于 `frameworks/anytouch1/mae/`，AnyTouch2 的视频 patch 转换位于
`frameworks/anytouch2/patching.py`。

公共协议和工具位于 `common/`：

- `backbone.py`：统一特征与重建接口
- `training.py`：训练 recipe、optimizer 和 scheduler
- `checkpoint.py`：checkpoint 读取与位置编码插值
- `pooling.py`：下游 spatial pooling

`registry.py` 是唯一的模型注册入口，同时绑定 backbone、training recipe、checkpoint
前缀和 checkpoint 参数恢复逻辑。新增编码器时增加一个独立目录，并在
`ENCODER_REGISTRY` 中注册一个 `EncoderSpec`。

## 训练

从仓库根目录运行：

```bash
bash scripts/process_backbone_data.sh <dataset_id> [--num_workers 4] [--overwrite]
bash scripts/train_backbone.sh \
  <dataset_id> <model_id> [num_processes] [batch_size] [epochs] \
  [lr] [image_size] [num_frames] [frame_stride] [resume]
```

训练输出位于 `playground/results/backbones/`。详细参数、预训练权重路径和 cache 契约见
[scripts/README.md](../../scripts/README.md#触觉-backbone-训练)。

AnyTouch1 checkpoint 转换工具的新入口为：

```bash
python -m vtla.tac_encoder.frameworks.anytouch1.tools.convert_anytouch_to_hf \
  --src <checkpoint.pth> --out <output_dir> --arch vit_l
```

旧的 `models.*` 和 `training.*` import 路径目前由薄兼容层转发；新代码应直接使用
`vtla.tac_encoder.registry`、`vtla.tac_encoder.common` 或对应编码器目录。
