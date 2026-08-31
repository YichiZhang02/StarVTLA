# pi0.5

`pi05` 是 StarVTLA 的 pi0.5 PyTorch policy：PaliGemma 视觉语言 prefix 负责多模态条件编码，
Gemma action expert 通过 flow matching 生成 action chunk。state 和 action 会补齐到固定维度，
真实维数仍由数据集 feature schema 决定。

共享的数据集、`robot_type`、state/action 表示和相机路由约定见
[VTLA Training](../../README.md)。

## 最小环境

当前仓库验证基线为 Python 3.10.19、PyTorch 2.7.1+cu128、torchvision
0.22.1+cu128 和 CUDA 12.8。pi0.5 依赖 `transformers==5.5.0` 与
`safetensors==0.7.0`；这里的实现使用 Transformers 5.x 的 Gemma/PaliGemma 接口，不建议单独
降级 Transformers。

```bash
conda create -n starvtla-pi05 python=3.10 -y
conda activate starvtla-pi05
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

安装检查：

```bash
python -c "import torch, transformers, safetensors; from vtla.frameworks.pi05.modeling_pi05 import PI05Policy; print(torch.__version__, transformers.__version__)"
```

`bfloat16` 训练需要支持 BF16 的 GPU。显存不足时优先减小 batch size 或启用
`--policy.gradient_checkpointing=true`；CPU 只适合导入检查。

## 预训练资产

`train.sh` 默认读取以下本地目录，不应在训练时临时访问 Hugging Face：

```text
playground/pretrained_models/pi05_base/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
├── policy_postprocessor.json
└── paligemma-3b-pt-224-tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

可通过 `PRETRAINED_PATH=/path/to/pi05_base` 覆盖底座。EE state/action 会触发 processor 重建，
此时本地 tokenizer 目录尤其重要；缺失时默认 tokenizer 名称可能触发网络访问。

## 结构与默认值

```text
images + task text -> PaliGemma prefix --+
padded state ----------------------------+-> Gemma action expert -> flow matching -> action chunk
```

| 配置 | 默认值 |
| --- | ---: |
| `image_resolution` | `224x224` |
| `chunk_size` | `32` |
| `n_action_steps` | `16` |
| `max_state_dim` / `max_action_dim` | `32` / `32` |
| `num_inference_steps` | `10` |
| `dtype` | `bfloat16` |

## 训练

```bash
bash train.sh <dataset_id> pi05 1 16 20000 \
  false none absolute_joint absolute_joint 6 strong
```

EE action 示例：

```bash
bash train.sh <processed_dataset_id> pi05 1 16 20000 \
  false none absolute_rot6d relative_rot6d 6 strong
```

支持 `tactile_mode=as_image` 和 `encode`。`encode` 模式会把触觉 token 加到 VLM prefix：

```bash
TACTILE_ENCODER_PATH=<backbone_checkpoint> \
  bash train.sh <dataset_id> pi05 1 8 20000 \
  true encode absolute_joint absolute_joint 6 strong
```

`train_expert_only=true` 会冻结整个 PaliGemma prefix，只训练 action expert 和投影层；
`freeze_vision_encoder=true` 只冻结视觉编码器。

## 推理

```bash
bash inference.sh <run_id> <step> async <robot_type> 16 0
```

checkpoint 自包含模型、processor、tokenizer 引用、归一化统计和动作表示。部署应直接加载训练输出
的 `pretrained_model/`，不要再指向 `pi05_base`。
