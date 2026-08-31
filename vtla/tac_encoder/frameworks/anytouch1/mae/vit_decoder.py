"""Self-contained ViT decoder block.

Reproduces the *old* HuggingFace ``transformers.models.vit.modeling_vit.ViTLayer``
(pre-LN, separate query/key/value, ``intermediate``/``output`` MLP) so that the
parameter names match exactly the AnyTouch stage1 checkpoint keys
``touch_decoder_blocks.{i}.*``.

We vendor it because newer transformers (>=5.x) renamed these submodules
(``attention.q_proj`` / ``mlp.fc1`` ...) and changed the forward return type,
which would break strict loading of the released AnyTouch weights.
"""
import math

import torch
from torch import nn


class ViTDecoderConfig:
    def __init__(self, hidden_size=512, intermediate_size=2048, num_attention_heads=16,
                 num_hidden_layers=8, patch_size=14, num_channels=3, layer_norm_eps=1e-12):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.layer_norm_eps = layer_norm_eps


class ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        probs = nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        return context


class ViTSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        return self.dense(hidden_states)


class ViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states):
        self_output = self.attention(hidden_states)
        return self.output(self_output)


class ViTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        return self.intermediate_act_fn(self.dense(hidden_states))


class ViTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states, input_tensor):
        return self.dense(hidden_states) + input_tensor


class ViTDecoderLayer(nn.Module):
    """Pre-LN transformer block, key-compatible with old HF ViTLayer."""

    def __init__(self, config):
        super().__init__()
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        attention_output = self.attention(self.layernorm_before(hidden_states))
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        return layer_output
