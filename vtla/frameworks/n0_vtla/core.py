"""N0-VTLA on StarVTLA's package-local PaliGemma/Gemma implementation.

No monkey patches, external source tree or OpenPI/JAX runtime are required.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from vtla.frameworks.pi05.modeling_pi05 import PI05Pytorch, make_att_2d_masks
from .tactile_encoder import FrozenDINOv2TactileEncoder
from .tactile_predictor import TactileActionPredictor


class N0VTLACore(PI05Pytorch):
    def __init__(self, config):
        super().__init__(config)
        # The local PiGemma wrapper replaces the language model after HF's
        # constructor. Restore native PaliGemma's tied embedding/head alias.
        paligemma = self.paligemma_with_expert.paligemma
        paligemma.lm_head.weight = paligemma.model.language_model.embed_tokens.weight
        # Match the native N0-VTLA path and the joint training attention. In
        # particular, do not let HF select SDPA for the cached prefix/expert.
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        width = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        if width % config.predictor_n_heads:
            raise ValueError("predictor_n_heads must divide the language hidden size.")
        self.tactile_encoder = FrozenDINOv2TactileEncoder(width, config.tactile_pool_grid, config.dinov2_config)
        self.tactile_predictor = TactileActionPredictor(
            width, config.n_latent, config.predictor_n_layers, config.predictor_n_heads, config.predictor_arch
        )
        self.z_proj = nn.Linear(width, self.action_in_proj.out_features)
        if config.z_gate_zero_init:
            self.z_gate = nn.Parameter(torch.zeros(1))
        if config.g_to_expert:
            self.g_proj = nn.Linear(width, self.action_in_proj.out_features)
            self.g_gate = nn.Parameter(torch.zeros(1))
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()

    def _prefix(self, images, image_masks, tokens, token_mask, *, cache):
        embeddings, pad, blocks = self.embed_prefix(images, image_masks, tokens, token_mask)
        embeddings = embeddings.to(self.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight.dtype)
        (context, _), kv = self.paligemma_with_expert(
            attention_mask=self._prepare_attention_masks_4d(make_att_2d_masks(pad, blocks)).to(embeddings.dtype),
            position_ids=pad.long().cumsum(1) - 1,
            inputs_embeds=[embeddings, None], use_cache=cache,
        )
        return context, embeddings, pad, blocks, kv

    def _tactile(self, context, prefix_pad, differences, view_masks):
        features = [self.tactile_encoder(diff) for diff in differences]
        g = torch.cat(features, dim=1)
        mask = torch.cat([m[:, None].expand(-1, f.shape[1]) for m, f in zip(view_masks, features)], dim=1)
        z = self.tactile_predictor(context.float(), g.float(), mask, prefix_pad)
        return self.z_proj(z), g, mask

    def _suffix(self, actions, time, z, g, g_mask):
        emb, pad, blocks, condition = self.embed_suffix(actions, time)
        z = z.to(emb.dtype)
        if hasattr(self, "z_gate"):
            z = z * self.z_gate
        cond = [z]
        cond_pad = [torch.ones(z.shape[:2], device=z.device, dtype=torch.bool)]
        if self.config.g_to_expert:
            cond.append(self.g_proj(g.float()).to(emb.dtype) * self.g_gate)
            cond_pad.append(g_mask)
        cond = torch.cat(cond, dim=1)
        cond_pad = torch.cat(cond_pad, dim=1)
        cond_blocks = torch.zeros_like(cond_pad, dtype=blocks.dtype)
        cond_blocks[:, 0] = 1
        return (torch.cat([cond, emb], 1), torch.cat([cond_pad, pad], 1),
                torch.cat([cond_blocks, blocks], 1), condition)

    def forward(self, images, image_masks, tokens, token_mask, differences, view_masks, actions,
                noise=None, time=None):
        # The original predictor consumes detached VL context. The second joint
        # pass still trains the VLM through the action expert's attention.
        with torch.no_grad():
            context, _, prefix_pad, _, _ = self._prefix(images, image_masks, tokens, token_mask, cache=False)
        z, g, g_mask = self._tactile(context.detach(), prefix_pad, differences, view_masks)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        x = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        prefix, prefix_pad, prefix_blocks = self.embed_prefix(images, image_masks, tokens, token_mask)
        suffix, suffix_pad, suffix_blocks, condition = self._suffix(x, time, z, g, g_mask)
        dtype = self.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight.dtype
        pad = torch.cat([prefix_pad, suffix_pad], 1)
        blocks = torch.cat([prefix_blocks, suffix_blocks], 1)
        (_, hidden), _ = self.paligemma_with_expert(
            attention_mask=self._prepare_attention_masks_4d(make_att_2d_masks(pad, blocks)).to(dtype),
            position_ids=pad.long().cumsum(1) - 1,
            inputs_embeds=[prefix.to(dtype), suffix.to(dtype)],
            use_cache=False, adarms_cond=[None, condition],
        )
        velocity = self.action_out_proj(hidden[:, -self.config.chunk_size:].float())
        return F.mse_loss(velocity, noise - actions, reduction="none")

    @torch.no_grad()
    def sample_actions(self, images, image_masks, tokens, token_mask, differences, view_masks, noise=None):
        context, _, prefix_pad, _, kv = self._prefix(images, image_masks, tokens, token_mask, cache=True)
        z, g, g_mask = self._tactile(context, prefix_pad, differences, view_masks)
        shape = (tokens.shape[0], self.config.chunk_size, self.config.max_action_dim)
        x = self.sample_noise(shape, tokens.device) if noise is None else noise
        if tuple(x.shape) != shape:
            raise ValueError(f"Noise shape must be {shape}, got {tuple(x.shape)}.")
        dt = -1.0 / self.config.num_inference_steps
        dtype = self.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight.dtype
        for step in range(self.config.num_inference_steps):
            time = torch.full((tokens.shape[0],), 1 + step * dt, device=tokens.device)
            suffix, pad, blocks, condition = self._suffix(x, time, z, g, g_mask)
            history_mask = prefix_pad[:, None, :].expand(-1, pad.shape[1], -1)
            mask = torch.cat([history_mask, make_att_2d_masks(pad, blocks)], dim=2)
            (_, hidden), _ = self.paligemma_with_expert(
                attention_mask=self._prepare_attention_masks_4d(mask).to(dtype),
                position_ids=prefix_pad.sum(1)[:, None] + pad.long().cumsum(1) - 1,
                past_key_values=copy.deepcopy(kv), inputs_embeds=[None, suffix.to(dtype)],
                use_cache=False, adarms_cond=[None, condition],
            )
            x = x + dt * self.action_out_proj(hidden[:, -self.config.chunk_size:].float())
        return x
