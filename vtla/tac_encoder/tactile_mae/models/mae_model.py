"""Tactile MAE model.

Structurally identical to AnyTouch stage1's ``TactileVideoMAE`` restricted to the
image path (``data_type == 0``). Only tactile images are supported; the video
patch-embedding / video prediction head are kept as (mostly inert) parameters so
that the released AnyTouch checkpoint can be loaded *strictly*.

Encoder  : CLIP ViT (assembled from transformers CLIPVisionEmbeddings + CLIPEncoder,
           whose state-dict keys match ``touch_model.*`` in the AnyTouch ckpt).
Decoder  : 8-layer ViT (vendored, keys match ``touch_decoder_blocks.*``).
Objective: masked-patch MSE reconstruction (MAE).
"""
import torch
import numpy as np
from torch import nn

from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings, CLIPEncoder

from .vit_decoder import ViTDecoderConfig, ViTDecoderLayer
from .pos_embed import get_2d_sincos_pos_embed


class _CLIPVisionCore(nn.Module):
    """Container mirroring HF CLIPVisionTransformer submodule names.

    Holds ``embeddings`` (only its submodules are used; the masking forward lives
    in TactileMAE), ``pre_layrnorm``, ``encoder`` and ``post_layernorm``. The
    resulting state-dict keys are exactly ``touch_model.*`` of the AnyTouch ckpt.
    """

    def __init__(self, vision_config):
        super().__init__()
        embed_dim = vision_config.hidden_size
        self.embeddings = CLIPVisionEmbeddings(vision_config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=vision_config.layer_norm_eps)
        self.encoder = CLIPEncoder(vision_config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=vision_config.layer_norm_eps)


class TactileMAE(nn.Module):
    def __init__(self, vision_config, decoder_config: ViTDecoderConfig,
                 mask_ratio=0.75, use_sensor_token=True, use_same_patchemb=True,
                 norm_pix_loss=False, visible_loss_weight=0.0):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.use_sensor_token = use_sensor_token
        self.use_same_patchemb = use_same_patchemb
        self.norm_pix_loss = norm_pix_loss
        # 0.0 = standard MAE (masked patches only); >0 also supervises visible
        # patches: loss = loss_masked + visible_loss_weight * loss_visible
        self.visible_loss_weight = visible_loss_weight

        embed_dim = vision_config.hidden_size
        proj_dim = vision_config.projection_dim
        dec_dim = decoder_config.hidden_size

        # ---- encoder ----
        self.touch_model = _CLIPVisionCore(vision_config)
        self.touch_projection = nn.Linear(embed_dim, proj_dim, bias=False)

        self.num_patches = self.touch_model.embeddings.num_patches
        self.patch_size = vision_config.patch_size

        # video patch-embedding (used for image path when use_same_patchemb=True,
        # by repeating the single frame 3x; matches AnyTouch stage1). Kept for
        # strict checkpoint compatibility.
        self.video_patch_embedding = nn.Conv3d(
            in_channels=vision_config.num_channels,
            out_channels=embed_dim,
            kernel_size=(3, self.patch_size, self.patch_size),
            stride=(3, self.patch_size, self.patch_size),
            bias=False,
        )
        self.video_position_embedding = nn.Embedding(self.num_patches + 1, embed_dim)

        # ---- decoder ----
        self.decoder_embed = nn.Linear(proj_dim, dec_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, dec_dim), requires_grad=False)
        self.touch_decoder_blocks = nn.ModuleList(
            [ViTDecoderLayer(decoder_config) for _ in range(decoder_config.num_hidden_layers)])
        self.decoder_norm = nn.LayerNorm(dec_dim, eps=decoder_config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            dec_dim, decoder_config.patch_size ** 2 * decoder_config.num_channels, bias=True)
        # inert (video) prediction head, kept for strict ckpt compatibility
        self.decoder_pred_video = nn.Linear(
            dec_dim, decoder_config.patch_size ** 2 * decoder_config.num_channels * 4, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        if self.use_sensor_token:
            self.sensor_token = nn.Parameter(torch.zeros(10, 5, embed_dim))

        self.n_prefix = 6 if self.use_sensor_token else 1  # cls (+ 5 sensor tokens)

    # ------------------------------------------------------------------ init
    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)
        if self.use_sensor_token:
            torch.nn.init.normal_(self.sensor_token, std=.02)

    # --------------------------------------------------------------- masking
    def random_masking(self, sequence, noise=None):
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.mask_ratio))
        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return sequence_unmasked, mask, ids_restore

    # -------------------------------------------------------------- encoder
    def embed(self, pixel_values, sensor_type, noise=None):
        batch_size = pixel_values.shape[0]
        target_dtype = self.video_patch_embedding.weight.dtype
        if self.use_same_patchemb:
            xv = pixel_values.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            patch_embeds = self.video_patch_embedding(xv.to(dtype=target_dtype))
        else:
            patch_embeds = self.touch_model.embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # (B, N, embed)

        pos_emb = self.touch_model.embeddings.position_embedding.weight.unsqueeze(0)  # (1, N+1, embed)
        embeddings = patch_embeds + pos_emb[:, 1:, :]
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        class_embeds = self.touch_model.embeddings.class_embedding + pos_emb[:, 0, :]
        class_embeds = class_embeds.expand(batch_size, 1, -1)

        if self.use_sensor_token:
            sensor_emb = self.sensor_token[sensor_type]  # (B, 5, embed)
            embeddings = torch.cat([class_embeds, sensor_emb, embeddings], dim=1)
        else:
            embeddings = torch.cat([class_embeds, embeddings], dim=1)
        return embeddings, mask, ids_restore

    def forward_encoder(self, x, sensor_type=None, noise=None):
        embeddings, mask, ids_restore = self.embed(x, sensor_type, noise)
        hidden = self.touch_model.pre_layrnorm(embeddings)
        enc_out = self.touch_model.encoder(inputs_embeds=hidden)
        last_hidden = enc_out.last_hidden_state
        out = self.touch_projection(last_hidden)
        return out, mask, ids_restore

    # -------------------------------------------------------------- decoder
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        p = self.n_prefix
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + p - x.shape[1], 1)
        x_ = torch.cat([x[:, p:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        if self.use_sensor_token:
            x = torch.cat([x[:, :6, :], x_], dim=1)
            x[:, 0, :] = x[:, 0, :] + self.decoder_pos_embed[:, 0, :]
            x[:, 6:, :] = x[:, 6:, :] + self.decoder_pos_embed[:, 1:, :]
        else:
            x = torch.cat([x[:, :1, :], x_], dim=1)
            x = x + self.decoder_pos_embed

        for blk in self.touch_decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, self.n_prefix:, :]
        return x

    # ----------------------------------------------------------------- loss
    def patchify(self, imgs):
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], 3, h * p, h * p))

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L] per-patch MSE
        masked_loss = (loss * mask).sum() / mask.sum()
        if self.visible_loss_weight > 0:
            visible = 1 - mask
            visible_loss = (loss * visible).sum() / visible.sum().clamp(min=1)
            return masked_loss + self.visible_loss_weight * visible_loss
        return masked_loss

    def forward(self, x, sensor_type=None, noise=None):
        latent, mask, ids_restore = self.forward_encoder(x, sensor_type=sensor_type, noise=noise)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

    # ------------------------------------------------------- feature extract
    @torch.no_grad()
    def extract_features(self, x, sensor_type=None):
        """Return the (unmasked) CLS feature for downstream / t-SNE use."""
        prev_ratio = self.mask_ratio
        self.mask_ratio = 0.0
        embeddings, _, _ = self.embed(x, sensor_type, noise=None)
        hidden = self.touch_model.pre_layrnorm(embeddings)
        enc_out = self.touch_model.encoder(inputs_embeds=hidden)
        cls = enc_out.last_hidden_state[:, 0, :]
        cls = self.touch_model.post_layernorm(cls)
        self.mask_ratio = prev_ratio
        return cls
