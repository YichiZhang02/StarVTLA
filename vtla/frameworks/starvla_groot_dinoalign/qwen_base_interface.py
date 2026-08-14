# Copyright 2025 starVLA community. All rights reserved.
# Ported into vtla from starVLA/model/modules/vlm/QWen3.py
#
# A thin, framework-agnostic wrapper around a HuggingFace image-text-to-text VLM
# (Qwen2.5-VL / Qwen3-VL). It centralizes:
#   - loading the backbone + processor
#   - building chat-template inputs from (images, instructions)
#   - exposing the language hidden size for action-head alignment
#
# Unlike the starVLA original it does not depend on OmegaConf configs, the
# overwatch logger, the fast-token vocabulary, or CoT prompt templating.

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen2-VL smart_resize: round H/W to a multiple of ``factor`` within [min,max] pixels.

    Mirrors ``transformers...qwen2_vl.image_processing_qwen2_vl.smart_resize`` so we can
    reproduce the processor's target resolution on GPU without importing it.
    """
    import math

    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class QwenVLInterface(nn.Module):
    """Wrapper around a Qwen-VL backbone used as the VLA prefix encoder.

    Image preprocessing (resize / normalize / patchify -> pixel_values + image_grid_thw)
    is reproduced on GPU from the batch tensors, and the text input_ids are built by
    expanding the ``<|image_pad|>`` placeholder by the (fixed) per-image token count.
    This removes the per-step tensor->CPU->PIL->HF-processor round-trip that otherwise
    serializes with and starves the GPU. Both paths are validated to match the reference
    ``AutoProcessor`` output exactly (pixel_values allclose; input_ids identical).
    """

    def __init__(
        self,
        base_vlm: str,
        attn_implementation: str = "sdpa",
        dtype: torch.dtype = torch.bfloat16,
        image_resolution: tuple[int, int] = (224, 224),
    ):
        super().__init__()

        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Fall back to sdpa if flash_attention_2 is requested but unavailable.
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                logger.warning("flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = AutoModelForImageTextToText.from_pretrained(
            base_vlm,
            attn_implementation=attn_implementation,
            dtype=dtype,
            ignore_mismatched_sizes=True,
        )
        processor = AutoProcessor.from_pretrained(base_vlm)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor

        # Align the hidden_size accessor across Qwen2.5-VL / Qwen3-VL configs.
        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            self.model.config.hidden_size = text_config.hidden_size

        self._init_gpu_preproc(image_resolution)

    # ------------------------------------------------------------------
    # GPU image preprocessing (replaces the per-step PIL + HF-processor path)
    # ------------------------------------------------------------------
    def _init_gpu_preproc(self, image_resolution: tuple[int, int]) -> None:
        """Cache the image-processor params and the fixed resize target / grid.

        The resize target is the processor's ``smart_resize`` of ``image_resolution``;
        for a square config (e.g. 224) under min_pixels this is a constant (e.g. 256),
        so ``image_grid_thw`` is identical for every image and tokens-per-image is fixed.
        """
        ip = self.processor.image_processor
        self._patch_size = int(ip.patch_size)
        self._merge_size = int(ip.merge_size)
        self._temporal_patch_size = int(ip.temporal_patch_size)
        self._img_mean = [float(m) for m in ip.image_mean]
        self._img_std = [float(s) for s in ip.image_std]

        factor = self._patch_size * self._merge_size
        size = getattr(ip, "size", None) or {}
        min_pixels = getattr(ip, "min_pixels", None) or size.get("shortest_edge", factor * factor)
        max_pixels = getattr(ip, "max_pixels", None) or size.get("longest_edge", 16777216)

        h, w = int(image_resolution[0]), int(image_resolution[1])
        self._target_h, self._target_w = _smart_resize(h, w, factor, int(min_pixels), int(max_pixels))
        self._grid_h = self._target_h // self._patch_size
        self._grid_w = self._target_w // self._patch_size
        # tokens consumed in the LLM sequence per image (after the spatial 2x2 merge)
        self._tokens_per_image = (self._grid_h * self._grid_w) // (self._merge_size**2)

        self._image_pad_token = self.processor.image_token  # e.g. "<|image_pad|>"

    def gpu_pixel_values(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Patchify a stack of images into Qwen ``(pixel_values, image_grid_thw)`` on GPU.

        Args:
            images: ``[N, C, H, W]`` float in [0, 1] (LeRobot frame convention), already
                ordered sample-major then view (to match the image-pad order in input_ids).
        Returns:
            pixel_values ``[N * grid_h * grid_w, C * temporal_patch_size * patch^2]`` and
            image_grid_thw ``[N, 3]`` (each ``[1, grid_h, grid_w]``).
        """
        n, c, h, w = images.shape
        p, m, tp = self._patch_size, self._merge_size, self._temporal_patch_size
        if (h, w) != (self._target_h, self._target_w):
            images = F.interpolate(
                images, size=(self._target_h, self._target_w), mode="bilinear",
                align_corners=False, antialias=True,
            )
            h, w = self._target_h, self._target_w

        mean = torch.tensor(self._img_mean, device=images.device, dtype=images.dtype).view(1, c, 1, 1)
        std = torch.tensor(self._img_std, device=images.device, dtype=images.dtype).view(1, c, 1, 1)
        x = (images - mean) / std

        gh, gw = h // p, w // p
        # Match Qwen2VLImageProcessor patch layout (validated allclose to the reference).
        x = x.unsqueeze(1).repeat(1, tp, 1, 1, 1)  # [N, tp, C, H, W] (single frame -> temporal)
        x = x.reshape(n, 1, tp, c, gh // m, m, p, gw // m, m, p)
        x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)  # N, grid_t, gh', gw', m, m, C, tp, p, p
        pixel_values = x.reshape(n * gh * gw, c * tp * p * p)

        image_grid_thw = torch.tensor([[1, gh, gw]] * n, dtype=torch.long, device=images.device)
        return pixel_values, image_grid_thw

    def build_text_inputs(self, instructions: List[str], num_views: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(input_ids, attention_mask)`` with image placeholders expanded.

        Builds the chat template *without* images then replaces each ``<|image_pad|>`` with
        ``tokens_per_image`` copies before tokenizing. Validated to match the reference
        processor's ``input_ids`` exactly. Text-only -> no per-step image processing.
        """
        pad = self._image_pad_token
        expanded = pad * self._tokens_per_image
        image_block = [{"type": "image"} for _ in range(num_views)]
        messages = [
            [{"role": "user", "content": image_block + [{"type": "text", "text": instr}]}]
            for instr in instructions
        ]
        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts = [t.replace(pad, expanded) for t in texts]
        enc = self.processor.tokenizer(texts, return_tensors="pt", padding=True)
        return enc["input_ids"].to(self.model.device), enc["attention_mask"].to(self.model.device)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**kwargs)

    def supports_prefix_injection(self) -> bool:
        """Whether the backbone exposes the Qwen3.5-style API needed to inject extra
        prefix tokens into the LLM input embeddings."""
        core = getattr(self.model, "model", None)
        return core is not None and all(
            hasattr(core, m)
            for m in ("get_rope_index", "get_image_features", "get_placeholder_mask",
                      "get_input_embeddings")
        )

    def forward_with_prefix_tokens(self, images: List[list], instructions: List[str], extra_embeds):
        """Run the VLM with ``extra_embeds`` injected into the LLM input sequence.

        ``extra_embeds`` (``[B, N, H]``) are placed at the end of the prompt as ``N``
        extra tokens that go through *all* Qwen-VL transformer layers (genuine
        encoder-side fusion), as opposed to being appended to the VLM output. Image
        features are merged by the model's own helpers and the extra tokens receive
        proper (text-continuation) M-RoPE positions.

        Returns ``(last_hidden_state, attention_mask)`` over the full
        ``image + language + extra`` sequence.
        """
        if not self.supports_prefix_injection():
            raise RuntimeError(
                "This VLM backbone does not expose the Qwen3.5-style API "
                "(get_rope_index / get_image_features / get_placeholder_mask) required for "
                "tactile_insert_location='encoder'. Use 'decoder' for this backbone."
            )

        core = self.model.model  # Qwen3_5Model
        inputs = self.build_qwenvl_inputs(images=images, instructions=instructions)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        device = input_ids.device
        bsize, seq_len = input_ids.shape
        n_extra = extra_embeds.shape[1]

        image_token_id = self.model.config.image_token_id
        video_token_id = getattr(self.model.config, "video_token_id", -1)

        # --- extend the token sequence with N filler *text* tokens at the end ---
        pad_id = self.processor.tokenizer.pad_token_id or 0
        filler = torch.full((bsize, n_extra), pad_id, dtype=input_ids.dtype, device=device)
        input_ids_ext = torch.cat([input_ids, filler], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attn_ext = torch.cat(
            [attention_mask, torch.ones(bsize, n_extra, dtype=attention_mask.dtype, device=device)], dim=1
        )

        # mm_token_type_ids: image=1, video=2, text=0 (filler tokens are text).
        mm_tt = torch.zeros_like(input_ids_ext)
        mm_tt[input_ids_ext == image_token_id] = 1
        if video_token_id >= 0:
            mm_tt[input_ids_ext == video_token_id] = 2

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # M-RoPE positions for the extended sequence (filler -> text positions).
            position_ids = core.get_rope_index(
                input_ids_ext,
                mm_tt,
                image_grid_thw=image_grid_thw,
                attention_mask=attn_ext,
            )[0]

            # Merge text + image embeddings exactly as the model would, then override
            # the filler slots with the tactile embeddings.
            inputs_embeds = core.get_input_embeddings()(input_ids_ext)
            if pixel_values is not None:
                image_out = core.get_image_features(pixel_values, image_grid_thw, return_dict=True)
                image_embeds = torch.cat(image_out.pooler_output, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                image_mask, _ = core.get_placeholder_mask(
                    input_ids_ext, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[:, seq_len:, :] = extra_embeds.to(inputs_embeds.dtype)

            outputs = core(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_ext,
                position_ids=position_ids,
                use_cache=False,
            )

        last_hidden = outputs.last_hidden_state
        return last_hidden, attn_ext.to(dtype=torch.bool)

    def build_qwenvl_inputs(self, images: torch.Tensor, instructions: List[str], **kwargs) -> dict:
        """Build batched model inputs from multi-view image tensors + instructions, on GPU.

        Args:
            images: ``[B, V, C, H, W]`` float in [0, 1] (V multi-view cameras per sample).
            instructions: list of B instruction strings.
        Returns:
            dict(input_ids, attention_mask, pixel_values, image_grid_thw) on the model device.

        Replaces the old PIL + ``apply_chat_template(tokenize=True, images=...)`` path: image
        tensors are patchified on GPU and input_ids are built text-only with the image-pad
        placeholder expanded by the fixed per-image token count.
        """
        if images.dim() != 5:
            raise ValueError(f"Expected images of shape [B, V, C, H, W], got {tuple(images.shape)}")
        bsize, num_views = images.shape[0], images.shape[1]
        assert bsize == len(instructions), "images batch and instructions length must match"

        # Flatten sample-major then view -> [B*V, C, H, W] so pixel_values rows align with
        # the per-sample, in-view-order image-pad placeholders in input_ids.
        flat = images.reshape(bsize * num_views, *images.shape[2:])
        pixel_values, image_grid_thw = self.gpu_pixel_values(flat)
        input_ids, attention_mask = self.build_text_inputs(instructions, num_views)

        # Qwen3.5 forward requires mm_token_type_ids (image=1, video=2, text=0) whenever
        # image_grid_thw is supplied, so M-RoPE can be computed. The processor normally
        # returns it; reproduce it here (no video tokens -> only image=1). Matches the
        # processor output exactly (mm==1 iff input_ids == image_pad).
        image_token_id = self.model.config.image_token_id
        mm_token_type_ids = (input_ids == image_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values.to(self.model.device),
            "image_grid_thw": image_grid_thw.to(self.model.device),
        }
