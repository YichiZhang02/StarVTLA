"""Qwen interface that exposes post-merger image tokens without a second vision pass."""

from __future__ import annotations

from contextlib import nullcontext
from typing import List

import torch

from .qwen_base_interface import QwenVLInterface


class DinoAlignQwenVLInterface(QwenVLInterface):
    """Qwen3.5 interface returning both VLM states and 64-token image features."""

    def forward_with_image_features(
        self,
        images: torch.Tensor,
        instructions: List[str],
        extra_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.supports_prefix_injection():
            raise RuntimeError(
                "starvla_groot_dinoalign requires the Qwen3.5-style get_image_features / "
                "get_placeholder_mask / get_rope_index API."
            )

        core = self.model.model
        inputs = self.build_qwenvl_inputs(images=images, instructions=instructions)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        device = input_ids.device
        base_seq_len = input_ids.shape[1]
        n_extra = 0 if extra_embeds is None else extra_embeds.shape[1]
        if n_extra:
            pad_id = self.processor.tokenizer.pad_token_id or 0
            filler = torch.full(
                (input_ids.shape[0], n_extra), pad_id, dtype=input_ids.dtype, device=device
            )
            input_ids = torch.cat((input_ids, filler), dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids[:, :base_seq_len])
        if n_extra:
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        input_ids.shape[0],
                        n_extra,
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ),
                dim=1,
            )

        image_token_id = self.model.config.image_token_id
        video_token_id = getattr(self.model.config, "video_token_id", -1)
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == image_token_id] = 1
        if video_token_id >= 0:
            mm_token_type_ids[input_ids == video_token_id] = 2

        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            position_ids = core.get_rope_index(
                input_ids,
                mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )[0]
            inputs_embeds = core.get_input_embeddings()(input_ids)
            if pixel_values is None:
                raise RuntimeError("DINO alignment requires image pixel values.")

            image_out = core.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            per_image = tuple(image_out.pooler_output)
            if not per_image:
                raise RuntimeError("Qwen vision encoder returned no image features.")
            token_counts = {tokens.shape[0] for tokens in per_image}
            if token_counts != {self._tokens_per_image}:
                raise RuntimeError(
                    f"Expected {self._tokens_per_image} Qwen tokens per image, got {sorted(token_counts)}"
                )
            image_tokens = torch.stack(per_image, dim=0)
            flat_image_tokens = image_tokens.flatten(0, 1).to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            image_mask, _ = core.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=flat_image_tokens,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, flat_image_tokens)

            if n_extra:
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[:, base_seq_len:, :] = extra_embeds.to(inputs_embeds.dtype)

            outputs = core(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )

        return outputs.last_hidden_state, attention_mask.to(torch.bool), image_tokens
