# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Open-loop only: hard-coded TeaCache-style residual reuse for DiT blocks.
# When num_denoising_steps_action == 5, only sub-steps 0 and 2 run full block stacks;
# steps 1,3,4 reuse cached block residual (x_after_blocks - x_before_blocks) from
# the last matching full step (step 1 uses residual from 0; steps 3,4 use residual from 2).
#
# Mirrors the idea in efficiency/teacache_sample_video_i2v.py (residual skip inside forward).
#
# --- Why "only 2 block forwards" is NOT ~same latency as "denoise 2 steps" -----------------
# 1) CosmosPolicySampler (cosmos_sampler.py) rewrites num_steps: for N>1 it does N->N-1 for nfe, then
#    adds one extra denoiser call when sample_clean. So **N user steps => N denoise() calls** for N>=2,
#    except N=2 becomes N-1=1 and hits the **special branch** => only **one** denoise() total (not two).
# 2) Each denoise() still runs: LVG concat, prepare_embedded_sequence (patch x_embedder), cross-attn
#    projection, **t_embedder** (timestep changes every step), optional tactile bias, **final_layer**,
#    unpatchify — on **every** sub-step. Hard cache only skips the **self.blocks** loop on 3/5 steps.
# 3) So vs 10 full steps you save mostly **8× DiT blocks**, but you still pay **5×** embed/t_embed/final
#    on the 5-step path; vs true 2-step you would pay **2×** those (and 2× blocks). The "2 block passes"
#    number does **not** mean only 2 end-to-end forwards of the non-block parts.

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.amp as amp
from einops import rearrange

from cosmos_policy._src.predict2.conditioner import DataType
from cosmos_policy._src.imaginaire.utils import log


def _lvg_concat_input_mask_if_needed(
    net: torch.nn.Module,
    x_B_C_T_H_W: torch.Tensor,
    timesteps_B_T: torch.Tensor,
    data_type: DataType,
    condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match MinimalV1LVGDiT.forward input concat before MiniTrainDIT.prepare_embedded_sequence."""
    from cosmos_policy._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT

    if isinstance(net, MinimalV1LVGDiT):
        if data_type == DataType.VIDEO:
            assert condition_video_input_mask_B_C_T_H_W is not None, "VIDEO mode requires condition mask"
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)],
                dim=1,
            )
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [
                    x_B_C_T_H_W,
                    torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device),
                ],
                dim=1,
            )
        timesteps_B_T = timesteps_B_T * net.timestep_scale
    return x_B_C_T_H_W, timesteps_B_T


def minimal_v4_dit_forward_with_hard_block_cache(
    net: torch.nn.Module,
    x_B_C_T_H_W: torch.Tensor,
    timesteps_B_T: torch.Tensor,
    crossattn_emb: torch.Tensor,
    *,
    call_idx: int,
    total_calls: int = 5,
    fps: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    data_type: Optional[DataType] = DataType.VIDEO,
    intermediate_feature_ids: Optional[List[int]] = None,
    img_context_emb: Optional[torch.Tensor] = None,
    tactile_self_attn_gate_B: Optional[torch.Tensor] = None,
    condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Same as MinimalV4DiT.forward except the transformer block loop may skip using cached residual.
    """
    assert isinstance(data_type, DataType), (
        f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
    )
    x_B_C_T_H_W, timesteps_B_T = _lvg_concat_input_mask_if_needed(
        net, x_B_C_T_H_W, timesteps_B_T, data_type, condition_video_input_mask_B_C_T_H_W
    )
    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = net.prepare_embedded_sequence(
        x_B_C_T_H_W,
        fps=fps,
        padding_mask=padding_mask,
    )

    if net.use_crossattn_projection:
        crossattn_emb = net.crossattn_proj(crossattn_emb)

    if img_context_emb is not None:
        assert net.extra_image_context_dim is not None, (
            "extra_image_context_dim must be set if img_context_emb is provided"
        )
        img_context_emb = net.img_context_proj(img_context_emb)
        context_input = (crossattn_emb, img_context_emb)
    else:
        context_input = crossattn_emb

    with amp.autocast("cuda", enabled=net.use_wan_fp32_strategy, dtype=torch.float32):
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = net.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = net.t_embedding_norm(t_embedding_B_T_D)

    affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_B_T_D.detach()}
    net.affline_scale_log_info = affline_scale_log_info
    net.affline_emb = t_embedding_B_T_D
    if getattr(net, "_capture_debug_mapped_t_embedding", False):
        if not hasattr(net, "_debug_mapped_t_embedding_history"):
            net._debug_mapped_t_embedding_history = []
        net._debug_mapped_t_embedding_history.append(t_embedding_B_T_D.detach().to(torch.float32).cpu())
    net.crossattn_emb = crossattn_emb

    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
            f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
        )

    tactile_kw = net._tactile_self_attn_block_kw(x_B_T_H_W_D, tactile_self_attn_gate_B)

    intermediate_features_outputs = []
    x_in_before_blocks = x_B_T_H_W_D

    # Hard schedule for total_calls == 5: full at 0 and 2; cache at 1,3,4
    middle = total_calls // 2
    full_indices = {0, middle}
    use_cache = total_calls == 5 and getattr(net, "_openloop_hard_block_cache_active", True)

    if use_cache and call_idx in full_indices:
        for i, block in enumerate(net.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                **tactile_kw,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                x_reshaped_for_disc = rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped_for_disc)
        delta = x_B_T_H_W_D - x_in_before_blocks
        if call_idx == 0:
            net._openloop_residual_after_first = delta
        elif call_idx == middle:
            net._openloop_residual_after_middle = delta
    elif use_cache:
        if call_idx == 1:
            res = getattr(net, "_openloop_residual_after_first", None)
        else:
            res = getattr(net, "_openloop_residual_after_middle", None)
        if res is None:
            log.warning(
                "openloop_hard_residual_cache: missing cached residual at call_idx=%s; running full blocks.",
                call_idx,
            )
            for i, block in enumerate(net.blocks):
                x_B_T_H_W_D = block(
                    x_B_T_H_W_D,
                    t_embedding_B_T_D,
                    context_input,
                    rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                    adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                    extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                    **tactile_kw,
                )
                if intermediate_feature_ids and i in intermediate_feature_ids:
                    x_reshaped_for_disc = rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d")
                    intermediate_features_outputs.append(x_reshaped_for_disc)
        else:
            x_B_T_H_W_D = x_in_before_blocks + res
    else:
        for i, block in enumerate(net.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                **tactile_kw,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                x_reshaped_for_disc = rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped_for_disc)

    x_B_T_H_W_O = net.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
    x_B_C_Tt_Hp_Wp = net.unpatchify(x_B_T_H_W_O)
    if intermediate_feature_ids:
        if len(intermediate_features_outputs) != len(intermediate_feature_ids):
            log.warning(
                "Collected %s intermediate features, but expected %s. Requested IDs: %s",
                len(intermediate_features_outputs),
                len(intermediate_feature_ids),
                intermediate_feature_ids,
            )
        return x_B_C_Tt_Hp_Wp, intermediate_features_outputs

    return x_B_C_Tt_Hp_Wp


def reset_openloop_denoise_counter(model: torch.nn.Module) -> None:
    """Call once before each diffusion sampling (e.g. each get_action)."""
    model._openloop_denoise_idx = 0
    if hasattr(model, "net"):
        net = model.net
        for name in (
            "_openloop_residual_after_first",
            "_openloop_residual_after_middle",
        ):
            if hasattr(net, name):
                delattr(net, name)


def apply_openloop_hard_residual_cache(model: torch.nn.Module, num_denoising_steps: int = 5) -> None:
    """
    Patch model.denoise for open-loop hard residual cache. Original denoise saved on model._denoise_orig.
    Only affects sampling when num_denoising_steps == 5 (other values fall back to full blocks every step).
    """
    import types

    if getattr(model, "_openloop_hard_residual_cache_patched", False):
        model._openloop_residual_cache_num_steps = num_denoising_steps
        return

    if num_denoising_steps == 5 and not getattr(model, "_openloop_hard_cache_cost_note_printed", False):
        print(
            "[openloop_hard_residual_cache] 5 user denoise steps => 5× denoise() calls; each call still "
            "runs patch embed + t_embed + final_layer; only DiT **blocks** are skipped on 3/5 calls. "
            "Latency vs 10 steps ≈ saved blocks, not 5× overall speedup. "
            "FRANKA_NUM_DENOISING_STEPS=2 uses sampler special-case => **1** denoise() call (see cosmos_sampler.py)."
        )
        model._openloop_hard_cache_cost_note_printed = True

    model._denoise_orig = model.denoise
    model.denoise = types.MethodType(denoise_openloop_hard_cache, model)
    model._openloop_hard_residual_cache_patched = True
    model._openloop_hard_residual_cache = True
    model._openloop_residual_cache_num_steps = num_denoising_steps


def remove_openloop_hard_residual_cache(model: torch.nn.Module) -> None:
    """Restore original denoise (e.g. between baseline vs cache benchmark runs)."""
    if getattr(model, "_openloop_hard_residual_cache_patched", False) and hasattr(model, "_denoise_orig"):
        model.denoise = model._denoise_orig
    model._openloop_hard_residual_cache_patched = False
    model._openloop_hard_residual_cache = False


def denoise_openloop_hard_cache(self: Any, xt_B_C_T_H_W: torch.Tensor, sigma: torch.Tensor, condition: Any):
    """
    Drop-in replacement for CosmosPolicyVideo2WorldModel.denoise with optional hard block cache.
    """
    from einops import rearrange

    from cosmos_policy._src.imaginaire.utils.denoise_prediction import DenoisePrediction

    if not getattr(self, "_openloop_hard_residual_cache", False):
        return self._denoise_orig(xt_B_C_T_H_W, sigma, condition)

    ns = int(getattr(self, "_openloop_residual_cache_num_steps", 5))
    if ns != 5:
        return self._denoise_orig(xt_B_C_T_H_W, sigma, condition)
    if sigma.ndim == 1:
        sigma_B_T = rearrange(sigma, "b -> b 1")
    elif sigma.ndim == 2:
        sigma_B_T = sigma
    else:
        raise ValueError(f"sigma shape {sigma.shape} is not supported")

    sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")
    c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling(sigma=sigma_B_1_T_1_1)

    net_state_in_B_C_T_H_W = xt_B_C_T_H_W * c_in_B_1_T_1_1
    condition_video_mask = None

    if condition.is_video:
        condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(net_state_in_B_C_T_H_W) / self.config.sigma_data
        if not condition.use_video_condition:
            condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

        _, C, _, _, _ = xt_B_C_T_H_W.shape
        condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
            net_state_in_B_C_T_H_W
        )

        net_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + net_state_in_B_C_T_H_W * (
            1 - condition_video_mask
        )
        sigma_cond_B_1_T_1_1 = torch.ones_like(sigma_B_1_T_1_1) * self.config.sigma_conditional
        _, _, _, c_noise_cond_B_1_T_1_1 = self.scaling(sigma=sigma_cond_B_1_T_1_1)
        condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
        c_noise_B_1_T_1_1 = c_noise_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + c_noise_B_1_T_1_1 * (
            1 - condition_video_mask_B_1_T_1_1
        )

    timesteps_B_T = c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(
        **{
            **self.tensor_kwargs,
            "dtype": torch.float32 if self.config.use_wan_fp32_strategy else self.tensor_kwargs["dtype"],
        },
    )
    if getattr(self, "_capture_debug_timesteps", False):
        if not hasattr(self, "_debug_timesteps_history"):
            self._debug_timesteps_history = []
        self._debug_timesteps_history.append(timesteps_B_T.detach().to(torch.float32).cpu())

    cond_dict = condition.to_dict()
    idx = int(getattr(self, "_openloop_denoise_idx", 0))
    self._openloop_denoise_idx = idx + 1

    # CosmosPolicySampler: for num_steps=5 -> 4 inner + 1 clean = 5 denoise calls
    setattr(self.net, "_openloop_hard_block_cache_active", True)
    net_output_B_C_T_H_W = minimal_v4_dit_forward_with_hard_block_cache(
        self.net,
        net_state_in_B_C_T_H_W.to(**self.tensor_kwargs),
        timesteps_B_T,
        cond_dict["crossattn_emb"],
        call_idx=idx,
        total_calls=5,
        fps=cond_dict.get("fps"),
        padding_mask=cond_dict.get("padding_mask"),
        data_type=cond_dict.get("data_type", DataType.VIDEO),
        intermediate_feature_ids=cond_dict.get("intermediate_feature_ids"),
        img_context_emb=cond_dict.get("img_context_emb"),
        tactile_self_attn_gate_B=cond_dict.get("tactile_self_attn_gate_B"),
        condition_video_input_mask_B_C_T_H_W=cond_dict.get("condition_video_input_mask_B_C_T_H_W"),
    ).float()

    x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
    if condition.is_video and self.config.denoise_replace_gt_frames:
        x0_pred_B_C_T_H_W = condition.gt_frames.type_as(
            x0_pred_B_C_T_H_W
        ) * condition_video_mask + x0_pred_B_C_T_H_W * (1 - condition_video_mask)

    eps_pred_B_C_T_H_W = (xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / sigma_B_1_T_1_1

    return DenoisePrediction(x0_pred_B_C_T_H_W, eps_pred_B_C_T_H_W, None)
