# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Chunked self-attention with outer-product logits bias: scores_ij += gamma_b * a_i * b_j.

## Can we use Dao-AILab FlashAttention (flash_attn_func / FA3) here?

**Not with the upstream public API today.**

- `flash_attn.flash_attn_interface.flash_attn_func` only supports optional **ALiBi**-style bias:
  ``(-alibi_slope * |i - j|)`` per head, not a general per-(i,j) matrix and **not** ``gamma * a_i * b_j``
  with arbitrary vectors ``a``, ``b`` on latent token positions.
- Hopper **FA3** (`flash_attn_3`) exposes the same ALiBi-style mechanism, not arbitrary additive logits.
- Community PRs (e.g. custom dense bias) are not a stable, drop-in API for this rank-1 outer product.

So **you cannot** call `flash_attn_func(q, k, v, ...)` and pass our tactile bias without either:
- materializing a full (or chunked) bias tile and using an op that accepts it (what we do with SDPA), or
- a **custom CUDA/Triton** kernel (or projects like FlashBias / future PyTorch AttnBias types) that fuses
  softmax with this bias structure.

## FlashBias-style SDPA (rank-1 outer bias) — recommended for speed

Our bias is **rank-1**: ``gamma_b * a_i * b_j = (sqrt(gamma_b)*a_i) * (sqrt(gamma_b)*b_j)``.

Following [FlashBias](https://arxiv.org/pdf/2505.12044) (NeurIPS 2025, Tsinghua), this is a self-contained
re-implementation of the trick (no external FlashBias dependency required):
use **concatenated Q/K** so the extra inner-product dimension contributes exactly that outer product, with
**no float attn_mask** — so PyTorch SDPA can select **Flash / cuDNN FMHA** backends (subject to head
dim rules: ``(headdim + rank)`` padded to a multiple of 8).

Backend ``flashbias_sdpa``: one fused SDPA over the **full** sequence (``chunk_q`` ignored).

## Other backends

- **flashbias_sdpa** (default): full-sequence SDPA with **concat** ``[q*scale, q_bias], [k, k_bias]`` (FlashBias trick); no float ``attn_mask`` — best chance to match **pre-bias FlashAttention speed**.
- **sdpa**: per query-chunk SDPA with float ``attn_mask`` (Flash often disabled).
- **eager**: explicit ``matmul -> +bias -> softmax -> @V`` (debug).

Env: ``COSMOS_TACTILE_SELF_ATTN_BACKEND`` = ``flashbias_sdpa`` | ``sdpa`` | ``eager``.
Optional: ``COSMOS_TACTILE_SELF_ATTN_SDP_PRIORITY`` (see below).
"""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _HAS_SDPA_KERNEL = True
except Exception:  # pragma: no cover
    SDPBackend = None  # type: ignore[misc, assignment]
    sdpa_kernel = None  # type: ignore[misc, assignment]
    _HAS_SDPA_KERNEL = False

TactileChunkBackend = Literal["sdpa", "eager", "flashbias_sdpa"]


def _sdp_priority_from_env() -> list | None:
    """Map COSMOS_TACTILE_SELF_ATTN_SDP_PRIORITY to backend order; None = PyTorch default."""
    if not _HAS_SDPA_KERNEL:
        return None
    key = os.environ.get("COSMOS_TACTILE_SELF_ATTN_SDP_PRIORITY", "").strip().lower()
    if not key:
        return None
    order = []
    for part in key.replace(",", " ").split():
        part = part.strip()
        if part in ("flash", "flash_attention"):
            order.append(SDPBackend.FLASH_ATTENTION)
        elif part in ("cudnn",):
            order.append(SDPBackend.CUDNN_ATTENTION)
        elif part in ("efficient", "mem_efficient", "memory_efficient"):
            order.append(SDPBackend.EFFICIENT_ATTENTION)
        elif part in ("math",):
            order.append(SDPBackend.MATH)
    return order or None


def _scaled_dot_product_attention_chunk(
    q_c: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    backends = _sdp_priority_from_env()
    if backends is not None:
        try:
            with sdpa_kernel(backends=backends, set_priority_order=True):
                return F.scaled_dot_product_attention(
                    q_c,
                    k,
                    v,
                    attn_mask=attn_bias,
                    is_causal=False,
                    scale=scale,
                )
        except Exception:
            pass
    return F.scaled_dot_product_attention(
        q_c,
        k,
        v,
        attn_mask=attn_bias,
        is_causal=False,
        scale=scale,
    )


def _chunk_eager(
    q_c: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    scores = torch.matmul(q_c, k.transpose(-2, -1)) * scale
    scores = scores + attn_bias
    attn_w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=q_c.dtype)
    return torch.matmul(attn_w, v)


def _flashbias_sdpa_full(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    q_bias_bsh1: torch.Tensor,
    k_bias_bsh1: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """
    FlashBias SDPA formulation (concat extra dim, scale=1): see FlashBias README / attention_func.flashbias_sdpa.

    q,k,v,q_bias,k_bias: (B, S, H, D) and (B, S, H, 1). Internally uses (B, H, S, *) for F.scaled_dot_product_attention.
    """
    q = q_bshd.transpose(1, 2)
    k = k_bshd.transpose(1, 2)
    v = v_bshd.transpose(1, 2)
    qb = q_bias_bsh1.transpose(1, 2)
    kb = k_bias_bsh1.transpose(1, 2)
    _, _, _, d_h = q.shape
    _, _, _, r_b = qb.shape
    total = d_h + r_b
    pad = (8 - (total % 8)) % 8

    def _run(q_cat: torch.Tensor, k_cat: torch.Tensor) -> torch.Tensor:
        backends = _sdp_priority_from_env()
        if backends is not None:
            try:
                with sdpa_kernel(backends=backends, set_priority_order=True):
                    return F.scaled_dot_product_attention(
                        q_cat,
                        k_cat,
                        v,
                        attn_mask=None,
                        dropout_p=0.0,
                        scale=1.0,
                        is_causal=causal,
                    )
            except Exception:
                pass
        return F.scaled_dot_product_attention(
            q_cat,
            k_cat,
            v,
            attn_mask=None,
            dropout_p=0.0,
            scale=1.0,
            is_causal=causal,
        )

    if pad == 0:
        out = _run(torch.cat([q * softmax_scale, qb], dim=-1), torch.cat([k, kb], dim=-1))
    else:
        blank = torch.zeros(q.shape[0], q.shape[1], q.shape[2], pad, device=q.device, dtype=q.dtype)
        out = _run(
            torch.cat([q * softmax_scale, qb, blank], dim=-1),
            torch.cat([k, kb, blank], dim=-1),
        )
    return out.transpose(1, 2).contiguous()


def dao_flash_attn_supports_tactile_outer_bias() -> bool:
    """
    Returns True only if we detect a *future* flash_attn API that accepts a general logits bias
    compatible with our chunked (B, H, Lq, Lk) bias. Upstream flash_attn_func / FA3: False.
    """
    try:
        import inspect

        from flash_attn.flash_attn_interface import flash_attn_func
    except Exception:
        try:
            import inspect

            from flash_attn_3.flash_attn_interface import flash_attn_func
        except Exception:
            return False
    try:
        params = inspect.signature(flash_attn_func).parameters
    except Exception:
        return False
    for name in ("attn_bias", "attention_bias", "bias", "attn_logits_bias"):
        if name in params:
            return True
    return False


def self_attention_with_tactile_outer_bias_chunked(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    a_S: torch.Tensor,
    b_S: torch.Tensor,
    gamma_B: torch.Tensor,
    chunk_q: int,
    output_proj: nn.Linear,
    output_dropout: nn.Module,
) -> torch.Tensor:
    """
    Self-attention with additive logits bias: scores_ij += gamma_b * a_i * b_j (batched gamma).
    q,k,v: (B, S, H, D).

    - ``flashbias_sdpa``: one SDPA call, no float mask; ``chunk_q`` ignored.
    - ``sdpa`` / ``eager``: query-axis chunking with ``chunk_q``.
    """
    # Default: FlashBias-style concat (no float attn_mask) so SDPA can use Flash/cuDNN FMHA when supported.
    backend: str = os.environ.get("COSMOS_TACTILE_SELF_ATTN_BACKEND", "flashbias_sdpa").strip().lower()
    if backend not in ("sdpa", "eager", "flashbias_sdpa"):
        backend = "sdpa"

    B, S, Hn, D = q_B_S_H_D.shape
    dtype = q_B_S_H_D.dtype
    device = q_B_S_H_D.device
    a = a_S.to(device=device, dtype=dtype)
    b = b_S.to(device=device, dtype=dtype)
    sqrt_gamma = torch.sqrt(gamma_B.to(device=device, dtype=dtype).clamp(min=0.0)).view(B, 1, 1, 1)
    q_bias = sqrt_gamma * a.view(1, S, 1, 1).expand(B, S, Hn, 1)
    k_bias = sqrt_gamma * b.view(1, S, 1, 1).expand(B, S, Hn, 1)

    if backend == "flashbias_sdpa":
        out_bshd = _flashbias_sdpa_full(
            q_B_S_H_D,
            k_B_S_H_D,
            v_B_S_H_D,
            q_bias,
            k_bias,
            softmax_scale=D**-0.5,
            causal=False,
        )
        flat = rearrange(out_bshd, "b s h d -> b s (h d)")
        return output_dropout(output_proj(flat))

    q = rearrange(q_B_S_H_D, "b s h d -> b h s d")
    k = rearrange(k_B_S_H_D, "b s h d -> b h s d")
    v = rearrange(v_B_S_H_D, "b s h d -> b h s d")
    scale = D**-0.5
    gamma = gamma_B.to(dtype=q.dtype).view(B, 1, 1, 1)
    b_row = b.view(1, 1, 1, S)
    out = torch.empty_like(q)

    for qs in range(0, S, chunk_q):
        qe = min(qs + chunk_q, S)
        q_c = q[:, :, qs:qe, :]
        a_c = a[qs:qe].view(1, 1, -1, 1)
        attn_bias = gamma * (a_c * b_row)
        if backend == "eager":
            out[:, :, qs:qe, :] = _chunk_eager(q_c, k, v, attn_bias, scale)
        else:
            out[:, :, qs:qe, :] = _scaled_dot_product_attention_chunk(q_c, k, v, attn_bias, scale)

    out_B_S_H_D = rearrange(out, "b h s d -> b s h d")
    flat = rearrange(out_B_S_H_D, "b s h d -> b s (h d)")
    return output_dropout(output_proj(flat))
