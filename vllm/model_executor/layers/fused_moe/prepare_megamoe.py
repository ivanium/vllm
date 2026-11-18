# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton input-staging kernels for DeepGEMM MegaMoE.

Quantize hidden states to fp8 with E8M0 group scales, to NVFP4 with per-token
global scales, or copy them as BF16, and repack the routing top-k tensors into
the int64/float32 layout that the DeepGEMM MegaMoE kernels consume. Given an
EPLB layer state, routing ids are mapped to physical replicas on the way.
"""

import functools
import os
from typing import NamedTuple

import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.eplb_triton import eplb_map_and_record
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    FP8_DTYPE,
    QuantKey,
    kMxfp8Dynamic,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

_E8M0_SCALE_DTYPES = (torch.uint8, torch.float8_e8m0fnu)
# Inverse of the NVFP4 range: E2M1 max (6) times E4M3 max (448).
_INV_NVFP4_MAX = tl.constexpr(1.0 / 2688.0)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true")


@functools.cache
def nvfp4_row_staging_enabled() -> bool:
    """`VLLM_MEGAMOE_NVFP4_ROW_STAGING=1` stages NVFP4 MegaMoE inputs with the
    row-block kernel (same bytes, fewer instructions per element)."""
    return _env_flag("VLLM_MEGAMOE_NVFP4_ROW_STAGING")


@functools.cache
def megamoe_staging_overlap_enabled() -> bool:
    """`VLLM_MEGAMOE_STAGING_OVERLAP=1` stages the NVFP4 MegaMoE hidden states
    on the aux stream while the router runs (row-block kernel, split in its
    hidden-state and routing halves)."""
    return _env_flag("VLLM_MEGAMOE_STAGING_OVERLAP")


@triton.jit
def _quantize_fp8_groups(
    hidden,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    """Return fp8 values, fp32 group scales and, for UE8M0, the scale exponents."""
    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [BLOCK_M, num_groups, GROUP_K])
    amax = tl.max(hidden_groups, axis=2)
    amax = tl.maximum(amax, 1.0e-4)

    # Divide correctly rounded like DeepGEMM's cast, so FP32 scales match bit for bit.
    scale = tl.math.div_rn(amax, 448.0)
    if USE_UE8M0:
        scale_bits = scale.to(tl.uint32, bitcast=True)
        scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
            tl.uint32
        )
        scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
        quant_scale = (scale_exp << 23).to(tl.float32, bitcast=True)
    else:
        scale_exp = scale.to(tl.uint32, bitcast=True)
        quant_scale = scale

    hidden_groups = tl.reshape(hidden, [BLOCK_M, num_groups, GROUP_K])
    scaled = hidden_groups * tl.math.div_rn(1.0, quant_scale)[:, :, None]
    scaled = tl.reshape(scaled, [BLOCK_M, BLOCK_K])
    return scaled.to(tl.float8e4nv), quant_scale, scale_exp


@triton.jit
def _pack_bytes(values, NUM_BYTES: tl.constexpr):
    """Pack the last dim of `[..., NUM_BYTES]` byte values into int32 words."""
    offsets = tl.arange(0, NUM_BYTES)
    return tl.sum(values.to(tl.uint32) << (offsets * 8), axis=-1).to(tl.int32)


@triton.jit
def _shared_sf_row(token_id, SHARED_BLOCK_M: tl.constexpr):
    """Row of `token_id` in the shared-expert MN-major UTCCP scale pages."""
    m_block_id = token_id // SHARED_BLOCK_M
    m_in_block = token_id % SHARED_BLOCK_M
    aligned_block_m: tl.constexpr = triton.cdiv(SHARED_BLOCK_M, 128) * 128
    transposed_m = (
        (m_in_block // 128) * 128 + (m_in_block % 32) * 4 + (m_in_block % 128) // 32
    )
    return m_block_id * aligned_block_m + transposed_m


@triton.jit
def _store_topk(
    token_id,
    token_mask,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    logical_to_physical,
    replica_count,
    expert_load,
    record_enabled,
    num_unpadded_tokens,
    num_logical,
    map_stride,
    num_physical,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    """Repack top-k routing as int64 ids and fp32 weights; padding is unrouted."""
    topk_offsets = tl.arange(0, BLOCK_TOPK)
    topk_mask = token_mask[:, None] & (topk_offsets[None, :] < top_k)
    token_is_padding = tl.full((BLOCK_M,), False, tl.int1)
    if is_padding is not None:
        token_is_padding = tl.load(
            is_padding + token_id * is_padding_stride_m,
            mask=token_mask,
            other=True,
        )

    ids = tl.load(
        topk_ids
        + token_id[:, None] * topk_ids_stride_m
        + topk_offsets[None, :] * topk_ids_stride_k,
        mask=topk_mask,
        other=0,
    ).to(tl.int64)
    ids = tl.where(token_is_padding[:, None], -1, ids)
    if logical_to_physical is not None:
        ids = eplb_map_and_record(
            ids,
            token_id[:, None],
            topk_mask,
            logical_to_physical,
            replica_count,
            num_logical,
            map_stride,
            expert_load,
            num_physical,
            record_enabled,
            num_unpadded_tokens,
        )
    tl.store(
        topk_idx_out
        + token_id[:, None] * topk_idx_stride_m
        + topk_offsets[None, :] * topk_idx_stride_k,
        ids,
        mask=topk_mask,
    )

    weights = tl.load(
        topk_weights
        + token_id[:, None] * topk_weights_stride_m
        + topk_offsets[None, :] * topk_weights_stride_k,
        mask=topk_mask,
        other=0.0,
    )
    weights = tl.where(token_is_padding[:, None], 0.0, weights)
    tl.store(
        topk_weights_out
        + token_id[:, None] * topk_weights_out_stride_m
        + topk_offsets[None, :] * topk_weights_out_stride_k,
        weights,
        mask=topk_mask,
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _prepare_megamoe_inputs_kernel(
    hidden_states,
    x_out,
    x_sf,
    shared_x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    logical_to_physical,
    replica_count,
    expert_load,
    record_enabled,
    num_unpadded_tokens,
    num_logical,
    map_stride,
    num_physical,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    shared_x_sf_stride_m: tl.constexpr,
    shared_x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    num_tokens,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    SHARED_BLOCK_M: tl.constexpr,
    QUANTIZE: tl.constexpr,
    USE_UE8M0: tl.constexpr,
) -> None:
    token_id = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    token_mask = token_id < num_tokens
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = token_mask[:, None] & (k_offsets[None, :] < hidden_size)
    hidden = tl.load(
        hidden_states
        + token_id[:, None] * hidden_stride_m
        + k_offsets[None, :] * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    if not QUANTIZE:
        tl.store(
            x_out + token_id[:, None] * x_stride_m + k_offsets[None, :] * x_stride_k,
            hidden.to(x_out.dtype.element_ty),
            mask=k_mask,
        )
    else:
        num_groups: tl.constexpr = BLOCK_K // GROUP_K
        fp8, quant_scale, scale_exp = _quantize_fp8_groups(
            hidden, BLOCK_M, BLOCK_K, GROUP_K, USE_UE8M0
        )
        tl.store(
            x_out + token_id[:, None] * x_stride_m + k_offsets[None, :] * x_stride_k,
            fp8,
            mask=k_mask,
        )

        if USE_UE8M0:
            packed_scale = _pack_bytes(scale_exp, num_groups)
            tl.store(
                x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
                packed_scale,
                mask=token_mask,
            )
        else:
            # FP32 scales; the bitcast fits a float32 or int32 buffer.
            sf_offset = tl.arange(0, num_groups)
            tl.store(
                x_sf
                + token_id[:, None] * x_sf_stride_m
                + k_block_id * x_sf_stride_k
                + sf_offset[None, :],
                quant_scale.to(x_sf.dtype.element_ty, bitcast=True),
                mask=token_mask[:, None],
            )

        # Shared-expert TMA loads need MN-major scales permuted by the runtime
        # BLOCK_M; write them here while resident to skip an extra kernel.
        if shared_x_sf is not None:
            if USE_UE8M0:
                shared_scale = packed_scale
            else:
                shared_scale = tl.reshape(quant_scale, [BLOCK_M]).to(
                    shared_x_sf.dtype.element_ty, bitcast=True
                )
            shared_row = _shared_sf_row(token_id, SHARED_BLOCK_M)
            tl.store(
                shared_x_sf
                + shared_row * shared_x_sf_stride_m
                + k_block_id * shared_x_sf_stride_k,
                shared_scale,
                mask=token_mask,
            )

    if k_block_id == 0:
        _store_topk(
            token_id,
            token_mask,
            topk_ids,
            topk_weights,
            is_padding,
            topk_idx_out,
            topk_weights_out,
            logical_to_physical,
            replica_count,
            expert_load,
            record_enabled,
            num_unpadded_tokens,
            num_logical,
            map_stride,
            num_physical,
            topk_ids_stride_m,
            topk_ids_stride_k,
            topk_weights_stride_m,
            topk_weights_stride_k,
            is_padding_stride_m,
            topk_idx_stride_m,
            topk_idx_stride_k,
            topk_weights_out_stride_m,
            topk_weights_out_stride_k,
            top_k,
            BLOCK_M,
            BLOCK_TOPK,
        )


@triton.jit(do_not_specialize=["num_tokens", "l2_activation_scale"])
def _prepare_megamoe_nvfp4_inputs_kernel(
    hidden_states,
    x_out,
    x_sf,
    x_scales,
    shared_x,
    shared_x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    logical_to_physical,
    replica_count,
    expert_load,
    record_enabled,
    num_unpadded_tokens,
    num_logical,
    map_stride,
    num_physical,
    w2_global_scale,
    l2_alpha,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    shared_x_stride_m: tl.constexpr,
    shared_x_sf_stride_m: tl.constexpr,
    shared_x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    num_tokens,
    l2_activation_scale,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    num_local_experts: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    SHARED_BLOCK_M: tl.constexpr,
    SHARED_MXFP8: tl.constexpr,
) -> None:
    # Grid (token, split): each program quantizes the BLOCK_K chunks split,
    # split + NUM_SPLITS, ... of one token's row.
    token = tl.program_id(0)
    split = tl.program_id(1)
    if l2_alpha is not None:
        experts = tl.arange(0, BLOCK_E)
        expert_mask = (experts < num_local_experts) & (token == 0) & (split == 0)
        alpha = tl.load(w2_global_scale + experts, mask=expert_mask)
        tl.store(l2_alpha + experts, alpha * l2_activation_scale, mask=expert_mask)
    if token >= num_tokens:
        return

    if split == 0:
        token_ids = token + tl.zeros([1], dtype=tl.int32)
        _store_topk(
            token_ids,
            token_ids < num_tokens,
            topk_ids,
            topk_weights,
            is_padding,
            topk_idx_out,
            topk_weights_out,
            logical_to_physical,
            replica_count,
            expert_load,
            record_enabled,
            num_unpadded_tokens,
            num_logical,
            map_stride,
            num_physical,
            topk_ids_stride_m,
            topk_ids_stride_k,
            topk_weights_stride_m,
            topk_weights_stride_k,
            is_padding_stride_m,
            topk_idx_stride_m,
            topk_idx_stride_k,
            topk_weights_out_stride_m,
            topk_weights_out_stride_k,
            top_k,
            1,
            BLOCK_TOPK,
        )

    # Every program reads the whole row for the token scale.
    row = hidden_states + token.to(tl.int64) * hidden_stride_m
    cols = tl.arange(0, BLOCK_H)
    hidden_row = tl.load(
        row + cols * hidden_stride_k, mask=cols < hidden_size, other=0.0
    )
    amax = tl.max(tl.abs(hidden_row.to(tl.float32)), axis=0)
    x_scale = tl.maximum(amax, 1.0e-12) * _INV_NVFP4_MAX
    if split == 0:
        tl.store(x_scales + token, x_scale)

    shared_sf_row = _shared_sf_row(token, SHARED_BLOCK_M)
    for k0 in tl.range(split * BLOCK_K, hidden_size, NUM_SPLITS * BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_ids < hidden_size
        hidden = tl.load(row + k_ids * hidden_stride_k, mask=k_mask, other=0.0)

        # E2M1 values with one E4M3 scale per 16, relative to the token scale.
        values = tl.reshape(
            tl.div_rn(hidden.to(tl.float32), x_scale), [BLOCK_K // 16, 16]
        )
        block_scale = tl.max(tl.abs(values), axis=1) * (1.0 / 6.0)
        block_scale = tl.minimum(tl.maximum(block_scale, 2.0**-9), 448.0)
        block_scale = block_scale.to(tl.float8e4nv)
        values = tl.div_rn(values, block_scale.to(tl.float32)[:, None])
        magnitude = tl.abs(values)
        # Round to nearest even on the grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
        code = (
            (magnitude > 0.25).to(tl.int32)
            + (magnitude >= 0.75).to(tl.int32)
            + (magnitude > 1.25).to(tl.int32)
            + (magnitude >= 1.75).to(tl.int32)
            + (magnitude > 2.5).to(tl.int32)
            + (magnitude >= 3.5).to(tl.int32)
            + (magnitude > 5.0).to(tl.int32)
        )
        code = tl.where((values < 0) & (code != 0), code | 8, code)
        low, high = tl.split(tl.reshape(code, [BLOCK_K // 2, 2]))
        packed_ids = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        tl.store(
            x_out + token * x_stride_m + packed_ids,
            (low | (high << 4)).to(tl.uint8),
            mask=packed_ids < hidden_size // 2,
        )
        # Four E4M3 scales per little-endian int32 word, one word per 64 values.
        sf_ids = k0 // 64 + tl.arange(0, BLOCK_K // 64)
        sf_bytes = tl.reshape(
            block_scale.to(tl.uint8, bitcast=True), [BLOCK_K // 64, 4]
        )
        tl.store(
            x_sf + token * x_sf_stride_m + sf_ids,
            _pack_bytes(sf_bytes, 4),
            mask=sf_ids < hidden_size // 64,
        )

        if shared_x is not None:
            shared_ptrs = shared_x + token * shared_x_stride_m + k_ids
            if SHARED_MXFP8:
                fp8, _, scale_exp = _quantize_fp8_groups(
                    tl.reshape(hidden.to(tl.float32), (1, BLOCK_K)),
                    1,
                    BLOCK_K,
                    32,
                    True,
                )
                tl.store(shared_ptrs, tl.reshape(fp8, [BLOCK_K]), mask=k_mask)
                word_ids = k0 // 128 + tl.arange(0, BLOCK_K // 128)
                tl.store(
                    shared_x_sf
                    + shared_sf_row * shared_x_sf_stride_m
                    + word_ids * shared_x_sf_stride_k,
                    _pack_bytes(tl.reshape(scale_exp, [BLOCK_K // 128, 4]), 4),
                    mask=word_ids < hidden_size // 128,
                )
            else:
                tl.store(shared_ptrs, hidden, mask=k_mask)


@triton.jit
def _div_rn_by_recip(a, b, b_inv):
    """`tl.div_rn(a, b)` given `b_inv = tl.div_rn(1.0, b)`, in one multiply and
    two FMAs instead of a division.

    Markstein's correction: with `b_inv` the correctly rounded reciprocal, the
    residual `a - b * q` of `q = a * b_inv` is exact and `q + residual * b_inv`
    rounds to the correctly rounded quotient, while quotient and residual are
    normal. Callers only use quotients whose code cannot depend on the
    subnormal range.
    """
    quotient = a * b_inv
    residual = tl.fma(-b, quotient, a)
    return tl.fma(residual, b_inv, quotient)


@triton.jit
def _e2m1_codes(values):
    """E2M1 codes (round to nearest even, saturating), `+0` for values that
    round to zero and for NaN, as int32."""
    magnitude = tl.abs(values)
    # Round to nearest even on the grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    code = (
        (magnitude > 0.25).to(tl.int32)
        + (magnitude >= 0.75).to(tl.int32)
        + (magnitude > 1.25).to(tl.int32)
        + (magnitude >= 1.75).to(tl.int32)
        + (magnitude > 2.5).to(tl.int32)
        + (magnitude >= 3.5).to(tl.int32)
        + (magnitude > 5.0).to(tl.int32)
    )
    return tl.where((values < 0) & (code != 0), code | 8, code)


@triton.jit
def _e2m1x2_cvt(high, low):
    """`_e2m1_codes(low) | _e2m1_codes(high) << 4` as uint8, through the SM100
    E2M1 conversion: it rounds to nearest even and saturates the same way, and
    the sign of every nibble that rounds to zero is cleared to match. Inputs
    must not be NaN (the conversion saturates NaN)."""
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 b0, b1, b2, b3;
        .reg .b32 w, m, s;
        cvt.rn.satfinite.e2m1x2.f32 b0, $1, $5;
        cvt.rn.satfinite.e2m1x2.f32 b1, $2, $6;
        cvt.rn.satfinite.e2m1x2.f32 b2, $3, $7;
        cvt.rn.satfinite.e2m1x2.f32 b3, $4, $8;
        mov.b32 w, {b0, b1, b2, b3};
        and.b32 m, w, 0x77777777;
        add.u32 s, m, 0x77777777;
        and.b32 s, s, 0x88888888;
        and.b32 s, s, w;
        or.b32 $0, m, s;
        }
        """,
        constraints="=r,r,r,r,r,r,r,r,r",
        args=[high, low],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _max_propagate_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _stage_nvfp4_chunk(
    raw,
    k0,
    rows,
    row_mask,
    x_scale,
    x_scale_inv,
    shared_sf_rows,
    x_out,
    x_sf,
    shared_x,
    shared_x_sf,
    x_stride_m: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    shared_x_stride_m: tl.constexpr,
    shared_x_sf_stride_m: tl.constexpr,
    shared_x_sf_stride_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SHARED_MXFP8: tl.constexpr,
    E2M1_CVT: tl.constexpr,
    NON_FINITE: tl.constexpr,
):
    """Quantize the BF16 `raw` = hidden[rows, k0:k0 + BLOCK_K] and store its
    NVFP4 bytes, E4M3 block scales and shared-expert input.

    Values are viewed as [row, group of 16, pair, value]: pairs share a byte,
    groups a scale; memory is accessed in contiguous rows. Scale bytes are
    stored one by one: byte j of a little-endian int32 scale word is the
    scale of the word's group j, as `_pack_bytes` packs them.
    """
    mask = row_mask[:, None]
    hidden = tl.reshape(raw.to(tl.float32), [BLOCK_M, BLOCK_K // 16, 8, 2])

    # E2M1 values with one E4M3 scale per 16, relative to the token scale; both
    # divisions equal tl.div_rn (see _div_rn_by_recip).
    values = _div_rn_by_recip(hidden, x_scale, x_scale_inv)
    block_scale = tl.max(tl.max(tl.abs(values), axis=3), axis=2) * (1.0 / 6.0)
    block_scale = tl.minimum(tl.maximum(block_scale, 2.0**-9), 448.0)
    block_scale = block_scale.to(tl.float8e4nv)
    scale = block_scale.to(tl.float32)
    scale_inv = tl.math.div_rn(1.0, scale)[:, :, None, None]
    values = _div_rn_by_recip(values, scale[:, :, None, None], scale_inv)
    if NON_FINITE:
        # NaN codes as +0; only rows with NaN or Inf produce NaN here.
        values = tl.where(values == values, values, 0.0)
    low, high = tl.split(values)
    if E2M1_CVT:
        packed = _e2m1x2_cvt(high, low)
    else:
        packed = (_e2m1_codes(low) | (_e2m1_codes(high) << 4)).to(tl.uint8)
    tl.store(
        x_out
        + rows[:, None] * x_stride_m
        + (k0 // 2 + tl.arange(0, BLOCK_K // 2))[None, :],
        tl.reshape(packed, [BLOCK_M, BLOCK_K // 2]),
        mask=mask,
    )
    tl.store(
        x_sf.to(tl.pointer_type(tl.uint8))
        + rows[:, None] * (x_sf_stride_m * 4)
        + (k0 // 16 + tl.arange(0, BLOCK_K // 16))[None, :],
        block_scale.to(tl.uint8, bitcast=True),
        mask=mask,
    )

    if shared_x is not None:
        shared_ptrs = (
            shared_x
            + rows[:, None] * shared_x_stride_m
            + (k0 + tl.arange(0, BLOCK_K))[None, :]
        )
        if SHARED_MXFP8:
            # As _quantize_fp8_groups: one UE8M0 scale per 32 values.
            group_amax = tl.max(tl.max(tl.abs(hidden), axis=3), axis=2)
            group_amax = tl.max(
                tl.reshape(group_amax, [BLOCK_M, BLOCK_K // 32, 2]), axis=2
            )
            group_amax = tl.maximum(group_amax, 1.0e-4)
            bits = tl.math.div_rn(group_amax, 448.0).to(tl.uint32, bitcast=True)
            scale_exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(
                tl.uint32
            )
            scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
            # tl.div_rn(1.0, 2^(e - 127)) bit for bit; 2^-127 is subnormal.
            inv_bits = tl.where(scale_exp == 254, 0x00400000, (254 - scale_exp) << 23)
            inv_scale = tl.reshape(
                tl.broadcast_to(
                    inv_bits.to(tl.float32, bitcast=True)[:, :, None],
                    [BLOCK_M, BLOCK_K // 32, 2],
                ),
                [BLOCK_M, BLOCK_K // 16],
            )
            fp8 = (hidden * inv_scale[:, :, None, None]).to(tl.float8e4nv)
            tl.store(shared_ptrs, tl.reshape(fp8, [BLOCK_M, BLOCK_K]), mask=mask)
            groups32 = tl.arange(0, BLOCK_K // 32)
            sf_offsets = (k0 // 128 + groups32 // 4) * (
                shared_x_sf_stride_k * 4
            ) + groups32 % 4
            tl.store(
                shared_x_sf.to(tl.pointer_type(tl.uint8))
                + shared_sf_rows[:, None] * (shared_x_sf_stride_m * 4)
                + sf_offsets[None, :],
                scale_exp.to(tl.uint8),
                mask=mask,
            )
        else:
            tl.store(shared_ptrs, raw, mask=mask)


@triton.jit
def _stage_nvfp4_rows(
    hidden_rows,
    rows,
    row_mask,
    split,
    first,
    amax,
    x_out,
    x_sf,
    x_scales,
    shared_x,
    shared_x_sf,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    shared_x_stride_m: tl.constexpr,
    shared_x_sf_stride_m: tl.constexpr,
    shared_x_sf_stride_k: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SHARED_BLOCK_M: tl.constexpr,
    SHARED_MXFP8: tl.constexpr,
    E2M1_CVT: tl.constexpr,
    NON_FINITE: tl.constexpr,
):
    """Store the token scales from `amax` (split 0) and quantize this split's
    chunks; `first` is the split's first chunk, already loaded."""
    x_scale = tl.maximum(amax, 1.0e-12) * _INV_NVFP4_MAX
    if split == 0:
        tl.store(x_scales + rows, x_scale, mask=row_mask)
    x_scale_inv = tl.math.div_rn(1.0, x_scale)[:, None, None, None]
    x_scale = x_scale[:, None, None, None]
    shared_sf_rows = rows
    if SHARED_MXFP8:
        shared_sf_rows = _shared_sf_row(rows, SHARED_BLOCK_M)
    _stage_nvfp4_chunk(
        first,
        split * BLOCK_K,
        rows,
        row_mask,
        x_scale,
        x_scale_inv,
        shared_sf_rows,
        x_out,
        x_sf,
        shared_x,
        shared_x_sf,
        x_stride_m,
        x_sf_stride_m,
        shared_x_stride_m,
        shared_x_sf_stride_m,
        shared_x_sf_stride_k,
        BLOCK_M,
        BLOCK_K,
        SHARED_MXFP8,
        E2M1_CVT,
        NON_FINITE,
    )
    for chunk in tl.range(1, hidden_size // (BLOCK_K * NUM_SPLITS)):
        k0 = (chunk * NUM_SPLITS + split) * BLOCK_K
        raw = tl.load(
            hidden_rows[:, None] + (k0 + tl.arange(0, BLOCK_K))[None, :] * hidden_stride_k,
            mask=row_mask[:, None],
            other=0.0,
        )
        _stage_nvfp4_chunk(
            raw,
            k0,
            rows,
            row_mask,
            x_scale,
            x_scale_inv,
            shared_sf_rows,
            x_out,
            x_sf,
            shared_x,
            shared_x_sf,
            x_stride_m,
            x_sf_stride_m,
            shared_x_stride_m,
            shared_x_sf_stride_m,
            shared_x_sf_stride_k,
            BLOCK_M,
            BLOCK_K,
            SHARED_MXFP8,
            E2M1_CVT,
            NON_FINITE,
        )


@triton.jit(do_not_specialize=["num_tokens", "l2_activation_scale"])
def _prepare_megamoe_nvfp4_rows_kernel(
    hidden_states,
    x_out,
    x_sf,
    x_scales,
    shared_x,
    shared_x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    logical_to_physical,
    replica_count,
    expert_load,
    record_enabled,
    num_unpadded_tokens,
    num_logical,
    map_stride,
    num_physical,
    w2_global_scale,
    l2_alpha,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    shared_x_stride_m: tl.constexpr,
    shared_x_sf_stride_m: tl.constexpr,
    shared_x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    num_tokens,
    l2_activation_scale,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    num_local_experts: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    AMAX_BLOCK_K: tl.constexpr,
    ROUTE_BLOCK_M: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    SHARED_BLOCK_M: tl.constexpr,
    SHARED_MXFP8: tl.constexpr,
    QUANTIZE: tl.constexpr,
    ROUTE: tl.constexpr,
    E2M1_CVT: tl.constexpr,
) -> None:
    """The bytes of `_prepare_megamoe_nvfp4_inputs_kernel`, from a 1D grid of
    quantize programs (BLOCK_M rows by every NUM_SPLITS-th BLOCK_K chunk) then
    routing programs (ROUTE_BLOCK_M rows of top-k), which run side by side
    instead of one after the other. QUANTIZE and ROUTE pick the halves, so the
    hidden-state half can run before routing is known.
    """
    pid = tl.program_id(0)
    num_quantize = 0
    if QUANTIZE:
        num_quantize = tl.cdiv(num_tokens, BLOCK_M) * NUM_SPLITS
        if l2_alpha is not None:
            experts = tl.arange(0, BLOCK_E)
            expert_mask = (experts < num_local_experts) & (pid == 0)
            alpha = tl.load(w2_global_scale + experts, mask=expert_mask)
            tl.store(l2_alpha + experts, alpha * l2_activation_scale, mask=expert_mask)

    if QUANTIZE:
        if pid < num_quantize:
            split = pid % NUM_SPLITS
            rows = (pid // NUM_SPLITS) * BLOCK_M + tl.arange(0, BLOCK_M)
            row_mask = rows < num_tokens
            rows = rows.to(tl.int64)
            hidden_rows = hidden_states + rows * hidden_stride_m
            # The split's first chunk is loaded with the row, for the amax.
            first = tl.load(
                hidden_rows[:, None]
                + (split * BLOCK_K + tl.arange(0, BLOCK_K))[None, :] * hidden_stride_k,
                mask=row_mask[:, None],
                other=0.0,
            )
            # The token scale needs the whole row: amax first, then quantize.
            row_max = tl.zeros([BLOCK_M, AMAX_BLOCK_K], dtype=tl.float32)
            for a0 in tl.static_range(0, hidden_size, AMAX_BLOCK_K):
                row_chunk = tl.load(
                    hidden_rows[:, None]
                    + (a0 + tl.arange(0, AMAX_BLOCK_K))[None, :] * hidden_stride_k,
                    mask=row_mask[:, None],
                    other=0.0,
                )
                row_max = _max_propagate_nan(row_max, tl.abs(row_chunk.to(tl.float32)))
            amax = tl.reduce(row_max, 1, _max_propagate_nan)
            # NaN and Inf fail the comparison.
            non_finite = (amax <= 3.4028234663852886e38) == 0
            if tl.max(non_finite.to(tl.int32), axis=0) != 0:
                # A NaN or Inf in the block: take the NaN-ignoring amax, as
                # _prepare_megamoe_nvfp4_inputs_kernel does, and code NaN as 0.
                row_max = tl.zeros([BLOCK_M, AMAX_BLOCK_K], dtype=tl.float32)
                for a0 in tl.static_range(0, hidden_size, AMAX_BLOCK_K):
                    row_chunk = tl.load(
                        hidden_rows[:, None]
                        + (a0 + tl.arange(0, AMAX_BLOCK_K))[None, :] * hidden_stride_k,
                        mask=row_mask[:, None],
                        other=0.0,
                    )
                    row_max = tl.maximum(row_max, tl.abs(row_chunk.to(tl.float32)))
                _stage_nvfp4_rows(
                    hidden_rows,
                    rows,
                    row_mask,
                    split,
                    first,
                    tl.max(row_max, axis=1),
                    x_out,
                    x_sf,
                    x_scales,
                    shared_x,
                    shared_x_sf,
                    hidden_stride_k,
                    x_stride_m,
                    x_sf_stride_m,
                    shared_x_stride_m,
                    shared_x_sf_stride_m,
                    shared_x_sf_stride_k,
                    hidden_size,
                    BLOCK_M,
                    BLOCK_K,
                    NUM_SPLITS,
                    SHARED_BLOCK_M,
                    SHARED_MXFP8,
                    E2M1_CVT,
                    True,
                )
            else:
                _stage_nvfp4_rows(
                    hidden_rows,
                    rows,
                    row_mask,
                    split,
                    first,
                    amax,
                    x_out,
                    x_sf,
                    x_scales,
                    shared_x,
                    shared_x_sf,
                    hidden_stride_k,
                    x_stride_m,
                    x_sf_stride_m,
                    shared_x_stride_m,
                    shared_x_sf_stride_m,
                    shared_x_sf_stride_k,
                    hidden_size,
                    BLOCK_M,
                    BLOCK_K,
                    NUM_SPLITS,
                    SHARED_BLOCK_M,
                    SHARED_MXFP8,
                    E2M1_CVT,
                    False,
                )

    if ROUTE:
        route_pid = pid - num_quantize
        if route_pid >= 0:
            token_ids = route_pid * ROUTE_BLOCK_M + tl.arange(0, ROUTE_BLOCK_M)
            _store_topk(
                token_ids,
                token_ids < num_tokens,
                topk_ids,
                topk_weights,
                is_padding,
                topk_idx_out,
                topk_weights_out,
                logical_to_physical,
                replica_count,
                expert_load,
                record_enabled,
                num_unpadded_tokens,
                num_logical,
                map_stride,
                num_physical,
                topk_ids_stride_m,
                topk_ids_stride_k,
                topk_weights_stride_m,
                topk_weights_stride_k,
                is_padding_stride_m,
                topk_idx_stride_m,
                topk_idx_stride_k,
                topk_weights_out_stride_m,
                topk_weights_out_stride_k,
                top_k,
                ROUTE_BLOCK_M,
                BLOCK_TOPK,
            )


def _resolve_hidden_quant(hidden_quant: QuantKey, block_k: int) -> tuple[int, bool]:
    """Map a hidden-state `QuantKey` to the kernel's `(GROUP_K, USE_UE8M0)`."""
    scale = hidden_quant.scale
    group_shape = scale.group_shape
    if (
        hidden_quant.dtype != FP8_DTYPE
        or not hidden_quant.symmetric
        or hidden_quant.scale2 is not None
        or scale.static
        or not group_shape.is_per_group()
        or block_k % group_shape.col != 0
        or scale.dtype not in (*_E8M0_SCALE_DTYPES, torch.float32)
    ):
        raise ValueError(
            "MegaMoE input staging requires symmetric dynamic fp8 "
            f"per-group quantization with a group size dividing {block_k} and "
            f"E8M0 or fp32 scales, got {hidden_quant}."
        )
    return group_shape.col, scale.dtype in _E8M0_SCALE_DTYPES


def _eplb_kernel_args(eplb_state: EplbLayerState | None) -> tuple:
    """Replica-map and load-recording kernel args; inert without EPLB."""
    if eplb_state is None or eplb_state.logical_to_physical_map is None:
        return (None,) * 5 + (0,) * 3
    mapping = eplb_state.logical_to_physical_map
    replicas = eplb_state.logical_replica_count
    load = eplb_state.expert_load_view
    record = eplb_state.should_record_tensor
    unpadded = eplb_state.num_unpadded_tokens_tensors
    assert replicas is not None and load is not None and record is not None
    assert unpadded is not None
    return (
        mapping,
        replicas,
        load,
        record,
        unpadded[dbo_current_ubatch_id()],
        mapping.shape[0],
        mapping.stride(0),
        load.numel(),
    )


def prepare_megamoe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor | None,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
    shared_x_sf: torch.Tensor | None = None,
    shared_block_m: int | None = None,
    hidden_quant: QuantKey | None = kMxfp8Dynamic,
    eplb_state: EplbLayerState | None = None,
) -> None:
    """Quantize hidden states and repack top-k routing for DeepGEMM MegaMoE.

    Args:
        hidden_states: Input activations of shape `[num_tokens, hidden_dim]`.
        topk_weights: Router top-k weights of shape `[num_tokens, top_k]`.
        topk_ids: Router top-k expert ids of shape `[num_tokens, top_k]`.
        x: Output buffer for the dispatched hidden states.
        x_sf: Output buffer for the hidden-state scale factors; `None` when
            `hidden_quant` is `None`.
        topk_idx_out: Output buffer for the repacked top-k expert ids.
        topk_weights_out: Output buffer for the repacked top-k weights.
        is_padding: Optional per-token mask; padded tokens are not routed.
        shared_x_sf: Optional output buffer for shared-expert scale factors.
            Must be given together with `shared_block_m`.
        shared_block_m: Block M used to lay out `shared_x_sf`.
        hidden_quant: Hidden-state quantization scheme. E8M0 scales are packed
            four per int32 into `x_sf`; fp32 scales are stored as their bits.
            `None` copies the hidden states into `x` unquantized.
        eplb_state: Optional EPLB layer state; maps ids to replicas and records load.

    """
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError(
            "MegaMoE input staging requires hidden_size to be a multiple of 128."
        )
    block_k = 128
    quantize = hidden_quant is not None
    if quantize != (x_sf is not None):
        raise ValueError("MegaMoE input staging needs x_sf iff hidden_quant is set.")
    group_k, use_ue8m0 = (
        (block_k, False)
        if hidden_quant is None
        else _resolve_hidden_quant(hidden_quant, block_k)
    )
    top_k = topk_ids.shape[1]
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )
    if (shared_x_sf is None) != (shared_block_m is None):
        raise ValueError(
            "MegaMoE shared input staging requires both shared_x_sf and shared_block_m."
        )
    if shared_x_sf is not None and not use_ue8m0 and group_k != block_k:
        raise ValueError(
            "MegaMoE shared input staging currently requires "
            f"UE8M0-packed or per-{block_k} fp32 hidden scales."
        )
    if shared_x_sf is not None:
        assert shared_block_m is not None
        if shared_block_m <= 0:
            raise ValueError("MegaMoE shared_block_m must be positive.")
        expected_sf_k = hidden_size // 128
        if shared_x_sf.ndim != 2 or shared_x_sf.shape[1] != expected_sf_k:
            raise ValueError(
                "MegaMoE shared_x_sf must have shape "
                f"(*, {expected_sf_k}), got {tuple(shared_x_sf.shape)}."
            )
        aligned_block_m = triton.cdiv(shared_block_m, 128) * 128
        required_rows = triton.cdiv(num_tokens, shared_block_m) * aligned_block_m
        if shared_x_sf.shape[0] < required_rows:
            raise ValueError(
                "MegaMoE shared_x_sf has insufficient rows: requires "
                f"{required_rows}, got {shared_x_sf.shape[0]}."
            )

    # On GB200, 8-row tiles win from 64 tokens and 32-row tiles from 1024 (1.3-1.6x
    # over 8 rows); smaller batches stay untiled.
    block_m = 32 if num_tokens >= 1024 else 8 if num_tokens >= 64 else 1
    grid = (triton.cdiv(num_tokens, block_m), triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(top_k)
    padding_stride_m = is_padding.stride(0) if is_padding is not None else 0
    _prepare_megamoe_inputs_kernel[grid](
        hidden_states,
        x,
        x_sf,
        shared_x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        *_eplb_kernel_args(eplb_state),
        hidden_states.stride(0),
        hidden_states.stride(1),
        x.stride(0),
        x.stride(1),
        x_sf.stride(0) if x_sf is not None else 0,
        x_sf.stride(1) if x_sf is not None else 0,
        shared_x_sf.stride(0) if shared_x_sf is not None else 0,
        shared_x_sf.stride(1) if shared_x_sf is not None else 0,
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        padding_stride_m,
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        num_tokens,
        hidden_size,
        top_k,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        GROUP_K=group_k,
        BLOCK_TOPK=block_topk,
        SHARED_BLOCK_M=shared_block_m or 1,
        QUANTIZE=quantize,
        USE_UE8M0=use_ue8m0,
        num_warps=4,
    )


def prepare_megamoe_nvfp4_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    x_scales: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
    shared_x: torch.Tensor | None = None,
    shared_x_sf: torch.Tensor | None = None,
    shared_block_m: int | None = None,
    w2_global_scale: torch.Tensor | None = None,
    l2_alpha: torch.Tensor | None = None,
    l2_activation_scale: float = 1.0,
    eplb_state: EplbLayerState | None = None,
) -> None:
    """Stage DeepGEMM NVFP4 MegaMoE inputs (`kNvfp4DynamicToken`).

    Each token is scaled by its own fp32 `x_scales` (row amax / 2688) and
    quantized to E2M1 with one E4M3 scale per 16 values.

    Args:
        hidden_states: BF16 activations of shape `[num_tokens, hidden_dim]`.
        topk_weights: Router top-k weights of shape `[num_tokens, top_k]`.
        topk_ids: Router top-k expert ids of shape `[num_tokens, top_k]`.
        x: Packed E2M1 output, `[num_tokens, hidden_dim // 2]`.
        x_sf: E4M3 scales packed four per int32, `[num_tokens, hidden_dim // 64]`.
        x_scales: Per-token fp32 dequantization scales, `[num_tokens]`.
        topk_idx_out: Output buffer for the repacked top-k expert ids.
        topk_weights_out: Output buffer for the repacked top-k weights.
        is_padding: Optional per-token mask; padded tokens are not routed.
        shared_x: Optional shared-expert input, fp8 (MXFP8 with `shared_x_sf`)
            or a BF16 copy of the hidden states.
        shared_x_sf: UE8M0 scales of an fp8 `shared_x` in the shared-expert
            TMA layout for `shared_block_m`.
        shared_block_m: Block M used to lay out `shared_x_sf`.
        w2_global_scale: Per-local-expert L2 weight global scales.
        l2_alpha: Output `w2_global_scale * l2_activation_scale`, the L2
            dequantization scale of each local expert. Written even without
            local tokens, as other ranks' tokens still reach these experts.
        l2_activation_scale: Global scale the kernel quantizes L2 inputs with.
        eplb_state: Optional EPLB layer state; maps ids to replicas and records load.

    """
    num_tokens, hidden_size = hidden_states.shape
    if hidden_size % 128 != 0:
        raise ValueError(
            "MegaMoE input staging requires hidden_size to be a multiple of 128."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )
    shared_mxfp8 = shared_x is not None and shared_x.dtype == FP8_DTYPE
    if shared_mxfp8 != (shared_x_sf is not None) or (shared_x_sf is None) != (
        shared_block_m is None
    ):
        raise ValueError(
            "MegaMoE NVFP4 staging needs shared_x_sf and shared_block_m exactly "
            "for an fp8 shared_x."
        )
    if (w2_global_scale is None) != (l2_alpha is None):
        raise ValueError("MegaMoE NVFP4 staging needs w2_global_scale with l2_alpha.")
    for name, tensor in (("x", x), ("x_sf", x_sf), ("shared_x", shared_x)):
        if tensor is not None and tensor.stride(-1) != 1:
            raise ValueError(f"MegaMoE NVFP4 staging requires a row-major {name}.")

    if nvfp4_row_staging_enabled():
        logger.info_once(
            "MegaMoE NVFP4 staging: row-block kernel "
            "(VLLM_MEGAMOE_NVFP4_ROW_STAGING=1)."
        )
        _launch_nvfp4_rows(
            num_tokens,
            hidden_states,
            x,
            x_sf,
            x_scales,
            shared_x,
            shared_x_sf,
            shared_block_m,
            w2_global_scale,
            l2_alpha,
            l2_activation_scale,
            topk_weights,
            topk_ids,
            topk_idx_out,
            topk_weights_out,
            is_padding,
            eplb_state,
        )
        return

    top_k = topk_ids.shape[1]
    num_local_experts = 0 if l2_alpha is None else l2_alpha.numel()
    # Up to 512 tokens each row is split over programs; larger batches fill the
    # GPU with one program per row looping over wider chunks.
    block_h = triton.next_power_of_2(hidden_size)
    if num_tokens <= 512:
        block_k = min(1024, block_h)
        num_splits = triton.cdiv(hidden_size, block_k)
    else:
        block_k = min(2048, block_h)
        num_splits = 1
    # One program even without tokens, to write l2_alpha.
    grid = (max(num_tokens, 1), num_splits)
    _prepare_megamoe_nvfp4_inputs_kernel[grid](
        hidden_states,
        x.view(torch.uint8),
        x_sf,
        x_scales,
        shared_x,
        shared_x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        *_eplb_kernel_args(eplb_state),
        w2_global_scale,
        l2_alpha,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x.stride(0),
        x_sf.stride(0),
        shared_x.stride(0) if shared_x is not None else 0,
        shared_x_sf.stride(0) if shared_x_sf is not None else 0,
        shared_x_sf.stride(1) if shared_x_sf is not None else 0,
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        is_padding.stride(0) if is_padding is not None else 0,
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        num_tokens,
        l2_activation_scale,
        hidden_size,
        top_k,
        num_local_experts,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        BLOCK_TOPK=triton.next_power_of_2(top_k),
        BLOCK_E=triton.next_power_of_2(max(num_local_experts, 1)),
        SHARED_BLOCK_M=shared_block_m or 1,
        SHARED_MXFP8=shared_mxfp8,
        num_warps=4,
    )


# Tokens per routing program of the row-block kernel.
_ROUTE_BLOCK_M = 64


def _nvfp4_rows_config(num_tokens: int, hidden_size: int) -> tuple[int, int, int, int]:
    """`(BLOCK_M, BLOCK_K, NUM_SPLITS, num_warps)` of the row-block kernel,
    by token band, from a GB300 sweep at hidden 6144 (every config writes
    the same bytes): 8-warp programs from 5 to 192 tokens, then one row per
    program over 2048-value chunks."""
    if num_tokens <= 4:
        block_m, block_k, chunks_per_program, num_warps = 1, 1024, 1, 4
    elif num_tokens <= 108 or 144 < num_tokens <= 192:
        block_m, block_k, chunks_per_program, num_warps = 2, 1024, 2, 8
    elif num_tokens <= 192:
        block_m, block_k, chunks_per_program, num_warps = 1, 2048, 3, 8
    else:
        block_m, block_k, chunks_per_program, num_warps = 1, 2048, 0, 4
    block_k = min(block_k, hidden_size & -hidden_size)  # a power of two dividing it
    num_chunks = hidden_size // block_k
    if chunks_per_program == 0:
        num_splits = 1  # each program loops over its rows' chunks
    elif num_chunks % chunks_per_program == 0:
        num_splits = num_chunks // chunks_per_program
    else:
        num_splits = num_chunks
    return block_m, block_k, num_splits, num_warps


@functools.cache
def _e2m1_cvt_supported() -> bool:
    """SM100-family GPUs convert fp32 pairs to E2M1 in one instruction."""
    from vllm.platforms import current_platform

    return current_platform.is_device_capability_family(100)


def _launch_nvfp4_rows(
    num_tokens: int,
    hidden_states: torch.Tensor | None,
    x: torch.Tensor | None,
    x_sf: torch.Tensor | None,
    x_scales: torch.Tensor | None,
    shared_x: torch.Tensor | None,
    shared_x_sf: torch.Tensor | None,
    shared_block_m: int | None,
    w2_global_scale: torch.Tensor | None,
    l2_alpha: torch.Tensor | None,
    l2_activation_scale: float,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    topk_idx_out: torch.Tensor | None,
    topk_weights_out: torch.Tensor | None,
    is_padding: torch.Tensor | None,
    eplb_state: EplbLayerState | None,
) -> None:
    """Launch `_prepare_megamoe_nvfp4_rows_kernel` for the halves given: the
    hidden-state half when `hidden_states` is set, routing when `topk_ids` is."""
    quantize = hidden_states is not None
    route = topk_ids is not None
    hidden_size = hidden_states.shape[1] if quantize else 128
    block_m, block_k, num_splits, num_warps = _nvfp4_rows_config(
        num_tokens, hidden_size
    )
    num_quantize = triton.cdiv(num_tokens, block_m) * num_splits if quantize else 0
    num_route = triton.cdiv(num_tokens, _ROUTE_BLOCK_M) if route else 0
    top_k = topk_ids.shape[1] if route else 1
    num_local_experts = 0 if l2_alpha is None else l2_alpha.numel()

    def stride(tensor: torch.Tensor | None, dim: int) -> int:
        return tensor.stride(dim) if tensor is not None else 0

    # One program even without tokens, to write l2_alpha.
    grid = (max(num_quantize + num_route, 1),)
    _prepare_megamoe_nvfp4_rows_kernel[grid](
        hidden_states,
        x.view(torch.uint8) if x is not None else None,
        x_sf,
        x_scales,
        shared_x,
        shared_x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        *_eplb_kernel_args(eplb_state if route else None),
        w2_global_scale,
        l2_alpha,
        stride(hidden_states, 0),
        stride(hidden_states, 1),
        stride(x, 0),
        stride(x_sf, 0),
        stride(shared_x, 0),
        stride(shared_x_sf, 0),
        stride(shared_x_sf, 1),
        stride(topk_ids, 0),
        stride(topk_ids, 1),
        stride(topk_weights, 0),
        stride(topk_weights, 1),
        stride(is_padding, 0),
        stride(topk_idx_out, 0),
        stride(topk_idx_out, 1),
        stride(topk_weights_out, 0),
        stride(topk_weights_out, 1),
        num_tokens,
        l2_activation_scale,
        hidden_size,
        top_k,
        num_local_experts,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        AMAX_BLOCK_K=min(2048, hidden_size & -hidden_size),
        ROUTE_BLOCK_M=_ROUTE_BLOCK_M,
        BLOCK_TOPK=triton.next_power_of_2(top_k),
        BLOCK_E=triton.next_power_of_2(max(num_local_experts, 1)),
        SHARED_BLOCK_M=shared_block_m or 1,
        SHARED_MXFP8=shared_x is not None and shared_x.dtype == FP8_DTYPE,
        QUANTIZE=quantize,
        ROUTE=route,
        # The conversion packs four output bytes per thread.
        E2M1_CVT=quantize
        and _e2m1_cvt_supported()
        and block_m * block_k // 2 >= 4 * 32 * num_warps,
        num_warps=num_warps,
    )


def stage_megamoe_nvfp4_hidden_states(
    hidden_states: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    x_scales: torch.Tensor,
    shared_x: torch.Tensor | None = None,
    shared_x_sf: torch.Tensor | None = None,
    shared_block_m: int | None = None,
    w2_global_scale: torch.Tensor | None = None,
    l2_alpha: torch.Tensor | None = None,
    l2_activation_scale: float = 1.0,
) -> None:
    """The hidden-state half of `prepare_megamoe_nvfp4_inputs`, with its
    arguments and bytes: `x`, `x_sf`, `x_scales`, the shared-expert input and
    `l2_alpha`. It needs no routing, so it can run alongside the router;
    `stage_megamoe_topk` stages the rest."""
    num_tokens, hidden_size = hidden_states.shape
    if hidden_size % 128 != 0:
        raise ValueError(
            "MegaMoE input staging requires hidden_size to be a multiple of 128."
        )
    shared_mxfp8 = shared_x is not None and shared_x.dtype == FP8_DTYPE
    if shared_mxfp8 != (shared_x_sf is not None) or (shared_x_sf is None) != (
        shared_block_m is None
    ):
        raise ValueError(
            "MegaMoE NVFP4 staging needs shared_x_sf and shared_block_m exactly "
            "for an fp8 shared_x."
        )
    if (w2_global_scale is None) != (l2_alpha is None):
        raise ValueError("MegaMoE NVFP4 staging needs w2_global_scale with l2_alpha.")
    for name, tensor in (("x", x), ("x_sf", x_sf), ("shared_x", shared_x)):
        if tensor is not None and tensor.stride(-1) != 1:
            raise ValueError(f"MegaMoE NVFP4 staging requires a row-major {name}.")
    _launch_nvfp4_rows(
        num_tokens,
        hidden_states,
        x,
        x_sf,
        x_scales,
        shared_x,
        shared_x_sf,
        shared_block_m,
        w2_global_scale,
        l2_alpha,
        l2_activation_scale,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def stage_megamoe_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
    eplb_state: EplbLayerState | None = None,
) -> None:
    """The routing half of `prepare_megamoe_nvfp4_inputs`: repack the top-k
    ids (EPLB-mapped, padding unrouted, load recorded) and weights."""
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )
    _launch_nvfp4_rows(
        topk_ids.shape[0],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        1.0,
        topk_weights,
        topk_ids,
        topk_idx_out,
        topk_weights_out,
        is_padding,
        eplb_state,
    )


# ---------------------------------------------------------------------------
# Fused routing (VLLM_MEGAMOE_FUSED_ROUTING): the router GEMM's split-K
# reduction, the sigmoid top-k gating and the NVFP4 staging in one kernel.


@functools.cache
def megamoe_fused_routing_enabled() -> bool:
    """`VLLM_MEGAMOE_FUSED_ROUTING=1` reduces the router GEMM's split-K
    partials, selects the sigmoid top-k experts and stages the NVFP4 MegaMoE
    inputs in one kernel (`route_and_stage_megamoe_nvfp4`) instead of three
    (`_splitk_reduce_kernel`, `topk_sigmoid`, the staging kernel), with the same
    bytes."""
    return _env_flag("VLLM_MEGAMOE_FUSED_ROUTING")


class MegaMoESigmoidRouting(NamedTuple):
    """Sigmoid top-k routing still to be selected from the router GEMM's
    split-K partials (`GateLinear.forward_split_k_partials`), with
    `fused_topk_bias(scoring_func="sigmoid")`'s parameters."""

    router_partials: torch.Tensor
    e_score_correction_bias: torch.Tensor
    top_k: int
    renormalize: bool
    routed_scaling_factor: float


# The order in which `_splitk_reduce_kernel` (bf16x3_router_gemm_sm100.py) adds
# the split-K partials of one logit, by its BLOCK_S (the split count rounded up
# to a power of two; the partials past the count are +0.0): the bits of the
# split index, folded first to last, each fold adding the two partial sums
# whose indices differ in that bit. On that kernel's blocked layout tl.sum folds
# a thread's repeats pairwise (low bit first), then the lanes and then the warps
# along the split (high bit first). Read from the kernel's PTX (Triton 3.8) for
# every config `_splitk_reduce_config` returns; checked on the GPU against the
# kernel before the fused path is used (`megamoe_fused_routing_verified`).
_SPLITK_REDUCE_FOLD_ORDER: dict[int, tuple[int, ...]] = {
    1: (),
    2: (0,),
    4: (1, 0),
    8: (1, 2, 0),
    16: (1, 2, 3, 0),
    32: (1, 2, 3, 4, 0),
    64: (4, 5, 1, 0, 3, 2),
}
MAX_FUSED_ROUTING_SPLIT_K = max(_SPLITK_REDUCE_FOLD_ORDER)
# topkGating marks the expert it picked with this choice value (scores are in
# [0, 1]); kept, so that rows whose biased scores fall below it route the same.
_TOPK_GATING_PICKED = tl.constexpr(-10000.0)


@triton.jit
def _split_partial(ptrs, mask, S: tl.constexpr, k_splits, split_stride):
    """Split-K partial `S` of the logits at `ptrs`; +0.0 past the split count,
    as `_splitk_reduce_kernel` loads it."""
    return tl.load(ptrs + S * split_stride, mask=mask & (S < k_splits), other=0.0)


@triton.jit
def _fold1(ptrs, mask, S: tl.constexpr, k_splits, split_stride, B0: tl.constexpr):
    lo = _split_partial(ptrs, mask, S, k_splits, split_stride)
    hi = _split_partial(ptrs, mask, S + (1 << B0), k_splits, split_stride)
    return lo + hi


@triton.jit
def _fold2(
    ptrs,
    mask,
    S: tl.constexpr,
    k_splits,
    split_stride,
    B0: tl.constexpr,
    B1: tl.constexpr,
):
    lo = _fold1(ptrs, mask, S, k_splits, split_stride, B0)
    hi = _fold1(ptrs, mask, S + (1 << B1), k_splits, split_stride, B0)
    return lo + hi


@triton.jit
def _fold3(
    ptrs,
    mask,
    S: tl.constexpr,
    k_splits,
    split_stride,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
):
    lo = _fold2(ptrs, mask, S, k_splits, split_stride, B0, B1)
    hi = _fold2(ptrs, mask, S + (1 << B2), k_splits, split_stride, B0, B1)
    return lo + hi


@triton.jit
def _fold4(
    ptrs,
    mask,
    S: tl.constexpr,
    k_splits,
    split_stride,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
):
    lo = _fold3(ptrs, mask, S, k_splits, split_stride, B0, B1, B2)
    hi = _fold3(ptrs, mask, S + (1 << B3), k_splits, split_stride, B0, B1, B2)
    return lo + hi


@triton.jit
def _fold5(
    ptrs,
    mask,
    S: tl.constexpr,
    k_splits,
    split_stride,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
    B4: tl.constexpr,
):
    lo = _fold4(ptrs, mask, S, k_splits, split_stride, B0, B1, B2, B3)
    hi = _fold4(ptrs, mask, S + (1 << B4), k_splits, split_stride, B0, B1, B2, B3)
    return lo + hi


@triton.jit
def _fold6(
    ptrs,
    mask,
    S: tl.constexpr,
    k_splits,
    split_stride,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
    B4: tl.constexpr,
    B5: tl.constexpr,
):
    lo = _fold5(ptrs, mask, S, k_splits, split_stride, B0, B1, B2, B3, B4)
    hi = _fold5(ptrs, mask, S + (1 << B5), k_splits, split_stride, B0, B1, B2, B3, B4)
    return lo + hi


@triton.jit
def _reduce_router_partials(
    ptrs,
    mask,
    k_splits,
    split_stride,
    NUM_FOLDS: tl.constexpr,
    F0: tl.constexpr,
    F1: tl.constexpr,
    F2: tl.constexpr,
    F3: tl.constexpr,
    F4: tl.constexpr,
    F5: tl.constexpr,
):
    """The router logits at `ptrs`: their split-K partials added in
    `_splitk_reduce_kernel`'s order (`_SPLITK_REDUCE_FOLD_ORDER`, F0 first).
    Every element sums its own partials, so any layout gives the same bits."""
    if NUM_FOLDS == 0:
        logits = tl.load(ptrs, mask=mask, other=0.0)
    elif NUM_FOLDS == 1:
        logits = _fold1(ptrs, mask, 0, k_splits, split_stride, F0)
    elif NUM_FOLDS == 2:
        logits = _fold2(ptrs, mask, 0, k_splits, split_stride, F0, F1)
    elif NUM_FOLDS == 3:
        logits = _fold3(ptrs, mask, 0, k_splits, split_stride, F0, F1, F2)
    elif NUM_FOLDS == 4:
        logits = _fold4(ptrs, mask, 0, k_splits, split_stride, F0, F1, F2, F3)
    elif NUM_FOLDS == 5:
        logits = _fold5(ptrs, mask, 0, k_splits, split_stride, F0, F1, F2, F3, F4)
    else:
        logits = _fold6(ptrs, mask, 0, k_splits, split_stride, F0, F1, F2, F3, F4, F5)
    return logits


@triton.jit
def _topk_gating_sigmoid(logits, EX2_ASM: tl.constexpr):
    """topkGating's score `1.0f / (1.0f + __expf(-x))` (topk_softmax_kernels.cu,
    built without fast math): __expf is ex2.approx.f32 of the product with
    log2(e) rounded to fp32, and the division is correctly rounded. NaN and
    Inf scores become 0, as there. EX2_ASM=False (Triton's interpreter) takes
    tl.math.exp2 instead of the instruction."""
    scaled = -logits * 1.4426950408889634
    if EX2_ASM:
        exp = tl.inline_asm_elementwise(
            asm="ex2.approx.f32 $0, $1;",
            constraints="=r,r",
            args=[scaled],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        exp = tl.math.exp2(scaled)
    scores = tl.math.div_rn(1.0, 1.0 + exp)
    # NaN and Inf fail the comparison.
    return tl.where(tl.abs(scores) <= 3.4028234663852886e38, scores, 0.0)


@triton.jit
def _pick_expert(choice0, expert0, score0, choice1, expert1, score1):
    """The larger choice, the lower expert on equal ones: topkGating's order. A
    total order on non-NaN choices (+0 and -0 tie), so any reduction tree finds
    the same winner; its score rides along."""
    take1 = (choice1 > choice0) | ((choice1 == choice0) & (expert1 < expert0))
    return (
        tl.where(take1, choice1, choice0),
        tl.where(take1, expert1, expert0),
        tl.where(take1, score1, score0),
    )


@triton.jit
def _sigmoid_topk_rows(
    logits,
    bias,
    experts,
    routed_scaling_factor,
    ROWS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    EX2_ASM: tl.constexpr,
):
    """topkGating's sigmoid top-k of each row of `logits` [ROWS, NUM_EXPERTS]
    with a correction bias: expert ids (int32) and weights by slot.

    Each slot takes the largest biased score, the lowest expert among equal
    ones (topkGating's tie rule: a strict `>` scan, then the lower expert on
    ties), and the winner's biased score becomes -10000 for the next slots.
    One reduction per slot in that order (`_pick_expert`; vllm#62912 selects by
    the same rule with redux.sync). The weights are the unbiased scores, times
    routed_scaling_factor over their sum in slot order (1 if not positive).
    The bias must have no NaN (topkGating's result then depends on its lanes).
    """
    scores = _topk_gating_sigmoid(logits, EX2_ASM)
    choice = scores + bias[None, :]
    slots = tl.arange(0, BLOCK_TOPK)[None, :]
    ids = tl.zeros([ROWS, BLOCK_TOPK], dtype=tl.int32)
    weights = tl.zeros([ROWS, BLOCK_TOPK], dtype=tl.float32)
    selected_sum = tl.zeros([ROWS], dtype=tl.float32)
    expert_ids = tl.broadcast_to(experts[None, :], [ROWS, NUM_EXPERTS])
    for k in tl.static_range(TOP_K):
        _, expert, weight = tl.reduce((choice, expert_ids, scores), 1, _pick_expert)
        picked = experts[None, :] == expert[:, None]
        ids = tl.where(slots == k, expert[:, None], ids)
        weights = tl.where(slots == k, weight[:, None], weights)
        selected_sum = selected_sum + weight
        choice = tl.where(picked, _TOPK_GATING_PICKED, choice)
    if RENORMALIZE:
        denom = tl.where(selected_sum > 0.0, selected_sum, 1.0)
        scale = tl.math.div_rn(tl.zeros_like(denom) + routed_scaling_factor, denom)
        weights = weights * scale[:, None]
    else:
        weights = weights * routed_scaling_factor
    return ids, weights


@triton.jit(
    do_not_specialize=["num_tokens", "routed_scaling_factor", "l2_activation_scale"]
)
def _route_and_stage_megamoe_nvfp4_kernel(
    router_partials,
    e_score_correction_bias,
    router_logits,
    topk_weights,
    topk_ids,
    hidden_states,
    x_out,
    x_sf,
    x_scales,
    shared_x,
    shared_x_sf,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    logical_to_physical,
    replica_count,
    expert_load,
    record_enabled,
    num_unpadded_tokens,
    num_logical,
    map_stride,
    num_physical,
    w2_global_scale,
    l2_alpha,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    shared_x_stride_m: tl.constexpr,
    shared_x_sf_stride_m: tl.constexpr,
    shared_x_sf_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    num_tokens,
    split_stride,
    k_splits,
    routed_scaling_factor,
    l2_activation_scale,
    hidden_size: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    NUM_FOLDS: tl.constexpr,
    F0: tl.constexpr,
    F1: tl.constexpr,
    F2: tl.constexpr,
    F3: tl.constexpr,
    F4: tl.constexpr,
    F5: tl.constexpr,
    GATE_TOKENS: tl.constexpr,
    num_local_experts: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    AMAX_BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    SHARED_BLOCK_M: tl.constexpr,
    SHARED_MXFP8: tl.constexpr,
    QUANTIZE: tl.constexpr,
    E2M1_CVT: tl.constexpr,
    EX2_ASM: tl.constexpr,
    USE_PDL: tl.constexpr,
) -> None:
    """One launch for the router's split-K reduction, its sigmoid top-k and the
    NVFP4 MegaMoE staging: a 1D grid of gating programs (GATE_TOKENS tokens, one
    logit per thread) then, with QUANTIZE, `_prepare_megamoe_nvfp4_rows_kernel`'s
    quantize programs, which run alongside.

    The bytes are those of `_splitk_reduce_kernel`, `topk_sigmoid` (topkGating)
    and `prepare_megamoe_nvfp4_inputs`: the reduction adds in the reduce
    kernel's order, the gating reproduces topkGating's arithmetic and tie rules,
    and the stores are `_store_topk`'s and the row-block kernel's.
    """
    pid = tl.program_id(0)
    if USE_PDL:
        # The partials come from the router GEMM just before.
        tl.extra.cuda.gdc_wait()
    num_gate = tl.cdiv(num_tokens, GATE_TOKENS)
    if QUANTIZE:
        if l2_alpha is not None:
            local = tl.arange(0, BLOCK_E)
            local_mask = (local < num_local_experts) & (pid == 0)
            alpha = tl.load(w2_global_scale + local, mask=local_mask)
            tl.store(l2_alpha + local, alpha * l2_activation_scale, mask=local_mask)

    if pid < num_gate:
        tokens = pid * GATE_TOKENS + tl.arange(0, GATE_TOKENS)
        token_mask = tokens < num_tokens
        experts = tl.arange(0, NUM_EXPERTS)
        offsets = tokens[:, None] * NUM_EXPERTS + experts[None, :]
        logit_mask = token_mask[:, None] & (experts[None, :] < NUM_EXPERTS)
        logits = _reduce_router_partials(
            router_partials + offsets,
            logit_mask,
            k_splits,
            split_stride,
            NUM_FOLDS,
            F0,
            F1,
            F2,
            F3,
            F4,
            F5,
        )
        if router_logits is not None:
            tl.store(router_logits + offsets, logits, mask=logit_mask)
        bias = tl.load(e_score_correction_bias + experts)
        ids, weights = _sigmoid_topk_rows(
            logits,
            bias,
            experts,
            routed_scaling_factor,
            GATE_TOKENS,
            NUM_EXPERTS,
            TOP_K,
            BLOCK_TOPK,
            RENORMALIZE,
            EX2_ASM,
        )

        token_is_padding = tl.full((GATE_TOKENS,), False, tl.int1)
        if is_padding is not None:
            token_is_padding = tl.load(
                is_padding + tokens * is_padding_stride_m,
                mask=token_mask,
                other=True,
            )
        slots = tl.arange(0, BLOCK_TOPK)
        topk_mask = token_mask[:, None] & (slots[None, :] < TOP_K)
        # As topkGating: padding rows get expert -1 and keep their weights.
        ids = tl.where(token_is_padding[:, None], -1, ids)
        if topk_ids is not None:
            tl.store(
                topk_ids + tokens[:, None] * TOP_K + slots[None, :],
                ids,
                mask=topk_mask,
            )
        if topk_weights is not None:
            tl.store(
                topk_weights + tokens[:, None] * TOP_K + slots[None, :],
                weights,
                mask=topk_mask,
            )
        # As _store_topk: int64 ids mapped to EPLB replicas (load recorded),
        # padding rows unrouted with weight 0.
        staged_ids = ids.to(tl.int64)
        if logical_to_physical is not None:
            staged_ids = eplb_map_and_record(
                staged_ids,
                tokens[:, None],
                topk_mask,
                logical_to_physical,
                replica_count,
                num_logical,
                map_stride,
                expert_load,
                num_physical,
                record_enabled,
                num_unpadded_tokens,
            )
        tl.store(
            topk_idx_out
            + tokens[:, None] * topk_idx_stride_m
            + slots[None, :] * topk_idx_stride_k,
            staged_ids,
            mask=topk_mask,
        )
        tl.store(
            topk_weights_out
            + tokens[:, None] * topk_weights_out_stride_m
            + slots[None, :] * topk_weights_out_stride_k,
            tl.where(token_is_padding[:, None], 0.0, weights),
            mask=topk_mask,
        )
    else:
        if QUANTIZE:
            # _prepare_megamoe_nvfp4_rows_kernel's quantize programs.
            quantize_pid = pid - num_gate
            if quantize_pid < tl.cdiv(num_tokens, BLOCK_M) * NUM_SPLITS:
                split = quantize_pid % NUM_SPLITS
                rows = (quantize_pid // NUM_SPLITS) * BLOCK_M + tl.arange(0, BLOCK_M)
                row_mask = rows < num_tokens
                rows = rows.to(tl.int64)
                hidden_rows = hidden_states + rows * hidden_stride_m
                first = tl.load(
                    hidden_rows[:, None]
                    + (split * BLOCK_K + tl.arange(0, BLOCK_K))[None, :]
                    * hidden_stride_k,
                    mask=row_mask[:, None],
                    other=0.0,
                )
                row_max = tl.zeros([BLOCK_M, AMAX_BLOCK_K], dtype=tl.float32)
                for a0 in tl.static_range(0, hidden_size, AMAX_BLOCK_K):
                    row_chunk = tl.load(
                        hidden_rows[:, None]
                        + (a0 + tl.arange(0, AMAX_BLOCK_K))[None, :] * hidden_stride_k,
                        mask=row_mask[:, None],
                        other=0.0,
                    )
                    row_max = _max_propagate_nan(
                        row_max, tl.abs(row_chunk.to(tl.float32))
                    )
                amax = tl.reduce(row_max, 1, _max_propagate_nan)
                non_finite = (amax <= 3.4028234663852886e38) == 0
                if tl.max(non_finite.to(tl.int32), axis=0) != 0:
                    row_max = tl.zeros([BLOCK_M, AMAX_BLOCK_K], dtype=tl.float32)
                    for a0 in tl.static_range(0, hidden_size, AMAX_BLOCK_K):
                        row_chunk = tl.load(
                            hidden_rows[:, None]
                            + (a0 + tl.arange(0, AMAX_BLOCK_K))[None, :]
                            * hidden_stride_k,
                            mask=row_mask[:, None],
                            other=0.0,
                        )
                        row_max = tl.maximum(row_max, tl.abs(row_chunk.to(tl.float32)))
                    _stage_nvfp4_rows(
                        hidden_rows,
                        rows,
                        row_mask,
                        split,
                        first,
                        tl.max(row_max, axis=1),
                        x_out,
                        x_sf,
                        x_scales,
                        shared_x,
                        shared_x_sf,
                        hidden_stride_k,
                        x_stride_m,
                        x_sf_stride_m,
                        shared_x_stride_m,
                        shared_x_sf_stride_m,
                        shared_x_sf_stride_k,
                        hidden_size,
                        BLOCK_M,
                        BLOCK_K,
                        NUM_SPLITS,
                        SHARED_BLOCK_M,
                        SHARED_MXFP8,
                        E2M1_CVT,
                        True,
                    )
                else:
                    _stage_nvfp4_rows(
                        hidden_rows,
                        rows,
                        row_mask,
                        split,
                        first,
                        amax,
                        x_out,
                        x_sf,
                        x_scales,
                        shared_x,
                        shared_x_sf,
                        hidden_stride_k,
                        x_stride_m,
                        x_sf_stride_m,
                        shared_x_stride_m,
                        shared_x_sf_stride_m,
                        shared_x_sf_stride_k,
                        hidden_size,
                        BLOCK_M,
                        BLOCK_K,
                        NUM_SPLITS,
                        SHARED_BLOCK_M,
                        SHARED_MXFP8,
                        E2M1_CVT,
                        False,
                    )


@functools.cache
def _launch_with_pdl() -> bool:
    """Programmatic dependent launch (SM90+): the kernel may start while the
    router GEMM finishes, and waits for it before reading its partials."""
    from vllm.platforms import current_platform

    return current_platform.is_cuda() and current_platform.has_device_capability(90)


@functools.cache
def _ex2_asm_supported() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_cuda()


def route_and_stage_megamoe_nvfp4(
    routing: MegaMoESigmoidRouting,
    hidden_states: torch.Tensor | None,
    x: torch.Tensor | None,
    x_sf: torch.Tensor | None,
    x_scales: torch.Tensor | None,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
    shared_x: torch.Tensor | None = None,
    shared_x_sf: torch.Tensor | None = None,
    shared_block_m: int | None = None,
    w2_global_scale: torch.Tensor | None = None,
    l2_alpha: torch.Tensor | None = None,
    l2_activation_scale: float = 1.0,
    eplb_state: EplbLayerState | None = None,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    router_logits: torch.Tensor | None = None,
) -> None:
    """`prepare_megamoe_nvfp4_inputs` with the routing selected in the same
    kernel from the router GEMM's split-K partials, as `splitk_reduce_triton`
    then `fused_topk_bias(scoring_func="sigmoid")` select it: every byte of
    `x`, `x_sf`, `x_scales`, the shared-expert input, `l2_alpha`,
    `topk_idx_out`, `topk_weights_out` and the EPLB load is the same.

    `hidden_states` None stages only the routing (the hidden-state half ran in
    `stage_megamoe_nvfp4_hidden_states`). `topk_weights` / `topk_ids`
    (`[num_tokens, top_k]` fp32 / int32) and `router_logits`
    (`[num_tokens, num_experts]` fp32), if given, receive the logical top-k and
    the logits.
    """
    partials = routing.router_partials
    bias = routing.e_score_correction_bias
    split_k, num_tokens, num_experts = partials.shape
    top_k = routing.top_k
    fold = _SPLITK_REDUCE_FOLD_ORDER.get(triton.next_power_of_2(split_k))
    if (
        fold is None
        or partials.dtype != torch.float32
        or not partials.is_contiguous()
        or num_experts & (num_experts - 1)
        or not 0 < top_k <= num_experts
        or bias.dtype != torch.float32
        or bias.shape != (num_experts,)
        or not bias.is_contiguous()
    ):
        raise ValueError(
            "MegaMoE fused routing needs contiguous fp32 router partials with at "
            f"most {MAX_FUSED_ROUTING_SPLIT_K} splits and a power-of-two expert "
            "count, top_k within it and a contiguous fp32 bias per expert; got "
            f"partials {tuple(partials.shape)} {partials.dtype}, top_k {top_k}, "
            f"bias {tuple(bias.shape)} {bias.dtype}."
        )
    for name, tensor, shape, dtype in (
        ("topk_weights", topk_weights, (num_tokens, top_k), torch.float32),
        ("topk_ids", topk_ids, (num_tokens, top_k), torch.int32),
        ("router_logits", router_logits, (num_tokens, num_experts), torch.float32),
    ):
        if tensor is not None and (
            tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"MegaMoE fused routing writes {name} as a contiguous {dtype} "
                f"tensor of shape {shape}."
            )
    if topk_idx_out.shape != (num_tokens, top_k) or topk_weights_out.shape != (
        num_tokens,
        top_k,
    ):
        raise ValueError("MegaMoE fused routing: top-k outputs of the wrong shape.")

    quantize = hidden_states is not None
    if quantize:
        assert x is not None and x_sf is not None and x_scales is not None
        hidden_size = hidden_states.shape[1]
        if hidden_states.shape[0] != num_tokens or hidden_size % 128 != 0:
            raise ValueError(
                "MegaMoE input staging requires hidden_size to be a multiple of "
                "128 and one hidden row per token."
            )
        shared_mxfp8 = shared_x is not None and shared_x.dtype == FP8_DTYPE
        if shared_mxfp8 != (shared_x_sf is not None) or (shared_x_sf is None) != (
            shared_block_m is None
        ):
            raise ValueError(
                "MegaMoE NVFP4 staging needs shared_x_sf and shared_block_m exactly "
                "for an fp8 shared_x."
            )
        if (w2_global_scale is None) != (l2_alpha is None):
            raise ValueError(
                "MegaMoE NVFP4 staging needs w2_global_scale with l2_alpha."
            )
        for name, tensor in (("x", x), ("x_sf", x_sf), ("shared_x", shared_x)):
            if tensor is not None and tensor.stride(-1) != 1:
                raise ValueError(f"MegaMoE NVFP4 staging requires a row-major {name}.")
        block_m, block_k, num_splits, num_warps = _nvfp4_rows_config(
            num_tokens, hidden_size
        )
        num_quantize = triton.cdiv(num_tokens, block_m) * num_splits
    else:
        hidden_size, shared_mxfp8 = 128, False
        block_m, block_k, num_splits, num_warps, num_quantize = 1, 128, 1, 4, 0
    num_local_experts = 0 if l2_alpha is None or not quantize else l2_alpha.numel()
    # One logit per thread: the split-K sum spreads over the most SMs, and each
    # thread keeps its BLOCK_S partials in flight.
    gate_tokens = max(1, 32 * num_warps // num_experts)
    num_gate = triton.cdiv(num_tokens, gate_tokens)
    folds = fold + (0,) * (6 - len(fold))

    def stride(tensor: torch.Tensor | None, dim: int) -> int:
        return tensor.stride(dim) if tensor is not None and quantize else 0

    use_pdl = _launch_with_pdl()
    # One program even without tokens, to write l2_alpha.
    grid = (max(num_gate + num_quantize, 1),)
    _route_and_stage_megamoe_nvfp4_kernel[grid](
        partials,
        bias,
        router_logits,
        topk_weights,
        topk_ids,
        hidden_states,
        x.view(torch.uint8) if quantize else None,
        x_sf if quantize else None,
        x_scales if quantize else None,
        shared_x if quantize else None,
        shared_x_sf if quantize else None,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        *_eplb_kernel_args(eplb_state),
        w2_global_scale if quantize else None,
        l2_alpha if quantize else None,
        stride(hidden_states, 0),
        stride(hidden_states, 1),
        stride(x, 0),
        stride(x_sf, 0),
        stride(shared_x, 0),
        stride(shared_x_sf, 0),
        stride(shared_x_sf, 1),
        is_padding.stride(0) if is_padding is not None else 0,
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        num_tokens,
        partials.stride(0),
        split_k,
        float(routing.routed_scaling_factor),
        float(l2_activation_scale),
        hidden_size,
        num_experts,
        top_k,
        triton.next_power_of_2(top_k),
        bool(routing.renormalize),
        len(fold),
        *folds,
        gate_tokens,
        num_local_experts,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        AMAX_BLOCK_K=min(2048, hidden_size & -hidden_size),
        BLOCK_E=triton.next_power_of_2(max(num_local_experts, 1)),
        SHARED_BLOCK_M=shared_block_m or 1,
        SHARED_MXFP8=shared_mxfp8,
        QUANTIZE=quantize,
        # The conversion packs four output bytes per thread.
        E2M1_CVT=quantize
        and _e2m1_cvt_supported()
        and block_m * block_k // 2 >= 4 * 32 * num_warps,
        EX2_ASM=_ex2_asm_supported(),
        USE_PDL=use_pdl,
        num_warps=num_warps,
        launch_pdl=use_pdl,
    )


# By (experts, top_k, renormalize, routed_scaling_factor).
_FUSED_ROUTING_VERIFIED: dict[tuple[int, int, bool, float], bool] = {}


def megamoe_fused_routing_verified(routing: MegaMoESigmoidRouting) -> bool | None:
    """Whether `route_and_stage_megamoe_nvfp4` selects routing bit for bit as
    `splitk_reduce_triton` then `topk_sigmoid` do, on this GPU and Triton.

    Checked once per process and routing shape (expert count, top_k,
    renormalize, scaling factor), outside CUDA graph capture (None until then):
    the logits on random partials at every split count up to
    MAX_FUSED_ROUTING_SPLIT_K, and the top-k ids and weights on tie-heavy
    logits (NaN, Inf, zeros, subnormal scores) with `routing`'s bias and
    parameters, and with biases that push choices below topkGating's -10000
    mark. A mismatch logs a warning and leaves the fused path off.
    """
    key = (
        routing.e_score_correction_bias.numel(),
        routing.top_k,
        bool(routing.renormalize),
        float(routing.routed_scaling_factor),
    )
    verified = _FUSED_ROUTING_VERIFIED.get(key)
    if verified is None and not torch.cuda.is_current_stream_capturing():
        mismatch = _fused_routing_mismatch(routing)
        verified = _FUSED_ROUTING_VERIFIED[key] = mismatch is None
        if mismatch is None:
            logger.info(
                "MegaMoE fused routing (VLLM_MEGAMOE_FUSED_ROUTING=1): the router's "
                "split-K reduction, sigmoid top-k and NVFP4 staging run in one "
                "kernel; checked bit for bit against splitk_reduce_triton and "
                "topk_sigmoid."
            )
        else:
            logger.warning(
                "VLLM_MEGAMOE_FUSED_ROUTING=1 is off: the fused routing kernel "
                "differs from splitk_reduce_triton + topk_sigmoid (%s).",
                mismatch,
            )
    return verified


def _fused_routing_mismatch(routing: MegaMoESigmoidRouting) -> str | None:
    """The first case where the fused kernel's logits, ids or weights differ
    from the unfused ops' bits, or None."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_sm100 import (  # noqa: E501
        splitk_reduce_triton,
    )

    bias = routing.e_score_correction_bias
    device = bias.device
    num_experts = bias.numel()
    top_k = routing.top_k
    if torch.isnan(bias).any():
        return "the correction bias has NaN"
    gen = torch.Generator(device=device).manual_seed(0)

    def bits(t: torch.Tensor) -> torch.Tensor:
        return t.contiguous().view(torch.int32)

    def fused(partials, bias_, is_padding=None):
        n = partials.shape[1]
        out = dict(
            router_logits=torch.empty(n, num_experts, device=device),
            topk_weights=torch.empty(n, top_k, device=device),
            topk_ids=torch.empty(n, top_k, dtype=torch.int32, device=device),
        )
        route_and_stage_megamoe_nvfp4(
            routing._replace(router_partials=partials, e_score_correction_bias=bias_),
            None,
            None,
            None,
            None,
            torch.empty(n, top_k, dtype=torch.int64, device=device),
            torch.empty(n, top_k, device=device),
            is_padding=is_padding,
            **out,
        )
        return out

    # The sums: magnitudes over six decades so the order shows, signed zeros.
    for split_k in (2, 3, 4, 5, 8, 9, 16, 17, 32, 47, MAX_FUSED_ROUTING_SPLIT_K):
        for n in (1, 37, 160):
            partials = torch.randn(
                split_k, n, num_experts, device=device, generator=gen
            )
            partials *= 10.0 ** (
                6 * torch.rand(split_k, n, 1, device=device, generator=gen) - 3
            )
            partials[:, 0, :8] = 0.0
            partials[0, 0, 8:16] = -0.0
            partials[:, 0, 16:24] = -0.0
            ref = torch.empty(n, num_experts, device=device)
            splitk_reduce_triton(partials, ref)
            if not torch.equal(bits(ref), bits(fused(partials, bias)["router_logits"])):
                return f"split-K sum of {split_k} partials, {n} tokens"

    # The routing: ties, NaN, Inf, signed zeros, tiny and huge logits.
    n = 160
    logits = torch.randint(-6, 7, (n, num_experts), device=device, generator=gen) * 0.5
    logits[1::3] = torch.randn(
        len(range(1, n, 3)), num_experts, device=device, generator=gen
    )
    logits[2] = 0.0
    logits[3, ::2] = -0.0
    logits[4, :5] = float("nan")
    logits[5, :3] = float("inf")
    logits[5, 3:6] = -float("inf")
    logits[6] = float("nan")
    logits[7, :16] = 1.0e-40
    logits[8] = 3.0e38
    # Scores from subnormal (x below about -87.3) to normal, and tied there.
    logits[9] = -88.7 + 1.4 * torch.arange(num_experts, device=device) / num_experts
    logits[10] = -88.0
    logits[10, ::9] = -87.5
    is_padding = torch.zeros(n, dtype=torch.bool, device=device)
    is_padding[::7] = True
    # All but two experts' choices below the -10000 mark: topkGating re-picks.
    deep = bias.clone()
    deep[2:] -= 20000.0
    biases = (bias, torch.zeros_like(bias), torch.round(bias * 4) / 4)
    for bias_ in (*biases, bias.abs() + 1.0, deep):
        for pad in (None, is_padding):
            ref_w = torch.empty(n, top_k, device=device)
            ref_ids = torch.empty(n, top_k, dtype=torch.int32, device=device)
            ops.topk_sigmoid(
                ref_w,
                ref_ids,
                torch.empty(n, top_k, dtype=torch.int32, device=device),
                logits,
                routing.renormalize,
                bias_,
                routing.routed_scaling_factor,
                is_padding=pad,
            )
            got = fused(logits.unsqueeze(0), bias_.contiguous(), pad)
            if not torch.equal(ref_ids, got["topk_ids"]):
                return "top-k expert ids"
            if not torch.equal(bits(ref_w), bits(got["topk_weights"])):
                return "top-k weights"
    return None
