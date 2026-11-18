# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combined Top-K and Top-P Triton kernels.

Based on the paper "Qrita: High-performance Top-k and Top-p Algorithm for GPUs
using Pivot-based Truncation and Selection" By Park et al.
(https://arxiv.org/abs/2602.01518)

"""

import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import (
    WarmupChoices,
    WarmupIntRange,
    _when,
)
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.platform_utils import num_compute_units

logger = init_logger(__name__)

_TRITON_TABLE_CACHE: dict[tuple[torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
_TRITON_BUFFER_CACHE: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}

# fmt: off
_NORMAL_CDF_TO_SIGMA_TABLE = [
  3.656,  3.650,  3.650,  3.650,  3.626,  3.626,  3.626,  3.514,  3.514,  3.503, 
  3.503,  3.434,  3.434,  3.428,  3.428,  3.387,  3.380,  3.380,  3.376,  3.373, 
  3.373,  3.356,  3.354,  3.354,  3.291,  3.249,  3.234,  3.214,  3.198,  3.198, 
  3.185,  3.177,  3.177,  3.165,  3.164,  3.161,  3.138,  3.120,  3.115,  3.113, 
  3.093,  3.066,  3.054,  3.043,  3.037,  3.023,  2.993,  2.991,  2.976,  2.970, 
  2.952,  2.946,  2.932,  2.908,  2.902,  2.895,  2.886,  2.874,  2.861,  2.844, 
  2.836,  2.810,  2.801,  2.790,  2.784,  2.779,  2.767,  2.757,  2.745,  2.733, 
  2.723,  2.716,  2.693,  2.678,  2.671,  2.656,  2.649,  2.629,  2.611,  2.595, 
  2.592,  2.585,  2.574,  2.550,  2.543,  2.534,  2.521,  2.518,  2.497,  2.485, 
  2.468,  2.450,  2.441,  2.430,  2.412,  2.402,  2.389,  2.383,  2.377,  2.364, 
  2.349,  2.338,  2.332,  2.319,  2.310,  2.301,  2.282,  2.274,  2.266,  2.250, 
  2.242,  2.236,  2.226,  2.215,  2.207,  2.196,  2.179,  2.171,  2.162,  2.147, 
  2.135,  2.121,  2.109,  2.095,  2.085,  2.073,  2.063,  2.045,  2.030,  2.016, 
  2.003,  1.992,  1.983,  1.972,  1.960,  1.949,  1.940,  1.928,  1.912,  1.897, 
  1.881,  1.869,  1.854,  1.838,  1.824,  1.807,  1.792,  1.779,  1.764,  1.751, 
  1.739,  1.726,  1.711,  1.697,  1.685,  1.668,  1.652,  1.636,  1.622,  1.603, 
  1.585,  1.568,  1.551,  1.534,  1.513,  1.499,  1.480,  1.464,  1.441,  1.422, 
  1.394,  1.373,  1.347,  1.320,  1.296,  1.270,  1.246,  1.219,  1.190,  1.163, 
  1.135,  1.104,  1.073,  1.041,  1.006,  0.969,  0.931,  0.894,  0.851,  0.806, 
  0.757,  0.702,  0.643,  0.574,  0.498,  0.405,  0.288,  0.134, -0.110, -3.813 
]

_PERCENTILE_TO_STD_TABLE = [
  2.576,  2.319,  2.178,  2.064,  1.968,  1.892,  1.819,  1.757,  1.708,  1.659, 
  1.616,  1.568,  1.526,  1.492,  1.456,  1.420,  1.382,  1.342,  1.309,  1.280, 
  1.249,  1.221,  1.193,  1.169,  1.145,  1.121,  1.095,  1.073,  1.050,  1.030, 
  1.008,  0.987,  0.966,  0.945,  0.926,  0.910,  0.891,  0.871,  0.854,  0.837, 
  0.819,  0.803,  0.784,  0.767,  0.753,  0.734,  0.719,  0.702,  0.690,  0.675, 
  0.658,  0.640,  0.625,  0.609,  0.595,  0.578,  0.564,  0.550,  0.537,  0.521, 
  0.509,  0.495,  0.481,  0.466,  0.453,  0.439,  0.424,  0.410,  0.397,  0.383, 
  0.370,  0.356,  0.343,  0.330,  0.316,  0.302,  0.289,  0.274,  0.261,  0.247, 
  0.235,  0.223,  0.209,  0.196,  0.184,  0.172,  0.159,  0.149,  0.137,  0.124, 
  0.112,  0.100,  0.086,  0.074,  0.062,  0.050,  0.035,  0.023,  0.009, -0.003, 
 -0.015, -0.027, -0.039, -0.052, -0.063, -0.074, -0.085, -0.097, -0.109, -0.122, 
 -0.134, -0.147, -0.158, -0.171, -0.184, -0.196, -0.210, -0.223, -0.235, -0.248, 
 -0.261, -0.275, -0.289, -0.302, -0.317, -0.328, -0.341, -0.353, -0.368, -0.382, 
 -0.396, -0.410, -0.426, -0.439, -0.452, -0.465, -0.480, -0.493, -0.507, -0.521, 
 -0.537, -0.551, -0.568, -0.582, -0.597, -0.614, -0.628, -0.643, -0.658, -0.673, 
 -0.691, -0.706, -0.721, -0.738, -0.754, -0.769, -0.789, -0.808, -0.824, -0.838, 
 -0.857, -0.877, -0.893, -0.912, -0.929, -0.947, -0.965, -0.983, -1.003, -1.027, 
 -1.050, -1.070, -1.092, -1.117, -1.139, -1.162, -1.189, -1.216, -1.241, -1.272, 
 -1.300, -1.330, -1.367, -1.404, -1.441, -1.485, -1.523, -1.564, -1.607, -1.658, 
 -1.710, -1.778, -1.832, -1.901, -1.978, -2.068, -2.174, -2.325, -2.577, -3.813 
]
# fmt: on


@triton.jit
def _update_min_larger_stats(data, above_mask, min_larger, num_min_larger, sentinel):
    """Update running (min, count) of values above a pivot across tiles.

    Tracks the smallest value strictly above a pivot and how many times
    it occurs.  Called once per tile per pivot; the running state is
    carried across tiles via `min_larger` / `num_min_larger`.

    Merge rule:
      - tile min < running min  → replace both
      - tile min == running min → accumulate count
      - tile min > running min  → keep running values
    """
    tile_min = tl.min(tl.where(above_mask, data, sentinel))
    tile_eq = above_mask & (tl.abs(data - tile_min) < 1e-9)
    tile_cnt = tl.sum(tile_eq)
    is_new = tile_min < min_larger
    is_same = tl.abs(tile_min - min_larger) < 1e-9
    num_min_larger = tl.where(is_new, tile_cnt, num_min_larger + tile_cnt * is_same)
    min_larger = tl.minimum(min_larger, tile_min)
    return min_larger, num_min_larger


@triton.jit(do_not_specialize_on_alignment=["BATCH_SIZE"])
def _topk_topp_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    BUFFER,
    PERCENTILE_TO_STD_TABLE,
    NORMAL_CDF_TO_SIGMA_TABLE,
    K,
    P,
    BATCH_SIZE,
    VOCAB_SIZE: tl.constexpr,
    MASK_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_TRUNC: tl.constexpr,
    TOPK_ENABLED: tl.constexpr,
    TOPP_ENABLED: tl.constexpr,
    SPLIT_COVERS_PONLY: tl.constexpr,
):
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, BATCH_SIZE, num_programs):
        LOGITS_ROW = LOGITS + row_id.to(tl.int64) * LOGITS_STRIDE_0
        BUFFER_ROW = BUFFER + pid * VOCAB_SIZE

        final_pivot = -float("inf")
        duplicate_logit = float("inf")
        num_duplicate_logit = tl.zeros((), dtype=tl.uint32)
        num_keep = tl.zeros((), dtype=tl.uint32)
        num_kept = tl.zeros((), dtype=tl.uint32)

        max_logit = -float("inf")
        min_logit = float("inf")

        if TOPK_ENABLED:
            k = tl.load(K + row_id)
            if k < VOCAB_SIZE:
                # Zeroth pass: Compute avg and std from a sample block
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                logits_blk0 = tl.load(
                    LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
                )
                # Exclude -inf values (e.g. from grammar bitmasks) from
                # statistics to avoid NaN in pivot computation.
                finite_mask = (logits_blk0 > -float("inf")) & mask_n
                num_finite = tl.sum(finite_mask)
                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                avg_logit = tl.where(
                    num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
                )
                sq_avg_logit = tl.where(
                    num_finite > 0,
                    tl.sum(finite_logits * finite_logits) / num_finite,
                    0.0,
                )
                std_logit = tl.sqrt(
                    tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
                )

                # Calculate outlier pivot t for Gaussian sigma-truncation
                percentile = tl.cast(k / VOCAB_SIZE * 200, tl.uint32)
                percentile = tl.minimum(percentile, 199)
                sigma = tl.load(PERCENTILE_TO_STD_TABLE + percentile)
                sigma = sigma + tl.abs(sigma) * -0.15
                outlier_pivot = avg_logit + std_logit * sigma
                num_outliers = tl.zeros((), dtype=tl.uint32)

                # First pass: compute max and min logits and gather outliers
                num_finite_total = tl.zeros((), dtype=tl.uint32)
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )

                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    # Exclude -inf from min to keep binary search bounds
                    # finite (avoids NaN pivots).
                    finite_blk_mask = logits_blk > -float("inf")
                    finite_blk = tl.where(finite_blk_mask, logits_blk, float("inf"))
                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))
                    num_finite_total += tl.sum(finite_blk_mask & mask_n)

                    outlier_mask = (logits_blk > outlier_pivot) & mask_n
                    cumulative_pos = tl.cast(
                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                    )
                    num_outliers += tl.sum(outlier_mask)
                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                    tl.store(BUFFER_ROW + write_pos, logits_blk, mask=outlier_mask)

                # If no finite logits exist (all -inf), clamp min to
                # max so the search converges to -inf (no masking).
                min_logit = tl.minimum(min_logit, max_logit)

                # Second passes: Ternary search for pivot
                num_iters = 0
                k_pivot = float("inf")
                k_pivots_num = tl.zeros((), dtype=tl.uint32)
                min_larger = float("inf")
                num_min_larger = tl.zeros((), dtype=tl.uint32)
                if num_finite_total <= k:
                    # All finite logits survive top-k; skip the cutoff search.
                    k_pivot = -float("inf")
                elif num_outliers > k:
                    max_range = max_logit
                    min_range = outlier_pivot
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )
                    found_pivot = 0
                    while found_pivot == 0:
                        k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute k_pivots_num,
                        # min_larger, and num_min_larger together to avoid
                        # a second data scan. See _update_min_larger_stats
                        # for the tile-level merge logic.
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            logits_blk2 = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=-float("inf")
                            )

                            above_0 = logits_blk2 > k_pivot_0
                            above_1 = logits_blk2 > k_pivot_1
                            k_pivots_num_0 += tl.sum(above_0)
                            k_pivots_num_1 += tl.sum(above_1)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                logits_blk2,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                float("inf"),
                            )
                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                                logits_blk2,
                                above_1,
                                min_larger_1,
                                num_min_larger_1,
                                float("inf"),
                            )

                        # Check if any of the pivots satisfy termination condition
                        if (
                            k_pivots_num_0 >= k
                            and k_pivots_num_0 - num_min_larger_0 < k
                        ):
                            k_pivot = k_pivot_0
                            k_pivots_num = k_pivots_num_0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                        if (
                            k_pivots_num_1 >= k
                            and k_pivots_num_1 - num_min_larger_1 < k
                        ):
                            k_pivot = k_pivot_1
                            k_pivots_num = k_pivots_num_1
                            min_larger = min_larger_1
                            num_min_larger = num_min_larger_1
                            found_pivot = 1

                        # Update range
                        if k_pivots_num_1 > k:
                            min_range = k_pivot_1
                        elif k_pivots_num_0 > k:
                            min_range = k_pivot_0

                        if k_pivots_num_0 < k:
                            max_range = k_pivot_0
                        elif k_pivots_num_1 < k:
                            max_range = k_pivot_1

                        num_iters += 1
                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                            k_pivot = (max_range + min_range) / 2.0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                else:
                    # If top-k outlier gathering failed, search whole logit space
                    max_range = max_logit
                    min_range = min_logit
                    found_pivot = 0
                    while found_pivot == 0:
                        k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass over full vocab (same approach
                        # as the buffer path above).
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            logits_blk2 = tl.load(
                                LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                            )

                            above_0 = logits_blk2 > k_pivot_0
                            above_1 = logits_blk2 > k_pivot_1
                            k_pivots_num_0 += tl.sum(above_0)
                            k_pivots_num_1 += tl.sum(above_1)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                logits_blk2,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                float("inf"),
                            )
                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                                logits_blk2,
                                above_1,
                                min_larger_1,
                                num_min_larger_1,
                                float("inf"),
                            )

                        # Check if any of the pivots satisfy termination condition
                        if (
                            k_pivots_num_0 >= k
                            and k_pivots_num_0 - num_min_larger_0 < k
                        ):
                            k_pivot = k_pivot_0
                            k_pivots_num = k_pivots_num_0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                        if (
                            k_pivots_num_1 >= k
                            and k_pivots_num_1 - num_min_larger_1 < k
                        ):
                            k_pivot = k_pivot_1
                            k_pivots_num = k_pivots_num_1
                            min_larger = min_larger_1
                            num_min_larger = num_min_larger_1
                            found_pivot = 1

                        # Update range
                        if k_pivots_num_1 > k:
                            min_range = k_pivot_1
                        elif k_pivots_num_0 > k:
                            min_range = k_pivot_0

                        if k_pivots_num_0 < k:
                            max_range = k_pivot_0
                        elif k_pivots_num_1 < k:
                            max_range = k_pivot_1

                        num_iters += 1
                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                            k_pivot = (max_range + min_range) / 2.0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1

                duplicate_logit = min_larger
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - (k_pivots_num - k)
                num_kept = tl.zeros((), dtype=tl.uint32)

                # Top-k only path.  If there are fewer finite values
                # than k (e.g. grammar mask), keep everything.
                final_pivot = k_pivot if num_finite_total > k else -float("inf")

                if TOPP_ENABLED and num_finite_total > k:
                    #### TOP-P SAMPLING AFTER TOP-K ####
                    p = tl.load(P + row_id)
                    if p < 1.0:
                        min_logit = k_pivot
                        sum_exp_logits = 0.0
                        num_outliers_2 = tl.zeros((), dtype=tl.uint32)
                        search_range = tl.cast(num_outliers, tl.int32)
                        search_iters = tl.cast(
                            (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                            tl.int32,
                        )

                        # Third pass: Calculate exp logits and sum, gather outliers
                        if num_outliers > k:
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n,
                                    mask=mask_n_2,
                                    other=-float("inf"),
                                )

                                outlier_mask = (probs_blk > min_logit) & mask_n_2

                                # Duplicate logit handling for Top-k
                                if num_keep < num_duplicate_logit:
                                    duplicate_mask = (
                                        tl.abs(probs_blk - duplicate_logit) < 1e-9
                                    )
                                    duplicate_count = (
                                        tl.cumsum(duplicate_mask) + num_kept
                                    )
                                    duplicate_keep_mask = (
                                        duplicate_count <= num_keep
                                    ) & duplicate_mask
                                    duplicate_remove_mask = (
                                        duplicate_mask & ~duplicate_keep_mask
                                    )
                                    outlier_mask = outlier_mask & (
                                        ~duplicate_remove_mask
                                    )
                                    num_kept += tl.sum(duplicate_keep_mask)

                                probs_blk = tl.where(
                                    outlier_mask, probs_blk, -float("inf")
                                )
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)

                            # Fourth pass: Calculate BUFFER and get outliers
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n,
                                    mask=mask_n_2,
                                    other=-float("inf"),
                                )

                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)
                        else:
                            # If top-k outlier gathering failed,
                            # retry gathering using top-k pivot
                            for i in range(0, NUM_TILES):
                                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                                mask_n = offs_n < VOCAB_SIZE

                                probs_blk = tl.load(
                                    LOGITS_ROW + offs_n,
                                    mask=mask_n,
                                    other=-float("inf"),
                                )

                                outlier_mask = (probs_blk > min_logit) & mask_n

                                # Duplicate logit handling for Top-k
                                duplicate_mask = (
                                    tl.abs(probs_blk - duplicate_logit) < 1e-9
                                )
                                duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                                duplicate_keep_mask = (
                                    duplicate_count <= num_keep
                                ) & duplicate_mask
                                duplicate_remove_mask = (
                                    duplicate_mask & ~duplicate_keep_mask
                                )
                                outlier_mask = outlier_mask & (~duplicate_remove_mask)
                                num_kept += tl.sum(duplicate_keep_mask)

                                probs_blk = tl.where(
                                    outlier_mask, probs_blk, -float("inf")
                                )
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)

                                cumulative_pos = tl.cast(
                                    tl.cumsum(outlier_mask) - 1 + num_outliers_2,
                                    tl.int32,
                                )
                                num_outliers_2 += tl.sum(outlier_mask)
                                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                                tl.store(
                                    BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask
                                )

                            search_range = tl.cast(num_outliers_2, tl.int32)
                            search_iters = tl.cast(
                                (num_outliers_2 + BLOCK_SIZE_TRUNC - 1)
                                // BLOCK_SIZE_TRUNC,
                                tl.int32,
                            )

                            # Fourth pass: Calculate BUFFER and get outliers
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                                )
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)

                        max_range = tl.exp(max_logit - max_logit) / sum_exp_logits
                        min_range = tl.exp(min_logit - max_logit) / sum_exp_logits

                        p_pivot = 1.0
                        num_iters = 0
                        min_larger_prob = 1.0
                        num_min_larger = tl.zeros((), dtype=tl.uint32)
                        p_pivots_sum = 0.0

                        # Fifth passes: Search for p_pivot
                        found_pivot = 0
                        while found_pivot == 0:
                            p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                            p_pivots_sum_0 = 0.0
                            min_larger_0 = 1.0
                            num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                            # Single fused pass: compute p_pivots_sum,
                            # min_larger, and num_min_larger together.
                            # See _update_min_larger_stats for the
                            # tile-level merge logic.
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range
                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                                )

                                above_0 = probs_blk > p_pivot_0
                                p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                                min_larger_0, num_min_larger_0 = (
                                    _update_min_larger_stats(
                                        probs_blk,
                                        above_0,
                                        min_larger_0,
                                        num_min_larger_0,
                                        1.0,
                                    )
                                )

                            # Check if the pivot satisfies termination condition
                            if p_pivots_sum_0 >= p and (
                                p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                            ):
                                p_pivot = p_pivot_0
                                min_larger_prob = min_larger_0
                                num_min_larger = num_min_larger_0
                                p_pivots_sum = p_pivots_sum_0
                                found_pivot = 1

                            # Update range
                            if p_pivots_sum_0 > p:
                                min_range = p_pivot_0
                            elif p_pivots_sum_0 < p:
                                max_range = p_pivot_0

                            num_iters += 1
                            if (max_range - min_range) < 1e-9 or num_iters >= 18:
                                p_pivot = (max_range + min_range) / 2.0
                                min_larger_prob = min_larger_0
                                num_min_larger = num_min_larger_0
                                p_pivots_sum = p_pivots_sum_0
                                found_pivot = 1

                        duplicate_logit = (
                            tl.log(min_larger_prob * sum_exp_logits) + max_logit
                        )
                        num_duplicate_logit = num_min_larger
                        num_keep = num_duplicate_logit - tl.cast(
                            (p_pivots_sum - p) / min_larger_prob, tl.uint32
                        )
                        num_kept = tl.zeros((), dtype=tl.uint32)

                        # Top-k + Top-p path
                        final_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit

        if TOPP_ENABLED and final_pivot == -float("inf"):
            #### STANDALONE TOP-P SAMPLING ####
            # When the split top-p pipeline co-runs (mixed top-k + top-p
            # batches), it covers p-only rows (k >= VOCAB_SIZE); skip those
            # here. Rows whose top-k was a no-op because they have <= k
            # finite logits (e.g. grammar masks) are NOT covered by the
            # split pipeline and must get standalone top-p here.
            run_standalone = True
            if SPLIT_COVERS_PONLY and TOPK_ENABLED:
                run_standalone = tl.load(K + row_id) < VOCAB_SIZE
            p = tl.load(P + row_id)
            if run_standalone and p < 1.0:
                # Zeroth pass: Compute avg and std from a sample block
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                logits_blk0 = tl.load(
                    LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
                )
                # Exclude -inf values (e.g. from grammar bitmasks) from
                # statistics to avoid NaN in pivot computation.
                finite_mask = (logits_blk0 > -float("inf")) & mask_n
                num_finite = tl.sum(finite_mask)
                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                avg_logit = tl.where(
                    num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
                )
                sq_avg_logit = tl.where(
                    num_finite > 0,
                    tl.sum(finite_logits * finite_logits) / num_finite,
                    0.0,
                )
                std_logit = tl.sqrt(
                    tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
                )
                max_sample = avg_logit + std_logit * 10.0
                sum_exp_logits = 0.0

                # First pass: compute max and min logits and sum_exp_logits
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    # Exclude -inf from min to keep binary search bounds
                    # finite (avoids NaN pivots).
                    finite_blk = tl.where(
                        logits_blk > -float("inf"), logits_blk, float("inf")
                    )
                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))

                    probs_blk = tl.exp(logits_blk - max_sample)
                    probs_blk = tl.where(mask_n, probs_blk, 0.0)
                    sum_exp_logits += tl.sum(probs_blk)

                # If no finite logits exist (all -inf), clamp min to
                # max so the search converges to -inf (no masking).
                min_logit = tl.minimum(min_logit, max_logit)

                idx = tl.cast(p * 200, tl.int32)
                idx = tl.maximum(0, tl.minimum(idx, 199))
                sigma = tl.load(NORMAL_CDF_TO_SIGMA_TABLE + idx)
                sigma = sigma + tl.abs(sigma) * -0.25
                outlier_pivot = avg_logit + std_logit * sigma

                outlier_prob = tl.exp(outlier_pivot - max_sample) / sum_exp_logits
                sum_outlier_probs = 0.0
                num_outliers = tl.zeros((), dtype=tl.uint32)

                # Second pass: Calculate softmax and gather outliers
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE

                    probs_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    probs_blk = tl.exp(probs_blk - max_sample)
                    probs_blk = probs_blk / sum_exp_logits

                    outlier_mask = (probs_blk > outlier_prob) & mask_n
                    sum_outlier_probs += tl.sum(outlier_mask * probs_blk)
                    cumulative_pos = tl.cast(
                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                    )
                    num_outliers += tl.sum(outlier_mask)
                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                    tl.store(BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask)

                max_range = tl.exp(max_logit - max_sample) / sum_exp_logits
                min_range = tl.exp(min_logit - max_sample) / sum_exp_logits

                p_pivot = 1.0
                num_iters = 0
                min_larger_prob = 1.0
                num_min_larger = tl.zeros((), dtype=tl.uint32)
                p_pivots_sum = 0.0

                # Third pass: Search for p_pivot
                if sum_outlier_probs > p:
                    min_range = outlier_prob
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        # See _update_min_larger_stats for the
                        # tile-level merge logic.
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                            )

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination condition
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1
                else:
                    # Re-populate the buffer with full softmax probabilities
                    for i in range(0, NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE

                        probs_blk = tl.load(
                            LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                        )
                        probs_blk = tl.exp(probs_blk - max_sample)
                        probs_blk = probs_blk / sum_exp_logits
                        tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        # See _update_min_larger_stats for the
                        # tile-level merge logic.
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n, other=0.0
                            )

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination condition
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_sample
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - tl.cast(
                    (p_pivots_sum - p) / min_larger_prob, tl.uint32
                )
                num_kept = tl.zeros((), dtype=tl.uint32)

                # Top-p only path
                final_pivot = tl.log(p_pivot * sum_exp_logits) + max_sample

        # Sixth pass: Apply mask and store final output.
        # If the pivot >= max logit (or is NaN), no token would
        # survive the strict `>` keep_mask.  Skip masking.
        # Using `not <` instead of `>=` so that NaN is also caught.
        if not (final_pivot < max_logit):
            final_pivot = -float("inf")
        elif final_pivot != -float("inf"):
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(
                    LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                )
                keep_mask = (logits_blk > final_pivot) & mask_n

                # Duplicate logit handling
                if num_keep < num_duplicate_logit:
                    duplicate_mask = (
                        tl.abs(logits_blk - duplicate_logit) < 1e-9
                    ) & mask_n
                    duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                    duplicate_keep_mask = (duplicate_count <= num_keep) & duplicate_mask
                    duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
                    num_kept += tl.sum(duplicate_keep_mask)
                    keep_mask = keep_mask & (~duplicate_remove_mask)

                logits_blk = tl.where(keep_mask, logits_blk, MASK_VALUE)
                tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)


def _max_sampler_batch_size(vllm_config: Any) -> int:
    return vllm_config.scheduler_config.max_num_seqs * (
        vllm_config.num_speculative_tokens + 1
    )


def _topk_topp_warmup_inputs(vllm_config: Any) -> dict[str, Any]:
    vocab_size = vllm_config.model_config.get_vocab_size()
    split_enabled = current_platform.is_cuda_alike()
    mode: Any = WarmupChoices((True, False), (True, True), (False, True))
    logits_stride: Any = WarmupChoices(16, 2)
    batch_size: Any = WarmupIntRange(1, _max_sampler_batch_size(vllm_config) + 1)
    topk_enabled = mode[0]
    topp_enabled = mode[1]
    _when(topk_enabled or not (split_enabled and batch_size <= _SPLIT_MAX_BATCH))
    logits = TritonWarmupTensor(
        torch.float32,
        shape=(batch_size, vocab_size),
        strides=(logits_stride, 1),
    )
    return dict(
        logits=logits,
        buffer=TritonWarmupTensor(torch.float32),
        percentile_to_std_table=TritonWarmupTensor(torch.float32),
        normal_cdf_to_sigma_table=TritonWarmupTensor(torch.float32),
        k=TritonWarmupTensor(torch.int32) if topk_enabled else None,
        p=TritonWarmupTensor(torch.float32) if topp_enabled else None,
        mask_value=float("-inf"),
        num_sm=num_compute_units(),
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_topk_topp_kernel,
    warmup_inputs=_topk_topp_warmup_inputs,
)
def _topk_topp(
    logits: torch.Tensor,
    buffer: torch.Tensor,
    percentile_to_std_table: torch.Tensor,
    normal_cdf_to_sigma_table: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    mask_value: float,
    num_sm: int,
) -> DispatchSpec:
    batch_size, vocab_size = logits.shape
    topk_enabled = k is not None
    topp_enabled = p is not None
    k_ptr = k if k is not None else logits
    p_ptr = p if p is not None else logits
    num_programs = min(num_sm, batch_size)
    # Smaller tiles compile and run faster on CPU; GPU benefits from larger tiles.
    # On XPU, large BLOCK_SIZE causes precision loss in the single-pass pivot
    # approximation; use smaller tiles for accurate top-p results.
    if logits.device.type == "cpu":
        block_size, block_size_trunc, num_warps = 256, 128, 4
    elif logits.device.type == "xpu":
        block_size, block_size_trunc, num_warps = 4096, 2048, 4
    else:
        block_size, block_size_trunc, num_warps = 8192, 4096, 8
        # Each program serially sweeps the vocab row in BLOCK_SIZE tiles, so
        # per-tile latency bounds kernel latency, and Triton's default of 4
        # warps leaves an 8192-wide tile at 16 elements per lane. 8 warps is
        # faster on every arch measured (SM90, SM100, SM120, gfx950); 16 is not.
    split_covers_ponly = (
        current_platform.is_cuda_alike()
        and topp_enabled
        and batch_size <= _SPLIT_MAX_BATCH
    )
    return (num_programs,), dict(
        K=k_ptr,
        P=p_ptr,
        BATCH_SIZE=batch_size,
        VOCAB_SIZE=vocab_size,
        BLOCK_SIZE=block_size,
        BLOCK_SIZE_TRUNC=block_size_trunc,
        TOPK_ENABLED=topk_enabled,
        TOPP_ENABLED=topp_enabled,
        SPLIT_COVERS_PONLY=split_covers_ponly,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# Split-row pipeline for small batches.
#
# The monolithic kernel above assigns one program per row, so at small batch
# sizes nearly all SMs idle and every full-vocab sweep is latency-bound
# (~25us each on SM100). For standalone top-p (no top-k truncation) the pivot
# search can hit its iteration cap while sweeping ~the whole vocab per
# iteration, which is slower than the PyTorch sort path.
#
# The pipeline below instead splits each row across S programs and runs the
# pivot search as parallel partial reductions over a fixed number of rounds
# (no host synchronization, CUDA-graph safe). The search state is never
# stored: the per-round combine is a tiny pure function of the stored
# partials, so every program redundantly recomputes the bracket chain from
# them. This is deterministic (same inputs, same code) and far cheaper than
# a vocab sweep, and it saves a kernel launch per combine. The pipeline
# handles only "p-only" rows (p < 1 and no active top-k); rows with an
# active top-k are left to the monolithic kernel, whose truncation path is
# already fast for them.
# ---------------------------------------------------------------------------

_SPLIT_MAX_BATCH = 64
_SPLIT_MAX_SPLITS = 32  # power of two
_SPLIT_FANOUT = 8
_SPLIT_ROUNDS = 5
# The step kernels are latency-bound at one program per SM, so keep splitting
# past that while slices span at least this many entries; shorter slices no
# longer amortize each program's row stats and combine replay.
_SPLIT_MIN_SLICE = 7168

_TRITON_SPLIT_CACHE: dict[torch.device, dict[str, torch.Tensor]] = {}


def _topp_split_count(batch_size: int, vocab_size: int, num_sm: int) -> int:
    splits = 1
    while splits < _SPLIT_MAX_SPLITS and (
        splits * 2 * batch_size <= num_sm or splits * 2 * _SPLIT_MIN_SLICE <= vocab_size
    ):
        splits *= 2
    return splits


@triton.jit
def _topp_sb_stats_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    STATS,
    K,
    P,
    HAS_K: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Per-slice online-softmax partials: (max, sum_exp, min, finite_count).
    pid = tl.program_id(0)
    row = pid // S
    slice_id = pid % S
    p = tl.load(P + row)
    if p >= 1.0:
        return
    if HAS_K and tl.load(K + row) < VOCAB_SIZE:
        return
    slice_len = (VOCAB_SIZE + S - 1) // S
    start = slice_id * slice_len
    end = tl.minimum(start + slice_len, VOCAB_SIZE)
    ROW = LOGITS + row.to(tl.int64) * LOGITS_STRIDE_0
    offs = tl.arange(0, BLOCK)
    m = -float("inf")
    exp_sum = 0.0
    mn = float("inf")
    cnt = tl.zeros((), dtype=tl.float32)
    for i in range(start, end, BLOCK):
        x = tl.load(ROW + i + offs, mask=i + offs < end, other=-float("inf"))
        finite = x > -float("inf")
        nm = tl.maximum(m, tl.max(x))
        rescale = tl.where(m == -float("inf"), 0.0, tl.exp(m - nm))
        e = tl.where(finite, tl.exp(x - nm), 0.0)
        exp_sum = exp_sum * rescale + tl.sum(e)
        m = nm
        mn = tl.minimum(mn, tl.min(tl.where(finite, x, float("inf"))))
        cnt += tl.sum(finite.to(tl.float32))
    base = (row * S + slice_id) * 4
    tl.store(STATS + base + 0, m)
    tl.store(STATS + base + 1, exp_sum)
    tl.store(STATS + base + 2, mn)
    tl.store(STATS + base + 3, cnt)


def _topp_split_stats_warmup_inputs(vllm_config: Any) -> dict[str, Any]:
    vocab_size = vllm_config.model_config.get_vocab_size()
    num_sm = num_compute_units()
    logits_stride: Any = WarmupChoices(16, 2)
    batch_size: Any = WarmupIntRange(
        1, min(_max_sampler_batch_size(vllm_config), _SPLIT_MAX_BATCH) + 1
    )
    has_k: Any = WarmupChoices(False, True)
    logits = TritonWarmupTensor(
        torch.float32,
        shape=(batch_size, vocab_size),
        strides=(logits_stride, 1),
    )
    return dict(
        logits=logits,
        stats=TritonWarmupTensor(torch.float32),
        k=TritonWarmupTensor(torch.int32) if has_k else None,
        p=TritonWarmupTensor(torch.float32),
        num_sm=num_sm,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_topp_sb_stats_kernel,
    warmup_inputs=_topp_split_stats_warmup_inputs,
)
def _topp_split_stats(
    logits: torch.Tensor,
    stats: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor,
    num_sm: int,
) -> DispatchSpec:
    batch_size, vocab_size = logits.shape
    splits = _topp_split_count(batch_size, vocab_size, num_sm)
    return (logits.shape[0] * splits,), dict(
        K=k if k is not None else logits,
        HAS_K=k is not None,
        VOCAB_SIZE=vocab_size,
        S=splits,
        BLOCK=8192,
        num_warps=8,
    )


@triton.jit
def _topp_sb_row_stats(STATS, row, S: tl.constexpr):
    """Combine per-slice partials into row (max, sum_exp, min, n_finite)."""
    sidx = tl.arange(0, S)
    sb = STATS + row * S * 4
    m_s = tl.load(sb + sidx * 4 + 0)
    exp_s = tl.load(sb + sidx * 4 + 1)
    mn_s = tl.load(sb + sidx * 4 + 2)
    c_s = tl.load(sb + sidx * 4 + 3)
    M = tl.max(m_s)
    w = tl.where(m_s == -float("inf"), 0.0, tl.exp(m_s - M))
    Z = tl.sum(exp_s * w)
    return M, Z, tl.min(mn_s), tl.sum(c_s)


@triton.jit
def _topp_sb_ladder(lo, hi, F: tl.constexpr):
    """Geometric ladder of F pivots inside (lo, hi)."""
    fidx = tl.arange(0, F).to(tl.float32)
    return lo * tl.exp(tl.log(hi / lo) * (fidx + 1.0) / (F + 1.0))


@triton.jit
def _topp_sb_combine(
    PART_BASE,
    Z,
    p,
    lo,
    hi,
    best,
    best_mass,
    best_minl,
    best_nmin,
    S: tl.constexpr,
    F: tl.constexpr,
    IS_LAST: tl.constexpr,
):
    """Combine one round's partials and advance the search state.

    Returns (done, nomask, pivot, dup, numdup, numkeep, lo, hi, best,
    best_mass, best_minl, best_nmin, sel_fidx, best_sel). `best*` tracks the
    tightest fallback: the largest evaluated pivot whose above-mass is >= p,
    which guarantees the top-p mass invariant if the search fails to
    converge. Pivots are recomputed from (lo, hi) via the same ladder the
    sweep used, so partials line up with pivots without storing them.
    `sel_fidx` / `best_sel` report which ladder slot produced `dup` /
    updated `best` this round (-1 if none), so callers can recover the
    per-slice (minl, nmin) partials behind the boundary value for
    deterministic duplicate handling.
    """
    fidx = tl.arange(0, F)
    sidx = tl.arange(0, S)
    offs2 = sidx[:, None] * (F * 3) + fidx[None, :] * 3
    mass_s = tl.load(PART_BASE + offs2 + 0)
    minl_s = tl.load(PART_BASE + offs2 + 1)
    nmin_s = tl.load(PART_BASE + offs2 + 2)
    mass = tl.sum(mass_s, axis=0)
    gmin = tl.min(minl_s, axis=0)
    nmin = tl.sum(tl.where(tl.abs(minl_s - gmin[None, :]) < 1e-9, nmin_s, 0.0), axis=0)
    pivs = _topp_sb_ladder(lo, hi, F)

    ok = (mass >= p) & (mass - gmin * nmin < p)
    ok_idx = tl.max(tl.where(ok, fidx, -1).to(tl.float32))
    ge = mass >= p
    ge_idx = tl.max(tl.where(ge, fidx, -1).to(tl.float32))

    best_sel = -1.0
    if ge_idx >= 0:
        bmask = fidx == ge_idx.to(tl.int32)
        bp = tl.sum(tl.where(bmask, pivs, 0.0))
        if bp > best:
            best = bp
            best_mass = tl.sum(tl.where(bmask, mass, 0.0))
            best_minl = tl.sum(tl.where(bmask, gmin, 0.0))
            best_nmin = tl.sum(tl.where(bmask, nmin, 0.0))
            best_sel = ge_idx

    done = 0.0
    nomask = 0.0
    pivot = 0.0
    sel_mass = 0.0
    dup = 1.0
    numdup = 1.0
    numkeep = 1.0
    sel_fidx = -1.0
    if ok_idx >= 0:
        bmask = fidx == ok_idx.to(tl.int32)
        pivot = tl.sum(tl.where(bmask, pivs, 0.0))
        sel_mass = tl.sum(tl.where(bmask, mass, 0.0))
        dup = tl.sum(tl.where(bmask, gmin, 0.0))
        numdup = tl.sum(tl.where(bmask, nmin, 0.0))
        sel_fidx = ok_idx
        done = 1.0
    else:
        new_lo = tl.maximum(lo, tl.max(tl.where(ge, pivs, -float("inf"))))
        new_hi = tl.minimum(hi, tl.min(tl.where(~ge, pivs, float("inf"))))
        if IS_LAST or not (new_hi > new_lo * (1.0 + 1e-7)):
            # Out of rounds or the range collapsed: fall back to the
            # tightest pivot with mass >= p, or keep the whole row if no
            # pivot qualified.
            if best > 0.0:
                pivot = best
                sel_mass = best_mass
                dup = best_minl
                numdup = best_nmin
            else:
                nomask = 1.0
            done = 1.0
        lo = new_lo
        hi = new_hi
    if (done != 0.0) and (nomask == 0.0):
        # Mirror the monolithic guard: a pivot at/above the max logit would
        # mask everything under the strict `>` comparison.
        nomask = tl.where(pivot * Z >= 1.0, 1.0, 0.0)
        # Duplicate (boundary-value) handling, same formula as the
        # monolithic kernel: keep only some of the boundary duplicates.
        numkeep = numdup - tl.cast(tl.maximum(sel_mass - p, 0.0) / dup, tl.uint32).to(
            tl.float32
        )
        numkeep = tl.minimum(tl.maximum(numkeep, 1.0), numdup)
    return (
        done,
        nomask,
        pivot,
        dup,
        numdup,
        numkeep,
        lo,
        hi,
        best,
        best_mass,
        best_minl,
        best_nmin,
        sel_fidx,
        best_sel,
    )


@triton.jit(do_not_specialize_on_alignment=["ROUND"])
def _topp_sb_step_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    STATS,
    PARTS,
    K,
    P,
    ROUND,
    HAS_K: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    NUM_ROUNDS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // S
    slice_id = pid % S
    p = tl.load(P + row)
    if p >= 1.0:
        return
    if HAS_K and tl.load(K + row) < VOCAB_SIZE:
        return
    M, Z, mn, num_finite = _topp_sb_row_stats(STATS, row, S)
    if (Z <= 0.0) or (num_finite < 1.0):
        return
    max_prob = 1.0 / Z
    min_prob = tl.maximum(tl.exp(mn - M) / Z, 1e-30)
    if max_prob <= min_prob * (1.0 + 1e-6):
        # (Nearly) uniform row: no pivot can differentiate; keep everything.
        return
    # Replay the combine chain for rounds 0..ROUND-1 (deterministic; each
    # is a tiny pure function of the stored partials). Partials from rounds
    # past a row's convergence are stale, so the chain stops at `done`.
    lo = min_prob
    hi = max_prob
    best = -1.0
    best_mass = 0.0
    best_minl = 1.0
    best_nmin = 0.0
    done = 0.0
    for j in range(ROUND):
        if done == 0.0:
            (
                done,
                nomask,
                pivot,
                dup,
                numdup,
                numkeep,
                lo,
                hi,
                best,
                best_mass,
                best_minl,
                best_nmin,
                sel_fidx,
                best_sel,
            ) = _topp_sb_combine(
                PARTS + (row * NUM_ROUNDS + j) * S * F * 3,
                Z,
                p,
                lo,
                hi,
                best,
                best_mass,
                best_minl,
                best_nmin,
                S=S,
                F=F,
                IS_LAST=False,
            )
    if done != 0.0:
        return
    # Sweep this slice against the ladder pivots derived from (lo, hi).
    pivs = _topp_sb_ladder(lo, hi, F)
    slice_len = (VOCAB_SIZE + S - 1) // S
    start = slice_id * slice_len
    end = tl.minimum(start + slice_len, VOCAB_SIZE)
    ROW = LOGITS + row.to(tl.int64) * LOGITS_STRIDE_0
    offs = tl.arange(0, BLOCK)
    fidx = tl.arange(0, F)
    mass = tl.zeros([F], dtype=tl.float32)
    minl = tl.full([F], 1.0, dtype=tl.float32)
    nmin = tl.zeros([F], dtype=tl.float32)
    for i in range(start, end, BLOCK):
        x = tl.load(ROW + i + offs, mask=i + offs < end, other=-float("inf"))
        pr = tl.exp(x - M) / Z
        pr2 = tl.broadcast_to(pr[None, :], (F, BLOCK))
        above = pr2 > pivs[:, None]
        mass += tl.sum(tl.where(above, pr2, 0.0), axis=1)
        cand = tl.where(above, pr2, float("inf"))
        tmin = tl.min(cand, axis=1)
        teq = above & (tl.abs(pr2 - tmin[:, None]) < 1e-9)
        tcnt = tl.sum(teq.to(tl.float32), axis=1)
        is_new = tmin < minl
        is_same = tl.abs(tmin - minl) < 1e-9
        nmin = tl.where(is_new, tcnt, nmin + tl.where(is_same, tcnt, 0.0))
        minl = tl.minimum(minl, tmin)
    pbase = PARTS + ((row * NUM_ROUNDS + ROUND) * S + slice_id) * F * 3
    tl.store(pbase + fidx * 3 + 0, mass)
    tl.store(pbase + fidx * 3 + 1, minl)
    tl.store(pbase + fidx * 3 + 2, nmin)


def _topp_split_step_warmup_inputs(vllm_config: Any) -> dict[str, Any]:
    vocab_size = vllm_config.model_config.get_vocab_size()
    num_sm = num_compute_units()
    logits_stride: Any = WarmupChoices(16, 2)
    batch_size: Any = WarmupIntRange(
        1, min(_max_sampler_batch_size(vllm_config), _SPLIT_MAX_BATCH) + 1
    )
    round: Any = WarmupIntRange(0, _SPLIT_ROUNDS)
    has_k: Any = WarmupChoices(False, True)
    logits = TritonWarmupTensor(
        torch.float32,
        shape=(batch_size, vocab_size),
        strides=(logits_stride, 1),
    )
    return dict(
        logits=logits,
        stats=TritonWarmupTensor(torch.float32),
        parts=TritonWarmupTensor(torch.float32),
        k=TritonWarmupTensor(torch.int32) if has_k else None,
        p=TritonWarmupTensor(torch.float32),
        round=round,
        num_sm=num_sm,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_topp_sb_step_kernel,
    warmup_inputs=_topp_split_step_warmup_inputs,
)
def _topp_split_step(
    logits: torch.Tensor,
    stats: torch.Tensor,
    parts: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor,
    round: int,
    num_sm: int,
) -> DispatchSpec:
    batch_size, vocab_size = logits.shape
    splits = _topp_split_count(batch_size, vocab_size, num_sm)
    return (logits.shape[0] * splits,), dict(
        K=k if k is not None else logits,
        HAS_K=k is not None,
        S=splits,
        F=_SPLIT_FANOUT,
        NUM_ROUNDS=_SPLIT_ROUNDS,
        VOCAB_SIZE=vocab_size,
        BLOCK=2048,
        num_warps=8,
    )


@triton.jit
def _topp_sb_mask_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    STATS,
    PARTS,
    K,
    P,
    HAS_K: tl.constexpr,
    MASK_VALUE: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    NUM_ROUNDS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // S
    slice_id = pid % S
    p = tl.load(P + row)
    if p >= 1.0:
        return
    if HAS_K and tl.load(K + row) < VOCAB_SIZE:
        return
    M, Z, mn, num_finite = _topp_sb_row_stats(STATS, row, S)
    if (Z <= 0.0) or (num_finite < 1.0):
        return
    max_prob = 1.0 / Z
    min_prob = tl.maximum(tl.exp(mn - M) / Z, 1e-30)
    if max_prob <= min_prob * (1.0 + 1e-6):
        return  # (nearly) uniform row: keep everything
    # Replay the full combine chain; the last round finalizes with fallback
    # semantics. Rows that converged early stop the chain at that round.
    # `fin` tracks the (round, ladder slot) whose per-slice partials define
    # the boundary value, for deterministic duplicate handling below.
    lo = min_prob
    hi = max_prob
    best = -1.0
    best_mass = 0.0
    best_minl = 1.0
    best_nmin = 0.0
    done = 0.0
    nomask = 0.0
    pivot = 0.0
    dup = 1.0
    numdup = 1.0
    numkeep = 1.0
    fin = -1.0
    best_fin = -1.0
    for j in range(NUM_ROUNDS):
        if done == 0.0:
            (
                done,
                nomask,
                pivot,
                dup,
                numdup,
                numkeep,
                lo,
                hi,
                best,
                best_mass,
                best_minl,
                best_nmin,
                sel_fidx,
                best_sel,
            ) = _topp_sb_combine(
                PARTS + (row * NUM_ROUNDS + j) * S * F * 3,
                Z,
                p,
                lo,
                hi,
                best,
                best_mass,
                best_minl,
                best_nmin,
                S=S,
                F=F,
                IS_LAST=j == NUM_ROUNDS - 1,
            )
            if sel_fidx >= 0.0:
                fin = j * F + sel_fidx
            if best_sel >= 0.0:
                best_fin = j * F + best_sel
    if nomask != 0.0:
        return
    if fin < 0.0:
        fin = best_fin
    # numdup/numkeep are exact small integers held in fp32.
    ties = numkeep < numdup
    num_preceding = 0.0
    if ties:
        # Deterministic, index-ordered duplicate budget: per-slice counts of
        # the boundary value come from the finalizing round's partials (a
        # slice holds copies of `dup` iff its minl matches `dup`, with the
        # count in nmin), so each slice knows exactly how many copies were
        # kept by lower-indexed slices.
        fin_round = tl.cast(fin / F, tl.int32)
        fin_slot = tl.cast(fin - tl.cast(fin_round, tl.float32) * F, tl.int32)
        pbase = PARTS + ((row * NUM_ROUNDS + fin_round) * S * F + fin_slot) * 3
        sidx = tl.arange(0, S)
        minl_s = tl.load(pbase + sidx * (F * 3) + 1)
        nmin_s = tl.load(pbase + sidx * (F * 3) + 2)
        dupc_s = tl.where(tl.abs(minl_s - dup) < 1e-9, nmin_s, 0.0)
        num_preceding = tl.sum(tl.where(sidx < slice_id, dupc_s, 0.0))
    slice_len = (VOCAB_SIZE + S - 1) // S
    start = slice_id * slice_len
    end = tl.minimum(start + slice_len, VOCAB_SIZE)
    ROW = LOGITS + row.to(tl.int64) * LOGITS_STRIDE_0
    offs = tl.arange(0, BLOCK)
    for i in range(start, end, BLOCK):
        mask_n = i + offs < end
        x = tl.load(ROW + i + offs, mask=mask_n, other=-float("inf"))
        # Mask in probability space (bitwise the step kernels' predicate);
        # the logit-space pivot rounds to M for huge |M|.
        prob = tl.exp(x - M) / Z
        keep = (prob > pivot) | (x == M)
        if ties:
            dmask = prob == dup
            cum = tl.cumsum(dmask.to(tl.int32)).to(tl.float32) + num_preceding
            keep_dup = dmask & (cum <= numkeep)
            keep = keep & (~dmask | keep_dup)
            num_preceding += tl.sum(dmask.to(tl.float32))
        tl.store(ROW + i + offs, tl.where(keep, x, MASK_VALUE), mask=mask_n)


def _topp_split_mask_warmup_inputs(vllm_config: Any) -> dict[str, Any]:
    vocab_size = vllm_config.model_config.get_vocab_size()
    num_sm = num_compute_units()
    logits_stride: Any = WarmupChoices(16, 2)
    batch_size: Any = WarmupIntRange(
        1, min(_max_sampler_batch_size(vllm_config), _SPLIT_MAX_BATCH) + 1
    )
    has_k: Any = WarmupChoices(False, True)
    logits = TritonWarmupTensor(
        torch.float32,
        shape=(batch_size, vocab_size),
        strides=(logits_stride, 1),
    )
    return dict(
        logits=logits,
        stats=TritonWarmupTensor(torch.float32),
        parts=TritonWarmupTensor(torch.float32),
        k=TritonWarmupTensor(torch.int32) if has_k else None,
        p=TritonWarmupTensor(torch.float32),
        mask_value=float("-inf"),
        num_sm=num_sm,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_topp_sb_mask_kernel,
    warmup_inputs=_topp_split_mask_warmup_inputs,
)
def _topp_split_mask(
    logits: torch.Tensor,
    stats: torch.Tensor,
    parts: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor,
    mask_value: float,
    num_sm: int,
) -> DispatchSpec:
    batch_size, vocab_size = logits.shape
    splits = _topp_split_count(batch_size, vocab_size, num_sm)
    return (logits.shape[0] * splits,), dict(
        K=k if k is not None else logits,
        HAS_K=k is not None,
        S=splits,
        F=_SPLIT_FANOUT,
        NUM_ROUNDS=_SPLIT_ROUNDS,
        VOCAB_SIZE=vocab_size,
        BLOCK=8192,
        num_warps=8,
    )


# ---------------------------------------------------------------------------
# The split-row pipeline in one launch (VLLM_TOPP_SPLIT_FUSED=1).
#
# Each program is one (row, slice), as in the kernels above, but it carries
# its slice through every phase: stats, the rounds, then the mask. After
# each phase it publishes its partials, meets the row's other S programs at
# a barrier (an arrival counter per row), and combines the published
# partials itself, as the next kernel of the pipeline would. Every phase
# runs the code of its kernel above on the same tiles, so every partial,
# every combine and the kept set are bitwise the pipeline's. What goes away
# is six launches, the row-stats reload of every kernel and the combine
# replay (15 combines per row become 5).
#
# Within a round, a tile whose probabilities all lie outside the round's
# pivot ladder takes an exact shortcut. With nothing above the lowest pivot
# it adds nothing to any partial (+0.0, an inf minimum and a zero count are
# no-ops), so it is skipped. With nothing between the lowest and the
# highest pivot, every pivot sees the same set above it, so one row of the
# (F, BLOCK) reduction, with the same layout and thus the same reduction
# tree as each row of the full one, gives all F partials.
#
# A program takes its (row, slice) from a ticket counter instead of its
# program id, so the programs that run always hold a prefix of the tickets
# and a row's barrier only waits for programs that are already running:
# whatever order the hardware starts them in, nothing deadlocks as long as
# S programs fit on the GPU at once. The last program out zeroes the
# counters for the next launch, so the workspace needs no reset (CUDA-graph
# safe). Calls must not overlap on one device, as for the pipeline above.
# ---------------------------------------------------------------------------

# Counters after the per-row arrival counters: the ticket, then the exits.
_FUSED_SYNC_SLOTS = _SPLIT_MAX_BATCH + 2
# The step kernel's tile (so its reductions' layout); the mask uses it too.
_SPLIT_FUSED_STEP_BLOCK = 2048
# Two programs per SM (8 warps each), so more rows run at once.
_SPLIT_FUSED_MAXNREG = 128

_topp_split_fused_enabled: bool | None = None
_topp_split_fused_logged = False


def topp_split_fused_enabled() -> bool:
    """VLLM_TOPP_SPLIT_FUSED=1, read once per process: run the small-batch
    top-p pipeline in one launch (CUDA only)."""
    global _topp_split_fused_enabled
    if _topp_split_fused_enabled is None:
        _topp_split_fused_enabled = os.environ.get(
            "VLLM_TOPP_SPLIT_FUSED", "0"
        ).lower() in ("1", "true")
        if _topp_split_fused_enabled and not current_platform.is_cuda():
            logger.info(
                "VLLM_TOPP_SPLIT_FUSED=1 is CUDA only; top-p masking for "
                "small batches keeps its multi-launch pipeline here."
            )
            _topp_split_fused_enabled = False
    return _topp_split_fused_enabled


def _topp_split_fused_fits(logits: torch.Tensor, num_sm: int) -> bool:
    """A row's programs must all fit on the GPU at once."""
    splits = _topp_split_count(logits.shape[0], logits.shape[1], num_sm)
    if splits <= num_sm:
        return True
    logger.info_once(
        "VLLM_TOPP_SPLIT_FUSED=1: %d programs per row exceed %d SMs; top-p "
        "masking for small batches keeps its multi-launch pipeline.",
        splits,
        num_sm,
    )
    return False


@triton.jit
def _topp_sb_row_barrier(COUNTER, target):
    """Arrive at a row's barrier and wait until `target` arrivals."""
    # Every thread's stores of this phase come before the arrival.
    tl.debug_barrier()
    arrived = tl.atomic_add(COUNTER, 1, sem="acq_rel", scope="gpu") + 1
    while arrived < target:
        arrived = tl.atomic_add(COUNTER, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()


@triton.jit
def _topp_sb_tile(
    ROW,
    i,
    end,
    offs,
    M,
    Z,
    pivs,
    piv_lo,
    piv_hi,
    pmax_t,
    first_round,
    F: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One round's partials of one tile: the step kernel's loop body, with
    exact shortcuts. Returns (sum, min, count) per pivot, an upper bound on
    the tile's probabilities, and (first round) the tile's largest logit."""
    tsum = tl.zeros([F], dtype=tl.float32)
    tmin = tl.full([F], float("inf"), dtype=tl.float32)
    tcnt = tl.zeros([F], dtype=tl.float32)
    pmax_new = piv_lo
    xmax = tl.full((), float("inf"), dtype=tl.float32)
    # Nothing above the lowest pivot: the tile adds nothing to any partial
    # (+0.0, an inf minimum and a zero count), so it is not even loaded.
    if pmax_t > piv_lo:
        x = tl.load(ROW + i + offs, mask=i + offs < end, other=-float("inf"))
        pr = tl.exp(x - M) / Z
        # 2: some probability inside the ladder (and all of the first round);
        # 1: the same set above every pivot; 0: nothing above any pivot.
        path = tl.full((), 2, dtype=tl.int32)
        if first_round:
            xmax = tl.max(x)
        else:
            above_lo = (pr > piv_lo).to(tl.int32)
            above_hi = (pr > piv_hi).to(tl.int32)
            counts = tl.sum(above_lo + above_hi * 4096)
            n_above = counts % 4096
            n_top = counts // 4096
            path = tl.where(n_above > n_top, 2, tl.where(n_top > 0, 1, 0))
        if path == 2:
            pr2 = tl.broadcast_to(pr[None, :], (F, BLOCK))
            above = pr2 > pivs[:, None]
            tsum = tl.sum(tl.where(above, pr2, 0.0), axis=1)
            cand = tl.where(above, pr2, float("inf"))
            tmin = tl.min(cand, axis=1)
            teq = above & (tl.abs(pr2 - tmin[:, None]) < 1e-9)
            tcnt = tl.sum(teq.to(tl.float32), axis=1)
            # The lowest pivot with nothing above it bounds the tile.
            pmax_new = tl.min(tl.where(tmin == float("inf"), pivs, float("inf")))
        elif path == 1:
            # Every pivot sees the same set: one row of the reduction, in the
            # same layout and so the same tree as each row of the full one.
            pr1 = pr[None, :]
            above1 = pr1 > piv_hi
            tsum1 = tl.sum(tl.where(above1, pr1, 0.0), axis=1)
            tmin1 = tl.min(tl.where(above1, pr1, float("inf")), axis=1)
            teq1 = above1 & (tl.abs(pr1 - tmin1[:, None]) < 1e-9)
            tcnt1 = tl.sum(teq1.to(tl.float32), axis=1)
            tsum = tl.zeros([F], dtype=tl.float32) + tsum1
            tmin = tl.zeros([F], dtype=tl.float32) + tmin1
            tcnt = tl.zeros([F], dtype=tl.float32) + tcnt1
            pmax_new = tl.full((), float("inf"), dtype=tl.float32)
    return tsum, tmin, tcnt, pmax_new, xmax


@triton.jit
def _topp_sb_fold(mass, minl, nmin, tsum, tmin, tcnt):
    """The step kernel's per-tile accumulation, in tile order."""
    mass = mass + tsum
    is_new = tmin < minl
    is_same = tl.abs(tmin - minl) < 1e-9
    nmin = tl.where(is_new, tcnt, nmin + tl.where(is_same, tcnt, 0.0))
    minl = tl.minimum(minl, tmin)
    return mass, minl, nmin


@triton.jit
def _topp_sb_fused_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    STATS,
    PARTS,
    SYNC,
    K,
    P,
    HAS_K: tl.constexpr,
    MASK_VALUE: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    NUM_ROUNDS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    STATS_BLOCK: tl.constexpr,
    STEP_BLOCK: tl.constexpr,
    TILES: tl.constexpr,
    TILES_P2: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    SYNC_SLOTS: tl.constexpr,
):
    ticket = tl.atomic_add(SYNC + MAX_ROWS, 1, sem="relaxed", scope="gpu")
    row = ticket // S
    slice_id = ticket % S
    p = tl.load(P + row)
    skip = p >= 1.0
    if HAS_K:
        skip = skip | (tl.load(K + row) < VOCAB_SIZE)
    if skip == 0:
        slice_len = (VOCAB_SIZE + S - 1) // S
        start = slice_id * slice_len
        end = tl.minimum(start + slice_len, VOCAB_SIZE)
        ROW = LOGITS + row.to(tl.int64) * LOGITS_STRIDE_0
        # Phase 1, _topp_sb_stats_kernel: this slice's online-softmax
        # partials (max, sum_exp, min, finite_count).
        offs_stats = tl.arange(0, STATS_BLOCK)
        m = -float("inf")
        exp_sum = 0.0
        mn = float("inf")
        cnt = tl.zeros((), dtype=tl.float32)
        for i in range(start, end, STATS_BLOCK):
            xs = tl.load(
                ROW + i + offs_stats, mask=i + offs_stats < end, other=-float("inf")
            )
            finite = xs > -float("inf")
            nm = tl.maximum(m, tl.max(xs))
            rescale = tl.where(m == -float("inf"), 0.0, tl.exp(m - nm))
            e = tl.where(finite, tl.exp(xs - nm), 0.0)
            exp_sum = exp_sum * rescale + tl.sum(e)
            m = nm
            mn = tl.minimum(mn, tl.min(tl.where(finite, xs, float("inf"))))
            cnt += tl.sum(finite.to(tl.float32))
        base = (row * S + slice_id) * 4
        tl.store(STATS + base + 0, m)
        tl.store(STATS + base + 1, exp_sum)
        tl.store(STATS + base + 2, mn)
        tl.store(STATS + base + 3, cnt)
        _topp_sb_row_barrier(SYNC + row, S)

        M, Z, mn, num_finite = _topp_sb_row_stats(STATS, row, S)
        max_prob = 1.0 / Z
        min_prob = tl.maximum(tl.exp(mn - M) / Z, 1e-30)
        # The step and mask kernels' early returns: no finite logit, or a
        # (nearly) uniform row, keeps everything.
        stop = (Z <= 0.0) | (num_finite < 1.0)
        stop = stop | (max_prob <= min_prob * (1.0 + 1e-6))
        if stop == 0:
            # Phase 2, _topp_sb_step_kernel per round, each round's combine
            # done once here instead of replayed by every later kernel.
            lo = min_prob
            hi = max_prob
            best = -1.0
            best_mass = 0.0
            best_minl = 1.0
            best_nmin = 0.0
            done = 0.0
            nomask = 0.0
            pivot = 0.0
            dup = 1.0
            numdup = 1.0
            numkeep = 1.0
            fin = -1.0
            best_fin = -1.0
            offs_step = tl.arange(0, STEP_BLOCK)
            fidx = tl.arange(0, F)
            # Exact bounds on the slice's tiles, refined every round: no
            # probability in tile t exceeds tile_pmax[t], and tile_xmax[t] is
            # its largest logit.
            tidx = tl.arange(0, TILES_P2)
            tile_pmax = tl.full([TILES_P2], float("inf"), dtype=tl.float32)
            tile_xmax = tl.full([TILES_P2], float("inf"), dtype=tl.float32)
            for j in range(NUM_ROUNDS):
                if done == 0.0:
                    pivs = _topp_sb_ladder(lo, hi, F)
                    piv_lo = tl.min(pivs)
                    piv_hi = tl.max(pivs)
                    mass = tl.zeros([F], dtype=tl.float32)
                    minl = tl.full([F], 1.0, dtype=tl.float32)
                    nmin = tl.zeros([F], dtype=tl.float32)
                    for t in range(TILES):
                        here = tidx == t
                        tsum, tmin, tcnt, pmax_new, xmax = _topp_sb_tile(
                            ROW,
                            start + t * STEP_BLOCK,
                            end,
                            offs_step,
                            M,
                            Z,
                            pivs,
                            piv_lo,
                            piv_hi,
                            tl.max(tl.where(here, tile_pmax, 0.0)),
                            j == 0,
                            F=F,
                            BLOCK=STEP_BLOCK,
                        )
                        tile_pmax = tl.where(
                            here, tl.minimum(tile_pmax, pmax_new), tile_pmax
                        )
                        if j == 0:
                            tile_xmax = tl.where(here, xmax, tile_xmax)
                        mass, minl, nmin = _topp_sb_fold(
                            mass, minl, nmin, tsum, tmin, tcnt
                        )
                    pslice = PARTS + ((row * NUM_ROUNDS + j) * S + slice_id) * F * 3
                    tl.store(pslice + fidx * 3 + 0, mass)
                    tl.store(pslice + fidx * 3 + 1, minl)
                    tl.store(pslice + fidx * 3 + 2, nmin)
                    _topp_sb_row_barrier(SYNC + row, S * (j + 2))
                    (
                        done,
                        nomask,
                        pivot,
                        dup,
                        numdup,
                        numkeep,
                        lo,
                        hi,
                        best,
                        best_mass,
                        best_minl,
                        best_nmin,
                        sel_fidx,
                        best_sel,
                    ) = _topp_sb_combine(
                        PARTS + (row * NUM_ROUNDS + j) * S * F * 3,
                        Z,
                        p,
                        lo,
                        hi,
                        best,
                        best_mass,
                        best_minl,
                        best_nmin,
                        S=S,
                        F=F,
                        IS_LAST=j == NUM_ROUNDS - 1,
                    )
                    if sel_fidx >= 0.0:
                        fin = j * F + sel_fidx
                    if best_sel >= 0.0:
                        best_fin = j * F + best_sel
            if nomask == 0.0:
                # Phase 3, _topp_sb_mask_kernel. The mask is elementwise (and
                # the tie budget an exact count in index order), so its tile
                # size does not change it; it uses the rounds' tiles.
                if fin < 0.0:
                    fin = best_fin
                ties = numkeep < numdup
                num_preceding = 0.0
                if ties:
                    fin_round = tl.cast(fin / F, tl.int32)
                    fin_slot = tl.cast(fin - tl.cast(fin_round, tl.float32) * F, tl.int32)
                    pbase = PARTS + ((row * NUM_ROUNDS + fin_round) * S * F + fin_slot) * 3
                    sidx = tl.arange(0, S)
                    minl_s = tl.load(pbase + sidx * (F * 3) + 1)
                    nmin_s = tl.load(pbase + sidx * (F * 3) + 2)
                    dupc_s = tl.where(tl.abs(minl_s - dup) < 1e-9, nmin_s, 0.0)
                    num_preceding = tl.sum(tl.where(sidx < slice_id, dupc_s, 0.0))
                for t in range(TILES):
                    here = tidx == t
                    pmax_t = tl.max(tl.where(here, tile_pmax, 0.0))
                    xmax_t = tl.max(tl.where(here, tile_xmax, -float("inf")))
                    i = start + t * STEP_BLOCK
                    mask_n = i + offs_step < end
                    # No probability above the pivot and no logit at the max:
                    # nothing is kept (the boundary value `dup` is above the
                    # pivot too), so the tile is masked without a load.
                    if (pmax_t <= pivot) & (xmax_t < M):
                        tl.store(
                            ROW + i + offs_step,
                            tl.full([STEP_BLOCK], MASK_VALUE, dtype=tl.float32),
                            mask=mask_n,
                        )
                    else:
                        xm = tl.load(ROW + i + offs_step, mask=mask_n, other=-float("inf"))
                        prob = tl.exp(xm - M) / Z
                        keep = (prob > pivot) | (xm == M)
                        if ties:
                            dmask = prob == dup
                            cum = (
                                tl.cumsum(dmask.to(tl.int32)).to(tl.float32)
                                + num_preceding
                            )
                            keep_dup = dmask & (cum <= numkeep)
                            keep = keep & (~dmask | keep_dup)
                            num_preceding += tl.sum(dmask.to(tl.float32))
                        tl.store(
                            ROW + i + offs_step,
                            tl.where(keep, xm, MASK_VALUE),
                            mask=mask_n,
                        )
    # The last program out zeroes the counters for the next launch.
    tl.debug_barrier()
    exits = tl.atomic_add(SYNC + MAX_ROWS + 1, 1, sem="acq_rel", scope="gpu")
    if exits == tl.num_programs(0) - 1:
        sync_idx = tl.arange(0, SYNC_SLOTS)
        tl.store(
            SYNC + sync_idx,
            tl.zeros([SYNC_SLOTS], dtype=tl.int32),
            mask=sync_idx < MAX_ROWS + 2,
        )


def _topp_split_fused_warmup_inputs(vllm_config: Any) -> dict[str, Any]:
    vocab_size = vllm_config.model_config.get_vocab_size()
    num_sm = num_compute_units()
    logits_stride: Any = WarmupChoices(16, 2)
    batch_size: Any = WarmupIntRange(
        1, min(_max_sampler_batch_size(vllm_config), _SPLIT_MAX_BATCH) + 1
    )
    has_k: Any = WarmupChoices(False, True)
    logits = TritonWarmupTensor(
        torch.float32,
        shape=(batch_size, vocab_size),
        strides=(logits_stride, 1),
    )
    return dict(
        logits=logits,
        stats=TritonWarmupTensor(torch.float32),
        parts=TritonWarmupTensor(torch.float32),
        sync=TritonWarmupTensor(torch.int32),
        k=TritonWarmupTensor(torch.int32) if has_k else None,
        p=TritonWarmupTensor(torch.float32),
        mask_value=float("-inf"),
        num_sm=num_sm,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_topp_sb_fused_kernel,
    warmup_inputs=_topp_split_fused_warmup_inputs,
)
def _topp_split_fused(
    logits: torch.Tensor,
    stats: torch.Tensor,
    parts: torch.Tensor,
    sync: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor,
    mask_value: float,
    num_sm: int,
) -> DispatchSpec:
    batch_size, vocab_size = logits.shape
    splits = _topp_split_count(batch_size, vocab_size, num_sm)
    tiles = triton.cdiv(triton.cdiv(vocab_size, splits), _SPLIT_FUSED_STEP_BLOCK)
    return (batch_size * splits,), dict(
        K=k if k is not None else logits,
        HAS_K=k is not None,
        S=splits,
        F=_SPLIT_FANOUT,
        NUM_ROUNDS=_SPLIT_ROUNDS,
        VOCAB_SIZE=vocab_size,
        STATS_BLOCK=8192,
        STEP_BLOCK=_SPLIT_FUSED_STEP_BLOCK,
        TILES=tiles,
        TILES_P2=next_power_of_2(tiles),
        MAX_ROWS=_SPLIT_MAX_BATCH,
        SYNC_SLOTS=next_power_of_2(_FUSED_SYNC_SLOTS),
        num_warps=8,
        maxnreg=_SPLIT_FUSED_MAXNREG,
    )


def _apply_topp_split(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor,
    mask_value: float,
    num_sm: int,
) -> None:
    """Split-row top-p pipeline for small batches; masks logits in place."""
    global _topp_split_fused_logged
    ws = _TRITON_SPLIT_CACHE.get(logits.device)
    if ws is None:
        ws = {
            "stats": logits.new_empty((_SPLIT_MAX_BATCH * _SPLIT_MAX_SPLITS, 4)),
            "parts": logits.new_empty(
                (_SPLIT_MAX_BATCH, _SPLIT_ROUNDS, _SPLIT_MAX_SPLITS, _SPLIT_FANOUT, 3)
            ),
        }
        _TRITON_SPLIT_CACHE[logits.device] = ws

    if topp_split_fused_enabled() and _topp_split_fused_fits(logits, num_sm):
        sync = ws.get("sync")
        if sync is None:
            # Zeroed once; every launch leaves it zeroed.
            sync = torch.zeros(
                _FUSED_SYNC_SLOTS, dtype=torch.int32, device=logits.device
            )
            ws["sync"] = sync
        if not _topp_split_fused_logged:
            _topp_split_fused_logged = True
            batch_size, vocab_size = logits.shape
            logger.info(
                "Top-p masking for small batches runs in one launch "
                "(VLLM_TOPP_SPLIT_FUSED=1): first call %d rows x %d vocab, %d "
                "programs per row, top-k %s.",
                batch_size,
                vocab_size,
                _topp_split_count(batch_size, vocab_size, num_sm),
                "on" if k is not None else "off",
            )
        _topp_split_fused(
            logits,
            ws["stats"],
            ws["parts"],
            sync,
            k,
            p,
            mask_value,
            num_sm,
        )
        return

    _topp_split_stats(
        logits,
        ws["stats"],
        k,
        p,
        num_sm,
    )
    for round_i in range(_SPLIT_ROUNDS):
        _topp_split_step(
            logits,
            ws["stats"],
            ws["parts"],
            k,
            p,
            round_i,
            num_sm,
        )
    _topp_split_mask(
        logits,
        ws["stats"],
        ws["parts"],
        k,
        p,
        mask_value,
        num_sm,
    )


def apply_top_k_top_p_triton(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    mask_value: float = float("-inf"),
    has_topp_only_rows: bool = True,
) -> torch.Tensor:
    """Apply combined top-k and top-p masking using Triton.

    Top-k is applied first (by logit value), then top-p is applied
    to the remaining k values (by probability).

    Args:
        logits: [batch_size, vocab_size] float32 tensor. The returned tensor
            may alias this input or be a new contiguous tensor for unsupported
            layouts.
        k: [batch_size] int32 tensor of top-k values per row, or None to disable top-k
        p: [batch_size] float32 tensor of top-p values per row (0 to 1),
            or None to disable top-p
        mask_value: Value for masked positions (default: -inf)
        has_topp_only_rows: Whether any row has p < 1 and k >= vocab_size.
            False lets small batches skip the split top-p pipeline; pass it
            only if no such row exists, as such a row would get no top-p.

    Returns:
        The masked logits tensor. It may or may not be modified in-place.

    """
    assert logits.ndim == 2
    assert logits.dtype == torch.float32
    batch_size, vocab_size = logits.shape
    topk_enabled = k is not None
    topp_enabled = p is not None

    if batch_size == 0 or not (topk_enabled or topp_enabled):
        return logits

    # The Triton kernel supports arbitrary row strides, but it still assumes
    # the vocab dimension is laid out contiguously within each row.
    if logits.stride(1) != 1:
        logits = logits.contiguous()

    if k is not None:
        assert k.ndim == 1 and k.shape[0] == batch_size
        k_ptr = k.to(torch.int32)
    else:
        k_ptr = logits  # Dummy pointer (won't be read)

    if p is not None:
        assert p.ndim == 1 and p.shape[0] == batch_size
        p_ptr = p.to(torch.float32)
    else:
        p_ptr = logits  # Dummy pointer (won't be read)

    num_sm = num_compute_units(logits.device.index)

    # At small batch sizes the monolithic kernel's standalone top-p path is
    # latency-bound (one program per row, serial vocab sweeps), so use the
    # split-row pipeline for rows without an active top-k. Rows with an
    # active top-k still go through the monolithic kernel (its truncation
    # path is already fast). The monolithic's standalone top-p branch then
    # covers only rows whose top-k is a no-op (k < vocab but <= k finite
    # logits, e.g. grammar masks), which the split pipeline skips.
    use_split = (
        topp_enabled and batch_size <= _SPLIT_MAX_BATCH and logits.device.type == "cuda"
    )
    if use_split and not topk_enabled:
        _apply_topp_split(logits, None, p_ptr, mask_value, num_sm)
        return logits

    NUM_PROGRAMS = min(num_sm, batch_size)

    # Cache per-Triton Program buffer on each device.
    buf_key = (logits.device, logits.dtype, vocab_size)
    buffer = _TRITON_BUFFER_CACHE.get(buf_key)
    if buffer is None or buffer.shape[0] < NUM_PROGRAMS:
        size = min(next_power_of_2(NUM_PROGRAMS), num_sm)
        buffer = logits.new_empty((size, vocab_size))
        _TRITON_BUFFER_CACHE[buf_key] = buffer
    if buffer.shape[0] > NUM_PROGRAMS:
        buffer = buffer[:NUM_PROGRAMS]

    # Cache lookup table entries on each device.
    tables = _TRITON_TABLE_CACHE.get(logits.device)
    if tables is None:
        with gpu_sync_allowed():
            normal_cdf_to_sigma_table = logits.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)
            percentile_to_std_table = logits.new_tensor(_PERCENTILE_TO_STD_TABLE)
            _TRITON_TABLE_CACHE[logits.device] = (
                normal_cdf_to_sigma_table,
                percentile_to_std_table,
            )
    else:
        normal_cdf_to_sigma_table, percentile_to_std_table = tables

    _topk_topp(
        logits,
        buffer,
        percentile_to_std_table,
        normal_cdf_to_sigma_table,
        k_ptr if topk_enabled else None,
        p_ptr if topp_enabled else None,
        mask_value,
        num_sm,
    )
    if use_split and has_topp_only_rows:
        _apply_topp_split(logits, k_ptr, p_ptr, mask_value, num_sm)

    return logits


def reset_buffer_cache():
    _TRITON_BUFFER_CACHE.clear()
    _TRITON_TABLE_CACHE.clear()
    _TRITON_SPLIT_CACHE.clear()
    torch.accelerator.empty_cache()
