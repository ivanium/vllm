# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocabulary-sharded LM heads and sampling over the data-parallel ranks.

With data parallelism and no tensor parallelism (a DEP decode instance), every
DP rank holds the whole LM head and streams all of it every step for its own
few rows, and a DSpark drafter does the same with its own head and Markov
head. The ranks run in lockstep (the MoE dispatch), so the rank with the most
rows sets the step, and its LM heads and full-vocabulary sampling are on it.

With VLLM_DP_SHARDED_LM_HEAD=1, DP rank r of W owns the vocabulary slice
[r V / W, (r + 1) V / W) and the ranks share the work of every rank's rows:

* Verification (`DPVocabShard.verify`): the ranks all-gather their logits rows'
  hidden states and sampling metadata, multiply all rows with their slice of the
  head, and run temperature, top-p and the rejection sampler on their slices.
  Top-p runs the split pipeline of `topk_topp_triton.py` with the slice as the
  pipeline's slice (at W = 16 they are the pipeline's own 16 slices of V =
  200064): each rank sweeps its slice with that pipeline's code, the per-slice
  partials are all-gathered and combined by the same `_topp_sb_combine`, so the
  kept set is the replicated one, bit for bit. Rejection sampling runs the
  rejection sampler's per-block kernels on the slice, exchanges their partials
  and the drafted tokens' logits (the idea of vLLM PR #65007's vocab-sharded
  verification over TP), and runs its per-request kernels on every rank. The
  Gumbel noise is keyed by the global token id, so it is the replicated
  sampler's noise. Every rank ends with every request's tokens and keeps its
  own.
* Drafting (`DPVocabShard.draft`, DSpark): the ranks all-gather the draft rows,
  multiply them with their slice of the draft head, and run the Markov chain on
  their slices, exchanging one 8-byte (value, token) word per request and
  position, as the replicated loop's Gumbel-max would pick. The draft logits
  cache holds every rank's requests on the local slice.

The exchanges are one-shot all-gathers over NVLink multicast (the Lamport
mailbox kernel of `models/deepseek_v41/nvidia/ops/cute_dsl/all_gather.py`, on
the DP group). Every rank must run the same exchanges in the same order: a
rank with no requests joins every step with no rows (`join_idle`). A step in
which some rank has a request that needs the full vocabulary (logprobs, top-k,
min-p, penalties and other logits processors, structured output) is verified
in full on every rank after an all-to-all completes its draft logits
(`complete_draft_logits`); the ranks agree on that in the DP padding sync.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.ops.topk_topp_triton import (
    _SPLIT_FANOUT,
    _SPLIT_ROUNDS,
    _topp_sb_combine,
    _topp_sb_ladder,
    _topp_sb_row_stats,
)
from vllm.v1.worker.gpu.sample.gumbel import (
    apply_temperature,
    gumbel_noised_argmax,
    spec_step_seed,
    tl_rand32,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_global_logsumexp,
    _compute_global_residual_mass,
    _compute_global_target_argmax,
    _compute_local_logits_stats_kernel,
    _compute_local_residual_mass_kernel,
    _insert_resampled_kernel,
    _seeded_resample_argmax,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

ENV_VAR = "VLLM_DP_SHARDED_LM_HEAD"

# The rejection sampler's statistics blocks and resampling blocks.
_STATS_BLOCK = 8192
_RESAMPLE_BLOCK = 1024
# Words of the per-row and per-request metadata in the verification payload.
_ROW_WORDS = 4
_REQ_WORDS = 8
# The slice GEMMs run on at least this many rows. Below 33 rows cuBLAS picks a
# different K split for the 12,504-row slice than for the full head (measured
# on GB300: 1-BF16-ulp differences on a few percent of logits); from 33 rows up
# the slice's logits equal the full head's bit for bit.
_MIN_GEMM_ROWS = 64
# Engagement counters are logged every this many verification steps.
_LOG_EVERY = 1000
# Words per request in the draft payload: anchor token, cache row, temperature,
# seed (2 words), then one position per draft step (at most 11 steps).
_DRAFT_REQ_WORDS = 16


def dp_sharded_lm_head_requested() -> bool:
    """VLLM_DP_SHARDED_LM_HEAD=1."""
    return os.environ.get(ENV_VAR, "0").lower() in ("1", "true")


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


# ----------------------------------------------------------------------------
# Exchange: a one-shot all-gather over NVLink multicast on any group.


class _MulticastAllGather:
    """All-gather along dim 0 over NVLink multicast, for int32 inputs of up to
    `capacity_bytes` a rank: DeepSeek-V4.1's Lamport-mailbox kernel
    (`_AllGatherKernel`) with a mailbox on `group`.

    Construction compiles the kernel (local); `attach` allocates the mailbox
    (collective). Calls must be issued in the same order on every rank and on
    one stream. Words equal to 0x80000000 arrive as 0 (FP32 -0.0 becomes +0.0;
    no index or count here takes that value)."""

    def __init__(
        self, group: dist.ProcessGroup, capacity_bytes: int, device: torch.device
    ) -> None:
        import cutlass.cute as cute
        from cutlass import Int32, Int64
        from cutlass.cute.runtime import make_fake_stream

        from vllm.models.common.ops.moe_tail_fusion.primitives import (
            current_cu_stream,
        )
        from vllm.models.deepseek_v41.nvidia.ops.cute_dsl import all_gather as ag

        self._Int32, self._Int64, self._stream = Int32, Int64, current_cu_stream
        self.group = group
        self.device = device
        self.world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        self.capacity_bytes = _round_up(capacity_bytes, ag.PACK_BYTES)
        self._packs = self.capacity_bytes // ag.PACK_BYTES
        self._mailbox_words = (
            ag.GENERATIONS * self.world * self._packs * ag.PACK_BYTES // 4
        )
        self.multicast = 0
        with torch.accelerator.device_index(device.index):
            self._kernel: Any = cute.compile(
                ag._AllGatherKernel(rank, self.world, self._packs),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int32(1),
                make_fake_stream(),
            )

    def allocate(self) -> None:
        """Allocate the mailbox in symmetric memory (local)."""
        import torch.distributed._symmetric_memory as symm_mem

        with torch.accelerator.device_index(self.device.index):
            self._mailbox = symm_mem.empty(
                (self._mailbox_words,), dtype=torch.int32, device=self.device
            )
            self._state = torch.zeros(2, dtype=torch.int32, device=self.device)

    def rendezvous(self) -> None:
        """Map every rank's mailbox (collective)."""
        import torch.distributed._symmetric_memory as symm_mem

        with torch.accelerator.device_index(self.device.index):
            handle = symm_mem.rendezvous(self._mailbox, self.group.group_name)
            self.multicast = int(handle.multicast_ptr or 0)
            self._mailbox.fill_(-(2**31))
            torch.accelerator.synchronize()

    def __call__(self, inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Every rank's `inp` (int32, contiguous) into `out` [world, *inp.shape]."""
        assert inp.dtype == torch.int32 and inp.is_contiguous()
        assert 0 < inp.nbytes <= self.capacity_bytes and inp.nbytes % 16 == 0
        assert out.is_contiguous() and out.numel() == inp.numel() * self.world
        i64, i32 = self._Int64, self._Int32
        self._kernel(
            i64(inp.data_ptr()),
            i64(out.data_ptr()),
            i64(self._mailbox.data_ptr()),
            i64(self.multicast),
            i64(self._state.data_ptr()),
            i32(inp.nbytes // 16),
            self._stream(),
        )
        return out


# ----------------------------------------------------------------------------
# Top-p on one slice: the split pipeline's stats, round and mask kernels
# (topk_topp_triton.py) with the slice id a parameter and the logits pointer on
# this rank's slice. The code per slice is the pipeline's, line for line, so
# with a row stride divisible by 16 (the same vector loads, hence the same
# reduction trees) every partial is bitwise the pipeline's.


@triton.jit
def _topp_shard_stats_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    STATS_OUT,
    P,
    SLICE_ID,
    VOCAB_SIZE: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slice_id = SLICE_ID
    p = tl.load(P + row)
    if p >= 1.0:
        return
    slice_len = (VOCAB_SIZE + S - 1) // S
    start = slice_id * slice_len
    end = tl.minimum(start + slice_len, VOCAB_SIZE)
    ROW = LOGITS + row.to(tl.int64) * LOGITS_STRIDE_0 - start
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
    base = row * 4
    tl.store(STATS_OUT + base + 0, m)
    tl.store(STATS_OUT + base + 1, exp_sum)
    tl.store(STATS_OUT + base + 2, mn)
    tl.store(STATS_OUT + base + 3, cnt)


@triton.jit(do_not_specialize_on_alignment=["ROUND"])
def _topp_shard_step_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    STATS,
    PARTS,
    PARTS_OUT,
    P,
    ROUND,
    SLICE_ID,
    S: tl.constexpr,
    F: tl.constexpr,
    NUM_ROUNDS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slice_id = SLICE_ID
    p = tl.load(P + row)
    if p >= 1.0:
        return
    M, Z, mn, num_finite = _topp_sb_row_stats(STATS, row, S)
    if (Z <= 0.0) or (num_finite < 1.0):
        return
    max_prob = 1.0 / Z
    min_prob = tl.maximum(tl.exp(mn - M) / Z, 1e-30)
    if max_prob <= min_prob * (1.0 + 1e-6):
        return
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
    pivs = _topp_sb_ladder(lo, hi, F)
    slice_len = (VOCAB_SIZE + S - 1) // S
    start = slice_id * slice_len
    end = tl.minimum(start + slice_len, VOCAB_SIZE)
    ROW = LOGITS + row.to(tl.int64) * LOGITS_STRIDE_0 - start
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
    pbase = PARTS_OUT + row * F * 3
    tl.store(pbase + fidx * 3 + 0, mass)
    tl.store(pbase + fidx * 3 + 1, minl)
    tl.store(pbase + fidx * 3 + 2, nmin)


@triton.jit
def _topp_shard_mask_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    STATS,
    PARTS,
    P,
    SLICE_ID,
    MASK_VALUE: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    NUM_ROUNDS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slice_id = SLICE_ID
    p = tl.load(P + row)
    if p >= 1.0:
        return
    M, Z, mn, num_finite = _topp_sb_row_stats(STATS, row, S)
    if (Z <= 0.0) or (num_finite < 1.0):
        return
    max_prob = 1.0 / Z
    min_prob = tl.maximum(tl.exp(mn - M) / Z, 1e-30)
    if max_prob <= min_prob * (1.0 + 1e-6):
        return
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
    slice_len = (VOCAB_SIZE + S - 1) // S
    start = slice_id * slice_len
    end = tl.minimum(start + slice_len, VOCAB_SIZE)
    ROW = LOGITS + row.to(tl.int64) * LOGITS_STRIDE_0 - start
    offs = tl.arange(0, BLOCK)
    for i in range(start, end, BLOCK):
        mask_n = i + offs < end
        x = tl.load(ROW + i + offs, mask=mask_n, other=-float("inf"))
        prob = tl.exp(x - M) / Z
        keep = (prob > pivot) | (x == M)
        if ties:
            dmask = prob == dup
            cum = tl.cumsum(dmask.to(tl.int32)).to(tl.float32) + num_preceding
            keep_dup = dmask & (cum <= numkeep)
            keep = keep & (~dmask | keep_dup)
            num_preceding += tl.sum(dmask.to(tl.float32))
        tl.store(ROW + i + offs, tl.where(keep, x, MASK_VALUE), mask=mask_n)


# ----------------------------------------------------------------------------
# Rejection sampling on one slice. The per-block kernels of
# rejection_sampler_utils.py run on the slice's blocks; these replace the ones
# that index the full vocabulary by token id.


@triton.jit(do_not_specialize=["num_logits"])
def _dp_token_logits_kernel(
    # [num_logits, 2] (row stride out_stride): the processed target logit and
    # the (pre-temperature) draft logit of the token drafted at each row, -inf
    # off this slice.
    out_ptr,
    out_stride,
    target_logits_ptr,
    target_logits_stride,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_reqs] the draft logits cache row of each request
    draft_logits_row_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    draft_sampled_ptr,
    num_logits,
    vocab_offset,
    shard_len,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < num_logits
    has_next = rows + 1 < num_logits
    next_pos = tl.load(expanded_local_pos_ptr + rows + 1, mask=has_next, other=0)
    token = tl.load(draft_sampled_ptr + rows + 1, mask=has_next, other=-1).to(tl.int64)
    col = token - vocab_offset
    # The next row belongs to the same request (it is not a bonus row), and its
    # token lies on this slice.
    owned = in_range & has_next & (next_pos != 0) & (col >= 0) & (col < shard_len)
    target = tl.load(
        target_logits_ptr + rows.to(tl.int64) * target_logits_stride + col,
        mask=owned,
        other=float("-inf"),
    ).to(tl.float32)
    req = tl.load(expanded_idx_mapping_ptr + rows, mask=in_range, other=0).to(tl.int64)
    cache_row = tl.load(draft_logits_row_mapping_ptr + req, mask=in_range, other=-1)
    step = tl.load(expanded_local_pos_ptr + rows, mask=in_range, other=0)
    draft = tl.load(
        draft_logits_ptr
        + tl.maximum(cache_row, 0).to(tl.int64) * draft_logits_stride_0
        + step * draft_logits_stride_1
        + col,
        mask=owned & (cache_row >= 0),
        other=float("-inf"),
    ).to(tl.float32)
    tl.store(out_ptr + rows * out_stride, target, mask=in_range)
    tl.store(out_ptr + rows * out_stride + 1, draft, mask=in_range)


@triton.jit
def _dp_cumulative_log_p_kernel(
    cumulative_log_p_ptr,
    # [num_logits, 2] (see _dp_token_logits_kernel), gathered over the slices
    token_logits_ptr,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_sampled_ptr,
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    temp_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    """_compute_cumulative_log_p_kernel with the drafted tokens' logits read
    from `token_logits_ptr`."""
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp == 0.0:
        return

    log_p = tl.zeros((), tl.float32)
    for step in range(num_draft_tokens):
        logit_idx = start_idx + step
        draft_token = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        if draft_token >= 0:
            target_logit = tl.load(token_logits_ptr + logit_idx * 2)
            target_lse = _compute_global_logsumexp(
                target_local_max_ptr,
                target_local_max_stride,
                target_local_sumexp_ptr,
                target_local_sumexp_stride,
                logit_idx,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            )
            draft_logit = tl.load(token_logits_ptr + logit_idx * 2 + 1) / temp
            draft_lse = _compute_global_logsumexp(
                draft_local_max_ptr,
                draft_local_max_stride,
                draft_local_sumexp_ptr,
                draft_local_sumexp_stride,
                logit_idx,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            )
            target_logprob = target_logit - target_lse
            draft_logprob = draft_logit - draft_lse
            log_p = tl.minimum(log_p + (target_logprob - draft_logprob), 0.0)
        tl.store(cumulative_log_p_ptr + logit_idx, log_p)


@triton.jit
def _dp_rejection_kernel(
    sampled_ptr,
    sampled_stride,
    rejected_steps_ptr,
    target_rejected_logsumexp_ptr,
    draft_rejected_logsumexp_ptr,
    token_logits_ptr,
    target_local_argmax_ptr,
    target_local_argmax_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_sampled_ptr,
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    temp_ptr,
    seed_ptr,
    pos_ptr,
    synthetic_conditional_rates_ptr,
    cumulative_log_p_ptr,
    local_residual_mass_ptr,
    local_residual_mass_stride,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    """_rejection_kernel (probabilistic drafts, no FLy) with the drafted
    tokens' logits read from `token_logits_ptr`."""
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0

    accepted_length = tl.zeros((), tl.int64)
    target_lse = 0.0
    draft_lse = 0.0
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        # -1 is used for placeholder draft token ids that should be rejected.
        is_valid_draft = draft_sampled >= 0
        draft_sampled = tl.maximum(0, draft_sampled)
        if not is_greedy:
            verifying &= is_valid_draft
        if verifying:
            target_argmax = tl.full((), 0, tl.int64)
            accepted = False
            pos = tl.load(pos_ptr + logit_idx)
            step_seed = spec_step_seed(seed, i) if USE_BLOCK_VERIFICATION else seed
            u = tl_rand32(step_seed, pos, includes_zero=False)
            if is_greedy:
                target_argmax = _compute_global_target_argmax(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_argmax_ptr,
                    target_local_argmax_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    accepted = target_argmax == draft_sampled
                accepted &= is_valid_draft
            elif USE_BLOCK_VERIFICATION:
                prefix_joint_ratio = tl.exp(
                    tl.load(cumulative_log_p_ptr + logit_idx).to(tl.float32)
                )
                next_draft_token = tl.load(
                    draft_sampled_ptr + logit_idx + 2,
                    mask=i < num_draft_tokens - 1,
                    other=-1,
                ).to(tl.int64)
                if next_draft_token >= 0:
                    residual_mass = _compute_global_residual_mass(
                        local_residual_mass_ptr,
                        local_residual_mass_stride,
                        prefix_joint_ratio,
                        token_logits_ptr,  # unused with draft logits
                        0,
                        target_local_max_ptr,
                        target_local_max_stride,
                        target_local_sumexp_ptr,
                        target_local_sumexp_stride,
                        next_draft_token,
                        logit_idx + 1,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                        True,  # HAS_DRAFT_LOGITS
                    )
                    denom = residual_mass + 1.0 - prefix_joint_ratio
                    h = tl.where(denom > 0.0, residual_mass / denom, 1.0)
                else:
                    h = prefix_joint_ratio
                accepted_length = tl.where(u <= h, i + 1, accepted_length)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            else:
                target_logit = tl.load(
                    token_logits_ptr + logit_idx * 2,
                    mask=is_valid_draft,
                    other=float("-inf"),
                )
                step_target_lse = _compute_global_logsumexp(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_sumexp_ptr,
                    target_local_sumexp_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                draft_logit = (
                    tl.load(
                        token_logits_ptr + logit_idx * 2 + 1,
                        mask=is_valid_draft,
                        other=float("-inf"),
                    )
                    / temp
                )
                step_draft_lse = _compute_global_logsumexp(
                    draft_local_max_ptr,
                    draft_local_max_stride,
                    draft_local_sumexp_ptr,
                    draft_local_sumexp_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                target_logprob = target_logit - step_target_lse
                draft_logprob = draft_logit - step_draft_lse
                target_lse = step_target_lse
                draft_lse = step_draft_lse
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    accepted = target_logprob > tl.log(u) + draft_logprob

            if not USE_BLOCK_VERIFICATION or is_greedy:
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    target_argmax if is_greedy and not accepted else draft_sampled,
                )

    tl.store(rejected_steps_ptr + req_idx, accepted_length)
    if USE_BLOCK_VERIFICATION and not is_greedy and accepted_length < num_draft_tokens:
        rejected_idx = start_idx + accepted_length
        target_lse = _compute_global_logsumexp(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sumexp_ptr,
            target_local_sumexp_stride,
            rejected_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        draft_lse = _compute_global_logsumexp(
            draft_local_max_ptr,
            draft_local_max_stride,
            draft_local_sumexp_ptr,
            draft_local_sumexp_stride,
            rejected_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
    tl.store(target_rejected_logsumexp_ptr + req_idx, target_lse)
    tl.store(draft_rejected_logsumexp_ptr + req_idx, draft_lse)


@triton.jit
def _dp_resample_kernel(
    # [num_reqs, pack_stride] int32: this slice's per-block argmax (global
    # token id) in [0, NB), the per-block maximum's bits in [NB, 2 * NB)
    pack_ptr,
    pack_stride,
    NB,
    target_logits_ptr,
    target_logits_stride,
    target_rejected_logsumexp_ptr,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_logits_row_mapping_ptr,
    draft_rejected_logsumexp_ptr,
    rejected_step_ptr,
    cu_num_logits_ptr,
    expanded_idx_mapping_ptr,
    draft_sampled_ptr,
    temp_ptr,
    seed_ptr,
    pos_ptr,
    cumulative_log_p_ptr,
    vocab_size,
    vocab_offset,
    shard_len,
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    """_resample_kernel (probabilistic drafts, no watermark) on this slice's
    blocks. The Gumbel noise is keyed by the global token id, so it is the
    replicated sampler's noise."""
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx).to(tl.int64)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        return

    rejected_draft_token = tl.load(
        draft_sampled_ptr + resample_token_idx + 1,
        mask=not is_bonus,
        other=0,
    )
    is_valid_rejected_draft = rejected_draft_token >= 0

    block_idx = tl.program_id(1)
    local = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = local < shard_len
    block = local + vocab_offset
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + local,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    out = pack_ptr + req_idx * pack_stride + block_idx
    if tl.max(target_logits, axis=0) == float("-inf"):
        # No target token survives in this block, so neither does the residual.
        tl.store(out, (vocab_offset + block_idx * BLOCK_SIZE).to(tl.int32))
        tl.store(out + NB, tl.full((), float("-inf"), tl.float32).to(tl.int32, bitcast=True))
        return

    if is_bonus or not is_valid_rejected_draft:
        residual_logits = target_logits
    else:
        cache_row = tl.load(draft_logits_row_mapping_ptr + req_state_idx)
        draft_logits = (
            tl.load(
                draft_logits_ptr
                + tl.maximum(cache_row, 0).to(tl.int64) * draft_logits_stride_0
                + resample_idx * draft_logits_stride_1
                + local,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            / temp
        )
        target_lse = tl.load(target_rejected_logsumexp_ptr + req_idx)
        draft_lse = tl.load(draft_rejected_logsumexp_ptr + req_idx)
        target_log_probs = target_logits - target_lse
        if USE_BLOCK_VERIFICATION:
            log_p_tau = 0.0
            if resample_idx > 0:
                log_p_tau = tl.load(cumulative_log_p_ptr + resample_token_idx - 1).to(
                    tl.float32
                )
            target_log_probs += log_p_tau
        draft_log_probs = draft_logits - draft_lse
        ratio = tl.exp(draft_log_probs - target_log_probs)
        residual_logits = tl.where(
            ratio < 1.0,
            target_log_probs + tldevice.log1p(-ratio),
            float("-inf"),
        ).to(tl.float32)

    value, idx = _seeded_resample_argmax(
        residual_logits,
        block,
        mask,
        resample_token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        vocab_size,
        USE_FP64=False,
    )
    tl.store(out, (vocab_offset + block_idx * BLOCK_SIZE + idx).to(tl.int32))
    tl.store(out + NB, value.to(tl.float32).to(tl.int32, bitcast=True))


# ----------------------------------------------------------------------------
# Packing and unpacking the verification exchanges (one launch each).


@triton.jit(do_not_specialize=["num_rows", "num_reqs"])
def _dp_pack_meta_kernel(
    row_meta_ptr,  # [rows, ROW_WORDS] int32 (payload)
    req_meta_ptr,  # [reqs, REQ_WORDS] int32 (payload)
    host_ptr,  # int32: req_local [rows], then the request words [reqs, REQ_WORDS]
    local_pos_ptr,
    draft_sampled_ptr,
    pos_ptr,
    num_rows,
    num_reqs,
    ROW_WORDS: tl.constexpr,
    REQ_WORDS: tl.constexpr,
):
    i = tl.program_id(0)
    if i < num_rows:
        out = row_meta_ptr + i * ROW_WORDS
        tl.store(out + 0, tl.load(host_ptr + i))
        tl.store(out + 1, tl.load(local_pos_ptr + i))
        tl.store(out + 2, tl.load(draft_sampled_ptr + i))
        tl.store(out + 3, tl.load(pos_ptr + i).to(tl.int32))
    else:
        q = i - num_rows
        w = tl.arange(0, REQ_WORDS)
        vals = tl.load(host_ptr + num_rows + q * REQ_WORDS + w)
        tl.store(req_meta_ptr + q * REQ_WORDS + w, vals)


@triton.jit(do_not_specialize=["num_rows", "num_reqs", "words_per_rank", "row_cap", "req_cap"])
def _dp_unpack_step_kernel(
    gathered_ptr,  # [world, words_per_rank] int32
    words_per_rank,
    row_cap,
    req_cap,
    # [2 * rows + 2 * reqs] int32 (see _Layout.host_index)
    index_ptr,
    num_rows,
    num_reqs,
    hidden_ptr,  # [rows, H_WORDS] int32
    expanded_req_ptr,  # [rows] int32
    local_pos_ptr,  # [rows] int32
    draft_sampled_ptr,  # [rows] int32
    pos_ptr,  # [rows] int64
    top_p_ptr,  # [rows] fp32
    cache_row_ptr,  # [reqs] int32
    temp_ptr,  # [reqs] fp32
    seed_ptr,  # [reqs] int64
    cu_ptr,  # [reqs + 1] int32
    H_WORDS: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    REQ_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    meta_base = row_cap * H_WORDS
    req_base = row_cap * (H_WORDS + ROW_WORDS)
    if i < num_rows:
        src = tl.load(index_ptr + i).to(tl.int64)
        rank = src // row_cap
        j = src - rank * row_cap
        base = gathered_ptr + rank * words_per_rank
        meta = base + meta_base + j * ROW_WORDS
        req = tl.load(meta + 0) + tl.load(index_ptr + num_rows + i)
        tl.store(expanded_req_ptr + i, req)
        tl.store(local_pos_ptr + i, tl.load(meta + 1))
        tl.store(draft_sampled_ptr + i, tl.load(meta + 2))
        tl.store(pos_ptr + i, tl.load(meta + 3).to(tl.int64))
        qsrc = tl.load(index_ptr + 2 * num_rows + req).to(tl.int64)
        qrank = qsrc // req_cap
        qj = qsrc - qrank * req_cap
        qmeta = gathered_ptr + qrank * words_per_rank + req_base + qj * REQ_WORDS
        tl.store(top_p_ptr + i, tl.load(qmeta + 2).to(tl.float32, bitcast=True))
        src_h = base + j * H_WORDS
        dst_h = hidden_ptr + i.to(tl.int64) * H_WORDS
        for k in range(0, H_WORDS, BLOCK):
            offs = k + tl.arange(0, BLOCK)
            m = offs < H_WORDS
            tl.store(dst_h + offs, tl.load(src_h + offs, mask=m), mask=m)
    else:
        q = i - num_rows
        qsrc = tl.load(index_ptr + 2 * num_rows + q).to(tl.int64)
        qrank = qsrc // req_cap
        qj = qsrc - qrank * req_cap
        qmeta = gathered_ptr + qrank * words_per_rank + req_base + qj * REQ_WORDS
        tl.store(cache_row_ptr + q, tl.load(qmeta + 0))
        tl.store(temp_ptr + q, tl.load(qmeta + 1).to(tl.float32, bitcast=True))
        seed_lo = tl.load(qmeta + 3).to(tl.uint32, bitcast=True).to(tl.int64)
        seed_hi = tl.load(qmeta + 4).to(tl.int64)
        tl.store(seed_ptr + q, (seed_hi << 32) | seed_lo)
        row_off = tl.load(index_ptr + 2 * num_rows + num_reqs + q)
        tl.store(cu_ptr + q, tl.load(qmeta + 6) + row_off)
        if q == num_reqs - 1:
            tl.store(cu_ptr + num_reqs, num_rows)


@triton.jit(do_not_specialize=["num_rows"])
def _dp_unpack_stats_kernel(
    gathered_ptr,  # [world, rows, WORDS] int32
    num_rows,
    shard_len,
    t_argmax_ptr,  # [rows, world * NB] int64
    t_max_ptr,  # [rows, world * NB] fp32
    t_sum_ptr,
    d_max_ptr,
    d_sum_ptr,
    token_logits_ptr,  # [rows, 2] fp32
    WORLD: tl.constexpr,
    NB: tl.constexpr,
    NB_P2: tl.constexpr,
    WORDS: tl.constexpr,
):
    """Rank r's per-block statistics of row g go to columns [r * NB, + NB),
    in token-id order; its block argmax is made a global token id. The drafted
    token's logits are the owning rank's (the others hold -inf)."""
    g = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, WORLD)[:, None]
    b = tl.arange(0, NB_P2)[None, :]
    mb = b < NB
    src = gathered_ptr + (r * num_rows + g) * WORDS
    dst = g * (WORLD * NB) + r * NB + b
    am = tl.load(src + b, mask=mb, other=0).to(tl.int64) + r * shard_len
    tl.store(t_argmax_ptr + dst, am, mask=mb)
    for k in tl.static_range(4):
        vals = tl.load(src + (1 + k) * NB + b, mask=mb, other=0).to(tl.float32, bitcast=True)
        if k == 0:
            tl.store(t_max_ptr + dst, vals, mask=mb)
        elif k == 1:
            tl.store(t_sum_ptr + dst, vals, mask=mb)
        elif k == 2:
            tl.store(d_max_ptr + dst, vals, mask=mb)
        else:
            tl.store(d_sum_ptr + dst, vals, mask=mb)
    rr = tl.arange(0, WORLD)
    tsrc = gathered_ptr + (rr * num_rows + g) * WORDS + 5 * NB
    tgt = tl.load(tsrc).to(tl.float32, bitcast=True)
    drf = tl.load(tsrc + 1).to(tl.float32, bitcast=True)
    tl.store(token_logits_ptr + g * 2, tl.max(tgt, axis=0))
    tl.store(token_logits_ptr + g * 2 + 1, tl.max(drf, axis=0))


@triton.jit(do_not_specialize=["num_reqs"])
def _dp_unpack_resample_kernel(
    gathered_ptr,  # [world, reqs, WORDS] int32: argmax (global id), max bits
    num_reqs,
    argmax_ptr,  # [reqs, world * NB] int64
    max_ptr,  # [reqs, world * NB] fp32
    WORLD: tl.constexpr,
    NB: tl.constexpr,
    NB_P2: tl.constexpr,
    WORDS: tl.constexpr,
):
    q = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, WORLD)[:, None]
    b = tl.arange(0, NB_P2)[None, :]
    mb = b < NB
    src = gathered_ptr + (r * num_reqs + q) * WORDS
    dst = q * (WORLD * NB) + r * NB + b
    tl.store(argmax_ptr + dst, tl.load(src + b, mask=mb, other=0).to(tl.int64), mask=mb)
    vals = tl.load(src + NB + b, mask=mb, other=0).to(tl.float32, bitcast=True)
    tl.store(max_ptr + dst, vals, mask=mb)


# ----------------------------------------------------------------------------
# DSpark Markov sampling on one slice (one position per launch).


@triton.jit(do_not_specialize=["col"])
def _dp_markov_block_kernel(
    local_argmax_ptr,
    local_max_ptr,
    num_blocks,
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    col,
    logits_ptr,
    logits_stride,
    bias_ptr,
    bias_stride,
    # [num_reqs, _DRAFT_REQ_WORDS] int32 request metadata (see DPVocabShard.draft)
    meta_ptr,
    vocab_offset,
    shard_len,
    META_WORDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """gumbel_sample's per-block kernel with the fused bias and the cache
    write, on this slice; the noise is keyed by the global token id."""
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    local = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = local < shard_len
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + local, mask=mask, other=float("-inf")
    )
    # Rounded to the logits dtype, as `logits + bias` would be.
    bias = tl.load(bias_ptr + token_idx * bias_stride + local, mask=mask, other=0.0)
    logits = (logits.to(tl.float32) + bias.to(tl.float32)).to(logits.dtype)
    logits = logits.to(tl.float32)

    meta = meta_ptr + token_idx * META_WORDS
    cache_row = tl.load(meta + 1)
    temp = tl.load(meta + 2).to(tl.float32, bitcast=True)
    seed_lo = tl.load(meta + 3).to(tl.uint32, bitcast=True).to(tl.int64)
    seed_hi = tl.load(meta + 4).to(tl.int64)
    seed = (seed_hi << 32) | seed_lo
    pos = tl.load(meta + 5 + col).to(tl.int64) - 1
    tl.store(
        cache_ptr + cache_row.to(tl.int64) * cache_stride_0 + col * cache_stride_1 + local,
        logits,
        mask=mask & (cache_row >= 0),
    )
    seed = spec_step_seed(seed, col)
    value, idx = gumbel_noised_argmax(
        logits,
        local + vocab_offset,
        mask,
        seed,
        pos,
        temp,
        IS_DRAFTING=True,
        USE_FP64=False,
        APPLY_TEMPERATURE=True,
    )
    tl.store(
        local_argmax_ptr + token_idx * num_blocks + block_idx,
        vocab_offset + block_idx * BLOCK_SIZE + idx,
    )
    tl.store(local_max_ptr + token_idx * num_blocks + block_idx, value)


@triton.jit
def _dp_argmax_pack_kernel(
    local_argmax_ptr,
    local_max_ptr,
    num_blocks,
    packed_ptr,
    PADDED_NUM_BLOCKS: tl.constexpr,
):
    """This slice's winner per row, as _gumbel_combine_kernel picks it (the
    first NaN block, else the first maximum), packed as (value bits, token)."""
    row = tl.program_id(0).to(tl.int64)
    block = tl.arange(0, PADDED_NUM_BLOCKS)
    mask = block < num_blocks
    values = tl.load(
        local_max_ptr + row * num_blocks + block, mask=mask, other=float("-inf")
    )
    nan_block = tl.min(tl.where(values != values, block, PADDED_NUM_BLOCKS), axis=0)
    if nan_block < PADDED_NUM_BLOCKS:
        winner = nan_block
    else:
        maximum = tl.max(values, axis=0)
        winner = tl.min(
            tl.where(mask & (values == maximum), block, PADDED_NUM_BLOCKS), axis=0
        )
    value = tl.load(local_max_ptr + row * num_blocks + winner)
    token = tl.load(local_argmax_ptr + row * num_blocks + winner)
    tl.store(packed_ptr + row * 2, value.to(tl.int32, bitcast=True))
    tl.store(packed_ptr + row * 2 + 1, token.to(tl.int32))


@triton.jit(do_not_specialize=["num_rows"])
def _dp_argmax_combine_kernel(
    gathered_ptr,
    num_rows,
    out_ptr,
    out_stride,
    WORLD: tl.constexpr,
):
    """Across the slices in token-id order: the first NaN, else the first
    maximum."""
    row = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, WORLD)
    vbits = tl.load(gathered_ptr + (r * num_rows + row) * 2)
    tokens = tl.load(gathered_ptr + (r * num_rows + row) * 2 + 1)
    values = vbits.to(tl.float32, bitcast=True)
    nan_r = tl.min(tl.where(values != values, r, WORLD), axis=0)
    if nan_r < WORLD:
        winner = nan_r
    else:
        maximum = tl.max(values, axis=0)
        winner = tl.min(tl.where(values == maximum, r, WORLD), axis=0)
    token = tl.sum(tl.where(r == winner, tokens, 0), axis=0)
    tl.store(out_ptr + row * out_stride, token.to(tl.int64))


# ----------------------------------------------------------------------------


@dataclass
class _Layout:
    """Where every rank's rows and requests sit in the gathered step."""

    rows: tuple[int, ...]
    reqs: tuple[int, ...]

    def __post_init__(self) -> None:
        self.row_off = np.concatenate([[0], np.cumsum(self.rows)]).astype(np.int64)
        self.req_off = np.concatenate([[0], np.cumsum(self.reqs)]).astype(np.int64)
        self.num_rows = int(self.row_off[-1])
        self.num_reqs = int(self.req_off[-1])
        self.row_cap = max(1, max(self.rows))
        self.req_cap = max(1, max(self.reqs))

    def host_index(self) -> np.ndarray:
        """Gathered rank r holds its rows at [r * row_cap, + rows[r]) and its
        requests at [r * req_cap, + reqs[r]). Returns, concatenated (int64):
        each global row's gathered row, each global row's first global request
        of its rank, each global request's gathered request, and each global
        request's first global row of its rank."""
        world = len(self.rows)
        row_src = np.concatenate(
            [np.arange(r * self.row_cap, r * self.row_cap + self.rows[r]) for r in range(world)]
        )
        row_req_off = np.repeat(self.req_off[:-1], self.rows)
        req_src = np.concatenate(
            [np.arange(r * self.req_cap, r * self.req_cap + self.reqs[r]) for r in range(world)]
        )
        req_row_off = np.repeat(self.row_off[:-1], self.reqs)
        return np.concatenate([row_src, row_req_off, req_src, req_row_off]).astype(
            np.int64
        )


class DPVocabShard:
    """See the module docstring."""

    @staticmethod
    def unsupported_reason(vllm_config: Any) -> str | None:
        """Configuration checks; identical on every rank."""
        parallel = vllm_config.parallel_config
        spec = vllm_config.speculative_config
        if parallel.data_parallel_size <= 1:
            return "data parallelism is 1"
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            return "it needs tensor and pipeline parallelism 1"
        if parallel.prefill_context_parallel_size != 1:
            return "prefill context parallelism is enabled"
        if getattr(parallel, "enable_elastic_ep", False):
            return "elastic EP is enabled"
        dp = parallel.data_parallel_size
        if dp & (dp - 1):
            return f"data parallelism {dp} is not a power of 2"
        if getattr(parallel, "enable_batch_sharded_sampling", False):
            return "batch-sharded sampling is enabled"
        if spec is None or spec.method != "dspark":
            return "it needs DSpark speculative decoding"
        if spec.draft_sample_method != "probabilistic":
            return "it needs probabilistic drafting"
        if spec.enable_adaptive_verification:
            return "adaptive verification is enabled"
        if getattr(spec, "num_speculative_tokens_per_batch_size", None):
            # A rank may draft K = 0 while its peers draft: their exchanges
            # would no longer pair up.
            return "dynamic speculative decoding is enabled"
        if spec.rejection_sample_method not in ("synthetic", "block", "standard"):
            return f"rejection_sample_method {spec.rejection_sample_method!r}"
        if spec.rejection_sample_method != "synthetic" and getattr(
            spec, "fly_window_size", None
        ):
            return "FLy verification is enabled"
        return None

    def __init__(
        self,
        *,
        dp_group: Any,
        device: torch.device,
        vocab_size: int,
        hidden_size: int,
        draft_hidden_size: int,
        max_num_reqs: int,
        num_spec: int,
        target_head: torch.Tensor,
        draft_head: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
        logits_dtype: torch.dtype,
        draft_logits_dtype: torch.dtype,
        synthetic_rates: torch.Tensor | None,
        use_block_verification: bool,
        target_processor: Any = None,
        draft_processor: Any = None,
    ) -> None:
        self.group = dp_group
        self.world = dp_group.world_size
        self.rank = dp_group.rank_in_group
        self.device = device
        assert vocab_size % self.world == 0
        self.vocab_size = vocab_size
        self.shard = vocab_size // self.world
        self.offset = self.rank * self.shard
        # Row pitch of the FP32 logits: divisible by 16 like the replicated
        # pipeline's, so Triton picks the same vector loads and reductions.
        self.pitch = _round_up(self.shard, 16)
        self.hidden_size = hidden_size
        self.draft_hidden_size = draft_hidden_size
        self.max_num_reqs = max_num_reqs
        self.num_spec = num_spec
        assert num_spec + 5 <= _DRAFT_REQ_WORDS
        self.max_rows = max_num_reqs * (num_spec + 1)
        self.synthetic_rates = synthetic_rates
        self.use_block_verification = use_block_verification

        sl = slice(self.offset, self.offset + self.shard)
        self.target_head = target_head[sl]
        self.draft_head = draft_head[sl]
        self.markov_w1 = markov_w1
        self.markov_w2 = markov_w2[sl]
        self.logits_dtype = logits_dtype
        # The logits processors whose soft cap and scale follow each head's GEMM
        # (the Markov bias goes through the drafter's).
        self.target_processor = target_processor
        self.draft_processor = draft_processor

        # Draft logits of every rank's requests on this slice: request state s
        # of rank q at row q * max_num_reqs + s.
        self.draft_logits_values = torch.zeros(
            self.world * max_num_reqs,
            num_spec,
            self.shard,
            dtype=draft_logits_dtype,
            device=device,
        )
        # The full-vocabulary draft logits of this rank's requests, completed
        # for a step that is verified in full.
        self.full_draft_logits: Any = None

        # Exchange payloads (int32 words).
        h_words = hidden_size // 2
        self.verify_words_per_rank = (
            self.max_rows * (h_words + _ROW_WORDS) + max_num_reqs * _REQ_WORDS
        )
        hd_words = draft_hidden_size // 2
        self.draft_words_per_rank = (
            max_num_reqs * num_spec * hd_words + max_num_reqs * _DRAFT_REQ_WORDS
        )
        self._verify_xchg = _MulticastAllGather(
            dp_group.device_group, self.verify_words_per_rank * 4, device
        )
        self._draft_xchg = _MulticastAllGather(
            dp_group.device_group, self.draft_words_per_rank * 4, device
        )

        self.num_rounds = _SPLIT_ROUNDS
        self.fanout = _SPLIT_FANOUT
        self._num_local_rs_blocks = triton.cdiv(self.shard, _RESAMPLE_BLOCK)
        self._num_local_stats_blocks = triton.cdiv(self.shard, _STATS_BLOCK)
        self._logged_verify = False
        self._logged_draft = False
        self._logged_full = False
        # Steps verified vocab-sharded and in full, logged every _LOG_EVERY.
        self.num_sharded_steps = 0
        self.num_full_steps = 0

    @staticmethod
    def postprocess(proc: Any, logits: torch.Tensor) -> torch.Tensor:
        """`LogitsProcessor.forward`'s soft cap and scale, elementwise, so the
        slice gets each logit the full processor gives it."""
        if proc is None:
            return logits
        if proc.soft_cap is not None:
            logits = logits / proc.soft_cap
            logits = torch.tanh(logits)
            logits = logits * proc.soft_cap
        if proc.scale != 1.0:
            logits *= proc.scale
        return logits

    @staticmethod
    def head_gemm(hidden: torch.Tensor, head_slice: torch.Tensor) -> torch.Tensor:
        """`F.linear(hidden, head_slice)`, run on at least _MIN_GEMM_ROWS rows so
        that each logit equals the full head's."""
        rows = hidden.shape[0]
        if rows >= _MIN_GEMM_ROWS:
            return F.linear(hidden, head_slice)
        padded = hidden.new_zeros(_MIN_GEMM_ROWS, hidden.shape[1])
        padded[:rows].copy_(hidden)
        return F.linear(padded, head_slice)[:rows]

    def allocate(self) -> None:
        """Allocate the exchanges' mailboxes (local). The caller agrees on
        success across the DP group before `rendezvous`."""
        self._verify_xchg.allocate()
        self._draft_xchg.allocate()

    def rendezvous(self) -> bool:
        """Map the mailboxes (collective over the DP group). Returns whether
        this rank got an NVLink multicast mapping."""
        self._verify_xchg.rendezvous()
        self._draft_xchg.rendezvous()
        return bool(self._verify_xchg.multicast and self._draft_xchg.multicast)

    # ------------------------------------------------------------------
    # Exchange helpers

    def _gather(self, xchg: _MulticastAllGather, words: torch.Tensor) -> torch.Tensor:
        """All-gather an int32 [n] tensor (padded to 16 bytes) into [world, n]."""
        n = words.numel()
        n_pad = _round_up(n, 4)
        if n_pad != n:
            padded = torch.zeros(n_pad, dtype=torch.int32, device=self.device)
            padded[:n].copy_(words.reshape(-1))
            words = padded
        out = torch.empty(self.world, n_pad, dtype=torch.int32, device=self.device)
        xchg(words.reshape(-1).contiguous(), out)
        return out if n_pad == n else out[:, :n].contiguous()

    # ------------------------------------------------------------------
    # Top-p

    def _top_p(self, logits: torch.Tensor, top_p: torch.Tensor) -> None:
        """Mask `logits` [rows, shard] (row pitch `self.pitch`) in place to the
        top-p nucleus of every row, over the whole vocabulary."""
        rows = logits.shape[0]
        S, F_, R = self.world, self.fanout, self.num_rounds
        dev = self.device
        stats_local = torch.empty(rows, 4, dtype=torch.float32, device=dev)
        _topp_shard_stats_kernel[(rows,)](
            logits,
            logits.stride(0),
            stats_local,
            top_p,
            self.rank,
            VOCAB_SIZE=self.vocab_size,
            S=S,
            BLOCK=8192,
            num_warps=8,
        )
        stats = torch.empty(rows, S, 4, dtype=torch.float32, device=dev)
        g = self._gather(self._verify_xchg, stats_local.view(torch.int32).view(-1))
        stats.copy_(g.view(S, rows, 4).view(torch.float32).transpose(0, 1))
        parts = torch.empty(rows, R, S, F_ * 3, dtype=torch.float32, device=dev)
        parts_local = torch.empty(rows, F_ * 3, dtype=torch.float32, device=dev)
        for r in range(R):
            _topp_shard_step_kernel[(rows,)](
                logits,
                logits.stride(0),
                stats,
                parts,
                parts_local,
                top_p,
                r,
                self.rank,
                S=S,
                F=F_,
                NUM_ROUNDS=R,
                VOCAB_SIZE=self.vocab_size,
                BLOCK=2048,
                num_warps=8,
            )
            g = self._gather(self._verify_xchg, parts_local.view(torch.int32).view(-1))
            parts[:, r].copy_(
                g.view(S, rows, F_ * 3).view(torch.float32).transpose(0, 1)
            )
        _topp_shard_mask_kernel[(rows,)](
            logits,
            logits.stride(0),
            stats,
            parts,
            top_p,
            self.rank,
            MASK_VALUE=float("-inf"),
            S=S,
            F=F_,
            NUM_ROUNDS=R,
            VOCAB_SIZE=self.vocab_size,
            BLOCK=8192,
            num_warps=8,
        )

    # ------------------------------------------------------------------
    # Verification

    @staticmethod
    def layout(dp_sync_state: DPSyncState) -> _Layout:
        """Every rank's rows and requests in the step. A rank with no rows is
        an idle rank running a dummy batch (it syncs its dummy requests): it
        has no requests here. A real request always has at least one row."""
        assert dp_sync_state.num_logits_across_dp is not None
        assert dp_sync_state.num_reqs_across_dp is not None
        rows = tuple(dp_sync_state.num_logits_across_dp)
        reqs = tuple(
            q if n else 0 for q, n in zip(dp_sync_state.num_reqs_across_dp, rows)
        )
        return _Layout(rows, reqs)

    def verify(
        self,
        hidden_states: torch.Tensor | None,
        input_batch: InputBatch | None,
        dp_sync_state: DPSyncState,
        sampling_states: Any = None,
        prefill_len: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Verify every rank's rows on this slice. Returns this rank's
        (sampled [num_reqs, num_spec + 1], num_sampled [num_reqs]), or None for
        a rank with no requests (which passes no batch)."""
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
            gather_draft_sampled,
        )

        lay = self.layout(dp_sync_state)
        if lay.num_rows == 0:
            return None
        dev = self.device
        H = self.hidden_size
        h_words = H // 2
        C, Cq = lay.row_cap, lay.req_cap
        G, R = lay.num_rows, lay.num_reqs
        n_me = lay.rows[self.rank]
        q_me = lay.reqs[self.rank]
        assert n_me > 0 or q_me == 0
        assert (n_me > 0) == (input_batch is not None)
        self.count_step(sharded=True)

        # Host data of the step, one copy: this rank's per-row request index
        # and per-request words, then the compaction index of the gathered step.
        host = [np.zeros(0, np.int32)]
        if n_me > 0:
            assert input_batch is not None and hidden_states is not None
            assert sampling_states is not None and prefill_len is not None
            cu = input_batch.cu_num_logits_np
            assert cu[-1] == n_me and input_batch.num_reqs == q_me
            idx = input_batch.idx_mapping_np
            temp_np = sampling_states.temperature.np[idx].astype(np.float32)
            top_p_np = sampling_states.top_p.np[idx].astype(np.float32)
            seed_np = sampling_states.seeds.np[idx].astype(np.int64)
            rq = np.zeros((q_me, _REQ_WORDS), dtype=np.int32)
            rq[:, 0] = self.rank * self.max_num_reqs + idx
            rq[:, 1] = temp_np.view(np.int32)
            # Top-p cannot change a greedy row's argmax: only sampled rows use it.
            rq[:, 2] = np.where(temp_np == 0.0, np.float32(1.0), top_p_np).view(
                np.int32
            )
            rq[:, 3] = (seed_np & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
            rq[:, 4] = (seed_np >> 32).astype(np.int32)
            rq[:, 5] = np.diff(cu)
            rq[:, 6] = cu[:-1]
            req_local = np.repeat(np.arange(q_me, dtype=np.int32), np.diff(cu))
            host = [req_local, rq.reshape(-1)]
        host.append(lay.host_index().astype(np.int32))
        host_np = np.concatenate(host)
        host_dev = async_tensor_h2d(host_np, device=dev)
        own_words = n_me + q_me * _REQ_WORDS
        index_dev = host_dev[own_words:]

        n_words = C * (h_words + _ROW_WORDS) + Cq * _REQ_WORDS
        payload = torch.empty(n_words, dtype=torch.int32, device=dev)
        if n_me > 0:
            hid = payload[: C * h_words].view(torch.bfloat16).view(C, H)
            torch.index_select(
                hidden_states, 0, input_batch.logits_indices, out=hid[:n_me]
            )
            draft_sampled, pos = gather_draft_sampled(
                input_batch.input_ids,
                input_batch.positions,
                input_batch.logits_indices,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                prefill_len,
            )
            _dp_pack_meta_kernel[(n_me + q_me,)](
                payload[C * h_words :],
                payload[C * (h_words + _ROW_WORDS) :],
                host_dev,
                input_batch.expanded_local_pos,
                draft_sampled,
                pos,
                n_me,
                q_me,
                ROW_WORDS=_ROW_WORDS,
                REQ_WORDS=_REQ_WORDS,
            )
        gathered = self._gather(self._verify_xchg, payload)

        hidden_all = torch.empty(G, H, dtype=torch.bfloat16, device=dev)
        expanded_req = torch.empty(G, dtype=torch.int32, device=dev)
        local_pos = torch.empty(G, dtype=torch.int32, device=dev)
        draft_sampled_g = torch.empty(G, dtype=torch.int32, device=dev)
        pos_g = torch.empty(G, dtype=torch.int64, device=dev)
        top_p = torch.empty(G, dtype=torch.float32, device=dev)
        cache_row = torch.empty(R, dtype=torch.int32, device=dev)
        temp_g = torch.empty(R, dtype=torch.float32, device=dev)
        seed_g = torch.empty(R, dtype=torch.int64, device=dev)
        cu_g = torch.empty(R + 1, dtype=torch.int32, device=dev)
        _dp_unpack_step_kernel[(G + R,)](
            gathered,
            n_words,
            C,
            Cq,
            index_dev,
            G,
            R,
            hidden_all.view(torch.int32),
            expanded_req,
            local_pos,
            draft_sampled_g,
            pos_g,
            top_p,
            cache_row,
            temp_g,
            seed_g,
            cu_g,
            H_WORDS=h_words,
            ROW_WORDS=_ROW_WORDS,
            REQ_WORDS=_REQ_WORDS,
            BLOCK=1024,
        )
        idx_mapping_g = torch.arange(R, dtype=torch.int32, device=dev)

        # Logits of every row on this slice, FP32 with a 16-divisible pitch.
        logits_bf = self.postprocess(
            self.target_processor, self.head_gemm(hidden_all, self.target_head)
        )
        buf = torch.empty(G, self.pitch, dtype=torch.float32, device=dev)
        logits = buf[:, : self.shard]
        logits.copy_(logits_bf)
        apply_temperature(logits, expanded_req, temp_g)
        self._top_p(logits, top_p)

        sampled, num_sampled = self._rejection_sample(
            logits,
            draft_sampled_g,
            cu_g,
            pos_g,
            idx_mapping_g,
            expanded_req,
            local_pos,
            temp_g,
            seed_g,
            cache_row,
        )
        if not self._logged_verify:
            self._logged_verify = True
            logger.info(
                "DP vocab-sharded verification (%s=1): DP rank %d of %d verifies "
                "vocabulary [%d, %d) of every rank's rows; first step %d rows of "
                "%d requests.",
                ENV_VAR,
                self.rank,
                self.world,
                self.offset,
                self.offset + self.shard,
                G,
                R,
            )
        if q_me == 0:
            return None
        lo = int(lay.req_off[self.rank])
        return sampled[lo : lo + q_me], num_sampled[lo : lo + q_me]

    def _rejection_sample(
        self,
        logits: torch.Tensor,
        draft_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
        pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        cache_row: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = self.device
        G = logits.shape[0]
        R = cu_num_logits.shape[0] - 1
        W = self.world
        cache = self.draft_logits_values
        nb = self._num_local_stats_blocks
        gnb = W * nb
        padded_gnb = triton.next_power_of_2(gnb)

        # This slice's per-block statistics and the drafted tokens' logits,
        # packed for one exchange: argmax (int32, this slice's ids), target max,
        # target sumexp, draft max, draft sumexp, token logits (target, draft).
        words = _round_up(5 * nb + 2, 4)
        pack = torch.empty(G, words, dtype=torch.int32, device=dev)
        pf = pack.view(torch.float32)
        argmax_v = pack[:, 0:nb]
        tmax_v = pf[:, nb : 2 * nb]
        tsum_v = pf[:, 2 * nb : 3 * nb]
        dmax_v = pf[:, 3 * nb : 4 * nb]
        dsum_v = pf[:, 4 * nb : 5 * nb]
        tok_v = pf[:, 5 * nb : 5 * nb + 2]
        _compute_local_logits_stats_kernel[(G, nb)](
            argmax_v,
            argmax_v.stride(0),
            tmax_v,
            tmax_v.stride(0),
            tsum_v,
            tsum_v.stride(0),
            dmax_v,
            dmax_v.stride(0),
            dsum_v,
            dsum_v.stride(0),
            logits,
            logits.stride(0),
            cache,
            cache.stride(0),
            cache.stride(1),
            cache_row,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            self.shard,
            G,
            BLOCK_SIZE=_STATS_BLOCK,
            HAS_DRAFT_LOGITS=True,
        )
        tl_block = 1024
        _dp_token_logits_kernel[(triton.cdiv(G, tl_block),)](
            tok_v,
            tok_v.stride(0),
            logits,
            logits.stride(0),
            cache,
            cache.stride(0),
            cache.stride(1),
            cache_row,
            expanded_idx_mapping,
            expanded_local_pos,
            draft_sampled,
            G,
            self.offset,
            self.shard,
            BLOCK=tl_block,
        )
        g = self._gather(self._verify_xchg, pack.view(-1))
        t_argmax = torch.empty(G, gnb, dtype=torch.int64, device=dev)
        t_max = torch.empty(G, gnb, dtype=torch.float32, device=dev)
        t_sum = torch.empty(G, gnb, dtype=torch.float32, device=dev)
        d_max = torch.empty(G, gnb, dtype=torch.float32, device=dev)
        d_sum = torch.empty(G, gnb, dtype=torch.float32, device=dev)
        token_logits = torch.empty(G, 2, dtype=torch.float32, device=dev)
        _dp_unpack_stats_kernel[(G,)](
            g,
            G,
            self.shard,
            t_argmax,
            t_max,
            t_sum,
            d_max,
            d_sum,
            token_logits,
            WORLD=W,
            NB=nb,
            NB_P2=triton.next_power_of_2(nb),
            WORDS=words,
        )

        rates = self.synthetic_rates
        synthetic = rates is not None
        block = self.use_block_verification
        if block:
            cumulative_log_p = torch.empty(G, dtype=torch.float32, device=dev)
            _dp_cumulative_log_p_kernel[(R,)](
                cumulative_log_p,
                token_logits,
                t_max,
                t_max.stride(0),
                t_sum,
                t_sum.stride(0),
                draft_sampled,
                d_max,
                d_max.stride(0),
                d_sum,
                d_sum.stride(0),
                cu_num_logits,
                idx_mapping,
                temperature,
                gnb,
                PADDED_VOCAB_NUM_BLOCKS=padded_gnb,
                num_warps=1,
            )
            nb_pad = _round_up(nb, 4)
            res_local = torch.zeros(G, nb_pad, dtype=torch.float32, device=dev)
            _compute_local_residual_mass_kernel[(G, nb)](
                res_local,
                res_local.stride(0),
                cumulative_log_p,
                logits,
                logits.stride(0),
                t_max,
                t_max.stride(0),
                t_sum,
                t_sum.stride(0),
                cache,
                cache.stride(0),
                cache.stride(1),
                cache_row,
                d_max,
                d_max.stride(0),
                d_sum,
                d_sum.stride(0),
                draft_sampled,
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                self.shard,
                G,
                gnb,
                BLOCK_SIZE=_STATS_BLOCK,
                PADDED_VOCAB_NUM_BLOCKS=padded_gnb,
            )
            gr = self._gather(self._verify_xchg, res_local.view(torch.int32).view(-1))
            residual = torch.empty(G, gnb, dtype=torch.float32, device=dev)
            residual.view(G, W, nb).copy_(
                gr.view(W, G, nb_pad)[:, :, :nb].view(torch.float32).transpose(0, 1)
            )
        else:
            cumulative_log_p = torch.zeros(1, dtype=torch.float32, device=dev)
            residual = torch.zeros(1, 1, dtype=torch.float32, device=dev)

        K = self.num_spec
        sampled = torch.empty(R, K + 1, dtype=torch.int64, device=dev)
        num_sampled = torch.empty(R, dtype=torch.int32, device=dev)
        t_rej_lse = torch.empty(R, dtype=torch.float32, device=dev)
        d_rej_lse = torch.empty(R, dtype=torch.float32, device=dev)
        _dp_rejection_kernel[(R,)](
            sampled,
            sampled.stride(0),
            num_sampled,
            t_rej_lse,
            d_rej_lse,
            token_logits,
            t_argmax,
            t_argmax.stride(0),
            t_max,
            t_max.stride(0),
            t_sum,
            t_sum.stride(0),
            draft_sampled,
            d_max,
            d_max.stride(0),
            d_sum,
            d_sum.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            seed,
            pos,
            rates if synthetic else temperature,
            cumulative_log_p,
            residual,
            residual.stride(0),
            gnb,
            PADDED_VOCAB_NUM_BLOCKS=padded_gnb,
            SYNTHETIC_MODE=synthetic,
            USE_BLOCK_VERIFICATION=block,
            num_warps=1,
        )

        rnb = self._num_local_rs_blocks
        rs_words = _round_up(2 * rnb, 4)
        rs_pack = torch.empty(R, rs_words, dtype=torch.int32, device=dev)
        _dp_resample_kernel[(R, rnb)](
            rs_pack,
            rs_words,
            rnb,
            logits,
            logits.stride(0),
            t_rej_lse,
            cache,
            cache.stride(0),
            cache.stride(1),
            cache_row,
            d_rej_lse,
            num_sampled,
            cu_num_logits,
            expanded_idx_mapping,
            draft_sampled,
            temperature,
            seed,
            pos,
            cumulative_log_p,
            self.vocab_size,
            self.offset,
            self.shard,
            BLOCK_SIZE=_RESAMPLE_BLOCK,
            USE_BLOCK_VERIFICATION=block,
        )
        grs = self._gather(self._verify_xchg, rs_pack.view(-1))
        gnr = W * rnb
        g_rs_argmax = torch.empty(R, gnr, dtype=torch.int64, device=dev)
        g_rs_max = torch.empty(R, gnr, dtype=torch.float32, device=dev)
        _dp_unpack_resample_kernel[(R,)](
            grs,
            R,
            g_rs_argmax,
            g_rs_max,
            WORLD=W,
            NB=rnb,
            NB_P2=triton.next_power_of_2(rnb),
            WORDS=rs_words,
        )
        _insert_resampled_kernel[(R,)](
            sampled,
            sampled.stride(0),
            num_sampled,
            g_rs_argmax,
            g_rs_argmax.stride(0),
            g_rs_max,
            g_rs_max.stride(0),
            gnr,
            cu_num_logits,
            expanded_idx_mapping,
            temperature,
            PADDED_RESAMPLE_NUM_BLOCKS=triton.next_power_of_2(gnr),
            num_warps=1,
        )
        return sampled, num_sampled

    # ------------------------------------------------------------------
    # A step verified in full: complete this rank's draft logits.

    def count_step(self, sharded: bool) -> None:
        """Count a verification step (every rank, every step) and log the
        totals every _LOG_EVERY steps."""
        if sharded:
            self.num_sharded_steps += 1
        else:
            self.num_full_steps += 1
        total = self.num_sharded_steps + self.num_full_steps
        if total % _LOG_EVERY == 0:
            logger.info(
                "DP vocab-sharded verification (%s=1): DP rank %d: %d steps "
                "verified vocab-sharded, %d verified in full, since start.",
                ENV_VAR,
                self.rank,
                self.num_sharded_steps,
                self.num_full_steps,
            )

    def complete_draft_logits(self) -> Any:
        """All-to-all of the cached draft-logit slices, so that every rank holds
        the full-vocabulary draft logits of its own requests. Collective."""
        from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache

        self.count_step(sharded=False)
        if self.full_draft_logits is None:
            self.full_draft_logits = DraftLogitsCache(
                self.max_num_reqs,
                self.num_spec,
                self.vocab_size,
                dtype=self.draft_logits_values.dtype,
                fill=0.0,
                device=self.device,
            )
        W, Q, K, Vs = self.world, self.max_num_reqs, self.num_spec, self.shard
        send = self.draft_logits_values.view(W, Q * K * Vs)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group.device_group)
        # recv[q] = rank q's slice of this rank's requests.
        self.full_draft_logits.values.view(Q, K, W, Vs).copy_(
            recv.view(W, Q, K, Vs).permute(1, 2, 0, 3)
        )
        if not self._logged_full:
            self._logged_full = True
            logger.info(
                "DP vocab-sharded verification: a step with a request that needs "
                "the full vocabulary is verified in full on every DP rank."
            )
        return self.full_draft_logits

    # ------------------------------------------------------------------
    # Drafting

    def draft(self, spec: Any, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """The DSpark Markov chain for every rank's `num_reqs` (equal on every
        rank) requests on this slice. Writes this rank's drafts into
        `spec.draft_tokens[:num_reqs]` and every rank's draft logits into
        `self.draft_logits_values`."""
        dev = self.device
        W, K, Hd = self.world, self.num_spec, self.draft_hidden_size
        NR = num_reqs
        num_sample = NR * K
        hd_words = Hd // 2
        payload = torch.empty(
            NR * K * hd_words + NR * _DRAFT_REQ_WORDS, dtype=torch.int32, device=dev
        )
        sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
        payload[: NR * K * hd_words].view(NR * K, hd_words).copy_(
            sample_hidden.contiguous().view(torch.int32).view(NR * K, hd_words)
        )
        meta = payload[NR * K * hd_words :].view(NR, _DRAFT_REQ_WORDS)
        idx_map = spec.sample_idx_mapping[:num_sample].view(NR, K)[:, 0].to(torch.int64)
        valid = idx_map >= 0
        safe = idx_map.clamp(min=0)
        anchor = spec.input_buffers.input_ids[spec._anchor_idx[:NR]]
        meta[:, 0] = anchor.to(torch.int32)
        meta[:, 1] = torch.where(
            valid, self.rank * self.max_num_reqs + idx_map, -1
        ).to(torch.int32)
        temp = torch.where(valid, spec.temperature[safe], 0.0).to(torch.float32)
        meta[:, 2] = temp.view(torch.int32)
        seed = torch.where(valid, spec.seeds[safe], 0)
        meta[:, 3] = (seed & 0xFFFFFFFF).to(torch.int32)
        meta[:, 4] = (seed >> 32).to(torch.int32)
        meta[:, 5 : 5 + K] = spec.sample_pos[:num_sample].view(NR, K).to(torch.int32)
        if 5 + K < _DRAFT_REQ_WORDS:
            meta[:, 5 + K :] = 0

        g = self._gather(self._draft_xchg, payload)
        hd_all = (
            g[:, : NR * K * hd_words]
            .reshape(W * NR * K, hd_words)
            .contiguous()
            .view(torch.bfloat16)
            .view(W * NR * K, Hd)
        )
        meta_all = g[:, NR * K * hd_words :].reshape(W * NR, _DRAFT_REQ_WORDS)
        meta_all = meta_all.contiguous()
        base = self.postprocess(
            self.draft_processor, self.head_gemm(hd_all, self.draft_head)
        ).view(W * NR, K, self.shard)

        rows = W * NR
        nb = self._num_local_rs_blocks
        la = torch.empty(rows, nb, dtype=torch.int64, device=dev)
        lm = torch.empty(rows, nb, dtype=torch.float32, device=dev)
        packed = torch.empty(rows, 2, dtype=torch.int32, device=dev)
        tokens = torch.empty(rows, K, dtype=torch.int64, device=dev)
        prev = meta_all[:, 0].to(torch.int64)
        cache = self.draft_logits_values
        for i in range(K):
            bias = self.postprocess(
                self.draft_processor,
                self.head_gemm(F.embedding(prev, self.markov_w1), self.markov_w2),
            )
            logits_i = base[:, i]
            _dp_markov_block_kernel[(rows, nb)](
                la,
                lm,
                nb,
                cache,
                cache.stride(0),
                cache.stride(1),
                i,
                logits_i,
                logits_i.stride(0),
                bias,
                bias.stride(0),
                meta_all,
                self.offset,
                self.shard,
                META_WORDS=_DRAFT_REQ_WORDS,
                BLOCK_SIZE=_RESAMPLE_BLOCK,
            )
            _dp_argmax_pack_kernel[(rows,)](
                la, lm, nb, packed, PADDED_NUM_BLOCKS=triton.next_power_of_2(nb)
            )
            gathered = self._gather(self._draft_xchg, packed.reshape(-1))
            _dp_argmax_combine_kernel[(rows,)](
                gathered, rows, tokens[:, i], tokens.stride(0), WORLD=W
            )
            prev = tokens[:, i]
        spec.draft_tokens[:NR] = tokens[self.rank * NR : (self.rank + 1) * NR]
        if not self._logged_draft:
            self._logged_draft = True
            logger.info(
                "DP vocab-sharded drafting (%s=1): DP rank %d of %d runs the DSpark "
                "Markov chain on vocabulary [%d, %d) for every rank's requests.",
                ENV_VAR,
                self.rank,
                self.world,
                self.offset,
                self.offset + self.shard,
            )
