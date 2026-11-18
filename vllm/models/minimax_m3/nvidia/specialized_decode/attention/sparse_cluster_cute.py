# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``sparse_cute.SparseAttention`` with each row's eight CTAs as one thread
block cluster (``VLLM_MINIMAX_M3_LOW_BATCH_ATTN_CLUSTER=1``).

The per-CTA kernel spends most of a row's time outside its attention: every
CTA ranks the row's whole page list, and the row's partials meet in global
memory (a release fence and a counter per CTA, then the last CTA loads them
all and finishes the row alone). Here the row's CTAs work as a cluster:

* Ranking: CTA ``c`` takes the row's pages ``c, c + 8, ..`` (at most four a
  thread, one up to 2048 pages) and finds its own top 15 by (score, page):
  each thread sorts its pages, each warp ranks its threads' best by counting
  over shuffles, and a binary search across the eight warp lists (their
  steps interleaved) ranks those. With more than one page a thread, the
  CTA's 15 best threads hold its top 15 among their at most 60 pages, which
  are ranked against each other. Each CTA then stores its list into every
  CTA's shared memory (``st.async``, completing on the receiver's
  mbarrier), and each ranks the 120 candidates the same way. The row's top
  15 are among them, so the selection is the per-CTA ranking's, page for
  page and in the same order.
* Merge: each CTA merges its warps exactly as before, then stores each
  head's partial (the maximum, the sum and the unnormalized output) into the
  shared memory of the CTA that finishes the head: CTA ``c`` takes head
  ``c`` (8 heads a row) or heads ``2c``, ``2c + 1`` (16). Once its mbarrier
  has the eight partials, that CTA merges them in the per-CTA kernel's last
  CTA's order and arithmetic and quantizes them to MXFP8. Every CTA arrives
  at a last cluster barrier right after its stores and waits for it before
  it exits.

The pages' attention (``SparseAttention.attend``) and the in-CTA merge are
the base class's, so o_proj's input is bitwise the per-CTA kernel's. Grids of
more than eight rows still trigger o_proj and prefetch its weights as there.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_front_cute import (  # noqa: E501
    block_scale,
    inverse_scale,
)

from . import HEAD_DIM, HEADS
from .indexer_cute import RECORD_BYTES, SCORE_STRIDE
from .ptx import (
    e4m3_byte,
    e4m3x4_f16,
    ld_f32,
    ld_f32_s32,
    ld_s32,
    ld_s32x2,
    ld_shared_f32x4,
    ld_u32x4,
    ranks_before,
    st_peer_f32,
    st_peer_s32,
    st_shared_f32,
    st_shared_f32x4,
    st_u8,
    st_u32,
)
from .sparse_cute import (
    HISTORY,
    RANKERS,
    STAGE_ROW,
    WARPS,
    SparseAttention,
    swizzled_scale_offset,
)

CLUSTER = 8  # CTAs a row: one thread block cluster
# Bytes a CTA receives: the cluster's lists (value, page, physical page),
# and per head it finishes the eight partials (128 values, maximum, sum).
RANK_BYTES = CLUSTER * HISTORY * 3 * 4
PARTIAL_WORDS = HEAD_DIM + 2
SLOTS = 4  # pages a ranking thread holds: CLUSTER x RANKERS x SLOTS = 8192
LIST = 16  # a sorted list's stride (15 entries; the binary search reads 0-14)
# Sorting network of a thread's four pairs.
SORT4 = ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2))


class ClusterSparseAttention(SparseAttention):
    """The module docstring's kernel, for ``heads`` query heads a row (16,
    or 8 in the first half of the 16-head tile); the launch arguments are
    ``SparseAttention``'s (it reads no ``partial`` or ``counters``)."""

    def __init__(self, heads: int = HEADS) -> None:
        super().__init__(CLUSTER, heads)
        assert SCORE_STRIDE == CLUSTER * RANKERS * SLOTS

    @cute.jit
    def __call__(
        self,
        q: Int64,
        kv: Int64,
        page_stride: Int64,
        records: Int64,
        scores: Int64,
        k_scale: Int64,
        v_scale: Int64,
        act: Int64,
        act_scale: Int64,
        partial: Int64,
        counters: Int64,
        logit_scale: Float32,
        weight: Int64,
        weight_bytes: Int32,
        weight_scale: Int64,
        scale_bytes: Int32,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q,
            kv,
            page_stride,
            records,
            scores,
            k_scale,
            v_scale,
            act,
            act_scale,
            logit_scale,
            weight,
            weight_bytes,
            weight_scale,
            scale_bytes,
        ).launch(
            grid=(CLUSTER, rows, 1),
            block=(WARPS * 32, 1, 1),
            cluster=(CLUSTER, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        q: Int64,
        kv: Int64,
        page_stride: Int64,
        records: Int64,
        scores: Int64,
        k_scale: Int64,
        v_scale: Int64,
        act: Int64,
        act_scale: Int64,
        logit_scale: Float32,
        weight: Int64,
        weight_bytes: Int32,
        weight_scale: Int64,
        scale_bytes: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        # The grid's x is the cluster's: c is the CTA's rank in its cluster.
        c, r, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        g = lane // 4
        t = lane % 4
        neg_inf = Float32(-math.inf)

        smem = cutlass.utils.SmemAllocator()
        sel = smem.allocate_tensor(Int32, cute.make_layout(16), 16)
        part_ml = smem.allocate_tensor(Float32, cute.make_layout((2, WARPS, HEADS)), 16)
        part_o = smem.allocate_tensor(
            Float32, cute.make_layout((WARPS, 16, 32, 4), stride=(2048, 128, 4, 1)), 16
        )
        # The partials of the heads this CTA finishes, from every CTA:
        # [source CTA, local head, word], words 0-127 the output, 128 the
        # maximum, 129 the sum.
        received = smem.allocate_tensor(
            Float32, cute.make_layout(CLUSTER * 2 * PARTIAL_WORDS), 16
        )
        stage = smem.allocate_tensor(Float32, cute.make_layout(2 * STAGE_ROW), 16)
        # Completions of the cluster's lists and of the partials.
        bars = smem.allocate_array(cutlass.Int64, 2, byte_alignment=8)
        rank_bar = bars
        part_bar = bars + 1
        # Ranking: the warps' sorted lists [rank, warp], each thread's rank
        # among the CTA's best, the best threads' pages [slot, rank], this
        # CTA's top 15 (its peers read it) and the cluster's lists [rank, CTA].
        lists = cute.make_layout((LIST, WARPS), stride=(1, LIST))
        wl_v = smem.allocate_tensor(Float32, lists, 16)
        wl_j = smem.allocate_tensor(Int32, lists, 16)
        wl_p = smem.allocate_tensor(Int32, lists, 16)
        wl_o = smem.allocate_tensor(Int32, lists, 16)
        owner_rank = smem.allocate_tensor(Int32, cute.make_layout(RANKERS), 16)
        slots = cute.make_layout((SLOTS, LIST), stride=(1, SLOTS))
        cand_v = smem.allocate_tensor(Float32, slots, 16)
        cand_j = smem.allocate_tensor(Int32, slots, 16)
        cand_p = smem.allocate_tensor(Int32, slots, 16)
        out_v = smem.allocate_tensor(Float32, cute.make_layout(LIST), 16)
        out_j = smem.allocate_tensor(Int32, cute.make_layout(LIST), 16)
        out_p = smem.allocate_tensor(Int32, cute.make_layout(LIST), 16)
        cluster_lists = cute.make_layout((LIST, CLUSTER), stride=(1, LIST))
        mg_v = smem.allocate_tensor(Float32, cluster_lists, 16)
        mg_j = smem.allocate_tensor(Int32, cluster_lists, 16)
        mg_p = smem.allocate_tensor(Int32, cluster_lists, 16)
        heads_here = 2 if self.heads == HEADS else 1
        if tid == 0:
            cute.arch.mbarrier_init(rank_bar, 1)
            cute.arch.mbarrier_init(part_bar, 1)
            cute.arch.mbarrier_init_fence()
            # The partials: a fixed size; the lists' arrival waits for the
            # count.
            cute.arch.mbarrier_arrive_and_expect_tx(
                part_bar, CLUSTER * heads_here * PARTIAL_WORDS * 4
            )
        cute.arch.sync_threads()
        # The peers store into these barriers only after seeing them
        # initialized (the wait before the first store).
        cute.arch.cluster_arrive_relaxed()

        _, rows, _ = cute.arch.grid_dim()
        if rows > 8:
            cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()

        # The row's record, its query and this CTA's (score, page) pairs, all
        # in flight together (the pairs past the row are masked once its
        # count is known; the row's score buffer holds 8192 of them).
        record = records + Int64(r) * RECORD_BYTES
        count, local_pos = ld_s32x2(record + 4 * HEADS)
        slot_phys = ld_s32(record + Int64(lane % 16) * 4)
        row_scores = scores + Int64(r) * SCORE_STRIDE * 8
        pv = cute.make_rmem_tensor((SLOTS,), Float32)
        pj = cute.make_rmem_tensor((SLOTS,), Int32)
        pp = cute.make_rmem_tensor((SLOTS,), Int32)
        for i in cutlass.range_constexpr(SLOTS):
            j = c + CLUSTER * (tid + RANKERS * i)
            pj[i] = j
            pair_v, pair_p = ld_f32_s32(row_scores + Int64(j) * 8)
            pv[i] = pair_v
            pp[i] = pair_p
        scale = logit_scale * ld_f32(k_scale)
        v_global = ld_f32(v_scale)
        q_words = cute.make_rmem_tensor((2, 2, 4), Uint32)
        for hh in cutlass.range_constexpr(2):
            for half in cutlass.range_constexpr(2):
                if cutlass.const_expr(8 * hh < self.heads):
                    words = ld_u32x4(
                        q
                        + Int64(r) * self.q_size
                        + Int64((g + 8 * hh) * HEAD_DIM + 32 * t + 16 * half)
                    )
                    for i in cutlass.range_constexpr(4):
                        q_words[hh, half, i] = words[i]
                else:
                    for i in cutlass.range_constexpr(4):
                        q_words[hh, half, i] = Uint32(0)
        # o_proj's launch, as the per-CTA kernel's.
        after_pages = Int32(0)
        if (rows <= 8) & (count <= RANKERS):
            after_pages = Int32(1)
        if (rows <= 8) & (after_pages == 0):
            cute.arch.griddepcontrol_launch_dependents()

        if tid < 16:
            sel[tid] = slot_phys
        cute.arch.cluster_wait()
        # Every CTA of the cluster reads the same count: they all rank, or
        # none does.
        if count > HISTORY:
            for i in cutlass.range_constexpr(SLOTS):
                if pj[i] >= count:
                    pv[i] = neg_inf
            self.rank(
                count,
                pv,
                pj,
                pp,
                c,
                tid,
                (wl_v, wl_j, wl_p, wl_o),
                owner_rank,
                (cand_v, cand_j, cand_p),
                (out_v, out_j, out_p),
                (mg_v, mg_j, mg_p),
                rank_bar,
                sel,
            )
        cute.arch.sync_threads()
        if after_pages == 1:
            cute.arch.griddepcontrol_launch_dependents()

        # From here to the in-CTA merge: SparseAttention.kernel's code.
        qw = cute.make_rmem_tensor((2, 16), Uint32)
        for hh in cutlass.range_constexpr(2):
            for half in cutlass.range_constexpr(2):
                for i in cutlass.range_constexpr(4):
                    pair = e4m3x4_f16(q_words[hh, half, i])
                    qw[hh, 8 * half + 2 * i] = pair[0]
                    qw[hh, 8 * half + 2 * i + 1] = pair[1]
        stat = cute.make_rmem_tensor((4,), Float32)
        stat[0] = neg_inf
        stat[1] = neg_inf
        stat[2] = Float32(0.0)
        stat[3] = Float32(0.0)
        o = cute.make_rmem_tensor((64,), Float32)
        o.fill(0.0)
        for i in cutlass.range_constexpr(self.chunks):
            chunk = warp + WARPS * i
            slot = c * self.slots + chunk // 4
            phys = sel[slot]
            limit = Int32(128 - 1)
            if slot == HISTORY:
                limit = local_pos
            if phys >= 0:
                self.attend(
                    kv + Int64(phys) * page_stride,
                    chunk % 4,
                    limit,
                    qw,
                    stat,
                    o,
                    scale,
                    g,
                    t,
                )
        for k in cutlass.range_constexpr(2):
            total = stat[2 + k]
            total = total + cute.arch.shuffle_sync_bfly(total, 1)
            total = total + cute.arch.shuffle_sync_bfly(total, 2)
            stat[2 + k] = total
        if (rows > 8) & (tid == 0):
            part = r * self.ctas + c
            parts = rows * self.ctas
            self.prefetch_share(weight, weight_bytes, part, parts)
            self.prefetch_share(weight_scale, scale_bytes, part, parts)

        # The CTA's warps merged, as the per-CTA kernel merges them, into this
        # CTA's shared memory instead of global memory.
        if t == 0:
            for k in cutlass.range_constexpr(2):
                part_ml[0, warp, g + 8 * k] = stat[k]
                part_ml[1, warp, g + 8 * k] = stat[2 + k]
        part_at = part_o.iterator.toint() + lane * 16
        for n in cutlass.range_constexpr(16):
            st_shared_f32x4(
                part_at + (warp * 16 + n) * 512,
                o[4 * n],
                o[4 * n + 1],
                o[4 * n + 2],
                o[4 * n + 3],
            )
        cute.arch.sync_threads()
        factor = cute.make_rmem_tensor((2, WARPS), Float32)
        for k in cutlass.range_constexpr(2):
            head = g + 8 * k
            top = part_ml[0, 0, head]
            for w in cutlass.range_constexpr(1, WARPS):
                top = cute.arch.fmax(top, part_ml[0, w, head])
            base = top
            if top == neg_inf:
                base = Float32(0.0)
            denom = Float32(0.0)
            for w in cutlass.range_constexpr(WARPS):
                factor[k, w] = cute.math.exp2(part_ml[0, w, head] - base, fastmath=True)
                denom = denom + factor[k, w] * part_ml[1, w, head]
            if cutlass.const_expr(8 * k < self.heads):
                if (warp == 0) & (t == 0):
                    at = self.partial_word(received, c, head, HEAD_DIM)
                    st_peer_f32(at, part_bar.toint(), self.owner(head), top)
                    st_peer_f32(at + 4, part_bar.toint(), self.owner(head), denom)
        for nn in cutlass.range_constexpr(2):
            n = 2 * warp + nn
            parts = [
                ld_shared_f32x4(part_at + (w * 16 + n) * 512) for w in range(WARPS)
            ]
            acc = [Float32(0.0)] * 4
            for e in cutlass.range_constexpr(4):
                for w in cutlass.range_constexpr(WARPS):
                    acc[e] = acc[e] + factor[e // 2, w] * parts[w][e]
            # Value e: head g + 8 (e // 2), dim 32 t + 16 (e % 2) + n.
            for e in cutlass.range_constexpr(4):
                if cutlass.const_expr(8 * (e // 2) < self.heads):
                    head = g + 8 * (e // 2)
                    dim = 32 * t + 16 * (e % 2) + n
                    st_peer_f32(
                        self.partial_word(received, c, head, dim),
                        part_bar.toint(),
                        self.owner(head),
                        acc[e],
                    )
        # This CTA's stores are issued: its peers may exit once they have
        # theirs.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.mbarrier_wait(part_bar, 0)
        self.finish_heads(
            received,
            stage.iterator.toint(),
            c,
            r,
            tid,
            act,
            act_scale,
            v_global,
        )
        cute.arch.cluster_wait()

    @cute.jit
    def owner(self, head: Int32) -> Int32:
        """The CTA that finishes ``head``."""
        result = head
        if cutlass.const_expr(self.heads == HEADS):
            result = head // 2
        return result

    @cute.jit
    def partial_word(
        self, received: cute.Tensor, source: Int32, head: Int32, word: Int32
    ) -> Int32:
        """Shared address of ``word`` of ``source``'s partial of ``head`` in
        the receive table of ``head``'s CTA (the same offset in every CTA)."""
        local = Int32(0)
        if cutlass.const_expr(self.heads == HEADS):
            local = head % 2
        return (
            received.iterator.toint()
            + ((source * 2 + local) * PARTIAL_WORDS + word) * 4
        )

    @cute.jit
    def rank(
        self,
        count: Int32,
        pv: cute.Tensor,
        pj: cute.Tensor,
        pp: cute.Tensor,
        c: Int32,
        tid: Int32,
        warp_lists: tuple,
        owner_rank: cute.Tensor,
        candidates: tuple,
        outs: tuple,
        merged: tuple,
        rank_bar: cute.Pointer,
        sel: cute.Tensor,
    ):
        """``sel[k]``, k < 15: the physical page of the row's rank-k pair
        (module docstring); every thread of every CTA of the cluster calls
        it. ``pv``, ``pj``, ``pp``: this thread's pairs (score, page, physical
        page), those past the row at -inf."""
        wl_v, wl_j, wl_p, wl_o = warp_lists
        cand_v, cand_j, cand_p = candidates
        out_v, out_j, out_p = outs
        mg_v, mg_j, mg_p = merged
        lane = tid % 32
        warp = tid // 32
        multi = count > CLUSTER * RANKERS
        if multi:
            # This thread's pairs best first (a five-comparator network).
            for a, b in SORT4:
                if ranks_before(pv[a], pj[a], pv[b], pj[b]) == 1:
                    tv = pv[a]
                    tj = pj[a]
                    tp = pp[a]
                    pv[a] = pv[b]
                    pj[a] = pj[b]
                    pp[a] = pp[b]
                    pv[b] = tv
                    pj[b] = tj
                    pp[b] = tp
        # Each thread's best pair ranked in its warp by counting the lanes'
        # that rank before it (the shuffles are independent); a warp's best
        # 15 in rank order.
        bv = pv[0]
        bj = pj[0]
        in_warp = Int32(0)
        for k in cutlass.range_constexpr(32):
            ov = cute.arch.shuffle_sync(bv, k)
            oj = cute.arch.shuffle_sync(bj, k)
            in_warp = in_warp + ranks_before(bv, bj, ov, oj)
        if in_warp < HISTORY:
            wl_v[in_warp, warp] = bv
            wl_j[in_warp, warp] = bj
            wl_p[in_warp, warp] = pp[0]
            wl_o[in_warp, warp] = tid
        owner_rank[tid] = Int32(-1)
        cute.arch.sync_threads()
        # The CTA's 15 best threads: rank of each warp's first 15 among all.
        if lane < HISTORY:
            lv = wl_v[lane, warp]
            lj = wl_j[lane, warp]
            best_place = self.ranked_before(lv, lj, wl_v, wl_j, WARPS)
            if best_place < HISTORY:
                if multi:
                    owner_rank[wl_o[lane, warp]] = best_place
                else:
                    out_v[best_place] = lv
                    out_j[best_place] = lj
                    out_p[best_place] = wl_p[lane, warp]
        cute.arch.sync_threads()
        if multi:
            # Their at most 60 pairs hold the CTA's top 15.
            mine = owner_rank[tid]
            if mine >= 0:
                for i in cutlass.range_constexpr(SLOTS):
                    cand_v[i, mine] = pv[i]
                    cand_j[i, mine] = pj[i]
                    cand_p[i, mine] = pp[i]
            cute.arch.sync_threads()
            if tid < SLOTS * HISTORY:
                cs = tid // SLOTS
                ck = tid % SLOTS
                cv = cand_v[ck, cs]
                cj = cand_j[ck, cs]
                cand_place = Int32(0)
                for s2 in cutlass.range_constexpr(HISTORY):
                    for i in cutlass.range_constexpr(SLOTS):
                        cand_place = cand_place + ranks_before(
                            cv, cj, cand_v[i, s2], cand_j[i, s2]
                        )
                if cand_place < HISTORY:
                    out_v[cand_place] = cv
                    out_j[cand_place] = cj
                    out_p[cand_place] = cand_p[ck, cs]
            cute.arch.sync_threads()
        # This CTA's list into every CTA's table (its column ``c``), then the
        # cluster's eight lists here.
        if tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(rank_bar, RANK_BYTES)
        if tid < CLUSTER * HISTORY:
            peer = tid // HISTORY
            k = tid % HISTORY
            st_peer_f32(
                mg_v.iterator.toint() + (k + LIST * c) * 4,
                rank_bar.toint(),
                peer,
                out_v[k],
            )
            st_peer_s32(
                mg_j.iterator.toint() + (k + LIST * c) * 4,
                rank_bar.toint(),
                peer,
                out_j[k],
            )
            st_peer_s32(
                mg_p.iterator.toint() + (k + LIST * c) * 4,
                rank_bar.toint(),
                peer,
                out_p[k],
            )
        cute.arch.mbarrier_wait(rank_bar, 0)
        if tid < CLUSTER * HISTORY:
            mpeer = tid // HISTORY
            mk = tid % HISTORY
            mv = mg_v[mk, mpeer]
            mj = mg_j[mk, mpeer]
            row_place = self.ranked_before(mv, mj, mg_v, mg_j, CLUSTER)
            if row_place < HISTORY:
                sel[row_place] = mg_p[mk, mpeer]

    @cute.jit
    def ranked_before(
        self,
        v: Float32,
        j: Int32,
        list_v: cute.Tensor,
        list_j: cute.Tensor,
        lists: cutlass.Constexpr[int],
    ) -> Int32:
        """The pairs of ``lists`` sorted lists [rank, list] (15 entries
        each) that rank before (``v``, ``j``): a prefix of each, found in four
        steps, the lists' interleaved."""
        at = [Int32(0)] * lists
        # Step-major: each step's loads of the lists are independent.
        for step in (8, 4, 2, 1):
            for w in cutlass.range_constexpr(lists):
                other_v = list_v[at[w] + step - 1, w]
                other_j = list_j[at[w] + step - 1, w]
                at[w] = at[w] + step * ranks_before(v, j, other_v, other_j)
        place = Int32(0)
        for w in cutlass.range_constexpr(lists):
            place = place + at[w]
        return place

    @cute.jit
    def finish_heads(
        self,
        received: cute.Tensor,
        stage: Int32,
        c: Int32,
        r: Int32,
        tid: Int32,
        act: Int64,
        act_scale: Int64,
        v_global: Float32,
    ):
        """This CTA's heads of row ``r`` from the cluster's partials in
        ``received``: the per-CTA kernel's ``finish``, per (head, dim), in its
        order and arithmetic. 8 heads: head ``c``, threads 0-127; 16: heads
        ``2c + tid // 128``."""
        local = tid // HEAD_DIM  # the CTA's first or second head
        dim = tid % HEAD_DIM
        if cutlass.const_expr(self.heads == HEADS):
            self.finish_value(received, stage, local, dim, v_global)
        else:
            if local == 0:
                self.finish_value(received, stage, local, dim, v_global)
        cute.arch.sync_threads()
        # MXFP8 blocks of 32 values, four threads a block (eight values each),
        # by warp 0: lanes 0-15 the first head, 16-31 the second (16 heads).
        if tid < 32:
            qlocal = tid // 16
            i = tid % 16
            out = [
                *ld_shared_f32x4(stage + (qlocal * STAGE_ROW + 8 * i) * 4),
                *ld_shared_f32x4(stage + (qlocal * STAGE_ROW + 8 * i + 4) * 4),
            ]
            amax = Float32(0.0)
            for x in out:
                amax = cute.arch.fmax(amax, cute.arch.fmax(x, -x))
            for d in cutlass.range_constexpr(2):
                amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, 1 << d))
            code = block_scale(amax)
            inverse = inverse_scale(code)
            words = []
            for half in cutlass.range_constexpr(2):
                word = Uint32(0)
                for k in cutlass.range_constexpr(4):
                    word = word | (e4m3_byte(out[4 * half + k] * inverse) << (8 * k))
                words.append(word)
            if cutlass.const_expr(self.heads == HEADS):
                self.store_block(
                    act, act_scale, r, 2 * c + qlocal, i, words[0], words[1], code
                )
            else:
                if qlocal == 0:
                    self.store_block(act, act_scale, r, c, i, words[0], words[1], code)

    @cute.jit
    def finish_value(
        self,
        received: cute.Tensor,
        stage: Int32,
        local: Int32,
        dim: Int32,
        v_global: Float32,
    ):
        """Output ``dim`` of this CTA's head ``local`` of the row, rounded to
        BF16, into the stage's row ``local``."""
        neg_inf = Float32(-math.inf)
        peaks = cute.make_rmem_tensor((CLUSTER,), Float32)
        sums = cute.make_rmem_tensor((CLUSTER,), Float32)
        vals = cute.make_rmem_tensor((CLUSTER,), Float32)
        for p in cutlass.range_constexpr(CLUSTER):
            at = (p * 2 + local) * PARTIAL_WORDS
            peaks[p] = received[at + HEAD_DIM]
            sums[p] = received[at + HEAD_DIM + 1]
            vals[p] = received[at + dim]
        top = peaks[0]
        for p in cutlass.range_constexpr(1, CLUSTER):
            top = cute.arch.fmax(top, peaks[p])
        base = top
        if top == neg_inf:
            base = Float32(0.0)
        denom = Float32(0.0)
        acc = Float32(0.0)
        for p in cutlass.range_constexpr(CLUSTER):
            f = cute.math.exp2(peaks[p] - base, fastmath=True)
            denom = denom + f * sums[p]
            acc = acc + f * vals[p]
        norm = Float32(0.0)
        if denom > Float32(0.0):
            norm = v_global / denom
        st_shared_f32(
            stage + (local * STAGE_ROW + dim) * 4,
            Float32(cutlass.BFloat16(acc * norm)),
        )

    @cute.jit
    def store_block(
        self,
        act: Int64,
        act_scale: Int64,
        r: Int32,
        head: Int32,
        i: Int32,
        low: Uint32,
        high: Uint32,
        code: Uint32,
    ):
        """Eight E4M3 values ``8i ..`` of ``head`` in row ``r`` of o_proj's
        input, and from the block's first thread its scale."""
        row = act + Int64(r) * self.q_size + Int64(head * HEAD_DIM + 8 * i)
        st_u32(row, low)
        st_u32(row + 4, high)
        if i % 4 == 0:
            block = head * (HEAD_DIM // 32) + i // 4
            st_u8(act_scale + Int64(swizzled_scale_offset(r, block)), code)
