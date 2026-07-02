"""ShardQ (sequence-CP sparse attention) implementation for the DeepseekV4
FlashMLA backend, split out of flashmla.py as a mixin (see DeepseekV4FlashMLAAttention).
"""
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, cast

import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import get_dcp_group, get_pcp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.a2a_transpose import (
    restore_shardq_to_heads,
    transpose_heads_to_shardq,
)
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    fused_indexer_q_rope_quant,
    fused_q_kv_rmsnorm,
)
from vllm.models.deepseek_v4.sequence_parallel import (
    ShardQTokenSplit,
    build_all_rank_shardq_token_splits,
    build_shardq_token_split,
    count_shardq_tokens_before,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
)
from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
    split_indexer_prefill_chunks,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    # The mixin only ever runs as part of DeepseekV4FlashMLAAttention; treat the
    # host class as the base so type-checking resolves the self.<attr> it relies
    # on. At runtime it stays a plain mixin (object) — no extra base, so no
    # import cycle and no subclass-registration side effects.
    _ShardQMixinBase = DeepseekV4Attention
else:
    _ShardQMixinBase = object

# Sentinel cached on the forward context when a shardq prefill plan has no
# prefill segments, so the empty result is memoized instead of recomputed.

_SHARDQ_NO_PREFILL = object()


class ShardQMixin(_ShardQMixinBase):
    """ShardQ methods mixed into DeepseekV4FlashMLAAttention."""

    def _forward_shardq(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm_parallel_execute(hidden_states)
        )
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )
        o = self._attention_impl_shardq(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
        )
        return self._o_proj(o, positions)

    def _reserve_shardq_profile_memory(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        tp_group = get_tp_group()
        world_size = tp_group.world_size
        if world_size == 1:
            return ()

        num_tokens = hidden_states.shape[0]
        max_local_tokens = (num_tokens + world_size - 1) // world_size
        full_padded_heads = self.get_padded_num_q_heads(self.n_heads)
        shapes = (
            # all-to-all input: full-token/local-head rows, packed by dest rank
            (num_tokens, self.n_local_heads, self.head_dim),
            # all-to-all output before reshaping to local-token/full-head rows
            (max_local_tokens * world_size, self.n_local_heads, self.head_dim),
            # shardq q and shardq attention output
            (max_local_tokens, full_padded_heads, self.head_dim),
            (max_local_tokens, full_padded_heads, self.head_dim),
            # restore all-to-all send and receive buffers
            (max_local_tokens * world_size, self.n_local_heads, self.head_dim),
            (num_tokens, self.n_local_heads, self.head_dim),
            # final scatter buffer before the existing output projection
            (num_tokens, self.n_local_heads, self.head_dim),
        )
        return tuple(
            torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device)
            for shape in shapes
        )

    def _validate_shardq_group(self) -> None:
        dcp_group = get_dcp_group()
        pcp_group = get_pcp_group()
        if dcp_group.world_size != 1 or pcp_group.world_size != 1:
            raise ValueError(
                "VLLM_DSV4_ENABLE_TP_SHARDQ currently requires replicated "
                "attention KV cache, so DCP and PCP must both be 1. Got "
                f"dcp={dcp_group.world_size}, pcp={pcp_group.world_size}."
            )

    @eager_break_during_capture
    def _attention_impl_shardq(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_shardq_group()
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        assert isinstance(attn_metadata, dict)
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None
        assert swa_metadata.query_start_loc is not None
        assert swa_metadata.seq_lens is not None

        tp_group = get_tp_group()
        num_tokens = hidden_states.shape[0]
        # Under SP the runner pads the flattened token count up to a tp_size
        # multiple; the shardq fold must run on the REAL (unpadded) query tokens
        # (query_start_loc[-1]). Slice the per-token inputs to the real count,
        # fold/attend on them, and re-pad the output back to num_tokens below.
        # Use the host-side token counts the metadata builder already computed
        # (python ints, always set). An .item() on the device query_start_loc
        # would force a per-layer GPU drain (exposed CPU<->GPU sync).
        real_num_tokens = (
            swa_metadata.num_prefill_tokens + swa_metadata.num_decode_tokens
        )
        sp_pad = num_tokens - real_num_tokens
        if sp_pad > 0:
            hidden_states = hidden_states[:real_num_tokens]
            qr = qr[:real_num_tokens]
            kv = kv[:real_num_tokens]
            kv_score = kv_score[:real_num_tokens]
            indexer_kv_score = indexer_kv_score[:real_num_tokens]
            indexer_weights = indexer_weights[:real_num_tokens]
            positions = positions[:real_num_tokens]
        # The shardq token splits depend only on forward-constant inputs
        # (real_num_tokens, world_size, and the per-forward query_start_loc /
        # seq_lens), so they are identical for every layer in a forward. Cache
        # them on the forward context (per-forward lifetime) so the ~30 layers
        # do not each rebuild the arange/cat index metadata on the host.
        splits_cache = None
        try:
            _fctx = get_forward_context()
            splits_cache = getattr(_fctx, "_shardq_splits_cache", None)
            if splits_cache is None:
                splits_cache = {}
                _fctx._shardq_splits_cache = splits_cache  # type: ignore[attr-defined]
        except Exception:
            splits_cache = None
        cached_splits = (
            splits_cache.get(real_num_tokens) if splits_cache is not None else None
        )
        if cached_splits is None:
            all_rank_splits = build_all_rank_shardq_token_splits(
                real_num_tokens, tp_group.world_size, device=hidden_states.device
            )
            split = build_shardq_token_split(
                swa_metadata.query_start_loc,
                swa_metadata.seq_lens,
                real_num_tokens,
                tp_group.world_size,
                tp_group.rank_in_group,
            )
            if splits_cache is not None:
                splits_cache[real_num_tokens] = (all_rank_splits, split)
        else:
            all_rank_splits, split = cached_splits

        # Transpose the heads into the q-projection stream so its
        # single-warp, SM-free cross-rank barrier overlaps the indexer /
        # compressor aux streams, instead of running fully exposed after the
        # parallel section joins. (Hides the transpose only when the aux work
        # extends past wq_b, e.g. larger indexer / longer context.)
        def q_proj_and_transpose() -> torch.Tensor:
            out = self.wq_b(qr)
            if self.shardq_born_local:
                # Replicated wq_b emits all n_heads; take this rank's
                # head-parallel slice to reproduce ColumnParallel for the
                # transpose path (used for decode / mixed batches).
                w = self.n_local_heads * self.head_dim
                rank = tp_group.rank_in_group
                # .contiguous(): the column slice is non-contiguous, and the
                # fused qnorm/rope/kv-insert kernel requires a contiguous q.
                out = out[:, rank * w : (rank + 1) * w].contiguous()
            q = out.view(-1, self.n_local_heads, self.head_dim)
            q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
            q = q[:, : self.n_local_heads, :].contiguous()
            return transpose_heads_to_shardq(q, all_rank_splits, tp_group)

        def q_proj_born_local() -> torch.Tensor:
            self._kv_insert_only(kv, positions, attn_metadata)
            qr_local = qr.index_select(0, split.local_to_global)
            q = self.wq_b(qr_local).view(-1, self.n_heads, self.head_dim)
            assert q.shape[-2] == self.n_heads
            qpos_cache = None
            try:
                fctx = get_forward_context()
                qpos_cache = getattr(fctx, "_shardq_qpos_cache", None)
                if qpos_cache is None:
                    qpos_cache = {}
                    fctx._shardq_qpos_cache = qpos_cache  # type: ignore[attr-defined]
            except Exception:
                qpos_cache = None
            q_pos = qpos_cache.get(real_num_tokens) if qpos_cache is not None else None
            if q_pos is None:
                q_pos = positions.index_select(0, split.local_to_global)
                if qpos_cache is not None:
                    qpos_cache[real_num_tokens] = q_pos
            q = self._fused_qnorm_rope_q_only(q, q_pos, attn_metadata)
            assert q.shape[-2] == self.get_padded_num_q_heads(self.n_heads)
            return q

        # Born-local Q is prefill-only: the fold-slice + q-only path assumes the
        # indexer's prefill (coalesced) layout. For decode / mixed batches fall
        # back to the validated transpose path (rank-sliced replicated wq_b).
        use_born_local = self.shardq_born_local and swa_metadata.num_decode_tokens == 0
        q_proj = q_proj_born_local if use_born_local else q_proj_and_transpose

        if self.indexer is not None:
            aux_streams = self.aux_stream_list
            indexer = self.indexer
            assert self.compressor is not None
            compressor = self.compressor

            if swa_metadata.num_decode_tokens == 0:
                indexer_fn = lambda: self._run_shardq_prefill_indexer(
                    indexer,
                    hidden_states,
                    qr,
                    indexer_kv_score,
                    indexer_weights,
                    positions,
                    split,
                    attn_metadata,
                )
            else:
                indexer_fn = lambda: indexer(
                    hidden_states,
                    qr,
                    indexer_kv_score,
                    indexer_weights,
                    positions,
                    self.indexer_rotary_emb,
                )

            q, _ = execute_in_parallel(
                q_proj,
                [
                    indexer_fn,
                    lambda: compressor(kv_score, positions, self.rotary_emb),
                ],
                self.ln_events[0],
                [self.ln_events[1], self.ln_events[2]],
                [aux_streams[0], aux_streams[1]] if aux_streams is not None else None,
                enable=aux_streams is not None,
            )
        elif self.compressor is not None:
            aux_stream = (
                self.aux_stream_list[0] if self.aux_stream_list is not None else None
            )
            compressor = self.compressor

            q, _ = maybe_execute_in_parallel(
                q_proj,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        else:
            q = q_proj()

        full_padded_heads = self.get_padded_num_q_heads(self.n_heads)
        if q.shape[-2] < full_padded_heads:
            q = torch.nn.functional.pad(
                q, (0, 0, 0, full_padded_heads - q.shape[-2]), value=0.0
            )
        if self.shardq_born_local:
            assert q.shape[-2] == full_padded_heads
        o_shardq = torch.empty_like(q)
        # Replicated full-head sink — no per-layer all-gather (see __init__).
        self._forward_mqa_shardq(q, o_shardq, split, self.attn_sink_full)
        o_shardq = o_shardq[:, : self.n_heads, :]
        o = restore_shardq_to_heads(o_shardq, all_rank_splits, tp_group)
        if sp_pad > 0:
            # Re-pad the SP dummy rows (zeros) so the caller's _o_proj +
            # downstream SP reduce_scatter see the padded row count.
            o = torch.nn.functional.pad(o, (0, 0, 0, 0, 0, sp_pad), value=0.0)
        return o

    def _build_shardq_prefill_indexer_metadata(
        self,
        indexer_metadata: DeepseekV32IndexerMetadata,
        split: ShardQTokenSplit,
        positions: torch.Tensor,
        indexer_compress_ratio: int,
        max_prefill_buffer_size: int,
        device: torch.device,
    ) -> tuple[DeepseekV32IndexerMetadata, torch.Tensor, torch.Tensor]:
        if indexer_metadata.block_table is None:
            raise ValueError("shardq local indexer requires indexer block_table")

        # The shardq query layout and local indexer metadata are identical for
        # every layer of the same compress_ratio in a forward. Cache the
        # host-heavy layout and fully-built metadata on the forward context so
        # the ~30 layers do not each pay the .cpu()/.item() host syncs, Python
        # loops, and metadata construction.
        cache = None
        try:
            fctx = get_forward_context()
            cache = getattr(fctx, "_shardq_idx_layout_cache", None)
            if cache is None:
                cache = {}
                fctx._shardq_idx_layout_cache = cache  # type: ignore[attr-defined]
        except Exception:
            cache = None
        layout = cache.get(indexer_compress_ratio) if cache is not None else None
        if layout is None:
            layout = self._build_shardq_indexer_layout(
                indexer_metadata,
                split,
                positions,
                indexer_compress_ratio,
                max_prefill_buffer_size,
                device,
            )
            if cache is not None:
                cache[indexer_compress_ratio] = layout

        built = layout.get("_built")
        if built is not None:
            return cast(
                tuple[DeepseekV32IndexerMetadata, torch.Tensor, torch.Tensor],
                built,
            )

        coalesced_indices = layout["coalesced_indices"]
        req_ids = layout["req_ids"]
        num_local_tokens = layout["num_local_tokens"]
        # Forward-constant gathers for the shared indexer metadata group.
        block_table = indexer_metadata.block_table.index_select(0, req_ids)
        local_seq_lens = indexer_metadata.seq_lens.index_select(0, req_ids)
        local_positions = positions.index_select(0, coalesced_indices)
        chunks = [
            DeepseekV32IndexerPrefillChunkMetadata(
                cu_seqlen_ks=cl["cu_seqlen_ks"],
                cu_seqlen_ke=cl["cu_seqlen_ke"],
                cu_seq_lens=cl["cu_seq_lens"],
                token_to_seq=cl["token_to_seq"],
                total_seq_lens=cl["total_seq_lens"],
                block_table=block_table[cl["req_start"] : cl["req_end"]],
                token_start=cl["token_start"],
                token_end=cl["token_end"],
                num_reqs=cl["req_end"] - cl["req_start"],
                skip_kv_gather=cl["skip_kv_gather"],
            )
            for cl in layout["chunk_layouts"]
        ]
        slot_mapping = torch.empty(
            (num_local_tokens,),
            dtype=indexer_metadata.slot_mapping.dtype,
            device=device,
        )
        built = (
            DeepseekV32IndexerMetadata(
                seq_lens=local_seq_lens,
                max_seq_len=indexer_metadata.max_seq_len,
                slot_mapping=slot_mapping,
                num_decodes=0,
                num_decode_tokens=0,
                num_prefills=local_seq_lens.shape[0],
                num_prefill_tokens=num_local_tokens,
                prefill=DeepseekV32IndexerPrefillMetadata(chunks),
                decode=None,
                block_table=block_table,
            ),
            coalesced_indices,
            local_positions,
        )
        layout["_built"] = built
        return built

    def _build_shardq_indexer_layout(
        self,
        indexer_metadata: DeepseekV32IndexerMetadata,
        split: ShardQTokenSplit,
        positions: torch.Tensor,
        indexer_compress_ratio: int,
        max_prefill_buffer_size: int,
        device: torch.device,
    ) -> dict:
        """Build the per-forward, layer-independent shardq indexer layout.

        Everything here depends only on forward-constant split, positions,
        seq_lens, and compress_ratio. Returns a dict of coalesced row indices,
        ordered request ids, and per-chunk tensors.
        """
        segment_req_ids_cpu = list(split.segment_req_ids_cpu)
        segment_qsl_cpu = list(split.local_query_start_loc_cpu)
        ordered_req_ids: list[int] = []
        req_segment_ids: dict[int, list[int]] = {}
        for segment_id, req_id in enumerate(segment_req_ids_cpu):
            if req_id not in req_segment_ids:
                ordered_req_ids.append(req_id)
                req_segment_ids[req_id] = []
            req_segment_ids[req_id].append(segment_id)

        local_parts: list[torch.Tensor] = []
        query_lens: list[int] = []
        for req_id in ordered_req_ids:
            req_query_len = 0
            for segment_id in req_segment_ids[req_id]:
                start = segment_qsl_cpu[segment_id]
                end = segment_qsl_cpu[segment_id + 1]
                if end <= start:
                    continue
                local_parts.append(split.local_to_global[start:end])
                req_query_len += end - start
            query_lens.append(req_query_len)

        if local_parts:
            coalesced_indices = torch.cat(local_parts)
        else:
            coalesced_indices = torch.empty(
                0, dtype=split.local_to_global.dtype, device=device
            )

        req_ids = torch.tensor(ordered_req_ids, dtype=torch.long, device=device)
        local_seq_lens = indexer_metadata.seq_lens.index_select(0, req_ids)
        compressed_seq_lens = local_seq_lens // indexer_compress_ratio
        coalesced_positions = positions.index_select(0, coalesced_indices).to(
            torch.int32
        )

        local_seq_lens_cpu = torch.tensor(
            [split.seq_lens_cpu[r] for r in ordered_req_ids], dtype=torch.int32
        )
        compressed_seq_lens_cpu = local_seq_lens_cpu // indexer_compress_ratio
        query_lens_cpu = torch.tensor(query_lens, dtype=torch.int32)
        query_lens_gpu = torch.tensor(query_lens, dtype=torch.int32, device=device)
        local_qsl_cpu = torch.empty(len(query_lens) + 1, dtype=torch.int32)
        local_qsl_cpu[0] = 0
        local_qsl_cpu[1:] = torch.cumsum(query_lens_cpu, dim=0)

        max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        # One packer call over ALL local requests: budget-packs multiple requests
        # into a chunk (num_reqs>1) instead of one chunk per request.
        chunk_specs = split_indexer_prefill_chunks(
            compressed_seq_lens_cpu,
            query_lens_cpu,
            max_prefill_buffer_size,
            max_logits_bytes,
            request_offset=0,
        )

        chunks = []
        for req_slice, query_slice in chunk_specs:
            start = req_slice.start
            end = req_slice.stop
            if start is None or end is None:
                raise ValueError("shardq local indexer request slice is incomplete")
            qs0 = query_slice.start
            qs1 = query_slice.stop
            if qs0 is None or qs1 is None:
                raise ValueError("shardq local indexer query slice is incomplete")

            num_reqs = end - start
            total_seq_lens = int(compressed_seq_lens_cpu[start:end].sum().item())
            if total_seq_lens == 0:
                continue

            # KV side: per-request compressed-KV segments in the gather workspace.
            arange_reqs = torch.arange(num_reqs, dtype=torch.int32, device=device)
            cu_seq_lens = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            cu_seq_lens[1:] = torch.cumsum(compressed_seq_lens[start:end], dim=0)
            token_to_seq = torch.repeat_interleave(
                arange_reqs,
                compressed_seq_lens[start:end],
                output_size=total_seq_lens,
            )

            # Query side: per-row request index (request-major) -> key window.
            # ks = the row's request KV base; ke = base + causal-compressed end from
            # the folded global positions (clamped to the request's compressed len).
            chunk_q_base = int(local_qsl_cpu[start].item())
            chunk_q_len = int(local_qsl_cpu[end].item()) - chunk_q_base
            b_local_full = torch.repeat_interleave(
                arange_reqs,
                query_lens_gpu[start:end],
                output_size=chunk_q_len,
            )
            pos_full = coalesced_positions[chunk_q_base : chunk_q_base + chunk_q_len]
            comp_per_row = compressed_seq_lens[start:end][b_local_full]
            ke_full = cu_seq_lens[b_local_full] + torch.minimum(
                (pos_full + 1) // indexer_compress_ratio,
                comp_per_row,
            ).to(torch.int32)
            ks_full = cu_seq_lens[b_local_full].to(torch.int32)

            token_start = chunk_q_base + qs0
            token_end = chunk_q_base + qs1
            cu_seq_len_ks = ks_full[qs0:qs1].contiguous()
            cu_seq_len_ke = ke_full[qs0:qs1].contiguous()

            chunks.append(
                {
                    "cu_seqlen_ks": cu_seq_len_ks,
                    "cu_seqlen_ke": cu_seq_len_ke,
                    "cu_seq_lens": cu_seq_lens,
                    "token_to_seq": token_to_seq,
                    "total_seq_lens": total_seq_lens,
                    "req_start": start,
                    "req_end": end,
                    "token_start": token_start,
                    "token_end": token_end,
                    "skip_kv_gather": qs0 > 0,
                }
            )

        num_local_tokens = coalesced_indices.numel()
        return {
            "coalesced_indices": coalesced_indices,
            "req_ids": req_ids,
            "num_local_tokens": num_local_tokens,
            "chunk_layouts": chunks,
        }

    def _run_shardq_prefill_indexer(
        self,
        indexer,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        split: ShardQTokenSplit,
        attn_metadata: dict,
    ) -> None:
        indexer_metadata = cast(
            DeepseekV32IndexerMetadata | None,
            attn_metadata.get(indexer.k_cache.prefix),
        )
        if indexer_metadata is None:
            raise ValueError("shardq local indexer requires indexer metadata")
        if indexer_metadata.num_decodes != 0:
            raise ValueError("shardq local indexer only handles pure-prefill batches")

        if split.local_to_global.numel() == 0:
            indexer.compressor(indexer_kv_score, positions, self.indexer_rotary_emb)
            return None

        local_metadata, local_indices, local_positions = (
            self._build_shardq_prefill_indexer_metadata(
                indexer_metadata,
                split,
                positions,
                indexer.compress_ratio,
                indexer.max_total_seq_len,
                hidden_states.device,
            )
        )
        assert self.topk_indices_buffer is not None
        topk_shape = (local_indices.numel(), indexer.topk_tokens)
        topk_dtype = self.topk_indices_buffer.dtype
        cache = None
        try:
            fctx = get_forward_context()
            cache = getattr(fctx, "_shardq_local_topk_cache", None)
            if cache is None:
                cache = {}
                fctx._shardq_local_topk_cache = cache  # type: ignore[attr-defined]
        except Exception:
            cache = None
        topk_key = (*topk_shape, topk_dtype)
        local_topk = cache.get(topk_key) if cache is not None else None
        if local_topk is None:
            local_topk = torch.empty(
                topk_shape,
                dtype=topk_dtype,
                device=hidden_states.device,
            )
            if cache is not None:
                cache[topk_key] = local_topk
        local_hidden_states = hidden_states.index_select(0, local_indices)
        local_qr = qr.index_select(0, local_indices)
        local_indexer_weights = indexer_weights.index_select(0, local_indices)
        compressor = indexer.compressor

        def wq_b_and_q_quant():
            q, _ = indexer.wq_b(local_qr)
            q = q.view(-1, indexer.n_head, indexer.head_dim)
            return fused_indexer_q_rope_quant(
                local_positions,
                q,
                self.indexer_rotary_emb.cos_sin_cache,
                local_indexer_weights,
                indexer.softmax_scale,
                indexer.n_head**-0.5,
                use_fp4=indexer.use_fp4_kv,
            )

        (q_quant, weights), k = maybe_execute_in_parallel(
            wq_b_and_q_quant,
            lambda: compressor(indexer_kv_score, positions, self.indexer_rotary_emb),
            indexer.ln_events[0],
            indexer.ln_events[1],
            indexer.aux_stream,
        )
        indexer.indexer_op.forward_with_metadata(
            local_hidden_states,
            q_quant,
            k,
            weights,
            local_metadata,
            local_topk,
        )
        self.topk_indices_buffer.index_copy_(0, local_indices, local_topk)
        return None

    def _forward_mqa_shardq(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        split: ShardQTokenSplit,
        attn_sink: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        assert isinstance(attn_metadata, dict)

        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache
        num_local_decode_tokens = count_shardq_tokens_before(
            split.ranges, swa_metadata.num_decode_tokens
        )

        if num_local_decode_tokens > 0:
            self._forward_decode_shardq(
                q=q[:num_local_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_local_decode_tokens],
                split=split,
                num_local_decode_tokens=num_local_decode_tokens,
                attn_sink=attn_sink,
            )
        if num_local_decode_tokens < q.shape[0]:
            self._forward_prefill_shardq(
                q=q[num_local_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_local_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
                split=split,
                num_local_decode_tokens=num_local_decode_tokens,
                attn_sink=attn_sink,
            )

    def _forward_decode_shardq(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
        split: ShardQTokenSplit,
        num_local_decode_tokens: int,
        attn_sink: torch.Tensor,
    ) -> None:
        local_decode_indices = split.local_to_global[:num_local_decode_tokens]
        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                assert swa_metadata.is_valid_token is not None
                assert swa_metadata.token_to_req_indices is not None
                topk_local = self.topk_indices_buffer.index_select(
                    0, local_decode_indices
                )
                token_to_req = swa_metadata.token_to_req_indices.index_select(
                    0, local_decode_indices
                )
                is_valid = swa_metadata.is_valid_token.index_select(
                    0, local_decode_indices
                )
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    topk_local,
                    token_to_req,
                    attn_metadata.block_table[: swa_metadata.num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_local_decode_tokens, 1, -1)
            else:
                assert attn_metadata.c128a_global_decode_topk_indices is not None
                assert attn_metadata.c128a_decode_topk_lens is not None
                c128a_decode_topk = attn_metadata.c128a_global_decode_topk_indices
                topk_indices = c128a_decode_topk.index_select(0, local_decode_indices)
                topk_lens = attn_metadata.c128a_decode_topk_lens.index_select(
                    0, local_decode_indices
                )

        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.decode_swa_lens is not None
        swa_indices = swa_metadata.decode_swa_indices.index_select(
            0, local_decode_indices
        )
        swa_lens = swa_metadata.decode_swa_lens.index_select(0, local_decode_indices)

        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        tile_metadata = self._get_shardq_decode_tile_metadata(
            swa_metadata, num_local_decode_tokens
        )
        flash_mla_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )

    def _get_shardq_decode_tile_metadata(
        self,
        swa_metadata: "DeepseekSparseSWAMetadata",
        num_local_decode_tokens: int,
    ):
        if num_local_decode_tokens == swa_metadata.num_decode_tokens:
            if self.compress_ratio <= 1:
                tile_metadata = swa_metadata.tile_sched_swaonly
            elif self.compress_ratio == 4:
                tile_metadata = swa_metadata.tile_sched_c4a
            elif self.compress_ratio == 128:
                tile_metadata = swa_metadata.tile_sched_c128a
            else:
                raise ValueError(
                    f"Unsupported compress_ratio={self.compress_ratio}; "
                    "expected 1, 4, or 128."
                )
            if tile_metadata is not None:
                return tile_metadata

        # The shared per-layer-type scheduler metadata is valid only when this
        # shardq rank is attending the same decode rows as the original batch.
        # Otherwise q/topk_length rows differ, so give FlashMLA a fresh metadata
        # object and let it plan for the shardq local rows.
        return get_mla_metadata()[0]

    def _get_shardq_prefill_plan(
        self,
        split: ShardQTokenSplit,
        swa_metadata: "DeepseekSparseSWAMetadata",
        num_local_decode_tokens: int,
        device: torch.device,
    ) -> dict | None:
        """Per-forward cache of the shardq prefill segment plan.

        The plan is forward-constant (it depends only on ``split``,
        ``swa_metadata`` and the SWA window), so the ~30 layers share one
        build. Returns None when there are no prefill segments.
        """
        cache = None
        try:
            fctx = get_forward_context()
            cache = getattr(fctx, "_shardq_prefill_plan_cache", None)
            if cache is None:
                cache = {}
                fctx._shardq_prefill_plan_cache = cache  # type: ignore[attr-defined]
        except Exception:
            cache = None
        key = (self.window_size, num_local_decode_tokens)
        if cache is not None and key in cache:
            cached = cache[key]
            return None if cached is _SHARDQ_NO_PREFILL else cached
        plan = self._build_shardq_prefill_plan(
            split, swa_metadata, num_local_decode_tokens, device
        )
        if cache is not None:
            cache[key] = _SHARDQ_NO_PREFILL if plan is None else plan
        return plan

    def _build_shardq_prefill_plan(
        self,
        split: ShardQTokenSplit,
        swa_metadata: "DeepseekSparseSWAMetadata",
        num_local_decode_tokens: int,
        device: torch.device,
    ) -> dict | None:
        segment_req_ids_cpu = list(split.segment_req_ids_cpu)
        segment_qsl_cpu = list(split.local_query_start_loc_cpu)
        prefill_segment_ids = [
            i
            for i, req_id in enumerate(segment_req_ids_cpu)
            if req_id >= swa_metadata.num_decodes
        ]
        if not prefill_segment_ids:
            return None

        seg_ids = torch.tensor(prefill_segment_ids, dtype=torch.long, device=device)
        req_ids = split.segment_req_ids.index_select(0, seg_ids).to(torch.long)
        local_seq_lens = split.local_seq_lens.index_select(0, seg_ids)
        segment_lengths_cpu = [
            segment_qsl_cpu[i + 1] - segment_qsl_cpu[i] for i in prefill_segment_ids
        ]
        segment_lengths = torch.tensor(
            segment_lengths_cpu, dtype=torch.int32, device=device
        )
        local_query_start_loc = torch.empty(
            len(prefill_segment_ids) + 1, dtype=torch.int32, device=device
        )
        local_query_start_loc[0] = 0
        local_query_start_loc[1:] = torch.cumsum(segment_lengths, dim=0)
        prefix_lens = local_seq_lens - segment_lengths
        gather_lens = segment_lengths + torch.clamp(
            prefix_lens, min=0, max=self.window_size - 1
        )
        local_prefill_indices = split.local_to_global[num_local_decode_tokens:]
        seg_seq_lens_cpu = [split.local_seq_lens_cpu[i] for i in prefill_segment_ids]
        prefix_lens_cpu = [
            sl - seg for sl, seg in zip(seg_seq_lens_cpu, segment_lengths_cpu)
        ]
        gather_lens_cpu = [
            seg + min(max(pl, 0), self.window_size - 1)
            for seg, pl in zip(segment_lengths_cpu, prefix_lens_cpu)
        ]
        local_qsl_cpu = [0]
        for seg in segment_lengths_cpu:
            local_qsl_cpu.append(local_qsl_cpu[-1] + seg)

        return {
            "req_ids": req_ids,
            "local_seq_lens": local_seq_lens,
            "local_query_start_loc": local_query_start_loc,
            "gather_lens": gather_lens,
            "local_prefill_indices": local_prefill_indices,
            "seq_lens_cpu": seg_seq_lens_cpu,
            "gather_lens_cpu": gather_lens_cpu,
            "local_qsl_cpu": local_qsl_cpu,
            "num_segments": len(prefill_segment_ids),
        }

    def _forward_prefill_shardq(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        split: ShardQTokenSplit,
        num_local_decode_tokens: int,
        attn_sink: torch.Tensor,
    ) -> None:
        swa_only = attn_metadata is None
        # The prefill segment plan (req ids, per-segment lengths, cu-seqlens,
        # gather lengths and their host copies) depends only on the forward-
        # constant ``split`` / ``swa_metadata`` / window, not on the layer.
        # Build it once per forward and reuse across the ~30 layers so each
        # layer skips the .cpu()/.item() host syncs and tensor constructions.
        plan = self._get_shardq_prefill_plan(
            split, swa_metadata, num_local_decode_tokens, q.device
        )
        if plan is None:
            return

        req_ids = plan["req_ids"]
        local_seq_lens = plan["local_seq_lens"]
        local_query_start_loc = plan["local_query_start_loc"]
        gather_lens = plan["gather_lens"]
        local_prefill_indices = plan["local_prefill_indices"]
        seq_lens_cpu = plan["seq_lens_cpu"]
        gather_lens_cpu = plan["gather_lens_cpu"]
        local_qsl_cpu = plan["local_qsl_cpu"]
        num_segments = plan["num_segments"]
        if not swa_only:
            assert attn_metadata is not None
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer.index_select(
                    0, local_prefill_indices
                )
            else:
                assert attn_metadata.c128a_prefill_topk_indices is not None
                prefill_rows = local_prefill_indices - swa_metadata.num_decode_tokens
                topk_indices = attn_metadata.c128a_prefill_topk_indices.index_select(
                    0, prefill_rows
                )
            top_k = topk_indices.shape[-1]
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer.index_select(
                0, local_prefill_indices
            )
            top_k = 0

        compressed_block_table = (
            attn_metadata.block_table.index_select(0, req_ids)
            if attn_metadata is not None
            else None
        )
        swa_block_table = swa_metadata.block_table.index_select(0, req_ids)
        workspace_manager = current_workspace_manager()

        for chunk_start in range(0, num_segments, self.PREFILL_CHUNK_SIZE):
            chunk_end = min(num_segments, chunk_start + self.PREFILL_CHUNK_SIZE)
            chunk_size = chunk_end - chunk_start
            chunk_max_compressed = 0
            if not swa_only:
                chunk_max_compressed = max(
                    seq_len // self.compress_ratio
                    for seq_len in seq_lens_cpu[chunk_start:chunk_end]
                )
            chunk_max_gather = max(gather_lens_cpu[chunk_start:chunk_end])
            chunk_m = chunk_max_compressed + chunk_max_gather
            kv = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_m, q.shape[-1]), torch.bfloat16),
            )[0]

            if not swa_only:
                assert compressed_block_table is not None
                assert compressed_k_cache is not None
                assert attn_metadata is not None
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=local_seq_lens[chunk_start:chunk_end]
                    // self.compress_ratio,
                    gather_lens=None,
                    block_table=compressed_block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=local_seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=chunk_max_compressed,
            )

            query_start = local_qsl_cpu[chunk_start]
            query_end = local_qsl_cpu[chunk_end]
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                local_query_start_loc[chunk_start : chunk_end + 1],
                local_seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_m,
                chunk_max_compressed,
            )
            flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=attn_sink,
                topk_length=combined_lens,
                out=output[query_start:query_end],
            )
