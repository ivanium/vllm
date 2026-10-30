# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)

NUM_PAGES = 10
BLOCK_SIZE = 16
NUM_KV_HEADS = 4
HEAD_SIZE = 64
DTYPE = torch.float16

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lp_group(
    layer_names: list[str],
    spec,
    row_bytes: int,
    blocks_per_page: int = 1,
    kernel_rows_per_block: int = 1,
) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        layer_names,
        spec,
        blocks_per_page=blocks_per_page,
        kernel_rows_per_block=kernel_rows_per_block,
        row_bytes=row_bytes,
    )


def _lp_config(kv_cache_groups: list[KVCacheGroupSpec]) -> KVCacheConfig:
    large_page_bytes = max(
        g.blocks_per_page * g.kernel_rows_per_block * g.row_bytes
        for g in kv_cache_groups
    )
    return KVCacheConfig(
        num_pages=NUM_PAGES,
        kv_cache_tensors=[],
        kv_cache_groups=[
            replace(
                g,
                num_blocks=NUM_PAGES
                * large_page_bytes
                // (g.kernel_rows_per_block * g.row_bytes),
            )
            for g in kv_cache_groups
        ],
        large_page_bytes=large_page_bytes,
        block_size=BLOCK_SIZE,
    )


def _register_views(kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
    """Per-layer strided row views over one shared backing buffer, as built
    by the worker's large-page binding."""
    backing = torch.empty(
        NUM_PAGES * kv_cache_config.large_page_bytes, dtype=torch.int8
    )
    kv_caches: dict[str, torch.Tensor] = {}
    for group in kv_cache_config.kv_cache_groups:
        row = group.row_bytes
        num_rows = backing.numel() // row
        layer_row = row // len(group.layer_names)
        for layer_idx, layer_name in enumerate(group.layer_names):
            kv_caches[layer_name] = torch.as_strided(
                backing,
                (num_rows, layer_row),
                (row, 1),
                storage_offset=layer_idx * layer_row,
            )
    return kv_caches


def _single_rank_vllm_config(total_kv_heads: int):
    """A one-rank (TP=1) parallel config, as canonical mappings are derived
    from it."""
    vllm_config = MagicMock()
    parallel_config = vllm_config.parallel_config
    parallel_config.tensor_parallel_size = 1
    parallel_config.decode_context_parallel_size = 1
    parallel_config.prefill_context_parallel_size = 1
    parallel_config.cp_kv_cache_interleave_size = 1
    parallel_config.world_size = 1
    parallel_config.rank = 0
    vllm_config.model_config.get_total_num_kv_heads.return_value = total_kv_heads
    return vllm_config


def _make_worker(
    kv_cache_config: KVCacheConfig,
    replicated_layout: bool = False,
    rank: int = 0,
    canonical_layout: bool = False,
):
    """Create an OffloadingConnectorWorker with mocked dependencies."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )

    spec = MagicMock(spec=OffloadingSpec)
    spec.replicated_layout = replicated_layout
    spec.config = MagicMock()
    spec.config.canonical_layout = canonical_layout
    spec.config.parallel.rank = rank
    spec.get_worker.return_value = MagicMock()

    worker = OffloadingConnectorWorker(
        spec=spec,
        vllm_config=_single_rank_vllm_config(NUM_KV_HEADS),
        kv_cache_config=kv_cache_config,
    )
    worker.worker = MagicMock()

    return worker, spec


def _store_metadata(job_id: int) -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={
            job_id: TransferJob(
                req_id="req",
                src_spec=GPULoadStoreSpec([0], group_sizes=(1,), block_indices=(0,)),
                dst_spec=LoadStoreSpec(),
            )
        },
    )


def _load_metadata(job_id: int) -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(
        load_jobs={
            job_id: TransferJob(
                req_id="req",
                src_spec=LoadStoreSpec(),
                dst_spec=GPULoadStoreSpec([0], group_sizes=(1,), block_indices=(0,)),
            )
        },
        store_jobs={},
    )


def _empty_metadata() -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(load_jobs={}, store_jobs={})


def _offloading_config(rank: int = 0) -> OffloadingConfig:
    return OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=False,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float16"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=2,
            tp_size=2,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
    )


class BareExternalOffloadingSpec(OffloadingSpec):
    def get_manager(self) -> OffloadingManager:
        raise NotImplementedError

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prepare_store_kv_non_writer_marks_completed_without_submit():
    worker, _ = _make_worker(
        KVCacheConfig(
            num_pages=0, kv_cache_tensors=[], kv_cache_groups=[], block_size=16
        ),
        replicated_layout=True,
        rank=1,
    )

    worker.prepare_store_kv(_store_metadata(7))
    worker.start_kv_transfers(_empty_metadata())

    assert worker._unsubmitted_store_jobs == []
    assert worker.worker is not None
    worker.worker.submit_store.assert_not_called()
    meta = worker.build_connector_worker_meta()
    assert meta is not None
    assert meta.completed_jobs == {7: 1}


@pytest.mark.parametrize("rank,canonical_layout", [(0, False), (0, True), (1, True)])
def test_prepare_store_kv_writer_submits_store(rank, canonical_layout):
    worker, _ = _make_worker(
        KVCacheConfig(
            num_pages=0, kv_cache_tensors=[], kv_cache_groups=[], block_size=16
        ),
        replicated_layout=True,
        rank=rank,
        canonical_layout=canonical_layout,
    )

    worker.prepare_store_kv(_store_metadata(8))
    assert worker.build_connector_worker_meta() is None
    worker.start_kv_transfers(_empty_metadata())

    assert worker.worker is not None
    worker.worker.submit_store.assert_called_once()


def test_prepare_store_kv_non_replicated_rank_gt_zero_queues_store():
    worker, _ = _make_worker(
        KVCacheConfig(
            num_pages=0, kv_cache_tensors=[], kv_cache_groups=[], block_size=16
        ),
        replicated_layout=False,
        rank=1,
    )

    worker.prepare_store_kv(_store_metadata(9))
    assert worker.build_connector_worker_meta() is None
    assert len(worker._unsubmitted_store_jobs) == 1

    worker.start_kv_transfers(_empty_metadata())

    assert worker.worker is not None
    worker.worker.submit_store.assert_called_once()
    assert worker._unsubmitted_store_jobs == []


def test_handle_preemptions_non_writer_acks_flushed_store():
    worker, _ = _make_worker(
        KVCacheConfig(
            num_pages=0, kv_cache_tensors=[], kv_cache_groups=[], block_size=16
        ),
        replicated_layout=True,
        rank=1,
    )
    metadata = _store_metadata(10)
    metadata.jobs_to_flush = {10}

    worker.handle_preemptions(metadata)

    assert worker.worker is not None
    worker.worker.submit_store.assert_not_called()
    worker.worker.wait.assert_called_once_with({10})
    assert metadata.store_jobs == {}
    meta = worker.build_connector_worker_meta()
    assert meta is not None
    assert meta.completed_jobs == {10: 1}


def test_start_kv_transfers_non_writer_still_submits_load():
    worker, _ = _make_worker(
        KVCacheConfig(
            num_pages=0, kv_cache_tensors=[], kv_cache_groups=[], block_size=16
        ),
        replicated_layout=True,
        rank=1,
    )

    worker.start_kv_transfers(_load_metadata(10))

    assert worker.worker is not None
    worker.worker.submit_load.assert_called_once()
    assert worker.build_connector_worker_meta() is None


def test_offloading_connector_worker_accepts_plugin_spec_default_layout():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )

    spec = BareExternalOffloadingSpec(_offloading_config(rank=1))

    OffloadingConnectorWorker(
        spec=spec,
        vllm_config=_single_rank_vllm_config(NUM_KV_HEADS),
        kv_cache_config=KVCacheConfig(
            num_pages=0, kv_cache_tensors=[], kv_cache_groups=[], block_size=16
        ),
    )

    assert spec.replicated_layout is False


def test_register_kv_caches():
    """register_kv_caches builds one block-view tensor and one ref per group.

    Covers a full-attention group, an MLA group with two kernel rows per
    manager block, and two Mamba groups. Block b of every group must be row
    b of its tensor, at byte offset b * block_bytes of the backing storage.
    """
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    mla_spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=NUM_KV_HEADS * HEAD_SIZE * 2,
        dtype=DTYPE,
    )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((BLOCK_SIZE, HEAD_SIZE),),
        dtypes=(DTYPE,),
    )

    kv_cache_groups = [
        _lp_group([f"attn_{i}" for i in range(3)], attn_spec, row_bytes=384),
        _lp_group(["mla_0"], mla_spec, row_bytes=256, kernel_rows_per_block=2),
        _lp_group(["mamba_0", "mamba_1"], mamba_spec, row_bytes=192),
        _lp_group(["mamba_2"], mamba_spec, row_bytes=512),
    ]
    kv_cache_config = _lp_config(kv_cache_groups)
    assert kv_cache_config.large_page_bytes == 512
    kv_caches = _register_views(kv_cache_config)

    worker, spec = _make_worker(kv_cache_config)
    worker.register_kv_caches(kv_caches)

    canonical = spec.get_worker.call_args[0][0]
    assert isinstance(canonical, CanonicalKVCaches)

    # One tensor per group over the shared backing storage, one row per
    # block: 384 B blocks give 5120 // 384 = 13 ids, 512 B blocks give 10.
    assert len(canonical.tensors) == len(kv_cache_groups)
    for group, tensor, num_blocks in zip(
        kv_cache_config.kv_cache_groups, canonical.tensors, [13, 10, 26, 10]
    ):
        block_bytes = group.kernel_rows_per_block * group.row_bytes
        assert tensor.tensor.dtype == torch.int8
        assert tensor.tensor.shape == (num_blocks, block_bytes)
        assert tensor.tensor.stride(0) == block_bytes
        assert tensor.tensor.data_ptr() == (
            kv_caches[group.layer_names[0]].untyped_storage().data_ptr()
        )
        assert tensor.page_size_bytes == block_bytes

    # One data ref per group covering the whole manager block.
    assert len(canonical.group_data_refs) == len(kv_cache_groups)
    for t_idx, (group, refs) in enumerate(
        zip(kv_cache_groups, canonical.group_data_refs)
    ):
        assert refs == [
            CanonicalKVCacheRef(
                tensor_idx=t_idx,
                page_size_bytes=group.kernel_rows_per_block * group.row_bytes,
                mapping=refs[0].mapping,
            )
        ]
        # Single-rank registration: byte-identical blocks, so the mapping is
        # the portable identity.
        assert refs[0].mapping is not None
        assert refs[0].mapping.parallelism_agnostic


def test_register_kv_caches_skips_scratch_group():
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    kv_cache_config = _lp_config(
        [
            _lp_group(["layer0", "layer1"], attn_spec, row_bytes=256),
            _lp_group(
                ["scratch"],
                CircularBufferSpec(
                    block_size=4, num_kv_heads=1, head_size=1, dtype=DTYPE
                ),
                row_bytes=64,
            ),
        ]
    )
    worker, spec = _make_worker(kv_cache_config)
    worker.register_kv_caches(_register_views(kv_cache_config))

    canonical = spec.get_worker.call_args[0][0]
    assert len(canonical.tensors) == 1
    assert canonical.tensors[0].tensor.shape == (NUM_PAGES, 256)
    assert len(canonical.group_data_refs) == 1
    refs = canonical.group_data_refs[0]
    assert [(r.tensor_idx, r.page_size_bytes) for r in refs] == [(0, 256)]


def test_register_kv_caches_pp_gap_group_gets_no_tensor():
    """A group this rank holds no layer of registers no tensor and an empty
    ref list (pipeline-parallel gap)."""
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    groups = [
        _lp_group(["local_0"], attn_spec, row_bytes=128),
        _lp_group(["remote_0"], attn_spec, row_bytes=128),
    ]
    kv_cache_config = _lp_config(groups)
    kv_caches = _register_views(kv_cache_config)
    del kv_caches["remote_0"]

    worker, spec = _make_worker(kv_cache_config)
    worker.register_kv_caches(kv_caches)

    canonical = spec.get_worker.call_args[0][0]
    assert len(canonical.tensors) == 1
    assert len(canonical.group_data_refs) == 2
    assert [(r.tensor_idx) for r in canonical.group_data_refs[0]] == [0]
    assert canonical.group_data_refs[1] == []
