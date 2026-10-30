# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mock end-to-end MoRIIO WRITE between prefill and decode of any TP sizes.

Every prefill rank runs the real writer and connector offset code on real
large-page cache views (kernel-row [K, H, N, C] views over a per-rank pool,
manager blocks expanded through each side's own group geometry); a fake MoRI
transport copies the planned bytes into the decode ranks' caches, and each
decode rank counts write_done with the real wrapper. Checks that every decode
rank ends up with exactly its shard of every block and recurrent-state slot,
nothing else is touched, every prefill rank frees its blocks, and a decode
rank completes only after all its writers did.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import ssm_conv_transfer_utils
from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MORIIO_CONNECTOR_VERSION,
    ROLE,
    MoRIIOAgentMetadata,
    RemoteAllocInfo,
    WriteTask,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout import (
    expand_block_kernel_rows,
    kda_conv_ssm,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_tp_mapping import (
    _rank_heads,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    derive_mamba_conv_split,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    GroupGeometry,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from .utils import make_moriio_writer

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 4  # manager block, tokens
# Kernel rows per manager block: attention groups split their 4-token block
# into 2-token rows, so row expansion is exercised; mamba states are one row.
ROWS_PER_ATTN_BLOCK = 2
KERNEL_TOKENS = BLOCK_SIZE // ROWS_PER_ATTN_BLOCK
NOTIFY_PORT = 20000
# Prefill blocks and the decode blocks they land in, per transfer group.
PREFILL_BLOCKS = {"full": [1, 4], "swa": [3], "mamba": [2]}
DECODE_BLOCKS = {"full": [5, 2], "swa": [0], "mamba": [3]}
MIN_BLOCKS = 6
# GDN state: Q and K are KEY_DIM channels, V is V_HEADS * V_DIM channels.
KEY_DIM, V_HEADS, V_DIM, K_DIM, CONV_ROWS = 4, 4, 2, 2, 2


@dataclass(frozen=True)
class Layer:
    name: str
    kind: str  # "full", "swa", "mla" or "mamba"
    group: str  # transfer group the layer's blocks come from


MODELS = {
    "gqa": (4, [Layer("attn0", "full", "full"), Layer("attn1", "full", "full")]),
    "gqa_replicated": (2, [Layer("attn0", "full", "full")]),
    "mla": (1, [Layer("mla0", "mla", "full"), Layer("mla1", "mla", "full")]),
    "sliding": (4, [Layer("attn0", "full", "full"), Layer("swa0", "swa", "swa")]),
    "gdn_hybrid": (
        4,
        [Layer("attn0", "full", "full"), Layer("gdn0", "mamba", "mamba")],
    ),
    "mla_hybrid": (1, [Layer("mla0", "mla", "full"), Layer("kda0", "mamba", "mamba")]),
}


def _groups(model: str, hma: bool) -> list[str]:
    _, layers = MODELS[model]
    groups = sorted({layer.group for layer in layers}, key=list(DECODE_BLOCKS).index)
    return groups if hma else ["full"]


def _spec(layer: Layer, num_heads: int, tp_size: int):
    if layer.kind == "full":
        return FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=num_heads,
            head_size=2,
            dtype=torch.bfloat16,
        )
    if layer.kind == "swa":
        return SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=num_heads,
            head_size=2,
            dtype=torch.bfloat16,
            sliding_window=8,
        )
    if layer.kind == "mla":
        return MLAAttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
        )
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(
            ((2 * KEY_DIM + V_HEADS * V_DIM) // tp_size, CONV_ROWS),
            (V_HEADS // tp_size, V_DIM, K_DIM),
        ),
        dtypes=(torch.uint8, torch.uint8),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )


def _global_state(layers: list[Layer], heads: int) -> dict[str, torch.Tensor]:
    """The unsharded content of every layer, per block / state slot."""
    gen = torch.Generator().manual_seed(0)
    state = {}
    for layer in layers:
        if layer.kind == "mamba":
            n = (2 * KEY_DIM + V_HEADS * V_DIM) * CONV_ROWS + V_HEADS * V_DIM * K_DIM
            shape: tuple[int, ...] = (MIN_BLOCKS, n)
        else:
            spec = _spec(layer, 1 if layer.kind == "mla" else heads, 1)
            shape = (
                MIN_BLOCKS,
                spec.num_heads,
                spec.num_states,
                spec.state_content_size_bytes,
            )
        state[layer.name] = torch.randint(
            1, 255, shape, dtype=torch.uint8, generator=gen
        )
    return state


def _mamba_shard(slot: torch.Tensor, tp_rank: int, tp_size: int) -> tuple:
    """(conv, ssm) bytes of ``tp_rank`` for one unsharded GDN state slot."""
    parts, offset = [], 0
    for channels in (KEY_DIM, KEY_DIM, V_HEADS * V_DIM):
        size = channels * CONV_ROWS
        shard = size // tp_size
        parts.append(slot[offset + tp_rank * shard : offset + (tp_rank + 1) * shard])
        offset += size
    ssm = slot[offset:].view(V_HEADS, -1)
    heads = V_HEADS // tp_size
    return torch.cat(parts), ssm[tp_rank * heads : (tp_rank + 1) * heads].reshape(-1)


@dataclass
class Rank:
    tp_rank: int
    tp_size: int
    caches: dict[str, torch.Tensor]  # registered tensor per layer
    regions: list[torch.Tensor]  # registered byte regions, in session order
    worker: MoRIIOConnectorWorker
    group_geometry: dict[str, GroupGeometry] = field(default_factory=dict)
    group_num_blocks: dict[str, int] = field(default_factory=dict)
    wrapper: Any = None
    sent: list = field(default_factory=list)

    def first_row(self, group: str, block_id: int) -> int:
        return expand_block_kernel_rows(
            self.group_geometry[group], self.group_num_blocks[group], [block_id]
        )[0]


def _byte_alias(view: torch.Tensor, pool: torch.Tensor, row_bytes: int) -> torch.Tensor:
    """Contiguous byte region covering ``view``'s rows of the pool."""
    offset = view.storage_offset()
    row_content = view[0].numel() * view.element_size()
    extent = (view.shape[0] - 1) * row_bytes + row_content
    return pool[offset : offset + extent]


def _build_rank(model, tp_rank, tp_size, layout, hma, state, is_producer) -> Rank:
    heads, layers = MODELS[model]
    groups = _groups(model, hma)
    specs = {
        layer.name: _spec(layer, len(_rank_heads(tp_rank, tp_size, heads)), tp_size)
        for layer in layers
    }
    group_layers = {
        g: [layer for layer in layers if (layer.group if hma else "full") == g]
        for g in groups
    }
    # A group's row is the concatenation of its layers' row content.
    row_bytes = {
        g: sum(
            specs[layer.name].page_size_bytes
            if layer.kind == "mamba"
            else specs[layer.name].page_size_bytes // ROWS_PER_ATTN_BLOCK
            for layer in members
        )
        for g, members in group_layers.items()
    }
    # One page size per rank (4.5x the largest row), so some blocks cross
    # page ends; block b of a group sits at b * block_bytes of its pool.
    large_page_bytes = 9 * max(row_bytes.values()) // 2
    group_geometry = {
        g: GroupGeometry(1 if g == "mamba" else ROWS_PER_ATTN_BLOCK, row_bytes[g])
        for g in groups
    }
    num_pages = max(
        -(-MIN_BLOCKS * gp.block_bytes // large_page_bytes)
        for gp in group_geometry.values()
    )
    group_num_blocks = {
        g: num_pages * large_page_bytes // gp.block_bytes
        for g, gp in group_geometry.items()
    }

    # One pool per group: a large page is owned by a single group at a time,
    # so co-locating two groups' live blocks in one pool would alias them.
    caches, regions, meta_counts = {}, [], {}
    for g in groups:
        gp = group_geometry[g]
        pool = torch.zeros(num_pages * large_page_bytes, dtype=torch.uint8)
        num_rows = pool.numel() // gp.row_bytes
        layer_offset = 0
        for layer in group_layers[g]:
            spec = specs[layer.name]
            if layer.kind == "mamba":
                cache = torch.as_strided(
                    pool,
                    (num_rows, 1, 1, spec.page_size_bytes),
                    (gp.row_bytes, spec.page_size_bytes, spec.page_size_bytes, 1),
                    storage_offset=layer_offset,
                )
                conv, ssm = kda_conv_ssm(cache, spec)
                if is_producer:
                    for slot in PREFILL_BLOCKS[g]:
                        row = expand_block_kernel_rows(gp, group_num_blocks[g], [slot])[
                            0
                        ]
                        c, s = _mamba_shard(state[layer.name][slot], tp_rank, tp_size)
                        conv[row].view(-1).copy_(c)
                        ssm[row].view(-1).copy_(s)
                regions += [
                    MoRIIOConnectorWorker._contiguous_byte_alias(conv),
                    MoRIIOConnectorWorker._contiguous_byte_alias(ssm),
                ]
                meta_counts[layer.name] = 2
                layer_offset += spec.page_size_bytes
            else:
                kernel_spec = spec.copy_with_new_block_size(KERNEL_TOKENS)
                strides = compute_layout_strides(
                    kernel_spec,
                    num_rows,
                    1,
                    layout,
                    fixed_strides=(gp.row_bytes, gp.row_bytes, None, None, None),
                )
                shape = (
                    num_rows,
                    spec.num_heads,
                    KERNEL_TOKENS,
                    spec.state_content_size_bytes,
                )
                cache = torch.as_strided(
                    pool, shape, strides[1:], storage_offset=layer_offset
                )
                if is_producer:
                    held = (
                        slice(None)
                        if layer.kind == "mla"
                        else _rank_heads(tp_rank, tp_size, heads)
                    )
                    for block in PREFILL_BLOCKS[g]:
                        row = expand_block_kernel_rows(
                            gp, group_num_blocks[g], [block]
                        )[0]
                        # [H, R * N, C] -> rows [R, H, N, C]
                        cache[row : row + gp.kernel_rows_per_block] = (
                            state[layer.name][block, held]
                            .reshape(
                                cache.shape[1],
                                gp.kernel_rows_per_block,
                                KERNEL_TOKENS,
                                -1,
                            )
                            .permute(1, 0, 2, 3)
                        )
                regions.append(_byte_alias(cache, pool, gp.row_bytes))
                meta_counts[layer.name] = 1
                layer_offset += spec.page_size_bytes // ROWS_PER_ATTN_BLOCK
            caches[layer.name] = cache

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = tp_rank
    worker.world_size = tp_size
    worker.use_mla = layers[0].kind == "mla"
    worker._has_mamba = any(layer.kind == "mamba" for layer in layers)
    worker.model_config = SimpleNamespace(get_total_num_kv_heads=lambda: heads)
    worker.kv_caches = caches
    worker.layer_to_spec = specs
    worker.layer_to_group = {
        layer.name: groups.index(layer.group if hma else "full") for layer in layers
    }
    worker.kv_cache_config = KVCacheConfig(
        num_pages=num_pages,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [layer.name for layer in group_layers[g]],
                specs[group_layers[g][0].name],
                num_blocks=group_num_blocks[g],
                kernel_rows_per_block=group_geometry[g].kernel_rows_per_block,
                row_bytes=group_geometry[g].row_bytes,
            )
            for g in groups
        ],
        large_page_bytes=large_page_bytes,
        block_size=BLOCK_SIZE,
    )
    worker._mamba_payload_index_by_layer = {
        layer.name: groups.index("mamba") for layer in layers if layer.kind == "mamba"
    }
    worker._conv_decomp = next(
        (
            derive_mamba_conv_split(spec, tp_size)
            for spec in specs.values()
            if isinstance(spec, MambaSpec)
        ),
        None,
    )
    worker.kv_layer_mr_offset = {}
    worker._mamba_offset_templates = {}
    worker._region_session_index = None
    worker.layer_name_to_local_kv_cache_metadata = {
        layer.name: [b""] * meta_counts[layer.name] for layer in layers
    }
    worker._record_slice_layouts(caches)
    # Built without __init__: there is no engine to shut down.
    worker.shutdown = lambda: None
    return Rank(
        tp_rank, tp_size, caches, regions, worker, group_geometry, group_num_blocks
    )


class _Session:
    def __init__(self, src: torch.Tensor, dst: torch.Tensor):
        self.src, self.dst = src, dst


def _connect(producer: Rank, consumers: list[Rank], dp_engine: str) -> None:
    """Route the producer's sessions and notifications to the decode ranks."""
    consumer = consumers[0].worker
    consumer_groups = consumer.kv_cache_config.transfer_groups
    metadata = MoRIIOAgentMetadata(
        version=MORIIO_CONNECTOR_VERSION,
        engine_id="decode",
        agent_metadata=b"",
        kv_caches_base_addr=[],
        attn_backend_name="test",
        group_num_blocks=[group.num_blocks for group in consumer_groups],
        group_geometries=consumer.kv_cache_config.transfer_group_geometries,
        kv_head_layouts={
            name: layout.to_wire() for name, layout in consumer.kv_head_layouts.items()
        },
        mamba_state_layouts={
            name: layout.to_wire()
            for name, layout in consumer.mamba_state_layouts.items()
        },
        attention_block_sizes=consumer.attention_block_sizes,
    )
    # The handshake carries the metadata as msgpack.
    metadata = msgspec.msgpack.decode(
        msgspec.msgpack.encode(metadata), type=MoRIIOAgentMetadata
    )

    def get_built_session(engine_id: str):
        if engine_id == f"{dp_engine}_dp0":
            (target,) = producer.worker._write_tp_mapping(len(consumers)).targets
            tp_rank = target.tp_rank
        else:
            tp_rank = int(engine_id.rsplit("_tp", 1)[1])
        remote = consumers[tp_rank]
        sessions = [
            _Session(src, dst)
            for src, dst in zip(producer.regions, remote.regions, strict=True)
        ]
        return sessions, metadata

    def write_remote_data(sizes, local_offsets, remote_offsets, session):
        for size, local, remote in zip(sizes, local_offsets, remote_offsets):
            session.dst[remote : remote + size] = session.src[local : local + size]
        return "status"

    def send_notify(transfer_id, ip, port, message_type, message_fields=None):
        producer.sent.append((port - NOTIFY_PORT, message_type))
        consumers[port - NOTIFY_PORT].wrapper._handle_structured_message(
            {"type": message_type, "transfer_id": transfer_id, **(message_fields or {})}
        )

    producer.worker._get_built_session = get_built_session
    producer.wrapper = SimpleNamespace(
        done_remote_allocate_req_dict={},
        lock=threading.Lock(),
        done_req_ids=[],
        write_remote_data=write_remote_data,
        send_notify=send_notify,
        waiting_for_transfer_complete=lambda statuses: None,
        _mark_transfer_terminal_locked=lambda transfer_id: None,
        _is_transfer_terminal_locked=lambda transfer_id: False,
    )
    producer.worker.moriio_wrapper = producer.wrapper


def _consumer_wrapper() -> MoRIIOWrapper:
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_write_cache_req_ids = []
    wrapper.failed_write_cache_req_ids = []
    wrapper._write_done_counts = {}
    wrapper._terminal_transfer_ids = OrderedDict()
    return wrapper


def _expected(
    layer: Layer,
    group_name: str,
    group: GroupGeometry,
    num_blocks: int,
    state,
    heads,
    tp_rank,
    tp_size,
    cache,
) -> torch.Tensor:
    """What the decode cache must hold after the transfer."""
    pairs = zip(PREFILL_BLOCKS[group_name], DECODE_BLOCKS[group_name])
    if layer.kind == "mamba":
        expected = torch.zeros(cache.shape[0], cache.shape[-1], dtype=torch.uint8)
        for src, dst in pairs:
            conv, ssm = _mamba_shard(state[layer.name][src], tp_rank, tp_size)
            row = expand_block_kernel_rows(group, num_blocks, [dst])[0]
            expected[row, : conv.numel() + ssm.numel()] = torch.cat([conv, ssm])
        return expected
    held = slice(None) if layer.kind == "mla" else _rank_heads(tp_rank, tp_size, heads)
    expected = torch.zeros_like(cache)
    for src, dst in pairs:
        row = expand_block_kernel_rows(group, num_blocks, [dst])[0]
        # [H, R * N, C] -> rows [R, H, N, C]
        expected[row : row + group.kernel_rows_per_block] = (
            state[layer.name][src, held]
            .reshape(cache.shape[1], group.kernel_rows_per_block, cache.shape[2], -1)
            .permute(1, 0, 2, 3)
        )
    return expected


SCENARIOS = [
    (model, producer_tp, consumer_tp, hma)
    for model in MODELS
    for producer_tp, consumer_tp in [(1, 2), (2, 4), (2, 1), (4, 2), (2, 2), (1, 4)]
    for hma in (True, False)
    # Recurrent state needs the hybrid KV cache manager.
    if hma or "hybrid" not in model
]


@pytest.mark.parametrize(
    ("layouts"),
    [
        (KVCacheLayout.LBNHC, KVCacheLayout.LBNHC),
        (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC),
        (KVCacheLayout.LBHNC, KVCacheLayout.LBHNC),
    ],
    ids=["token-major", "mixed", "head-major"],
)
@pytest.mark.parametrize(("model", "producer_tp", "consumer_tp", "hma"), SCENARIOS)
def test_write_lands_every_shard_byte_exact(
    monkeypatch, model, producer_tp, consumer_tp, hma, layouts
):
    monkeypatch.setattr(
        ssm_conv_transfer_utils, "is_conv_state_dim_first", lambda: True
    )
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.CONSUMER)
    heads, layers = MODELS[model]
    state = _global_state(layers, heads)
    groups = _groups(model, hma)
    consumers = [
        _build_rank(model, d, consumer_tp, layouts[1], hma, state, False)
        for d in range(consumer_tp)
    ]
    for consumer in consumers:
        consumer.wrapper = _consumer_wrapper()
    producers = [
        _build_rank(model, p, producer_tp, layouts[0], hma, state, True)
        for p in range(producer_tp)
    ]

    writers: dict[int, set[int]] = {d: set() for d in range(consumer_tp)}
    pending = []
    for producer in producers:
        _connect(producer, consumers, "decode")
        mapping = producer.worker._write_tp_mapping(consumer_tp)
        for target in mapping.targets:
            writers[target.tp_rank].add(producer.tp_rank)
        info = RemoteAllocInfo(block_ids=[DECODE_BLOCKS[g] for g in groups])
        producer.wrapper.done_remote_allocate_req_dict["t"] = info
        writer = make_moriio_writer(producer.worker)
        for layer in layers:
            writer._execute_write_task(
                WriteTask(
                    request_id="req",
                    transfer_id="t",
                    dst_engine_id="decode",
                    local_block_ids=[PREFILL_BLOCKS[g] for g in groups],
                    remote_block_ids_hint=None,
                    layer_name=layer.name,
                    event=SimpleNamespace(synchronize=lambda: None),
                    remote_notify_port=NOTIFY_PORT,
                    remote_ip="decode",
                    remote_tp_size=consumer_tp,
                    tp_mapping=mapping,
                )
            )
        info.writes_expected = len(layers)
        pending.append((producer, writer, info))

    finished: set[int] = set()
    for producer, writer, info in pending:
        writer._finalize_if_complete("t", info)
        finished.add(producer.tp_rank)
        # Prefill frees its blocks whether or not it had anything to write.
        assert [ack.transfer_id for ack in producer.wrapper.done_req_ids] == ["t"]
        for d, consumer in enumerate(consumers):
            done = consumer.wrapper.pop_finished_write_req_ids()
            assert done == (
                {"t"}
                if writers[d] <= finished and (producer.tp_rank in writers[d])
                else set()
            ), (d, finished)

    for consumer in consumers:
        for layer in layers:
            cache = consumer.caches[layer.name]
            received = (
                cache.view(cache.shape[0], -1) if layer.kind == "mamba" else cache
            )
            group_name = layer.group if hma else "full"
            expected = _expected(
                layer,
                group_name,
                consumer.group_geometry[group_name],
                consumer.group_num_blocks[group_name],
                state,
                heads,
                consumer.tp_rank,
                consumer_tp,
                received,
            )
            assert torch.equal(received, expected), (layer.name, consumer.tp_rank)
