# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM-based (Triton) all-to-all permute for the DeepSeek-V4 shardQ transpose.

The shardq transpose/restore move strided row blocks into peer symmetric-memory
buffers. The copy-engine path issues one ``cudaMemcpy2DAsync`` per descriptor;
this module does the whole permute in a single Triton kernel that issues
vectorized NVLink stores from CUDA cores, encoding the permute as two
forward-constant index arrays:

  src_off[row]   element offset into the (flat) source tensor for output row
  dst_addr[row]  absolute byte address of element 0 of that row in the
                 destination peer's symmetric buffer

One launch covers every peer; ``width_elems`` contiguous elements move per row.
Both directions reduce to the same kernel because the descriptors already encode
the per-direction offsets and pitches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

import vllm.envs as envs
from vllm.models.deepseek_v4.sequence_parallel import AllRankShardQTokenSplit

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - triton always present with torch GPU.
    triton = None
    tl = None

try:
    import torch.distributed._symmetric_memory as torch_symm_mem
except Exception:  # pragma: no cover - depends on local PyTorch build.
    torch_symm_mem = None

try:
    from cuda.bindings import runtime as crt
except Exception:  # pragma: no cover - optional fast copy path.
    crt = None

if triton is not None:

    @triton.jit
    def _a2a_permute_kernel(
        src_ptr,
        src_off_ptr,  # int64[n_rows] element offset into src per row
        dst_addr_ptr,  # int64[n_rows] absolute byte addr of dst row start
        width,
        BLOCK_W: tl.constexpr,
    ):
        row = tl.program_id(0)
        cb = tl.program_id(1)
        col = cb * BLOCK_W + tl.arange(0, BLOCK_W)
        mask = col < width
        soff = tl.load(src_off_ptr + row)
        daddr = tl.load(dst_addr_ptr + row)
        dptr = daddr.to(tl.pointer_type(src_ptr.dtype.element_ty))
        v = tl.load(src_ptr + soff + col, mask=mask)
        tl.store(dptr + col, v, mask=mask)


def sm_a2a_available() -> bool:
    return triton is not None


def build_sm_indices(
    descriptors,
    peer_bufs,
    *,
    elem_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Per-row (src element offset, dst absolute byte address) for a permute.

    Works for both transpose and restore: each descriptor already carries the
    starting offsets and pitches for its direction.
    """
    src_offs: list[torch.Tensor] = []
    dst_addrs: list[torch.Tensor] = []
    width = 0
    for d in descriptors:
        if d.length <= 0 or d.width_elems == 0:
            continue
        width = d.width_elems
        rows = torch.arange(d.length, device=device, dtype=torch.int64)
        src_offs.append(d.src_offset_elems + rows * d.src_pitch_elems)
        base = int(peer_bufs[d.peer_rank].data_ptr())
        dst_elems = d.dst_offset_elems + rows * d.dst_pitch_elems
        dst_addrs.append(base + dst_elems * elem_size)
    if not src_offs:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty, 0
    return torch.cat(src_offs).contiguous(), torch.cat(dst_addrs).contiguous(), width


def launch_sm_a2a(
    src_flat: torch.Tensor,
    src_off: torch.Tensor,
    dst_addr: torch.Tensor,
    width: int,
    *,
    block_w: int = 512,
) -> None:
    n_rows = src_off.numel()
    if n_rows == 0 or width == 0:
        return
    grid = (n_rows, triton.cdiv(width, block_w))
    _a2a_permute_kernel[grid](src_flat, src_off, dst_addr, width, BLOCK_W=block_w)


@dataclass(frozen=True)
class ShardQA2ACopyDescriptor:
    peer_rank: int
    src_offset_elems: int
    dst_offset_elems: int
    length: int
    width_elems: int
    src_pitch_elems: int
    dst_pitch_elems: int


@dataclass
class _SymmetricShardQBuffer:
    storage: torch.Tensor
    tensor: torch.Tensor
    handle: Any
    capacity: int


_SHARDQ_WS: dict[tuple, torch.Tensor] = {}


def _shardq_ws(
    name: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Persistent scratch buffer reused across layers (one per name/device).

    The shardq transpose/restore allocate identically-shaped send/recv buffers
    on every layer; caching them removes ~30x allocation churn per forward.
    Reallocates only when the requested shape/dtype grows or changes.
    """
    key = (name, device, dtype)
    t = _SHARDQ_WS.get(key)
    if t is None or tuple(t.shape) != tuple(shape):
        t = torch.empty(shape, dtype=dtype, device=device)
        _SHARDQ_WS[key] = t
    return t


_SYMM_SHARDQ_WS: dict[tuple, _SymmetricShardQBuffer] = {}
_PEER_VIEW_CACHE: dict[tuple, list[torch.Tensor]] = {}


def _shardq_fused_a2a_available(tensor: torch.Tensor, group: GroupCoordinator) -> bool:
    # The fused path uses cuda-python cudaMemcpy2DAsync exclusively (no torch-op
    # copy fallback); require it, else defer to the NCCL reference path.
    if not tensor.is_cuda or torch_symm_mem is None or crt is None:
        return False
    device_group = getattr(group, "device_group", None)
    return getattr(device_group, "group_name", None) is not None


def _get_symm_group_name(group: GroupCoordinator) -> str:
    group_name = getattr(group.device_group, "group_name", None)
    if group_name is None:
        raise RuntimeError("symmetric memory rendezvous requires group_name")
    return str(group_name)


def _validate_shardq_a2a_rows(tensor: torch.Tensor, name: str) -> None:
    if tensor.stride(-1) != 1 or tensor.stride(-2) != tensor.shape[-1]:
        raise ValueError(
            f"{name} must store each local head row contiguously, "
            f"got shape={tuple(tensor.shape)} stride={tuple(tensor.stride())}"
        )


def _all_rank_splits_signature(
    all_rank_splits: AllRankShardQTokenSplit,
) -> tuple:
    return (
        tuple(int(count) for count in all_rank_splits.counts),
        tuple(
            tuple((token_range.start, token_range.end) for token_range in ranges)
            for ranges in all_rank_splits.ranges
        ),
    )


def _get_forward_descriptor_cache() -> (
    dict[tuple, tuple[ShardQA2ACopyDescriptor, ...]] | None
):
    try:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
    except Exception:
        return None

    cache = getattr(forward_context, "_dsv4_shardq_fused_a2a_desc_cache", None)
    if cache is None:
        cache = {}
        forward_context._dsv4_shardq_fused_a2a_desc_cache = cache  # type: ignore[attr-defined]
    return cache


def _get_fused_a2a_descriptors(
    role: str,
    all_rank_splits: AllRankShardQTokenSplit,
    *,
    rank: int,
    world_size: int,
    local_heads: int,
    full_heads: int,
    head_dim: int,
    src_pitch_elems: int,
    dst_pitch_elems: int,
) -> tuple[ShardQA2ACopyDescriptor, ...]:
    cache_key = (
        role,
        rank,
        world_size,
        local_heads,
        full_heads,
        head_dim,
        src_pitch_elems,
        dst_pitch_elems,
        _all_rank_splits_signature(all_rank_splits),
    )
    cache = _get_forward_descriptor_cache()
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    width_elems = local_heads * head_dim
    descriptors: list[ShardQA2ACopyDescriptor] = []
    if role == "transpose":
        for peer_rank, rank_ranges in enumerate(all_rank_splits.ranges):
            lt0 = 0
            for token_range in rank_ranges:
                length = token_range.length
                if length > 0:
                    descriptors.append(
                        ShardQA2ACopyDescriptor(
                            peer_rank=peer_rank,
                            src_offset_elems=token_range.start * src_pitch_elems,
                            dst_offset_elems=(
                                lt0 * dst_pitch_elems + rank * width_elems
                            ),
                            length=length,
                            width_elems=width_elems,
                            src_pitch_elems=src_pitch_elems,
                            dst_pitch_elems=dst_pitch_elems,
                        )
                    )
                lt0 += length
    elif role == "restore":
        lt0 = 0
        for token_range in all_rank_splits.ranges[rank]:
            length = token_range.length
            if length > 0:
                for peer_rank in range(world_size):
                    descriptors.append(
                        ShardQA2ACopyDescriptor(
                            peer_rank=peer_rank,
                            src_offset_elems=(
                                lt0 * src_pitch_elems + peer_rank * width_elems
                            ),
                            dst_offset_elems=token_range.start * dst_pitch_elems,
                            length=length,
                            width_elems=width_elems,
                            src_pitch_elems=src_pitch_elems,
                            dst_pitch_elems=dst_pitch_elems,
                        )
                    )
            lt0 += length
    else:
        raise ValueError(f"unknown shardq fused A2A descriptor role: {role}")

    result = tuple(descriptors)
    if cache is not None:
        cache[cache_key] = result
    return result


def _symmetric_shardq_ws(
    group_name: str,
    role: str,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[object, torch.Tensor]:
    if torch_symm_mem is None:
        raise RuntimeError("torch.distributed._symmetric_memory is unavailable")

    capacity, heads, head_dim = (int(dim) for dim in shape)
    base_key = (group_name, role, heads, head_dim, dtype, device)
    selected: _SymmetricShardQBuffer | None = None
    for key, buffer in _SYMM_SHARDQ_WS.items():
        if key[:-1] != base_key or buffer.capacity < capacity:
            continue
        if selected is None or buffer.capacity < selected.capacity:
            selected = buffer

    if selected is None:
        storage_elems = max(1, capacity * heads * head_dim)
        storage = torch_symm_mem.empty(storage_elems, dtype=dtype, device=device)
        tensor = storage[: capacity * heads * head_dim].view(
            capacity,
            heads,
            head_dim,
        )
        handle = torch_symm_mem.rendezvous(storage, group_name)
        handle.barrier()
        selected = _SymmetricShardQBuffer(
            storage=storage,
            tensor=tensor,
            handle=handle,
            capacity=capacity,
        )
        _SYMM_SHARDQ_WS[(*base_key, capacity)] = selected

    return selected.handle, selected.tensor[:capacity]


def _handle_get_buffer(
    handle: Any,
    peer_rank: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    storage_offset: int,
) -> torch.Tensor:
    try:
        return handle.get_buffer(peer_rank, shape, dtype, storage_offset)
    except TypeError:
        return handle.get_buffer(
            peer_rank,
            shape,
            dtype=dtype,
            storage_offset=storage_offset,
        )


def _get_peer_buffers(
    handle: Any,
    tensor: torch.Tensor,
    world_size: int,
) -> list[torch.Tensor]:
    shape = tuple(int(dim) for dim in tensor.shape)
    key = (id(handle), tensor.data_ptr(), shape, tensor.dtype)
    cached = _PEER_VIEW_CACHE.get(key)
    if cached is not None:
        return cached

    peer_buffers: list[torch.Tensor] = []
    for peer_rank in range(world_size):
        peer = _handle_get_buffer(handle, peer_rank, shape, tensor.dtype, 0)
        if tuple(peer.shape) != shape:
            peer = peer.view(shape)
        peer_buffers.append(peer)
    _PEER_VIEW_CACHE[key] = peer_buffers
    return peer_buffers


def _cuda_d2d_kind() -> object:
    assert crt is not None
    kind = getattr(crt, "cudaMemcpyDeviceToDevice", None)
    if kind is not None:
        return kind
    return crt.cudaMemcpyKind.cudaMemcpyDeviceToDevice


def _cuda_check(result: Any, call: str) -> None:
    if result is None:
        return
    status = result[0] if isinstance(result, tuple) else result
    code = int(status.value) if hasattr(status, "value") else int(status)
    if code != 0:
        raise RuntimeError(f"{call} failed with CUDA status {status}")


def _copy_block_cuda(
    src: torch.Tensor,
    dst: torch.Tensor,
    desc: ShardQA2ACopyDescriptor,
    *,
    stream_handle: int,
) -> None:
    assert crt is not None
    elem_size = src.element_size()
    result = crt.cudaMemcpy2DAsync(
        dst.data_ptr() + desc.dst_offset_elems * elem_size,
        desc.dst_pitch_elems * elem_size,
        src.data_ptr() + desc.src_offset_elems * elem_size,
        desc.src_pitch_elems * elem_size,
        desc.width_elems * elem_size,
        desc.length,
        _cuda_d2d_kind(),
        stream_handle,
    )
    _cuda_check(result, "cudaMemcpy2DAsync")


def _copy_block(
    src: torch.Tensor,
    dst: torch.Tensor,
    desc: ShardQA2ACopyDescriptor,
    *,
    stream_handle: int,
) -> None:
    if desc.length == 0 or desc.width_elems == 0:
        return
    # cuda-python only: the availability gate guarantees crt is not None here.
    if crt is None:
        raise RuntimeError(
            "fused shardq A2A requires cuda-python (cuda.bindings.runtime); "
            "the NCCL reference path should have been selected instead."
        )
    _copy_block_cuda(src, dst, desc, stream_handle=stream_handle)


def _barrier_after_shardq_copies(handle: Any, stream: torch.cuda.Stream) -> None:
    # The strided peer copies and handle.barrier() are enqueued on the same
    # (current) stream, so the device-side barrier is already ordered after the
    # copies complete on the GPU. A host-blocking stream.synchronize() here is
    # redundant and, in eager mode, stalled the CPU ~25 ms/forward (the dominant
    # cudaStreamSynchronize cost that erased the fusion's GPU-work win).
    del stream
    handle.barrier()


def _barrier_before_shardq_copies(handle: Any, stream: torch.cuda.Stream) -> None:
    # WAR hazard: the symmetric workspace is reused across layers. Before
    # scattering into peers buffers, wait until all ranks finished reading the
    # previous exchange so a fast rank cannot clobber a slow peers buffer
    # mid-read. Enqueued on the current stream (after the prior read), so it
    # passes only once every ranks prior consumption has completed.
    del stream
    handle.barrier()


def _get_sm_index_cache() -> dict | None:
    try:
        from vllm.forward_context import get_forward_context

        fctx = get_forward_context()
    except Exception:
        return None
    cache = getattr(fctx, "_dsv4_shardq_sm_idx_cache", None)
    if cache is None:
        cache = {}
        fctx._dsv4_shardq_sm_idx_cache = cache  # type: ignore[attr-defined]
    return cache


def _get_sm_indices(
    src: torch.Tensor,
    peer_bufs: list[torch.Tensor],
    descriptors: tuple[ShardQA2ACopyDescriptor, ...],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    key = (id(descriptors), tuple(int(b.data_ptr()) for b in peer_bufs))
    cache = _get_sm_index_cache()
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    result = build_sm_indices(
        descriptors,
        peer_bufs,
        elem_size=src.element_size(),
        device=src.device,
    )
    if cache is not None:
        cache[key] = result
    return result


def _issue_shardq_copies(
    src: torch.Tensor,
    peer_bufs: list[torch.Tensor],
    descriptors: tuple[ShardQA2ACopyDescriptor, ...],
    stream: torch.cuda.Stream,
) -> None:
    """Issue the per-peer strided P2P copies on the selected fused backend."""
    backend = envs.VLLM_DSV4_TP_SHARDQ_A2A
    if backend == "sm" and sm_a2a_available() and src.is_contiguous():
        # One SM kernel issues all peers' strided NVLink stores. The barrier is
        # enqueued by the caller on the same (current) stream afterwards, so it
        # stays ordered after these stores.
        src_off, dst_addr, width = _get_sm_indices(src, peer_bufs, descriptors)
        launch_sm_a2a(src.view(-1), src_off, dst_addr, width, block_w=512)
        return
    sh = stream.cuda_stream
    for desc in descriptors:
        _copy_block(src, peer_bufs[desc.peer_rank], desc, stream_handle=sh)


def fused_transpose_to_shardq(
    q: torch.Tensor,
    all_rank_splits: AllRankShardQTokenSplit,
    group: GroupCoordinator,
) -> torch.Tensor:
    """Push Q shards directly into local-token/full-head shardq layout."""
    world_size = group.world_size
    if world_size == 1:
        return q

    if q.dim() != 3:
        raise ValueError(f"q must be [tokens, heads, dim], got {q.shape}")
    num_tokens, local_heads, head_dim = q.shape
    if sum(all_rank_splits.counts) != num_tokens:
        raise ValueError(
            "shardq split token counts must add up to q.shape[0], "
            f"got {sum(all_rank_splits.counts)} and {num_tokens}"
        )
    _validate_shardq_a2a_rows(q, "q")

    rank = group.rank_in_group
    full_heads = world_size * local_heads
    max_local_tokens = max(all_rank_splits.counts) if all_rank_splits.counts else 0
    local_count = all_rank_splits.counts[rank]
    handle, shardq_buf = _symmetric_shardq_ws(
        _get_symm_group_name(group),
        "transpose",
        (max_local_tokens, full_heads, head_dim),
        q.dtype,
        q.device,
    )
    descriptors = _get_fused_a2a_descriptors(
        "transpose",
        all_rank_splits,
        rank=rank,
        world_size=world_size,
        local_heads=local_heads,
        full_heads=full_heads,
        head_dim=head_dim,
        src_pitch_elems=q.stride(0),
        dst_pitch_elems=full_heads * head_dim,
    )
    peer_shardq_bufs = _get_peer_buffers(handle, shardq_buf, world_size)
    stream = torch.cuda.current_stream(q.device)
    _barrier_before_shardq_copies(handle, stream)
    _issue_shardq_copies(q, peer_shardq_bufs, descriptors, stream)
    _barrier_after_shardq_copies(handle, stream)
    return shardq_buf[:local_count]


def fused_restore_from_shardq(
    output: torch.Tensor,
    all_rank_splits: AllRankShardQTokenSplit,
    group: GroupCoordinator,
) -> torch.Tensor:
    """Push shardq output head blocks back into full-token/local-head layout."""
    world_size = group.world_size
    if world_size == 1:
        return output

    if output.dim() != 3:
        raise ValueError(f"output must be [tokens, heads, dim], got {output.shape}")
    rank = group.rank_in_group
    local_count = all_rank_splits.counts[rank]
    if output.shape[0] != local_count:
        raise ValueError(
            "output token count must match current shardq split, "
            f"got {output.shape[0]} and {local_count}"
        )
    if output.shape[1] % world_size != 0:
        raise ValueError(
            f"output head count {output.shape[1]} must be divisible by {world_size}"
        )
    _validate_shardq_a2a_rows(output, "output")

    local_heads = output.shape[1] // world_size
    head_dim = output.shape[2]
    num_tokens = sum(all_rank_splits.counts)
    handle, restored = _symmetric_shardq_ws(
        _get_symm_group_name(group),
        "restore",
        (num_tokens, local_heads, head_dim),
        output.dtype,
        output.device,
    )
    descriptors = _get_fused_a2a_descriptors(
        "restore",
        all_rank_splits,
        rank=rank,
        world_size=world_size,
        local_heads=local_heads,
        full_heads=world_size * local_heads,
        head_dim=head_dim,
        src_pitch_elems=output.stride(0),
        dst_pitch_elems=local_heads * head_dim,
    )
    peer_outputs = _get_peer_buffers(handle, restored, world_size)
    stream = torch.cuda.current_stream(output.device)
    _barrier_before_shardq_copies(handle, stream)
    _issue_shardq_copies(output, peer_outputs, descriptors, stream)
    _barrier_after_shardq_copies(handle, stream)
    return restored[:num_tokens]


def transpose_heads_to_shardq(
    q: torch.Tensor,
    all_rank_splits: AllRankShardQTokenSplit,
    group: GroupCoordinator,
) -> torch.Tensor:
    """Transpose full-token/local-head Q into local-token/full-head Q.

    Every TP rank starts with all token rows for its local head shard. The
    shardq sequence-CP owner of a token needs all head shards for that token, so
    each rank sends its local-head rows for rank ``r``'s token split to rank
    ``r``. The returned tensor is ordered by the current rank's
    ``local_to_global`` rows and has the TP head shards concatenated in rank
    order.
    """
    world_size = group.world_size
    if world_size == 1:
        return q

    backend = envs.VLLM_DSV4_TP_SHARDQ_A2A
    if backend != "nccl" and _shardq_fused_a2a_available(
        q,
        group,
    ):
        return fused_transpose_to_shardq(q, all_rank_splits, group)

    if q.dim() != 3:
        raise ValueError(f"q must be [tokens, heads, dim], got {q.shape}")
    num_tokens = q.shape[0]
    if sum(all_rank_splits.counts) != num_tokens:
        raise ValueError(
            "shardq split token counts must add up to q.shape[0], "
            f"got {sum(all_rank_splits.counts)} and {num_tokens}"
        )

    # The per-rank token ownership is a handful of contiguous TokenRanges, so we
    # pack the rank-grouped send buffer with contiguous slice copies instead of
    # an index_select gather + cat (no index tensors, no temporaries).
    H, D = q.shape[1], q.shape[2]
    send = _shardq_ws("t_send", (num_tokens, H, D), q.dtype, q.device)
    off = 0
    for rank_ranges in all_rank_splits.ranges:
        for r in rank_ranges:
            length = r.end - r.start
            send[off : off + length].copy_(q[r.start : r.end])
            off += length
    local_count = all_rank_splits.counts[group.rank_in_group]
    recv = _shardq_ws("t_recv", (local_count * world_size, H, D), q.dtype, q.device)
    dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=[local_count] * world_size,
        input_split_sizes=all_rank_splits.counts,
        group=group.device_group,
    )
    return (
        recv.view(world_size, local_count, H, D)
        .permute(1, 0, 2, 3)
        .reshape(local_count, world_size * H, D)
        .contiguous()
    )


def restore_shardq_to_heads(
    output: torch.Tensor,
    all_rank_splits: AllRankShardQTokenSplit,
    group: GroupCoordinator,
) -> torch.Tensor:
    """Restore local-token/full-head output to full-token/local-head layout."""
    world_size = group.world_size
    if world_size == 1:
        return output

    backend = envs.VLLM_DSV4_TP_SHARDQ_A2A
    if backend != "nccl" and _shardq_fused_a2a_available(
        output,
        group,
    ):
        return fused_restore_from_shardq(output, all_rank_splits, group)

    if output.dim() != 3:
        raise ValueError(f"output must be [tokens, heads, dim], got {output.shape}")
    local_count = all_rank_splits.counts[group.rank_in_group]
    if output.shape[0] != local_count:
        raise ValueError(
            "output token count must match current shardq split, "
            f"got {output.shape[0]} and {local_count}"
        )
    if output.shape[1] % world_size != 0:
        raise ValueError(
            f"output head count {output.shape[1]} must be divisible by {world_size}"
        )
    local_heads = output.shape[1] // world_size
    D = output.shape[2]
    num_tokens = sum(all_rank_splits.counts)
    head_chunks = output[:, : world_size * local_heads, :].view(
        local_count, world_size, local_heads, D
    )
    # Write the head-interleave transpose straight into a persistent send buffer
    # (one copy of a non-contiguous view, same as before but reused).
    send = _shardq_ws(
        "r_send",
        (world_size * local_count, local_heads, D),
        output.dtype,
        output.device,
    )
    send.view(world_size, local_count, local_heads, D).copy_(
        head_chunks.permute(1, 0, 2, 3)
    )

    recv = _shardq_ws(
        "r_recv", (num_tokens, local_heads, D), output.dtype, output.device
    )
    dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=all_rank_splits.counts,
        input_split_sizes=[local_count] * world_size,
        group=group.device_group,
    )

    # Scatter rank-grouped recv back to global token order via contiguous slice
    # writes (ranges are contiguous spans), not an index_copy_ scatter.
    restored = torch.empty(
        (num_tokens, local_heads, D), dtype=output.dtype, device=output.device
    )
    off = 0
    for rank_ranges in all_rank_splits.ranges:
        for r in rank_ranges:
            length = r.end - r.start
            restored[r.start : r.end].copy_(recv[off : off + length])
            off += length
    return restored


def gather_full_attn_sink(
    local_attn_sink: torch.Tensor,
    n_local_heads: int,
    padded_heads: int,
    group: GroupCoordinator,
) -> torch.Tensor:
    """Gather local sink values into the full-head order used after Q transpose."""
    local = local_attn_sink[:n_local_heads].contiguous()
    full = local if group.world_size == 1 else group.all_gather(local, dim=0)
    if full.shape[0] < padded_heads:
        return F.pad(full, (0, padded_heads - full.shape[0]), value=-float("inf"))
    return full[:padded_heads]
