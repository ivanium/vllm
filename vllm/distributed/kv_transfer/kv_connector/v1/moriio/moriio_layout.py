# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Mapping
from typing import NamedTuple

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    GroupGeometry,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


class LayerTransferGeometry(NamedTuple):
    """Row-level byte geometry of one layer's kernel-block view [K, H, N, C]."""

    num_rows: int  # K: rows of the group's row grid covered by the view
    tokens_per_row: int  # tokens one kernel row holds (the kernel block size)
    row_len: int  # bytes one kernel row of this layer spans
    row_stride: int  # bytes between consecutive rows (the group's row_bytes)


def expand_block_kernel_rows(
    group: GroupGeometry, num_blocks: int, block_ids: list[int]
) -> list[int]:
    """Kernel row ids of the group's manager blocks, in order.

    Block ``b`` expands to rows ``b * R`` to ``b * R + R - 1`` (``R`` is
    ``kernel_rows_per_block``). ``num_blocks`` is the owning pool's manager
    block count (per-deployment, so it differs across peers) and bounds the
    accepted block ids.
    """
    if not block_ids:
        return []
    invalid = next((b for b in block_ids if not 0 <= b < num_blocks), None)
    if invalid is not None:
        raise ValueError(f"block id {invalid} is outside [0, {num_blocks})")
    return group.rows(block_ids).tolist()


class MambaTransferGeometry(NamedTuple):
    """Geometry of a hybrid (mamba/KDA) layer's recurrent state.

    A KDA layer's packed page is exposed as ``conv_state`` and ``ssm_state``
    views. Each is slot-strided, so a slot's bytes live at
    ``slot * slot_stride * element_size`` and span ``slot_bytes``. The two
    tensors are registered as two separate MoRIIO regions per layer.
    """

    conv_slot_stride: int
    ssm_slot_stride: int
    ssm_slot_bytes: int
    conv_element_size: int
    ssm_element_size: int
    conv_region_len: int
    ssm_region_len: int


def get_mamba_transfer_geometry(
    conv: torch.Tensor,
    ssm: torch.Tensor,
) -> MambaTransferGeometry:
    """Derive the slot-strided geometry of a KDA layer's conv + ssm state.

    ``conv``/``ssm`` are zero-copy views over the layer's packed page. Slot byte
    offset is ``slot * slot_stride * element_size`` and each
    slot spans ``subtensor[0].numel() * element_size`` bytes; the whole tensor
    is registered as one region (``*_region_len`` bytes).
    """
    if conv.ndim == 0 or ssm.ndim == 0 or conv.shape[0] == 0 or ssm.shape[0] == 0:
        raise ValueError("KDA conv/ssm caches must contain at least one slot")
    if conv.shape[0] != ssm.shape[0]:
        raise ValueError(
            f"KDA conv/ssm slot count mismatch: {conv.shape[0]} != {ssm.shape[0]}"
        )
    conv_slot_bytes = conv[0].numel() * conv.element_size()
    ssm_slot_bytes = ssm[0].numel() * ssm.element_size()
    conv_available_bytes = (
        conv.untyped_storage().nbytes() - conv.storage_offset() * conv.element_size()
    )
    ssm_available_bytes = (
        ssm.untyped_storage().nbytes() - ssm.storage_offset() * ssm.element_size()
    )
    if conv.shape[0]:
        conv_required_bytes = (conv.shape[0] - 1) * conv.stride(
            0
        ) * conv.element_size() + conv_slot_bytes
        ssm_required_bytes = (ssm.shape[0] - 1) * ssm.stride(
            0
        ) * ssm.element_size() + ssm_slot_bytes
        if conv_required_bytes > conv_available_bytes:
            raise ValueError("KDA conv slots exceed the registered storage extent")
        if ssm_required_bytes > ssm_available_bytes:
            raise ValueError("KDA ssm slots exceed the registered storage extent")

    return MambaTransferGeometry(
        conv_slot_stride=conv.stride(0),
        ssm_slot_stride=ssm.stride(0),
        ssm_slot_bytes=ssm_slot_bytes,
        conv_element_size=conv.element_size(),
        ssm_element_size=ssm.element_size(),
        # Registered region must span the full slot-strided extent: stride(0)
        # may exceed the per-slot element count, so numel() would under-size the
        # region and truncate the highest-slot transfers. shape[0]*stride(0)
        # covers every byte a ``slot * slot_stride * elem`` offset can address.
        # Cap at the bytes remaining from each view's storage offset. A packed
        # conv+ssm page places ssm at a non-zero intra-buffer offset,
        # so do not register past the end of the shared backing allocation.
        conv_region_len=min(
            conv.shape[0] * conv.stride(0) * conv.element_size(),
            conv_available_bytes,
        ),
        ssm_region_len=min(
            ssm.shape[0] * ssm.stride(0) * ssm.element_size(),
            ssm_available_bytes,
        ),
    )


def build_layer_to_spec(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    layer_to_spec: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        layer_specs = group.layer_kv_cache_specs or {}
        for layer_name in group.layer_names:
            layer_to_spec[layer_name] = layer_specs.get(layer_name, group.kv_cache_spec)
    return layer_to_spec


def is_mla_cache_layer(
    layer_to_spec: Mapping[str, KVCacheSpec], layer_name: str
) -> bool:
    try:
        spec = layer_to_spec[layer_name]
    except KeyError as e:
        raise ValueError(f"Missing KV cache spec for layer {layer_name}") from e
    return isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec))


def get_layer_transfer_geometry(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
) -> LayerTransferGeometry:
    """Row geometry of a layer's large-page kernel-block view.

    Every attention layer view is the standardized ``[K, H, N, C]``: K kernel
    rows of the group's row grid (one row per kernel block), strided by the
    group's row bytes. Manager block ids do not address it directly; expand
    them with ``expand_block_kernel_rows``.
    """
    spec = layer_to_spec.get(layer_name)
    if spec is None:
        raise ValueError(f"Missing KV cache spec for layer {layer_name}")
    shape = kv_cache.shape
    stride = kv_cache.stride()
    element_size = kv_cache.element_size()
    if (
        isinstance(spec, AttentionSpec)
        and len(shape) == 4
        and shape[1] == spec.num_heads
        and shape[3] * element_size == spec.state_content_size_bytes
    ):
        return LayerTransferGeometry(
            num_rows=shape[0],
            tokens_per_row=shape[2],
            row_len=shape[1] * shape[2] * shape[3] * element_size,
            row_stride=stride[0] * element_size,
        )
    raise ValueError(
        f"Unsupported MoRIIO KV cache shape for layer {layer_name}: "
        f"{tuple(shape)}; large-page layer views must be [K, H, N, C]"
    )


def kda_conv_ssm(
    kv_cache: torch.Tensor,
    spec: MambaSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return zero-copy conv and SSM views over a packed two-state page."""
    from math import prod

    from vllm.utils.torch_utils import get_dtype_size

    if len(spec.shapes) != 2 or len(spec.dtypes) != 2:
        raise ValueError(
            "MoRIIO KDA transfer requires exactly two states (conv and SSM)"
        )
    if kv_cache.element_size() != 1:
        raise ValueError("MoRIIO KDA packed cache must use a byte-sized dtype")
    if kv_cache.ndim != 4 or kv_cache.shape[1:3] != (1, 1) or kv_cache.stride(3) != 1:
        raise ValueError(
            "MoRIIO KDA packed cache must have shape [B, 1, 1, C] with "
            "a contiguous byte dimension"
        )
    # Mirror MambaBase.bind_kv_cache: blocks may be strided within a shared
    # multi-layer allocation, while every individual byte page is contiguous.
    pages = kv_cache.squeeze(dim=(1, 2))
    states: list[torch.Tensor] = []
    offset = 0
    for shp, dt in zip(spec.shapes, spec.dtypes, strict=True):
        nbytes = prod(shp) * get_dtype_size(dt)
        if offset + nbytes > pages.shape[1]:
            raise ValueError(
                "MoRIIO KDA state layout exceeds the packed cache page size"
            )
        state = pages[:, offset : offset + nbytes].view(dt)
        states.append(state.view(-1, *shp))
        offset += nbytes
    return states[0], states[1]


def iter_layer_registration_regions(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
) -> list[tuple[torch.Tensor, int]]:
    spec = layer_to_spec[layer_name]
    if isinstance(spec, MambaSpec):
        # KDA layer: register the conv and ssm tensors as two whole-tensor
        # regions (the conv 3-subprojection split is expressed as transfer
        # offset triples, not extra registrations).
        conv, ssm = kda_conv_ssm(kv_cache, spec)
        geom = get_mamba_transfer_geometry(conv, ssm)
        return [(conv, geom.conv_region_len), (ssm, geom.ssm_region_len)]
    geometry = get_layer_transfer_geometry(layer_name, kv_cache, layer_to_spec)
    # The strided span of every kernel row, ending at the last row's bytes so
    # the region stays inside the shared pool allocation.
    region_len = (geometry.num_rows - 1) * geometry.row_stride + geometry.row_len
    return [(kv_cache, region_len)]


def merge_contiguous_offsets(
    offsets_local: list[int],
    offsets_remote: list[int],
    sizes: list[int],
) -> tuple[list[int], list[int], list[int]]:
    if not offsets_local:
        return [], [], []
    if not (len(offsets_local) == len(offsets_remote) == len(sizes)):
        raise ValueError("Input list lengths mismatch")

    rows = sorted(zip(offsets_local, offsets_remote, sizes), key=lambda row: row[0])
    merged: list[list[int]] = []
    for local, remote, size in rows:
        if (
            merged
            and local == merged[-1][0] + merged[-1][2]
            and remote == merged[-1][1] + merged[-1][2]
        ):
            merged[-1][2] += size
        else:
            merged.append([local, remote, size])

    return (
        [row[0] for row in merged],
        [row[1] for row in merged],
        [row[2] for row in merged],
    )


def compute_block_transfer_offsets(
    geometry: LayerTransferGeometry,
    local_group: GroupGeometry,
    local_num_blocks: int,
    remote_group: GroupGeometry,
    remote_num_blocks: int,
    local_block_ids: list[int],
    remote_block_ids: list[int],
    merge_fn: Callable[
        [list[int], list[int], list[int]], tuple[list[int], list[int], list[int]]
    ] = merge_contiguous_offsets,
) -> tuple[list[int], list[int], list[int]]:
    """Byte offsets moving whole manager blocks of one layer between peers.

    Manager block ids expand to kernel rows with each side's own group
    geometry and pool size (a peer's row bytes and block count can
    differ). Row ``k`` of the layer's registered region starts at
    ``k * row stride`` and spans ``geometry.row_len`` bytes, so both sides
    must split manager blocks into the same number of kernel rows for the
    rows to pair 1:1.
    """
    if local_group.kernel_rows_per_block != remote_group.kernel_rows_per_block:
        raise ValueError(
            "MoRIIO peers split manager blocks into a different number of "
            f"kernel rows: {local_group.kernel_rows_per_block} != "
            f"{remote_group.kernel_rows_per_block}"
        )
    # A shorter/empty local list transfers only what decode allocated (READ-mode
    # partial or full-prefix-hit); a longer local list is a genuine bug.
    if len(local_block_ids) > len(remote_block_ids):
        raise ValueError(
            "local_block_ids longer than remote_block_ids: "
            f"{len(local_block_ids)} > {len(remote_block_ids)}"
        )
    remote_block_ids = remote_block_ids[: len(local_block_ids)]
    local_rows = expand_block_kernel_rows(
        local_group, local_num_blocks, local_block_ids
    )
    remote_rows = expand_block_kernel_rows(
        remote_group, remote_num_blocks, remote_block_ids
    )
    offset_local = [row * geometry.row_stride for row in local_rows]
    offset_remote = [row * remote_group.row_bytes for row in remote_rows]
    sizes = [geometry.row_len] * len(local_rows)
    return merge_fn(offset_local, offset_remote, sizes)


class MambaOffsetTemplate(NamedTuple):
    """Slot-independent conv+ssm offset decomposition for a KDA layer.

    Homogeneous-TP geometry (conv/ssm slot strides, conv sub-projection
    offsets, ssm per-slot size) is identical across requests and across the
    homogeneous GDN layers, so it is computed once and reused; only the
    per-request slot bases vary (see ``apply_mamba_offset_template``).
    """

    num_slots: int
    conv_slot_stride_bytes: int
    ssm_slot_stride_bytes: int
    ssm_read_bytes: int
    conv_subprojs: tuple[tuple[int, int, int], ...]


def build_mamba_offset_template(
    conv: torch.Tensor,
    ssm: torch.Tensor,
    split_info: MambaConvSplitInfo,
    tp_ratio: int,
) -> MambaOffsetTemplate:
    """Precompute the slot-independent conv+ssm offset decomposition.

    Only homogeneous TP (``tp_ratio == 1``) is supported; heterogeneous TP is
    gated with ``NotImplementedError`` (see the design doc). The returned
    template is a pure function of the layer geometry and ``split_info``, so it
    can be cached and reused across requests without recomputation.
    """
    if tp_ratio != 1:
        # Heterogeneous-TP recurrent-state transfer needs the remote page's
        # slot stride (which differs from the local page) and, for
        # P_TP > D_TP, a multi-rank gather of conv/ssm slices. Both are
        # follow-ups. The mamba path bypasses the attention KV-head validator,
        # so fail loudly here instead of silently transferring wrong bytes.
        raise NotImplementedError(
            "heterogeneous-TP mamba/KDA transfer is not supported yet "
            f"(tp_ratio={tp_ratio}); use equal prefill/decode TP for "
            "hybrid models"
        )
    geom = get_mamba_transfer_geometry(conv, ssm)
    conv_local_offsets = split_info.local_conv_offsets
    conv_remote_offsets = split_info.remote_conv_offsets(0, 1)
    ssm_read_bytes = geom.ssm_slot_bytes
    conv_subprojs = tuple(
        (loff, roff, rsz)
        for (loff, _lsz), (roff, rsz) in zip(conv_local_offsets, conv_remote_offsets)
    )
    return MambaOffsetTemplate(
        num_slots=conv.shape[0],
        conv_slot_stride_bytes=geom.conv_slot_stride * geom.conv_element_size,
        ssm_slot_stride_bytes=geom.ssm_slot_stride * geom.ssm_element_size,
        ssm_read_bytes=ssm_read_bytes,
        conv_subprojs=conv_subprojs,
    )


def apply_mamba_offset_template(
    template: MambaOffsetTemplate,
    local_slots: list[int],
    remote_slots: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Apply per-request slot bases to a cached ``MambaOffsetTemplate``.

    A "slot" here is a kernel row of the registered conv/ssm view: callers
    expand the mamba group's manager block ids with
    ``expand_block_kernel_rows`` (each side with its own group geometry)
    first. Conv sub-projection entries (all slots) come first, followed by
    one ssm entry per slot.
    """
    if len(local_slots) != len(remote_slots):
        raise ValueError(
            "local and remote mamba slot counts must match: "
            f"{len(local_slots)} != {len(remote_slots)}"
        )
    invalid_local = next(
        (slot for slot in local_slots if not 0 <= slot < template.num_slots), None
    )
    if invalid_local is not None:
        raise ValueError(
            f"local mamba slot {invalid_local} is outside [0, {template.num_slots})"
        )
    invalid_remote = next((slot for slot in remote_slots if slot < 0), None)
    if invalid_remote is not None:
        raise ValueError(f"remote mamba slot {invalid_remote} must be non-negative")
    local_offs: list[int] = []
    remote_offs: list[int] = []
    sizes: list[int] = []

    # Conv sub-projections (region 0), all slots first.
    conv_slot_stride_bytes = template.conv_slot_stride_bytes
    for ls, rs in zip(local_slots, remote_slots):
        lbase = ls * conv_slot_stride_bytes
        rbase = rs * conv_slot_stride_bytes
        for loff, roff, rsz in template.conv_subprojs:
            local_offs.append(lbase + loff)
            remote_offs.append(rbase + roff)
            sizes.append(rsz)

    # SSM temporal state (region 1), one entry per slot.
    ssm_slot_stride_bytes = template.ssm_slot_stride_bytes
    ssm_read_bytes = template.ssm_read_bytes
    for ls, rs in zip(local_slots, remote_slots):
        local_offs.append(ls * ssm_slot_stride_bytes)
        remote_offs.append(rs * ssm_slot_stride_bytes)
        sizes.append(ssm_read_bytes)

    return local_offs, remote_offs, sizes


def compute_mamba_conv_split_count(
    local_slots: list[int],
    split_info: MambaConvSplitInfo,
) -> int:
    """Number of leading conv entries in a KDA layer transfer plan.

    Entries ``[:count]`` are conv sub-projections (conv region/session);
    entries ``[count:]`` are the per-slot ssm state (ssm region/session).
    """
    return len(local_slots) * len(split_info.local_conv_offsets)


class HeadLayout(NamedTuple):
    """Byte addressing of one attention layer's KV heads in its region.

    Offsets are relative to the layer's registered region and rows are
    addressed by kernel row id (``block_stride`` is the group's row stride);
    expand manager block ids with ``expand_block_kernel_rows`` first.
    ``plane_offsets`` holds the start of the K and V planes when they are
    stored apart, or ``(0,)`` when K and V are packed into each head's
    content.
    """

    num_heads: int
    num_tokens: int
    head_bytes: int
    block_stride: int
    head_stride: int
    token_stride: int
    plane_offsets: tuple[int, ...]

    @property
    def is_dense(self) -> bool:
        """Each block plane is one contiguous run of all its heads and tokens."""
        hb = self.head_bytes
        return (
            self.head_stride == hb and self.token_stride == self.num_heads * hb
        ) or (self.token_stride == hb and self.head_stride == self.num_tokens * hb)

    def to_wire(self) -> list[int]:
        return [*self[:6], *self.plane_offsets]

    @classmethod
    def from_wire(cls, values: list[int]) -> "HeadLayout":
        heads, tokens, head_bytes, block_stride, head_stride, token_stride, *planes = (
            values
        )
        return cls(
            heads,
            tokens,
            head_bytes,
            block_stride,
            head_stride,
            token_stride,
            tuple(planes),
        )


def get_layer_head_layout(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
) -> HeadLayout:
    """Describe where each KV head of a kernel row lives, from the strides."""
    spec = layer_to_spec.get(layer_name)
    if spec is None:
        raise ValueError(f"Missing KV cache spec for layer {layer_name}")
    shape = kv_cache.shape
    stride = kv_cache.stride()
    element_size = kv_cache.element_size()
    if (
        isinstance(spec, AttentionSpec)
        and len(shape) == 4
        and shape[1] == spec.num_heads
        and shape[3] * element_size == spec.state_content_size_bytes
    ):
        return HeadLayout(
            num_heads=shape[1],
            num_tokens=shape[2],
            head_bytes=shape[3] * element_size,
            block_stride=stride[0] * element_size,
            head_stride=stride[1] * element_size,
            token_stride=stride[2] * element_size,
            plane_offsets=(0,),
        )
    raise ValueError(
        f"MoRIIO cannot slice KV heads of layer {layer_name} with cache shape "
        f"{tuple(shape)}"
    )


def _head_runs(
    layout: HeadLayout, first_head: int, num_heads: int
) -> tuple[str, list[tuple[int, int]]]:
    """Coarsest (offset in block plane, size) runs covering num_heads heads.

    Returns the run kind with the runs; two layouts can be paired run by run
    only when their kinds match.
    """
    hb = layout.head_bytes
    heads = range(first_head, first_head + num_heads)
    if layout.token_stride == hb and layout.head_stride == layout.num_tokens * hb:
        # Head-major with packed tokens: the heads are one contiguous run.
        return "block", [
            (first_head * layout.head_stride, num_heads * layout.head_stride)
        ]
    if layout.token_stride == hb:
        # Head-major: one run of all tokens per head.
        return "head", [(h * layout.head_stride, layout.num_tokens * hb) for h in heads]
    if layout.head_stride == hb:
        # Token-major: one run of adjacent heads per token.
        return "token", [
            (t * layout.token_stride + first_head * hb, num_heads * hb)
            for t in range(layout.num_tokens)
        ]
    return "element", _element_runs(layout, first_head, num_heads)


def _element_runs(
    layout: HeadLayout, first_head: int, num_heads: int
) -> list[tuple[int, int]]:
    """One run per (token, head), ordered token-major."""
    return [
        (t * layout.token_stride + h * layout.head_stride, layout.head_bytes)
        for t in range(layout.num_tokens)
        for h in range(first_head, first_head + num_heads)
    ]


def compute_head_slice_offsets(
    local: HeadLayout,
    remote: HeadLayout,
    local_row_ids: list[int],
    remote_row_ids: list[int],
    local_head: int,
    remote_head: int,
    num_heads: int,
    merge_fn: Callable[
        [list[int], list[int], list[int]], tuple[list[int], list[int], list[int]]
    ] = merge_contiguous_offsets,
) -> tuple[list[int], list[int], list[int]]:
    """Offsets copying ``num_heads`` heads of each local row into a remote one.

    Local heads ``[local_head, local_head + num_heads)`` land on remote heads
    ``[remote_head, remote_head + num_heads)``; every other remote head is
    left for the producer rank that holds it. Rows are kernel rows; expand
    manager block ids with ``expand_block_kernel_rows`` first.
    """
    if (
        local.num_tokens != remote.num_tokens
        or local.head_bytes != remote.head_bytes
        or len(local.plane_offsets) != len(remote.plane_offsets)
    ):
        raise ValueError(
            "MoRIIO head slicing needs the same row length, head size and K/V "
            f"planes on both sides, got local {local} and remote {remote}"
        )
    if not (
        local_head >= 0
        and local_head + num_heads <= local.num_heads
        and remote_head >= 0
        and remote_head + num_heads <= remote.num_heads
    ):
        raise ValueError(
            f"Head slice local [{local_head}, {local_head + num_heads}) / remote "
            f"[{remote_head}, {remote_head + num_heads}) is outside "
            f"{local.num_heads} / {remote.num_heads} heads"
        )
    if len(local_row_ids) > len(remote_row_ids):
        raise ValueError(
            "local_row_ids longer than remote_row_ids: "
            f"{len(local_row_ids)} > {len(remote_row_ids)}"
        )
    local_kind, local_runs = _head_runs(local, local_head, num_heads)
    remote_kind, remote_runs = _head_runs(remote, remote_head, num_heads)
    if local_kind != remote_kind:
        # The two sides order heads and tokens differently: pair single
        # (token, head) runs, which merge_fn coalesces where both are adjacent.
        local_runs = _element_runs(local, local_head, num_heads)
        remote_runs = _element_runs(remote, remote_head, num_heads)

    offset_local: list[int] = []
    offset_remote: list[int] = []
    sizes: list[int] = []
    for local_plane, remote_plane in zip(local.plane_offsets, remote.plane_offsets):
        for lb, rb in zip(local_row_ids, remote_row_ids):
            local_base = local_plane + lb * local.block_stride
            remote_base = remote_plane + rb * remote.block_stride
            for (local_off, size), (remote_off, _) in zip(local_runs, remote_runs):
                offset_local.append(local_base + local_off)
                offset_remote.append(remote_base + remote_off)
                sizes.append(size)
    return merge_fn(offset_local, offset_remote, sizes)


class MambaStateLayout(NamedTuple):
    """Byte addressing of one mamba layer's per-slot conv and SSM state."""

    conv_slot_stride: int
    ssm_slot_stride: int
    ssm_bytes: int
    conv_proj_bytes: tuple[int, ...]

    def to_wire(self) -> list[int]:
        return [*self[:3], *self.conv_proj_bytes]

    @classmethod
    def from_wire(cls, values: list[int]) -> "MambaStateLayout":
        conv_slot_stride, ssm_slot_stride, ssm_bytes, *conv_proj_bytes = values
        return cls(conv_slot_stride, ssm_slot_stride, ssm_bytes, tuple(conv_proj_bytes))


def get_mamba_state_layout(
    conv: torch.Tensor, ssm: torch.Tensor, split_info: MambaConvSplitInfo
) -> MambaStateLayout:
    geom = get_mamba_transfer_geometry(conv, ssm)
    return MambaStateLayout(
        conv_slot_stride=geom.conv_slot_stride * geom.conv_element_size,
        ssm_slot_stride=geom.ssm_slot_stride * geom.ssm_element_size,
        ssm_bytes=geom.ssm_slot_bytes,
        conv_proj_bytes=split_info.proj_bytes,
    )


def _state_slice(
    local_bytes: int,
    remote_bytes: int,
    local_slice: int,
    local_slices: int,
    remote_slice: int,
    remote_slices: int,
) -> tuple[int, int, int]:
    """(local offset, remote offset, size) of the bytes one target receives."""
    if local_bytes == remote_bytes:
        # Equal TP, or a part replicated on both sides (e.g. Mamba2 B/C with
        # fewer groups than ranks): copy it whole.
        return 0, 0, local_bytes
    if local_slices > 1 and local_bytes == local_slices * remote_bytes:
        return local_slice * remote_bytes, 0, remote_bytes
    if remote_slices > 1 and remote_bytes == remote_slices * local_bytes:
        return 0, remote_slice * local_bytes, local_bytes
    raise ValueError(
        f"MoRIIO cannot map {local_bytes} local state bytes onto {remote_bytes} "
        f"remote bytes for slice {local_slice}/{local_slices} -> "
        f"{remote_slice}/{remote_slices}"
    )


def compute_mamba_slice_offsets(
    local: MambaStateLayout,
    remote: MambaStateLayout,
    local_slots: list[int],
    remote_slots: list[int],
    local_slice: int,
    local_slices: int,
    remote_slice: int,
    remote_slices: int,
) -> tuple[list[int], list[int], list[int], int]:
    """(local, remote, sizes, n_conv) moving one target's share of the state.

    ``local_slots``/``remote_slots`` are kernel rows of the registered
    conv/ssm views; expand manager block ids with
    ``expand_block_kernel_rows`` first. Entries ``[:n_conv]`` address the conv
    region; the rest the SSM region. Each conv sub-projection and the SSM
    state are sharded across TP ranks independently, so each is sliced or
    gathered on its own.
    """
    if len(local_slots) != len(remote_slots):
        raise ValueError(
            "local and remote mamba slot counts must match: "
            f"{len(local_slots)} != {len(remote_slots)}"
        )
    if len(local.conv_proj_bytes) != len(remote.conv_proj_bytes):
        raise ValueError(
            "local and remote conv states have different sub-projections: "
            f"{local.conv_proj_bytes} vs {remote.conv_proj_bytes}"
        )
    slices = (local_slice, local_slices, remote_slice, remote_slices)
    conv_parts = []
    local_base = remote_base = 0
    for local_bytes, remote_bytes in zip(local.conv_proj_bytes, remote.conv_proj_bytes):
        local_off, remote_off, size = _state_slice(local_bytes, remote_bytes, *slices)
        conv_parts.append((local_base + local_off, remote_base + remote_off, size))
        local_base += local_bytes
        remote_base += remote_bytes
    ssm_local, ssm_remote, ssm_size = _state_slice(
        local.ssm_bytes, remote.ssm_bytes, *slices
    )

    local_offs: list[int] = []
    remote_offs: list[int] = []
    sizes: list[int] = []
    for ls, rs in zip(local_slots, remote_slots):
        for local_off, remote_off, size in conv_parts:
            local_offs.append(ls * local.conv_slot_stride + local_off)
            remote_offs.append(rs * remote.conv_slot_stride + remote_off)
            sizes.append(size)
    n_conv = len(sizes)
    for ls, rs in zip(local_slots, remote_slots):
        local_offs.append(ls * local.ssm_slot_stride + ssm_local)
        remote_offs.append(rs * remote.ssm_slot_stride + ssm_remote)
        sizes.append(ssm_size)
    return local_offs, remote_offs, sizes, n_conv
