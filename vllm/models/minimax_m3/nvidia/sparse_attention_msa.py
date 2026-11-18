# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA (SM100 family) block-sparse attention for MiniMax M3.

`MiniMaxM3SparseMSAImpl` serves BF16 and FP8 KV caches: prefill attends with
`fmha_sm100` (`build_k2q_csr` + `sparse_atten_func`), and decode uses the
CUTLASS `fmha_sm100` path by default for regular decode and speculative
verification, with Triton split-K as the fallback.

`MiniMaxM3SparseMSANvfp4Impl` serves NVFP4 KV caches, reading vLLM's packed
pages directly with only the E4M3 query: decode and prefill both run the CUTLASS
Q8KV4 kernels.
"""

import functools
import os
import weakref
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.attention import MiniMaxM3MSADecodeBackend
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn_decode,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseDecodeMetadata,
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMetadataBuilder,
    MiniMaxM3SparsePrefillMetadata,
)
from vllm.models.minimax_m3.nvidia.msa_cutlass_sparse_decode import (
    NVFP4_BLOCK_SCALE_SHIFT,
    MSACutlassDecodeMetadata,
    MSACutlassDecodePlanCache,
    msa_cutlass_sparse_decode,
    nvfp4_kv_cache_views,
    prepare_decode_metadata,
    should_prepare_decode_metadata,
    supports_cutlass_sparse_decode,
    uses_q8kv4_decode,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_dtype_size, nvfp4_kv_cache_full_dim
from vllm.v1.attention.backend import (
    AttentionLayer,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

if TYPE_CHECKING:
    from vllm.third_party.fmha_sm100.prefill_q8kv4 import PrefillWorkspace

logger = init_logger(__name__)


@functools.cache
def _q8kv4_prefill_blocker(
    num_q_heads: int, num_kv_heads: int, page_size: int, topk_blocks: int
) -> str | None:
    """Why the Q8KV4 kernel cannot prefill this layout on this device, or None."""
    from vllm.third_party.fmha_sm100.q8kv4_prefill_adapter import _plan_blocker

    return _plan_blocker(
        num_qo_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        kv_block_num=topk_blocks,
        causal=True,
        output_maxscore=False,
        device=None,
    )


def _get_msa_cutlass_min_batch_size() -> int | None:
    value = os.getenv("VLLM_MINIMAX_M3_MSA_CUTLASS_MIN_BATCH")
    return int(value) if value is not None else None


# Q heads per KV head the Q8KV4 prefill kernel serves, and the ratio it can
# serve with a padded query (the Q8KV4 decode serves both).
_Q8KV4_PREFILL_Q_HEADS_PER_KV = 16
_Q8KV4_PADDED_PREFILL_Q_HEADS_PER_KV = 8


def _q8kv4_prefill_pad_heads_enabled() -> bool:
    """VLLM_MINIMAX_M3_MSA_PREFILL_PAD_HEADS=1: on a rank with 8 Q heads per KV
    head (e.g. TP8), the NVFP4 prefill pads its query to the 16 the Q8KV4
    kernel serves."""
    return os.getenv("VLLM_MINIMAX_M3_MSA_PREFILL_PAD_HEADS", "0") == "1"


def _dequantize_query(
    query_fp8: torch.Tensor, q_scale: float, dtype: torch.dtype
) -> torch.Tensor:
    """`query_fp8 * q_scale` in `dtype` (E4M3 is exact in BF16)."""
    out = query_fp8.to(dtype)
    if q_scale != 1.0:
        out.mul_(q_scale)
    return out


class MiniMaxM3SparseMSABackend(MiniMaxM3SparseBackend):
    """MiniMax M3 backend with NVIDIA MSA-specific decode metadata."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3SparseMSAMetadataBuilder"]:
        return MiniMaxM3SparseMSAMetadataBuilder


class MiniMaxM3SparseMSANvfp4Backend(MiniMaxM3SparseMSABackend):
    """MSA backend over HND NVFP4 KV cache pages in per-head K/V slots."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["nvfp4"]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """Store each head's K and V as two slots of packed data + block scales
        (see `nvfp4_kv_cache_views`)."""
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
            return spec
        return replace(
            spec,
            num_head_slots=2 * spec.num_kv_heads,
            state_content_bytes=nvfp4_kv_cache_full_dim(spec.head_size)
            * get_dtype_size(spec.dtype),
            kv_head_major_slots=True,
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...] | None:
        # Every head's K/V slots are one contiguous run of the page; the
        # kernels take the page stride from the cache view.
        return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)


class MiniMaxM3SparseCutlassBackend(MiniMaxM3SparseMSABackend):
    """Attention-backend alias selecting CUTLASS MSA sparse decode."""

    @staticmethod
    def get_name() -> str:
        return "CUTLASS_MSA"


class MiniMaxM3SparseTritonBackend(MiniMaxM3SparseMSABackend):
    """Attention-backend alias selecting Triton MSA sparse decode."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_MSA"


@dataclass
class MiniMaxM3SparseMSADecodeMetadata(MiniMaxM3SparseDecodeMetadata):
    msa_cutlass: MSACutlassDecodeMetadata | None = None


# WeakValueDictionary: live impls keep the workspace alive; the last model
# teardown releases it instead of caching GPU memory forever.
_prefill_workspaces: weakref.WeakValueDictionary[tuple, "PrefillWorkspace"] = (
    weakref.WeakValueDictionary()
)


def _shared_prefill_workspace(
    device_index: int, topk_blocks: int, max_tokens: int, num_q_heads: int
) -> "PrefillWorkspace":
    """The one Q8KV4 prefill split workspace per process, reserved for the
    largest batch (attention impls are per layer)."""
    from vllm.third_party.fmha_sm100.prefill_q8kv4 import PrefillWorkspace

    key = (device_index, topk_blocks, max_tokens, num_q_heads)
    if (workspace := _prefill_workspaces.get(key)) is not None:
        return workspace
    workspace = PrefillWorkspace(
        device_index, block_scale_shift=NVFP4_BLOCK_SCALE_SHIFT
    )
    workspace.split_buffers(topk_blocks, max_tokens, num_q_heads)
    _prefill_workspaces[key] = workspace
    return workspace


class _PaddedPrefillBuffers:
    """The padded Q8KV4 prefill's query and output, reserved for the largest
    batch and shared by the layers of a process (they run in order on one
    stream, as the split workspace does).

    `query` is E4M3 `[max_tokens, num_kv_heads, 16, head_size]`: each KV head's
    first `num_heads_per_kv` heads take the real query, and the rest stay zero
    from the allocation on (nothing writes them). `output` is the kernel's BF16
    output over all 16 heads; only the real heads are read back.
    """

    def __init__(
        self,
        device_index: int,
        max_tokens: int,
        num_kv_heads: int,
        head_size: int,
    ) -> None:
        device = torch.device("cuda", device_index)
        self.max_tokens = max_tokens
        self.query = torch.zeros(
            (max_tokens, num_kv_heads, _Q8KV4_PREFILL_Q_HEADS_PER_KV, head_size),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.output = torch.empty(
            (max_tokens, num_kv_heads * _Q8KV4_PREFILL_Q_HEADS_PER_KV, head_size),
            dtype=torch.bfloat16,
            device=device,
        )


_padded_prefill_buffers: weakref.WeakValueDictionary[tuple, _PaddedPrefillBuffers] = (
    weakref.WeakValueDictionary()
)


def _shared_padded_prefill_buffers(
    device_index: int, max_tokens: int, num_kv_heads: int, head_size: int
) -> _PaddedPrefillBuffers:
    key = (device_index, max_tokens, num_kv_heads, head_size)
    if (buffers := _padded_prefill_buffers.get(key)) is not None:
        return buffers
    buffers = _PaddedPrefillBuffers(device_index, max_tokens, num_kv_heads, head_size)
    _padded_prefill_buffers[key] = buffers
    return buffers


def _padded_prefill_views(
    tokens: int,
    num_kv_heads: int,
    head_size: int,
    buffers: _PaddedPrefillBuffers,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The padded query `[tokens, num_kv_heads, 16, head_size]` (zero past each
    group's real heads) and the BF16 output `[tokens, num_kv_heads * 16,
    head_size]` for one prefill, from the reservation when it fits."""
    if tokens <= buffers.max_tokens:
        return buffers.query[:tokens], buffers.output[:tokens]
    # Past the reservation (not reached by the scheduler's batches).
    shape = (tokens, num_kv_heads, _Q8KV4_PREFILL_Q_HEADS_PER_KV, head_size)
    return (
        torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device),
        torch.empty(
            (tokens, num_kv_heads * _Q8KV4_PREFILL_Q_HEADS_PER_KV, head_size),
            dtype=torch.bfloat16,
            device=device,
        ),
    )


def _pad_prefill_query(query: torch.Tensor, padded: torch.Tensor) -> torch.Tensor:
    """Write `query` `[tokens, num_kv_heads * n, head_size]` into the first n
    heads of each KV head's group of `padded` `[tokens, num_kv_heads, 16,
    head_size]` and return `padded` as the contiguous `[tokens, num_kv_heads *
    16, head_size]` query the Q8KV4 prefill reads."""
    tokens, num_kv_heads, _, head_size = padded.shape
    heads_per_kv = query.shape[1] // num_kv_heads
    padded[:, :, :heads_per_kv].copy_(
        query.view(tokens, num_kv_heads, heads_per_kv, head_size)
    )
    return padded.view(tokens, -1, head_size)


def _unpad_prefill_output(padded: torch.Tensor, out: torch.Tensor) -> None:
    """Copy the real heads of the padded prefill's output `padded` `[tokens,
    num_kv_heads * 16, head_size]` into `out` `[tokens, num_kv_heads * n,
    head_size]`: the first n heads of each KV head's group."""
    tokens, num_heads, head_size = out.shape
    num_kv_heads = padded.shape[1] // _Q8KV4_PREFILL_Q_HEADS_PER_KV
    heads_per_kv = num_heads // num_kv_heads
    out.view(tokens, num_kv_heads, heads_per_kv, head_size).copy_(
        padded.view(tokens, num_kv_heads, _Q8KV4_PREFILL_Q_HEADS_PER_KV, head_size)[
            :, :, :heads_per_kv
        ]
    )


class MiniMaxM3SparseMSAMetadataBuilder(MiniMaxM3SparseMetadataBuilder):
    """Prepare MSA plans only for decode shapes supported by `fmha_sm100`."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        config = vllm_config.model_config.hf_text_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_q_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.topk_blocks = config.index_topk_blocks
        # AttentionSpec stores every FP8 mode as uint8, so retain the configured
        # format to distinguish E4M3 (supported) from E5M2 before planning.
        self.kv_cache_dtype = vllm_config.cache_config.cache_dtype
        self.decode_backend = vllm_config.attention_config.minimax_m3_msa_decode_backend
        self.q8kv4_decode = uses_q8kv4_decode(
            kv_cache_dtype=self.kv_cache_dtype,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
        )
        self.msa_cutlass_plan_cache = MSACutlassDecodePlanCache()
        self.msa_cutlass_min_batch_size = _get_msa_cutlass_min_batch_size()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3SparseMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        decode = metadata.decode
        if decode is None:
            return metadata

        msa_cutlass = None
        if should_prepare_decode_metadata(
            metadata.num_decodes,
            decode.decode_query_len,
            decode_backend=self.decode_backend,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            page_size=SPARSE_BLOCK_SIZE,
            topk_blocks=self.topk_blocks,
            min_batch_size=self.msa_cutlass_min_batch_size,
        ):
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            msa_cutlass = prepare_decode_metadata(
                decode.block_table,
                decode.seq_lens,
                seq_lens_cpu[: metadata.num_decodes],
                decode.decode_query_len,
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                page_size=SPARSE_BLOCK_SIZE,
                topk_blocks=self.topk_blocks,
                q8kv4=self.q8kv4_decode,
                plan_cache=self.msa_cutlass_plan_cache,
            )
        metadata.decode = MiniMaxM3SparseMSADecodeMetadata(
            seq_lens=decode.seq_lens,
            block_table=decode.block_table,
            decode_query_len=decode.decode_query_len,
            msa_cutlass=msa_cutlass,
        )
        return metadata


class MiniMaxM3SparseMSABaseImpl(MiniMaxM3SparseImpl):
    """What the MSA impls share: CUTLASS decode plans and the prefill CSR."""

    use_cutlass_decode: bool
    # The prefill kernels ``fmha_sm100.warmup`` prebuilds for this impl.
    prefill_backend: str

    def should_use_msa_decode(self, layer_name: str) -> bool:
        if not self.use_cutlass_decode:
            return False
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return False
        main_md = attn_metadata[layer_name]
        if not isinstance(main_md, MiniMaxM3SparseMetadata):
            return False
        decode = main_md.decode
        return (
            isinstance(decode, MiniMaxM3SparseMSADecodeMetadata)
            and decode.msa_cutlass is not None
        )

    def _build_prefill_csr(
        self,
        topk: torch.Tensor,
        nd: int,
        num_tokens: int,
        p: MiniMaxM3SparsePrefillMetadata,
        qhead_per_kv: int | None = None,
    ):
        """`qhead_per_kv` is the Q heads per KV head the prefill kernel runs
        (its tile packs 128 // qhead_per_kv tokens); default the layer's."""
        from vllm.third_party.fmha_sm100.sparse import build_k2q_csr

        if qhead_per_kv is None:
            qhead_per_kv = self.num_heads // self.num_kv_heads
        # [H, prefill, MK] transposed view; build_k2q_csr consumes the strided
        # view directly (topK stays innermost-contiguous).
        return build_k2q_csr(
            topk[nd:num_tokens].transpose(0, 1),
            p.cu_seqlens_q,
            p.cu_seqlens_k,
            SPARSE_BLOCK_SIZE,
            total_k=0,
            max_seqlen_k=p.max_seq_len,
            max_seqlen_q=p.max_query_len,
            total_rows=p.total_kv_blocks,
            qhead_per_kv=qhead_per_kv,
            return_schedule=True,
        )


class MiniMaxM3SparseMSAImpl(MiniMaxM3SparseMSABaseImpl):
    """MSA block-sparse attention over a BF16 or FP8 KV cache, with guarded
    CUTLASS sparse decode."""

    prefill_backend = "cute_dsl"

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_blocks: int,
        sparse_block_size: int,
        msa_decode_backend: MiniMaxM3MSADecodeBackend = "cutlass",
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
        )
        self.use_cutlass_decode = supports_cutlass_sparse_decode(
            decode_backend=msa_decode_backend,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            page_size=self.block_size,
            topk_blocks=self.topk_blocks,
        )
        logger.info_once(
            "MiniMax M3 MSA sparse decode selected %s",
            "CUTLASS" if self.use_cutlass_decode else "Triton",
        )

    def _bf16_query(
        self, q: torch.Tensor, q_scale: float, dtype: torch.dtype
    ) -> torch.Tensor:
        # With CUTLASS decode the query is E4M3; the other kernels read BF16.
        if not self.use_cutlass_decode:
            return q
        return _dequantize_query(q, q_scale, dtype)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        # Indexer top-k from the shared token-major buffer [total_q, H, MK]; the
        # kernels want [H, tokens, MK], so slice tokens on dim 0 then transpose.
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        q_scale_float = getattr(layer, "_q_scale_float", 1.0)
        if self.use_fp8_kv:
            kv_cache = kv_cache.view(self.kv_cache_fp8_dtype)
        k_scale = getattr(layer, "_k_scale", None) if self.use_fp8_kv else None
        v_scale = getattr(layer, "_v_scale", None) if self.use_fp8_kv else None

        # Decode [:nd]: CUTLASS for planned shapes, otherwise Triton.
        if main_md.num_decodes > 0:
            d = main_md.decode
            assert isinstance(d, MiniMaxM3SparseMSADecodeMetadata)
            msa_metadata = d.msa_cutlass
            if self.use_cutlass_decode and msa_metadata is not None:
                msa_cutlass_sparse_decode(
                    q[:nd],
                    kv_cache,
                    topk[:nd],
                    out[:nd],
                    msa_metadata,
                    scale=self.scale,
                    q_scale_float=q_scale_float,
                    k_scale_float=getattr(layer, "_k_scale_float", 1.0),
                    v_scale_float=getattr(layer, "_v_scale_float", 1.0),
                )
            else:
                minimax_m3_sparse_attn_decode(
                    self._bf16_query(q[:nd], q_scale_float, output.dtype),
                    kv_cache,
                    topk[:nd].transpose(0, 1),
                    d.block_table,
                    d.seq_lens,
                    self.num_kv_heads,
                    self.scale,
                    out[:nd],
                    d.decode_query_len,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

        # Prefill [nd:]: MSA sparse FMHA over the selected blocks.
        if main_md.num_prefills > 0:
            from vllm.third_party.fmha_sm100.sparse import sparse_atten_func

            p = main_md.prefill
            assert p is not None
            k2q_row_ptr, k2q_q_indices, schedule = self._build_prefill_csr(
                topk, nd, num_tokens, p
            )
            # An E4M3 query attends as is, its scale folded into softmax_scale.
            qp = q[nd:]
            softmax_scale = (
                self.scale * q_scale_float if self.use_cutlass_decode else self.scale
            )
            k_cache, v_cache = kv_cache.split(self.head_size, dim=-1)
            sparse_atten_func(
                qp,
                k_cache,
                v_cache,
                k2q_row_ptr,
                k2q_q_indices,
                topK=self.topk_blocks,
                blk_kv=SPARSE_BLOCK_SIZE,
                causal=True,
                softmax_scale=softmax_scale,
                cu_seqlens_q=p.cu_seqlens_q,
                cu_seqlens_k=p.cu_seqlens_k,
                max_seqlen_q=p.max_query_len,
                max_seqlen_k=p.max_seq_len,
                page_table=p.block_table,
                seqused_k=p.seq_lens,
                schedule=schedule,
                out=out[nd:],
            )
        return output


class MiniMaxM3SparseMSANvfp4Impl(MiniMaxM3SparseMSABaseImpl):
    """MSA block-sparse attention over an NVFP4 KV cache, SM100 family only.

    Decode and prefill run the CUTLASS Q8KV4 kernels, which read the E4M3
    query; there is no BF16-query kernel and no Triton fallback. Both can
    write the output as MXFP8 instead of BF16 (`forward(output_mxfp8=...)`).
    """

    use_cutlass_decode = True
    prefill_backend = "q8kv4"
    supports_mxfp8_output = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "nvfp4",
        *,
        topk_blocks: int,
        sparse_block_size: int,
        msa_decode_backend: MiniMaxM3MSADecodeBackend = "cutlass",
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
        )
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            raise ValueError("MiniMax M3 NVFP4 KV cache requires an SM100-family GPU")
        # NVFP4 decodes only on CUTLASS, whatever msa_decode_backend says.
        if not supports_cutlass_sparse_decode(
            decode_backend="cutlass",
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            page_size=self.block_size,
            topk_blocks=self.topk_blocks,
        ):
            raise ValueError(
                "MiniMax M3 NVFP4 KV cache requires CUTLASS MSA sparse decode "
                f"(num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
                f"block_size={self.block_size}, topk_blocks={self.topk_blocks})"
            )
        # With 8 Q heads per KV head (e.g. TP8) the prefill can pad its query
        # to the 16 the Q8KV4 kernel serves: each row of its tiles is one
        # (token, head) whose softmax, P x V and split combine read no other
        # row, so the real heads come out as an unpadded call computes them.
        self.prefill_pad_heads = (
            self.num_heads == _Q8KV4_PADDED_PREFILL_Q_HEADS_PER_KV * self.num_kv_heads
            and _q8kv4_prefill_pad_heads_enabled()
        )
        self.prefill_num_heads = (
            _Q8KV4_PREFILL_Q_HEADS_PER_KV * self.num_kv_heads
            if self.prefill_pad_heads
            else self.num_heads
        )
        blocker = _q8kv4_prefill_blocker(
            self.prefill_num_heads, self.num_kv_heads, self.block_size, self.topk_blocks
        )
        if blocker is not None:
            raise ValueError(
                "MiniMax M3 NVFP4 KV cache requires the CUTLASS Q8KV4 prefill, "
                f"which cannot serve this layout or device: {blocker}"
            )
        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        device_index = torch.accelerator.current_device_index()
        self._padded_prefill: _PaddedPrefillBuffers | None = None
        if self.prefill_pad_heads:
            # The padded call writes BF16 over all 16 heads, so its MXFP8
            # output (VLLM_MINIMAX_M3_MSA_MXFP8_OUTPUT) is not offered.
            self.supports_mxfp8_output = False
            self._padded_prefill = _shared_padded_prefill_buffers(
                device_index, max_tokens, self.num_kv_heads, self.head_size
            )
            logger.info_once(
                "MiniMax M3 NVFP4 MSA: CUTLASS Q8KV4 decode at %d Q heads per KV "
                "head; the Q8KV4 prefill pads its query to %d with zero heads "
                "(VLLM_MINIMAX_M3_MSA_PREFILL_PAD_HEADS=1).",
                self.num_heads // self.num_kv_heads,
                _Q8KV4_PREFILL_Q_HEADS_PER_KV,
            )
        else:
            logger.info_once("MiniMax M3 NVFP4 MSA: CUTLASS Q8KV4 decode and prefill")
        # NVFP4_BLOCK_SCALE_SHIFT assumes unit global scales; checked on the
        # first forward, once the checkpoint's KV scales are loaded.
        self._check_global_scales = True
        self._prefill_workspace = _shared_prefill_workspace(
            device_index,
            self.topk_blocks,
            max_tokens,
            self.prefill_num_heads,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor | None,
        output_mxfp8: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Attend into the BF16 `output`, or, with `output_mxfp8` given and
        `output` None, into o_proj's MXFP8 input `(data, scale)`: E4M3 `data`
        `[tokens, num_heads * head_size]` and a flat uint8 `scale` of UE8M0
        scales in FlashInfer's 128x4 swizzled layout, rows padded to 128."""
        assert query.dtype == torch.float8_e4m3fn
        assert (output is None) != (output_mxfp8 is None)
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            if output_mxfp8 is not None:
                # Zeros rather than uninitialized E4M3/UE8M0 bytes, which can
                # hold NaN codes (a BF16 output is sanitized by its quantize).
                output_mxfp8[0].zero_()
                output_mxfp8[1].zero_()
            return output  # profiling run; caches unbound
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        hd = self.head_size
        q_fp8 = query[:num_tokens].view(-1, self.num_heads, hd)
        out = (
            None if output is None else output[:num_tokens].view(-1, self.num_heads, hd)
        )
        if output_mxfp8 is not None and output_mxfp8[0].shape[0] > num_tokens:
            # Rows past the batch (piecewise cudagraph padding) get zeros. The
            # kernels clear the scales up to the next 128-row tile; the tiles
            # after it are a contiguous byte range of the swizzled layout.
            mx_data, mx_scale = output_mxfp8
            mx_data[num_tokens:].zero_()
            tile_rows = (num_tokens + 127) // 128 * 128
            mx_scale[tile_rows * (mx_data.shape[1] // 32) :].zero_()
        q_scale_float = getattr(layer, "_q_scale_float", 1.0)
        k_scale = getattr(layer, "_k_scale", None)
        v_scale = getattr(layer, "_v_scale", None)
        if self._check_global_scales:
            k_global = getattr(layer, "_k_scale_float", 1.0)
            v_global = getattr(layer, "_v_scale_float", 1.0)
            if k_global != 1.0 or v_global != 1.0:
                raise ValueError(
                    "MiniMax M3 reads its NVFP4 KV cache on the Q8KV4 kernels with "
                    f"block_scale_shift={NVFP4_BLOCK_SCALE_SHIFT}, which needs unit "
                    f"global scales; got k_scale={k_global}, v_scale={v_global}"
                )
            self._check_global_scales = False

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert isinstance(d, MiniMaxM3SparseMSADecodeMetadata)
            msa_metadata = d.msa_cutlass
            if msa_metadata is None:
                raise RuntimeError(
                    "MiniMax M3 NVFP4 KV cache has no Triton decode fallback; "
                    f"CUTLASS decode was not planned for {main_md.num_decodes} "
                    f"requests with query length {d.decode_query_len}"
                )
            msa_cutlass_sparse_decode(
                q_fp8[:nd],
                kv_cache,
                topk[:nd],
                None if out is None else out[:nd],
                msa_metadata,
                scale=self.scale,
                q_scale_float=q_scale_float,
                k_scale_float=getattr(layer, "_k_scale_float", 1.0),
                v_scale_float=getattr(layer, "_v_scale_float", 1.0),
                k_scale=k_scale,
                v_scale=v_scale,
                output_mxfp8=(
                    None
                    if output_mxfp8 is None
                    else (output_mxfp8[0][:nd], output_mxfp8[1])
                ),
            )

        if main_md.num_prefills > 0:
            from vllm.third_party.fmha_sm100.prefill_q8kv4 import run_prefill

            p = main_md.prefill
            assert p is not None
            k2q_row_ptr, _, schedule = self._build_prefill_csr(
                topk,
                nd,
                num_tokens,
                p,
                qhead_per_kv=self.prefill_num_heads // self.num_kv_heads,
            )
            k_data, k_sf, v_data, v_sf = nvfp4_kv_cache_views(kv_cache)
            prefill_q = q_fp8[nd:]
            prefill_out = None if out is None else out[nd:]
            if self._padded_prefill is not None:
                assert out is not None and output_mxfp8 is None
                logger.info_once(
                    "MiniMax M3 NVFP4 MSA: prefill runs the Q8KV4 kernel on the "
                    "zero-padded query."
                )
                padded_q, prefill_out = _padded_prefill_views(
                    num_tokens - nd,
                    self.num_kv_heads,
                    hd,
                    self._padded_prefill,
                    q_fp8.device,
                )
                prefill_q = _pad_prefill_query(prefill_q, padded_q)
            mxfp8_kwargs: dict[str, Any] = {}
            if output_mxfp8 is not None:
                # Decode rows come first: the combine writes rows [nd, num_tokens)
                # of the same activation (scales indexed from row nd) and clears
                # the scales of its padding rows.
                mx_data, mx_scale = output_mxfp8
                mx_rows = mx_data[nd:num_tokens].view(-1, self.num_heads, hd)
                mxfp8_kwargs = dict(
                    out_mxfp8=(mx_rows, mx_scale), out_mxfp8_first_row=nd
                )
            run_prefill(
                prefill_q,
                (k_data, v_data),
                (k_sf, v_sf),
                kv_indices=p.block_table,
                kv_indptr=None,
                cu_seqlens_q=p.cu_seqlens_q,
                cu_seqlens_k=p.cu_seqlens_k,
                k2q_row_ptr=k2q_row_ptr,
                schedule=schedule,
                sm_scale=self.scale * q_scale_float,
                topk=self.topk_blocks,
                seqused_k=p.seq_lens,
                k_global_scale=k_scale,
                v_global_scale=v_scale,
                block_scale_shift=NVFP4_BLOCK_SCALE_SHIFT,
                out=prefill_out,
                workspace=self._prefill_workspace,
                **mxfp8_kwargs,
            )
            if self._padded_prefill is not None:
                assert out is not None and prefill_out is not None
                _unpad_prefill_output(prefill_out, out[nd:])
        return output
