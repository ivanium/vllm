# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The row-block NVFP4 MegaMoE staging kernel (VLLM_MEGAMOE_NVFP4_ROW_STAGING=1)
and its split halves write the same bytes as the default staging kernel."""

import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe import prepare_megamoe as pm
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="NVFP4 MegaMoE input staging only feeds the SM100 kernel",
)

TOP_K, NUM_EXPERTS, NUM_LOCAL = 4, 128, 6


def _inputs(num_tokens, hidden_size, eplb):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(num_tokens + hidden_size)
    # Rows span four decades, plus zero, tiny, -0, BF16-subnormal, NaN and Inf.
    row_scale = torch.logspace(-3, 1, max(num_tokens, 1), device=device)
    hidden = (
        torch.randn(num_tokens, hidden_size, device=device, generator=generator)
        * row_scale[:num_tokens, None]
    ).bfloat16()
    if num_tokens >= 8:
        hidden[0] = 0
        hidden[1, :64] = 1.0e-30
        hidden[1, 64:96] = -1.0e-38
        hidden[2, ::7] = -0.0
        hidden[3, 5] = float("nan")
        hidden[4, 9] = float("inf")
        hidden[5, :32] = 2.0**-133
        hidden[6, 100] = -float("inf")
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (num_tokens, TOP_K), device=device, generator=generator
    ).int()
    if num_tokens > 2:
        topk_ids[1, 0], topk_ids[2, 0] = -1, NUM_EXPERTS
    topk_weights = torch.rand(num_tokens, TOP_K, device=device, generator=generator)
    is_padding = torch.arange(num_tokens, device=device) % 5 == 1
    state = None
    if eplb:
        replicas = torch.ones(NUM_EXPERTS, dtype=torch.int64)
        replicas[:16] = 2
        mapping = torch.full((NUM_EXPERTS, 3), -1, dtype=torch.int64)
        mapping[:, 0] = torch.arange(NUM_EXPERTS)
        mapping[:16, 1] = NUM_EXPERTS + torch.arange(16)
        state = EplbLayerState(
            logical_to_physical_map=mapping.to(device),
            logical_replica_count=replicas.to(device),
            expert_load_view=torch.zeros(
                NUM_EXPERTS + 16, dtype=torch.int32, device=device
            ),
            should_record_tensor=torch.tensor(True, device=device),
            num_unpadded_tokens_tensors=[
                torch.tensor(max(num_tokens - 3, 0), dtype=torch.int32, device=device)
            ]
            * 2,
        )
    w2_global_scale = torch.rand(NUM_LOCAL, device=device, generator=generator)
    return hidden, topk_weights, topk_ids, is_padding, state, w2_global_scale


def _outputs(num_tokens, hidden_size, shared, shared_block_m):
    device = torch.device("cuda")
    out = dict(
        x=torch.full(
            (num_tokens, hidden_size // 2), 0x5A, dtype=torch.uint8, device=device
        ).view(torch.int8),
        x_sf=torch.full(
            (num_tokens, hidden_size // 64), -7, dtype=torch.int32, device=device
        ),
        x_scales=torch.full((num_tokens,), -3.0, device=device),
        topk_idx=torch.full((num_tokens, TOP_K), -9, dtype=torch.int64, device=device),
        topk_weights=torch.full((num_tokens, TOP_K), -9.0, device=device),
        l2_alpha=torch.full((NUM_LOCAL,), -1.0, device=device),
        shared_x=None,
        shared_x_sf=None,
    )
    if shared == "mxfp8":
        out["shared_x"] = torch.full(
            (num_tokens, hidden_size), 0x33, dtype=torch.uint8, device=device
        ).view(torch.float8_e4m3fn)
        aligned = triton.cdiv(shared_block_m, 128) * 128
        rows = triton.cdiv(num_tokens, shared_block_m) * aligned
        # MN-major, as the shared-expert TMA layout.
        out["shared_x_sf"] = torch.full(
            (hidden_size // 128, rows), -1, dtype=torch.int32, device=device
        ).t()
    elif shared == "bf16":
        out["shared_x"] = torch.full(
            (num_tokens, hidden_size), 7.0, dtype=torch.bfloat16, device=device
        )
    return out


def _stage(monkeypatch, mode, inputs, out, shared_block_m):
    hidden, topk_weights, topk_ids, is_padding, state, w2_global_scale = inputs
    if state is not None:
        state.expert_load_view.zero_()
    shared_kwargs = dict(
        shared_x=out["shared_x"],
        shared_x_sf=out["shared_x_sf"],
        shared_block_m=shared_block_m if out["shared_x_sf"] is not None else None,
        w2_global_scale=w2_global_scale,
        l2_alpha=out["l2_alpha"],
        l2_activation_scale=0.37,
    )
    if mode == "split":
        pm.stage_megamoe_nvfp4_hidden_states(
            hidden, out["x"], out["x_sf"], out["x_scales"], **shared_kwargs
        )
        pm.stage_megamoe_topk(
            topk_weights,
            topk_ids,
            out["topk_idx"],
            out["topk_weights"],
            is_padding=is_padding,
            eplb_state=state,
        )
    else:
        monkeypatch.setenv(
            "VLLM_MEGAMOE_NVFP4_ROW_STAGING", "1" if mode == "rows" else "0"
        )
        pm.nvfp4_row_staging_enabled.cache_clear()
        pm.prepare_megamoe_nvfp4_inputs(
            hidden,
            topk_weights,
            topk_ids,
            out["x"],
            out["x_sf"],
            out["x_scales"],
            out["topk_idx"],
            out["topk_weights"],
            is_padding=is_padding,
            eplb_state=state,
            **shared_kwargs,
        )
        pm.nvfp4_row_staging_enabled.cache_clear()
    torch.accelerator.synchronize()
    result = {k: v.clone() for k, v in out.items() if v is not None}
    if state is not None:
        result["expert_load"] = state.expert_load_view.clone()
    return result


@pytest.mark.parametrize("shared", ["mxfp8", "bf16", None])
@pytest.mark.parametrize(
    "num_tokens", [0, 1, 7, 64, 120, 136, 160, 192, 513, 1027, 4096, 16384]
)
@pytest.mark.parametrize("hidden_size", [6144, 5120, 256])
def test_row_staging_writes_the_same_bytes(
    monkeypatch, hidden_size, num_tokens, shared
):
    eplb = num_tokens <= 512
    shared_block_m = 64 if num_tokens <= 512 else 192
    inputs = _inputs(num_tokens, hidden_size, eplb)
    results = {
        mode: _stage(
            monkeypatch,
            mode,
            inputs,
            _outputs(num_tokens, hidden_size, shared, shared_block_m),
            shared_block_m,
        )
        for mode in ("default", "rows", "split")
    }
    for mode in ("rows", "split"):
        for name, expected in results["default"].items():
            actual = results[mode][name]
            assert torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            ), f"{mode}: {name} differs"


@triton.jit
def _e2m1_check_kernel(mismatches, start, BLOCK: tl.constexpr, SWAP: tl.constexpr):
    bits = (
        start
        + tl.program_id(0).to(tl.int64) * (2 * BLOCK)
        + tl.arange(0, 2 * BLOCK).to(tl.int64)
    )
    values = bits.to(tl.uint32).to(tl.float32, bitcast=True)
    values = tl.where(values == values, values, 0.0)
    even, odd = tl.split(tl.reshape(values, [BLOCK, 2]))
    if SWAP:
        low, high = odd, even
    else:
        low, high = even, odd
    expected = (pm._e2m1_codes(low) | (pm._e2m1_codes(high) << 4)).to(tl.uint8)
    actual = pm._e2m1x2_cvt(high, low)
    tl.atomic_add(mismatches, tl.sum((expected != actual).to(tl.int32), axis=0))


@pytest.mark.parametrize("swap", [False, True])
def test_e2m1_conversion_matches_threshold_codes(swap):
    """Every fp32 bit pattern but NaN, which the kernel zeroes first, in each
    nibble: every E2M1 tie and grid point is an even pattern."""
    mismatches = torch.zeros(1, dtype=torch.int32, device="cuda")
    block, chunk = 4096, 1 << 30
    for start in range(0, 1 << 32, chunk):
        _e2m1_check_kernel[(chunk // (2 * block),)](
            mismatches, start, BLOCK=block, SWAP=swap
        )
    assert mismatches.item() == 0


@triton.jit
def _division_check_kernel(
    mismatches, a_ptr, b_ptr, num_a, num_b, scale_a, scale_b, BLOCK: tl.constexpr
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < num_a * num_b
    a = tl.load(a_ptr + i // num_b, mask=mask, other=1.0) * scale_a
    b = tl.load(b_ptr + i % num_b, mask=mask, other=1.0) * scale_b
    actual = pm._div_rn_by_recip(a, b, tl.math.div_rn(1.0, b))
    expected = tl.math.div_rn(a, b)
    differ = actual.to(tl.uint32, bitcast=True) != expected.to(tl.uint32, bitcast=True)
    tl.atomic_add(mismatches, tl.sum((differ & mask).to(tl.int32), axis=0))


@pytest.mark.parametrize(
    "dividend, divisor, scales",
    [
        ("bf16", "fp32", [(1.0, 1.0), (2.0**-20, 2.0**-30), (2.0**40, 2.0**30)]),
        ("fp32", "e4m3", [(1.0, 1.0), (2.0**-12, 2.0**-9), (2.0**11, 2.0**8)]),
    ],
)
def test_div_rn_by_recip_matches_div_rn(dividend, divisor, scales):
    """Every significand pair of both divisions, at several exponents."""
    device = torch.device("cuda")
    significands = {
        "fp32": (torch.arange(1 << 23, dtype=torch.int32, device=device) | 0x3F800000),
        "bf16": (torch.arange(128, dtype=torch.int32, device=device) << 16) | 0x3F800000,
    }
    a = significands[dividend].view(torch.float32)
    b = (
        torch.tensor([1 + k / 8 for k in range(8)], device=device)
        if divisor == "e4m3"
        else significands[divisor].view(torch.float32)
    )
    mismatches = torch.zeros(1, dtype=torch.int32, device=device)
    block = 1024
    for scale_a, scale_b in scales:
        _division_check_kernel[(triton.cdiv(a.numel() * b.numel(), block),)](
            mismatches, a, b, a.numel(), b.numel(), scale_a, scale_b, BLOCK=block
        )
    assert mismatches.item() == 0


@pytest.mark.parametrize("graph", [False, True])
def test_staging_alongside_the_router_stages_the_same_bytes(monkeypatch, graph):
    """DeepGemmMegaMoEExperts.staging_hidden_states stages on the aux stream and
    forward(..., hidden_states_staged=True) only the routing; the MegaMoE launch
    sees the same buffer bytes as with one staging call, eagerly and replayed."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as dgmm

    device = torch.device("cuda")
    num_tokens, hidden_size, shared_block_m = 136, 6144, 64
    hidden, topk_weights, topk_ids, _, state, w2_global_scale = _inputs(
        num_tokens, hidden_size, eplb=True
    )
    buffer = _outputs(num_tokens, hidden_size, "mxfp8", shared_block_m)
    symm_buffer = SimpleNamespace(
        x=buffer["x"],
        x_sf=buffer["x_sf"],
        x_scales=buffer["x_scales"],
        topk_idx=buffer["topk_idx"],
        topk_weights=buffer["topk_weights"],
        shared_l1_acts=buffer["shared_x"],
        shared_l1_acts_sf=buffer["shared_x_sf"],
    )
    experts = dgmm.DeepGemmMegaMoEExperts.__new__(dgmm.DeepGemmMegaMoEExperts)
    torch.nn.Module.__init__(experts)
    experts.prefix = "model.layers.3.ffn.experts"
    experts.max_num_tokens = num_tokens
    experts.num_shared_experts = 1
    experts.shared_spec = dgmm.MegaMoEWeightSpec(
        torch.float8_e4m3fn, scale_group_k=32, scale_dtype=torch.float8_e8m0fnu
    )
    experts.weight_format = dgmm.MEGA_MOE_NVFP4
    experts._kernel_kwargs = {}
    experts._staging_events = None
    experts.capture_fn = None
    experts.eplb_state = state
    experts.fast_math = True
    experts.w13_weight_scale_2 = torch.ones(NUM_LOCAL, device=device)
    experts.w2_weight_scale_2 = w2_global_scale
    experts._l2_alpha = buffer["l2_alpha"]
    experts.l2_activation_scale = 0.37
    monkeypatch.setattr(experts, "_synchronize_first_launch", lambda: None)
    monkeypatch.setattr(experts, "get_symm_buffer", lambda: symm_buffer)
    monkeypatch.setattr(experts, "get_block_m", lambda n: shared_block_m)
    monkeypatch.setattr(experts, "_kernel_weights", lambda name: None)
    monkeypatch.setattr(dgmm.DeepGemmMegaMoEExperts, "l1_weights", None)
    monkeypatch.setattr(dgmm.DeepGemmMegaMoEExperts, "l2_weights", None)
    seen = []

    def nvfp4_mega_moe(y, l1, l2, buf, **kwargs):
        # Stream-ordered copies: what the kernel would read when it starts.
        seen.append({k: v.clone() for k, v in vars(buf).items()} | {
            "l2_alpha": kwargs["l2_alpha"].clone(),
            "expert_load": state.expert_load_view.clone(),
        })

    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm",
        lambda: SimpleNamespace(nvfp4_mega_moe=nvfp4_mega_moe),
    )
    gate = torch.randn(hidden_size, NUM_EXPERTS, device=device)

    def layer(overlap):
        with experts.staging_hidden_states(hidden) as staged:
            torch.matmul(hidden.float(), gate)  # the router, on the main stream
        assert staged == overlap
        return experts(hidden, topk_weights, topk_ids, hidden_states_staged=staged)

    def reset():
        for tensor in buffer.values():
            if tensor is not None:
                # Any strides: shared_x_sf is a transposed int32 view.
                pattern = torch.full(
                    (tensor.numel() * tensor.element_size(),),
                    0xA5,
                    dtype=torch.uint8,
                    device=device,
                )
                tensor.copy_(pattern.view(tensor.dtype).view(tensor.shape))
        state.expert_load_view.zero_()

    reset()
    monkeypatch.setenv("VLLM_MEGAMOE_STAGING_OVERLAP", "0")
    dgmm.megamoe_staging_overlap_enabled.cache_clear()
    layer(False)
    monkeypatch.setenv("VLLM_MEGAMOE_STAGING_OVERLAP", "1")
    dgmm.megamoe_staging_overlap_enabled.cache_clear()
    reset()
    if graph:
        layer(True)  # compile the kernels outside the capture
        seen.pop()
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            layer(True)  # its copies are written on replay
        reset()
        cuda_graph.replay()
    else:
        layer(True)
    torch.accelerator.synchronize()
    dgmm.megamoe_staging_overlap_enabled.cache_clear()
    expected, actual = seen[0], seen[-1]
    for name, tensor in expected.items():
        assert torch.equal(
            actual[name].contiguous().view(torch.uint8),
            tensor.contiguous().view(torch.uint8),
        ), name
