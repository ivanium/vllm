# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_MEGAMOE_FUSED_ROUTING: one kernel after the router GEMM gives the bytes
of splitk_reduce_triton + topk_sigmoid + prepare_megamoe_nvfp4_inputs; with the
switch off, or where the path does not apply, nothing is launched."""

import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as dgmm
from vllm.model_executor.layers.fused_moe import prepare_megamoe as pm
from vllm.platforms import current_platform

sm100 = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the fused routing kernel feeds the SM100 NVFP4 MegaMoE kernel",
)

HIDDEN, NUM_EXPERTS, TOP_K, NUM_LOCAL, RSF = 6144, 128, 4, 9, 2.0


class _UnusedGate:
    weight = torch.empty(NUM_EXPERTS, 8)

    def forward_split_k_partials(self, *args):
        raise AssertionError("the router GEMM ran")


def _experts(weight_format):
    cls = dgmm.DeepGemmMegaMoEExperts
    experts = cls.__new__(cls)
    torch.nn.Module.__init__(experts)
    experts.weight_format = weight_format
    experts.max_num_tokens = 64
    experts._fused_routing_bias_ok = None
    return experts


@pytest.mark.parametrize("value", [None, "0"])
def test_switch_off_launches_nothing(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("VLLM_MEGAMOE_FUSED_ROUTING", raising=False)
    else:
        monkeypatch.setenv("VLLM_MEGAMOE_FUSED_ROUTING", value)
    pm.megamoe_fused_routing_enabled.cache_clear()
    experts = _experts(dgmm.MEGA_MOE_NVFP4)
    routing = experts.sigmoid_router_partials(
        _UnusedGate(), torch.empty(4, 8), torch.zeros(NUM_EXPERTS), TOP_K, True, RSF
    )
    assert routing is None
    pm.megamoe_fused_routing_enabled.cache_clear()


def test_inapplicable_launches_nothing(monkeypatch):
    monkeypatch.setenv("VLLM_MEGAMOE_FUSED_ROUTING", "1")
    pm.megamoe_fused_routing_enabled.cache_clear()
    try:
        # Not NVFP4 (or not CUDA): the gate is never asked for partials.
        experts = _experts(dgmm.MEGA_MOE_MXFP4)
        assert (
            experts.sigmoid_router_partials(
                _UnusedGate(), torch.empty(4, 8), torch.zeros(128), TOP_K, True, RSF
            )
            is None
        )
        # No split-K entry point on the gate.
        experts = _experts(dgmm.MEGA_MOE_NVFP4)
        assert (
            experts.sigmoid_router_partials(
                torch.nn.Linear(8, NUM_EXPERTS),
                torch.empty(4, 8),
                torch.zeros(NUM_EXPERTS),
                TOP_K,
                True,
                RSF,
            )
            is None
        )
    finally:
        pm.megamoe_fused_routing_enabled.cache_clear()


def test_fold_orders_cover_every_reduce_config():
    """Every BLOCK_S `_splitk_reduce_config` returns for up to
    MAX_FUSED_ROUTING_SPLIT_K splits has an order over all of its index bits."""
    for split_k in range(1, pm.MAX_FUSED_ROUTING_SPLIT_K + 1):
        block_s = 1 << (split_k - 1).bit_length()
        order = pm._SPLITK_REDUCE_FOLD_ORDER[block_s]
        assert sorted(order) == list(range(block_s.bit_length() - 1))


# ----------------------------------------------------------------------------- GPU


def _routing_inputs(num_tokens, seed):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(seed)
    hidden = (
        torch.randn(num_tokens, HIDDEN, device=device, generator=gen)
        * torch.logspace(-2, 1, max(num_tokens, 1), device=device)[:num_tokens, None]
    ).bfloat16()
    if num_tokens >= 8:
        hidden[2] = 0
        hidden[3, 5] = float("nan")
        hidden[4, 9] = float("inf")
    weight = (
        torch.randn(NUM_EXPERTS, HIDDEN, device=device, generator=gen) * HIDDEN**-0.5
    )
    bias = torch.randn(NUM_EXPERTS, device=device, generator=gen) * 0.05
    weight[1:4] = weight[0]  # tied experts
    bias[1:4] = bias[0]
    replicas = torch.ones(NUM_EXPERTS, dtype=torch.int64)
    replicas[:16] = 2
    mapping = torch.full((NUM_EXPERTS, 3), -1, dtype=torch.int64)
    mapping[:, 0] = torch.arange(NUM_EXPERTS)
    mapping[:16, 1] = NUM_EXPERTS + torch.arange(16)

    def state():
        return EplbLayerState(
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

    is_padding = torch.arange(num_tokens, device=device) % 5 == 1
    return hidden, weight, bias, is_padding, state


def _outputs(num_tokens, shared_block_m):
    device = torch.device("cuda")
    rows = (num_tokens + shared_block_m - 1) // shared_block_m * 128
    return dict(
        x=torch.full((num_tokens, HIDDEN // 2), 0x5A, dtype=torch.uint8, device=device),
        x_sf=torch.full(
            (num_tokens, HIDDEN // 64), -7, dtype=torch.int32, device=device
        ),
        x_scales=torch.full((num_tokens,), -3.0, device=device),
        topk_idx=torch.full((num_tokens, TOP_K), -9, dtype=torch.int64, device=device),
        topk_w=torch.full((num_tokens, TOP_K), -9.0, device=device),
        l2_alpha=torch.full((NUM_LOCAL,), -1.0, device=device),
        shared_x=torch.full(
            (num_tokens, HIDDEN), 0x33, dtype=torch.uint8, device=device
        ).view(torch.float8_e4m3fn),
        shared_x_sf=torch.full(
            (HIDDEN // 128, rows), -1, dtype=torch.int32, device=device
        ).t(),
        logits=torch.full((num_tokens, NUM_EXPERTS), -5.0, device=device),
        ids=torch.full((num_tokens, TOP_K), -5, dtype=torch.int32, device=device),
        weights=torch.full((num_tokens, TOP_K), -5.0, device=device),
    )


@sm100
@pytest.mark.parametrize("num_tokens", [1, 16, 33, 64, 136, 192, 513, 4096, 8192])
def test_fused_routing_stages_the_same_bytes(num_tokens):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.router import gate_linear as gl

    hidden, weight, bias, is_padding, state = _routing_inputs(num_tokens, num_tokens)
    w2 = torch.rand(NUM_LOCAL, device="cuda")
    sbm = 64 if num_tokens <= 192 else 192
    results = []
    for fused in (False, True):
        out, eplb = _outputs(num_tokens, sbm), state()
        common = dict(
            is_padding=is_padding,
            shared_x=out["shared_x"],
            shared_x_sf=out["shared_x_sf"],
            shared_block_m=sbm,
            w2_global_scale=w2,
            l2_alpha=out["l2_alpha"],
            l2_activation_scale=0.37,
            eplb_state=eplb,
        )
        if fused:
            partials = gl.fp32_router_gemm_split_k_partials(
                hidden, weight, pm.MAX_FUSED_ROUTING_SPLIT_K
            )
            assert partials is not None
            pm.route_and_stage_megamoe_nvfp4(
                pm.MegaMoESigmoidRouting(partials, bias, TOP_K, True, RSF),
                hidden,
                out["x"],
                out["x_sf"],
                out["x_scales"],
                out["topk_idx"],
                out["topk_w"],
                router_logits=out["logits"],
                topk_ids=out["ids"],
                topk_weights=out["weights"],
                **common,
            )
        else:
            out["logits"].copy_(gl.fp32_router_gemm_dispatch_impl(hidden, weight, True))
            ops.topk_sigmoid(
                out["weights"],
                out["ids"],
                torch.empty_like(out["ids"]),
                out["logits"],
                True,
                bias,
                RSF,
                is_padding=is_padding,
            )
            pm.prepare_megamoe_nvfp4_inputs(
                hidden,
                out["weights"],
                out["ids"],
                out["x"],
                out["x_sf"],
                out["x_scales"],
                out["topk_idx"],
                out["topk_w"],
                **common,
            )
        torch.cuda.synchronize()
        results.append(
            {k: v.contiguous().view(torch.uint8).clone() for k, v in out.items()}
            | {"expert_load": eplb.expert_load_view.clone()}
        )
    for name, ref in results[0].items():
        assert torch.equal(ref, results[1][name]), name


@sm100
def test_startup_check_passes():
    _, _, bias, _, _ = _routing_inputs(8, 0)
    routing = pm.MegaMoESigmoidRouting(None, bias, TOP_K, True, RSF)
    assert pm._fused_routing_mismatch(routing) is None
