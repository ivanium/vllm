# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Step3p5MTPProposer's per-group slot-mapping math."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID
from vllm.v1.worker.utils import AttentionGroup

pytest.importorskip("triton")
if not current_platform.is_xpu() and not torch.cuda.is_available():
    pytest.skip("CUDA/XPU required for the EAGLE step kernel", allow_module_level=True)

DEVICE_TYPE = current_platform.device_type

MANAGER_BLOCK_SIZE = 8


def _make_group(group_id: int, layer_name: str) -> AttentionGroup:
    spec = FullAttentionSpec(
        block_size=MANAGER_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float16,
    )
    return AttentionGroup(
        backend=None,  # type: ignore[arg-type]
        layer_names=[layer_name],
        kv_cache_spec=spec,
        kv_cache_group_id=group_id,
    )


def test_step3p5_secondary_group_slot_mapping_uses_own_block_table():
    """The secondary group's slot ids use its own block table, not the
    primary group's."""
    device = torch.device(DEVICE_TYPE)
    primary = _make_group(0, "d0.attn")
    secondary = _make_group(1, "d1.attn")

    proposer = Step3p5MTPProposer.__new__(Step3p5MTPProposer)
    proposer.uses_mrope = False
    proposer.device = device
    proposer.max_model_len = 64
    proposer.max_positions = 2
    proposer.positions = torch.zeros(2, dtype=torch.int64, device=device)
    proposer._slot_mapping_buffer = torch.full(
        (2,), PADDING_SLOT_ID, dtype=torch.int64, device=device
    )
    proposer.draft_attn_groups = [primary, secondary]
    proposer.kv_cache_gid = 0
    proposer._per_group_block_tables = {
        1: torch.tensor([[3, 4], [6, 9]], dtype=torch.int32, device=device)
    }
    proposer._per_group_slot_mappings = {}
    proposer._per_group_slot_mapping_buffers = {}

    cad = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([5, 12], dtype=torch.int32, device=device),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=12,
        block_table_tensor=torch.tensor(
            [[1, 7], [5, 8]], dtype=torch.int32, device=device
        ),
        slot_mapping=torch.zeros(2, dtype=torch.int64, device=device),
    )

    old_positions = torch.tensor([3, 10], dtype=torch.int64, device=device)
    proposer._update_positions_dependent_metadata(old_positions, cad, 2, 2, 8)

    # Positions advance by one step (3 -> 4, 10 -> 11).
    assert cad.slot_mapping.tolist() == [1 * 8 + 4, 8 * 8 + 3]
    assert proposer._per_group_slot_mappings[1].tolist() == [3 * 8 + 4, 9 * 8 + 3]
