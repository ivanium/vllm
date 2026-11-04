# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for kernel_warmup's KV-block zeroer warmup."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.kernel_warmup import kernel_warmup

pytestmark = pytest.mark.cpu_test


def test_kernel_warmup_warms_each_v1_kv_block_zeroer():
    """The V1 runner keeps one zeroer per KV cache group in
    ``kv_block_zeroers``; warmup must reach every one of them."""
    warmed: list[int] = []
    zeroer = SimpleNamespace(warmup=lambda num_blocks: warmed.append(num_blocks))
    worker = SimpleNamespace(
        use_v2_model_runner=False,
        get_model=torch.nn.Module,
        model_runner=SimpleNamespace(
            kv_block_zeroers={0: zeroer, 2: zeroer},
            kv_cache_config=SimpleNamespace(num_pages=8),
            device=torch.device("cpu"),
            get_model=torch.nn.Module,
            is_pooling_model=False,
            attn_groups=[],
        ),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(
                enable_jit_warmup=False,
                enable_cutedsl_warmup=False,
                enable_flashinfer_autotune=False,
            ),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type=""),
                hf_config=SimpleNamespace(model_type=""),
            ),
        ),
    )

    kernel_warmup(worker, process_local_only=True)

    assert warmed == [8, 8]
