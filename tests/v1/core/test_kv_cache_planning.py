# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.core.kv_cache_planning import _project_kv_cache_groups_to_worker
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)


def _new_kv_cache_spec(num_kv_heads: int = 2) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=64,
        dtype=torch.float32,
    )


def test_project_kv_cache_groups_to_worker():
    spec_a = _new_kv_cache_spec()
    spec_b = _new_kv_cache_spec(num_kv_heads=4)

    global_groups = [
        KVCacheGroupSpec(["layer1", "layer2", "layer3"], spec_a),
    ]
    worker_spec = {"layer1": spec_a, "layer2": spec_a}
    projected = _project_kv_cache_groups_to_worker(global_groups, worker_spec)
    assert len(projected) == 1
    assert projected[0].layer_names == ["layer1", "layer2"]
    assert projected[0].kv_cache_spec is spec_a

    projected = _project_kv_cache_groups_to_worker(global_groups, {"layer4": spec_a})
    assert len(projected) == 1
    assert projected[0].layer_names == []
    assert projected[0].kv_cache_spec is spec_a

    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"layer1": spec_a, "layer2": spec_b, "layer3": spec_a},
    )
    global_groups_uniform = [
        KVCacheGroupSpec(["layer1", "layer2", "layer3"], uniform_spec),
    ]
    projected = _project_kv_cache_groups_to_worker(
        global_groups_uniform, {"layer1": spec_a, "layer3": spec_a}
    )
    assert len(projected) == 1
    assert projected[0].layer_names == ["layer1", "layer3"]
    proj_spec = projected[0].kv_cache_spec
    assert isinstance(proj_spec, UniformTypeKVCacheSpecs)
    assert set(proj_spec.kv_cache_specs.keys()) == {"layer1", "layer3"}
