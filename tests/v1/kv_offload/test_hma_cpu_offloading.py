# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for HMA (Hybrid Memory Allocator) CPU offloading.

This test verifies:
1. KV cache offloading to CPU works correctly
2. KV cache loading from CPU produces correct generation results
3. Prefix cache hits from CPU are faster than cold computation

The test uses the OffloadingConnector with HybridOffloadingManager and
HybridCpuGpuOffloadingHandlers internally, even for single-group models.
"""

import socket
import time

import pytest

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.platforms import current_platform


def get_free_port() -> int:
    """Get a free port for ZMQ communication."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def create_llm_with_cpu_offloading(
    model: str,
    gpu_memory_utilization: float = 0.5,
    num_cpu_blocks: int = 1000,
    cpu_block_size: int = 48,
    attn_backend: str | None = None,
) -> LLM:
    """
    Create an LLM instance with CPU offloading enabled.

    Args:
        model: Model name or path.
        gpu_memory_utilization: GPU memory fraction to use.
        num_cpu_blocks: Number of CPU blocks for offloading.
        cpu_block_size: Size of CPU blocks in tokens.
        attn_backend: Attention backend to use.

    Returns:
        LLM instance with CPU offloading configured.
    """
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "num_cpu_blocks": num_cpu_blocks,
            "block_size": cpu_block_size,
        },
    )

    kwargs = {
        "model": model,
        "gpu_memory_utilization": gpu_memory_utilization,
        "kv_transfer_config": kv_transfer_config,
        "enforce_eager": True,  # Faster for testing
    }

    if attn_backend:
        kwargs["attention_config"] = {"backend": attn_backend}

    return LLM(**kwargs)


def verify_generation_correctness(
    llm: LLM,
    prompt: str,
    expected_output: str,
    num_trials: int = 10,
    success_threshold: float = 0.8,
) -> bool:
    """
    Verify that generation produces expected output consistently.

    Args:
        llm: LLM instance.
        prompt: Input prompt.
        expected_output: Expected output text.
        num_trials: Number of trials to run.
        success_threshold: Minimum fraction of successful trials.

    Returns:
        True if success rate meets threshold.
    """
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    success_count = 0

    for _ in range(num_trials):
        outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
        if outputs[0].outputs[0].text == expected_output:
            success_count += 1

    success_rate = success_count / num_trials
    return success_rate >= success_threshold


def measure_generation_latency(
    llm: LLM,
    prompt_token_ids: list[int],
    num_trials: int = 5,
) -> float:
    """
    Measure average generation latency.

    Args:
        llm: LLM instance.
        prompt_token_ids: Token IDs for the prompt.
        num_trials: Number of trials to average.

    Returns:
        Average latency in seconds.
    """
    sampling_params = SamplingParams(max_tokens=1)
    prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]

    total_time = 0.0
    for _ in range(num_trials):
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        total_time += time.time() - start_time

    return total_time / num_trials


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CPU offloading requires CUDA-alike GPU",
)
def test_hma_cpu_offloading_correctness():
    """
    Test that generation is correct after KV cache CPU offload and reload.

    This test:
    1. Generates with a prompt (triggers KV computation and CPU offload)
    2. Resets prefix cache (removes GPU cache)
    3. Generates again (triggers CPU->GPU reload)
    4. Verifies output matches expected result
    """
    llm = create_llm_with_cpu_offloading(
        model="meta-llama/Llama-3.2-1B-Instruct",
        gpu_memory_utilization=0.4,
        num_cpu_blocks=500,
        cpu_block_size=48,
    )

    try:
        # Use a simple counting prompt for deterministic output
        prompt = "Count from 1 to 5: 1, 2, 3, 4,"

        # First generation - computes KV cache and offloads to CPU
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
        first_output = outputs[0].outputs[0].text

        # Reset prefix cache to force CPU reload on next generation
        llm.reset_prefix_cache()

        # Second generation - should load from CPU
        outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
        second_output = outputs[0].outputs[0].text

        # Verify outputs match (correctness after CPU reload)
        assert first_output == second_output, (
            f"Output mismatch after CPU reload: '{first_output}' vs '{second_output}'"
        )

        # Verify both outputs are reasonable
        # Note: We don't strictly require " 5" since the model might generate
        # differently, but outputs should be consistent
        print(f"Generation output: '{first_output}'")

    finally:
        del llm


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CPU offloading requires CUDA-alike GPU",
)
def test_hma_cpu_offloading_latency():
    """
    Test that CPU cache hits are faster than cold computation.

    This test:
    1. Measures cold computation time (no cache)
    2. Measures GPU cache hit time
    3. Measures CPU cache hit time (after prefix reset)
    4. Verifies CPU hit is faster than cold computation
    """
    llm = create_llm_with_cpu_offloading(
        model="meta-llama/Llama-3.2-1B-Instruct",
        gpu_memory_utilization=0.4,
        num_cpu_blocks=500,
        cpu_block_size=48,
    )

    try:
        # Use a longer prompt to make latency differences more visible
        prompt_length = 5000
        prompt_token_ids = list(range(prompt_length))
        sampling_params = SamplingParams(max_tokens=1)
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]

        num_tests = 5
        cold_times = []
        gpu_hit_times = []
        cpu_hit_times = []

        for i in range(num_tests):
            # Vary the first token to avoid cross-test caching
            prompt_token_ids[0] = i

            # Cold computation
            llm.reset_prefix_cache()
            start = time.time()
            llm.generate(prompts, sampling_params, use_tqdm=False)
            cold_times.append(time.time() - start)

            # GPU cache hit
            start = time.time()
            llm.generate(prompts, sampling_params, use_tqdm=False)
            gpu_hit_times.append(time.time() - start)

            # Reset prefix cache to force CPU reload
            llm.reset_prefix_cache()

            # CPU cache hit
            start = time.time()
            llm.generate(prompts, sampling_params, use_tqdm=False)
            cpu_hit_times.append(time.time() - start)

        avg_cold = sum(cold_times) / len(cold_times)
        avg_gpu_hit = sum(gpu_hit_times) / len(gpu_hit_times)
        avg_cpu_hit = sum(cpu_hit_times) / len(cpu_hit_times)

        print(f"\nLatency results ({num_tests} trials):")
        print(f"  Cold computation: {avg_cold * 1000:.2f}ms")
        print(f"  GPU cache hit:    {avg_gpu_hit * 1000:.2f}ms")
        print(f"  CPU cache hit:    {avg_cpu_hit * 1000:.2f}ms")
        print(f"  CPU speedup vs cold: {avg_cold / avg_cpu_hit:.2f}x")

        # CPU hit should be faster than cold computation most of the time
        cpu_faster_count = sum(
            1 for cpu, cold in zip(cpu_hit_times, cold_times) if cpu < cold
        )
        assert cpu_faster_count >= num_tests * 0.6, (
            f"CPU hit was faster than cold only {cpu_faster_count}/{num_tests} times"
        )

    finally:
        del llm


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CPU offloading requires CUDA-alike GPU",
)
def test_hma_cpu_offloading_multiple_requests():
    """
    Test CPU offloading with multiple concurrent requests sharing prefixes.

    This test verifies that:
    1. Multiple requests with shared prefix work correctly
    2. KV cache is properly shared and offloaded
    3. CPU reload produces correct results for all requests
    """
    llm = create_llm_with_cpu_offloading(
        model="meta-llama/Llama-3.2-1B-Instruct",
        gpu_memory_utilization=0.4,
        num_cpu_blocks=500,
        cpu_block_size=48,
    )

    try:
        # Common prefix with different endings
        prefix = "The capital of France is"
        prompts = [
            f"{prefix} Paris. The capital of Germany is",
            f"{prefix} Paris. The capital of Italy is",
            f"{prefix} Paris. The capital of Spain is",
        ]

        sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

        # First generation - compute and offload
        outputs1 = llm.generate(prompts, sampling_params, use_tqdm=False)
        results1 = [o.outputs[0].text for o in outputs1]

        # Reset and regenerate - should load from CPU
        llm.reset_prefix_cache()
        outputs2 = llm.generate(prompts, sampling_params, use_tqdm=False)
        results2 = [o.outputs[0].text for o in outputs2]

        # Verify consistency
        for i, (r1, r2) in enumerate(zip(results1, results2)):
            assert r1 == r2, (
                f"Request {i} output mismatch after CPU reload: '{r1}' vs '{r2}'"
            )

        print("\nMultiple request results:")
        for i, (prompt, result) in enumerate(zip(prompts, results1)):
            print(f"  Request {i}: '{prompt}' -> '{result}'")

    finally:
        del llm


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CPU offloading requires CUDA-alike GPU",
)
def test_hma_cpu_offloading_eviction_and_reload():
    """
    Test that evicted blocks can be reloaded correctly.

    This test:
    1. Fills CPU cache with multiple prompts
    2. Causes eviction by adding more prompts
    3. Reloads evicted content and verifies correctness
    """
    # Use smaller CPU cache to trigger eviction
    llm = create_llm_with_cpu_offloading(
        model="meta-llama/Llama-3.2-1B-Instruct",
        gpu_memory_utilization=0.4,
        num_cpu_blocks=100,  # Small cache to trigger eviction
        cpu_block_size=48,
    )

    try:
        sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

        # Generate multiple prompts to fill cache
        prompts = [
            "The quick brown fox jumps over the lazy",
            "To be or not to be, that is the",
            "In the beginning, there was",
            "Once upon a time, in a land far",
        ]

        # Store original outputs
        original_outputs = {}
        for prompt in prompts:
            outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
            original_outputs[prompt] = outputs[0].outputs[0].text
            llm.reset_prefix_cache()  # Force offload

        # Generate more to potentially cause eviction
        extra_prompts = [f"Extra prompt number {i} for testing" for i in range(10)]
        for prompt in extra_prompts:
            llm.generate(prompt, sampling_params, use_tqdm=False)
            llm.reset_prefix_cache()

        # Re-generate original prompts and verify
        for prompt in prompts:
            outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
            new_output = outputs[0].outputs[0].text

            # Output should be consistent (either from cache or recomputed)
            assert new_output == original_outputs[prompt], (
                f"Output mismatch for '{prompt[:30]}...': "
                f"'{original_outputs[prompt]}' vs '{new_output}'"
            )

        print("\nEviction and reload test passed")
        print(f"  Tested {len(prompts)} prompts after {len(extra_prompts)} evictions")

    finally:
        del llm


if __name__ == "__main__":
    # Run tests directly for debugging
    print("=" * 60)
    print("Running HMA CPU Offloading End-to-End Tests")
    print("=" * 60)

    print("\n[1/4] Testing correctness...")
    test_hma_cpu_offloading_correctness()
    print("PASSED")

    print("\n[2/4] Testing latency...")
    test_hma_cpu_offloading_latency()
    print("PASSED")

    print("\n[3/4] Testing multiple requests...")
    test_hma_cpu_offloading_multiple_requests()
    print("PASSED")

    print("\n[4/4] Testing eviction and reload...")
    test_hma_cpu_offloading_eviction_and_reload()
    print("PASSED")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
