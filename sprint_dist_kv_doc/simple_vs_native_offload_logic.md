# Comparison of Simple and Native Offload Logic

## Purpose

This document specifically compares two sets of CPU offload implementations in vLLM:

- `OffloadingConnector`, hereafter referred to as `native`
- `SimpleCPUOffloadConnector`, hereafter referred to as `simple`

Key 4 things to clarify:

1. How do both sides understand `cpu_bytes_to_use`
2. How to calculate CPU block capacity on both sides
3. What is the relationship between the CPU block id and TP rank local data on both sides?
4. When do both parties actually initiate load/store copying?

## One sentence summary

The two sets of implementations do the same type of thing, but with different ideas:

- `native` is more like "first have a logical CPU block cache, and then configure a copy backend"
- `simple` is more like "first move and run the local GPU/CPU block, and then stack the store/load strategy on top"

Therefore, they are different in budget semantics, scheduling methods, copy timing, and failure modes.

## Overview comparison

| Dimensions | Native (`OffloadingConnector`) | Simple (`SimpleCPUOffloadConnector`) |
| --- | --- | --- |
| Top-level entry | `OffloadingConnector` | `SimpleCPUOffloadConnector` |
| scheduler core | `CPUOffloadingManager` + `CPUOffloadingSpec` | `SimpleCPUOffloadScheduler` |
| worker core | `CpuGpuOffloadingHandlers` | `SimpleCPUOffloadWorker` |
| Copy backend | `ops.swap_blocks` -> `cudaMemcpyAsync` | `DmaCopyBackend` |
| store strategy | request/hash driver | eager or lazy |
| `cpu_bytes_to_use` default semantics | server-level total budget | server-level total budget |
| Explicit per-rank override | None | `cpu_bytes_to_use_per_rank` |
| Load initiation time | `start_load_kv()` | `get_finished()` |
| store initiation time | `wait_for_save()` followed by next submission | `get_finished()` |
| Does the native bug exist this time | Yes, it has been fixed | There is no such bug |

## Common basic model

The two sets of implementations have one thing in common:

- The scheduler maintains a "logical CPU block id space"
- Each TP rank on the worker side has its own local CPU tensors
- The same logical block id will be mapped to the same row number of the local CPU tensor on each rank.

That is, if the scheduler says:```text
CPU block ids = [1280 .. 1535]
```
Then its meaning is not:

- There is only one global host memory, which is shared by all ranks.

Instead:

- rank0 targets row `[1280 .. 1535]` of the local CPU tensor
- rank1 also takes the `[1280 .. 1535]` row of its own local CPU tensor as the target
- rank2 Same reason
- rank3 Same reason

So:

- The block id space is "logically shared"
- Physical memory is "per-rank local"

This is key to understanding why `world_size` is multiplied.

## Semantics of `cpu_bytes_to_use`

This is the easiest place to get confused.

## Native: Treat `cpu_bytes_to_use` as the total budget of the entire TP group

native goes like this:

- `OffloadingConnector`
- `OffloadingSpecFactory`
- `CPUOffloadingSpec`

The key code is:

- [spec.py](../vllm/v1/kv_offload/cpu/spec.py:17)

The core formula of native is:```python
kv_bytes_per_block = page_size_bytes * world_size
num_blocks = cpu_bytes_to_use // kv_bytes_per_block
```
The meaning here is:

- `page_size_bytes`: How many bytes does a logical block occupy in "single rank local"
- `world_size`: How many TP ranks there are, each of which must store a copy of local KV data

Therefore:```text
Total cost of a logical CPU block
= Single rank local block cost * TP rank number
```
So native defaults to:```text
cpu_bytes_to_use
```
understood as:```text
Total CPU offload budget for the entire TP group
```
It's not "each rank gets this much".

## Simple: First divide the total budget by `world_size`, and then calculate it from the local perspective

Simple splits the budget into per-rank at the connector entry:

- [simple_cpu_offload_connector.py](../vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:69)

The core logic is:```python
cpu_capacity_bytes = cpu_bytes_to_use
cpu_capacity_per_rank = cpu_capacity_bytes // world_size
```
Then both scheduler and worker only use this `cpu_capacity_per_rank` to continue calculation.

So simple also defaults to:```text
cpu_bytes_to_use
```
understood as:```text
Server level / total budget of the entire TP group
```
It's just that it puts the "dividing by `world_size`" step at the connector entrance.

In addition, simple also supports:```text
cpu_bytes_to_use_per_rank
```
This is an explicit override; native currently has no corresponding option.

## Why native needs to be multiplied by `world_size`

Misunderstandings are most likely to occur here.

Let’s talk about the general facts first:

- CPU offload does not create a globally shared host tensor, but each worker rank allocates local CPU tensors.
- So a "logical CPU block" physically always corresponds to `world_size` local CPU slots

Whether this `world_size` local data is a "redundant copy" depends on the KV cache semantics of the model itself:

- For caches such as ordinary multi-head / grouped-query / draft attention, the common situation is that each rank holds different head shards
- For Kimi's main model MLA cache this time, it is not the head shard, but each rank holds the same latent KV cache.

That is to say, in this Kimi experiment:

- The draft attention layer is per-rank shard
- The main model 61 layer MLA is the big one, and it is indeed 4 physical copies at TP=4

So native multiplied by `world_size` here can be directly understood as:```text
The main MLA cache stores one copy on each of the four TP ranks.
```
Because of this, the total budget for the entire TP group must be calculated as 4 times the local cost.

What happens if we don't multiply by `world_size`?

Assume that each logical block of a single rank is `X` bytes, TP=4:

- The correct total cost should be `4X`
- If you use `X` to remove incorrectly, the total number of blocks will expand 4 times
- Then each rank will be allocated too large local CPU tensors
- In the end, the total CPU usage will be close to "each of the 4 ranks takes a whole budget"

So native is multiplied by `world_size` here to ensure:```text
cpu_bytes_to_use represents the total budget of the entire TP group
```
## Kimi In this experiment, how big is a block?

In Kimi’s experiment this time, `group_page_size_bytes` refers to:```text
A single rank local, the total number of bytes across all local KV tensors in a logical block
```
Not the total size of the entire TP group.

## Main model MLA layer

The local shape of the main model MLA layer can be understood as:```text
[num_blocks, 32, 576]
```
So the size of a single layer, single rank, and single block is:```text
32 * 576 * 1 = 18,432 bytes
```
Among them:

- `32`: There are 32 tokens in each block
- `576`: local latent dimension of MLA
- `1`: fp8, 1 byte per element

## Kimi Main MLA Layer: Why it can be clearly said that it is "redundant replication"

This needs to be stated clearly separately.

The MLA cache path of Kimi's main model is not the ordinary attention method of cutting TP by KV heads. Instead, the latent KV is first generated and then the latent KV is written into the cache.

There are 3 key pieces of evidence:

1. The projection `kv_a_proj_with_mqa` that generates KV latent is `ReplicatedLinear`, not `ColumnParallelLinear`. This means that each TP rank holds the same weight and calculates the same output:
   - [kimi_linear.py](../vllm/model_executor/models/kimi_linear.py:217)
   - [linear.py](../vllm/model_executor/layers/linear.py:287)

2. In the main MLA layer forward, the actual cache path is:```python
kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
```
That is, the `kv_c + k_pe` output by replicated projection:
   - [mla.py](../vllm/model_executor/layers/mla.py:141)
   - [mla.py](../vllm/model_executor/layers/mla.py:147)
   - [mla.py](../vllm/model_executor/layers/mla.py:150)

3. The MLA cache specification itself does not have the TP shard dimension. vLLM defines MLA as:
   - `num_kv_heads=1`
   - `kv_cache_shape = (num_blocks, block_size, head_size)`

Corresponding code:
   - [mla_attention.py](../vllm/model_executor/layers/attention/mla_attention.py:835)
   - [mla_attention.py](../vllm/model_executor/layers/attention/mla_attention.py:1132)

Therefore, for the Kimi main MLA layer, the currently implemented physical storage semantics when TP=4 is:```text
rank0 saves a latent KV cache
rank1 saves another copy of the same latent KV cache
rank2 saves another copy of the same latent KV cache
rank3 saves another copy of the same latent KV cache
```
So if we only look at the real 61 main MLA layers in this experiment, the conclusion can be clearly written as:```text
Under the current implementation, the CPU offload of the Kimi primary MLA is 4x physical redundant replication when TP=4
```
Although the draft attention layer is not a redundant copy but a per-rank head shard, it only accounts for a small portion of the block cost and does not affect the judgment of the main cost structure.

## What does `131,072 B` in the Draft attention layer mean?

What this number represents is:```text
The draft attention layer is local to a single rank, the number of bytes of a block
```
Its local shape can be understood as:```text
[num_blocks, 2, 32, 16, 128]
```
The meaning of each dimension is:

- `2`: K and V two caches
- `32`: 32 tokens in each block
- `16`: the number of KV heads on the current rank
  This time the total KV heads=64, TP=4, so each rank is `64 / 4 = 16`
- `128`: Dimensions of each head
- `1`: fp8, 1 byte per element

So:```text
page_size_draft_layer
= 2 * 32 * 16 * 128 * 1
= 131,072 bytes
```
This `131,072 B` is a single rank local value.  
If you want to calculate the total cost of the entire TP=4 draft layer on a logical block, you need to multiply it by 4.

## The overall block size of a single rank in this experiment

In this experiment:

- 61 master model MLA layers
- 1 draft attention layer

So the total size of a single rank local, one logical block across all local KV tensors is:```text
61 * 18,432 + 131,072
= 1,255,424 bytes
```
This is the `group_page_size_bytes` we talked about before.

## Why the correct value of native is `85528`

Because the current TP=4, the total cost of a logical block in the entire TP group is:```text
1,255,424 * 4 = 5,021,696 bytes
```
And because in the configuration:```text
cpu_bytes_to_use = 429,496,729,600   # 400 GiB
```
so:```text
num_blocks = floor(429,496,729,600 / 5,021,696)
           = 85,528
```
therefore:```text
85528
```
This is the correct logical block capacity of native in this experiment.

It’s also very intuitive when viewed from another angle:

- Single rank local: `1,255,424 * 85528 ≈ 100 GiB`
- Total of 4 ranks: about `400 GiB`

This is exactly the same as `cpu_bytes_to_use=400 GiB`.

## Native’s capacity calculation logic

The key files of native are:

- [spec.py](../vllm/v1/kv_offload/cpu/spec.py:17)
- [cpu_gpu.py](../vllm/v1/kv_offload/worker/cpu_gpu.py:374)

The process is:

1. `CPUOffloadingSpec` calculates the number of logical blocks `num_blocks`
2. The worker side `CpuGpuOffloadingHandlers` are allocated on each rank:```python
cpu_tensor = torch.zeros((num_cpu_blocks, cpu_page_size_bytes), ...)
```
That is:

- Each rank has its own local CPU tensors
- The `num_cpu_blocks` of each rank are the same
- But the physical memory is owned by each rank.

## Simple capacity calculation logic

The key files for simple are:

- [simple_cpu_offload_connector.py](../vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:69)
- [simple_kv_offload/manager.py](../vllm/v1/simple_kv_offload/manager.py:159)
- [simple_kv_offload/worker.py](../vllm/v1/simple_kv_offload/worker.py:153)

The simple method is divided into two steps:

1. Do the connector entrance first:```python
cpu_capacity_per_rank = cpu_bytes_to_use // world_size
```
2. Then the scheduler/worker only calculates the number of local blocks based on "local KV tensors of this rank".

scheduler side:```python
gpu_total_bytes = sum(t.size for t in gpu_config.kv_cache_tensors)
num_cpu_blocks = num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes
```
worker side:```python
total_bytes_per_block = sum(local_unique_tensor_bytes_per_block)
self.num_cpu_blocks = self.cpu_capacity_bytes // total_bytes_per_block
```
So simple is essentially "the total budget is evenly distributed to each rank according to `world_size`", but the implementation form is different from native.

## Will Simple count twice in the `num_layers` dimension like native?

No.

This is one of the most important comparisons of this time.

The bug before native was essentially:```python
page_size_bytes # Already the aggregate value of the entire KV group
* len(kv_cache_tensors) # Multiply the number of layers again
```
Simple does not have this logic.

The calculation methods of simple are:

- Scheduler side: Do a `sum(...)` on the total size of local KV tensors
- Worker side: Do a `sum(...)` of the number of bytes per block of the local unique tensor

It doesn't have this kind of native:```python
already_aggregated_group_page_size * number_of_layers
```
of double counting.

So:

- simple has its own semantics and implementation differences
- But it doesn't have the native "layer double counting" bug this time

## Store / Load timing comparison

## Native

Key documents:

- [offloading_connector.py](../vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:77)
- [offloading/worker.py](../vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:387)

Typical timing for native is:```text
prepare_store_kv()
-> First put the store job into the deferred queue
-> Next step is to submit in handle_preemptions() / start_kv_transfers()
-> worker.transfer_async()
-> swap_blocks()
```
That is to say:

-Load/store and engine step boundaries are more tightly coupled
- store is submitted "one beat later"

## Simple

Key documents:

- [simple_cpu_offload_connector.py](../vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:145)
- [simple_kv_offload/worker.py](../vllm/v1/simple_kv_offload/worker.py:221)

These simple interfaces are basically empty:

- `start_load_kv()`
- `wait_for_save()`

The real load/store commit happens at:```text
get_finished()
```
That is, after forward ends.

So the idea of simple is:

- forward run first
- Copy and submit at the end of step
- Use background DMA thread to run copy

## Store strategy comparison

## Native

Native is more of a "logical block cache" design:

- Centered on request / block hash
- Determine which blocks should store / load / evict via `CPUOffloadingManager`

Key documents:

- [cpu/manager.py](../vllm/v1/kv_offload/cpu/manager.py:24)

## Simple

simple has two store modes:

- eager: Go to the store according to the confirmed completed block of request
- lazy: scan the GPU free queue and prioritize blocks close to the eviction frontier

Key documents:

- [simple_kv_offload/manager.py](../vllm/v1/simple_kv_offload/manager.py:369)

So simple is closer:```text
GPU block pool / eviction behavior
```
And native is closer:```text
Logical offload cache / request block hash
```
## Why does the native bug not appear in simple this time?

The reason is very straightforward:

1. simple does not use `CPUOffloadingSpec`
2. Simple does not have the native formula of "group page size multiplied by the number of layers"
3. The local capacity calculation of simple only performs tensor summary once, without multiplying the number of layers.

So although both sides are finally doing "GPU<->CPU KV block transfer", but:

- The native bug is in the "budget formula"
- simple does not use this budget formula at all

## Final conclusion

### Native

- More versatile and spec-driven
- `cpu_bytes_to_use` defaults to the total budget of the entire TP group
- multiply `world_size` in `kv_bytes_per_block`
- Each rank is allocated local CPU tensors
- This bug is caused by recalculating the group page size that has been aggregated by the number of layers.

### Simple

- More direct and localized
- `cpu_bytes_to_use` also represents the total budget of the entire TP group by default
- But first divide by `world_size` at the connector entry
- Each rank calculates the number of blocks based on its own local budget and local tensors
- There is no native layer duplication counting bug this time

### The most important sentence

Neither side is a design where "a global CPU cache is shared by all TP ranks".  
Both sides are essentially:```text
Logical block id sharing
Physical CPU tensors per-rank local
```
The main differences are:

- native is calculated using "total budget / entire TP group block cost"
- simple is calculated using "the total budget is first spread equally to per-rank, and then based on the local block cost"