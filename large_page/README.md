# [RFC]: Large-page KV cache allocator: one KV cache group per spec

**Status:** draft · **Implementation:** two stacked PRs to follow (PR1 opt-in, PR2 large pages only) · **Related:** [#62100](https://github.com/vllm-project/vllm/issues/62100) (pluggable KV cache planning)

## TL;DR

vLLM requires every KV cache group to have the same page size in bytes. For hybrid models, this one rule is the source of most of the grouping complexity in core: layer-count grouping, page padding, raised block sizes, `UniformTypeKVCacheSpecs`, and a second hash granularity (`hash_block_size`). Each new hybrid model, drafter or cache precision tends to need another special case.

This RFC proposes to remove the rule. The KV cache pool becomes a set of equal **large pages**. A page belongs to one KV cache group at a time, and the group cuts it into blocks of its own size. Then grouping becomes trivial: **one group per `KVCacheSpec`**, with one block size for every prefix-cacheable group. Block ids stay affine (block `b` is kernel row `b * R`), so attention kernels and slot mapping keep their upstream form.

KV cache capacity stays within 1% of main on the models we checked. The motivation is simpler code, not memory.

## 1. Background

Main stores the KV cache of each group in per-layer tensors that several groups share. A block id `b` addresses page `b` of every layer in its group, and one tensor can back one layer from each group. This sharing works only if all groups have the same page size in bytes. Core enforces this rule (`assert len(page_sizes) == 1`) and makes models satisfy it in three ways:

- **Group by layer count.** Layers of the same type are split into groups of equal size, so that each shared tensor holds one layer per group.
- **Raise the block size.** A narrow attention page grows in tokens until it is as wide as the widest page, for example a Mamba state.
- **Pad pages.** A page that still does not match is padded up to the widest page.

When different groups then have different block sizes, prefix caching hashes at a smaller common granularity (`hash_block_size`) and scales hits back up.

## 2. Problem

**The grouping is byte arithmetic, not model structure.** On Kimi-Linear-48B (TP4; 20 KDA layers, 7 MLA layers), one KDA state is 561,152 bytes. To give MLA the same page size, main raises the block size to 512 tokens (512 × 1,152 B = 589,824 B) and pads every KDA state by 5.1%. The 20 KDA layers become 3 groups because the MLA group has 7 layers. The model has two kinds of KV cache, but main runs four groups with 512-token blocks.

**Small changes re-cut the whole plan.** On DSV4.1-Flash, a change of the indexer precision from fp8 to mxfp4 changes the plan from 7 groups to 9, although no SWA layer changed ([#62100](https://github.com/vllm-project/vllm/issues/62100) has the details). A drafter can do the same (Kimi-K3 with DSpark: 4 groups become 20).

**The workarounds add concepts that every component must understand.** `UniformTypeKVCacheSpecs` merges layers of one type with different sizes into one spec. `hash_block_size` and `prefix_match_unit` add a third token granularity next to the scheduler block size and the group block size. Platforms override `_align_hybrid_block_size`, and models declare custom grouping through `SupportsKVCacheGrouping`.

[#62100](https://github.com/vllm-project/vllm/issues/62100) makes the grouping pluggable, so a model or platform can own its plan. That moves the heuristic out of core, but it keeps the equal-page rule that makes a heuristic necessary.

## 3. Goals and non-goals

Goals:

- One KV cache group per `KVCacheSpec`. Same-spec drafter layers join the target model's group.
- One block size `B` for all prefix-cacheable groups, so that a block hash means the same tokens in every group.
- No page padding, no block-size raise for byte reasons, no sub-block hashing.
- Kernel addressing, slot mapping and block tables keep their upstream form.
- KV cache capacity close to main, with a bounded loss.

Non-goals:

- Higher KV cache capacity. Capacity is a constraint, not the motivation.
- A new eviction policy. LRU over cached blocks stays.
- Changes to attention kernels, apart from 64-bit offsets where a pool can exceed 2 GiB per view.

## 4. Design

<p align="center"><img src="https://raw.githubusercontent.com/ivanium/vllm/845cdab0c6cc93a89ec6e52b874f784f50b25c88/large_page/fig-pages.svg" alt="Kimi-Linear-48B at TP4. Main pads every KDA state to the 512-token MLA page and runs four groups over seven shared per-layer tensors. With large pages, each 32.1 MiB page belongs to one of two groups and holds 3 KDA blocks or 64 MLA blocks of 64 tokens." width="100%"></p>

*Figure 1. Kimi-Linear-48B, TP4. Left: main pads every KDA state to the 512-token MLA page and runs four groups over shared per-layer tensors. Right: the pool is a set of 32.1 MiB large pages. Each page belongs to one group. The KDA group fits 3 blocks in a page, and the MLA group fits 64 blocks of 64 tokens.*

**Groups.** The planner makes one group for each distinct `KVCacheSpec` (after normalization) and kernel block size. A group's block holds the data of all its layers for `B` tokens, or one state for a state-based spec. On Kimi-Linear, the KDA block is 20 layers × 561,152 B = 10.7 MiB, and the MLA block is 7 layers × 64 tokens × 1,152 B = 504 KiB.

**Block size.** `B` is the least common multiple of the model's hard constraints (ring capacity, Mamba checkpoint alignment, DCP), raised to at least 64 tokens and then to the next size that every layer's attention kernel supports. An explicit `--block-size` is used as given, also below 64, if it is feasible. Groups that are not prefix-cacheable (rings, bounded-replay SWA, state pools) keep their own size.

**Large pages.** The pool is `N` pages of `S` bytes. Page 0 holds every group's null block. A page has exactly one owner group at a time; when all blocks of a page are free, the page returns to the shared free list and any group can take it. Ownership changes over time, so a block id is unique only within its group, and every component identifies a block as `(group_id, block_id)`.

**Page size.** `S` is the smallest multiple of any group's block size that satisfies two conditions: it is at least the largest block, and its demand-weighted waste is at most 2%. Waste is the page tail that a group cannot fill, weighted by how many blocks of each group one max-length request needs. The planner also checks that one max-length request fits on every rank. On Kimi-Linear, `S` = 3 KDA blocks = 32.1 MiB; the MLA group uses 64 blocks of a page and loses a 1.9% tail. The 2% bound is the only tuning constant.

<p align="center"><img src="https://raw.githubusercontent.com/ivanium/vllm/845cdab0c6cc93a89ec6e52b874f784f50b25c88/large_page/fig-ids.svg" alt="Affine block ids: block b starts at byte b times its block size, so its kernel row is b times R; ids that cross a page end are holes; each page belongs to one group." width="100%"></p>

*Figure 2. Affine block ids. Block `b` of a group starts at byte `b × block_bytes` of the pool, so the kernel row is `b × R`. An id that crosses a page end belongs to no page and is never allocated (a hole). The page owner decides which group's ids are live on that page.*

**Affine block ids.** Block `b` of group `g` occupies bytes `[b × block_bytes_g, (b + 1) × block_bytes_g)` of one pool buffer. Each layer gets one strided view over that buffer, with the stride equal to the group's row size. Thus the kernel row of block `b` is `b × R_g`, where `R_g` is the number of kernel blocks in one manager block. There is no lookup table and no second address formula. When `S` is not a multiple of a group's block size, the id that crosses a page end is a hole. The pool never allocates holes, and admission uses the minimum number of blocks per page.

**Rank-uniform geometry.** `S` and each group's row size are equal on all ranks; a group's row is its maximum over ranks. Therefore all TP and PP ranks, and both sides of a P/D pair, agree on block ids and page boundaries. Startup asserts this.

**Allocation.** A group takes a block from the first source that has one:

1. A free block on a page that the group already owns. The fullest page goes first, so that pages empty and return to the free list.
2. A free page with no cached blocks.
3. The older of two candidates: the group's least recently used cached block, or the oldest page in the free list. If the page wins, the pool evicts the cached blocks of all groups on that page.

**Admission.** `can_allocate` checks a request against free pages with a closed form, independent of the order in which groups then allocate. For each group, the uncovered demand is the demand minus the group's free blocks. This gives a lower and an upper page count, depending on how many cached blocks the group can reuse. The request is admitted if the free pages cover the sum of the upper counts minus the smallest slack (upper minus lower) of any group. A naive per-group sum is not sound, because one group's allocation can consume pages that another group counted. Models with one block per page per group take a scalar fast path.

**Prefix caching.** Every prefix-cacheable group hashes at `B`. Cache hits, KV events and connector transfers use whole blocks. Partial-block hits at a finer granularity are removed.

**Workers and connectors.** Workers allocate one buffer for the pool and bind per-layer views. Zeroing, copy-on-write, warmup and CUDA graph capture use the affine ids. Connectors receive per-group geometry (`kernel_rows_per_block`, `row_bytes`) and address blocks as `(group_id, block_id)`. NIXL sends the geometry in its handshake and rejects peers with a different version or geometry.

## 5. Costs

**Single-owner pages strand free blocks.** A page returns to the free list only when all its blocks are free. In CPU replays of multi-turn traffic with 4 to 8 GiB of KV cache, prefix hit rates were 5 to 6 points below an ideal LRU over the same bytes (Kimi-Linear, Qwen3.8-Flash-Next). Smaller pages would reduce the gap, but the largest state sets the minimum page size.

**Page tails.** By construction, the demand-weighted tail is at most 2%. On Kimi-Linear, the branch has 473.28x maximum concurrency at 32K tokens against 472.24x on main. Main pads each KDA state by 5.1%; the branch loses 1.6% (demand-weighted) to page tails.

**A coarser default block size.** Dense models move from 16-token blocks to 64-token blocks unless `--block-size` is set. Capacity does not change, but prefix hits round down to 64 tokens. A connector configured with a block size that is not a multiple of 64 now fails at startup.

**Compatibility.** The NIXL handshake version changes, so P and D instances must upgrade together. In PR2, external connectors that read the KV layout (LMCache, FlexKV, UMBP) need a matching release. Models that declare custom grouping (`SupportsKVCacheGrouping`) lose that hook.

## 6. Rollout

**PR1: opt-in allocator.** PR1 adds `--kv-cache-allocator large_page`. The default stays `uniform_page`. PR1 contains the planner, the page-owning block pool and admission, worker views, kernel changes for 64-bit offsets, NIXL and SimpleCPU offload. A platform opts in through `Platform.support_large_page_kv_cache()`. Startup rejects the unported cases: other connectors, NIXL host buffers, HiSparse and FlexAttention. PR1 has 7 commits: +3.4k/−0.6k lines in `vllm/` and about the same again in tests.

**PR2: one allocator.** PR2 makes large pages the only allocator. It ports the other platforms, every in-tree connector (Mooncake, Offloading, LMCache, HF3FS, MoRIIO, UMBP, FlexKV) and HiSparse. Then it deletes the equal-page path: `UniformTypeKVCacheSpecs`, `hash_block_size`, `prefix_match_unit`, page padding and the layer-count grouping. PR2 has 10 commits, one per area. Over the two PRs, `vllm/` gains 6.9k lines and loses 10.6k.

We propose to merge PR1 first and run it opt-in on the hybrid models for one release, then merge PR2.

**Validation.** At an earlier revision of the branches, GSM8K accuracy matched main on DSV4.1-Flash, GLM-5.3-Flash, Kimi-Linear-48B, Qwen3.8-Flash-Next and MiniMax-M3. NIXL P/D and SimpleCPU offload runs also passed GSM8K. The current revision adds affine ids and the 2% page rule. It passes the unit tests and CPU replays of the planner and the block pool, but GPU end-to-end runs are not done yet.

## 7. Alternatives

**Keep the equal-page rule and make grouping pluggable ([#62100](https://github.com/vllm-project/vllm/issues/62100)).** This lets a model or platform fix its own plan, but each new model still needs a plan, and core still needs `hash_block_size` for mixed block sizes. The two proposals compose: with one group per spec, a planner override is needed only for placement, not for grouping.

**Pack unequal pages into one block slot ([#64307](https://github.com/vllm-project/vllm/issues/64307), [#56156](https://github.com/vllm-project/vllm/pull/56156)).** These proposals give one block id a slot that holds every group's page at its exact size, so padding goes away. One block id still spans all groups, so the layer-count grouping and `hash_block_size` stay. This RFC removes the shared block id instead.

**Per-group pools with fixed sizes.** A static split of memory between groups is simple, but the right split depends on the workload. A wrong split wastes memory that the other group needs. Shared large pages move memory between groups at run time.

**Pages smaller than the largest state.** A state that spans several smaller pages would reduce stranding. It needs a block to map to more than one page, which breaks affine ids and puts a lookup back into every kernel. We think the stranding cost does not justify that today.

## 8. Open questions

1. **Default block size.** Is 64 tokens the right floor for dense models, or should the floor apply only to hybrid models?
2. **Performance.** Kimi-Linear now hashes 8 times more blocks per request (64-token instead of 512-token blocks). The per-step cost at the current revision is not yet measured on GPU. At an earlier revision, a prefill-heavy GLM-5.2 HiSparse workload was 4.7% slower than main.
3. **Release order of external connectors.** Should PR2 wait for LMCache and FlexKV releases, or merge with those connectors gated off?
4. **Out-of-tree models.** Do any out-of-tree models depend on `SupportsKVCacheGrouping` or on the KV events at sub-block granularity?

AI assistance (Claude) was used to prepare this RFC and the code.
