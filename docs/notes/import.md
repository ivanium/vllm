现在有了完整的答案，三个问题逐一回答：

1. 什么时候开始 offload？
没有基于内存占用率的百分比阈值，有两种模式：

Eager 模式（默认）：无阈值，完全主动。每一步 _prepare_eager_store_specs 都扫描当前 running 请求里已确认计算完成的满块，只要 CPU 还有空闲 block 就直接塞：


# manager.py:465
num_free = cpu_block_pool.get_num_free_blocks()
...
if num_free <= 0:
    out_of_space = True
    break
num_free -= 1
不管 GPU 内存剩多少，新算完一个满块就立刻安排 offload。

Lazy 模式（需显式开启 lazy_offload=True）：有一个 _target_free，计算方式是：


# manager.py:191-201
WATERMARK_RATIO = 1.0
target = cdiv(max_num_batched_tokens, block_size)  # 每步最多消耗的 blocks
return int(target * (1 + WATERMARK_RATIO))          # = 2 × target
例如 max_num_batched_tokens=8192, block_size=16 → _target_free = 1024 blocks。每步从 GPU free queue 的 LRU 尾端走一遍，只要 free queue 里覆盖的块数不足 1024，就往 CPU offload，直到够为止。

2. 还有多少轮才真正满？
以 Lazy 模式为例估算（Eager 模式更难估算，因为它总是主动 offload）：


GPU 总 blocks = N
_target_free  = 2 × (max_num_batched_tokens / block_size)
每轮消耗 blocks ≈ max_num_batched_tokens / block_size = T

触发 offload 的时机：free blocks < 2T
"真正满"的时机：free blocks = 0（allocate_slots 返回 None）

从触发 offload 到真正满 ≈ 2T / T = 2 轮
这是理想情况——offload 能跟上消耗速度。如果 CPU 也满了（num_cpu_free=0），lazy offload 不会再 offload 新块，GPU free queue 继续缩小，还是会用完。

3. GPU 满了但 offload 还没完成，机制是什么？
这是最关键的部分，分两层：

第一层：GPU blocks 被 touch 保护

正在 DMA 中的 GPU blocks 在提交 copy 前会调用 touch()（ref_cnt +1），它们不在 free queue 里，allocator 无法分配它们。所以满的判断实际上排除了这些 in-flight blocks。

第二层：allocate_slots 失败 → preempt → 强制同步

当真的分配不到 block 时（allocate_slots 返回 None），scheduler 触发 preempt：


# scheduler.py:500-502
preempted_req = self.running.pop()
self._preempt_request(preempted_req, ...)  # free 其所有 blocks
被 preempt 的请求放回 waiting 队列，其 blocks 释放。SchedulerOutput.preempted_req_ids 非空，触发 Worker 侧在 forward 之前调用：


# gpu_model_runner.py:3812
get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
→ SimpleCPUOffloadWorker.handle_preemptions() 检查 need_flush=True：


# simple_kv_offload/worker.py:271
def handle_preemptions(self, metadata):
    if not metadata.need_flush:
        return
    self._flush_and_sync_all()   # 同步等待所有 load/store CUDA event 完成
_flush_and_sync_all() 对 _load_events 和 _store_events 里所有 event 逐个调 event.synchronize()，完全阻塞直到所有在途 DMA 结束，然后被 touch 的 GPU blocks 才真正释放回 free queue。

一句话总结：GPU 满时的兜底是 preempt + 强制全局 stream sync，代价是一个 step 的 stall（所有在途 DMA 都要等完），这正是 offload 比 baseline 有延迟的来源之一。

我说的满是指的是gpu的kvcache满，所以请调研一下，gpu的默认kvcache策略是什么，prefix tree以及LRU吗？然后cpu的默认kvcache策略又是什么？
The user is asking about the default KV cache eviction strategies for GPU and CPU in vLLM v1. Let me look at the relevant code.

Agent:KV cache eviction policy research
IN
在 /home/aoshen/setup_new_cluster/vllm/vllm/v1/core/ 目录下，调研 GPU KV cache 和 CPU KV cache（simple_kv_offload）的默认淘汰策略。

具体要找：
1. GPU block pool 的 free_block_queue 是什么数据结构，append/popleft 的顺序是 LRU 还是别的？是否有 prefix tree？evict 时选择哪个 block？
2. CPU block pool（SimpleCPUOffloadScheduler 里的 cpu_coordinator/cpu_block_pool）的淘汰策略，是否与 GPU 相同？
3. 当 prefix caching 开启时，cached_block 如何参与 free_block_queue？block 的 ref_cnt 变成 0 后是进 free queue 的头部还是尾部？

重点文件：
- vllm/v1/core/block_pool.py
- vllm/v1/core/kv_cache_coordinator.py （或类似名称）
- vllm/v1/core/kv_cache_utils.py 中 FreeKVCacheBlockQueue
- vllm/v1/simple_kv_offload/manager.py 中 cpu_coordinator 的初始化

只做代码阅读，不要修改任何文件。报告要精确：数据结构是什么，LRU/FIFO/其他，prefix tree 在哪里体现，GPU 和 CPU 策略是否一致。
Bash
IN
find /home/aoshen/setup_new_cluster/vllm/vllm/v1/core -type f -name "*.py" | head -20

OUT
Command running in background with ID: bqv8ys30m. Output is being written to: /tmp/claude-1557948685/-home-aoshen-setup-new-cluster/a81dc08b-a34a-470e-961b-956b1596ed42/tasks/bqv8ys30m.output
Bash
IN
ls -la /home/aoshen/setup_new_cluster/vllm/vllm/v1/core/ | grep -E "\.py$"

OUT
Command running in background with ID: bk5u3jga7. Output is being written to: /tmp/claude-1557948685/-home-aoshen-setup-new-cluster/a81dc08b-a34a-470e-961b-956b1596ed42/tasks/bk5u3jga7.output
Globpattern: "/home/aoshen/setup_new_cluster/vllm/vllm/v1/core/**/*.py"
Found 16 files
Read block_pool.py
Read kv_cache_utils.py
Grep "class FreeKVCacheBlockQueue" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/core/kv_cache_utils.py)
12 lines of output
Read kv_cache_utils.py (lines 159-458)
Read kv_cache_coordinator.py
Globpattern: "/home/aoshen/setup_new_cluster/vllm/vllm/v1/simple_kv_offload/**/*.py"
Found 7 files
Read manager.py
Read manager.py (lines 401-800)
Grep "class KVCacheBlock" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/core/kv_cache_utils.py)
7 lines of output
Read kv_cache_utils.py (lines 108-257)
Grep "def free_blocks" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/core/block_pool.py)
22 lines of output
Read block_pool.py (lines 410-459)
Bash
IN
grep -n "free_blocks\|free(" /home/aoshen/setup_new_cluster/vllm/vllm/v1/core/single_type_kv_cache_manager.py | head -30

OUT
Command running in background with ID: bg4yuur1u. Output is being written to: /tmp/claude-1557948685/-home-aoshen-setup-new-cluster/a81dc08b-a34a-470e-961b-956b1596ed42/tasks/bg4yuur1u.output
Grep "def free" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/core/single_type_kv_cache_manager.py)
64 lines of output
Read kv_cache_utils.py (lines 301-350)
Read single_type_kv_cache_manager.py (lines 277-306)
Grep "get_kv_cache_coordinator" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/core/kv_cache_coordinator.py)
12 lines of output
Read kv_cache_coordinator.py (lines 548-647)
Grep "prefix.*tree|trie|Trie" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/core)
5 lines of output
Grep "prefix.*tree|trie|Trie" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/core/kv_cache_utils.py)
8 lines of output
Grep "cpu_block_pool|cpu_coordinator" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/simple_kv_offload/manager.py)
80 lines of output
Read kv_cache_utils.py (from line [158, 175]1)
Read kv_cache_utils.py (lines 159-178)
Bash
IN
grep -n "ref_cnt.*0\|prefix.*cache\|cached_block" /home/aoshen/setup_new_cluster/vllm/vllm/v1/core/block_pool.py | head -20

OUT
Command running in background with ID: b064amnpg. Output is being written to: /tmp/claude-1557948685/-home-aoshen-setup-new-cluster/a81dc08b-a34a-470e-961b-956b1596ed42/tasks/b064amnpg.output
Read block_pool.py (from line [392, 408]1)
Read block_pool.py (lines 393-412)
Read block_pool.py (lines 353-402)
Read block_pool.py (lines 321-355)
The agent did a thorough investigation. Let me now provide a clear, concise summary to the user.

GPU KV Cache 默认策略
没有 Prefix Tree，用哈希表 + LRU 双向链表。

核心数据结构是 FreeKVCacheBlockQueue（kv_cache_utils.py），一个带虚拟头尾节点的双向链表，代码注释明确写道：


1. The least recent used block is at the front (LRU).
2. If two blocks have the same last accessed time, the one with
   more hash tokens (the tail of a block chain) is at the front.
淘汰：从头部 popleft()，最久未用的先被驱逐
释放：block ref_cnt 降到 0 时，append_n() 追加到尾部（最近释放的最后被淘汰）
Prefix caching：用 BlockHashToBlockMap（哈希表，不是树）做前缀命中查找。cached block 仍然留在 free queue 里，只是 evict 时会先调用 _maybe_evict_cached_block() 清掉哈希映射
所以"prefix tree"的说法并不准确，实际是哈希表做 O(1) 前缀命中查找 + LRU 双向链表决定驱逐顺序。

CPU KV Cache 策略
和 GPU 完全一致，CPU BlockPool 和 GPU BlockPool 是同一套代码，SimpleCPUOffloadScheduler 的初始化：


# simple_kv_offload/manager.py:110
self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
    kv_cache_config=self.cpu_kv_cache_config,
    enable_caching=True,   # 同样开了前缀缓存
    ...
)
self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool
CPU 侧独立维护一套 FreeKVCacheBlockQueue + BlockHashToBlockMap，block 数量由 CPU 内存容量决定（_derive_cpu_config 按比例换算 GPU/CPU 总字节数），淘汰顺序同样是 LRU。

对你之前问题的影响
GPU 满时触发 preempt 的根因是：free queue 里所有块要么 ref_cnt > 0（被 in-flight DMA touch），要么已经是 cached block 但都被 touch 住了。preempt 强制 _flush_and_sync_all() 之后，in-flight 的 GPU blocks 才从 touch 状态释放回 free queue，allocator 才能再次 popleft() 得到可用块。

1. cpu的ref_cnt是怎么增加和减少的，有request在cpu的时候，就+1?这个request fetch回gpu的时候，就-1？
2. dma touch是什么意思，解释一下
The user has two specific questions:

How does CPU block ref_cnt work - when does it increase/decrease?
What does "DMA touch" mean?
Let me read the relevant code carefully.

Grep "cpu_block_pool\.touch|cpu_pool\.touch|gpu_pool\.touch|\.touch\(" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/simple_kv_offload/manager.py)
39 lines of output
Grep "def touch|def free_blocks|def get_new_blocks|ref_cnt" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/core/block_pool.py)
91 lines of output
Read manager.py (lines 254-313)
Read manager.py (lines 626-725)
现在答案完全清晰了。

1. CPU ref_cnt 的增减时机
不是按 request 是否在 CPU 上来增减的，而是按 block 是否正在参与 DMA 传输来控制。具体有三个场景：

场景 A：GPU→CPU store（offload）

cpu_pool.get_new_blocks(n)   → cpu block ref_cnt: 0 → 1  (分配即+1)
gpu_pool.touch(gpu_blocks)   → gpu block ref_cnt: +1      (保护 DMA 源)
DMA 提交（异步执行）
...
store event 完成后:
  cpu_pool.free_blocks(cpu_blocks)  → cpu block ref_cnt: 1 → 0
  gpu_pool.free_blocks(gpu_blocks)  → gpu block ref_cnt: -1
CPU block 在 store 完成后 ref_cnt 降为 0，同时注册进 cached_block_hash_to_block，变成 LRU 尾部的 cached block——可以被命中，也可以被驱逐。

场景 B：CPU→GPU load（fetch）

cpu_pool.touch(cpu_blocks)   → cpu block ref_cnt: +1  (防止被驱逐)
gpu_pool.touch(gpu_blocks)   → gpu block ref_cnt: +1  (防止被覆写)
DMA 提交（异步执行）
...
load event 完成，_cleanup_load_request() 被调用:
  cpu_pool.free_blocks(cpu_blocks)  → cpu block ref_cnt: -1
  gpu_pool.free_blocks(gpu_blocks)  → gpu block ref_cnt: -1
场景 C：CPU block 被命中（prefix cache hit on CPU）
find_longest_cache_hit 返回的 cpu blocks 直接走场景 B 的 touch 路径，不会额外 +1。

2. "DMA touch" 的含义
touch() 的本质就是给 block 的 ref_cnt +1，并把它从 free queue 里摘出来：


# block_pool.py:392
def touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)  # 从 LRU 队列摘出
        block.ref_cnt += 1
为什么叫 "DMA touch"：DMA（直接内存访问）传输是异步的，在 copy 进行期间，源 block 和目标 block 的内存不能被别人碰。如果不把这些 block 从 free queue 摘出来，LRU evict 可能会把它们分配给别的 request，导致：

store 场景：GPU block 正在被 DMA 读取传往 CPU，但同时被 allocator 分配给新 request 写入 → 数据损坏
load 场景：CPU block 正在被 DMA 读取传往 GPU，但同时被 CPU LRU 逻辑驱逐并重用 → 读到错误数据；目标 GPU block 正在被写入，但可能同时被 forward 用于 attention 计算 → 数据混乱
touch() 通过 ref_cnt +1 把 block "钉住"，使它在 DMA 期间对 allocator 不可见。DMA 完成后，free_blocks() 把 ref_cnt -1，block 重新进入 free queue 尾部，恢复可分配/可驱逐状态。

一句话总结：CPU ref_cnt 不跟 request 绑定，而是跟 DMA 生命周期绑定。touch = ref_cnt+1 + 从 LRU 队列摘除；free = ref_cnt-1 + 若归零则插回 LRU 队列尾部。

所以ref count只可能是0和1吗？怎么理解这两种情况：GPU 满时触发 preempt 的根因是：free queue 里所有块要么 ref_cnt > 0（被 in-flight DMA touch），要么已经是 cached block 但都被 touch 住了
The user is asking whether ref_cnt can only be 0 or 1, and wants to understand the statement about GPU full triggering preempt. Let me think about this carefully.

ref_cnt can be > 1 in some cases:

When a block is shared between multiple requests (prefix caching - multiple requests share the same prefix blocks)
When a block gets touch()'d multiple times
Let me look at the code more carefully to understand when ref_cnt can be > 1.

Looking at the code:

get_new_blocks: ref_cnt 0→1

touch(): ref_cnt +1 (can go to 2, 3, etc.)

free_blocks(): ref_cnt -1

So ref_cnt can definitely be > 1 when multiple requests share the same cached prefix block or when a block is touched for DMA while also being used by a request. Now I need to reconsider the "GPU full triggers preempt" statement more carefully - my earlier characterization wasn't quite right. The free_block_queue only contains blocks with ref_cnt == 0, so I should think about when that queue becomes empty and what triggers preemption in that scenario. I see now—cached blocks sit in the free_block_queue with ref_cnt == 0 as eviction candidates, and when GPU memory is full, it means every block has ref_cnt > 0, either actively used by requests or touched for DMA operations. The "cached block but touched" scenario happens when a cached block gets touched either because another request hits it as a prefix cache or because it's being transferred to CPU, at which point its ref_cnt increments and it leaves the free queue.

So GPU full simply means all blocks are occupied with ref_cnt > 0—no available blocks in the free queue to allocate. A block can have ref_cnt > 1 when multiple requests share it as a prefix cache hit.

Looking at the allocate_slots code, when a request hits a cached block, it gets touched to increment ref_cnt, so if 5 requests share the same prefix block, that block's ref_cnt becomes 5. Let me trace through what happens with cached blocks more carefully.

A cached block starts in the free queue with ref_cnt=0. When allocated (popleft), ref_cnt becomes 1. After the request finishes, ref_cnt goes back to 0 and the block returns to the LRU tail. The blocks that get "pinned" are those with ref_cnt=0 in the free queue that get touched by either a DMA store operation (GPU block temporarily pinned while copying to CPU) or a prefix cache hit (block now held by a request with ref_cnt > 0, removed from the free queue).

Grep "ref_cnt|touch\b" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/core/kv_cache_manager.py)
19 lines of output
Grep "coordinator\.touch|block_pool\.touch|\.touch\(computed|touch.*computed_block|computed.*touch" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/core, glob: *.py)
11 lines of output
Read single_type_kv_cache_manager.py (lines 141-260)
现在答案非常清楚了。

ref_cnt 不只是 0 和 1
ref_cnt 可以是任意非负整数。两种情况会让它超过 1：

情况一：多个 request 共享同一个 prefix cached block

当 N 个 request 都命中同一段 prefix（比如同一个 system prompt），allocate_new_computed_blocks 里会对这些 cached blocks 调用 touch()：


# single_type_kv_cache_manager.py:191
# Touch the computed blocks to make sure they won't be evicted.
self.block_pool.touch(new_computed_blocks)
每个 request 进来命中都 +1，所以一个被 10 个 request 共享的 prefix block 的 ref_cnt = 10。等这些 request 都结束调用 free_blocks() 后才逐步 -1，降到 0 才回到 free queue。

情况二：DMA touch 叠加在已有 ref_cnt 上

一个正在被某 request 使用的 GPU block（ref_cnt=1），同时被安排 store 到 CPU，touch() 再 +1，此时 ref_cnt=2。

重新理解"GPU 满时触发 preempt 的根因"
我之前说的那句话其实不准确，正确的理解是：

GPU 满 = free_block_queue 为空，即所有 block 的 ref_cnt > 0。block 的 ref_cnt > 0 有且只有两个来源：


ref_cnt > 0 的原因
├── 1. 被 running request 持有
│        └── get_new_blocks()          → ref_cnt: 0→1
│            touch(prefix_hit_blocks) → ref_cnt: +1 per request
│
└── 2. 被 DMA 操作 touch 住
         └── store: gpu_pool.touch()  → 保护"即将被读的 GPU 源 block"
             load:  gpu_pool.touch()  → 保护"即将被写的 GPU 目标 block"
当 GPU 满时，两种情况都可能存在，但主要原因通常是第 1 种（running requests 消耗了所有 blocks），第 2 种（DMA touch）只在 in-flight DMA 期间临时占用少量 blocks。

之前说的"cached block 但都被 touch 住了"其实是一个误导性表述。准确说法是：cached block 本身 ref_cnt=0，它们就在 free_block_queue 里可以被驱逐，分配时 popleft() 会驱逐它们并把内存分配给新 request——这是正常的 LRU 驱逐路径。只有当这些 cached block 也被 touch（被某个 request 命中复用，或被 DMA 读取），它们才从 free queue 里消失、ref_cnt>0。

真正 GPU 满的场景：所有 block 都被 running requests 持有（ref_cnt≥1），free queue 里一个 cached block 也没有了，allocate_slots 拿不到新块，触发 preempt。

kv_cache_utils.py
我是说cpu中的ref count。怎么能够产生情况2的，就是说在eager的时候，一个block是cpu和gpu各会有一份对吗，然后正在被使用，确定分配后，就马上store到gpu，然后这个store的过程中，就会ref_cnt = 2?
The user is asking specifically about CPU ref_cnt, and asking about eager mode: when a block is in both CPU and GPU, and during the store (GPU→CPU) process, does the CPU block's ref_cnt become 2?

Let me re-read the eager store flow carefully:

In eager mode (_prepare_eager_store_specs):

A new GPU block has been computed (forward pass finished)

The scheduler wants to store it to CPU

cpu_block_pool.get_new_blocks(n) is called → new CPU block ref_cnt: 0→1

gpu_pool.touch(gpu_blocks) increments the GPU block's ref_cnt, then DMA transfers the data from GPU to CPU

Once the transfer completes in _process_store_completion(), the CPU block's ref_cnt drops back to 1 (moving it to the LRU cache as a stored block) while the GPU block's ref_cnt decreases by 1

So the CPU block never reaches ref_cnt of 2 during this flow—it starts at 1 when allocated and stays at 1 after the store completes.

For the load path, the sequence reverses: a request finds cached KV blocks on CPU (which have ref_cnt=0 in the LRU), touches them to increment ref_cnt to 1, then initiates a CPU→GPU DMA transfer. Once the load finishes, the cleanup process decrements the CPU block's ref_cnt back to 0, returning it to the free pool.

The question is whether a CPU block's ref_cnt can actually reach 2. This would only happen if two requests simultaneously touched the same CPU block for loading—but that's tricky because once a block is touched and removed from the free queue for a load operation, a second request shouldn't be able to touch it at the same time. I'm trying to figure out if there's a race condition where two requests could both initiate loads on the same CPU block concurrently. cached block? Looking at the code:

update_state_after_alloc is called per request
Each call does cpu_pool.touch(cpu_blocks_to_touch)
If two requests have the same prefix, they'd be loading the same CPU blocks
But actually, in find_longest_cache_hit, the same CPU block would be returned for both requests. Both would touch it, so ref_cnt=2. Both would then DMA it to their respective GPU blocks (different destination GPU blocks). When load completes for each, free_blocks is called, ref_cnt goes back down.

But actually this is the GPU side - both requests need to bring the SAME CPU block to DIFFERENT GPU blocks. The CPU block is just being READ, so two simultaneous reads are fine. The ref_cnt=2 just means two readers are holding it.

So yes, CPU ref_cnt CAN reach 2+ when multiple requests load the same cached CPU block at the same time.

Now for eager mode when a GPU block gets stored to CPU: a new CPU block is allocated that didn't previously exist, while the GPU block remains with an incremented ref_cnt. The DMA copies data from GPU to CPU, and once complete, the CPU block's ref_cnt drops from 1 to 0 (becoming cached) while the GPU block's ref_cnt decreases by 1. During the store operation itself, the GPU block has an elevated ref_cnt from the touch, but the newly allocated CPU block only has ref_cnt=1 since no other request is using it yet. GPU block stays with the running request throughout—the store operation just creates a CPU copy of that data for future prefix cache hits from other requests, while the GPU block continues serving its original request until completion.

让我具体看 eager store 的完整流程，确认 CPU block 的分配时机。

Read manager.py (lines 444-583)
现在完全清楚了。你的理解有两处需要纠正：

eager store 时，CPU block 的 ref_cnt 只是 1，不是 2
因为 CPU block 是全新分配的，不存在"被 request 使用"的情况：


_prepare_eager_store_specs() 里：

cpu_block_pool.get_new_blocks(n)   → CPU block ref_cnt: 0 → 1  (全新块)
gpu_pool.touch(gpu_blocks)         → GPU block ref_cnt: +1      (保护 DMA 源)

DMA 异步执行（GPU→CPU copy）...

store event 完成后 (_process_store_completion)：
  cpu_pool.free_blocks(cpu_blocks) → CPU block ref_cnt: 1 → 0
  gpu_pool.free_blocks(gpu_blocks) → GPU block ref_cnt: -1
  并将 CPU block 注册进 cached_block_hash_to_block
CPU block 整个生命周期 ref_cnt 都是 1，DMA 结束后降为 0，成为 LRU 尾部的 cached block。

关于"cpu 和 gpu 各有一份"
你说的是对的，但含义要准确：

GPU block：被 running request 持有（ref_cnt≥1），request 结束才释放
CPU block：是新分配的"副本"，DMA 完成后就独立存活（ref_cnt=0，在 CPU LRU 里等待被命中或驱逐）
store 操作是复制而不是移动。GPU block 完成 store 之后，GPU 上的数据仍然在，running request 继续用 GPU block 做 attention 计算；与此同时 CPU 上多了一份副本，供未来其他 request prefix cache 命中时 load 回 GPU。

CPU ref_cnt 什么时候会 = 2？
是 load 路径，且多个 request 同时命中同一个 CPU cached block：


request A: cpu_pool.touch(cpu_block)  → ref_cnt: 0 → 1
request B: cpu_pool.touch(cpu_block)  → ref_cnt: 1 → 2  （同一块数据）

两个 load DMA 各往不同的 GPU block 拷贝同一份 CPU 数据

A 的 load 完成: cpu_pool.free_blocks → ref_cnt: 2 → 1
B 的 load 完成: cpu_pool.free_blocks → ref_cnt: 1 → 0  → 回到 LRU
就是说cpu的block一直是0，除了load回gpu的时候被1-n个request命中，就ref count变成1-n了对吗
对，总结得很准确。

CPU block 的 ref_cnt 状态机就是这样：


get_new_blocks()           → ref_cnt: 0 → 1   (store DMA 期间)
store 完成 free_blocks()   → ref_cnt: 1 → 0   (进 LRU，成为 cached block)

[稳定状态：ref_cnt = 0，躺在 CPU LRU 里等待命中]

request A load 命中 touch() → ref_cnt: 0 → 1
request B load 命中 touch() → ref_cnt: 1 → 2
...
A load 完成 free_blocks()   → ref_cnt: n → n-1
B load 完成 free_blocks()   → ref_cnt: n-1 → n-2
...全部完成后回到 0
唯一补充：store 期间 ref_cnt 短暂变成 1，但这段时间它还不是 cached block（还没注册进哈希表），对外不可见，不会被任何 request 命中，所以这个 1 是"内部流程用"，不会和 load 的 1~n 叠加。

被load回gpu的cpu block，是复制吗对吗？
对，是复制。

load（CPU→GPU）和 store（GPU→CPU）都是 DMA copy，数据在两边都保留。

load 完成后：

GPU block：有了数据，被 request 用于 attention 计算
CPU block：数据还在，ref_cnt 降回 0，继续躺在 CPU LRU 里
所以同一份 CPU cached block 可以被多次 load，每次都是复制给不同的 GPU block，自身不消耗。只有 CPU LRU 空间不足时，CPU block 才会被驱逐（数据丢失）。

handoff_prompt.md
说明一下block从cpu到disk的状态机
The user wants to understand the state machine for blocks going from CPU to disk. This is related to the Mooncake disk offloading feature. Let me look at the relevant code - this would be in the Mooncake store worker since the disk offloading is handled by Mooncake's own mechanisms, not by vLLM's simple_kv_offload.

Let me check the Mooncake store worker for disk offload handling, and also check if there's any vLLM-native disk offload code.

Grep "disk|offload|ssd|OFFLOAD" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/simple_kv_offload)
5 lines of output
Globpattern: "/home/aoshen/setup_new_cluster/vllm/vllm/v1/kv_offload/**/*.py"
Found 16 files
Grep "disk|ssd|DISK|SSD" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/simple_kv_offload/manager.py)
1 line of output
Grep "disk|ssd|DISK|file_storage|offload_path" (in /home/aoshen/setup_new_cluster/vllm/vllm/v1/kv_offload)
1 line of output
There's no disk offload in the vLLM simple_kv_offload or kv_offload code. The disk offloading is handled entirely within Mooncake's own infrastructure. Let me look at the Mooncake-related code and the environment variables for disk offloading in the benchmark scripts.

The disk offloading for the Mooncake connector is managed by the Mooncake library itself - MooncakeDistributedStore. The enable_offload flag in the config and MOONCAKE_ENABLE_OFFLOAD env var enable disk offloading within Mooncake. vLLM doesn't have its own CPU→disk state machine.

Now I need to check the setup_vllm_env.sh script and the MooncakeStoreConfig to see how the disk offload mechanism is actually configured.

Read setup_vllm_env.sh
Grep "enable_offload|disk_offload|OFFLOAD|offload_buffer|staging" (in /home/aoshen/setup_new_cluster/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py)
61 lines of output
现在答案很清晰了。

CPU→Disk 没有 vLLM 自己的状态机
vLLM 代码里完全没有 CPU block → Disk 的逻辑。CPU 到 Disk 的 offload 是 Mooncake 内部自己管的，对 vLLM 完全透明。

两个独立的 offload 层次

GPU KV cache
    ↓  store (DMA copy, vLLM 的 simple_kv_offload 管)
CPU Memory (MooncakeDistributedStore 的 global_segment)
    ↓  evict to disk (Mooncake 内部自动管理，vLLM 不知道)
Disk (SSD)
第一层（GPU→CPU）：vLLM 的 SimpleCPUOffloadConnector 或 MooncakeStoreConnector 负责，有完整的 block ref_cnt、LRU、DMA 状态机。

第二层（CPU→Disk）：完全在 Mooncake 内部。vLLM 只是通过环境变量告诉 Mooncake "你可以用多少 disk"：


MOONCAKE_ENABLE_OFFLOAD=1
MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/mnt/data/...
MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=...
MOONCAKE_BUCKET_EVICTION_POLICY=lru          # disk 上的淘汰策略
MOONCAKE_USE_URING=true                      # io_uring 异步 I/O
Mooncake 自己决定什么时候把 CPU memory 里的数据写到 disk，vLLM 对此无感知。

vLLM 唯一感知 disk offload 存在的地方
KVCacheStoreSendingThread 里有一个背压检测，当 batch_put 返回 MOONCAKE_NO_AVAILABLE_HANDLE（CPU 内存不足，正在往 disk offload，暂时没有可用 handle），vLLM 会暂停该 request 的后续 store，等 disk offload 腾出空间后再恢复：


# mooncake_store_worker.py
if MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes:
    self._mark_request_skipped_for_pressure(req_id)
    # 跳过后续 store，直到下一次 store 成功才恢复
这就是 discuss.md 里 yk 说的"put 速度 > SSD offload 速度导致内存满"——vLLM 感知到的只是一个错误码，具体 CPU→Disk 的状态机完全在 Mooncake 侧。