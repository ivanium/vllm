# Comparison of Simple and Native Offload Logic

## 目的

这篇文档专门对比 vLLM 里的两套 CPU offload 实现：

- `OffloadingConnector`，下文简称 `native`
- `SimpleCPUOffloadConnector`，下文简称 `simple`

重点澄清 4 件事：

1. 两边如何理解 `cpu_bytes_to_use`
2. 两边如何计算 CPU block 容量
3. 两边的 CPU block id 和 TP rank 本地数据是什么关系
4. 两边什么时候真正发起 load/store 拷贝

## 一句话总结

两套实现做的是同一类事情，但思路不同：

- `native` 更像“先有一个逻辑 CPU block cache，再配一个 copy backend”
- `simple` 更像“先把本地 GPU/CPU block 搬运跑起来，再在上面叠 store/load 策略”

因此它们在预算语义、调度方式、拷贝时机、故障模式上都不一样。

## 总览对比

| 维度 | Native (`OffloadingConnector`) | Simple (`SimpleCPUOffloadConnector`) |
| --- | --- | --- |
| 顶层入口 | `OffloadingConnector` | `SimpleCPUOffloadConnector` |
| scheduler 核心 | `CPUOffloadingManager` + `CPUOffloadingSpec` | `SimpleCPUOffloadScheduler` |
| worker 核心 | `CpuGpuOffloadingHandlers` | `SimpleCPUOffloadWorker` |
| 拷贝后端 | `ops.swap_blocks` -> `cudaMemcpyAsync` | `DmaCopyBackend` |
| store 策略 | request/hash 驱动 | eager 或 lazy |
| `cpu_bytes_to_use` 默认语义 | server 级总预算 | server 级总预算 |
| 显式 per-rank 覆盖 | 无 | `cpu_bytes_to_use_per_rank` |
| load 发起时机 | `start_load_kv()` | `get_finished()` |
| store 发起时机 | `wait_for_save()` 后下一步提交 | `get_finished()` |
| 这次 native bug 是否存在 | 有，已修复 | 没有这个 bug |

## 共同的基本模型

两套实现有一个共同点：

- scheduler 侧维护一个“逻辑 CPU block id 空间”
- worker 侧每个 TP rank 都有自己本地的 CPU tensors
- 相同的逻辑 block id，会在每个 rank 上映射到本地 CPU tensor 的同一行号

也就是说，如果 scheduler 说：

```text
CPU block ids = [1280 .. 1535]
```

那它的含义不是：

- 全局只有一份 host memory，被所有 rank 共用

而是：

- rank0 把本地 CPU tensor 的第 `[1280 .. 1535]` 行当作目标
- rank1 也把自己本地 CPU tensor 的第 `[1280 .. 1535]` 行当作目标
- rank2 同理
- rank3 同理

所以：

- block id 空间是“逻辑共享”的
- 物理内存是“per-rank 本地”的

这点对理解为什么要乘 `world_size` 很关键。

## `cpu_bytes_to_use` 的语义

这是最容易混的地方。

## Native：把 `cpu_bytes_to_use` 当成整个 TP 组的总预算

native 走的是：

- `OffloadingConnector`
- `OffloadingSpecFactory`
- `CPUOffloadingSpec`

关键代码在：

- [spec.py](../vllm/v1/kv_offload/cpu/spec.py:17)

native 的核心公式是：

```python
kv_bytes_per_block = page_size_bytes * world_size
num_blocks = cpu_bytes_to_use // kv_bytes_per_block
```

这里的含义是：

- `page_size_bytes`：一个逻辑 block 在“单个 rank 本地”占多少字节
- `world_size`：有多少个 TP rank 都要各自存一份本地 KV 数据

因此：

```text
一个逻辑 CPU block 的总成本
= 单 rank 本地 block 成本 * TP rank 数
```

所以 native 默认把：

```text
cpu_bytes_to_use
```

理解成：

```text
整个 TP 组的总 CPU offload 预算
```

不是“每个 rank 各拿这么多”。

## Simple：先把总预算除以 `world_size`，再按本地视角算

simple 在 connector 入口就把预算拆成 per-rank：

- [simple_cpu_offload_connector.py](../vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:69)

核心逻辑是：

```python
cpu_capacity_bytes = cpu_bytes_to_use
cpu_capacity_per_rank = cpu_capacity_bytes // world_size
```

然后 scheduler 和 worker 都只拿这个 `cpu_capacity_per_rank` 继续算。

所以 simple 默认也是把：

```text
cpu_bytes_to_use
```

理解成：

```text
server 级 / 整个 TP 组的总预算
```

只是它把“除以 `world_size`”这一步放在 connector 入口做了。

另外 simple 还支持：

```text
cpu_bytes_to_use_per_rank
```

这个显式覆盖项；native 目前没有对应选项。

## 为什么 native 要乘 `world_size`

这里最容易产生误解。

先说通用事实：

- CPU offload 不是搞一个全局共享的 host tensor，而是每个 worker rank 自己分配本地 CPU tensors
- 所以一个“逻辑 CPU block”在物理上总会对应 `world_size` 份本地 CPU slot

但这 `world_size` 份本地数据，是否属于“冗余复制”，要看模型本身的 KV cache 语义：

- 对普通 multi-head / grouped-query / draft attention 这类 cache，常见情况是各 rank 持有不同 head shard
- 对 Kimi 这次主模型 MLA cache，则不是 head shard，而是每个 rank 都持有同一份 latent KV cache

也就是说，在这次 Kimi 实验里：

- draft attention 那层是 per-rank shard
- 主模型 61 层 MLA 才是大头，而且它在 TP=4 下确实是 4 份物理副本

所以 native 这里乘 `world_size`，在 Kimi 这个 case 里可以直接理解成：

```text
主 MLA cache 在 4 个 TP rank 上都各存一份
```

也正因为如此，整个 TP 组的总预算必须按 4 倍本地成本来算。

如果不乘 `world_size`，会发生什么？

假设单 rank 每个逻辑 block 是 `X` 字节，TP=4：

- 正确总成本应该是 `4X`
- 如果错用 `X` 去除，总 block 数会膨胀 4 倍
- 那每个 rank 都会分到过大的本地 CPU tensors
- 最后总 CPU 占用会接近“4 个 rank 每个都拿了一整份预算”

所以 native 这里乘 `world_size` 是为了保证：

```text
cpu_bytes_to_use 表示整个 TP 组的总预算
```

## Kimi 这个实验里，一个 block 到底有多大

这次 Kimi 的实验里，`group_page_size_bytes` 指的是：

```text
单个 rank 本地，一个逻辑 block 跨全部本地 KV tensors 的总字节数
```

不是整个 TP 组的总大小。

## 主模型 MLA 层

主模型 MLA 层的本地 shape 可以理解成：

```text
[num_blocks, 32, 576]
```

所以单层、单 rank、单 block 的大小是：

```text
32 * 576 * 1 = 18,432 bytes
```

其中：

- `32`：每个 block 里有 32 个 token
- `576`：MLA 的本地 latent 维度
- `1`：fp8，每元素 1 byte

## Kimi 主 MLA 层：为什么这里可以明确说是“冗余复制”

这点需要单独说清楚。

Kimi 主模型的 MLA cache 路径不是普通 attention 那种按 KV heads 切 TP 的形式，而是先生成 latent KV，再把 latent KV 写进 cache。

关键证据有 3 条：

1. 生成 KV latent 的投影 `kv_a_proj_with_mqa` 是 `ReplicatedLinear`，不是 `ColumnParallelLinear`。这意味着每个 TP rank 都持有同一份权重，并各自算出同一份输出：
   - [kimi_linear.py](../vllm/model_executor/models/kimi_linear.py:217)
   - [linear.py](../vllm/model_executor/layers/linear.py:287)

2. 主 MLA 层 forward 里，真正进入 cache 路径的是：

```python
kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
```

也就是 replicated 投影产出的 `kv_c + k_pe`：
   - [mla.py](../vllm/model_executor/layers/mla.py:141)
   - [mla.py](../vllm/model_executor/layers/mla.py:147)
   - [mla.py](../vllm/model_executor/layers/mla.py:150)

3. MLA cache 规格本身不带 TP shard 维度。vLLM 给 MLA 定义的是：
   - `num_kv_heads=1`
   - `kv_cache_shape = (num_blocks, block_size, head_size)`

对应代码：
   - [mla_attention.py](../vllm/model_executor/layers/attention/mla_attention.py:835)
   - [mla_attention.py](../vllm/model_executor/layers/attention/mla_attention.py:1132)

因此，对 Kimi 主 MLA 层来说，TP=4 时当前实现的物理存储语义就是：

```text
rank0 存一份 latent KV cache
rank1 再存一份同样的 latent KV cache
rank2 再存一份同样的 latent KV cache
rank3 再存一份同样的 latent KV cache
```

所以如果只看这次实验里真正的大头 61 个主 MLA 层，那么结论可以明确写成：

```text
当前实现下，Kimi 主 MLA 的 CPU offload 在 TP=4 时是 4x 物理冗余复制
```

draft attention 那层虽然不是冗余复制，而是 per-rank head shard，但它只占这次 block 成本中的小头，不影响对主成本结构的判断。

## Draft attention 层的 `131,072 B` 是什么意思

这个数字表示的是：

```text
draft attention 层在单个 rank 本地，一个 block 的字节数
```

它的本地 shape 可以理解成：

```text
[num_blocks, 2, 32, 16, 128]
```

每一维含义是：

- `2`：K 和 V 两份 cache
- `32`：每个 block 里 32 个 token
- `16`：当前 rank 上的 KV heads 数  
  这次总 KV heads=64，TP=4，所以每 rank 是 `64 / 4 = 16`
- `128`：每个 head 的维度
- `1`：fp8，每元素 1 byte

所以：

```text
page_size_draft_layer
= 2 * 32 * 16 * 128 * 1
= 131,072 bytes
```

这个 `131,072 B` 是单 rank 本地数值。  
如果要算整个 TP=4 组这个 draft layer 在一个逻辑 block 上的总成本，还要再乘 4。

## 本次实验里单 rank 的整体 block 大小

本次实验中：

- 61 个主模型 MLA 层
- 1 个 draft attention 层

所以单 rank 本地，一个逻辑 block 跨所有本地 KV tensors 的总大小是：

```text
61 * 18,432 + 131,072
= 1,255,424 bytes
```

这就是我们之前说的 `group_page_size_bytes`。

## 为什么 native 的正确值是 `85528`

因为当前是 TP=4，所以整个 TP 组一个逻辑 block 的总成本是：

```text
1,255,424 * 4 = 5,021,696 bytes
```

又因为配置里：

```text
cpu_bytes_to_use = 429,496,729,600   # 400 GiB
```

所以：

```text
num_blocks = floor(429,496,729,600 / 5,021,696)
           = 85,528
```

因此：

```text
85528
```

就是 native 在这次实验里的正确逻辑 block 容量。

换个角度看也很直观：

- 单 rank 本地：`1,255,424 * 85528 ≈ 100 GiB`
- 4 个 rank 合计：约 `400 GiB`

这和 `cpu_bytes_to_use=400 GiB` 完全对上。

## Native 的容量计算逻辑

native 的关键文件是：

- [spec.py](../vllm/v1/kv_offload/cpu/spec.py:17)
- [cpu_gpu.py](../vllm/v1/kv_offload/worker/cpu_gpu.py:374)

其流程是：

1. `CPUOffloadingSpec` 计算逻辑 block 数 `num_blocks`
2. worker 侧 `CpuGpuOffloadingHandlers` 在每个 rank 上都分配：

```python
cpu_tensor = torch.zeros((num_cpu_blocks, cpu_page_size_bytes), ...)
```

也就是：

- 每个 rank 都有自己的本地 CPU tensors
- 每个 rank 的 `num_cpu_blocks` 相同
- 但物理内存是各 rank 各自拥有

## Simple 的容量计算逻辑

simple 的关键文件是：

- [simple_cpu_offload_connector.py](../vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:69)
- [simple_kv_offload/manager.py](../vllm/v1/simple_kv_offload/manager.py:159)
- [simple_kv_offload/worker.py](../vllm/v1/simple_kv_offload/worker.py:153)

simple 的做法分两步：

1. connector 入口先做：

```python
cpu_capacity_per_rank = cpu_bytes_to_use // world_size
```

2. 然后 scheduler / worker 都只按“本 rank 本地 KV tensors”去算本地 block 数。

scheduler 侧：

```python
gpu_total_bytes = sum(t.size for t in gpu_config.kv_cache_tensors)
num_cpu_blocks = num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes
```

worker 侧：

```python
total_bytes_per_block = sum(local_unique_tensor_bytes_per_block)
self.num_cpu_blocks = self.cpu_capacity_bytes // total_bytes_per_block
```

所以 simple 本质上也是“总预算按 `world_size` 平摊到每个 rank”，只是实现形式不同于 native。

## Simple 会不会像 native 一样在 `num_layers` 维度算两遍

不会。

这是这次最重要的对比之一。

native 之前的 bug 本质上是：

```python
page_size_bytes   # 已经是整个 KV group 的聚合值
* len(kv_cache_tensors)   # 又把层数乘了一遍
```

而 simple 没有这条逻辑。

simple 的计算方式都是：

- scheduler 侧：把本地 KV tensors 的总 size 做一次 `sum(...)`
- worker 侧：把本地 unique tensor 的每 block 字节数做一次 `sum(...)`

它没有 native 这种：

```python
already_aggregated_group_page_size * number_of_layers
```

的重复计算。

所以：

- simple 有自己的语义和实现差异
- 但它没有 native 这次这个“层数重复计数”的 bug

## Store / Load 时机对比

## Native

关键文件：

- [offloading_connector.py](../vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:77)
- [offloading/worker.py](../vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:387)

native 的典型时序是：

```text
prepare_store_kv()
-> 先把 store job 放进 deferred queue
-> 下一步在 handle_preemptions() / start_kv_transfers() 中提交
-> worker.transfer_async()
-> swap_blocks()
```

也就是说：

- load/store 和 engine step 边界耦合得更紧
- store 是“延后一拍”提交的

## Simple

关键文件：

- [simple_cpu_offload_connector.py](../vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:145)
- [simple_kv_offload/worker.py](../vllm/v1/simple_kv_offload/worker.py:221)

simple 这几个接口基本是空的：

- `start_load_kv()`
- `wait_for_save()`

真正的 load/store 提交发生在：

```text
get_finished()
```

也就是 forward 结束之后。

因此 simple 的思路是：

- forward 先跑
- copy 提交放到 step 末尾
- 用后台 DMA thread 去跑拷贝

## Store 策略对比

## Native

native 更偏“逻辑 block cache”设计：

- 以 request / block hash 为中心
- 通过 `CPUOffloadingManager` 决定哪些 block 该 store / load / evict

关键文件：

- [cpu/manager.py](../vllm/v1/kv_offload/cpu/manager.py:24)

## Simple

simple 有两种 store 模式：

- eager：按 request 已确认完成的 block 去 store
- lazy：扫描 GPU free queue，优先处理靠近 eviction 前沿的 block

关键文件：

- [simple_kv_offload/manager.py](../vllm/v1/simple_kv_offload/manager.py:369)

所以 simple 更贴近：

```text
GPU block pool / eviction 行为
```

而 native 更贴近：

```text
逻辑 offload cache / request block hash
```

## 为什么这次 native bug 没出现在 simple

原因很直接：

1. simple 不走 `CPUOffloadingSpec`
2. simple 没有 native 那条“group page size 又乘一次 layer 数”的公式
3. simple 的本地容量计算只做一次 tensor 汇总，没有再额外乘层数

所以虽然两边最后都在做“GPU<->CPU KV block 搬运”，但：

- native 的 bug 出在“预算公式”
- simple 压根没用这套预算公式

## 最后的结论

### Native

- 更通用、更 spec-driven
- `cpu_bytes_to_use` 默认表示整个 TP 组总预算
- 在 `kv_bytes_per_block` 里乘 `world_size`
- 每个 rank 各自分配本地 CPU tensors
- 这次 bug 是因为把已经聚合过的 group page size 又按层数重复算了一次

### Simple

- 更直接、更本地化
- `cpu_bytes_to_use` 默认也表示整个 TP 组总预算
- 但在 connector 入口先除以 `world_size`
- 每个 rank 按自己的本地预算和本地 tensors 计算 block 数
- 没有 native 这次这个层数重复计数 bug

### 最重要的一句

两边都不是“全局一份 CPU cache 被所有 TP rank 共用”的设计。  
两边本质上都是：

```text
逻辑 block id 共享
物理 CPU tensors per-rank 本地
```

差别主要在于：

- native 用“总预算 / 整个 TP 组 block 成本”来算
- simple 用“总预算先平摊到 per-rank，再按本地 block 成本”来算
