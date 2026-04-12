# SiMM 替代 Mooncake 的性能结论

只看性能，SiMM 最值得强调的优势只有一句话：

**SiMM 没有中心化元数据服务，IO 路径更短。**

作者给出的解释是，Mooncake 的路径更接近：

`client -> master/metadata -> data node`

而 SiMM 做了全局 shard 分区，client 本地 hash 后可直接找到目标 data server，因此更接近：

`client -> data server`

这意味着 SiMM 的远端 KV hit 更容易少一次控制面交互，固定开销更低。  
在 vLLM 的 KV cache 命中场景里，TTFT 往往就输赢在这种固定开销上，所以这应当作为第一结论。

在这个核心优势之上，README 里另外 3 个点可以作为增强项：

1. **小 IO batch**
SiMM 明确强调会把小 KV I/O 聚合成 batch，用来摊薄每个 chunk 的请求开销。这对 KV cache reload 很重要。

2. **端到端 zero-copy**
SiMM 强调 `end-to-end zero-copy` 和 MR buffer，目的是减少额外 copy，避免命中后时间又浪费在数据搬运上。

3. **充分利用全部 RDMA NIC**
SiMM 强调 `ALL RDMA NICs`，这会进一步提高大对象读取和高并发下的带宽上限。

所以文档最精简的表达应该是：

**SiMM 最可能更快的根本原因，是它把 Mooncake 更接近“两跳”的 IO 路径，缩成了“一跳”；batch、zero-copy 和多 NIC 利用，是建立在这一点之上的进一步放大。**

需要同时注明边界：

- SiMM 目前主要是分布式内存 KV IO 能力
- 现在是通过 LMCache 接入 vLLM
- 相关接入 PR 还没有提交到社区
- 这个设计是为了性能做 tradeoff，资源管控能力没有 Mooncake 那么强

## 参考

- [README.md](/home/aoshen/setup_new_cluster/SiMM/README.md)
- [clnt_messenger.cc](/home/aoshen/setup_new_cluster/SiMM/src/client/clnt_messenger.cc)
- [mooncake-store.md](/home/aoshen/setup_new_cluster/Mooncake/docs/source/design/mooncake-store.md)
- [master_client.cpp](/home/aoshen/setup_new_cluster/Mooncake/mooncake-store/src/master_client.cpp)
