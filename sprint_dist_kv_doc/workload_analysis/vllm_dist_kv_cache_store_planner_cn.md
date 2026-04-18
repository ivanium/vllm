# [Internal] vLLM Distributed KV Cache Store Planner

- 来源：https://docs.google.com/document/d/1xMBWD_KFnroOHiD2gHg0sk3L1DufTqiRGqegYduFngs/edit?pli=1&tab=t.b4tw2dvgiuxz
- 同步到仓库：2026-04-14
- 注意：这是从连接的 Google 文档派生的本地 Markdown 快照，以便更轻松地在存储库内浏览。

## 规划

### vLLM分布式KV缓存存储规划

#### 总体目标

总体目标是使 vLLM 能够与多层分布式 KV 缓存存储一起运行，以避免多轮和代理工作负载的任何前缀缓存缺失。换句话说，系统提示和所有会话上下文都应该获得缓存命中。

此次冲刺计划紧凑，为期一个月，重点是将 Mooncake-store 集成到 vLLM 中并优化性能。在冲刺结束时，我们将交付：

- 开源代码
- 具有代表性的代理工作负载
- 以及一篇博客文章：
  - 总结并突出技术和优化
  - 我们的绩效数据
  - 以及人们尝试代码并重现结果的说明

#### 非目标

- 我们主要关注 vLLM 端的优化，而不是 Mooncake 内部的优化，除非它们能解锁关键的性能改进。
- 我们将寻求月饼团队的帮助进行月饼端的优化。

#### 有针对性的硬件设置和性能目标

硬件：

- 主要针对GB200进行多节点NVLink优化

设置：

- 单节点DP
- PD
- vLLM路由器+多节点DP

绩效目标：

- 指标：吞吐量、延迟、前缀缓存命中率
- 分布式 KV 存储的 SLO：
  - 当没有前缀缓存命中时零开销
  - 根据空间允许缓存所有有用的内容
  - 一个节点存储的缓存应该尽快对其他节点可见
- E2E SLA目标：Crusoe工作负载#### 支持的关键型号

- DSV3.2
- 基米2.5
-（可选）Qwen3.5

### 关键里程碑和时间表

#### 0.5：模拟代理工作负载和基准测试工具

紫荆已经收集到了药剂痕迹，并进行了设置。

#### 1：基本月饼店集成到 vLLM

- 实现月饼店连接器
- 支持RDMA和TCP以及磁盘后端
- 使用单节点 DP 进行验证
- 验证DP+EP和多个vLLM DP实例可以共享KV缓存内容
- 从 Llama 和 Qwen3 开始，然后尝试 Kimi-2.5 NVFP4

#### 2：分布式KV存储+PD

- 与 vLLM MultiConnector 良好集成
- 解码节点可以从分布式存储中获取部分缓存命中，并从预填充节点中获取剩余的KV
- 支持预填充/解码节点的不同 TP 和 KV 缓存布局

#### 3：分布式KV存储+用于多节点DP的vLLM路由器

- 路由器查询 Mooncake master 的前缀缓存位置
- 需要缓存感知的负载平衡策略

#### 4：性能优化

- 分析和优化：
  - 控制路径通信
  - 数据路径通信
  - 路由器集成
  - 缓存驱逐策略

## 开发结果

### 2026 年 4 月 9 日

Kimi-k2.5、DP2 TP4：基准测试结果显示 Mooncake 提高了吞吐量和平均延迟，而一些尾部指标仍然较差或持平。

### 2026 年 4 月 8 日

单节点DP2和TP2实验：

- 基线与 Mooncake 内存卸载
- 有和没有跨层
- 报告的运行中 TTFT 和 E2EL 得到显着改进

## 会议记录

### 2026 年 4 月 13 日 | Dist KV 商店站立

要点：

- Eagle 推测解码在随机数据集上的接受率较低
- NIXL使用UCX进行PD数据传输
- NIXL 连接器初始化期间出现段错误，并非特定于卸载
- 工作负载定义需要统一和校准
- 计划对更大的 PD 和 MNDP 部署进行基准测试### 2026 年 4 月 9 日

CPU-GPU 数据路径注意事项：

- C2C 带宽较高，但小量传输的延迟可能并不比 PCIe 更好
- 当协议为 RDMA 时，即使本地 CPU 发起的缓存负载仍可能涉及 NIC 路径
- 需要更多基准测试来量化 CPU-GPU 通信延迟

### 2026 年 4 月 8 日 |区月饼店同步

当前限制：

- 单节点、多 GPU：CPU 内存卸载有效，磁盘卸载无效
- 多节点，每个节点 1 个 GPU：CPU 和磁盘卸载工作
- 多节点、多GPU：CPU内存卸载有效

附加说明：

- 需要可运行的设置而不会崩溃
- 需要更好地观察 PD 带宽和传输行为
- 需要足够大的基准设置来触发卸载和外部缓存命中

## 开发运行手册

### 设置

开发分支：

- https://github.com/ivanium/vllm/tree/feat/mooncake-store-int

基本设置摘要：

- 克隆“守夜”
- 将`ivanium/vllm`克隆为`vllm-mooncake`
- 结帐`feat/mooncake-store-int`
- 创建 `uv` Python 3.12 环境
- 安装可编辑的vLLM
- 安装月饼传送引擎轮
- 安装 Kimi sprint 依赖项：
  - `nixl[cu13]`
  - `flashinfer-cubin==0.6.7`
  - `flashinfer-jit-cache==0.6.7`
  - `fastsafetensors`
- 克隆“路由器内部”
- 克隆“crusoe-inference”
- 克隆并安装`vllm-bench`

### PD 分解设置

- 在“recipes/crusoe/kimik25/”下创建“dist_kv/”食谱
- 使用“pd_dev.yaml”作为设置参考
- 为 Mooncake 环境变量和主启动添加更详细的说明

### 多节点 DP 设置

- 在`recipes/crusoe/kimik25/`下为DP创建`dist_kv/`食谱
- 使用`tp4_eagle_fa4_offloading_c8.yaml`作为参考
- 为 Mooncake 环境变量和主启动添加更详细的说明