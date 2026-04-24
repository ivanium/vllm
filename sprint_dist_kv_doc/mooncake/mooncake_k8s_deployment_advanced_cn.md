---
title: Mooncake k8s 部署进阶 (HA Master + etcd + 健康探测)
audience: 部署 Mooncake 的 SRE / 运维工程师 + 想深入 HA 机制的开发者
last_verified: 2026-04-23
sources:
  - vllm/mooncake-standalone.yaml
  - vllm/mooncake_high_availablity.yaml
  - vllm/third_partys/Mooncake/
related_docs:
  - mooncake_k8s_yaml_learning_notes_cn.md  (主文档: YAML 配置对比 + 运行时机制速通)
  - sglang_mooncake_env_config_deep_dive_cn.md  (SGLang 消费 Mooncake 的 env)
scope: |
  把主文档不是"日常查阅"必需的三块内容集中收纳:
  1. Master 角色 HA vs standalone 完整对比 (含启动参数详解)
  2. etcd 侧的 master / client 交互机制 + key schema
  3. 健康探测 (startupProbe / livenessProbe / readinessProbe) 策略
---

# Mooncake k8s 部署进阶

> 导读: 这是 [主文档 mooncake_k8s_yaml_learning_notes_cn.md](./mooncake_k8s_yaml_learning_notes_cn.md) 的姊妹文档,收纳三块"部署/运维进阶"内容。如果你只想快速看懂一份 Mooncake YAML 在写什么,看主文档就够了;如果要理解 HA 选主、etcd 键值、健康探测策略背后的机制,来这里。

---

## § 1 · Master 角色: HA vs Standalone

### 1.1 Standalone master (1 副本, 无 HA)

```yaml
containers:
- command:
  - mooncake_master
  - --default_kv_lease_ttl=300000   # 5 分钟租约
  ports:
  - containerPort: 50051  # master gRPC
  - containerPort: 9003   # metrics
  livenessProbe:
    httpGet: { path: /health, port: 9003 }
```

特点:
- **单副本, 单点**: master 挂 → 控制面全挂 → 所有 store 数据不可读写 (数据还在, 但没人分配了)
- **无状态外置**: master 的 KV 元数据、allocation map 全在进程内存
- 用 `httpGet /health` 做 liveness probe (Mooncake master 内置 HTTP metrics endpoint)

### 1.2 HA master (3 副本, etcd 外置状态)

```yaml
containers:
- command:
  - sh -c
  - |
    mooncake_master \
      --enable_ha=true \
      --etcd_endpoints=http://etcd-client.mooncake-ha.svc.cluster.local:2379 \
      --rpc_address=$(POD_IP) \
      --rpc_port=50051 \
      --eviction_high_watermark_ratio=0.9 \
      --default_kv_lease_ttl=10000      # 10 秒租约 (比 standalone 激进)
  livenessProbe:
    exec: { command: [sh, -c, "pgrep -x mooncake_master"] }
```

关键差异:

| 项 | Standalone | HA |
|---|---|---|
| `--enable_ha` | 无 | **true** |
| 状态持久化 | 进程内存 | **etcd 集群** |
| Leader 选举 | 无 | etcd + Raft 选主 |
| `--rpc_address` | 无 (默认 localhost) | `$(POD_IP)` 注册进 etcd 供发现 |
| Liveness 探测 | httpGet /health | pgrep 进程名 |
| 租约 TTL | 5 min (宽松) | 10 s (激进, 适合高周转) |

**为什么 HA 模式下 master 要 3 副本 + podAntiAffinity?**

参考分布式共识基本原理:
- quorum (多数派): N 副本需要 `⌊N/2⌋+1` 同意才能决策
- N=3 quorum=2, 能容忍挂 1 个; N=5 quorum=3, 能容忍挂 2 个
- **偶数 N 没好处**: N=4 quorum=3, 挂 1 个剩 3 够, 挂 2 个剩 2 不够 —— 容错能力和 N=3 一样
- 而且 **偶数 N 在网络分区下有可能两边 2-2 平分 → 都没 majority → 全挂**
- 所以分布式系统一律用奇数副本

`podAntiAffinity` + `topologyKey: kubernetes.io/hostname` 强制 3 副本调度到 3 台**不同物理机**:
- 单机挂 → 只丢 1 个副本, 剩 2 个仍够 quorum → 服务继续
- 否则如果 3 副本在同 1 台机, 等于 1 副本可用性

### 1.3 Master 关键启动参数

| 参数 | 含义 | standalone 值 | HA 值 |
|---|---|---|---|
| `--enable_ha` | 开启 HA 模式 | 无 | `true` |
| `--etcd_endpoints` | HA 模式下 etcd 地址 | — | `http://etcd-client...:2379` |
| `--rpc_address` | master 对外 RPC 地址 (注册进 etcd) | — | `$(POD_IP)` |
| `--rpc_port` | master RPC 端口 | 默认 50051 | `50051` |
| `--eviction_high_watermark_ratio` | 空间使用率到多少开始 evict | 默认 0.95 | **0.9** (给碎片留空间) |
| `--default_kv_lease_ttl` | KV 默认租约 (ms) | `300000` (5min) | `10000` (10s) |

**租约 TTL 的取舍**: 长租约减少续期开销但内存占用久; 短租约周转快但续期 RPC 压力大。生产 HA 选短租约因为 master 是共享资源, 不能被一个长期未读的 key 占着不放。

---

## § 2 · etcd HA 集成:master / client 两侧怎么用

**master 侧** (`mooncake_master --enable_ha=true --etcd_endpoints=... --rpc_address=$(POD_IP)`):
- 启动时向 etcd 注册自己, 通过 etcd lease 机制参与 leader 选举 (见 `EtcdLeaderCoordinator`)
- 实际 etcd key = **`mooncake-store/<cluster_namespace>/master_view`** ([etcd_leader_coordinator.cpp:377-384](../../Mooncake/mooncake-store/src/ha/leadership/backends/etcd/etcd_leader_coordinator.cpp#L377))
  - `cluster_namespace` 默认 `"mooncake"`, 可由 env `MC_STORE_CLUSTER_ID` 覆盖
  - value = leader 的 `rpc_address:rpc_port`
- master 通过 `EtcdHelper::GrantLease(DEFAULT_MASTER_VIEW_LEASE_TTL_SEC=5s, ...)` 申请一个 5 秒租约, 背景线程 `KeepAlive` 续期; leader 进程挂掉 → 5s 后 lease 失效 → etcd 触发 key 删除 → 其他 master 抢占
- Leader master 处理所有 RPC, follower 在失去 leadership 时挂起工作 (见 `ReleaseLeadership`)
- 还有 Redis backend 备份实现 (`RedisLeaderCoordinator`), 选择由 `HABackendSpec` 决定

**client 侧** (store container / SGLang worker):
- `MOONCAKE_MASTER=etcd://host:2379` → client 启动时连 etcd, `ReadCurrentView()` 读出当前 leader 地址, 再连 leader
- Client 启 `leader_monitor_thread_` (`client_service.h:686` + `LeaderMonitorThreadMain` 入口在 `client_service.cpp:316`) 后台 watch etcd, leader 切换后**自动 reconnect** 到新 leader
- Client **不直连 etcd 做数据读写**, etcd 在这里纯粹是服务发现

**etcd 本身也要 HA**: 通常 3/5 节点, 但不在 Mooncake YAML 范围, 需要单独用 etcd-operator / bitnami etcd chart 部署。

---

## § 3 · 健康探测策略

### 3.1 Probe 配置速查表 (对照 §1.3-H 迁过来)

| 探针 | standalone | HA | 机制一行 |
|---|---|---|---|
| master `readinessProbe` | `tcpSocket 50051` | 无 | standalone: master gRPC 端口通就 ready |
| master `livenessProbe` | `httpGet /health:9003` | `pgrep mooncake_master` | httpGet 能区分"进程在但 HA 异常"; pgrep 只查进程存在 |
| store `startupProbe` | 无 | `nc -z 127.0.0.1 8099`, 90×10s 宽限 | 给 RDMA 注册 1TB 内存留冷启动时间 |
| store `livenessProbe` | 无 | `nc -z 127.0.0.1 8099` | 进程在 ≠ 服务可用, 必须查端口真 listen |

### 3.2 Standalone 的探测

```yaml
# master
readinessProbe: { tcpSocket: { port: 50051 } }   # gRPC 端口通就 ready
livenessProbe:  { httpGet: { path: /health, port: 9003 } }  # HTTP /health 端点
```

### 3.3 HA 的探测

```yaml
# master
livenessProbe: { exec: [sh, -c, "pgrep -x mooncake_master"] }

# store
startupProbe:  { exec: [sh, -c, "nc -z 127.0.0.1 8099"], periodSeconds: 10, failureThreshold: 90 }
livenessProbe: { exec: [sh, -c, "nc -z 127.0.0.1 8099"], initialDelaySeconds: 10 }
```

### 3.4 可学习点

| 场景 | 探测策略 | 原因 |
|---|---|---|
| master (有 HTTP 端点) | httpGet /health | 能区分"进程在但 HA 异常"(会返回非 200) |
| master (没 HTTP 端点) | pgrep 进程名 | 单 binary, 进程在 = 健康 |
| store (Python + C++ 混合) | nc -z port | 进程在 ≠ 服务可用, 必须查端口真的 listen |
| store 冷启动慢 | startupProbe 单独给 15 分钟宽限 | RDMA 注册 1 TB 内存慢, 不能用 livenessProbe 30s 宽限打死 |

**startupProbe 和 livenessProbe 的分工**: k8s 保证 startupProbe 成功前, livenessProbe 不会开始探测。慢启动服务必用 startupProbe 做"第一关", 通过后再用严格的 livenessProbe 做"持续监控"。

---

## § 4 · 部署基础设施 (RBG + Downward API)

主文档默认读者已经知道 k8s RBG / Downward API 的存在, 这里集中讲一下这两个机制的原理。

### 4.1 RoleBasedGroup (RBG) 是什么

`kind: RoleBasedGroup` 是 **sgl-project/rbg** (github.com/sgl-project/rbg) 提供的 CRD, apiVersion `workloads.x-k8s.io/v1alpha1`。**专门管理"一个系统里多种角色"的 workload**。

- 原生 k8s 要写 4 个 StatefulSet + 一堆 label/topology 协调
- RBG 把它们收敛成一个资源, 共享调度顺序、重启策略、滚动升级

类似的思路还有 `LeaderWorkerSet` (LWS, 用在 SGLang 的多机推理)、`PodGroup` (training 用)。**核心洞察**: 分布式系统里"角色"是一等公民, 需要专门的编排层。

### 4.2 Downward API 注入 pod IP 作 hostname

两份 YAML 都有这段:
```yaml
- name: MOONCAKE_LOCAL_HOSTNAME
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
```

**它为什么存在**: Mooncake 的 `local_hostname` 会被 **上报给 master**, 其他 client 通过 master 发现 segment 后, **用这个 hostname 回连你**。如果用默认值 `"localhost"`:
- master 存: `segment X at "localhost:12847"`
- client B 去连自己的 `127.0.0.1:12847` → 连不到
- **GET 永远超时**

Downward API 在 pod 启动时把 **实际运行时 IP** 注入成 env:
- `hostNetwork: true` 下 `status.podIP` = 宿主机 IP (物理网卡 IP, RDMA 对端能直接 ARP 到)
- 不能硬编码: 每个 pod IP 都不同, 且重建可能变

---

## § 5 · 延伸阅读

- [主文档 mooncake_k8s_yaml_learning_notes_cn.md](./mooncake_k8s_yaml_learning_notes_cn.md) — YAML 配置对比 + 运行时机制速通 (P2PHANDSHAKE / Lease / Port / Env / Segment 等)
- [SGLang env config deep dive](./sglang_mooncake_env_config_deep_dive_cn.md) — SGLang 侧消费 Mooncake 的 env 变量全景
- [k8s Mooncake HA 部署 notes](./k8s_mooncake_ha_deployment_notes.md) — HA 部署上层说明
- Mooncake 源码: `vllm/third_partys/Mooncake/`
