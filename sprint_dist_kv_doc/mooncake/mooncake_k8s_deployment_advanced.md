---
title: Advanced Mooncake k8s Deployment (HA Master + etcd + Health Probes)
audience: SRE / operations engineers deploying Mooncake, and developers who want to understand the HA mechanism in depth
last_verified: 2026-04-23
sources:
  - vllm/mooncake-standalone.yaml
  - vllm/mooncake_high_availablity.yaml
  - vllm/third_partys/Mooncake/
related_docs:
  - mooncake_k8s_yaml_learning_notes.md  (main document: YAML configuration comparison + runtime mechanism quick tour)
  - sglang_mooncake_env_config_deep_dive.md  (SGLang env variables for consuming Mooncake)
scope: |
  This document collects three topics that are not required for daily lookup in the main document:
  1. Full comparison of the Master role in HA vs standalone mode, including startup parameters.
  2. Master / client interaction with etcd, plus the key schema.
  3. Health probe strategy: startupProbe / livenessProbe / readinessProbe.
---

# Advanced Mooncake k8s Deployment

> Guide: this is a companion document to [the main document, mooncake_k8s_yaml_learning_notes.md](./mooncake_k8s_yaml_learning_notes.md). It collects three "advanced deployment / operations" topics. If you only want to quickly understand what a Mooncake YAML says, the main document is enough. If you need the mechanisms behind HA leader election, etcd key-value state, and health probe policy, read this document.

---

## § 1 · Master Role: HA vs Standalone

### 1.1 Standalone master (1 replica, no HA)

```yaml
containers:
- command:
  - mooncake_master
  - --default_kv_lease_ttl=300000   # 5-minute lease
  ports:
  - containerPort: 50051  # master gRPC
  - containerPort: 9003   # metrics
  livenessProbe:
    httpGet: { path: /health, port: 9003 }
```

Characteristics:
- **Single replica, single point of failure**: if master dies, the control plane is down. All store data becomes unreadable/unwritable. The data still exists, but no one can allocate it.
- **No externalized state**: the master's KV metadata and allocation map live entirely in process memory.
- Uses `httpGet /health` as the liveness probe. Mooncake master has a built-in HTTP metrics endpoint.

### 1.2 HA master (3 replicas, state externalized to etcd)

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
      --default_kv_lease_ttl=10000      # 10-second lease, more aggressive than standalone
  livenessProbe:
    exec: { command: [sh, -c, "pgrep -x mooncake_master"] }
```

Key differences:

| Item | Standalone | HA |
|---|---|---|
| `--enable_ha` | absent | **true** |
| State persistence | process memory | **etcd cluster** |
| Leader election | none | etcd + Raft leader election |
| `--rpc_address` | absent (defaults to localhost) | `$(POD_IP)`, registered into etcd for discovery |
| Liveness probe | httpGet /health | pgrep process name |
| Lease TTL | 5 min (relaxed) | 10 s (aggressive, suitable for high churn) |

**Why does HA mode use 3 master replicas plus podAntiAffinity?**

From basic distributed consensus principles:
- quorum (majority): N replicas require `⌊N/2⌋+1` votes before the system can make a decision.
- For N=3, quorum=2, so the system tolerates 1 failed replica. For N=5, quorum=3, so it tolerates 2 failed replicas.
- **Even N does not help**: for N=4, quorum=3. Losing 1 leaves 3, which is enough; losing 2 leaves 2, which is not enough. This has the same fault tolerance as N=3.
- Also, **even N can split 2-2 during a network partition, leaving neither side with a majority, so the whole system stalls**.
- Distributed systems therefore normally use an odd replica count.

`podAntiAffinity` plus `topologyKey: kubernetes.io/hostname` forces the 3 replicas onto 3 **different physical machines**:
- If one machine dies, only 1 replica is lost. The remaining 2 still form a quorum, so service continues.
- If all 3 replicas landed on the same machine, availability would effectively be the same as a single replica.

### 1.3 Key master startup parameters

| Parameter | Meaning | Standalone value | HA value |
|---|---|---|---|
| `--enable_ha` | Enable HA mode | absent | `true` |
| `--etcd_endpoints` | etcd address in HA mode | — | `http://etcd-client...:2379` |
| `--rpc_address` | Externally reachable master RPC address, registered into etcd | — | `$(POD_IP)` |
| `--rpc_port` | Master RPC port | default 50051 | `50051` |
| `--eviction_high_watermark_ratio` | Start eviction when space usage reaches this ratio | default 0.95 | **0.9** (leaves room for fragmentation) |
| `--default_kv_lease_ttl` | Default KV lease in ms | `300000` (5 min) | `10000` (10 s) |

**Lease TTL tradeoff**: long leases reduce renewal overhead but keep memory occupied longer. Short leases reclaim memory faster but increase renewal RPC pressure. Production HA uses a shorter lease because the master is a shared resource, and stale keys that are no longer being read should not keep occupying it.

---

## § 2 · etcd HA Integration: How Master and Client Use It

**Master side** (`mooncake_master --enable_ha=true --etcd_endpoints=... --rpc_address=$(POD_IP)`):
- On startup, the master registers itself in etcd and participates in leader election through the etcd lease mechanism. See `EtcdLeaderCoordinator`.
- The actual etcd key is **`mooncake-store/<cluster_namespace>/master_view`** ([etcd_leader_coordinator.cpp:377-384](../../third_partys/Mooncake/mooncake-store/src/ha/leadership/backends/etcd/etcd_leader_coordinator.cpp#L377)).
  - `cluster_namespace` defaults to `"mooncake"` and can be overridden by env `MC_STORE_CLUSTER_ID`.
  - The value is the leader's `rpc_address:rpc_port`.
- The master obtains a 5-second lease through `EtcdHelper::GrantLease(DEFAULT_MASTER_VIEW_LEASE_TTL_SEC=5s, ...)`, and a background `KeepAlive` thread renews it. If the leader process dies, the lease expires after 5 s, etcd deletes the key, and another master can take over.
- The leader master handles all RPCs. Followers suspend work when they lose leadership. See `ReleaseLeadership`.
- There is also an alternate Redis backend implementation (`RedisLeaderCoordinator`). `HABackendSpec` selects which backend to use.

**Client side** (store container / SGLang worker):
- `MOONCAKE_MASTER=etcd://host:2379` means the client connects to etcd on startup, calls `ReadCurrentView()` to read the current leader address, then connects to that leader.
- After the client starts `leader_monitor_thread_` (`client_service.h:686`, with `LeaderMonitorThreadMain` entry in `client_service.cpp:316`), it watches etcd in the background and **automatically reconnects** to the new leader after a leader switch.
- Clients **do not connect directly to etcd for data-path reads/writes**. Here, etcd is purely service discovery.

**etcd itself also needs HA**: production deployments commonly use a 3- or 5-node cluster, but that is outside the Mooncake YAML. Deploy it separately with etcd-operator or the Bitnami etcd chart.

---

## § 3 · Health Probe Strategy

### 3.1 Probe quick reference table (moved from §1.3-H)

| Probe | Standalone | HA | One-line mechanism |
|---|---|---|---|
| master `readinessProbe` | `tcpSocket 50051` | absent | Standalone: if the master gRPC port is reachable, it is ready. |
| master `livenessProbe` | `httpGet /health:9003` | `pgrep mooncake_master` | httpGet can distinguish "process exists but HA is unhealthy"; pgrep only checks that the process exists. |
| store `startupProbe` | absent | `nc -z 127.0.0.1 8099`, 90 x 10 s grace | Leaves cold-start time for registering 1 TB of RDMA memory. |
| store `livenessProbe` | absent | `nc -z 127.0.0.1 8099` | Process existence does not mean service availability; the port must actually listen. |

### 3.2 Standalone probes

```yaml
# master
readinessProbe: { tcpSocket: { port: 50051 } }   # ready when the gRPC port is reachable
livenessProbe:  { httpGet: { path: /health, port: 9003 } }  # HTTP /health endpoint
```

### 3.3 HA probes

```yaml
# master
livenessProbe: { exec: [sh, -c, "pgrep -x mooncake_master"] }

# store
startupProbe:  { exec: [sh, -c, "nc -z 127.0.0.1 8099"], periodSeconds: 10, failureThreshold: 90 }
livenessProbe: { exec: [sh, -c, "nc -z 127.0.0.1 8099"], initialDelaySeconds: 10 }
```

### 3.4 What to learn from the probe choices

| Scenario | Probe strategy | Reason |
|---|---|---|
| master with HTTP endpoint | httpGet /health | Can distinguish "process exists but HA is unhealthy" by returning non-200. |
| master without HTTP endpoint | pgrep process name | Single binary; process exists is treated as healthy. |
| store, Python + C++ mixed | nc -z port | Process existence does not mean service availability; the port must really listen. |
| store has slow cold start | Give a separate 15-minute grace through startupProbe | RDMA registration of 1 TB memory is slow; a 30 s livenessProbe grace would kill it too early. |

**startupProbe vs livenessProbe division of labor**: Kubernetes guarantees that livenessProbe does not start before startupProbe succeeds. Slow-starting services should use startupProbe as the "first gate"; after it passes, a stricter livenessProbe can do continuous monitoring.

---

## § 4 · Deployment Infrastructure (RBG + Downward API)

The main document assumes readers already know about k8s RBG and Downward API. This section collects the mechanics behind those two features.

### 4.1 What is RoleBasedGroup (RBG)?

`kind: RoleBasedGroup` is a CRD provided by **sgl-project/rbg** (github.com/sgl-project/rbg), with apiVersion `workloads.x-k8s.io/v1alpha1`. It is built specifically for managing workloads with **multiple roles in one system**.

- Native k8s would require 4 StatefulSets plus many labels and topology rules.
- RBG consolidates them into one resource, with shared scheduling order, restart policy, and rolling update semantics.

Similar ideas include `LeaderWorkerSet` (LWS, used for SGLang multi-node inference) and `PodGroup` (used in training). **Core insight**: in distributed systems, "role" is a first-class concept and needs a dedicated orchestration layer.

### 4.2 Downward API injects pod IP as hostname

Both YAMLs contain this block:
```yaml
- name: MOONCAKE_LOCAL_HOSTNAME
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
```

**Why it exists**: Mooncake's `local_hostname` is **reported to the master**. After other clients discover a segment through the master, they **connect back to you** using this hostname. If the default value `"localhost"` were used:
- master stores: `segment X at "localhost:12847"`.
- client B tries to connect to its own `127.0.0.1:12847`, which fails.
- **GET times out forever**.

Downward API injects the **actual runtime IP** as an env var when the pod starts:
- With `hostNetwork: true`, `status.podIP` is the host's physical NIC IP, which an RDMA peer can ARP directly.
- It cannot be hardcoded: every pod has a different IP, and recreation may change it.

---

## § 5 · Further Reading

- [Main document, mooncake_k8s_yaml_learning_notes.md](./mooncake_k8s_yaml_learning_notes.md) — YAML configuration comparison + runtime mechanism quick tour (P2PHANDSHAKE / Lease / Port / Env / Segment, etc.)
- [SGLang env config deep dive](./sglang_mooncake_env_config_deep_dive.md) — full view of the SGLang-side env variables for consuming Mooncake
- `k8s_mooncake_ha_deployment_notes.md` — higher-level HA deployment notes; referenced by the source doc but not present in this checkout
- Mooncake source: `vllm/third_partys/Mooncake/`
