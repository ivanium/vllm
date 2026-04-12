# Simple Top-Down System Design for vLLM + MooncakeStoreConnector

## 1. Goal

Design a simple V1 integration for using `MooncakeStoreConnector` in vLLM on a multi-node GB200/Grace cluster.

V1 priorities:

- Performance first.
- Operational simplicity second.
- Higher complexity is acceptable if it removes major data-path bottlenecks.

This document proposes a **node-scoped Mooncake ownership model**:

- one Mooncake **master server** per cluster
- one Mooncake **owner client** per node
- multiple vLLM worker-local **dummy clients** per node

This is intentionally different from a pure "one store client per worker" model. The main reason is to centralize RDMA ownership, NIC binding, and host-memory management at the node level.

## 2. Design Assumptions

- "Owner client" maps to a node-scoped Mooncake `RealClient` service.
- "Dummy client" maps to a per-worker Mooncake `DummyClient` frontend that forwards requests to the local owner.
- A GB200 node has **two Grace CPUs**, so we treat the node as **two NUMA/locality domains**.
- The owner client can bind **four RDMA NICs** on the node and expose one shared transfer service to all local workers.
- If the RDMA stack for a node fails, V1 accepts the node as the failure domain.

## 3. High-Level Architecture

```
                         +---------------------------+
                         | Mooncake Master Server    |
                         | metadata + allocation     |
                         +-------------+-------------+
                                       |
             -----------------------------------------------------
             |                                                   |
             v                                                   v
 +-------------------------------+                 +-------------------------------+
 | Node A                        |                 | Node B                        |
 |                               |                 |                               |
 |  vLLM workers                 |                 |  vLLM workers                 |
 |  +---------+  +---------+     |                 |  +---------+  +---------+     |
 |  | Worker0 |  | Worker1 |     |                 |  | Worker0 |  | Worker1 |     |
 |  +----+----+  +----+----+     |                 |  +----+----+  +----+----+     |
 |       |            |          |                 |       |            |          |
 |   +---v------------v---+      |                 |   +---v------------v---+      |
 |   | Dummy Clients      |      |                 |   | Dummy Clients      |      |
 |   +---+------------+---+      |                 |   +---+------------+---+      |
 |       | local RPC  | IPC/SHM  |                 |       | local RPC  | IPC/SHM  |
 |   +---v------------v---+      |     RDMA        |   +---v------------v---+      |
 |   | Owner Client        +<--------------------->+   | Owner Client        |      |
 |   | (node-scoped)       |      |                 |   | (node-scoped)       |      |
 |   | - 4 NIC binding     |      |                 |   | - 4 NIC binding     |      |
 |   | - shared CPU pool   |      |                 |   | - shared CPU pool   |      |
 |   +---------------------+      |                 |   +---------------------+      |
 +-------------------------------+                 +-------------------------------+
```

## 4. Node-Internal Layout

```
                 One Node

   +-------------------------------------------------------------+
   | Owner Client / RealClient                                   |
   | - owns RDMA transports                                      |
   | - mounts node CPU global segments                           |
   | - manages shared local buffer pool                          |
   | - exports local RPC + IPC/SHM endpoint for dummy clients    |
   +---------------------------+---------------------------------+
                               |
          -------------------------------------------------
          |                     |                         |
          v                     v                         v
   +-------------+       +-------------+           +-------------+
   | DummyClient |       | DummyClient |    ...    | DummyClient |
   | Worker/GPU0 |       | Worker/GPU1 |           | Worker/GPU3 |
   +------+------+       +------+------+           +------+------+
          |                     |                         |
          v                     v                         v
      GPU KV cache         GPU KV cache              GPU KV cache

   Locality model:
   - Grace 0 <-> GPU0/GPU1
   - Grace 1 <-> GPU2/GPU3
   - owner segments and transfer queues should be NUMA-aware
```

## 5. End-to-End Data Flow

### Save / Offload Path

1. A vLLM worker decides to offload KV blocks.
2. The worker calls its local dummy client.
3. The dummy client forwards the request to the node owner client.
4. The owner client asks the cluster master for placement.
5. The owner client performs RDMA write to the selected memory segment.

Target placement may be:

- local node CPU memory
- remote node CPU memory

### Load / Reuse Path

1. A vLLM worker needs KV blocks.
2. The worker calls its local dummy client.
3. The owner client resolves replicas through the master metadata.
4. The owner client performs RDMA read into the destination GPU buffer.

V1 policy:

- prefer the shortest local path when metadata allows it
- otherwise read from remote node memory directly
- do not optimize for every corner case before the main path is stable

## 6. Key Design Choices

### 6.1 One Owner Client Per Node

Why:

- avoids duplicating RDMA setup per worker
- lets us bind all node NICs in one place
- gives one shared CPU memory pool per node
- makes NUMA-aware placement implementable

### 6.2 RDMA Terminates at the Node, Not at Each Worker

All RDMA resources should be owned by the node-scoped owner client.

This keeps:

- NIC binding
- queue-pair lifecycle
- memory registration
- retry and error handling

in one place instead of spreading them across all worker processes.

### 6.3 NUMA-Aware Data Path

The owner client should allocate CPU global segments with NUMA awareness and align the preferred path with the nearest Grace domain and RDMA NIC.

Practical intent:

- GPU0/GPU1 prefer Grace0-local CPU memory
- GPU2/GPU3 prefer Grace1-local CPU memory
- segmenting by NIC NUMA domain helps use all four NICs efficiently

### 6.4 Shared Buffering and Copy Model

V1 accepts **one additional node-local copy** into owner-visible shared memory if needed, but avoids paying one extra copy per worker client.

So the model is:

- shared staging/buffer management at the owner level
- no separate full data-path copy stack inside every dummy client

### 6.5 Two-Stage Allocation, Shared Locally

There are two allocation layers:

1. cluster allocation by the master server for the final Mooncake replica placement
2. node-local allocation by the owner client for shared transfer/staging resources

The second layer is shared across all local dummy clients.

## 7. Failure Model

V1 intentionally uses a coarse failure domain:

- if the owner client or node-level RDMA path fails, the whole node instance is considered unavailable
- dummy clients are lightweight frontends, not independent recovery domains

This is acceptable in V1 because it reduces control-plane complexity and keeps the fast path simple.

## 8. Why This Is a Good V1

- It matches Mooncake's existing `RealClient` / `DummyClient` split well.
- It gives a clean place to implement NUMA-aware CPU segment mounting.
- It allows one owner to bind multiple RDMA NICs.
- It keeps the worker integration simple on the vLLM side.
- It is easy to evolve later toward smarter locality-aware allocation and finer-grained failover.

## 9. Out of Scope for V1

- fine-grained per-worker failover
- SSD offload policy redesign
- advanced replica scoring across topology layers
- full scheduler changes for perfect locality placement

## 10. Summary

The recommended V1 design is:

- **one cluster, one master server**
- **one node, one owner client**
- **one worker, one dummy client**
- **all RDMA owned by the node owner**
- **NUMA-aware CPU segments across the two Grace domains**

This gives us a simple top-down architecture that is fast enough to be worth building, while keeping the implementation boundary clear between vLLM workers and Mooncake node services.
