# NVL72 GB200 集群存储与 KV Cache 分层缓存架构文档

## 1. 集群概览

- **机型**: NVIDIA GB200 NVL72
- **整柜配置**: 36 Grace CPU + 72 Blackwell GPU，18 个 Compute Tray
- **单 Compute Tray（节点）**: 2 个 GB200 Grace Blackwell Superchip → 4 GPU + 2 Grace CPU（2 socket）
- **每节点 CPU 内存 (DDR)**: 约 882 GiB LPDDR5X（`/dev/shm` 为 442G）
- **每 GPU 显存 (HBM)**: 192G HBM3e，单节点 4 GPU 共 768G
- **共享文件系统**: Lustre（10.56.1.3@tcp:/lustre），35T；`/mnt/lustre` 为根挂载，`/home` 为其子路径挂载（`10.56.1.3@tcp:/lustre[/home]`）
- **多用户环境**: 每台机器有多个用户并发登录使用（观察到 4-7 个 /run/user 挂载）

---

## 2. GB200 Superchip 与互联架构

### 2.1 单 Superchip 内部结构

每个 GB200 Grace Blackwell Superchip 由 1 个 Grace CPU + 2 个 B200 GPU 组成，三者通过 **NVLink-C2C** 连接：

```
┌─────────────────────── GB200 Superchip ───────────────────────┐
│                                                               │
│   ┌──────────┐   NVLink-C2C    ┌──────────┐                  │
│   │ B200 GPU │◄──900 GB/s ───►│Grace CPU │                  │
│   │ 192G HBM │  (双向, 7x     │ LPDDR5X  │                  │
│   └──────────┘   PCIe Gen5)   └────┬─────┘                  │
│                                     │ NVLink-C2C             │
│   ┌──────────┐   NVLink-C2C        │ 900 GB/s               │
│   │ B200 GPU │◄──900 GB/s ────────┘                         │
│   │ 192G HBM │                                               │
│   └──────────┘                                               │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 2.2 单 Compute Tray（节点）结构

每个 Compute Tray 包含 2 个 Superchip：

```
┌──────────────────── Compute Tray (节点) ─────────────────────┐
│                                                              │
│  Superchip 0                    Superchip 1                  │
│  ┌────────────────────┐        ┌────────────────────┐       │
│  │ Grace CPU 0        │        │ Grace CPU 1        │       │
│  │   ↕ C2C 900GB/s    │        │   ↕ C2C 900GB/s    │       │
│  │ B200 GPU 0  GPU 1  │        │ B200 GPU 2  GPU 3  │       │
│  └────────────────────┘        └────────────────────┘       │
│           │                            │                     │
│           └──── NVLink (rack-level) ───┘                     │
│                  ↕ 1.8 TB/s per GPU                          │
│           连接到 9 个 NVLink Switch Tray                      │
│                                                              │
│  本地存储:                                                    │
│  ├── 系统盘: nvme4n1 (2T)  → /                               │
│  └── 数据盘: nvme0n1~3 (4×2.9T) → md0 RAID0 ~12T → /mnt/data│
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 GPU-CPU 数据传输：NVLink-C2C（重点）

GB200 架构中 GPU 和 CPU 之间的互联**不是传统的 PCIe，而是 NVLink-C2C**，这是理解 KV Cache 分层缓存的关键：

| 维度 | NVLink-C2C (GB200) | PCIe Gen5 (传统 x86 服务器) |
|------|--------------------|-----------------------------|
| 带宽 | **900 GB/s 双向** | 128 GB/s (x16) |
| 倍数 | **~7x PCIe Gen5** | 基准 |
| 内存一致性 | **硬件级 cache coherent** | 无（需显式数据搬运） |
| 地址空间 | **CPU/GPU 统一虚拟地址空间** | 独立地址空间 |
| 数据搬运 | 无需显式 memcpy，硬件自动处理 | 需要 cudaMemcpy 等显式操作 |
| 能效 | 比 PCIe Gen5 高 25x | 基准 |

**NVLink-C2C 的关键特性**：

1. **统一内存地址空间**: CPU 和 GPU 共享同一个进程页表（通过 ATS - Address Translation Services），CPU 线程和 GPU 线程可以透明地访问对方的内存，无需显式数据拷贝。

2. **Cache Coherent**: 硬件级缓存一致性（基于 Arm AMBA CHI 协议），CPU 和 GPU 对共享数据的修改对彼此立即可见，无需软件层面的同步。

3. **对 KV Cache 的意义**: GPU 可以直接以 900 GB/s 的带宽访问 CPU 端的 DDR 内存（LPDDR5X），当 GPU HBM 不够用时，KV Cache 可以溢出到 CPU 内存而不需要昂贵的显式数据搬运。这使得 CPU 内存成为天然的 L2 KV Cache 层。

### 2.4 整柜拓扑

```
┌────────────────────── NVL72 机柜 ──────────────────────┐
│                                                        │
│  ┌─────────────────────────────────────────────┐      │
│  │         9 × NVLink Switch Tray               │      │
│  │    每 Tray: 2 NVLink Switch, 144 ports       │      │
│  │    总计: 130 TB/s GPU-GPU 带宽                │      │
│  │    72 GPU 全互联, 任意 GPU 对等通信            │      │
│  └──────────────────┬──────────────────────────┘      │
│           NVLink    │   1.8 TB/s per GPU               │
│  ┌──────────────────┴──────────────────────────┐      │
│  │  Tray 01: [GPU0 GPU1]─CPU0 [GPU2 GPU3]─CPU1 │      │
│  │  Tray 02: [GPU4 GPU5]─CPU2 [GPU6 GPU7]─CPU3 │      │
│  │  ...                                         │      │
│  │  Tray 18: [GPU68 69]─CPU34 [GPU70 71]─CPU35  │      │
│  └──────────────────────────────────────────────┘      │
│                                                        │
│  网络:                                                  │
│  ├── GPU-GPU (柜内): NVLink 5th gen, 1.8 TB/s/GPU     │
│  ├── GPU-CPU (Superchip 内): NVLink-C2C, 900 GB/s     │
│  ├── 跨柜 GPU-GPU: RoCE (RDMA)                        │
│  └── 存储网络: TCP (10.56.1.x) → Lustre               │
│                                                        │
│  共享存储:                                              │
│  └── Lustre: 10.56.1.3@tcp:/lustre → /mnt/lustre      │
│              /home → 10.56.1.3@tcp:/lustre[/home]     │
│              35T, 18 节点共享                           │
└────────────────────────────────────────────────────────┘
```

---

## 3. 单节点存储布局

### 3.1 块设备结构

```
┌─────────────────── 单节点块设备 ──────────────────────┐
│                                                       │
│  系统盘 (nvme4n1 · 2T)                                │
│  ├── nvme4n1p1   2T    /           (根目录)            │
│  ├── nvme4n1p15  99M   /boot/efi   (EFI 引导)         │
│  └── nvme4n1p16  923M  /boot       (内核引导)          │
│                                                       │
│  数据盘 (4 × 2.9T NVMe · RAID0)                       │
│  ├── nvme0n1  2.9T ─┐                                 │
│  ├── nvme1n1  2.9T ─┤                                 │
│  ├── nvme2n1  2.9T ─┼─ md0 (RAID0) ~12T → /mnt/data  │
│  └── nvme3n1  2.9T ─┘                                 │
│                                                       │
│  网络文件系统                                           │
│  ├── 10.56.1.3@tcp:/lustre         35T → /mnt/lustre  │
│  └── 10.56.1.3@tcp:/lustre[/home]      → /home        │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### 3.2 系统盘 vs 数据盘

两者硬件上都是 NVMe SSD，区别在于用途：

- **系统盘 (nvme4n1, 2T)**: 装操作系统、内核、系统库、conda 环境等。相当于 C 盘，不应存放大量数据。
- **数据盘 (nvme0n1~3, 各 2.9T)**: 纯粹用于数据存储，组 RAID0 后提供高带宽本地存储，用作 Mooncake L3 Disk Cache。

### 3.3 为什么用 RAID0

4 块盘组 RAID0 追求**最大带宽**（单盘 ~6-7 GB/s，RAID0 理论 25+ GB/s），牺牲冗余性。任何一块盘故障整个 md0 丢失。本地数据盘被定位为**可丢失的缓存层**，重要数据不应只存在这里。

### 3.4 /etc/fstab 配置

```
LABEL=cloudimg-rootfs   /              ext4   discard,commit=30,errors=remount-ro  0 1
LABEL=BOOT              /boot          ext4   defaults                              0 2
LABEL=UEFI              /boot/efi      vfat   umask=0077                            0 1
/dev/md0                /mnt/data      xfs    defaults,nofail                       0 0
10.56.1.3@tcp:/lustre   /mnt/lustre    lustre nochecksum,_netdev                    0 0
/mnt/lustre/home        /home          none   bind,x-systemd.requires-mounts-for=/mnt/lustre,_netdev,nofail 0 0
```

注意：md0 行的 `nofail` 选项意味着即使 RAID 不存在也能正常启动，导致未组 RAID 的节点不会被发现。

补充：虽然 `fstab` 里 `/home` 这一行写的是 bind mount，但当前机器运行时 `findmnt -T /home` 显示其来源为 `10.56.1.3@tcp:/lustre[/home]`，因此 `/home` 实际位于 Lustre 上，不是本地磁盘目录。

---

## 4. 查看存储的常用命令

| 命令 | 用途 | 备注 |
|------|------|------|
| `lsblk` | 查看本地块设备（磁盘、分区、RAID） | **看不到**网络文件系统 |
| `df -h` | 查看所有已挂载文件系统 | 能看到 Lustre 等网络存储 |
| `fdisk -l` | 查看磁盘分区表 | 需要 root |
| `blkid` | 查看块设备 UUID 和文件系统类型 | |
| `mdadm --detail /dev/md0` | 查看 RAID 详情 | |
| `mount \| grep lustre` | 确认 Lustre 挂载状态 | |

**关键区别**: `lsblk` 只显示本地块设备，`df -h` 显示所有已挂载的文件系统。确认 Lustre 等网络存储必须用 `df -h`。

---

## 5. KV Cache 分层缓存架构 (Mooncake)

### 5.1 分层总览

```
速度快 ──────────────────────────────────────────── 速度慢
容量小 ──────────────────────────────────────────── 容量大

┌─────────────────────────────────────────────────────────┐
│  L1 · GPU HBM (每 GPU 192G HBM3e)                       │
│  延迟: ~ns  |  带宽: 8 TB/s (HBM 内部)                   │
│  用途: 活跃 KV Cache, 正在参与计算                        │
│  作用域: 单 GPU                                          │
├─────────────────────────────────────────────────────────┤
│  L2 · CPU DDR 内存 (每节点约 882 GiB LPDDR5X)            │
│  延迟: ~μs  |  带宽: 900 GB/s (NVLink-C2C, 非 PCIe!)    │
│  用途: 温 KV Cache, 最近用过但暂时不活跃                   │
│  作用域: 本节点 (跨节点通过 RDMA 复用)                     │
│  关键: GPU 可通过统一内存地址空间直接访问，无需显式拷贝      │
├─────────────────────────────────────────────────────────┤
│  L3 · 本地 NVMe RAID0 (每节点 ~12T)                      │
│  延迟: ~100μs  |  带宽: 25+ GB/s (4 盘 RAID0)           │
│  用途: Mooncake Disk Cache, 大容量冷 KV Cache             │
│  作用域: 本节点直接访问; 其他节点可通过 CPU 内存中转访问    │
│  路径: /mnt/data                                         │
└─────────────────────────────────────────────────────────┘

注: Lustre（`/mnt/lustre` 及其 `/home` 子路径）不参与 KV Cache 存储，仅用于模型权重、
数据集、checkpoint 等通用共享文件。
```

### 5.2 各层对比

| 层级 | 存储介质 | 延迟 | 带宽 | 容量/节点 | 作用域 | 互联方式 |
|------|---------|------|------|----------|--------|---------|
| L1 | GPU HBM3e | ~ns | 8 TB/s | 768G (4×192G) | 单 GPU | GPU 内部 |
| L2 | CPU LPDDR5X | ~μs | 900 GB/s | 约 882 GiB | 本节点; 跨节点 RDMA | **NVLink-C2C** |
| L3 | 本地 NVMe RAID0 | ~100μs | 25+ GB/s | ~12T | 本节点直接; 跨节点经 CPU 中转 | PCIe |

### 5.3 跨节点 KV Cache 复用策略

```
节点 A                                  节点 B
┌──────────────┐                       ┌──────────────┐
│ L1: GPU HBM  │                       │ L1: GPU HBM  │
│      ↕       │  NVLink (柜内72GPU    │      ↕       │
│  NVLink-C2C  │   全互联 1.8TB/s)     │  NVLink-C2C  │
│      ↕       │◄─────────────────────►│      ↕       │
│ L2: CPU DDR  │                       │ L2: CPU DDR  │
│      ↕       │◄──── RDMA ──────────►│      ↕       │
│ L3: NVMe     │    (RoCE RDMA)        │ L3: NVMe     │
│  /mnt/data   │                       │  /mnt/data   │
└──────────────┘                       └──────────────┘
```

**跨节点访问 L3 的路径**：

节点 B 的 GPU 需要访问节点 A 的 NVMe 上的 KV Cache 时，数据流为：

```
节点A NVMe ─(PCIe)→ 节点A CPU DDR ─(RDMA/NVLink)→ 节点B CPU DDR ─(C2C)→ 节点B GPU HBM
```

另一种可能的路径（柜内 NVLink 直通）：

```
节点A NVMe ─(PCIe)→ 节点A CPU DDR ─(C2C)→ 节点A GPU ─(NVLink)→ 节点B GPU HBM
```

由于柜内 72 GPU 通过 NVLink Switch 全互联（1.8 TB/s/GPU），第二条路径可能在某些场景下更快，取决于 CPU DDR→GPU 和 GPU→GPU 的实际可用带宽。

**跨节点复用原则**：

1. **L2 跨节点**: CPU DDR 之间走 RDMA，或经 GPU NVLink 中转
2. **L3 跨节点**: NVMe 数据先读到本地 CPU DDR (L2)，再通过上述路径传输到远端节点
3. **柜内优先 NVLink**: 同一 NVL72 柜内的节点间优先利用 NVLink 全互联的高带宽

### 5.4 GPU-CPU 之间的 KV Cache 传输（NVLink-C2C 详解）

在 GB200 架构中，KV Cache 从 GPU HBM (L1) 溢出到 CPU DDR (L2) 的路径是 NVLink-C2C，**不是 PCIe**：

```
┌─────────── GB200 Superchip 内部 KV Cache 流动 ──────────┐
│                                                         │
│  B200 GPU                     Grace CPU                 │
│  ┌─────────────┐              ┌─────────────┐          │
│  │ HBM3e 192G  │  NVLink-C2C  │ LPDDR5X     │          │
│  │             │◄─────────────►│             │          │
│  │ 活跃 KV     │  900 GB/s    │ 溢出 KV     │          │
│  │ Cache (L1)  │  双向        │ Cache (L2)   │          │
│  │             │  cache       │             │          │
│  │ 带宽: 8TB/s │  coherent    │ 带宽: ~500   │          │
│  │ (HBM 内部)  │  统一地址空间 │ GB/s (DDR)  │          │
│  └─────────────┘              └─────────────┘          │
│                                                         │
│  关键: GPU 访问 CPU 内存不需要 cudaMemcpy，              │
│  通过统一虚拟地址空间直接访问，硬件自动处理数据搬运。      │
│  瓶颈在 NVLink-C2C 的 900 GB/s（远快于 PCIe 的 128GB/s） │
└─────────────────────────────────────────────────────────┘
```

---

## 6. 网络拓扑观察

| 网络类型 | 用途 | 协议 | 依据 |
|---------|------|------|------|
| NVLink 5th gen | 柜内 72 GPU 全互联 | NVLink | 1.8 TB/s/GPU |
| NVLink-C2C | Superchip 内 CPU↔GPU | NVLink | 900 GB/s, cache coherent |
| RoCE (RDMA) | 跨柜 GPU-GPU, RDMA | RDMA | 本机 `rocep*` HCA 与 `gpu0rdma0`~`gpu3rdma0` |
| TCP (10.56.1.x) | Lustre 存储网络 | TCP | 挂载信息 `@tcp`（非 `@o2ib`） |

Lustre 走 TCP 而非 RDMA（`@o2ib`），意味着存储网络性能受 TCP 协议栈开销影响。跨节点 KV Cache 复用应优先走 RDMA 而非 Lustre。

---

## 7. 当前问题与待办

### 问题 1: 部分节点（10+ 台）未组 RAID

**现象**: `gb200-rack-03` 等 10+ 台节点的 4 块 NVMe 数据盘处于裸盘状态。`/etc/fstab` 中已有 md0 条目但实际 RAID 未创建，`nofail` 导致问题被隐藏。

**影响**:
- 浪费 12T 本地高速存储
- Mooncake L3 disk cache 无法工作
- 节点间缓存能力不一致
- 数据被写到系统盘（根目录已用 755G / 39%）

**排查**:
```bash
pdsh -w node[01-18] 'df -h /mnt/data 2>&1'
```

**修复**:
```bash
# 1. 创建 RAID0
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 \
  /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1

# 2. 格式化 (注意 fstab 中配的是 xfs)
sudo mkfs.xfs /dev/md0

# 3. 挂载
sudo mkdir -p /mnt/data
sudo mount /dev/md0 /mnt/data
```

> 注意: fstab 中 md0 的文件系统类型是 **xfs**（不是 ext4），格式化时要用 `mkfs.xfs`。

### 问题 2: Lustre 空间使用率偏高

35T 已用 16T（46%），18 节点共享，平均每节点余量不到 1T。若用 Lustre 存冷 KV Cache 需提前规划空间预算。

### 问题 3: 未组 RAID 节点的系统盘占用异常

755G / 39%，远高于正常节点 314G / 16%。需排查并迁移数据：
```bash
du -sh /* 2>/dev/null | sort -rh | head -20
```

### 问题 4: 多用户管理

4-7 个用户并发，需制定 /mnt/data 和 /mnt/lustre 的使用规范和配额。

---

## 8. Mooncake L3 Disk Cache 配置要点

1. **路径**: 指向 `/mnt/data`（本地 RAID0, xfs 文件系统）
2. **前提**: 确保所有 18 个节点的 /mnt/data 均已正确挂载
3. **容量**: 每节点约 12T
4. **跨节点复用**: L3 本身不跨节点，跨节点走 RDMA (L2)

---

## 附录 A: 关键信息速查

```
系统盘:        nvme4n1           2T        /
数据盘:        nvme0n1~3         4×2.9T    → md0 RAID0 ~12T → /mnt/data (xfs)
共享存储:      Lustre            35T       → /mnt/lustre
                                     /home → 10.56.1.3@tcp:/lustre[/home]
Lustre 地址:   10.56.1.3@tcp:/lustre
CPU 内存 (DDR): 约 882 GiB LPDDR5X / 节点 (`/dev/shm` 为 442G)
GPU 显存 (HBM): 192G HBM3e / GPU, 768G / 节点
CPU-GPU 互联:  NVLink-C2C, 900 GB/s, cache coherent
GPU-GPU 互联:  NVLink 5th gen, 1.8 TB/s / GPU, 柜内 72 GPU 全互联
跨柜互联:      RoCE (RDMA)
存储网络:      TCP (非 RDMA)
节点数:        18
每节点:        4 GPU (B200) + 2 Grace CPU (2 socket)
```

## 附录 B: 术语对照

| 缩写 | 全称 | 说明 |
|------|------|------|
| HBM | High Bandwidth Memory | GPU 显存，本文档中指 GPU memory |
| DDR / LPDDR5X | Low-Power DDR5X | CPU 内存，本文档中指 CPU memory |
| NVLink-C2C | NVLink Chip-to-Chip | Superchip 内 CPU↔GPU 互联，900 GB/s |
| NVLink | NVLink 5th gen | 柜内 GPU↔GPU 互联，1.8 TB/s/GPU |
| RDMA | Remote Direct Memory Access | 跨节点内存直接访问 |
| md0 | Linux Software RAID | 本地 NVMe 组成的 RAID0 阵列 |

---

## 附录 C: NVLink-C2C 与 KV Cache Offload 的实际考量

### C.1 统一内存的工作方式

在 GB200 上，CPU 和 GPU 通过 NVLink-C2C 共享统一虚拟地址空间。实际使用中有两种模式：

1. **隐式访问（Unified Memory / Managed Memory）**: 通过 CUDA Unified Memory 或 RMM 分配的内存，GPU 和 CPU 都可以直接访问。当 GPU 访问 CPU 端内存时，硬件自动通过 NVLink-C2C 拉取数据，无需 cudaMemcpy。适合 KV Cache 溢出场景——GPU HBM 满了之后，KV Cache 自然溢出到 CPU DDR，GPU 需要时按需拉回。

2. **显式传输（DMA Copy Engine）**: 应用主动调度数据搬运，利用 Grace CPU 的 DMA 引擎通过 NVLink-C2C 做批量传输。吞吐更可控，适合提前预取即将使用的 KV Cache。

### C.2 实际带宽预期

NVLink-C2C 标称 900 GB/s 是双向总带宽（每方向 450 GB/s）。实际可用带宽取决于：

- **访问模式**: 大块顺序读写接近峰值，小块随机访问会打折
- **一个 Grace CPU 连接 2 个 GPU**: 900 GB/s 由 2 个 GPU 共享，单 GPU 到 CPU 的峰值约 450 GB/s 单向
- **CPU DDR 带宽**: LPDDR5X 本身带宽约 500 GB/s，可能成为瓶颈

### C.3 与 Mooncake 的结合

Mooncake 的分层架构中：

```
Mooncake Transfer Engine
├── GPU HBM (L1) ←→ CPU DDR (L2): NVLink-C2C, 900 GB/s
│   硬件 cache coherent, 可用 unified memory
│
├── CPU DDR (L2) ←→ 远端 CPU DDR / GPU: RDMA 或 NVLink
│   跨节点 KV Cache 复用 (柜内可走 NVLink 全互联)
│
├── CPU DDR (L2) ←→ 本地 NVMe (L3): PCIe / DMA
│   本节点 disk cache 读写
│
└── 远端 NVMe (L3) 跨节点访问:
    远端 NVMe → 远端 CPU DDR → RDMA/NVLink → 本地 GPU
    通过 CPU 内存中转实现跨节点 L3 复用
```

---

## 附录 D: 完整带宽层级参考

```
带宽层级 (单向, 近似值):

GPU HBM 内部          ████████████████████████████████████████  8,000 GB/s
GPU die-to-die        ██████████████████████████████████████    5,000 GB/s  (10 TB/s 双向)
NVLink GPU-GPU        ██████                                     900 GB/s  (1.8 TB/s 双向)
NVLink-C2C GPU-CPU    ███                                        450 GB/s  (900 GB/s 双向)
CPU LPDDR5X           ███                                        500 GB/s
PCIe Gen5 x16         █                                          64 GB/s   (128 GB/s 双向)
NVMe RAID0 (4盘)      ▌                                          25+ GB/s
NVMe SSD (单盘)                                                  6-7 GB/s
InfiniBand NDR400     ▌                                          50 GB/s
TCP 网络 (25GbE)                                                 3 GB/s

注: Lustre 不参与 KV Cache 链路，其带宽仅影响模型加载等通用 I/O。
```

---

## 附录 E: 排查脚本汇总

### E.1 批量检查所有节点的 RAID 和挂载状态

```bash
#!/bin/bash
# check_storage.sh - 在所有节点上检查存储配置

echo "=== 检查 /mnt/data 挂载 ==="
pdsh -w node[01-18] 'mountpoint -q /mnt/data && echo "OK: /mnt/data mounted" || echo "FAIL: /mnt/data NOT mounted"'

echo ""
echo "=== 检查 md0 RAID 状态 ==="
pdsh -w node[01-18] 'test -e /dev/md0 && echo "OK: md0 exists" || echo "FAIL: md0 NOT found"'

echo ""
echo "=== 检查 Lustre 挂载 ==="
pdsh -w node[01-18] 'mountpoint -q /mnt/lustre && echo "OK: lustre mounted" || echo "FAIL: lustre NOT mounted"'

echo ""
echo "=== 检查系统盘使用率 ==="
pdsh -w node[01-18] "df -h / | tail -1 | awk '{print \$5, \$3\"/\"\$2}'"

echo ""
echo "=== 检查裸盘状态 ==="
pdsh -w node[01-18] 'lsblk -d -n -o NAME,SIZE,TYPE /dev/nvme[0-3]n1 2>/dev/null'
```

### E.2 单节点 RAID 修复脚本

```bash
#!/bin/bash
# fix_raid.sh - 在未组 RAID 的节点上执行
# !! 执行前确认该节点的 nvme0n1~3 确实是空盘 !!

set -e

echo "Creating RAID0 array..."
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 \
  /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 --force

echo "Formatting as xfs..."
sudo mkfs.xfs -f /dev/md0

echo "Mounting to /mnt/data..."
sudo mkdir -p /mnt/data
sudo mount /dev/md0 /mnt/data

echo "Verifying..."
df -h /mnt/data
mdadm --detail /dev/md0

echo "Done. /mnt/data is ready."
echo "Note: /etc/fstab already has the md0 entry, reboot will auto-mount."
```

### E.3 系统盘占用排查

```bash
#!/bin/bash
# check_disk_usage.sh - 排查系统盘占用

echo "=== Top 20 largest directories under / ==="
sudo du -sh /* 2>/dev/null | sort -rh | head -20

echo ""
echo "=== Large files (>1G) on system disk ==="
sudo find / -xdev -type f -size +1G -exec ls -lh {} \; 2>/dev/null | sort -k5 -rh | head -20
```
