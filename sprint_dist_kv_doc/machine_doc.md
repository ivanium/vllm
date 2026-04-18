# NVL72 GB200 Cluster Storage and Layered KV Cache Architecture

## 1. Cluster overview

- **Model**: NVIDIA GB200 NVL72
- **Full cabinet configuration**: 36 Grace CPU + 72 Blackwell GPU, 18 Compute Tray
- **Single Compute Tray (node)**: 2x GB200 Grace Blackwell Superchip → 4 GPU + 2 Grace CPU (2 socket)
- **CPU memory (DDR) per node**: ~882 GiB LPDDR5X (442G for `/dev/shm`)
- **Video Memory (HBM) per GPU**: 192G HBM3e, single node 4 GPUs total 768G
- **Shared file system**: Lustre (10.56.1.3@tcp:/lustre), 35T; `/mnt/lustre` is mounted as the root, `/home` is mounted as its subpath (`10.56.1.3@tcp:/lustre[/home]`)
- **Multi-user environment**: Each machine has multiple users logging in concurrently (4-7 /run/user mounts observed)

---

## 2. GB200 Superchip and interconnection architecture

### 2.1 Single Superchip internal structure

Each GB200 Grace Blackwell Superchip consists of 1 Grace CPU + 2 B200 GPUs, which are connected through **NVLink-C2C**:
```
┌─────────────────────── GB200 Superchip ───────────────────────┐
│                                                               │
│   ┌──────────┐   NVLink-C2C    ┌──────────┐                  │
│   │ B200 GPU │◄──900 GB/s ───►│Grace CPU │                  │
│ │ 192G HBM │ (bidirectional, 7x │ LPDDR5X │ │
│   └──────────┘   PCIe Gen5)   └────┬─────┘                  │
│                                     │ NVLink-C2C             │
│   ┌──────────┐   NVLink-C2C        │ 900 GB/s               │
│   │ B200 GPU │◄──900 GB/s ────────┘                         │
│   │ 192G HBM │                                               │
│   └──────────┘                                               │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```
### 2.2 Single Compute Tray (node) structure

Each Compute Tray contains 2 Superchips:
```
┌──────────────────── Compute Tray (node) ──────────────────────────
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
│ Connects to 9 NVLink Switch Trays │
│                                                              │
│ Local Storage: │
│ ├── System disk: nvme4n1 (2T) → / │
│ └── Data disk: nvme0n1~3 (4×2.9T) → md0 RAID0 ~12T → /mnt/data│
│                                                              │
└──────────────────────────────────────────────────────────────┘
```
### 2.3 GPU-CPU data transmission: NVLink-C2C (key)

The interconnection between the GPU and CPU in the GB200 architecture is not traditional PCIe, but NVLink-C2C. This is the key to understanding the KV Cache hierarchical cache:

| Dimensions | NVLink-C2C (GB200) | PCIe Gen5 (legacy x86 server) |
|------|--------------------|--------------------------------|
| Bandwidth | **900 GB/s bidirectional** | 128 GB/s (x16) |
| Multiples | **~7x PCIe Gen5** | Benchmarks |
| Memory consistency | **Hardware level cache coherent** | None (requires explicit data movement) |
| Address space | **CPU/GPU unified virtual address space** | Independent address space |
| Data handling | No need for explicit memcpy, hardware handles it automatically | Requires explicit operations such as cudaMemcpy |
| Energy Efficiency | 25x better than PCIe Gen5 | Benchmarks |

**NVLink-C2C Key Features**:

1. **Unified memory address space**: CPU and GPU share the same process page table (through ATS - Address Translation Services), and CPU threads and GPU threads can transparently access each other's memory without explicit data copying.

2. **Cache Coherent**: Hardware-level cache coherency (based on Arm AMBA CHI protocol), modifications to shared data by the CPU and GPU are immediately visible to each other, without software-level synchronization.

3. **Significance for KV Cache**: The GPU can directly access the CPU-side DDR memory (LPDDR5X) with a bandwidth of 900 GB/s. When the GPU HBM is not enough, the KV Cache can overflow to the CPU memory without expensive explicit data transfer. This makes CPU memory a natural L2 KV Cache layer.

### 2.4 Whole cabinet topology
```
┌────────────────────── NVL72 Cabinet ────────────────────────┐
│                                                        │
│  ┌─────────────────────────────────────────────┐      │
│  │         9 × NVLink Switch Tray               │      │
│ │ Per Tray: 2 NVLink Switch, 144 ports │ │
│ │ Total: 130 TB/s GPU-GPU bandwidth │ │
│ │ 72 GPU fully interconnected, any GPU peer-to-peer communication │ │
│  └──────────────────┬──────────────────────────┘      │
│           NVLink    │   1.8 TB/s per GPU               │
│  ┌──────────────────┴──────────────────────────┐      │
│  │  Tray 01: [GPU0 GPU1]─CPU0 [GPU2 GPU3]─CPU1 │      │
│  │  Tray 02: [GPU4 GPU5]─CPU2 [GPU6 GPU7]─CPU3 │      │
│  │  ...                                         │      │
│  │  Tray 18: [GPU68 69]─CPU34 [GPU70 71]─CPU35  │      │
│  └──────────────────────────────────────────────┘      │
│                                                        │
│ Network: │
│ ├── GPU-GPU (inside the cabinet): NVLink 5th gen, 1.8 TB/s/GPU │
│ ├── GPU-CPU (within Superchip): NVLink-C2C, 900 GB/s │
│ ├── Cross-cabinet GPU-GPU: RoCE (RDMA) │
│ └── Storage network: TCP (10.56.1.x) → Luster │
│                                                        │
│ Shared storage: │
│  └── Lustre: 10.56.1.3@tcp:/lustre → /mnt/lustre      │
│              /home → 10.56.1.3@tcp:/lustre[/home]     │
│ 35T, 18 nodes shared │
└────────────────────────────────────────────────────────┘
```
---

## 3. Single node storage layout

### 3.1 Block device structure
```
┌─────────────────── Single node block device ────────────────────────┐
│                                                       │
│ System disk (nvme4n1 · 2T) │
│ ├── nvme4n1p1 2T / (root directory) │
│ ├── nvme4n1p15 99M /boot/efi (EFI boot) │
│ └── nvme4n1p16 923M /boot (kernel boot) │
│                                                       │
│ Data disk (4 × 2.9T NVMe · RAID0) │
│  ├── nvme0n1  2.9T ─┐                                 │
│  ├── nvme1n1  2.9T ─┤                                 │
│  ├── nvme2n1  2.9T ─┼─ md0 (RAID0) ~12T → /mnt/data  │
│  └── nvme3n1  2.9T ─┘                                 │
│                                                       │
│ Network File System │
│  ├── 10.56.1.3@tcp:/lustre         35T → /mnt/lustre  │
│  └── 10.56.1.3@tcp:/lustre[/home]      → /home        │
│                                                       │
└───────────────────────────────────────────────────────┘
```
### 3.2 System disk vs data disk

Both hardware are NVMe SSD, the difference lies in the use:

- **System disk (nvme4n1, 2T)**: Install operating system, kernel, system library, conda environment, etc. Equivalent to C drive, it should not store large amounts of data.
- **Data disk (nvme0n1~3, 2.9T each)**: purely used for data storage. After grouping into RAID0, it provides high-bandwidth local storage and is used as Mooncake L3 Disk Cache.

### 3.3 Why use RAID0

4-disk group RAID0 pursues **maximum bandwidth** (single disk ~6-7 GB/s, RAID0 theory 25+ GB/s), sacrificing redundancy. If any disk fails, the entire md0 will be lost. The local data disk is positioned as a **losable cache layer**, and important data should not only exist here.

### 3.4 /etc/fstab configuration
```
LABEL=cloudimg-rootfs   /              ext4   discard,commit=30,errors=remount-ro  0 1
LABEL=BOOT              /boot          ext4   defaults                              0 2
LABEL=UEFI              /boot/efi      vfat   umask=0077                            0 1
/dev/md0                /mnt/data      xfs    defaults,nofail                       0 0
10.56.1.3@tcp:/lustre   /mnt/lustre    lustre nochecksum,_netdev                    0 0
/mnt/lustre/home        /home          none   bind,x-systemd.requires-mounts-for=/mnt/lustre,_netdev,nofail 0 0
```
Note: The `nofail` option in the md0 line means that it can start normally even if RAID does not exist, causing nodes that are not RAID-enabled to be discovered.

Supplement: Although the line `/home` in `fstab` says bind mount, when the current machine is running `findmnt -T /home` shows that its source is `10.56.1.3@tcp:/lustre[/home]`, so `/home` is actually located on Luster, not the local disk directory.

---

## 4. View stored common commands

| Command | Purpose | Remarks |
|------|------|------|
| `lsblk` | View local block devices (disks, partitions, RAID) | **cannot see** network file systems |
| `df -h` | View all mounted file systems | Can see network storage such as Luster |
| `fdisk -l` | View disk partition table | Requires root |
| `blkid` | View block device UUID and file system type | |
| `mdadm --detail /dev/md0` | View RAID details | |
| `mount \| grep lustre` | Confirm Luster mounting status | |

**Key difference**: `lsblk` only displays local block devices, `df -h` displays all mounted file systems. Confirm that network storage such as Luster must use `df -h`.

---

## 5. KV Cache hierarchical cache architecture (Mooncake)

### 5.1 Layered Overview
```
Fast ─────────────────────────────────────────── Slow
Small capacity ──────────────────────────────────────────── Large capacity

┌─────────────────────────────────────────────────────────┐
│ L1 · GPU HBM (192G HBM3e per GPU) │
│ Latency: ~ns | Bandwidth: 8 TB/s (HBM internal) │
│ Purpose: Active KV Cache, participating in calculation │
│Scope: Single GPU │
├─────────────────────────────────────────────────────────┤
│ L2 · CPU DDR Memory (~882 GiB LPDDR5X per node) │
│ Latency: ~μs | Bandwidth: 900 GB/s (NVLink-C2C, non-PCIe!) │
│ Purpose: Warm KV Cache, recently used but temporarily inactive │
│ Scope: this node (multiplexed across nodes through RDMA) │
│ Key: GPU directly accessible via unified memory address space, no explicit copy required │
├─────────────────────────────────────────────────────────┤
│ L3 · Local NVMe RAID0 (~12T per node) │
│ Latency: ~100μs | Bandwidth: 25+ GB/s (4-disk RAID0) │
│Usage: Mooncake Disk Cache, large capacity cold KV Cache │
│ Scope: Direct access to this node; other nodes can be accessed through CPU memory transfer │
│ Path: /mnt/data │
└─────────────────────────────────────────────────────────┘

Note: Lustre (`/mnt/lustre` and its `/home` subpath) does not participate in KV Cache storage and is only used for model weights,
Common shared files such as data sets and checkpoints.
```
### 5.2 Comparison of each layer

| Tier | Storage media | Latency | Bandwidth | Capacity/node | Scope | Interconnection method |
|------|----------|------|------|----------|--------|---------|
| L1 | GPU HBM3e | ~ns | 8 TB/s | 768G (4×192G) | Single GPU | GPU internal |
| L2 | CPU LPDDR5X | ~μs | 900 GB/s | ~882 GiB | Local node; cross-node RDMA | **NVLink-C2C** |
| L3 | Local NVMe RAID0 | ~100μs | 25+ GB/s | ~12T | Direct to this node; transitive across nodes via CPU | PCIe |

### 5.3 Cross-node KV Cache reuse strategy
```
Node A Node B
┌──────────────┐                       ┌──────────────┐
│ L1: GPU HBM  │                       │ L1: GPU HBM  │
│ ↕ │ NVLink (72GPU in the cabinet │ ↕ │
│ NVLink-C2C │ Fully connected 1.8TB/s) │ NVLink-C2C │
│      ↕       │◄─────────────────────►│      ↕       │
│ L2: CPU DDR  │                       │ L2: CPU DDR  │
│      ↕       │◄──── RDMA ──────────►│      ↕       │
│ L3: NVMe     │    (RoCE RDMA)        │ L3: NVMe     │
│  /mnt/data   │                       │  /mnt/data   │
└──────────────┘                       └──────────────┘
```
**Path to access L3 across nodes**:

When the GPU of node B needs to access the KV Cache on the NVMe of node A, the data flow is:
```
Node A NVMe ─(PCIe)→ Node A CPU DDR ─(RDMA/NVLink)→ Node B CPU DDR ─(C2C)→ Node B GPU HBM
```
Another possible path (in-cabinet NVLink pass-through):
```
Node A NVMe ─(PCIe)→ Node A CPU DDR ─(C2C)→ Node A GPU ─(NVLink)→ Node B GPU HBM
```
Since the 72 GPUs in the cabinet are fully interconnected via NVLink Switch (1.8 TB/s/GPU), the second path may be faster in some scenarios, depending on the actual available bandwidth of CPU DDR→GPU and GPU→GPU.

**Cross-node reuse principle**:

1. **L2 cross-node**: RDMA between CPU DDR, or via GPU NVLink
2. **L3 cross-node**: NVMe data is first read into the local CPU DDR (L2), and then transmitted to the remote node through the above path
3. **NVLink within the cabinet is preferred**: Nodes within the same NVL72 cabinet are given priority to utilize the high bandwidth of NVLink full interconnection.

### 5.4 KV Cache transmission between GPU-CPU (NVLink-C2C detailed explanation)

In the GB200 architecture, the KV Cache path from GPU HBM (L1) to CPU DDR (L2) is NVLink-C2C, not PCIe:
```
┌─────────── GB200 Superchip internal KV Cache flow ──────────┐
│                                                         │
│  B200 GPU                     Grace CPU                 │
│  ┌─────────────┐              ┌─────────────┐          │
│  │ HBM3e 192G  │  NVLink-C2C  │ LPDDR5X     │          │
│  │             │◄─────────────►│             │          │
│ │ Active KV │ 900 GB/s │ Overflow KV │ │
│ │ Cache (L1) │ Bidirectional │ Cache (L2) │ │
│  │             │  cache       │             │          │
│ │ Bandwidth: 8TB/s │ coherent │ Bandwidth: ~500 │ │
│ │ (HBM internal) │ Unified address space │ GB/s (DDR) │ │
│  └─────────────┘              └─────────────┘          │
│                                                         │
│ Key: cudaMemcpy is not required for GPU access to CPU memory, │
│ Direct access through the unified virtual address space, and the hardware automatically handles data transfer.      │
│The bottleneck is NVLink-C2C’s 900 GB/s (much faster than PCIe’s 128GB/s) │
└─────────────────────────────────────────────────────────┘
```
---

## 6. Network topology observation

| Network type | Purpose | Protocol | Basis |
|---------|------|------|------|
| NVLink 5th gen | 72 GPU fully interconnected in the cabinet | NVLink | 1.8 TB/s/GPU |
| NVLink-C2C | Superchip internal CPU↔GPU | NVLink | 900 GB/s, cache coherent |
| RoCE (RDMA) | Cross-cabinet GPU-GPU, RDMA | RDMA | Native `rocep*` HCA and `gpu0rdma0`~`gpu3rdma0` |
| TCP (10.56.1.x) | Luster storage network | TCP | Mount information `@tcp` (not `@o2ib`) |

Luster uses TCP instead of RDMA (`@o2ib`), which means that storage network performance is affected by the overhead of the TCP protocol stack. For cross-node KV Cache reuse, RDMA should be used instead of Lustre.

---

## 7. Current issues and to-dos

### Problem 1: Some nodes (10+ units) are not configured with RAID

**Phenomena**: 4 NVMe data disks of 10+ nodes such as `gb200-rack-03` are in the bare disk state. There is already an md0 entry in `/etc/fstab` but the actual RAID is not created, `nofail` causes the problem to be hidden.

**Impact**:
- Waste of 12T local high-speed storage
- Mooncake L3 disk cache not working
- Inconsistent caching capabilities between nodes
- Data is written to the system disk (the root directory is used 755G / 39%)

**Troubleshooting**:```bash
pdsh -w node[01-18] 'df -h /mnt/data 2>&1'
```
**repair**:```bash
# 1. Create RAID0
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 \
  /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1

# 2. Format (note that fstab is configured with xfs)
sudo mkfs.xfs /dev/md0

# 3. Mount
sudo mkdir -p /mnt/data
sudo mount /dev/md0 /mnt/data
```
> Note: The file system type of md0 in fstab is **xfs** (not ext4), and `mkfs.xfs` must be used when formatting.

### Problem 2: Luster space usage is high

35T has been used, 16T (46%), shared by 18 nodes, and the average margin per node is less than 1T. If you use Luster to store KV Cache, you need to plan the space budget in advance.

### Problem 3: The system disk of the non-RAID node is occupied abnormally

755G / 39%, much higher than the normal node 314G / 16%. Data needs to be checked and migrated:```bash
du -sh /* 2>/dev/null | sort -rh | head -20
```
### Question 4: Multi-user management

For 4-7 concurrent users, usage specifications and quotas for /mnt/data and /mnt/lustre need to be established.

---

## 8. Mooncake L3 Disk Cache configuration points

1. **Path**: Point to `/mnt/data` (local RAID0, xfs file system)
2. **Prerequisite**: Make sure /mnt/data of all 18 nodes is mounted correctly
3. **Capacity**: About 12T per node
4. **Cross-node reuse**: L3 itself does not cross nodes, but uses RDMA (L2) across nodes.

---

## Appendix A: Quick Check of Key Information
```
System disk: nvme4n1 2T /
Data disk: nvme0n1~3 4×2.9T → md0 RAID0 ~12T → /mnt/data (xfs)
Shared storage: Luster 35T → /mnt/lustre
                                     /home → 10.56.1.3@tcp:/lustre[/home]
Luster address: 10.56.1.3@tcp:/lustre
CPU Memory (DDR): ~882 GiB LPDDR5X/node (442G for `/dev/shm`)
GPU memory (HBM): 192G HBM3e/GPU, 768G/node
CPU-GPU interconnect: NVLink-C2C, 900 GB/s, cache coherent
GPU-GPU interconnect: NVLink 5th gen, 1.8 TB/s / GPU, 72 GPU full interconnect in the cabinet
Cross-cabinet interconnection: RoCE (RDMA)
Storage network: TCP (non-RDMA)
Number of nodes: 18
Per node: 4 GPU (B200) + 2 Grace CPU (2 socket)
```
## Appendix B: Terminology comparison

| Abbreviation | Full name | Description |
|------|------|------|
| HBM | High Bandwidth Memory | GPU memory, referred to in this document as GPU memory |
| DDR / LPDDR5X | Low-Power DDR5X | CPU memory, referred to in this document as CPU memory |
| NVLink-C2C | NVLink Chip-to-Chip | Superchip intra-CPU↔GPU interconnect, 900 GB/s |
| NVLink | NVLink 5th gen | In-cabinet GPU↔GPU interconnect, 1.8 TB/s/GPU |
| RDMA | Remote Direct Memory Access | Cross-node memory direct access |
| md0 | Linux Software RAID | Local NVMe RAID0 array |

---

## Appendix C: Practical considerations for NVLink-C2C and KV Cache Offload

### C.1 How unified memory works

On GB200, the CPU and GPU share a unified virtual address space via NVLink-C2C. There are two modes in actual use:

1. **Implicit access (Unified Memory / Managed Memory)**: Memory allocated through CUDA Unified Memory or RMM can be directly accessed by both GPU and CPU. When the GPU accesses CPU-side memory, the hardware automatically pulls the data through NVLink-C2C without cudaMemcpy. Suitable for KV Cache overflow scenarios - after the GPU HBM is full, the KV Cache will naturally overflow to the CPU DDR, and the GPU will pull it back on demand when needed.

2. **Explicit transfer (DMA Copy Engine)**: The application actively schedules data transfer and uses the DMA engine of Grace CPU to perform batch transfer through NVLink-C2C. The throughput is more controllable and is suitable for prefetching the KV Cache that will be used in advance.

### C.2 Actual bandwidth expectations

NVLink-C2C nominal 900 GB/s is total bi-directional bandwidth (450 GB/s in each direction). Actual available bandwidth depends on:

- **Access Mode**: Large block sequential read and write is close to the peak, small block random access will be discounted
- **One Grace CPU connected to 2 GPUs**: 900 GB/s shared by 2 GPUs, single GPU to CPU peak of ~450 GB/s one way
- **CPU DDR Bandwidth**: LPDDR5X itself has a bandwidth of about 500 GB/s, which may become a bottleneck

### C.3 Combination with Mooncake

In Mooncake’s layered architecture:
```
Mooncake Transfer Engine
├── GPU HBM (L1) ←→ CPU DDR (L2): NVLink-C2C, 900 GB/s
│ Hardware cache coherent, unified memory available
│
├── CPU DDR (L2) ←→ Remote CPU DDR / GPU: RDMA or NVLink
│ Cross-node KV Cache reuse (removable NVLink full interconnection within the cabinet)
│
├── CPU DDR (L2) ←→ Local NVMe (L3): PCIe / DMA
│ This node disk cache read and write
│
└── Remote NVMe (L3) cross-node access:
Remote NVMe → Remote CPU DDR → RDMA/NVLink → Local GPU
Cross-node L3 reuse via CPU memory transfer
```
---

## Appendix D: Complete Bandwidth Tier Reference
```
Bandwidth level (unidirectional, approximate):

GPU HBM internal ██████████████████████████████████████ 8,000 GB/s
GPU die-to-die ████ onto
NVLink GPU-GPU ██████ 900 GB/s (1.8 TB/s bidirectional)
NVLink-C2C GPU-CPU ███ 450 GB/s (900 GB/s bidirectional)
CPU LPDDR5X           ███                                        500 GB/s
PCIe Gen5 x16 █ 64 GB/s (128 GB/s bidirectional)
NVMe RAID0 (4 disks) ▌ 25+ GB/s
NVMe SSD (single drive) 6-7 GB/s
InfiniBand NDR400     ▌                                          50 GB/s
TCP network (25GbE) 3 GB/s

Note: Luster does not participate in the KV Cache link, and its bandwidth only affects general I/O such as model loading.
```
---

## Appendix E: Summary of troubleshooting scripts

### E.1 Batch check the RAID and mount status of all nodes```bash
#!/bin/bash
# check_storage.sh - Check storage configuration on all nodes

echo "=== Check /mnt/data mount ==="
pdsh -w node[01-18] 'mountpoint -q /mnt/data && echo "OK: /mnt/data mounted" || echo "FAIL: /mnt/data NOT mounted"'

echo ""
echo "=== Check md0 RAID status ==="
pdsh -w node[01-18] 'test -e /dev/md0 && echo "OK: md0 exists" || echo "FAIL: md0 NOT found"'

echo ""
echo "=== Check Luster mount ==="
pdsh -w node[01-18] 'mountpoint -q /mnt/lustre && echo "OK: lustre mounted" || echo "FAIL: lustre NOT mounted"'

echo ""
echo "=== Check system disk usage ==="
pdsh -w node[01-18] "df -h / | tail -1 | awk '{print \$5, \$3\"/\"\$2}'"

echo ""
echo "=== Check bare disk status ==="
pdsh -w node[01-18] 'lsblk -d -n -o NAME,SIZE,TYPE /dev/nvme[0-3]n1 2>/dev/null'
```
### E.2 Single node RAID repair script```bash
#!/bin/bash
# fix_raid.sh - executed on a node without RAID
# !! Before execution, confirm that nvme0n1~3 of the node is indeed an empty disk!!

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
### E.3 System disk usage troubleshooting```bash
#!/bin/bash
# check_disk_usage.sh - Check system disk usage

echo "=== Top 20 largest directories under / ==="
sudo du -sh /* 2>/dev/null | sort -rh | head -20

echo ""
echo "=== Large files (>1G) on system disk ==="
sudo find / -xdev -type f -size +1G -exec ls -lh {} \; 2>/dev/null | sort -k5 -rh | head -20
```
