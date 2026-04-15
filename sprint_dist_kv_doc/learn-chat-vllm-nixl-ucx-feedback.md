# Chat Learning Report

> **Source**: Email thread "vLLM User Feedback for NIXL/UCX"
> **Participants**: 8 people — Kaichao You (Inferact), Bin Lei, Mikhail Brinskii, Yossi Itigin, Pen Chung Li, Aviad Yehezkel, Omri Kahalon, sima.mt (Alibaba)
> **Time span**: 2026-03-09 — 2026-03-13+ (the last email is not dated and should be later than 3/13)
> **Analysis depth**: standard (with codebase cross-reference)

---

## TL;DR

- **The transparency and controllability of UCX on the GB200 cluster is extremely poor**: gdrcopy version mismatch directly crashes, PCI ACS causes RDMA to fail silently, dependencies between transports are opaque (for example, `cuda_copy` is an implicit dependency of GPU memory detection)
- **Coupling of control plane and data plane is the core pain point**: Users hope that the data plane only uses NVLink/cuda_ipc, and the control plane uses TCP, but UCX does not support separate configuration, `UCX_TLS` is a global setting
- **NVIDIA UCX team has claimed short-term action items** (within 2-3 weeks): disable gdrcopy, PCI ACS graceful degradation, streamline UCX protocol stack
- **UCX's mmap hook causes a memory leak** (publicly reported by the Mistral team), another example of UCX's "too many features"
- **`UCX_TLS=rc,cuda_copy,^cuda_ipc`** is the key recipe to force GPU Direct RDMA (bypass NVLink), where `cuda_copy` cannot be omitted

---

## Topic Overview
| Topic | Participants | Proportion | Is there a conclusion |
|------|--------|------|-----------|
| Actual list of issues with UCX on GB200 cluster | Kaichao, Pen | ~25% | Yes (issues listed) |
| UCX transport configuration is opaque | Kaichao, Mikhail, Bin | ~25% | Partial (short-term plan available, long-term undecided) |
| Control plane vs data plane separation | Kaichao, Mikhail, Bin | ~20% | No (UCX does not support it yet, has been added to TODO) |
| NVIDIA Short/Long Term Action Plan | Bin, Yossi | ~15% | Yes (short term 2-3 weeks, long term TBD) |
| GPU Direct RDMA forced use | Kaichao | ~10% | Yes (available recipes found) |
| UCX log noise problem | Kaichao | ~5% | No (no reply) |

---

## Core Knowledge Points / Key Knowledge Points

### 1. Three major UCX pitfalls on GB200 clusters

**Background**: Kaichao encountered a series of problems when deploying vLLM disaggregated inference on the GB200 rack provided by Crusoe.

**Key Points**:
- **gdrcopy version mismatch**: The cloud vendor has installed gdrcopy, but the driver version used during compilation is inconsistent with the runtime, causing UCX to directly exit with an error. UCX does not have a graceful degradation mechanism.
- **PCI ACS silently blocks GPU Direct RDMA**: Although Crusoe provides a rack, PCI Access Control Services is enabled. The IB driver reports that the RDMA NIC can access the GPU (`ibv_query_device` returns success), but an error is reported when the actual transmission occurs. Users can only manually disable some RDMA NICs.
- **"no remote ep address" metaphysical error**: This error is reported without setting any UCX flag on a certain GB200 cluster, but the other cluster is normal. Finally, it was found that the error disappeared after setting `UCX_TLS=rc,cuda_copy,^cuda_ipc`, but the reason was unknown.**Conclusion**: The maturity of the cloud vendor's GB200 cluster software stack is very low, and UCX's error handling and diagnostic capabilities are insufficient.

**Codebase verification**: It is confirmed that `UCX_TLS=cuda_copy,cuda_ipc,tcp` + `UCX_CUDA_IPC_ENABLE_MNNVL=y` is the current standard configuration in the `manual_nixl_prefill_01.sh` and vigil recipe of the local warehouse, indicating that these problems have been solved and solidified into the configuration template.

### 2. Implicit dependencies between UCX Transport

**Background**: Kaichao found that `UCX_TLS=rc` failed to register GPU memory when trying to force GPU Direct RDMA.

**Key Points**:
- The `cuda_copy` transport is not only used for GPU<->CPU copy, but is also a necessary condition for UCX to detect GPU memory**. Without it, UCX cannot recognize the pointer pointing to the GPU, causing memory registration to fail.
- This dependency is not stated in the UCX documentation, and Kaichao discovered it through GitHub issue openucx/ucx#10929.
- Final recipe: `UCX_TLS=rc,cuda_copy,^cuda_ipc`
  - `rc` = Reliable Connection (InfiniBand verb, go GPU Direct RDMA)
  - `cuda_copy` = required for UCX to recognize GPU memory
  - `^cuda_ipc` = exclude cuda_ipc to prevent NVLink

**Conclusion**: UCX transport names and interdependencies are "arcane knowledge" and extremely user-unfriendly.

### 3. Control plane vs data plane separation requirements

**Background**: In vLLM disaggregated inference, the control plane (handshake/notification) of NIXL uses TCP, and the data plane (KV cache transfer) should use NVLink or RDMA. But `UCX_TLS` is a global configuration.

**Key Points**:
- Kaichao hopes: data plane = `cuda_ipc` only, control plane = `tcp`. However, after setting `UCX_TLS=cuda_ipc,cuda_copy,tcp`, the data plane may silently fall back to tcp to transmit GPU data, and users can only know this by looking at the logs.
- Mikhail (UCX team) confirmed: Currently UCX will automatically fallback to tcp when cuda_ipc is unavailable, this is by design.
- Bin Lei proposed: The data path and control path should be supported to configure transport separately. If the data path cannot find cuda_ipc, an error will be reported instead of fallback.
- Mikhail confirmed: **Not currently supported**, has been added to TODO.

**Conclusion**: This is an architectural level deficiency. Currently, it can only be verified through logs whether the expected transport is used.

**Codebase verification**: `ai-infra-skills/knowledge/serving/disaggregated-serving.md` clearly requires "Separate control-plane addressing from KV-data transfer", indicating that this is a known architectural principle and is not yet supported at the UCX level. The vLLM layer separates the control plane through `VLLM_NIXL_SIDE_CHANNEL_HOST/PORT`, but the data plane transport selection still depends on the UCX global configuration.

### 4. NVIDIA’s short-term action plan (2-3 weeks)

**Background**: Bin Lei gave a formal response on behalf of the NVIDIA UCX/NIXL team.

**Key Points**:
1. **Disable gdrcopy** — NIXL processes do not use it and should not crash with it
2. **Detect PCI ACS and gracefully downgrade to NVLink (cuda_ipc)** — when GDR is not available
3. **Improve UCX_TLS user experience**
4. **Streamlined UCX feature set**:
   - Disable memory hooks (addresses mmap leak issue reported by Mistral)
   - Limit available protocols to those actually used by NIXL
   - Force GPU data to only go through IB or NVLink**Conclusion**: The short-term fix focuses on "subtraction" - removing UCX functions that NIXL does not need to avoid side effects.

### 5. Long-term plans and controversies

**Background**: Bin Lei also mentioned two long-term plans.

**Key Points**:
- **Send notification through NVLink**: Kaichao believes that it is not necessary. The notification is on the CPU. Using NVLink will increase the delay. It is recommended to use TCP directly.
- **Further refactor UCX, retaining only the functions required by NIXL**: This is a more radical direction.

**Conclusion**: Divergence in long-term direction. Kaichao's feedback (notifications should not use NVLink) is reasonable - notifications are small messages, low frequency, and initiated by the CPU. NVLink's high-bandwidth and low-latency advantages cannot be used here.

### 6. UCX mmap Hook causes memory leaks

**Background**: Kaichao cited the Mistral team's public blog.

**Key Points**:
- UCX hooks all `mmap` calls of a process (used to track memory registrations), which leads to undesirable memory leaks in LLM inference scenarios.
- Reference: https://mistral.ai/news/debugging-memory-leak-in-vllm
- This is another example of UCX's "too many functions" - UCX was originally designed for HPC, and many functions are harmful in AI inference scenarios.

**Conclusion**: NVIDIA's short-term plan to "disable memory hooks" directly responds to this problem.

### 7. UCX release rhythm

**Key Points**:
- NIXL: Published once a month
- UCX 1.21: Target April-May 2026
- UCX PR openucx/ucx#11100 is useful for vLLM, you need to wait for the next release
- NVIDIA is working to shorten UCX release cycles to match NIXL

---

## Technical Insights
1. **UCX's transport abstraction leaks seriously**: `cuda_copy` is both a "GPU<->CPU copy transport" and a "GPU memory detection mechanism". This dual role violates the principle of least surprise and is a source of user confusion.

2. **The maturity of cloud vendor GB200 software stack is the bottleneck**: Hardware (NVLink 1.8 TB/s) far exceeds software capabilities. The nvidia-imex service was accidentally killed, the gdrcopy version was mismatched, the PCI ACS was misconfigured—these were not problems that users could diagnose on their own. This is completely consistent with the judgment of "new hardware is often blocked more by software maturity than by hardware capability" in `ai-infra-skills/knowledge/hardware/b300-blackwell-notes.md`.

3. **IB driver reporting capability != actually available**: In the PCI ACS scenario, `ibv_query_device` returns that the GPU is reachable, but the actual transmission fails. This is a dangerous "false positive", which means that the IB driver's capability reports cannot be trusted and end-to-end transmission testing must be done.

4. **Conflict between UCX’s HPC genes and AI inference scenarios**: UCX was originally designed for general communication of MPI/HPC, and a large number of functions (memory hooks, multi-transport automatic selection, active messages) are not only useless in LLM inference, but also cause problems. NVIDIA's long-term direction (cutting UCX for NIXL) is correct.

5. **The value of configuration solidification**: vigil's YAML recipes in the local warehouse have solidified UCX configurations (`UCX_TLS`, `UCX_CUDA_IPC_ENABLE_MNNVL`, RC timeout, etc.) into templates, indicating that these problems have evolved from "manual debugging every time" to the "standardized configuration" stage.---

## Decisions and Rationale
| Decision | Rationale | Owner | Status |
|----------|-----------|-------|--------|
| gdrcopy disabled in NIXL process | NIXL does not use gdrcopy, but its presence causes crashes | NVIDIA UCX team | Short-term TODO (2-3 weeks) |
| Detect PCI ACS and downgrade to cuda_ipc | When GDR is unavailable, NVLink should be used automatically instead of reporting an error | NVIDIA UCX team | Short-term TODO |
| Disable UCX memory hooks | Causing mmap memory leak (Mistral report) | NVIDIA UCX team | Short-term TODO |
| Limit GPU data transport to IB or NVLink | Avoid tcp fallback to transfer GPU data | NVIDIA UCX team | Short-term TODO |
| Data plane/control plane separation configuration | Users need to precisely control which transports are used for data transmission | NVIDIA UCX team | Long-term TODO (**not supported yet**) |
| Notification does not use NVLink | Kaichao objects, believing that it is more reasonable to use TCP for CPU notification | To be discussed | Open |

---

## Debates and Disagreements
### Is NVLink Notification necessary?

- **Bin Lei (NVIDIA)**: Long-term plans include "Implementing notifications via NVLink"
- **Kaichao You (Inferact)**: Objection. The notification message is on the CPU, and the delay through NVLink is higher and unnecessary. It is recommended that NIXL directly uses TCP to send notifications and does not use the active message function of UCX.
- **Resolution**: No consensus, but Kaichao's argument is technically more sound.

### Default behavior of tcp Fallback

- **Mikhail (UCX)**: UCX is designed to automatically fallback to tcp when cuda_ipc is unavailable. The question is "What behavior does the user want?"
- **Kaichao + Bin Lei**: I hope the data path will not fallback, but report an error. Users would rather see errors than silently transfer GPU data via tcp.
- **Resolution**: Mikhail confirmed that detached configuration is not currently supported and has been added to TODO.

---

## Verification Results
| Claim | Source | Status | Notes |
|-------|--------|--------|-------|
| `cuda_copy` is a necessary condition for UCX to detect GPU memory | Kaichao cited GitHub issue | **confirmed** | openucx/ucx#10929 confirmed |
| `UCX_TLS=cuda_ipc,cuda_copy,tcp` is a standard recipe | Email discussion | **confirmed** | Local warehouse `manual_nixl_prefill_01.sh`, vigil recipes all use this configuration |
| UCX memory hooks cause memory leaks | Kaichao quoted Mistral blog | **confirmed** | Reference link exists, NVIDIA has included short-term fixes |
| UCX does not support control/data path separation configuration | Mikhail confirmed | **confirmed** | Mikhail clearly said "not currently supported" |
| NIXL monthly release, UCX 1.21 target April-May | Yossi | **unverified** | From NVIDIA official staff, credible but the time point may change |
| The nvidia-imex service may be killed, causing NVLink to break | Kaichao | **confirmed** | The local `machine_doc.md` is supported by the GB200 architecture document |---

## Glossary
| Term | Meaning | Context |
|------|---------|---------|
| **UCX** | Unified Communication X — NVIDIA's unified communications framework, supporting multiple transports | NIXL underlying communication library |
| **NIXL** | NVIDIA Inference eXchange Library — KV cache transport library for LLM inference | Transport layer for vLLM disaggregated inference |
| **UCX_TLS** | UCX Transport Layer Selection — environment variable that controls which transport UCX uses | For example `cuda_ipc,cuda_copy,tcp` |
| **cuda_ipc** | CUDA Inter-Process Communication — Zero-copy sharing between GPUs, based on NVLink | Preferred transport between GPUs on the same node |
| **cuda_copy** | GPU<->CPU memory copy transport in UCX, **also a necessary component for GPU memory detection** | Often mistaken as optional, but actually necessary |
| **rc** | Reliable Connection — Reliable connection for InfiniBand verb | The transport used by GPU Direct RDMA |
| **GPU Direct RDMA (GDR)** | Allows RDMA NIC to directly read and write GPU memory, bypassing the CPU | Optimal path for cross-node GPU data transfer |
| **gdrcopy** | NVIDIA's user-mode GPU memory copy library, used for low-latency GPU access to small data | NIXL is not used but UCX may try to call |
| **PCI ACS** | PCI Access Control Services — PCIe-level access control | When enabled, prevents RDMA NICs from directly accessing the GPU |
| **MNNVL** | Multi-Node NVLink — Cross-node NVLink connection for GB200 | `UCX_CUDA_IPC_ENABLE_MNNVL=y` enabled |
| **nvidia-imex** | NVIDIA Infrastructure Manager and Executor - a daemon that manages multi-node NVLink | NVLink breaks silently after being killed |
| **active message** | UCX's remote procedure call mechanism, used to send notification/control messages | Kaichao believes that TCP should be used instead |
| **Disaggregated Inference** | Inference architecture that separates prefill and decode to different GPU groups | P/D separated deployment of vLLM |

---

## Open Questions and Further Study

1. What is the root cause of the **"no remote ep address" error? ** Kaichao said that the error disappeared after setting `UCX_TLS=rc,cuda_copy,^cuda_ipc`, but did not explain why. Inconsistent behavior between two GB200 clusters, possibly related to RDMA NIC configuration or IB subnet.
2. What is the specific content of **UCX PR openucx/ucx#11100? ** The email mentioned that vLLM was "pretty useful", but did not explain what specific changes were made.
3. **IPv4/IPv6 noise issue with `UCX_LOG_LEVEL=debug`** — raised by Kaichao but did not receive a reply. This problem may reduce debugging efficiency.
4. **Multi-threading in vLLM + NIXL bug that prevents UCX from using CUDA IPC** - mentioned in the first email, but the details were not discussed later. This is a vLLM level issue, not a UCX level issue.
5. **Have NVIDIA’s short-term action items (2-3 weeks commitment) been implemented? **The email is from March 2026, the current date is April, it should be verifiable.---

## Action Items

| Item | Owner | Status |
|------|-------|--------|
| Disable gdrcopy in NIXL flows | NVIDIA UCX team (Bin Lei) | Short-term TODO |
| PCI ACS detection + graceful degradation | NVIDIA UCX team | Short-term TODO |
| Improve UCX_TLS user experience | NVIDIA UCX team | Short-term TODO |
| Disable memory hooks | NVIDIA UCX team | Short-term TODO |
| Limit GPU transport to IB/NVLink only | NVIDIA UCX team | Short-term TODO |
| NVLink notification (long-term) | NVIDIA UCX team | Long-term plans (disputed) |
| UCX refactored to retain only NIXL functionality (long-term) | NVIDIA UCX team | Long-term plan |
| Data plane/control plane separation configuration | NVIDIA UCX team (Mikhail) | TODO (no timetable) |

---

## Recommended further reading

- `ai-infra-skills/knowledge/serving/disaggregated-serving.md` — P/D disaggregated architecture mode
- `vllm/sprint_dist_kv_doc/machine_doc.md` — Detailed explanation of GB200 hardware architecture
- `ai-infra-skills/knowledge/hardware/nvidia-datacenter-gpu-matrix.md` — GPU specs comparison
- `vllm/docs/features/nixl_connector_usage.md` — NIXL connector configuration guide
- `serving-systems-expert` agent — in-depth discussion of serving architecture