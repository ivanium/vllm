# Handoff Prompt — Phased `NIXL + MooncakeStore` Bring-up on `gb200-rack1-01` + `gb200-rack1-03`

## 2026-04-08 最新状态（新的 Agent session 先读）

- 当前运行环境已经收敛到单一环境：
  - `/home/aoshen/code/uv_envs/py312`
- 不要再用 `/home/aoshen/setup_new_cluster/vllm/.venv` 做 bring-up
  - 这个 repo-local `.venv` 是后来为定制 `vllm` 建的专用环境
  - 但它和顶层 `README.md` 指定的主环境分叉后，已经造成过排障混乱
- `py312` 里当前已经确认可用：
  - `vllm`
  - `vigil`
  - `vllm-router`
  - `nixl`
- 当前最值得接着跑的 spec 不是远端双机，而是本地单机最小链路：
  - `/home/aoshen/setup_new_cluster/vigil/examples/pd_1p1d_nixl_only_local_01.yaml`
- 最新一次最有价值的实测是：
  - 日志目录：`/home/aoshen/setup_new_cluster/vigil/logs/pd_1p1d_nixl_only_local_01_20260408T065431Z`
  - `NIXL is available`
  - 但随后在 backend 初始化阶段失败，报：
    - `nixl_cu12._bindings.nixlBackendError: NIXL_ERR_BACKEND`
- 这说明当前主阻塞已经不是：
  - `vllm` / `vigil` / `router` 没装
  - 也不是 `NIXL is not available`
- 当前真正主阻塞是：
  - `NIXL` backend 初始化 / register_memory 失败
  - 下一轮应优先排查 `nixl` CUDA 变体、底层 runtime、以及 `NIXL_ERR_BACKEND` 的直接触发条件
- 远端双机已经有一次历史实验：
  - `01 + 04` 的 `vigil` dry-run 通过
  - 但 Slurm step launch 失败，job `2225`
  - 这是次要线索，当前不应优先于单机 `NIXL_ERR_BACKEND`

## 已确认的事实

- 当前机器就是 `gb200-rack1-01`，IP 是 `192.168.0.101`
- 目标第二台机器是 `gb200-rack1-03`，IP 是 `192.168.0.103`
- `/etc/hosts` 已经有整排机器映射，不需要再猜 hostname
- Slurm 可用，`vigil` 的多机模式应使用 `mode: remote`，不是 `local + SSH`
- `vigil` 目前只支持 Slurm 作为 remote launcher
- `vigil` 在 remote 模式下按 `serving.roles` 的顺序分配节点
  - 如果 `nodelist: gb200-rack1-[01,03]`
  - 且 role 顺序是 `prefill` 然后 `decode`
  - 那么 prefill 会落到 `01`，decode 会落到 `03`
- 最终目标不是直接做 Mooncake PD，而是分阶段演进到 `MultiConnector(NIXL, MooncakeStoreConnector)`
- 当前版本不支持 `multi-GPU node + disk offloading`
  - 这项明确排除在本轮之外
- `gb200-rack1-03` 的 Slurm 状态不要写死
  - 本轮排障过程中它出现过 `alloc`，也出现过 `idle`
  - 每次正式远端 bring-up 前都必须重新用 `sinfo` / `squeue` 现查

## 本地代码与环境现状

- 当前唯一推荐使用的运行环境是：
  - `/home/aoshen/code/uv_envs/py312`
  - 这是 `/home/aoshen/setup_new_cluster/README.md` 指定的主环境
- `vllm` repo: `/home/aoshen/setup_new_cluster/vllm`
  - branch/commit: `feat/mooncake-store-int` 系列，当前 `9998c1db5`
  - 当前应通过 `py312` 中的 editable install 使用这个 repo
  - 不要优先使用 `/home/aoshen/setup_new_cluster/vllm/.venv`
- `vigil` repo: `/home/aoshen/setup_new_cluster/vigil`
  - 当前 commit: `a6bfa06`
  - CLI 已安装进 `/home/aoshen/code/uv_envs/py312`
- `router` repo 不在当前 workspace 根目录，但存在于：
  - `/home/aoshen/code/router`
  - commit: `4df9bbc`
  - 当前 bring-up 应直接使用：
    - `/home/aoshen/code/uv_envs/py312/bin/vllm-router`
  - 不要再把 `./target/release/vllm-router` 当成前置条件
- `vllm-bench` repo 在：
  - `/home/aoshen/code/vllm-bench`
  - commit: `22f7597`
  - `target/release/vllm-bench` 已经 build 好
- `py312` 里当前已经确认存在：
  - `vllm`
  - `vigil`
  - `vllm-router`
  - `nixl`
- 当前版本已经支持普通 `multi-GPU node + CPU offloading`
  - 因此它适合作为 `Phase B` 的中间验证阶段
- `crusoe-inference` 数据集仓库目前不在 `/home/aoshen` 下
  - 如果要跑文档里的 `synthetic_sharegpt_v3.json` / `codex_swebenchpro.json`
  - 需要额外 clone

## 从 `[INTERNAL] Crusoe Inference.docx` 提炼出来的结论

- 这份内部文档的主线是：先把 `benchmarking + vigil` runbook 跑顺，再继续做更激进的优化
- Crusoe benchmark 的组织方式是以 `vigil recipe` 为核心
- 文档明确建议先聚焦 `PD-disagg spec`
- 文档里的 benchmark 形态分三层：
  - `Simple test mode`: 固定 `70k input / 200 output`
  - 全局共享前缀：前 `12k`
  - 对话内 prefix cache hit：逐步到 `90%+`
  - `Loadtest mode`: 按 traffic 分布模拟
  - `Shadow mode`: 使用真实 swebenchpro 导出的 traffic dump
- 文档里的 Mooncake 章节明确写了：
  - Mooncake 只在单机场景测试过
  - `PD is not yet verified at all`
- 因此这次用 `01 + 03` 跑两机 bring-up，第一步仍然应该是先把 **NIXL Connector 的 1P1D recipe 跑通**
- 但最终目标已经更新为：
  - 在跑通 PD 后，再把 `MooncakeStoreConnector` 通过 `MultiConnector` 接进来
  - `[INTERNAL] Crusoe Inference.docx` 里的 Mooncake 内容只能作为方向参考
  - 不能直接当作“已验证的双机 PD 方案”照搬

## 这次任务的最终目标

- 最终目标是把 `NIXL connector` 和 `MooncakeStoreConnector` 通过 `MultiConnector` 组装起来
- 目标拓扑固定为：
  - `gb200-rack1-01 = prefill`
  - `gb200-rack1-03 = decode`
- 当前阶段默认：
  - `1P1D`
  - 每机 1 卡
  - 模型先用 `Qwen/Qwen3-8B`
- 这轮不做 `multi-GPU node + disk offloading`
- 这轮不使用 `MooncakeConnector` 作为主传输链路
- `MooncakeStore` 这轮只作为 store / fallback，不作为主 PD transport

## 分阶段执行计划

### Phase A: 先跑通 `1P1D + NIXL`

- `prefill`: `NixlConnector`
- `decode`: `NixlConnector`
- 目标是先把 `vigil`、Slurm、router、PD 基本链路跑通
- benchmark 只做 smoke，不直接上 Crusoe 大流量

### Phase B: 用 CPU offload 验证 `MultiConnector`

- `prefill`: `MultiConnector(NixlConnector, SimpleCPUOffloadConnector)`
- `decode`: `NixlConnector`
- 这是过渡阶段，目的不是 offload 本身，而是先确认 `MultiConnector` 的配置方式正确
- 只改 `prefill`，不同时改两边

### Phase C: `prefill` 接入 `MooncakeStore`

- `prefill`: `MultiConnector(NixlConnector, MooncakeStoreConnector)`
- `decode`: `NixlConnector`
- 这是最终目标的第一半
- 先验证 `NIXL` 主链路 + `MooncakeStore` store / fallback 的最小可用性
- 只启用当前支持的内存 / CPU 路径，不启用 disk offload

### Phase D: `decode` 也接入 `MooncakeStore`

- `prefill`: `MultiConnector(NixlConnector, MooncakeStoreConnector)`
- `decode`: `MultiConnector(NixlConnector, MooncakeStoreConnector)`
- 这是“真正 distributed / decode fallback”阶段
- 只有 `Phase C` 稳定后才能进入

## 配置约定

- `MultiConnector` 的 connector 顺序固定为：
  - `NixlConnector`
  - `MooncakeStoreConnector`
- 原因是：
  - `load` 按顺序找 source
  - `save` 写入所有 connector
  - 所以 `NIXL` 必须排第一，`MooncakeStore` 必须排第二
- `Phase B` 的 connector 顺序固定为：
  - `NixlConnector`
  - `SimpleCPUOffloadConnector`
- 改动顺序固定为：
  - 先改 `prefill`
  - 再改 `decode`
- router 继续走普通 PD / NIXL 路线
  - 不加 `--kv-connector mooncake`

## 每阶段操作步骤

每个阶段都按同一模板执行：

1. 确认 `gb200-rack1-03` 是否空闲
2. 准备对应阶段的 `vigil` spec
3. 先跑 `--dry-run`
4. 再正式启动服务
5. 跑 smoke benchmark
6. 检查 router / prefill / decode 日志
7. 满足通过标准后再进入下一阶段

固定说明：

- 如果 `03` 一直忙：
  - 可以先做 dry-run
  - 或者只做临时节点 smoke
  - 正式结果仍以 `01 + 03` 为准
- `Phase A` 的第一份 spec 应该是最小 NIXL 基线
- 不允许跳过 `Phase B` 直接上 `MooncakeStore`

## 详细操作计划

### Phase A: `1P1D + NIXL` 基线

目标：

- 先把 `vigil + Slurm + router + PD` 最小链路跑通

建议 spec 文件：

- `/home/aoshen/setup_new_cluster/vigil/examples/pd_1p1d_nixl_only_01_03.yaml`

操作步骤：

1. 确认 `03` 是否空闲

```bash
sinfo -N -o '%N %t %P' | grep gb200-rack1-03
```

2. 准备环境并补齐工具链

```bash
source /home/aoshen/code/uv_envs/py312/bin/activate
export HF_HOME=/mnt/lustre/hf-models
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=/home/aoshen/.local/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export VLLM_RPC_BASE_PATH=/tmp/aoshen_rpc

cd /home/aoshen/setup_new_cluster/vigil
/home/aoshen/.local/bin/uv pip install -e .
```

3. 补充检查

```bash
source /home/aoshen/code/uv_envs/py312/bin/activate
which vigil
which vllm
which vllm-router
test -x /home/aoshen/code/vllm-bench/target/release/vllm-bench && echo ok
```

4. 写最小 `NixlConnector` spec

- `mode: remote`
- `model: Qwen/Qwen3-8B`
- `prefill` 和 `decode` 都用：

```text
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}}'
```

- `nodelist` 固定：

```yaml
gb200-rack1-[01,03]
```

- `roles` 顺序固定：
  - `prefill`
  - `decode`

5. 先做 dry-run

```bash
source /home/aoshen/code/uv_envs/py312/bin/activate
cd /home/aoshen/setup_new_cluster/vigil
vigil -c examples/pd_1p1d_nixl_only_01_03.yaml --dry-run
```

6. 再做 smoke-run

- 小 workload 即可：
  - `random-input-len 8192`
  - `random-output-len 128`
  - `max-concurrency 1`

7. 检查日志

- `precheck.log`
- `env_info.log`
- `slurm.out`
- router log
- prefill engine log
- decode engine log

通过标准：

- 服务能完整起停
- NIXL 不报握手或 side-channel 错误
- smoke 请求全部成功

### Phase B: `MultiConnector + SimpleCPUOffloadConnector`

目标：

- 不碰 Mooncake，先验证 `MultiConnector` 在 PD 场景下的接线方式

建议 spec 文件：

- `/home/aoshen/setup_new_cluster/vigil/examples/pd_1p1d_nixl_cpuoffload_01_03.yaml`

操作步骤：

1. 复制 `Phase A` 的 spec
2. 只改 `prefill` 的 `kv-transfer-config`
3. `decode` 保持 `NixlConnector`

`prefill` 的 connector 顺序固定为：

```text
NixlConnector -> SimpleCPUOffloadConnector
```

执行顺序：

```bash
source /home/aoshen/code/uv_envs/py312/bin/activate
cd /home/aoshen/setup_new_cluster/vigil
vigil -c examples/pd_1p1d_nixl_cpuoffload_01_03.yaml --dry-run
vigil -c examples/pd_1p1d_nixl_cpuoffload_01_03.yaml
```

检查重点：

- `MultiConnector` 初始化成功
- `SimpleCPUOffloadConnector` 正常注册
- decode 仍稳定使用 NIXL

通过标准：

- 行为不退化
- `prefill` 侧日志可以确认 `MultiConnector` 已生效

### Phase C: `prefill` 接入 `MooncakeStoreConnector`

目标：

- 验证 `NIXL` 主链路 + `MooncakeStore` 旁路存储是否成立

建议 spec 文件：

- `/home/aoshen/setup_new_cluster/vigil/examples/pd_1p1d_nixl_mooncakestore_prefill_01_03.yaml`

操作步骤：

1. 复制 `Phase B` 的 spec
2. 只把 `prefill` 的第二个 connector 从 `SimpleCPUOffloadConnector` 换成 `MooncakeStoreConnector`
3. `decode` 仍保持 `NixlConnector`
4. 补齐 Mooncake 需要的环境变量

关键环境变量：

```bash
export MOONCAKE_CONFIG_PATH=<mooncake config path>
```

说明：

- 这一阶段只启用当前支持的内存 / CPU 路径
- 不启用 disk offload
- 不引入 `MooncakeConnector`

执行顺序：

```bash
source /home/aoshen/code/uv_envs/py312/bin/activate
cd /home/aoshen/setup_new_cluster/vigil
vigil -c examples/pd_1p1d_nixl_mooncakestore_prefill_01_03.yaml --dry-run
vigil -c examples/pd_1p1d_nixl_mooncakestore_prefill_01_03.yaml
```

检查重点：

- `MooncakeStoreConnector` 能成功初始化
- 请求成功率不低于 `Phase A/B`
- 日志里能看到 store / load 路径被触发

通过标准：

- `prefill` 可以同时写 `NIXL` 和 `MooncakeStore`
- 服务稳定，无明显 connector 错误

### Phase D: `decode` 也接入 `MooncakeStoreConnector`

目标：

- 实现真正的 distributed fallback

建议 spec 文件：

- `/home/aoshen/setup_new_cluster/vigil/examples/pd_1p1d_nixl_mooncakestore_both_01_03.yaml`

操作步骤：

1. 复制 `Phase C` 的 spec
2. 把 `decode` 侧也改成：

```text
MultiConnector(NixlConnector, MooncakeStoreConnector)
```

3. 保持 connector 顺序不变：

```text
NixlConnector -> MooncakeStoreConnector
```

4. 先 dry-run，再正式启动

```bash
source /home/aoshen/code/uv_envs/py312/bin/activate
cd /home/aoshen/setup_new_cluster/vigil
vigil -c examples/pd_1p1d_nixl_mooncakestore_both_01_03.yaml --dry-run
vigil -c examples/pd_1p1d_nixl_mooncakestore_both_01_03.yaml
```

5. 先跑 smoke benchmark，再跑轻量 prefix-heavy workload

检查重点：

- decode 侧 `MultiConnector` 初始化成功
- decode 侧具备从 `MooncakeStore` 读取的能力
- 正常路径仍优先使用 NIXL

通过标准：

- decode 侧没有因为第二个 connector 引入新的 load 失败
- 请求链路保持稳定
- 日志里能确认 decode 具备 `MooncakeStore` fallback 能力

### Phase 结果归档

每个阶段都至少保留以下产物：

- `precheck.log`
- `env_info.log`
- `slurm.out`
- router log
- prefill / decode engine log
- `vllm-bench` 结果目录

建议固定关注这些指标：

- `TTFT`
- `TPOT`
- `ITL`
- `E2EL`
- 成功率

### 2026-04-08 当前实测产物

- 单机旧失败样本：
  - `/home/aoshen/setup_new_cluster/vigil/logs/pd_1p1d_nixl_only_local_01_20260408T064433Z`
- 单机旧失败样本：
  - `/home/aoshen/setup_new_cluster/vigil/logs/pd_1p1d_nixl_only_local_01_20260408T064726Z`
  - 这个阶段的关键信号仍是 `NIXL is not available`
- 单机最新样本：
  - `/home/aoshen/setup_new_cluster/vigil/logs/pd_1p1d_nixl_only_local_01_20260408T065431Z`
  - 这是当前最重要的排障基线
  - 关键信号是 `NIXL is available`
  - 但随后失败于 `nixl_cu12._bindings.nixlBackendError: NIXL_ERR_BACKEND`
  - 最关键日志：
    - `/home/aoshen/setup_new_cluster/vigil/logs/pd_1p1d_nixl_only_local_01_20260408T065431Z/prefill_0.log`
- 远端历史样本：
  - `/home/aoshen/setup_new_cluster/vigil/logs/pd_1p1d_nixl_only_01_04_20260408T063944Z`
  - `dry-run` 通过
  - 但正式 Slurm job `2225` 失败于 step launch

## `vigil` YAML 模板

下面给出 4 份建议 spec 骨架，目标是让接手的人可以直接复制后再做少量参数微调。

统一约定：

- 一次只跑一个 phase，所以 4 份模板都复用：
  - `router_port: 30000`
  - `prefill base_port: 8100`
  - `decode base_port: 8200`
- 如果机器上已有同端口进程，再整体改端口，不要只改其中一个
- router 固定用已经安装好的：
  - `/home/aoshen/code/uv_envs/py312/bin/vllm-router`
- `nodelist` 固定：

```yaml
gb200-rack1-[01,03]
```

### Phase A YAML: `pd_1p1d_nixl_only_01_03.yaml`

```yaml
model: Qwen/Qwen3-8B
mode: remote
precheck: true
collect_env: true
router_port: 30000

serving:
  roles:
    - role: prefill
      count: 1
      nodes_per_instance: 1
      gpus_per_node: 1
      base_port: 8100
      vllm_engine:
        repo_path: /home/aoshen/setup_new_cluster/vllm
        cmd: >-
          /home/aoshen/code/uv_envs/py312/bin/vllm serve {model}
          --port {port}
          -tp 1
          --max-model-len 32768
          --enable-prefix-caching
          --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}}'
          --disable-uvicorn-access-log
        env:
          CUDA_VISIBLE_DEVICES: "0"
          HF_HOME: /mnt/lustre/hf-models
          CUDA_HOME: /usr/local/cuda-13.1
          LD_LIBRARY_PATH: /usr/local/cuda-13.1/lib64
          VLLM_RPC_BASE_PATH: /tmp/aoshen_rpc
          VLLM_ENGINE_READY_TIMEOUT_S: "1800"
          VLLM_RPC_TIMEOUT: "600000"
        health_check:
          url: http://localhost:{port}/health
          timeout_s: 900
          poll_interval_s: 5

    - role: decode
      count: 1
      nodes_per_instance: 1
      gpus_per_node: 1
      base_port: 8200
      vllm_engine:
        repo_path: /home/aoshen/setup_new_cluster/vllm
        cmd: >-
          /home/aoshen/code/uv_envs/py312/bin/vllm serve {model}
          --port {port}
          -tp 1
          --max-model-len 32768
          --enable-prefix-caching
          --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}}'
          --disable-uvicorn-access-log
        env:
          CUDA_VISIBLE_DEVICES: "0"
          HF_HOME: /mnt/lustre/hf-models
          CUDA_HOME: /usr/local/cuda-13.1
          LD_LIBRARY_PATH: /usr/local/cuda-13.1/lib64
          VLLM_RPC_BASE_PATH: /tmp/aoshen_rpc
          VLLM_ENGINE_READY_TIMEOUT_S: "1800"
          VLLM_RPC_TIMEOUT: "600000"
        health_check:
          url: http://localhost:{port}/health
          timeout_s: 900
          poll_interval_s: 5

  router:
    repo_path: /home/aoshen/code/router
    cmd: >-
      /home/aoshen/code/uv_envs/py312/bin/vllm-router
      --policy round_robin
      --host 0.0.0.0
      --port {router_port}
      --intra-node-data-parallel-size 1
      --vllm-pd-disaggregation
      {prefill_flags}
      {decode_flags}
    health_check:
      url: http://localhost:{router_port}/health
      timeout_s: 120
      poll_interval_s: 5

  remote:
    launcher: slurm
    slurm:
      partition: batch
      time_limit: "02:00:00"
      grace_period_s: 120
      nodelist: "gb200-rack1-[01,03]"

post_serve:
  - cmd: >-
      ./target/release/vllm-bench
      --backend openai-chat
      --base-url http://localhost:{router_port}
      --model {model}
      --dataset-name random
      --random-input-len 8192
      --random-output-len 128
      --num-prompts 4
      --max-concurrency 1
      --result-dir "{log_dir}/smoke"
      --save-result
    repo_path: /home/aoshen/code/vllm-bench
```

### Phase B YAML: `pd_1p1d_nixl_cpuoffload_01_03.yaml`

在 `Phase A` 基础上只改 `prefill` 的 `kv-transfer-config`，其他字段保持不动。

```yaml
model: Qwen/Qwen3-8B
mode: remote
precheck: true
collect_env: true
router_port: 30000

serving:
  roles:
    - role: prefill
      count: 1
      nodes_per_instance: 1
      gpus_per_node: 1
      base_port: 8100
      vllm_engine:
        repo_path: /home/aoshen/setup_new_cluster/vllm
        cmd: >-
          /home/aoshen/code/uv_envs/py312/bin/vllm serve {model}
          --port {port}
          -tp 1
          --max-model-len 32768
          --enable-prefix-caching
          --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}},{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":85899345920,"lazy_offload":true}}]}}'
          --disable-uvicorn-access-log
        env:
          CUDA_VISIBLE_DEVICES: "0"
          HF_HOME: /mnt/lustre/hf-models
          CUDA_HOME: /usr/local/cuda-13.1
          LD_LIBRARY_PATH: /usr/local/cuda-13.1/lib64
          VLLM_RPC_BASE_PATH: /tmp/aoshen_rpc
          VLLM_ENGINE_READY_TIMEOUT_S: "1800"
          VLLM_RPC_TIMEOUT: "600000"
        health_check:
          url: http://localhost:{port}/health
          timeout_s: 900
          poll_interval_s: 5

    - role: decode
      count: 1
      nodes_per_instance: 1
      gpus_per_node: 1
      base_port: 8200
      vllm_engine:
        repo_path: /home/aoshen/setup_new_cluster/vllm
        cmd: >-
          /home/aoshen/code/uv_envs/py312/bin/vllm serve {model}
          --port {port}
          -tp 1
          --max-model-len 32768
          --enable-prefix-caching
          --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}}'
          --disable-uvicorn-access-log
        env:
          CUDA_VISIBLE_DEVICES: "0"
          HF_HOME: /mnt/lustre/hf-models
          CUDA_HOME: /usr/local/cuda-13.1
          LD_LIBRARY_PATH: /usr/local/cuda-13.1/lib64
          VLLM_RPC_BASE_PATH: /tmp/aoshen_rpc
          VLLM_ENGINE_READY_TIMEOUT_S: "1800"
          VLLM_RPC_TIMEOUT: "600000"
        health_check:
          url: http://localhost:{port}/health
          timeout_s: 900
          poll_interval_s: 5

  router:
    repo_path: /home/aoshen/code/router
    cmd: >-
      /home/aoshen/code/uv_envs/py312/bin/vllm-router
      --policy round_robin
      --host 0.0.0.0
      --port {router_port}
      --intra-node-data-parallel-size 1
      --vllm-pd-disaggregation
      {prefill_flags}
      {decode_flags}
    health_check:
      url: http://localhost:{router_port}/health
      timeout_s: 120
      poll_interval_s: 5

  remote:
    launcher: slurm
    slurm:
      partition: batch
      time_limit: "02:00:00"
      grace_period_s: 120
      nodelist: "gb200-rack1-[01,03]"

post_serve:
  - cmd: >-
      ./target/release/vllm-bench
      --backend openai-chat
      --base-url http://localhost:{router_port}
      --model {model}
      --dataset-name random
      --random-input-len 8192
      --random-output-len 128
      --num-prompts 4
      --max-concurrency 1
      --result-dir "{log_dir}/smoke"
      --save-result
    repo_path: /home/aoshen/code/vllm-bench
```

### Phase C/D 前的一次性 Mooncake 准备

这部分只需要做一次，建议在进入 `Phase C` 之前完成。

1. 安装 Mooncake wheel

```bash
source /home/aoshen/code/uv_envs/py312/bin/activate
cd /home/aoshen/setup_new_cluster/vllm
/home/aoshen/.local/bin/uv pip install scripts/mooncake/mooncake_transfer_engine-0.3.10.post1-cp312-cp312-manylinux_2_39_aarch64.whl
```

2. 启动 Mooncake master

```bash
cd /home/aoshen/setup_new_cluster/vllm
bash scripts/mooncake/start_mooncake_master.sh --bg
```

3. 准备 CPU-memory-only 的 Mooncake config

```bash
cd /home/aoshen/setup_new_cluster/vllm
source scripts/mooncake/setup_vllm_env.sh --cpu-mem-size 80
cat scripts/mooncake/mooncake_config.json
```

4. 当前阶段不要传 `--disk-size`

- `Phase C/D` 默认只做 CPU memory，不做 disk offload
- 如果 master 已经被别人启动，不要重复起多个实例
- 结束后如需手动清理，可执行：

```bash
cd /home/aoshen/setup_new_cluster/vllm
bash scripts/mooncake/start_mooncake_master.sh --stop
```

### Phase C YAML: `pd_1p1d_nixl_mooncakestore_prefill_01_03.yaml`

`Phase C` 只改 `prefill` 侧；`decode` 仍保持 `NixlConnector`。

```yaml
model: Qwen/Qwen3-8B
mode: remote
precheck: true
collect_env: true
router_port: 30000

serving:
  roles:
    - role: prefill
      count: 1
      nodes_per_instance: 1
      gpus_per_node: 1
      base_port: 8100
      vllm_engine:
        repo_path: /home/aoshen/setup_new_cluster/vllm
        cmd: >-
          /home/aoshen/code/uv_envs/py312/bin/vllm serve {model}
          --port {port}
          -tp 1
          --max-model-len 32768
          --enable-prefix-caching
          --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}]}}'
          --disable-uvicorn-access-log
        env:
          CUDA_VISIBLE_DEVICES: "0"
          HF_HOME: /mnt/lustre/hf-models
          CUDA_HOME: /usr/local/cuda-13.1
          LD_LIBRARY_PATH: /usr/local/cuda-13.1/lib64
          VLLM_RPC_BASE_PATH: /tmp/aoshen_rpc
          VLLM_ENGINE_READY_TIMEOUT_S: "1800"
          VLLM_RPC_TIMEOUT: "600000"
          MOONCAKE_CONFIG_PATH: /home/aoshen/setup_new_cluster/vllm/scripts/mooncake/mooncake_config.json
          MC_TCP_ENABLE_CONNECTION_POOL: "1"
        health_check:
          url: http://localhost:{port}/health
          timeout_s: 900
          poll_interval_s: 5

    - role: decode
      count: 1
      nodes_per_instance: 1
      gpus_per_node: 1
      base_port: 8200
      vllm_engine:
        repo_path: /home/aoshen/setup_new_cluster/vllm
        cmd: >-
          /home/aoshen/code/uv_envs/py312/bin/vllm serve {model}
          --port {port}
          -tp 1
          --max-model-len 32768
          --enable-prefix-caching
          --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}}'
          --disable-uvicorn-access-log
        env:
          CUDA_VISIBLE_DEVICES: "0"
          HF_HOME: /mnt/lustre/hf-models
          CUDA_HOME: /usr/local/cuda-13.1
          LD_LIBRARY_PATH: /usr/local/cuda-13.1/lib64
          VLLM_RPC_BASE_PATH: /tmp/aoshen_rpc
          VLLM_ENGINE_READY_TIMEOUT_S: "1800"
          VLLM_RPC_TIMEOUT: "600000"
        health_check:
          url: http://localhost:{port}/health
          timeout_s: 900
          poll_interval_s: 5

  router:
    repo_path: /home/aoshen/code/router
    cmd: >-
      /home/aoshen/code/uv_envs/py312/bin/vllm-router
      --policy round_robin
      --host 0.0.0.0
      --port {router_port}
      --intra-node-data-parallel-size 1
      --vllm-pd-disaggregation
      {prefill_flags}
      {decode_flags}
    health_check:
      url: http://localhost:{router_port}/health
      timeout_s: 120
      poll_interval_s: 5

  remote:
    launcher: slurm
    slurm:
      partition: batch
      time_limit: "02:00:00"
      grace_period_s: 120
      nodelist: "gb200-rack1-[01,03]"

post_serve:
  - cmd: >-
      ./target/release/vllm-bench
      --backend openai-chat
      --base-url http://localhost:{router_port}
      --model {model}
      --dataset-name random
      --random-input-len 8192
      --random-output-len 128
      --num-prompts 4
      --max-concurrency 1
      --result-dir "{log_dir}/smoke"
      --save-result
    repo_path: /home/aoshen/code/vllm-bench
```

### Phase D YAML: `pd_1p1d_nixl_mooncakestore_both_01_03.yaml`

`Phase D` 的重点是把 `decode` 也切成 `MultiConnector(NIXL, MooncakeStore)`。

```yaml
model: Qwen/Qwen3-8B
mode: remote
precheck: true
collect_env: true
router_port: 30000

serving:
  roles:
    - role: prefill
      count: 1
      nodes_per_instance: 1
      gpus_per_node: 1
      base_port: 8100
      vllm_engine:
        repo_path: /home/aoshen/setup_new_cluster/vllm
        cmd: >-
          /home/aoshen/code/uv_envs/py312/bin/vllm serve {model}
          --port {port}
          -tp 1
          --max-model-len 32768
          --enable-prefix-caching
          --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}]}}'
          --disable-uvicorn-access-log
        env:
          CUDA_VISIBLE_DEVICES: "0"
          HF_HOME: /mnt/lustre/hf-models
          CUDA_HOME: /usr/local/cuda-13.1
          LD_LIBRARY_PATH: /usr/local/cuda-13.1/lib64
          VLLM_RPC_BASE_PATH: /tmp/aoshen_rpc
          VLLM_ENGINE_READY_TIMEOUT_S: "1800"
          VLLM_RPC_TIMEOUT: "600000"
          MOONCAKE_CONFIG_PATH: /home/aoshen/setup_new_cluster/vllm/scripts/mooncake/mooncake_config.json
          MC_TCP_ENABLE_CONNECTION_POOL: "1"
        health_check:
          url: http://localhost:{port}/health
          timeout_s: 900
          poll_interval_s: 5

    - role: decode
      count: 1
      nodes_per_instance: 1
      gpus_per_node: 1
      base_port: 8200
      vllm_engine:
        repo_path: /home/aoshen/setup_new_cluster/vllm
        cmd: >-
          /home/aoshen/code/uv_envs/py312/bin/vllm serve {model}
          --port {port}
          -tp 1
          --max-model-len 32768
          --enable-prefix-caching
          --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}]}}'
          --disable-uvicorn-access-log
        env:
          CUDA_VISIBLE_DEVICES: "0"
          HF_HOME: /mnt/lustre/hf-models
          CUDA_HOME: /usr/local/cuda-13.1
          LD_LIBRARY_PATH: /usr/local/cuda-13.1/lib64
          VLLM_RPC_BASE_PATH: /tmp/aoshen_rpc
          VLLM_ENGINE_READY_TIMEOUT_S: "1800"
          VLLM_RPC_TIMEOUT: "600000"
          MOONCAKE_CONFIG_PATH: /home/aoshen/setup_new_cluster/vllm/scripts/mooncake/mooncake_config.json
          MC_TCP_ENABLE_CONNECTION_POOL: "1"
        health_check:
          url: http://localhost:{port}/health
          timeout_s: 900
          poll_interval_s: 5

  router:
    repo_path: /home/aoshen/code/router
    cmd: >-
      /home/aoshen/code/uv_envs/py312/bin/vllm-router
      --policy round_robin
      --host 0.0.0.0
      --port {router_port}
      --intra-node-data-parallel-size 1
      --vllm-pd-disaggregation
      {prefill_flags}
      {decode_flags}
    health_check:
      url: http://localhost:{router_port}/health
      timeout_s: 120
      poll_interval_s: 5

  remote:
    launcher: slurm
    slurm:
      partition: batch
      time_limit: "02:00:00"
      grace_period_s: 120
      nodelist: "gb200-rack1-[01,03]"

post_serve:
  - cmd: >-
      ./target/release/vllm-bench
      --backend openai-chat
      --base-url http://localhost:{router_port}
      --model {model}
      --dataset-name random
      --random-input-len 8192
      --random-output-len 128
      --num-prompts 4
      --max-concurrency 1
      --result-dir "{log_dir}/smoke"
      --save-result
    repo_path: /home/aoshen/code/vllm-bench
```

## 验收标准

阶段通用标准：

- `dry-run` 成功
- 服务可以完整起停
- smoke benchmark 成功率正常
- 没有 connector 初始化失败
- 没有明显的 handshake / load / save 错误

阶段特有标准：

- `Phase C`
  - 日志里要能看到 `MooncakeStoreConnector` 确实参与了 store / load 路径
- `Phase D`
  - 要能确认 decode 侧具备从 `MooncakeStore` 读取的能力

## 当前明确排除项

- 不做 `multi-GPU node + disk offloading`
- 不在第一轮引入 `MooncakeConnector` 主传输链路
- 不把 speculative decoding / MTP 和当前 PD bring-up 混在一起
- 不在 `Phase A / Phase B` 就做 Crusoe full workload

## 下一位接手的人第一步

1. 先激活唯一推荐环境：
   - `source /home/aoshen/code/uv_envs/py312/bin/activate`
2. 先读最新单机失败日志：
   - `/home/aoshen/setup_new_cluster/vigil/logs/pd_1p1d_nixl_only_local_01_20260408T065431Z/prefill_0.log`
3. 以本地单机 spec 作为当前主线：
   - `/home/aoshen/setup_new_cluster/vigil/examples/pd_1p1d_nixl_only_local_01.yaml`
4. 优先排查并解决：
   - `nixl_cu12._bindings.nixlBackendError: NIXL_ERR_BACKEND`
5. 只有在本地单机 `NIXL only` 稳定后，才回到远端 `01 + 03` 的 `Phase A`
6. 不允许跳过 `Phase B` 直接上 `MooncakeStore`

## 关键路径速查

- 当前工作区：`/home/aoshen/setup_new_cluster`
- 内部文档：`/home/aoshen/setup_new_cluster/[INTERNAL] Crusoe Inference.docx`
- vLLM：`/home/aoshen/setup_new_cluster/vllm`
- vigil：`/home/aoshen/setup_new_cluster/vigil`
- router：`/home/aoshen/code/router`
- vllm-bench：`/home/aoshen/code/vllm-bench`
- HF cache：`/mnt/lustre/hf-models`
- Mooncake offload path：`/mnt/data/aoshen/mooncake_offload`
