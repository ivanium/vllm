# Mooncake 本地安装 + 可观测性验证（给其它机器的 agent）

本文档给 **需要在新机器上装一套带 B/C/D/quorum 可观测性补丁的 Mooncake，并跑 smoke test** 的 agent 使用。目标是 20-30 分钟内拿到一个可验证通过的 master 实例。

对应分支：`aoshen524/Mooncake:feat/observability-logs`（PR #1 to `ivanium/Mooncake:yifan/dev`）。
涉及的补丁：
- Method B — `transfer_metadata.cpp` 打 segment descriptor 404 时加 `metadata_key='<key>'`
- Method C — `http_metadata_server.cpp` 每次 GET/PUT/DELETE 打一行 access log（`metadata GET MISS / PUT NEW / PUT UPDATE / DELETE OK / DELETE MISS`）
- Method D — `transfer_task.cpp` + `client_service.cpp` 在 openSegment 失败 / replica 提交失败时打 `endpoint='...' replica_idx=...`
- Quorum 修复 — `client_service.cpp` `WaitForTransfers` 从"任一 replica 失败 = 整 op 失败"改为"至少一个 replica 成功 = op 成功"

## 0. 前置条件

- Linux (aarch64 或 x86_64)，GPU 机器
- CUDA toolkit 可用（验证：`nvcc --version`）
- `cmake` ≥ 3.20，`g++` ≥ 11
- `uv` 已安装；Python 3.12 venv 已激活
- 足够磁盘空间（build 目录 + wheel ≈ 2 GB）

如果系统上已经有 `mooncake_master` 在跑，先停：

```bash
pkill -f mooncake_master || true   # 确保停干净（Python wrapper 和 C++ 子进程都要杀）
```

## 1. Clone + 切到正确分支

```bash
# 选一个工作目录，例如 ~/src
cd ~/src
git clone git@github.com:aoshen524/Mooncake.git
cd Mooncake
git checkout feat/observability-logs
```

确认 HEAD 对：

```bash
git log --oneline -3
# 应该看到（哈希可能已变）：
#   3795eac client_service: quorum write on WaitForTransfers + richer failure logging
#   f21e66d observability: enrich put-path error logs and add HTTP metadata access log
#   71b6212 fix: correct segment name                       <- 这是 ivanium/Mooncake yifan/dev 的 base
```

如果 base 不是 `fix: correct segment name`，说明上游又 rebase 过。按 §8 流程 rebase 一次。

## 2. 编译

```bash
rm -rf build
bash scripts/dev_compile.sh
```

- 耗时约 4-8 分钟（取决于机器）
- 成功标志：最后几行 `[100%] Built target task_integration_test`，无 error
- 产物：
  - `build/mooncake-integration/engine.cpython-312-*.so`
  - `build/mooncake-integration/store.cpython-312-*.so`
  - `build/mooncake-store/src/mooncake_master`
  - `build/mooncake-common/libasio.so`

如果报 `-DWITH_NVIDIA_PEERMEM=OFF` 但 build 失败，看 `dev_compile.sh` 最前面的 cmake 参数是否和本机环境匹配。

## 3. 打 wheel

```bash
bash scripts/build_wheel.sh 3.12 dist
```

- 耗时约 1-2 分钟
- 成功标志：最后 `Wheel package built and repaired successfully!`
- 产物：`mooncake-wheel/dist/mooncake_transfer_engine-0.3.10.post2-cp312-cp312-manylinux_2_39_<arch>.whl`
- 这个 wheel 是 **auditwheel-repaired** 的，已经 bundle 了 libjsoncpp 等 runtime 依赖，**不要**再手动 patchelf

## 4. 安装 wheel

```bash
uv pip install --force-reinstall \
    mooncake-wheel/dist/mooncake_transfer_engine-0.3.10.post2-cp312-cp312-manylinux_2_39_*.whl
```

注意 `--force-reinstall`：旧版 Mooncake 的 `.so` 必须被替换掉。

快速验证 Python import + master binary 可用：

```bash
python -c 'from mooncake.engine import TransferEngine; from mooncake.store import MooncakeDistributedStore; print("import OK")'
which mooncake_master
# 应输出类似 /home/<user>/code/uv_envs/py312/bin/mooncake_master
```

## 5. 启动 master

仓库里自带了启动脚本（在 vLLM repo 的 `scripts/mooncake/` 里，而不是 Mooncake repo 里）：

```bash
# 假设 vllm 仓库 checkout 在 $VLLM_ROOT
export VLLM_ROOT=/path/to/vllm_repo   # 改成你的路径
cd $VLLM_ROOT
MC_METRICS_PORT=9103 bash scripts/mooncake/start_mooncake_master.sh --bg
```

启动成功的标志（脚本输出里看）：

```
PID: <xxxxxx> (written to .../mooncake_master.pid)
Log: .../mooncake_master.log
```

进程验证：

```bash
pgrep -a mooncake_master
# 应看到两个进程：Python wrapper + C++ master
# Python wrapper 在 bin/mooncake_master，C++ 在 site-packages/mooncake/mooncake_master
```

端口验证：
- RPC：50051
- HTTP metadata：8080
- Metrics：9103

```bash
curl -s http://localhost:8080/health   # 应返回 "OK"
```

## 6. Smoke test：验证 B/C 确实 fire

用仓库自带的 probe 脚本（在 vLLM repo 里）：

```bash
python $VLLM_ROOT/scripts/mooncake/probe_observability_logs.py --skip-d
```

输出应包含：

```
[C] GET MISS on a brand-new key (expect HTTP 404 + log 'metadata GET MISS')
    http=404
[C] PUT NEW (expect HTTP 200 + log 'metadata PUT NEW')
    http=200
[C] PUT UPDATE same key (expect HTTP 200 + log 'metadata PUT UPDATE')
    http=200
[C] DELETE OK (expect HTTP 200 + log 'metadata DELETE OK')
    http=200
[C] DELETE MISS on the now-deleted key (expect HTTP 404 + log 'metadata DELETE MISS')
    http=404

[B] transfer_sync_write to a ghost hostname ...
    transfer returned status=-1 (log should still be emitted)
```

然后查 master log（注意是 vLLM repo 下的路径）：

```bash
grep -E 'metadata (GET MISS|PUT NEW|PUT UPDATE|DELETE OK|DELETE MISS)' \
    $VLLM_ROOT/scripts/mooncake/mooncake_master.log | tail -20
```

应该看到类似：

```
W ... http_metadata_server.cpp:38] metadata GET MISS   key='probe-C-<ts>' store_size=0
I ... http_metadata_server.cpp:78] metadata PUT NEW    key='probe-C-<ts>' body_bytes=15 store_size=1
I ... http_metadata_server.cpp:78] metadata PUT UPDATE key='probe-C-<ts>' body_bytes=26 store_size=1
I ... http_metadata_server.cpp:111] metadata DELETE OK key='probe-C-<ts>' store_size=0
W ... http_metadata_server.cpp:101] metadata DELETE MISS key='probe-C-<ts>' store_size=0
W ... http_metadata_server.cpp:38] metadata GET MISS   key='mooncake%2Fram%2Fghost-probe-<ts>%3A65500'
```

最后一行是 Method B 间接触发：probe 对 ghost hostname 发 transfer_sync_write，client 端去 master 查 segment → master 返 404（C 的 `GET MISS` 打出来）→ client 端 `transfer_metadata.cpp` 打出 B 的 `Failed to retrieve segment descriptor ... metadata_key='mooncake/ram/ghost-probe-...:65500'`。

## 7. 可选：验证 Method D

Method D（`Failed to open segment endpoint='...' replica_idx=...`）需要 **race-delete** 或者 **多 client 环境**才能触发，单 client 本地 probe 触发不了（master 的本地 segment cache 会挡）。建议：

- 要在真实 benchmark / CI 流里看 D，跑一次 vigil PD-disagg run 就会在 worker 端触发（只要有 openSegment 失败）
- 或者跑 Mooncake 自带的 `ctest -R batch_remove_test` / `storage_backend_e2e_test`（在 build/ 下），它们会 inject ghost segment

probe 脚本里有 `probe_method_d` 函数保留了 race-delete 的 recipe 注释，可以参考。

## 8. 如果上游 `yifan/dev` 又 force-push 过

表现：你的分支 base 不再是 `71b6212`，`git pull` 或 `git rebase origin/yifan/dev` 报 conflict，而且 conflict 在 yifan **自己的** 老提交上（像 `feat: fix & support disk offloading` 这种）。

正确处理：**只 replay 你自己的 2 个提交**，不要让 git 尝试 replay yifan 被改写的老提交：

```bash
cd ~/src/Mooncake
git fetch origin yifan/dev

# 找到你自己 2 个提交之前的 base（在 `git log feat/observability-logs` 里看，应该是最后一个非你的提交）
OLD_BASE=$(git log feat/observability-logs --format=%H --not --author=aoshen524 -1)

# 用 --onto 精准 rebase
git rebase --onto origin/yifan/dev $OLD_BASE feat/observability-logs
```

然后重新跑 §2-§6 即可。

## 9. Clean-up

停 master：

```bash
bash $VLLM_ROOT/scripts/mooncake/start_mooncake_master.sh --stop
# 如果 .pid 文件丢了或 stop 脚本没杀干净：
pkill -f mooncake_master
```

卸载 wheel：

```bash
uv pip uninstall mooncake-transfer-engine
```

## 参考

- PR：https://github.com/ivanium/Mooncake/pull/1
- Probe 脚本源码：`$VLLM_ROOT/scripts/mooncake/probe_observability_logs.py`
- Master 启动脚本源码：`$VLLM_ROOT/scripts/mooncake/start_mooncake_master.sh`
- 原始 bug 报告 / observability plan：`$VLLM_ROOT/sprint_dist_kv_doc/bug_analysis/`（如果有访问权限）
