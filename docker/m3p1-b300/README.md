# MiniMax-M3.1-Flash on B300

Build this branch for one B300 node with eight GPUs. The image uses the amd64
variant of the same CUDA 13.5 nightly as `Dockerfile.m3p1-gb300`, with the same
vLLM and DeepGEMM/MSA source manifests. It does not copy ARM binaries.

## Build

From a clean checkout:

```bash
docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.m3p1-b300 \
  --build-arg BRANCH_COMMIT="$(git rev-parse HEAD)" \
  --build-arg BUILD_JOBS=16 \
  -t "vllm-m3p1-b300:$(git rev-parse --short=12 HEAD)" --load .
```

The build needs network access and enough disk for the CUDA nightly and two
connector builds. It does not require a GPU. Source checksums, native imports,
CUDA major versions, EFA provider availability and library paths are checked
inside the build. These checks do not validate GPU execution or EFA transfers.

| Component | Pin / behavior |
| --- | --- |
| Base | `cu135-nightly-153dc046462c4724a9a4eb393cadd60169ed943c`, amd64 digest `sha256:f2036e7a2b69217d9fbf048caf4b994fbd996d7821939938e33a4f4a2ce009e3` |
| vLLM / kernels | `docker/m3p1-gb300/vllm-files.sha256` and `kernel-sources.sha256` |
| EFA | Base image's 1.51.0 userspace; NIXL must resolve `/opt/amazon/efa/lib/libfabric.so.1` |
| Mooncake | `ed1b346b0cf3f6b04a1aa33c877b6e0f212a39a4`, CUDA 13 wheel built with EFA and the classic store |
| NIXL | v1.5.2 (`e72c52c78478d2e1ca9e9c57b872114e113ad767`) plus the reconnect backport; CUDA 13, LIBFABRIC backend only |

The connector wheels are rebuilt for the base image's Python ABI. Mooncake's
EFA transport must use `dlopen`, not a direct libfabric dependency. The ARM-only
Mooncake HBM payload is omitted: this configuration does not enable `hbm_pool`.
Do not enable HBM-pool offload without a compatible Mooncake build.

`nixl-1.5.2-reconnect.patch` backports
[ai-dynamo/nixl#2438](https://github.com/ai-dynamo/nixl/pull/2438)
(commit `94266532814b823b543fa341c8dffcb486e59b41`). It retains the reply queue,
CM thread teardown and upstream regression test. The backport preserves 1.5.2's
four-argument control-message API, assigned-index wire format and CUDA ABI.
The patch fixes missing replies during connection re-establishment; it does
not establish that first-connection failures are fixed. The optional vLLM
`background_receiver` is not enabled.

## TP8 serving configuration

Use `tp8.yaml` with `vllm serve --config` and pass `tp8.env` as the container's
`--env-file`. Supply the model path separately. The image keeps the vLLM server
entrypoint. For example, after granting GPU access and mounting the model:

```bash
docker run --rm --gpus all --ipc=host \
  --env-file docker/m3p1-b300/tp8.env \
  -v "$PWD/docker/m3p1-b300/tp8.yaml:/config/tp8.yaml:ro" \
  -v "$MODEL_PATH:/model:ro" -p 8000:8000 \
  "$IMAGE" /model --config /config/tp8.yaml
```

Use benchmark concurrency **4**; `max-num-seqs: 8` is the scheduler limit.
The specialized path requires `decode_backend: latency_specialized`, all four
TP8 environment switches, FULL decode graphs and NVLink multicast on all ranks.
Use one node, eight visible GPUs and CPU affinity that covers all eight GPUs.
This is an aggregate TP-only configuration, not a DP/EP disaggregated preset.
Dynamo parser options belong to the sidecar, not the vLLM engine arguments.

Before reporting performance, confirm the specialized kernels execute and run
accuracy checks with real speculative acceptance. Compare B300 TP8 with the
specialized path off/on under the same workload. Validate NIXL initial connect,
reconnect and disaggregated evaluation separately. Image construction alone
does not establish any of those results.
