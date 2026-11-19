# MiniMax-M3.1-Flash on GB300: `docker/Dockerfile.m3p1-gb300`

This builds an image of this branch for GB300 (linux/arm64) without compiling anything. It starts from the vLLM
CUDA 13.5 nightly this branch is based on and copies in four things: this branch's changed `vllm/` files, the DeepGEMM
and MSA kernel sources the branch needs, and Mooncake `0.3.13.post1+hbm`. Every file it adds is checked against a
sha256 list in this directory.

## Inputs

| Part | Source | Checked by |
| --- | --- | --- |
| Base | `vllm/vllm-openai:cu135-nightly-153dc046462c4724a9a4eb393cadd60169ed943c`, pinned to its linux/arm64 image `sha256:18dfff30…` (index `sha256:712ef89a…`): vLLM 153dc046, CUDA 13.5, torch `2.16.0.dev20261009+cu134`, FlashInfer 0.7.4 | the digest |
| vLLM | the 77 files under `vllm/` that this branch changes from 153dc046 (50 modified, 27 added), copied from the build context over the installed package; the stale bytecode of the modified files is removed | `vllm-files.sha256` |
| JIT kernel sources | 5 MegaMoE headers at vllm-project/DeepGEMM `196f338` (this branch's pin) and 22 `fmha_sm100` files at the head of vllm-project/MSA#49 (`56bf386f`), fetched with `ADD --checksum` | each `ADD`, then `kernel-sources.sha256` |
| Mooncake | `mooncake-transfer-engine-cuda13` `0.3.13.post1+hbm`, whose store binding has `mount_external_segment` (used by the Mooncake store connector's `hbm_pool`). It comes from the last layer (`sha256:65bd369b…`, 13 MB) of the public image `vllm/vllm-openai:cu134-arm64-1107-mdev` (`sha256:1b4a9b46…`), which the published InferenceX GB300 recipe for this model runs. It replaces the base's PyPI Mooncake and adds the five libraries it links. Its source is not public. | the layer digest, then `mooncake-hbm.sha256` |

## Build

From the root of a clean checkout of this branch:

```bash
git status --porcelain -- vllm docker   # must print nothing
docker buildx build --platform linux/arm64 -f docker/Dockerfile.m3p1-gb300 \
    --build-arg BRANCH_COMMIT=$(git rev-parse HEAD) \
    -t vllm-m3p1-gb300:$(git rev-parse --short=10 HEAD) --load .
```

Requirements:

- Docker with BuildKit, building for linux/arm64. On an amd64 host the `RUN` steps need QEMU; they only copy and hash
  files.
- Network access to Docker Hub and raw.githubusercontent.com.
- About 45 GB of free disk: the base is 12.8 GB compressed and about 28 GB unpacked.

After the base pull, the steps take a few minutes.

The build checks each of these, and stops at the first that fails:

- The layer, and every downloaded file, matches its pinned sha256.
- The checkout's `vllm/` files match `vllm-files.sha256`, so a dirty checkout or another commit stops the build.
- Every copied file matches its list.
- nvcc is release 13.5.
- `_store` has `mount_external_segment`.
- There is exactly one Mooncake distribution, and it is `0.3.13.post1+hbm`.

The last step prints the package versions.

## Check the image

```bash
IMG=vllm-m3p1-gb300:$(git rev-parse --short=10 HEAD)
docker run --rm --entrypoint sh -v "$PWD/docker/m3p1-gb300:/m:ro" "$IMG" -c \
    'cd /usr/local/lib/python3.12/dist-packages && sha256sum -c --quiet /m/vllm-files.sha256 /m/kernel-sources.sha256 /m/mooncake-hbm.sha256 && echo "155 files OK"'
docker run --rm --entrypoint python3 "$IMG" -I -B -c \
    'import importlib.metadata as m; [print(n, m.version(n)) for n in ("vllm", "torch", "flashinfer-python", "mooncake-transfer-engine-cuda13")]'
docker inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMG"
```

Expected versions:

- vllm `0.34.0rc1.dev398+g153dc0464.d20261113.cu134`. This string, the `VLLM_*` environment and the `ai.vllm.*` labels
  stay the base nightly's. The labels `org.opencontainers.image.revision` and `org.opencontainers.image.version` carry
  this branch's commit.
- torch `2.16.0.dev20261009+cu134`.
- flashinfer-python `0.7.4`.
- mooncake-transfer-engine-cuda13 `0.3.13.post1+hbm`.
- nvcc `V13.5.37`.

To run the image under Slurm with enroot or pyxis, convert it with `enroot import dockerd://<image>` on the Docker
host, or push it to a registry and import it from there.

## When the branch changes

- **A commit changes `vllm/`.** From a clean checkout, regenerate the list:

  ```bash
  git diff --no-renames --diff-filter=d --name-only 153dc046462c4724a9a4eb393cadd60169ed943c HEAD -- vllm/ \
      | LC_ALL=C sort | xargs sha256sum > docker/m3p1-gb300/vllm-files.sha256
  ```

  If the branch deletes or renames a file under `vllm/`, step 3 also needs to remove the old file. No commit does
  that today.
- **The DeepGEMM or MSA pin moves.** Update `DEEPGEMM_COMMIT` or `MSA_COMMIT`, the `ADD` lines and
  `kernel-sources.sha256` together.
- **The pinned images age out.** The nightly's tag is deleted as newer nightlies arrive, because vLLM keeps the latest
  14 per variant. Both digests still pull, but Docker Hub does not promise to keep them. To keep this build
  reproducible, mirror the base and the 1107-mdev image, or just its layer.
