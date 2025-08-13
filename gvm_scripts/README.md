## Usage

Please clone `git@github.com:ivanium/kvcached-public-fork` to `kvcached/` directory and check out to the `uvm-backend` branch.

To install kvcached:

```shell
pip install kvcached --no-build-isolation -e .
```

After the first time installation, later we can re-compile c++ code without re-installation:

```shell
python setup.py build_ext --inplace
```

To add signal handler support, please refer to `csrc/inc/ftensor.hpp:25-29`:

```c++
  // [GVM] swap out `size` bytes from the tensor.
  bool reclaim_handler(size_t size);

  // [GVM] call UVM prefetch to host or internal swap interface.
  bool swapout(void *addr, size_t size);
```

Install vLLM (if not installed):

```shell
# Clone this repo and check out this branch
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

To run vLLM with kvcached:

```shell
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export ENABLE_KVCACHED=true
export KVCACHED_IPC_NAME=VLLM

# Original vLLM command. For example,
vllm serve meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --no-enable-prefix-caching --enforce-eager --port=12345
```

Note that `--no-enable-prefix-caching` is necessary, as kvcached must run without prefix caching enabled. `--enforce-eager` is optional. It disables CUDA graphs, which will touch all KV cache pool space during capturing.

I also provided a `start_server.sh` and a `start_client.sh` for reference. One can directly run `start_server.sh` and then `start_client.sh` in another terminal to benchmark with ShareGPT dataset.