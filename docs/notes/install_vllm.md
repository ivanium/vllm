vllm installation needs some customizations, so we need to write a separate installation script. usually i prefer to manually install vllm.


go to the vllm directory and install vllm, and save the installation log to a separate file called `vllm_installation.log`, with both stdout and stderr redirected to the file. this can take a while (like 10~20 minutes), run it in the background and check every 5 minutes to see if it's done.
```bash
uv pip install -vvv -e . --reinstall-package vllm --torch-backend=cu130 > vllm_installation.log 2>&1
```

be careful, the default pytorch release does not support aarch64 + GPU. in such case, you need to install pytorch with a custom uv flag `--torch-backend=cu130` or `--torch-backend=cu131` depending on the cuda version in the system.

A sanity check is to check if the pytorch version is installed correctly by running `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`. it should print the pytorch version and True.

If you want to use pre-compiled wheels:

```bash
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 uv pip install -vvv -e . --reinstall-package vllm --torch-backend=cu130
```

If somehow you still end up with a CPU version torch:

```bash
uv pip install torch==2.10.0 --torch-backend=cu130 --reinstall-package torch
```

