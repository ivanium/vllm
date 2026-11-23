# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata as metadata
import platform
import subprocess

import torch
from mooncake import engine
from mooncake.store import MooncakeDistributedStore
from nixl import _api

assert platform.machine() == "x86_64"
assert torch.version.cuda is not None and torch.version.cuda.startswith("13.")
assert metadata.version("nixl-cu13") == "1.5.2+reconnect"
assert metadata.version("mooncake-transfer-engine-cuda13") == "0.3.14+efa.ed1b346b"
assert engine.SUPPORT_EFA
assert MooncakeDistributedStore is not None
assert _api.nixl_agent is not None

packages = {
    dist.metadata["Name"].lower().replace("_", "-") for dist in metadata.distributions()
}
assert "nixl-cu12" not in packages
assert "mooncake-transfer-engine" not in packages
assert "nvidia-cutlass-dsl-libs-cu12" not in packages

mooncake = metadata.distribution("mooncake-transfer-engine-cuda13")
for file in mooncake.files or ():
    if ".so" in file.name:
        dynamic = subprocess.check_output(
            ["readelf", "-d", str(mooncake.locate_file(file))], text=True
        )
        assert "Shared library: [libfabric.so" not in dynamic, file

for name in ("vllm", "torch", "flashinfer-python", "nixl-cu13", mooncake.name):
    print(f"{name}=={metadata.version(name)}")
