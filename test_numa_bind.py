#!/usr/bin/env python3
"""Test _try_bind_numa under various CUDA_VISIBLE_DEVICES configs."""

import json, os, subprocess, sys

NUMA_CORES = {0: list(range(0, 70)), 1: list(range(70, 140))}
SCENARIOS = [
    ("all_gpus", None, [(0, 0, 0), (1, 1, 0), (2, 2, 1), (3, 3, 1)]),
    ("gpu_0_1", "0,1", [(0, 0, 0), (1, 1, 0)]),
    ("gpu_2_3", "2,3", [(0, 2, 1), (1, 3, 1)]),
    ("reversed", "3,2,1,0", [(0, 3, 1), (1, 2, 1), (2, 1, 0), (3, 0, 0)]),
    ("single_n1", "2", [(0, 2, 1)]),
    ("single_n0", "0", [(0, 0, 0)]),
    ("cross", "0,3", [(0, 0, 0), (1, 3, 1)]),
]

def parse_cpulist(s):
    cores = []
    for p in s.split(","):
        p = p.strip()
        if "-" in p:
            lo, hi = p.split("-", 1); cores.extend(range(int(lo), int(hi)+1))
        elif p:
            cores.append(int(p))
    return sorted(cores)

def get_gpu_numa_node(physical_id):
    """Same logic as _get_gpu_numa_node in mooncake_store_worker."""
    from vllm.third_party import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_id)
    try:
        n = pynvml.nvmlDeviceGetNumaNodeId(handle)
        if n >= 0: return n
    except Exception: pass
    pci = pynvml.nvmlDeviceGetPciInfo(handle)
    bus_id = pci.busId.decode("utf-8") if isinstance(pci.busId, bytes) else pci.busId
    bus_id = bus_id.strip().lower()
    parts = bus_id.split(":")
    if len(parts) >= 2 and len(parts[0]) > 4:
        parts[0] = parts[0][-4:]; bus_id = ":".join(parts)
    p = f"/sys/bus/pci/devices/{bus_id}/numa_node"
    if os.path.exists(p):
        with open(p) as f:
            v = int(f.read().strip())
        if v >= 0: return v
    return None

def run_single(cuda_idx):
    import torch
    from vllm.platforms import current_platform
    torch.cuda.set_device(cuda_idx)
    phys = current_platform.device_id_to_physical_device_id(cuda_idx)
    numa = get_gpu_numa_node(phys)
    with open(f"/sys/devices/system/node/node{numa}/cpulist") as f:
        cores = parse_cpulist(f.read().strip())
    reserved = cores[-2:] if len(cores) > 2 else cores
    os.sched_setaffinity(0, reserved)
    actual = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, list(range(os.cpu_count() or 1)))
    return {"cuda_idx": cuda_idx, "physical_id": phys, "numa_node": numa,
            "reserved_cores": reserved, "actual_affinity": actual,
            "bind_ok": actual == sorted(reserved)}

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        import torch
        r = []
        for i in range(torch.cuda.device_count()):
            os.sched_setaffinity(0, list(range(os.cpu_count() or 1)))
            r.append(run_single(i))
        print(json.dumps(r)); return

    ok_all = True
    for name, cvd, expected in SCENARIOS:
        print(f"\n{'='*60}\nScenario: {name}  (CVD={cvd or '<unset>'})")
        env = os.environ.copy()
        if cvd is not None: env["CUDA_VISIBLE_DEVICES"] = cvd
        elif "CUDA_VISIBLE_DEVICES" in env: del env["CUDA_VISIBLE_DEVICES"]
        r = subprocess.run([sys.executable, __file__, "--worker"],
                           capture_output=True, text=True, timeout=30, env=env)
        if r.returncode != 0:
            print(f"  FAIL: {r.stderr[:300]}"); ok_all = False; continue
        for res, (_, ep, en) in zip(json.loads(r.stdout), expected):
            ok = res["physical_id"]==ep and res["numa_node"]==en and res["bind_ok"] and res["reserved_cores"]==NUMA_CORES[en][-2:]
            print(f"  CUDA {res['cuda_idx']} → phys={res['physical_id']} NUMA={res['numa_node']} cores={res['reserved_cores']} [{'PASS' if ok else 'FAIL'}]")
            if not ok: ok_all = False
    print(f"\n{'='*60}\nOverall: {'ALL PASS' if ok_all else 'SOME FAILED'}")
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
