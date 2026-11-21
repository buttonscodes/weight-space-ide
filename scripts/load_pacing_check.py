"""Model-free check of scripts/load_pacing: both paths onto mps are paced, values are unchanged, other copies are not.

    PYTHONPATH=scripts/load_pacing LOAD_GIB_PER_S=1.5 python scripts/load_pacing_check.py <scratch directory>

The scratch directory receives one safetensors file of random numbers (1 GiB). Run it under a memory cap, with and
without LOAD_GIB_PER_S, under each interpreter that will load a model with the pacing.
"""
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

state = getattr(torch, "_load_pacing", None)
print("torch", torch.__version__, "pacing installed:", state is not None, "LOAD_GIB_PER_S =", os.environ.get("LOAD_GIB_PER_S"))
torch.manual_seed(0)
chunks = [torch.randn(64 * 2 ** 20, dtype=torch.float32).to(torch.bfloat16) for _ in range(8)]      # 8 x 128 MiB in bfloat16
total = sum(c.numel() * c.element_size() for c in chunks)

# path 1: a tensor on the CPU moved with Tensor.to
t = time.perf_counter()
moved = [c.to("mps") for c in chunks]
torch.mps.synchronize()
dt = time.perf_counter() - t
print(f"path 1, Tensor.to: {total / 2 ** 30:.2f} GiB to mps in {dt:.2f} s, {total / 2 ** 30 / dt:.2f} GiB/s; values equal:",
      all(torch.equal(m.cpu(), c) for m, c in zip(moved, chunks)))
t = time.perf_counter()
again = [m.to("mps") for m in moved] + [c.to("cpu") for c in chunks] + [c.to(torch.float32) for c in chunks[:2]]
print(f"copies that are not onto mps from elsewhere: {time.perf_counter() - t:.2f} s")
del moved, again

# path 2: the way transformers loads with device_map "mps": safetensors' pread backend on mps, then _materialize_copy
path = os.path.join(sys.argv[1], "random.safetensors")
if not os.path.exists(path):
    save_file({f"t{i:02d}": c for i, c in enumerate(chunks)}, path)
import transformers.core_model_loading as loading                                              # noqa: E402

t = time.perf_counter()
with safe_open(path, framework="pt", device="mps", backend="pread") as f:
    read = [loading._materialize_copy(f.get_slice(k), "mps", torch.bfloat16) for k in sorted(f.keys())]
torch.mps.synchronize()
dt = time.perf_counter() - t
print(f"path 2, pread onto mps: {total / 2 ** 30:.2f} GiB in {dt:.2f} s, {total / 2 ** 30 / dt:.2f} GiB/s; on",
      {str(r.device) for r in read}, "values equal:", all(torch.equal(r.cpu(), c) for r, c in zip(read, chunks)))
if state is not None:
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in state.items() if k != "free_at"}, "of", total, "each")
