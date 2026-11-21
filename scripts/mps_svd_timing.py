"""Model-free: how long does torch.svd_lowrank take on a device, at the shapes Heretic calls it with?

    python scripts/mps_svd_timing.py <rows> <cols> [cpu|mps] [attempts]

Heretic's directional ablation with row_normalization "full" calls torch.svd_lowrank(D, q = 2 r + 4, niter = 6) with
r = 3 once per module per trial, on the device the model is on. Under torch 2.14.1 on mps one call took 12.5 s on
an [896, 896] matrix (0.002 s on the CPU, 0.01 s under torch 2.15.0) and did not return within 170 s on
[2560, 2560] and [2560, 10240]: the SVD of the small projected matrix falls back to the CPU there, slowly. A call
that has not returned after STALL_AFTER seconds (default 170) prints where it is and ends the process. Run it
under each interpreter of interest, under a memory cap.
"""
import faulthandler
import os
import sys
import time

import torch

faulthandler.dump_traceback_later(int(os.environ.get("STALL_AFTER", "170")), exit=True)
rows, cols = int(sys.argv[1]), int(sys.argv[2])
device = sys.argv[3] if len(sys.argv) > 3 else "mps"
print("torch", torch.__version__, "device", device, "shape", (rows, cols), flush=True)
torch.manual_seed(12345)
D = torch.randn(rows, cols, dtype=torch.float32).to(device)
for attempt in range(int(sys.argv[4]) if len(sys.argv) > 4 else 3):
    t = time.perf_counter()
    torch.manual_seed(12345)
    U, S, V = torch.svd_lowrank(D, q=10, niter=6)
    if device == "mps":
        torch.mps.synchronize()
    print(f"attempt {attempt}: {time.perf_counter() - t:.3f} s, S[:3] = {[round(float(x), 4) for x in S[:3].cpu()]}", flush=True)
print("DONE", flush=True)
