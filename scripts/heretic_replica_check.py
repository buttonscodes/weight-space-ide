"""Does Heretic's randomized rank-r approximation explain the IDE's distance from Heretic's change for
row_normalization "full"? No model is loaded: only tensors read from files.

    python scripts/heretic_replica_check.py <checkpoint dir> <out.json> full=<heretic run dir> [adapter=<dir>]
        [device=<cpu|mps>]

Run it from the IDE's `weight-space-ide` directory, with either tool's interpreter (it needs torch, safetensors
and numpy only; no runner, no workspace), so the same arithmetic can be repeated under each tool's torch version.
The run directory holds heretic_dump.py's dump (dump/dump.json: Heretic's means, the exported trial's parameters,
the seed, the module names, the library versions) and Heretic's adapter export (`adapter=<dir>`, else the first of
<run>/adapter, <run>/<save_directory>, <run>/model with adapter_config.json; refused unless r == lora_alpha). The
checkpoint is read one tensor at a time (one model.safetensors or the shards of its index); its config.json maps
the dump's module names to checkpoint tensors through wsp.runtime_layouts (a Qwen3.5 checkpoint's
model.language_model.* names and linear_attn.out_proj, or the plain Qwen2 names).

For every module the exported trial changed (its layer weight not 0), from Heretic's means and the trial's
parameters, with the product's direction and layer-weight functions (wsp.directions.table, direction_at,
layer_weight):
  - D, the full change before truncation, from the IDE's closed form (formats_shared.full_change); its exact
    rank-r truncation is checked against the IDE's delta, which wsp.directions.change computes here on the tensor
    (premise "B-S4 replica input is the IDE's change", relative 1e-6, always asserted);
  - R, Heretic's three lines (torch.manual_seed(seed); torch.svd_lowrank(D, q = 2r + 4, niter = 6); B = U_r
    sqrt(S_r), A = sqrt(S_r) V_r^T) on D in float32 on `device` (below), with the run's seed and the adapter's r;
  - replica_vs_heretic |R - B A| / |B A| against Heretic's stored product, replica_vs_ide |R - delta_IDE| /
    |delta_IDE|, ide_vs_heretic (what B-S4 "change from Heretic's means" asserts), tail sigma_{r+1} / sigma_r
    and sigma_r / sigma_1 of D's exact singular values (how hard D is for a randomized method), and, reported,
    the same replica from D formed by Heretic's own float32 lines.

Bound (the stage "B-S4 replica of Heretic's randomized approximation", heretic_compare.BOUNDS by the run's dtype):
for a bfloat16 run replica_vs_heretic at most 1e-5, asserted only when this interpreter's torch equals the run's
(dump.json "versions"["torch"], compared without a "+..." local suffix; formats_shared.torch_matches_run), since
the random stream and LAPACK may differ between torch versions, AND the replica ran on the device Heretic's run
was made on (dump.json "device", compared by type: "mps:0" is mps, a dump without one is cpu); with either
different it is reported with the reason in words (formats_shared.replica_applies). Basis:
the same algorithm, seed and float32 input up to the 1e-6 agreement of the unnormalised change, so the replica
should reproduce Heretic's stored product to rounding; if it does, the whole of ide_vs_heretic is the randomized
method's distance from the best rank-r approximation. Added after the first two bfloat16 runs on the 0.5B model, to
test that explanation of the failed 4e-4 bound of "B-S4 change from Heretic's means"; that bound is unchanged.
heretic_compare.py runs the same stage in its comparison (with the IDE's delta from its patch). Every assertion
prints PASS or FAIL; <out.json> holds every number; the exit status is 1 when one fails.

`device=<cpu|mps>` (default cpu, the arithmetic above) is where Heretic's lines are replicated. Heretic runs them
on the device of the module's weight: v (float32) is moved there, W is the float32 weight there, D is formed there
(its own lines: row norms, F.normalize, the rank-1 update, F.normalize, the norms restored, W subtracted) and the
seed call immediately precedes svd_lowrank, whose random test matrix comes from that device's generator (on mps a
different stream from the CPU's; float32 products also differ there in their last bits). With device=mps the
replica does the same on the Apple GPU: D from the IDE's closed form (float64, on the CPU) is moved there in
float32, D from Heretic's lines is formed there; the factors come back to the CPU, where every distance and the
premise are computed. No operation falls back: what torch raises on the device is raised. <out.json> gains
"device", "run_device", "run_device_recorded" and "replica_rule".
"""
import json
import os
import sys
from pathlib import Path

if len(sys.argv) < 4 or sys.argv[1] in ("-h", "--help"):
    print(__doc__)
    sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 2)

sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from safetensors import safe_open

import formats_shared as FS
import heretic_compare as HC          # its bounds table only (its heavy imports are inside its main)

ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
args = dict(a.split("=", 1) for a in sys.argv[3:])
if "full" not in args:
    sys.exit("give the Heretic run as full=<run dir> (this stage is for row_normalization \"full\")")
run = Path(args.pop("full"))
adapter_arg = args.pop("adapter", None)
DEVICE = args.pop("device", "cpu")
if args:
    sys.exit(f"unknown arguments: {sorted(args)}")
if DEVICE not in ("cpu", "mps"):
    sys.exit(f"device={DEVICE}: the device is cpu or mps")
if DEVICE == "mps" and not torch.backends.mps.is_available():
    sys.exit("device=mps: torch.backends.mps.is_available() is False in this interpreter")

dump = json.loads((run / "dump/dump.json").read_text())
settings = dump["modifier_settings"].get("Abliteration")
if not settings or settings.get("row_normalization") != "full":
    sys.exit(f"{run}: not an abliteration run with row_normalization \"full\"")
with safe_open(str(run / "dump/dump.safetensors"), "pt") as f:
    good_mean, bad_mean = f.get_tensor("good_mean"), f.get_tensor("bad_mean")
adapter = FS.Adapter(FS.find_adapter(run, adapter_arg))
base = FS.Checkpoint(ckpt)
cfg = json.loads((ckpt / "config.json").read_text()) if (ckpt / "config.json").is_file() else {}
run_dtype = "bfloat16" if "bfloat16" in str(dump.get("dtype")) else "float32"
rank_setting = settings.get("full_normalization_lora_rank")
print("torch", torch.__version__, " run's torch", (dump.get("versions") or {}).get("torch"), " run", run.name,
      " dtype", run_dtype, " seed", dump["seed"], " r", adapter.config["r"], " (setting %s)" % rank_setting, flush=True)
RUN_DEVICE = FS.device_kind(dump.get("device"))      # where Heretic's lines ran: the device of the module weights
print("replica on", DEVICE, " run's device", RUN_DEVICE, f"({dump.get('device')})", flush=True)

rows = FS.replica_rows(base, cfg, dump, good_mean, bad_mean, adapter,
                       lambda tensor, W, v, w, r: FS.ide_full_delta(W, v, w, r, tensor), device=DEVICE)
print("  tensor                                              replica-heretic  replica-ide  ide-heretic   tail      "
      "s_r/s_1   exact-ide")
for r in rows:
    if "error" in r:
        print("  %-50s %s" % (r["tensor"] or r["module"], r["error"]))
        continue
    print("  %-50s %.3e        %.3e    %.3e   %s  %s  %.1e" % (
        r["tensor"], r["replica_vs_heretic"], r["replica_vs_ide"], r["ide_vs_heretic"],
        "%.3e" % r["tail"] if r["tail"] is not None else "   n/a   ",
        "%.3e" % r["sigma_r_over_sigma_1"] if r["sigma_r_over_sigma_1"] is not None else "   n/a   ",
        r["exact_truncation_vs_ide"]))

rep = FS.Report()
numbers = FS.replica_check(rep, run.name, rows, FS.bound(HC.BOUNDS, FS.REPLICA_STAGE, run_dtype), torch.__version__,
                           dump.get("versions"), DEVICE, dump.get("device"))
applies, why = FS.replica_applies(torch.__version__, dump.get("versions"), DEVICE, dump.get("device"))
ok = rep.ok()
FS.write_json(out, {"torch": torch.__version__, "run_torch": numbers["run_torch"], "run": str(run.name),
                    "device": DEVICE, "run_device": RUN_DEVICE, "run_device_recorded": dump.get("device"),
                    "replica_rule": {"applies": applies, "reason": why},
                    "run_dtype": run_dtype, "seed": dump["seed"], "r": adapter.config["r"],
                    "full_normalization_lora_rank": rank_setting, "rows": rows, "summary": numbers,
                    "bounds": {FS.REPLICA_STAGE: FS.bound(HC.BOUNDS, FS.REPLICA_STAGE, run_dtype),
                               FS.REPLICA_INPUT: FS.REPLICA_INPUT_BOUND},
                    "checks": rep.labels, "pass": ok})
sys.exit(0 if ok else 1)
