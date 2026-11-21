"""Where do the IDE's ARA fit and Heretic's part? No model is loaded: only tensors read from files.

    python scripts/heretic_fit_check.py <base model.safetensors | base model dir> <heretic run dir> \\
        [<out.json> [<parameter>=<value> ...]]

Run it from the IDE's `weight-space-ide` directory, with either tool's interpreter (it needs torch and
safetensors only), so the same arithmetic can be repeated under each tool's torch version.

For every matrix Heretic's exported trial fitted, from Heretic's own captured inputs and outputs (the dump of
heretic_dump.py) and the base weight, it runs two fits in this interpreter:

  replica : Heretic's fitting loop for one module, in Heretic's layout (W [d, D_in]). `replica` and
            `heretic_loss` below are copied from Heretic (src/heretic/modifiers/ara.py at 64e9991, AGPL-3.0)
            with only the model plumbing removed; they are here for this comparison and nothing imports them.
  ide     : wsp.directions_ara.fit on V = W^T, the IDE's layout and loss.

and reports, per optimiser budget (steps x max_iter), in float32 and in float64:
  - ide against replica (the two formulations in one interpreter),
  - replica against the same replica with the captures changed in their last digits (relative 1e-7), which is
    how far apart two runs of ONE formulation land when their inputs differ by rounding,
and at the run's own budget, in float32, each against what Heretic actually exported: the change
(merged - base, which holds a small change only to about 3e-8 of the weight), and the merged weight itself,
element for element (base + change, added in float32 as the export adds it).

A trailing `<parameter>=<value>` replaces one of the exported trial's parameters (for example
`overcorrect_relative_weight=0`), to see what the sensitivity depends on; the comparison with Heretic's export
is then left out, since Heretic did not run those parameters. `budgets=1x1,5x20` chooses the budgets (default:
1x1, 1x2, 1x5, 1x20 and the run's own); `dtypes=float32` leaves float64 out. Both save time on a larger model.

Runs exported as an adapter, shards and other layouts (added for the format comparisons; a run directory with a
merged model and a dump without module names is checked exactly as before):
  - The base is a model.safetensors file as before, or a model directory (one model.safetensors, or the shards of
    its model.safetensors.index.json), read one tensor at a time.
  - A run without model/model.safetensors (export_strategy "adapter") is compared with its adapter: the directory
    given as `adapter=<dir>`, else the first of <run>/adapter, <run>/<save_directory>, <run>/model that holds
    adapter_config.json; refused unless r == lora_alpha. Heretic's change is then its fitted lora_B @ lora_A
    (float32), and each row of the run's own budget also says whether the IDE's fitted factors, and their product
    B @ A, equal Heretic's bit for bit.
  - When the dump records module names (dump.json "modules", heretic_dump.py), every module Heretic fitted is
    checked under its own name (a hybrid layer's linear_attn.out_proj, a Qwen3.5 checkpoint's
    model.language_model.* names, a layer's several modules by index), matched to the checkpoint tensor through
    wsp.runtime_layouts.module_path, and the initial values are drawn in Heretic's module order. Without them the
    modules are model.layers.<l>.self_attn.o_proj and model.layers.<l>.mlp.down_proj, index 0, as before.
  - BOUNDS below, by the dtype Heretic's run was made in (dump.json "dtype"): for a bfloat16 run, B-S5 of the
    formats build spec asserts that the IDE's fit from Heretic's captures, at the run's own budget in float32,
    gives Heretic's fitted B A bit for bit (this needs the adapter). Each assertion prints PASS or FAIL,
    <out.json> gains "checks" and "pass", and the exit status is 1 when one fails. The float32 row has no bound:
    a float32 run is reported, as this script always did.
  - B-S5 is stated under the torch Heretic's run was made with, so it is asserted only when this interpreter's
    torch.__version__ equals the run's (dump.json "versions"["torch"]), compared after stripping a "+..." local
    suffix from both (formats_shared.b_s5_applies). Under another torch, or when the dump records no torch
    version, it is reported only: a note says why, and <out.json> "b_s5_reported" holds the counts the check would
    have carried (modules, not_equal, factors_equal, products_equal), so the torch-version spread is on record;
    the exit status is then decided by the other checks. <out.json> "run_torch" is the run's recorded torch.
  - `device=<cpu|mps>` (default cpu, the behaviour above) is the device the fits run on. Heretic fits on the
    device of its adapter weights, the model's device (src/heretic/modifiers/ara.py): the base weight, its row norms,
    the captures (moved there) and the adapter factors are on that device, while the initial A is drawn on the CPU
    and copied into lora_A. With device=mps the replica does the same on the Apple GPU (its factors and their product
    are then taken on the CPU, as before), and the default dtypes are float32 only (the GPU has no float64). The
    IDE's fit (wsp.directions_ara.fit, unchanged) is given its tensors on that device too; if its first fit raises
    there, it runs on the CPU and <out.json> "ide_device_note" says why. B-S5 is also stated on the device Heretic's
    fit ran on, so it is asserted only when, besides the torch version, the device the IDE's fit ran on is the run's
    (dump.json "device", compared by type: "mps:0" is mps; a dump without one is cpu; formats_shared.b_s5_applies);
    otherwise it is reported with the reason, as for another torch. <out.json> gains "device", "ide_device",
    "run_device" and "b_s5_rule".
  - Whenever the fit is compared with Heretic's adapter, every module's row also holds how far the fitted change is
    from Heretic's, ide_product_rel_heretic and replica_product_rel_heretic (||B A - B_h A_h|| / ||B_h A_h||, float64
    norms), and whether the replica's factors equal Heretic's (replica_factors_equal_heretic); B-S5's line and
    "b_s5_reported" carry the worst and the median of each, the replica's equal count, and "per_module".
"""
import json
import math
import os
import statistics
import sys
from pathlib import Path

if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
    print(__doc__)
    sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 2)

sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.linalg as LA
from safetensors import safe_open
from safetensors.torch import load_file
from torch.optim import LBFGS

import formats_shared as FS          # standard library only at its top: imports under either tool's interpreter
from wsp import directions_ara as ARA

# ---- BOUNDS[(stage, dtype)], dtype = the dtype of Heretic's run; None = reported only (this script's float32 checks)
BOUNDS = {
    ("B-S5 IDE fit from Heretic's captures", "float32"): None,
    ("B-S5 IDE fit from Heretic's captures", "bfloat16"): "bit-equal",   # the IDE's B @ A against Heretic's B @ A
}

base_arg = Path(sys.argv[1])
if base_arg.is_dir() or base_arg.name.endswith(".index.json"):
    base_dir = base_arg if base_arg.is_dir() else base_arg.parent
    base = FS.Checkpoint(base_dir)                  # one tensor at a time, from one file or its shards
else:
    base_dir = base_arg.parent
    base = safe_open(sys.argv[1], "pt")             # one tensor at a time: the whole file is not needed
base_config = json.loads((base_dir / "config.json").read_text()) if (base_dir / "config.json").is_file() else {}
run = Path(sys.argv[2])
dump = json.loads((run / "dump/dump.json").read_text())
tens = load_file(str(run / "dump/dump.safetensors"))
trial = next(t for t in dump["trials"] if t["index"] == dump["exported_trial_index"])
P, conf, seed = dict(trial["parameters"]), dump["modifier_settings"]["ARA"], dump["seed"]
replaced = dict(a.split("=", 1) for a in sys.argv[4:])
chosen = {k: replaced.pop(k) for k in ("budgets", "dtypes", "adapter", "device") if k in replaced}
DEVICE = chosen.get("device", "cpu")
if DEVICE not in ("cpu", "mps"):
    sys.exit(f"device={DEVICE}: the device is cpu or mps")
if DEVICE == "mps" and not torch.backends.mps.is_available():
    sys.exit("device=mps: torch.backends.mps.is_available() is False in this interpreter")
if DEVICE == "mps" and "float64" in chosen.get("dtypes", ""):
    sys.exit("device=mps: the Apple GPU has no float64; leave float64 out (dtypes=float32)")
RUN_DEVICE = FS.device_kind(dump.get("device"))      # where Heretic's fit ran: the device of its adapter weights
for key, value in replaced.items():
    assert key in P, f"{key} is not one of {sorted(P)}"
    P[key] = type(P[key])(value)
TENSOR = {"attn.o_proj": "model.layers.{}.self_attn.o_proj.weight", "mlp.down_proj": "model.layers.{}.mlp.down_proj.weight"}
NUDGE = 1e-7
RUN_DTYPE = "bfloat16" if "bfloat16" in str(dump.get("dtype")) else "float32"
merged_file = run / "model/model.safetensors"
merged = safe_open(str(merged_file), "pt") if merged_file.is_file() and "adapter" not in chosen else None
adapter_dir = None if merged is not None else FS.find_adapter(run, chosen.get("adapter"))
adapter = FS.Adapter(adapter_dir) if adapter_dir is not None else None      # AdapterRefused unless r == lora_alpha
if merged is None and adapter is None:
    sys.exit(f"{run}: neither model/model.safetensors nor an adapter export (adapter_config.json)")
MODULES = FS.dump_modules(dump)
names = set(base.keys())


def fitted_modules(layer):
    """[(component, module index, checkpoint tensor, Heretic's module name or None)] of one layer, in Heretic's order:
    by the dump's module names when it has them, else the two dense names (index 0), as before."""
    if MODULES is None:
        return [(comp, 0, pattern.format(layer), None) for comp, pattern in TENSOR.items()]
    out = []
    for comp, mods in (MODULES[layer] if layer < len(MODULES) else {}).items():
        for index, module in enumerate(mods):
            tensor = FS.tensor_of_module(base_config, module, names) if module else None
            assert tensor is not None, f"layer {layer} {comp} module {index} ({module}): no checkpoint tensor"
            out.append((comp, index, tensor, module))
    return out


def heretic_factors(tensor, module):
    """Heretic's (lora_B, lora_A) of the module that changes ``tensor``, from its adapter."""
    if module is not None and module in adapter.modules:
        return adapter.factors(module)
    hits = [m for m in adapter.modules if FS.tensor_of_module(base_config, m, names) == tensor]
    assert len(hits) == 1, f"{tensor}: {len(hits)} adapter modules"
    return adapter.factors(hits[0])


# ---- copied from Heretic (see the module docstring) ---------------------------------------------------------------
def mean_distances_to_knn(a, b, k):
    distances = torch.cdist(a, b)
    nearest_distances, _ = distances.topk(k, dim=1, largest=False)
    return nearest_distances.mean(1)


def heretic_loss(good_output, bad_output, new_good_output, new_bad_output):
    preserve_good_behavior = ((new_good_output - good_output) ** 2).mean()
    steer_bad_behavior = (
        mean_distances_to_knn(new_bad_output, good_output, P["neighbor_count"]).mean()
        + P["overcorrect_relative_weight"] * -mean_distances_to_knn(new_bad_output, bad_output, P["neighbor_count"]).mean()
    )
    return P["preserve_good_behavior_weight"] * preserve_good_behavior + P["steer_bad_behavior_weight"] * steer_bad_behavior


def replica(W_base, initial_A, good_input, good_output, bad_input, bad_output, steps, max_iter):
    # W_base and the captures are on the fit's device (Heretic: the adapter weights' device, the captures moved there);
    # the adapter weights are created on that device and initial_A, drawn on the CPU, is copied into lora_A.
    lora_A = torch.zeros_like(initial_A, device=W_base.device, requires_grad=True)
    lora_B = torch.zeros(W_base.shape[0], initial_A.shape[0], dtype=W_base.dtype, device=W_base.device,
                         requires_grad=True)
    W_row_norms = LA.vector_norm(W_base, dim=1, keepdim=True).detach()
    with torch.no_grad():
        lora_A.copy_(initial_A)
    good_product = good_input @ W_base.T
    bad_product = bad_input @ W_base.T

    def objective(A, B):
        new_good_output = good_product + (good_input @ A.T) @ B.T
        new_bad_output = bad_product + (bad_input @ A.T) @ B.T
        if conf["preserve_row_magnitudes"]:
            eff_row_norms = LA.vector_norm(W_base + (B @ A), dim=1)
            scale = W_row_norms.squeeze(1) / eff_row_norms
            new_good_output = new_good_output * scale
            new_bad_output = new_bad_output * scale
        return heretic_loss(good_output, bad_output, new_good_output, new_bad_output)

    optimizer = LBFGS([lora_A, lora_B], lr=conf["learning_rate"], max_iter=max_iter, history_size=conf["history_size"],
                      line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = objective(lora_A, lora_B)
        loss.backward()
        return loss

    for _ in range(steps):
        optimizer.step(closure)
    return lora_B.detach(), lora_A.detach()        # the fitted factors, on the fit's device
# ---- end of the copied part ---------------------------------------------------------------------------------------


def replica_on_cpu(W_base, initial_A, *captured_and_budget):
    """The replica's fitted (lora_B, lora_A) copied to the CPU, and their product B @ A taken there (as before)."""
    B, A = (t.cpu() for t in replica(W_base, initial_A, *captured_and_budget))
    return B, A, B @ A


IDE_DEVICE, IDE_DEVICE_NOTE, ide_fits = DEVICE, None, 0


def ide_fit(W, A0, B0, caps, p, name, dtype):
    """wsp.directions_ara.fit (unchanged) on IDE_DEVICE -> (A, B, info), the factors copied to the CPU. When the first
    fit on another device than the CPU raises, the IDE's fit runs on the CPU from then on, and IDE_DEVICE_NOTE says why
    (a failure after a fit completed on that device is raised: the modules would mix devices)."""
    global IDE_DEVICE, IDE_DEVICE_NOTE, ide_fits
    while True:
        dev = torch.device(IDE_DEVICE)
        Wd, xg, yg, xb, yb = (t.to(dev) for t in (W, *caps))
        try:
            data = ARA.prepare_data(Wd.T.contiguous(), torch.zeros(W.shape[0], dtype=dtype, device=dev), xg, yg, xb, yb)
            A, B, info = ARA.fit(data, A0.to(dtype).to(dev), B0.to(dtype).to(dev), p, name)
            ide_fits += 1
            return A.cpu(), B.cpu(), info
        except ARA.DR.DirectionError:
            raise
        except Exception as e:                                        # noqa: BLE001 - recorded, then the CPU
            if IDE_DEVICE == "cpu" or ide_fits:
                raise
            IDE_DEVICE_NOTE = (f"wsp.directions_ara.fit raised on {IDE_DEVICE} ({type(e).__name__}: {e}); the IDE's "
                               "fit runs on the cpu (wsp/ is not changed for this check)")
            print("note:", IDE_DEVICE_NOTE, flush=True)
            IDE_DEVICE = "cpu"


def differences(own_rows):
    """Per module, how far the IDE's and the replica's fitted change B @ A are from Heretic's (float64 norm of the
    difference over the norm of Heretic's), with the worst and the median; and the replica's factor equality."""
    ide = [r["ide_product_rel_heretic"] for r in own_rows]
    rep_ = [r["replica_product_rel_heretic"] for r in own_rows]
    return {"ide_product_rel_worst": max(ide), "ide_product_rel_median": statistics.median(ide),
            "replica_product_rel_worst": max(rep_), "replica_product_rel_median": statistics.median(rep_),
            "replica_factors_equal": sum(r["replica_factors_equal_heretic"] for r in own_rows),
            "per_module": [{"tensor": r["tensor"], "ide_product_rel_heretic": r["ide_product_rel_heretic"],
                            "replica_product_rel_heretic": r["replica_product_rel_heretic"]} for r in own_rows]}


def rel(a, b):
    d, n = float((a.double() - b.double()).norm()), float(b.double().norm())
    return d / n if n > 0 else (0.0 if d == 0 else math.inf)


def captures(layer, comp, dtype, nudge, index=0):
    out = []
    g = torch.Generator().manual_seed(7)
    for key in (f"io.good.L{layer}.{comp}.{index}.x", f"io.good.L{layer}.{comp}.{index}.y",
                f"io.bad.L{layer}.{comp}.{index}.x", f"io.bad.L{layer}.{comp}.{index}.y"):
        t = tens[key].to(dtype)
        if nudge:
            t = t * (1 + NUDGE * torch.randn(t.shape, generator=g, dtype=torch.float64)).to(dtype)
        out.append(t.to(DEVICE))                    # Heretic: .float().to(lora_A.device)
    return out


own = (conf["n_optimization_steps"], conf["max_iter"])
budgets = [(1, 1), (1, 2), (1, 5), (1, 20), own]
if "budgets" in chosen:
    budgets = [tuple(int(x) for x in b.split("x")) for b in chosen["budgets"].split(",")]
dtypes = [getattr(torch, t) for t in chosen.get("dtypes", "float32,float64" if DEVICE == "cpu" else "float32").split(",")]
rows = []
for dtype in dtypes:
    for steps, max_iter in budgets:
        gens = [torch.Generator().manual_seed(seed) for _ in range(2)]   # one per call, consumed in fit order
        for layer in range(P["start_layer_index"], P["end_layer_index"]):
            for comp, index, name, module in fitted_modules(layer):
                assert name.replace(".weight", ".bias") not in base.keys(), "this check has no bias term"
                W = base.get_tensor(name).to(dtype)                       # on the CPU; Wd on the fit's device
                Wd = W.to(DEVICE)
                r = conf["lora_rank"]
                initial_A = torch.empty(r, W.shape[1])                    # drawn in float32 on the CPU, as both tools do
                torch.nn.init.kaiming_uniform_(initial_A, a=math.sqrt(5), generator=gens[0])
                A0, B0 = ARA.initial_values(gens[1], r, W.shape[1], W.shape[0])
                assert torch.equal(A0, initial_A), "the two initial values differ"
                caps = captures(layer, comp, dtype, False, index)          # on the fit's device
                rB, rA, dW_r = replica_on_cpu(Wd, initial_A.to(dtype), *caps, steps, max_iter)
                _, _, dW_n = replica_on_cpu(Wd, initial_A.to(dtype), *captures(layer, comp, dtype, True, index),
                                            steps, max_iter)
                p = {**ARA.DEFAULTS, **P, "preserve_row_magnitudes": conf["preserve_row_magnitudes"], "lora_rank": r,
                     "n_optimization_steps": steps, "learning_rate": conf["learning_rate"], "max_iter": max_iter,
                     "history_size": conf["history_size"], "seed": seed}
                A, B, info = ide_fit(W, A0, B0, caps, p, name, dtype)       # factors on the CPU
                dW_i = (A.T @ B.T).T
                row = {"dtype": str(dtype).replace("torch.", ""), "budget": [steps, max_iter], "tensor": name,
                       "change_over_weight": float(dW_r.norm() / W.norm()), "ide_vs_replica": rel(dW_i, dW_r),
                       "nudged_replica_vs_replica": rel(dW_n, dW_r),
                       "loss_before": info["loss_before"], "loss_after_ide": info["loss_after"]}
                if MODULES is not None:
                    row.update(component=comp, module_index=index, module=module)
                if (steps, max_iter) == own and dtype == torch.float32 and not replaced:
                    if merged is not None:
                        exported = merged.get_tensor(name).float()
                        dW_h = exported - W
                        row.update(replica_vs_heretic=rel(dW_r, dW_h), ide_vs_heretic=rel(dW_i, dW_h),
                                   heretic_change_over_weight=float(dW_h.norm() / W.norm()), elements=W.numel(),
                                   replica_merged_differs=int(((W + dW_r) != exported).sum()),
                                   ide_merged_differs=int(((W + dW_i) != exported).sum()))
                    else:
                        hB, hA = heretic_factors(name, module)
                        dW_h = hB @ hA                                       # Heretic's fitted B A (scaling 1)
                        prod_i = B @ A                                        # the IDE's factors, multiplied alike
                        row.update(replica_vs_heretic=rel(dW_r, dW_h), ide_vs_heretic=rel(dW_i, dW_h),
                                   heretic_change_over_weight=float(dW_h.norm() / W.norm()), elements=W.numel(),
                                   heretic_from="adapter",
                                   ide_factors_equal_heretic=bool(A.shape == hA.shape and B.shape == hB.shape
                                                                  and torch.equal(A, hA) and torch.equal(B, hB)),
                                   ide_product_equals_heretic=bool(torch.equal(prod_i, dW_h)),
                                   ide_product_differs=int((prod_i != dW_h).sum()),
                                   replica_product_differs=int((dW_r != dW_h).sum()),
                                   replica_factors_equal_heretic=bool(rA.shape == hA.shape and rB.shape == hB.shape
                                                                      and torch.equal(rA, hA) and torch.equal(rB, hB)),
                                   ide_product_rel_heretic=rel(prod_i, dW_h),     # B @ A, float64 norms
                                   replica_product_rel_heretic=rel(dW_r, dW_h))
                rows.append(row)

print("torch", torch.__version__, " run", run.name, " seed", seed, " parameters", P,
      f" (replaced: {sorted(replaced)})" if replaced else " (the exported trial's)")
print("device", DEVICE, " ide fit on", IDE_DEVICE, " run fitted on", RUN_DEVICE, f"({dump.get('device')})",
      f" ({IDE_DEVICE_NOTE})" if IDE_DEVICE_NOTE else "")
print("  dtype   budget tensor                                    change/weight  ide vs replica  nudged replica vs replica")
for row in rows:
    extra = ""
    if "replica_merged_differs" in row:
        extra = "   vs Heretic's export: replica %.2e  ide %.2e; merged weight differs in %d (replica) %d (ide) of %d" % (
            row["replica_vs_heretic"], row["ide_vs_heretic"], row["replica_merged_differs"], row["ide_merged_differs"],
            row["elements"])
    elif "ide_product_equals_heretic" in row:
        extra = "   vs Heretic's adapter: replica %.2e  ide %.2e; B A differs in %d (replica) %d (ide) of %d" % (
            row["replica_vs_heretic"], row["ide_vs_heretic"], row["replica_product_differs"], row["ide_product_differs"],
            row["elements"])
    print("  %-7s %-6s %-41s %.3e      %.2e        %.2e%s" % (
        row["dtype"], "%dx%d" % tuple(row["budget"]), row["tensor"], row["change_over_weight"], row["ide_vs_replica"],
        row["nudged_replica_vs_replica"], extra))

# ---- B-S5, by the run's dtype (None = reported only, as this script always reported)
rep = FS.Report()
stage = "B-S5 IDE fit from Heretic's captures"
bound = FS.bound(BOUNDS, stage, RUN_DTYPE)
own_rows = [r for r in rows if r["budget"] == list(own) and r["dtype"] == "float32" and "ide_vs_heretic" in r]
RUN_TORCH = dump["versions"].get("torch") if isinstance(dump.get("versions"), dict) else None
# asserted only under the run's torch AND with the IDE's fit on the run's device (formats_shared.b_s5_applies)
applies, why = FS.b_s5_applies(torch.__version__, dump.get("versions"), IDE_DEVICE, dump.get("device"))
b_s5_reported = None
if DEVICE != "cpu":
    rep.note(run.name, "device of the fits", {"device": DEVICE, "ide_device": IDE_DEVICE, "run_device": RUN_DEVICE,
                                              "ide_device_note": IDE_DEVICE_NOTE
                                              or f"wsp.directions_ara.fit ran on {IDE_DEVICE} unchanged"})
if bound is not None:
    if replaced:
        rep.note(run.name, stage, "parameters replaced: Heretic did not run them, so its export is not compared")
    elif not applies:
        rep.note(run.name, stage, f"reported only: {why}")
        b_s5_reported = {"reason": why}
        if own_rows and all("ide_product_equals_heretic" in r for r in own_rows):
            b_s5_reported.update(
                modules=len(own_rows),
                not_equal=[r["tensor"] for r in own_rows if not r["ide_product_equals_heretic"]][:5],
                factors_equal=sum(r["ide_factors_equal_heretic"] for r in own_rows),
                products_equal=sum(r["ide_product_equals_heretic"] for r in own_rows), **differences(own_rows))
            print(f"     {run.name} {stage} (reported) {len(own_rows)} modules: factors equal "
                  f"{b_s5_reported['factors_equal']} (ide) {b_s5_reported['replica_factors_equal']} (replica); "
                  f"B @ A relative to Heretic's: ide worst {b_s5_reported['ide_product_rel_worst']:.3e} median "
                  f"{b_s5_reported['ide_product_rel_median']:.3e}, replica worst "
                  f"{b_s5_reported['replica_product_rel_worst']:.3e} median "
                  f"{b_s5_reported['replica_product_rel_median']:.3e}", flush=True)
        else:
            b_s5_reported["unavailable"] = ("no adapter export" if adapter is None
                                            else "the run's own budget in float32 was left out")
    elif adapter is None:
        rep.check(run.name, stage, False, error="B-S5 compares with Heretic's fitted B A: the run has no adapter export")
    elif not own_rows:
        rep.check(run.name, stage, False, error="B-S5 is at the run's own budget in float32: budgets or dtypes left it out")
    else:
        rep.check(run.name, stage, all(r["ide_product_equals_heretic"] for r in own_rows), modules=len(own_rows),
                  not_equal=[r["tensor"] for r in own_rows if not r["ide_product_equals_heretic"]][:5],
                  factors_equal=sum(r["ide_factors_equal_heretic"] for r in own_rows), **differences(own_rows))
ok = rep.ok()
if len(sys.argv) > 3:
    reported = {} if b_s5_reported is None else {"b_s5_reported": b_s5_reported}
    Path(sys.argv[3]).write_text(json.dumps({"torch": torch.__version__, "run_torch": RUN_TORCH, **reported,
                                             "device": DEVICE, "ide_device": IDE_DEVICE,
                                             "ide_device_note": IDE_DEVICE_NOTE, "run_device": RUN_DEVICE,
                                             "run_device_recorded": dump.get("device"), "b_s5_rule": {"applies": applies, "reason": why},
                                             "run": run.name, "seed": seed, "parameters": P,
                                             "replaced": sorted(replaced), "settings": {k: v for k, v in conf.items() if "prompts" not in k},
                                             "nudge": NUDGE, "rows": rows,
                                             "run_dtype": RUN_DTYPE, "heretic_export": "merged" if merged is not None else "adapter",
                                             "modules_by": "names" if MODULES is not None else "index",
                                             "bounds": {k[0]: v for k, v in BOUNDS.items() if k[1] == RUN_DTYPE},
                                             "checks": rep.labels, "pass": ok}, indent=1))
sys.exit(0 if ok else 1)
