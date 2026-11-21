"""What the format comparisons share: the report, the bounds lookup, a sharded checkpoint read one tensor at a time,
Heretic's exported adapter as float32 deltas, the bfloat16 one-step rule and the merged model PEFT would make.

Imported by scripts/heretic_compare.py, heretic_compare_ara.py, heretic_compare_scores.py, formats_bf16_self.py,
formats_gguf_compare.py, formats_timing.py and formats_transport.py. It is also a small model-free command, run with
the IDE's interpreter from the IDE's `weight-space-ide` directory (offline, under the memory guard):

    python scripts/formats_shared.py merge <base model dir> <heretic adapter dir> <out dir>

which writes the merged model of a Heretic run exported as an adapter (export_strategy "adapter") the way PEFT merges
it (`base_layer.weight.data += B @ A * (lora_alpha / r)`: a float32 delta added in place to the checkpoint's tensor,
so a bfloat16 weight is rounded once). Only the weight files that hold a changed tensor are written (streamed,
tensor by tensor); every other file of the base directory is a symbolic link to the base's. <out dir>/merge.json
lists the changed tensors and the SHA-256 of every tensor written. Give the out directory to heretic_dump.py as its
merged model (its second argument) and to heretic_compare_scores.py (it finds <run>/heretic-merged).

Nothing here loads a model or reaches the network.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

ADAPTER_PREFIX = "base_model.model."
ADAPTER_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
HERETIC_SYSTEM_PROMPT = "You are a helpful assistant."   # Heretic's system_prompt default (config.py at 64e9991)
DTYPES = ("float32", "bfloat16")


# ------------------------------------------------------------------ the report
class Report:
    """Every number and a pass flag per assertion, by label; a stage without a bound is a note (reported only)."""

    def __init__(self):
        self.labels = {}

    def check(self, label, name, ok, **numbers):
        self.labels.setdefault(label, {})[name] = {"pass": bool(ok), **numbers}
        shown = {k: v for k, v in numbers.items()
                 if not isinstance(v, (list, dict)) or len(json.dumps(v, default=str)) < 300}
        print(f"{'PASS' if ok else 'FAIL'} {label} {name} {json.dumps(shown, default=str)}", flush=True)
        return bool(ok)

    def note(self, label, name, value):
        self.labels.setdefault(label, {})[name] = value
        if not isinstance(value, (list, dict)) or len(json.dumps(value, default=str)) < 300:
            print(f"     {label} {name} (reported) {json.dumps(value, default=str)}", flush=True)

    def bounded(self, label, name, bound, value, ok_fn, **numbers):
        """Assert ``ok_fn(value, bound)`` when the bounds table gives a bound, else report the value."""
        if bound is None:
            self.note(label, f"{name} (no bound for this dtype)", {"value": value, **numbers})
            return True
        return self.check(label, name, ok_fn(value, bound), value=value, bound=bound, **numbers)

    def ok(self):
        return all(v["pass"] for e in self.labels.values() for v in e.values() if isinstance(v, dict) and "pass" in v)


def bound(table, stage, dtype):
    """``table[(stage, dtype)]``; a missing key is a programming error (every script lists every row it reads)."""
    if (stage, dtype) not in table:
        raise KeyError(f"the bounds table has no row ({stage!r}, {dtype!r})")
    return table[(stage, dtype)]


def torch_release(v):
    """A torch version string without its local suffix ("2.14.1+cpu" -> "2.14.1"); None stays None."""
    return None if v is None else str(v).split("+", 1)[0].strip()


def torch_matches_run(this_torch, run_versions, claim="bit equality"):
    """Whether a claim stated under the torch Heretic's run was made with is asserted in this interpreter: only
    when this interpreter's torch equals the run's recorded one (dump.json "versions"["torch"], importlib.metadata's
    form), compared after stripping a "+..." local suffix from both. -> (applies, reason)."""
    run_torch = run_versions.get("torch") if isinstance(run_versions, dict) else None
    if not run_torch:
        return False, "the run's dump records no torch version"
    if torch_release(this_torch) == torch_release(run_torch):
        return True, f"this interpreter's torch {this_torch} is the run's {run_torch}"
    return False, (f"this interpreter's torch {this_torch} is not the run's {run_torch}; {claim} is stated "
                   "under the run's own torch")


def device_kind(device):
    """The type of a torch device as a dump records it (dump.json "device", str(model.device)): "mps:0" -> "mps",
    "cpu" -> "cpu", "cuda:1" -> "cuda". A missing or empty device is "cpu" (a dump that records none was made on
    the CPU)."""
    text = "" if device is None else str(device).strip().lower()
    return text.split(":", 1)[0].strip() or "cpu"


def device_matches_run(this_device, run_device, claim="bit equality"):
    """Whether a claim stated on the device Heretic's fit ran on is asserted for a fit computed on ``this_device``.
    Heretic fits on the device of the adapter weights, the model's device (modifiers/ara.py moves the captures
    there), which the dump records as "device"; the two are compared by type (device_kind). -> (applies, reason)."""
    mine, theirs = device_kind(this_device), device_kind(run_device)
    if mine == theirs:
        return True, f"this check fits on {mine}, the run's device"
    return False, (f"this check fits on {mine}, the run fitted on {theirs}; {claim} is stated on the run's own "
                   "device")


def b_s5_applies(this_torch, run_versions, this_device="cpu", run_device=None):
    """Whether B-S5 (the IDE's fit from Heretic's captures bit-equal to Heretic's fitted B A) is asserted in this
    interpreter: only when this interpreter's torch is the run's (torch_matches_run) AND the IDE's fit is computed
    on the device Heretic's fit ran on (device_matches_run; ``run_device`` is dump.json "device", missing = "cpu").
    -> (applies, reason); when either differs the reason names each difference."""
    torch_ok, torch_why = torch_matches_run(this_torch, run_versions, "bit equality")
    device_ok, device_why = device_matches_run(this_device, run_device, "bit equality")
    return _both_apply((torch_ok, torch_why), (device_ok, device_why))


def _both_apply(torch_rule, device_rule):
    """(applies, reason) of two (ok, reason) rules: asserted only when both hold; else the reasons that fail."""
    (torch_ok, torch_why), (device_ok, device_why) = torch_rule, device_rule
    if torch_ok and device_ok:
        return True, f"{torch_why}, and {device_why}"
    return False, "; ".join(why for ok, why in ((torch_ok, torch_why), (device_ok, device_why)) if not ok)


def replica_applies(this_torch, run_versions, this_device="cpu", run_device=None):
    """Whether the bound of the replica of Heretic's randomized approximation (REPLICA_STAGE) is asserted: only when
    this interpreter's torch is the run's (torch_matches_run) AND the replica ran on the device Heretic's run was
    made on (``run_device`` dump.json "device", missing = "cpu"; compared by device_kind). svd_lowrank draws its
    random test matrix from the device's generator, a different stream on mps than on the CPU, and float32
    products differ in their last bits between devices. -> (applies, reason); the reason names each difference."""
    claim = "the replica's agreement"
    mine, theirs = device_kind(this_device), device_kind(run_device)
    if mine == theirs:
        device_rule = (True, f"the replica ran on {mine}, the run's device")
    else:
        device_rule = (False, f"the replica ran on {mine}, Heretic's run on {theirs}; {claim} is stated on the run's "
                              "own device")
    return _both_apply(torch_matches_run(this_torch, run_versions, claim), device_rule)


def replica_device(run_device, mps_available=None):
    """The device heretic_compare.py's inline replica runs on: the run's (dump.json "device", by device_kind) when
    this process has it (mps: torch.backends.mps.is_available(), or ``mps_available`` when given), else the CPU.
    Only cpu and mps are used; a run on any other device is replicated on the CPU (and its bound reported)."""
    kind = device_kind(run_device)
    if kind == "mps":
        if mps_available is None:
            import torch
            mps_available = torch.backends.mps.is_available()
        return "mps" if mps_available else "cpu"
    return "cpu"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    os.replace(tmp, path)


def set_run_dtype(dtype):
    """WSP_RUN_DTYPE for the runner the workspace builds later (it reads the variable when it is created)."""
    if dtype not in DTYPES:
        raise SystemExit(f"--dtype must be one of {', '.join(DTYPES)}")
    os.environ["WSP_RUN_DTYPE"] = dtype


def rel(a, b):
    d = float((a.double() - b.double()).norm())
    n = float(b.double().norm())
    return d / n if n > 0 else (0.0 if d == 0 else math.inf)


# ------------------------------------------------------------------ a checkpoint, one tensor at a time
class Checkpoint:
    """The safetensors weights of a model directory: one model.safetensors, or the shards its
    model.safetensors.index.json names. Tensors are read one at a time (never a whole file)."""

    def __init__(self, directory):
        from safetensors import safe_open
        self.dir = Path(directory)
        index = self.dir / "model.safetensors.index.json"
        single = self.dir / "model.safetensors"
        if single.is_file():
            self.files = {"model.safetensors": safe_open(str(single), "pt")}
            self.where = {k: "model.safetensors" for k in self.files["model.safetensors"].keys()}
        elif index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            self.files = {f: safe_open(str(self.dir / f), "pt") for f in sorted(set(weight_map.values()))}
            self.where = dict(weight_map)
        else:
            raise FileNotFoundError(f"{self.dir}: no model.safetensors and no model.safetensors.index.json")

    def keys(self):
        return list(self.where)

    def __contains__(self, name):
        return name in self.where

    def get_slice(self, name):
        return self.files[self.where[name]].get_slice(name)

    def get_tensor(self, name):
        return self.files[self.where[name]].get_tensor(name)

    def shape(self, name):
        return list(self.get_slice(name).get_shape())

    def dtype(self, name):
        return self.get_slice(name).get_dtype()


def open_checkpoint(directory):
    return Checkpoint(directory)


# ------------------------------------------------------------------ Heretic's adapter
class AdapterRefused(ValueError):
    pass


def adapter_modules(keys):
    return sorted({k[len(ADAPTER_PREFIX):-len(s)] for k in keys for s in ADAPTER_SUFFIXES
                   if k.startswith(ADAPTER_PREFIX) and k.endswith(s)})


def adapter_config(directory):
    """adapter_config.json of a Heretic adapter, refused unless it applies every module at full strength:
    ``r == lora_alpha`` and no rank_pattern or alpha_pattern (Heretic sets lora_alpha = r, model.py apply_lora)."""
    directory = Path(directory)
    files = [directory / "adapter_config.json", directory / "adapter_model.safetensors"]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        raise AdapterRefused(f"{directory}: not an adapter export (missing {', '.join(missing)}); configure the "
                             f"Heretic run with export_strategy = \"adapter\"")
    cfg = json.loads(files[0].read_text())
    if cfg.get("r") != cfg.get("lora_alpha"):
        raise AdapterRefused(f"{directory}: r = {cfg.get('r')} and lora_alpha = {cfg.get('lora_alpha')}; the comparison "
                             f"reads B @ A as the change, which holds only when r == lora_alpha")
    for key in ("rank_pattern", "alpha_pattern"):
        if cfg.get(key):
            raise AdapterRefused(f"{directory}: {key} is set ({sorted(cfg[key])[:3]}); the comparison reads B @ A as "
                                 f"the change of every module, which holds only without per-module patterns")
    return cfg


class Adapter:
    """A Heretic adapter export: per module (PEFT's module name without base_model.model.) the float32 change
    ``B @ A`` (the scaling is 1, checked by adapter_config), computed as PEFT's get_delta_weight computes it."""

    def __init__(self, directory):
        from safetensors import safe_open
        self.dir = Path(directory)
        self.config = adapter_config(self.dir)
        self.file = safe_open(str(self.dir / "adapter_model.safetensors"), "pt")
        self.modules = adapter_modules(self.file.keys())
        bad = [m for m in self.modules if not all(f"{ADAPTER_PREFIX}{m}{s}" in set(self.file.keys())
                                                  for s in ADAPTER_SUFFIXES)]
        if bad:
            raise AdapterRefused(f"{self.dir}: modules without both factors: {bad[:3]}")

    def factors(self, module):
        return (self.file.get_tensor(f"{ADAPTER_PREFIX}{module}.lora_B.weight"),
                self.file.get_tensor(f"{ADAPTER_PREFIX}{module}.lora_A.weight"))

    def delta(self, module):
        import torch
        b, a = self.factors(module)
        if b.dtype != torch.float32 or a.dtype != torch.float32:
            raise AdapterRefused(f"{self.dir}: {module} factors are {b.dtype}/{a.dtype}, not float32")
        return b @ a                     # (B @ A) * scaling with scaling 1.0: the multiplication is exact

    def dtypes(self):
        return sorted({str(self.file.get_slice(k).get_dtype()) for k in self.file.keys()})


def find_adapter(run, explicit=None):
    """The adapter directory of a Heretic run: ``explicit`` when given, else the first of <run>/adapter,
    <run>/<save_directory of its config.toml>, <run>/model that holds adapter_config.json; else None."""
    if explicit:
        return Path(explicit)
    run = Path(run)
    candidates = [run / "adapter"]
    cfg = run / "config.toml"
    if cfg.is_file():
        import tomllib
        save = tomllib.loads(cfg.read_text()).get("save_directory")
        if save:
            candidates.append(run / save)
    candidates.append(run / "model")
    return next((c for c in candidates if (c / "adapter_config.json").is_file()), None)


def parse_adapters(values, modes):
    """--heretic-adapter values: ``<dir>`` (one run only) or ``<mode>=<dir>`` -> {mode: dir}."""
    out = {}
    for v in values or []:
        mode, sep, d = v.partition("=")
        if sep and mode in modes:
            out[mode] = d
        elif len(modes) == 1 and not sep:
            out[modes[0]] = v
        else:
            raise SystemExit(f"--heretic-adapter {v}: give <dir> with one run, or <mode>=<dir> naming one of "
                             f"{', '.join(modes)}")
    return out


# ------------------------------------------------------------------ module names: Heretic's to the workspace's
def layer_relative(name, layer):
    """``model.language_model.layers.3.linear_attn.out_proj`` (or the loaded-model form) -> ``linear_attn.out_proj``;
    None when the name is not under layer ``layer``."""
    m = re.search(rf"(?:^|\.)layers\.{int(layer)}\.(.+)$", name)
    return m.group(1) if m else None


def tensor_of_module(cfg, module, tensor_names):
    """Heretic's full module name -> the checkpoint tensor ``<module>.weight`` the workspace addresses: the exact
    name when the checkpoint has it, else the tensor whose module path maps to the same loaded-model path
    (wsp.runtime_layouts.module_path on both sides); None when there is none or more than one. It lives in
    wsp.peft_adapter, which the IDE's scoring of an adapter directory uses (wsp does not import scripts/)."""
    from wsp.peft_adapter import tensor_of_module as of
    return of(cfg, module, tensor_names)


def dump_modules(dump):
    """dump.json["modules"] (heretic_dump.py): per layer {component: [full module name per module index]}, or None
    for a dump written before it was recorded (the comparison then matches by index, as before)."""
    mods = dump.get("modules")
    return mods if isinstance(mods, list) and all(isinstance(m, dict) for m in mods) else None


def io_entries(L):
    """The module entries of one layer of a module_io measurement's io.json, version 1 or 2, as
    [(component, module path relative to the layer or None, entry with captured/d_in/x/y)]."""
    out = []
    if isinstance(L.get("modules"), list):                     # a flat module list
        for e in L["modules"]:
            out.append((e.get("component"), e.get("module"), e))
        return out
    for comp, c in (L.get("components") or {}).items():
        if isinstance(c, list):                                 # per component, one entry per module
            out.extend((comp, e.get("module"), e) for e in c)
        elif isinstance(c, dict) and isinstance(c.get("modules"), list):
            out.extend((comp, e.get("module"), e) for e in c["modules"])
        else:                                                   # version 1: one module per component
            out.append((comp, c.get("module") if isinstance(c, dict) else None, c))
    return out


def adapter_changes(ws, base, adapter, patch, rounding=False):
    """Heretic's adapter change B @ A against the IDE's patch delta (its float32 pieces, ``ws.read(variant="delta")``)
    per tensor; with ``rounding`` also round_bf16(W + delta) of the two, element by element.
    ``base`` is a Checkpoint. -> (rows, tensors unchanged in both, adapter modules with no tensor)."""
    import torch
    names = set(base.keys())
    targets, missing = {}, []
    for m in adapter.modules:
        t = tensor_of_module(ws.hf_config, m, names)
        if t is None:
            missing.append(m)
        else:
            targets[t] = m
    ide = {t for t, ps in patch.deltas.items() if any(not p.is_zero() for p in ps)}
    rows = []
    for name in sorted(set(targets) | ide):
        W = base.get_tensor(name)
        d_h = adapter.delta(targets[name]) if name in targets else torch.zeros(W.shape)
        d_i = ws.read(name, variant="delta").float() if name in ide else torch.zeros(W.shape)
        h, i = bool(d_h.any()), bool(d_i.any())
        if not h and not i:
            continue
        if h != i:
            rows.append({"tensor": name, "heretic_changed": h, "ide_changed": i})
            continue
        wn = float(W.double().norm())
        row = {"tensor": name, "rel_diff": rel(d_i, d_h),
               "change_over_weight": float(d_h.double().norm()) / wn if wn > 0 else None,
               "ide_change_over_weight": float(d_i.double().norm()) / wn if wn > 0 else None,
               "max_abs_diff": float((d_i - d_h).abs().max()), "elements": W.numel(),
               "elements_that_differ": int((d_i != d_h).sum())}
        if rounding:
            row["rounded"] = rounded_sums(W, d_i, d_h)
        rows.append(row)
        del W, d_h, d_i
    touched = {r["tensor"] for r in rows}
    return rows, len([k for k in base.keys() if k not in touched]), missing


def ara_components(record=None, output=""):
    """The per-module rows of every ablate_ara call: from a patch record or run_script's ``calls`` (any nesting;
    dicts with ``mismatch_good``), else parsed from the script's output lines (``L<l> <component>: loss ...
    mismatch good <g> bad <b>``). -> [{layer, component, module?, mismatch_good, mismatch_bad, ...}]."""
    found = []

    def walk(x):
        if isinstance(x, dict):
            if "mismatch_good" in x and "mismatch_bad" in x:
                found.append(x)
                return
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(record)
    if found:
        return found
    rows = []
    for m in re.finditer(r"L(\d+) (\S+?)(?: (\S+))?: loss .*?mismatch good (\S+) bad (\S+)", output or ""):
        rows.append({"layer": int(m.group(1)), "component": m.group(2), "module": m.group(3),
                     "mismatch_good": float(m.group(4)), "mismatch_bad": float(m.group(5)), "from": "output"})
    return rows


# ------------------------------------------------------------------ bfloat16 rounding and the one-step rule
def bf16_order(t):
    """A monotonic integer per bfloat16 value (adjacent representable values differ by 1; +0 and -0 are both 0)."""
    import torch
    u = t.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    mag = u & 0x7FFF
    return torch.where((u & 0x8000) != 0, -mag, mag)


def bf16_steps(a, b):
    """|steps| between two tensors of bfloat16 values, element by element (int32)."""
    return (bf16_order(a) - bf16_order(b)).abs()


ONE_STEP_FRACTION = 0.01        # B-S4: at most 1% of a tensor's elements may differ after rounding
ROUNDED_CHUNK = 1 << 21         # elements per row chunk of rounded_sums (8 MiB per float32 temporary)


def step_at(m):
    """The bfloat16 spacing at a positive value m: 2 ** (floor(log2(m)) - 7), exact (frexp, no log rounding)."""
    _, e = math.frexp(m)            # m = f * 2**e with 0.5 <= f < 1, so floor(log2(m)) = e - 1
    return math.ldexp(1.0, e - 1 - 7)


def spacing(t):
    """Element by element, the bfloat16 spacing at |t| (2 ** (floor(log2 |t|) - 7), as step_at), 0 where t is 0;
    float32. Exact: torch.frexp gives the exponent, no log is rounded."""
    import torch
    t = t.to(torch.float32)
    _, e = torch.frexp(t)           # |t| = f * 2**e with 0.5 <= f < 1, so floor(log2 |t|) = e - 1
    s = torch.ldexp(torch.ones_like(t), e - 8)
    return torch.where(t == 0, torch.zeros_like(t), s)


def rounded_sums(W, d1, d2, chunk=None):
    """round_bf16(W + d1) against round_bf16(W + d2), W the base values (exact in float32), d1, d2 float32 deltas.
    Both sums in float32, one rounding each (PEFT's merge). With a, b the two rounded tensors (b from d2):

    The rule (one_step_ok): (a) every element has |a - b| <= max(sp(a), sp(b), s_m), sp(x) the bfloat16 spacing
    at |x| (0 for 0) and s_m the spacing at m, the median of |b| over the tensor (torch.median, the lower middle
    value, so a bfloat16 value; if m is 0, the smallest non-zero |b|; if every element is 0 the tensor passes only
    when a == b): an element at or above the median magnitude may flip by one of its own rounding steps, a smaller
    one may differ by at most one step of the median magnitude; and (b) at most ONE_STEP_FRACTION of the elements
    have a != b. Two exact sums that differ by much less than a rounding step round to the same value or to
    neighbouring values, so one own step is the most a non-identical pair of deltas can be held to; for elements
    near zero the own step is far smaller than the difference of the deltas, so there the bound is absolute, one
    step at the tensor's median magnitude.

    Corrected twice before being applied: the first rule (every element equal or one own step apart) cannot hold
    for elements near zero; the second (every element within one step at the median magnitude) cannot hold for
    elements above the median, where one flip of an element's own rounding is already more than one median step.

    -> {elements, equal, one_step, more_than_one, max_steps (own-step counts, reported), median_abs,
    step_at_median, max_abs_difference, max_abs_difference_in_median_steps (reported), differing,
    largest_abs_over_median (the largest |b| among the elements more than one own step apart, over m; None when
    there are none), outside (elements violating (a)), max_excess (the largest |a - b| / max(sp(a), sp(b), s_m);
    0 when none differ)}. Computed in row chunks of at most ``chunk`` elements (ROUNDED_CHUNK); the numbers do not
    depend on the chunk."""
    import torch
    # Row chunks: on a [2560, 9216] matrix the thirty-odd whole-tensor temporaries of this function came to about
    # 3 GiB. Every number returned is a count or a maximum over elements, so chunks give the same numbers; only
    # |b| is held whole (float32), for its median.
    if W.dim() == 0:
        W, d1, d2 = W.reshape(1), d1.reshape(1), d2.reshape(1)
    n = W.shape[0]
    per = max(1, (chunk or ROUNDED_CHUNK) // max(1, math.prod(W.shape[1:])))
    parts = [slice(lo, min(n, lo + per)) for lo in range(0, n, per)]

    def rounded(z, d):
        return (W[z].to(torch.float32) + d[z].to(torch.float32)).to(torch.bfloat16)

    babs = torch.empty(tuple(W.shape), dtype=torch.float32)
    for z in parts:
        babs[z] = rounded(z, d2).float().abs()
    m = float(babs.median()) if babs.numel() else 0.0
    if m == 0.0:
        nz = babs[babs > 0]
        m = float(nz.min()) if nz.numel() else 0.0
        del nz
    s_m = step_at(m) if m > 0 else None
    out = {"elements": int(W.numel()), "equal": 0, "one_step": 0, "more_than_one": 0, "max_steps": 0}
    mx, largest_abs, differing, outside, max_excess = 0.0, None, 0, 0, 0.0
    for z in parts:
        a, b = rounded(z, d1), rounded(z, d2)
        s = bf16_steps(a, b)
        out["equal"] += int((s == 0).sum())
        out["one_step"] += int((s == 1).sum())
        out["more_than_one"] += int((s > 1).sum())
        out["max_steps"] = max(out["max_steps"], int(s.max()) if s.numel() else 0)
        af, bf = a.float(), b.float()
        del a, b
        # |a - b| in float32: rounding is monotonic and every allowed step is a power of two representable in
        # float32, so (rounded |a - b|) <= step exactly when |a - b| <= step, and the maximum of the rounded values
        # is the rounded maximum; dividing by a power of two is exact.
        diff = (af - bf).abs()
        mx = max(mx, float(diff.max()) if diff.numel() else 0.0)
        far = s > 1
        if bool(far.any()):
            top = float(babs[z][far].max())
            largest_abs = top if largest_abs is None else max(largest_abs, top)
        differs = s != 0
        del s, far
        allowed = torch.maximum(spacing(af), spacing(bf))
        del af, bf
        if s_m is not None:
            allowed = torch.clamp(allowed, min=s_m)
        if bool(differs.any()):
            excess = diff[differs] / allowed[differs]   # allowed > 0 where a != b: one of the two is non-zero
            outside += int((excess > 1).sum())
            max_excess = max(max_excess, float(excess.max()))
            differing += int(differs.sum())
        del diff, differs, allowed
    largest = None if largest_abs is None else (largest_abs / m if m > 0 else math.inf)
    out.update({"median_abs": m, "step_at_median": s_m, "max_abs_difference": mx,
                "max_abs_difference_in_median_steps": (mx / s_m) if s_m else (0.0 if mx == 0 else math.inf),
                "differing": differing, "largest_abs_over_median": largest,
                "outside": outside, "max_excess": max_excess})
    return out


def merge_counts(rows):
    """Totals over tensors: sums of the counts and the largest of the maxima (median_abs and step_at_median are
    per tensor and not totalled)."""
    out = {"elements": 0, "equal": 0, "one_step": 0, "more_than_one": 0, "max_steps": 0, "differing": 0,
           "outside": 0, "max_abs_difference": 0.0, "max_abs_difference_in_median_steps": 0.0, "max_excess": 0.0,
           "largest_abs_over_median": None}
    for r in rows:
        for k in ("elements", "equal", "one_step", "more_than_one", "differing", "outside"):
            out[k] += r[k]
        for k in ("max_steps", "max_abs_difference", "max_abs_difference_in_median_steps", "max_excess"):
            out[k] = max(out[k], r[k])
        if r["largest_abs_over_median"] is not None:
            out["largest_abs_over_median"] = max(out["largest_abs_over_median"] or 0.0, r["largest_abs_over_median"])
    return out


def one_step_ok(counts, fraction=ONE_STEP_FRACTION, allowed_steps=1):
    """B-S4's rounded rule on one tensor's row of rounded_sums: (a) every element |a - b| <= allowed_steps *
    max(sp(a), sp(b), s_m), its own bfloat16 step or, below the median magnitude, one step at the median (max_excess
    <= allowed_steps), and (b) at most ``fraction`` of the elements with a != b. An all-zero b (no s_m) passes only
    when a == b. Corrected twice before being applied (see rounded_sums): first for elements near zero, then for
    elements above the median."""
    if counts["step_at_median"] is None:
        return counts["differing"] == 0
    return counts["max_excess"] <= allowed_steps and counts["differing"] <= fraction * counts["elements"]


# ------------------------------------------------------------------ row_normalization "full": Heretic's randomized rank r
REPLICA_STAGE = "B-S4 replica of Heretic's randomized approximation"
REPLICA_INPUT = "B-S4 replica input is the IDE's change"
REPLICA_INPUT_BOUND = 1e-6      # the exact rank-r truncation of D against the IDE's delta (the script's own premise)


def heretic_lowrank(D, r, seed, device="cpu"):
    """Heretic's rank-r approximation of a full change D [d_out, d_in] (src/heretic/modifiers/abliteration.py at
    64e9991, lines 421-452, AGPL-3.0, copied for this comparison): reseed, torch.svd_lowrank(q = 2r + 4, niter = 6),
    split the singular values evenly. On ``device`` (default cpu: D as given, today's arithmetic): Heretic runs these
    lines on the device of the module's weight, where D was formed, so D is moved there first and the seed call
    comes immediately before svd_lowrank, which draws its test matrix from that device's generator. The factors are
    returned on the CPU (moving them is exact). -> (lora_B [d_out, r], lora_A [r, d_in])."""
    import torch
    D = D.to(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    U, S, V = torch.svd_lowrank(D, q=2 * r + 4, niter=6)
    U = U[:, :r]
    S = S[:r]
    Vh = V[:, :r].T
    sqrt_S = torch.sqrt(S)
    return (U @ torch.diag(sqrt_S)).cpu(), (torch.diag(sqrt_S) @ Vh).cpu()


def full_change(W, v, w):
    """The full change D [d_out, d_in] (float64) of row_normalization "full" for weight W [d_out, d_in] (the
    residual stream first, as stored), unit direction v [d_out] and strength w: the IDE's closed form of
    wsp.directions.change, D = diag(g) W - u (W^T x)^T with x = v / n, n the row norms, which is Heretic's
    normalise, project, renormalise, restore the row norms, subtract W, written out. The IDE never forms D (it
    truncates through D's Gram matrix); its delta is D's best rank-r approximation."""
    import torch
    W64, v64 = W.double(), v.double()
    nrm = W64.norm(dim=1)
    ok = nrm > 0
    one = torch.ones_like(nrm)
    x = torch.where(ok, v64 / torch.where(ok, nrm, one), torch.zeros_like(nrm))
    a = W64.T @ x                                  # V x in the IDE's layout
    t = W64 @ a                                    # G x
    alpha = float(x @ t)
    c = torch.where(ok, t / torch.where(ok, nrm, one), torch.zeros_like(t))
    s = torch.sqrt(torch.clamp(1 - 2 * w * v64 * c + w * w * v64 * v64 * alpha, min=1e-24))
    g = torch.where(ok, 1 / s - 1, torch.zeros_like(s))
    u = torch.where(ok, nrm * w * v64 / s, torch.zeros_like(s))
    return g[:, None] * W64 - u[:, None] * a[None, :]


def heretic_full_change(W, v, w, device="cpu"):
    """D as Heretic's own float32 lines form it (abliteration.py at 64e9991, the "full" branch before the
    low-rank call): normalise the rows, add -w v (v^T W), normalise, restore the row norms, subtract W. On
    ``device`` (default cpu, today's arithmetic), as Heretic does on the module weight's device: v (float32) is moved
    there, W is the float32 weight there, and every step of D runs there. D is returned on the CPU (exact)."""
    import torch
    import torch.nn.functional as F
    W = W.to(device=device, dtype=torch.float32)
    v = v.to(device=device, dtype=torch.float32)
    W_org = W
    W_row_norms = torch.linalg.vector_norm(W, dim=1, keepdim=True)
    W = F.normalize(W, p=2, dim=1)
    lora_A = (v @ W).view(1, -1)
    lora_B = (-w * v).view(-1, 1)
    W = W + lora_B @ lora_A
    W = F.normalize(W, p=2, dim=1)
    W = W * W_row_norms
    return (W - W_org).cpu()


def exact_lowrank(D, r):
    """D's best rank-r approximation, exactly (float64, through the Gram matrix of D's smaller side, as the IDE
    truncates), and all of D's singular values in descending order (torch.linalg.svdvals, accurate also far below
    sigma_1, where the Gram matrix's eigenvalues are not). -> (D_r float64, sigma float64)."""
    import torch
    D = D.double()
    if D.shape[0] <= D.shape[1]:
        _, U = torch.linalg.eigh(D @ D.T)
        Ur = U[:, -r:]
        Dr = Ur @ (Ur.T @ D)
    else:
        _, V = torch.linalg.eigh(D.T @ D)
        Vr = V[:, -r:]
        Dr = (D @ Vr) @ Vr.T
    return Dr, torch.linalg.svdvals(D)


class _TensorView:
    """The few members of a workspace view that wsp.directions.change reads, over a tensor in the IDE's layout
    ([d_in, d], the residual stream last), so the product's own function runs on tensors read from files."""

    def __init__(self, t, name=""):
        self.t, self.name = t, name
        self.shape, self.size = tuple(t.shape), t.numel()

    def sub(self, idx):
        return _TensorView(self.t[idx], self.name)

    def read(self, variant="current", extra=None):
        return self.t

    def contract(self, M, axis=-1, variant="current", extra=None):
        import torch
        return torch.tensordot(self.t.movedim(axis, -1), M, dims=1)


def ide_full_delta(W, v, w, r, name=""):
    """The IDE's delta for row_normalization "full" (wsp.directions.change, the product's function, on W read from
    a file; no workspace, no runner, no model) in W's own layout [d_out, d_in], float32."""
    import types
    import torch
    from wsp import directions as DR
    view = _TensorView(W.to(torch.float32).T.contiguous(), name)
    pieces, _ = DR.change(types.SimpleNamespace(), [view], v.to(torch.float32), w, "full", r)
    if not pieces:
        return torch.zeros(W.shape, dtype=torch.float32)
    (_, A, B), = pieces
    return (A @ B).T.contiguous()


def replica_row(W, v, w, r, seed, heretic_BA, ide_delta, device="cpu"):
    """One tensor of the replica stage: D from the IDE's closed form, its exact rank-r truncation against the IDE's
    delta (the premise: the replica's input is the IDE's change), Heretic's randomized approximation R of D with
    the run's seed and rank, and the distances between R, Heretic's stored B A and the IDE's delta. ``device`` is
    where Heretic's lines run (heretic_lowrank, heretic_full_change): the closed form and its truncation are float64
    on the CPU, R's input is that D in float32 moved to the device; every distance is computed on the CPU."""
    import torch
    D64 = full_change(W, v, w)
    Dr, sig = exact_lowrank(D64, r)
    lb, la = heretic_lowrank(D64.to(torch.float32), r, seed, device)
    R = lb @ la
    Dh = heretic_full_change(W, v, w, device)
    hb, ha = heretic_lowrank(Dh, r, seed, device)
    Rh = hb @ ha
    k = len(sig)
    row = {"shape": list(W.shape), "rank": r, "weight": w,
           "exact_truncation_vs_ide": rel(Dr, ide_delta),
           "replica_vs_heretic": rel(R, heretic_BA), "replica_vs_ide": rel(R, ide_delta),
           "ide_vs_heretic": rel(ide_delta, heretic_BA), "replica_bit_equal_heretic": bool(torch.equal(R, heretic_BA)),
           "tail": float(sig[r] / sig[r - 1]) if k > r and sig[r - 1] > 0 else None,
           "sigma_r_over_sigma_1": float(sig[r - 1] / sig[0]) if k >= r and sig[0] > 0 else None,
           "heretic_formula_vs_ide_formula": rel(Dh, D64),
           "replica_of_heretic_formula_vs_heretic": rel(Rh, heretic_BA)}
    del D64, Dr, R, Dh, Rh
    return row


def adapter_module_of(adapter, cfg, module, tensor, names):
    """Heretic's adapter module for a dump module name: the name itself when the adapter has it, else the adapter
    module whose checkpoint tensor is ``tensor``. -> name or None."""
    if module in adapter.modules:
        return module
    hits = [m for m in adapter.modules if tensor_of_module(cfg, m, names) == tensor]
    return hits[0] if len(hits) == 1 else None


def replica_rows(base, cfg, dump, good_mean, bad_mean, adapter, ide_delta, dense_tensor=None, device="cpu"):
    """The replica stage over every module Heretic's exported trial changed. ``base`` a Checkpoint, ``cfg`` the
    model's config.json, ``dump`` heretic_dump.py's dump.json with Heretic's means, ``adapter`` an Adapter,
    ``ide_delta(tensor, W, v, w, r)`` the IDE's delta of that tensor in W's layout, ``device`` where Heretic's lines
    are replicated (replica_row). Modules by the dump's names (dump.json "modules", through wsp.runtime_layouts),
    else model.layers.<l> o_proj and down_proj."""
    from wsp import directions as DR
    settings = dump["modifier_settings"]["Abliteration"]
    trial = next(t for t in dump["trials"] if t["index"] == dump["exported_trial_index"])
    P = trial["parameters"]
    r, seed = int(adapter.config["r"]), int(dump["seed"])
    tab = DR.table(good_mean, bad_mean, settings.get("orthogonalize_direction", True))
    mods = dump_modules(dump)
    dense = dense_tensor or {"attn.o_proj": "model.layers.{}.self_attn.o_proj", "mlp.down_proj": "model.layers.{}.mlp.down_proj"}
    names = set(base.keys())
    rows = []
    for layer in range(int(dump["layers"])):
        for comp, dist in P["weight_distributions"].items():
            w, _ = DR.layer_weight(dist, layer)
            if w is None:
                continue
            v = DR.direction_at(tab, P.get("direction_index"), layer)
            modules = mods[layer].get(comp, []) if mods is not None else [dense[comp].format(layer)]
            for i, module in enumerate(modules):
                tensor = tensor_of_module(cfg, module, names)
                row = {"layer": layer, "component": comp, "module_index": i, "module": module, "tensor": tensor}
                hm = adapter_module_of(adapter, cfg, module, tensor, names) if tensor else None
                if tensor is None or hm is None:
                    rows.append({**row, "error": "no checkpoint tensor" if tensor is None else "not in Heretic's adapter"})
                    continue
                W = base.get_tensor(tensor).float()
                if W.dim() != 2 or W.shape[0] != v.shape[0]:
                    rows.append({**row, "error": f"weight {list(W.shape)} is not [d_model, d_in]"})
                    continue
                rows.append({**row, **replica_row(W, v, w, r, seed, adapter.delta(hm), ide_delta(tensor, W, v, w, r),
                                                  device)})
                del W
    return rows


def replica_check(rep, label, rows, bound, this_torch, run_versions, device="cpu", run_device=None):
    """The replica stage's checks: its premise (every tensor's exact truncation of D within REPLICA_INPUT_BOUND of
    the IDE's delta, always asserted) and B-S4's replica bound (``bound``, from the bounds table by the run's
    dtype; asserted only under the run's own torch with the replica on the run's device, replica_applies; else
    reported with the reason). ``device`` the device the replica ran on, ``run_device`` dump.json "device".
    -> the numbers."""
    good = [r for r in rows if "error" not in r]
    worst = (lambda k: max((r[k] for r in good if r[k] is not None), default=None))
    numbers = {"tensors": len(rows), "errors": [{k: r[k] for k in ("module", "tensor", "error")} for r in rows
                                                if "error" in r][:5],
               "worst_replica_vs_heretic": worst("replica_vs_heretic"), "worst_replica_vs_ide": worst("replica_vs_ide"),
               "worst_ide_vs_heretic": worst("ide_vs_heretic"), "worst_tail": worst("tail"),
               "smallest_sigma_r_over_sigma_1": min((r["sigma_r_over_sigma_1"] for r in good
                                                     if r["sigma_r_over_sigma_1"] is not None), default=None),
               "worst_exact_truncation_vs_ide": worst("exact_truncation_vs_ide"),
               "worst_replica_of_heretic_formula_vs_heretic": worst("replica_of_heretic_formula_vs_heretic"),
               "replicas_bit_equal": sum(r["replica_bit_equal_heretic"] for r in good),
               "torch": this_torch, "run_torch": (run_versions or {}).get("torch") if isinstance(run_versions, dict) else None,
               "device": device_kind(device), "run_device": device_kind(run_device)}
    premise = bool(good) and len(good) == len(rows) and numbers["worst_exact_truncation_vs_ide"] <= REPLICA_INPUT_BOUND
    rep.check(label, REPLICA_INPUT, premise, bound=REPLICA_INPUT_BOUND, tensors=len(rows),
              worst=numbers["worst_exact_truncation_vs_ide"], errors=numbers["errors"])
    if bound is None:
        rep.note(label, f"{REPLICA_STAGE} (no bound for this dtype)", numbers)
        return numbers
    applies, why = replica_applies(this_torch, run_versions, device, run_device)
    if not applies:
        rep.note(label, REPLICA_STAGE, {"note": f"reported only: {why}", **numbers})
        print(f"     {label} {REPLICA_STAGE} (reported) reported only: {why}; replica on {numbers['device']}, "
              f"run on {numbers['run_device']}; worst replica_vs_heretic "
              f"{numbers['worst_replica_vs_heretic']}, replica_vs_ide {numbers['worst_replica_vs_ide']}, "
              f"ide_vs_heretic {numbers['worst_ide_vs_heretic']}", flush=True)
        return numbers
    ok = premise and numbers["worst_replica_vs_heretic"] is not None and \
        numbers["worst_replica_vs_heretic"] <= bound["replica_vs_heretic"]
    rep.check(label, REPLICA_STAGE, ok, bound=bound, **numbers)
    return numbers


# ------------------------------------------------------------------ the merged model of an adapter, as PEFT makes it
_ST_DTYPES = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16", "I64": "int64", "I32": "int32",
              "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool"}


def _st_name(torch_dtype):
    rev = {v: k for k, v in _ST_DTYPES.items()}
    return rev[str(torch_dtype).replace("torch.", "")]


def _write_streamed(path, names, get, metadata):
    """Write a safetensors file tensor by tensor (header first, then each tensor's bytes); ``get(name)`` returns a
    contiguous CPU tensor. -> {name: SHA-256 of its bytes}."""
    import torch
    header, offset, shapes = {}, 0, {}
    for name in names:
        t = get(name, meta_only=True)
        n = math.prod(t["shape"]) * t["itemsize"]
        header[name] = {"dtype": t["st"], "shape": t["shape"], "data_offsets": [offset, offset + n]}
        shapes[name] = n
        offset += n
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    digests = {}
    tmp = Path(str(path) + ".part")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name in names:
            t = get(name).contiguous()
            raw = t.view(torch.uint8).numpy().tobytes() if t.numel() else b""
            assert len(raw) == shapes[name], (name, len(raw), shapes[name])
            digests[name] = hashlib.sha256(raw).hexdigest()
            f.write(raw)
            del t, raw
    os.replace(tmp, path)
    return digests


def merge_adapter(base_dir, adapter_dir, out_dir, log=print, as_dtype=None):
    """Materialise PEFT's merge of a Heretic adapter into ``out_dir`` (see the module docstring) -> merge.json's
    content. Each changed tensor is merged in the checkpoint's own dtype (the run's dtype for a bfloat16 run of a
    bfloat16 checkpoint), or first converted to ``as_dtype`` ("float32" for a run Heretic made in float32, which
    merges into the upcast weights and saves float32). An ``out_dir`` that already holds a merge.json for the same
    base, adapter, adapter SHA-256 and dtype is reused after its written tensors are hashed again; anything else in
    it is refused (a fresh directory is expected)."""
    import torch
    base_dir, adapter_dir, out_dir = Path(base_dir).resolve(), Path(adapter_dir).resolve(), Path(out_dir)
    ad = Adapter(adapter_dir)
    adapter_sha = hashlib.sha256((adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest()
    ck = Checkpoint(base_dir)
    cfg = json.loads((base_dir / "config.json").read_text()) if (base_dir / "config.json").is_file() else {}
    names = set(ck.keys())
    targets, unmatched = {}, []
    for m in ad.modules:
        t = tensor_of_module(cfg, m, names)
        if t is None:
            unmatched.append(m)
        else:
            targets[t] = m
    if unmatched:
        raise AdapterRefused(f"adapter modules with no checkpoint tensor: {unmatched[:5]}")
    done = out_dir / "merge.json"
    if done.is_file():
        old = json.loads(done.read_text())
        if (old.get("base"), old.get("adapter"), old.get("adapter_sha256"), old.get("as_dtype")) != \
                (str(base_dir), str(adapter_dir), adapter_sha, as_dtype):
            raise SystemExit(f"{out_dir} holds another merge; give a fresh directory")
        again = Checkpoint(out_dir)
        for name, digest in old["sha256"].items():
            t = again.get_tensor(name).contiguous()
            if hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() != digest:
                raise SystemExit(f"{out_dir}: {name} no longer has its recorded SHA-256; give a fresh directory")
        log(f"     merged model reused: {out_dir}")
        return old
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} exists and is not empty; give a fresh directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    changed = {}
    files = {}
    for t, f in ck.where.items():
        files.setdefault(f, []).append(t)
    digests, written = {}, []
    for f, tensors in sorted(files.items()):
        if not any(t in targets for t in tensors):
            continue
        meta = ck.files[f].metadata()

        def get(name, meta_only=False):
            sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
            if meta_only:
                sl = ck.get_slice(name)
                st = sl.get_dtype()
                if name in targets and as_dtype:
                    st = _st_name(getattr(torch, as_dtype))
                return {"shape": list(sl.get_shape()), "st": st, "itemsize": sizes[st]}
            w = ck.get_tensor(name)
            if name in targets:
                delta = ad.delta(targets[name])
                if tuple(delta.shape) != tuple(w.shape):
                    raise AdapterRefused(f"{targets[name]}: B @ A is {list(delta.shape)}, {name} is {list(w.shape)}")
                w = w.to(getattr(torch, as_dtype)) if as_dtype else w.clone()
                w += delta               # PEFT's merge: base_layer.weight.data += delta_weight (float32 delta)
                changed[name] = {"module": targets[name], "dtype": str(w.dtype)}
            return w
        log(f"     writing {f} ({sum(t in targets for t in tensors)} of {len(tensors)} tensors changed)")
        digests.update(_write_streamed(out_dir / f, sorted(tensors), get, meta))
        written.append(f)
    linked = []
    for p in sorted(base_dir.iterdir()):
        if p.name in written or not p.is_file():
            continue
        os.symlink(p.resolve(), out_dir / p.name)
        linked.append(p.name)
    info = {"base": str(base_dir), "adapter": str(adapter_dir), "adapter_sha256": adapter_sha, "as_dtype": as_dtype,
            "adapter_config": {k: ad.config.get(k) for k in ("r", "lora_alpha", "target_modules")},
            "rule": "weight.data += lora_B @ lora_A (float32, scaling 1), PEFT's merge (lora/layer.py)",
            "changed": changed, "written_files": written, "linked_files": linked, "sha256": digests}
    write_json(done, info)
    return info


# ------------------------------------------------------------------ guard logs and Heretic logs
def guard_log(path):
    """The memory guard's JSON lines -> {preflight, samples, teardown, peak_footprint_bytes, peak_rss_bytes}."""
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    samples = [r for r in rows if r.get("event") == "sample"]
    return {"preflight": next((r for r in rows if r.get("event") == "preflight"), None),
            "samples": len(samples), "teardown": next((r for r in reversed(rows) if r.get("event") == "teardown"), None),
            "peak_footprint_bytes": max((r.get("tree_footprint", 0) for r in samples), default=None),
            "peak_rss_bytes": max((r.get("tree_rss", 0) for r in samples), default=None)}


def parse_duration(text):
    """Heretic's format_duration ("42s", "1m 31s", "2h 3m 4s") -> seconds."""
    total, found = 0.0, False
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", text):
        total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
        found = True
    return total if found else None


STAMP = re.compile(r"^(\d+\.\d+)\t(.*)$")


def heretic_log(path):
    """A Heretic run.log -> [(seconds since the first line or None, line)], plain or stamped by formats_timing.py
    --stage stamp ("<seconds>\\t<line>")."""
    out = []
    for raw in Path(path).read_text(errors="replace").splitlines():
        m = STAMP.match(raw)
        out.append((float(m.group(1)), m.group(2)) if m else (None, raw))
    return out


# ------------------------------------------------------------------ the command
def main(argv):
    import argparse
    p = argparse.ArgumentParser(prog="formats_shared.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("merge", help="materialise PEFT's merge of a Heretic adapter (model-free)")
    m.add_argument("base", help="the base model directory (config.json and safetensors weights)")
    m.add_argument("adapter", help="the Heretic adapter directory (adapter_config.json, adapter_model.safetensors)")
    m.add_argument("out", help="a fresh output directory")
    m.add_argument("--as-dtype", choices=DTYPES, default=None,
                   help="merge into the weights converted to this dtype (default: the checkpoint's own dtype)")
    a = p.parse_args(argv)
    sys.path.insert(0, os.getcwd())
    info = merge_adapter(a.base, a.adapter, a.out, as_dtype=a.as_dtype)
    print(f"merged: {len(info['changed'])} tensors changed, {len(info['written_files'])} files written, "
          f"{len(info['linked_files'])} linked -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
