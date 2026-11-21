"""Compare the IDE's PEFT LoRA adapter export (wsp.adapter_export) with Heretic's adapter export and with PEFT itself.

Three stages, each run by the orchestrator under the memory guard, one process at a time, with the isolated
Hugging Face cache (HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=<isolated hf-home>),
from the IDE's `weight-space-ide` directory:

    python -m wsp.memory_guard --log <log> --forecast-gib <g> -- <IDE python> scripts/heretic_compare_adapter.py \\
        --stage ide <model snapshot dir> <isolated IDE root> <out dir> <label>=<new run dir>,<original run dir> ...
    python -m wsp.memory_guard --log <log> --forecast-gib <g> -- <Heretic python> scripts/heretic_compare_adapter.py \\
        --stage ide-captures <model snapshot dir> <isolated IDE root> <out dir> <ARA label>=<new run dir>,<original run dir> ...
    python -m wsp.memory_guard --log <log> --forecast-gib <g> -- <Heretic python> scripts/heretic_compare_adapter.py \\
        --stage peft <model snapshot dir> <out dir> <label>=<new run dir>,<original run dir> ...

A label names one Heretic run twice: the new run (export_strategy "adapter", save_directory "adapter"; its
adapter/ is read) and the original run of the same config (its dump/, written by heretic_dump.py, and its merged
model/ are read). Before anything else, every stage asserts without a model that the two journals hold the same
parameter draws, trial values, trial parameters and scores (J; else the label stops: the oracle moved).

Stage ide (the IDE's interpreter; loads the model): per label, in its own isolated root <root>/<label>, the model
is measured as the run measured it: chat, the run's system prompt and response prefix, the run's good and bad
prompt sets given as their dataset specifications from the dump's recorded modifier settings (resolved offline from
the cache; the resolved prompts must equal the dump's, M0; if the datasets cannot be resolved the dump's texts are
measured instead and the record is then not Heretic-runnable), ARA with module_io; then the dump's exported trial is
applied through Session.apply (ARA with the run's settings and seed), the patch alone is exported as the adapter
`ide-<label>` (layout auto), and <out>/<label>/ holds ide_current.safetensors (ws.read(name, (), "current") in float32
of every touched tensor), ide.json (patch id, digest, record, the export's answer) and, when the record is
Heretic-runnable, ide_reproduce.json and ide_reproduce_adapter.json (to_heretic with the run's model and
model_commit, and with the adapter; what GET /api/patches/{pid}/record/heretic answers).

Stage ide-captures (Heretic's interpreter, ARA labels; loads the model): the same in <root>/<label>-captures, but the
fit runs on Heretic's captures (dump/dump.safetensors io.*), written into the measurement through the code path of
scripts/heretic_compare_ara.py ("IDE from Heretic's captures": scripts/heretic_captures.py), so the fit runs under
Heretic's libraries on Heretic's inputs; the adapter is `ide-captures-<label>`, the files go to <out>/<label>/captures/.

Stage peft (Heretic's interpreter; loads the model, never two at once): the base is loaded with
AutoModelForCausalLM.from_pretrained(<snapshot>, dtype=torch.float32), a fresh copy per check; base, merged model and
ide_current files are read one tensor at a time.

    P0  Heretic's new adapter: r, lora_alpha == r, tensors float32, metadata {"format": "pt"}, the modules its tensors
        name are every self_attn.o_proj and mlp.down_proj of every layer                               exact
    P3  Heretic's adapter merged into a fresh base equals the original run's merged model, every
        tensor of the merged file (the oracle did not move; else the label stops)                      bit-equal
    P1  PeftModel.from_pretrained(base, IDE dir) raises and warns nothing; the modules carrying the
        adapter are the file's target_modules (as a set); each one's r and scaling are the file's
        (rank_pattern and alpha_pattern resolved by PEFT's get_pattern_key); peft_config["default"]
        has the file's r, lora_alpha, rank_pattern and alpha_pattern                                   exact
    P2  set(load_peft_weights(IDE dir)) == set(get_peft_model_state_dict(model)); every value float32   exact
    P5  first-token logits on the dump's KL prompts (Heretic's token ids): the IDE adapter attached
        (unmerged) vs a fresh base with ide_current loaded          max abs <= 1e-5 * max(1, max |logit|)
        (an attached adapter computes W x + B (A x), merged weights (W + B A) x: float32 rounds the two
        differently, in proportion to the logits; reported beside it, the same difference for Heretic's
        own adapter, attached against merged, on the same prompts)
    P4  merge_and_unload() of the IDE adapter, every touched tensor vs ide_current                     bit-equal; else
        relative change difference <= 1e-6 and max abs <= 1e-6 * max |delta| (passes, "not bit-equal")
    P4b every other tensor of the checkpoint file bit-equal to the base                               exact
    P6  ablation: per module of Heretic's adapter, lora_B @ lora_A * (alpha / r) against the IDE's
        product (zero when the module is absent from the IDE adapter; Heretic's product must then be
        exactly 0)                         relative <= 5e-5 (tiny fixture), <= 1e-3 (qwen05b, row_normalization full)
    P6b ARA from Heretic's captures (stage ide-captures): every module the IDE adapter has, lora_A and
        lora_B equal Heretic's; every module it lacks has lora_A == 0 or lora_B == 0 in Heretic's        bit-equal
    P6c ARA from each tool's own captures and libraries: per module relative difference of the products  reported

warnings.catch_warnings(record=True) with simplefilter("always") is around every PEFT call, and a recorded warning
fails P1 / P3 / P4 (they are printed). Every assertion prints a PASS or FAIL line with its numbers; <out dir>/
<stage>.json holds every number and a pass flag per assertion; the exit status is 1 when one fails.
"""
import gc
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.getcwd())

P6_BOUNDS = {"qwen05b": 1e-3}          # by the original run's model directory; else the fixture's bound
P6_BOUND = 5e-5
P5_RELATIVE = 1e-5                     # of max(1, the largest |logit|)
P4_RELATIVE = 1e-6
MODULES = re.compile(r"model\.layers\.\d+\.(self_attn\.o_proj|mlp\.down_proj)")
PREFIX, SUFFIXES = "base_model.model.", (".lora_A.weight", ".lora_B.weight")


# ------------------------------------------------------------------ shared
class Report:
    def __init__(self):
        self.labels = {}

    def check(self, label, name, ok, **numbers):
        self.labels.setdefault(label, {})[name] = {"pass": bool(ok), **numbers}
        shown = {k: v for k, v in numbers.items() if not isinstance(v, (list, dict)) or len(json.dumps(v, default=str)) < 300}
        print(f"{'PASS' if ok else 'FAIL'} {label} {name} {json.dumps(shown, default=str)}", flush=True)
        return bool(ok)

    def note(self, label, name, value):
        self.labels.setdefault(label, {})[name] = value

    def ok(self):
        return all(v["pass"] for e in self.labels.values() for v in e.values() if isinstance(v, dict) and "pass" in v)


def parse_labels(args) -> dict:
    out = {}
    for a in args:
        label, _, dirs = a.partition("=")
        new, _, orig = dirs.partition(",")
        if not label or not new or not orig:
            raise SystemExit(f"{a}: a label is <label>=<new run dir>,<original run dir>")
        out[label] = (Path(new), Path(orig))
    return out


def _journal_rows(path):
    """What two runs of one config must share, from an Optuna journal read as JSON lines (not opened with Optuna, so
    the files are never locked or written): per trial its parameters as drawn, its final state and values, and its
    Heretic user attributes "parameters" and "scores"."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "param_name" in r:
            rows.append(["param", r.get("trial_id"), r["param_name"], r.get("param_value_internal"), r.get("distribution")])
        elif "values" in r and "trial_id" in r:
            rows.append(["values", r["trial_id"], r.get("state"), r.get("values")])
        elif "trial_id" in r and isinstance(r.get("user_attr"), dict):
            kept = {k: v for k, v in r["user_attr"].items() if k in ("parameters", "scores", "index")}
            if kept:
                rows.append(["user_attr", r["trial_id"], kept])
    return rows


def journals_equal(rep, label, new, orig) -> bool:
    files = [sorted((d / "checkpoints").glob("*.jsonl")) for d in (new, orig)]
    if any(len(f) != 1 for f in files):
        return rep.check(label, "J journals equal", False, journals=[len(f) for f in files])
    a, b = _journal_rows(files[0][0]), _journal_rows(files[1][0])
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return rep.check(label, "J journals equal", a == b, rows=[len(a), len(b)], first_difference=first)


def rel(a, b):
    d = float((a.double() - b.double()).norm())
    n = float(b.double().norm())
    return d / n if n > 0 else (0.0 if d == 0 else math.inf)


def write(out, name, data):
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(data, indent=1, default=str))


# ------------------------------------------------------------------ stages ide and ide-captures (the IDE's code)
def measure_as_the_run(ws, dump, ara, rep, label):
    """Measure "oracle" as the run did: its dataset specifications when they resolve, else the dump's texts."""
    from wsp import directions as DR
    from wsp import directions_datasets as DD
    system, prefix = dump["system_prompt"], dump["response_prefix"] or None
    cls = "ARA" if ara else "Abliteration"
    table = (dump.get("modifier_settings") or {}).get(cls) or {}
    extra = {} if ara else {"winsorization_quantile": table.get("winsorization_quantile", 1.0)}
    texts = {k: [p["user"] for p in v] for k, v in dump["prompts"].items()}
    source, error = "datasets", None
    try:
        DR.measure(ws, "oracle", table["good_prompts"], table["bad_prompts"], chat=True, system=system, prefix=prefix,
                   variant="base", replace=True, module_io=ara, **extra)
    except (KeyError, DR.DirectionError) as e:
        source, error = "texts", f"{type(e).__name__}: {e}"
        print(f"     {label}: the dataset specifications did not resolve ({error}); measuring the dump's texts, so "
              f"this patch's record is not Heretic-runnable", flush=True)
        DR.measure(ws, "oracle", texts["good"], texts["bad"], chat=True, system=system, prefix=prefix, variant="base",
                   replace=True, module_io=ara, **extra)
    got = DR.prompts(ws, "oracle")
    same = all([DD.user_of(p) for p in got[k]] == texts[k] and
               all(DD.system_of(p, system) == q["system"] for p, q in zip(got[k], dump["prompts"][k]))
               for k in ("good", "bad"))
    rep.check(label, "M0 measured prompts are the dump's", same, source=source, error=error,
              counts=[len(got["good"]), len(got["bad"])])
    return source


def commit_of(dump, snap) -> str | None:
    """The model commit the run loaded: the dump's, or, for a run that pinned none, the name of the cache snapshot
    directory the model is read from here (.../snapshots/<commit>), which is what an offline run resolves to."""
    if dump.get("model_commit"):
        return dump["model_commit"]
    snap = Path(snap)
    return snap.name if snap.parent.name == "snapshots" else None


def modifier_of(dump) -> str:
    """The run's modifier class: the dump's "modifier", or, for a dump made before that key existed, the one key
    of its modifier_settings (a Heretic run has exactly one modifier)."""
    if dump.get("modifier"):
        return dump["modifier"]
    (cls,) = dump["modifier_settings"]
    return cls


def params_of(dump, ara):
    trial = next(t for t in dump["trials"] if t["index"] == dump["exported_trial_index"])
    if ara:
        conf = (dump.get("modifier_settings") or {}).get("ARA") or {}
        defaults = {"preserve_row_magnitudes": True, "lora_rank": 50, "n_optimization_steps": 5, "learning_rate": 1.0,
                    "max_iter": 20, "history_size": 10}
        return {"method": "ara", "direction": "oracle", **trial["parameters"],
                **{k: conf.get(k, v) for k, v in defaults.items()}, "seed": dump["seed"]}, trial
    conf = (dump.get("modifier_settings") or {}).get("Abliteration") or {}
    return {"direction": "oracle", **trial["parameters"], "orthogonalize_direction": conf.get("orthogonalize_direction", True),
            "row_normalization": conf.get("row_normalization", "full"),
            "full_normalization_lora_rank": conf.get("full_normalization_lora_rank", 3)}, trial


def stage_ide(snap, root, out, labels, captures):
    import torch
    from safetensors.torch import load_file, save_file
    from wsp import adapter_export as AX
    from wsp import directions as DR
    from wsp import directions_record as REC
    from wsp import directions_search as DSR
    from wsp.session import Session
    rep = Report()
    stage = "ide-captures" if captures else "ide"
    for label, (new, orig) in labels.items():
        if not journals_equal(rep, label, new, orig):
            continue
        dump = json.loads((orig / "dump" / "dump.json").read_text())
        ara = modifier_of(dump) == "ARA"
        if captures and not ara:
            rep.note(label, "skipped", "not an ARA run")
            continue
        here = out / label / ("captures" if captures else "")
        sess = Session(str(root / (f"{label}-captures" if captures else label)))
        os.makedirs(sess.root, exist_ok=True)
        ws = sess.open(str(snap), log=lambda *a, **k: None)
        try:
            for p in list(ws.patches):                     # the edit is made on the base (Heretic's H3)
                if p.enabled:
                    ws.set_patch(p.id, enabled=False)
            source = measure_as_the_run(ws, dump, ara, rep, label)
            if captures:
                import heretic_captures as HC
                n, d = ws.struct.n_layers, ws.struct.d_model
                meta, _, _ = DR.load(ws, "oracle")
                info = DR.load_io(ws, "oracle", meta)
                tens = load_file(str(orig / "dump" / "dump.safetensors"))
                rows, heretic_io = HC.heretic_io_of(ws, "oracle", info, tens, n, d)
                HC.use_heretic_io(ws, "oracle", info, heretic_io, n, d)
                rep.note(label, "module_io", rows)
                del tens
            params, trial = params_of(dump, ara)
            rep_ = sess.apply(DR.script_for(params), message=f"heretic adapter comparison ({label})", dry_run=False)
            if not rep.check(label, "I1 committed through Session.apply", rep_.get("ok") and rep_.get("committed"),
                             error=rep_.get("error"), record=rep_.get("record"), record_error=rep_.get("record_error")):
                continue
            pid = rep_["patch"]
            name = f"ide-captures-{label}" if captures else f"ide-{label}"
            version = ws.version
            ans = AX.export(ws, name, patches=[pid], layout="auto", replace=True)
            rep.check(label, "I2 the export changes nothing in the workspace", ws.version == version)
            touched = [m["tensor"] for m in ans["modules"]]
            here.mkdir(parents=True, exist_ok=True)
            save_file({t: ws.read(t, (), "current").float().contiguous() for t in touched},
                      str(here / "ide_current.safetensors"))
            rec, stale = REC.read(ws, ws.patch(pid))
            # a snapshot opened by path has no Hub id: the run's model is the query's, as the route would get it
            reasons = REC.heretic_reasons(ws, rec, pid, model=dump["model"]) if rec else ["no record"]
            status = {"runnable": not reasons, "reasons": reasons}
            info = {"label": label, "stage": stage, "patch": pid, "digest": DSR.piece_digest(ws.patch(pid).deltas),
                    "record": rec, "stale": stale, "heretic": status, "measured_from": source,
                    "adapter_dir": ans["dir"], "adapter": {k: ans[k] for k in ("layout", "r", "rank_pattern", "modules",
                                                                               "notes", "files", "base_model_name_or_path",
                                                                               "family", "validated_family")},
                    "parameters": params, "trial": trial["index"], "method": "ara" if ara else "ablation",
                    "row_normalization": None if ara else params["row_normalization"], "touched": touched,
                    "dtype_of_base": str(ws.store.dtype(touched[0])) if touched else None}
            if rec and status and status["runnable"]:
                for suffix, kw in (("", {}), ("_adapter", {"adapter": name})):
                    data = REC.to_heretic(ws, rec, pid, model=dump["model"], model_commit=commit_of(dump, snap), **kw)
                    (here / f"ide_reproduce{suffix}.json").write_text(json.dumps(data, indent=4))
            write(here, "ide.json", info)
            rep.note(label, "ide", {k: info[k] for k in ("patch", "digest", "measured_from", "adapter_dir", "heretic")})
            print(f"     {label}: patch {pid}, {len(touched)} tensors, layout {ans['layout']} r {ans['r']}, record "
                  f"{'Heretic-runnable' if status and status['runnable'] else status}", flush=True)
        finally:
            sess.unload()
            del ws
            gc.collect()
    return rep, stage


# ------------------------------------------------------------------ stage peft (PEFT and transformers only)
def modules_of(keys):
    return sorted({k[len(PREFIX):-len(s)] for k in keys for s in SUFFIXES if k.startswith(PREFIX) and k.endswith(s)})


def factors(tensors, m):
    return tensors[f"{PREFIX}{m}.lora_B.weight"], tensors[f"{PREFIX}{m}.lora_A.weight"]


def peft_rank(cfg, m):
    from peft.utils.other import get_pattern_key
    r = cfg["rank_pattern"].get(get_pattern_key(list(cfg["rank_pattern"]), m), cfg["r"])
    a = cfg["alpha_pattern"].get(get_pattern_key(list(cfg["alpha_pattern"]), m), cfg["lora_alpha"])
    return r, a


def warned(caught):
    return [f"{w.category.__name__}: {str(w.message)[:300]}" for w in caught]


def stage_peft(snap, out, labels):
    import torch
    from peft import PeftModel
    from peft.utils import get_peft_model_state_dict, load_peft_weights
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    rep = Report()

    def fresh():
        return AutoModelForCausalLM.from_pretrained(str(snap), dtype=torch.float32)

    base_f = safe_open(str(snap / "model.safetensors"), "pt")
    base_keys = set(base_f.keys())
    expected_modules = sorted(k[:-len(".weight")] for k in base_keys if k.endswith(".weight")
                              and MODULES.fullmatch(k[:-len(".weight")]))

    def first_token_logits(model, ids):
        with torch.inference_mode():
            return torch.stack([model(input_ids=torch.tensor([x])).logits[0, -1].float() for x in ids])

    for label, (new, orig) in labels.items():
        if not journals_equal(rep, label, new, orig):
            continue
        ide = json.loads((out / label / "ide.json").read_text())
        ide_dir = Path(ide["adapter_dir"])
        dump = json.loads((orig / "dump" / "dump.json").read_text())
        kl_ids = dump["ids"]["kl"]
        hdir = new / "adapter"
        # ---- P0: Heretic's own adapter file
        hcfg = json.loads((hdir / "adapter_config.json").read_text())
        metadata = safe_open(str(hdir / "adapter_model.safetensors"), "pt").metadata()
        hsd = load_file(str(hdir / "adapter_model.safetensors"))
        hkeys = list(hsd)
        hdtypes = sorted({str(v.dtype) for v in hsd.values()})
        rep.check(label, "P0 Heretic's adapter", hcfg["r"] == hcfg["lora_alpha"] and hdtypes == ["torch.float32"]
                  and metadata == {"format": "pt"} and modules_of(hkeys) == expected_modules,
                  r=hcfg["r"], lora_alpha=hcfg["lora_alpha"], dtypes=hdtypes, metadata=metadata,
                  modules=len(modules_of(hkeys)), expected=len(expected_modules),
                  missing=sorted(set(expected_modules) - set(modules_of(hkeys)))[:5],
                  extra=sorted(set(modules_of(hkeys)) - set(expected_modules))[:5])
        # ---- P3: Heretic's adapter merged equals Heretic's merged model (the oracle did not move)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hmodel = PeftModel.from_pretrained(fresh(), str(hdir))
            h_attached = first_token_logits(hmodel, kl_ids)        # P5's baseline: Heretic's own adapter, attached
            hmodel = hmodel.merge_and_unload()
        h_merged = first_token_logits(hmodel, kl_ids)
        baseline = {"max_abs": float((h_attached - h_merged).abs().max()), "scale": float(h_merged.abs().max())}
        del h_attached, h_merged
        hm = hmodel.state_dict()
        orig_f = safe_open(str(orig / "model" / "model.safetensors"), "pt")
        okeys = sorted(orig_f.keys())
        equal, worst, missing = 0, 0.0, [k for k in okeys if k not in hm]
        for k in okeys:
            if k in hm:
                o = orig_f.get_tensor(k).float()
                equal += bool(torch.equal(hm[k], o))
                worst = max(worst, float((hm[k] - o).abs().max()) if o.numel() else 0.0)
                del o
        p3 = rep.check(label, "P3 Heretic's adapter merged equals its merged model",
                       equal == len(okeys) and not missing, bit_equal=equal, tensors=len(okeys), worst_max_abs=worst,
                       missing=missing[:5], warnings=warned(caught))
        del hmodel, hm, orig_f
        gc.collect()
        if not p3:
            print(f"     {label}: the oracle moved (P3); this label stops here", flush=True)
            continue
        # ---- P1, P2, P5 (unmerged), P4, P4b: the IDE's adapter in PEFT
        cfg = json.loads((ide_dir / "adapter_config.json").read_text())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = PeftModel.from_pretrained(fresh(), str(ide_dir))
                error = None
            except Exception as e:
                model, error = None, f"{type(e).__name__}: {e}"
        if model is None:
            rep.check(label, "P1 PEFT loads the IDE's adapter", False, error=error, warnings=warned(caught))
            continue
        layers = {n[len(PREFIX):]: m for n, m in model.named_modules()
                  if hasattr(m, "lora_A") and hasattr(m.lora_A, "keys") and "default" in m.lora_A}
        ranks = {n: [int(m.r["default"]), float(m.scaling["default"])] for n, m in layers.items()}
        want = {m: [peft_rank(cfg, m)[0], peft_rank(cfg, m)[1] / peft_rank(cfg, m)[0]] for m in cfg["target_modules"]}
        pc = model.peft_config["default"]
        same_config = (pc.r, pc.lora_alpha, dict(pc.rank_pattern or {}), dict(pc.alpha_pattern or {})) == \
            (cfg["r"], cfg["lora_alpha"], cfg["rank_pattern"], cfg["alpha_pattern"])
        rep.check(label, "P1 PEFT loads the IDE's adapter", not caught and set(layers) == set(cfg["target_modules"])
                  and ranks == want and same_config, warnings=warned(caught), modules=len(layers),
                  targets=len(cfg["target_modules"]), only_peft=sorted(set(layers) - set(cfg["target_modules"]))[:5],
                  only_file=sorted(set(cfg["target_modules"]) - set(layers))[:5],
                  rank_and_scaling=sorted({tuple(v) for v in ranks.values()}), config_equal=same_config)
        file_sd = load_peft_weights(str(ide_dir), device="cpu")   # PEFT would pick MPS, which has no float64
        peft_keys = set(get_peft_model_state_dict(model))
        dtypes = sorted({str(v.dtype) for v in file_sd.values()})
        rep.check(label, "P2 the file's keys are PEFT's", set(file_sd) == peft_keys and dtypes == ["torch.float32"],
                  keys=len(file_sd), only_file=sorted(set(file_sd) - peft_keys)[:4],
                  only_peft=sorted(peft_keys - set(file_sd))[:4], dtypes=dtypes)
        attached = first_token_logits(model, kl_ids)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            merged_model = model.merge_and_unload()
        merged = merged_model.state_dict()
        cur_f = safe_open(str(out / label / "ide_current.safetensors"), "pt")
        cur = sorted(cur_f.keys())
        rows = []
        for name in cur:
            c, m, b = cur_f.get_tensor(name), merged[name], base_f.get_tensor(name).float()
            delta = c - b
            bit = bool(torch.equal(m, c))
            row = {"tensor": name, "bit_equal": bit, "max_abs": float((m - c).abs().max()),
                   "relative_change": rel(m - b, delta), "max_abs_delta": float(delta.abs().max())}
            row["within"] = bit or (row["relative_change"] <= P4_RELATIVE
                                    and row["max_abs"] <= P4_RELATIVE * row["max_abs_delta"])
            rows.append(row)
            del c, b, delta
        rep.check(label, "P4 merged IDE adapter equals the IDE's current weights",
                  not caught and rows and all(r["within"] for r in rows), tensors=len(rows),
                  bit_equal=sum(r["bit_equal"] for r in rows), not_bit_equal=[r for r in rows if not r["bit_equal"]][:5],
                  worst_relative=max((r["relative_change"] for r in rows), default=None), warnings=warned(caught))
        rep.note(label, "P4 rows", rows)
        others = sorted(k for k in base_keys if k not in cur and k in merged)
        differ = [k for k in others if not torch.equal(merged[k], base_f.get_tensor(k).float())]
        rep.check(label, "P4b every other tensor equals the base", not differ, tensors=len(others), differ=differ[:5])
        del model, merged_model, merged, layers, cur_f
        gc.collect()
        # ---- P5: the IDE's current weights loaded into a fresh base
        model = fresh()
        cur_f = safe_open(str(out / label / "ide_current.safetensors"), "pt")
        sd = model.state_dict()
        with torch.no_grad():
            for name in cur:
                sd[name].copy_(cur_f.get_tensor(name))
        loaded = first_token_logits(model, kl_ids)
        diff = float((attached - loaded).abs().max())
        scale = float(loaded.abs().max())
        bound = P5_RELATIVE * max(1.0, scale)
        rep.check(label, "P5 first-token logits, adapter attached vs current weights", diff <= bound,
                  max_abs=diff, bound=bound, relative=diff / max(1.0, scale), prompts=len(kl_ids), scale=scale,
                  heretic_adapter_attached_vs_merged=baseline)
        del model, sd, cur_f, attached, loaded
        gc.collect()
        # ---- P6, P6b, P6c: the two tools' factors and products, module by module
        hmods = modules_of(hsd)
        hscale = hcfg["lora_alpha"] / hcfg["r"]
        imods = set(modules_of(file_sd))
        rows = []
        for m in hmods:
            hb, ha = factors(hsd, m)
            hp = hb @ ha * hscale
            if m in imods:
                ib, ia = factors(file_sd, m)
                r_m, a_m = peft_rank(cfg, m)
                rows.append({"module": m, "in_ide": True, "relative": rel(ib @ ia * (a_m / r_m), hp),
                             "heretic_norm": float(hp.norm())})
            else:
                rows.append({"module": m, "in_ide": False, "heretic_zero": bool((hp == 0).all()),
                             "heretic_norm": float(hp.norm())})
        extra = sorted(imods - set(hmods))
        in_ide = [r for r in rows if r["in_ide"]]
        worst = max((r["relative"] for r in in_ide), default=0.0)
        if ide["method"] == "ablation":
            bound = P6_BOUNDS.get(orig.parent.name, P6_BOUND)
            rep.check(label, "P6 products per module (ablation, each tool's own measurement)",
                      worst <= bound and all(r["heretic_zero"] for r in rows if not r["in_ide"]) and not extra,
                      worst_relative=worst, bound=bound, modules=len(rows), in_ide=len(in_ide),
                      absent_nonzero=[r["module"] for r in rows if not r["in_ide"] and not r["heretic_zero"]][:5],
                      only_ide=extra[:5])
        else:
            rep.note(label, "P6c products per module (ARA, each tool's own captures and libraries; reported)",
                     {"worst_relative": worst, "rows": rows, "only_ide": extra})
            print(f"     {label} P6c (reported) worst relative difference of the products {worst:.3e} over "
                  f"{len(in_ide)} modules", flush=True)
            cap = out / label / "captures" / "ide.json"
            if cap.is_file():
                cinfo = json.loads(cap.read_text())
                csd = load_peft_weights(cinfo["adapter_dir"], device="cpu")
                cmods = set(modules_of(csd))
                bad = []
                for m in sorted(cmods):
                    if m not in hmods:
                        bad.append({"module": m, "why": "not in Heretic's adapter"})
                        continue
                    (cb, ca), (hb, ha) = factors(csd, m), factors(hsd, m)
                    if not (cb.shape == hb.shape and ca.shape == ha.shape and torch.equal(cb, hb) and torch.equal(ca, ha)):
                        bad.append({"module": m, "why": "factors differ", "shapes": [list(cb.shape), list(hb.shape)],
                                    "max_abs_B": float((cb - hb).abs().max()) if cb.shape == hb.shape else None,
                                    "max_abs_A": float((ca - ha).abs().max()) if ca.shape == ha.shape else None})
                lacking = [m for m in hmods if m not in cmods
                           and not (bool((factors(hsd, m)[0] == 0).all()) or bool((factors(hsd, m)[1] == 0).all()))]
                rep.check(label, "P6b ARA from Heretic's captures: the factors equal Heretic's", not bad and not lacking,
                          modules=len(cmods), differ=bad[:5], lacking_but_nonzero=lacking[:5])
            else:
                rep.check(label, "P6b ARA from Heretic's captures: the factors equal Heretic's", False,
                          error="no captures/ide.json: run --stage ide-captures first")
        del hsd, file_sd
        gc.collect()
    return rep, "peft"


# ------------------------------------------------------------------ main
def main(argv):
    if len(argv) < 2 or argv[0] != "--stage" or argv[1] not in ("ide", "ide-captures", "peft"):
        raise SystemExit(__doc__)
    stage = argv[1]
    if stage == "peft":
        if len(argv) < 5:
            raise SystemExit(__doc__)
        snap, out, labels = Path(argv[2]), Path(argv[3]), parse_labels(argv[4:])
        rep, name = stage_peft(snap, out, labels)
    else:
        if len(argv) < 6:
            raise SystemExit(__doc__)
        snap, root, out, labels = Path(argv[2]), Path(argv[3]), Path(argv[4]), parse_labels(argv[5:])
        rep, name = stage_ide(snap, root, out, labels, captures=stage == "ide-captures")
    from importlib.metadata import version
    versions = {}
    for p in ("torch", "transformers", "safetensors", "peft"):
        try:
            versions[p] = version(p)
        except Exception:
            versions[p] = None
    result = {"stage": stage, "versions": versions, "python": sys.version.split()[0], "labels": rep.labels,
              "pass": rep.ok()}
    write(out, f"{name}.json", result)
    print(f"{'ok' if result['pass'] else 'NOT OK'}: {out / (name + '.json')}", flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
