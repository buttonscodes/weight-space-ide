"""What Heretic does on a mixture-of-experts model: its modules, and what its ARA capture receives. A reproduction.

Three modes, each with its own interpreter:

    python scripts/heretic_moe_probe.py heretic <run dir> <out.json> [<batch size>]         # Heretic's interpreter
    python scripts/heretic_moe_probe.py ide <model dir> <isolated IDE root> <out.json>      # the IDE's interpreter
    python scripts/heretic_moe_probe.py compare <heretic.json> <ide.json> [<out.json>]       # either

``heretic``: from the run directory's config.toml (any Heretic config; the good prompts of its [modifier.ARA] or
[modifier.Abliteration] section), loads Heretic's own Model and records, with Heretic's own code only:
  - per layer, the modules get_layer_modules returns, by component and module index, under their full names (the
    names apply_lora targets), and for every ``experts`` attribute of ``mlp``, ``block_sparse_moe``, ``feed_forward``
    or ``moe`` (the owners get_layer_modules iterates) its type, its own parameters and what iterating it does (the
    loops in get_layer_modules that add routed experts);
  - on the first batch of the good prompts at the batch size given (default 1; one batch size per process), first an
    observation: the generate call get_module_io makes (Heretic's own Model.generate, max_new_tokens=1,
    use_cache=False) with recording hooks only: the shape of each hooked module's input and output, in call order;
    on the fused expert module of every MoE block the shapes of its arguments; on the router (``mlp.gate``) the
    experts it selects for each prompt's last token (row b * S + S - 1 of its flattened [B * S, k] output, S the
    padded length; Heretic pads on the left);
  - then Heretic's get_module_io on the same batch, a recording hook registered before each of Heretic's, and the
    exception it raises, if any, with the module whose hook was running (the last one recorded);
  - when that succeeds, Heretic's get_module_io_batched on all the good prompts, and what it returns or raises.
``ide``: opens the model as an IDE workspace in the isolated root and records the IDE's modules per layer and
component (wsp.directions.modules, model-free) for expert_modules "all" and "shared", as module paths in the loaded
model's naming (wsp.runtime_layouts.module_path of model.layers.<l>.<path>).
``compare``: the two side by side. Bound, stated before the first run: Heretic's and the IDE's modules are the same
modules in the same order, per layer and component (G2b (i)); asserted for each value of expert_modules; each prints
PASS or FAIL and the exit status is 1 when one fails.

No weight is changed and nothing is written but <out.json>. Run offline, under the memory cap.
"""
import json
import sys
from pathlib import Path


def heretic(run, out, size):
    import os
    import tomllib
    import traceback
    os.chdir(run)
    sys.argv = sys.argv[:1]                      # Settings() parses the command line
    import torch
    import transformers
    from heretic.config import Settings, SingleDatasetSpecification
    from heretic.model import Model
    from heretic.utils import batchify, load_prompts

    settings = Settings()
    cfg = tomllib.loads(Path("config.toml").read_text())
    section = cfg["modifier"]["ARA" if "ARA" in cfg["modifier"] else "Abliteration"]
    prompts = load_prompts(settings, SingleDatasetSpecification(**section["good_prompts"]))
    model = Model(settings)
    by_id = {id(m): n for n, m in model.model.named_modules()}
    result = {"versions": {"torch": torch.__version__, "transformers": transformers.__version__},
              "prompts": len(prompts), "layers": []}

    for li, layer in enumerate(model.get_layers()):
        row = {"modules": {c: [by_id.get(id(m)) for m in ms] for c, ms in model.get_layer_modules(li).items()},
               "experts": {}}
        for owner in ("mlp", "block_sparse_moe", "feed_forward", "moe"):
            ex = getattr(getattr(layer, owner, None), "experts", None)
            if ex is None:
                continue
            try:
                iter(ex)
                it = "iterable"
            except TypeError as e:
                it = f"TypeError: {e}"
            row["experts"][f"{owner}.experts"] = {
                "type": type(ex).__name__, "iterating": it,
                "parameters": {n: list(p.shape) for n, p in ex.named_parameters(recurse=False)}}
        result["layers"].append(row)

    def observe(batch):
        """Heretic's own generate call of get_module_io (max_new_tokens=1, use_cache=False) with recording hooks only."""
        calls, routes, fused, handles = [], [], [], []
        for li in range(len(model.get_layers())):
            for comp, ms in model.get_layer_modules(li).items():
                for mi, m in enumerate(ms):
                    def rec(mod, inp, outp, li=li, comp=comp, mi=mi):
                        calls.append({"layer": li, "component": comp, "index": mi, "module": by_id.get(id(mod)),
                                      "input": list(inp[0].shape), "output": list(outp.shape)})
                    handles.append(m.register_forward_hook(rec))
            mlp = getattr(model.get_layers()[li], "mlp", None)
            if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp, "experts"):
                def gate(mod, inp, outp, li=li):
                    routes.append({"layer": li, "top_k_index": outp[2].tolist()})
                def ex(mod, inp, outp, li=li):
                    fused.append({"layer": li, "module": by_id.get(id(mod)), "arguments": [list(a.shape) for a in inp],
                                  "output": list(outp.shape)})
                handles += [mlp.gate.register_forward_hook(gate), mlp.experts.register_forward_hook(ex)]
        try:
            enc, _ = model.generate(batch, max_new_tokens=1, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        B, S = enc["input_ids"].shape
        last = [{"layer": r["layer"], "rows": len(r["top_k_index"]),
                 "last_token_experts": [r["top_k_index"][b * S + S - 1] for b in range(B)]} for r in routes]
        return {"prompts": B, "padded_length": S, "prompt_lengths": enc["attention_mask"].sum(1).tolist(),
                "calls": calls, "fused_experts": fused, "router": last}

    def capture(batch):
        """Heretic's get_module_io on ``batch``, a recording hook registered before each of Heretic's."""
        calls, handles = [], []
        for li in range(len(model.get_layers())):
            for comp, ms in model.get_layer_modules(li).items():
                for mi, m in enumerate(ms):
                    def rec(mod, inp, outp, li=li, comp=comp, mi=mi):
                        calls.append({"layer": li, "component": comp, "index": mi, "module": by_id.get(id(mod)),
                                      "input": list(inp[0].shape), "output": list(outp.shape)})
                    handles.append(m.register_forward_hook(rec))   # before Heretic's: runs first
        error, io = None, None
        try:
            io = model.get_module_io(batch)
        except Exception as e:                     # what Heretic's capture raises
            error = {"type": type(e).__name__, "message": str(e),
                     "raised_in_hook_of": calls[-1] if calls else None,
                     "traceback_tail": traceback.format_exc().strip().splitlines()[-4:]}
        finally:
            for h in handles:
                h.remove()
        got = io and [{c: {str(i): [list(x.shape), list(y.shape)] for i, (x, y) in v.items()} for c, v in L.items()}
                      for L in io]
        return {"prompts": len(batch), "calls": calls, "error": error, "captured": got}

    # One batch size per process: after an exception in a hook, Heretic's get_module_io leaves its hooks registered
    # (their removal follows generate, outside a finally), so a second capture in the same process is not clean.
    first = batchify(prompts, size)[0]
    result["batch_size"] = size
    result["observed"] = observe(first)
    result["capture"] = capture(first)
    if result["capture"]["error"] is None:
        try:
            io = model.get_module_io_batched(prompts)
            result["batched"] = {"error": None, "captured": [{c: {str(i): [list(x.shape), list(y.shape)]
                                                                    for i, (x, y) in v.items()} for c, v in L.items()}
                                                             for L in io]}
        except Exception as e:
            result["batched"] = {"error": {"type": type(e).__name__, "message": str(e)}}
    else:
        result["batched"] = {"error": "not run: get_module_io on the first batch raised"}
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps({"layers": [r["modules"] for r in result["layers"]],
                      "experts": result["layers"][0]["experts"],
                      "batch_size": size, "error": (result["capture"]["error"] or {}).get("message"),
                      "raised_in_hook_of": (result["capture"]["error"] or {}).get("raised_in_hook_of"),
                      "batched_error": result["batched"]["error"]}, indent=1))


def ide(model_dir, root, out):
    import os
    sys.path.insert(0, os.getcwd())
    from wsp import directions as DR
    from wsp import runtime_layouts as RL
    from wsp.session import Session
    ws = Session(str(root)).open(str(model_dir), log=lambda *a, **k: None)
    result = {"layers": {}}
    for em in DR.EXPERT_MODULES:
        rows = []
        for l in range(ws.struct.n_layers):
            row = {}
            for comp in DR.COMPONENT_SOURCES:
                ms = DR.modules(ws, l, comp, em)
                if ms:
                    row[comp] = [RL.module_path(ws.hf_config, f"model.layers.{l}.{m['path']}") for m in ms]
            rows.append(row)
        result["layers"][em] = rows
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def compare(h_path, i_path, out=None):
    h, i = json.loads(Path(h_path).read_text()), json.loads(Path(i_path).read_text())
    strip = lambda n: n.replace("model.language_model.", "model.", 1) if n else n  # noqa: E731  the loaded text model's names
    her = [{c: [strip(n) for n in v] for c, v in L["modules"].items()} for L in h["layers"]]
    result, ok = {"heretic": her, "checks": {}}, True
    for em, rows in i["layers"].items():
        diffs = []
        for l, (a, b) in enumerate(zip(her, rows)):
            for comp in sorted(set(a) | set(b)):
                if a.get(comp, []) != b.get(comp, []):
                    diffs.append({"layer": l, "component": comp, "heretic": a.get(comp, []), "ide": b.get(comp, [])})
        same = not diffs and len(her) == len(rows)
        ok &= same
        result["checks"][em] = {"pass": same, "differences": diffs,
                                "modules_heretic": sum(len(v) for L in her for v in L.values()),
                                "modules_ide": sum(len(v) for L in rows for v in L.values())}
        print(f"{'PASS' if same else 'FAIL'}  (i) same modules in the same order, expert_modules={em!r}: "
              f"Heretic {result['checks'][em]['modules_heretic']} modules, IDE {result['checks'][em]['modules_ide']}, "
              f"{len(diffs)} layer/component lists differ")
        for d in diffs[:4]:
            print("     ", d)
    result["pass"] = ok
    if out:
        Path(out).write_text(json.dumps(result, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) >= 2 else 2)
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "heretic":
        heretic(Path(args[0]).resolve(), Path(args[1]).resolve(), int(args[2]) if len(args) > 2 else 1)
    elif mode == "ide":
        ide(args[0], args[1], args[2])
    elif mode == "compare":
        sys.exit(compare(*args[:3]))
    else:
        sys.exit(f"unknown mode {mode}")
