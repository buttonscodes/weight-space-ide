"""Compare the IDE's fitted low-rank ablation (ARA) with finished Heretic ARA runs on the same model, stage by stage.

Run with the IDE's interpreter, with the IDE's `weight-space-ide` directory as the current directory, offline and
under the memory guard:

    python scripts/heretic_compare_ara.py [--dtype float32|bfloat16] [--heretic-adapter [<label>=]<dir> ...] \\
        [--expert-modules all|shared] <model snapshot dir> <isolated IDE root> <out.json> <label>=<heretic run dir> ...

Each Heretic run directory holds the dump written by heretic_dump.py (dump/dump.json, dump/dump.safetensors) and
either Heretic's merged model (model/model.safetensors) or its adapter (export_strategy "adapter": the directory
given with --heretic-adapter, else the first of <run>/adapter, <run>/<save_directory>, <run>/model that holds
adapter_config.json; refused unless r == lora_alpha; Heretic's change is then B @ A). Stages: token ids, mean
residuals, the saved module inputs and outputs, the weight change per tensor (each tool from its own captures, then
the IDE from Heretic's captures), the consistency check of every fitted module, the value of the fitted objective at
each tool's change, base logits and the KL score; with --dtype bfloat16 also the refusals of the IDE's own fit
(C4-5, reported) and the time of each stage (C4-6).

--expert-modules is the IDE fit's expert_modules (default "all", as before). The IDE refuses ARA over routed experts
with "all"; on a mixture-of-experts model Heretic under transformers 5.x has no routed-expert modules (they are one
fused module its module discovery skips), so "shared" is the IDE fit that corresponds to Heretic's there.

--dtype sets WSP_RUN_DTYPE before the workspace's runner exists (default float32, today's comparison) and selects
the row of BOUNDS below. Modules are matched by Heretic's module names (dump.json "modules") through
wsp.runtime_layouts.module_path; a dump without them is matched by index, as before (a component Heretic captured
as one module). A model of several weight files (model.safetensors.index.json) is read one tensor at a time.

The objective here is written a second time, in float64 and in the checkpoint's own [d, D_in] layout, from the
formula in docs/DIRECTIONS.md; it does not call wsp.directions_ara.

Every assertion prints PASS or FAIL; <out.json> holds every number and a pass flag per assertion; the exit status
is 1 when one fails. None in BOUNDS = reported only.
"""
import argparse
import math
import os
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

# ---- BOUNDS[(stage, dtype)]: float32 rows are today's (this script asserted nothing before), bfloat16 rows 7.3/10.3
BOUNDS = {
    ("B-S1 token ids", "float32"): None,
    ("B-S1 token ids", "bfloat16"): "identical",
    ("B-S2 mean residuals", "float32"): None,
    ("B-S2 mean residuals", "bfloat16"): None,                  # reported; expected relative difference below 2e-2
    ("B-S6 consistency check", "float32"): None,
    ("B-S6 consistency check", "bfloat16"): (0.0, 5e-3),         # 0 < mismatch <= 5e-3, every captured module
    ("C4-2 module inputs and outputs", "float32"): None,
    ("C4-2 module inputs and outputs", "bfloat16"): None,        # reported; expected of the order of 2^-8
    ("C4-3 objective reduction", "float32"): None,
    ("C4-3 objective reduction", "bfloat16"): 0.10,              # median over matrices of |r_IDE - r_H| / |r_H|
    ("C4-5 refusals of the IDE's own fit", "float32"): None,
    ("C4-5 refusals of the IDE's own fit", "bfloat16"): None,    # reported
}
SETTINGS = ("preserve_row_magnitudes", "lora_rank", "n_optimization_steps", "learning_rate", "max_iter", "history_size")


def objective(W, dW, bias, Xg, Yg, Xb, Yb, p, preserve):
    """The fitted objective at a change dW of W [d, D_in], in float64."""
    import torch
    W, dW, Xg, Yg, Xb, Yb = (t.double() for t in (W, dW, Xg, Yg, Xb, Yb))
    new = W + dW
    og, ob = Xg @ new.T, Xb @ new.T
    if preserve:
        scale = W.norm(dim=1) / new.norm(dim=1)
        og, ob = og * scale, ob * scale
    if bias is not None:
        og, ob = og + bias.double(), ob + bias.double()
    k = p["neighbor_count"]
    near = lambda a, b: torch.cdist(a, b).topk(k, dim=1, largest=False).values.mean(1)  # noqa: E731
    keep = ((og - Yg) ** 2).mean()
    steer = near(ob, Yg).mean() - p["overcorrect_relative_weight"] * near(ob, Yb).mean()
    return {"total": float(p["preserve_good_behavior_weight"] * keep + p["steer_bad_behavior_weight"] * steer),
            "preserve": float(keep), "steer": float(steer)}


def parse(argv):
    p = argparse.ArgumentParser(prog="heretic_compare_ara.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--heretic-adapter", action="append", default=[], metavar="[LABEL=]DIR")
    p.add_argument("--expert-modules", choices=("all", "shared"), default="all")
    p.add_argument("snapshot")
    p.add_argument("root")
    p.add_argument("out")
    p.add_argument("runs", nargs="+", metavar="LABEL=RUN")
    a = p.parse_args(argv)
    return a, dict(r.split("=", 1) for r in a.runs)


def main(argv):
    a, runs = parse(argv)
    sys.path.insert(0, os.getcwd())
    import formats_shared as FS
    FS.set_run_dtype(a.dtype)

    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import load_file

    from wsp import directions as DR
    from wsp import directions_record as REC
    from wsp import directions_scores as DS
    from wsp import runtime_layouts as RL
    from wsp.session import Session

    import heretic_captures as HC                # this directory: the captures stages, shared with heretic_compare_adapter

    adapters = FS.parse_adapters(a.heretic_adapter, list(runs))
    snap, root, out = Path(a.snapshot), Path(a.root), Path(a.out)
    rel = FS.rel
    B = lambda s: FS.bound(BOUNDS, s, a.dtype)  # noqa: E731

    def stage(*what):
        print("[compare]", *what, file=sys.stderr, flush=True)

    CHUNK = 1 << 24

    def changes(ws, base, merged):
        """Heretic's change against the IDE's, tensor by tensor and in row chunks of at most CHUNK values (as
        heretic_compare.py). -> (rows of changed tensors, how many were unchanged in both tools)."""
        changed, same = [], 0
        for name in sorted(set(base.keys()) & set(merged.keys())):
            bs, ms = base.get_slice(name), merged.get_slice(name)
            shape = list(bs.get_shape())
            assert shape and shape == list(ms.get_shape()), (name, shape, ms.get_shape())
            per = max(1, CHUNK // max(1, math.prod(shape[1:])))
            h_changed = i_changed = False
            a2 = {"cur_m": 0.0, "m_b": 0.0, "cur_b": 0.0, "b": 0.0, "m": 0.0}
            worst, differ = 0.0, 0
            for lo in range(0, shape[0], per):
                z = slice(lo, min(shape[0], lo + per))
                b, m = bs[z].float(), ms[z].float()
                cur = ws.read(name, (z,), variant="current").float()
                h, i = not torch.equal(m, b), not torch.equal(cur, b)
                h_changed, i_changed = h_changed or h, i_changed or i
                if h or i:
                    e = (cur - m).double()
                    a2["cur_m"] += float((e * e).sum())
                    a2["m_b"] += float(((m - b).double() ** 2).sum())
                    a2["cur_b"] += float(((cur - b).double() ** 2).sum())
                    worst, differ = max(worst, float(e.abs().max())), differ + int((cur != m).sum())
                a2["b"] += float((b.double() ** 2).sum())
                a2["m"] += float((m.double() ** 2).sum())
                del b, m, cur
            if not h_changed and not i_changed:
                same += 1
            elif h_changed != i_changed:
                changed.append({"tensor": name, "heretic_changed": h_changed, "ide_changed": i_changed})
            else:
                changed.append({"tensor": name, "rel_diff": math.sqrt(a2["cur_m"] / a2["m_b"]),
                                "change_over_weight": math.sqrt(a2["m_b"] / a2["b"]),
                                "ide_change_over_weight": math.sqrt(a2["cur_b"] / a2["b"]), "max_abs_diff": worst,
                                "elements": math.prod(shape), "elements_that_differ": differ,
                                "edited_weight_rel_diff": math.sqrt(a2["cur_m"] / a2["m"])})
        return changed, same

    def tensor_for(ws, l, comp, path):
        """The checkpoint tensor of the IDE module (layer, component, path in the layer; None: the component's
        matrix, a version-1 measurement)."""
        if path is None:
            views = DR.component_views(ws, l, comp)
            return views[0].name if views else None
        hits = [t for t in base.keys() if t.endswith(".weight")
                and FS.layer_relative(RL.module_path(ws.hf_config, t[:-len(".weight")]), l) == path]
        return hits[0] if len(hits) == 1 else None

    def refusal_count(ws, texts, system, prefix, run):
        import json
        import tomllib
        sc = run / "dump/scores.json"
        M = json.loads(sc.read_text())["max_response_length"] if sc.is_file() else \
            tomllib.loads((run / "config.toml").read_text()).get("max_response_length", 100)
        got = DS.score(ws, {"state": "current", "chat": True, "system": system, "prefix": prefix,
                            "max_response_length": M, "scores": [{"plugin": "KeywordRate", "prompts": texts}]})
        return got["scores"][0]["matches"]

    def mismatches(ws, res):
        rec, _ = REC.read(ws, ws.patch(res["patch"]))
        return FS.ara_components(rec, res.get("output", ""))

    rep = FS.Report()
    session = Session(str(root))
    ws = session.open(str(snap), log=lambda *a_, **k: None)
    n, d = ws.struct.n_layers, ws.struct.d_model
    base = FS.Checkpoint(snap)                    # read one tensor at a time, never a whole file
    result = {"model": {"layers": n, "d_model": d, "tensors": len(base.keys()),
                        "parameters": sum(math.prod(base.get_slice(k).get_shape()) for k in base.keys())},
              "ide_versions": {p: version(p) for p in ("torch", "transformers", "tokenizers", "safetensors")},
              "interpreter": sys.version.split()[0], "dtype": a.dtype, "expert_modules": a.expert_modules,
              "bounds": {k[0]: v for k, v in BOUNDS.items() if k[1] == a.dtype}, "runner": None, "runs": {}}

    import json
    for label, run in runs.items():
        run = Path(run)
        dump = json.loads((run / "dump/dump.json").read_text())
        tens = load_file(str(run / "dump/dump.safetensors"))
        assert dump["modifier"] == "ARA" and dump["layers"] == n, (dump["modifier"], dump["layers"], n)
        r = {"heretic_versions": dump["versions"], "heretic_dtype": dump["dtype"], "heretic_device": dump["device"],
             "timing": {}}
        system, prefix = dump["system_prompt"], dump["response_prefix"] or None
        texts = {k: [p["user"] for p in v] for k, v in dump["prompts"].items()}
        assert all(p["system"] == system for v in dump["prompts"].values() for p in v)
        merged_file = run / "model/model.safetensors"
        adapter_dir = None if merged_file.is_file() and label not in adapters else FS.find_adapter(run, adapters.get(label))
        if adapter_dir is None and not merged_file.is_file():
            rep.check(label, "Heretic's export found", False, error=f"{run}: neither model/model.safetensors nor an adapter")
            continue
        adapter = FS.Adapter(adapter_dir) if adapter_dir else None
        merged = None if adapter else safe_open(str(merged_file), "pt")
        r["heretic_export"] = "adapter" if adapter else "merged"
        modules = FS.dump_modules(dump)
        r["matched_by"] = "module names" if modules is not None else "index"

        # ---- stage 1: token ids (the prompt as each tool feeds it to the model)
        checked = DR.check_measure(ws, "oracle", texts["good"], texts["bad"], chat=True, system=system, prefix=prefix,
                                   variant="base", replace=True, module_io=True)
        ids = DR._rendered_ids(ws, checked)
        kl_checked = DR.check_measure(ws, "oracle-kl", texts["kl"], texts["kl"], chat=True, system=system,
                                      prefix=prefix, variant="base", replace=True)
        kl_ids = DR._rendered_ids(ws, kl_checked)["good"]
        r["tokens"] = {k: {"prompts": len(ids[k]), "identical": sum(x == y for x, y in zip(ids[k], dump["io_ids"][k]))}
                       for k in ("good", "bad")}
        r["tokens"]["kl"] = {"prompts": len(kl_ids), "identical": sum(x == y for x, y in zip(kl_ids, dump["ids"]["kl"]))}
        rep.bounded(label, "B-S1 token ids", B("B-S1 token ids"),
                    all(v["identical"] == v["prompts"] for v in r["tokens"].values()), lambda v, b: v,
                    counts={k: f"{v['identical']}/{v['prompts']}" for k, v in r["tokens"].items()})

        # ---- stage 2: mean residuals, [n + 1, d]
        stage(label, "measuring with module_io")
        t0 = time.monotonic()
        DR.measure(ws, "oracle", texts["good"], texts["bad"], chat=True, system=system, prefix=prefix, variant="base",
                   replace=True, module_io=True)
        r["timing"]["measure_seconds"] = time.monotonic() - t0
        result["runner"] = DR.runner_description(ws)
        meta, good, bad = DR.load(ws, "oracle")
        r["means"] = {"good_rel_worst": max(rel(good[k], tens["good_mean"][k]) for k in range(n + 1)),
                      "bad_rel_worst": max(rel(bad[k], tens["bad_mean"][k]) for k in range(n + 1))}
        rep.note(label, "B-S2 mean residuals", {**r["means"], "expected_below": 2e-2})

        # ---- stage 3: the saved module inputs and outputs, per layer, component and module
        info = DR.load_io(ws, "oracle", meta)
        io_rows, heretic_io = HC.heretic_io_of(ws, "oracle", info, tens, n, d, modules=modules)
        r["module_io"] = io_rows
        worst_io = [max((row.get(f"{s}_{x}_rel", 0.0) for row in io_rows for s in ("good", "bad")), default=None)
                    for x in "xy"]
        rep.note(label, "C4-2 module inputs and outputs", {"worst_input_rel": worst_io[0], "worst_output_rel": worst_io[1],
                                                           "captured_by_both": sum("good_x_rel" in row for row in io_rows),
                                                           "ide_modules": len(io_rows), "expected_order": 2 ** -8})
        own_io = HC.ide_io_of(ws, "oracle", info, n, d)

        # ---- stage 4: the weight change, tensor by tensor, each tool from its own captures
        trial = next(t for t in dump["trials"] if t["index"] == dump["exported_trial_index"])
        conf = dump["modifier_settings"]["ARA"]
        defaults = {"preserve_row_magnitudes": True, "lora_rank": 50, "n_optimization_steps": 5, "learning_rate": 1.0,
                    "max_iter": 20, "history_size": 10}
        params = {"method": "ara", "direction": "oracle", **trial["parameters"],
                  **{k: conf.get(k, defaults[k]) for k in SETTINGS}, "seed": dump["seed"]}
        if a.expert_modules != "all":
            params["expert_modules"] = a.expert_modules
        r["parameters"] = {k: v for k, v in params.items() if k not in ("method", "direction")}
        p = trial["parameters"]
        script = DR.script_for(params)
        stage(label, "fitting (own captures)")
        t0 = time.monotonic()
        res = session.apply(script, message=f"heretic comparison (ARA, {label})", dry_run=False)
        r["timing"]["fit_own_seconds"] = time.monotonic() - t0
        assert res.get("ok") and res.get("committed"), res
        if adapter is None:
            r["keys"] = {"base": len(base.keys()), "merged": len(merged.keys()),
                         "only_base": sorted(set(base.keys()) - set(merged.keys()))[:5],
                         "only_merged": sorted(set(merged.keys()) - set(base.keys()))[:5]}
            stage(label, "comparing tensors")
            changed, same = changes(ws, base, merged)
        else:
            r["adapter"] = {"dir": str(adapter_dir), "r": adapter.config["r"], "lora_alpha": adapter.config["lora_alpha"],
                            "modules": len(adapter.modules), "dtypes": adapter.dtypes()}
            stage(label, "comparing tensors (Heretic's adapter)")
            changed, same, r["adapter"]["unmatched_modules"] = FS.adapter_changes(ws, base, adapter, ws.patch(res["patch"]),
                                                                                  rounding=a.dtype != "float32")
        r["weights"] = {"changed": changed, "unchanged_in_both": same,
                        "changed_in_one_only": sum("rel_diff" not in c for c in changed),
                        "ide_stored_values": res.get("stored"), "ide_output": res.get("output", "")[-1500:]}

        # ---- B-S6: the consistency check ||X V + b - Y|| / ||Y|| of every fitted module, as the fit reported it
        comps = mismatches(ws, res)
        values = [c[k] for c in comps for k in ("mismatch_good", "mismatch_bad")]
        lim = B("B-S6 consistency check")
        rep.bounded(label, "B-S6 consistency check", lim, values,
                    lambda v, b: bool(v) and all(b[0] < x <= b[1] for x in v), modules=len(comps),
                    smallest=min(values, default=None), largest=max(values, default=None))
        r["consistency"] = comps

        # ---- stage 6 (part 1): the objective at the IDE's own change, on Heretic's captures and on its own
        def target(l, comp, path):
            return tensor_for(ws, l, comp, path)

        fitted = sorted({(l, comp, path) for (s, l, comp, path) in heretic_io
                         if s == "good" and p["start_layer_index"] <= l < p["end_layer_index"]},
                        key=lambda k: (k[0], list(DR.COMPONENTS).index(k[1]) if k[1] in DR.COMPONENTS else 9, str(k[2])))
        fits, shapes = [], {}
        names = set(base.keys())
        amap = {FS.tensor_of_module(ws.hf_config, m, names): m for m in adapter.modules} if adapter is not None else {}
        for l, comp, path in fitted:
            name = target(l, comp, path)
            if name is None:
                fits.append({"layer": l, "component": comp, "module": path, "error": "no checkpoint tensor"})
                continue
            bias_name = name[:-len(".weight")] + ".bias"
            W = base.get_tensor(name).float()
            bias = base.get_tensor(bias_name).float() if bias_name in base else None
            (xg, yg), (xb, yb) = heretic_io[("good", l, comp, path)], heretic_io[("bad", l, comp, path)]
            at = lambda dW, cap=(xg, yg, xb, yb), W=W, bias=bias: objective(  # noqa: E731
                W, dW, bias, *cap, p, params["preserve_row_magnitudes"])
            zero = torch.zeros_like(W)
            if adapter is not None:
                d_h = adapter.delta(amap[name]) if name in amap else zero
            else:
                d_h = merged.get_tensor(name).float() - W
            d_i = ws.read(name, variant="current").float() - W
            row = {"layer": l, "component": comp, "module": path, "tensor": name, "shape": list(W.shape),
                   "bias": bias is not None, "start": at(zero), "heretic": at(d_h), "ide": at(d_i)}
            own = own_io.get(("good", l, comp, path)), own_io.get(("bad", l, comp, path))
            if own[0] is not None and own[1] is not None:
                cap = (own[0][0], own[0][1], own[1][0], own[1][1])
                row["own_captures"] = {"start": at(zero, cap), "ide": at(d_i, cap)}
            fits.append(row)
            shapes["%dx%d" % (W.shape[1], W.shape[0])] = shapes.get("%dx%d" % (W.shape[1], W.shape[0]), 0) + 1
            del W, d_h, d_i
        r["timing"]["fitted_shapes"] = shapes     # V [D_in, d] of every fitted matrix, for formats_fit_timing.py

        # ---- stage 5: behaviour: base logits, and KL(first token) of the edited model against the base
        hl = tens["kl_logits"]
        lb = torch.stack([ws.runner.logits(x, "base", tail=1)[-1].float() for x in kl_ids])
        lc = torch.stack([ws.runner.logits(x, "current", tail=1)[-1].float() for x in kl_ids])
        kl = float(F.kl_div(F.log_softmax(lc, -1), F.log_softmax(lb, -1), reduction="batchmean", log_target=True))
        r["behaviour"] = {"base_logits_max_abs_diff": float((lb - hl).abs().max()), "base_logits_scale": float(hl.abs().max()),
                          "kl_ide": kl,
                          "kl_heretic": next(s["score"]["value"] for s in trial["scores"] if s["name"] == "KL divergence"),
                          "refusals_heretic": next(s["score"]["value"] for s in trial["scores"] if s["name"] == "Refusals")}
        if a.dtype != "float32":                  # C4-5: the IDE's own fit scored for refusals (reported)
            t0 = time.monotonic()
            ide_count = refusal_count(ws, texts.get("refusal") or [], system, prefix, run)
            r["timing"]["refusal_score_seconds"] = time.monotonic() - t0
            her_count = round(r["behaviour"]["refusals_heretic"] * len(texts.get("refusal") or []))
            rep.note(label, "C4-5 refusals of the IDE's own fit", {"ide": ide_count, "heretic": her_count,
                                                                    "prompts": len(texts.get("refusal") or [])})
        ws.delete_patch(res["patch"])

        # ---- stage 4b: the same fit from Heretic's own captures (separates the fit from the capture)
        HC.use_heretic_io(ws, "oracle", info, heretic_io, n, d)
        stage(label, "fitting (Heretic's captures)")
        t0 = time.monotonic()
        res2 = session.apply(script, message=f"heretic comparison (ARA, {label}, Heretic's captures)", dry_run=False)
        r["timing"]["fit_heretic_captures_seconds"] = time.monotonic() - t0
        assert res2.get("ok") and res2.get("committed"), res2
        stage(label, "comparing tensors (from Heretic's captures)")
        if adapter is None:
            changed2, _ = changes(ws, base, merged)
        else:
            changed2, _, _ = FS.adapter_changes(ws, base, adapter, ws.patch(res2["patch"]), rounding=a.dtype != "float32")
        r["weights_from_heretic_io"] = changed2
        lc2 = torch.stack([ws.runner.logits(x, "current", tail=1)[-1].float() for x in kl_ids])
        r["behaviour"]["kl_ide_from_heretic_io"] = float(F.kl_div(F.log_softmax(lc2, -1), F.log_softmax(lb, -1),
                                                                  reduction="batchmean", log_target=True))

        # ---- stage 6 (part 2): the objective at the IDE's change from Heretic's captures
        for row in fits:
            if "tensor" not in row:
                continue
            W = base.get_tensor(row["tensor"]).float()
            bias_name = row["tensor"][:-len(".weight")] + ".bias"
            bias = base.get_tensor(bias_name).float() if bias_name in base else None
            (xg, yg), (xb, yb) = (heretic_io[("good", row["layer"], row["component"], row["module"])],
                                  heretic_io[("bad", row["layer"], row["component"], row["module"])])
            row["ide_from_heretic_io"] = objective(W, ws.read(row["tensor"], variant="current").float() - W, bias,
                                                   xg, yg, xb, yb, p, params["preserve_row_magnitudes"])
            del W
        ws.delete_patch(res2["patch"])
        r["objective"] = fits

        # ---- C4-3: the reduction of the objective, each tool from its own captures
        red = []
        for row in fits:
            if "own_captures" not in row:
                continue
            r_h = row["start"]["total"] - row["heretic"]["total"]
            r_i = row["own_captures"]["start"]["total"] - row["own_captures"]["ide"]["total"]
            red.append({"tensor": row["tensor"], "heretic": r_h, "ide": r_i,
                        "relative": abs(r_i - r_h) / abs(r_h) if r_h != 0 else (0.0 if r_i == 0 else math.inf)})
        r["objective_reduction"] = red
        med = statistics.median([x["relative"] for x in red]) if red else None
        rep.bounded(label, "C4-3 objective reduction", B("C4-3 objective reduction"), med,
                    lambda v, b: v is not None and v <= b, matrices=len(red),
                    worst=max((x["relative"] for x in red), default=None))
        r["checks"] = rep.labels.get(label, {})
        result["runs"][label] = r
        result["pass"] = rep.ok()
        FS.write_json(out, result)

    result["pass"] = rep.ok()
    FS.write_json(out, result)
    for label, r in result["runs"].items():
        print(f"== {label} ({a.dtype}, Heretic's {r['heretic_export']}, modules by {r['matched_by']})   {r['parameters']}")
        print("  tokens identical:", {k: f"{v['identical']}/{v['prompts']}" for k, v in r["tokens"].items()})
        print("  means worst rel diff: good %.2e  bad %.2e" % (r["means"]["good_rel_worst"], r["means"]["bad_rel_worst"]))
        worst = [max((row.get(f"{s}_{x}_rel", 0.0) for row in r["module_io"] for s in ("good", "bad")), default=0.0)
                 for x in "xy"]
        print("  module I/O worst rel diff: inputs %.2e  outputs %.2e   (%d of %d modules captured by both)" % (
            worst[0], worst[1], sum("good_x_rel" in row for row in r["module_io"]), len(r["module_io"])))
        for own, other in zip(r["weights"]["changed"], r["weights_from_heretic_io"]):
            print("  %-40s change rel diff %.3e (own captures; %d of %d elements differ)  %.3e "
                  "(Heretic's captures; %d differ)  change/weight %s" % (
                      own["tensor"], own.get("rel_diff", math.nan), own.get("elements_that_differ", -1), own.get("elements", -1),
                      other.get("rel_diff", math.nan), other.get("elements_that_differ", -1), own.get("change_over_weight")))
        for f in r["objective"]:
            if "start" in f:
                print("  L%d %-13s %-22s objective: start %.9g  heretic %.9g  ide %.9g  ide from Heretic's captures %.9g" % (
                    f["layer"], f["component"], f["module"] or "", f["start"]["total"], f["heretic"]["total"],
                    f["ide"]["total"], f["ide_from_heretic_io"]["total"]))
        print("  tensors unchanged in both: %d   changed in one tool only: %d" % (r["weights"]["unchanged_in_both"], r["weights"]["changed_in_one_only"]))
        b = r["behaviour"]
        print("  base logits max abs diff %.2e (scale %.2f)   KL ide %.6g  ide from Heretic's captures %.6g  heretic %.6g" % (
            b["base_logits_max_abs_diff"], b["base_logits_scale"], b["kl_ide"], b["kl_ide_from_heretic_io"], b["kl_heretic"]))
        print("  timing (C4-6):", {k: v for k, v in r["timing"].items()})
    print(f"{'ok' if result['pass'] else 'NOT OK'}: {out}", flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
