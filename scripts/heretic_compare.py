"""Compare the IDE's behaviour directions with finished Heretic runs on the same model, stage by stage.

Run with the IDE's interpreter, with the IDE's `weight-space-ide` directory as the current directory, offline and
under the memory guard:

    python scripts/heretic_compare.py [--dtype float32|bfloat16] [--heretic-adapter [<mode>=]<dir> ...] \\
        [--expert-modules all|shared] <model snapshot dir> <isolated IDE root> <out.json> <mode>=<heretic run dir> ...

Each Heretic run directory holds the dump written by heretic_dump.py (dump/dump.json, dump/dump.safetensors) and
either Heretic's merged model (model/, export_strategy "merge") or its adapter (export_strategy "adapter": the
directory given with --heretic-adapter, else the first of <run>/adapter, <run>/<save_directory>, <run>/model that
holds adapter_config.json; refused unless r == lora_alpha). ``mode`` is the run's row_normalization (none, pre,
full). Stages: token ids, mean residuals, directions, weight changes per tensor (from the IDE's own measurement,
then from Heretic's means), base logits and the KL score; with --dtype bfloat16 also the refusals of the IDE's own
edit (C4-4).

--expert-modules is the IDE ablation's expert_modules (default "all", the IDE's default, as before); C4-1 then
compares Heretic's modules with the IDE's modules for that value. On a mixture-of-experts model under transformers
5.x Heretic changes only the shared expert (its module discovery skips the fused routed experts), so "shared" is the
IDE edit that corresponds to Heretic's there; with "all" the routed experts' tensors are changed by the IDE only.

--dtype sets WSP_RUN_DTYPE before the workspace's runner exists (default float32, today's comparison) and selects
the row of BOUNDS below for every stage. Modules are matched by Heretic's module names (dump.json "modules")
through wsp.runtime_layouts.module_path; a dump without them is matched by index, as before. A model of several
weight files (model.safetensors.index.json) is read one tensor at a time.

With an adapter, Heretic's change of a tensor is B @ A in float32 (scaling 1), and the IDE's is its patch's
float32 delta; B-S4 compares the two from Heretic's means, and then round_bf16(W + delta_IDE) against
round_bf16(W + delta_Heretic) element by element (both sums computed here, one rounding each). The rounded rule,
per touched tensor, with a and b the two rounded tensors, sp(x) the bfloat16 spacing at |x| and s_m the spacing at
the median of |b|: every element |a - b| <= max(sp(a), sp(b), s_m), and at most 1% of the elements with a != b
(formats_shared.rounded_sums, one_step_ok). An element at or above the median magnitude may flip by one of its own
rounding steps; a smaller one may differ by at most one step of the median magnitude, since near zero the own step
is far smaller than the difference of the deltas. Corrected twice before being applied: after the first bfloat16
run, because the first rule (every element equal or one own step apart) cannot hold for elements near zero; then,
after a synthetic check, because one step at the median magnitude for every element cannot hold for elements above
the median. The own-step counts and the distance in median steps are still reported, with no bound.

For row_normalization "full" with an adapter, the stage "B-S4 replica of Heretic's randomized approximation"
(formats_shared.replica_rows, replica_check; standalone and model-free: scripts/heretic_replica_check.py) forms
the full change D from the IDE's closed form, checks that its exact rank-r truncation is the IDE's delta (relative
1e-6), runs Heretic's three lines (manual_seed(seed); svd_lowrank(D, q = 2r + 4, niter = 6); B = U_r sqrt(S_r),
A = sqrt(S_r) V_r^T) on D in float32 with the run's seed and rank, and reports replica_vs_heretic, replica_vs_ide,
ide_vs_heretic and D's spectral tail. Heretic runs those lines on the device of the module's weight, so the replica
runs on the run's device (dump.json "device", by formats_shared.device_kind) when this process has it (mps:
torch.backends.mps.is_available()), else on the CPU (formats_shared.replica_device); distances are computed on the
CPU. For a bfloat16 run replica_vs_heretic is bounded by 1e-5, asserted only under the run's own torch version and
with the replica on the run's device (formats_shared.replica_applies; else reported with the reason: svd_lowrank's
random stream is the device's, and float32 products differ between devices). Basis: the same algorithm, seed and
float32 input up to the 1e-6
agreement of the unnormalised change, so the replica should reproduce Heretic's stored product to rounding; if it
does, the whole of ide_vs_heretic is the randomized method's distance from the best rank-r approximation. Added
after the first two bfloat16 runs on the 0.5B model, to test that explanation of the failed 4e-4 bound of "B-S4
change from Heretic's means"; that bound is unchanged.

Bounds (docs: the formats build spec, 7.3 and 10.3; None = reported only; a stage outside its bound is a finding,
not a threshold to move). Every assertion prints PASS or FAIL; <out.json> holds every number and a pass flag per
assertion; the exit status is 1 when one fails.
"""
import argparse
import json
import math
import os
import sys
from importlib.metadata import version
from pathlib import Path

# ---- BOUNDS[(stage, dtype)]: float32 rows are today's (this script asserted nothing before), bfloat16 rows 7.3/10.3
BOUNDS = {
    ("B-S1 token ids", "float32"): None,
    ("B-S1 token ids", "bfloat16"): "identical",
    ("B-S2 mean residuals", "float32"): None,
    ("B-S2 mean residuals", "bfloat16"): None,          # reported; expected relative difference below 2e-2
    ("B-S3 directions", "float32"): None,
    ("B-S3 directions", "bfloat16"): None,              # reported; expected cosine at least 0.99 at indices 1..n
    ("B-S4 change from Heretic's means", "float32"): None,
    ("B-S4 change from Heretic's means", "bfloat16"): {"none": 1e-6, "pre": 1e-6, "full": 4e-4},   # relative
    ("B-S4 rounded sums", "float32"): None,
    # every element |a - b| <= max(own bfloat16 step of a or b, step at the tensor's median |b|); at most 1% of the
    # tensor's elements with a != b
    ("B-S4 rounded sums", "bfloat16"): {"max_difference_in_allowed_steps": 1, "differing_fraction": 0.01},
    # row_normalization "full" with an adapter: Heretic's randomized rank-r call replicated on the IDE's D, against
    # Heretic's stored B A; asserted only under the run's own torch, on the run's device (formats_shared.replica_check)
    ("B-S4 replica of Heretic's randomized approximation", "float32"): None,
    ("B-S4 replica of Heretic's randomized approximation", "bfloat16"): {"replica_vs_heretic": 1e-5},
    ("C4-1 modules and tensors", "float32"): None,
    ("C4-1 modules and tensors", "bfloat16"): "equal sets",
    ("C4-4 refusals of the IDE's own edit", "float32"): None,
    ("C4-4 refusals of the IDE's own edit", "bfloat16"): 2,   # |IDE - Heretic| refusal count, of the scored prompts
}
EXPECTED = {"B-S2 mean residuals": {"relative_below": 2e-2}, "B-S3 directions": {"cosine_at_least": 0.99}}
CHUNK = 1 << 24                 # values compared per step (64 MB as float32)


def parse(argv):
    p = argparse.ArgumentParser(prog="heretic_compare.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--heretic-adapter", action="append", default=[], metavar="[MODE=]DIR")
    p.add_argument("--expert-modules", choices=("all", "shared"), default="all")
    p.add_argument("snapshot")
    p.add_argument("root")
    p.add_argument("out")
    p.add_argument("runs", nargs="+", metavar="MODE=RUN")
    a = p.parse_args(argv)
    runs = dict(r.split("=", 1) for r in a.runs)
    return a, runs


def ide_module_tensors(ws, expert_modules="all"):
    """The tensor of every module the IDE edits with ``expert_modules``, by layer and component (part 1's DR.modules
    when present)."""
    from wsp import directions as DR
    out_ = set()
    for l in range(ws.struct.n_layers):
        for comp in DR.COMPONENTS:
            if hasattr(DR, "modules"):
                for m in DR.modules(ws, l, comp, expert_modules):
                    views = m.get("views") or []
                    if views and getattr(views[0], "name", None):
                        out_.add(views[0].name)
            else:
                views = DR.component_views(ws, l, comp)
                if views:
                    out_.add(views[0].name)
    return out_


def ide_edit_params(trial_parameters, mode, ortho=True, expert_modules="all"):
    """The IDE ablation's parameters for Heretic's exported trial; expert_modules only when it is not the default."""
    params = {"direction": "oracle", **trial_parameters, "orthogonalize_direction": ortho,
              "row_normalization": mode, "full_normalization_lora_rank": 3}
    if expert_modules != "all":
        params["expert_modules"] = expert_modules
    return params


def main(argv):
    a, runs = parse(argv)
    sys.path.insert(0, os.getcwd())
    import formats_shared as FS
    FS.set_run_dtype(a.dtype)                    # before the runner exists

    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import load_file

    from wsp import directions as DR
    from wsp import directions_scores as DS
    from wsp.session import Session

    adapters = FS.parse_adapters(a.heretic_adapter, list(runs))
    snap, root, out = Path(a.snapshot), Path(a.root), Path(a.out)
    rel = FS.rel

    def cos(x, y):
        return float(F.cosine_similarity(x.double(), y.double(), dim=-1).min())

    def stage(*what):
        print("[compare]", *what, file=sys.stderr, flush=True)

    def changes(ws, base, merged):
        """Heretic's change against the IDE's, tensor by tensor and in row chunks of at most CHUNK values, so that no
        tensor is ever whole in memory (an embedding read whole from the checkpoint, the merged model and the
        workspace at once is several GiB within a second, which a memory guard rightly stops). ``base`` and
        ``merged`` are open safetensors files; everything is compared in float32 (a bfloat16 checkpoint converts
        exactly). -> (rows of changed tensors, how many were unchanged in both tools)."""
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

    def max_response_length(run):
        sc = run / "dump/scores.json"
        if sc.is_file():
            return json.loads(sc.read_text())["max_response_length"]
        import tomllib
        return tomllib.loads((run / "config.toml").read_text()).get("max_response_length", 100)

    rep = FS.Report()
    session = Session(str(root))
    ws = session.open(str(snap), log=lambda *a_, **k: None)
    n, d = ws.struct.n_layers, ws.struct.d_model
    base = FS.Checkpoint(snap)                    # read one tensor at a time, never a whole file
    result = {"model": {"layers": n, "d_model": d, "tensors": len(base.keys()),
                        "parameters": sum(math.prod(base.get_slice(k).get_shape()) for k in base.keys())},
              "ide_versions": {p: version(p) for p in ("torch", "transformers", "tokenizers", "safetensors")},
              "dtype": a.dtype, "expert_modules": a.expert_modules, "bounds": {f"{k[0]}": v for k, v in BOUNDS.items() if k[1] == a.dtype},
              "expected": EXPECTED, "runner": None, "modes": {}}
    B = lambda s: FS.bound(BOUNDS, s, a.dtype)  # noqa: E731

    for mode, run in runs.items():
        run = Path(run)
        dump = json.loads((run / "dump/dump.json").read_text())
        tens = load_file(str(run / "dump/dump.safetensors"))
        r = {"heretic_versions": dump["versions"], "heretic_dtype": dump["dtype"], "heretic_device": dump["device"]}
        assert dump["layers"] == n, (dump["layers"], n)
        system, prefix = dump["system_prompt"], dump["response_prefix"] or None
        texts = {k: [p["user"] for p in v] for k, v in dump["prompts"].items()}
        assert all(p["system"] == system for v in dump["prompts"].values() for p in v)
        merged_file = run / "model/model.safetensors"
        adapter_dir = None if merged_file.is_file() and mode not in adapters else FS.find_adapter(run, adapters.get(mode))
        r["heretic_export"] = "adapter" if adapter_dir else "merged"
        if adapter_dir is None and not merged_file.is_file():
            rep.check(mode, "Heretic's export found", False, error=f"{run}: neither model/model.safetensors nor an adapter")
            continue

        # ---- stage 1: token ids (the prompt as each tool feeds it to the model)
        checked = DR.check_measure(ws, "oracle", texts["good"], texts["bad"], chat=True, system=system, prefix=prefix,
                                   variant="base", replace=True)
        ids = DR._rendered_ids(ws, checked)
        kl_checked = DR.check_measure(ws, "oracle-kl", texts["kl"], texts["kl"], chat=True, system=system,
                                      prefix=prefix, variant="base", replace=True)
        kl_ids = DR._rendered_ids(ws, kl_checked)["good"]
        r["tokens"] = {k: {"prompts": len(ids[k]), "identical": sum(x == y for x, y in zip(ids[k], dump["ids"][k])),
                           "lengths_ide": [len(x) for x in ids[k]], "lengths_heretic": [len(x) for x in dump["ids"][k]]}
                       for k in ("good", "bad")}
        r["tokens"]["kl"] = {"prompts": len(kl_ids), "identical": sum(x == y for x, y in zip(kl_ids, dump["ids"]["kl"]))}
        all_same = all(v["identical"] == v["prompts"] == len(dump["ids"][k]) for k, v in r["tokens"].items())
        rep.bounded(mode, "B-S1 token ids", B("B-S1 token ids"), all_same, lambda v, b: v,
                    counts={k: f"{v['identical']}/{v['prompts']}" for k, v in r["tokens"].items()})

        # ---- stage 2: mean residuals, [n + 1, d]
        stage(mode, "measuring")
        DR.measure(ws, "oracle", texts["good"], texts["bad"], chat=True, system=system, prefix=prefix, variant="base",
                   replace=True)
        result["runner"] = DR.runner_description(ws)
        _, good, bad = DR.load(ws, "oracle")
        hg, hb = tens["good_mean"], tens["bad_mean"]
        assert tuple(hg.shape) == tuple(good.shape) == (n + 1, d), (hg.shape, good.shape)
        r["means"] = {"good_rel_per_index": [rel(good[k], hg[k]) for k in range(n + 1)],
                      "bad_rel_per_index": [rel(bad[k], hb[k]) for k in range(n + 1)],
                      "good_max_abs": float((good - hg).abs().max()), "bad_max_abs": float((bad - hb).abs().max()),
                      "typical_norm": float(hg.norm(dim=-1).median())}
        worst_mean = max(r["means"]["good_rel_per_index"] + r["means"]["bad_rel_per_index"])
        rep.note(mode, "B-S2 mean residuals", {"worst_relative": worst_mean,
                                               "expected_below": EXPECTED["B-S2 mean residuals"]["relative_below"]})

        # ---- stage 3: directions (Heretic's formula on Heretic's means, against the IDE's table on the IDE's means)
        ortho = True
        hr = F.normalize(hb - hg, p=2, dim=1)
        hgood = F.normalize(hg, p=2, dim=1)
        hr = F.normalize(hr - (hr * hgood).sum(1, keepdim=True) * hgood, p=2, dim=1)
        tab = DR.table(good, bad, ortho)
        r["directions"] = {"cosine_per_index": [cos(tab[k], hr[k]) for k in range(n + 1)],
                           "max_abs": float((tab - hr).abs().max())}
        rep.note(mode, "B-S3 directions", {"min_cosine_1_to_n": min(r["directions"]["cosine_per_index"][1:]),
                                           "expected_at_least": EXPECTED["B-S3 directions"]["cosine_at_least"]})

        # ---- stage 4: the weight change, tensor by tensor
        trial = next(t for t in dump["trials"] if t["index"] == dump["exported_trial_index"])
        params = ide_edit_params(trial["parameters"], mode, ortho, a.expert_modules)
        r["parameters"] = trial["parameters"]
        script = DR.script_for(params)
        stage(mode, "applying the ablation")
        res = session.apply(script, message=f"heretic comparison ({mode})", dry_run=False)
        assert res.get("ok") and res.get("committed"), res
        adapter = FS.Adapter(adapter_dir) if adapter_dir else None
        if adapter is None:
            merged = safe_open(str(merged_file), "pt")
            r["keys"] = {"base": len(base.keys()), "merged": len(merged.keys()),
                         "only_base": sorted(set(base.keys()) - set(merged.keys()))[:5],
                         "only_merged": sorted(set(merged.keys()) - set(base.keys()))[:5]}
            stage(mode, "comparing tensors")
            changed, same = changes(ws, base, merged)
        else:
            r["adapter"] = {"dir": str(adapter_dir), "r": adapter.config["r"], "lora_alpha": adapter.config["lora_alpha"],
                            "modules": len(adapter.modules), "dtypes": adapter.dtypes()}
            stage(mode, "comparing tensors (Heretic's adapter)")
            changed, same, unmatched = FS.adapter_changes(ws, base, adapter, ws.patch(res["patch"]), False)
            r["adapter"]["unmatched_modules"] = unmatched
        one_sided = [c for c in changed if "rel_diff" not in c]
        r["weights"] = {"changed": changed, "unchanged_in_both": same, "changed_in_one_only": len(one_sided),
                        "ide_stored_values": res.get("stored"), "ide_output": res.get("output", "")[-600:]}

        # ---- C4-1: the modules of both tools, and the tensors each changes
        mods = FS.dump_modules(dump)
        if mods is not None:
            names = set(base.keys())
            heretic_mods = [FS.tensor_of_module(ws.hf_config, nm, names) for layer in mods for v in layer.values() for nm in v]
            ide_mods = ide_module_tensors(ws, a.expert_modules)
            h_set = {t for t in heretic_mods if t}
            sets = {"heretic_modules": len(heretic_mods), "heretic_unmatched": heretic_mods.count(None),
                    "ide_modules": len(ide_mods), "only_heretic": sorted(h_set - ide_mods)[:5],
                    "only_ide": sorted(ide_mods - h_set)[:5], "tensors_changed_in_one_only": len(one_sided)}
            rep.bounded(mode, "C4-1 modules and tensors", B("C4-1 modules and tensors"),
                        h_set == ide_mods and not heretic_mods.count(None) and not one_sided, lambda v, b: v, **sets)
        else:
            rep.note(mode, "C4-1 modules and tensors", "the dump has no module names (an older dump); not compared")

        # ---- stage 5: behaviour: base logits, and KL(first token) of the edited model against the base
        stage(mode, "logits and KL")
        hl = tens["kl_logits"]
        lb = torch.stack([ws.runner.logits(x, "base", tail=1)[-1].float() for x in kl_ids])
        lc = torch.stack([ws.runner.logits(x, "current", tail=1)[-1].float() for x in kl_ids])
        kl = float(F.kl_div(F.log_softmax(lc, -1), F.log_softmax(lb, -1), reduction="batchmean", log_target=True))
        hkl = next(s["score"]["value"] for s in trial["scores"] if s["name"] == "KL divergence")
        # Heretic's merged model run by the IDE's own runner arithmetic is not available here; compare scores.
        r["behaviour"] = {"base_logits_max_abs_diff": float((lb - hl).abs().max()), "base_logits_scale": float(hl.abs().max()),
                          "kl_ide": kl, "kl_heretic": hkl,
                          "refusals_heretic": next(s["score"]["value"] for s in trial["scores"] if s["name"] == "Refusals")}

        # ---- C4-4: refusals of the IDE's own edit (its measurement, the trial's parameters), against Heretic's count
        if a.dtype != "float32":
            stage(mode, "refusals of the IDE's own edit")
            refusal = texts.get("refusal") or []
            got = DS.score(ws, {"state": "current", "chat": True, "system": system, "prefix": prefix,
                                "max_response_length": max_response_length(run),
                                "scores": [{"plugin": "KeywordRate", "prompts": refusal}]})
            ide_count = got["scores"][0]["matches"]
            her_count = round(r["behaviour"]["refusals_heretic"] * len(refusal))
            r["behaviour"]["refusals_ide_count"], r["behaviour"]["refusals_heretic_count"] = ide_count, her_count
            rep.bounded(mode, "C4-4 refusals of the IDE's own edit", B("C4-4 refusals of the IDE's own edit"),
                        abs(ide_count - her_count), lambda v, b: v <= b, ide=ide_count, heretic=her_count,
                        prompts=len(refusal))
        ws.delete_patch(res["patch"])

        # ---- stage 4b: the same ablation from Heretic's own means (isolates the weight arithmetic from the measurement)
        meta, _, _ = DR.load(ws, "oracle")
        DR.save(ws, "oracle", hg, hb, meta, DR.prompts(ws, "oracle"), replace=True)
        res2 = session.apply(script, message=f"heretic comparison ({mode}, Heretic's means)", dry_run=False)
        assert res2.get("ok") and res2.get("committed"), res2
        stage(mode, "comparing tensors (from Heretic's means)")
        again = []
        if adapter is None:
            for c in changed:
                if "rel_diff" in c:
                    b = base.get_tensor(c["tensor"]).float()
                    again.append({"tensor": c["tensor"], "rel_diff": rel(ws.read(c["tensor"], variant="current").float() - b,
                                                                         merged.get_tensor(c["tensor"]).float() - b)})
        else:
            again, _, _ = FS.adapter_changes(ws, base, adapter, ws.patch(res2["patch"]), True)
        r["weights_from_heretic_means"] = again
        both = [c for c in again if "rel_diff" in c]
        worst = max((c["rel_diff"] for c in both), default=None)
        lim = B("B-S4 change from Heretic's means")
        rep.bounded(mode, "B-S4 change from Heretic's means", None if lim is None else lim.get(mode), worst,
                    lambda v, b: v is not None and v <= b and len(both) == len(again), tensors=len(again),
                    changed_in_one_only=len(again) - len(both))
        if adapter is not None:
            counts = [c["rounded"] for c in both]
            rule = BOUNDS[("B-S4 rounded sums", "bfloat16")]
            per_ok = [FS.one_step_ok(c, rule["differing_fraction"], rule["max_difference_in_allowed_steps"])
                      for c in counts]
            total = FS.merge_counts(counts)
            rep.bounded(mode, "B-S4 rounded sums", B("B-S4 rounded sums"), all(per_ok) and bool(counts),
                        lambda v, b: v, tensors=len(counts), tensors_outside=per_ok.count(False), **total,
                        tensors_outside_names=[c0["tensor"] for c0, ok in zip(both, per_ok) if not ok],
                        worst_one_step_fraction=max((c["one_step"] / c["elements"] for c in counts), default=None),
                        worst_differing_fraction=max((c["differing"] / c["elements"] for c in counts), default=None),
                        per_tensor=[{"tensor": c0["tensor"], **{k: c[k] for k in (
                            "median_abs", "step_at_median", "max_abs_difference",
                            "max_abs_difference_in_median_steps", "differing", "more_than_one",
                            "largest_abs_over_median", "outside", "max_excess")}} for c0, c in zip(both, counts)])
        if adapter is not None and mode == "full":
            # ---- B-S4 replica: Heretic's randomized rank-r call on the IDE's D, against Heretic's stored B A
            rep_device = FS.replica_device(dump.get("device"))   # the run's device when this process has it
            stage(mode, f"replicating Heretic's randomized approximation on {rep_device} (the run's device "
                        f"{FS.device_kind(dump.get('device'))})")
            rows_rep = FS.replica_rows(base, ws.hf_config, dump, hg, hb, adapter,
                                       lambda tensor, W, v, w, rk: ws.read(tensor, variant="delta").float(),
                                       device=rep_device)
            r["replica"] = {"rows": rows_rep, "device": rep_device, "run_device": FS.device_kind(dump.get("device")),
                            "summary": FS.replica_check(rep, mode, rows_rep, B(FS.REPLICA_STAGE), torch.__version__,
                                                        dump.get("versions"), rep_device, dump.get("device"))}
        ws.delete_patch(res2["patch"])
        r["difference_over_mean"] = [float((hb[k] - hg[k]).norm() / hg[k].norm()) for k in range(1, n + 1)]
        r["checks"] = rep.labels.get(mode, {})
        result["modes"][mode] = r
        result["pass"] = rep.ok()
        FS.write_json(out, result)

    result["pass"] = rep.ok()
    FS.write_json(out, result)
    for mode, r in result["modes"].items():
        print(f"== {mode} ({a.dtype}, Heretic's {r['heretic_export']})")
        print("  tokens identical:", {k: f"{v['identical']}/{v['prompts']}" for k, v in r["tokens"].items()})
        print("  means   worst rel diff: good %.2e  bad %.2e" % (max(r["means"]["good_rel_per_index"]), max(r["means"]["bad_rel_per_index"])))
        print("  directions min cosine, indices 1..n: %.9f" % min(r["directions"]["cosine_per_index"][1:]))
        rows = [c for c in r["weights"]["changed"] if "rel_diff" in c]
        if rows:
            print("  tensors changed in both: %d   worst rel diff of the change %.3e" % (
                len(rows), max(c["rel_diff"] for c in rows)))
        for c in r["weights"]["changed"]:
            if "rel_diff" not in c:
                print("  ", c)
        again = [c["rel_diff"] for c in r["weights_from_heretic_means"] if "rel_diff" in c]
        if again:
            print("  from Heretic's own means, worst rel diff: %.2e" % max(again))
        if "replica" in r:
            sm = r["replica"]["summary"]
            print("  replica of Heretic's randomized approximation on %s (the run's device %s): worst replica_vs_heretic"
                  " %s  replica_vs_ide %s" % (r["replica"]["device"], r["replica"]["run_device"],
                                             sm["worst_replica_vs_heretic"], sm["worst_replica_vs_ide"]))
        print("  |bad - good| / |good| per index:", ["%.3f" % x for x in r["difference_over_mean"]])
        print("  tensors unchanged in both: %d   changed in one tool only: %d" % (r["weights"]["unchanged_in_both"], r["weights"]["changed_in_one_only"]))
        print("  base logits max abs diff %.2e (scale %.2f)   KL ide %.6g  heretic %.6g" % (
            r["behaviour"]["base_logits_max_abs_diff"], r["behaviour"]["base_logits_scale"], r["behaviour"]["kl_ide"], r["behaviour"]["kl_heretic"]))
    print(f"{'ok' if result['pass'] else 'NOT OK'}: {out}", flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
