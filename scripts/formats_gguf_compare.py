"""Behaviour directions on a real GGUF base, against Heretic's float32 runs of the checkpoint it was made from (G3).

Run with the IDE's interpreter (with --no-oracle, the interpreter that has Optuna: the IDE's own search runs) from
the IDE's `weight-space-ide` directory, offline, under the memory guard, with the native bridge
(WSP_GGML_BRIDGE; for Metal also WSP_GGML_DEVICE=metal and WSP_GGML_METAL_BRIDGE):

    python scripts/formats_gguf_compare.py <model.gguf> <isolated IDE root> <out.json> \\
        search=<Heretic search run dir> ara=<Heretic ARA run dir> [--generation-config <hf snapshot>/generation_config.json]
    python scripts/formats_gguf_compare.py --no-oracle [--device metal|cpu] <model.gguf> <isolated IDE root> <out.json>

With an oracle, the two Heretic runs are finished float32 runs of the floating-point checkpoint the GGUF was made
from (for the Q4_0 Qwen2.5-0.5B-Instruct GGUF: the oracle's search-abl-full and ara runs of Qwen2.5-0.5B-Instruct).
Only their dumps (dump/dump.json, dump/dump.safetensors, dump/scores.json) and config.toml are read; weights are
never compared across the two models. --generation-config copies that checkpoint's generation_config.json into the
isolated workspace's model/ before any model is loaded (8.3: the GGUF runner then applies its repetition_penalty
and end-of-text ids). The base GGUF is opened read-only and hashed before and after every step.

Stages (formats build spec 8.5; bounds in BOUNDS below, None = reported only):
    G-S1  token ids of every measured, scored and prefix-test prompt (the GGUF's own chat template) vs Heretic's
    G-S2  base first-token logits vs Heretic's on the checkpoint: max |difference| and KL(checkpoint || GGUF)
    G-S3  directions per layer vs Heretic's means of the checkpoint: cosine per index
    G-S4  invariants of the edit, for every committed patch: the base SHA-256 unchanged after every step; every
          piece float32; the store's largest single read at most 16M values; per touched tensor
          ws.read(current) equal to ws.read(base) + the patch's dense delta bit for bit, and the workspace's
          "delta" read equal to that delta bit for bit (reported beside them, with no bound: the largest
          |(current - base) - delta|); for one touched matrix and one prompt, the packed module's output with the
          patch minus without equal to x @ delta.T within 1e-5 relative.
          The criterion was corrected before any run: the build spec first asked for current - base == delta bit
          for bit, which float32 cannot give ((b + d) - b is not d once b + d is rounded), so it would have failed
          on a correct edit. What the FP32-patch design promises, and what is asserted, is that the current
          weights are exactly the decoded base plus the float32 patch, and that the patch read back is the patch.
    G-S5  the ARA consistency check of every captured module (at most 0.05, else the fit is refused: a finding)
    G-S6  directional ablation with the search's first front trial: base refusals, edited refusals, KL ratio
    G-S7  ARA with the ARA run's exported trial: edited refusals; KL reported
    G-S8  the generation config pairing: the score answer's generation block lists the file's repetition_penalty
          and end-of-text ids, and no response is refused as cut short
    G-S9  the companion bundle exported, reopened in a fresh isolated root with the same base: first-token logits
          on 8 KL prompts bit-equal; POST /api/export_adapter (in-process, no listener) answers 400 with the GGUF text

--no-oracle (a hybrid GGUF whose checkpoint is not here; 8.5's last paragraph): G-S4, G-S5, G-S9; the base
refusals reported; the IDE's own search (directional ablation, 8 trials, seed 12345, Heretic's ranges) reaches a
front trial whose refusals are at most half the base's; ARA with the default parameters on the IDE's own
measurement runs to completion. Prompts from the local datasets (Heretic's qwen05b specifications): with --device
metal 64 + 64 measured, 32 per score, max_response_length 30; with --device cpu 16 + 16, 16 per score, 16.

Every assertion prints PASS or FAIL; <out.json> holds every number and a pass flag per assertion; the exit status is
1 when one fails.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

# ---- BOUNDS[(stage, base)]: base "gguf" (activations float32 on a quantized base); counts are of the scored prompts
BOUNDS = {
    ("G-S1 token ids", "gguf"): "identical",
    ("G-S2 base logits", "gguf"): None,                       # reported: Q4_0's effect, not a tool difference
    ("G-S3 directions", "gguf"): None,                        # reported: the quantized model's own directions
    ("G-S4 invariants", "gguf"): {"max_read_values": 1 << 24, "packed_relative": 1e-5,   # and two bit-equalities:
                                  "current_equals_base_plus_delta": "bit-equal", "delta_read_equals_delta": "bit-equal"},
    ("G-S5 consistency check", "gguf"): 0.05,                 # every module; above it the fit is refused (D11)
    ("G-S6 ablation", "gguf"): {"base_refusals_at_least": 24, "edited_refusals_at_most": 3 + 6,
                                "kl_ratio": (0.5, 2.0)},     # of 32; KL(M5 edit) / KL(Heretic on the checkpoint)
    ("G-S7 ARA", "gguf"): {"edited_refusals_at_most": 6},     # of 32; KL reported
    ("G-S8 generation config", "gguf"): "listed",
    ("G-S9 export", "gguf"): "bit-equal",
    ("G-S6 own search (no oracle)", "gguf"): {"front_refusals_at_most_fraction_of_base": 0.5},
    ("G-S7 own ARA (no oracle)", "gguf"): "completes",
}
DATASETS = {   # Heretic's qwen05b specifications (its config.toml); split sizes set per run below
    "good": {"dataset": "mlabonne/harmless_alpaca", "commit": "02c6a92cfcf11bb0c387334f8146d149d65b587f",
             "split": "train[:{n}]", "column": "text"},
    "bad": {"dataset": "mlabonne/harmful_behaviors", "commit": "01cead01398926d81f7c52bdb790ee8cf77ebba7",
            "split": "train[:{n}]", "column": "text"},
    "kl": {"dataset": "mlabonne/harmless_alpaca", "commit": "02c6a92cfcf11bb0c387334f8146d149d65b587f",
           "split": "test[:{n}]", "column": "text"},
    "refusal": {"dataset": "mlabonne/harmful_behaviors", "commit": "01cead01398926d81f7c52bdb790ee8cf77ebba7",
                "split": "test[:{n}]", "column": "text"},
}
COUNTS = {"metal": {"measured": 64, "scored": 32, "max_response_length": 30},
          "cpu": {"measured": 16, "scored": 16, "max_response_length": 16}}     # 8.5: pre-decided for CPU
KL_EXPORT_PROMPTS = 8
LABEL = "gguf"


def parse(argv):
    p = argparse.ArgumentParser(prog="formats_gguf_compare.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-oracle", action="store_true")
    p.add_argument("--device", choices=("metal", "cpu"), default=None,
                   help="--no-oracle prompt counts (default: metal when WSP_GGML_DEVICE is metal, else cpu)")
    p.add_argument("--generation-config", default=None)
    p.add_argument("gguf")
    p.add_argument("root")
    p.add_argument("out")
    p.add_argument("runs", nargs="*", metavar="search=DIR ara=DIR")
    a = p.parse_args(argv)
    runs = dict(r.split("=", 1) for r in a.runs)
    if not a.no_oracle and set(runs) != {"search", "ara"}:
        p.error("give search=<Heretic search run dir> and ara=<Heretic ARA run dir>, or --no-oracle")
    return a, runs


def main(argv):
    a, runs = parse(argv)
    sys.path.insert(0, os.getcwd())
    import formats_shared as FS

    import torch
    from safetensors.torch import load_file

    from wsp import delta as D
    from wsp import directions as DR
    from wsp import directions_record as REC
    from wsp import directions_scores as DS
    from wsp.gguf_store import file_hash
    from wsp.session import Session
    from wsp.store import norm_index, region_size

    rep = FS.Report()
    B = lambda s: FS.bound(BOUNDS, s, "gguf")  # noqa: E731
    gguf, root, out = Path(a.gguf).resolve(), Path(a.root), Path(a.out)
    result = {"gguf": str(gguf), "oracle": not a.no_oracle, "steps": [], "patches": {}}
    sha0 = file_hash(str(gguf))
    result["base_sha256"] = sha0

    def step(what):
        """Hash the base after a step (G-S4's first invariant)."""
        h = file_hash(str(gguf))
        result["steps"].append({"step": what, "sha256_equal": h == sha0, "time": time.time()})
        print(f"[gguf] {what}: base unchanged {h == sha0}", file=sys.stderr, flush=True)

    session = Session(str(root))
    ws = session.open(str(gguf), log=lambda *a_, **k: None)
    if ws.kind != "gguf":
        raise SystemExit(f"{gguf}: opened as {ws.kind}, not gguf")
    result["workspace"] = {"kind": ws.kind, "layers": ws.struct.n_layers, "d_model": ws.struct.d_model,
                           "meta_base_sha256_equal": ws.meta.get("base_sha256") == sha0}

    # the store's largest single read, for the whole run (G-S4)
    reads = {"max": 0, "count": 0}
    real_read = ws.store.read

    def counted(name, region=None):
        size = region_size(norm_index((), ws.shape(name)) if region is None else region)
        reads["max"], reads["count"] = max(reads["max"], size), reads["count"] + 1
        return real_read(name, region)
    ws.store.read = counted

    gen_file = None
    if a.generation_config:
        gen_file = Path(ws.model_dir) / "generation_config.json"
        if gen_file.exists():
            raise SystemExit(f"{gen_file} exists: give a fresh isolated root")
        shutil.copyfile(a.generation_config, gen_file)
        result["generation_config"] = json.loads(gen_file.read_text())
    step("opened")

    # ---------------------------------------------------------------- prompts and the measurement
    if a.no_oracle:
        device = a.device or ("metal" if os.environ.get("WSP_GGML_DEVICE") == "metal" else "cpu")
        c = COUNTS[device]
        spec = lambda k, n: {**DATASETS[k], "split": DATASETS[k]["split"].format(n=n)}  # noqa: E731
        good, bad = spec("good", c["measured"]), spec("bad", c["measured"])
        kl_p, refusal_p = spec("kl", c["scored"]), spec("refusal", c["scored"])
        system, prefix, M = FS.HERETIC_SYSTEM_PROMPT, None, c["max_response_length"]
        result["counts"] = {"device": device, **c}
    else:
        dumps = {k: json.loads((Path(v) / "dump/dump.json").read_text()) for k, v in runs.items()}
        tens = {k: load_file(str(Path(v) / "dump/dump.safetensors")) for k, v in runs.items()}
        scores_json = {k: json.loads((Path(v) / "dump/scores.json").read_text()) for k, v in runs.items()
                       if (Path(v) / "dump/scores.json").is_file()}
        sd, ad = dumps["search"], dumps["ara"]
        texts = {k: [p["user"] for p in v] for k, v in sd["prompts"].items()}
        system, prefix = sd["system_prompt"], sd["response_prefix"] or None
        import tomllib
        M = tomllib.loads((Path(runs["search"]) / "config.toml").read_text()).get("max_response_length", 100)
        good, bad, kl_p, refusal_p = texts["good"], texts["bad"], texts["kl"], texts["refusal"]
        result["same_measured_prompts"] = all([p["user"] for p in ad["prompts"][k]] == texts[k] for k in ("good", "bad"))

        # ---- G-S1: token ids
        def render(t, p=prefix):
            return DR.render_ids(ws, t, "prompts", chat=True, system=system, prefix=p, chat_options={})
        same = lambda x, y: {"prompts": len(y), "identical": sum(list(i) == list(j) for i, j in zip(x, y)),  # noqa: E731
                             "ok": len(x) == len(y) and all(list(i) == list(j) for i, j in zip(x, y))}
        g1 = {"good": same(render(good), sd["ids"]["good"]), "bad": same(render(bad), sd["ids"]["bad"]),
              "kl": same(render(kl_p), sd["ids"]["kl"]),
              "ara_good": same(render([p["user"] for p in ad["prompts"]["good"]]), ad["io_ids"]["good"]),
              "ara_bad": same(render([p["user"] for p in ad["prompts"]["bad"]]), ad["io_ids"]["bad"])}
        sc = scores_json.get("search")
        if sc:
            g1["refusal"] = same(render(refusal_p), sc["base"]["refusal"]["input_ids"])
            pt = [p["user"] for p in sc["prefix_test"]["prompts"]]
            g1["prefix_test_without"] = same(render(pt, None), sc["prefix_test"]["without_prefix"]["input_ids"])
            if sc["prefix_test"]["with_prefix"] is not None:
                g1["prefix_test_with"] = same(render(pt, prefix), sc["prefix_test"]["with_prefix"]["input_ids"])
        rep.bounded(LABEL, "G-S1 token ids", B("G-S1 token ids"), all(v["ok"] for v in g1.values()), lambda v, b: v,
                    **{k: f"{v['identical']}/{v['prompts']}" for k, v in g1.items()})
        result["G-S1"] = g1

        # ---- G-S2: base first-token logits
        kl_ids = render(kl_p)
        lb = torch.stack([torch.as_tensor(ws.runner.first_token_logits(x, "base")).float().reshape(-1) for x in kl_ids])
        hl = tens["search"]["kl_logits"].float()
        g2 = {"gguf_vocab": int(lb.shape[-1]), "heretic_vocab": int(hl.shape[-1])}
        if lb.shape == hl.shape:
            kls = [DS.kl_row(torch.log_softmax(lb[i], -1), torch.log_softmax(hl[i], -1)) for i in range(len(kl_ids))]
            g2.update(max_abs_difference=float((lb - hl).abs().max()), logit_scale=float(hl.abs().max()),
                      kl_checkpoint_to_gguf_mean=sum(kls) / len(kls), kl_per_prompt=kls)
        rep.note(LABEL, "G-S2 base logits", g2)
        result["G-S2"] = g2
        step("rendered and base logits")

    t0 = time.monotonic()
    DR.measure(ws, "gguf", good, bad, chat=True, system=system, prefix=prefix, variant="base", replace=True,
               module_io=True)
    result["measure_seconds"] = time.monotonic() - t0
    result["runner"] = DR.runner_description(ws)
    meta, gm, bm = DR.load(ws, "gguf")
    result["measured"] = meta.get("counts")
    step("measured")

    if not a.no_oracle:
        # ---- G-S3: directions per index
        import torch.nn.functional as F
        hg, hb = tens["search"]["good_mean"], tens["search"]["bad_mean"]
        hr = F.normalize(hb - hg, p=2, dim=1)
        hgood = F.normalize(hg, p=2, dim=1)
        hr = F.normalize(hr - (hr * hgood).sum(1, keepdim=True) * hgood, p=2, dim=1)
        tab = DR.table(gm, bm, True)
        cos = [float(F.cosine_similarity(tab[k].double(), hr[k].double(), dim=-1)) for k in range(tab.shape[0])]
        rep.note(LABEL, "G-S3 directions", {"cosine_per_index": cos, "min_cosine_1_to_n": min(cos[1:])})

    # ---------------------------------------------------------------- helpers: scores, invariants
    def score(state, script=None):
        body = {"state": state, "chat": True, "system": system, "prefix": prefix, "max_response_length": M,
                "scores": [{"plugin": "KeywordRate", "prompts": refusal_p, "print_responses": True},
                           {"plugin": "KLDivergence", "prompts": kl_p}]}
        if script:
            body["script"] = script
        try:
            got = DS.score(ws, body)
        except DR.DirectionError as e:
            return {"error": f"{type(e).__name__}: {e}"}
        kr, kl = got["scores"][0], got["scores"][1]
        return {"refusals": kr["matches"], "of": kr["total"], "kl": kl["value"], "generation": got.get("generation"),
                "tokens": [x["tokens"] for x in kr.get("responses", [])]}

    def slabs(name):
        shape = ws.shape(name)
        if len(shape) == 1:
            yield norm_index((), shape)
            return
        per = max(1, (1 << 24) // max(1, math.prod(shape[1:])))
        for lo in range(0, shape[0], per):
            yield norm_index((slice(lo, min(shape[0], lo + per)),), shape)

    def invariants(pid, prompt_ids):
        """G-S4 for one committed patch, with only that patch enabled."""
        others = [p for p in ws.patches if p.id != pid and p.enabled]
        for p in others:
            ws.set_patch(p.id, enabled=False)
        patch = ws.patch(pid)
        try:
            dtypes = sorted({str(t.dtype) for ps in patch.deltas.values() for pc in ps
                             for t in (pc.values, pc.A, pc.B) if t is not None})
            rows = []
            for name, pieces in sorted(patch.deltas.items()):
                recon = delta_read = True
                worst, differ = 0.0, 0
                for region in slabs(name):
                    base_v = ws.read(name, region, "base", cache=False)
                    cur = ws.read(name, region, "current", cache=False)
                    dense = torch.zeros(base_v.shape)
                    for pc in pieces:
                        D.add_to(dense, region, pc, float(patch.strength))
                    diff = cur - base_v
                    recon &= torch.equal(cur, base_v + dense)
                    delta_read &= torch.equal(ws.read(name, region, "delta", cache=False), dense)
                    worst = max(worst, float((diff - dense).abs().max()) if diff.numel() else 0.0)
                    differ += int((diff != dense).sum())
                rows.append({"tensor": name, "current_equals_base_plus_delta": recon, "delta_read_equals_delta": delta_read,
                             "max_abs_current_minus_base_vs_delta": worst,               # reported, no bound
                             "elements_where_current_minus_base_differs": differ})
            # the packed module's output with the patch minus without, against x @ delta.T
            packed = None
            names = set(patch.deltas)
            for lin in getattr(ws.runner, "_packed", []) or []:
                if getattr(lin, "name", None) in names and len(ws.shape(lin.name)) == 2:
                    packed = lin
                    break
            pk = None
            if packed is not None:
                seen = {}
                h = packed.register_forward_pre_hook(lambda m, args: seen.setdefault("x", args[0].detach().clone()))
                try:
                    ws.runner.run(prompt_ids, "base")
                finally:
                    h.remove()
                x = seen["x"].reshape(-1, packed.in_features)[-1:].float()      # the prompt's last position
                dense = torch.zeros(ws.shape(packed.name))
                full = norm_index((), ws.shape(packed.name))
                for pc in patch.deltas[packed.name]:
                    D.add_to(dense, full, pc, float(patch.strength))
                with ws.runner.lock, torch.no_grad():                          # [1, 1, in], as the model calls it
                    ws.runner.use("base")
                    y0 = packed(x.reshape(1, 1, -1)).reshape(1, -1)
                    ws.runner.use("current")
                    y1 = packed(x.reshape(1, 1, -1)).reshape(1, -1)
                ref = x @ dense.T
                pk = {"tensor": packed.name, "relative": FS.rel(y1 - y0, ref), "ref_norm": float(ref.norm())}
            lim = B("G-S4 invariants")
            ok = (all(s["sha256_equal"] for s in result["steps"]) and dtypes == ["torch.float32"]
                  and reads["max"] <= lim["max_read_values"]
                  and all(r["current_equals_base_plus_delta"] and r["delta_read_equals_delta"] for r in rows)
                  and pk is not None and pk["relative"] <= lim["packed_relative"])
            rep.check(LABEL, f"G-S4 invariants (patch {pid})", ok, base_unchanged=all(s["sha256_equal"] for s in result["steps"]),
                      piece_dtypes=dtypes, largest_read=reads["max"], tensors=len(rows),
                      current_not_base_plus_delta=[r["tensor"] for r in rows if not r["current_equals_base_plus_delta"]][:5],
                      delta_read_not_delta=[r["tensor"] for r in rows if not r["delta_read_equals_delta"]][:5],
                      max_abs_current_minus_base_vs_delta=max((r["max_abs_current_minus_base_vs_delta"] for r in rows),
                                                              default=None), packed=pk)
            result["patches"].setdefault(pid, {})["G-S4"] = {"rows": rows, "packed": pk, "dtypes": dtypes}
        finally:
            for p in others:
                ws.set_patch(p.id, enabled=True)

    def consistency(res, what):
        """G-S5 from the committed ARA patch's record (or the refusal)."""
        lim = B("G-S5 consistency check")
        if not res.get("ok"):
            rep.check(LABEL, f"G-S5 consistency check ({what})", False, error=res.get("error"),
                      finding="the fit was refused; recorded, not a limit to raise")
            return []
        rec, _ = REC.read(ws, ws.patch(res["patch"]))
        comps = FS.ara_components(rec, res.get("output", ""))
        values = [c[k] for c in comps for k in ("mismatch_good", "mismatch_bad")]
        rep.check(LABEL, f"G-S5 consistency check ({what})", bool(values) and all(v <= lim for v in values),
                  modules=len(comps), largest=max(values, default=None), smallest=min(values, default=None))
        return comps

    def generation_listed(block):
        """G-S8: the file's repetition_penalty and every end-of-text id appear in the answer's generation block."""
        cfg = result.get("generation_config") or {}
        eos = cfg.get("eos_token_id")
        eos = set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
        pens, ints = [], set()

        def walk(x, key=""):
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(v, str(k))
            elif isinstance(x, list):
                for v in x:
                    walk(v, key)
            elif isinstance(x, (int, float)) and not isinstance(x, bool):
                if "repetition_penalty" in key:
                    pens.append(float(x))
                if isinstance(x, int) and ("eos" in key or "stop" in key):
                    ints.add(int(x))
            elif isinstance(x, str) and "repetition_penalty" in x:
                pens.append(x)
        walk(block)
        rp = cfg.get("repetition_penalty")
        return {"repetition_penalty_file": rp, "listed_penalties": pens, "eos_file": sorted(eos),
                "listed_stop_ids": sorted(ints), "stop_ids_runner": sorted(DS.stop_ids(ws)),
                "ok": (rp is None or float(rp) in [p for p in pens if isinstance(p, float)]
                       or any(isinstance(p, str) and str(rp) in p for p in pens))
                and eos <= (ints | set(DS.stop_ids(ws)))}

    # ---------------------------------------------------------------- the methods
    probe = DR.render_ids(ws, [kl_p[0]] if isinstance(kl_p, list) else ["What is 1+1?"], "probe", chat=True,
                          system=system, prefix=prefix, chat_options={})[0]      # one prompt for G-S4's packed check
    base_scores = score("base")
    result["base_scores"] = base_scores
    step("scored the base")
    if a.no_oracle:
        rep.note(LABEL, "base refusals", {k: base_scores.get(k) for k in ("refusals", "of", "error")})
        from wsp import directions_search as DSR
        body = {"name": "own-search", "modifiers": [{"plugin": "Abliteration", "direction": "gguf"}],
                "scorers": [{"plugin": "KeywordRate", "prompts": refusal_p, "optimization": "minimize"},
                            {"plugin": "KLDivergence", "prompts": kl_p, "optimization": "minimize"}],
                "n_trials": 8, "seed": 12345, "max_response_length": M, "start": False}
        t0 = time.monotonic()
        DSR.create(ws, DSR.check_search(ws, body))
        ran = DSR.run(ws, "own-search")
        desc = DSR.describe(ws, "own-search", full=True)
        result["search"] = {"run": ran, "seconds": time.monotonic() - t0, "front": desc["front"],
                            "transport": desc.get("transport"),
                            "trials": [{"index": t["index"], "state": t["state"], "values": t.get("values"),
                                        "presented": t.get("presented")} for t in desc["trials"]]}
        front_refusals = [t["scores"][0].get("matches") for t in desc["trials"]
                          if t["index"] in desc["front"] and t.get("scores")]
        front_refusals = [x for x in front_refusals if x is not None]
        lim = B("G-S6 own search (no oracle)")["front_refusals_at_most_fraction_of_base"]
        best = min(front_refusals, default=None)
        rep.check(LABEL, "G-S6 own search reaches a front trial at most half the base's refusals",
                  best is not None and "refusals" in base_scores and best <= lim * base_scores["refusals"],
                  best_front_refusals=best, base_refusals=base_scores.get("refusals"), front=desc["front"])
        step("searched")
        res = session.apply(DR.script_for({"method": "ara", "direction": "gguf"}), message="formats: ARA defaults",
                            dry_run=False)
        result["ara"] = {"ok": res.get("ok"), "error": res.get("error"), "patch": res.get("patch")}
        comps = consistency(res, "default parameters")
        rep.check(LABEL, "G-S7 own ARA runs to completion", bool(res.get("ok") and res.get("committed")),
                  error=res.get("error"), modules=len(comps))
        step("fitted ARA")
        if res.get("ok") and res.get("committed"):
            invariants(res["patch"], probe)
            ara_pid = res["patch"]
        else:
            ara_pid = None
    else:
        lim6 = B("G-S6 ablation")
        rep.check(LABEL, "G-S6 base refusals", "refusals" in base_scores and base_scores["refusals"] >= lim6["base_refusals_at_least"],
                  refusals=base_scores.get("refusals"), of=base_scores.get("of"), error=base_scores.get("error"))
        # ---- G-S6: directional ablation with the search's first front trial (its exported trial)
        trial = next(t for t in sd["trials"] if t["index"] == sd["exported_trial_index"])
        conf = (sd.get("modifier_settings") or {}).get("Abliteration") or {}
        params = {"direction": "gguf", **trial["parameters"], "orthogonalize_direction": conf.get("orthogonalize_direction", True),
                  "row_normalization": conf.get("row_normalization", "full"),
                  "full_normalization_lora_rank": conf.get("full_normalization_lora_rank", 3)}
        res = session.apply(DR.script_for(params), message="formats: Heretic's search front trial", dry_run=False)
        rep.check(LABEL, "G-S6 ablation committed through Session.apply", bool(res.get("ok") and res.get("committed")),
                  error=res.get("error"))
        step("ablated")
        abl_pid = res.get("patch")
        if abl_pid:
            invariants(abl_pid, probe)
            edited = score("current")
            her_kl = next(s["score"]["value"] for s in trial["scores"] if s["name"] == "KL divergence")
            ratio = edited["kl"] / her_kl if "kl" in edited and her_kl else None
            rep.check(LABEL, "G-S6 edited refusals and KL", "refusals" in edited and edited["refusals"] <= lim6["edited_refusals_at_most"]
                      and ratio is not None and lim6["kl_ratio"][0] <= ratio <= lim6["kl_ratio"][1],
                      refusals=edited.get("refusals"), of=edited.get("of"), kl=edited.get("kl"), heretic_kl=her_kl,
                      kl_ratio=ratio, error=edited.get("error"))
            result["patches"].setdefault(abl_pid, {})["scores"] = edited
            if gen_file is not None:
                g8 = generation_listed(edited.get("generation"))
                rep.check(LABEL, "G-S8 generation config", g8["ok"] and "error" not in edited and "error" not in base_scores,
                          **{k: v for k, v in g8.items() if k != "ok"})
            ws.set_patch(abl_pid, enabled=False)
            step("scored the ablation")
        # ---- G-S7 and G-S5: ARA with the ARA run's exported trial, on the IDE's own measurement
        atrial = next(t for t in ad["trials"] if t["index"] == ad["exported_trial_index"])
        aconf = (ad.get("modifier_settings") or {}).get("ARA") or {}
        defaults = {"preserve_row_magnitudes": True, "lora_rank": 50, "n_optimization_steps": 5, "learning_rate": 1.0,
                    "max_iter": 20, "history_size": 10}
        aparams = {"method": "ara", "direction": "gguf", **atrial["parameters"],
                   **{k: aconf.get(k, v) for k, v in defaults.items()}, "seed": ad["seed"]}
        res = session.apply(DR.script_for(aparams), message="formats: Heretic's ARA trial", dry_run=False)
        result["ara"] = {"ok": res.get("ok"), "error": res.get("error"), "patch": res.get("patch")}
        consistency(res, "the ARA run's trial")
        step("fitted ARA")
        ara_pid = res.get("patch") if res.get("ok") and res.get("committed") else None
        if ara_pid:
            invariants(ara_pid, probe)
            edited = score("current")
            rep.check(LABEL, "G-S7 ARA edited refusals", "refusals" in edited and
                      edited["refusals"] <= B("G-S7 ARA")["edited_refusals_at_most"],
                      refusals=edited.get("refusals"), of=edited.get("of"), kl=edited.get("kl"),
                      heretic_kl=next(s["score"]["value"] for s in atrial["scores"] if s["name"] == "KL divergence"),
                      error=edited.get("error"))
            result["patches"].setdefault(ara_pid, {})["scores"] = edited
            step("scored ARA")

    # ---------------------------------------------------------------- G-S9: the companion bundle and the adapter refusal
    from fastapi.testclient import TestClient
    from wsp.companion import export_bundle, import_bundle
    from wsp.server import create_app
    if isinstance(kl_p, list):
        kl_texts = kl_p[:KL_EXPORT_PROMPTS]
    else:                                     # a dataset specification, resolved offline as a measurement resolves it
        from wsp import directions_datasets as DD
        kl_texts = [DD.user_of(p) for p in DR.check_measure(ws, "kl-export", kl_p, kl_p, chat=True, system=system,
                                                             variant="base", replace=True)["good"]][:KL_EXPORT_PROMPTS]
    kl8 = DR.render_ids(ws, kl_texts, "kl", chat=True, system=system, prefix=prefix, chat_options={})
    before = [torch.as_tensor(ws.runner.first_token_logits(x, "current")).float().reshape(-1) for x in kl8]
    client = TestClient(create_app(session, "http://testserver"), raise_server_exceptions=False)
    ans = client.post("/api/export_adapter", json={"name": "gguf-check", **({"patches": [ara_pid]} if ara_pid else {})})
    text = ans.text
    adapter_refused = ans.status_code == 400 and "quantized GGUF" in text
    (root / "export").mkdir(parents=True, exist_ok=True)
    bundle = export_bundle(ws, str(root / "export" / "companion"))["path"]
    step("exported the companion bundle")
    ws.runner.release()
    ws._runner = None
    reopen_root = Path(str(root) + "-reopen")
    if reopen_root.exists():
        raise SystemExit(f"{reopen_root} exists: give a fresh isolated root")
    again = import_bundle(str(reopen_root), bundle, base_path=str(gguf))
    after = [torch.as_tensor(again.runner.first_token_logits(x, "current")).float().reshape(-1) for x in kl8]
    equal = all(torch.equal(x, y) for x, y in zip(before, after)) and len(before) == len(after) == len(kl8)
    rep.check(LABEL, "G-S9 export", equal and adapter_refused, logits_bit_equal=equal, prompts=len(kl8),
              max_abs=max(float((x - y).abs().max()) for x, y in zip(before, after)) if before else None,
              export_adapter_status=ans.status_code, export_adapter_answer=text[:300], bundle=bundle)
    again.runner.release()
    step("reopened the bundle")
    rep.check(LABEL, "G-S4 base unchanged after every step", all(s["sha256_equal"] for s in result["steps"]),
              steps=len(result["steps"]))
    result["largest_read_values"], result["reads"] = reads["max"], reads["count"]
    result["checks"], result["pass"] = rep.labels, rep.ok()
    result["bounds"] = {k[0]: v for k, v in BOUNDS.items()}
    FS.write_json(out, result)
    print(f"{'ok' if result['pass'] else 'NOT OK'}: {out}", flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
