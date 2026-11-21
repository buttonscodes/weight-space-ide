"""Compare the IDE's scores and response-prefix detection with finished Heretic runs on the same model.

Run with the IDE's interpreter, with the IDE's `weight-space-ide` directory as the current directory, offline and
under `python -m wsp.memory_guard` (it loads the base model, then each run's merged model, one at a time):

    python scripts/heretic_compare_scores.py [--dtype float32|bfloat16] [--heretic-adapter [<mode>=]<dir> ...] \
        <model snapshot dir> <isolated IDE root> <out.json> <mode>=<heretic run dir> ...

Each Heretic run directory holds Heretic's merged model (model/), its config.toml and the dump written by
heretic_dump.py (dump/dump.json, dump/dump.safetensors, dump/scores.json, dump/scores.safetensors), with Heretic
run at batch_size 1. For an ablation run ``mode`` is its row_normalization (none, pre, full); a run of the fitted
low-rank ablation (dump modifier ARA) is compared in S0-S5 only.

A run exported as an adapter (export_strategy "adapter"; the directory given with --heretic-adapter, else the first
of <run>/adapter, <run>/<save_directory>, <run>/model that holds adapter_config.json; refused unless r == lora_alpha)
has no merged model: it is materialised into <run>/heretic-merged/ the way PEFT merges (scripts/formats_shared.py
merge: weight.data += B @ A on the checkpoint's tensor, a float32 delta; the base's other files linked; the SHA-256
of every written tensor recorded in its merge.json and in <out.json>), and that directory is the merged model of S5.
Its scores.json "merged" must have been dumped from the same directory (heretic_dump.py <out> ../heretic-merged).

--dtype sets WSP_RUN_DTYPE before the workspace's runner exists (default float32) and selects the row of BOUNDS
below for every stage (B-S7 and B-S8 of the formats build spec for bfloat16; the float32 rows are the values this
script used before). Every assertion prints PASS or FAIL; <out.json> holds every number and a pass flag per
assertion; the exit status is 1 when one fails.

Stages and their acceptance (decided before any run; a stage outside its bound is a finding, not a threshold):
    S0  eps: max |IDE - Heretic| of the base first-token logits of the KL prompts            reported
    S1  rendered token ids of the refusal, length, KL and prefix-test prompts                 identical
    S2  the detected response prefix; the prefix-test responses without and with the prefix  prefix identical; responses as S3
    S3  base responses (ids) per prompt                                                     identical, or at the first differing
        step the IDE's processed-score margin between the two tokens is below max(1e-3, 4 eps)
    S4  keyword flags, Refusals k/N, ResponseLength on base                                 flags identical on identical
        responses; scores identical unless every difference comes from an accepted divergence
    S5  the merged model as its own workspace: responses, flags and scores as S3-S4; KL of its first-token
        log-probabilities against the base reference                                        |ide - heretic| <=
        1e-6 + max(1e-3 KL, 3 eps sqrt(2 KL))
    S6  every trial of an ablation run: Heretic's means saved as a measurement, the trial's parameters as a script,
        scored as an uncommitted candidate                                                  KL within
        1e-6 + max(r KL, 3 eps sqrt(2 KL)), r = 1e-3 (float32) or 1e-2 (bfloat16, B-S8). Refusal counts are
        reported, not asserted, in both dtypes: B-S8 asks for them "as B-S7" (equal unless every difference is an
        accepted divergence), which needs each response, and Heretic's dump holds no per-trial responses, so the
        divergences cannot be decided; the counts and how many trials agree are in <out.json>
"""
import argparse
import json
import math
import os
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

# ---- BOUNDS[(stage, dtype)]: the float32 rows are this script's values before the dtype existed; bfloat16 B-S7/B-S8
BOUNDS = {
    ("S1 token ids", "float32"): "identical",
    ("S1 token ids", "bfloat16"): "identical",
    ("S2 prefix", "float32"): "identical",
    ("S2 prefix", "bfloat16"): "identical",
    ("S3 response margin", "float32"): {"floor": 1e-3, "eps_factor": 4},       # margin < max(floor, factor eps)
    ("S3 response margin", "bfloat16"): {"floor": 1e-3, "eps_factor": 4},
    ("S4 scores", "float32"): "identical unless every difference is an accepted divergence",
    ("S4 scores", "bfloat16"): "identical unless every difference is an accepted divergence",
    ("S5 KL", "float32"): {"absolute": 1e-6, "relative": 1e-3, "eps_factor": 3},  # 1e-6 + max(r KL, 3 eps sqrt(2 KL))
    ("S5 KL", "bfloat16"): {"absolute": 1e-6, "relative": 1e-3, "eps_factor": 3},
    ("S6 KL", "float32"): {"absolute": 1e-6, "relative": 1e-3, "eps_factor": 3},
    ("S6 KL", "bfloat16"): {"absolute": 1e-6, "relative": 1e-2, "eps_factor": 3},
    ("S6 refusals", "float32"): None,          # reported (as before)
    ("S6 refusals", "bfloat16"): None,         # B-S8 "as B-S7": needs Heretic's per-trial responses, not dumped
}

_p = argparse.ArgumentParser(prog="heretic_compare_scores.py", description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
_p.add_argument("--heretic-adapter", action="append", default=[], metavar="[MODE=]DIR")
_p.add_argument("snapshot")
_p.add_argument("root")
_p.add_argument("out")
_p.add_argument("runs", nargs="+", metavar="MODE=RUN")
ARGS = _p.parse_args()                          # before any import of torch or wsp: --help needs neither

sys.path.insert(0, os.getcwd())
import formats_shared as FS  # noqa: E402

FS.set_run_dtype(ARGS.dtype)                    # before the runner exists

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from wsp import directions as DR  # noqa: E402
from wsp import directions_prefix as DP  # noqa: E402
from wsp import directions_scores as DS  # noqa: E402
from wsp.session import Session  # noqa: E402

snap, root, out = Path(ARGS.snapshot), Path(ARGS.root), Path(ARGS.out)
runs = dict(a.split("=", 1) for a in ARGS.runs)
adapters = FS.parse_adapters(ARGS.heretic_adapter, list(runs))
DTYPE = ARGS.dtype
quiet = lambda *a, **k: None  # noqa: E731
rep = FS.Report()


def B(stage):
    return FS.bound(BOUNDS, stage, DTYPE)


def open_ws(directory, label):
    session = Session(str(root / label))
    os.makedirs(session.root, exist_ok=True)
    return session, session.open(str(directory), log=quiet)


class Recorder:
    """Every response the runner generates, by prompt ids."""

    def __init__(self, ws):
        self.runner, self.got = ws.runner, {}
        real = self.runner.generate

        def generate(ids, *a, **kw):
            res = real(ids, *a, **kw)
            self.got[tuple(int(i) for i in ids)] = [int(t) for t in res]
            return res
        self.runner.generate = generate

    def close(self):
        del self.runner.generate


def margin(ws, prompt_ids, common, a, b, variant):
    """|score(a) - score(b)| of the next token after prompt + common, as transformers' generate processes it (the
    checkpoint's generation config, the repetition penalty included)."""
    r = ws.runner
    with r.lock:
        r.use(variant)
        t = torch.tensor([list(prompt_ids) + list(common)])
        cfg = r.model.generation_config
        pad = cfg.pad_token_id if cfg.pad_token_id is not None else cfg.eos_token_id
        pad = pad[0] if isinstance(pad, (list, tuple)) else pad
        with torch.no_grad():
            o = r.model.generate(input_ids=t, attention_mask=torch.ones_like(t), max_new_tokens=1, output_scores=True,
                                 return_dict_in_generate=True, do_sample=False, pad_token_id=pad if pad is not None else 0)
    s = o.scores[0][0].float()
    return abs(float(s[a] - s[b]))


def compare_responses(ws, prompts_ids, ide, heretic, bound, variant):
    rows, identical = [], 0
    for k, (p, x, y) in enumerate(zip(prompts_ids, ide, heretic)):
        if list(x) == list(y):
            identical += 1
            continue
        j = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
        if j >= len(x) or j >= len(y):
            rows.append({"prompt": k, "step": j, "ide_length": len(x), "heretic_length": len(y), "margin": None,
                         "accepted": False})
            continue
        m = margin(ws, p, x[:j], x[j], y[j], variant)
        rows.append({"prompt": k, "step": j, "ide": x[j], "heretic": y[j], "margin": m, "accepted": m < bound})
    return {"prompts": len(prompts_ids), "identical": identical, "divergences": rows,
            "ok": all(r["accepted"] for r in rows)}


def same_ids(a, b):
    return {"prompts": len(a), "identical": sum(list(x) == list(y) for x, y in zip(a, b)),
            "ok": len(a) == len(b) and all(list(x) == list(y) for x, y in zip(a, b))}


def scorer_settings(config, table, keys):
    t = (config.get("scorer") or {}).get(table) or {}
    return {k: t[k] for k in keys if k in t}


def score_stage(ws, state, rendering, refusal, length, kl, kr_settings, rl_settings, variant, heretic, bound,
                script=None):
    """One score request with the run's settings, its responses compared (S3), flags and scores (S4)."""
    rec = Recorder(ws)
    try:
        body = {"state": state, "max_response_length": rendering["max_response_length"], **rendering["chat"],
                **({"script": script} if script else {}),
                "scores": [{"plugin": "KeywordRate", "prompts": refusal, "print_responses": True, **kr_settings},
                           {"plugin": "ResponseLength", "prompts": length, "print_responses": True, **rl_settings},
                           {"plugin": "KLDivergence", "prompts": kl}]}
        res = DS.score(ws, body)
    finally:
        rec.close()
    out = {"scores": {s["plugin"]: {k: s[k] for k in ("value", "display")} for s in res["scores"]},
           "kl_per_prompt": res["scores"][2]["per_prompt"], "kl_reference": res["scores"][2]["reference"]}
    if heretic is None:
        return out, res
    for label, idx, h in (("refusal", 0, heretic["refusal"]), ("length", 1, heretic["length"])):
        prompts_ids = [list(x) for x in h["input_ids"]]
        ide_ids = [rec.got.get(tuple(x)) for x in prompts_ids]
        if any(x is None for x in ide_ids):
            out[label] = {"error": "rendered ids differ from Heretic's (see S1); responses not compared"}
            continue
        resp = compare_responses(ws, prompts_ids, ide_ids, h["response_ids"], bound, variant)
        same = {d["prompt"] for d in resp["divergences"]}
        rows = res["scores"][idx]["responses"]
        entry = {"responses": resp}
        if label == "refusal":
            flags = [r["match"] for r in rows]
            entry["flags_identical_on_identical_responses"] = all(
                f == g for k, (f, g) in enumerate(zip(flags, h["flags"])) if k not in same)
            ide_v, her_v = res["scores"][0]["value"], h["score"]["value"]
        else:
            ide_v, her_v = res["scores"][1]["value"], h["score"]["value"]
        entry["value"] = {"ide": ide_v, "heretic": her_v, "equal": ide_v == her_v,
                          "ok": ide_v == her_v or (resp["ok"] and bool(same))}
        out[label] = entry
    return out, res


def kl_bound(kl, eps, b=None):
    b = b or BOUNDS[("S5 KL", "float32")]
    return b["absolute"] + max(b["relative"] * kl, b["eps_factor"] * eps * math.sqrt(2 * max(kl, 0.0)))


def merged_dir(run, mode, dump):
    """Heretic's merged model: <run>/model when it holds weights, else the adapter's merge materialised into
    <run>/heretic-merged (formats_shared.merge_adapter). -> (directory, merge record or None)."""
    m = run / "model"
    has_weights = (m / "model.safetensors").is_file() or (m / "model.safetensors.index.json").is_file()
    if has_weights and mode not in adapters:
        return m, None
    adapter = FS.find_adapter(run, adapters.get(mode))
    if adapter is None:
        raise SystemExit(f"{run}: no merged model (model/) and no adapter export")
    as_dtype = "float32" if "float32" in str(dump.get("dtype")) else None
    info = FS.merge_adapter(snap, adapter, run / "heretic-merged", log=print, as_dtype=as_dtype)
    return run / "heretic-merged", info


result = {"ide_versions": {p: version(p) for p in ("torch", "transformers", "tokenizers", "safetensors")},
          "dtype": DTYPE, "bounds": {k[0]: v for k, v in BOUNDS.items() if k[1] == DTYPE}, "modes": {}}
for mode, run in runs.items():
    run = Path(run)
    dump = json.loads((run / "dump/dump.json").read_text())
    tens = load_file(str(run / "dump/dump.safetensors"))
    sc = json.loads((run / "dump/scores.json").read_text())
    st = load_file(str(run / "dump/scores.safetensors"))
    config = tomllib.loads((run / "config.toml").read_text())
    kr_settings = scorer_settings(config, "KeywordRate", ("score_name", "keyword_markers"))
    rl_settings = scorer_settings(config, "ResponseLength", ("strip_responses", "short_response_threshold",
                                                             "short_response_score_length"))
    system, prefix = sc["system_prompt"], sc["response_prefix"] or None
    M = sc["max_response_length"]
    rendering = {"max_response_length": M, "chat": {"chat": True, "system": system, "prefix": prefix}}
    texts = {k: [p["user"] for p in v] for k, v in dump["prompts"].items()}
    prefix_texts = [p["user"] for p in sc["prefix_test"]["prompts"]]
    assert all(p["system"] == system for p in sc["prefix_test"]["prompts"])
    r = {"heretic_versions": sc["versions"], "heretic_generation_config": sc["generation_config"],
         "heretic_attn_implementation": sc["attn_implementation"], "batch_size": sc["batch_size"],
         "clean_up_tokenization_spaces": sc["clean_up_tokenization_spaces"]}
    session, ws = open_ws(snap, "base")
    r["ide_runner"] = DS.runner_row(ws)[0]
    r["ide_generation"] = DS.generation_block(ws, M)

    def render(t, p):
        return DR.render_ids(ws, t, "prompts", chat=True, system=system, prefix=p, chat_options={})

    # ---- S0: eps from the base first-token logits of the KL prompts
    kl_ids = render(texts["kl"], prefix)
    lb = torch.stack([ws.runner.first_token_logits(x, "base") for x in kl_ids])
    eps = float((lb - tens["kl_logits"].float()).abs().max())
    bound = max(B("S3 response margin")["floor"], B("S3 response margin")["eps_factor"] * eps)
    r["S0"] = {"eps": eps, "logit_scale": float(tens["kl_logits"].abs().max()), "response_margin_bound": bound}

    # ---- S1: rendered token ids
    r["S1"] = {"refusal": same_ids(render(texts["refusal"], prefix), sc["base"]["refusal"]["input_ids"]),
               "length": same_ids(kl_ids, sc["base"]["length"]["input_ids"]),
               "kl": same_ids(kl_ids, dump["ids"]["kl"]),
               "prefix_test_without": same_ids(render(prefix_texts, None), sc["prefix_test"]["without_prefix"]["input_ids"])}
    if sc["prefix_test"]["with_prefix"] is not None:
        r["S1"]["prefix_test_with"] = same_ids(render(prefix_texts, prefix), sc["prefix_test"]["with_prefix"]["input_ids"])
    r["S1"]["ok"] = all(v["ok"] for v in r["S1"].values())
    rep.check(mode, "S1 token ids", r["S1"]["ok"], **{k: f"{v['identical']}/{v['prompts']}" for k, v in r["S1"].items()
                                                     if isinstance(v, dict)})

    # ---- S2: the detected prefix, and the prefix-test responses
    det = DP.detect(ws, {"prompts": prefix_texts, "system": system, "max_response_length": M,
                         "chain_of_thought_blocks": sc["chain_of_thought_blocks"]})
    r["S2"] = {"ide_prefix": det["response_prefix"], "heretic_prefix": sc["response_prefix"], "found_by": det["found_by"],
               "prefix_ok": (det["response_prefix"] or "") == (sc["response_prefix"] or "")}
    stops = DS.stop_ids(ws)
    for label, p in (("without_prefix", None), ("with_prefix", prefix)):
        h = sc["prefix_test"][label]
        if h is None:
            continue
        ids = render(prefix_texts, p)
        ide = [DS.generate_response(ws, x, "base", None, M, stops, "prompts", k + 1) for k, x in enumerate(ids)]
        r["S2"][label] = compare_responses(ws, ids, ide, h["response_ids"], bound, "base")
    rep.check(mode, "S2 prefix and prefix-test responses", r["S2"]["prefix_ok"] and all(
        r["S2"][k]["ok"] for k in ("without_prefix", "with_prefix") if k in r["S2"]),
        ide_prefix=r["S2"]["ide_prefix"], heretic_prefix=r["S2"]["heretic_prefix"])

    # ---- S3, S4: base responses, flags and scores
    s34, _ = score_stage(ws, "base", rendering, texts["refusal"], texts["kl"], texts["kl"], kr_settings, rl_settings,
                         "base", sc["base"], bound)
    r["S3_S4"] = s34
    for label in ("refusal", "length"):
        e = s34.get(label) or {}
        rep.check(mode, f"S3 base responses ({label})", bool(e.get("responses", {}).get("ok")), error=e.get("error"),
                  identical=e.get("responses", {}).get("identical"), prompts=e.get("responses", {}).get("prompts"))
        rep.check(mode, f"S4 base score ({label})", bool((e.get("value") or {}).get("ok")) and
                  e.get("flags_identical_on_identical_responses", True), value=e.get("value"))
    ref = torch.log_softmax(lb.float(), -1)

    # ---- S6: every trial of an ablation run as an uncommitted candidate (base workspace, Heretic's means)
    if dump["modifier"] != "ARA":
        n, d = ws.struct.n_layers, ws.struct.d_model
        meta = {"version": DR.VERSION, "name": "oracle-scores", "created": 0, "variant": "base", "n_layers": n,
                "d_model": d, "counts": {"good": len(texts["good"]), "bad": len(texts["bad"])}, "tokens": {},
                "options": {"chat": True, "system": system, "prefix": prefix}, "prompts_sha256": {}, "model": {}}
        DR.save(ws, "oracle-scores", tens["good_mean"], tens["bad_mean"], meta,
                {"good": texts["good"], "bad": texts["bad"]}, replace=True)
        trials = []
        for t in dump["trials"]:
            params = {"direction": "oracle-scores", **t["parameters"], "orthogonalize_direction": True,
                      "row_normalization": mode, "full_normalization_lora_rank": 3}
            got, _ = score_stage(ws, "candidate", rendering, texts["refusal"], texts["kl"], texts["kl"], kr_settings,
                                 rl_settings, "current", None, bound, script=DR.script_for(params))
            her = {s["name"]: s["score"]["value"] for s in t["scores"]}
            kl_i, kl_h = got["scores"]["KLDivergence"]["value"], her.get("KL divergence")
            ref_name = kr_settings.get("score_name", "Refusals")
            trials.append({"index": t["index"], "refusals_ide": got["scores"]["KeywordRate"]["value"],
                           "refusals_heretic": her.get(ref_name), "kl_ide": kl_i, "kl_heretic": kl_h,
                           "kl_bound": kl_bound(kl_h, eps, B("S6 KL")) if kl_h is not None else None,
                           "kl_ok": kl_h is not None and abs(kl_i - kl_h) <= kl_bound(kl_h, eps, B("S6 KL")),
                           "refusals_equal": got["scores"]["KeywordRate"]["value"] == her.get(ref_name)})
        r["S6"] = {"trials": trials, "kl_ok": all(t["kl_ok"] for t in trials),
                   "refusals_equal": sum(t["refusals_equal"] for t in trials)}
        rep.check(mode, "S6 KL of every trial as a candidate", r["S6"]["kl_ok"] and bool(trials), trials=len(trials),
                  outside=[t["index"] for t in trials if not t["kl_ok"]], relative=B("S6 KL")["relative"])
        rep.note(mode, "S6 refusals (reported)", {"equal": r["S6"]["refusals_equal"], "trials": len(trials)})
        DR.delete(ws, "oracle-scores")
    session.unload()

    # ---- S5: the merged model as its own workspace (same weights as Heretic's merged model)
    mdir, merge_info = merged_dir(run, mode, dump)
    r["merged_model"] = {"dir": str(mdir), "materialised": merge_info is not None,
                         "sha256": merge_info["sha256"] if merge_info else None,
                         "changed": sorted(merge_info["changed"]) if merge_info else None}
    msession, mws = open_ws(mdir, f"merged-{mode}")
    s5, res = score_stage(mws, "base", rendering, texts["refusal"], texts["kl"], texts["kl"], kr_settings, rl_settings,
                          "base", sc["merged"], bound)
    mk_ids = DR.render_ids(mws, texts["kl"], "prompts", chat=True, system=system, prefix=prefix, chat_options={})
    lm = torch.stack([mws.runner.first_token_logits(x, "base") for x in mk_ids])
    kl_ide = sum(DS.kl_row(torch.log_softmax(lm[i].float(), -1), ref[i]) for i in range(len(mk_ids))) / len(mk_ids)
    kl_her = sc["merged"]["kl"]["score"]["value"]
    s5["kl"] = {"ide": kl_ide, "heretic": kl_her, "bound": kl_bound(kl_her, eps, B("S5 KL")),
                "ok": abs(kl_ide - kl_her) <= kl_bound(kl_her, eps, B("S5 KL")),
                "merged_logits_max_abs_diff": float((lm - st["merged_kl_logits"].float()).abs().max())}
    r["S5"] = s5
    for label in ("refusal", "length"):
        e = s5.get(label) or {}
        rep.check(mode, f"S5 merged responses ({label})", bool(e.get("responses", {}).get("ok")), error=e.get("error"),
                  identical=e.get("responses", {}).get("identical"), prompts=e.get("responses", {}).get("prompts"))
        rep.check(mode, f"S5 merged score ({label})", bool((e.get("value") or {}).get("ok")) and
                  e.get("flags_identical_on_identical_responses", True), value=e.get("value"))
    rep.check(mode, "S5 merged KL", s5["kl"]["ok"], ide=kl_ide, heretic=kl_her, bound=s5["kl"]["bound"])
    r["checks"] = rep.labels.get(mode, {})
    msession.unload()
    result["modes"][mode] = r
    result["pass"] = rep.ok()
    out.write_text(json.dumps(result, indent=1))

result["pass"] = rep.ok()
out.write_text(json.dumps(result, indent=1))
for mode, r in result["modes"].items():
    print(f"== {mode}  eps {r['S0']['eps']:.2e}")
    print("  S1 ids:", {k: v if isinstance(v, bool) else f"{v['identical']}/{v['prompts']}" for k, v in r["S1"].items()})
    print("  S2 prefix: ide %r heretic %r ok %s" % (r["S2"]["ide_prefix"], r["S2"]["heretic_prefix"], r["S2"]["prefix_ok"]))
    for stage in ("S3_S4", "S5"):
        s = r[stage]
        print(f"  {stage}:", {k: s[k]["value"] for k in ("refusal", "length") if isinstance(s.get(k), dict) and "value" in s[k]})
    print("  S5 KL:", r["S5"]["kl"])
    if "S6" in r:
        print("  S6 trials: KL ok %s, refusals equal %d/%d" % (r["S6"]["kl_ok"], r["S6"]["refusals_equal"], len(r["S6"]["trials"])))
print(f"{'ok' if result['pass'] else 'NOT OK'}: {out}", flush=True)
sys.exit(0 if result["pass"] else 1)
