"""The IDE in bfloat16 against the IDE in float32 on the same checkpoint (G4: B-S1, B-S3, B-S6, B-S9).

Run with the IDE's interpreter from the IDE's `weight-space-ide` directory, offline, one process per dtype, each
under the memory guard (it loads the model), each in its own isolated root:

    python scripts/formats_bf16_self.py --stage run --dtype float32|bfloat16 <model dir> <isolated IDE root> \\
        <out dir> abl=<Heretic ablation run dir> ara=<Heretic ARA run dir>
    python scripts/formats_bf16_self.py --stage compare <float32 out dir> <bfloat16 out dir> <out.json>

The two Heretic runs are finished float32 runs of the same checkpoint (for Qwen2.5-0.5B-Instruct, the oracle's
abl-full and ara runs); only their dump/dump.json (and config.toml) are read: the prompts, the system prompt, the
response prefix, the exported trial of each and its settings. Nothing is written under them.

Stage run (no model in the compare stage): sets WSP_RUN_DTYPE before the runner exists; renders every measured,
KL and refusal prompt (chat, the run's system prompt) and records the token ids; measures the ARA run's 64 + 64
prompts once, as a chat with module_io; runs the two exported trials (directional ablation with the ablation run's
row_normalization, ARA with the ARA run's settings and seed) on that measurement as uncommitted candidates, the ARA
script once through the weight-script interpreter for the consistency check of every module it fits; and scores
the base and both candidates (KeywordRate on the refusal prompts, KLDivergence on the KL prompts, the run's
max_response_length). <out dir>/run.json holds every number; <out dir>/means.safetensors the two means and the
direction table.

Stage compare: the bounds below, for the bfloat16 run against the float32 run. Every assertion prints PASS or FAIL;
<out.json> holds every number and a pass flag per assertion; the exit status is 1 when one fails.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---- BOUNDS[(stage, dtype)], dtype = the run compared with float32 (formats build spec 7.3); None = reported only
BOUNDS = {
    ("B-S1 token ids", "float32"): "identical",
    ("B-S1 token ids", "bfloat16"): "identical",       # IDE bf16 vs IDE fp32 vs Heretic's dumps
    ("B-S3 directions", "float32"): None,
    ("B-S3 directions", "bfloat16"): None,             # reported; expected cosine at least 0.99 at indices 1..n
    ("B-S6 consistency check", "float32"): None,
    ("B-S6 consistency check", "bfloat16"): (0.0, 5e-3),   # 0 < mismatch <= 5e-3 for every captured module
    ("B-S9 refusals and KL", "float32"): None,
    ("B-S9 refusals and KL", "bfloat16"): None,        # reported; refusals expected within 3 of 32
}
EXPECTED = {"B-S3 directions": {"cosine_at_least": 0.99}, "B-S9 refusals and KL": {"refusals_within": 3}}


def parse(argv):
    p = argparse.ArgumentParser(prog="formats_bf16_self.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("run", "compare"), required=True)
    p.add_argument("--dtype", choices=("float32", "bfloat16"))
    p.add_argument("args", nargs="+")
    a = p.parse_args(argv)
    if a.stage == "run" and (a.dtype is None or len(a.args) != 5):
        p.error("--stage run needs --dtype and <model dir> <root> <out dir> abl=<run dir> ara=<run dir>")
    if a.stage == "compare" and len(a.args) != 3:
        p.error("--stage compare needs <float32 out dir> <bfloat16 out dir> <out.json>")
    return a


def run_stage(dtype, model, root, out, runs):
    import formats_shared as FS
    FS.set_run_dtype(dtype)
    import tomllib

    from safetensors.torch import save_file

    from wsp import directions as DR
    from wsp import directions_scores as DS
    from wsp import wscript
    from wsp.session import Session

    dumps = {k: json.loads((Path(v) / "dump/dump.json").read_text()) for k, v in runs.items()}
    if set(dumps) != {"abl", "ara"} or dumps["ara"].get("modifier") != "ARA":
        raise SystemExit("give abl=<Heretic ablation run dir> and ara=<Heretic ARA run dir>")
    ara, abl = dumps["ara"], dumps["abl"]
    texts = {k: [p["user"] for p in v] for k, v in ara["prompts"].items()}
    system, prefix = ara["system_prompt"], ara["response_prefix"] or None
    M = tomllib.loads((Path(runs["ara"]) / "config.toml").read_text()).get("max_response_length", 100)
    out.mkdir(parents=True, exist_ok=True)
    session = Session(str(root))
    ws = session.open(str(model), log=lambda *a_, **k: None)
    for p in list(ws.patches):                       # candidates on the base
        if p.enabled:
            ws.set_patch(p.id, enabled=False)
    n = ws.struct.n_layers
    result = {"dtype": dtype, "max_response_length": M, "system": system, "prefix": prefix,
              "same_prompts_in_both_runs": all([q["user"] for q in abl["prompts"][k]] == texts[k] for k in ("good", "bad")),
              "timing": {}}

    # ---- token ids of every prompt (B-S1)
    ids = {}
    for k in ("good", "bad", "kl", "refusal"):
        ids[k] = DR.render_ids(ws, texts[k], k, chat=True, system=system, prefix=prefix, chat_options={})
    result["ids"] = ids
    heretic = {"good": ara["io_ids"]["good"], "bad": ara["io_ids"]["bad"], "kl": ara["ids"]["kl"]}
    result["ids_identical_to_heretic"] = {k: sum(list(x) == list(y) for x, y in zip(ids[k], heretic[k])) == len(ids[k])
                                          == len(heretic[k]) for k in heretic}

    # ---- one measurement with module_io
    t0 = time.monotonic()
    DR.measure(ws, "self", texts["good"], texts["bad"], chat=True, system=system, prefix=prefix, variant="base",
               replace=True, module_io=True)
    result["timing"]["measure_seconds"] = time.monotonic() - t0
    result["runner"] = DR.runner_description(ws)
    _, good, bad = DR.load(ws, "self")
    save_file({"good_mean": good.contiguous(), "bad_mean": bad.contiguous(),
               "table": DR.table(good, bad, True).contiguous()}, str(out / "means.safetensors"))

    # ---- the two exported trials as scripts on that measurement
    def exported(dump):
        return next(t for t in dump["trials"] if t["index"] == dump["exported_trial_index"])
    conf = (abl.get("modifier_settings") or {}).get("Abliteration") or {}
    abl_params = {"direction": "self", **exported(abl)["parameters"],
                  "orthogonalize_direction": conf.get("orthogonalize_direction", True),
                  "row_normalization": conf.get("row_normalization", "full"),
                  "full_normalization_lora_rank": conf.get("full_normalization_lora_rank", 3)}
    aconf = (ara.get("modifier_settings") or {}).get("ARA") or {}
    defaults = {"preserve_row_magnitudes": True, "lora_rank": 50, "n_optimization_steps": 5, "learning_rate": 1.0,
                "max_iter": 20, "history_size": 10}
    ara_params = {"method": "ara", "direction": "self", **exported(ara)["parameters"],
                  **{k: aconf.get(k, v) for k, v in defaults.items()}, "seed": ara["seed"]}
    scripts = {"ablation": DR.script_for(abl_params), "ara": DR.script_for(ara_params)}
    result["scripts"] = scripts

    # ---- B-S6: the consistency check of every module the ARA script fits (run once, nothing committed)
    t0 = time.monotonic()
    res = wscript.run_script(ws, scripts["ara"])
    result["timing"]["ara_fit_seconds"] = time.monotonic() - t0
    result["consistency"] = FS.ara_components(res.get("calls"), res.get("output", ""))
    del res

    # ---- scores: base and both candidates (B-S9)
    scores = {}
    for state, script in (("base", None), ("ablation", scripts["ablation"]), ("ara", scripts["ara"])):
        body = {"state": "base" if state == "base" else "candidate", "chat": True, "system": system, "prefix": prefix,
                "max_response_length": M,
                "scores": [{"plugin": "KeywordRate", "prompts": texts["refusal"], "print_responses": True},
                           {"plugin": "KLDivergence", "prompts": texts["kl"]}]}
        if script:
            body["script"] = script
        t0 = time.monotonic()
        got = DS.score(ws, body)
        result["timing"][f"score_{state}_seconds"] = time.monotonic() - t0
        kr, kl = got["scores"][0], got["scores"][1]
        scores[state] = {"refusals": kr["matches"], "of": kr["total"], "kl": kl["value"], "kl_per_prompt": kl["per_prompt"],
                         "responses": [{"tokens": x["tokens"], "match": x["match"], "response": x["response"]}
                                       for x in kr.get("responses", [])],
                         "generation": got.get("generation")}
    result["scores"] = scores
    result["heretic"] = {"ablation": {s["name"]: s["score"]["value"] for s in exported(abl)["scores"]},
                         "ara": {s["name"]: s["score"]["value"] for s in exported(ara)["scores"]}}
    result["layers"] = n
    FS.write_json(out / "run.json", result)
    session.unload()
    print(f"ok: {out / 'run.json'}  (refusals base {scores['base']['refusals']}/{scores['base']['of']}, "
          f"ablation {scores['ablation']['refusals']}, ara {scores['ara']['refusals']})", flush=True)
    return 0


def compare_stage(f32_dir, bf16_dir, out):
    import formats_shared as FS
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    runs = {d: json.loads((Path(p) / "run.json").read_text()) for d, p in (("float32", f32_dir), ("bfloat16", bf16_dir))}
    if runs["float32"]["dtype"] != "float32" or runs["bfloat16"]["dtype"] != "bfloat16":
        raise SystemExit("the first out dir must be a float32 run, the second a bfloat16 run")
    rep = FS.Report()
    dt = "bfloat16"
    B = lambda s: FS.bound(BOUNDS, s, dt)  # noqa: E731
    f, b = runs["float32"], runs["bfloat16"]
    label = "bfloat16 vs float32"

    # B-S1
    same = {k: f["ids"][k] == b["ids"][k] for k in f["ids"]}
    heretic = {d: r["ids_identical_to_heretic"] for d, r in runs.items()}
    rep.bounded(label, "B-S1 token ids", B("B-S1 token ids"),
                all(same.values()) and all(all(v.values()) for v in heretic.values()), lambda v, bd: v,
                ide_bf16_vs_fp32=same, ide_vs_heretic=heretic)

    # B-S3
    tf = load_file(str(Path(f32_dir) / "means.safetensors"))["table"]
    tb = load_file(str(Path(bf16_dir) / "means.safetensors"))["table"]
    cos = [float(F.cosine_similarity(tb[k].double(), tf[k].double(), dim=-1)) for k in range(tf.shape[0])]
    rep.note(label, "B-S3 directions", {"cosine_per_index": cos, "min_cosine_1_to_n": min(cos[1:]),
                                        "expected_at_least": EXPECTED["B-S3 directions"]["cosine_at_least"]})

    # B-S6
    values = [c[k] for c in b["consistency"] for k in ("mismatch_good", "mismatch_bad")]
    lim = B("B-S6 consistency check")
    rep.bounded(label, "B-S6 consistency check", lim, values, lambda v, bd: bool(v) and all(bd[0] < x <= bd[1] for x in v),
                modules=len(b["consistency"]), smallest=min(values, default=None), largest=max(values, default=None),
                float32_largest=max((c[k] for c in f["consistency"] for k in ("mismatch_good", "mismatch_bad")),
                                    default=None))

    # B-S9
    rows = {}
    for state in ("base", "ablation", "ara"):
        rows[state] = {"refusals_float32": f["scores"][state]["refusals"], "refusals_bfloat16": b["scores"][state]["refusals"],
                       "of": f["scores"][state]["of"], "kl_float32": f["scores"][state]["kl"],
                       "kl_bfloat16": b["scores"][state]["kl"]}
    rep.note(label, "B-S9 refusals and KL", {"states": rows, "expected": EXPECTED["B-S9 refusals and KL"],
                                             "generation_bfloat16": b["scores"]["base"].get("generation")})
    result = {"compared": {"float32": str(f32_dir), "bfloat16": str(bf16_dir)}, "runners": {d: r.get("runner") for d, r in runs.items()},
              "timing": {d: r.get("timing") for d, r in runs.items()}, "bounds": {k[0]: v for k, v in BOUNDS.items() if k[1] == dt},
              "checks": rep.labels, "pass": rep.ok()}
    FS.write_json(out, result)
    print(f"{'ok' if result['pass'] else 'NOT OK'}: {out}", flush=True)
    del torch
    return 0 if result["pass"] else 1


def main(argv):
    a = parse(argv)
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if a.stage == "run":
        model, root, out = (Path(x) for x in a.args[:3])
        runs = dict(x.split("=", 1) for x in a.args[3:])
        return run_stage(a.dtype, model, root, out, runs)
    return compare_stage(Path(a.args[0]), Path(a.args[1]), Path(a.args[2]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
