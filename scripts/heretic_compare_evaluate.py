"""Compare the IDE's scoring of a modified checkpoint (wsp.directions_evaluate) with Heretic's evaluate_model.

Run by the orchestrator only (it loads models), with the IDE's interpreter, with the IDE's `weight-space-ide`
directory as the current directory, under `python -m wsp.memory_guard`, one process at a time, offline, with the
isolated Hugging Face cache:

    python scripts/heretic_compare_evaluate.py <base snapshot dir> <isolated IDE root> <out.json> \\
        [previous=<heretic_compare_scores out.json>] <label>=<heretic run dir> ...

Each Heretic run directory holds Heretic's merged model (model/), its config.toml and the dump written by
heretic_dump.py (dump/dump.json, dump/dump.safetensors, dump/scores.json), Heretic run at batch_size 1;
dump/scores.json "merged" holds exactly what evaluate_model prints for model/. A run exported as an adapter
(export_strategy "adapter": adapter/ holds adapter_config.json) has as model/ PEFT's own merge of that adapter
(scripts/peft_merge_adapter.py, PEFT's merge_and_unload on the base loaded in float32) and its dump made on it
(heretic_dump.py dump model); such a run, or one whose adapter/ is an adapter exported by the IDE, also gets the A
stages below: the IDE's scoring of adapter/ itself (H4b), against its scoring of model/ and against Heretic's. `previous` is the output of
scripts/heretic_compare_scores.py on the same runs (its modes[<label>].S5.kl.ide, the merged model opened as its own
workspace); without it, or for a label it does not have, E5 is "not compared".

Per run, in its own isolated root <root>/<label>: the base snapshot is opened (only the base model is ever loaded),
then E0, then one directions_evaluate.evaluate request with path <run>/model, chat true, Heretic's system prompt,
its response prefix (or null), its max_response_length, and the scorers as heretic_dump.py scored them: KeywordRate
on the refusal prompts (print_responses, the run's [scorer.KeywordRate] settings), ResponseLength on the KL prompts
(print_responses, the run's [scorer.ResponseLength] settings), KLDivergence on the KL prompts.

Stages and their acceptance (decided before any run; a stage outside its bound is a finding, not a threshold):
    E0  eps: max |IDE - Heretic| of the base first-token logits of the KL prompts            reported
    E1  the weight comparison: the set compare.rows[].param equals the set of tensors this script finds different
        by reading both files itself (safetensors.safe_open, float32, row chunks of at most 16M values), the tied
        head excluded; each row's changed equal; norm within relative 1e-6 of this script's float64 norm. The
        counts against docs/DIRECTIONS.md (tiny ablation 5, Qwen2.5 ablation 31, Qwen2.5 ARA 8, Qwen3 ARA 44)
                                                                                                reported
    E1b reconstruction                     reconstruction.max_abs <= 2**-23 * m, m the largest |weight| of the
        changed tensors (one float32 ulp at the largest magnitude); inexact                     reported
    E2  base responses                     identical to scores.json base responses (ids), or diverging only where the
        margin between the two tokens is below max(1e-3, 4 eps)
    E3  modified responses                 identical to scores.json merged responses, or near-tie as E2 with the
        margin computed on the candidate (runner.use("current", directions_evaluate.build_candidate(...)[0]))
    E4  scores                             base Refusals and mean length equal scores.json base exactly and KL
        "0 (by definition)"; modified Refusals and mean length equal scores.json merged exactly unless every
        difference comes from an E2/E3 accepted divergence; modified KL within
        1e-6 + max(1e-3 KL, 3 eps sqrt(2 KL)) of scores.json merged kl
    E5  against the IDE's own S5           when previous has modes[<label>].S5.kl.ide:
        |KL(H4) - KL(S5)| <= 1e-6 + 1e-4 KL(S5); else "not compared"
    E6  nothing written                    ws.version and the patches/ listing unchanged; the merged directory's
        files have the same size and mtime before and after
    E7  refusals                           the base snapshot as path -> E-H16; <run> (no model at the top level)
        -> the refusal of an unrecognised directory, each text exactly (the listing of <run> in the second text is
        built here from os.listdir: sorted, a directory with "/", at most 8 names then " and <k> more")
    The A stages (H4b; only when <run>/adapter/adapter_config.json exists): a second request, the same body with
    path <run>/adapter, its responses recorded separately (decided before any run, like the others):
    A0  input                              the adapter request's input.kind is "adapter", the checkpoint request's
        "checkpoint"                                                                                  exact
    A1  the weight comparison              the adapter request's compare.rows have the checkpoint request's params
        in the same order and the same changed counts; each norm within relative 1e-6 of the checkpoint request's
    A1b the logical weights                every tensor the adapter changes, read through the adapter request's own
        store (base + adapter as the IDE reads it), against the same tensor of model/ read here (safetensors,
        float32): max |difference| <= 2**-23 * m, m the largest |weight| of these tensors in model/; the count
        of bit-equal tensors reported
    A2  responses                          the adapter request's base and modified responses equal, token for token,
        the checkpoint request's and Heretic's (scores.json base and merged); no near-tie allowance
    A3  scores                             base Refusals, mean length and KL equal the checkpoint request's exactly;
        modified Refusals and mean length equal Heretic's exactly; modified KL within
        1e-6 + max(1e-3 KL, 3 eps sqrt(2 KL)) of scores.json merged kl (E4's bound) and within 1e-6 + 1e-4 KL of
        the checkpoint request's (E5's form); the difference from the checkpoint request reported
    A4  nothing written in adapter/        its files have the same size and mtime before and after

Copied from scripts/heretic_compare_scores.py (not imported; that script is unchanged): Recorder, changed to keep
a list of responses per prompt ids in call order with the variant of each call (the base is generated before the
modified model); margin, changed to take the candidate's pieces (extra); compare_responses (passing extra on);
scorer_settings; kl_bound; and its S0 computation.

Every assertion prints a PASS or FAIL line with its numbers; <out.json> holds every number and a pass flag per
assertion; the exit status is 1 when one fails.
"""
import json
import math
import os
import sys
from pathlib import Path

CHUNK = 16_000_000                     # E1: values per read, at most
NORM_RELATIVE = 1e-6                   # E1
ULP = 2.0 ** -23                       # E1b
E5_RELATIVE, E5_FLOOR = 1e-4, 1e-6     # E5
HEADS = ("lm_head.weight", "embed_out.weight")
INDEX = "model.safetensors.index.json"
# docs/DIRECTIONS.md's counts of differing tensors (reported, E1): (model, method) -> tensors.
DOC_COUNTS = {("tiny", "ablation"): 5, ("qwen2.5", "ablation"): 31, ("qwen2.5", "ARA"): 8, ("qwen3", "ARA"): 44}
E_H16 = "path holds the open workspace's own base checkpoint: there is nothing to compare"
UNRECOGNISED = ("path holds no model the IDE reads: no PEFT adapter (adapter_config.json), no workspace (workspace.json) "
                "and no safetensors weights (model.safetensors, an index, or *.safetensors files); ")
LISTED = 8


def unrecognised(directory) -> str:
    """E7's expected refusal of a directory that holds no model (docs/DIRECTIONS.md), built from its listing."""
    names = sorted(os.listdir(directory))
    if not names:
        return UNRECOGNISED + "it is empty"
    shown = [n + ("/" if os.path.isdir(os.path.join(directory, n)) else "") for n in names[:LISTED]]
    more = f" and {len(names) - LISTED} more" if len(names) > LISTED else ""
    return UNRECOGNISED + f"it holds {len(names)} {'entry' if len(names) == 1 else 'entries'}: {', '.join(shown)}{more}"


# ------------------------------------------------------------------ the report
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
        return bool(self.labels) and all(v["pass"] for e in self.labels.values() for v in e.values()
                                         if isinstance(v, dict) and "pass" in v)


# ------------------------------------------------------------------ pure helpers (no model)
def parse_args(argv):
    """-> (snap, root, out, previous or None, {label: run dir})."""
    if len(argv) < 4:
        raise ValueError("give <base snapshot dir> <isolated IDE root> <out.json> and at least one <label>=<run dir>")
    snap, root, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    previous, runs = None, {}
    for a in argv[3:]:
        label, sep, value = a.partition("=")
        if not sep or not label or not value:
            raise ValueError(f"{a}: an argument is previous=<file> or <label>=<heretic run dir>")
        if label == "previous":
            previous = Path(value)
        elif label in runs:
            raise ValueError(f"{label}: given twice")
        else:
            runs[label] = Path(value)
    if not runs:
        raise ValueError("give at least one <label>=<heretic run dir>")
    return snap, root, out, previous, runs


def weight_files(directory) -> dict:
    """{tensor name: file} of a checkpoint directory: its index, else model.safetensors, else every *.safetensors
    (read independently of the IDE's reader)."""
    from safetensors import safe_open
    directory = Path(directory)
    if (directory / INDEX).is_file():
        wm = json.loads((directory / INDEX).read_text())["weight_map"]
        return {k: directory / v for k, v in wm.items()}
    files = [directory / "model.safetensors"] if (directory / "model.safetensors").is_file() else \
        sorted(p for p in directory.glob("*.safetensors") if p.is_file())
    out = {}
    for p in files:
        with safe_open(str(p), "pt") as f:
            for k in f.keys():
                out[k] = p
    return out


def ties(config) -> bool:
    """The base ties its head to the token embedding (config.json, top level or text_config)."""
    if not isinstance(config, dict):
        return False
    t = config.get("text_config")
    return config.get("tie_word_embeddings") is True or (isinstance(t, dict) and t.get("tie_word_embeddings") is True)


def row_chunks(shape, limit=CHUNK):
    """Row ranges (start, stop) of a tensor of ``shape``, each at most ``limit`` values (one row when a row is larger;
    None for a scalar)."""
    if len(shape) == 0:
        return [None]
    per_row = 1
    for s in shape[1:]:
        per_row *= int(s)
    step = max(1, limit // max(per_row, 1))
    return [(a, min(a + step, int(shape[0]))) for a in range(0, int(shape[0]), step)] or [(0, 0)]


def diff_stats(pairs):
    """One tensor's comparison from (base chunk, modified chunk) pairs, each converted to float32 first ->
    {changed, of, norm, max_abs, base_max_abs, modified_max_abs} (norm in float64)."""
    changed, of, sq, max_abs, bmax, omax = 0, 0, 0.0, 0.0, 0.0, 0.0
    for b, o in pairs:
        b, o = b.float(), o.float()
        changed += int((o != b).sum())
        of += b.numel()
        d = o.double() - b.double()
        sq += float(d.pow(2).sum())
        if d.numel():
            max_abs = max(max_abs, float(d.abs().max()))
            bmax, omax = max(bmax, float(b.abs().max())), max(omax, float(o.abs().max()))
    return {"changed": changed, "of": of, "norm": math.sqrt(sq), "max_abs": max_abs, "base_max_abs": bmax,
            "modified_max_abs": omax}


def read_compare(base_dir, modified_dir, tied_heads, limit=CHUNK):
    """E1's independent reading: every tensor in both checkpoints (the tied heads excepted), chunk by chunk ->
    {"rows": {name: diff_stats}, "only_base": [...], "only_modified": [...], "excluded": [...]}."""
    from safetensors import safe_open
    bf, mf = weight_files(base_dir), weight_files(modified_dir)
    excluded = sorted(n for n in set(bf) | set(mf) if n in tied_heads)
    names = [n for n in sorted(bf) if n in mf and n not in tied_heads]
    rows, handles = {}, {}

    def handle(p):
        if p not in handles:
            handles[p] = safe_open(str(p), "pt")
        return handles[p]
    for n in names:
        bs, ms = handle(bf[n]).get_slice(n), handle(mf[n]).get_slice(n)
        shape = list(bs.get_shape())
        if list(ms.get_shape()) != shape:
            rows[n] = {"shape_differs": [shape, list(ms.get_shape())]}
            continue
        def pairs(n=n, bs=bs, ms=ms, shape=shape):
            for r in row_chunks(shape, limit):
                if r is None:                                   # a scalar
                    yield handle(bf[n]).get_tensor(n), handle(mf[n]).get_tensor(n)
                else:
                    yield bs[r[0]:r[1]], ms[r[0]:r[1]]
        rows[n] = diff_stats(pairs())
    return {"rows": rows, "only_base": sorted(n for n in bf if n not in mf and n not in tied_heads),
            "only_modified": sorted(n for n in mf if n not in bf and n not in tied_heads), "excluded": excluded}


def e1_rows(ide_rows, mine):
    """E1's assertion from the IDE's compare.rows and read_compare's rows -> (ok, numbers)."""
    differ = {n for n, r in mine["rows"].items() if r.get("changed") or "shape_differs" in r}
    ide = {r["param"]: r for r in ide_rows}
    bad_changed, bad_norm, worst = [], [], 0.0
    for n in sorted(differ & set(ide)):
        m, i = mine["rows"][n], ide[n]
        if i["changed"] != m.get("changed"):
            bad_changed.append({"param": n, "ide": i["changed"], "script": m.get("changed")})
        if m.get("norm"):
            rel = abs(i["norm"] - m["norm"]) / m["norm"]
            worst = max(worst, rel)
            if rel > NORM_RELATIVE:
                bad_norm.append({"param": n, "ide": i["norm"], "script": m["norm"], "relative": rel})
    ok = set(ide) == differ and not bad_changed and not bad_norm
    return ok, {"ide_differ": len(ide), "script_differ": len(differ), "only_ide": sorted(set(ide) - differ)[:10],
                "only_script": sorted(differ - set(ide))[:10], "changed_unequal": bad_changed[:10],
                "norm_outside": bad_norm[:10], "worst_norm_relative": worst, "bound": NORM_RELATIVE}


def e1b(reconstruction, changed_rows) -> tuple[bool, dict]:
    """E1b: reconstruction.max_abs <= 2**-23 * m, m the largest |weight| of the changed tensors (the modified
    checkpoint's values, as read by read_compare)."""
    m = max((r["modified_max_abs"] for r in changed_rows), default=0.0)
    bound = ULP * m
    return reconstruction["max_abs"] <= bound, {"max_abs": reconstruction["max_abs"], "m": m, "bound": bound,
                                                "inexact": reconstruction["inexact"]}


def doc_count(dump):
    """The docs' count of differing tensors for this run's model and method, or None."""
    model, method = str(dump.get("model", "")).lower(), "ARA" if dump.get("modifier") == "ARA" else "ablation"
    key = "tiny" if "tiny" in model else "qwen2.5" if "qwen2.5" in model else "qwen3" if "qwen3" in model else None
    return DOC_COUNTS.get((key, method)), {"model": key, "method": method}


def kl_bound(kl, eps):
    return 1e-6 + max(1e-3 * kl, 3 * eps * math.sqrt(2 * max(kl, 0.0)))


def e5(kl_h4, previous, label):
    """E5 -> (ok or None when not compared, numbers)."""
    try:
        kl_s5 = previous["modes"][label]["S5"]["kl"]["ide"]
    except (KeyError, TypeError):
        return None, {"compared": "not compared", "kl_h4": kl_h4}
    bound = E5_FLOOR + E5_RELATIVE * abs(kl_s5)
    return abs(kl_h4 - kl_s5) <= bound, {"kl_h4": kl_h4, "kl_s5": kl_s5, "difference": abs(kl_h4 - kl_s5),
                                         "bound": bound}


def listing(directory) -> dict:
    """{relative path: [size, mtime_ns]} of every file under ``directory`` (E6)."""
    out = {}
    for dirpath, _, files in os.walk(directory):
        for fn in files:
            p = os.path.join(dirpath, fn)
            st = os.stat(p)
            out[os.path.relpath(p, directory)] = [st.st_size, st.st_mtime_ns]
    return dict(sorted(out.items()))


def scorer_settings(config, table, keys):
    t = (config.get("scorer") or {}).get(table) or {}
    return {k: t[k] for k in keys if k in t}


# ------------------------------------------------------------------ copied from heretic_compare_scores.py
class Recorder:
    """Every response the runner generates, by prompt ids: a list in call order, with each call's variant (the
    base is generated before the modified model)."""

    def __init__(self, ws):
        self.runner, self.got = ws.runner, {}
        real = self.runner.generate

        def generate(ids, *a, **kw):
            res = real(ids, *a, **kw)
            variant = a[0] if a else kw.get("variant", "current")
            self.got.setdefault(tuple(int(i) for i in ids), []).append(
                {"variant": variant, "response": [int(t) for t in res]})
            return res
        self.runner.generate = generate

    def response(self, ids, base):
        """The first response to ``ids`` generated on the base variant (``base``) or on another one, else None."""
        for e in self.got.get(tuple(int(i) for i in ids), []):
            if (e["variant"] == "base") == base:
                return e["response"]
        return None

    def close(self):
        del self.runner.generate


def margin(ws, prompt_ids, common, a, b, variant, extra=None):
    """|score(a) - score(b)| of the next token after prompt + common, as transformers' generate processes it (the
    checkpoint's generation config, the repetition penalty included); ``extra`` the candidate's pieces."""
    import torch
    r = ws.runner
    with r.lock:
        r.use(variant, extra)
        t = torch.tensor([list(prompt_ids) + list(common)])
        cfg = r.model.generation_config
        pad = cfg.pad_token_id if cfg.pad_token_id is not None else cfg.eos_token_id
        pad = pad[0] if isinstance(pad, (list, tuple)) else pad
        with torch.no_grad():
            o = r.model.generate(input_ids=t, attention_mask=torch.ones_like(t), max_new_tokens=1, output_scores=True,
                                 return_dict_in_generate=True, do_sample=False, pad_token_id=pad if pad is not None else 0)
    s = o.scores[0][0].float()
    return abs(float(s[a] - s[b]))


def compare_responses(ws, prompts_ids, ide, heretic, bound, variant, extra=None):
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
        m = margin(ws, p, x[:j], x[j], y[j], variant, extra)
        rows.append({"prompt": k, "step": j, "ide": x[j], "heretic": y[j], "margin": m, "accepted": m < bound})
    return {"prompts": len(prompts_ids), "identical": identical, "divergences": rows,
            "ok": all(r["accepted"] for r in rows)}


# ------------------------------------------------------------------ one run
def by_plugin(scores):
    return {s["plugin"]: s for s in scores}


def responses_stage(ws, rec, h, base, bound, variant, extra=None):
    """E2 / E3 for one scorer's prompts: the IDE's recorded responses against Heretic's."""
    prompts_ids = [list(x) for x in h["input_ids"]]
    ide_ids = [rec.response(x, base) for x in prompts_ids]
    if any(x is None for x in ide_ids):
        return {"error": "rendered ids differ from Heretic's, or a response was not generated; responses not compared",
                "missing": sum(x is None for x in ide_ids), "ok": False}
    return compare_responses(ws, prompts_ids, ide_ids, h["response_ids"], bound, variant, extra)


def score_equal(ide_value, her_value, resp, allow_divergence):
    accepted = isinstance(resp, dict) and resp.get("ok") and bool(resp.get("divergences"))
    return {"ide": ide_value, "heretic": her_value, "equal": ide_value == her_value,
            "difference_from_accepted_divergences": bool(accepted),
            "ok": ide_value == her_value or (allow_divergence and bool(accepted))}


def run_one(rep, label, run, snap, root, previous):
    import tomllib
    import torch
    from safetensors.torch import load_file

    from wsp import directions as DR
    from wsp import directions_evaluate as de_
    from wsp import directions_scores as DS
    from wsp.directions import DirectionError
    from wsp.session import Session

    quiet = lambda *a, **k: None  # noqa: E731
    dump = json.loads((run / "dump/dump.json").read_text())
    tens = load_file(str(run / "dump/dump.safetensors"))
    sc = json.loads((run / "dump/scores.json").read_text())
    config = tomllib.loads((run / "config.toml").read_text())
    kr_settings = scorer_settings(config, "KeywordRate", ("score_name", "keyword_markers"))
    rl_settings = scorer_settings(config, "ResponseLength", ("strip_responses", "short_response_threshold",
                                                             "short_response_score_length"))
    system, prefix = sc["system_prompt"], sc["response_prefix"] or None
    M = sc["max_response_length"]
    texts = {k: [p["user"] for p in v] for k, v in dump["prompts"].items()}
    merged = run / "model"
    rep.note(label, "run", {"run": str(run), "heretic_versions": sc.get("versions"), "batch_size": sc.get("batch_size"),
                            "max_response_length": M, "system_prompt": system, "response_prefix": prefix})

    session = Session(str(root / label))
    os.makedirs(session.root, exist_ok=True)
    ws = session.open(str(snap), log=quiet)
    try:
        rep.note(label, "ide_runner", DS.runner_row(ws)[0])

        # ---- E0: eps from the base first-token logits of the KL prompts (heretic_compare_scores.py's S0)
        kl_ids = DR.render_ids(ws, texts["kl"], "prompts", chat=True, system=system, prefix=prefix, chat_options={})
        lb = torch.stack([ws.runner.first_token_logits(x, "base") for x in kl_ids])
        eps = float((lb - tens["kl_logits"].float()).abs().max())
        bound = max(1e-3, 4 * eps)
        rep.note(label, "E0", {"eps": eps, "logit_scale": float(tens["kl_logits"].abs().max()),
                               "response_margin_bound": bound})
        del lb

        # ---- E6 before
        pdir = os.path.join(ws.dir, "patches")
        before = {"version": ws.version, "patches": sorted(os.listdir(pdir)) if os.path.isdir(pdir) else [],
                  "merged": listing(merged)}

        # ---- the request
        body = {"path": str(merged), "chat": True, "system": system, "prefix": prefix, "max_response_length": M,
                "scores": [{"plugin": "KeywordRate", "prompts": texts["refusal"], "print_responses": True, **kr_settings},
                           {"plugin": "ResponseLength", "prompts": texts["kl"], "print_responses": True, **rl_settings},
                           {"plugin": "KLDivergence", "prompts": texts["kl"]}]}
        rec = Recorder(ws)
        try:
            ans = de_.evaluate(ws, body)
        finally:
            rec.close()
        compare = ans["compare"]
        rep.note(label, "answer", {"compare": compare, "notes": ans["notes"], "changes": ans["changes"],
                                   "generation": ans["generation"], "seconds": ans["seconds"],
                                   "base": {s["plugin"]: {k: s.get(k) for k in ("value", "display", "per_prompt")}
                                            for s in ans["base"]["scores"]},
                                   "modified": {s["plugin"]: {k: s.get(k) for k in ("value", "display", "per_prompt",
                                                                                    "reference")}
                                                for s in ans["modified"]["scores"]},
                                   "modified_state": ans["modified"]["state"]})

        # ---- E1: the weight comparison, read independently
        base_cfg = json.loads((snap / "config.json").read_text()) if (snap / "config.json").is_file() else {}
        tied_heads = set(HEADS) if ties(base_cfg) else set()
        mine = read_compare(snap, merged, tied_heads)
        ok, numbers = e1_rows(compare["rows"], mine)
        rep.check(label, "E1 the differing tensors, their changed counts and norms", ok, **numbers,
                  only_base=mine["only_base"][:10], only_modified=mine["only_modified"][:10],
                  script_excluded=mine["excluded"], ide_tied=compare["tied"])
        expected, key = doc_count(dump)
        rep.note(label, "E1 counts against the docs (reported)",
                 {**key, "docs": expected, "found": compare["differ"],
                  "equal": None if expected is None else expected == compare["differ"]})

        # ---- E1b: reconstruction
        changed_rows = [r for r in mine["rows"].values() if r.get("changed")]
        ok, numbers = e1b(compare["reconstruction"], changed_rows)
        rep.check(label, "E1b the applied change reconstructs the checkpoint to one float32 ulp", ok, **numbers)

        # ---- E2: base responses
        e2 = {k: responses_stage(ws, rec, sc["base"][k], True, bound, "base") for k in ("refusal", "length")}
        rep.check(label, "E2 base responses identical or near-tie", all(v["ok"] for v in e2.values()),
                  **{k: {x: v.get(x) for x in ("prompts", "identical", "divergences", "error")} for k, v in e2.items()},
                  bound=bound)

        # ---- E3: modified responses (margins on the candidate)
        state = ans["modified"]["state"]
        unchanged = all(rec.response(x, False) == list(y) for k in ("refusal", "length")
                        for x, y in zip(sc["merged"][k]["input_ids"], sc["merged"][k]["response_ids"]))
        deltas = None
        try:
            if not unchanged and state == "candidate":
                checked = de_.check_evaluate(ws, body)
                deltas = de_.build_candidate(ws, checked["other"], de_.held_names(ws, checked["other"], compare))[0]
            e3 = {k: responses_stage(ws, rec, sc["merged"][k], False, bound, "current", deltas)
                  for k in ("refusal", "length")}
        finally:
            DS._release(ws, deltas)
            deltas = None
        rep.check(label, "E3 modified responses identical or near-tie", all(v["ok"] for v in e3.values()),
                  **{k: {x: v.get(x) for x in ("prompts", "identical", "divergences", "error")} for k, v in e3.items()},
                  bound=bound)

        # ---- E4: scores
        b, m = by_plugin(ans["base"]["scores"]), by_plugin(ans["modified"]["scores"])
        base_ref = score_equal(b["KeywordRate"]["value"], sc["base"]["refusal"]["score"]["value"], e2["refusal"], False)
        base_len = score_equal(b["ResponseLength"]["value"], sc["base"]["length"]["score"]["value"], e2["length"], False)
        base_kl = {"value": b["KLDivergence"]["value"], "display": b["KLDivergence"]["display"],
                   "ok": b["KLDivergence"]["value"] == 0 and b["KLDivergence"]["display"] == "0 (by definition)"}
        mod_ref = score_equal(m["KeywordRate"]["value"], sc["merged"]["refusal"]["score"]["value"], e3["refusal"], True)
        mod_len = score_equal(m["ResponseLength"]["value"], sc["merged"]["length"]["score"]["value"], e3["length"], True)
        kl_her, kl_ide = sc["merged"]["kl"]["score"]["value"], m["KLDivergence"]["value"]
        mod_kl = {"ide": kl_ide, "heretic": kl_her, "difference": abs(kl_ide - kl_her), "bound": kl_bound(kl_her, eps)}
        mod_kl_ok = mod_kl["difference"] <= mod_kl["bound"]
        rep.check(label, "E4 base Refusals and mean length equal Heretic's exactly, KL 0 by definition",
                  base_ref["ok"] and base_len["ok"] and base_kl["ok"], refusals=base_ref, length=base_len, kl=base_kl)
        rep.check(label, "E4 modified Refusals and mean length equal Heretic's evaluate_model",
                  mod_ref["ok"] and mod_len["ok"], refusals=mod_ref, length=mod_len)
        rep.check(label, "E4 modified KL within the bound of Heretic's evaluate_model", mod_kl_ok, **mod_kl)

        # ---- E5: against the IDE's own S5
        if previous is not None:
            ok, numbers = e5(kl_ide, previous, label)
        else:
            ok, numbers = None, {"compared": "not compared (no previous)", "kl_h4": kl_ide}
        if ok is None:
            rep.note(label, "E5 KL against the merged model opened as its own workspace", numbers)
        else:
            rep.check(label, "E5 KL against the merged model opened as its own workspace", ok, **numbers)

        # ---- E7: refusals
        for name, path, want in (("E7 the base snapshot as path is refused (E-H16)", snap, E_H16),
                                 ("E7 the run directory as path is refused (unrecognised)", run, unrecognised(run))):
            try:
                de_.check_evaluate(ws, {"path": str(path), "check": True})
                got = None
            except DirectionError as e:
                got = str(e)
            rep.check(label, name, got == want, got=got, want=want)

        # ---- A0 to A4: the adapter itself (H4b)
        adapter = run / "adapter"
        if (adapter / "adapter_config.json").is_file():
            adapter_stages(rep, label, ws, de_, body, ans, rec, sc, merged, adapter, eps)

        # ---- E6 after
        after = {"version": ws.version, "patches": sorted(os.listdir(pdir)) if os.path.isdir(pdir) else [],
                 "merged": listing(merged)}
        rep.check(label, "E6 nothing written", before == after, version=[before["version"], after["version"]],
                  patches=[len(before["patches"]), len(after["patches"])],
                  merged_files=len(after["merged"]), merged_same=before["merged"] == after["merged"])
    finally:
        session.unload()


def adapter_stages(rep, label, ws, de_, body, ans, rec, sc, merged, adapter, eps):
    """A0 to A4: the same request with path <run>/adapter, against the checkpoint request ``ans`` (its responses in
    ``rec``) and Heretic's scores ``sc``."""
    import torch
    from safetensors import safe_open
    before = listing(adapter)
    rec2 = Recorder(ws)
    try:
        got = de_.evaluate(ws, {**body, "path": str(adapter)})
    finally:
        rec2.close()
    rep.note(label, "A answer", {"input": got["input"], "compare": got["compare"], "notes": got["notes"],
                                 "changes": got["changes"], "modified_state": got["modified"]["state"]})
    rep.check(label, "A0 the adapter is read as an adapter, model/ as a checkpoint",
              got["input"]["kind"] == "adapter" and ans["input"]["kind"] == "checkpoint",
              adapter=got["input"]["kind"], checkpoint=ans["input"]["kind"])
    a_rows, c_rows = got["compare"]["rows"], ans["compare"]["rows"]
    rel = [abs(x["norm"] - y["norm"]) / y["norm"] if y["norm"] else abs(x["norm"]) for x, y in zip(a_rows, c_rows)]
    ok = [r["param"] for r in a_rows] == [r["param"] for r in c_rows] and \
        [r["changed"] for r in a_rows] == [r["changed"] for r in c_rows] and all(x <= NORM_RELATIVE for x in rel)
    rep.check(label, "A1 the weight comparison equals the checkpoint request's", ok,
              params=[len(a_rows), len(c_rows)], worst_norm_relative=max(rel, default=0.0), bound=NORM_RELATIVE)
    other = de_.check_evaluate(ws, {"path": str(adapter), "check": True})["other"]
    files = weight_files(merged)
    worst, m, equal, names = 0.0, 0.0, 0, sorted(other.may_differ or ())
    for n in names:
        with safe_open(str(files[n]), "pt") as f:
            want = f.get_tensor(n).float()
        have = other.store.read(n)
        worst = max(worst, float((have - want).abs().max()))
        m = max(m, float(want.abs().max()))
        equal += bool(torch.equal(have, want))
    rep.check(label, "A1b base + adapter as the IDE reads it equals PEFT's merge to one float32 ulp",
              worst <= ULP * m, max_abs=worst, m=m, bound=ULP * m, tensors=len(names), bit_equal=equal)
    rows = {}
    ok = True
    for k, plugin in (("refusal", "KeywordRate"), ("length", "ResponseLength")):
        for side, base in (("base", True), ("merged", False)):
            prompts = [list(x) for x in sc[side][k]["input_ids"]]
            mine = [rec2.response(x, base) for x in prompts]
            theirs = [rec.response(x, base) for x in prompts]
            heretic = [list(y) for y in sc[side][k]["response_ids"]]
            same = sum(a == b == c for a, b, c in zip(mine, theirs, heretic))
            rows[f"{k}.{side}"] = {"prompts": len(prompts), "identical": same}
            ok = ok and same == len(prompts)
    rep.check(label, "A2 responses token for token equal the checkpoint request's and Heretic's", ok, **rows)
    ba, ma = by_plugin(got["base"]["scores"]), by_plugin(got["modified"]["scores"])
    bc, mc = by_plugin(ans["base"]["scores"]), by_plugin(ans["modified"]["scores"])
    base_same = all(ba[p]["value"] == bc[p]["value"] for p in ("KeywordRate", "ResponseLength", "KLDivergence"))
    ref = ma["KeywordRate"]["value"] == sc["merged"]["refusal"]["score"]["value"]
    length = ma["ResponseLength"]["value"] == sc["merged"]["length"]["score"]["value"]
    kl_a, kl_c, kl_h = ma["KLDivergence"]["value"], mc["KLDivergence"]["value"], sc["merged"]["kl"]["score"]["value"]
    to_h, to_c = abs(kl_a - kl_h), abs(kl_a - kl_c)
    bh, bc_ = kl_bound(kl_h, eps), E5_FLOOR + E5_RELATIVE * abs(kl_c)
    rep.check(label, "A3 scores: base as the checkpoint request's, modified as Heretic's, KL within the bounds",
              base_same and ref and length and to_h <= bh and to_c <= bc_, base_equal=base_same,
              refusals=[ma["KeywordRate"]["value"], sc["merged"]["refusal"]["score"]["value"]],
              length=[ma["ResponseLength"]["value"], sc["merged"]["length"]["score"]["value"]],
              kl={"adapter": kl_a, "checkpoint": kl_c, "heretic": kl_h, "to_heretic": to_h, "bound_heretic": bh,
                  "to_checkpoint": to_c, "bound_checkpoint": bc_})
    rep.check(label, "A4 nothing written in the adapter directory", listing(adapter) == before, files=len(before))


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    try:
        snap, root, out, previous_path, runs = parse_args(argv)
    except ValueError as e:
        print(f"{e}\n\n{__doc__}", file=sys.stderr)
        return 2
    sys.path.insert(0, os.getcwd())
    from importlib.metadata import version
    previous = json.loads(previous_path.read_text()) if previous_path is not None else None
    rep = Report()
    result = {"ide_versions": {p: version(p) for p in ("torch", "transformers", "tokenizers", "safetensors")},
              "base": str(snap), "previous": str(previous_path) if previous_path else None}
    out.parent.mkdir(parents=True, exist_ok=True)
    for label, run in runs.items():
        try:
            run_one(rep, label, run, snap, root, previous)
        except Exception as e:                       # a run that cannot finish is a failed assertion, not a stop
            rep.check(label, "the run finished", False, error=f"{type(e).__name__}: {e}")
        out.write_text(json.dumps({**result, "labels": rep.labels, "pass": rep.ok()}, indent=1, default=str))
    result = {**result, "labels": rep.labels, "pass": rep.ok()}
    out.write_text(json.dumps(result, indent=1, default=str))
    print(f"{'ok' if result['pass'] else 'NOT OK'}: {out}", flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
