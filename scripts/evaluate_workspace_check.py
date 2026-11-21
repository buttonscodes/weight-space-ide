"""Check the IDE's scoring of another workspace (wsp.directions_evaluate, path = a workspace; plan item H4c).

Run by the orchestrator only (it loads models), with the IDE's interpreter, with the IDE's `weight-space-ide`
directory as the current directory, under `python -m wsp.memory_guard`, one process at a time, offline:

    python scripts/evaluate_workspace_check.py <base snapshot dir> <isolated IDE root> <out.json> \\
        <label>=<workspace dir>,<heretic run dir> ...

<workspace dir> is a workspace over <base snapshot dir> with committed patches (for example the workspaces
scripts/heretic_compare_adapter.py --stage ide leaves in its root: a directional ablation, an ARA fit); <heretic run
dir> a finished Heretic run with its dump (heretic_dump.py), whose prompts, rendering and scorer settings make the
request, as scripts/heretic_compare_evaluate.py builds it. Per label, in <root>/<label>, the base snapshot is opened
as a second workspace B (so both are over the same files) and scored with path <workspace dir>; then the workspace
itself is opened in a session of its own and scores its own edit (state current, then state base) with the same
fields. Acceptance, decided before any run:

    W0  read-only        every file under <workspace dir>: size, mtime and SHA-256 equal before and after B's
                         request, and no file or directory added or removed                              exact
    W1  input            input.kind "workspace", same_base true, candidate "patches", the enabled patches' ids in
                         stack order                                                                    exact
    W2  the comparison   compare.rows' params are the tensors whose current weights differ from the base, as this
                         script reads them from the workspace (Workspace.read current against base); each row's
                         changed equal, norm within relative 1e-6 of this script's float64 norm
    W3  responses        every response B's request generates (base and modified, by prompt ids) equals, token for
                         token, the one the workspace's own scoring generates for the same ids and state  exact
    W4  scores           B's base and modified scores equal the workspace's own state base and state current
                         scores, every field but the KL reference's metadata (values, displays, per-prompt KL,
                         responses)                                                                      exact

Every assertion prints a PASS or FAIL line with its numbers; <out.json> holds every number and a pass flag per
assertion; the exit status is 1 when one fails.
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

NORM_RELATIVE = 1e-6


def tree(directory) -> dict:
    """{relative path: [size, mtime_ns, sha256]} of every file and {relative dir/: None} of every directory (W0)."""
    out = {}
    for dirpath, dirs, files in os.walk(directory):
        for d in dirs:
            out[os.path.relpath(os.path.join(dirpath, d), directory) + "/"] = None
        for fn in files:
            p = os.path.join(dirpath, fn)
            st = os.stat(p)
            with open(p, "rb") as f:
                out[os.path.relpath(p, directory)] = [st.st_size, st.st_mtime_ns, hashlib.sha256(f.read()).hexdigest()]
    return dict(sorted(out.items()))


def strip(scores):
    return [{k: v for k, v in s.items() if k != "reference"} for s in scores]


def parse_args(argv):
    if len(argv) < 4:
        raise ValueError("give <base snapshot dir> <isolated IDE root> <out.json> and at least one "
                         "<label>=<workspace dir>,<heretic run dir>")
    snap, root, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    labels = {}
    for a in argv[3:]:
        label, sep, value = a.partition("=")
        wsd, comma, run = value.partition(",")
        if not sep or not comma or not label or not wsd or not run or label in labels:
            raise ValueError(f"{a}: an argument is <label>=<workspace dir>,<heretic run dir>, each label once")
        labels[label] = (Path(wsd), Path(run))
    return snap, root, out, labels


def run_one(rep, label, wsd, run, snap, root):
    import heretic_compare_evaluate as H
    from wsp import directions_evaluate as de_
    from wsp import directions_scores as DS
    from wsp.session import Session
    from wsp.workspace import Workspace

    quiet = lambda *a, **k: None  # noqa: E731
    body = request_body(run)
    rep.note(label, "request", {k: v for k, v in body.items() if k != "scores"})

    # ---- B over the same base scores the other workspace
    sb = Session(str(root / label))
    os.makedirs(sb.root, exist_ok=True)
    wb = sb.open(str(snap), log=quiet)
    before = tree(wsd)
    rec_b = H.Recorder(wb)
    try:
        ans = de_.evaluate(wb, {**body, "path": str(wsd)})
    finally:
        rec_b.close()
    after = tree(wsd)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    rep.check(label, "W0 the other workspace is unchanged", before == after, files=len(before), differ=changed[:10])
    rep.note(label, "answer", {"input": ans["input"], "compare": ans["compare"], "notes": ans["notes"],
                               "changes": ans["changes"], "modified_state": ans["modified"]["state"]})
    sb.unload()

    # ---- the workspace read here (read-only), for W1 and W2
    other = Workspace.load(str(wsd), metadata_only=True)
    enabled = [p.id for p in other.patches if p.enabled and p.strength]
    inp = ans["input"]
    rep.check(label, "W1 the input is the workspace over the same base, scored by its patches",
              inp.get("kind") == "workspace" and inp.get("same_base") is True and inp.get("candidate") == "patches"
              and [p["id"] for p in inp.get("patches", [])] == enabled, input=inp, enabled=enabled)
    mine = {}
    for n in other.touched():
        c, b = other.read(n, (), "current", cache=False), other.read(n, (), "base", cache=False)
        k = int((c != b).sum())
        if k:
            mine[n] = {"changed": k, "norm": math.sqrt(float((c.double() - b.double()).pow(2).sum()))}
    rows = {r["param"]: r for r in ans["compare"]["rows"]}
    worst = max((abs(rows[n]["norm"] - m["norm"]) / m["norm"] for n, m in mine.items() if n in rows and m["norm"]),
                default=0.0)
    ok = set(rows) == set(mine) and all(rows[n]["changed"] == m["changed"] for n, m in mine.items()) and \
        worst <= NORM_RELATIVE
    rep.check(label, "W2 the differing tensors, their changed counts and norms", ok, ide=sorted(rows), script=sorted(mine),
              worst_norm_relative=worst, bound=NORM_RELATIVE)
    del other

    # ---- the workspace scores its own edit
    sa = Session(str(wsd.parent))
    wa = sa.open(str(wsd), log=quiet)
    rec_a = H.Recorder(wa)
    try:
        own_current = DS.score(wa, {**body, "state": "current"})
        own_base = DS.score(wa, {**body, "state": "base"})
    finally:
        rec_a.close()
    sa.unload()
    keys = sorted(set(rec_b.got) | set(rec_a.got))
    bad, compared = [], 0
    for ids in keys:
        for base in (True, False):
            x, y = rec_b.response(list(ids), base), rec_a.response(list(ids), base)
            if x is None and y is None:
                continue
            compared += 1
            if x != y:
                bad.append({"prompt_tokens": len(ids), "base": base, "b": x, "own": y})
    rep.check(label, "W3 responses token for token equal the workspace's own", not bad and compared > 0,
              compared=compared, differ=bad[:5])
    sm, sbase = strip(ans["modified"]["scores"]), strip(ans["base"]["scores"])
    om, ob = strip(own_current["scores"]), strip(own_base["scores"])
    kl = lambda ss: next(s["value"] for s in ss if s["plugin"] == "KLDivergence")  # noqa: E731
    rep.check(label, "W4 the scores equal the workspace's own scoring of its edit", sm == om and sbase == ob,
              modified_equal=sm == om, base_equal=sbase == ob, kl={"b": kl(sm), "own": kl(om)},
              values={"b": [s["value"] for s in sm], "own": [s["value"] for s in om]})


def request_body(run):
    """The scores request of scripts/heretic_compare_evaluate.py for one Heretic run (without path)."""
    import tomllib
    import heretic_compare_evaluate as H
    dump = json.loads((run / "dump/dump.json").read_text())
    sc = json.loads((run / "dump/scores.json").read_text())
    config = tomllib.loads((run / "config.toml").read_text())
    kr = H.scorer_settings(config, "KeywordRate", ("score_name", "keyword_markers"))
    rl = H.scorer_settings(config, "ResponseLength", ("strip_responses", "short_response_threshold",
                                                      "short_response_score_length"))
    texts = {k: [p["user"] for p in v] for k, v in dump["prompts"].items()}
    return {"chat": True, "system": sc["system_prompt"], "prefix": sc["response_prefix"] or None,
            "max_response_length": sc["max_response_length"],
            "scores": [{"plugin": "KeywordRate", "prompts": texts["refusal"], "print_responses": True, **kr},
                       {"plugin": "ResponseLength", "prompts": texts["kl"], "print_responses": True, **rl},
                       {"plugin": "KLDivergence", "prompts": texts["kl"]}]}


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    try:
        snap, root, out, labels = parse_args(argv)
    except ValueError as e:
        print(f"{e}\n\n{__doc__}", file=sys.stderr)
        return 2
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib.metadata import version
    import heretic_compare_evaluate as H
    rep = H.Report()
    result = {"ide_versions": {p: version(p) for p in ("torch", "transformers")}}
    out.parent.mkdir(parents=True, exist_ok=True)
    for label, (wsd, run) in labels.items():
        try:
            run_one(rep, label, wsd, run, snap, root)
        except Exception as e:                       # a run that cannot finish is a failed assertion, not a stop
            rep.check(label, "the run finished", False, error=f"{type(e).__name__}: {e}")
    result = {**result, "labels": rep.labels, "pass": rep.ok()}
    out.write_text(json.dumps(result, indent=1, default=str))
    print(f"{'ok' if result['pass'] else 'NOT OK'}: {out}", flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
