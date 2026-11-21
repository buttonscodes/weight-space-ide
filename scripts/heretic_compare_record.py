"""Compare the IDE's reproduction records with Heretic's reproduce.json, and re-run the IDE's edits from their records.

Run from the IDE's `weight-space-ide` directory with the IDE's interpreter (stage search also needs Optuna 5,
requirements-search.txt), with HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
HF_HOME=<isolated hf-home>:

    python -m wsp.memory_guard --log <log> --forecast-gib <g> -- python scripts/heretic_compare_record.py \\
        --stage search <model snapshot dir> <isolated IDE root> <out dir> <label>=<heretic search run dir> ...
    python scripts/heretic_compare_record.py --stage map <out.json> <label>=<IDE reproduce.json>,<Heretic reproduce.json> ...
    python -m wsp.memory_guard --log <log> --forecast-gib <g> -- python scripts/heretic_compare_record.py \\
        --stage rerun <model snapshot dir> <isolated IDE root> <adapter out dir> <out.json> <label> ...

Stage search (loads the model) makes the IDE side of a search run for stage map: in <root>/<label>, the measurement
"oracle" from the run's dataset specifications (config.toml's modifier table; chat, the dump's system prompt and
prefix), a search with the run's seed, n_trials, n_startup_trials, max_response_length, modifier settings and
scorers (their prompts given as the config's dataset specifications) as scripts/heretic_compare_search.py --stage R
makes one, run to the end; its first front trial committed with apply_trial; then <out>/<label>/ide_reproduce.json
is what GET /api/patches/{pid}/record/heretic?model=<run's model>&model_commit=<run's commit> answers, and ide.json
the patch, digest and record. Asserted: SS1 the search finished, SS2 the trial committed, SS3 the record is
Heretic-runnable.

Stage map (no model): the IDE's file (written by stage search, or by scripts/heretic_compare_adapter.py --stage ide
as <label>/ide_reproduce.json) against Heretic's file for the same run (scripts/heretic_reproduce_json.py), field by
field as docs/DIRECTIONS.md's table of the mapping says (written out again here, not imported from wsp):

    S-C1  settings.model, model_commit, response_prefix, system_prompt, modifiers[0].plugin    equal
    S-C2  each key of settings.modifier.<Class>; dataset specifications by dataset, commit,
          split, column, prefix, suffix and system_prompt (a single one as a list of one)      equal
          (print_loss is reported: it only prints each step's loss, modifiers/ara.py:381)
    S-C3  parameters                                                                          JSON-equal
    S-C4  settings.seed: ARA and search runs                                                  equal
          an ablation made by hand has no seed in the IDE: its file says 0 (asserted), Heretic's
          seed is reported; Heretic's seed reaches an ablation only through row_normalization
          full's randomized SVD (modifiers/abliteration.py:435), and for full the IDE's file
          must name that difference in wsp.differences (asserted)
    S-C5  a search run (the IDE file lists scorers): n_trials, n_startup_trials, the scorers'
          plugins and optimizations, each trial score's name and value (refusal counts
          exact; KL within 1e-6 + 1e-3 * KL)                                                  as stated
    S-C6  each settings key only in Heretic's file equals Heretic's default for that field
          (typed below from heretic/config.py); else "the IDE omitted a non-default setting"  one assertion per key
          Reported instead, because they cannot change the weights heretic --reproduce makes
          (NOT_WEIGHTS below, each with where Heretic reads it): the answers to Heretic's
          prompts (checkpoint_action, trial_index, model_action), and, for an edit made by hand
          (the IDE's file lists no scorers), max_response_length and the scorer tables;
          max_response_length is reported only when the IDE's file sets response_prefix
          (asserted), since Heretic's prefix detection generates responses
    reported only: timestamp, environment.*, hashes, system; settings keys only in the IDE's file; the values of the
    other settings both files hold.

Stage rerun (loads the model): for each label of scripts/heretic_compare_adapter.py --stage ide (its <adapter out
dir>/<label>/ide.json names the patch; the workspace is <root>/<label>):

    S-E1  the patch switched off; reproduce(dry_run=True): verdict identical, digest == recorded_digest, no new patch
    S-E2  measured again from GET .../record?remeasure=true's body under the suggested name;
          reproduce(measurement=<name>, dry_run=True): means.equal true and verdict identical
    S-E3  reproduce(dry_run=False): a new patch whose pieces are torch.equal to the original's, origin.kind
          "reproduction"

Every assertion prints a PASS or FAIL line with its numbers; the JSON output holds every number and a pass flag per
assertion; the exit status is 1 when one fails.
"""
import json
import os
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, os.getcwd())

SPEC_KEYS = ("dataset", "commit", "split", "column", "prefix", "suffix", "system_prompt")
# Heretic's defaults of every settings field that is serialized (heretic/config.py at 64e9991, the fields without
# exclude=True). "model" has no default.
HERETIC_DEFAULTS = {
    "model_commit": None, "dtypes": ["auto", "float16", "bfloat16", "float32"], "quantization": "none",
    "device_map": "auto", "max_memory": None, "offload_outputs_to_cpu": True, "batch_size": 0,
    "max_response_length": 100, "response_prefix": None,
    "scorers": [{"plugin": "heretic.scorers.keyword_rate.KeywordRate", "optimization": "minimize", "instance_name": None},
                {"plugin": "heretic.scorers.kl_divergence.KLDivergence", "optimization": "minimize",
                 "instance_name": None}],
    "modifiers": [{"plugin": "heretic.modifiers.ara.ARA", "instance_name": None}],
    "n_trials": 100, "n_startup_trials": 30, "seed": None, "modified_model_type": "decensored",
    "max_shard_size": "5GB", "export_strategy": None, "checkpoint_action": None, "trial_index": None,
    "n_additional_trials": None, "model_action": None, "upload_repo_private": None,
    "upload_reproducibility_information": None, "upload_tags": ["uncensored", "decensored", "abliterated"],
    "ignore_mismatches": None, "system_prompt": "You are a helpful assistant."}
REPORTED = ("timestamp", "environment", "hashes", "system")
# Settings that cannot change the weights `heretic --reproduce` makes from the file, with where Heretic reads them
# (64e9991). The first three answer Heretic's prompts; the last two matter to scoring only and are reported for an
# edit made by hand, which has no scorers.
RUN_CONTROL = {
    "checkpoint_action": "the answer to Heretic's prompt about an existing checkpoint (config.py:517)",
    "trial_index": "the answer to Heretic's prompt for the trial of the front to use (config.py:522)",
    "model_action": "the answer to Heretic's prompt for what to do with the modified model (config.py:532)",
}
SCORING_ONLY = {
    "max_response_length": "read only where responses are generated (model.py:651): by the scorers, and by the "
                           "prefix detection, which runs only when response_prefix is unset (main.py:482)",
    "scorer": "the scorers' own settings; an edit made by hand has no scorers",
}
LOGGING = {"print_loss": "prints each optimization step's loss (modifiers/ara.py:381)"}
FULL_DIFFERENCE = "svd_lowrank"        # the text of wsp.differences that names row_normalization full's randomized SVD


class Report:
    def __init__(self):
        self.labels = {}

    def check(self, label, name, ok, **numbers):
        self.labels.setdefault(label, {})[name] = {"pass": bool(ok), **numbers}
        print(f"{'PASS' if ok else 'FAIL'} {label} {name} {json.dumps(numbers, default=str)[:600]}", flush=True)
        return bool(ok)

    def note(self, label, name, value):
        self.labels.setdefault(label, {})[name] = value

    def ok(self):
        return bool(self.labels) and all(v["pass"] for e in self.labels.values() for v in e.values()
                                         if isinstance(v, dict) and "pass" in v)


def finish(rep, out, stage):
    result = {"stage": stage, "labels": rep.labels, "pass": rep.ok()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    print(f"{'ok' if result['pass'] else 'NOT OK'}: {out}", flush=True)
    return 0 if result["pass"] else 1


# ------------------------------------------------------------------ stage map (no model)
def as_list(v):
    return v if isinstance(v, list) else [v]


def is_spec(v):
    return isinstance(v, dict) and "dataset" in v or isinstance(v, list) and bool(v) and all(
        isinstance(x, dict) and "dataset" in x for x in v)


def specs_equal(a, b):
    la, lb = as_list(a), as_list(b)
    return len(la) == len(lb) and all(isinstance(x, dict) and isinstance(y, dict) and
                                      all(x.get(k) == y.get(k) for k in SPEC_KEYS) for x, y in zip(la, lb))


def kl_ok(ide, her):
    return abs(ide - her) <= 1e-6 + 1e-3 * abs(her)


def stage_map(out, pairs):
    rep = Report()
    for label, (ide_path, her_path) in pairs.items():
        ide, her = json.loads(Path(ide_path).read_text()), json.loads(Path(her_path).read_text())
        si, sh = ide["settings"], her["settings"]
        for key in ("model", "model_commit", "response_prefix", "system_prompt"):
            rep.check(label, f"S-C1 settings.{key}", si.get(key) == sh.get(key), ide=si.get(key), heretic=sh.get(key))
        pi, ph = [(x.get("modifiers") or [{}])[0].get("plugin") for x in (si, sh)]
        rep.check(label, "S-C1 settings.modifiers[0].plugin", pi == ph, ide=pi, heretic=ph)
        cls = (ph or "").rsplit(".", 1)[-1]
        ti, th = (si.get("modifier") or {}).get(cls) or {}, (sh.get("modifier") or {}).get(cls) or {}
        not_weights = {}
        for key in sorted(set(ti) | set(th)):
            a, b = ti.get(key, "<absent>"), th.get(key, "<absent>")
            if key in LOGGING:
                not_weights[f"modifier.{cls}.{key}"] = {"ide": a, "heretic": b, "why": LOGGING[key]}
                continue
            same = specs_equal(a, b) if is_spec(a) or is_spec(b) else a == b
            rep.check(label, f"S-C2 settings.modifier.{cls}.{key}", same, ide=a, heretic=b)
        rep.check(label, "S-C3 parameters", json.dumps(ide.get("parameters"), sort_keys=True)
                  == json.dumps(her.get("parameters"), sort_keys=True), ide=ide.get("parameters"),
                  heretic=her.get("parameters"))
        by_hand = not si.get("scorers")
        if by_hand and cls == "Abliteration":
            rep.check(label, "S-C4 settings.seed of an ablation made by hand is 0", si.get("seed") == 0,
                      ide=si.get("seed"), heretic=sh.get("seed"))
            if ti.get("row_normalization") == "full":
                named = [d for d in (ide.get("wsp") or {}).get("differences") or [] if FULL_DIFFERENCE in d]
                rep.check(label, "S-C4 the file names row_normalization full's seeded SVD as a difference",
                          bool(named), differences=named)
        else:
            rep.check(label, "S-C4 settings.seed", si.get("seed") == sh.get("seed"), ide=si.get("seed"),
                      heretic=sh.get("seed"))
        if si.get("scorers"):
            for key in ("n_trials", "n_startup_trials"):
                rep.check(label, f"S-C5 settings.{key}", si.get(key) == sh.get(key), ide=si.get(key), heretic=sh.get(key))
            a = [(s.get("plugin"), s.get("optimization")) for s in si.get("scorers") or []]
            b = [(s.get("plugin"), s.get("optimization")) for s in sh.get("scorers") or []]
            rep.check(label, "S-C5 scorers' plugins and optimizations", a == b, ide=a, heretic=b)
            have = {s["name"]: s["score"]["value"] for s in ide.get("scores") or []}
            for s in her.get("scores") or []:
                name, value = s["name"], s["score"]["value"]
                mine = have.get(name)
                if "kl" in name.lower():
                    ok = mine is not None and kl_ok(mine, value)
                    bound = 1e-6 + 1e-3 * abs(value)
                else:
                    ok, bound = mine == value, 0
                rep.check(label, f"S-C5 score {name}", ok, ide=mine, heretic=value, bound=bound)
        for key in sorted(set(sh) - set(si)):
            why = RUN_CONTROL.get(key) or (SCORING_ONLY.get(key) if by_hand else None)
            if why is not None:
                not_weights[key] = {"heretic": sh[key], "default": HERETIC_DEFAULTS.get(key, "<a plugin table>"),
                                    "why": why}
                if key == "max_response_length":
                    rep.check(label, "S-C6 settings.max_response_length cannot reach the weights: response_prefix "
                                     "is set", si.get("response_prefix") is not None, ide=si.get("response_prefix"))
                continue
            if key in HERETIC_DEFAULTS:
                ok = sh[key] == HERETIC_DEFAULTS[key]
                rep.check(label, f"S-C6 settings.{key} only in Heretic's file", ok, heretic=sh[key],
                          default=HERETIC_DEFAULTS[key], **({} if ok else {"why": "the IDE omitted a non-default setting"}))
            else:
                rep.check(label, f"S-C6 settings.{key} only in Heretic's file", False, heretic=sh[key],
                          default="<no default: a plugin table>", why="the IDE omitted a non-default setting")
        rep.note(label, "reported, cannot change the weights", not_weights)
        print(f"     {label}: reported, cannot change the weights: {', '.join(sorted(not_weights)) or 'none'}", flush=True)
        rep.note(label, "reported", {
            **{k: {"ide": ide.get(k), "heretic": her.get(k)} for k in REPORTED},
            "settings only in the IDE's file": {k: si[k] for k in sorted(set(si) - set(sh))},
            "other settings in both": {k: {"ide": si[k], "heretic": sh[k]} for k in sorted(set(si) & set(sh))
                                       if k not in ("model", "model_commit", "response_prefix", "system_prompt",
                                                    "modifiers", "modifier", "seed")},
            "top-level keys only in the IDE's file": sorted(set(ide) - set(her))})
    return rep


# ------------------------------------------------------------------ stage search (loads the model)
def stage_search(snap, root, out, runs):
    from wsp import directions as DR
    from wsp import directions_record as REC
    from wsp import directions_search as DSR
    from wsp.session import Session
    rep = Report()
    for label, run in runs.items():
        run = Path(run)
        config = tomllib.loads((run / "config.toml").read_text())
        dump = json.loads((run / "dump" / "dump.json").read_text())
        modifiers = config.get("modifiers") or [{"plugin": "heretic.modifiers.ara.ARA"}]
        plugin = modifiers[0]["plugin"].rsplit(".", 1)[-1]
        row = DSR.MODIFIERS[plugin]
        table = (config.get("modifier") or {}).get(plugin) or {}
        settings = {**row["settings"], **{k: v for k, v in table.items() if k in row["settings"]}}
        ara = row["method"] == "ara"
        system, prefix = dump["system_prompt"], dump["response_prefix"] or None
        sess = Session(str(root / label))
        os.makedirs(sess.root, exist_ok=True)
        ws = sess.open(str(snap), log=lambda *a, **k: None)
        try:
            for p in list(ws.patches):
                if p.enabled:
                    ws.set_patch(p.id, enabled=False)
            extra = {} if ara else {"winsorization_quantile": table.get("winsorization_quantile", 1.0)}
            DR.measure(ws, "oracle", table["good_prompts"], table["bad_prompts"], chat=True, system=system,
                       prefix=prefix, variant="base", replace=True, module_io=ara, **extra)
            scorers = []
            tables = config.get("scorer") or {}
            for s in config.get("scorers") or [{"plugin": "heretic.scorers.keyword_rate.KeywordRate",
                                                "optimization": "minimize"},
                                               {"plugin": "heretic.scorers.kl_divergence.KLDivergence",
                                                "optimization": "minimize"}]:
                cls = s["plugin"].rsplit(".", 1)[-1]
                key = cls + (f"_{s['instance_name']}" if s.get("instance_name") else "")
                t = tables.get(key) or {}
                scorers.append({"plugin": cls, "optimization": s["optimization"], "prompts": t["prompts"],
                                **({"instance_name": s["instance_name"]} if s.get("instance_name") else {}),
                                **{k: v for k, v in t.items() if k != "prompts" and not isinstance(v, dict)}})
            name = "s1"
            if name in [e["name"] for e in DSR.entries(ws)]:
                DSR.delete(ws, name)
            body = {"name": name, "modifiers": [{"plugin": plugin, "direction": "oracle", **settings}], "scorers": scorers,
                    "n_trials": config.get("n_trials", 100), "n_startup_trials": config.get("n_startup_trials", 30),
                    "seed": config.get("seed"), "max_response_length": config.get("max_response_length", 100),
                    "chat": True, "system": system, "prefix": prefix}
            DSR.create(ws, DSR.check_search(ws, body))
            res = DSR.run(ws, name)
            if not rep.check(label, "SS1 the search finished", res.get("state") == "finished", state=res.get("state"),
                             reason=res.get("reason")):
                continue
            recs = DSR.trial_records(ws, name)
            s = DSR.load(ws, name)
            front = DSR.front(recs, s["objectives"])
            report = DSR.apply_trial(sess, name, front[0]) if front else {"ok": False, "error": "empty front"}
            if not rep.check(label, "SS2 the first front trial committed", report.get("committed"),
                             trial=front[0] if front else None, error=report.get("error"),
                             edit_equals_scored=report.get("edit_equals_scored")):
                continue
            pid = report["patch"]
            rec, _ = REC.read(ws, ws.patch(pid))
            reasons = REC.heretic_reasons(ws, rec, pid, model=dump["model"]) if rec else ["no record"]
            status = {"runnable": not reasons, "reasons": reasons}
            if rep.check(label, "SS3 the record is Heretic-runnable", status["runnable"], reasons=status["reasons"]):
                commit = dump.get("model_commit") or (Path(snap).name if Path(snap).parent.name == "snapshots" else None)
                data = REC.to_heretic(ws, rec, pid, model=dump["model"], model_commit=commit)
                here = out / label
                here.mkdir(parents=True, exist_ok=True)
                (here / "ide_reproduce.json").write_text(json.dumps(data, indent=4))
                (here / "ide.json").write_text(json.dumps({"label": label, "patch": pid, "trial": front[0],
                                                           "digest": rec["pieces"]["digest"], "record": rec}, indent=1))
            ws.set_patch(pid, enabled=False)
        finally:
            sess.unload()
    return rep


# ------------------------------------------------------------------ stage rerun (loads the model)
def stage_rerun(snap, root, adapter_out, labels):
    import torch
    from wsp import directions as DR
    from wsp import directions_record as REC
    from wsp.session import Session
    rep = Report()

    def same_pieces(a, b):
        if sorted(a) != sorted(b):
            return False
        for name in a:
            if len(a[name]) != len(b[name]):
                return False
            for p, q in zip(a[name], b[name]):
                if p.kind != q.kind or p.region != q.region:
                    return False
                for t in ("values", "A", "B", "coords"):
                    x, y = getattr(p, t), getattr(q, t)
                    if (x is None) != (y is None) or (x is not None and not torch.equal(x, y)):
                        return False
        return True

    for label in labels:
        ide = json.loads((Path(adapter_out) / label / "ide.json").read_text())
        pid = ide["patch"]
        sess = Session(str(root / label))
        ws = sess.open(str(snap), log=lambda *a, **k: None)
        try:
            for p in list(ws.patches):                    # the weights it was made on: the base (stage ide made it so)
                if p.enabled:
                    ws.set_patch(p.id, enabled=False)
            n = len(ws.patches)
            r = REC.reproduce(sess, pid, dry_run=True)
            rep.check(label, "S-E1 dry run identical, nothing committed", r.get("verdict") == "identical"
                      and r.get("digest") == r.get("recorded_digest") and len(ws.patches) == n and not r.get("committed"),
                      verdict=r.get("verdict"), digest=r.get("digest"), recorded=r.get("recorded_digest"),
                      error=r.get("error"), environment=[x for x in r.get("environment") or [] if x["same"] is False])
            body = (REC.answer(ws, pid, remeasure_too=True)["remeasure"] or [{}])[0].get("body")
            if body is None:
                rep.check(label, "S-E2 measured again from the record", False, error="no remeasure body")
            else:
                DR.measure(ws, **{**DR.measure_body(body), "replace": True})
                r2 = REC.reproduce(sess, pid, measurement=body["name"], dry_run=True)
                means = r2.get("means") or {}
                rep.check(label, "S-E2 measured again from the record", means.get("equal") is True
                          and r2.get("verdict") == "identical", measurement=body["name"], means=means,
                          verdict=r2.get("verdict"),
                          differences=(r2.get("differences") or [])[:5])
            r3 = REC.reproduce(sess, pid, dry_run=False)
            new = ws.patch(r3["patch"]) if r3.get("committed") else None
            rep.check(label, "S-E3 committed re-run equals the original", new is not None
                      and new.source.get("origin", {}).get("kind") == "reproduction"
                      and same_pieces(new.deltas, ws.patch(pid).deltas),
                      patch=r3.get("patch"), verdict=r3.get("verdict"), error=r3.get("error"))
            if new is not None:
                ws.set_patch(new.id, enabled=False)
        finally:
            sess.unload()
    return rep


# ------------------------------------------------------------------ main
def main(argv):
    if len(argv) < 2 or argv[0] != "--stage" or argv[1] not in ("search", "map", "rerun"):
        raise SystemExit(__doc__)
    stage = argv[1]
    if stage == "map":
        if len(argv) < 4:
            raise SystemExit(__doc__)
        out, pairs = Path(argv[2]), {}
        for a in argv[3:]:
            label, _, files = a.partition("=")
            ide, _, her = files.partition(",")
            if not (label and ide and her):
                raise SystemExit(f"{a}: a label is <label>=<IDE reproduce.json>,<Heretic reproduce.json>")
            pairs[label] = (ide, her)
        return finish(stage_map(out, pairs), out, stage)
    if stage == "search":
        if len(argv) < 6:
            raise SystemExit(__doc__)
        snap, root, out = Path(argv[2]), Path(argv[3]), Path(argv[4])
        runs = dict(a.split("=", 1) for a in argv[5:])
        return finish(stage_search(snap, root, out, runs), out / "search.json", stage)
    if len(argv) < 7:
        raise SystemExit(__doc__)
    snap, root, adapter_out, out = Path(argv[2]), Path(argv[3]), Path(argv[4]), Path(argv[5])
    return finish(stage_rerun(snap, root, adapter_out, argv[6:]), out, stage)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
