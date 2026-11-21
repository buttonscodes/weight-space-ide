"""Compare the IDE's automatic search (wsp.directions_search) with finished Heretic searches.

Run with the IDE's interpreter (Optuna 5 installed: requirements-search.txt), with the IDE's `weight-space-ide`
directory as the current directory:

    python scripts/heretic_compare_search.py --stage Z <out.json> <heretic run dir> ...
    python -m wsp.memory_guard ... -- python scripts/heretic_compare_search.py --stage R <model snapshot dir> <isolated IDE root> <out.json> <heretic run dir> ...

A Heretic run directory holds its config.toml, its Optuna journal (checkpoints/*.jsonl) and the dump written by
heretic_dump.py (dump/dump.json; stage R also reads dump/dump.safetensors). The journal is copied next to the
output and read from the copy with Optuna's JournalStorage; the original is never opened. The seed, n_trials,
n_startup_trials, the scorers, the modifier and its settings come from config.toml (Heretic's defaults where it is
silent); the layers, components and prompts from dump.json.

Stage Z needs no model. It is the model-free proof that the search's space, sampler, parameter transformation and
Pareto front are Heretic's, on Heretic's own records. Stage R loads the model (one run at a time; give it the runs of
one model) and runs the IDE's own search on it.

Acceptance (decided before any run; a stage outside its bound is a finding to investigate, not a threshold to move):

    Z1  space() from the dump's layers, components and prompt counts, against the
        distributions of every journal trial                                         equal, every run
    Z2  the IDE's Sampler, told each journal trial's outcome in turn, against the
        journal's draws                                                              equal bit for bit, every trial
    Z3  heretic_parameters(to_parameters(...)) of each trial's draws against
        dump.json's parameters                                                       equal, every trial
    Z4  front() of the journal's trials against Optuna's best_trials of the journal  the same set; the order of
                                                                                     Heretic's sort
    R0  eps: max |IDE - Heretic| of the base first-token logits of the KL prompts    reported (feeds R2)
    R1  the IDE's search with the run's seed, n_trials and n_startup_trials: the
        startup trials' draws against the journal's                                  identical (required)
    R2  each trial's scores. Ablation: for every trial whose draws are identical,
        Refusals equal and KL within 1e-6 + max(1e-3 KL, 3 eps sqrt(2 KL)) of
        Heretic's                                                                    required; ARA: reported (the fit
                                                                                     is reproducible only under the
                                                                                     same torch version)
    R3  the TPE trials' draws. Ablation: identical whenever every earlier trial
        passed R2 (the draws depend on the ranking only)                             required; otherwise, and for
                                                                                     ARA, the first trial where they
                                                                                     part and both rankings reported
    R4  the sampler isolated: the same search driven with Heretic's recorded values
        told instead of the IDE's (Sampler, DS.candidate and DS.evaluate called
        directly)                                                                    every trial's draws identical
                                                                                     (required); the IDE's scores
                                                                                     against Heretic's (ablation: as
                                                                                     R2, required; ARA: reported)
    R5  the front reached: the IDE's (R1) against Heretic's best_trials              ablation: the same trial indices
                                                                                     when R3 passed (required); ARA:
                                                                                     reported
    R6  stop and resume (ablation runs with row_normalization full): a second
        search stopped inside trial 2 at its 5th response, then run again           the final records equal R1's:
                                                                                     draws, values, digests (required)
    R7  commit: the first front trial through apply_trial, then a score of the
        current weights with the same scorers                                        digest equal to the scored one;
                                                                                     scores equal to the trial's
                                                                                     recorded scores exactly (required)

The output JSON holds every stage per run with its verdict ("ok") and the overall verdict; a one-line summary per
run is printed.
"""
import json
import math
import os
import shutil
import struct
import sys
import time
import tomllib
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, os.getcwd())

from wsp import directions_sampler as S
from wsp import directions_search as DSR

SCORERS = [{"plugin": "heretic.scorers.keyword_rate.KeywordRate", "optimization": "minimize"},
           {"plugin": "heretic.scorers.kl_divergence.KLDivergence", "optimization": "minimize"}]   # Heretic's default
PROMPTS_OF = {"KeywordRate": "refusal", "KLDivergence": "kl", "ResponseLength": "kl"}
MEASURED = ("good_prompts", "bad_prompts", "winsorization_quantile")


# ------------------------------------------------------------------ reading a Heretic run
def label_of(run: Path) -> str:
    return f"{run.parent.name}/{run.name}"


def read_run(run: Path, out_dir: Path) -> dict:
    """config.toml, dump.json and the journal's trials (read from a copy) of one Heretic run."""
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config = tomllib.loads((run / "config.toml").read_text())
    dump = json.loads((run / "dump" / "dump.json").read_text())
    journals = sorted((run / "checkpoints").glob("*.jsonl"))
    if len(journals) != 1:
        raise SystemExit(f"{run}: expected one journal in checkpoints/, found {len(journals)}")
    copy = out_dir / "journals" / (label_of(run).replace("/", "_") + ".jsonl")
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(journals[0], copy)
    study = optuna.load_study(study_name=S.SAMPLER["study_name"], storage=JournalStorage(JournalFileBackend(str(copy))))
    modifiers = config.get("modifiers") or [{"plugin": "heretic.modifiers.ara.ARA"}]
    plugin = modifiers[0]["plugin"].split(".")[-1]
    if plugin not in DSR.MODIFIERS or len(modifiers) != 1:
        raise SystemExit(f"{run}: one modifier, Abliteration or ARA, is compared; this run has {modifiers}")
    row = DSR.MODIFIERS[plugin]
    table = (config.get("modifier") or {}).get(plugin) or {}
    settings = {**row["settings"], **{k: v for k, v in table.items() if k in row["settings"]}}
    scorers = config.get("scorers") or SCORERS
    texts = {k: [p["user"] for p in v] for k, v in dump["prompts"].items()}
    return {"label": label_of(run), "config": config, "dump": dump, "study": study, "plugin": plugin,
            "method": row["method"], "settings": settings, "measured": {k: table[k] for k in MEASURED if k in table
                                                                        and not isinstance(table[k], dict)},
            "scorers": scorers, "texts": texts, "seed": config.get("seed"), "n_trials": config.get("n_trials", 100),
            "n_startup_trials": config.get("n_startup_trials", 30),
            "directions": [s["optimization"] for s in scorers if s["optimization"] in ("minimize", "maximize")],
            "components": list(dump["components"]), "layers": int(dump["layers"])}


def same_bits(a: dict, b: dict) -> bool:
    """Equal keys in equal order and every value equal bit for bit (floats compared as their IEEE bytes)."""
    if list(a) != list(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if type(x) is not type(y):
            return False
        if isinstance(x, float) and struct.pack("<d", x) != struct.pack("<d", y):
            return False
        if not isinstance(x, float) and x != y:
            return False
    return True


def journal_space(trial) -> list[dict]:
    from optuna.distributions import distribution_to_json
    return [{"name": n, "distribution": json.loads(distribution_to_json(d))} for n, d in trial.distributions.items()]


def heretic_front(study, objective_names) -> list[int]:
    """Heretic's front: study.best_trials sorted by their scores in objective order (main.py), as 1-based indices."""
    def key(t):
        scores = t.user_attrs.get("scores")
        if scores:
            return tuple(next((s["score"]["value"] for s in scores if s["name"] == n), None) for n in objective_names)
        return tuple(t.values)
    return [t.number + 1 for t in sorted(study.best_trials, key=key)]


# ------------------------------------------------------------------ stage Z (no model)
def stage_z(r: dict) -> dict:
    from optuna.trial import TrialState
    study, dump = r["study"], r["dump"]
    n_good, n_bad = len(r["texts"].get("good", [])), len(r["texts"].get("bad", []))
    space = DSR.space(r["method"], n_layers=r["layers"], components=r["components"], n_good=n_good, n_bad=n_bad)
    trials = study.trials
    z1 = [k + 1 for k, t in enumerate(trials) if journal_space(t) != space]
    directions = [d.name.lower() for d in study.directions]
    smp = S.Sampler(seed=r["seed"], n_startup_trials=r["n_startup_trials"], directions=r["directions"])
    rows, complete = [], 0
    for k, t in enumerate(trials):
        kind = "startup" if complete < r["n_startup_trials"] else "tpe"
        handle, draws = smp.ask(space)
        row = {"index": k + 1, "kind": kind, "state": t.state.name, "draws_equal": same_bits(draws, t.params)}
        if not row["draws_equal"]:
            row["ide"], row["heretic"] = draws, t.params
        params = DSR.to_parameters(r["method"], t.params, r["settings"], direction="m", components=r["components"],
                                   seed=r["seed"])
        hp = DSR.heretic_parameters(r["method"], params)
        want = dump["trials"][k]["parameters"] if k < len(dump["trials"]) else None
        row["parameters_equal"] = hp == want
        attrs = t.user_attrs.get("parameters")
        row["journal_parameters_equal"] = None if attrs is None else hp == attrs
        if not row["parameters_equal"]:
            row["ide_parameters"], row["heretic_parameters"] = hp, want
        rows.append(row)
        if t.state == TrialState.COMPLETE:
            smp.tell(handle, values=t.values)
            complete += 1
        elif t.state == TrialState.FAIL:
            smp.tell(handle, failed=True)
        else:
            row["unsupported_state"] = True
            break
    objectives = [{"index": i, "name": n, "direction": d}
                  for i, (n, d) in enumerate(zip(dump.get("objective_names") or ["a", "b"], r["directions"]))]
    records = [{"index": t.number + 1, "state": "complete" if t.state == TrialState.COMPLETE else "failed",
                "values": t.values} for t in trials]
    ide_front = DSR.front(records, objectives)
    her_front = heretic_front(study, [o["name"] for o in objectives])
    z = {"Z1": {"space": space == journal_space(trials[0]) if trials else None, "trials_differing": z1,
                "ok": bool(trials) and not z1},
         "Z2": {"trials": len(rows), "startup": sum(x["kind"] == "startup" for x in rows),
                "tpe": sum(x["kind"] == "tpe" for x in rows), "equal": sum(x["draws_equal"] for x in rows),
                "startup_equal": sum(x["draws_equal"] for x in rows if x["kind"] == "startup"),
                "tpe_equal": sum(x["draws_equal"] for x in rows if x["kind"] == "tpe"),
                "directions": {"config": r["directions"], "journal": directions},
                "ok": len(rows) == len(trials) and all(x["draws_equal"] for x in rows) and r["directions"] == directions},
         "Z3": {"equal": sum(x["parameters_equal"] for x in rows), "trials": len(rows),
                "ok": all(x["parameters_equal"] for x in rows) and len(dump["trials"]) == len(trials)},
         "Z4": {"ide": ide_front, "heretic": her_front, "same_set": set(ide_front) == set(her_front),
                "ok": ide_front == her_front},
         "rows": rows}
    z["ok"] = all(z[k]["ok"] for k in ("Z1", "Z2", "Z3", "Z4"))
    return z


# ------------------------------------------------------------------ stage R (loads the model)
def kl_bound(kl, eps):
    return 1e-6 + max(1e-3 * kl, 3 * eps * math.sqrt(2 * max(kl, 0.0)))


def scorer_entries(r) -> list[dict]:
    """The run's scorers as the IDE's entries: the dump's prompts of each scorer, its config table's settings."""
    out = []
    tables = r["config"].get("scorer") or {}
    for s in r["scorers"]:
        cls = s["plugin"].split(".")[-1]
        name = cls + (f"_{s['instance_name']}" if s.get("instance_name") else "")
        settings = {k: v for k, v in (tables.get(name) or {}).items() if k != "prompts" and not isinstance(v, dict)}
        out.append({"plugin": cls, "optimization": s["optimization"], "prompts": r["texts"][PROMPTS_OF[cls]],
                    **({"instance_name": s["instance_name"]} if s.get("instance_name") else {}), **settings})
    return out


def heretic_scores(r, k) -> list[float]:
    by_name = {x["name"]: x["score"]["value"] for x in r["dump"]["trials"][k]["scores"]}
    return [by_name.get(n) for n in r["dump"]["objective_names"]]


def ranking(values) -> list[int]:
    return [i + 1 for i, _ in sorted(enumerate(values), key=lambda p: tuple(p[1]))]


def stage_r(r: dict, snap: Path, root: Path) -> dict:
    import torch
    from safetensors.torch import load_file
    from wsp import directions as DR
    from wsp import directions_scores as DS
    from wsp.session import Session
    out = {"ide_versions": {p: version(p) for p in ("torch", "transformers", "optuna", "numpy")},
           "heretic_versions": r["dump"].get("versions")}
    session = Session(str(root / r["label"].replace("/", "_")))
    os.makedirs(session.root, exist_ok=True)
    ws = session.open(str(snap), log=lambda *a, **k: None)
    try:
        system, prefix = r["dump"]["system_prompt"], r["dump"]["response_prefix"] or None
        texts, ablation = r["texts"], r["method"] == "ablation"
        DR.measure(ws, "oracle", texts["good"], texts["bad"], chat=True, system=system, prefix=prefix, variant="base",
                   replace=True, module_io=not ablation, **({"winsorization_quantile": r["measured"]["winsorization_quantile"]}
                                                          if "winsorization_quantile" in r["measured"] else {}))
        settings = {k: v for k, v in r["settings"].items()}
        body = {"modifiers": [{"plugin": r["plugin"], "direction": "oracle", **settings}], "scorers": scorer_entries(r),
                "n_trials": r["n_trials"], "n_startup_trials": r["n_startup_trials"], "seed": r["seed"],
                "max_response_length": r["config"].get("max_response_length", 100), "chat": True, "system": system,
                "prefix": prefix}
        # ---- R0: eps from the base first-token logits of the KL prompts
        tens = load_file(str(Path(r["run"]) / "dump" / "dump.safetensors"))
        kl_ids = DR.render_ids(ws, texts["kl"], "prompts", chat=True, system=system, prefix=prefix, chat_options={})
        lb = torch.stack([ws.runner.first_token_logits(x, "base") for x in kl_ids])
        eps = float((lb - tens["kl_logits"].float()).abs().max())
        out["R0"] = {"eps": eps}
        # ---- R1: the IDE's own search
        t0 = time.monotonic()
        DSR.create(ws, DSR.check_search(ws, {"name": "r1", **body}))
        res = DSR.run(ws, "r1")
        recs = DSR.trial_records(ws, "r1")
        jt = r["study"].trials
        rows = []
        for k, rec in enumerate(recs):
            her = heretic_scores(r, k) if k < len(r["dump"]["trials"]) else None
            same = k < len(jt) and same_bits(rec["draws"], jt[k].params)
            row = {"index": k + 1, "kind": rec["kind"], "state": rec["state"], "draws_equal": same,
                   "ide": rec["values"], "heretic": her}
            if same and rec["values"] is not None and her is not None:
                row["refusals_equal"] = rec["values"][0] == her[0]
                row["kl_difference"] = abs(rec["values"][1] - her[1])
                row["kl_bound"] = kl_bound(her[1], eps)
                row["scores_ok"] = row["refusals_equal"] and row["kl_difference"] <= row["kl_bound"]
            rows.append(row)
        startup = [x for x in rows if x["kind"] == "startup"]
        out["R1"] = {"run": res, "seconds": time.monotonic() - t0, "startup": len(startup),
                     "startup_equal": sum(x["draws_equal"] for x in startup),
                     "ok": len(recs) == len(jt) and all(x["draws_equal"] for x in startup)}
        compared = [x for x in rows if "scores_ok" in x]
        out["R2"] = {"rows": rows, "compared": len(compared), "passed": sum(x["scores_ok"] for x in compared),
                     "ok": (all(x["scores_ok"] for x in compared) and len(compared) == sum(x["draws_equal"] for x in rows))
                     if ablation else None}
        # ---- R3: the TPE trials' draws
        part = next((x["index"] for x in rows if not x["draws_equal"]), None)
        clean = all(x.get("scores_ok") for x in rows[:(part - 1) if part else len(rows)])
        out["R3"] = {"tpe": sum(x["kind"] == "tpe" for x in rows), "tpe_equal": sum(x["draws_equal"] for x in rows
                                                                                  if x["kind"] == "tpe"),
                     "first_parting": part, "earlier_all_passed_R2": clean,
                     "ide_ranking": ranking([x["ide"] for x in rows if x["ide"] is not None]),
                     "heretic_ranking": ranking([x["heretic"] for x in rows if x["heretic"] is not None]),
                     "ok": (part is None or not clean) if ablation else None}
        # ---- R4: the sampler isolated (Heretic's values told)
        s = DSR.load(ws, "r1")
        smp = S.Sampler(seed=s["seed"], n_startup_trials=s["n_startup_trials"],
                        directions=[o["direction"] for o in s["objectives"]])
        prepared = DSR._prepare(ws, s, None, None)
        r4 = []
        for k, t in enumerate(jt):
            handle, draws = smp.ask(s["space"])
            params = DSR.to_parameters(s["method"], draws, s["modifier"]["settings"], direction="oracle",
                                       components=s["model"]["components"], seed=s["seed"])
            with ws.lock:
                deltas, _ = DS.candidate(ws, DR.script_for(params))
                try:
                    ev = DS.evaluate(ws, prepared, "candidate", deltas)
                finally:
                    DS._release(ws, deltas)
            ide = [ev["scores"][o["index"]]["value"] for o in s["objectives"]]
            her = heretic_scores(r, k)
            row = {"index": k + 1, "draws_equal": same_bits(draws, t.params), "ide": ide, "heretic": her}
            row["refusals_equal"] = ide[0] == her[0]
            row["kl_difference"], row["kl_bound"] = abs(ide[1] - her[1]), kl_bound(her[1], eps)
            row["scores_ok"] = row["refusals_equal"] and row["kl_difference"] <= row["kl_bound"]
            r4.append(row)
            smp.tell(handle, values=list(t.values))
        out["R4"] = {"rows": r4, "draws_equal": sum(x["draws_equal"] for x in r4), "trials": len(r4),
                     "ok": all(x["draws_equal"] for x in r4) and (all(x["scores_ok"] for x in r4) if ablation else True)}
        # ---- R5: the fronts
        ide_front = DSR.front(recs, s["objectives"])
        her_front = heretic_front(r["study"], r["dump"]["objective_names"])
        out["R5"] = {"ide": ide_front, "heretic": her_front,
                     "ide_scores": {i: recs[i - 1]["values"] for i in ide_front},
                     "heretic_scores": {i: heretic_scores(r, i - 1) for i in her_front},
                     "ok": (ide_front == her_front) if ablation and out["R3"]["ok"] and part is None else None}
        # ---- R6: stop inside trial 2 at its 5th response, then resume
        if ablation and r["settings"].get("row_normalization") == "full":
            DSR.create(ws, DSR.check_search(ws, {"name": "r6", **body}))
            flag = [False]

            def progress(index, phase, done, total):
                if index == 2 and phase == "responses" and done >= 5:
                    flag[0] = True
            first = DSR.run(ws, "r6", progress=progress, stopped=lambda: flag[0])
            second = DSR.run(ws, "r6")
            again = DSR.trial_records(ws, "r6")
            keys = ("draws", "values")
            out["R6"] = {"first": first, "second": second,
                         "trial_2_attempts": again[1]["attempts"] if len(again) > 1 else None,
                         "ok": first["state"] == "stopped" and len(again) == len(recs)
                         and all(a[k] == b[k] for a, b in zip(again, recs) for k in keys)
                         and all(a["candidate"]["digest"] == b["candidate"]["digest"] for a, b in zip(again, recs))}
        # ---- R7: commit the first front trial, then score the current weights
        if ide_front:
            index = ide_front[0]
            rec = recs[index - 1]
            report = DSR.apply_trial(session, "r1", index)
            r7 = {"trial": index, "committed": report.get("committed"), "error": report.get("error"),
                  "digest": report.get("digest"), "scored_digest": report.get("scored_digest"),
                  "edit_equals_scored": report.get("edit_equals_scored")}
            if report.get("committed"):
                now = DS.score(ws, {"state": "current", **s["scoring"]["request"]})
                r7["scores_now"] = [x["value"] for x in now["scores"]]
                r7["scores_then"] = [x["value"] for x in rec["scores"]]
                r7["ok"] = bool(report["edit_equals_scored"]) and r7["scores_now"] == r7["scores_then"]
                ws.set_patch(report["patch"], enabled=False)
            else:
                r7["ok"] = False
            out["R7"] = r7
        required = [out[k]["ok"] for k in ("R1", "R2", "R3", "R4", "R5", "R6", "R7") if k in out]
        out["ok"] = all(v for v in required if v is not None)
        return out
    finally:
        session.unload()


# ------------------------------------------------------------------ main
def main(argv):
    if len(argv) < 2 or argv[0] != "--stage" or argv[1] not in ("Z", "R"):
        raise SystemExit(__doc__)
    S.require()
    stage = argv[1]
    if stage == "Z":
        out, runs = Path(argv[2]), [Path(a) for a in argv[3:]]
        snap = root = None
    else:
        snap, root, out, runs = Path(argv[2]), Path(argv[3]), Path(argv[4]), [Path(a) for a in argv[5:]]
    if not runs:
        raise SystemExit(__doc__)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"stage": stage, "heretic": DSR.HERETIC, "versions": S.versions(), "runs": {}}
    for run in runs:
        r = read_run(run, out.parent)
        r["run"] = str(run)
        entry = stage_z(r) if stage == "Z" else stage_r(r, snap, root)
        result["runs"][r["label"]] = entry
        if stage == "Z":
            z2 = entry["Z2"]
            print(f"{r['label']}: {r['method']}, {z2['trials']} trials ({z2['startup']} startup, {z2['tpe']} TPE); "
                  f"draws equal {z2['startup_equal']}/{z2['startup']} startup, {z2['tpe_equal']}/{z2['tpe']} TPE; "
                  f"space {'ok' if entry['Z1']['ok'] else 'DIFFERS'}; parameters {entry['Z3']['equal']}/"
                  f"{entry['Z3']['trials']}; front {entry['Z4']['ide']} vs {entry['Z4']['heretic']}; "
                  f"{'ok' if entry['ok'] else 'NOT OK'}", flush=True)
        else:
            print(f"{r['label']}: " + ", ".join(f"{k} {entry[k].get('ok')}" for k in ("R1", "R2", "R3", "R4", "R5", "R6",
                                                                                     "R7") if k in entry)
                  + f"; {'ok' if entry['ok'] else 'NOT OK'}", flush=True)
        out.write_text(json.dumps(result, indent=1, default=str))
    result["ok"] = all(e["ok"] for e in result["runs"].values())
    if stage == "Z":
        result["totals"] = {k: sum(e["Z2"][k] for e in result["runs"].values())
                            for k in ("trials", "startup", "tpe", "equal", "startup_equal", "tpe_equal")}
        print(f"total: {result['totals']}", flush=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    print(f"{'ok' if result['ok'] else 'NOT OK'}: {out}", flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
