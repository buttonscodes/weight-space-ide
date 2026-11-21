"""Run Heretic's own reproduce.json files in the IDE (wsp.directions_heretic) and compare the patches (plan item H3b).

Run with the IDE's interpreter plus pyarrow (the datasets are parquet), with the IDE's `weight-space-ide` directory as
the current directory, offline (HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1, HF_HOME the cache that holds the model and
the datasets), under a memory cap:

    python scripts/heretic_run_reproduce.py <model snapshot dir> <isolated IDE root> <out.json> \\
        <label>=<new run dir>,<original run dir>,<reproduce.json> ...

A label names one Heretic configuration three times: the new run (export_strategy "adapter", save_directory
"adapter"; its adapter/ and its journal are read), the original run of the same config (its dump/, written by
scripts/heretic_dump.py, is what the hand-pasted comparison reads) and the reproduce.json Heretic's own code wrote
for the new run (scripts/heretic_reproduce_json.py: heretic.utils.generate_reproduce_json from the run's journal).
The IDE's workspaces are made in <root>/<label>/ with Workspace.create_hf over the snapshot, with the repository the
file names and the commit the snapshot directory is named by (the Hugging Face cache names a snapshot by its
commit), which is what opening the model by its Hub id records; so the model check compares both.

Per label, in order (a failed J or R1 stops the label):

    J   the new run's journal holds the same draws, values, parameters and scores as the original run's
        (heretic_compare_adapter.journals_equal): the original run's dump describes the new run    exact
    F   the file is the new run's: its parameters equal those of a trial in the new run's journal, and
        hashes["adapter_model.safetensors"] is the SHA-256 of the new run's adapter file             exact
    R0  check_heretic_reproduce of the file in a fresh workspace: no refusal; the model's repository and
        commit "same"; the environment's differences and the notes are reported (no bound)           exact
    R1  heretic_reproduce(dry_run=False) of the file: committed through Session.apply, with a record   exact
    R2  that patch against the adapter Heretic's own run exported (formats_shared.adapter_changes: per
        tensor ||delta_IDE - B A||_F / ||B A||_F), each tool from its own measurement. Directional
        ablation (none, full): every tensor at most the bound the hand-pasted comparison uses for the
        same model (heretic_compare_adapter.py P6: P6_BOUNDS by the original run's model directory,
        1e-3 for qwen05b, else P6_BOUND, 5e-5, the tiny fixture's), and no tensor changed by one tool
        only (a module Heretic stores with a zero product is not a change); heretic_compare.py and
        heretic_compare_ara.py assert no weight bound in float32 (their float32 rows report only).
        ARA: reported only, as heretic_compare_ara.py's float32 rows and P6c report it (a fit moves
        with the torch version: Heretic ran torch 2.14.1, the IDE 2.15.0)
    R3  the hand-pasted path in a second fresh workspace: heretic_compare_adapter.measure_as_the_run and
        params_of from the original run's dump (Heretic's parameters taken by hand), DR.script_for and
        Session.apply: its piece digest equals R1's (nothing pasted by hand changes nothing)          identical
    R4  round trip: R3's patch (an edit made in the IDE) written as a Heretic reproduce.json by the
        existing export (directions_record.to_heretic) and read back by heretic_reproduce in a third
        fresh workspace: the same piece digest as R3's patch, every piece torch.equal, the answer's
        same_as_file true                                                                           bit for bit

Every assertion prints PASS or FAIL with its numbers; <out.json> holds every number and a pass flag per assertion and
no path; the exit status is 1 when one fails. Heretic's own check of the file it wrote is not run again
(scripts/heretic_check_reproduce.py does that).
"""
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))



def journal_parameters(run):
    """Every trial's Heretic parameters in the new run's journal (read as JSON lines, never opened with Optuna)."""
    (journal,) = sorted((Path(run) / "checkpoints").glob("*.jsonl"))
    out = []
    for line in journal.read_text().splitlines():
        r = json.loads(line) if line.strip() else {}
        if isinstance(r.get("user_attr"), dict) and "parameters" in r["user_attr"]:
            out.append(r["user_attr"]["parameters"])
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv):
    if len(argv) < 4:
        raise SystemExit(__doc__)
    snap, root, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    labels = {}
    for a in argv[3:]:
        label, _, dirs = a.partition("=")
        new, orig, file = dirs.split(",")
        labels[label] = (Path(new), Path(orig), Path(file))

    import torch
    import formats_shared as FS
    import heretic_compare_adapter as HCA
    from wsp import directions as DR
    from wsp import directions_heretic as DH
    from wsp import directions_record as REC
    from wsp import directions_search as DSR
    from wsp.session import Session
    from wsp.workspace import Workspace

    rep = HCA.Report()
    r2_bound = {label: HCA.P6_BOUNDS.get(orig.parent.name, HCA.P6_BOUND) for label, (_, orig, _) in labels.items()}
    result = {"bounds": {"R2 ablation per tensor": r2_bound}, "torch": str(torch.__version__), "labels": {}}

    def fresh(label, n, data):
        d = root / label
        Workspace.create_hf(str(d / n), n, str(snap), repo=data["settings"]["model"], sha=snap.name)
        sess = Session(str(d))
        sess.open(n)
        return sess

    def pieces_equal(a, b):
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
                    if not ((x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))):
                        return False
        return True

    for label, (new, orig, file) in labels.items():
        info = {}
        result["labels"][label] = info
        if not HCA.journals_equal(rep, label, new, orig):
            continue
        data = json.loads(file.read_text())
        adapter_file = new / "adapter" / "adapter_model.safetensors"
        rep.check(label, "F the file is the new run's",
                  data["parameters"] in journal_parameters(new)
                  and data["hashes"].get("adapter_model.safetensors") == sha256(adapter_file),
                  modifier=data["settings"]["modifiers"][0]["plugin"])
        ara = data["settings"]["modifiers"][0]["plugin"].endswith(".ARA")

        # ---- R0, R1: the file in a fresh workspace
        s1 = fresh(label, "file", data)
        try:
            try:
                plan = DH.check_heretic_reproduce(s1.ws, data)
                refusal = None
            except REC.RecordError as e:
                plan, refusal = None, str(e)
            rep.check(label, "R0 checked without refusal",
                      plan is not None and plan["model"]["repository"] == plan["model"]["commit"] == "same",
                      refusal=refusal, model=None if plan is None else {k: plan["model"][k] for k in ("repository", "commit")})
            if plan is not None:
                info["R0"] = {"environment": {"severity": plan["environment"]["severity"],
                                              "same": plan["environment"]["same"],
                                              "differences": [{k: r[k] for k in ("name", "file", "ide", "severity")}
                                                              for r in plan["environment"]["differences"]
                                                              if r["severity"] in ("critical", "high", "medium")],
                                              "low_differences": sum(1 for r in plan["environment"]["differences"]
                                                                     if r["severity"] == "low")},
                              "notes": plan["notes"], "script": plan["script"], "settings": plan["settings"]}
            ans = DH.heretic_reproduce(s1, data, dry_run=False)
            if not rep.check(label, "R1 committed through Session.apply", ans.get("ok") and ans.get("committed")
                             and "record" in ans, error=ans.get("error"), record_error=ans.get("record_error")):
                continue
            p1 = s1.ws.patch(ans["patch"])
            d1 = DSR.piece_digest(p1.deltas)
            info["R1"] = {"digest": d1, "measured": ans["ran"]["measured"], "settings_reported": ans["heretic"]["settings"]["reported"]}

            # ---- R2: against Heretic's own adapter
            rows, unchanged, missing = FS.adapter_changes(s1.ws, FS.Checkpoint(snap), FS.Adapter(new / "adapter"), p1)
            both = [r for r in rows if "rel_diff" in r]
            worst = max((r["rel_diff"] for r in both), default=None)
            info["R2"] = {"tensors": len(rows), "one_sided": len(rows) - len(both), "worst_relative": worst,
                          "per_tensor": [{k: r.get(k) for k in ("tensor", "rel_diff", "max_abs_diff",
                                                                "heretic_changed", "ide_changed")} for r in rows],
                          "adapter_modules_without_tensor": missing}
            if ara:
                rep.note(label, "R2 against Heretic's adapter (ARA, reported)", {"worst_relative": worst,
                                                                                  "one_sided": len(rows) - len(both)})
            else:
                rep.check(label, "R2 against Heretic's adapter", worst is not None and worst <= r2_bound[label]
                          and len(both) == len(rows) and not missing, worst_relative=worst, bound=r2_bound[label],
                          tensors=len(rows), one_sided=len(rows) - len(both))
        finally:
            s1.unload()
            gc.collect()

        # ---- R3: the hand-pasted path
        s3 = fresh(label, "hand", data)
        try:
            dump = json.loads((orig / "dump" / "dump.json").read_text())
            source = HCA.measure_as_the_run(s3.ws, dump, ara, rep, label)
            params, _ = HCA.params_of(dump, ara)
            r3 = s3.apply(DR.script_for(params), message=f"hand-pasted ({label})", dry_run=False)
            ok3 = rep.check(label, "R3 the hand-pasted patch is the file's", r3.get("committed")
                            and DSR.piece_digest(s3.ws.patch(r3["patch"]).deltas) == d1, measured_from=source,
                            digest=r3.get("committed") and DSR.piece_digest(s3.ws.patch(r3["patch"]).deltas), file_digest=d1)
            if not ok3 and not r3.get("committed"):
                continue
            p3 = s3.ws.patch(r3["patch"])
            d3 = DSR.piece_digest(p3.deltas)
            rec, _ = REC.read(s3.ws, p3)
            written = json.loads(json.dumps(REC.to_heretic(s3.ws, rec, p3.id)))
        finally:
            s3.unload()
            gc.collect()

        # ---- R4: the round trip, in a third workspace
        s4 = fresh(label, "back", data)
        try:
            a4 = DH.heretic_reproduce(s4, written, dry_run=False)
            ok = a4.get("committed")
            p4 = s4.ws.patch(a4["patch"]) if ok else None
            rep.check(label, "R4 round trip bit for bit", ok and DSR.piece_digest(p4.deltas) == d3
                      and pieces_equal(p3.deltas, p4.deltas) and a4["same_as_file"] is True,
                      digest=a4.get("digest"), written_digest=written["wsp"]["digest"], hand_digest=d3,
                      same_as_file=a4.get("same_as_file"))
        finally:
            s4.unload()
            gc.collect()
        info["checks"] = rep.labels.get(label, {})
        result["pass"] = rep.ok()
        FS.write_json(out, result)

    for label in labels:
        result["labels"][label]["checks"] = rep.labels.get(label, {})
    result["pass"] = rep.ok()
    FS.write_json(out, result)
    print(f"{'ok' if result['pass'] else 'NOT OK'}: {out}", flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
