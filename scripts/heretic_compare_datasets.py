"""Compare the IDE's prompt sets from datasets (wsp.directions_datasets) with Heretic's.

Model-free and offline: nothing is downloaded and no model is loaded. Run from the IDE's `weight-space-ide`
directory with the Hugging Face cache that the Heretic runs filled, read only:

    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=<cache home> HF_DATASETS_CACHE=<scratch> \\
        <heretic python> -I -B scripts/heretic_compare_datasets.py --write-oracle <oracle.json> --fixtures <scratch dir>
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=<cache home> \\
        <IDE python> scripts/heretic_compare_datasets.py --oracle <oracle.json> --runs <heretic runs> --out <result.json>

The first command runs in Heretic's environment (``datasets`` 5.0.1 is the oracle) and imports nothing from
``wsp``: it writes the local fixtures and what Heretic's side produces for them (stage 3), the values of
``datasets.ReadInstruction`` (stage 2) and ``load_dataset`` on the cached Hub datasets (stage 4). The second runs the
IDE's resolution under the IDE's own libraries against that file and against the runs' dumps (stage 1). Without
``--oracle`` and ``--write-oracle``, one process does both (``--runs``, ``--fixtures``, ``--out``), for a run under
Heretic's interpreter. Every result records the library versions of both sides.

A run directory holds Heretic's config.toml and the dump written by heretic_dump.py (dump/dump.json, whose
``prompts`` are the {system, user} lists Heretic built). The IDE resolves with an empty scratch state directory,
so a Hub dataset is listed from the cache.

Acceptance (written before any run; a stage outside it is a finding, not a bound to move):

    1  every good, bad, KL and refusal specification of every dumped run, resolved
       with the run's system_prompt (Heretic's default when it has none), as
       {system, user} lists                                                         equal to the dump's, every set;
                                                                                    every record: the pinned commit,
                                                                                    listing "cache", rows_checked true,
                                                                                    each file's sha256 that of the
                                                                                    cached blob
    2  the split grammar: parse_split + absolute against ReadInstruction on 506
       split strings x 8 lengths                                                    the same ranges, or both refuse
    3  local layouts against Heretic's own code path (heretic() below)              the same prompts, or both refuse;
                                                                                    except exactly two, refused by the
                                                                                    IDE with the stated message: a
                                                                                    save_to_disk split "train[:3]+
                                                                                    train[5:6]" (E25), and the txt
                                                                                    directory with empty lines (E41)
    4  14 splits on both cached Hub datasets against load_dataset                   the same rows, or both refuse, or
                                                                                    an empty selection that the IDE
                                                                                    refuses (E40) where datasets
                                                                                    refuses or returns 0 rows

Added after the first run, which found a case the acceptance above did not list: a whole split of 6,265 rows,
which the IDE refuses by its limit of 1,024 prompts per set where load_dataset returns it. The limit is the IDE's
by design, so the refusal alone says nothing about the rows. For such a case stage 4 now resolves the split again
with the limit lifted in this process only, and the rows must equal load_dataset's (``over_limit_rows_equal``); a
refusal by the limit whose rows differ, or cannot be read, stays a difference.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import tomllib
from importlib.metadata import PackageNotFoundError, version

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM = "You are a helpful assistant."                  # Heretic's default system_prompt (config.py)
MARKER = ".heretic-compare-datasets"
LIBRARIES = ("datasets", "pyarrow", "fsspec", "huggingface_hub", "pyyaml")

# ------------------------------------------------------------------ stage 2: the split grammar grid
BOUNDS = ["", "0", "1", "5", "-1", "-5", "99", "100", "101", "1_0", "-0", "_"]
PCTS = ["0%", "1%", "10%", "33%", "50%", "-10%", "99%", "100%", "-100%", "101%"]
SPLITS = ["train", "train[:]", "x.y", "a b", "train[", "train[1]", "train[::2]", "[:5]", "train[ :5]",
          "train[:10%](pct1_dropremainder)", "train[:10%](closest)", "train[:10](closest)", "train[:10%](foo)",
          "train[:10%]+test[:10%](pct1_dropremainder)", "train[:5] + test[-3:]", "train+test", "train+train",
          "train[:10%]+test[:5]", "test[:5]+train[:10%](pct1_dropremainder)", "Train", "train1", "_x[:3]"]
SPLITS += [f"train[{a}:{b}]" for a in BOUNDS + PCTS for b in BOUNDS + PCTS]
LENGTHS = [0, 1, 7, 99, 100, 101, 250, 25058]


def lengths_of(n):
    return {"train": n, "test": max(n // 4, 0), "x.y": n, "Train": n, "train1": n, "_x": n}


# ------------------------------------------------------------------ stage 3: local layouts
ROWS = [f"prompt {i} é中" for i in range(30)]
CASES = [("flat", "train", None), ("flat", "test[:3]", None), ("sharded", "train", None), ("sharded", "test[-2:]", None),
         ("dirs", "train", None), ("dirs", "validation[1:3]", None), ("onejson", "train", None), ("txtdir", "train", None),
         ("cfg", "train", None), ("cfg", "train", "en"), ("cfg", "test", "en"), ("mixed", "train", None),
         ("hidden", "train", None), ("tsv", "train", None), ("keywords", "train", None), ("keywords", "validation", None),
         ("keywords", "test", None), ("blank", "train", None), ("bom", "train", None), ("saved", "train[2:5]", None),
         ("saved", "train[:3]+train[5:6]", None), ("textfile.txt", None, None), ("textfile.txt", "[1:]", None),
         ("textfile.txt", "train[:2]", None), ("sharded", "train+test", None),
         ("overlap", "train", None), ("literal", "train", None)]
DELIBERATE = {("saved", "train[:3]+train[5:6]", None): "E25", ("txtdir", "train", None): "E41"}


def jsonl(rows):
    return "".join(json.dumps({"text": r, "id": i}, ensure_ascii=False) + "\n" for i, r in enumerate(rows))


def write_fixtures(root):
    """The local layouts of the design's experiments E3 and E5 (pyarrow and datasets.save_to_disk: Heretic's side)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset
    if os.path.exists(root):
        if not os.path.isfile(os.path.join(root, MARKER)):
            sys.exit(f"{root} exists and was not written by this script; give an empty or new directory")
        shutil.rmtree(root)
    os.makedirs(root)
    open(os.path.join(root, MARKER), "w").close()

    def w(rel, text, newline=None):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline=newline) as f:
            f.write(text)

    def pqw(rel, rows):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        pq.write_table(pa.table({"text": rows, "n": list(range(len(rows)))}), p)
    w("flat/train.jsonl", jsonl(ROWS[:20]))
    w("flat/test.jsonl", jsonl(ROWS[20:]))
    pqw("sharded/data/train-00001-of-00002.parquet", ROWS[10:20])          # written out of order
    pqw("sharded/data/train-00000-of-00002.parquet", ROWS[:10])
    pqw("sharded/data/test-00000-of-00001.parquet", ROWS[20:])
    csv_text = 'text,id\n"has, comma",1\n"has ""quote""",2\n"multi\nline",3\nplain,4\n'
    w("dirs/train/b.csv", csv_text)
    w("dirs/train/a.csv", "text,id\nfirst,0\n")
    w("dirs/validation/x.csv", csv_text)
    w("onejson/prompts.json", json.dumps([{"text": r} for r in ROWS[:7]]))
    w("txtdir/train.txt", "one\r\n\r\n  two  \r\nthree\n\n", newline="")
    w("cfg/README.md", "---\nconfigs:\n- config_name: en\n  data_files:\n  - split: train\n    path: en/train*.jsonl\n"
                       "  - split: test\n    path: en/test.jsonl\n- config_name: de\n  default: true\n"
                       "  data_files: de/*.jsonl\n---\n# card\n")
    w("cfg/en/train1.jsonl", jsonl(ROWS[:3]))
    w("cfg/en/train2.jsonl", jsonl(ROWS[3:5]))
    w("cfg/en/test.jsonl", jsonl(ROWS[5:6]))
    w("cfg/de/x.jsonl", jsonl(["hallo", "welt"]))
    w("mixed/a.csv", "text\nc1\nc2\n")
    w("mixed/b.csv", "text\nc3\n")
    w("mixed/c.json", json.dumps([{"text": "j"}]))
    w("hidden/train.jsonl", jsonl(["visible"]))
    w("hidden/.h/train.jsonl", jsonl(["hidden"]))
    w("hidden/__pycache__/train.jsonl", jsonl(["special"]))
    w("tsv/train.tsv", "text\tid\na, b\t1\nc\t2\n")
    w("keywords/my_train_data.jsonl", jsonl(["tr"]))
    w("keywords/dev.jsonl", jsonl(["dv"]))
    w("keywords/eval-set.jsonl", jsonl(["ev"]))
    w("blank/train.jsonl", jsonl(["x"]) + "\n" + jsonl(["y"]))
    w("bom/train.jsonl", "﻿" + jsonl(["bom"]))
    w("textfile.txt", "  alpha \n\n beta\r\ngamma\n   \n")
    w("overlap/train-train.jsonl", '{"text": "a"}\n{"text": "b"}\n')         # matched by two default patterns
    w("literal/README.md", "---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n"
                           "    path: [x.jsonl, missing.jsonl]\n---\n")       # a literal path without a file
    w("literal/x.jsonl", '{"text": "x"}\n')
    Dataset.from_dict({"text": ROWS[:12]}).save_to_disk(os.path.join(root, "saved"))


def heretic(spec):
    """Heretic's own loading of one local specification, user texts only: the local branches of
    ``_load_prompts_single`` and ``get_split_slice`` (heretic/src/heretic/utils.py at 64e9991), without prefix,
    suffix and system (stage 1 covers those through the dumps)."""
    from datasets import ReadInstruction, load_dataset, load_from_disk
    from datasets.download.download_manager import DownloadMode
    from datasets.utils.info_utils import VerificationMode

    def get_split_slice(split_str, length):
        split_name = split_str.split("[")[0]
        a = ReadInstruction.from_spec(split_str).to_absolute({split_name: length})[0]
        return a.from_, a.to
    path, split_str = spec["dataset"], spec["split"]
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as file:
            prompts = [line.strip() for line in file if line.strip()]
        if split_str is not None:
            start, end = get_split_slice(f"_{split_str}", len(prompts))
            prompts = prompts[start:end]
        return prompts
    if os.path.exists(os.path.join(path, "state.json")):
        dataset = load_from_disk(path)
        start, end = get_split_slice(split_str, len(dataset))
        return list(dataset[start:end][spec["column"]])
    dataset = load_dataset(path, name=spec["config"], split=split_str, verification_mode=VerificationMode.NO_CHECKS,
                           download_mode=DownloadMode.FORCE_REDOWNLOAD)
    return list(dataset[spec["column"]])


# ------------------------------------------------------------------ stage 4: the cached Hub datasets
REPOS = {"mlabonne/harmless_alpaca": "02c6a92cfcf11bb0c387334f8146d149d65b587f",
         "mlabonne/harmful_behaviors": "01cead01398926d81f7c52bdb790ee8cf77ebba7"}
HUB_SPLITS = ["train[:400]", "test[:100]", "train[200:250]", "train[-7:]", "test[:10%]", "train[10%:12%]", "test",
              "train[:3]+test[:3]", "test[-5:]+train[:2]", "train[:2%](pct1_dropremainder)", "train[5:5]", "train[7:3]",
              "validation[:5]", "train[1000:1100]"]


# ------------------------------------------------------------------ helpers
def versions():
    out = {"python": sys.version.split()[0]}
    for name in LIBRARIES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def error(e):
    return ["error", type(e).__name__, str(e)]


def is_error(x):
    return isinstance(x, list) and len(x) == 3 and x[0] == "error"


# ------------------------------------------------------------------ Heretic's side
def make_oracle(fixtures):
    from datasets import ReadInstruction, load_dataset
    started = time.monotonic()
    stage2 = []
    for s in SPLITS:
        for n in LENGTHS:
            try:
                want = [[i.splitname, i.from_, i.to] for i in ReadInstruction.from_spec(s).to_absolute(lengths_of(n))]
            except Exception as e:
                want = error(e)
            stage2.append({"split": s, "length": n, "want": want})
    write_fixtures(fixtures)
    stage3 = []
    for d, split, config in CASES:
        spec = {"dataset": os.path.join(fixtures, d), "split": split, "column": "text", "config": config}
        try:
            want = heretic(spec)
        except Exception as e:
            want = error(e)
        stage3.append({"case": [d, split, config], "specification": spec, "want": want})
    stage4 = []
    for repo, commit in REPOS.items():
        for s in HUB_SPLITS:
            try:                                # Heretic's Hub branch, with the commit its configs pin
                want = list(load_dataset(repo, name=None, revision=commit, split=s)["text"])
            except Exception as e:
                want = error(e)
            stage4.append({"repository": repo, "commit": commit, "split": s, "want": want})
    return {"versions": versions(), "fixtures": fixtures, "seconds": round(time.monotonic() - started, 3),
            "stage2": stage2, "stage3": stage3, "stage4": stage4}


# ------------------------------------------------------------------ the IDE's side
def specs_of(cfg):
    """The good, bad, KL and refusal specifications of a Heretic config, as the dump names its prompt sets."""
    out = {}
    for _, v in (cfg.get("modifier") or {}).items():
        if "good_prompts" in v:
            out["good"] = v["good_prompts"]
        if "bad_prompts" in v:
            out["bad"] = v["bad_prompts"]
    scorers = cfg.get("scorer") or {}
    out["kl"] = (scorers.get("KLDivergence") or {}).get("prompts")
    out["refusal"] = (scorers.get("KeywordRate") or {}).get("prompts")
    return out


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage1(DD, runs, state):
    cache = DD.hub_cache_dir()
    sets, equal, diffs, record_faults, files_checked, run_names = 0, 0, [], [], 0, []
    for dump_path in sorted(glob.glob(os.path.join(runs, "*", "*", "dump", "dump.json"))):
        run = os.path.dirname(os.path.dirname(dump_path))
        if not os.path.isfile(os.path.join(run, "config.toml")):
            continue
        label = os.path.relpath(run, runs)
        run_names.append(label)
        with open(dump_path, encoding="utf-8") as f:
            dump = json.load(f)
        with open(os.path.join(run, "config.toml"), "rb") as f:
            cfg = tomllib.load(f)
        system = cfg.get("system_prompt", SYSTEM)
        for key, value in specs_of(cfg).items():
            sets += 1
            want = dump["prompts"][key]
            try:
                canon, rec = DD.resolve(DD.items(value, key, key), key, key, system=system, state=state)
                got = [{"system": DD.system_of(p, system), "user": DD.user_of(p)} for p in canon]
            except DD.DatasetError as e:
                diffs.append({"run": label, "set": key, "error": str(e)})
                continue
            if got == want:
                equal += 1
            else:
                first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
                diffs.append({"run": label, "set": key, "got": len(got), "want": len(want), "first_difference": first})
            for part in rec["parts"]:
                given = part["specification"]["commit"]
                if part["kind"] != "hub" or part["commit"] != given or part["listing"] != "cache" \
                        or part["rows_checked"] is not True:
                    record_faults.append({"run": label, "set": key, "kind": part["kind"], "commit": part["commit"],
                                          "pinned": given, "listing": part["listing"],
                                          "rows_checked": part["rows_checked"]})
                snap = os.path.join(cache, "datasets--" + part["repository"].replace("/", "--"), "snapshots",
                                    part["commit"])
                for f in part["files"]:
                    blob = os.path.realpath(os.path.join(snap, f["path"]))
                    files_checked += 1
                    if sha256_of(blob) != f["sha256"] or os.path.getsize(blob) != f["bytes"]:
                        record_faults.append({"run": label, "set": key, "file": f["path"], "sha256": f["sha256"]})
    ok = sets > 0 and equal == sets and not record_faults
    return {"runs": len(run_names), "run_names": run_names, "sets": sets, "equal": equal, "differences": diffs,
            "files_checked": files_checked, "record_faults": record_faults, "ok": ok}


def stage2(DD, oracle):
    agree, diffs = 0, []
    for e in oracle["stage2"]:
        try:
            got = [list(x) for x in DD.absolute(DD.parse_split(e["split"]), lengths_of(e["length"]))]
        except DD.DatasetError as x:
            got = error(x)
        except Exception as x:                          # not a refusal: always a difference
            got = ["crash", type(x).__name__, str(x)]
        if got == e["want"] or (is_error(got) and is_error(e["want"])):
            agree += 1
        else:
            diffs.append({"split": e["split"], "length": e["length"], "ide": got, "datasets": e["want"]})
    return {"cases": len(oracle["stage2"]), "agree": agree, "differences": diffs,
            "ok": agree == len(oracle["stage2"]) == 4048}


def resolve_users(DD, spec, state):
    try:
        canon, _ = DD.resolve(DD.items(spec, "prompts", "prompts"), "prompts", "prompts", system=None, state=state)
        return [DD.user_of(p) for p in canon]
    except DD.DatasetError as e:
        return error(e)


def stage3(DD, oracle, state):
    counts, diffs, rows = {"same": 0, "both_refuse": 0, "deliberate": 0}, [], []
    for e in oracle["stage3"]:
        case, spec, want = tuple(e["case"]), e["specification"], e["want"]
        got = resolve_users(DD, spec, state)
        kind = None
        if case in DELIBERATE:
            if DELIBERATE[case] == "E25":
                message = ("prompts.split: a dataset saved with save_to_disk takes one slice, not parts joined by + "
                           "(Heretic would read only the first part)")
            else:                                       # the first empty row of Heretic's own list
                k = next((i + 1 for i, t in enumerate(want) if t == ""), None) if isinstance(want, list) else None
                message = f"prompts prompt {k} ({spec['dataset']}, split train row {k}) must be text of 1..20000 characters"
            if isinstance(want, list) and is_error(got) and got[2] == message:
                kind = "deliberate"
        elif is_error(want) and is_error(got):
            kind = "both_refuse"
        elif got == want:
            kind = "same"
        if kind is None:
            diffs.append({"case": list(case), "ide": got, "heretic": want})
        else:
            counts[kind] += 1
        rows.append({"case": list(case), "result": kind or "different",
                     "count": len(got) if isinstance(got, list) and not is_error(got) else None})
    ok = not diffs and counts["deliberate"] == len(DELIBERATE)
    return {"cases": len(oracle["stage3"]), **counts, "differences": diffs, "rows": rows, "ok": ok}


def resolve_unlimited(DD, spec, state):
    """``resolve_users`` with the IDE's limit of prompts per set lifted, in this process only."""
    limit = DD.MAX_PROMPTS
    DD.MAX_PROMPTS = 1 << 40
    try:
        return resolve_users(DD, spec, state)
    finally:
        DD.MAX_PROMPTS = limit


def stage4(DD, oracle, state):
    counts, diffs, rows = {"same_rows": 0, "both_refuse": 0, "empty_refused": 0, "over_limit_rows_equal": 0}, [], []
    for e in oracle["stage4"]:
        spec = {"dataset": e["repository"], "commit": e["commit"], "split": e["split"], "column": "text"}
        got, want = resolve_users(DD, spec, state), e["want"]
        empty = "must list 1..1024 prompts; the prompt set gives 0"
        over = (f"must list 1..{DD.MAX_PROMPTS} prompts; the prompt set gives {len(want):,} (give a smaller split, "
                f"such as train[:400])") if not is_error(want) and len(want) > DD.MAX_PROMPTS else None
        if is_error(got) and got[2].endswith(empty) and (is_error(want) or want == []):
            kind = "empty_refused"
        elif is_error(got) and is_error(want):
            kind = "both_refuse"
        elif not is_error(got) and got == want:
            kind = "same_rows"
        elif over and is_error(got) and got[2].endswith(over) and resolve_unlimited(DD, spec, state) == want:
            kind = "over_limit_rows_equal"
        else:
            kind = None
            diffs.append({"repository": e["repository"], "split": e["split"],
                          "ide": got if is_error(got) else f"{len(got)} rows",
                          "datasets": want if is_error(want) else f"{len(want)} rows"})
        if kind:
            counts[kind] += 1
        rows.append({"repository": e["repository"], "split": e["split"], "result": kind or "different",
                     "datasets_rows": None if is_error(want) else len(want)})
    return {"cases": len(oracle["stage4"]), **counts, "differences": diffs, "rows": rows, "ok": not diffs}


def compare(oracle, runs, out, mode):
    sys.path.insert(0, APP)
    from wsp import directions_datasets as DD
    started = time.monotonic()
    work = tempfile.mkdtemp(prefix="compare-state-", dir=os.path.dirname(os.path.abspath(out)))
    state = os.path.join(work, "directions", "datasets")       # empty: Hub datasets are listed from the cache
    result = {"mode": mode, "versions": {"oracle": oracle["versions"], "ide": versions()},
              "cache": DD.hub_cache_dir(), "fixtures": oracle["fixtures"], "stages": {}}
    result["stages"]["1"] = stage1(DD, runs, state)
    result["stages"]["2"] = stage2(DD, oracle)
    result["stages"]["3"] = stage3(DD, oracle, state)
    result["stages"]["4"] = stage4(DD, oracle, state)
    result["state_written"] = sorted(os.listdir(work))
    shutil.rmtree(work)
    result["seconds"] = round(time.monotonic() - started, 3)
    result["ok"] = all(s["ok"] for s in result["stages"].values())
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    s = result["stages"]
    print(f"stage 1: {s['1']['equal']} of {s['1']['sets']} sets equal over {s['1']['runs']} runs, "
          f"{s['1']['files_checked']} file hashes checked, {len(s['1']['record_faults'])} record faults")
    print(f"stage 2: {s['2']['agree']} of {s['2']['cases']} agree")
    print(f"stage 3: {s['3']['same']} same, {s['3']['both_refuse']} both refuse, {s['3']['deliberate']} deliberate "
          f"refusals, {len(s['3']['differences'])} different, of {s['3']['cases']}")
    print(f"stage 4: {s['4']['same_rows']} same rows, {s['4']['both_refuse']} both refuse, {s['4']['empty_refused']} "
          f"empty selections refused, {s['4']['over_limit_rows_equal']} over the limit of prompts per set with equal "
          f"rows, {len(s['4']['differences'])} different, of {s['4']['cases']}")
    print(f"ok: {result['ok']} ({mode}; {result['seconds']} s)")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--write-oracle", help="write Heretic's side (Heretic's interpreter) to this file")
    p.add_argument("--oracle", help="compare the IDE against this file (the IDE's interpreter)")
    p.add_argument("--runs", help="the directory of Heretic runs (<model>/<run>/dump/dump.json and config.toml)")
    p.add_argument("--fixtures", help="a scratch directory for the local layouts (written by this script)")
    p.add_argument("--out", help="the result file")
    a = p.parse_args()
    if a.write_oracle:
        if not a.fixtures:
            p.error("--write-oracle needs --fixtures")
        oracle = make_oracle(os.path.abspath(a.fixtures))
        with open(a.write_oracle, "w", encoding="utf-8") as f:
            json.dump(oracle, f, ensure_ascii=False)
        print(f"oracle: {len(oracle['stage2'])} split cases, {len(oracle['stage3'])} local cases, "
              f"{len(oracle['stage4'])} Hub cases ({oracle['seconds']} s)")
        return
    if not (a.runs and a.out):
        p.error("--runs and --out are needed to compare")
    if a.oracle:
        with open(a.oracle, encoding="utf-8") as f:
            oracle = json.load(f)
        mode = "oracle file"
    else:
        if not a.fixtures:
            p.error("one process needs --fixtures")
        oracle = make_oracle(os.path.abspath(a.fixtures))
        mode = "one process"
    result = compare(oracle, a.runs, a.out, mode)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
