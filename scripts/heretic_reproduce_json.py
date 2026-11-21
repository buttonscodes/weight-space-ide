"""Rebuild a finished Heretic run's reproduce.json with Heretic's own code, offline and without a model.

Run with Heretic's own interpreter, with the run directory (config.toml, checkpoints/) as the current directory:

    python heretic_reproduce_json.py <out.json> <timestamp> <weights dir> [<model commit>]

Heretic writes reproduce.json only when it uploads a model; this script makes the same file from what the run left:
config.toml is read by Settings() (the command line cleared first, as heretic_dump.py does); the Optuna journal in
checkpoints/ is copied next to <out.json> and read from the copy with JournalStorage (the run's own journal is never
opened); the recorded settings are rebuilt with Settings.model_validate_init_only(json.loads(study.user_attrs
["settings"]), settings); the trial is the one Heretic exports: the Pareto front sorted by the objective values as
heretic/main.py sorts it, indexed by the config's trial_index; settings.n_trials is set to the number of trials in the
study and settings.export_strategy to the config's, as heretic/main.py does before an upload; hashes are
heretic.utils.get_file_sha256 of every <weights dir>/*.safetensors (an adapter run: adapter/, a merged run: model/).
The file is heretic.utils.generate_reproduce_json(settings, trial, timestamp, hashes,
include_system_information=False). The timestamp is given (ISO 8601, UTC, no microseconds), so two runs of this
script give the same bytes.

Heretic fetches the base model's commit from the Hub before it writes the file (heretic/utils.py,
create_reproduce_folder: settings.model_commit = huggingface_hub.model_info(settings.model).sha). Nothing is fetched
here: the commit is the config's model_commit, or, for a run whose config does not pin one, the <model commit>
argument, which must be what the run loaded offline: the commit the local cache's refs/main names for the model,
with that snapshot present (read from the cache directory, no request).

Asserted (exit 1 otherwise): settings.model_commit is set (by the config, or by the argument as above); exactly one journal;
trial_index is set and within the front; every scorer is an objective (the front's sort key is then the scores in
scorer order); the weights directory holds a safetensors file. Each assertion prints a PASS or FAIL line;
<out>.checks.json holds every number with a pass flag per assertion (the journal's copy is <out>.journal.jsonl).
Uses Heretic's code only; imports nothing from wsp.
"""
import json
import shutil
import sys
from pathlib import Path

if len(sys.argv) not in (4, 5):
    raise SystemExit(__doc__)
out, timestamp, weights = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]).resolve()
given_commit = sys.argv[4] if len(sys.argv) == 5 else None
sys.argv = sys.argv[:1]                      # Settings() parses the command line

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

from heretic.config import Settings
from heretic.utils import generate_reproduce_json, get_file_sha256

checks = {}


def check(name, ok, **numbers):
    checks[name] = {"pass": bool(ok), **numbers}
    print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(numbers, default=str)}", flush=True)
    return ok


optuna.logging.set_verbosity(optuna.logging.WARNING)
current = Settings()
journals = sorted(Path(current.study_checkpoint_dir).glob("*.jsonl"))
if not check("one journal", len(journals) == 1, journals=len(journals)):
    raise SystemExit(1)
out.parent.mkdir(parents=True, exist_ok=True)
copy = out.parent / (out.stem + ".journal.jsonl")
shutil.copyfile(journals[0], copy)
study = optuna.load_study(study_name="heretic", storage=JournalStorage(JournalFileBackend(str(copy))))
settings = Settings.model_validate_init_only(json.loads(study.user_attrs["settings"]), current)
source = "config"
if not settings.model_commit and given_commit:
    # what an offline run without a pinned commit loads: the snapshot the cache's refs/main names
    from huggingface_hub.constants import HF_HUB_CACHE
    repo = Path(HF_HUB_CACHE) / ("models--" + settings.model.replace("/", "--"))
    ref = repo / "refs" / "main"
    cached = ref.read_text().strip() if ref.is_file() else None
    if cached == given_commit and (repo / "snapshots" / given_commit).is_dir():
        settings.model_commit, source = given_commit, "argument, equal to the cache's refs/main"
    else:
        source = f"argument refused: the cache's refs/main is {cached}"
ok = check("model_commit is set", bool(settings.model_commit), model=settings.model, model_commit=settings.model_commit,
           source=source)

complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
optimizations = [s.optimization for s in settings.scorers]
names = [s["name"] for s in complete[0].user_attrs["scores"]] if complete else []
ok &= check("every scorer is an objective", complete and all(o != "none" for o in optimizations)
            and len(names) == len(optimizations), optimizations=optimizations, score_names=names)
if not ok:
    raise SystemExit(1)


def key(trial):                              # heretic/main.py: the front sorted by the objectives' values
    return tuple(next((s["score"]["value"] for s in trial.user_attrs["scores"] if s["name"] == n), None) for n in names)


front = sorted(study.best_trials, key=key)
index = current.trial_index
if not check("trial_index within the front", index is not None and 0 <= index < len(front), trial_index=index,
             front=[t.user_attrs.get("index") for t in front]):
    raise SystemExit(1)
trial = front[index]
settings.n_trials = len(study.trials)
settings.export_strategy = current.export_strategy
files = sorted(weights.glob("*.safetensors"))
hashes = {f.name: get_file_sha256(f) for f in files}
check("weight files found", bool(files), weights=[f.name for f in files])
text = generate_reproduce_json(settings, trial, timestamp, hashes, include_system_information=False)
out.write_text(text, encoding="utf-8")
summary = {"out": str(out), "trial": trial.user_attrs.get("index"), "n_trials": settings.n_trials,
           "export_strategy": str(settings.export_strategy), "hashes": hashes, "checks": checks,
           "pass": all(c["pass"] for c in checks.values())}
out.with_suffix(".checks.json").write_text(json.dumps(summary, indent=1, default=str))
print(f"{'ok' if summary['pass'] else 'NOT OK'}: {out} (trial {summary['trial']}, {len(hashes)} weight files)",
      flush=True)
sys.exit(0 if summary["pass"] else 1)
