"""The automatic search for a behaviour edit's parameters, as Heretic searches them.

A search draws the parameters of one method (directional ablation or ARA) from Heretic's search space with
Heretic's sampler (wsp.directions_sampler: Optuna's multivariate TPE, seeded), runs each draw's weight script as an
uncommitted candidate on top of the current weights, scores it with the scores of wsp.directions_scores, and tells
the sampler the scores whose ``optimization`` is ``minimize`` or ``maximize``. The trials that no other complete
trial dominates form the Pareto front, sorted as Heretic sorts it.

The spaces (SPACES) and the transformation from draws to parameters (``to_parameters``) are Heretic's, written as
Heretic writes them, so with the same seed the startup draws are Heretic's whatever the scores, and the later draws
stay Heretic's as long as the trials rank the same way.

Storage, in the workspace: ``directions/searches/<name>/search.json`` (the settings, scorers with their prompts,
seed, sampler and its versions, space and the identities of what the trials describe; written last at creation),
``measurement.json`` and ``measurement.prompts.json`` (the measurement as it was at creation) and
``trials/<index:06d>.json``, one record per trial, rewritten atomically as the trial goes on. The record of a trial
is written before the sampler is told its outcome.

Resume by replay: a run makes a fresh sampler and replays every finished trial (ask with Heretic's suggest calls,
check the draws equal the recorded ones, tell the recorded outcome), so its generators stand where an uninterrupted
search's stand. A trial stopped inside (by the caller's ``stopped()``, at a script progress point or a scoring unit)
or interrupted by the machine (an admission refusal, a lost model worker, a killed process) is not a trial: nothing
is told and the next run asks again, which draws the same parameters, and reruns it. A trial that raises is
``failed``: told to the sampler as FAIL (it counts towards n_trials, not towards the startup trials), and three in a
row end the run. One search runs per workspace at a time, across processes (an flock on ``searches/build.lock``);
each trial is one admitted operation holding the workspace lock, released between trials.

Nothing here commits an edit: a trial is a candidate, its pieces held for its scores and dropped, and the search
keeps no pieces (only their digest). Nothing here calls Workspace.commit, Session.commit_deltas or Session.apply
except ``apply_trial``, which commits a trial through Session.apply.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import statistics
import time
import uuid

import torch

from . import directions as DR
from . import directions_sampler as S
from . import directions_scores as DS
from .runtime_events import RuntimeRefusal, job_scope

SEARCH_VERSION = 1
DIRECTORY = "searches"
HERETIC = "64e9991"                    # the Heretic commit whose spaces, sampler and semantics these are
FINAL = ("complete", "failed")
STATES = ("complete", "failed", "stopped", "interrupted", "running")
MAX_CONSECUTIVE_FAILURES = 3
MAX_TRIALS = 10_000
DEFAULT_TRIALS = 100                   # Heretic's n_trials default
DEFAULT_STARTUP_TRIALS = 30            # Heretic's n_startup_trials default
MAX_SEED = 2 ** 32 - 1                 # the range of Optuna's sampler seed
TRIAL_RECORD_BYTES = 8 << 20           # free disk (beyond DR.DISK_RESERVE) before creating a search and each trial
RESPONSE_CHARS = 1_000_000             # a trial's response texts above this are dropped from its record
OUTPUT_CHARS = 20_000                  # a trial's script output is kept to this
TRANSPORT_SHARE_LIMIT = 0.1
TMP_SECONDS = 3600                     # a temporary file older than this in a search directory is removed at a run
CREATE_FIELDS = ("name", "modifiers", "scorers", "n_trials", "n_startup_trials", "seed", "chat", "system", "prefix",
                 "chat_options", "max_response_length", "start", "stream")
RENDERING = ("chat", "system", "prefix", "chat_options")
RUN_FIELDS = ("n_additional_trials", "accept_changed_libraries")
APPLY_FIELDS = ("dry_run", "message", "enforce_scope", "accept_different_edit")

# Heretic plugin -> the IDE's method and the settings the search accepts (Heretic's names and defaults). Keys of
# "measured" belong to the measurement (POST /api/directions/measure) and are refused in a search.
MODIFIERS = {
    "Abliteration": {"method": "ablation", "path": "heretic.modifiers.abliteration.Abliteration",
                     "settings": {"orthogonalize_direction": True, "row_normalization": "full",
                                  "full_normalization_lora_rank": 3},
                     "measured": ("good_prompts", "bad_prompts", "winsorization_quantile"),
                     "ignored": ()},
    "ARA": {"method": "ara", "path": "heretic.modifiers.ara.ARA",
            "settings": {"preserve_row_magnitudes": True, "lora_rank": 50, "n_optimization_steps": 5,
                         "learning_rate": 1.0, "max_iter": 20, "history_size": 10, "print_loss": False},
            "measured": ("good_prompts", "bad_prompts"),
            "ignored": ("print_loss",)},      # accepted and recorded; the script's output always has the losses
}

# Heretic's search spaces, in its suggest order (heretic/modifiers/abliteration.py and ara.py, suggest_parameters).
# m: n (layers), L (n - 1), n_good, n_bad. The arithmetic is Heretic's, so the bounds are the same floats.
SPACES = {
    "ablation": {
        "head": [
            ("direction_scope", "categorical", lambda m: ["global", "per layer"]),
            ("direction_index", "float", lambda m: (0.4 * m["L"], 0.9 * m["L"])),
        ],
        "per_component": [                   # names "<component>.<key>", components in sorted order
            ("max_weight", "float", lambda m, c: (-0.25 if c == "mlp.down_proj" else 0.8, 1.5)),
            ("max_weight_position", "float", lambda m, c: (0.6 * m["L"], 1.0 * m["L"])),
            ("min_weight", "float", lambda m, c: (0.0, 1.0)),
            ("min_weight_distance", "float", lambda m, c: (1.0, max(0.6 * m["L"], 1.0))),
        ],
    },
    "ara": {
        "head": [
            ("start_layer_index", "int", lambda m: (0, m["n"] // 2)),
            ("end_layer_index", "int", lambda m: (m["n"] // 2 + 1, m["n"])),
            ("preserve_good_behavior_weight", "float", lambda m: (0.0, 1.0)),
            ("steer_bad_behavior_weight", "float_log", lambda m: (0.0001, 1.0)),
            ("overcorrect_relative_weight", "float", lambda m: (0.0, 1.3)),
            ("neighbor_count", "int", lambda m: (1, min(15, m["n_good"], m["n_bad"]))),
        ],
        "per_component": [],
    },
}
ARA_DRAWN = tuple(row[0] for row in SPACES["ara"]["head"])
ARA_SETTINGS = ("preserve_row_magnitudes", "lora_rank", "n_optimization_steps", "learning_rate", "max_iter",
                "history_size")
ABLATION_SETTINGS = ("orthogonalize_direction", "row_normalization", "full_normalization_lora_rank")


class SearchError(DR.DirectionError):
    pass


class SearchMissing(SearchError):
    """No search of that name (the routes answer 404)."""


class SearchBusy(SearchError):
    """A search runs (the routes answer 409)."""


# ------------------------------------------------------------------ the space, the parameters (pure)
def _distribution(kind, bounds):
    if kind == "categorical":
        return {"name": "CategoricalDistribution", "attributes": {"choices": list(bounds)}}
    lo, hi = bounds
    if kind == "int":
        return {"name": "IntDistribution", "attributes": {"log": False, "step": 1, "low": int(lo), "high": int(hi)}}
    return {"name": "FloatDistribution",
            "attributes": {"step": None, "low": float(lo), "high": float(hi), "log": kind == "float_log"}}


def space(method, *, n_layers, components=(), n_good=0, n_bad=0) -> list[dict]:
    """The parameters Heretic draws for ``method``, in its suggest order, each with its distribution in Optuna's
    ``distribution_to_json`` form (built without Optuna)."""
    if method not in SPACES:
        raise SearchError(f"method must be one of {', '.join(SPACES)}")
    m = {"n": n_layers, "L": n_layers - 1, "n_good": n_good, "n_bad": n_bad}
    rows = [{"name": name, "distribution": _distribution(kind, f(m))} for name, kind, f in SPACES[method]["head"]]
    for c in sorted(components):
        for key, kind, f in SPACES[method]["per_component"]:
            rows.append({"name": f"{c}.{key}", "distribution": _distribution(kind, f(m, c))})
    return rows


def to_parameters(method, draws, settings, *, direction, components, seed) -> dict:
    """Heretic's transformation of one trial's draws into the method's parameters (abliteration.py and ara.py,
    suggest_parameters), plus the settings: the input of ``DR.script_for``."""
    if method == "ablation":
        di = None if draws["direction_scope"] == "per layer" else draws["direction_index"]
        wd = {}
        for c in sorted(components):
            mw = max(0.0, draws[f"{c}.max_weight"])
            wd[c] = {"max_weight": mw, "max_weight_position": draws[f"{c}.max_weight_position"],
                     "min_weight": draws[f"{c}.min_weight"] * mw,
                     "min_weight_distance": draws[f"{c}.min_weight_distance"]}
        return {"method": "ablation", "direction": direction, "direction_index": di, "weight_distributions": wd,
                **{k: settings[k] for k in ABLATION_SETTINGS}}
    if method == "ara":
        return {"method": "ara", "direction": direction, **{k: draws[k] for k in ARA_DRAWN},
                **{k: settings[k] for k in ARA_SETTINGS}, "seed": seed}
    raise SearchError(f"method must be one of {', '.join(SPACES)}")


def heretic_parameters(method, parameters) -> dict:
    """What Heretic records as a trial's parameters (``user_attrs["parameters"]``)."""
    if method == "ablation":
        return {"direction_index": parameters["direction_index"],
                "weight_distributions": {c: dict(d) for c, d in parameters["weight_distributions"].items()}}
    return {k: parameters[k] for k in ARA_DRAWN}


def present(method, heretic_parameters) -> dict:
    """Heretic's ``to_presentation_dict`` of those parameters."""
    p = heretic_parameters
    if method == "ablation":
        out = {"direction_index": "per layer" if p["direction_index"] is None else f"{p['direction_index']:.2f}"}
        for c, d in p["weight_distributions"].items():
            for k, v in d.items():
                out[f"{c}.{k}"] = f"{v:.2f}"
        return out
    return {k: (f"{v:.4f}" if isinstance(v, float) else f"{v}") for k, v in p.items()}


def front(records, objectives) -> list[int]:
    """The Pareto front of the complete trials (Optuna's best_trials: no other complete trial dominates them after
    ``maximize`` objectives are negated; equal value vectors are both on it), as trial indices sorted by the raw
    objective values ascending (Heretic's order), ties by index."""
    pool = sorted((r for r in records if r["state"] == "complete"), key=lambda r: r["index"])
    dirs = [o["direction"] for o in objectives]

    def loss(r):
        return [v if d == "minimize" else -v for v, d in zip(r["values"], dirs)]
    losses = [loss(r) for r in pool]
    on = []
    for t, lt in zip(pool, losses):
        dominated = any(all(a <= b for a, b in zip(lu, lt)) and any(a < b for a, b in zip(lu, lt)) for lu in losses)
        if not dominated:
            on.append(t)
    return [r["index"] for r in sorted(on, key=lambda r: tuple(r["values"]))]


def piece_digest(deltas) -> str:
    """SHA-256 of a candidate's pieces as they are packed (zero pieces and tensors left without pieces dropped): the
    packing index as compact sorted JSON, then per packed tensor (sorted by key) its key, dtype, shape and bytes."""
    from . import delta as D
    kept = {k: [p for p in ps if not p.is_zero()] for k, ps in deltas.items()}
    kept = {k: ps for k, ps in kept.items() if ps}
    index, tensors = D.pack(kept)
    h = hashlib.sha256(json.dumps(index, sort_keys=True, separators=(",", ":")).encode())
    for key in sorted(tensors):
        t = tensors[key].detach().contiguous().cpu()
        h.update(key.encode())
        h.update(str(t.dtype).encode())
        h.update(json.dumps(list(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment() -> dict:
    """The library versions, sampler versions and thread count a trial ran with (no host, user or path)."""
    import numpy
    return {"libraries": {"torch": str(torch.__version__), "transformers": _version("transformers")},
            "sampler": {"optuna": S.version(), "numpy": numpy.__version__}, "python": platform.python_version(),
            "torch_threads": torch.get_num_threads(), "machine": platform.machine(), "system": platform.system()}


LIMITS = {"modifiers": list(MODIFIERS), "methods": list(SPACES), "n_trials": [1, MAX_TRIALS],
          "n_startup_trials": [0, MAX_TRIALS], "seed": [0, MAX_SEED], "scorers_per_search": DS.MAX_SCORERS,
          "sampler": {"class": S.SAMPLER["class"], "n_ei_candidates": S.SAMPLER["n_ei_candidates"],
                      "multivariate": S.SAMPLER["multivariate"]},
          "optuna": {"accepted": ">=5.0,<6", "validated": S.VALIDATED}, "consecutive_failures": MAX_CONSECUTIVE_FAILURES,
          "trial_record_bytes": TRIAL_RECORD_BYTES, "response_characters": RESPONSE_CHARS,
          "transport_share_limit": TRANSPORT_SHARE_LIMIT,
          # each modifier's settings with Heretic's defaults, and the counts' defaults, for a client's form
          "settings": {p: dict(MODIFIERS[p]["settings"]) for p in MODIFIERS},
          "defaults": {"n_trials": DEFAULT_TRIALS, "n_startup_trials": DEFAULT_STARTUP_TRIALS}}


# ------------------------------------------------------------------ identities and what a change means
def _sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def measurement_identity(ws, name) -> dict:
    """What a search's trials were measured from: hashes of the measurement's meta, means and io.json (the module
    inputs' arrays are not hashed: they are written once with io.json last and replaced only with a new meta)."""
    meta = DR._meta(ws, DR.check_name(name))
    paths = DR._paths(ws, name)
    try:
        means = _sha_file(paths["tensors"])
    except OSError:
        raise DR.DirectionError(f"direction {name!r}: its vectors file is missing") from None
    io_json = os.path.join(paths["io"], DR.IO_FILE)
    return {"name": name, "meta_sha256": hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "means_sha256": means, "io_sha256": _sha_file(io_json) if os.path.isfile(io_json) else None,
            "prompts_sha256": meta.get("prompts_sha256"), "created": meta.get("created"), "counts": meta.get("counts")}


def _plain(value):
    return json.loads(json.dumps(value))


def _model_now(ws, s):
    from .steering import model_identity
    return _plain(model_identity(ws))


def _current_now(ws, s):
    from .atlas import wiring
    return _plain(wiring.patch_fingerprint(ws))


MEASUREMENT_KEYS = ("meta_sha256", "means_sha256", "io_sha256")


def measurement_sha(ws, s):
    """The measurement's hashes now, or None when it is gone or unreadable (a change)."""
    try:
        m = measurement_identity(ws, s["modifier"]["direction"])
    except DR.DirectionError:
        return None
    return {k: m[k] for k in MEASUREMENT_KEYS}


def _libraries_now(ws, s):
    return environment()["libraries"]


# key, how it is read now, what adding trials does ("run"), what committing a trial does ("apply"). The scoring row is
# checked after DS.prepare (the rendered ids and the runner); the sampler row is the replay itself.
COMPAT = [
    ("model", _model_now, "refuse", "refuse"),
    ("current", _current_now, "refuse", "refuse"),
    ("measurement", measurement_sha, "refuse", "refuse"),
    ("scoring", None, "refuse", "allow"),
    ("libraries", _libraries_now, "refuse*", "digest"),
    ("sampler", None, "replay", "allow"),
]
CHECKED = {"run": ("model", "current", "measurement", "libraries"), "trial": ("model", "current", "measurement"),
           "apply": ("model", "current", "measurement")}
APPLY_TAIL = "its script would now make another edit than the one it scored"


def _accepted_libraries(s):
    changes = s.get("environment_changes") or []
    return changes[-1]["environment"]["libraries"] if changes else s["identity"]["libraries"]


def _recorded(s, key):
    if key == "measurement":
        m = s["identity"]["measurement"]
        return {k: m.get(k) for k in MEASUREMENT_KEYS}
    if key == "libraries":
        return _accepted_libraries(s)
    return s["identity"][key]


def _patch_ids(fingerprint):
    return ", ".join(str(p[0]) for p in fingerprint) if fingerprint else "none"


def _refusal(s, key, now) -> tuple[str, str]:
    """(what changed, what to do) of a refused row."""
    name = s["name"]
    if key == "model":
        return (f"search '{name}' was made on other base weights than this workspace has now",
                "its trials do not describe this model: start a new search")
    if key == "current":
        return (f"the enabled patches changed since search '{name}' was made (then: "
                f"{_patch_ids(s['identity']['current'])}; now: {_patch_ids(now)})",
                "switch them back, or start a new search on these weights")
    if key == "measurement":
        return (f"measurement '{s['modifier']['direction']}' changed since search '{name}' was made (measured again "
                f"or deleted)", "start a new search")
    if key == "libraries":
        then = _accepted_libraries(s)
        return (f"search '{name}' was scored with torch {then.get('torch')} and transformers {then.get('transformers')}; "
                f"this environment has torch {now.get('torch')} and transformers {now.get('transformers')}. Scores, and "
                f"an ARA trial's fit, can differ across versions",
                "pass accept_changed_libraries to continue anyway (each trial records its versions)")
    raise AssertionError(key)


def _changes(ws, s, when):
    """[(key, now, message for adding trials, message for a commit)] of the rows of ``when`` that changed."""
    out = []
    for key, read, _, _ in COMPAT:
        if key not in CHECKED[when]:
            continue
        now = read(ws, s)
        if now != _recorded(s, key):
            head, tail = _refusal(s, key, now)
            out.append((key, now, f"{head}; {tail}", f"{head}; {APPLY_TAIL}"))
    return out


def check_compat(ws, s, when, accept_changed_libraries=False) -> None:
    """Refuse (R7 for "run" and "trial", A6 for "apply") when what the search's trials describe changed: the base
    weights, the enabled patches, the measurement and, for "run", the library versions; with
    ``accept_changed_libraries`` a library change is recorded in ``environment_changes`` and accepted."""
    if when not in CHECKED:
        raise AssertionError(when)
    for key, now, run_message, apply_message in _changes(ws, s, when):
        if when == "apply":
            raise SearchError(apply_message)
        if key == "libraries" and accept_changed_libraries:
            final = sum(1 for r in trial_records(ws, s["name"], s) if r["state"] in FINAL)
            s.setdefault("environment_changes", []).append({"from_index": final + 1, "environment": environment()})
            _write_search(ws, s)
            continue
        raise SearchError(run_message)


def compatibility(ws, s) -> dict:
    """What adding trials and committing a trial would answer now, from the rows that need no model."""
    run_rows = _changes(ws, s, "run")
    apply_rows = [r for r in run_rows if r[0] in CHECKED["apply"]]
    return {"run": run_rows[0][2] if run_rows else None, "apply": apply_rows[0][3] if apply_rows else None,
            "checked_without_model": list(CHECKED["run"])}


# ------------------------------------------------------------------ storage
_RUNNING: dict = {}                    # realpath of a searches directory -> the search this process runs there


def _dir(ws):
    return os.path.join(ws.dir, DR.DIRECTORY, DIRECTORY)


def _key(ws):
    return os.path.realpath(_dir(ws))


def _name(name):
    if not isinstance(name, str) or not DR.NAME.match(name):
        raise SearchError("name must be 1..64 characters from A-Z, a-z, 0-9, '.', '_' and '-', starting with a letter or "
                          "digit")
    return name


def _sdir(ws, name):
    return os.path.join(_dir(ws), name)


def _trials_dir(ws, name):
    return os.path.join(_sdir(ws, name), "trials")


def _write_search(ws, s):
    DR._write_json(os.path.join(_sdir(ws, s["name"]), "search.json"), s)


def _write_trial(ws, name, rec):
    DR._write_json(os.path.join(_trials_dir(ws, name), f"{rec['index']:06d}.json"), rec)


def _reason_of(e):
    if isinstance(e, FileNotFoundError):
        return "the file is missing"
    if isinstance(e, OSError):
        return e.strerror or type(e).__name__
    return str(e)


def load(ws, name) -> dict:
    """The search record (search.json), or R1 (SearchMissing) / R2."""
    d = _sdir(ws, _name(name))
    if not os.path.isdir(d):
        raise SearchMissing(f"no search named '{name}'")
    try:
        with open(os.path.join(d, "search.json")) as f:
            s = json.load(f)
    except (OSError, ValueError) as e:
        raise SearchError(f"search '{name}': its record cannot be read ({_reason_of(e)})") from None
    if not isinstance(s, dict) or s.get("version") != SEARCH_VERSION or s.get("name") != name \
            or not isinstance(s.get("id"), str):
        raise SearchError(f"search '{name}': its record is not a version 1 search")
    return s


def trial_records(ws, name, s=None) -> list[dict]:
    """Every trial record in order, checked (R9): file names ``<index:06d>.json`` with index == number + 1 ==
    position + 1, this search's id, known states, at most one non-final record and that one last."""
    s = s if s is not None else load(ws, name)
    d = _trials_dir(ws, name)
    files = sorted(fn for fn in os.listdir(d) if fn.endswith(".json")) if os.path.isdir(d) else []
    recs = []
    for pos, fn in enumerate(files):
        try:
            with open(os.path.join(d, fn)) as f:
                rec = json.load(f)
        except (OSError, ValueError) as e:
            raise SearchError(f"search '{name}': trial file {fn} cannot be read ({_reason_of(e)})") from None
        if not isinstance(rec, dict) or rec.get("version") != 1:
            raise SearchError(f"search '{name}': trial file {fn} cannot be read (it is not a version 1 trial record)")
        what = None
        if fn != f"{pos + 1:06d}.json":
            what = f"trial file {fn} is in place {pos + 1}"
        elif rec.get("index") != pos + 1 or rec.get("number") != pos:
            what = f"trial file {fn} holds trial {rec.get('index')!r} (number {rec.get('number')!r})"
        elif rec.get("search_id") != s["id"]:
            what = f"trial file {fn} belongs to another search"
        elif rec.get("state") not in STATES:
            what = f"trial file {fn} has state {rec.get('state')!r}"
        if what:
            raise SearchError(f"search '{name}': its trial records are inconsistent ({what})")
        recs.append(rec)
    open_ = [r for r in recs if r["state"] not in FINAL]
    if len(open_) > 1 or (open_ and open_[0] is not recs[-1]):
        raise SearchError(f"search '{name}': its trial records are inconsistent (trial {open_[0]['index']} is "
                          f"{open_[0]['state']} but is not the last)")
    return recs


def running(ws) -> str | None:
    """Whether a search of this workspace runs now: "this IDE", "another process" or None (never waits)."""
    from .atlas import wiring
    return wiring.build_running(_dir(ws))


def _holder(ws, name):
    """Who runs search ``name`` now, as far as can be told: this process names its search; another process holding
    the lock is reported for every search of the workspace."""
    r = running(ws)
    if r == "this IDE":
        return r if _RUNNING.get(_key(ws)) in (name, None) else None
    return r


@contextlib.contextmanager
def search_lock(ws, name=None):
    """One search per workspace, across threads and processes (wiring.build_lock on the searches directory);
    refused at once with SearchBusy (R3)."""
    from .atlas import wiring
    key = _key(ws)
    stack = contextlib.ExitStack()
    try:
        stack.enter_context(wiring.build_lock(_dir(ws)))
    except wiring.Busy as e:
        if wiring.WHERE["this IDE"] in str(e):
            other = _RUNNING.get(key)
            where = wiring.WHERE["this IDE"] + (f" (search '{other}')" if other else "")
        else:
            where = wiring.WHERE["another process"]
        raise SearchBusy(f"a search is already running on this workspace{where}; stop it first") from None
    with stack:
        before = _RUNNING.get(key)
        _RUNNING[key] = name
        try:
            yield
        finally:
            if before is None:
                _RUNNING.pop(key, None)
            else:
                _RUNNING[key] = before


def _shown(rec, holder):
    return "interrupted" if rec["state"] == "running" and holder is None else rec["state"]


def _counts(recs, holder):
    out = {}
    for r in recs:
        st = _shown(r, holder)
        out[st] = out.get(st, 0) + 1
    return out


def _transport(recs):
    shares = [r["transport"]["share"] for r in recs
              if r["state"] == "complete" and r.get("transport") and r["transport"].get("share") is not None]
    if not shares:
        return None
    med = statistics.median(shares)
    return {"trials": len(shares), "median_share": med, "max_share": max(shares), "limit": TRANSPORT_SHARE_LIMIT,
            "over": med > TRANSPORT_SHARE_LIMIT}


def describe(ws, name, *, full=False, job=None) -> dict:
    """The search record, its trials (summaries, or full records with ``full``), the front, who runs it, what a
    run or a commit would answer now (without a model) and the transport summary."""
    s = load(ws, name)
    recs = trial_records(ws, name, s)
    holder = _holder(ws, name)
    fr = front(recs, s["objectives"])
    trials = []
    for r in recs:
        if full:
            trials.append({**r, "state": _shown(r, holder)})
            continue
        trials.append({"index": r["index"], "state": _shown(r, holder), "kind": r["kind"], "presented": r["presented"],
                       "values": r["values"], "displays": [x["display"] for x in r["scores"]] if r["scores"] else None,
                       "seconds": (r.get("seconds") or {}).get("total"), "on_front": r["index"] in fr})
    return {"search": s, "trials": trials, "front": fr, "running": holder, "job": job,
            "compatible": compatibility(ws, s), "transport": _transport(recs)}


def trial(ws, name, index) -> dict:
    """One full trial record, a running record without a holder shown as interrupted; A3 (SearchMissing) when the
    search has no trial ``index``."""
    s = load(ws, name)
    recs = trial_records(ws, name, s)
    if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(recs):
        raise SearchMissing(f"search '{name}' has no trial {index}")
    rec = recs[index - 1]
    return {**rec, "state": _shown(rec, _holder(ws, name))}


def front_entries(ws, name) -> dict:
    """The Pareto front in Heretic's order, with what each of its trials drew and its script (Heretic's
    ``trial_index`` setting is a position in this list)."""
    s = load(ws, name)
    recs = trial_records(ws, name, s)
    out = []
    for position, index in enumerate(front(recs, s["objectives"])):
        r = recs[index - 1]
        out.append({"position": position, "index": index, "values": r["values"],
                    "displays": [x["display"] for x in r["scores"]], "presented": r["presented"], "script": r["script"]})
    return {"objectives": s["objectives"], "front": out}


def check_run(ws, name, *, n_additional_trials=None, accept_changed_libraries=False) -> dict:
    """What ``run`` refuses before its first trial without a model, checked without writing anything (the routes
    answer these before a job starts): Optuna, the record and its trials, the trial count, the rows of COMPAT that
    need no model (a library change passes when it is accepted) and the disk -> {"final", "n_trials"} as the run
    would start."""
    S.require()
    if n_additional_trials is not None:
        _check_additional(n_additional_trials)
    if not isinstance(accept_changed_libraries, bool):
        raise SearchError("accept_changed_libraries must be true or false")
    s = load(ws, name)
    final = sum(1 for r in trial_records(ws, name, s) if r["state"] in FINAL)
    n_trials = s["n_trials"]
    if n_additional_trials is not None:
        n_trials = final + n_additional_trials
        if n_trials > MAX_TRIALS:
            raise SearchError(f"n_additional_trials: search '{name}' would have {n_trials} trials, more than {MAX_TRIALS}")
    elif final >= n_trials:
        raise SearchError(f"search '{name}' has run all its {n_trials} trials; ask for more with n_additional_trials")
    for key, _, message, _ in _changes(ws, s, "run"):
        if not (key == "libraries" and accept_changed_libraries):
            raise SearchError(message)
    _check_disk(ws)
    return {"final": final, "n_trials": n_trials}


def entries(ws) -> list[dict]:
    """Every search of the workspace; a damaged one is listed with ``error`` (and can be deleted)."""
    d = _dir(ws)
    out = []
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not os.path.isdir(os.path.join(d, fn)):
            continue
        try:
            s = load(ws, fn)
            recs = trial_records(ws, fn, s)
            holder = _holder(ws, fn)
            out.append({"name": fn, "id": s["id"], "created": s["created"], "method": s["method"],
                        "n_trials": s["n_trials"], "n_startup_trials": s["n_startup_trials"], "seed": s["seed"],
                        "objectives": s["objectives"], "trials": _counts(recs, holder),
                        "front": front(recs, s["objectives"]), "running": holder, "compatible": compatibility(ws, s)})
        except SearchError as e:
            out.append({"name": fn, "error": str(e)})
    return out


def delete(ws, name) -> None:
    """Remove a search (search.json first, then its directory); D1 while it runs."""
    d = _sdir(ws, _name(name))
    if not os.path.isdir(d):
        raise SearchMissing(f"no search named '{name}'")
    if _holder(ws, name) is not None:
        raise SearchBusy(f"search '{name}' is running; stop it first")
    path = os.path.join(d, "search.json")
    if os.path.exists(path):
        os.remove(path)
    shutil.rmtree(d)


# ------------------------------------------------------------------ creating a search (model-free checks)
def _model_components(ws):
    """The components some layer has a module of (wsp.directions.modules: a linear-attention mixer's output projection
    counts as attn.o_proj, an expert's down projection as mlp.down_proj)."""
    n = ws.struct.n_layers
    return sorted(c for c in DR.COMPONENTS if any(DR.modules(ws, l, c) for l in range(n)))


def _int_field(value, lo, hi, message):
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise SearchError(message)
    return value


def _check_modifier(ws, raw) -> dict:
    from . import directions_ara as ARA
    if not isinstance(raw, list) or len(raw) != 1:
        raise SearchError("modifiers must list exactly one modifier; using multiple modifiers is not yet supported (as "
                          "in Heretic)")
    m = raw[0]
    if not isinstance(m, dict):
        raise SearchError("modifiers[0] must be an object")
    plugin = m.get("plugin")
    paths = {row["path"]: p for p, row in MODIFIERS.items()}
    plugin = paths.get(plugin, plugin) if isinstance(plugin, str) else None
    if plugin not in MODIFIERS:
        raise SearchError(f"modifiers[0].plugin must be one of {', '.join(MODIFIERS)} (or their heretic.modifiers "
                          f"import paths)")
    row = MODIFIERS[plugin]
    for key in row["measured"]:
        if key in m:
            if key == "winsorization_quantile":
                raise SearchError("modifiers[0].winsorization_quantile: winsorization is part of the measurement "
                                  "(POST /api/directions/measure); the search uses the measurement as it was made")
            raise SearchError(f"modifiers[0].{key}: the IDE measures the prompts first (POST /api/directions/measure); "
                              f"name that measurement in modifiers[0].direction")
    fields = ("plugin", "instance_name", "direction") + tuple(row["settings"])
    unknown = sorted(set(m) - set(fields), key=str)
    if unknown:
        raise SearchError(f"modifiers[0].{unknown[0]}: unknown field; the fields of {plugin} are {', '.join(fields)}")
    if "direction" not in m:
        raise SearchError("modifiers[0].direction: required (the name of a saved measurement)")
    try:
        instance = DS._instance_name(m.get("instance_name"), 0, "modifiers")
    except DR.DirectionError as e:
        raise SearchError(str(e)) from None
    if "print_loss" in m and not isinstance(m["print_loss"], bool):
        raise SearchError("modifiers[0].print_loss must be true or false")
    given = {**row["settings"], **{k: m[k] for k in row["settings"] if k in m}}
    try:                                   # the settings, checked by the method's own normalizer (placeholder draws)
        if row["method"] == "ablation":
            placeholder = {"max_weight": 1.0, "max_weight_position": 0.0, "min_weight": 0.0, "min_weight_distance": 1.0}
            p = DR.normalize({"direction": m["direction"], "direction_index": None,
                              "weight_distributions": {c: dict(placeholder) for c in DR.COMPONENTS},
                              **{k: given[k] for k in ABLATION_SETTINGS}})
            settings = {k: p[k] for k in ABLATION_SETTINGS}
        else:
            p = ARA.normalize({"method": "ara", "direction": m["direction"], "start_layer_index": 0, "end_layer_index": None,
                               **{k: ARA.DEFAULTS[k] for k in ("preserve_good_behavior_weight", "steer_bad_behavior_weight",
                                                               "overcorrect_relative_weight")},
                               "neighbor_count": 1, "seed": 0, **{k: given[k] for k in ARA_SETTINGS}})
            settings = {**{k: p[k] for k in ARA_SETTINGS}, "print_loss": given["print_loss"]}
    except DR.DirectionError as e:
        raise SearchError(f"modifiers[0].{e}") from None
    return {"plugin": plugin, "instance_name": instance, "direction": p["direction"], "method": row["method"],
            "settings": settings}


def _check_create_disk(ws, direction):
    paths = DR._paths(ws, direction)
    need = TRIAL_RECORD_BYTES + sum(os.path.getsize(paths[k]) for k in ("meta", "prompts") if os.path.isfile(paths[k]))
    free = DR._free_bytes(ws.dir)
    if need > free - DR.DISK_RESERVE:
        raise SearchError(f"the search needs {need:,} bytes on disk; the workspace's file system has {free:,} bytes free, "
                          f"of which {DR.DISK_RESERVE:,} stay free")


def check_search(ws, body: dict) -> dict:
    """Everything about a create request that needs no model (C1-C27), in this order: the fields, the name, the
    modifier and its settings, the counts and the seed, the method's plan for its widest parameters (which reads the
    measurement), the scorers (as POST /api/directions/scores checks them, labelled ``scorers``) and their rendering,
    the disk, and last Optuna, so every other refusal is reported without it -> the normalized request."""
    if not isinstance(body, dict):
        raise SearchError("the request must be an object")
    unknown = sorted(set(body) - set(CREATE_FIELDS), key=str)
    if unknown:
        raise SearchError(f"{unknown[0]}: unknown field; the fields are {', '.join(CREATE_FIELDS[:-1])} and "
                          f"{CREATE_FIELDS[-1]}")
    for key in ("name", "modifiers", "scorers"):
        if key not in body:
            raise SearchError(f"{key}: required")
    name = _name(body["name"])
    if os.path.exists(_sdir(ws, name)):
        raise SearchError(f"a search named '{name}' exists; run more of its trials with POST "
                          f"/api/directions/searches/{name}/run, or delete it first")
    mod = _check_modifier(ws, body["modifiers"])
    n_trials = _int_field(body.get("n_trials", DEFAULT_TRIALS), 1, MAX_TRIALS,
                          f"n_trials must be an integer from 1 to {MAX_TRIALS}")
    n_startup = _int_field(body.get("n_startup_trials", DEFAULT_STARTUP_TRIALS), 0, MAX_TRIALS,
                           f"n_startup_trials must be an integer from 0 to {MAX_TRIALS}")
    seed = body.get("seed")
    if seed is not None:
        _int_field(seed, 0, MAX_SEED, f"seed must be null or an integer from 0 to {MAX_SEED} (the range of Optuna's "
                                      f"sampler seed)")
    start = body.get("start", True)
    if not isinstance(start, bool):
        raise SearchError("start must be true or false")
    n = ws.struct.n_layers
    comps = _model_components(ws)
    if mod["method"] == "ablation":
        widest = {"direction": mod["direction"], "direction_index": None,
                  "weight_distributions": {c: {"max_weight": 1.0, "max_weight_position": float(n - 1), "min_weight": 1.0,
                                               "min_weight_distance": float(n)} for c in comps},
                  **{k: mod["settings"][k] for k in ABLATION_SETTINGS}}
    else:
        widest = {"method": "ara", "direction": mod["direction"], "start_layer_index": 0, "end_layer_index": None,
                  **{k: mod["settings"][k] for k in ARA_SETTINGS}}
    try:
        DR.plan(ws, widest)
        meta = DR._meta(ws, mod["direction"])
    except DR.DirectionError as e:
        raise SearchError(f"modifiers[0]: {e}") from None
    given = {k: body[k] for k in RENDERING if k in body}
    opts = meta.get("options") or {}
    measured = {k: opts.get(k) for k in RENDERING}
    request = {"state": "current", "scores": body["scorers"], **(given or measured)}
    if "max_response_length" in body:
        request["max_response_length"] = body["max_response_length"]
    try:
        spec = DS.check_scores(ws, request, field="scorers")
    except DR.DirectionError as e:
        raise SearchError(str(e)) from None
    for i, raw in enumerate(body["scorers"]):
        if "optimization" not in raw:
            raise SearchError(f"scorers[{i}].optimization: required ('minimize', 'maximize' or 'none'; Heretic requires it)")
    objectives = [{"index": i, "name": e["name"], "direction": e["optimization"]} for i, e in enumerate(spec["scores"])
                  if e["optimization"] in ("minimize", "maximize")]
    if not objectives:
        raise SearchError('No optimization objectives configured. At least one scorer must set optimization to "maximize" '
                          'or "minimize".')
    rendering = {k: spec[k] for k in RENDERING}
    notes = []
    if given and rendering != measured:
        notes.append("scored with another rendering than the measurement was made with")
    _check_create_disk(ws, mod["direction"])
    version = S.require()
    counts = meta.get("counts") or {}
    model = {"n_layers": n, "components": comps, "n_good": counts.get("good", 0), "n_bad": counts.get("bad", 0)}
    return {"name": name, "modifier": {k: mod[k] for k in ("plugin", "instance_name", "direction", "settings")},
            "method": mod["method"],
            # the scorers' resolved prompts, inline, so a run never reads a dataset again; their records beside them
            "scoring": {"request": _plain({"scores": [{**raw, "prompts": spec["scores"][i]["prompts"]}
                                                      for i, raw in enumerate(body["scorers"])],
                                           **rendering, "max_response_length": spec["max_response_length"]}),
                        "names": [e["name"] for e in spec["scores"]], "rendering_from": "request" if given else "measurement",
                        "datasets": [e["datasets"] for e in spec["scores"]]},
            "objectives": objectives, "n_trials": n_trials, "n_startup_trials": n_startup, "seed": seed, "start": start,
            "model": model, "space": space(mod["method"], n_layers=n, components=comps, n_good=model["n_good"],
                                           n_bad=model["n_bad"]),
            "notes": notes, "optuna": version}


def create(ws, spec: dict) -> dict:
    """Write a new search from ``check_search``'s answer: the directory, the two measurement copies, then
    search.json -> the search record. Nothing is run."""
    import numpy
    version = S.require()
    name = _name(spec["name"])
    d = _sdir(ws, name)
    if os.path.exists(d):
        raise SearchError(f"a search named '{name}' exists; run more of its trials with POST "
                          f"/api/directions/searches/{name}/run, or delete it first")
    direction = spec["modifier"]["direction"]
    _check_create_disk(ws, direction)
    try:
        meta = DR._meta(ws, direction)
        prompts = DR.prompts(ws, direction)
        measurement = measurement_identity(ws, direction)
    except DR.DirectionError as e:
        raise SearchError(f"modifiers[0]: {e}") from None
    seed, seed_from = spec["seed"], "request"
    if seed is None:
        seed, seed_from = random.SystemRandom().randint(0, MAX_SEED), "random"
    env = environment()
    s = {"version": SEARCH_VERSION, "id": uuid.uuid4().hex, "name": name, "created": time.time(), "heretic": HERETIC,
         "method": spec["method"], "modifier": spec["modifier"], "scoring": spec["scoring"],
         "objectives": spec["objectives"], "n_trials": spec["n_trials"], "n_startup_trials": spec["n_startup_trials"],
         "seed": seed, "seed_from": seed_from,
         "sampler": {"class": S.SAMPLER["class"], "n_startup_trials": spec["n_startup_trials"],
                     "n_ei_candidates": S.SAMPLER["n_ei_candidates"], "multivariate": S.SAMPLER["multivariate"],
                     "seed": seed, "study_name": S.SAMPLER["study_name"], "optuna": version, "numpy": numpy.__version__,
                     "validated": version == S.VALIDATED},
         "model": spec["model"], "space": spec["space"],
         "identity": {"model": _model_now(ws, None), "current": _current_now(ws, None), "measurement": measurement,
                      "libraries": env["libraries"], "scoring": None},
         "environment": env, "environment_changes": [], "notes": list(spec["notes"])}
    s = _plain(s)
    os.makedirs(os.path.join(d, "trials"))
    DR._write_json(os.path.join(d, "measurement.json"), meta)
    DR._write_json(os.path.join(d, "measurement.prompts.json"), prompts)
    _write_search(ws, s)
    return s


# ------------------------------------------------------------------ running trials
def _check_additional(n):
    _int_field(n, 1, MAX_TRIALS, f"n_additional_trials must be an integer from 1 to {MAX_TRIALS}")


def _check_disk(ws):
    need = TRIAL_RECORD_BYTES
    free = DR._free_bytes(ws.dir)
    if need > free - DR.DISK_RESERVE:
        raise SearchError(f"the search needs {need:,} bytes free for its next trial record; the workspace's file system "
                          f"has {free:,} bytes free, of which {DR.DISK_RESERVE:,} stay free")


def _remove_old_tmp(ws, name):
    now = time.time()
    for d in (_sdir(ws, name), _trials_dir(ws, name)):
        for fn in os.listdir(d) if os.path.isdir(d) else []:
            if not fn.endswith(".tmp"):
                continue
            path = os.path.join(d, fn)
            try:
                if now - os.stat(path).st_mtime > TMP_SECONDS:
                    os.remove(path)
            except OSError:
                pass


def _scoring_spec(ws, s) -> dict:
    """The scoring spec of every trial: the search's request checked again, state candidate, and responses kept
    (print_responses) in the search's own copy for the records' rows."""
    spec = DS.check_scores(ws, {"state": "current", **s["scoring"]["request"]}, field="scorers")
    spec["state"] = "candidate"
    for e in spec["scores"]:
        if e["plugin"] in ("KeywordRate", "ResponseLength"):
            e["settings"] = {**e["settings"], "print_responses": True}
    return spec


@job_scope("direction_search")
def _prepare(ws, s, progress, stopped):
    with ws.lock:
        spec = _scoring_spec(ws, s)
        DS.runner_row(ws)
        unit = (lambda phase, done, total: progress(None, phase, done, total)) if progress is not None else None
        return DS.prepare(ws, spec, progress=unit, stopped=stopped)


def _runner_text(r):
    return f"{r.get('class')} {r.get('dtype')} {r.get('device')}"


def _check_scoring(ws, s, prepared):
    """The scoring row of COMPAT: the rendered prompts and the runner, stored by the first run, compared later."""
    now = {"rendered_sha256": hashlib.sha256(json.dumps(prepared["ids"]).encode()).hexdigest(),
           "runner": _plain(DS.reference_key(ws, [])[1]["runner"])}
    then = s["identity"].get("scoring")
    if then is None:
        s["identity"]["scoring"] = now
        _write_search(ws, s)
        return
    if then["rendered_sha256"] != now["rendered_sha256"]:
        raise SearchError(f"the scored prompts of search '{s['name']}' render differently now (the tokenizer or the chat "
                          f"template changed, or the search was scored before the IDE rendered prompts as Heretic does); "
                          f"start a new search")
    if then["runner"] != now["runner"]:
        raise SearchError(f"search '{s['name']}' was scored on {_runner_text(then['runner'])}; the model now runs on "
                          f"{_runner_text(now['runner'])}, so its scores would not be comparable")


def _mismatch(s, index):
    now = S.versions()
    return SearchError(f"the sampler does not reproduce trial {index} of search '{s['name']}' (made with Optuna "
                       f"{s['sampler']['optuna']} and numpy {s['sampler']['numpy']}; this environment has Optuna "
                       f"{now['optuna']} and numpy {now['numpy']}): its next draws would not be Heretic's. Start a new "
                       f"search with the same seed")


def _kind(final, s):
    return "startup" if sum(1 for r in final if r["state"] == "complete") < s["n_startup_trials"] else "tpe"


def _attempt(prior):
    state = "interrupted" if prior["state"] == "running" else prior["state"]
    reason = prior.get("attempt_reason") or prior.get("error")
    if prior["state"] == "running" and not reason:
        reason = "the process running the trial ended before it finished"
    return {"state": state, "started": prior.get("started"), "ended": prior.get("ended"), "reason": reason,
            "done": prior.get("done"), "total": prior.get("total")}


def _new_record(s, number, draws, kind, prior=None) -> dict:
    mod = s["modifier"]
    params = to_parameters(s["method"], draws, mod["settings"], direction=mod["direction"],
                           components=s["model"]["components"], seed=s["seed"])
    hp = heretic_parameters(s["method"], params)
    attempts = (list(prior.get("attempts") or []) + [_attempt(prior)]) if prior is not None else []
    return {"version": 1, "search_id": s["id"], "index": number + 1, "number": number, "state": "running", "kind": kind,
            "draws": dict(draws), "parameters": hp, "presented": present(s["method"], hp),
            "script": DR.script_for(params), "values": None, "scores": None, "responses_omitted": None,
            "candidate": None, "transport": None, "seconds": None, "started": time.time(), "ended": None,
            "attempts": attempts, "error": None, "event": None, "environment": None, "reference_key": None,
            "attempt_reason": None, "done": None, "total": None}


def _trailing_failures(final):
    n = 0
    for r in reversed(final):
        if r["state"] != "failed":
            break
        n += 1
    return n


@job_scope("direction_search")
def _trial(ws, prepared, rec, scope, progress, stopped):
    """One admitted operation holding the workspace lock (as DS.score). Raises; the caller classifies.
    -> (the scores' answer, (script seconds, scoring seconds), the transfer meter or None in this process)."""
    from . import worker_jobs
    from .remote_runner import TransferMeter
    with ws.lock:
        deltas = None
        meter = TransferMeter() if worker_jobs.uses_worker(ws) else None
        try:
            t0 = time.monotonic()
            deltas, report = DS.candidate(ws, rec["script"], scope, stopped=stopped)     # ScriptStopped passes through
            rec["candidate"] = {**{k: v for k, v in report.items() if k != "note"}, "output": report["output"][:OUTPUT_CHARS],
                                "digest": piece_digest(deltas) if deltas else None, "note": report["note"]}
            t1 = time.monotonic()
            if meter is not None:
                ws.runner.meter = meter
            unit = (lambda phase, d, t: progress(rec["index"], phase, d, t)) if progress is not None else None
            out = DS.evaluate(ws, prepared, "candidate", deltas, progress=unit, stopped=stopped)
            t2 = time.monotonic()
            return out, (t1 - t0, t2 - t1), meter
        finally:
            if meter is not None:
                ws.runner.meter = None
            DS._release(ws, deltas)


def _stop_reason(stopped):
    """The text ``stopped()`` answers (the server's says who stopped the search), or a plain stop."""
    try:
        r = stopped() if stopped is not None else None
    except Exception:
        r = None
    return r if isinstance(r, str) and r else "stopped by the person"


def _trim_scores(scores):
    """The scores as a record keeps them: per-prompt rows without the prompt text (the prompts are the search's),
    with the response texts unless together they are above RESPONSE_CHARS."""
    out, chars = [], 0
    for x in scores:
        x = dict(x)
        responses = x.pop("responses", None)
        if responses is not None:
            x["rows"] = [{k: v for k, v in r.items() if k != "prompt"} for r in responses]
            chars += sum(len(r["response"]) for r in responses)
        out.append(x)
    if chars <= RESPONSE_CHARS:
        return out, None
    for x in out:
        for r in x.get("rows") or []:
            r.pop("response", None)
    return out, f"{chars:,} characters of responses, above {RESPONSE_CHARS:,}"


def _run_trial(ws, s, prepared, rec, scope, progress, stopped) -> str:
    """Run one trial and fill its record -> its outcome: complete, failed, stopped or interrupted."""
    from .wscript import ScriptStopped
    started = time.monotonic()
    try:
        out, (t_script, t_scoring), meter = _trial(ws, prepared, rec, scope, progress, stopped)
    except ScriptStopped:
        rec.update(state="stopped", attempt_reason=_stop_reason(stopped), done=0, total=None, ended=time.time())
        return "stopped"
    except RuntimeRefusal as e:
        rec.update(state="interrupted", error=str(e), event=e.record, ended=time.time())
        return "interrupted"
    except Exception as e:
        rec.update(state="failed", error=f"{type(e).__name__}: {e}", ended=time.time())
        return "failed"
    total = time.monotonic() - started
    if out.get("stopped"):
        rec.update(state="stopped", attempt_reason=_stop_reason(stopped), done=out.get("done"), total=out.get("total"),
                   ended=time.time())
        return "stopped"
    scores, omitted = _trim_scores(out["scores"])
    transport = None
    if meter is not None:                  # 8.2: the share of the trial spent carrying its pieces to the worker
        m = meter.as_dict()
        transport = {**m, "share": m["seconds"] / total if total > 0 else None}
    rec.update(state="complete", scores=scores, responses_omitted=omitted,
               values=[scores[o["index"]]["value"] for o in s["objectives"]],
               seconds={"script": t_script, "scoring": t_scoring, "total": total}, transport=transport,
               environment=environment(), reference_key=prepared["reference"]["key"] if prepared["reference"] else None,
               ended=time.time())
    return "complete"


def _end(ws, s, state, reason, ran, event=None) -> dict:
    recs = trial_records(ws, s["name"], s)
    return {"state": state, "reason": reason, "ran": list(ran), "front": front(recs, s["objectives"]),
            "trials": _counts(recs, "this IDE"), "event": event}


def run(ws, name, *, n_additional_trials=None, accept_changed_libraries=False, scope=None,
        progress=None, stopped=None, after_trial=None, event=None) -> dict:
    """Run trials of a search until it has n_trials final trials, a stop, or an end condition.

    ``progress(index, phase, done, total)`` as often as the scores call it (index None while the KL reference is
    computed); ``event("trial", trial, trial_state, values, displays, front)`` after each trial; ``stopped()`` is
    asked at every stop point (between trials, during the KL reference, before each statement and component of the
    trial's script, before every scoring unit) and may answer the reason as text; ``after_trial()`` between trials.
    -> {"state": "finished" | "stopped" | "failed", "reason", "ran": [indices], "front": [indices], "trials": counts
    by state, "event"}. Raises SearchError (or the scores' DirectionError, or a RuntimeRefusal) for a refusal before
    the first trial of this run; a refusal between later trials ends the run "failed" with it as the reason."""
    S.require()
    if n_additional_trials is not None:
        _check_additional(n_additional_trials)
    if not isinstance(accept_changed_libraries, bool):
        raise SearchError("accept_changed_libraries must be true or false")
    with search_lock(ws, name):
        s = load(ws, name)
        recs = trial_records(ws, name, s)
        final = [r for r in recs if r["state"] in FINAL]
        pending = next((r for r in recs if r["state"] not in FINAL), None)
        if n_additional_trials is not None:
            total = len(final) + n_additional_trials          # Heretic: len(study.trials) + n_additional_trials
            if total > MAX_TRIALS:
                raise SearchError(f"n_additional_trials: search '{name}' would have {total} trials, more than {MAX_TRIALS}")
            s["n_trials"] = total
            _write_search(ws, s)
        elif len(final) >= s["n_trials"]:
            raise SearchError(f"search '{name}' has run all its {s['n_trials']} trials; ask for more with "
                              f"n_additional_trials")
        _remove_old_tmp(ws, name)
        check_compat(ws, s, "run", accept_changed_libraries)
        _check_disk(ws)
        prepared = _prepare(ws, s, progress, stopped)
        if prepared.get("stopped"):
            return _end(ws, s, "stopped", "stopped while the KL reference was computed", [])
        _check_scoring(ws, s, prepared)
        smp = S.Sampler(seed=s["seed"], n_startup_trials=s["n_startup_trials"],
                        directions=[o["direction"] for o in s["objectives"]])
        try:
            smp.replay(final, s["space"])
        except S.ReplayMismatch as e:
            raise _mismatch(s, e.index) from None
        failures = _trailing_failures(final)
        ran = []
        while len(final) < s["n_trials"]:
            if stopped is not None and stopped():
                return _end(ws, s, "stopped", "stopped between trials", ran)
            if after_trial is not None and after_trial():
                return _end(ws, s, "stopped", "stopped after a trial", ran)
            try:
                check_compat(ws, s, "trial", accept_changed_libraries)
                _check_disk(ws)
            except SearchError as e:
                if not ran:
                    raise
                return _end(ws, s, "failed", str(e), ran)
            number = len(final)
            handle, draws = smp.ask(s["space"])
            prior = pending if pending is not None and pending["number"] == number else None
            if prior is not None and prior["draws"] != draws:
                raise _mismatch(s, number + 1)
            rec = _new_record(s, number, draws, _kind(final, s), prior)
            pending = None
            _write_trial(ws, name, rec)                        # state "running"
            outcome = _run_trial(ws, s, prepared, rec, scope, progress, stopped)
            _write_trial(ws, name, rec)                        # written before the sampler is told
            if outcome == "complete":
                smp.tell(handle, values=rec["values"])
                final.append(rec)
                failures = 0
            elif outcome == "failed":
                smp.tell(handle, failed=True)
                final.append(rec)
                failures += 1
            ran.append(rec["index"])
            if event is not None:
                event("trial", trial=rec["index"], trial_state=rec["state"], values=rec["values"],
                      displays=[x["display"] for x in rec["scores"]] if rec["scores"] else None,
                      front=front(final, s["objectives"]))
            if outcome == "stopped":
                return _end(ws, s, "stopped", rec["attempt_reason"], ran)
            if outcome == "interrupted":
                return _end(ws, s, "failed", rec["error"], ran, rec["event"])
            if failures >= MAX_CONSECUTIVE_FAILURES:
                return _end(ws, s, "failed", f"{failures} trials in a row failed (the last: {rec['error']}); the search "
                                             f"stopped. Fix the cause and run it again", ran)
        return _end(ws, s, "finished", None, ran)


# ------------------------------------------------------------------ committing a trial (through Session.apply)
def _nonzero(deltas) -> bool:
    return any(not p.is_zero() for ps in (deltas or {}).values() for p in ps)


def _why_different(s, rec) -> str:
    """A7's explanation: what can make the edit made now differ from the one the trial scored."""
    then, now = rec.get("environment") or {}, environment()
    a, b = (then.get("libraries") or {}), now["libraries"]
    if a == b and then.get("torch_threads") == now["torch_threads"]:
        return ("The library versions and thread count are those it was scored with, so this is not expected; report "
                "it.")
    if s["method"] == "ara":
        return (f"An ARA fit is reproducible only from the same captures with the same torch version and thread count "
                f"(scored with torch {a.get('torch')} and {then.get('torch_threads')} threads; now torch {b['torch']} and "
                f"{now['torch_threads']} threads).")
    return (f"The closed form rounds differently with other library versions (scored with torch {a.get('torch')}; now "
            f"torch {b['torch']}).")


def apply_trial(sess, name, index, *, dry_run=False, message=None, enforce_scope=False,
                accept_different_edit=False) -> dict:
    """Commit a complete trial through Session.apply (scope, the largest-patch setting, edit_preflight, candidate
    tests and drift, ws.commit, patch recording): its script is run again, and the pieces it makes now must have the
    digest of the pieces the trial scored, else the commit is refused (A7) unless ``accept_different_edit``. The
    patch's source records ``origin``: the search, the trial, both digests and both environments."""
    ws = sess.ws
    s = load(ws, name)
    recs = trial_records(ws, name, s)
    if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(recs):
        raise SearchMissing(f"search '{name}' has no trial {index}")
    rec = recs[index - 1]
    if rec["state"] != "complete":
        raise SearchError(f"trial {index} of search '{name}' is {_shown(rec, _holder(ws, name))}; only a complete trial "
                          f"can be committed")
    busy = SearchBusy("a search is running on this workspace; stop it before committing a trial")
    job = getattr(sess, "search_job", None)
    if job and job.get("running") and job.get("workspace") == ws.name:
        raise busy
    held = contextlib.ExitStack()
    try:
        held.enter_context(search_lock(ws, name))     # held while the trial is committed: no search starts meanwhile
    except SearchBusy:
        raise busy from None
    with held:
        check_compat(ws, s, "apply")
        displays = [x["display"] for x in rec["scores"]]
        origin = {"kind": "direction_search", "search": name, "search_id": s["id"], "trial": index,
                  "method": s["method"], "parameters": rec["parameters"], "seed": s["seed"], "values": rec["values"],
                  "displays": displays, "scored_digest": rec["candidate"]["digest"], "digest": None,
                  "edit_equals_scored": None, "scored_environment": rec["environment"], "environment": environment(),
                  "measurement": s["identity"]["measurement"]}

        def check(deltas):
            d = piece_digest(deltas) if _nonzero(deltas) else None
            origin["digest"] = d
            origin["edit_equals_scored"] = d == origin["scored_digest"]
            if origin["edit_equals_scored"] or accept_different_edit:
                return None
            return (f"the edit made now differs from the one trial {index} scored (pieces {(d or 'none')[:12]} against "
                    f"{(origin['scored_digest'] or 'none')[:12]}). {_why_different(s, rec)} Its scores do not describe "
                    f"this edit: score it again (POST /api/directions/scores with the trial's script), or pass "
                    f"accept_different_edit")
        report = sess.apply(rec["script"], message or f"search {name}: trial {index} ({', '.join(displays)})",
                            author="human", dry_run=dry_run, enforce_scope=enforce_scope, origin=origin, check=check)
    return {**report, "search": name, "trial": index, "digest": origin["digest"],
            "scored_digest": origin["scored_digest"], "edit_equals_scored": origin["edit_equals_scored"]}


# ------------------------------------------------------------------ the routes' field checks
def run_body(body) -> dict:
    if not isinstance(body, dict):
        raise SearchError("the request must be an object")
    unknown = sorted(set(body) - set(RUN_FIELDS) - {"stream"}, key=str)
    if unknown:
        raise SearchError(f"{unknown[0]}: unknown field; the fields are n_additional_trials, accept_changed_libraries "
                          f"and stream")
    n = body.get("n_additional_trials")
    if n is not None:
        _check_additional(n)
    accept = body.get("accept_changed_libraries", False)
    if not isinstance(accept, bool):
        raise SearchError("accept_changed_libraries must be true or false")
    return {"n_additional_trials": n, "accept_changed_libraries": accept}


def stop_body(body) -> dict:
    if not isinstance(body, dict):
        raise SearchError("the request must be an object")
    unknown = sorted(set(body) - {"when"}, key=str)
    if unknown:
        raise SearchError(f"{unknown[0]}: unknown field; the fields are when")
    when = body.get("when", "now")
    if when not in ("now", "after_trial"):
        raise SearchError("when must be 'now' or 'after_trial'")
    return {"when": when}


def apply_body(body) -> dict:
    if not isinstance(body, dict):
        raise SearchError("the request must be an object")
    unknown = sorted(set(body) - set(APPLY_FIELDS), key=str)
    if unknown:
        raise SearchError(f"{unknown[0]}: unknown field; the fields are dry_run, message, enforce_scope and "
                          f"accept_different_edit")
    out = {}
    for key in ("dry_run", "enforce_scope", "accept_different_edit"):
        out[key] = body.get(key, False)
        if not isinstance(out[key], bool):
            raise SearchError(f"{key} must be true or false")
    message = body.get("message")
    if message is not None and (not isinstance(message, str) or len(message) > 1000):
        raise SearchError("message must be text of at most 1000 characters")
    out["message"] = message
    return out
