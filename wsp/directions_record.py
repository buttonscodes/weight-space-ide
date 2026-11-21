"""The reproduction record of a committed direction edit (plan item H3; docs/DIRECTIONS.md, "The reproduction record").

Heretic writes `reproduce.json` beside an uploaded model: its settings, the trial's parameters and scores, the
library versions, and the hashes of the weights. The IDE keeps the same knowledge for every committed edit whose
script read a measurement (`ablate_direction`, `ablate_ara`, `direction`, `direction_means`), in a file of its own:

    <workspace>/directions/records/<patch id>.json            the record (version 1, below)
    <workspace>/directions/records/prompts/<sha>.json          the measurement's prompts, by content
    <workspace>/directions/records/means/<sha>.safetensors     a byte copy of the measurement's means, by content
    <workspace>/directions/records/base.json                   the base files' SHA-256, cached by name, size, mtime

The interpreter lists the calls (`wscript.Interp.calls`); `Session.apply` calls `prepare` before the commit (the
weights the edit was made on, the base's identity, the libraries, the measurements' snapshots read and checked)
and `write` after it. A dry run, a refused edit or a script that read no measurement writes nothing, and a record
that cannot be written never undoes the commit (the report says `record_error`). The module inputs of an ARA
measurement are not copied (they can be gigabytes); their hash identifies them.

The record is valid for its patch while `record.patch.id` and `record.patch.created` are the patch's: an id can be
given again after a deletion, and the record of the old patch is then stale (the next commit with that id
overwrites it).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import time

import torch

from . import directions as DR
from . import directions_rendering as RD

VERSION = 1
RECORDS = ("directions", "records")
HASH_AT_COMMIT_MAX_BYTES = 8 * 2 ** 30   # above this the base is not hashed inside a commit (hashed by a re-run)
BLOCK = 1 << 20
ABLATE = ("ablate_direction", "ablate_ara")
READS = ("direction", "direction_means")
FIELDS = ("version", "patch", "created", "script", "script_sha256", "rebuildable", "calls", "pieces", "made_on", "base",
          "environment", "origin", "search", "heretic")
PACKAGES = ("torch", "transformers", "safetensors", "numpy", "tokenizers", "optuna", "pyarrow")
CODE_FILES = ("wsp/directions.py", "wsp/directions_ara.py", "wsp/wscript.py", "wsp/delta.py", "wsp/workspace.py")
APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODIFIER = {"ablation": "Abliteration", "ara": "ARA"}
# expert_modules "shared": what Heretic at 64e9991 does on such a model under transformers 5.x, by method (run on tiny
# random Qwen3.5-MoE, Qwen2-MoE and Qwen3-MoE models; docs/DIRECTIONS.md, "Mixture-of-experts models in Heretic")
EXPERT_MODULES_NOTE = {
    "ablation": ("expert_modules is an IDE field; Heretic at 64e9991 under transformers 5.x ablates only the shared expert "
                 "of such a model (it does not list the fused routed experts as modules)"),
    "ara": ("expert_modules is an IDE field; Heretic at 64e9991 under transformers 5.x lists no routed experts of such a "
            "model, and its ARA raises while capturing a shared expert's inputs (it reads them as [prompt, position, "
            "width]; they are [tokens, width]), so it fits such a model only when it has no shared expert"),
}
ABLATION_SETTINGS = ("orthogonalize_direction", "row_normalization", "full_normalization_lora_rank")
ARA_SETTINGS = ("preserve_row_magnitudes", "lora_rank", "n_optimization_steps", "learning_rate", "max_iter",
                "history_size")
MEASURED_KEYS = ("variant", "tokens", "options", "datasets", "module_io", "model", "environment", "rendering")
STREAM_ELEMS = 16_000_000          # values per row slab when a re-run's change is compared with the recorded one
# The largest relative difference per tensor that still counts as rounding, by row_normalization (directional
# ablation only; an ARA fit has no rounding bound). The strictest over the script's calls applies.
BOUNDS = {"none": 1e-5, "pre": 1e-5, "full": 1e-4}
REPRODUCE_FIELDS = ("measurement", "dry_run", "message", "enforce_scope", "accept_different")
NO_RECORD = ("patch {pid} has no reproduction record: it was not made by ablate_direction, ablate_ara, direction or "
             "direction_means through the edit path, or it was made before records existed")


class RecordError(ValueError):
    pass


class RecordMissing(RecordError):
    """No such patch, or no (valid) record for it: the routes answer 404."""


class RecordBusy(RecordError):
    """A search runs on the workspace: the routes answer 409."""


# ------------------------------------------------------------------ paths and small helpers
def _dir(ws, *parts) -> str:
    return os.path.join(ws.dir, *RECORDS, *parts)


def record_path(ws, pid) -> str:
    return _dir(ws, f"{pid}.json")


def sha256_file(path) -> str:
    """SHA-256 of a file in 1 MiB blocks (one block in memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def prompts_sha(prompts) -> str:
    """The content address of a prompts snapshot {"good": [...], "bad": [...]}."""
    text = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plain(value):
    return json.loads(json.dumps(value))


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


# ------------------------------------------------------------------ the base's identity (6.3; addendum B)
def _cache_rows(ws) -> list:
    try:
        with open(_dir(ws, "base.json")) as f:
            rows = json.load(f).get("files") or []
    except (OSError, ValueError, AttributeError):
        return []
    return [r for r in rows if isinstance(r, list) and len(r) == 4 and isinstance(r[0], str)]


def _write_cache(ws, new_rows) -> None:
    """Merge [name, bytes, mtime_ns, sha256] rows into base.json (a row per file name)."""
    if not new_rows:
        return
    rows = {r[0]: r for r in _cache_rows(ws)}
    for r in new_rows:
        rows[r[0]] = list(r)
    DR._write_json(_dir(ws, "base.json"), {"files": [rows[k] for k in sorted(rows)]})


def _cached(rows, name, size, mtime_ns):
    for r in rows:
        if r[:3] == [name, size, mtime_ns] and isinstance(r[3], str) and r[3]:
            return r[3]
    return None


def _local_bytes(ws) -> int:
    total = 0
    for f in getattr(ws.store, "files", None) or []:
        path = getattr(f, "path", None)
        if not getattr(f, "remote", False) and path and os.path.isfile(path):
            total += os.path.getsize(path)
    return total


def _base(ws, hash: bool) -> tuple[dict, list]:
    """-> (the base's identity, the cache rows hashed now). Nothing is written."""
    meta = getattr(ws, "meta", None) or {}
    out = {"kind": ws.kind, "name": ws.name, "repo": meta.get("repo") or None, "sha": meta.get("sha") or None,
           "base_sha256": meta.get("base_sha256") if ws.kind == "gguf" else None, "files": [], "hashed": True,
           "hash_seconds": 0.0}
    if ws.kind == "gguf":
        out["hashed"] = bool(out["base_sha256"])
        return out, []
    rows, new, spent = _cache_rows(ws), [], 0.0
    for f in getattr(ws.store, "files", None) or []:
        if getattr(f, "remote", False):
            size = f.data_start + max((m["data_offsets"][1] for m in f.tensors.values()), default=0)
            out["files"].append({"name": os.path.basename(str(f.location).split("?")[0]), "bytes": int(size),
                                 "mtime_ns": None, "sha256": None})
            continue
        path = getattr(f, "path", None)
        if not path or not os.path.isfile(path):
            continue
        st = os.stat(path)
        name = os.path.basename(path)
        sha = _cached(rows, name, st.st_size, st.st_mtime_ns)
        if sha is None and hash:
            t0 = time.monotonic()
            sha = sha256_file(path)
            spent += time.monotonic() - t0
            after = os.stat(path)
            if (after.st_size, after.st_mtime_ns) == (st.st_size, st.st_mtime_ns):
                new.append([name, st.st_size, st.st_mtime_ns, sha])
            else:                                   # written to while it was read: no hash for it
                sha = None
        out["files"].append({"name": name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha})
    out["hashed"] = all(e["sha256"] for e in out["files"] if e["mtime_ns"] is not None)
    out["hash_seconds"] = round(spent, 6)
    return out, new


def base_identity(ws, hash: bool = True) -> dict:
    """The base weights by content: a GGUF base by its base_sha256; every local weight file by its SHA-256 (cached in
    base.json by name, size and modification time; with ``hash`` a missing or stale entry is hashed and cached,
    without it a stale one is null); weights still on the Hub by repository and commit (sha256 null)."""
    out, new = _base(ws, hash)
    _write_cache(ws, new)
    return out


# ------------------------------------------------------------------ the environment
def environment(ws) -> dict:
    """The libraries and code an edit is computed with (no host, user or path): directions_search.environment, the
    packages' installed versions, the platform string, the Python implementation, the loaded runner and the
    SHA-256 of the modules that compute a direction edit."""
    from . import directions_search as DSR
    code = {}
    for rel in CODE_FILES:
        try:
            code[rel] = sha256_file(os.path.join(APP, *rel.split("/")))
        except OSError:
            code[rel] = None
    return {**DSR.environment(), "packages": {p: _version(p) for p in PACKAGES}, "platform": platform.platform(),
            "implementation": platform.python_implementation(), "runner": DR.runner_description(ws),
            "code_sha256": code}


# ------------------------------------------------------------------ prepare (before the commit) and write (after)
def _heretic(ws, call) -> tuple[dict | None, int | None]:
    """Heretic's view of an ablate call: its modifier, its trial parameters and its modifier settings."""
    from . import directions_search as DSR
    if call["function"] not in ABLATE:
        return None, None
    method, p = call["method"], call["parameters"]
    hp = DSR.heretic_parameters(method, p)
    hp.pop("expert_modules", None)                     # an IDE field: never one of Heretic's parameters
    if method == "ara":
        if hp.get("end_layer_index") is None:
            hp["end_layer_index"] = ws.struct.n_layers
        settings, seed = {k: p[k] for k in ARA_SETTINGS}, p["seed"]
    else:
        settings, seed = {k: p[k] for k in ABLATION_SETTINGS}, None
    out = {"modifier": MODIFIER[method], "parameters": hp, "settings": settings}
    if p.get("expert_modules") == "shared":
        out["note"] = EXPERT_MODULES_NOTE[method]
    return out, seed


def _measurement(ws, ident, snapshots) -> dict:
    """The call's measurement identity plus what its meta says about how it was made, with its prompts and means
    read now (checked against the hashes the call saw) for the snapshots."""
    name = ident["name"]
    meta = DR._meta(ws, DR.check_name(name))
    sha = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if sha != ident["meta_sha256"]:
        raise RecordError(f"measurement '{name}' changed while the edit ran")
    prompts = DR.prompts(ws, name)
    psha = prompts_sha(prompts)
    with open(DR._paths(ws, name)["tensors"], "rb") as f:
        means = f.read()
    if hashlib.sha256(means).hexdigest() != ident["means_sha256"]:
        raise RecordError(f"measurement '{name}' changed while the edit ran")
    snapshots["prompts"][psha] = prompts
    snapshots["means"][ident["means_sha256"]] = means
    return {**ident, **{k: meta.get(k) for k in MEASURED_KEYS}, "measured_on": meta.get("measured_on"),
            "prompts_snapshot": psha, "means_snapshot": ident["means_sha256"]}


def rebuildable(script: str, calls) -> bool:
    """Whether the script is exactly the ablate calls' own one-line scripts (so it can be rebuilt for another
    measurement)."""
    if not calls or any(c["function"] not in ABLATE for c in calls):
        return False
    try:
        lines = [DR.script_for({**c["parameters"]}) for c in calls]
    except DR.DirectionError:
        return False
    return script.strip() == "\n".join(lines)


def _search(ws, origin) -> dict | None:
    """What a committed search trial was found and scored with (search.json and the trial's record)."""
    if not isinstance(origin, dict) or origin.get("kind") != "direction_search":
        return None
    from . import directions_scores as DS
    from . import directions_search as DSR
    name = origin["search"]
    s = DSR.load(ws, name)
    recs = DSR.trial_records(ws, name, s)
    rec = recs[int(origin["trial"]) - 1]
    req = s["scoring"]["request"]
    datasets = s["scoring"].get("datasets") or []
    scorers = []
    for i, e in enumerate(req["scores"]):
        scorers.append({"plugin": e.get("plugin"), "instance_name": e.get("instance_name"),
                        "optimization": e.get("optimization"),
                        "settings": {k: v for k, v in e.items() if k not in DS.COMMON_FIELDS},
                        "prompts_sha256": DR._sha(e["prompts"]), "datasets": datasets[i] if i < len(datasets) else None})
    scores = []
    for i, x in enumerate(rec.get("scores") or []):
        plugin = x.get("plugin") or (req["scores"][i].get("plugin") if i < len(req["scores"]) else None)
        scores.append({"plugin": plugin, "name": x.get("name"), "value": x.get("value"), "display": x.get("display")})
    return {"name": name, "id": s["id"], "n_trials": s["n_trials"], "n_startup_trials": s["n_startup_trials"],
            "seed": s["seed"], "seed_from": s.get("seed_from"), "sampler": s.get("sampler"),
            "objectives": s.get("objectives"), "max_response_length": req.get("max_response_length"),
            "rendering": {k: req.get(k) for k in DSR.RENDERING}, "scorers": scorers,
            "finished_trials": sum(1 for r in recs if r["state"] in DSR.FINAL),
            "trial": {"index": rec["index"], "values": rec.get("values"),
                      "displays": [x.get("display") for x in rec.get("scores") or []], "scores": scores}}


def prepare(ws, script: str, calls, deltas, origin) -> dict:
    """Everything of the record that is known before the commit -> the pending record (nothing written). The base is
    hashed when its cache is stale and its local files hold at most HASH_AT_COMMIT_MAX_BYTES."""
    from . import directions_search as DSR
    from .atlas import wiring
    with ws.lock:
        made_on = _plain(wiring.patch_fingerprint(ws))
    snapshots = {"prompts": {}, "means": {}}
    out_calls = []
    for c in calls:
        c = copy.deepcopy(c)
        c["measurement"] = _measurement(ws, c["measurement"], snapshots)
        c["heretic"], c["seed"] = _heretic(ws, c)
        out_calls.append(c)
    kept = {k: [p for p in ps if not p.is_zero()] for k, ps in deltas.items()}
    kept = {k: ps for k, ps in kept.items() if ps}
    base, cache = _base(ws, hash=_local_bytes(ws) <= HASH_AT_COMMIT_MAX_BYTES)
    record = {"script": script, "script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
              "rebuildable": rebuildable(script, out_calls), "calls": out_calls,
              "pieces": {"digest": DSR.piece_digest(deltas), "stored": sum(p.numel() for ps in kept.values() for p in ps),
                         "tensors": sorted(kept)},
              "made_on": made_on, "base": base, "environment": environment(ws),
              "origin": _plain(origin) if origin is not None else None, "search": _plain(_search(ws, origin)),
              "heretic": DSR.HERETIC}
    return {"record": _plain(record), "snapshots": snapshots, "cache": cache}


def write(ws, pid: str, pending: dict) -> dict:
    """Write the snapshots (each once), the base hash cache, then the record of the committed patch ``pid``."""
    p = ws.patch(pid)
    for sha, prompts in pending["snapshots"]["prompts"].items():
        path = _dir(ws, "prompts", f"{sha}.json")
        if not os.path.isfile(path):
            DR._write_json(path, prompts)
    for sha, data in pending["snapshots"]["means"].items():
        path = _dir(ws, "means", f"{sha}.safetensors")
        if not os.path.isfile(path):
            def put(tmp, data=data):
                with open(tmp, "wb") as f:
                    f.write(data)
            DR._atomic(path, put)
    _write_cache(ws, pending["cache"])
    rec = {"version": VERSION, "patch": {"id": p.id, "created": p.created, "message": p.message, "author": p.author},
           "created": time.time(), **pending["record"]}
    rec = {k: rec[k] for k in FIELDS}
    DR._write_json(record_path(ws, pid), rec)
    return {"version": VERSION, "calls": len(rec["calls"]), "digest": rec["pieces"]["digest"]}


# ------------------------------------------------------------------ reading a record, and what changed since
def read(ws, patch) -> tuple[dict | None, bool]:
    """-> (the patch's record, stale): stale when the file is another patch's (an id given again)."""
    try:
        with open(record_path(ws, patch.id)) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return None, False
    meta = rec.get("patch") if isinstance(rec, dict) else None
    try:
        if isinstance(meta, dict) and meta.get("id") == patch.id and float(meta.get("created")) == float(patch.created):
            return rec, False
    except (TypeError, ValueError):
        pass
    return None, True


def _fill_base(ws, base) -> dict:
    """The recorded base with each null local sha256 filled from the cache when the file still has the recorded
    size and modification time and the cache holds a hash for exactly those; `sha256_from` says where each came from."""
    base = copy.deepcopy(base)
    rows = _cache_rows(ws)
    paths = {os.path.basename(f.path): f.path for f in getattr(ws.store, "files", None) or []
             if not getattr(f, "remote", False) and getattr(f, "path", None)}
    for e in base.get("files") or []:
        if e.get("sha256"):
            e["sha256_from"] = "commit"
            continue
        e["sha256_from"] = None
        path = paths.get(e["name"])
        if e.get("mtime_ns") is None or path is None or not os.path.isfile(path):
            continue
        st = os.stat(path)
        if [st.st_size, st.st_mtime_ns] == [e["bytes"], e["mtime_ns"]]:
            sha = _cached(rows, e["name"], e["bytes"], e["mtime_ns"])
            if sha:
                e["sha256"], e["sha256_from"] = sha, "later, file unchanged since"
    return base


def _base_view(b) -> dict:
    return {"kind": b.get("kind"), "base_sha256": b.get("base_sha256"), "repo": b.get("repo"), "sha": b.get("sha"),
            "files": [{"name": f["name"], "bytes": f["bytes"], "sha256": f.get("sha256")} for f in b.get("files") or []]}


def _base_row(ws, then, pid, hash) -> dict:
    now = base_identity(ws, hash=True) if hash else _base(ws, False)[0]
    row = {"key": "base", "then": _base_view(then), "now": _base_view(now), "same": True, "effect": "refuse",
           "message": None}
    differs = changed = None
    unknown = False
    if then.get("kind") != now["kind"]:
        differs = "kind"
    elif now["kind"] == "gguf":
        if then.get("base_sha256") and now["base_sha256"]:
            differs = None if then["base_sha256"] == now["base_sha256"] else "base_sha256"
        else:
            unknown = True
    else:
        tf, nf = {f["name"]: f for f in then.get("files") or []}, {f["name"]: f for f in now["files"]}
        if set(tf) != set(nf):
            differs = sorted(set(tf) ^ set(nf))[0]
        for name in sorted(tf) if differs is None else ():
            a, b = tf[name], nf[name]
            if a.get("mtime_ns") is None:                       # still on the Hub: the repository and commit
                if (then.get("repo"), then.get("sha"), a["bytes"]) != (now["repo"], now["sha"], b["bytes"]):
                    differs = name
            elif a["bytes"] != b["bytes"]:
                differs = name
            elif a.get("sha256") and b.get("sha256"):
                if a["sha256"] != b["sha256"]:
                    differs = name
            elif not a.get("sha256") and b.get("mtime_ns") != a.get("mtime_ns"):
                changed = changed or name
            else:
                unknown = True
            if differs:
                break
    if differs is not None:
        row.update(same=False, message=f"the base weights are not those patch {pid} was made on ({differs} differs)")
    elif changed is not None:
        row.update(same=False, message=f"the base was not hashed when the edit was committed and its file {changed} has "
                                       f"changed since (size or modification time): it cannot be shown to be the same")
    elif unknown and not hash:
        row.update(same=None, message="the base's content hash is not cached; a re-run hashes it")
    elif unknown or any(not f.get("sha256") for f in then.get("files") or [] if f.get("mtime_ns") is not None):
        gib = sum(f["bytes"] for f in then.get("files") or [] if f.get("mtime_ns") is not None) / 2 ** 30
        row["message"] = (f"the base was not hashed when the edit was committed ({gib:.2f} GiB); its files have the same "
                          f"size and modification time, and their hash now is recorded")
    return row


def _patch_ids(fp) -> str:
    return ", ".join(str(e[0]) for e in fp) if fp else "none"


def _made_on_row(ws, then, pid) -> dict:
    from .atlas import wiring
    now = _plain(wiring.patch_fingerprint(ws))
    same = _plain(then) == now
    msg = None if same else (f"the enabled patches are not those patch {pid} was made on (then: {_patch_ids(then)}; now: "
                             f"{_patch_ids(now)}). Switch patch {pid} off (and the others back as they were) to re-run it "
                             f"on the weights it was made on")
    return {"key": "made_on", "then": then, "now": now, "same": same, "effect": "refuse", "message": msg}


def _measurement_rows(ws, rec, pid) -> list:
    from . import directions_search as DSR
    rows, seen = [], set()
    for c in rec.get("calls") or []:
        m = c["measurement"]
        if m["name"] in seen:
            continue
        seen.add(m["name"])
        then = {k: m.get(k) for k in DSR.MEASUREMENT_KEYS}
        try:
            ident = DSR.measurement_identity(ws, m["name"])
            now = {k: ident[k] for k in DSR.MEASUREMENT_KEYS}
        except DR.DirectionError:
            now = None
        same = then == now
        msg = None if same else (f"measurement '{m['name']}' changed since patch {pid} was made (measured again or "
                                 f"deleted): measure it again from the record (GET /api/patches/{pid}/record?remeasure="
                                 f"true) and pass the new name as measurement")
        rows.append({"key": f"measurement:{m['name']}", "then": then, "now": now, "same": same, "effect": "refuse",
                     "message": msg})
    return rows


def checks(ws, rec, pid, hash=False) -> list:
    """The table CHECKS: the rows that refuse a re-run when they differ (the base, the patches below, each
    measurement). ``hash`` False (every GET) never reads a weight file."""
    return [_base_row(ws, rec["base"], pid, hash), _made_on_row(ws, rec["made_on"], pid), *_measurement_rows(ws, rec, pid)]


# Severity of a difference in the environment (Heretic's scheme, heretic/reproduce.py, extended).
SEVERITY = {"code_sha256": "critical", "torch": "high", "transformers": "high", "numpy": "medium",
            "safetensors": "medium", "tokenizers": "medium", "optuna": "medium", "runner": "medium", "python": "low",
            "implementation": "low", "platform": "low", "machine": "low", "system": "low", "pyarrow": "low"}


def _leaves(env) -> list[tuple[str, str, object]]:
    """[(row name, severity key, value)] of an environment, a name once (libraries before packages)."""
    env = env or {}
    out, seen = [], set()

    def add(name, key, value):
        if name not in seen:
            seen.add(name)
            out.append((name, key, value))
    for group in ("libraries", "sampler"):
        for k, v in (env.get(group) or {}).items():
            add(k, k, v)
    for k in ("python", "torch_threads", "machine", "system"):
        add(k, k, env.get(k))
    for k, v in (env.get("packages") or {}).items():
        add(k, k, v)
    for k in ("platform", "implementation", "runner"):
        add(k, k, env.get(k))
    for path, v in (env.get("code_sha256") or {}).items():
        add(path, "code_sha256", v)
    return out


def environment_rows(then, now, ara: bool) -> list:
    """The environment then and now, leaf by leaf, each with its severity; reported, never refused. ``same`` is None
    for the runner while the model is not loaded: what it will run on is not known yet, which is not a difference."""
    have = {name: v for name, _, v in _leaves(now)}
    rows = []
    for name, key, v in _leaves(then):
        severity = ("high" if ara else "low") if key == "torch_threads" else SEVERITY.get(key, "low")
        current = have.get(name)
        same = None if key == "runner" and current is None and v is not None else v == current
        rows.append({"name": name, "then": v, "now": current, "severity": severity, "same": same})
    return rows


# ------------------------------------------------------------------ GET /api/patches/{pid}/record
def suggested_name(name: str, pid: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]", "-", f"{name}-r{pid}")[:64]
    return s if DR.NAME.match(s) else ("m" + s)[:64]


MEASURE_DEFAULTS = {"chat": False, "system": None, "prefix": None, "chat_options": None, "winsorization_quantile": 1.0}


def remeasure(ws, rec, pid) -> tuple[list | None, list]:
    """-> (per distinct measurement {"measurement", "body"}: the measure request that makes it again from the
    record's prompts snapshot and recorded rendering, body null when the snapshot is missing; the reasons)."""
    if not rec.get("rebuildable"):
        return None, [f"patch {pid}'s script does more than ablate_direction and ablate_ara calls, so it cannot be "
                      f"rebuilt for another measurement"]
    out, reasons, seen = [], [], set()
    for c in rec["calls"]:
        m = c["measurement"]
        if m["name"] in seen:
            continue
        seen.add(m["name"])
        try:
            with open(_dir(ws, "prompts", f"{m['prompts_snapshot']}.json")) as f:
                snap = json.load(f)
        except (OSError, ValueError, TypeError):
            reasons.append(f"the prompts of measurement '{m['name']}' are not in the workspace (snapshot "
                           f"{str(m.get('prompts_snapshot'))[:12]} is missing)")
            out.append({"measurement": m["name"], "body": None})
            continue
        opts = m.get("options") or {}
        body = {"name": suggested_name(m["name"], pid), "good": snap["good"], "bad": snap["bad"],
                **{k: opts.get(k, d) for k, d in MEASURE_DEFAULTS.items() if k != "winsorization_quantile"},
                "variant": m.get("variant") or "base",
                "winsorization_quantile": opts.get("winsorization_quantile", MEASURE_DEFAULTS["winsorization_quantile"]),
                "module_io": m.get("io_sha256") is not None}
        out.append({"measurement": m["name"], "body": body})
    return out, reasons


def answer(ws, pid: str, *, remeasure_too: bool = False) -> dict:
    """GET /api/patches/{pid}/record (KeyError for no patch). Never hashes a weight file and writes nothing."""
    p = ws.patch(pid)
    rec, stale = read(ws, p)
    out = {"record": None, "stale": stale, "checks": None, "environment": None, "remeasure": None,
           "remeasure_unavailable": None, "heretic": None}      # heretic: the Heretic file's rules, with a record
    if rec is None:
        return out
    rec = {**rec, "base": _fill_base(ws, rec["base"])}
    ara = any(c.get("method") == "ara" for c in rec.get("calls") or [])
    out.update(record=rec, checks=checks(ws, rec, pid, hash=False),
               environment=environment_rows(rec.get("environment"), environment(ws), ara),
               heretic=heretic_status(ws, rec, pid))
    if remeasure_too:
        out["remeasure"], out["remeasure_unavailable"] = remeasure(ws, rec, pid)
    return out


# ------------------------------------------------------------------ POST /api/patches/{pid}/reproduce
def reproduce_body(body) -> dict:
    """The re-run's arguments of a request body; an unknown field is refused."""
    if not isinstance(body, dict):
        raise RecordError("the request must be an object")
    unknown = sorted(set(body) - set(REPRODUCE_FIELDS), key=str)
    if unknown:
        raise RecordError(f"{unknown[0]}: unknown field; the fields are measurement, dry_run, message, enforce_scope "
                          f"and accept_different")
    out = {"measurement": body.get("measurement"), "dry_run": body.get("dry_run", True),
           "enforce_scope": body.get("enforce_scope", False), "accept_different": body.get("accept_different", False),
           "message": body.get("message")}
    if out["measurement"] is not None and not (isinstance(out["measurement"], str) and DR.NAME.match(out["measurement"])):
        raise RecordError(f"measurement must be null or a measurement name matching {DR.NAME.pattern}")
    for key in ("dry_run", "enforce_scope", "accept_different"):
        if not isinstance(out[key], bool):
            raise RecordError(f"{key} must be true or false")
    m = out["message"]
    if m is not None and (not isinstance(m, str) or len(m) > 1000):
        raise RecordError("message must be null or text of at most 1000 characters")
    return out


def _tensor_difference(shape, old, new) -> dict:
    """||new - old||_F / ||old||_F and max |new - old| of one tensor's pieces (each at strength 1), built in row
    slabs of at most STREAM_ELEMS values with delta.add_to and summed in float64. ``relative`` is None where the
    ratio does not exist: only one side has pieces (``only_in`` names it, "recorded" or "rerun"), or the recorded
    change has norm 0 and the re-run's has not. Every value is finite, so the answer is JSON."""
    from . import delta as D
    from .store import norm_index, region_shape
    full = norm_index((), tuple(shape))
    rows = len(full[0]) if full else 1
    per_row = 1
    for r in full[1:]:
        per_row *= len(r)
    step = max(1, STREAM_ELEMS // max(1, per_row))
    dd = rr = 0.0
    worst = 0.0
    for a in range(0, rows, step):
        region = (range(a, min(rows, a + step)),) + tuple(full[1:]) if full else full
        o = torch.zeros(region_shape(region))
        n = torch.zeros(region_shape(region))
        for p in old:
            D.add_to(o, region, p, 1.0)
        for p in new:
            D.add_to(n, region, p, 1.0)
        e = (n - o).double()
        dd += float((e * e).sum())
        rr += float((o.double() ** 2).sum())
        if e.numel():
            worst = max(worst, float(e.abs().max()))
    only = None if bool(old) == bool(new) else ("recorded" if old else "rerun")
    if only is not None:
        rel = None
    elif not old:
        rel = 0.0
    else:
        rel = (dd ** 0.5) / (rr ** 0.5) if rr > 0 else (0.0 if dd == 0 else None)
    return {"relative": rel, "max_abs": worst, "only_in": only}


def differences(ws, recorded, rerun) -> list[dict]:
    """Per tensor in either set: the re-run's change against the recorded patch's pieces (both at strength 1)."""
    def kept(deltas):
        out = {k: [p for p in ps if not p.is_zero()] for k, ps in (deltas or {}).items()}
        return {k: ps for k, ps in out.items() if ps}
    a, b = kept(recorded), kept(rerun)
    return [{"tensor": name, **_tensor_difference(ws.shape(name), a.get(name, []), b.get(name, []))}
            for name in sorted(set(a) | set(b))]


def verdict(rec, digest, recorded, diffs) -> tuple[str, float | None]:
    """-> (identical | within_rounding | different, the rounding bound or None). Rounding is granted to directional
    ablation only: every call ablate_direction, every tensor within the strictest bound of its calls."""
    calls = rec.get("calls") or []
    bound = None
    if calls and all(c["function"] == "ablate_direction" for c in calls):
        bound = min(BOUNDS[c["parameters"]["row_normalization"]] for c in calls)
    if digest == recorded:
        return "identical", bound
    if bound is not None and diffs and all(d["relative"] is not None and d["relative"] <= bound for d in diffs):
        return "within_rounding", bound
    return "different", bound


def _largest(diffs) -> str:
    """The largest difference of a different edit, for its refusal: a tensor without a ratio comes first."""
    if not diffs:
        return "the pieces differ although neither edit changes a weight"
    worst = max(diffs, key=lambda x: math.inf if x["relative"] is None else x["relative"])
    if worst["only_in"] is not None:
        return f"{worst['tensor']} is changed by the {'recorded edit' if worst['only_in'] == 'recorded' else 're-run'} only"
    if worst["relative"] is None:
        return f"the recorded change to {worst['tensor']} is zero and the re-run's is not"
    return f"largest relative difference {worst['relative']:.3g} in {worst['tensor']}"


def _why_different(rec, now) -> str:
    then = rec.get("environment") or {}
    a, b = then.get("libraries") or {}, now.get("libraries") or {}
    if any(c.get("method") == "ara" for c in rec.get("calls") or []):
        return (f"an ARA fit is reproducible only from the same captures with the same torch version and thread count "
                f"(recorded: torch {a.get('torch')}, {then.get('torch_threads')} threads; now: torch {b.get('torch')}, "
                f"{now.get('torch_threads')} threads)")
    if a == b:
        return "the library versions are those it was made with, so this is not expected; report it"
    return (f"the closed form rounds differently with other library versions (recorded torch {a.get('torch')}; now "
            f"torch {b.get('torch')})")


def _distinct_measurements(rec) -> list[str]:
    out = []
    for c in rec.get("calls") or []:
        name = c["measurement"]["name"]
        if name not in out:
            out.append(name)
    return out


def _snapshot_means(ws, rec):
    """The recorded means snapshot of the record's (one) measurement -> (good, bad) or None when it is missing."""
    from safetensors.torch import load_file
    m = rec["calls"][0]["measurement"]
    path = _dir(ws, "means", f"{m.get('means_snapshot')}.safetensors")
    try:
        t = load_file(path)
        return t["good_mean"], t["bad_mean"]
    except Exception:
        return None


def _new_measurement_row(ws, rec, pid, name) -> dict:
    """Mode measurement: the new measurement must have been made from the record's prompts and rendering (prompts,
    options, variant, the means' shape, whether module inputs were saved, and for a current measurement the patches
    it was measured on)."""
    m = rec["calls"][0]["measurement"]
    meta, good, _ = DR.load(ws, name)
    snap = _snapshot_means(ws, rec)
    shape = list(snap[0].shape) if snap is not None else [ws.struct.n_layers + 1, ws.struct.d_model]
    then = {"prompts_sha256": m.get("prompts_sha256"), "options": m.get("options"), "variant": m.get("variant") or "base",
            "shape": shape, "module_io": m.get("io_sha256") is not None}
    now = {"prompts_sha256": meta.get("prompts_sha256"), "options": meta.get("options"), "variant": meta.get("variant"),
           "shape": list(good.shape), "module_io": bool(meta.get("module_io"))}
    if m.get("measured_on") is not None:
        then["measured_on"], now["measured_on"] = m.get("measured_on"), meta.get("measured_on")
    then["rendering"], now["rendering"] = m.get("rendering"), meta.get("rendering")     # None: version 1
    differs = next((k for k in then if k != "rendering" and _plain(then[k]) != _plain(now[k])), None)
    msg = None if differs is None else (f"measurement '{name}' was not made from patch {pid}'s prompts and rendering: "
                                        f"{differs} differs")
    if differs is None and not RD.compatible(then["rendering"], now["rendering"]):
        differs = "rendering"
        msg = (f"measurement '{name}' and patch {pid}'s measurement were rendered with different versions (patch {pid}'s: "
               f"{_rendering_version(then['rendering'])}, this one: {_rendering_version(now['rendering'])}; version 2 "
               f"renders prompts as Heretic does) and on this model the two give different token ids for these prompts, "
               f"so it would not reproduce the edit: re-run patch {pid} without measurement (from its recorded means)")
    return {"key": f"measurement:{name}", "then": then, "now": now, "same": differs is None, "effect": "refuse",
            "message": msg}


def _rendering_version(entry) -> int:
    return (entry or {}).get("version", 1)


def _means(ws, rec, name) -> tuple[dict | None, list]:
    """The new measurement's means against the recorded snapshot: equal, and per set the largest over indices of
    ||new - snapshot|| / ||snapshot||."""
    snap = _snapshot_means(ws, rec)
    if snap is None:
        return None, ["the recorded means are not in the workspace"]
    _, good, bad = DR.load(ws, name)
    rel = {}
    for label, new, old in (("good", good, snap[0]), ("bad", bad, snap[1])):
        worst = 0.0
        for k in range(old.shape[0]):
            d, n = float((new[k].double() - old[k].double()).norm()), float(old[k].double().norm())
            worst = max(worst, d / n if n > 0 else (0.0 if d == 0 else float("inf")))
        rel[label] = worst
    return {"equal": bool(torch.equal(good, snap[0]) and torch.equal(bad, snap[1])), "relative": rel}, []


def reproduce(sess, pid, *, measurement=None, dry_run=True, message=None, enforce_scope=False,
              accept_different=False) -> dict:
    """Re-run a committed direction edit from its record, through Session.apply only (docs/DIRECTIONS.md,
    "Re-running an edit"): on the weights it was made on (the base and the enabled patches as recorded, the patch
    itself switched off by the person) and the recorded measurement, or, with ``measurement``, a new measurement
    made from the record's prompts and rendering. A dry run (the default) commits nothing; a commit makes a new
    patch whose origin says it is a re-run. Nothing here writes a weight or switches a patch."""
    from . import directions_search as DSR
    ws = sess.ws
    try:
        patch = ws.patch(pid)
    except KeyError:
        raise RecordMissing(f"no patch '{pid}'") from None
    rec, _ = read(ws, patch)
    if rec is None:
        raise RecordMissing(NO_RECORD.format(pid=pid))
    busy = RecordBusy("a search is running on this workspace; stop it before re-running an edit")
    job = getattr(sess, "search_job", None)
    if (job and job.get("running") and job.get("workspace") == ws.name) or DSR.running(ws):
        raise busy
    names = _distinct_measurements(rec)
    if measurement is not None:
        if not rec.get("rebuildable"):
            raise RecordError(f"patch {pid}'s script does more than ablate_direction and ablate_ara calls, so it cannot "
                              f"be rebuilt for another measurement: re-run it without measurement")
        if len(names) != 1:
            raise RecordError(f"patch {pid} reads {len(names)} measurements; measurement mode takes one")
    held = contextlib.ExitStack()
    try:
        held.enter_context(DSR.search_lock(ws))           # no search starts while the edit is re-run
    except DSR.SearchBusy:
        raise busy from None
    with held:
        rows = [_base_row(ws, rec["base"], pid, hash=True), _made_on_row(ws, rec["made_on"], pid)]
        rows += _measurement_rows(ws, rec, pid) if measurement is None else [_new_measurement_row(ws, rec, pid, measurement)]
        for row in rows:
            if row["same"] is not True:
                raise RecordError(row["message"])
        means, notes = (None, []) if measurement is None else _means(ws, rec, measurement)
        if measurement is None:
            script = rec["script"]
        else:
            script = "\n".join(DR.script_for({**c["parameters"], "direction": measurement}) for c in rec["calls"])
        now = environment(ws)
        recorded = rec["pieces"]["digest"]
        origin = {"kind": "reproduction", "of": pid, "of_created": patch.created, "recorded_digest": recorded,
                  "digest": None, "verdict": None,
                  "measurement": measurement if measurement is not None else (names[0] if len(names) == 1 else None),
                  "environment": now}
        seen = {"differences": None, "bound": None}

        def check(deltas):
            d = DSR.piece_digest(deltas)
            diffs = None if d == recorded else differences(ws, patch.deltas, deltas)
            v, bound = verdict(rec, d, recorded, diffs)
            origin["digest"], origin["verdict"] = d, v
            seen.update(differences=diffs, bound=bound)
            if v != "different" or dry_run or accept_different:
                return None
            return (f"the re-run made a different edit ({_largest(diffs)}). {_why_different(rec, now)}. Pass "
                    f"accept_different to commit it anyway")
        report = sess.apply(script, message or f"re-run of patch {pid}", author="human", dry_run=dry_run,
                            enforce_scope=enforce_scope, origin=origin, check=check)
    ara = any(c.get("method") == "ara" for c in rec.get("calls") or [])
    return {**report, "verdict": origin["verdict"], "digest": origin["digest"], "recorded_digest": recorded,
            "differences": seen["differences"], "bound": seen["bound"],
            "environment": environment_rows(rec.get("environment"), now, ara), "means": means, "checks": rows,
            "notes": notes}


# ------------------------------------------------------------------ Heretic's reproduce.json (version 4)
HERETIC_VERSION = "2.2.0.dev0"                                     # heretic/pyproject.toml at HERETIC_COMMIT
HERETIC_COMMIT = "64e999152ff7918efc73576814709b6b6a71f15c"
HERETIC_URL = "https://github.com/p-e-w/heretic"
HERETIC_NOTE = ("written by the weight-space IDE from its reproduction record; Heretic reproduces it with heretic "
                "--reproduce; the IDE's own result is the patch with this digest")
HERETIC_DIFFERENCES = {
    "full": ("row_normalization full: Heretic stores a randomized rank-r approximation (svd_lowrank seeded with the "
             "seed), the IDE the best rank-r one"),
    "ara": "ARA: a fit is reproducible only with the same torch version and thread count, in either tool",
    "eager": "the IDE loads transformers models with eager attention, Heretic with the default",
    "rendering": ("the measurement was rendered as the IDE rendered prompts before it rendered them as Heretic does "
                  "(rendering version 1): where the tokenizer adds its own special tokens, the template prints the date "
                  "or has no system role, its token ids are not those Heretic generates from"),
}
SPEC_FIELDS = ("dataset", "commit", "config", "split", "column", "prefix", "suffix", "system_prompt")
LOCAL_KINDS = {"inline": "inline texts", "text_file": "a local file", "directory": "a directory",
               "saved_to_disk": "a directory"}
HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")
QUERY_FIELDS = ("adapter", "model", "model_commit")
NEEDS_MODEL = "no Hub repository is known for this checkpoint: pass model (and model_commit)"


def heretic_query(adapter=None, model=None, model_commit=None) -> dict:
    """The query of GET /api/patches/{pid}/record/heretic, checked. ``model`` must be a Hub repository id: a local
    path would put a path of this machine into the file."""
    out = {"adapter": adapter, "model": model, "model_commit": model_commit}
    for key, v in out.items():
        if v is not None and (not isinstance(v, str) or not 1 <= len(v) <= 256 or any(c < " " or c == "\x7f" for c in v)):
            raise RecordError(f"{key} must be text of 1..256 characters without control characters")
    if model is not None and not HUB_ID.match(model):
        raise RecordError("model must be a Hugging Face repository id (a local path would put a path of this machine "
                          "into the file)")
    return out


def _local_kinds(record) -> list[str]:
    """What of a prompt set's dataset record is not a pinned Hub dataset (texts typed in, local files)."""
    if not isinstance(record, dict) or not record.get("parts"):
        return ["inline texts"]
    out = []
    for part in record["parts"]:
        kind = part.get("kind")
        what = LOCAL_KINDS.get(kind) if kind != "hub" else (None if part.get("commit") else "a Hub dataset without a "
                                                                                            "commit")
        if kind != "hub" and what is None:
            what = f"a {kind} part"
        if what is not None and what not in out:
            out.append(what)
    return out


def _spec(record) -> dict | list:
    """A prompt set's Hub parts as Heretic's dataset specification (one dict for one part), each with the commit
    the IDE resolved."""
    specs = []
    for part in record["parts"]:
        s = part.get("specification") or {}
        specs.append({k: (part.get("commit") if k == "commit" else s.get(k, "" if k in ("prefix", "suffix") else None))
                      for k in SPEC_FIELDS})
    return specs[0] if len(specs) == 1 else specs


def _adapter_sha(ws, name, pid) -> str | None:
    """SHA-256 of the adapter export ``name``'s weights file when its provenance lists patch ``pid``, else None."""
    if not isinstance(name, str) or not DR.NAME.match(name):
        return None
    folder = os.path.join(ws.dir, "export", "adapters", name)
    try:
        with open(os.path.join(folder, "wsp_adapter.json")) as f:
            prov = json.load(f)
        if not any(isinstance(p, dict) and p.get("id") == pid for p in prov.get("patches") or []):
            return None
        return sha256_file(os.path.join(folder, "adapter_model.safetensors"))
    except (OSError, ValueError, AttributeError):
        return None


def heretic_reasons(ws, rec, pid, *, adapter=None, model=None) -> list[str]:
    """The rules HERETIC_RULES (H1 to H10): why this record cannot be written as a Heretic reproduce.json."""
    from . import directions_scores as DS
    out = []
    calls = rec.get("calls") or []
    if not (len(calls) == 1 and calls[0]["function"] in ABLATE and rec.get("rebuildable")):
        out.append("the script is not one ablate_direction or ablate_ara call; Heretic applies exactly one modifier")
    base = rec.get("base") or {}
    if base.get("kind") != "hf":
        out.append(f"Heretic runs transformers checkpoints only (this workspace is {base.get('kind')})")
    elif not (model or base.get("repo")):
        out.append(NEEDS_MODEL)
    if rec.get("made_on"):
        out.append(f"the edit was made on top of other patches ({_patch_ids(rec['made_on'])}); Heretic edits the base "
                   f"model")
    if calls:
        m = calls[0]["measurement"]
        opts = m.get("options") or {}
        if (m.get("variant") or "base") != "base":
            out.append("the measurement was made on the current weights; Heretic measures the base model")
        if not opts.get("chat"):
            out.append("the measurement did not use the chat template; Heretic always renders prompts as a chat")
        if opts.get("system") is None:
            out.append("the measurement used no system message; Heretic always sends one (system_prompt)")
        if opts.get("chat_options"):                  # a chat measurement records {} when no option is set
            out.append(f"the chat template options {json.dumps(opts['chat_options'], sort_keys=True)} have no Heretic "
                       f"setting")
        datasets = m.get("datasets") or {}
        for label in ("good", "bad"):
            kinds = _local_kinds(datasets.get(label))
            if kinds:
                out.append(f"the {label} prompts include {' and '.join(kinds)}; Heretic's settings name Hub datasets "
                           f"(a local path would put a path of this machine into the file)")
    for i, s in enumerate((rec.get("search") or {}).get("scorers") or [], 1):
        if s.get("plugin") not in DS.PLUGIN_PATHS.values():
            out.append(f"scorer {i} ({s.get('plugin')}) is not one of Heretic's built-in scorers")
        kinds = _local_kinds(s.get("datasets"))
        if kinds:
            out.append(f"the scorer {i} prompts include {' and '.join(kinds)}; Heretic's settings name Hub datasets "
                       f"(a local path would put a path of this machine into the file)")
    if adapter is not None and _adapter_sha(ws, adapter, pid) is None:
        out.append(f"no adapter named '{adapter}' contains patch {pid}")
    return out


def heretic_status(ws, rec, pid) -> dict:
    """The Heretic rules evaluated without a query (GET /api/patches/{pid}/record's ``heretic``). ``needs_model``:
    the only thing missing is the checkpoint's Hub repository, so the file can be had by naming it in the query."""
    reasons = heretic_reasons(ws, rec, pid)
    return {"runnable": not reasons, "reasons": reasons, "needs_model": reasons == [NEEDS_MODEL]}


def _timestamp(created) -> str:
    import datetime
    t = datetime.datetime.fromtimestamp(float(created), datetime.timezone.utc)
    return t.replace(microsecond=0, tzinfo=None).isoformat()


def to_heretic(ws, rec, pid, *, adapter=None, model=None, model_commit=None) -> dict:
    """The record as Heretic's version-4 reproduce.json (docs/DIRECTIONS.md, "The mapping to Heretic's
    reproduce.json"), without system information (Heretic's "basic"); RecordError with every reason when a rule
    fails. Nothing of this machine is written: no path, host or user."""
    from . import directions_scores as DS
    from . import directions_search as DSR
    reasons = heretic_reasons(ws, rec, pid, adapter=adapter, model=model)
    if reasons:
        raise RecordError(f"this record cannot be written as a Heretic reproduce.json: {'; '.join(reasons)}")
    call = rec["calls"][0]
    m, h, base = call["measurement"], call["heretic"], rec["base"]
    opts = m.get("options") or {}
    env = rec.get("environment") or {}
    search = rec.get("search")
    cls = h["modifier"]
    model_id = model or base.get("repo")
    commit = model_commit if model_commit is not None else (base.get("sha") if model_id == base.get("repo") else None)
    settings = {"model": model_id, "model_commit": commit}
    runner = (m.get("environment") or {}).get("runner") or {}
    dtype = runner.get("dtype")
    if isinstance(dtype, str) and dtype.startswith("torch."):
        settings["dtypes"] = [dtype[len("torch."):]]
    if runner.get("device") == "cpu":
        settings["device_map"] = "cpu"
    settings["batch_size"] = 1
    if search:
        settings["max_response_length"] = search.get("max_response_length")
    settings["response_prefix"] = opts.get("prefix") or ""
    paths = {v: k for k, v in DS.PLUGIN_PATHS.items()}
    if (search or {}).get("scorers"):
        # An edit made by hand has no scorers and the key is left out, so Heretic scores its reproduction with its
        # own (the reproducing person's) scorers: Heretic refuses an empty list (heretic/evaluator.py:56).
        settings["scorers"] = [{"plugin": paths[s["plugin"]], "optimization": s.get("optimization"),
                                "instance_name": s.get("instance_name")} for s in search["scorers"]]
    settings["modifiers"] = [{"plugin": DSR.MODIFIERS[cls]["path"], "instance_name": None}]
    if search:
        settings["n_trials"] = search.get("finished_trials", search.get("n_trials"))
        settings["n_startup_trials"] = search.get("n_startup_trials")
        settings["seed"] = search.get("seed")
    else:
        settings["n_trials"], settings["n_startup_trials"] = 1, 1
        settings["seed"] = call.get("seed") if call.get("seed") is not None else 0
    settings["export_strategy"] = "adapter" if adapter is not None else "merge"
    settings["system_prompt"] = opts.get("system")
    if search:
        settings["scorer"] = {}
        for s in search["scorers"]:
            key = s["plugin"] + (f"_{s['instance_name']}" if s.get("instance_name") else "")
            settings["scorer"][key] = {**(s.get("settings") or {}), "prompts": _spec(s["datasets"])}
    datasets = m.get("datasets") or {}
    table = {"good_prompts": _spec(datasets["good"]), "bad_prompts": _spec(datasets["bad"]), **h["settings"]}
    if cls == "Abliteration":
        table["winsorization_quantile"] = opts.get("winsorization_quantile", 1.0)
    else:
        table["print_loss"] = False
    settings["modifier"] = {cls: table}
    scores = [{"name": x.get("name"), "score": {"value": x.get("value"), "rich_display": x.get("display"),
                                                "md_display": x.get("display")}, "baseline": None}
              for x in ((search or {}).get("trial") or {}).get("scores") or []]
    hashes = {"adapter_model.safetensors": _adapter_sha(ws, adapter, pid)} if adapter is not None else {}
    differences = []
    if call["method"] == "ablation" and call["parameters"].get("row_normalization") == "full":
        differences.append(HERETIC_DIFFERENCES["full"])
    if call["method"] == "ara":
        differences.append(HERETIC_DIFFERENCES["ara"])
    if opts.get("chat") and _rendering_version(m.get("rendering")) == 1:
        differences.append(HERETIC_DIFFERENCES["rendering"])
    if dtype not in ("torch.bfloat16", "torch.float16"):
        # bfloat16 / float16 runs load the library's default attention, as Heretic does (wsp.runner); a measurement
        # without a recorded runner or dtype was made before those runs existed, on eager attention
        differences.append(HERETIC_DIFFERENCES["eager"])
    out = {"version": "4", "timestamp": _timestamp(rec["created"]),
           "environment": {"heretic": {"version": HERETIC_VERSION, "is_standard_pypi": False,
                                       "metadata": {"type": "git", "url": HERETIC_URL, "commit_hash": HERETIC_COMMIT,
                                                    "requested_revision": None}},
                           "pytorch_version": (env.get("libraries") or {}).get("torch"),
                           "requirements": {k: v for k, v in (env.get("packages") or {}).items()
                                            if v is not None and k != "torch"}},
           "settings": settings, "parameters": h["parameters"], "scores": scores, "hashes": hashes,
           "wsp": {"record_version": rec.get("version"), "patch": pid, "digest": rec["pieces"]["digest"],
                   "note": HERETIC_NOTE, "differences": differences}}
    text = json.dumps(out)
    for path in (ws.dir, os.path.dirname(ws.dir), os.path.expanduser("~")):
        if path and len(path) > 1 and path in text:             # a last guard: the rules above should make it moot
            raise RecordError("this record cannot be written as a Heretic reproduce.json: it would contain a path of "
                              "this machine")
    return out
