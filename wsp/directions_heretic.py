"""Running a Heretic reproduce.json natively (plan item H3b; docs/DIRECTIONS.md, "Running a Heretic reproduce.json").

Heretic writes reproduce.json (format version 4) beside a model it uploads, and `heretic --reproduce <file>` makes
that model again: it takes the file's settings, measures with the file's prompt specifications, applies the file's
modifier with the chosen trial's parameters (no search) and saves. The IDE does the same with its own code: the file
is translated into a measure request (POST /api/directions/measure, the Hub dataset specifications resolved offline
from the cache by wsp.directions_datasets) and into the one-line weight script of the file's modifier
(DR.script_for), and the script is applied through Session.apply like every other edit, so the patch is scoped,
tested, recorded and has its own reproduction record (wsp.directions_record).

What cannot be honoured is refused before anything runs, one reason per item (RecordError, every reason joined).
Library and system differences are reported with Heretic's severities (heretic/reproduce.py at 64e9991), never
refused. The file's scores and hashes are reported as they are: a reproduction scores nothing and writes no
Heretic file.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
import json
import platform

import torch

from . import directions as DR
from . import directions_record as REC
from .directions_record import RecordBusy, RecordError

FILE_VERSION = "4"
BODY_FIELDS = ("reproduce", "measurement", "check", "dry_run", "message", "enforce_scope")
HERETIC_SYSTEM_PROMPT = "You are a helpful assistant."          # heretic/config.py Settings.system_prompt default
# The prompt sets a modifier table without them gets (heretic/modifiers/abliteration.py and ara.py at 64e9991).
DEFAULT_PROMPTS = {"good_prompts": {"dataset": "mlabonne/harmless_alpaca", "split": "train[:400]", "column": "text"},
                   "bad_prompts": {"dataset": "mlabonne/harmful_behaviors", "split": "train[:400]", "column": "text"}}
TOP_KEYS = ("version", "timestamp", "system", "environment", "settings", "parameters", "scores", "hashes")
# settings.<key>: what the IDE does with it. Honoured: it decides the measurement or the edit. Reported: compared with
# the IDE's run and listed, never refused. The rest is not used, for the reason given.
HONOURED = ("model", "model_commit", "quantization", "response_prefix", "system_prompt", "modifiers", "modifier", "seed")
REPORTED = ("dtypes", "device_map", "batch_size")
NOT_USED = {
    "scorers": "scoring: a reproduction applies the file's trial and scores nothing that changes the patch",
    "scorer": "scoring: a reproduction applies the file's trial and scores nothing that changes the patch",
    "max_response_length": "scoring: only generated responses are bounded by it",
    "n_trials": "Heretic's search: a reproduction applies the file's one trial",
    "n_startup_trials": "Heretic's search: a reproduction applies the file's one trial",
    "trial_index": "Heretic's trial menu: a reproduction applies the file's one trial",
    "n_additional_trials": "Heretic's search: a reproduction applies the file's one trial",
    "checkpoint_action": "Heretic's study journal: a reproduction opens none",
    "export_strategy": "what Heretic does with the model afterwards (the IDE's result is a patch)",
    "model_action": "what Heretic does with the model afterwards (the IDE's result is a patch)",
    "max_shard_size": "what Heretic does with the model afterwards (the IDE's result is a patch)",
    "modified_model_type": "Heretic's model card",
    "upload_repo_private": "Heretic's upload",
    "upload_reproducibility_information": "Heretic's upload",
    "upload_tags": "Heretic's upload",
    "max_memory": "where Heretic keeps tensors while it runs; it does not change the values",
    "offload_outputs_to_cpu": "where Heretic keeps tensors while it runs; it does not change the values",
    "ignore_mismatches": "Heretic's answer to its own environment check; the IDE reports differences and never stops on "
                         "them",
}
SEVERITIES = ("low", "medium", "high", "critical")
SEED_NOTE = "Heretic's directional ablation uses the seed only for row_normalization full"
HASHES_NOTE = ("Heretic compares these with the weight files it saves; the IDE makes a patch, not Heretic's files, so "
               "they are not compared")
SEARCH_NOTE = ("the file describes one trial (its parameters and scores) of a search of {n} trials (seed {seed}); the "
               "search's other trials are in Heretic's study journal beside the file, not in reproduce.json, and "
               "nothing of the search is run again")
NO_SYSTEM_NOTE = ("the file has no system information (Heretic's \"basic\"): the Python version, operating system, CPU "
                  "and accelerator are not compared")
GGUF_NOTE = ("this workspace's base is a quantized GGUF file, not the transformers checkpoint the file names: the "
             "measurement and the edit run on its weights")


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def file_sha256(data) -> str:
    """The content address of a reproduce.json's content (compact, sorted JSON)."""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_name(data) -> str:
    return f"heretic-{file_sha256(data)[:12]}"


# ------------------------------------------------------------------ the request body
def heretic_reproduce_body(body) -> dict:
    """POST /api/directions/reproduce's arguments; an unknown field is refused."""
    if not isinstance(body, dict):
        raise RecordError("the request must be an object")
    unknown = sorted(set(body) - set(BODY_FIELDS), key=str)
    if unknown:
        raise RecordError(f"{unknown[0]}: unknown field; the fields are reproduce, measurement, check, dry_run, message "
                          f"and enforce_scope")
    if "reproduce" not in body:
        raise RecordError("reproduce: required (the content of a Heretic reproduce.json)")
    out = {"data": body["reproduce"], "measurement": body.get("measurement"), "check": body.get("check", False),
           "dry_run": body.get("dry_run", True), "message": body.get("message"),
           "enforce_scope": body.get("enforce_scope", False)}
    if not isinstance(out["data"], dict):
        raise RecordError("reproduce must be the content of a Heretic reproduce.json (an object)")
    m = out["measurement"]
    if m is not None and not (isinstance(m, str) and DR.NAME.match(m)):
        raise RecordError(f"measurement must be null or a measurement name matching {DR.NAME.pattern}")
    for key in ("check", "dry_run", "enforce_scope"):
        if not isinstance(out[key], bool):
            raise RecordError(f"{key} must be true or false")
    msg = out["message"]
    if msg is not None and (not isinstance(msg, str) or len(msg) > 1000):
        raise RecordError("message must be null or text of at most 1000 characters")
    return out


# ------------------------------------------------------------------ the model
def _model(ws, settings, reasons, notes) -> dict:
    """The file's model against the workspace's base: repository and commit where the workspace knows them,
    "not verified" where it does not (nothing is guessed from a path)."""
    meta = getattr(ws, "meta", None) or {}
    model, commit = settings.get("model"), settings.get("model_commit")
    repo, sha = meta.get("repo") or None, meta.get("sha") or None
    out = {"file": {"model": model, "model_commit": commit}, "workspace": {"kind": ws.kind, "repo": repo, "sha": sha},
           "repository": "not verified", "commit": "not verified", "notes": []}
    if not isinstance(model, str) or not model:
        reasons.append("settings.model: missing (the file names no model)")
        return out
    if ws.kind == "gguf":
        notes.append(GGUF_NOTE)
    if not REC.HUB_ID.match(model):
        out["notes"].append("the file's model is not a Hugging Face repository id, so it cannot be compared")
    elif repo is None:
        out["notes"].append("the workspace does not know its Hugging Face repository (it was opened from a local "
                            "directory), so the base is not verified to be the file's model")
    elif repo != model:
        reasons.append(f"the file's model is {model}; this workspace's base is {repo}")
        out["repository"] = "different"
        return out
    else:
        out["repository"] = "same"
    if out["repository"] != "same":
        return out
    if not commit:
        out["notes"].append("the file pins no model commit")
    elif sha is None:
        out["notes"].append("the workspace does not know its commit, so the base is not verified to be the file's commit")
    elif sha != commit:
        reasons.append(f"the file's model commit is {commit}; this workspace's base is {repo} at {sha}")
        out["commit"] = "different"
    else:
        out["commit"] = "same"
    return out


# ------------------------------------------------------------------ the modifier and its table
def _modifier(settings, reasons) -> dict | None:
    from . import directions_search as DSR
    mods = settings.get("modifiers")
    if not isinstance(mods, list) or len(mods) != 1 or not isinstance(mods[0], dict):
        n = len(mods) if isinstance(mods, list) else 0
        reasons.append(f"settings.modifiers lists {n} modifiers; Heretic applies exactly one")
        return None
    plugin, instance = mods[0].get("plugin"), mods[0].get("instance_name") or None
    paths = {row["path"]: cls for cls, row in DSR.MODIFIERS.items()}
    if plugin not in paths:
        reasons.append(f"the modifier {plugin} is not one the IDE has (heretic.modifiers.abliteration.Abliteration, "
                       f"heretic.modifiers.ara.ARA); it makes the edit, so the file cannot be reproduced without it")
        return None
    cls = paths[plugin]
    row = DSR.MODIFIERS[cls]
    tables = settings.get("modifier") if isinstance(settings.get("modifier"), dict) else {}
    keys = [cls] + ([f"{cls}_{instance}"] if instance else [])
    defaults = {**copy.deepcopy(DEFAULT_PROMPTS), **row["settings"]}
    if cls == "Abliteration":
        defaults["winsorization_quantile"] = 1.0
    table, ignored = dict(defaults), []
    for key in keys:                     # Heretic's get_settings_raw: the class table, then the instance's over it
        t = tables.get(key)
        if t is None:
            continue
        if not isinstance(t, dict):
            reasons.append(f"settings.modifier.{key} must be a table")
            continue
        for k, v in t.items():
            if k in defaults:
                table[k] = v
            else:
                ignored.append(f"settings.modifier.{key}.{k}")
    return {"class": cls, "method": row["method"], "plugin": plugin, "instance_name": instance, "table": table,
            "defaulted": sorted(k for k in defaults if not any(isinstance(tables.get(x), dict) and k in tables[x]
                                                                for x in keys)),
            "ignored": ignored}


def _ide_parameters(mod, params, name, seed, expert_modules) -> dict:
    """The file's modifier, its table and the trial's parameters as DR.script_for's input."""
    t = mod["table"]
    if mod["method"] == "ablation":
        p = {"direction": name, **{k: params.get(k) for k in ("direction_index", "weight_distributions")},
             **{k: t[k] for k in ("orthogonalize_direction", "row_normalization", "full_normalization_lora_rank")}}
        extra = sorted(set(params) - {"direction_index", "weight_distributions"})
    else:
        from . import directions_search as DSR
        p = {"method": "ara", "direction": name, **{k: params.get(k) for k in DSR.ARA_DRAWN},
             **{k: t[k] for k in DSR.ARA_SETTINGS}, "seed": seed}
        extra = sorted(set(params) - set(DSR.ARA_DRAWN))
    if extra:
        raise DR.DirectionError(f"{extra[0]}: not a parameter of {mod['class']}")
    if expert_modules != "all":
        p["expert_modules"] = expert_modules
    return p


# ------------------------------------------------------------------ the environment (heretic/reproduce.py)
def package_severity(name) -> str:
    """heretic/reproduce.py get_package_mismatch_severity at 64e9991."""
    if name == "heretic-llm":
        return "critical"
    if name in ("torch", "transformers"):
        return "high"
    if name in ("accelerate", "bitsandbytes", "kernels", "optuna", "peft", "tokenizers", "triton"):
        return "medium"
    return "low"


def heretic_version_text(info) -> str | None:
    """heretic/reproduce.py format_version_information, without the random suffix it gives a local or unknown
    installation (which makes two of them always differ): here they never equal the IDE's either."""
    if not isinstance(info, dict):
        return None
    meta = info.get("metadata") if isinstance(info.get("metadata"), dict) else {}
    v = info.get("version")
    kind = meta.get("type")
    if kind == "pypi":
        return f"{v}"
    if kind == "git":
        return f"{v}-git+{meta.get('url')}@{meta.get('commit_hash')}"
    return f"{v}-{'local' if kind == 'local' else 'unknown'}"


IDE_HERETIC = f"{REC.HERETIC_VERSION}-git+{REC.HERETIC_URL}@{REC.HERETIC_COMMIT}"


def _accelerator() -> str | None:
    """Heretic's accelerator type, for the devices this torch can see: CUDA or ROCm, MPS, else none (XPU, MLU, SDAA and
    the other types Heretic knows are not detected here)."""
    if torch.cuda.is_available():
        return "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "MPS"
    return None


def environment(data, notes) -> dict:
    """The file's environment against the IDE's, row by row as heretic/reproduce.py check_environment compares
    them (the file's packages and torch, Heretic itself, and with system information the Python version, operating
    system, CPU and accelerator), each graded with Heretic's severity. Only the rows that differ are listed."""
    rows = []
    env = data.get("environment") if isinstance(data.get("environment"), dict) else None
    system = data.get("system") if isinstance(data.get("system"), dict) else None

    def row(name, then, now, severity, same=None):
        rows.append({"name": name, "file": then, "ide": now, "severity": severity,
                     "same": (then == now) if same is None else same})
    if system is not None:
        s = system
        row("Python version", ((s.get("python") or {}).get("version")), platform.python_version(), "low")
        row("Operating system", ((s.get("os") or {}).get("platform")), platform.platform(), "low")
        rows.append({"name": "CPU", "file": (s.get("cpu") or {}).get("brand"), "ide": None, "severity": "low",
                     "same": None, "note": "the IDE does not read the CPU's brand"})
        row("Accelerator type", ((s.get("accelerators") or {}).get("type")), _accelerator(), "high")
    else:
        notes.append(NO_SYSTEM_NOTE)
    if env is None:
        notes.append("the file has no environment section: the library versions are not compared")
    else:
        req = env.get("requirements") if isinstance(env.get("requirements"), dict) else {}
        file_pk = {**req, "torch": env.get("pytorch_version")}
        for name in sorted(file_pk):
            now = str(torch.__version__) if name == "torch" else _version(name)
            row(name, file_pk[name], now, package_severity(name))
        row("heretic-llm", heretic_version_text(env.get("heretic")), IDE_HERETIC, "critical")
    differ = [r for r in rows if r["same"] is not True]
    worst = max((SEVERITIES.index(r["severity"]) for r in differ if r["same"] is False), default=None)
    differ.sort(key=lambda r: (-SEVERITIES.index(r["severity"]), r["name"]))
    return {"severity": None if worst is None else SEVERITIES[worst], "differences": differ,
            "same": sum(1 for r in rows if r["same"] is True)}


def _run_dtype(ws) -> tuple[str | None, str | None, str]:
    """(dtype, device, "runner" | "configured"): the loaded runner's, else what a runner would load with."""
    import os
    r = DR.runner_description(ws)
    if r is not None:
        dtype = r.get("dtype")
        return (dtype[len("torch."):] if isinstance(dtype, str) and dtype.startswith("torch.") else dtype,
                (r.get("device") or "").split(":")[0] or None, "runner")
    if ws.kind == "hf":
        return os.environ.get("WSP_RUN_DTYPE", "float32"), torch.device(os.environ.get("WSP_DEVICE", "cpu")).type, \
            "configured"
    return "float32", "cpu", "configured"


def settings_rows(ws, settings, method) -> dict:
    """What the IDE does with each of the file's settings: honoured, reported (compared with the IDE's run) or not
    used, with the reason. The seed is honoured for ARA only: an ablation of the IDE uses none."""
    dtype, device, source = _run_dtype(ws)
    reported = []
    dtypes = settings.get("dtypes")
    first = dtypes[0] if isinstance(dtypes, list) and dtypes else None
    reported.append({"field": "dtypes", "file": dtypes, "ide": dtype, "ide_from": source,
                     "same": None if first in (None, "auto") else first == dtype,
                     "note": "Heretic loads with the first of these that works (\"auto\": the checkpoint's own dtype); "
                             "the IDE runs with WSP_RUN_DTYPE"})
    dm = settings.get("device_map")
    reported.append({"field": "device_map", "file": dm, "ide": device, "ide_from": source,
                     "same": (dm == device) if dm in ("cpu", "mps", "cuda") else None,
                     "note": "the device the model runs on; float32 products differ between devices"})
    bs = settings.get("batch_size")
    reported.append({"field": "batch_size", "file": bs, "ide": 1, "ide_from": "the IDE measures one prompt at a time",
                     "same": bs == 1, "note": "prompts batched with padding round differently from one at a time"})
    honoured = [k for k in HONOURED if k in settings and (k != "seed" or method == "ara")]
    not_used = [{"field": k, "why": NOT_USED[k]} for k in settings if k in NOT_USED]
    if method != "ara" and "seed" in settings:
        not_used.append({"field": "seed", "why": f"{SEED_NOTE}, and the IDE computes the best rank-r one, which needs "
                                                 f"no seed"})
    other = [k for k in settings if k not in HONOURED and k not in REPORTED and k not in NOT_USED]
    not_used += [{"field": k, "why": "not one of Heretic's settings at 64e9991"} for k in other]
    return {"honoured": honoured, "reported": reported, "not_used": not_used}


# ------------------------------------------------------------------ the measurement: new, reused, or refused
def _existing(ws, name, checked, module_io) -> tuple[str, str | None]:
    """("new", None) | ("reuse", None) | ("refuse", reason): a measurement of that name is reused only when it was
    made from exactly this request (prompts, rendering, variant, module inputs for ARA) on this base."""
    from .steering import model_identity
    if not DR.exists(ws, name):
        return "new", None
    meta = DR._meta(ws, DR.check_name(name))
    want = {"prompts_sha256": {"good": DR._sha(checked["good"]), "bad": DR._sha(checked["bad"])},
            "options": {"chat": True, "system": checked["system"], "prefix": checked["prefix"],
                        "chat_options": checked["chat_options"], "winsorization_quantile": checked["winsorization_quantile"]},
            "variant": "base", "model": model_identity(ws)}
    got = {k: meta.get(k) for k in want}
    differs = next((k for k in want if json.loads(json.dumps(want[k])) != json.loads(json.dumps(got[k]))), None)
    if differs is None and module_io and not meta.get("module_io"):
        differs = "module_io"
    if differs is not None:
        return "refuse", (f"a measurement named '{name}' exists and was not made from this file's prompts and settings "
                          f"({differs} differs): pass another measurement name")
    return "reuse", None


# ------------------------------------------------------------------ the check (no model, nothing written)
def check_heretic_reproduce(ws, data, *, measurement=None) -> dict:
    """Everything about running a Heretic reproduce.json that needs no model: the file's version, the model against
    the workspace's base, the enabled patches, the quantization, the modifier and its table, the response prefix,
    the trial's parameters (checked by the IDE's own normalizer), the prompt sets resolved offline (each on its own),
    the measurement name -> the plan (what will be measured and run, what is reported); RecordError with every reason
    otherwise. Reads no weight and writes nothing."""
    from .atlas import wiring
    if not isinstance(data, dict):
        raise RecordError("reproduce must be the content of a Heretic reproduce.json (an object)")
    reasons, notes = [], []
    version = data.get("version")
    if version != FILE_VERSION:
        reasons.append(f"file format version {json.dumps(version)}: the IDE reads version 4 (plugin-based) reproduce.json "
                       f"files, as Heretic at 64e9991 does (older files were written before Heretic's plugins)")
    settings = data.get("settings")
    if not isinstance(settings, dict):
        reasons.append("settings: missing (a reproduce.json holds Heretic's settings)")
        settings = {}
    params = data.get("parameters")
    if not isinstance(params, dict):
        reasons.append("parameters: missing (a reproduce.json holds the chosen trial's parameters)")
    model = _model(ws, settings, reasons, notes)
    on = wiring.patch_fingerprint(ws)
    if on:
        reasons.append(f"patches {', '.join(str(e[0]) for e in on)} are switched on; Heretic edits the base model: switch "
                       f"them off to reproduce the file")
    quant = settings.get("quantization", "none")
    if quant not in (None, "none"):
        reasons.append(f"settings.quantization is {json.dumps(quant)}: the IDE does not load quantized transformers "
                       f"models, and the measurement and the edit would be of other weights")
    mod = _modifier(settings, reasons)
    prefix = settings.get("response_prefix")
    if prefix is None:
        reasons.append("settings.response_prefix is not set: Heretic would detect it by generating responses; detect it "
                       "with POST /api/directions/prefix and put it into the file")
    elif not isinstance(prefix, str):
        reasons.append("settings.response_prefix must be text")
    system = settings.get("system_prompt", HERETIC_SYSTEM_PROMPT)
    if not isinstance(system, str):
        reasons.append("settings.system_prompt must be text")
    seed = settings.get("seed")
    name = measurement if measurement is not None else default_name(data)
    script = ide = None
    expert_modules = "all"
    if mod is not None and isinstance(params, dict):
        if mod["method"] == "ara" and (isinstance(seed, bool) or not isinstance(seed, int)):
            reasons.append("settings.seed must be an integer: ARA's fit is seeded with it")
        else:
            env = data.get("environment") if isinstance(data.get("environment"), dict) else {}
            tf = (env.get("requirements") if isinstance(env.get("requirements"), dict) else {}).get("transformers")
            if ws.struct.recognised and DR.has_routed_experts(ws):
                # under transformers 5.x Heretic's module discovery skips the fused routed experts (EXPERT_MODULES_NOTE)
                head = str(tf).split(".")[0]
                if not head.isdigit() or int(head) >= 5:
                    expert_modules = "shared"
                    notes.append(REC.EXPERT_MODULES_NOTE[mod["method"]] + "; the reproduction passes expert_modules=\"shared\" "
                                 f"(the file's transformers is {tf})")
            try:
                ide = _ide_parameters(mod, params, name, seed, expert_modules)
                script = DR.script_for(ide)
            except DR.DirectionError as e:
                reasons.append(f"parameters: {e}")
    measure = checked = None
    state = None
    if mod is not None:
        t = mod["table"]
        module_io = mod["method"] == "ara"
        q = t.get("winsorization_quantile", 1.0) if mod["method"] == "ablation" else 1.0
        good, bad = t["good_prompts"], t["bad_prompts"]
        resolved = True
        for label, value, key in (("good", good, "good_prompts"), ("bad", bad, "bad_prompts")):
            field = f"settings.modifier.{mod['class']}.{key}"
            try:
                DR._prompt_set(ws, DR._prompt_value(value, label, field), label, field,
                               system=system if isinstance(system, str) else None, chat=True)
            except DR.DirectionError as e:
                reasons.append(str(e))
                resolved = False
        if resolved and isinstance(system, str) and isinstance(prefix, str):
            measure = {"name": name, "good": good, "bad": bad, "chat": True, "system": system, "prefix": prefix or None,
                       "variant": "base", "winsorization_quantile": q, "module_io": module_io}
            try:
                checked = DR.check_measure(ws, **{k: v for k, v in measure.items() if k != "name"}, name=name,
                                           replace=True)
            except DR.DirectionError as e:
                reasons.append(str(e))
            else:
                state, why = _existing(ws, name, checked, module_io)
                if why:
                    reasons.append(why)
    if reasons:
        raise RecordError(f"this reproduce.json cannot be run in the IDE: {'; '.join(reasons)}")
    env = environment(data, notes)
    # statements: what the file holds that the IDE does not use, and the differences by design
    for s in settings.get("scorers") or []:
        from . import directions_scores as DS
        plugin = s.get("plugin") if isinstance(s, dict) else s
        if plugin not in DS.PLUGIN_PATHS:
            notes.append(f"the scorer {plugin} is not one the IDE has; a reproduction scores nothing, so the patch is the "
                         f"same without it")
    n = settings.get("n_trials")
    if isinstance(n, int) and n > 1:
        notes.append(SEARCH_NOTE.format(n=n, seed=seed))
    if mod["method"] == "ablation" and mod["table"].get("row_normalization") == "full":
        notes.append(REC.HERETIC_DIFFERENCES["full"])
    elif mod["method"] == "ablation":
        notes.append(f"settings.seed is not used: {SEED_NOTE}")
    if mod["method"] == "ara":
        notes.append(REC.HERETIC_DIFFERENCES["ara"])
    dtype = _run_dtype(ws)[0]
    if ws.kind == "hf" and dtype not in ("bfloat16", "float16"):
        notes.append(REC.HERETIC_DIFFERENCES["eager"])
    for key in mod["ignored"]:
        notes.append(f"{key} is not a setting of {mod['class']}; Heretic ignores it too")
    if mod["defaulted"]:
        notes.append(f"settings.modifier.{mod['class']} does not set {', '.join(mod['defaulted'])}: Heretic's defaults "
                     f"are used, as Heretic uses them")
    for key in data:
        if key not in TOP_KEYS and key != "wsp":
            notes.append(f"{key} is not part of Heretic's reproduce.json; not used")
    wsp = data.get("wsp") if isinstance(data.get("wsp"), dict) else None
    hashes = data.get("hashes") if isinstance(data.get("hashes"), dict) else {}
    return {"file": {"version": version, "timestamp": data.get("timestamp"), "sha256": file_sha256(data),
                     "modifier": mod["class"], "instance_name": mod["instance_name"], "parameters": params,
                     "scores": data.get("scores"), "hashes": hashes, "hashes_note": HASHES_NOTE if hashes else None,
                     "wsp": None if wsp is None else {k: wsp.get(k) for k in ("record_version", "patch", "digest")}},
            "model": model, "settings": settings_rows(ws, settings, mod["method"]),
            "measurement": {"name": name, "state": state, "request": measure},
            "method": mod["method"], "parameters": ide, "script": script,
            "environment": env, "notes": notes}


def heretic_script(ws, data, *, measurement=None) -> str:
    """The one-line weight script a Heretic reproduce.json translates to (ablate_direction or ablate_ara on the
    measurement the file's prompts make), after every check: what POST /api/directions/reproduce applies."""
    return check_heretic_reproduce(ws, data, measurement=measurement)["script"]


# ------------------------------------------------------------------ the run
def heretic_reproduce(sess, data, *, measurement=None, dry_run=True, message=None, enforce_scope=False) -> dict:
    """Run a Heretic reproduce.json natively: the checks, the measurement from the file's prompt specifications and
    settings (or the reuse of one made from exactly them), then the file's modifier with the file's parameters as a
    weight script through Session.apply (a dry run by default, which commits nothing). The origin of a commit says
    it came from the file; nothing here writes a weight or switches a patch."""
    from . import directions_search as DSR
    ws = sess.ws
    busy = RecordBusy("a search is running on this workspace; stop it before running a reproduce.json")
    job = getattr(sess, "search_job", None)
    if (job and job.get("running") and job.get("workspace") == ws.name) or DSR.running(ws):
        raise busy
    held = contextlib.ExitStack()
    try:
        held.enter_context(DSR.search_lock(ws))
    except DSR.SearchBusy:
        raise busy from None
    with held:
        plan = check_heretic_reproduce(ws, data, measurement=measurement)
        m = plan["measurement"]
        if m["state"] == "new":
            DR.measure(ws, **m["request"])
        wsp = plan["file"]["wsp"]
        origin = {"kind": "heretic_reproduce", "file_sha256": plan["file"]["sha256"], "version": plan["file"]["version"],
                  "timestamp": plan["file"]["timestamp"], "model": plan["model"]["file"]["model"],
                  "model_commit": plan["model"]["file"]["model_commit"], "modifier": plan["file"]["modifier"],
                  "measurement": m["name"], "digest": None, "file_digest": (wsp or {}).get("digest")}

        def check(deltas):
            origin["digest"] = DSR.piece_digest(deltas)
            return None
        report = sess.apply(plan["script"], message or f"Heretic reproduce.json ({plan['file']['modifier']})",
                            author="human", dry_run=dry_run, enforce_scope=enforce_scope, origin=origin, check=check)
    plan["settings"] = settings_rows(ws, data.get("settings") or {}, plan["method"])        # the runner exists now
    digest, file_digest = origin["digest"], origin["file_digest"]
    return {**report, "ran": {"measurement": m["name"], "measured": m["state"] == "new", "script": plan["script"],
                              "dry_run": dry_run},
            "digest": digest, "file_digest": file_digest,
            "same_as_file": None if file_digest is None or digest is None else digest == file_digest,
            "heretic": plan}
