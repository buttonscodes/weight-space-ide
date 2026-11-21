"""Scoring an already modified checkpoint against its base, as Heretic's ``evaluate_model`` does.

The modified checkpoint (a directory of floating-point safetensors with the base's tensor names and shapes: a
Heretic merge, a fine-tune) is opened read-only through a Store of its own; only the base model is ever loaded.
``path`` may also be a PEFT LoRA adapter directory (the base plus the adapter, as PEFT merges it) or another
workspace of the IDE (its base with its enabled patches): wsp.directions_evaluate_inputs detects which and reads
it through a store of the same shape, and the answer says which it took (``input``).

    check      everything that needs no weight beyond the headers: the workspace, the request, the path, the
               tensor names, shapes and dtypes, the configuration that changes how the model runs, the
               generation configuration, the scorers (wsp.directions_scores.check_scores)
    pass 1     every tensor compared with the base, slab by slab, holding nothing: which tensors differ, by how much
    forecast   the memory the changes would hold while applied (12 bytes per weight), refused before anything is built
    pass 2     the pieces ``other - current`` of the differing (and patched) tensors, so current + pieces = other
    scoring    the base's scores and the checkpoint's, scored as an uncommitted candidate on top of the loaded model

The tokenizer and the chat template are the base's, as in Heretic (its tokenizer is loaded once, from the base);
a different tokenizer or template in the checkpoint is a note. Nothing here writes a weight, a patch,
workspace.json or the checkpoint's files; a KL reference entry may be written, as by any score request. GGUF is
refused on both sides, never dequantized. A tensor the loaded text model does not load (wsp.runtime_layouts: a
vision tower, a multi-token prediction head) is compared but never held (``compare.not_loaded``, a note); a run whose
patches take the side path (bfloat16, float16) says so in a note. Definitions and limits: docs/DIRECTIONS.md.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field

import torch

from . import arch
from . import delta as D
from . import directions_scores as DS
from . import edit_limit, edits, runtime_layouts
from .directions import DirectionError
from .store import Store, norm_index
from .workspace import STREAM_ELEMS

FIELDS = ("path", "check", "scores", "chat", "system", "prefix", "chat_options", "max_response_length")
SCORE_KEYS = ("scores", "chat", "system", "prefix", "chat_options", "max_response_length")
SCRATCH = 4 * 4 * STREAM_ELEMS        # four float32 slabs at once (base, current, other, difference): 256 MiB
MAX_PATH = 4096
NOTE_KEYS = 8                         # differing config keys listed in a note
INDEX = "model.safetensors.index.json"

# The configuration that changes how the model runs: (row, the keys it reads, in order, its normalisation).
CONFIG_KEYS = (
    ("model_type", ("model_type",), "as_is"),
    ("architectures", ("architectures",), "as_is"),
    ("hidden_size", ("hidden_size", "n_embd", "d_model"), "int"),
    ("num_hidden_layers", ("num_hidden_layers", "n_layer", "num_layers"), "int"),
    ("num_attention_heads", ("num_attention_heads", "n_head", "num_heads"), "int"),
    ("num_key_value_heads", ("num_key_value_heads",), "kv_heads"),
    ("head_dim", ("head_dim",), "int"),
    ("intermediate_size", ("intermediate_size", "n_inner"), "int"),
    ("vocab_size", ("vocab_size",), "int"),
    ("max_position_embeddings", ("max_position_embeddings", "n_positions"), "int"),
    ("norm_eps", ("rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon"), "float"),
    ("hidden_act", ("hidden_act", "hidden_activation", "activation_function"), "as_is"),
    ("tie_word_embeddings", ("tie_word_embeddings",), "bool"),
    ("attention_bias", ("attention_bias",), "bool"),
    ("rope_theta", ("rope_theta", "rope_parameters.rope_theta"), "float"),
    ("rope_type", ("rope_scaling.rope_type", "rope_scaling.type", "rope_parameters.rope_type"), "rope_type"),
    ("rope_rest", ("rope_scaling", "rope_parameters"), "rope_rest"),
    ("sliding_window", ("sliding_window", "use_sliding_window"), "sliding_window"),
    ("layer_types", ("layer_types",), "layer_types"),
)
CONFIG_IGNORED = frozenset({
    "_name_or_path", "transformers_version", "torch_dtype", "dtype", "_commit_hash", "_attn_implementation",
    "attn_implementation", "use_cache", "rope_parameters", "rope_scaling", "rope_theta", "sliding_window",
    "use_sliding_window", "layer_types", "quantization_config", "quantization"})
# Every top-level key a row reads (never listed in the note of other differing keys).
ROW_KEYS = frozenset(k.split(".")[0] for _, keys, _ in CONFIG_KEYS for k in keys)


@dataclass
class Other:                          # the modified checkpoint, opened read-only (headers only)
    path: str
    files: list[str]
    store: Store
    config: dict | None
    generation: dict | None
    tokenizer: object | None
    template: str | None
    tied: list[str] = field(default_factory=list)
    kind: str = "checkpoint"           # checkpoint | adapter | workspace (wsp.directions_evaluate_inputs)
    input: dict = field(default_factory=lambda: {"kind": "checkpoint"})
    may_differ: set | None = None      # the only tensors that can differ from the base (None: any); pass 1 reads these
    pieces: dict | None = None         # the change relative to the base as pieces, scored as they are (H4c, same base)
    notes: list = field(default_factory=list)


# ------------------------------------------------------------------ the request (no model, no weights)
def evaluate_body(body) -> dict:
    """The fields of a request (``stream`` is the route's) -> {path, check, and the score fields given}."""
    if not isinstance(body, dict):
        raise DirectionError("the request must be an object")
    unknown = sorted(set(body) - set(FIELDS) - {"stream"}, key=str)
    if unknown:
        raise DirectionError(f"{unknown[0]}: unknown field; the fields are {', '.join(FIELDS[:-1])}, "
                             f"{FIELDS[-1]} and stream")
    if "path" not in body:
        raise DirectionError("path: required: the absolute path of the modified checkpoint's directory")
    path = body["path"]
    if not isinstance(path, str):
        raise DirectionError("path must be text")
    if len(path) > MAX_PATH:
        raise DirectionError(f"path is longer than {MAX_PATH} characters")
    check = body.get("check", False)
    if not isinstance(check, bool):
        raise DirectionError("check must be true or false")
    if not check and "scores" not in body:
        raise DirectionError("scores: required (unless check is true)")
    return {"path": path, "check": check, **{k: body[k] for k in SCORE_KEYS if k in body}}


def _read_json(path, what):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise DirectionError(f"{what} cannot be read: {e}") from None


def _weight_files(path) -> list[str]:
    """The checkpoint's safetensors files: its index's, else model.safetensors, else every *.safetensors."""
    idx = os.path.join(path, INDEX)
    if os.path.exists(idx):
        index = _read_json(idx, f"the checkpoint's index {INDEX}")
        try:
            names = sorted(set(index["weight_map"].values()))
            if not all(isinstance(n, str) for n in names):
                raise TypeError("a file name is not text")
        except (KeyError, TypeError, AttributeError) as e:
            raise DirectionError(f"the checkpoint's index {INDEX} cannot be read: {type(e).__name__}: {e}") from None
        if not names:
            raise DirectionError("path has no safetensors weights (model.safetensors, an index, or *.safetensors files)")
        for n in names:
            if os.path.basename(n) != n or not os.path.isfile(os.path.join(path, n)):
                raise DirectionError(f"the checkpoint's index names {n}, which is not in path")
        return [os.path.join(path, n) for n in names]
    one = os.path.join(path, "model.safetensors")
    if os.path.isfile(one):
        return [one]
    files = sorted(os.path.join(path, n) for n in os.listdir(path)
                   if n.endswith(".safetensors") and os.path.isfile(os.path.join(path, n)))
    if not files:
        raise DirectionError("path has no safetensors weights (model.safetensors, an index, or *.safetensors files)")
    return files


def _quantization(config) -> str | None:
    if not isinstance(config, dict):
        return None
    q = config.get("quantization_config") or config.get("quantization")
    if not q:
        return None
    method = q.get("quant_method") if isinstance(q, dict) and config.get("quantization_config") else None
    return method if isinstance(method, str) and method else "quantization"


def template_text(directory) -> str | None:
    """A directory's chat template: chat_template.jinja, else chat_template.json's, else tokenizer_config.json's
    (a list as sorted JSON); None when there is none."""
    if not directory:
        return None
    jinja = os.path.join(directory, "chat_template.jinja")
    if os.path.isfile(jinja):
        try:
            with open(jinja) as f:
                return f.read()
        except OSError:
            return None
    for fn in ("chat_template.json", "tokenizer_config.json"):
        p = os.path.join(directory, fn)
        if not os.path.isfile(p):
            continue
        try:
            with open(p) as f:
                value = json.load(f).get("chat_template")
        except (OSError, ValueError, AttributeError):
            value = None
        if value is None and fn == "chat_template.json":
            continue
        if isinstance(value, list):
            return json.dumps(value, sort_keys=True)
        return value if isinstance(value, str) else None
    return None


def _tokenizer(ws, path):
    """The checkpoint's tokenizer, built as Workspace.load builds the base's; None when its file is missing or
    cannot be read."""
    from .tokenizer import BPETokenizer, load_tokenizer
    try:
        if ws.kind == "wsp":
            if not os.path.isfile(os.path.join(path, "tokenizer_meta.json")):
                return None
            return load_tokenizer(path)
        tp = os.path.join(path, "tokenizer.json")
        if not os.path.isfile(tp):
            return None
        with open(tp) as f:
            return BPETokenizer(f.read(), special=True)
    except Exception:                               # a damaged tokenizer file: scored with the base's, as any
        return None


def _base_dir(ws):
    return getattr(ws, "model_dir", None) if ws.kind != "wsp" else ws.dir


def _generation(directory, what):
    p = os.path.join(directory, "generation_config.json") if directory else None
    if not p or not os.path.exists(p):
        return {}
    g = _read_json(p, what)
    return g if isinstance(g, dict) else {}


def open_other(ws, path) -> Other:
    """The directory's checks (E-H8 to E-H17), its kind (a checkpoint, a PEFT adapter, a workspace:
    wsp.directions_evaluate_inputs), its headers and its configuration files; no weight read."""
    if not os.path.isabs(path):
        raise DirectionError("path must be an absolute path of a local directory; a Hub id is not looked up (give its "
                             "snapshot directory in the Hugging Face cache)")
    if not os.path.exists(path):
        raise DirectionError(f"path: no such directory: {path}")
    if not os.path.isdir(path):
        if path.lower().endswith(".gguf"):
            raise DirectionError("path names a GGUF file; the modified model must be a directory of floating-point "
                                 "safetensors files (a GGUF model is not read as floating-point weights)")
        raise DirectionError("path names a file; give the checkpoint's directory")
    from . import directions_evaluate_inputs as EI
    kind = EI.detect(path)
    if kind == "adapter":
        return EI.open_adapter(ws, path)
    if kind == "workspace":
        return EI.open_workspace(ws, path)
    files = _weight_files(path)
    cp = os.path.join(path, "config.json")
    if ws.kind == "hf" and not os.path.isfile(cp):
        raise DirectionError("path has no config.json")
    base = {os.path.realpath(f.path) for f in ws.store.files}
    if {os.path.realpath(f) for f in files} == base:
        raise DirectionError("path holds the open workspace's own base checkpoint: there is nothing to compare")
    config = None
    if os.path.isfile(cp):
        if ws.kind == "hf":
            config = _read_json(cp, "the modified checkpoint's config.json")
            if not isinstance(config, dict):
                raise DirectionError("the modified checkpoint's config.json cannot be read: it is not an object")
        else:
            try:
                with open(cp) as f:
                    config = json.load(f)
            except (OSError, ValueError):
                config = None
    method = _quantization(config)
    if method:
        raise DirectionError(f"the modified checkpoint's config.json declares quantization ({method}); its integer "
                             f"storage is not read as floating-point weights")
    try:
        store = Store(files)
    except Exception as e:
        raise DirectionError(f"the checkpoint's safetensors headers cannot be read: {type(e).__name__}: {e}") from None
    generation = _generation(path, "the modified checkpoint's generation_config.json") if ws.kind == "hf" else None
    return Other(path=path, files=files, store=store, config=config, generation=generation,
                 tokenizer=_tokenizer(ws, path), template=template_text(path))


# ------------------------------------------------------------------ the configuration table
_MISSING = object()


def _dotted(cfg, key):
    v = cfg
    for part in key.split("."):
        if not isinstance(v, dict) or v.get(part) is None:
            return _MISSING
        v = v[part]
    return v


def _first(cfg, keys):
    for k in keys:
        v = _dotted(cfg, k)
        if v is not _MISSING:
            return v
    return _MISSING


def _number(v, kind):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    if kind == "float":
        return float(v)
    return int(v) if math.isfinite(v) and float(v).is_integer() else v


def _row(cfg, row):
    """(present, normalised value) of one CONFIG_KEYS row on one side."""
    name, keys, how = row
    if how in ("as_is", "bool"):
        v = _first(cfg, keys)
        return (v is not _MISSING), (None if v is _MISSING else v)
    if how in ("int", "float"):
        v = _first(cfg, keys)
        return (v is not _MISSING), (None if v is _MISSING else _number(v, how))
    if how == "kv_heads":
        v = _first(cfg, keys)
        if v is not _MISSING:
            return True, _number(v, "int")
        return _row(cfg, CONFIG_KEYS[4])
    if how == "rope_type":
        v = _first(cfg, keys)
        return True, ("default" if v is _MISSING else v)
    if how == "rope_rest":
        rs, rp = cfg.get("rope_scaling"), cfg.get("rope_parameters")
        if rs is not None:
            return True, ({k: x for k, x in rs.items() if k not in ("type", "rope_type")} if isinstance(rs, dict) else rs)
        if rp is not None:
            return True, ({k: x for k, x in rp.items() if k not in ("rope_theta", "rope_type")}
                          if isinstance(rp, dict) else rp)
        return True, {}
    if how == "sliding_window":
        return True, (cfg.get("sliding_window") if cfg.get("use_sliding_window") is True else None)
    if how == "layer_types":
        v = cfg.get("layer_types")
        if v is not None:
            return True, v
        _, n = _row(cfg, CONFIG_KEYS[3])
        return True, (["full_attention"] * n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None)
    raise AssertionError(how)


def _scopes(base, other):
    out = [("", base, other)]
    tb, to = base.get("text_config"), other.get("text_config")
    if isinstance(tb, dict) or isinstance(to, dict):
        out.append(("text_config.", tb if isinstance(tb, dict) else {}, to if isinstance(to, dict) else {}))
    return out


def compare_config(base: dict, other: dict) -> dict:
    """CONFIG_KEYS on both sides (and on text_config when either has one): a differing row -> E-H22. -> {'same',
    'other'}: 'other' the sorted keys that differ (or are present on one side only) and change nothing the IDE runs."""
    base = base if isinstance(base, dict) else {}
    other = other if isinstance(other, dict) else {}
    noted = set()
    scopes = _scopes(base, other)
    for prefix, b, o in scopes:
        for row in CONFIG_KEYS:
            pb, vb = _row(b, row)
            po, vo = _row(o, row)
            if not pb and not po:
                continue
            if pb != po:
                noted.add(prefix + row[0])
            elif vb != vo:
                raise DirectionError(f"the modified checkpoint's config.json differs from the base's in {prefix}{row[0]} "
                                     f"(base {json.dumps(vb)}, modified {json.dumps(vo)}): it would run differently from "
                                     f"the configuration the IDE runs it with, the base's")
    for prefix, b, o in scopes:
        for k in set(b) | set(o):
            if k in CONFIG_IGNORED or k in ROW_KEYS or k == "text_config":
                continue
            if (k in b) != (k in o) or b[k] != o[k]:
                noted.add(prefix + k)
    keys = sorted(noted)
    return {"same": not keys, "other": keys}


def compare_generation(base: dict, other: dict) -> dict:
    """The greedy keys of both generation_config.json files (a missing key and a null one alike): a difference ->
    E-H23 naming the first in GREEDY_KEYS order."""
    for k in DS.GREEDY_KEYS:
        vb, vo = base.get(k), other.get(k)
        if vb != vo:
            raise DirectionError(f"the modified checkpoint's generation_config.json differs from the base's in {k} (base "
                                 f"{json.dumps(vb)}, modified {json.dumps(vo)}): Heretic's evaluate_model would generate "
                                 f"with the modified checkpoint's settings and the IDE would use the base's; this is not "
                                 f"supported yet")
    return {"same": True}


# ------------------------------------------------------------------ the checks before any weight read
def _is_tied_head(ws, name) -> bool:
    """A head tensor of a base whose structure is tied: the loaded model does not use it (its head is the token
    embedding), so it is never compared and never gets a piece, in either checkpoint or in both."""
    if not getattr(ws.struct, "tied", False):
        return False
    m = arch._HEAD.search(name)
    return m is not None and m.group(1) == "weight"


def _not_loaded(ws, name) -> str | None:
    """The part (``the vision tower``, ...) of checkpoint tensor ``name`` that the text model the runner loads does
    not load, from wsp.runtime_layouts on the workspace's config; None for a tensor it loads (and for every tensor
    of a layout without a runtime name table)."""
    try:
        runtime_layouts.runtime_name(getattr(ws, "hf_config", None) or {}, name, tuple(ws.shape(name)))
    except runtime_layouts.Skipped as e:
        return e.part or e.reason
    return None


def _texts(spec) -> list[str]:
    """Every distinct user and system text of the scorers' prompts, in order."""
    seen = {}
    for e in spec["scores"]:
        for p in e["prompts"]:
            user, system = (p, spec["system"]) if isinstance(p, str) else (p.get("user"), p.get("system"))
            for t in (user, system):
                if isinstance(t, str):
                    seen.setdefault(t, None)
    return list(seen)


def _ids(encode, text):
    try:
        return [int(t) for t in encode(text)]
    except Exception as e:                          # a text one tokenizer cannot encode differs
        return ("cannot encode", type(e).__name__)


def compare_tokenizer(ws, other, spec) -> dict:
    texts = _texts(spec) if spec is not None else []
    out = {"prompts": len(texts), "differ": None if other.tokenizer is None else 0}
    if other.tokenizer is not None:
        out["differ"] = sum(1 for t in texts if _ids(ws.tok.encode, t) != _ids(other.tokenizer.encode, t))
    out["template_same"] = template_text(_base_dir(ws)) == other.template
    return out


def check_evaluate(ws, args) -> dict:
    """Everything before any weight is read beyond the headers, and before the model (H.3.3's order) ->
    {'other', 'check', 'spec', 'config', 'generation_config', 'tokenizer'}."""
    if ws.kind == "gguf":
        raise DirectionError("scoring a modified checkpoint needs a floating-point base: this workspace's base is GGUF "
                             "(quantized) and is never dequantized to compare it with another checkpoint")
    if ws.store.remote:
        raise DirectionError("the open model's weights are still on the Hub; pull them first (the comparison reads "
                             "every base tensor)")
    for name in ws.names():
        if not ws.store.is_float(name):
            raise DirectionError(f"the open model's tensor {name} is stored as {ws.store.dtype(name)}; only a checkpoint "
                                 f"of floating-point tensors can be compared")
    args = evaluate_body(args)
    other = open_other(ws, args["path"])
    base_names, other_names = ws.names(), other.store.names()
    in_base, in_other = set(base_names), set(other_names)
    tied, missing, extra = [], [], []
    for n in base_names:
        if _is_tied_head(ws, n):
            tied.append(n)
        elif n not in in_other:
            missing.append(n)
    for n in other_names:
        if n not in in_base:
            (tied if _is_tied_head(ws, n) else extra).append(n)
    if missing:
        raise DirectionError(f"the modified checkpoint lacks {len(missing)} tensor(s) of the base, for example "
                             f"{missing[0]}; it is not the same architecture")
    if extra:
        raise DirectionError(f"the modified checkpoint has {len(extra)} tensor(s) the base does not, for example "
                             f"{extra[0]}; it is not the same architecture")
    other.tied = sorted(tied)
    compared = [n for n in base_names if n in in_other and n not in other.tied]
    for n in compared:
        a, b = other.store.shape(n), ws.shape(n)
        if tuple(a) != tuple(b):
            raise DirectionError(f"{n} has shape {json.dumps(list(a))} in the modified checkpoint and "
                                 f"{json.dumps(list(b))} in the base")
    for n in compared:
        if not other.store.is_float(n):
            raise DirectionError(f"{n} is stored as {other.store.dtype(n)} in the modified checkpoint; only "
                                 f"floating-point tensors are compared (a quantized checkpoint is not dequantized)")
    config, generation = {"same": True, "other": []}, {"same": True}
    if ws.kind == "hf":
        config = compare_config(getattr(ws, "hf_config", None) or {}, other.config or {})
        generation = compare_generation(_generation(_base_dir(ws), "the open model's generation_config.json"),
                                        other.generation or {})
    spec = None
    if "scores" in args or not args["check"]:
        spec = DS.check_scores(ws, {"state": "current", **{k: args[k] for k in SCORE_KEYS if k in args}})
    return {"other": other, "check": args["check"], "spec": spec, "config": config, "generation_config": generation,
            "tokenizer": compare_tokenizer(ws, other, spec)}


# ------------------------------------------------------------------ pass 1, the forecast, pass 2
def _slab_region(whole, rows):
    return (whole[0][rows.start:rows.stop],) + whole[1:] if whole else whole


def compare_weights(ws, other, *, progress=None, stopped=None) -> dict:
    """Pass 1: every compared tensor of the checkpoint against the base, slab by slab, holding nothing."""
    started = time.monotonic()
    names = [n for n in ws.names() if n not in other.tied]
    total = len(names)
    rows, changed_all, of_all = [], 0, 0
    for done, name in enumerate(names):
        if stopped is not None and stopped():
            return {"stopped": True, "done": done, "total": total}
        if other.may_differ is not None and name not in other.may_differ:      # equal to the base by construction
            of_all += ws.store.numel(name)
            if progress is not None:
                progress("compare", done + 1, total)
            continue
        whole = norm_index((), ws.shape(name))
        changed, sq, base_sq, max_abs = 0, 0.0, 0.0, 0.0
        for slab, b in ws.slabs(name, (), "base", max_elems=STREAM_ELEMS):
            o = other.store.read(name, _slab_region(whole, slab))
            if not bool(torch.isfinite(o).all()):
                raise DirectionError(f"{name} contains NaN or inf in the modified checkpoint")
            d = o.double() - b.double()
            changed += int((o != b).sum())
            sq += float(d.pow(2).sum())
            base_sq += float(b.double().pow(2).sum())
            if d.numel():
                max_abs = max(max_abs, float(d.abs().max()))
        n = ws.store.numel(name)
        of_all += n
        if changed:
            changed_all += changed
            norm = math.sqrt(sq)
            rows.append({"param": name, "changed": changed, "of": n, "norm": norm, "max_abs": max_abs,
                         "relative": norm / math.sqrt(base_sq) if base_sq > 0 else None})
        if progress is not None:
            progress("compare", done + 1, total)
    not_loaded = sorted(r["param"] for r in rows if _not_loaded(ws, r["param"]) is not None)
    return {"tensors": total, "differ": len(rows), "changed": changed_all, "of": of_all, "rows": rows,
            "dtypes": {"base": dict(sorted(Counter(ws.store.dtype(n) for n in names).items())),
                       "modified": dict(sorted(Counter(other.store.dtype(n) for n in names).items()))},
            "tied": list(other.tied), "not_loaded": not_loaded, "seconds": round(time.monotonic() - started, 3)}


def held_names(ws, other, compared) -> list[str]:
    """The tensors that get a piece: the differing ones and every patched one (its piece is other - current even
    where the checkpoint equals the base), tied heads and tensors the text model does not load excepted, in the
    workspace's order."""
    want = {r["param"] for r in compared["rows"]} | set(ws.touched())
    return [n for n in ws.names() if n in want and n not in other.tied and _not_loaded(ws, n) is None]


def held_weights(ws, other, compared) -> dict:
    """The forecast: what the pieces hold while applied (4 bytes a weight, 12 with the runner's saved originals),
    against what memory allows now (edit_limit's measurement, as an edit is admitted)."""
    names = held_names(ws, other, compared)
    W = sum(ws.store.numel(n) for n in names)
    fits, why = True, None
    if W > edit_limit.SMALL_EDIT:
        _, w, detail = edit_limit._memory(ws, SCRATCH)
        if w is None:
            fits, why = False, (f"memory could not be measured ({detail.get('reason')}), so the modified checkpoint's "
                                f"{W:,} changed weights are not loaded")
        elif W > w:
            fits = False
            why = (f"the modified checkpoint differs from the current weights in {len(names)} tensors ({W:,} weights); "
                   f"scoring it holds them as FP32 changes applied to the loaded model, about "
                   f"{edit_limit.gib(edit_limit.MEMORY_BYTES * W)} at {edit_limit.MEMORY_BYTES} bytes per weight, and "
                   f"memory allows {w:,} weights now ({edit_limit.gib(detail.get('available', 0))} available, "
                   f"{edit_limit.gib(detail.get('preferred', 0))} kept free by the {edit_limit.RESERVE}). Free memory "
                   f"on the machine, or score a checkpoint that changes fewer tensors")
    return {"tensors": len(names), "weights": W, "bytes": edit_limit.PATCH_BYTES * W,
            "applied_bytes": edit_limit.MEMORY_BYTES * W, "fits": fits, "why": why}


def build_candidate(ws, other, names, *, stopped=None) -> tuple[dict | None, dict]:
    """Pass 2: for each tensor of ``names``, the pieces other - current (current = the base with the enabled
    patches, one consistent set) -> (the pieces, or None when nothing differs; {'inexact', 'max_abs'}: how far
    current + pieces, as the runner will hold it in float32, is from the checkpoint). A stop ->
    (None, {'stopped': True, 'done', 'total'})."""
    current = ws.piece_index()
    deltas, inexact, max_abs = {}, 0, 0.0
    for done, name in enumerate(names):
        if stopped is not None and stopped():
            return None, {"stopped": True, "done": done, "total": len(names)}
        pieces = []
        for _, region, cur in edits._current_slabs(ws, name, current):
            o = other.store.read(name, region)
            d = o - cur
            if bool((d != 0).any()):
                pieces.append(D.block(region, d))
            back = cur + d
            inexact += int((back != o).sum())
            if back.numel():
                max_abs = max(max_abs, float((back - o).abs().max()))
        ps = [p for p in D.compact(pieces, ws.shape(name)) if not p.is_zero()] if pieces else []
        if ps:
            deltas[name] = ps
    return (deltas or None), {"inexact": inexact, "max_abs": max_abs}


def candidate(ws, other, compared, *, stopped=None) -> tuple[dict | None, dict]:
    """The pieces scored for the modified model: its own pieces relative to the base when it has them (a workspace
    over the same base) and the open workspace has no enabled patch, else pass 2's ``other - current``."""
    names = held_names(ws, other, compared)
    if other.pieces is not None and not ws.touched():
        deltas = {n: other.pieces[n] for n in names if n in other.pieces}
        return (deltas or None), {"inexact": 0, "max_abs": 0.0}
    return build_candidate(ws, other, names, stopped=stopped)


# ------------------------------------------------------------------ the notes and the whole request
def _fmt(x: float) -> str:
    return f"{x:.6g}"


def notes_of(ws, compare, *, reconstruction=None, other=None, runner=None) -> list[str]:
    out = list(other.notes) if other is not None else []
    tok = compare["tokenizer"]
    if tok["differ"] is None:
        out.append("the modified checkpoint has no tokenizer file; it is scored with the base's tokenizer, as Heretic's "
                   "evaluate_model does")
    elif tok["differ"] > 0:
        out.append(f"the modified checkpoint's tokenizer encodes {tok['differ']} of the {tok['prompts']} prompt texts "
                   f"differently from the base's; it is scored with the base's tokenizer, as Heretic's evaluate_model does")
    if not tok["template_same"]:
        out.append("the modified checkpoint's chat template differs from the base's; prompts are rendered with the "
                   "base's template, as Heretic's evaluate_model does")
    if compare["differ"] == 0:
        out.append("the modified checkpoint's weights equal the base's in every tensor")
    if ws.touched():
        p = sum(1 for x in ws.patches if x.enabled and x.strength)
        out.append(f"{p} patch(es) are enabled in this workspace: the modified checkpoint is scored as itself (its "
                   f"weights replace the current ones for this scoring), against the base weights")
    names = set(ws.names())
    held = set(other.store.names()) if other is not None else set()
    absent = [n for n in compare["tied"] if n in names and n not in held]
    extra = [n for n in compare["tied"] if n not in names]
    both = [n for n in compare["tied"] if n in names and n in held]
    if absent:
        out.append(f"{', '.join(absent)} is not in the modified checkpoint; the base ties it to the token embedding, so "
                   f"it is not compared")
    if extra:
        out.append(f"{', '.join(extra)} is in the modified checkpoint but not in the base; the base ties it to the token "
                   f"embedding, so it is not compared")
    if both:
        out.append(f"{', '.join(both)} is in both checkpoints; the base ties it to the token embedding, so the model does "
                   f"not use it and it is not compared")
    skipped = compare.get("not_loaded") or []
    if skipped:
        parts = list(dict.fromkeys(_not_loaded(ws, n) for n in skipped))
        shown = ", ".join(skipped[:NOTE_KEYS]) + (f" and {len(skipped) - NOTE_KEYS} more"
                                                   if len(skipped) > NOTE_KEYS else "")
        if len(skipped) == 1:
            out.append(f"1 differing tensor belongs to a part the text model does not load ({parts[0]}) and is not "
                       f"scored: {shown}")
        else:
            out.append(f"{len(skipped)} differing tensors belong to parts the text model does not load "
                       f"({'; '.join(parts)}) and are not scored: {shown}")
    keys = compare["config"]["other"]
    if keys:
        shown = ", ".join(keys[:NOTE_KEYS]) + (f" and {len(keys) - NOTE_KEYS} more" if len(keys) > NOTE_KEYS else "")
        out.append(f"the modified checkpoint's config.json also differs in {shown}; these do not change how the IDE "
                   f"runs it")
    if reconstruction is not None and reconstruction["inexact"] > 0:
        out.append(f"{reconstruction['inexact']} weights of the modified checkpoint come out differently by at most "
                   f"{_fmt(reconstruction['max_abs'])} once applied as a change in float32")
    if runner is not None and getattr(runner, "patch_apply", None) == "side":
        dtype = str(getattr(runner, "dtype", "")).removeprefix("torch.")
        out.append(f"this run is in {dtype}: the checkpoint's changes are applied as float32 differences beside the "
                   f"{dtype} base weights (the side path), not as the checkpoint's weights rounded to {dtype}, which is "
                   f"what loading it would give")
    return out


def evaluate(ws, body, *, progress=None, stopped=None) -> dict:
    """The whole request: the checks, pass 1, the forecast; with ``check`` the comparison only (no model). Otherwise
    pass 2 and the scores of the base and of the checkpoint (an uncommitted candidate on top of current), against
    the base's KL reference. ``progress(phase, done, total)``: phases compare, reference, base_responses, responses,
    logits. A stop -> {'stopped': True, 'done', 'total'} of the step it stopped in."""
    started = time.monotonic()
    with ws.lock:
        checked = check_evaluate(ws, body)
        other, spec, check = checked["other"], checked["spec"], checked["check"]
        compared = compare_weights(ws, other, progress=progress, stopped=stopped)
        if compared.get("stopped"):
            return compared
        compare = {**compared, "held": held_weights(ws, other, compared), "config": checked["config"],
                   "generation_config": checked["generation_config"], "tokenizer": checked["tokenizer"]}
        if check:
            return {"check": True, "path": other.path, "input": other.input, "version": ws.version, "compare": compare,
                    "notes": notes_of(ws, compare, other=other), "seconds": round(time.monotonic() - started, 3)}
        if not compare["held"]["fits"]:
            raise DirectionError(compare["held"]["why"])
        deltas, reconstruction = candidate(ws, other, compared, stopped=stopped)
        if reconstruction.get("stopped"):
            return reconstruction
        compare["reconstruction"] = reconstruction
        DS.runner_row(ws)
        prepared = DS.prepare(ws, spec, progress=progress, stopped=stopped)
        if prepared.get("stopped"):
            return prepared

        def base_progress(phase, done, total):
            progress("base_responses" if phase == "responses" else phase, done, total)
        try:
            base = DS.evaluate(ws, prepared, "base", None, progress=base_progress if progress is not None else None,
                               stopped=stopped)
            if base.get("stopped"):
                return base
            state = "candidate" if deltas else "current"
            modified = DS.evaluate(ws, prepared, state, deltas, progress=progress, stopped=stopped)
            if modified.get("stopped"):
                return modified
        finally:
            DS._release(ws, deltas)
        changes = [{"name": b["name"], "base": b["value"], "modified": m["value"], "change": m["value"] - b["value"]}
                   for b, m in zip(base["scores"], modified["scores"])]
        return {"check": False, "path": other.path, "input": other.input, "version": ws.version,
                "seconds": round(time.monotonic() - started, 3), "compare": compare,
                "rendering": {"chat": spec["chat"], "system": spec["system"], "prefix": spec["prefix"],
                              "chat_options": spec["chat_options"]},
                "generation": prepared["generation"], "base": {"state": "base", "scores": base["scores"]},
                "modified": {"state": state, "scores": modified["scores"]}, "changes": changes,
                "datasets": [e["datasets"] for e in spec["scores"]],
                "notes": notes_of(ws, compare, reconstruction=reconstruction, other=other, runner=ws.runner),
                "done": modified["done"], "total": modified["total"]}
