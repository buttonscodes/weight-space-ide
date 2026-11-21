"""What ``path`` of POST /api/directions/evaluate names, and the two inputs beside a checkpoint (H4b, H4c).

    checkpoint   a directory of safetensors weights (wsp.directions_evaluate.open_other, H4)
    adapter      a PEFT LoRA adapter directory (adapter_config.json): the base plus the adapter, as PEFT merges it
                 (wsp.peft_adapter), read on the fly; no merged checkpoint is written
    workspace    a workspace of the IDE (workspace.json): its logical weights, its base with its enabled patches

Each is presented to the comparison as an ``Other`` whose store reads the modified model's float32 weights region
by region, so the checks, pass 1, the forecast and pass 2 of wsp.directions_evaluate are the checkpoint's. Two
shortcuts follow from what the input is: an adapter, and a workspace over the same base (its weight files are the
open workspace's, by real path), can differ from the base only in the tensors they change (``may_differ``; pass 1
reads only those); and such a workspace's change relative to the base is its patch pieces (``pieces``), scored as
they are when the open workspace has no enabled patch.

The other workspace is read, never written: it is loaded with Workspace.load (no lock, no migration: a GGUF
workspace, whose load may record its base's hash, is refused before), and nothing in it changes.
"""
from __future__ import annotations

import json
import os

from . import delta as D
from . import edit_limit
from .directions import DirectionError
from .store import norm_index, region_index
from .workspace import _is_region

ADAPTER = "adapter_config.json"
WORKSPACE = "workspace.json"
INDEX = "model.safetensors.index.json"
LISTED = 8                             # entries of an unrecognised directory named in its refusal
KINDS = ("adapter", "workspace", "checkpoint")        # the order an ambiguous directory's refusal names them


# ------------------------------------------------------------------ which kind of directory
def markers(path) -> dict[str, list[str]]:
    """{kind: [the files that make it one]} of a directory, for every kind it looks like."""
    names = set(os.listdir(path))
    files = {n for n in names if os.path.isfile(os.path.join(path, n))}
    out: dict[str, list[str]] = {}
    if ADAPTER in files:
        out["adapter"] = [ADAPTER]
    if WORKSPACE in files:
        out["workspace"] = [WORKSPACE]
    elif "config.json" in files and "base.safetensors" in files:      # wsp.session.is_workspace's older layout
        out["workspace"] = ["config.json", "base.safetensors"]
    own = {"adapter_model.safetensors"} | ({"base.safetensors"} if "workspace" in out else set())
    weights = sorted(n for n in files if n == INDEX or (n.endswith(".safetensors") and n not in own))
    if weights:
        out["checkpoint"] = weights
    return out


def _what(kind, files) -> str:
    shown = ", ".join(files[:3]) + (f" and {len(files) - 3} more" if len(files) > 3 else "")
    return {"adapter": "a PEFT adapter", "workspace": "a workspace", "checkpoint": "checkpoint weights"}[kind] + \
        f" ({shown})"


def _entries(path) -> str:
    names = sorted(os.listdir(path))
    if not names:
        return "it is empty"
    shown = [n + ("/" if os.path.isdir(os.path.join(path, n)) else "") for n in names[:LISTED]]
    more = f" and {len(names) - LISTED} more" if len(names) > LISTED else ""
    return f"it holds {len(names)} {'entry' if len(names) == 1 else 'entries'}: {', '.join(shown)}{more}"


def detect(path) -> str:
    """The kind of directory ``path`` is; an ambiguous or unrecognised one is refused with what was found."""
    found = markers(path)
    if len(found) > 1:
        raise DirectionError(f"path holds more than one kind of model: "
                             f"{'; '.join(_what(k, found[k]) for k in KINDS if k in found)}; give the directory of one")
    if not found:
        raise DirectionError(f"path holds no model the IDE reads: no PEFT adapter ({ADAPTER}), no workspace ({WORKSPACE}) "
                             f"and no safetensors weights (model.safetensors, an index, or *.safetensors files); "
                             f"{_entries(path)}")
    return next(iter(found))


# ------------------------------------------------------------------ stores that read the modified model
class _Over:
    """The headers of ``base`` (a Store), read through ``fn(name, region)``; ``changed`` names hold float32."""

    def __init__(self, base, fn, changed):
        self.base, self.fn, self.changed = base, fn, set(changed)
        self.files = []

    def names(self):
        return self.base.names()

    def __contains__(self, name):
        return name in self.base

    def shape(self, name):
        return self.base.shape(name)

    def numel(self, name):
        return self.base.numel(name)

    def is_float(self, name):
        return self.base.is_float(name)

    def dtype(self, name):
        return "F32" if name in self.changed else self.base.dtype(name)

    def read(self, name, region=()):
        return self.fn(name, region)


# ------------------------------------------------------------------ H4b: a PEFT adapter
def open_adapter(ws, path):
    from . import adapter_export as AE
    from . import peft_adapter as PA
    from .directions_evaluate import Other, _is_tied_head
    if ws.kind != "hf":
        raise DirectionError("a PEFT adapter applies to a transformers checkpoint; this workspace is one of this "
                             "project's own models")
    try:
        adapter = PA.Adapter(path)
        bound = adapter.bind(ws.hf_config, ws.names(), ws.shape, tied=lambda n: _is_tied_head(ws, n),
                             conv1d=lambda n: AE.tensor_reason(ws, n) == AE.CONV1D)
    except PA.AdapterError as e:
        raise DirectionError(str(e)) from None
    base = ws.store
    cache = {}

    def read(name, region=()):
        if not _is_region(region, base.shape(name)):
            region = norm_index(region, base.shape(name))
        t = base.read(name, region)
        e = bound.get(name)
        if e is None:
            return t
        if cache.get("name") != name:              # one module's whole product kept while its slabs are read
            cache.clear()
            cache.update(name=name, whole=adapter.delta(e) if e["out"] * e["in"] <= PA.WHOLE_ELEMS else None)
        d = cache["whole"][region_index(region)] if cache["whole"] is not None else adapter.delta(e, region)
        return t + d                                # PEFT's merge: weight.data += delta_weight, in float32

    cfg = adapter.config
    info = {"kind": "adapter", "peft_type": cfg.get("peft_type"), "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"),
            "use_rslora": bool(cfg.get("use_rslora")), "base_model_name_or_path": cfg.get("base_model_name_or_path"),
            "modules": [{k: e[k] for k in ("module", "param", "r", "alpha", "scaling", "transposed", "dtype")}
                        for e in sorted(bound.values(), key=lambda e: ws.names().index(e["param"]))]}
    notes = []
    named, repo = cfg.get("base_model_name_or_path"), ws.meta.get("repo")
    if isinstance(named, str) and isinstance(repo, str) and named != repo:
        notes.append(f"the adapter names {named} as its base (base_model_name_or_path); it is applied to the open "
                     f"workspace's base, {repo}")
    from .directions_evaluate import _base_dir, _generation, template_text
    return Other(path=path, files=[], store=_Over(base, read, bound), config=getattr(ws, "hf_config", None) or {},
                 generation=_generation(_base_dir(ws), "the open model's generation_config.json"), tokenizer=ws.tok,
                 template=template_text(_base_dir(ws)), kind="adapter", input=info, may_differ=set(bound),
                 notes=notes)


# ------------------------------------------------------------------ H4c: another workspace
def _scaled(piece, s):
    """A piece times a patch strength, so that adding it once (strength 1, as a candidate's pieces are added) does the
    arithmetic the runner does for a committed piece of that strength (``s * piece.dense()`` / ``s * values``)."""
    if piece.kind == "sparse":
        return D.Piece("sparse", piece.region, values=s * piece.values, coords=piece.coords)
    return D.Piece("block", piece.region, values=s * piece.dense())


def _patch_bytes(path) -> int:
    pdir = os.path.join(path, "patches")
    if not os.path.isdir(pdir):
        return 0
    return sum(os.path.getsize(os.path.join(pdir, n)) for n in os.listdir(pdir) if n.endswith(".safetensors")
               and os.path.isfile(os.path.join(pdir, n)))


def open_workspace(ws, path):
    """The other workspace, read-only: its meta checked before it is loaded (GGUF, weights on the Hub, the open
    workspace itself), then Workspace.load, then the same-base decision."""
    from .directions_evaluate import (Other, SCRATCH, _generation, _quantization, _read_json, _tokenizer,
                                      template_text)
    from .workspace import Workspace
    real = os.path.realpath(path)
    if real == os.path.realpath(ws.dir):
        raise DirectionError("path is the open workspace itself: its patches are scored against its base by the scores "
                             "(state current), not here")
    mp = os.path.join(path, WORKSPACE)
    meta = _read_json(mp, f"the workspace's {WORKSPACE}") if os.path.isfile(mp) else {"kind": "wsp"}
    if not isinstance(meta, dict):
        raise DirectionError(f"the workspace's {WORKSPACE} cannot be read: it is not an object")
    kind = meta.get("kind", "wsp")
    if kind == "gguf":
        raise DirectionError("path is a GGUF workspace: its base is quantized and is never dequantized to compare it with "
                             "this base")
    if kind not in ("wsp", "hf"):
        raise DirectionError(f"path is a workspace of kind {json.dumps(kind)}, which is not read")
    if kind == "hf" and meta.get("weights"):
        md = os.path.normpath(os.path.join(path, str(meta.get("model_dir", ""))))
        if any(not os.path.exists(os.path.join(md, str(w))) for w in meta["weights"]):
            raise DirectionError("the workspace at path has weights still on the Hub; nothing is fetched (pull them in "
                                 "that workspace first)")
    held = _patch_bytes(path)
    if held > edit_limit.PATCH_BYTES * edit_limit.SMALL_EDIT:
        _, w, detail = edit_limit._memory(ws, SCRATCH)
        if w is None or held > edit_limit.MEMORY_BYTES * w:
            raise DirectionError(f"the workspace at path holds {edit_limit.gib(held)} of patch files, which are read into "
                                 f"memory to score it, and memory allows "
                                 f"{edit_limit.gib(edit_limit.MEMORY_BYTES * (w or 0))} now")
    try:
        other = Workspace.load(path, metadata_only=True)
    except Exception as e:
        raise DirectionError(f"the workspace at path cannot be read: {type(e).__name__}: {e}") from None
    if other.store.remote:
        raise DirectionError("the workspace at path has weights still on the Hub; nothing is fetched (pull them in "
                             "that workspace first)")
    files = lambda w: {os.path.realpath(getattr(f, "path", f.location)) for f in w.store.files}  # noqa: E731
    same = other.kind == ws.kind and files(other) == files(ws)
    touched = other.touched()
    if same and not touched:
        raise DirectionError("path is a workspace over the open workspace's own base with no enabled patch: there is "
                             "nothing to compare")
    enabled = [p for p in other.patches if p.enabled and p.strength]
    pieces = {n: [pc if s == 1.0 else _scaled(pc, s) for pc, s in ps] for n, ps in other.piece_index().items()} \
        if same else None
    candidate = "patches" if same and not ws.touched() else "difference"
    info = {"kind": "workspace", "name": other.name, "workspace_kind": other.kind, "same_base": same,
            "patches": [{"id": p.id, "message": p.message, "strength": p.strength} for p in enabled],
            "disabled": len(other.patches) - len(enabled), "candidate": candidate}
    notes = []
    if not same:
        notes.append(f"the workspace at path is over another base (its weight files are not this workspace's): its "
                     f"base with its {len(enabled)} enabled patch(es) is compared with this base as a checkpoint")
    read = lambda name, region=(): other.read(name, region, "current", cache=False)  # noqa: E731
    if kind == "hf":
        md = other.model_dir
        config = other.hf_config if isinstance(other.hf_config, dict) else {}
        generation = _generation(md, "the other workspace's generation_config.json")
    else:
        md, generation = other.dir, None
        try:
            with open(os.path.join(md, "config.json")) as f:
                config = json.load(f)
        except (OSError, ValueError):
            config = None
    method = _quantization(config)
    if method:
        raise DirectionError(f"the workspace at path declares quantization ({method}) in its config.json; its integer "
                             f"storage is not read as floating-point weights")
    return Other(path=path, files=[], store=_Over(other.store, read, touched), config=config, generation=generation,
                 tokenizer=_tokenizer(ws, md), template=template_text(md), kind="workspace", input=info,
                 may_differ=set(touched) if same else None, pieces=pieces, notes=notes)
