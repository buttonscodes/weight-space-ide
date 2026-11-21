"""The workspace's patches as a LoRA adapter in PEFT's format (plan item H2; docs/DIRECTIONS.md, "A LoRA adapter").

Heretic's "Export the LoRA only" writes the trial as a PEFT adapter: `adapter_config.json` and
`adapter_model.safetensors`, which `PeftModel.from_pretrained(base, directory)` loads on the base model. This
module writes the same two files from the patches, without importing `peft` (the format is two small, fixed
files, and `peft` would need the base model loaded only to write them), plus `wsp_adapter.json`: where the
adapter came from (the patches, their digests, the base files, the piece kinds per module). PEFT ignores it.

Every piece kind becomes LoRA factors exactly (the table FACTORS), with `delta W = lora_B @ lora_A` and
`lora_alpha = r` per module (scaling 1):

    lowrank  A [m, r] @ B [r, n]   lora_B rows = s * A, lora_A columns = B                          rank r
    block    values [m, n], m <= n lora_B[rows[i], i] = s (one-hot), lora_A row i = values[i]      rank m
    block    values [m, n], m > n  lora_B rows = s * values, lora_A[j, cols[j]] = 1 (one-hot)       rank n
    sparse   densified into the box of its distinct rows and columns, then the block rule

Products with one-hot factors are sums of one value and exact zeros, so a block or sparse change comes back bit
for bit; a strength s that is not a power of two is folded into lora_B (one rounding per value, where the IDE
rounds the scaled product instead), and the module says `exact: false`. Several pieces of one tensor (from one
patch or several) are concatenated along the rank axis in stack order.

Only the modules the patches change are targets, and only linear layers' weight matrices (the table TARGETS);
a GGUF base is refused (the patches were computed against the quantized weights, and the IDE never builds the
dequantized model an adapter would need), and so is one of this project's own models. Nothing here reads a base
weight, writes a weight, a patch file or the workspace's meta; the files go to
`<workspace>/export/adapters/<name>/`, written in a temporary sibling directory and moved in place.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import time
import unicodedata

import numpy as np
import torch

from . import delta as D
from . import directions as DR
from . import store as ST
from .store import NP_DTYPE

VERSION = 1
LAYOUTS = ("auto", "uniform", "per_module")
PEFT_VALIDATED = "0.21.2"                 # the PEFT version the acceptance loaded these files with
VALIDATED_FAMILIES = ("qwen2",)           # model types the acceptance loaded with PEFT; extended only with a run
DIRECTORY = ("export", "adapters")
FIELDS = ("name", "patches", "layout", "base_model_name_or_path", "replace", "dry_run")
RECORDS = ("directions", "records")       # the reproduction records (written by the record capture)
STREAM_ELEMS = 16_000_000                 # the largest box a sparse piece is densified into
SLACK = 1 << 20                           # room for the two JSON files beside the safetensors
MAX_BASE_NAME = 256

CONFIG_FILE = "adapter_config.json"
WEIGHTS_FILE = "adapter_model.safetensors"
PROVENANCE_FILE = "wsp_adapter.json"
KEY = "base_model.model.{module}.lora_{AB}.weight"


class AdapterError(ValueError):
    pass


class AdapterMissing(AdapterError):
    """A patch id that is not in the workspace (the route answers 404)."""


# ------------------------------------------------------------------ which tensors are linear layers' weights
CONV1D = ("a Conv1D weight stored [in, out] (GPT-2 style); PEFT needs fan_in_fan_out for it, which this export "
          "does not write")
UNKNOWN = "not a matrix the IDE recognises in this model"
NOT_2D = "not a 2-D weight matrix"
ROUTED_EXPERT = ("transformers loads a layer's routed experts as one fused parameter, so a LoRA adapter has no module "
                 "to attach to; export the checkpoint instead")
MULTIMODAL = ("the adapter's module names are the checkpoint's (model.language_model...): load it on the model class "
              "the checkpoint names, not on the text-only model")


def _resid_at(axis):
    def rule(part, struct):
        roles = tuple(part.roles)
        if "resid" not in roles:
            return UNKNOWN
        return None if roles.index("resid") == axis else CONV1D
    return rule


def _refused(reason):
    return lambda part, struct: reason


# role -> rule(part, struct) -> None (a linear layer's weight) or the reason it is not. `b_*` covers every bias role.
TARGETS = {
    "W_Q": _resid_at(1), "W_K": _resid_at(1), "W_V": _resid_at(1),
    "W_in": _resid_at(1), "W_gate": _resid_at(1), "W_up": _resid_at(1),
    "W_O": _resid_at(0), "W_out": _resid_at(0),
    "W_U": lambda part, struct: "the unembedding tied to the token embedding" if struct.tied else None,
    "W_E": _refused("the token embedding (PEFT's embedding adapters are not written by this export)"),
    "W_pos": _refused("a position embedding"),
    "norm_w": _refused("a normalisation weight"),
    "norm_b": _refused("a normalisation weight"),
    "b_*": _refused("a bias"),
}


def target_reason(part, struct) -> str | None:
    """None when this part of a tensor is a linear layer's weight under TARGETS, else the reason it is not."""
    rule = TARGETS.get(part.role) or (TARGETS["b_*"] if part.role.startswith("b_") else None)
    return UNKNOWN if rule is None else rule(part, struct)


def tensor_reason(ws, name: str) -> str | None:
    """None when the tensor is one linear module's weight: every part of it accepted by TARGETS (a fused tensor
    such as qkv_proj is one module), 2-D and named `<module>.weight`."""
    parts = ws.struct.parts.get(name) or []
    if not parts or name in ws.struct.other:
        return UNKNOWN
    if _routed_expert(ws.struct, name):
        return ROUTED_EXPERT
    for part in parts:
        reason = target_reason(part, ws.struct)
        if reason is not None:
            return reason
    if len(ws.shape(name)) != 2 or not name.endswith(".weight"):
        return NOT_2D
    return None


def _routed_expert(struct, name: str) -> bool:
    """Whether the tensor is one of a mixture-of-experts layer's routed experts (wsp.arch Layer.experts). The checkpoint
    keeps one tensor per expert, but transformers 5.x fuses a layer's experts into one parameter at load, so there
    is no per-expert module. The shared expert is an ordinary MLP and is not one of them."""
    return any(part.name == name for L in struct.layers for ex in getattr(L, "experts", None) or [] for part in ex.values())


def module_of(name: str) -> str:
    """The module path of a tensor `<module>.weight` (the transformers module of a dense checkpoint's weight)."""
    return name[:-len(".weight")]


# ------------------------------------------------------------------ PEFT matches names on their ending
# PEFT attaches an adapter to every module whose name is a target or ends in "." + a target, and gives a module the
# rank of the first rank_pattern key k (in the dict's order) with re.match(rf"(.*\.)?({k})$", module): an escaped
# key still matches every module whose name ends in "." + the key's module.
def pattern_collision(modules, r: int):
    """The first (a, m), in sorted module order, where target m's name ends in "." + a, a has a rank_pattern entry
    (its rank differs from r) and m's rank is not a's: PEFT would give m a's rank. ``modules``: {module: rank}."""
    ranks = dict(modules)
    names = sorted(ranks)
    for m in names:
        for a in names:
            if a != m and m.endswith("." + a) and ranks[a] != r and ranks[m] != ranks[a]:
                return a, m
    return None


def also_matched(targets, tensors) -> list[tuple[str, str]]:
    """[(a, m)], sorted by m: every module m that is not a target, has a 2-D weight `<m>.weight`, and whose name ends
    in "." + a for a target a (the first such a in sorted order). PEFT attaches an adapter to m too, with no tensors
    in the file for it. ``tensors``: {tensor name: shape}."""
    targets = sorted(set(targets))
    out = []
    for name in sorted(tensors):
        if not name.endswith(".weight") or len(tensors[name]) != 2:
            continue
        m = module_of(name)
        if m in targets:
            continue
        a = next((t for t in targets if m.endswith("." + t)), None)
        if a is not None:
            out.append((a, m))
    return out


MAX_ALSO_NOTES = 5


def _layout(ranks: dict, layout: str):
    """-> (resolved layout, r, rank_pattern, notes) for {module: rank}; rank_pattern keys sorted by module name."""
    r = max(ranks.values())
    resolved = layout if layout != "auto" else ("uniform" if len(set(ranks.values())) == 1 else "per_module")
    notes = []
    if resolved == "per_module":
        hit = pattern_collision(ranks, r)
        if hit is not None:
            a, m = hit
            if layout == "per_module":
                raise AdapterError(f"layout 'per_module' cannot be written: PEFT matches a rank pattern on the end of a "
                                   f"module's name, and {m} ends in .{a} with another rank; use 'uniform' or 'auto'")
            resolved = "uniform"
            notes.append(f"module names {a} and {m} end alike and have different ranks, which PEFT's rank patterns "
                         f"cannot tell apart: the uniform layout is used")
    rank_pattern = {} if resolved == "uniform" else \
        {re.escape(m): ranks[m] for m in sorted(ranks) if ranks[m] != r}
    return resolved, r, rank_pattern, notes


def _also_notes(pairs) -> list[str]:
    notes = [f"module {m} is not changed by these patches but its name ends in .{a}: PEFT will attach an empty adapter "
             f"to it too (it changes nothing) and warn that its weights are missing" for a, m in pairs[:MAX_ALSO_NOTES]]
    if len(pairs) > MAX_ALSO_NOTES:
        notes.append(f"and {len(pairs) - MAX_ALSO_NOTES} more")
    return notes


# ------------------------------------------------------------------ pieces to factors (the table FACTORS)
def _axis(r) -> list[int]:
    return [r] if isinstance(r, int) else list(r)


def _power_of_two(s: float) -> bool:
    return s != 0 and math.isfinite(s) and math.frexp(abs(s))[0] == 0.5


def _sparse_box(piece):
    """(distinct rows, distinct columns) of a sparse piece's changed weights."""
    c = piece.coords
    return torch.unique(c[:, 0]), torch.unique(c[:, 1])


def _lowrank_rank(piece) -> int:
    return int(piece.A.shape[1])


def _block_rank(piece) -> int:
    m, n = (len(_axis(r)) for r in piece.region)
    return m if m <= n else n


def _sparse_rank(piece) -> int:
    R, C = _sparse_box(piece)
    return min(len(R), len(C))


def _place_block(lora_B, lora_A, k0, rows, cols, values, s):
    m, n = len(rows), len(cols)
    ri, ci = torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)
    values = values.reshape(m, n).to(torch.float32)
    if m <= n:
        lora_B[ri, torch.arange(k0, k0 + m)] = float(s)
        lora_A[k0:k0 + m, ci] = values
    else:
        lora_B[ri, k0:k0 + n] = s * values
        lora_A[torch.arange(k0, k0 + n), ci] = 1.0


def _lowrank_factors(lora_B, lora_A, k0, piece, s):
    rows, cols = (_axis(r) for r in piece.region)
    r = piece.A.shape[1]
    lora_B[torch.tensor(rows, dtype=torch.long), k0:k0 + r] = s * piece.A.to(torch.float32)
    lora_A[k0:k0 + r, torch.tensor(cols, dtype=torch.long)] = piece.B.to(torch.float32)


def _block_factors(lora_B, lora_A, k0, piece, s):
    rows, cols = (_axis(r) for r in piece.region)
    _place_block(lora_B, lora_A, k0, rows, cols, piece.values, s)


def _sparse_factors(lora_B, lora_A, k0, piece, s):
    R, C = _sparse_box(piece)
    box = torch.zeros(len(R), len(C))
    i = torch.searchsorted(R, piece.coords[:, 0].contiguous())
    j = torch.searchsorted(C, piece.coords[:, 1].contiguous())
    box.index_put_((i, j), piece.values.to(torch.float32), accumulate=True)
    _place_block(lora_B, lora_A, k0, R.tolist(), C.tolist(), box, s)


# piece kind -> (its rank in the adapter, writer(lora_B, lora_A, first rank index, piece, strength))
FACTORS = {
    "lowrank": (_lowrank_rank, _lowrank_factors),
    "block": (_block_rank, _block_factors),
    "sparse": (_sparse_rank, _sparse_factors),
}


def factors(shape, pieces, k: int):
    """The (lora_B [out, k], lora_A [k, in]) of one module from its [(piece, strength)] in order, zero beyond the
    pieces' ranks (the uniform layout's padding)."""
    out, inp = shape
    lora_B, lora_A = torch.zeros(out, k), torch.zeros(k, inp)
    k0 = 0
    for piece, s in pieces:
        rank, write = FACTORS[piece.kind]
        write(lora_B, lora_A, k0, piece, s)
        k0 += rank(piece)
    return lora_B, lora_A


# ------------------------------------------------------------------ files the record capture writes (read only)
def record_of(ws, patch) -> dict | None:
    """The patch's reproduction record, `<workspace>/directions/records/<id>.json`, when there is one for this
    patch (same id and commit time; an id given again after a deletion has a stale record), else None."""
    path = os.path.join(ws.dir, *RECORDS, f"{patch.id}.json")
    try:
        with open(path) as f:
            rec = json.load(f)
        meta = rec.get("patch") or {}
        if meta.get("id") == patch.id and float(meta.get("created")) == float(patch.created):
            return rec
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def cached_base_hashes(ws) -> dict:
    """{file name: SHA-256} of the base files whose hash the record capture cached in
    `<workspace>/directions/records/base.json` (`{"files": [[name, bytes, mtime_ns, sha256], ...]}`) and that are
    unchanged since (same size and modification time). The export never hashes the base itself."""
    try:
        with open(os.path.join(ws.dir, *RECORDS, "base.json")) as f:
            rows = json.load(f).get("files") or []
    except (OSError, ValueError, AttributeError):
        return {}
    out = {}
    for f in ws.store.files:
        if f.remote:
            continue
        try:
            st = os.stat(f.path)
        except OSError:
            continue
        name = os.path.basename(f.path)
        for row in rows:
            if isinstance(row, list) and len(row) == 4 and row[:3] == [name, st.st_size, st.st_mtime_ns] and row[3]:
                out[name] = row[3]
    return out


def base_files(ws) -> list[dict]:
    hashes = cached_base_hashes(ws)
    out = []
    for f in ws.store.files:
        name = os.path.basename(f.location.split("?")[0])
        if f.remote:
            size = f.data_start + max((m["data_offsets"][1] for m in f.tensors.values()), default=0)
        else:
            size = os.path.getsize(f.path)
        out.append({"name": name, "bytes": int(size), "sha256": None if f.remote else hashes.get(name)})
    return out


# ------------------------------------------------------------------ checks
def _num(x) -> str:
    return repr(float(x))


def check_layout(layout) -> str:
    if layout not in LAYOUTS:
        raise AdapterError("layout must be 'auto', 'uniform' or 'per_module'")
    return layout


def check_base_name(value):
    if value is None:
        return None
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_BASE_NAME or \
            any(unicodedata.category(c) == "Cc" for c in value):
        raise AdapterError("base_model_name_or_path must be text of 1..256 characters without control characters, or null")
    return value


def check_name(name) -> str:
    if not isinstance(name, str) or not DR.NAME.match(name):
        raise AdapterError(f"name must match {DR.NAME.pattern}")
    return name


def _check_kind(ws) -> None:
    if ws.kind == "wsp":
        raise AdapterError("a PEFT adapter is for a transformers checkpoint; this is one of this project's own models: "
                           "use the existing export (GET /api/export)")
    if ws.kind == "gguf":
        raise AdapterError("this workspace's base is a quantized GGUF file: a PEFT adapter applies to a floating-point "
                           "transformers checkpoint, and these patches were computed against the quantized weights, so "
                           "the adapter's base would be the dequantized model, which the IDE never builds. Export the "
                           "WSP companion bundle (GET /api/export), or apply the edit to the floating-point checkpoint "
                           "the GGUF was made from")
    if ws.kind != "hf":
        raise AdapterError(f"a PEFT adapter is for a transformers checkpoint; this workspace is of kind {ws.kind!r}")
    for n in ws.names():
        if not ws.store.is_float(n):
            raise AdapterError(f"this checkpoint stores packed or scaled weights ({ws.store.dtype(n)} in {n}), which "
                               f"the IDE does not decode; no adapter can be exported")


def _select(ws, patches):
    """[(patch, strength, enabled)] in stack order, and the notes about switched-off patches."""
    if patches is not None and (not isinstance(patches, list) or not all(isinstance(p, str) for p in patches)
                                or len(set(patches)) != len(patches)):
        raise AdapterError("patches must be null or a list of distinct patch ids")
    with ws.lock:
        stack = [(p, float(p.strength), bool(p.enabled)) for p in ws.patches]
    if patches is None:
        chosen = [e for e in stack if e[2] and e[1] != 0]
        if not chosen:
            raise AdapterError("nothing to export: no patch is enabled")
        return chosen, []
    if not patches:
        raise AdapterError("nothing to export: patches is empty")
    ids = {e[0].id for e in stack}
    for pid in patches:
        if pid not in ids:
            raise AdapterMissing(f"no patch '{pid}'")
    chosen = [e for e in stack if e[0].id in set(patches)]
    notes = []
    for p, s, enabled in chosen:
        if s == 0:
            raise AdapterError(f"patch {p.id} has strength 0: it changes nothing")
        if not enabled:
            notes.append(f"patch {p.id} is switched off in the workspace; the adapter includes it at strength {_num(s)}")
    return chosen, notes


# ------------------------------------------------------------------ the plan
def _plan(ws, patches=None, layout="auto", base_model_name_or_path=None) -> tuple[dict, dict]:
    """-> (the public answer, the internal plan {module: (tensor, shape, [(piece, strength)], k)})."""
    layout = check_layout(layout)
    request_base = check_base_name(base_model_name_or_path)
    _check_kind(ws)
    chosen, notes = _select(ws, patches)
    per_tensor: dict[str, list] = {}
    rows: dict[str, list] = {}
    for p, s, _ in chosen:
        for name, pieces in p.deltas.items():
            pieces = [pc for pc in pieces if not pc.is_zero()]
            if not pieces:
                continue
            reason = tensor_reason(ws, name) if name in ws.store else UNKNOWN
            if reason is not None:
                raise AdapterError(f"patch {p.id} changes {name}, {reason}: a LoRA adapter holds changes of linear "
                                   f"layers' weight matrices only")
            for pc in pieces:
                if pc.kind == "sparse":
                    R, C = _sparse_box(pc)
                    if len(R) * len(C) > STREAM_ELEMS:
                        raise AdapterError(f"patch {p.id} changes {name} at {pc.values.numel()} scattered weights spread "
                                           f"over {len(R)} rows and {len(C)} columns: too wide to factor exactly")
                k = FACTORS[pc.kind][0](pc)
                per_tensor.setdefault(name, []).append((pc, s))
                rows.setdefault(name, []).append({"patch": p.id, "kind": pc.kind, "rank": k, "strength": s})
    if not per_tensor:
        raise AdapterError("nothing to export: the selected patches change no weight")
    modules, internal = [], {}
    for name in sorted(per_tensor, key=module_of):
        shape = tuple(ws.shape(name))
        k = sum(r["rank"] for r in rows[name])
        if k > min(shape):
            raise AdapterError(f"{name} would need rank {k} in the adapter, above min(out, in) = {min(shape)}: "
                               f"export fewer patches at once")
        m = module_of(name)
        exact = all(_power_of_two(r["strength"]) for r in rows[name])
        modules.append({"module": m, "tensor": name, "shape": list(shape), "rank": k, "exact": exact,
                        "pieces": rows[name]})
        internal[m] = (name, shape, per_tensor[name], k)
    resolved, r, rank_pattern, layout_notes = _layout({e["module"]: e["rank"] for e in modules}, layout)
    notes += layout_notes
    notes += _also_notes(also_matched(internal, {n: ws.shape(n) for n in ws.names()}))
    repo, sha = ws.meta.get("repo") or None, ws.meta.get("sha") or None
    base_name = request_base if request_base is not None else repo
    revision = sha if base_name is not None and base_name == repo else None
    exported = {p.id for p, _, _ in chosen}
    if base_name is None:
        notes.insert(0, "no Hub repository is known for this checkpoint, so base_model_name_or_path is null: pass it, "
                        "or load the base yourself and give PeftModel.from_pretrained this directory")
    patch_rows = []
    from . import directions_search as DSR
    for p, s, enabled in chosen:
        rec = record_of(ws, p)
        if rec is not None:
            below = [str(e[0]) for e in rec.get("made_on") or [] if isinstance(e, list) and e and e[0] not in exported]
            if below:
                notes.append(f"patch {p.id} was made on top of patches {', '.join(below)}, which this adapter does "
                             f"not include")
        patch_rows.append({"id": p.id, "created": p.created, "message": p.message, "strength": s, "enabled": enabled,
                           "digest": DSR.piece_digest(p.deltas), "record": rec is not None})
    for p, s, _ in chosen:
        if not _power_of_two(s):
            notes.append(f"patch {p.id} has strength {_num(s)}: its factors are scaled, which rounds each value once "
                         f"(the IDE scales the product instead)")
    if isinstance((ws.hf_config or {}).get("text_config"), dict):
        notes.append(MULTIMODAL)
    family = ws.struct.family
    validated = family in VALIDATED_FAMILIES
    if not validated:
        notes.append(f"this model type ({family}) has not been loaded with PEFT by the IDE's acceptance; PEFT refuses "
                     f"an adapter whose module names do not exist")
    items = _items(modules, resolved, r)
    answer = {"dry_run": True, "written": False, "files": [], "layout": resolved, "r": r, "rank_pattern": rank_pattern,
              "bytes": _safetensors_bytes(items), "modules": modules, "patches": patch_rows,
              "base_model_name_or_path": base_name, "revision": revision, "family": family,
              "validated_family": validated, "notes": notes}
    return answer, internal


def _items(modules, layout, r) -> list[tuple[str, str, tuple]]:
    """(key, dtype, shape) of every tensor of adapter_model.safetensors, keys sorted."""
    items = []
    for e in modules:
        out, inp = e["shape"]
        k = r if layout == "uniform" else e["rank"]
        items.append((KEY.format(module=e["module"], AB="A"), "F32", (k, inp)))
        items.append((KEY.format(module=e["module"], AB="B"), "F32", (out, k)))
    return sorted(items)


METADATA = {"format": "pt"}


def _safetensors_bytes(items) -> int:
    """The size store.write_safetensors gives these tensors (the same header, padded to 8 bytes)."""
    header, off = {}, 0
    for name, dtype, shape in items:
        n = int(np.prod(shape)) * np.dtype(NP_DTYPE[dtype]).itemsize
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {k: str(v) for k, v in METADATA.items()}
    blob = len(json.dumps(header, separators=(",", ":")).encode())
    return 8 + blob + (8 - blob % 8) % 8 + off


def config(answer) -> dict:
    """adapter_config.json: every key that changes PEFT's forward pass set explicitly, plus identity keys."""
    return {"alpha_pattern": dict(answer["rank_pattern"]), "base_model_name_or_path": answer["base_model_name_or_path"],
            "bias": "none", "exclude_modules": None, "fan_in_fan_out": False, "inference_mode": True,
            "init_lora_weights": True, "layers_pattern": None, "layers_to_transform": None, "lora_alpha": answer["r"],
            "lora_bias": False, "lora_dropout": 0.0, "modules_to_save": None, "peft_type": "LORA", "r": answer["r"],
            "rank_pattern": dict(answer["rank_pattern"]), "revision": answer["revision"],
            "target_modules": sorted(e["module"] for e in answer["modules"]), "target_parameters": None,
            "task_type": "CAUSAL_LM", "use_dora": False, "use_rslora": False}


def plan(ws, patches=None, layout="auto", base_model_name_or_path=None) -> dict:
    """What `export` would write, and nothing written."""
    return _plan(ws, patches, layout, base_model_name_or_path)[0]


# ------------------------------------------------------------------ writing
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _file(path) -> dict:
    return {"name": os.path.basename(path), "bytes": os.path.getsize(path), "sha256": _sha256(path)}


def _write(tmp, ws, answer, internal) -> None:
    items = _items(answer["modules"], answer["layout"], answer["r"])
    kept: dict = {}

    def data(key):
        m, ab = key[len("base_model.model."):].rsplit(".lora_", 1)
        if kept.get("module") != m:                     # one module's two factors in memory at a time
            name, shape, pieces, k = internal[m]
            kept.clear()
            kept["module"] = m
            kept["B"], kept["A"] = factors(shape, pieces, answer["r"] if answer["layout"] == "uniform" else k)
        return kept[ab[0]].contiguous().numpy().tobytes()

    ST.write_safetensors(os.path.join(tmp, WEIGHTS_FILE), items, data, METADATA)
    kept.clear()
    with open(os.path.join(tmp, CONFIG_FILE), "w") as f:
        f.write(json.dumps(config(answer), indent=2, sort_keys=True))
    files = {fn: {k: v for k, v in _file(os.path.join(tmp, fn)).items() if k != "name"} for fn in (CONFIG_FILE, WEIGHTS_FILE)}
    from . import directions_search as DSR
    provenance = {"version": VERSION, "format": "peft-lora", "peft_validated": PEFT_VALIDATED, "heretic": DSR.HERETIC,
                  "created": time.time(), "workspace": ws.name, "family": answer["family"],
                  "validated_family": answer["validated_family"],
                  "base": {"repo": ws.meta.get("repo") or None, "sha": ws.meta.get("sha") or None, "files": base_files(ws)},
                  "patches": answer["patches"], "layout": answer["layout"], "r": answer["r"], "modules": answer["modules"],
                  "files": files, "notes": answer["notes"]}
    with open(os.path.join(tmp, PROVENANCE_FILE), "w") as f:
        f.write(json.dumps(provenance, indent=1, sort_keys=True))


def export(ws, name, patches=None, layout="auto", base_model_name_or_path=None, replace=False) -> dict:
    """Write the adapter to `<workspace>/export/adapters/<name>/` -> plan's answer plus dir, files, written."""
    name = check_name(name)
    answer, internal = _plan(ws, patches, layout, base_model_name_or_path)
    parent = os.path.join(ws.dir, *DIRECTORY)
    target = os.path.join(parent, name)
    if os.path.lexists(target) and not replace:
        raise AdapterError(f"an adapter named '{name}' exists in this workspace; pass replace, or choose another name")
    need = answer["bytes"] + SLACK
    free = DR._free_bytes(ws.dir)
    if need > free - DR.DISK_RESERVE:
        raise AdapterError(f"the adapter needs {need:,} bytes; the workspace's file system has {free:,} bytes free, "
                           f"of which {DR.DISK_RESERVE:,} stay free")
    os.makedirs(parent, exist_ok=True)
    stem = f".{name}.{os.getpid()}.{secrets.token_hex(4)}"
    tmp = os.path.join(parent, stem + ".tmp")
    os.mkdir(tmp)
    old = None
    try:
        _write(tmp, ws, answer, internal)
        files = [_file(os.path.join(tmp, fn)) for fn in (CONFIG_FILE, WEIGHTS_FILE, PROVENANCE_FILE)]
        if os.path.lexists(target):
            old = os.path.join(parent, stem + ".old")
            os.replace(target, old)
        try:
            os.replace(tmp, target)
        except BaseException:
            if old is not None:
                os.replace(old, target)
            raise
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
    return {**answer, "dry_run": False, "written": True, "dir": target, "files": files}


def request(ws, body) -> dict:
    """POST /api/export_adapter: the body checked, then `plan` (dry_run) or `export`."""
    if not isinstance(body, dict):
        raise AdapterError("the request must be an object")
    for field in body:
        if field not in FIELDS:
            raise AdapterError(f"{field}: unknown field; the fields are {', '.join(FIELDS[:-1])} and {FIELDS[-1]}")
    dry_run, replace = body.get("dry_run", False), body.get("replace", False)
    for field, v in (("dry_run", dry_run), ("replace", replace)):
        if not isinstance(v, bool):
            raise AdapterError(f"{field} must be true or false")
    kw = dict(patches=body.get("patches"), layout="auto" if body.get("layout") is None else body["layout"],
              base_model_name_or_path=body.get("base_model_name_or_path"))
    if dry_run:
        if body.get("name") is not None:
            check_name(body["name"])
        return plan(ws, **kw)
    return export(ws, body.get("name"), replace=replace, **kw)
