"""Steering: push neurons and heads while the model runs, without changing a weight.

A steering specification (contract C3) is a JSON list; each entry pushes one target:

    {"addr": "L34N3838",      "mode": "add", "dose": 0.75, "unit": "resid"}   add to a neuron's activation
    {"addr": "L34N3838",      "mode": "mul", "dose": 2}                       multiply a neuron's activation
    {"addr": "blocks.12.mlp", "mode": "mul", "dose": 0}                       multiply a whole MLP's activations
    {"addr": "L11H3",         "mode": "mul", "dose": 0}                       multiply one head's output (attn.hook_z)
    {"addr": "group:refusal", "mode": "add", "dose": 0.5}                     every neuron of a group
    {"addr": "direction:refusal@12", "mode": "add", "dose": -1, "unit": "diff"}  a measured direction

A ``direction:<measurement>@<k>`` entry adds a vector to the residual stream entering layer k
(``blocks.k.hook_resid_pre``, where the measurement read its index k): ``c = dose * scale * v``
with ``v`` row k of the measurement's direction table (``directions.table``, with
``orthogonalize_direction``, default true) and ``scale`` the separation of the bad mean from the
good mean along ``v`` (``unit: "diff"``, the default) or 1 (``unit: "raw"``); ``dose < 0`` moves
towards the good mean.  An optional ``means_sha256`` pins the measurement's means file.  The
vector is resolved here, on the server, and crosses to the worker as numbers (``add_vector``).

``add`` on a neuron works at ``blocks.L.mlp.hook_post`` (the input of the MLP's output
projection), at every position.  With ``unit: "resid"`` (the default) the constant is
``dose * p50(L) / ||w_out column||``: p50(L) is the median residual-stream norm leaving layer L
(position 0 excluded), measured once on the base model; the column norm is read from the
logical weights of the variant being run.  ``unit: "raw"`` adds ``dose``.

Relation to neuron-explorer (``explorer/runtime.py``, ``_push_entry`` / ``layer_resid_p50``):
the same formula and the same reference text (``REFERENCE_TEXT`` below, used verbatim), tokenized
the same way, so a dose here equals a dose on neuron-explorer's path *without* sweep statistics.
When neuron-explorer has sweep statistics it takes p50 from them (``resid_norm_q10_50_90[1]``);
a recipe made that way differs from ours by the ratio of the two medians, unless that p50 is
given explicitly (``prepare(..., p50={layer: value})``, reported as ``source: "given"``).
Two documented differences remain: a neuron that is both multiplied and pushed is ``x * m + c``
here (neuron-explorer applies its pushes in list order), and ``probe_reads`` reports values after
a same-layer push (neuron-explorer's ``read()`` captures before it).

The specification is normalised into a compact *wire form* that crosses the model-worker
boundary as JSON (``normalize`` / ``prepare``), resolved into per-layer vectors where the
weights are (``resolve``), and applied as one vectorised hook per layer and hook point
(``make_hooks``):

    {"version": 1,
     "layers": {"34": {"add_resid": [[3838], [0.75]],      neuron indices, doses (summed per neuron)
                       "add_raw":   [[17], [1.5]],
                       "mul":       [[3838], [2.0]],       multipliers (multiplied per neuron)
                       "mlp_mul":   0.5,                   whole-MLP multiplier (product)
                       "head_mul":  [[3], [0.0]],          head multipliers
                       "add_vector": [0.1, ...]}},         d_model numbers added at hook_resid_pre (directions)
     "p50": {"34": 41.2},                                  only layers that have add_resid
     "calibration": "<key>"}

At a layer the MLP hook computes ``x * m + c`` (m: the whole-MLP multiplier times the
per-neuron multipliers; c: the resolved constants), so a neuron that is both multiplied and
pushed is multiplied first.  Only the ``current`` variant is ever steered.  Nothing here
writes a weight or records a patch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from contextlib import contextmanager

import torch

from .runtime_events import job_scope
from .worker_jobs import worker_job

VERSION = 1
MAX_ENTRIES = 256
MAX_NEURONS = 131072
MAX_DOSE = 1e6
MAX_ADDR = 200
READ_WINDOW = 4_000_000          # logical values per weight read when measuring output-column norms
READ_GAP = 64                    # unquantized storage: neurons closer than this share one read
MAX_P50 = 1e9
MAX_VECTOR = 1e9                 # |component| of a residual-stream push (add_vector)
DIRECTION_UNITS = ('diff', 'raw')  # diff: dose x the means' separation along the direction; raw: dose x the unit direction
CALIBRATION_FILE = os.path.join('steering', 'calibration.json')
CALIBRATION_KEEP = 8             # calibrations kept per workspace (different models / texts)
PROGRESS_AFTER = 0.5             # seconds before a column-norm read starts publishing progress
PROGRESS_EVERY = 0.25            # and then at most this often

# The calibration text of neuron-explorer, verbatim, so that a resid dose means the same thing in
# both tools.  Source: neuron-explorer, explorer/runtime.py, REFERENCE_TEXT.
# MIT License, Copyright (c) 2026 lyramakesmusic.
# It is tokenized as neuron-explorer's layer_resid_p50 does: the tokenizer's own special tokens
# (transformers' default add_special_tokens=True), no truncation (unless the model's context is
# shorter; reference_tokens reports it), position 0 excluded from the median.
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)
REFERENCE_SOURCE = 'neuron-explorer explorer/runtime.py REFERENCE_TEXT (MIT, Copyright (c) 2026 lyramakesmusic)'


class SteeringError(ValueError):
    """A steering specification that cannot be applied; the message names the entry."""


# ------------------------------------------------------------------ targets
_SPACED = re.compile(r"^L(\d+)\s+([NH])(\d+)$")
_NEURON = re.compile(r"^blocks\.(\d+)\.mlp\.neuron\.(\d+)$")
_RANGE = re.compile(r"^blocks\.(\d+)\.mlp\.neurons\.(\d+)-(\d+)$")
_MLP = re.compile(r"^blocks\.(\d+)\.mlp$")
_HEAD = re.compile(r"^blocks\.(\d+)\.attn\.head\.(\d+)$")
_DIRECTION = re.compile(r"^direction:([A-Za-z0-9][A-Za-z0-9._-]{0,63})@(0|[1-9][0-9]*)$", re.ASCII)
DIRECTION_HELP = "a direction is written direction:<measurement>@<index>, for example direction:refusal@12"


def _canonical(addr: str) -> str:
    from .addressing import normalize
    addr = addr.strip()
    m = _SPACED.match(addr)
    if m:
        addr = f"L{m.group(1)}{m.group(2)}{m.group(3)}"
    return normalize(addr)


def _layer_types(ws) -> list | None:
    cfg = getattr(ws, 'hf_config', None) or {}
    types = cfg.get('layer_types') or (cfg.get('text_config') or {}).get('layer_types')
    return list(types) if isinstance(types, (list, tuple)) else None


def neurons_hookable(ws, l: int) -> bool:
    """The layer has neurons and a hook point at the input of its output projection."""
    s = ws.struct
    if not 0 <= l < s.n_layers or not s.has_neurons(l):
        return False
    return ws.kind == 'wsp' or bool(s.layers[l].modules.get('mlp_out'))


def resid_hookable(ws, l: int) -> bool:
    """The layer has a hook point at its input (blocks.l.hook_resid_pre: the decoder layer's forward pre-hook)."""
    s = ws.struct
    if not 0 <= l < s.n_layers:
        return False
    return ws.kind == 'wsp' or bool(s.layers[l].modules.get('layer'))


def head_problem(ws, l: int) -> str | None:
    """None when layer l's heads can be multiplied at attn.hook_z; else why not."""
    s = ws.struct
    L = s.layers[l]
    types = _layer_types(ws)
    if (types and l < len(types) and 'linear' in str(types[l])) or str(L.modules.get('attn', '')).endswith('linear_attn'):
        return f"layer {l} uses linear attention, which has no per-head output to multiply"
    if not s.has_heads(l):
        return f"layer {l} has no recognised attention heads"
    if ws.kind != 'wsp' and not L.modules.get('o_proj'):
        return f"layer {l} has no attn.hook_z hook point (no separate output projection)"
    return None


SKIPPED_SHOWN = 5          # skipped group members listed by name in an entry's report


def group_neurons(ws, name: str) -> tuple[list[tuple[int, int]], dict | None]:
    """A group's neurons as sorted (layer, index) pairs, and what was not pushed (None if nothing).

    The expansion is the selections module's (contract C2: the same one `neurons_of` uses): nested groups
    (each once, so a cycle stops), ranges expanded. Heads and tensor slices are not neurons and members that
    no longer resolve are left out; both are counted in the second value so the report can say so."""
    from .selections import expand
    sel = expand(ws, 'group:' + name)                 # an unknown group raises KeyError
    pairs = [(int(l), int(n)) for l in sorted(sel.neurons) for n in sorted(sel.neurons[l])]
    if not (sel.other or sel.heads or sel.slices):
        return pairs, None
    skipped = {'unresolved': len(sel.other), 'heads': len(sel.heads), 'slices': len(sel.slices)}
    if sel.other:
        skipped['members'] = [dict(o) for o in sel.other[:SKIPPED_SHOWN]]
    return pairs, skipped


def skipped_text(skipped: dict) -> str:
    """'2 members that no longer resolve, 1 head' -- what a group entry does not push."""
    parts = []
    for key, one, many in (('unresolved', 'member that no longer resolves', 'members that no longer resolve'),
                           ('heads', 'head', 'heads'), ('slices', 'tensor slice', 'tensor slices')):
        n = skipped.get(key, 0)
        if n:
            parts.append(f"{n} {one if n == 1 else many}")
    return ', '.join(parts)


def target(ws, addr: str, report: dict | None = None):
    """An address -> ('neurons', [(layer, index), ...]) | ('mlp', layer) | ('head', layer, head).

    For a group, `report` (when given) receives 'skipped': what the group holds that is not pushed.
    A direction ``direction:<measurement>@<k>`` -> ('direction', measurement, k); only the form is checked here.
    Raises SteeringError (without an entry prefix) when the address is not steerable here."""
    s = ws.struct
    if addr.strip().startswith('direction:'):
        m = _DIRECTION.match(addr.strip())
        if not m:
            raise SteeringError(DIRECTION_HELP)
        return ('direction', m.group(1), int(m.group(2)))
    a = _canonical(addr)

    def layer_ok(l):
        if not 0 <= l < s.n_layers:
            raise SteeringError(f"layer {l} does not exist (the model has {s.n_layers})")

    if a.startswith('group:'):
        try:
            pairs, skipped = group_neurons(ws, a[6:])
        except KeyError:
            raise SteeringError(f"no group named {a[6:]!r}") from None
        if not pairs:
            raise SteeringError(f"group {a[6:]!r} has no neurons to push"
                                + (f" (it holds {skipped_text(skipped)})" if skipped else ""))
        if report is not None and skipped:
            report['skipped'] = skipped
        if len(pairs) > MAX_NEURONS:
            raise SteeringError(f"group {a[6:]!r} has {len(pairs)} neurons; at most {MAX_NEURONS} can be pushed")
        for l, n in pairs:
            _neuron_ok(ws, l, n)
        return ('neurons', pairs)
    m = _NEURON.match(a)
    if m:
        l, n = int(m.group(1)), int(m.group(2))
        _neuron_ok(ws, l, n)
        return ('neurons', [(l, n)])
    m = _RANGE.match(a)
    if m:
        l, lo, hi = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if hi < lo:
            raise SteeringError(f"neuron range {lo}-{hi} is empty")
        if hi - lo + 1 > MAX_NEURONS:
            raise SteeringError(f"range has {hi - lo + 1} neurons; at most {MAX_NEURONS} can be pushed")
        _neuron_ok(ws, l, lo)
        _neuron_ok(ws, l, hi)
        return ('neurons', [(l, n) for n in range(lo, hi + 1)])
    m = _MLP.match(a)
    if m:
        l = int(m.group(1))
        layer_ok(l)
        if not neurons_hookable(ws, l):
            raise SteeringError(f"layer {l} has no MLP activations that can be steered")
        return ('mlp', l)
    m = _HEAD.match(a)
    if m:
        l, h = int(m.group(1)), int(m.group(2))
        layer_ok(l)
        why = head_problem(ws, l)
        if why:
            raise SteeringError(why)
        if not 0 <= h < s.n_heads:
            raise SteeringError(f"head {h} does not exist (layer {l} has {s.n_heads})")
        return ('head', l, h)
    try:
        ws.resolve(addr)
    except Exception:
        raise SteeringError("unknown address") from None
    raise SteeringError("cannot be steered: only neurons (L3N17, ranges, groups), whole MLPs (blocks.3.mlp) "
                        "and heads (L3H5) can be pushed")


def _neuron_ok(ws, l, n):
    s = ws.struct
    if not 0 <= l < s.n_layers:
        raise SteeringError(f"layer {l} does not exist (the model has {s.n_layers})")
    if not neurons_hookable(ws, l):
        raise SteeringError(f"layer {l} has no MLP neurons that can be steered")
    width = s.mlp_width(l)
    if not 0 <= n < width:
        raise SteeringError(f"neuron {n} does not exist (layer {l} has {width})")


# ------------------------------------------------------------------ the specification -> wire form
def _dose(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SteeringError(f"{where}: dose must be a number")
    try:
        value = float(value)
    except OverflowError:
        raise SteeringError(f"{where}: dose is larger than {MAX_DOSE:g}") from None
    if not math.isfinite(value):
        raise SteeringError(f"{where}: dose must be a finite number")
    if abs(value) > MAX_DOSE:
        raise SteeringError(f"{where}: dose {value:g} is larger than {MAX_DOSE:g}")
    return value


def normalize(ws, spec):
    """A C3 list -> (wire without p50, entries). Raises SteeringError naming the entry."""
    if spec is None:
        spec = []
    if not isinstance(spec, list):
        raise SteeringError("steer must be a list of entries")
    if len(spec) > MAX_ENTRIES:
        raise SteeringError(f"steer has {len(spec)} entries; at most {MAX_ENTRIES}")
    layers: dict[int, dict] = {}
    entries, total = [], 0
    for i, e in enumerate(spec):
        where = f"steer entry {i + 1}"
        if not isinstance(e, dict):
            raise SteeringError(f"{where}: must be an object with addr, mode and dose")
        addr = e.get('addr')
        if not isinstance(addr, str) or not addr.strip():
            raise SteeringError(f"{where}: addr must be a nonempty address")
        if len(addr) > MAX_ADDR:
            raise SteeringError(f"{where}: addr is longer than {MAX_ADDR} characters")
        where = f"steer entry {i + 1} ({addr.strip()})"
        if addr.strip().startswith('direction:'):
            entries.append(_direction_entry(ws, e, i, addr.strip(), where, layers))
            continue
        unknown = sorted(set(e) - {'addr', 'mode', 'dose', 'unit'})
        if unknown:
            raise SteeringError(f"{where}: unknown field {', '.join(unknown)}")
        mode = e.get('mode')
        if mode not in ('add', 'mul'):
            raise SteeringError(f"{where}: mode must be 'add' or 'mul'")
        if 'dose' not in e:
            raise SteeringError(f"{where}: dose is missing")
        dose = _dose(e['dose'], where)
        unit = e.get('unit')
        if mode == 'mul' and unit is not None:
            raise SteeringError(f"{where}: unit applies to add only; mul multiplies by dose")
        if mode == 'add':
            unit = 'resid' if unit is None else unit
            if unit not in ('resid', 'raw'):
                raise SteeringError(f"{where}: unit must be 'resid' or 'raw'")
        found = {}
        try:
            tgt = target(ws, addr, found)
        except SteeringError as exc:
            raise SteeringError(f"{where}: {exc}") from None
        kind = tgt[0]
        if mode == 'add' and kind != 'neurons':
            raise SteeringError(f"{where}: add pushes neurons only; use mul for a whole MLP or a head")
        info = dict(index=i, addr=addr.strip(), mode=mode, dose=dose, kind=kind)
        if unit:
            info['unit'] = unit
        if kind == 'neurons':
            pairs = tgt[1]
            total += len(pairs)
            if total > MAX_NEURONS:
                raise SteeringError(f"{where}: the specification pushes more than {MAX_NEURONS} neurons")
            per = {}
            for l, n in pairs:
                per.setdefault(l, []).append(n)
                slot = layers.setdefault(l, {})
                if mode == 'mul':
                    t = slot.setdefault('mul', {})
                    t[n] = t.get(n, 1.0) * dose
                else:
                    t = slot.setdefault('add_' + unit, {})
                    t[n] = t.get(n, 0.0) + dose
            info['neurons'] = len(pairs)
            info['layers'] = {str(l): len(v) for l, v in sorted(per.items())}
            info['targets'] = [f"L{l}N{n}" for l, n in pairs[:32]]
            if found.get('skipped'):
                info['skipped'] = found['skipped']
        elif kind == 'mlp':
            l = tgt[1]
            slot = layers.setdefault(l, {})
            slot['mlp_mul'] = slot.get('mlp_mul', 1.0) * dose
            info['layer'] = l
        else:
            _, l, h = tgt
            t = layers.setdefault(l, {}).setdefault('head_mul', {})
            t[h] = t.get(h, 1.0) * dose
            info.update(layer=l, head=h)
        entries.append(info)
    wire_layers = {}
    for l in sorted(layers):
        slot, out = layers[l], {}
        for key in ('add_resid', 'add_raw', 'mul', 'head_mul'):
            if slot.get(key):
                idx = sorted(slot[key])
                out[key] = [idx, [float(slot[key][n]) for n in idx]]
        if 'mlp_mul' in slot:
            out['mlp_mul'] = float(slot['mlp_mul'])
        if 'add_vector' in slot:
            c32 = slot['add_vector'].to(torch.float32)          # the float64 sum of the entries, cast once
            for x in c32.tolist():
                if not (math.isfinite(x) and abs(x) <= MAX_VECTOR):
                    raise SteeringError(f"{slot['vector_where']}: the push at layer {l} has a component of {x:g}; "
                                        f"at most {MAX_VECTOR:g}")
            out['add_vector'] = [float(x) for x in c32.tolist()]
        wire_layers[str(l)] = out
    return {'version': VERSION, 'layers': wire_layers}, entries


def _direction_entry(ws, e, i, addr, where, layers) -> dict:
    """One direction:<measurement>@<k> entry: its fields checked, its vector (float64) added to layer k's slot
    (``layers[k]['add_vector']``) -> the entry's info. Reads the measurement's files only; no model."""
    from . import directions as DR
    from .directions_search import measurement_identity
    unknown = sorted(set(e) - {'addr', 'mode', 'dose', 'unit', 'orthogonalize_direction', 'means_sha256'})
    if unknown:
        raise SteeringError(f"{where}: unknown field {', '.join(unknown)}")
    if e.get('mode') != 'add':
        raise SteeringError(f"{where}: a direction is added: mode must be 'add'")
    if 'dose' not in e:
        raise SteeringError(f"{where}: dose is missing")
    dose = _dose(e['dose'], where)
    unit = 'diff' if e.get('unit') is None else e['unit']
    if unit == 'resid':
        raise SteeringError(f"{where}: unit 'resid' is not available for a direction; use 'diff' (the difference of the "
                            f"two means along the direction, the default) or 'raw' (the unit direction times the dose)")
    if unit not in DIRECTION_UNITS:
        raise SteeringError(f"{where}: unit must be 'diff' or 'raw' for a direction")
    ortho = e.get('orthogonalize_direction', True)
    if not isinstance(ortho, bool):
        raise SteeringError(f"{where}: orthogonalize_direction must be true or false")
    pin = e.get('means_sha256')
    if 'means_sha256' in e and not (isinstance(pin, str) and re.fullmatch(r'[0-9a-f]{64}', pin)):
        raise SteeringError(f"{where}: means_sha256 must be the 64 hexadecimal digits of the measurement's means file")
    try:
        _, name, k = target(ws, addr)
    except SteeringError as exc:
        raise SteeringError(f"{where}: {exc}") from None
    try:
        meta, good, bad = DR.load(ws, name)
        DR.matches(ws, meta, name)
        ident = measurement_identity(ws, name)
    except DR.DirectionError as exc:
        raise SteeringError(f"{where}: {exc}") from None
    if pin is not None and pin != ident['means_sha256']:
        raise SteeringError(f"{where}: measurement '{name}' changed since this entry was made: its means are "
                            f"{ident['means_sha256'][:12]}… now and the entry names {pin[:12]}…; add the entry again "
                            f"from the Layers view, or drop means_sha256, to steer along the new measurement")
    n = meta['n_layers']
    if k == n:
        raise SteeringError(f"{where}: index {n} is the residual stream after the final norm, which no layer reads; "
                            f"steer at an index from 0 to {n - 1}")
    if k > n:
        raise SteeringError(f"{where}: index {k} does not exist (the measurement has indices 0 to {n})")
    if not resid_hookable(ws, k):
        raise SteeringError(f"{where}: layer {k} has no hook point at its input (hook_resid_pre), so a direction cannot "
                            f"be added there")
    tab = DR.table(good, bad, ortho)
    if not bool((tab[k] != 0).any()):
        raise SteeringError(f"{where}: index {k} of direction '{name}' is unusable: the good and bad means do not differ "
                            f"there (after the projection, when it is asked for)")
    v = tab[k].double()
    d = bad[k].double() - good[k].double()
    scale = float(d @ v) if unit == 'diff' else 1.0
    c = dose * scale * v
    slot = layers.setdefault(k, {})
    slot['add_vector'] = c if 'add_vector' not in slot else slot['add_vector'] + c
    slot['vector_where'] = where                         # the last entry at this layer names a sum that is too large
    return dict(index=i, addr=addr, mode='add', dose=dose, kind='direction', unit=unit, direction=name,
                measurement_index=k, layer=k, orthogonalize_direction=ortho, scale=scale, norm=float(c.norm()),
                means_sha256=ident['means_sha256'], meta_sha256=ident['meta_sha256'])


_LAYER_KEY = re.compile(r'(0|[1-9][0-9]*)', re.ASCII)


def needs_calibration(wire) -> list[int]:
    return [int(l) for l, slot in wire['layers'].items() if slot.get('add_resid')]


def _inside_worker(ws) -> bool:
    from .worker_lease import WorkerLease
    return isinstance(getattr(ws, '_runtime_client', None), WorkerLease)


def _layer_key(ws, key, what):
    """A layer given as an int or its decimal string (ASCII, no leading zero) -> its int."""
    if isinstance(key, bool):
        raise SteeringError(f"{what} {key!r} is not a layer")
    if isinstance(key, int):
        l = key
    elif isinstance(key, str) and _LAYER_KEY.fullmatch(key):
        l = int(key)
    else:
        raise SteeringError(f"{what} {key!r} is not a layer (use its number, e.g. \"12\")")
    if not 0 <= l < ws.struct.n_layers:
        raise SteeringError(f"{what} {key!r}: layer {l} does not exist (the model has {ws.struct.n_layers})")
    return l


def _positive(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SteeringError(f"{what} must be a number")
    try:
        v = float(v)
    except OverflowError:
        raise SteeringError(f"{what} is too large") from None
    if not math.isfinite(v) or not 0 < v <= MAX_P50:
        raise SteeringError(f"{what} must be a finite number above 0 and at most {MAX_P50:g} (it is {v:g})")
    return v


def given_p50(ws, p50) -> dict[str, float]:
    """An explicit per-layer p50 ({layer: residual norm}) -> {"layer": float}, validated.

    This is how a recipe made with neuron-explorer's sweep statistics gets the same doses here:
    give the sweep's p50 (``resid_norm_q10_50_90[1]``) for the layers it pushes."""
    if p50 is None:
        return {}
    if not isinstance(p50, dict):
        raise SteeringError("p50 must be an object {layer: residual norm}")
    out = {}
    for key, v in p50.items():
        l = _layer_key(ws, key, "p50 layer")
        if str(l) in out:
            raise SteeringError(f"p50 gives layer {l} twice")
        out[str(l)] = _positive(v, f"p50 of layer {l}")
    return out


def prepare(ws, spec, calibrate=True, on_event=None, p50=None, recalibrate=False):
    """A C3 list -> (wire with p50 for every layer that has a resid push, info).

    ``p50`` ({layer: value}) gives the residual norm for some or all layers; those layers are not
    measured and their entries report ``source: "given"``.  For the rest, calibration runs on the
    base model once per model and reference text (``info['calibration']`` says whether it was cached
    or ran); ``recalibrate=True`` measures again even when cached.  With ``calibrate=False`` a missing
    calibration is reported, not run, and the wire lacks those layers' p50."""
    if _inside_worker(ws):
        raise SteeringError("a steering specification is prepared by the server; the model worker takes its wire form")
    given = given_p50(ws, p50)
    wire, entries = normalize(ws, spec)
    info = {'entries': entries, 'calibration': {'needed': False}}
    want = needs_calibration(wire)
    values = {str(l): given[str(l)] for l in want if str(l) in given}
    measure = [l for l in want if str(l) not in given]
    if measure:
        got, cal = calibration(ws, run=calibrate, on_event=on_event, recalibrate=recalibrate, need=measure)
        info['calibration'] = cal
        if got is not None:
            missing = [l for l in measure if str(l) not in got]
            if missing:
                raise SteeringError(f"calibration has no residual norm for layer {missing[0]}; a resid push cannot be sized there "
                                    f"(give p50 for it, or use unit raw)")
            values.update({str(l): float(got[str(l)]) for l in measure})
            wire['calibration'] = cal['key']
    if values:
        wire['p50'] = {str(l): values[str(l)] for l in want if str(l) in values}
    if want:
        info['calibration']['given'] = sorted(int(l) for l in values if l in given)
    for e in entries:
        if e.get('unit') == 'resid':
            src = {'given' if l in given else 'measured' for l in e['layers']}
            e['source'] = src.pop() if len(src) == 1 else 'mixed'
    return wire, info


def check_wire(ws, wire):
    """Validate a wire form (as the model worker receives it) -> the same dict, or SteeringError.

    Strict on purpose: the worker trusts nothing it is sent, and only SteeringError leaves here."""
    try:
        return _check_wire(ws, wire)
    except SteeringError:
        raise
    except (TypeError, ValueError, KeyError, IndexError, OverflowError, AttributeError) as exc:
        raise SteeringError(f"steer wire form is malformed ({type(exc).__name__}: {exc})") from None


def _check_wire(ws, wire):
    s = ws.struct
    if not isinstance(wire, dict) or wire.get('version') != VERSION or not isinstance(wire.get('layers'), dict):
        raise SteeringError(f"steer wire form must be an object with version {VERSION} and layers")
    if set(wire) - {'version', 'layers', 'p50', 'calibration'}:
        raise SteeringError(f"steer wire form has unknown fields {sorted(set(wire) - {'version', 'layers', 'p50', 'calibration'})}")
    cal = wire.get('calibration')
    if cal is not None and (not isinstance(cal, str) or len(cal) > 128):
        raise SteeringError("steer calibration must be a calibration key")
    p50 = wire.get('p50', {})
    if not isinstance(p50, dict):
        raise SteeringError("steer p50 must be an object")
    for key in p50:
        if not isinstance(key, str) or not _LAYER_KEY.fullmatch(key):
            raise SteeringError(f"steer p50 key {key!r} is not a layer")
    total, seen = 0, set()

    def number(v, what):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise SteeringError(f"{what} must be a number")
        try:
            f = float(v)
        except OverflowError:
            raise SteeringError(f"{what} must be at most {MAX_DOSE:g}") from None
        if not math.isfinite(f) or abs(f) > MAX_DOSE:
            raise SteeringError(f"{what} must be a finite number of at most {MAX_DOSE:g}")
        return f

    for key, slot in wire['layers'].items():
        if not isinstance(key, str) or not _LAYER_KEY.fullmatch(key) or not 0 <= int(key) < s.n_layers:
            raise SteeringError(f"steer layer {key!r} does not exist")
        l = int(key)
        if l in seen:
            raise SteeringError(f"steer layer {l} is given twice")
        seen.add(l)
        if not isinstance(slot, dict):
            raise SteeringError(f"steer layer {l} must be an object")
        if set(slot) - {'add_resid', 'add_raw', 'mul', 'mlp_mul', 'head_mul', 'add_vector'}:
            raise SteeringError(f"steer layer {l} has unknown fields")
        if 'add_vector' in slot:
            vec = slot['add_vector']
            if not isinstance(vec, list) or len(vec) != s.d_model:
                raise SteeringError(f"steer layer {l} add_vector must be a list of {s.d_model} numbers (d_model)")
            for v in vec:
                # not (|v| <= MAX): also NaN and inf; an int of any size compares without overflow
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not abs(v) <= MAX_VECTOR:
                    raise SteeringError(f"steer layer {l} add_vector value must be a finite number of at most "
                                        f"{MAX_VECTOR:g}")
            if not resid_hookable(ws, l):
                raise SteeringError(f"layer {l} has no hook point at its input (hook_resid_pre), so a direction cannot "
                                    f"be added there")
        for k in ('add_resid', 'add_raw', 'mul', 'head_mul'):
            if k not in slot:
                continue
            pair = slot[k]
            if (not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], list) or not isinstance(pair[1], list)
                    or len(pair[0]) != len(pair[1]) or not all(type(i) is int for i in pair[0])):
                raise SteeringError(f"steer layer {l} {k} must be [[indices], [values]] of equal length")
            if not pair[0]:
                raise SteeringError(f"steer layer {l} {k} is empty; leave it out")
            if len(set(pair[0])) != len(pair[0]):
                raise SteeringError(f"steer layer {l} {k} repeats an index")
            for v in pair[1]:
                number(v, f"steer layer {l} {k} value")
            if k == 'head_mul':
                why = head_problem(ws, l)
                if why:
                    raise SteeringError(why)
                if not all(0 <= h < s.n_heads for h in pair[0]):
                    raise SteeringError(f"steer layer {l} head index out of range")
            else:
                total += len(pair[0])
                if not neurons_hookable(ws, l):
                    raise SteeringError(f"layer {l} has no MLP neurons that can be steered")
                width = s.mlp_width(l)
                if not all(0 <= n < width for n in pair[0]):
                    raise SteeringError(f"steer layer {l} {k} neuron index out of range (width {width})")
        if 'mlp_mul' in slot:
            number(slot['mlp_mul'], f"steer layer {l} mlp_mul")
            if not neurons_hookable(ws, l):
                raise SteeringError(f"layer {l} has no MLP activations that can be steered")
        if 'add_resid' in slot:
            if key not in p50:
                raise SteeringError(f"layer {l} has a resid push but no p50; prepare the specification first")
            _positive(p50[key], f"steer p50 of layer {l}")
    if total > MAX_NEURONS:
        raise SteeringError(f"the steer wire form pushes more than {MAX_NEURONS} neurons")
    return wire


def as_wire(ws, steer, on_event=None, p50=None):
    """None -> None; a wire dict -> validated; a C3 list -> prepared here (calibrating when needed)."""
    if steer is None:
        return None
    if isinstance(steer, dict):
        return check_wire(ws, steer)
    wire, info = prepare(ws, steer, calibrate=True, on_event=on_event, p50=p50)
    return wire


def is_empty(wire) -> bool:
    return wire is None or not wire.get('layers')


# ------------------------------------------------------------------ resolution where the weights are
def _read_unit(ws, W) -> int:
    """Neurons per decode unit along the neuron axis of W: the quantization block size when the
    store packs that axis in blocks (GGUF), else 1. The selection code's own rule (one read path)."""
    from .selections import read_unit
    return read_unit(ws, W.name, W.kept[W.perm[0]])


def plan_reads(indices, d: int, unit: int = 1, window: int | None = None, gap: int | None = None):
    """Sorted unique neuron indices -> [(a, b, [indices in a..b])], one weight read each.

    ``unit > 1``: the neuron axis is stored in blocks of ``unit`` (a quantized tensor). A read
    covers only blocks that hold a needed neuron (consecutive such blocks share a read), and a
    read never splits a block, so no block is decoded twice and none is decoded needlessly.
    ``unit == 1``: neurons at most ``gap`` apart share a read.  Either way a read holds at most
    ``max(window, unit * d)`` logical values.  The planner is ``selections.windows``, the one the
    selection edits (scale_neurons) use for their column reads."""
    import numpy as np
    from .selections import windows
    window = READ_WINDOW if window is None else window
    gap = READ_GAP if gap is None else gap
    pos = np.asarray(list(indices), dtype=np.int64)
    if not len(pos):
        return []
    plan = windows(pos, d, window, gap=None if unit > 1 else gap, unit=unit)
    return [(int(w[0]), int(w[-1]) + 1, [int(n) for n in w]) for w in plan]


class Progress:
    """``reading`` events for a column-norm read that takes more than a moment (after
    PROGRESS_AFTER seconds, then at most every PROGRESS_EVERY), and one ``read`` at the end if any
    was sent.  A failing callback never stops the read."""

    def __init__(self, on_event, total: int):
        self.on_event, self.total, self.done = on_event, int(total), 0
        self.start, self.last = time.monotonic(), None

    def _send(self, e):
        try:
            self.on_event(e)
        except Exception:
            self.on_event = None

    def add(self, n: int, layer: int) -> None:
        self.done += int(n)
        if not self.on_event:
            return
        now = time.monotonic()
        if now - self.start < PROGRESS_AFTER or (self.last is not None and now - self.last < PROGRESS_EVERY):
            return
        self.last = now
        self._send({'state': 'reading', 'layer': int(layer), 'done': self.done, 'total': self.total,
                    'seconds': round(now - self.start, 3)})

    def finish(self) -> None:
        if self.on_event and self.last is not None:
            self._send({'state': 'read', 'done': self.done, 'total': self.total,
                        'seconds': round(time.monotonic() - self.start, 3)})


def column_norms(ws, l: int, indices, variant: str = 'current', extra=None, progress: Progress | None = None):
    """-> (float64 norms of the output-weight columns of neurons ``indices`` in layer l, cost).

    Reads the logical weights of the variant (base + its patches + extra) through the store's
    logical read path, planned by ``plan_reads``: for a quantized store only the packed blocks that
    hold the needed columns are decoded, each once, at most ``max(READ_WINDOW, block * d_model)``
    values per read."""
    W = ws.sem.mlp(l)['W_out']                      # canonical [neuron, d_model]
    d = W.shape[1]
    cache = None
    if extra is None and variant != 'base':
        cache = ws.calib.setdefault('steer_norms', {}).setdefault(int(l), {})
    elif extra is None:
        cache = ws.keep.setdefault('steer_norms', {}).setdefault(int(l), {})
    idx = sorted(set(int(n) for n in indices))
    need = [n for n in idx if cache is None or n not in cache]
    found = {} if cache is None else cache
    unit = _read_unit(ws, W) if need else 1
    cost = dict(reads=0, elements=0, seconds=0.0, cached=len(idx) - len(need), block=unit)
    if progress is not None and len(idx) > len(need):
        progress.add(len(idx) - len(need), l)
    started = time.monotonic()
    for a, b, ns in plan_reads(need, d, unit):
        t = W.sub(slice(a, b)).read(variant, extra)
        norms = t.double().norm(dim=1)
        for n in ns:
            found[n] = float(norms[n - a])
        cost['reads'] += 1
        cost['elements'] += (b - a) * d
        if progress is not None:
            progress.add(len(ns), l)
    cost['seconds'] = round(time.monotonic() - started, 4)
    return torch.tensor([found[int(n)] for n in indices], dtype=torch.float64), cost


def _resid_count(wire) -> int:
    return sum(len(slot['add_resid'][0]) for slot in wire['layers'].values() if slot.get('add_resid'))


class LayerPlan:
    __slots__ = ('m', 'c', 'hm', 'r', 'norms')

    def __init__(self):
        self.m = self.c = self.hm = self.r = None
        self.norms = {}


def resolve(ws, wire, variant: str = 'current', extra=None, cost=None, on_event=None):
    """A wire form -> {layer: LayerPlan} of dense per-layer vectors (float32)."""
    check_wire(ws, wire)
    plan = {}
    progress = Progress(on_event, _resid_count(wire)) if on_event else None
    for key, slot in wire['layers'].items():
        l = int(key)
        p = LayerPlan()
        if any(k in slot for k in ('add_resid', 'add_raw', 'mul', 'mlp_mul')):
            width = ws.struct.mlp_width(l)
            if 'mul' in slot or 'mlp_mul' in slot:
                m = torch.full((width,), float(slot.get('mlp_mul', 1.0)), dtype=torch.float64)
                if 'mul' in slot:
                    idx, val = slot['mul']
                    m[torch.tensor(idx, dtype=torch.long)] *= torch.tensor(val, dtype=torch.float64)
                p.m = m.float()
            if 'add_resid' in slot or 'add_raw' in slot:
                c = torch.zeros(width, dtype=torch.float64)
                if 'add_raw' in slot:
                    idx, val = slot['add_raw']
                    c[torch.tensor(idx, dtype=torch.long)] += torch.tensor(val, dtype=torch.float64)
                if 'add_resid' in slot:
                    idx, val = slot['add_resid']
                    norms, spent = column_norms(ws, l, idx, variant, extra, progress)
                    if cost is not None:
                        for k2 in ('reads', 'elements', 'cached'):
                            cost[k2] = cost.get(k2, 0) + spent[k2]
                        cost['seconds'] = round(cost.get('seconds', 0.0) + spent['seconds'], 4)
                    for n, v in zip(idx, norms.tolist()):
                        if not (math.isfinite(v) and v > 0):
                            raise SteeringError(f"L{l}N{n}: its output weights have norm {v:g}, so a resid dose has no "
                                                f"size; use unit raw")
                        p.norms[n] = v
                    c[torch.tensor(idx, dtype=torch.long)] += (torch.tensor(val, dtype=torch.float64) * float(wire['p50'][key])
                                                               / norms)
                p.c = c.float()
        if 'head_mul' in slot:
            hm = torch.ones(ws.struct.n_heads, dtype=torch.float64)
            idx, val = slot['head_mul']
            hm[torch.tensor(idx, dtype=torch.long)] = torch.tensor(val, dtype=torch.float64)
            p.hm = hm.float()
        if 'add_vector' in slot:
            p.r = torch.tensor(slot['add_vector'], dtype=torch.float32)
        plan[l] = p
    if progress is not None:
        progress.finish()
    return plan


def make_hooks(plan) -> dict:
    """{hook name: fn(x, name) -> x'}: one vectorised hook per layer and hook point."""
    hooks = {}
    for l, p in sorted(plan.items()):
        if p.m is not None or p.c is not None:
            def mlp(x, name, m=p.m, c=p.c):
                if m is not None:
                    x = x * m.to(device=x.device, dtype=x.dtype)
                if c is not None:
                    x = x + c.to(device=x.device, dtype=x.dtype)
                return x
            hooks[f'blocks.{l}.mlp.hook_post'] = mlp
        if p.hm is not None:
            def heads(z, name, hm=p.hm):                 # z [batch, pos, head, d_head]
                return z * hm.to(device=z.device, dtype=z.dtype)[:, None]
            hooks[f'blocks.{l}.attn.hook_z'] = heads
        if p.r is not None:
            def resid(x, name, r=p.r):                   # x [batch, pos, d_model], entering layer l
                return x + r.to(device=x.device, dtype=x.dtype)
            hooks[f'blocks.{l}.hook_resid_pre'] = resid
    return hooks


def hooks_for(ws, steer, variant: str = 'current', extra=None, on_event=None) -> dict:
    """The hooks that apply ``steer`` (None, a wire dict or a C3 list) to ``variant``."""
    wire = as_wire(ws, steer)
    if is_empty(wire):
        return {}
    if variant == 'base':
        raise SteeringError("the base model is never steered; steer the current variant")
    return make_hooks(resolve(ws, wire, variant, extra, on_event=on_event))


def compose(*fns):
    """Hooks at one point, applied in order; a hook that returns None leaves the value as it was."""
    fns = [f for f in fns if f is not None]

    def hook(x, name):
        changed = False
        for f in fns:
            y = f(x, name)
            if y is not None:
                x, changed = y, True
        return x if changed else None
    return hook


def module_pre_hook(fn, name, heads=None):
    """A torch forward pre-hook (with_kwargs=True) that hands the module's input to ``fn``;
    ``heads`` reshapes [.., heads * d_head] to [.., heads, d_head] for attn.hook_z."""
    def pre(module, args, kwargs):
        if args:
            x, key = args[0], None
        else:
            key = 'hidden_states' if 'hidden_states' in kwargs else next(iter(kwargs))
            x = kwargs[key]
        v = x.reshape(*x.shape[:-1], heads, x.shape[-1] // heads) if heads else x
        out = fn(v, name)
        if out is None:
            return None
        out = out.reshape(x.shape).to(x.dtype)
        if key is None:
            return (out,) + tuple(args[1:]), kwargs
        return args, {**kwargs, key: out}
    return pre


@contextmanager
def removing(handles):
    try:
        yield handles
    finally:
        for h in handles:
            h.remove()


def describe(ws, wire, info, variant: str = 'current', on_event=None):
    """What a prepared specification will do, entry by entry: the resolved constants and the column
    norms they came from (read here, through the store's logical read path), plus the read cost."""
    import statistics
    if variant == 'base' and not is_empty(wire):
        raise SteeringError("the base model is never steered; steer the current variant")
    p50 = wire.get('p50') or {}
    cost = dict(reads=0, elements=0, seconds=0.0, cached=0)
    out = []
    total = sum(e.get('neurons', 0) for e in info['entries'] if e.get('unit') == 'resid')
    progress = Progress(on_event, total) if on_event else None
    for e in info['entries']:
        d = {k: e[k] for k in ('index', 'addr', 'mode', 'dose', 'kind', 'unit', 'source', 'layer', 'head', 'neurons', 'layers',
                               'skipped', 'direction', 'measurement_index', 'orthogonalize_direction', 'scale', 'norm',
                               'means_sha256', 'meta_sha256') if k in e}
        if e['kind'] != 'neurons' or e['mode'] == 'mul':
            d['value'] = e['dose']
            out.append(d)
            continue
        # One target list per entry (a group may span layers); the first 32 are listed.
        pairs = _entry_pairs(ws, e)
        values, norms, listed = [], [], []
        by_layer = {}
        for l, n in pairs:
            by_layer.setdefault(l, []).append(n)
        resolved = {}
        for l, ns in by_layer.items():
            if e['unit'] == 'raw':
                for n in ns:
                    resolved[(l, n)] = (None, e['dose'])
                continue
            nv, spent = column_norms(ws, l, ns, variant, progress=progress)
            for k2 in ('reads', 'elements', 'cached'):
                cost[k2] += spent[k2]
            cost['seconds'] = round(cost['seconds'] + spent['seconds'], 4)
            p = p50.get(str(l))
            for n, v in zip(ns, nv.tolist()):
                if not (math.isfinite(v) and v > 0):
                    raise SteeringError(f"steer entry {e['index'] + 1} ({e['addr']}): L{l}N{n} has output weights of norm "
                                        f"{v:g}, so a resid dose has no size; use unit raw")
                resolved[(l, n)] = (v, None if p is None else e['dose'] * p / v)
        for l, n in pairs:
            v, c = resolved[(l, n)]
            if v is not None:
                norms.append(v)
            if c is not None:
                values.append(c)
            if len(listed) < 32:
                listed.append({'addr': f"L{l}N{n}", 'norm': v, 'value': c})
        if e['unit'] == 'resid':
            d['p50'] = {str(l): p50.get(str(l)) for l in by_layer}
        if len(pairs) == 1:
            d['norm'], d['value'] = listed[0]['norm'], listed[0]['value']
        else:
            d['targets'] = listed
            for key, vals in (('values', values), ('norms', norms)):
                if vals:
                    d[key] = {'min': min(vals), 'median': statistics.median(vals), 'max': max(vals)}
        if e['unit'] == 'resid' and not values:
            d['value'] = None
            d['note'] = 'not calibrated yet; the constant is known once the base model has been measured'
        out.append(d)
    if progress is not None:
        progress.finish()
    layers = wire['layers']
    return {'ok': True, 'entries': out, 'calibration': info['calibration'], 'wire': wire,
            'layers': len(layers),
            'neurons': len({(l, n) for l, slot in layers.items() for k in ('add_resid', 'add_raw', 'mul') for n in slot.get(k, [[]])[0]}),
            'heads': sum(len(slot.get('head_mul', [[]])[0]) for slot in layers.values()),
            'mlps': sum('mlp_mul' in slot for slot in layers.values()),
            'directions': sum(e['kind'] == 'direction' for e in info['entries']),
            'reads': cost}


def _entry_pairs(ws, e):
    return target(ws, e['addr'])[1]


# ------------------------------------------------------------------ calibration
def reference_tokens(ws) -> tuple[list[int], dict]:
    """The reference text as token ids, tokenized as neuron-explorer's ``layer_resid_p50`` does
    (``tok(REFERENCE_TEXT)``: the tokenizer's own special tokens, no truncation), and how exact
    that is: ``{'exact': bool, 'special_tokens': bool, 'truncated_to': n | None, 'fallback': str | None}``.

    Not exact when the model's context is shorter than the text (cut to the context) or when a toy
    vocabulary cannot spell it (lower case, then only the characters it can spell)."""
    tok = ws.tok
    if tok is None:
        raise SteeringError("this model has no tokenizer, so it cannot be calibrated; give p50 or use unit raw")
    how = {'special_tokens': False, 'truncated_to': None, 'fallback': None}
    ids = None
    try:
        try:
            ids = list(tok.encode(REFERENCE_TEXT, special=True))   # BPETokenizer: add_special_tokens=True
            how['special_tokens'] = True
        except TypeError:
            ids = list(tok.encode(REFERENCE_TEXT))                  # a plain vocabulary has no special tokens
    except (ValueError, KeyError):
        ids = None
    if ids is None:
        try:
            ids = list(tok.encode(REFERENCE_TEXT.lower()))
            how['fallback'] = 'lower case'
        except (ValueError, KeyError):
            ok = {}
            for ch in REFERENCE_TEXT.lower():
                if ch not in ok:
                    try:
                        ok[ch] = list(tok.encode(ch))
                    except (ValueError, KeyError):
                        ok[ch] = []
            ids = [i for ch in REFERENCE_TEXT.lower() for i in ok[ch]]
            how['fallback'] = 'characters the vocabulary can spell'
    ids = [int(i) for i in ids]
    n_ctx = int(getattr(ws.struct, 'n_ctx', 0) or 0)
    if n_ctx and len(ids) > n_ctx:
        ids = ids[:n_ctx]
        how['truncated_to'] = n_ctx
    if len(ids) < 2:
        raise SteeringError("the reference text has too few tokens for this tokenizer; give p50 or use unit raw")
    how['exact'] = how['fallback'] is None and how['truncated_to'] is None
    return ids, how


def reference_ids(ws) -> list[int]:
    return reference_tokens(ws)[0]


def model_identity(ws) -> dict:
    """What a calibration is valid for: the base weights (not the patches; it is measured on base)."""
    meta = getattr(ws, 'meta', None) or {}
    if meta.get('base_sha256'):
        return {'kind': ws.kind, 'base_sha256': meta['base_sha256']}
    files = []
    for f in getattr(ws.store, 'files', None) or []:
        path = getattr(f, 'path', None)
        if path and os.path.exists(path):
            st = os.stat(path)
            files.append([os.path.basename(path), st.st_size, st.st_mtime_ns])
        else:
            files.append([str(getattr(f, 'location', None) or getattr(f, 'name', '') or ''), None, None])
    return {'kind': ws.kind, 'name': ws.name, 'repo': meta.get('repo'), 'sha': meta.get('sha'), 'files': files}


def calibration_key(ws, ids) -> str:
    value = {'identity': model_identity(ws), 'reference': hashlib.sha256(REFERENCE_TEXT.encode()).hexdigest(),
             'ids': hashlib.sha256(json.dumps(ids).encode()).hexdigest(), 'version': VERSION}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _usable(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 < v <= MAX_P50


def calibration(ws, run=True, on_event=None, recalibrate=False, need=None):
    """-> ({layer: p50} or None, info). Cached in the workspace; runs on the base model when missing
    (or when ``recalibrate``).  A measurement with any non-finite or non-positive layer is never
    cached; if a layer in ``need`` is one of them this raises, naming it."""
    ids, how = reference_tokens(ws)
    key = calibration_key(ws, ids)
    info = {'needed': True, 'cached': False, 'ran': False, 'key': key, 'tokens': len(ids),
            'reference': REFERENCE_SOURCE, 'reference_sha256': hashlib.sha256(REFERENCE_TEXT.encode()).hexdigest(), **how}
    try:
        data = ws.read_json(CALIBRATION_FILE, {}) or {}
    except (OSError, ValueError):
        data = {}                 # an unreadable cache is measured again and rewritten
    if not isinstance(data, dict):
        data = {}
    hit = data.get(key)
    if (not recalibrate and isinstance(hit, dict) and isinstance(hit.get('p50'), dict) and hit['p50']
            and all(_usable(v) for v in hit['p50'].values())):
        info.update(cached=True, seconds=hit.get('seconds'), measured=hit.get('time'))
        return {str(k): float(v) for k, v in hit['p50'].items()}, info
    if not run:
        return None, info
    if _inside_worker(ws):
        raise SteeringError("calibration is started by the server; the model worker only runs its forward pass")
    if on_event:
        on_event({'state': 'calibrating', 'tokens': len(ids)})
    started = time.monotonic()
    out = calibrate_job(ws, ids)
    seconds = round(time.monotonic() - started, 3)
    p50 = {str(k): float(v) for k, v in out['p50'].items()}
    bad = {k: v for k, v in p50.items() if not _usable(v)}
    info.update(ran=True, recalibrated=bool(recalibrate and hit is not None), seconds=seconds)
    if bad:
        info.update(stored=False, unusable=sorted(int(k) for k in bad))
        wanted = [l for l in (need or []) if str(l) in bad]
        if on_event:
            on_event({'state': 'calibration_unusable', 'layers': info['unusable']})
        if wanted:
            l = wanted[0]
            raise SteeringError(f"the measured residual norm of layer {l} is {bad[str(l)]:g}, so a resid dose has no size there; "
                                f"nothing was cached. Give p50 for layer {l}, or use unit raw")
        return {k: v for k, v in p50.items() if k not in bad}, info
    data = {k: v for k, v in data.items() if isinstance(v, dict)}
    data[key] = {'p50': p50, 'tokens': int(out['tokens']), 'seconds': seconds, 'time': time.time(),
                 'identity': model_identity(ws), 'reference_sha256': info['reference_sha256'], 'exact': how['exact']}
    while len(data) > CALIBRATION_KEEP:
        oldest = min(data, key=lambda k: data[k].get('time', 0))
        data.pop(oldest)
    ws.write_json(CALIBRATION_FILE, data)
    info.update(stored=True, measured=data[key]['time'])
    if on_event:
        on_event({'state': 'calibrated', 'seconds': seconds, 'tokens': len(ids)})
    return p50, info


@job_scope('steer_calibrate')
@worker_job('steer_calibrate')
def calibrate_job(ws, ids):
    """Worker job: one forward pass of the base model over ``ids``; only the medians come back."""
    from .worker_jobs import _ids
    ids = _ids(list(ids))
    if not 2 <= len(ids) <= 4096:
        raise SteeringError("calibration needs 2..4096 reference tokens")
    r = ws.runner
    s = ws.struct
    p50 = {}

    def reducer(l):
        def hook(x, name):
            h = x[0, 1:] if x.shape[1] > 1 else x[0]
            p50[str(l)] = float(h.float().norm(dim=-1).median())
            return None
        return hook
    hooks = {f'blocks.{l}.hook_resid_post': reducer(l) for l in range(s.n_layers) if r.available(f'blocks.{l}.hook_resid_post')}
    with torch.no_grad():
        r.run(ids, 'base', None, hooks=hooks)
    return {'p50': p50, 'tokens': len(ids)}

