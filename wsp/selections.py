"""Selections: any address, including a named group, as something that can be scaled, ablated or reverted.

A selection is an ordinary workspace group (`Workspace.set_group`, address `group:<name>`); there is
no second store.  This module reads a selection as

    neurons   {layer: {index, ...}}       from L3N5, blocks.3.mlp.neurons.64-127, ...
    heads     {(layer, head), ...}         from L3H5, blocks.3.attn.kvgroup.1
    slices    [(tensor, index), ...]       everything else: tensors, slices, sublayers, layers

and turns it into one weight script (`script`), which the browser applies through `/api/edit`
like any other edit, so task scope, candidate tests, drift checks and patch recording are the
same as for a typed script.  `actions` says, without reading any weight, which of scale /
ablate / revert / console can be done and, if not, why.

It also holds the small amount of machinery that makes edits over thousands of neurons cheap:
`positions` along one axis of a tensor, grouped into windows that are read once
(`windows`) and runs that become one stored piece each (`runs`).  Whether a weight has
been edited is decided from the patch pieces, without reading the checkpoint, so it costs the
same on quantized models (whose base is never rewritten).
"""
from __future__ import annotations

import itertools
import json
import logging
import math
import re
from dataclasses import dataclass, field

import numpy as np
import torch

from . import addressing as A
from . import delta as D
from .store import norm_index, region_size

log = logging.getLogger(__name__)

# How many weights one selection edit may write is wsp.edit_limit's (the setting, memory and disk), see _limit.
WINDOW = 4_000_000             # weights read at once when walking many neurons of one tensor
GAP = 64                       # unselected positions a window may read through between two selected ones
SHOW_MEMBERS = 200             # members listed on a selection's page (the selection itself is never capped)

_NEURON = re.compile(r"^blocks\.(\d+)\.mlp\.neuron\.(\d+)$")
_NEURONS = re.compile(r"^blocks\.(\d+)\.mlp\.neurons\.(\d+)-(\d+)$")
_HEAD = re.compile(r"^blocks\.(\d+)\.attn\.head\.(\d+)$")
_KV = re.compile(r"^blocks\.(\d+)\.attn\.kvgroup\.(\d+)$")
NEURON_ROLES = ("W_in", "b_in", "W_gate", "b_gate", "W_up", "b_up", "W_out")


class Refusal(ValueError):
    """An action that cannot be done on this selection; the message says why in plain words."""

    def __init__(self, msg: str, **counts):
        super().__init__(msg)
        self.counts = counts


@dataclass
class Selection:
    addr: str
    neurons: dict = field(default_factory=dict)        # layer -> set of indices
    heads: set = field(default_factory=set)            # (layer, head)
    slices: list = field(default_factory=list)         # (tensor, index), first occurrence order
    other: list = field(default_factory=list)          # {"member", "reason"}
    members: list = field(default_factory=list)        # the group's own members (or [addr])

    @property
    def n_neurons(self) -> int:
        return sum(len(v) for v in self.neurons.values())


# ------------------------------------------------------------------------------ reading an address
def expand(ws, addr: str) -> Selection:
    """Classify what an address selects. Group members that no longer resolve are reported, not raised."""
    addr = A.normalize(addr)
    sel = Selection(addr)
    seen_slices: set = set()
    seen_groups: set = set()

    def add(a: str, top: bool) -> None:
        a = A.normalize(a)
        if a.startswith("group:"):
            g = a[6:]
            if g in seen_groups:
                return
            seen_groups.add(g)
            members = ws.group_members(g)          # unknown: KeyError; not a list of members: ValueError
            if top:
                sel.members = list(members)
            for m in members:
                if not isinstance(m, str):         # e.g. null in a hand-edited groups.json: unresolved, with a reason
                    sel.other.append({"member": json.dumps(m, default=str)[:80],
                                      "reason": "not an address: a selection member must be text"})
                    continue
                try:
                    add(m, False)
                except (KeyError, IndexError, ValueError) as e:
                    sel.other.append({"member": m, "reason": _plain(e)})
            return
        if top:
            sel.members = [a]
        if a.startswith("blocks.*") or ".*." in a:
            for l in range(ws.struct.n_layers):
                add(a.replace(".*", f".{l}", 1), False)
            return
        m = _NEURON.match(a) or _NEURONS.match(a)
        if m:
            l, a0 = int(m.group(1)), int(m.group(2))
            b0 = int(m.group(3)) if m.re is _NEURONS else a0
            if not _neurons_exist(ws.struct, l, a0, b0):
                ws.resolve(a)                      # says why (raises IndexError or KeyError)
            sel.neurons.setdefault(l, set()).update(range(a0, b0 + 1))
            return
        m = _HEAD.match(a)
        if m:
            ws.resolve(a)
            sel.heads.add((int(m.group(1)), int(m.group(2))))
            return
        m = _KV.match(a)
        if m:
            ws.resolve(a)
            l, g = int(m.group(1)), int(m.group(2))
            per = ws.struct.n_heads // max(1, ws.struct.n_kv_heads)
            sel.heads.update((l, h) for h in range(g * per, (g + 1) * per))
            return
        for name, idx in ws.resolve(a):
            key = (name, idx)
            if key not in seen_slices:
                seen_slices.add(key)
                sel.slices.append(key)

    add(addr, True)
    return sel


def _neurons_exist(st, l: int, a: int, b: int) -> bool:
    """Bounds check of a neuron run without resolving it (thousands of members): False means "ask resolve"."""
    try:
        return 0 <= l < st.n_layers and st.has_neurons(l) and 0 <= a <= b < st.mlp_width(l)
    except (AttributeError, TypeError):
        return False


def neurons_of(ws, name: str) -> list[tuple[int, int]]:
    """The neurons of a group (or of any address), as sorted (layer, index) pairs.

    Ranges are expanded; heads, tensors and members that no longer resolve are skipped."""
    addr = name if name.startswith("group:") or not _is_group(ws, name) else "group:" + name
    sel = expand(ws, addr)
    return [(l, n) for l in sorted(sel.neurons) for n in sorted(sel.neurons[l])]


def _is_group(ws, name: str) -> bool:
    return name in getattr(ws, "groups", {})


def members_from_neurons(pairs) -> list[str]:
    """(layer, index) pairs -> group members, contiguous runs written as blocks.L.mlp.neurons.a-b."""
    by: dict[int, set] = {}
    for l, n in pairs:
        by.setdefault(int(l), set()).add(int(n))
    out = []
    for l in sorted(by):
        for a, b in runs(np.asarray(sorted(by[l]), dtype=np.int64)):
            out.append(f"L{l}N{a}" if b - a == 1 else f"blocks.{l}.mlp.neurons.{a}-{b - 1}")
    return out


def _index(x, it) -> int:
    """A whole number given as an int (or an integral float, as JSON may send it); anything else is refused."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or (isinstance(x, float) and not x.is_integer()):
        raise ValueError(f"{it!r} is not a neuron: layer and index must be whole numbers")
    return int(x)


def parse_neurons(items) -> list[tuple[int, int]]:
    """[[L, n], ...] or ["L3N5", ...] -> pairs. Non-integers are refused, never truncated."""
    out = []
    for it in items:
        if isinstance(it, str):
            m = re.match(r"^L(\d+)N(\d+)$", it.strip())
            if not m:
                raise ValueError(f"{it!r} is not a neuron (expected L<layer>N<index>)")
            out.append((int(m.group(1)), int(m.group(2))))
        else:
            try:
                l, n = it
            except (TypeError, ValueError):
                raise ValueError(f"{it!r} is not a neuron (expected [layer, index])") from None
            out.append((_index(l, it), _index(n, it)))
    return out


def member_key(ws, m: str) -> str:
    """One spelling per member, so the same slice written two ways is stored once: a slice of one tensor is
    written as A.slice_str writes it (`x[:, 3]`, whatever spacing it was typed with); other addresses normalised."""
    a = A.normalize(m)
    if "[" in a and not a.startswith("group:"):
        try:
            sl = ws.resolve(a)
        except (KeyError, IndexError, ValueError):
            return a
        if len(sl) == 1:
            return A.slice_str(sl[0])
    return a


def merge_members(ws, existing: list[str], new: list[str]) -> list[str]:
    """Union of two member lists; neuron members of both are merged into one set of runs, other members are
    kept once by their normalised key (member_key): a slice is stored as A.slice_str writes it, any other
    member as it was given (`L0H1` stays `L0H1`)."""
    keep, keys, pairs = [], set(), []
    for m in list(existing) + list(new):
        a = A.normalize(m)
        mm = _NEURON.match(a) or _NEURONS.match(a)
        if mm:
            l, a0 = int(mm.group(1)), int(mm.group(2))
            b0 = int(mm.group(3)) if mm.re is _NEURONS else a0
            pairs += [(l, n) for n in range(a0, b0 + 1)]
            continue
        k = member_key(ws, m)
        if k not in keys:
            keys.add(k)
            keep.append(k if "[" in a else str(m).strip())
    return members_from_neurons(pairs) + keep


# ------------------------------------------------------------------------------ positions along one axis
def runs(pos: np.ndarray) -> list[tuple[int, int]]:
    """Sorted positions -> [(start, stop)) runs of consecutive values."""
    if len(pos) == 0:
        return []
    cut = np.nonzero(np.diff(pos) != 1)[0] + 1
    starts = np.concatenate([[0], cut])
    stops = np.concatenate([cut, [len(pos)]])
    return [(int(pos[a]), int(pos[b - 1]) + 1) for a, b in zip(starts, stops)]


def windows(pos: np.ndarray, other: int, budget: int | None = None, gap: int | None = GAP,
            unit: int = 1) -> list[np.ndarray]:
    """Group sorted positions into windows read in one go: a window spans at most `budget` weights
    (`other` weights per position) and, unless `gap` is None, reads through at most `gap` unselected
    positions at a time. Greedy from the left, so with `gap=None` it is the fewest windows the budget allows.

    `unit > 1`: the axis is stored in packed blocks of `unit` positions (a quantized matrix, see `read_unit`),
    and a read decodes every block it touches. Windows then follow the blocks: a window never splits a block
    (no block is decoded twice), a block holding no selected position ends the window (none is decoded
    needlessly; `gap` is not used), and a window spans at most max(budget, unit * other) weights."""
    budget = WINDOW if budget is None else budget                 # looked up per call, so a test can shrink it
    out, start = [], 0
    other = max(1, other)
    if unit > 1:
        per = max(1, budget // (unit * other))                    # blocks a window may span
        blk = np.asarray(pos, dtype=np.int64) // unit
        for i in range(1, len(pos) + 1):
            if i == len(pos) or blk[i] - blk[i - 1] > 1 or blk[i] - blk[start] + 1 > per:
                out.append(pos[start:i])
                start = i
        return out
    for i in range(1, len(pos) + 1):
        if (i == len(pos) or (gap is not None and pos[i] - pos[i - 1] > gap)
                or (pos[i] - pos[start] + 1) * other > budget):
            out.append(pos[start:i])
            start = i
    return out


FLOAT_DTYPES = {"F32", "F16", "BF16", "F64"}


def read_gap(ws, name: str) -> int | None:
    """How far a read may run through unselected positions on storage that is not packed in blocks along the
    read axis: GAP for float storage; for other storage the fewest windows win (no gap limit)."""
    return GAP if str(ws.store.dtype(name)).upper() in FLOAT_DTYPES else None


def read_unit(ws, name: str, axis: int) -> int:
    """Positions per packed block along `axis` of a stored tensor: the store's `read_block` (a GGUF store
    answers (1, block) for a quantized matrix), else 1."""
    fn = getattr(ws.store, "read_block", None)
    got = fn(name) if fn is not None else None
    if not got:
        return 1
    a, size = got
    return int(size) if size > 1 and a == axis else 1


def read_windows(ws, name: str, axis: int, pos: np.ndarray, other: int, budget: int | None = None) -> list[np.ndarray]:
    """The windows a read of positions `pos` along `axis` of `name` is planned in: by packed block when the
    store packs that axis in blocks, else by `read_gap`. The one plan for column reads of a quantized store
    (steering.plan_reads, used for output-column norms, is the same function). Every window is at most
    `budget` weights: when one block across `other` weights is already more than that, the windows are not
    aligned to blocks (a block may then be decoded by two reads)."""
    budget = WINDOW if budget is None else budget
    unit = read_unit(ws, name, axis)
    if unit > 1 and unit * max(1, other) <= budget:
        return windows(pos, other, budget, gap=None, unit=unit)
    return windows(pos, other, budget, gap=read_gap(ws, name))


def with_axis(region: tuple, axis: int, r) -> tuple:
    return region[:axis] + (r,) + region[axis + 1:]


def other_size(region: tuple, axis: int) -> int:
    return math.prod(len(r) for d, r in enumerate(region) if d != axis and isinstance(r, range))


def kept_before(region: tuple, axis: int) -> int:
    """Index of `axis` among the axes a read of `region` keeps."""
    return sum(isinstance(r, range) for r in region[:axis])


def pieces_of(ws, name: str, extra: dict | None = None) -> list:
    out = [p for p, s in ws.pieces(name) if s]
    if extra and name in extra:
        out += list(extra[name])
    return out


def _member(x: np.ndarray, r) -> np.ndarray:
    if isinstance(r, int):
        return x == r
    rel = x - r.start
    return (rel % r.step == 0) & (rel // r.step >= 0) & (rel // r.step < len(r))


def axis_hits(pieces, template: tuple, axis: int, pos: np.ndarray) -> np.ndarray:
    """Which of `pos` (sorted positions along `axis`) some piece reaches inside `template`'s other axes.
    From piece regions only: a hit may still be an unchanged weight (see `edited_positions`)."""
    hit = np.zeros(len(pos), dtype=bool)
    for p in pieces:
        if p.kind == "sparse":
            c = p.coords.numpy()
            keep = p.values.numpy() != 0
            for d, t in enumerate(template):
                if d != axis:
                    keep &= _member(c[:, d], t)
            if keep.any():
                hit |= np.isin(pos, c[keep, axis])
            continue
        if any(d != axis and D._hits(p.region[d], t) is None for d, t in enumerate(template)):
            continue
        r = p.region[axis]
        a, b, step = (r, r + 1, 1) if isinstance(r, int) else (r.start, r.stop, r.step)
        if step == 1:
            i, j = np.searchsorted(pos, [a, b])
            hit[i:j] = True
        else:
            hit |= np.isin(pos, np.arange(a, b, step))
    return hit


class DeltaReader:
    """The delta (current - base) of one tensor over many windows along one axis, from the patch pieces alone.
    Each read adds only the pieces whose extent along that axis reaches the window, so walking thousands of
    windows over thousands of pieces does not cost windows x pieces. Same values as ws.read(..., "delta")."""

    def __init__(self, ws, name: str, axis: int, extra: dict | None = None):
        self.shape = ws.shape(name)
        self.pieces = [(p, s) for p, s in ws.pieces(name) if s] + [(p, 1.0) for p in (extra or {}).get(name, ())]
        lo, hi = [], []
        for p, _ in self.pieces:
            if p.kind == "sparse":
                c = p.coords[:, axis]
                a, b = (int(c.min()), int(c.max()) + 1) if len(c) else (0, 0)
            else:
                r = p.region[axis]
                a, b = (r, r + 1) if isinstance(r, int) else (min(r[0], r[-1]), max(r[0], r[-1]) + 1) if len(r) else (0, 0)
            lo.append(a)
            hi.append(b)
        self.lo, self.hi = np.asarray(lo, dtype=np.int64), np.asarray(hi, dtype=np.int64)
        self.axis = axis

    def read(self, region: tuple) -> torch.Tensor:
        out = torch.zeros(tuple(len(r) for r in region if isinstance(r, range)))
        r = region[self.axis]
        if isinstance(r, int):
            a, b = r, r + 1
        elif not len(r):
            return out
        else:
            a, b = min(r[0], r[-1]), max(r[0], r[-1]) + 1
        for i in np.nonzero((self.lo < b) & (self.hi > a))[0]:
            p, s = self.pieces[i]
            D.add_to(out, region, p, s)
        return out


def delta_windows(ws, name: str, template: tuple, axis: int, pos: np.ndarray, extra: dict | None = None):
    """(window region, kept axis, first position, positions in the window, delta over the window).

    Reads only the patch pieces (variant "delta"): the checkpoint is never touched, so windows are as wide
    as the budget allows."""
    k = kept_before(template, axis)
    reader = DeltaReader(ws, name, axis, extra)
    for w in windows(pos, other_size(template, axis), gap=None):
        lo, hi = int(w[0]), int(w[-1]) + 1
        region = with_axis(template, axis, range(lo, hi))
        yield region, k, lo, w, reader.read(region)


def edited_positions(ws, name: str, template: tuple, axis: int, pos: np.ndarray, extra: dict | None = None) -> np.ndarray:
    """Exactly those of `pos` where the current weights differ from base (within `template`)."""
    pieces = pieces_of(ws, name, extra)
    if not pieces or len(pos) == 0:
        return pos[:0]
    cand = pos[axis_hits(pieces, template, axis, pos)]
    out = [cand[:0]]
    for _, k, lo, w, d in delta_windows(ws, name, template, axis, cand, extra):
        rows = d.ne(0).movedim(k, 0).reshape(d.shape[k], -1).any(1) if d.dim() else d.ne(0).reshape(1)
        out.append(w[rows[torch.as_tensor(w - lo)].numpy()])
    return np.concatenate(out)


def best_axis(pieces, region: tuple) -> int | None:
    """The kept axis along which the pieces reach the smallest share of the region (fewest positions to walk)."""
    best, share = None, 2.0
    for d, r in enumerate(region):
        if not isinstance(r, range) or len(r) == 0:
            continue
        pos = np.sort(np.asarray(r, dtype=np.int64))
        s = axis_hits(pieces, region, d, pos).sum() / len(pos)
        if s < share:
            best, share = d, s
    return best


def _is_region(index, shape) -> bool:
    from .workspace import _is_region as is_region
    return is_region(index, shape)


def neuron_axis(st, part) -> tuple[int, range, tuple]:
    """(native axis, native position of neuron 0..width-1 along it, region of the whole part)."""
    shape = tuple(st.shapes[part.name])
    base = list(part.base) + [slice(None)] * (len(shape) - len(part.base))
    kept = [d for d, i in enumerate(base) if isinstance(i, slice)]
    axis = kept[part.roles.index("neuron")]
    template = norm_index(tuple(base), shape)
    return axis, template[axis], template


def neuron_parts(st, l: int, roles=NEURON_ROLES) -> list:
    L = st.layers[l]
    return [L.mlp[r] for r in roles if r in L.mlp]


# ------------------------------------------------------------------------------ what a selection is
def _layer_of(st) -> dict:
    out = {}
    for l, L in enumerate(st.layers):
        for p in list(L.attn.values()) + list(L.mlp.values()) + list(L.norms):
            out.setdefault(p.name, l)
        for n in L.other:
            out.setdefault(n, l)
    return out


# ------------------------------------------------------------------------------ the weights an action reaches
@dataclass
class Union:
    """The weights an action reaches in one tensor, each exactly once: `lines` (positions along `axis`, with all
    of every other axis) and `boxes` (disjoint products of positions per axis, outside the lines)."""
    shape: tuple
    axis: int | None
    lines: np.ndarray
    boxes: list

    @property
    def size(self) -> int:
        n = 0
        if self.axis is not None and len(self.lines):
            n = len(self.lines) * math.prod(d for k, d in enumerate(self.shape) if k != self.axis)
        return n + sum(math.prod(len(a) for a in b) for b in self.boxes)


def _box(ws, name: str, idx) -> tuple:
    """A (tensor, index) as sorted, unique positions per axis (an int index is one position)."""
    shape = ws.shape(name)
    region = idx if _is_region(idx, shape) else norm_index(idx, shape)      # wraps negative ints (W[4, -17])
    out = tuple(np.unique(np.asarray([r] if isinstance(r, int) else r, dtype=np.int64)) for r in region)
    for d, (a, n) in enumerate(zip(out, shape)):
        if len(a) and (a[0] < 0 or a[-1] >= n):
            raise IndexError(f"index out of range for axis {d} of {name} (size {n})")
    return out


def _full(box: tuple, shape: tuple) -> list[bool]:
    return [len(a) == n for a, n in zip(box, shape)]


def union(shape: tuple, boxes: list, hint: int | None = None) -> Union:
    """Union of boxes in one tensor, as lines along one axis plus disjoint boxes: each weight once."""
    shape = tuple(shape)
    nd = len(shape)
    boxes = [b for b in boxes if all(len(a) for a in b)]
    if not boxes:
        return Union(shape, None, np.zeros(0, dtype=np.int64), [])
    line_axes = []                                     # for each box, the axes it is a line along
    for b in boxes:
        f = _full(b, shape)
        line_axes.append([d for d in range(nd) if all(f[k] for k in range(nd) if k != d)])
    count = [0] * nd
    for b, ax in zip(boxes, line_axes):
        for d in ax:
            count[d] += len(b[d])
    if hint is not None and count[hint]:
        axis = hint
    elif any(count):
        axis = max(range(nd), key=lambda d: (count[d], d))
    else:
        axis = None
    lines = np.zeros(0, dtype=np.int64)
    rest = []
    for b, ax in zip(boxes, line_axes):
        if axis is not None and axis in ax:
            lines = np.union1d(lines, b[axis])
        else:
            rest.append(b)
    if axis is not None and len(lines):
        rest = [b[:axis] + (np.setdiff1d(b[axis], lines, assume_unique=True),) + b[axis + 1:] for b in rest]
        rest = [b for b in rest if all(len(a) for a in b)]
    return Union(shape, axis, lines, _disjoint(rest, shape))


def _overlaps(a: tuple, b: tuple) -> bool:
    return all(len(np.intersect1d(x, y, assume_unique=True)) for x, y in zip(a, b))


def _disjoint(boxes: list, shape: tuple) -> list:
    if len(boxes) <= 1:
        return boxes
    if len(shape) == 1:
        return [(np.unique(np.concatenate([b[0] for b in boxes])),)]
    if len(shape) != 2:
        if any(_overlaps(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]):
            raise Refusal("Members of this selection overlap inside a tensor with more than two axes; "
                          "save them as separate selections.")
        return boxes
    by_rows: dict = {}                                 # boxes over the same rows: one box, columns merged
    for r, c in boxes:
        k = r.tobytes()
        by_rows[k] = (r, np.union1d(by_rows[k][1], c) if k in by_rows else c)
    boxes = list(by_rows.values())
    if len(boxes) <= 1 or not any(_overlaps(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]):
        return boxes
    rows = np.unique(np.concatenate([r for r, _ in boxes]))
    if len(rows) * len(boxes) > 50_000_000:
        raise Refusal("Members of this selection overlap in too many separate pieces of one tensor; "
                      "save fewer overlapping slices.")
    member = np.stack([np.isin(rows, r, assume_unique=True) for r, _ in boxes], 1)
    groups, inverse = np.unique(member, axis=0, return_inverse=True)
    out = []
    for g, which in enumerate(groups):
        if which.any():
            cols = np.unique(np.concatenate([boxes[i][1] for i in np.nonzero(which)[0]]))
            out.append((rows[inverse.reshape(-1) == g], cols))
    return out


def _targets(ws, sel: Selection, action: str, other: list) -> tuple[dict, dict]:
    """{tensor: [boxes]} an action reaches, and {tensor: axis} to prefer as the line axis.

    scale / ablate: what each neuron writes (its output-projection column), what each head writes (its
    output-projection columns), the selected weights themselves. revert (and the selection's size): every
    part of each neuron and head, and the selected weights."""
    st = ws.struct
    out: dict = {}
    hint: dict = {}
    every = action in ("revert", "all")
    for l in sorted(sel.neurons):
        ns = np.asarray(sorted(sel.neurons[l]), dtype=np.int64)
        parts = neuron_parts(st, l) if every else [st.layers[l].mlp["W_out"]]
        for part in parts:
            if not every and not ws.editable(part.name):
                other.append({"member": f"{len(ns)} neurons of layer {l}", "reason": _not_editable(ws, part.name)})
                continue
            axis, positions, template = neuron_axis(st, part)
            box = list(_box(ws, part.name, template))
            box[axis] = np.sort(positions.start + ns * positions.step)
            out.setdefault(part.name, []).append(tuple(box))
            hint[part.name] = axis
    for l, h in sorted(sel.heads):
        if every:
            for name, idx in st.head_slices(l, h):
                out.setdefault(name, []).append(_box(ws, name, idx))
            continue
        o = st.layers[l].attn["W_O"]
        if not ws.editable(o.name):
            other.append({"member": f"L{l}H{h}", "reason": _not_editable(ws, o.name)})
            continue
        out.setdefault(o.name, []).append(_box(ws, o.name, st.part_index(o, head=h)))
    for name, idx in sel.slices:
        if not ws.editable(name):
            if action != "all":
                other.append({"member": A.slice_str((name, idx)), "reason": _not_editable(ws, name)})
            continue
        out.setdefault(name, []).append(_box(ws, name, idx))
    return out, hint


def unions(ws, sel: Selection, action: str, other: list | None = None) -> dict:
    other = [] if other is None else other
    t, hint = _targets(ws, sel, action, other)
    return {name: union(ws.shape(name), boxes, hint.get(name)) for name, boxes in t.items()}


def _sub_boxes(box: tuple):
    """A box as products of runs: [(a, b) per axis]."""
    per = [runs(a) for a in box]
    for combo in itertools.product(*per):
        yield combo


def _chunks(sub: list, budget: int | None = None):
    """Split [(a, b) per axis] into pieces of at most `budget` weights (each one store read; default WINDOW)."""
    budget = WINDOW if budget is None else budget
    size = math.prod(b - a for a, b in sub)
    if size <= budget:
        yield sub
        return
    for d, (a, b) in enumerate(sub):
        if b - a > 1:
            per = math.prod(y - x for k, (x, y) in enumerate(sub) if k != d)
            step = max(1, budget // per)
            for x in range(a, b, step):
                yield from _chunks(sub[:d] + [(x, min(b, x + step))] + sub[d + 1:], budget)
            return
    yield sub


def _expr(name: str, shape: tuple, sub: list) -> str:
    return A.to_script(name, tuple(slice(None) if (a, b) == (0, n) else slice(a, b) for (a, b), n in zip(sub, shape)))


def _line_box(u: Union, positions: np.ndarray) -> tuple:
    return tuple(positions if d == u.axis else np.arange(n, dtype=np.int64) for d, n in enumerate(u.shape))


def _neuron_ids(st, name: str, u: Union, layer_parts: dict, positions: np.ndarray):
    """(layer, neuron ids) when `positions` along the union's axis are exactly neurons of a neuron part."""
    if name not in layer_parts or not len(positions):
        return None
    l, part = layer_parts[name]
    axis, pr, template = neuron_axis(st, part)
    if axis != u.axis or any(len(r) != n for d, (r, n) in enumerate(zip(template, u.shape)) if d != axis):
        return None
    rel = positions - pr.start
    if (rel % pr.step).any() or (rel < 0).any() or (rel // pr.step >= len(pr)).any():
        return None
    return l, rel // pr.step


def _layer_parts(st, layers, roles) -> dict:
    return {p.name: (l, p) for l in layers for p in neuron_parts(st, l, roles)}


def _delta_counts(ws, name: str, u: Union, extra: dict | None = None):
    """(changed weights, squared norm of the change, edited line positions, edited boxes) inside a union.
    From the patch pieces alone: the checkpoint is never read."""
    n, sq = 0, 0.0
    edited_lines, edited_boxes = np.zeros(0, dtype=np.int64), []
    if not pieces_of(ws, name, extra):
        return n, sq, edited_lines, edited_boxes
    fast = _counts_from_pieces(ws, name, u, extra)
    if fast is not None:
        return fast
    whole = tuple(range(d) for d in u.shape)
    if u.axis is not None and len(u.lines):
        reader = DeltaReader(ws, name, u.axis, extra)
        pos = u.lines[axis_hits([p for p, _ in reader.pieces], whole, u.axis, u.lines)]
        got = []
        for w in windows(pos, other_size(whole, u.axis), gap=None):
            lo, hi = int(w[0]), int(w[-1]) + 1
            d = reader.read(with_axis(whole, u.axis, range(lo, hi))).index_select(u.axis, torch.as_tensor(w - lo))
            nz = d.ne(0)
            n, sq = n + int(nz.sum()), sq + float(d.double().pow(2).sum())
            got.append(w[nz.movedim(u.axis, 0).reshape(len(w), -1).any(1).numpy()])
        if got:
            edited_lines = np.concatenate(got)
    if u.boxes:
        reader = DeltaReader(ws, name, 0, extra)
        for box in u.boxes:
            for sub in _sub_boxes(box):
                for ch in _chunks(list(sub)):
                    d = reader.read(tuple(range(a, b) for a, b in ch))
                    c = int(d.ne(0).sum())
                    if c:
                        n, sq = n + c, sq + float(d.double().pow(2).sum())
                        edited_boxes.append(ch)
    return n, sq, edited_lines, edited_boxes


def _line_flags(p: D.Piece, axis: int) -> np.ndarray:
    """For a block piece: which of its positions along `axis` hold a non-zero value (kept on the piece)."""
    cache = p.__dict__.setdefault("_lines_kept", {})
    if axis not in cache:
        v = p.values.ne(0)
        if isinstance(p.region[axis], int):     # an integer index: one position, and that axis is not in the values
            cache[axis] = np.array([bool(v.any())])
        else:
            k = kept_before(p.region, axis)
            cache[axis] = v.movedim(k, 0).reshape(v.shape[k], -1).any(1).numpy()
    return cache[axis]


def _counts_from_pieces(ws, name: str, u: Union, extra: dict | None):
    """The counts of `_delta_counts` without building any delta, when that is exact: every piece is a block whose
    positions along the union's axis are consecutive lines of the union, and no two pieces meet along that axis
    (so no weight is counted twice). Otherwise None. This is the common case of an edit made by selection."""
    if u.boxes or u.axis is None or not len(u.lines):
        return None
    ps = [(p, s) for p, s in ws.pieces(name) if s] + [(p, 1.0) for p in (extra or {}).get(name, ())]
    lo, hi = [], []
    for p, _ in ps:
        if p.kind != "block":
            return None
        r = p.region[u.axis]
        a, b = (r, r + 1) if isinstance(r, int) else (r.start, r.stop) if r.step == 1 else (None, None)
        if a is None:
            return None
        lo.append(a)
        hi.append(b)
    lo, hi = np.asarray(lo, dtype=np.int64), np.asarray(hi, dtype=np.int64)
    order = np.argsort(lo, kind="stable")
    if len(order) > 1 and (lo[order][1:] < np.maximum.accumulate(hi[order])[:-1]).any():
        return None                                          # pieces meet: count from the delta itself
    i, j = np.searchsorted(u.lines, lo), np.searchsorted(u.lines, hi)
    if ((j - i) != (hi - lo)).any():
        return None                                          # a piece reaches past the union: not all of it counts
    n, sq, lines = 0, 0.0, []
    for (p, s), a in zip(ps, lo):
        st = p.stats(rank=False)
        n, sq = n + st["changed"], sq + st["sq"] * s * s
        f = _line_flags(p, u.axis)
        lines.append(a + np.nonzero(f)[0])
    return n, sq, np.unique(np.concatenate(lines)) if lines else np.zeros(0, dtype=np.int64), []


def summary(ws, addr: str, sel: Selection | None = None) -> dict:
    """Size of a selection (neurons, heads, layers it touches, weights in every part of them, each counted
    once) and its members, listed up to SHOW_MEMBERS."""
    sel = sel or expand(ws, addr)
    st = ws.struct
    layer_of = _layer_of(st) if st.recognised else {}
    layers = set(sel.neurons) | {l for l, _ in sel.heads} | {layer_of[n] for n, _ in sel.slices if n in layer_of}
    try:
        weights = sum(u.size for u in unions(ws, sel, "all").values())
    except Refusal:                                   # overlapping members of a >2-D tensor: counted per member
        weights = None
    return {"neurons": sel.n_neurons, "heads": len(sel.heads), "tensors": len(sel.slices),
            "layers": sorted(layers), "weights": weights,
            "by_layer": {str(l): len(sel.neurons[l]) for l in sorted(sel.neurons)},
            "n_members": len(sel.members), "members": sel.members[:SHOW_MEMBERS],
            "other": sel.other[:SHOW_MEMBERS], "n_other": len(sel.other)}


def changed(ws, addr: str, sel: Selection | None = None) -> tuple[int, float]:
    """(edited weights, squared norm of their change) in a selection (every part, each weight once), exactly,
    from the patch pieces alone. Where the members cannot be put together (they overlap in a tensor of more than
    two axes) the count is made over the delta of each such tensor with a mask of what the selection reaches,
    still exact; a tensor above WINDOW weights is counted slice by slice (ws.changed), which may count an
    overlap twice. Never 0 for an edited selection."""
    sel = sel or expand(ws, addr)
    n, sq = 0, 0.0
    try:
        us = unions(ws, sel, "all")
    except Refusal:
        return _changed_by_mask(ws, sel)
    for name, u in us.items():
        c, s2, _, _ = _delta_counts(ws, name, u)
        n, sq = n + c, sq + s2
    return n, sq


def _changed_by_mask(ws, sel: Selection) -> tuple[int, float]:
    targets, _ = _targets(ws, sel, "all", [])
    n, sq = 0, 0.0
    for name, boxes in targets.items():
        if not ws.pieces(name):
            continue
        shape = tuple(ws.shape(name))
        if math.prod(shape) <= WINDOW:
            mask = np.zeros(shape, dtype=bool)
            for b in boxes:
                mask[np.ix_(*b)] = True
            d = ws.read(name, (), "delta")[torch.as_tensor(mask)]
            n, sq = n + int(d.ne(0).sum()), sq + float(d.double().pow(2).sum())
        else:
            for b in boxes:
                for sub in _sub_boxes(b):
                    c, s2 = ws.changed(name, tuple(slice(a, z) for a, z in sub))
                    n, sq = n + c, sq + s2
    return n, sq


STEER_INSTEAD = "To change it without a weight patch, scale it at run time in the Run tab's steering panel."


def _limit(ws, written: int):
    """The largest patch this selection edit may write now (wsp.edit_limit), with the old fixed limit as the
    fallback where memory cannot be measured. Its script reads at most WINDOW weights at once (_chunks,
    read_windows), which is the scratch its preflights charge. Memory and disk are measured only above
    edit_limit.SMALL_EDIT weights, as in the script guard (wscript.Interp._add), so buttons and console agree."""
    from . import edit_limit
    return edit_limit.allowed(ws, written, scratch=min(written, WINDOW) * edit_limit.MEMORY_BYTES, fallback=True)


def _too_large(weights: int, limit, steerable: bool = False) -> str:
    """`limit`: the edit_limit.Limit refused. `steerable`: run-time steering can scale the selection instead."""
    from . import edit_limit
    return edit_limit.refusal(weights, limit, STEER_INSTEAD if steerable else None)


def _steerable(sel: Selection, addr: str) -> bool:
    """Run-time steering can scale this instead: it holds neurons or heads, or it is a whole MLP (blocks.L.mlp)."""
    return bool(sel.n_neurons or sel.heads or re.fullmatch(r"blocks\.\d+\.mlp", addr))


def _plain(e: Exception) -> str:
    s = str(e)
    return s[1:-1] if len(s) > 1 and s[0] == s[-1] and s[0] in "'\"" else s


def _not_editable(ws, name: str) -> str:
    return f"{name} is stored as {ws.store.dtype(name)}, which cannot be edited"


MAX_LINES = 20_000                 # script lines one action may emit (scattered slices; neurons are one line per layer)


def plan(ws, addr: str, action: str, sel: Selection | None = None) -> dict:
    """What an action would do, each weight once, without reading the checkpoint:

    {"ops": [("neurons", L, ids) | ("box", name, [(a, b) per axis])], "weights": weights it changes (scale:
    every weight it reaches; revert: every weight that differs from base), "written": weights it stores,
    "neuron_count", "head_count", "slice_count", "other", "changed_sq"}.

    For revert only edited weights are kept (exact, from the patch pieces)."""
    if action not in ("scale", "ablate", "revert"):
        raise ValueError(f"unknown action {action!r} (scale, ablate or revert)")
    sel = sel or expand(ws, addr)
    st = ws.struct
    other = list(sel.other)
    us = unions(ws, sel, action, other)
    ops, weights, written, sq = [], 0, 0, 0.0
    if action == "revert":
        lp = _layer_parts(st, sel.neurons, NEURON_ROLES)
        per_layer: dict = {}
        rest: list = []
        for name, u in us.items():
            c, s2, lines, boxes = _delta_counts(ws, name, u)
            if not c:
                continue
            weights, sq = weights + c, sq + s2
            written += sum(math.prod(b - a for a, b in ch) for ch in boxes)
            hit = _neuron_ids(st, name, u, lp, lines) if len(lines) else None
            if hit:
                l, ids = hit
                keep = np.isin(ids, np.asarray(sorted(sel.neurons[l]), dtype=np.int64))
                per_layer.setdefault(l, set()).update(int(i) for i in ids[keep])
                rest.append((name, u, lines, ids, keep))
            elif len(lines):
                rest.append((name, u, lines, None, None))
            ops += [("revert", name, ch) for ch in boxes]
        line_ops = []
        for name, u, lines, ids, keep in rest:
            other_n = math.prod(n for d, n in enumerate(u.shape) if d != u.axis)
            written += len(lines) * other_n
            if ids is not None:
                l = _neuron_ids(st, name, u, lp, lines)[0]
                inside = np.isin(ids, np.asarray(sorted(per_layer.get(l, ())), dtype=np.int64))
                lines = lines[~inside]                     # those are reverted by revert_neurons below
            for a, b in runs(lines):
                line_ops.append(("revert", name, [(a, b) if d == u.axis else (0, n) for d, n in enumerate(u.shape)]))
        ops = [("revert_neurons", l, np.asarray(sorted(v), dtype=np.int64)) for l, v in sorted(per_layer.items()) if v] \
            + line_ops + ops
        neuron_count = sum(len(v) for v in per_layer.values())
        head_count = sum(1 for l, h in sorted(sel.heads)
                         if any(pieces_of(ws, n) and ws.changed(n, i)[0] for n, i in st.head_slices(l, h)))
    else:
        lp = _layer_parts(st, sel.neurons, ("W_out",))
        heads: dict = {}
        for l, h in sorted(sel.heads):
            heads.setdefault(st.layers[l].attn["W_O"].name, []).append((l, h))
        for name, u in us.items():
            weights += u.size
            hit = _neuron_ids(st, name, u, lp, u.lines) if len(u.lines) else None
            left = u.lines
            if hit:
                ops.append(("neurons", hit[0], hit[1]))
                left = left[:0]
            for l, h in heads.get(name, ()):          # a head's whole output block: written as scale_head
                hb = _box(ws, name, st.part_index(st.layers[l].attn["W_O"], head=h))
                if (u.axis is not None and all(len(a) == n for d, (a, n) in enumerate(zip(hb, u.shape)) if d != u.axis)
                        and np.isin(hb[u.axis], left, assume_unique=True).all()):
                    ops.append(("head", l, h))
                    left = np.setdiff1d(left, hb[u.axis], assume_unique=True)
            if len(left):
                ops += [("box", name, ch) for sub in _sub_boxes(_line_box(u, left)) for ch in _chunks(list(sub))]
            ops += [("box", name, ch) for box in u.boxes for sub in _sub_boxes(box) for ch in _chunks(list(sub))]
        written = weights
        neuron_count = sum(len(v) for l, v in sel.neurons.items() if ws.editable(st.layers[l].mlp["W_out"].name))
        head_count = sum(1 for l, h in sel.heads if ws.editable(st.layers[l].attn["W_O"].name))
    return {"ops": ops, "weights": weights, "written": written, "changed_sq": sq, "other": other,
            "neuron_count": neuron_count, "head_count": head_count, "slice_count": len(sel.slices)}


def _ints(ns) -> str:
    ns = [int(n) for n in ns]
    if len(ns) > 1 and ns[-1] - ns[0] == len(ns) - 1:
        return f"range({ns[0]}, {ns[-1] + 1})"
    return "[" + ", ".join(str(n) for n in ns) + "]"


def _render(ws, op, f) -> str:
    kind = op[0]
    if kind == "neurons":
        return f"scale_neurons({op[1]}, {_ints(op[2])}, {f!r})"
    if kind == "revert_neurons":
        return f"revert_neurons({op[1]}, {_ints(op[2])})"
    if kind == "head":
        return f"scale_head(({op[1]}, {op[2]}), {f!r})"
    x = _expr(op[1], ws.shape(op[1]), op[2])
    return f"revert({x})" if kind == "revert" else f"{x} *= {f!r}"


def script(ws, addr: str, action: str, factor=None) -> dict:
    """Contract C2: one weight script for an action on any selection, each weight reached once.

    -> {"script", "neurons", "heads", "weights", "other"}; raises Refusal when the action cannot be done."""
    addr = A.normalize(addr)
    sel = expand(ws, addr)
    if action == "scale":
        try:
            f = float(factor)
        except (TypeError, ValueError):
            raise Refusal("Give a number to scale by.") from None
        if not math.isfinite(f):
            raise Refusal("The scale factor must be a finite number.")
        if f == 1.0:
            raise Refusal("Scaling by 1 changes nothing.")
    elif action == "ablate":
        f = 0.0
    else:
        f = None
    p = plan(ws, addr, action, sel)
    counts = {"neurons": p["neuron_count"], "heads": p["head_count"], "weights": p["weights"], "other": p["other"]}
    if not p["ops"]:
        if action == "revert":
            raise Refusal("Nothing in this selection has been edited, so there is nothing to revert.", **counts)
        raise Refusal(_nothing_editable(p["other"]), **counts)
    limit = _limit(ws, p["written"])
    if over := limit.exceeded(p["written"]):
        raise Refusal(_too_large(p["written"], over, action != "revert" and _steerable(sel, addr)), **counts)
    if len(p["ops"]) > MAX_LINES:
        raise Refusal(f"This selection is scattered over {len(p['ops']):,} separate pieces of tensors (the limit is "
                      f"{MAX_LINES:,}); save fewer, larger slices.", **counts)
    if action != "revert" and _all_zero(ws, p):
        raise Refusal("These weights are already zero." if f == 0 else "These weights are all zero, so scaling them changes nothing.", **counts)
    from . import edit_limit
    return {"script": "\n".join(_render(ws, op, f) for op in p["ops"]), **counts,
            "cost": edit_limit.cost(ws, p["written"])}


def _nothing_editable(other: list) -> str:
    if not other:
        return "This selection is empty."
    reasons = sorted({o["reason"] for o in other})
    return "Nothing in this selection can be edited: " + "; ".join(reasons[:3]) + ("; …" if len(reasons) > 3 else "")


def _all_zero(ws, p: dict, limit: int = 1_000_000) -> bool:
    """True when the (small) set of weights an action would multiply is all zero already."""
    if p["weights"] > limit:
        return False
    st = ws.struct
    for op in p["ops"]:
        if op[0] == "neurons":
            part = st.layers[op[1]].mlp["W_out"]
            axis, positions, template = neuron_axis(st, part)
            step = max(1, WINDOW // other_size(template, axis))      # each read within the store's read limit
            for a, b in runs(np.asarray(op[2], dtype=np.int64)):
                for x in range(a, b, step):
                    if bool(ws.read(part.name, with_axis(template, axis, positions[x:min(b, x + step)])).any()):
                        return False
        elif op[0] == "head":
            o = st.layers[op[1]].attn["W_O"]
            if bool(ws.read(o.name, st.part_index(o, head=op[2])).any()):
                return False
        elif bool(ws.read(op[1], tuple(range(a, b) for a, b in op[2])).any()):
            return False
    return True


def _reach(ws, sel: Selection) -> dict:
    """What revert reaches beyond the click, in words, for the revert button's title."""
    st = ws.struct
    notes = []
    if sel.neurons:
        notes.append("for each neuron, the weights that read its input (gate and up rows) as well as what it writes")
    per = st.n_heads // max(1, getattr(st, "n_kv_heads", None) or st.n_heads) if sel.heads else 1
    if sel.heads and per > 1:
        shared = []
        for l, h in sorted(sel.heads):
            g = h // per
            mates = [f"L{l}H{x}" for x in range(g * per, (g + 1) * per) if x != h and (l, x) not in sel.heads]
            if mates:
                shared.append(f"L{l}H{h} with {', '.join(mates)}")
        if shared:
            notes.append("the key and value rows each head shares with the other heads of its group ("
                         + "; ".join(shared[:4]) + ("; …" if len(shared) > 4 else "") + ")")
    return notes


REVERT_NOTE = ("Adds a patch that cancels the current edits here. It is worked out from the patches as they are "
               "now: disabling, re-strengthening or deleting them later leaves this patch cancelling an edit that "
               "is no longer there.")


def actions(ws, addr: str, sel: Selection | None = None) -> dict:
    """For each inspector action: {"enabled", "reason"}. Costs the patch pieces, never a checkpoint read."""
    addr = A.normalize(addr)
    sel = sel or expand(ws, addr)
    out = {}
    try:
        w = plan(ws, addr, "scale", sel)
    except Refusal as e:
        off = {"enabled": False, "reason": str(e)}
        # no "changed": the plan could not count it, so the node counts slice by slice (selections.changed)
        return {k: off for k in ("scale", "ablate", "console", "revert")} | {"weights": 0, "other": [], "n_other": 0}
    r = plan(ws, addr, "revert", sel)
    # One limit for the view: memory is asked at most once, for the larger of the two edits (scale and revert
    # read in the same windows, so the scratch is the same); each is then checked against it.
    shared = []
    def limit():
        if not shared:
            shared.append(_limit(ws, max(w["written"], r["written"])))
        return shared[0]
    from . import edit_limit
    def ask(written):
        return (f" (a patch of about {edit_limit.gib(written * edit_limit.PATCH_BYTES)}; you are asked to confirm first)"
                if written > edit_limit.CONFIRM_ABOVE else "")
    if not w["ops"]:
        why = _nothing_editable(w["other"])
        out["scale"] = out["ablate"] = out["console"] = {"enabled": False, "reason": why}
    elif over := limit().exceeded(w["written"]):
        why = _too_large(w["written"], over, _steerable(sel, addr))
        out["scale"] = out["ablate"] = out["console"] = {"enabled": False, "reason": why}
    else:
        what = _what(sel, w, sum(u.size for u in unions(ws, sel, "all").values()))
        out["scale"] = {"enabled": True, "reason": f"multiplies {what}{ask(w['written'])}"}
        out["ablate"] = {"enabled": True, "reason": f"sets {what} to zero{ask(w['written'])}"}
        out["console"] = {"enabled": True, "reason": "puts the scaling script in the console to edit before applying"}
    reach = _reach(ws, sel)
    if not r["ops"]:
        out["revert"] = {"enabled": False, "reason": "nothing here has been edited"}
    elif over := limit().exceeded(r["written"]):
        out["revert"] = {"enabled": False, "reason": _too_large(r["written"], over)}
    else:
        out["revert"] = {"enabled": True, "reason": f"puts {r['weights']:,} edited weights back to their base values"
                         + (" — including " + "; and ".join(reach) if reach else "") + ask(r["written"]) + ". "
                         + REVERT_NOTE}
    out["weights"] = w["weights"]
    out["changed"], out["changed_sq"] = r["weights"], r["changed_sq"]
    out["other"] = w["other"][:20]
    out["n_other"] = len(w["other"])
    return out


def node_extras(ws, addr: str, sel: Selection | None = None) -> dict:
    """The node answer's actions, size summary, breadcrumb and shortcuts. They need the model's structure
    (parts, roles, shapes); where it cannot describe this address the node is still shown, with every action
    disabled and the reason given (fail closed: an enabled button must change weights), and no breadcrumb."""
    from . import hierarchy
    out = {}
    unavailable = {}
    # Only a structure that lacks what these need (a part without roles, no shapes: AttributeError) is caught;
    # any other error is a planning bug and surfaces.
    try:
        out["actions"] = actions(ws, addr, sel)
    except AttributeError as e:
        log.warning("actions for %s: %s", addr, _plain(e))
        unavailable["actions"] = why = f"Cannot work out what this selection edits ({_plain(e)})."
        off = {"enabled": False, "reason": why}
        out["actions"] = {k: off for k in ("scale", "ablate", "console", "revert")} | {"weights": 0, "other": [],
                                                                                        "n_other": 0}
    for key, fn, empty in (("selection", lambda: summary(ws, addr, sel), None),
                           ("crumbs", lambda: hierarchy.crumbs(ws, addr), []),
                           ("jumps", lambda: hierarchy.jumps(ws, addr), [])):
        try:
            out[key] = fn()
        except AttributeError as e:
            log.warning("%s for %s: %s", key, addr, _plain(e))
            unavailable[key] = _plain(e)
            out[key] = empty
    if unavailable:
        out["unavailable"] = unavailable
    return out


def _what(sel: Selection, p: dict, every: int | None = None) -> str:
    """What scale / ablate reach, in words, with both counts when they differ: the weights the action changes
    (what neurons and heads write) and the weights in every part of the selection (the size shown above)."""
    parts = []
    k = p["slice_count"]
    if p["neuron_count"]:
        parts.append("what the neuron writes" if p["neuron_count"] == 1 and not p["head_count"] and not k
                     else f"what {p['neuron_count']:,} neurons write")
    if p["head_count"]:
        parts.append("what the head writes" if p["head_count"] == 1 and not parts and not k
                     else "what 1 head writes" if p["head_count"] == 1 else f"what {p['head_count']} heads write")
    if k:
        parts.append("the selected weights" if k == 1 and not parts else f"{k} selected tensor slices")
    n = f"{p['weights']:,} weights"
    if every is not None and every != p["weights"]:
        n += f" of the {every:,} in every part of the selection"
    return " and ".join(parts) + f" ({n})"


# ------------------------------------------------------------------------------ what a revert depends on
def _extents(pieces) -> np.ndarray:
    """[pieces, axes, 2] (lo, hi) of each piece along each axis."""
    out = []
    for p in pieces:
        if p.kind == "sparse":
            c = p.coords.numpy()
            if not len(c):
                continue
            out.append(np.stack([c.min(0), c.max(0) + 1], 1))
            continue
        ext = []
        for r in p.region:
            if isinstance(r, int):
                ext.append((r, r + 1))
            elif len(r):
                ext.append((min(r[0], r[-1]), max(r[0], r[-1]) + 1))
            else:
                ext = None
                break
        if ext is not None:
            out.append(np.asarray(ext, dtype=np.int64))
    return np.stack(out) if out else np.zeros((0, 0, 2), dtype=np.int64)


def _meet(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> bool:
    """Whether some extent of `a` overlaps some extent of `b` on every axis."""
    if not len(a) or not len(b):
        return False
    for i in range(0, len(a), chunk):
        x = a[i:i + chunk, None]
        if ((x[..., 0] < b[None, :, :, 1]) & (b[None, :, :, 0] < x[..., 1])).all(-1).any():
            return True
    return False


def compensates(ws, deltas: dict, names) -> list[dict]:
    """The enabled patches a revert was worked out against: those with pieces that meet the revert's pieces in the
    reverted tensors (by extent, so it may name a patch whose overlap is only near). Recorded on the revert patch,
    because it cancels their edits as they are now: changing them later leaves it cancelling something else."""
    mine = {n: _extents(deltas.get(n, ())) for n in names}
    out = []
    for p in ws.patches:
        if not p.enabled or not p.strength:
            continue
        if any(n in p.deltas and _meet(mine[n], _extents(p.deltas[n])) for n in mine):
            out.append({"id": p.id, "strength": p.strength, "message": p.message})
    return out


def revert_warning(ws, pid: str, action: str, before: dict) -> str | None:
    """A plain warning when a patch that an enabled revert patch was worked out against is deleted, disabled or
    re-strengthened. The change is made; the revert patch is left as it is."""
    deps = [(q, d) for q in ws.patches if q.enabled and q.id != pid
            for d in (q.source or {}).get("compensates", ()) if d.get("id") == pid]
    if not deps:
        return None
    p = None if action == "delete" else ws.patch(pid)
    if p is not None and p.enabled == before.get("enabled") and p.strength == before.get("strength"):
        return None                                          # nothing changed
    what = "was deleted" if p is None else "was disabled" if not p.enabled else f"is now at ×{p.strength:g}"
    deps = [(q, d) for q, d in deps if p is None or not p.enabled or p.strength != d.get("strength")]
    if not deps:
        return None                                          # back to what the reverts were worked out against
    names = ", ".join(f'{q.id} ("{q.message}", worked out at ×{d.get("strength", 1):g})' for q, d in deps[:3])
    more = f" and {len(deps) - 3} more" if len(deps) > 3 else ""
    return (f"Patch {pid} {what}, but revert patch {names}{more} cancels its edits as they were: it now cancels an "
            f"edit that is no longer there. Disable or delete the revert patch too, or set {pid} back.")
