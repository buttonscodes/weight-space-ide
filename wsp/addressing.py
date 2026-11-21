"""Addresses: one string names any part of any model, at any level of the hierarchy.

Semantic addresses (available when the architecture is recognised; the same for every model):

    model                              every tensor
    embed | pos_embed | unembed | final_norm
    blocks.3                           a layer
    blocks.3.attn | blocks.3.mlp       a sublayer
    blocks.3.attn.head.5   (= L3H5)    a head: its rows of Q, the K / V it uses, its columns of O
    blocks.3.attn.kvgroup.1            heads that share keys and values (grouped-query attention)
    blocks.3.mlp.neuron.17 (= L3N17)   a neuron: its read vector(s), bias, write vector
    blocks.3.mlp.neurons.64-127        a run of neurons
    blocks.*.mlp                       * matches every layer

Native addresses (always available; exactly the checkpoint's names):

    model.layers.3.mlp.down_proj.weight                 a tensor
    model.layers.3.mlp.down_proj.weight[:, 17]          a slice (python indexing: ints, a:b, :)
    raw:model.layers.3                                  every tensor under a name prefix
    raw:                                                the whole checkpoint as a name tree

    group:<name>                       a user-defined group of addresses

`resolve` turns an address into (tensor name, index) pairs, which is all the rest of the IDE
needs: views, chunks, task scopes, edits and exports are built on that one function.
"""
from __future__ import annotations

import json
import logging
import re

Slice = tuple[str, tuple]

log = logging.getLogger(__name__)

_ALIAS = re.compile(r"^L(\d+)([HN])(\d+)$")


def normalize(addr: str) -> str:
    if not isinstance(addr, str):                     # e.g. a null member in a hand-edited groups.json
        raise ValueError(f"an address must be text, not {type(addr).__name__}")
    addr = addr.strip()
    addr = re.sub(r"^blocks\[(\d+|\*)\]", r"blocks.\1", addr)
    m = _ALIAS.match(addr)
    if m:
        l, kind, i = m.groups()
        return f"blocks.{l}.attn.head.{i}" if kind == "H" else f"blocks.{l}.mlp.neuron.{i}"
    if addr == "ln_final":
        return "final_norm"
    return addr


def parse_index(text: str) -> tuple:
    """'17, 0:8' -> (17, slice(0, 8))"""
    out = []
    for part in text.split(","):
        part = part.strip()
        if part == "" or part == ":":
            out.append(slice(None))
        elif ":" in part:
            bits = [int(b) if b.strip() else None for b in part.split(":")]
            out.append(slice(*bits))
        else:
            out.append(int(part))
    return tuple(out)


def index_str(idx: tuple) -> str:
    if not idx:
        return ""
    parts = []
    for i in idx:
        if isinstance(i, slice):
            parts.append(":" if i == slice(None) else
                         f"{'' if i.start is None else i.start}:{'' if i.stop is None else i.stop}"
                         + (f":{i.step}" if i.step not in (None, 1) else ""))
        else:
            parts.append(str(i))
    return "[" + ", ".join(parts) + "]"


def region_str(region: tuple, shape: tuple) -> str:
    """A normalised region as a minimal index string ('' for the whole tensor)."""
    parts = []
    for r, n in zip(region, shape):
        if isinstance(r, int):
            parts.append(str(r))
        elif len(r) == n and r.step == 1:
            parts.append(":")
        elif len(r) == 0:
            parts.append("0:0")
        else:
            parts.append(f"{r.start}:{r[-1] + 1}" + (f":{r.step}" if r.step != 1 else ""))
    while parts and parts[-1] == ":":
        parts.pop()
    return "[" + ", ".join(parts) + "]" if parts else ""


def slice_str(s: Slice) -> str:
    return s[0] + index_str(s[1])


def slice_shape(shape: tuple[int, ...], idx: tuple) -> tuple[int, ...]:
    out = []
    for d, n in enumerate(shape):
        if d < len(idx):
            i = idx[d]
            if isinstance(i, slice):
                out.append(len(range(*i.indices(n))))
        else:
            out.append(n)
    return tuple(out)


def count(ws, slices: list[Slice]) -> int:
    total = 0
    for name, idx in slices:
        n = 1
        for d in slice_shape(ws.shape(name), idx):
            n *= d
        total += n
    return total


def _part_slices(ws, parts) -> list[Slice]:
    """Parts -> unique (tensor, index) slices (a fused tensor that is wholly covered appears once, whole)."""
    s = ws.struct
    by_name: dict[str, list] = {}
    for p in parts:
        by_name.setdefault(p.name, []).append(p)
    out = []
    for name, ps in by_name.items():
        if len(ps) == len(s.parts.get(name, ps)):
            out.append((name, ()))
        else:
            out += [(name, s.part_index(p)) for p in ps]
    return out


def layer_slices(ws, l: int, what: str = "") -> list[Slice]:
    L = ws.struct.layers[l]
    if what == "attn":
        return _part_slices(ws, L.attn.values())
    if what == "mlp":
        return _part_slices(ws, L.mlp.values())
    if what == "norms":
        return _part_slices(ws, L.norms)
    seen, out = set(), []
    for name, idx in _part_slices(ws, list(L.norms) + list(L.attn.values()) + list(L.mlp.values())) + [(n, ()) for n in L.other]:
        if (name, idx) not in seen:
            seen.add((name, idx))
            out.append((name, idx))
    return out


def resolve(ws, addr: str, _within: tuple = ()) -> list[Slice]:
    s = ws.struct
    addr = normalize(addr)
    if addr.startswith("group:"):
        g = addr[6:]
        if g in _within:
            raise ValueError(f"selection {g!r} contains itself")
        out: list[Slice] = []
        for member in ws.group_members(g):            # an unknown group named directly raises KeyError
            try:
                out += resolve(ws, member, _within + (g,))
            except (KeyError, IndexError, ValueError) as e:
                # A member that no longer resolves (a deleted inner selection, a cycle) is skipped; the
                # selection's page lists it with the reason (selections.expand).
                log.info("group:%s skips %s: %s", g, member, e)
        return out
    if addr.startswith("raw:"):
        pre = addr[4:]
        out = [(n, ()) for n in ws.names() if not pre or n == pre or n.startswith(pre + ".")]
        if not out:
            raise KeyError(f"no tensors under {pre!r}")
        return out
    m = re.match(r"^(.*?)\[(.*)\]$", addr)
    if m and m.group(1) in ws.store:
        name, idx = m.group(1), parse_index(m.group(2))
        if len(idx) > len(ws.shape(name)):
            raise IndexError(f"{addr}: {len(idx)} indices for a tensor of shape {list(ws.shape(name))}")
        from .store import norm_index
        norm_index(idx, ws.shape(name))            # bounds check
        return [(name, idx)]
    if addr in ws.store:
        return [(addr, ())]
    if addr == "model":
        return [(n, ()) for n in ws.names()]
    if ".*" in addr and addr.startswith("blocks."):
        out = []
        for l in range(s.n_layers):
            out += resolve(ws, addr.replace(".*", f".{l}", 1))
        return out
    if s.recognised:
        if addr == "embed":
            return [(s.embed.name, ())]
        if addr == "pos_embed" and s.pos:
            return [(s.pos.name, ())]
        if addr == "unembed":
            return [(s.unembed.name, ())] + ([(s.unembed_bias.name, ())] if s.unembed_bias else [])
        if addr == "final_norm" and s.final_norm:
            return [(p.name, ()) for p in s.final_norm]
        m = re.match(r"^blocks\.(\d+)(?:\.(attn|mlp|norms))?$", addr)
        if m:
            l = int(m.group(1))
            _check(l < s.n_layers, addr)
            out = layer_slices(ws, l, m.group(2) or "")
            if not out:
                raise KeyError(f"{addr}: this layer has no recognised {m.group(2)} weights")
            return out
        m = re.match(r"^blocks\.(\d+)\.attn\.head\.(\d+)$", addr)
        if m:
            l, h = int(m.group(1)), int(m.group(2))
            _check(l < s.n_layers and s.has_heads(l) and h < s.n_heads, addr)
            return s.head_slices(l, h)
        m = re.match(r"^blocks\.(\d+)\.attn\.kvgroup\.(\d+)$", addr)
        if m:
            l, g = int(m.group(1)), int(m.group(2))
            _check(l < s.n_layers and s.has_heads(l) and g < s.n_kv_heads, addr)
            per = s.n_heads // s.n_kv_heads
            out, seen = [], set()
            for h in range(g * per, (g + 1) * per):
                for sl in s.head_slices(l, h):
                    if sl not in seen:
                        seen.add(sl)
                        out.append(sl)
            return out
        m = re.match(r"^blocks\.(\d+)\.mlp\.neuron\.(\d+)$", addr)
        if m:
            l, n = int(m.group(1)), int(m.group(2))
            _check(l < s.n_layers and s.has_neurons(l) and n < s.mlp_width(l), addr)
            return s.neuron_slices(l, n)
        m = re.match(r"^blocks\.(\d+)\.mlp\.neurons\.(\d+)-(\d+)$", addr)
        if m:
            l, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
            _check(l < s.n_layers and s.has_neurons(l) and a <= b < s.mlp_width(l), addr)
            return s.neuron_slices(l, slice(a, b + 1))
    out = [(n, ()) for n in ws.names() if n.startswith(addr + ".")]       # any name prefix
    if not out:
        raise KeyError(f"unknown address {addr!r}")
    return out


def _check(ok: bool, addr: str) -> None:
    if not ok:
        raise IndexError(f"{addr} does not exist in this model")


def to_script(name: str, idx: tuple = ()) -> str:
    """A (tensor, index) pair as a weight-script expression."""
    if re.match(r"^blocks\.\d+\.(attn|mlp|ln\d)\.\w+$", name) or name in ("embed.W_E", "pos_embed.W_pos", "unembed.W_U", "unembed.b_U",
                                                                       "ln_final.w", "ln_final.b"):
        return re.sub(r"^blocks\.(\d+)", r"blocks[\1]", name) + index_str(idx)
    return f"P({json.dumps(name)})" + index_str(idx)
