"""The hierarchy: the same weights at seven levels of organisation, for any checkpoint.

    level 0  model
    level 1  section      embed / pos_embed / blocks / final_norm / unembed / a name prefix
    level 2  block        blocks.L
    level 3  component    attn / mlp / a name prefix inside a layer
    level 4  unit group   a run of neurons, a key-value group of heads, a user-defined group
    level 5  unit         a head or a neuron: several slices of several tensors that act together
    level 6  tensor       a tensor or a slice of one; below that, rows / columns / scalars

Recognised architectures get the semantic tree (blocks, heads, neurons). Every checkpoint,
recognised or not, also has the name tree (`raw:`), which is the checkpoint's own dotted names.

Listing the tree costs nothing but the file headers: parameter counts come from shapes and
"what was edited" from the patch pieces. Tree expansion never reads weights. Callers may explicitly request a norm for a small node.
"""
from __future__ import annotations

from . import addressing as A

NEURON_RUN = 64
NORM_LIMIT = 6_000_000            # weights a tree node may have and still get its norm computed on the spot
LEVELS = ["model", "section", "block", "component", "unit group", "unit", "tensor"]


def node(ws, addr: str, label: str | None = None, kind: str = "tensor", has_children: bool = False, norm: bool = False) -> dict:
    slices = ws.resolve(addr)
    n = A.count(ws, slices)
    sq, dsq, changed = 0.0, 0.0, 0
    measure = norm and n <= NORM_LIMIT and all(ws.store.local(name) for name, _ in slices)    # never download to draw a tree
    if (addr.startswith("group:") or len(slices) > 64) and not measure:
        from .selections import changed as selection_changed       # neurons by the thousand: walk positions, not slices
        if any(ws.pieces(name) for name in {name for name, _ in slices}):
            changed, dsq = selection_changed(ws, addr)
        slices = []
    for name, idx in slices:
        if ws.pieces(name):
            c, d = ws.changed(name, idx)
            changed += c
            dsq += d
        if measure and ws.editable(name):
            sq += float(ws.read(name, idx).double().pow(2).sum())
    return {"addr": addr, "label": label or addr.rsplit(".", 1)[-1], "kind": kind, "level": LEVELS.index(kind),
            "n_params": n, "norm": sq ** 0.5 if measure else None, "dnorm": dsq ** 0.5, "changed": changed,
            "has_children": has_children}


def _tensor(ws, name: str, idx: tuple = (), label: str | None = None) -> dict:
    addr = name + A.index_str(idx)
    n = node(ws, addr, label if label is not None else name.rsplit(".", 1)[-1] + A.index_str(idx), "tensor")
    n["shape"] = list(A.slice_shape(ws.shape(name), idx))
    n["dtype"] = ws.store.dtype(name)
    return n


def _short(name: str, prefix: str) -> str:
    return name[len(prefix) + 1:] if prefix and name.startswith(prefix + ".") else name


def children(ws, addr: str) -> list[dict]:
    s = ws.struct
    addr = A.normalize(addr)
    parts = addr.split(".")
    if addr == "model":
        if not s.recognised:
            return _raw_children(ws, "")
        out = [node(ws, "embed", "embed", "section", True)]
        if s.pos:
            out.append(node(ws, "pos_embed", "pos_embed", "section", True))
        out += [node(ws, f"blocks.{l}", f"block {l}", "block", True) for l in range(s.n_layers)]
        if s.final_norm:
            out.append(node(ws, "final_norm", "final norm", "section", True))
        if not s.tied or s.unembed_bias:
            out.append(node(ws, "unembed", "unembed", "section", True))
        if ws.kind != "wsp":
            out.append({"addr": "raw:", "label": "all tensors by name", "kind": "section", "level": 1,
                        "n_params": ws.n_params(), "norm": None, "dnorm": 0.0, "changed": 0, "has_children": True})
        return out
    if addr.startswith("raw:"):
        return _raw_children(ws, addr[4:])
    if addr.startswith("group:"):
        from .selections import SHOW_MEMBERS                        # listing only: the group itself is never cut
        out = []
        for m in ws.group_members(addr[6:])[:SHOW_MEMBERS]:
            try:
                out.append(node(ws, m, m, kind_of(ws, m), kind_of(ws, m) != "tensor"))
            except (KeyError, IndexError, ValueError):     # no longer resolves: the selection's page says why
                continue
        return out
    if addr in ("embed", "pos_embed", "unembed", "final_norm"):
        return [_tensor(ws, name, idx, name if ws.kind != "wsp" else None) for name, idx in ws.resolve(addr)]
    if parts[0] == "blocks" and len(parts) == 2 and s.recognised:
        l = int(parts[1])
        L = s.layers[l]
        out = []
        if L.attn:
            out.append(node(ws, f"blocks.{l}.attn", f"attn · {s.n_heads} heads" if s.has_heads(l) else "attn", "component", True))
        if L.mlp:
            out.append(node(ws, f"blocks.{l}.mlp", f"mlp · {s.mlp_width(l)} neurons" if s.has_neurons(l) else "mlp", "component", True))
        for p in L.norms:
            out.append(_tensor(ws, p.name, (), _short(p.name, L.prefix)))
        for name in L.other:
            out.append(_tensor(ws, name, (), _short(name, L.prefix)))
        return out
    if len(parts) == 3 and parts[2] == "attn" and s.recognised:
        l = int(parts[1])
        L = s.layers[l]
        out = []
        if s.has_heads(l):
            o = L.attn["W_O"]
            for h in range(s.n_heads):
                n = node(ws, f"{addr}.head.{h}", f"head {h}", "unit", True)
                n["thumb"] = A.slice_str((o.name, s.part_index(o, head=h)))       # what the head writes, for a thumbnail
                out.append(n)
        for name, idx in A.layer_slices(ws, l, "attn"):
            out.append(_tensor(ws, name, idx, _short(name, L.prefix) + A.index_str(idx)))
        return out
    if len(parts) == 5 and parts[3] in ("head", "kvgroup", "neuron"):
        l = int(parts[1])
        L = s.layers[l]
        return [_tensor(ws, name, idx, _short(name, L.prefix) + A.index_str(idx)) for name, idx in ws.resolve(addr)]
    if len(parts) == 3 and parts[2] == "mlp" and s.recognised:
        l = int(parts[1])
        L = s.layers[l]
        out = [_tensor(ws, name, idx, _short(name, L.prefix) + A.index_str(idx)) for name, idx in A.layer_slices(ws, l, "mlp")]
        if not s.has_neurons(l):
            return out
        w = s.mlp_width(l)
        if w <= NEURON_RUN:
            return out + [node(ws, f"{addr}.neuron.{n}", f"neuron {n}", "unit", True) for n in range(w)]
        run = NEURON_RUN if w <= 64 * NEURON_RUN else -(-w // 64)
        for a in range(0, w, run):
            b = min(a + run, w) - 1
            out.append(node(ws, f"{addr}.neurons.{a}-{b}", f"neurons {a}–{b}", "unit group", True))
        return out
    if len(parts) == 5 and parts[3] == "neurons":
        a, b = (int(x) for x in parts[4].split("-"))
        base = ".".join(parts[:3])
        if b - a + 1 > NEURON_RUN:
            step = max(NEURON_RUN, -(-(b - a + 1) // 64))
            return [node(ws, f"{base}.neurons.{x}-{min(x + step, b + 1) - 1}", f"neurons {x}–{min(x + step, b + 1) - 1}", "unit group", True)
                    for x in range(a, b + 1, step)]
        return [node(ws, f"{base}.neuron.{n}", f"neuron {n}", "unit", True, norm=False) for n in range(a, b + 1)]
    if addr in ws.store or "[" in addr:
        return []
    return _raw_children(ws, addr)


def _raw_children(ws, prefix: str) -> list[dict]:
    """One level of the checkpoint's dotted names under a prefix."""
    names = [n for n in ws.names() if not prefix or n.startswith(prefix + ".")]
    cut = len(prefix) + 1 if prefix else 0
    seen: dict[str, list[str]] = {}
    for n in names:
        seen.setdefault(n[cut:].split(".", 1)[0], []).append(n)
    out = []
    for comp, members in seen.items():
        full = f"{prefix}.{comp}" if prefix else comp
        if len(members) == 1 and members[0] == full:
            out.append(_tensor(ws, full, (), comp))
        else:
            k = "block" if comp.isdigit() else "section" if not prefix else "component"
            out.append(node(ws, "raw:" + full, comp, k, True))
    return out


def kind_of(ws, addr: str) -> str:
    addr = A.normalize(addr)
    if addr == "model":
        return "model"
    if addr.startswith("group:"):
        return "unit group"
    base = addr.split("[", 1)[0]
    if base in ws.store:
        return "tensor"
    parts = addr.split(".")
    if parts[0] == "blocks" and len(parts) >= 4 and parts[3] in ("head", "neuron"):
        return "unit"
    if parts[0] == "blocks" and len(parts) >= 4 and parts[3] in ("neurons", "kvgroup"):
        return "unit group"
    if parts[0] == "blocks" and len(parts) == 3:
        return "component"
    if parts[0] == "blocks" and len(parts) == 2:
        return "block"
    if addr.startswith("raw:"):
        p = addr[4:].split(".")
        return "block" if p[-1].isdigit() else "section" if len(p) <= 1 else "component"
    return "section"


def _neuron_groups(w: int, n: int) -> list[tuple[int, int]]:
    """The runs of neurons the tree shows above neuron n of a layer w neurons wide, outermost first."""
    out: list[tuple[int, int]] = []
    if w <= NEURON_RUN:
        return out
    run = NEURON_RUN if w <= 64 * NEURON_RUN else -(-w // 64)
    a = n // run * run
    b = min(a + run, w) - 1
    out.append((a, b))
    while b - a + 1 > NEURON_RUN:
        step = max(NEURON_RUN, -(-(b - a + 1) // 64))
        a = a + (n - a) // step * step
        b = min(a + step - 1, b)
        out.append((a, b))
    return out


def _home(ws, name: str):
    """Where a tensor sits: (layer, "attn" | "mlp" | None), a section address, or None."""
    s = ws.struct
    if not s.recognised:
        return None
    for l, L in enumerate(s.layers):
        if any(p.name == name for p in L.attn.values()):
            return (l, "attn")
        if any(p.name == name for p in L.mlp.values()):
            return (l, "mlp")
        if any(p.name == name for p in L.norms) or name in L.other:
            return (l, None)
    for sec, parts in (("embed", [s.embed]), ("pos_embed", [s.pos]), ("unembed", [s.unembed, s.unembed_bias]), ("final_norm", s.final_norm)):
        if any(p is not None and p.name == name for p in parts):
            return sec
    return None


def crumbs(ws, addr: str) -> list[dict]:
    """The path from the model down to an address, as the tree shows it: [{"addr", "label"}], last = addr itself."""
    s = ws.struct
    addr = A.normalize(addr)
    out = [{"addr": "model", "label": "model"}]
    if addr == "model":
        return out
    if addr.startswith("group:"):
        return out + [{"addr": addr, "label": f"selection {addr[6:]}"}]
    if addr.startswith("raw:"):
        out.append({"addr": "raw:", "label": "all tensors by name"})
        parts = [p for p in addr[4:].split(".") if p]
        out += [{"addr": "raw:" + ".".join(parts[:i + 1]), "label": parts[i]} for i in range(len(parts))]
        return out
    parts = addr.split(".")
    if s.recognised and parts[0] == "blocks" and len(parts) >= 2 and parts[1].isdigit():
        l = int(parts[1])
        out.append({"addr": f"blocks.{l}", "label": f"block {l}"})
        if len(parts) >= 3:
            out.append({"addr": f"blocks.{l}.{parts[2]}", "label": parts[2]})
        if len(parts) == 5 and parts[2] == "mlp" and parts[3] in ("neuron", "neurons") and s.has_neurons(l):
            a, b = (int(x) for x in parts[4].split("-")) if parts[3] == "neurons" else (int(parts[4]),) * 2
            for x, y in _neuron_groups(s.mlp_width(l), a):
                if x <= a and b <= y and (x, y) != (a, b):
                    out.append({"addr": f"blocks.{l}.mlp.neurons.{x}-{y}", "label": f"neurons {x}–{y}"})
            out.append({"addr": addr, "label": f"neuron {a}" if parts[3] == "neuron" else f"neurons {a}–{b}"})
        elif len(parts) == 5 and parts[2] == "attn" and parts[3] in ("head", "kvgroup"):
            out.append({"addr": addr, "label": f"head {parts[4]}" if parts[3] == "head" else f"kv group {parts[4]}"})
        elif len(parts) > 3:
            out.append({"addr": addr, "label": ".".join(parts[3:])})
        return out
    base = addr.split("[", 1)[0]
    if base in ws.store:
        home = _home(ws, base)
        label = base
        if isinstance(home, tuple):
            l, comp = home
            out.append({"addr": f"blocks.{l}", "label": f"block {l}"})
            if comp:
                out.append({"addr": f"blocks.{l}.{comp}", "label": comp})
            label = _short(base, s.layers[l].prefix)
        elif isinstance(home, str):
            out.append({"addr": home, "label": home.replace("_", " ")})
        out.append({"addr": base, "label": label})
        if addr != base:
            out.append({"addr": addr, "label": addr[len(base):]})
        return out
    return out + [{"addr": addr, "label": addr}]


def jumps(ws, addr: str) -> list[dict]:
    """Shortcuts across the levels: from a slice to the neuron or head it is, from a unit to its tensors."""
    s = ws.struct
    addr = A.normalize(addr)
    if not s.recognised or addr.startswith(("group:", "raw:")):
        return []
    kind = kind_of(ws, addr)
    if kind in ("unit", "unit group") and addr.startswith("blocks."):
        l = int(addr.split(".")[1])
        prefix = s.layers[l].prefix
        return [{"addr": A.slice_str(x), "label": _short(x[0], prefix) + A.index_str(x[1]), "kind": "tensor"} for x in ws.resolve(addr)]
    if kind != "tensor":
        return []
    name, idx = ws.resolve(addr)[0]
    home = _home(ws, name)
    if not isinstance(home, tuple) or home[1] is None:
        return []
    from .store import norm_index
    from .selections import neuron_axis
    l, comp = home
    L = s.layers[l]
    shape = ws.shape(name)
    region = norm_index(idx, shape)
    out = []
    if comp == "mlp" and s.has_neurons(l):
        for p in L.mlp.values():
            if p.name != name or "neuron" not in p.roles:
                continue
            axis, positions, template = neuron_axis(s, p)
            if any(region[d] != template[d] for d in range(len(shape)) if d != axis):
                continue
            r = region[axis]
            ids = [r] if isinstance(r, int) else list(r)
            if not ids or any(i not in positions for i in ids):
                continue
            a, b = positions.index(ids[0]), positions.index(ids[-1])
            if len(ids) == 1:
                out.append({"addr": f"blocks.{l}.mlp.neuron.{a}", "label": f"L{l}N{a}", "kind": "unit"})
            elif b - a + 1 == len(ids) and b > a:
                out.append({"addr": f"blocks.{l}.mlp.neurons.{a}-{b}", "label": f"neurons {a}–{b} of layer {l}", "kind": "unit group"})
    if comp == "attn" and s.has_heads(l):
        for p in L.attn.values():
            if p.name != name or not ({"head", "headdh"} & set(p.roles)):
                continue
            hs = [h for h in range(s.n_heads) if norm_index(s.part_index(p, head=h), shape) == region]
            if len(hs) == 1:
                out.append({"addr": f"blocks.{l}.attn.head.{hs[0]}", "label": f"L{l}H{hs[0]}", "kind": "unit"})
            elif hs and s.n_kv_heads and s.n_kv_heads != s.n_heads:
                g = hs[0] * s.n_kv_heads // s.n_heads
                out.append({"addr": f"blocks.{l}.attn.kvgroup.{g}", "label": f"kv group {g} of layer {l} (heads {hs[0]}–{hs[-1]})", "kind": "unit group"})
    seen, uniq = set(), []
    for j in out:
        if j["addr"] not in seen:
            seen.add(j["addr"])
            uniq.append(j)
    return uniq


def partition(ws, addr: str) -> list[str]:
    """Children of an address that together cover exactly the same weights, with no overlap (for chunk planning)."""
    s = ws.struct
    addr = A.normalize(addr)
    parts = addr.split(".")
    if addr == "model":
        if not s.recognised:
            return [c["addr"] for c in _raw_children(ws, "")]
        out = ["embed"] + (["pos_embed"] if s.pos else []) + [f"blocks.{l}" for l in range(s.n_layers)]
        out += (["final_norm"] if s.final_norm else []) + (["unembed"] if not s.tied or s.unembed_bias else [])
        if s.tied and s.unembed_bias:
            out[-1] = s.unembed_bias.name
        return out + s.other
    if addr.startswith("group:"):
        return list(ws.group_members(addr[6:]))
    if addr.startswith("raw:"):
        return [c["addr"] for c in _raw_children(ws, addr[4:])]
    if s.recognised and parts[0] == "blocks":
        l = int(parts[1]) if parts[1].isdigit() else -1
        if len(parts) == 2:
            L = s.layers[l]
            return ([f"{addr}.attn"] if L.attn else []) + ([f"{addr}.mlp"] if L.mlp else []) + [p.name for p in L.norms] + L.other
        L = s.layers[l]
        if len(parts) == 3 and parts[2] == "attn":
            if not s.has_heads(l):
                return [A.slice_str(x) for x in A.layer_slices(ws, l, "attn")]
            grouped = s.n_kv_heads and s.n_kv_heads != s.n_heads
            units = [f"{addr}.kvgroup.{g}" for g in range(s.n_kv_heads)] if grouped else [f"{addr}.head.{h}" for h in range(s.n_heads)]
            rest = [p.name for r, p in L.attn.items() if r == "b_O"]
            return units + rest
        if len(parts) == 3 and parts[2] == "mlp":
            if not s.has_neurons(l):
                return [A.slice_str(x) for x in A.layer_slices(ws, l, "mlp")]
            w = s.mlp_width(l)
            run = 32 if w <= 4096 else 128
            return [f"{addr}.neurons.{a}-{min(a + run, w) - 1}" for a in range(0, w, run)] + [p.name for r, p in L.mlp.items() if r == "b_out"]
        if len(parts) == 5 and parts[3] == "neurons":
            a, b = (int(x) for x in parts[4].split("-"))
            if b > a:
                mid = (a + b) // 2
                base = ".".join(parts[:3])
                return [f"{base}.neurons.{a}-{mid}", f"{base}.neurons.{mid + 1}-{b}"]
    slices = ws.resolve(addr)
    if len(slices) > 1:
        return [A.slice_str(x) for x in slices]
    return []


def outline(ws, depth: int = 2) -> str:
    """The hierarchy as indented text (for the programmer model's overview)."""
    lines: list[str] = []

    def walk(n: dict, d: int):
        mark = f"  Δ{n['dnorm']:.3g} ({n['changed']} changed)" if n["changed"] else ""
        shape = f" {n['shape']}" if "shape" in n else ""
        norm = f"  |w|={n['norm']:.4g}" if n.get("norm") is not None else ""
        lines.append(f"{'  ' * d}{n['addr']}{shape}  params={n['n_params']}{norm}{mark}")
        if d < depth and n["has_children"] and len(lines) < 400:
            for c in children(ws, n["addr"]):
                if c["kind"] in ("unit", "unit group") or (c["addr"] == "raw:"):
                    continue
                walk(c, d + 1)

    root = node(ws, "model", "model", "model", True, norm=False)
    walk(root, 0)
    return "\n".join(lines)
