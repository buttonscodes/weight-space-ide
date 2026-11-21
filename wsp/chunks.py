"""Weights as text for the programmer model: formats, token budgets, chunk plans.

Token costs were measured, not guessed: 1,200 Gaussian weights were sent through the Claude
Code CLI in each format and the input-token difference against an empty prompt recorded
(November 2026, the session's default model).

    format      example                 tokens / weight   notes
    q   (q999)  `-412 87 0 913`              2.5          integers in [-999, 999] x one scale per tensor;
                                                          1, 2 and 3 digits cost the same, so use 3 (~10 bits)
    g3          `-0.0412 0.0087`             5.3          3 significant digits
    g2          `-0.041 0.0087`              4.6
    sparse      `17:-0.041 203:0.0087`       6.8 / entry  bare; ~11 as rendered here with token labels
    (4-digit integers cost 3.2, scientific notation 7.5; a minus sign costs about one token.)

Whole rendered chunks were then checked the same way (actual tokens vs the planner's estimate):
a head in q 12.4k vs 14.2k, 32 neurons in g3 22.1k vs 25.8k, a 16-head attention layer in q
62.5k vs 66.6k.  The planner is 7-17% high on dense formats by design (SAFETY), so a plan
that says it fits, fits.

So a 1M-token context holds roughly 300k weights densely at ~10-bit precision with a quarter
of the window left for instructions, tool traffic and thinking; anything larger has to be
read in parts or in a lossy view (SVD factors, top-k entries, per-row statistics).

A *chunk* is a set of addresses that belong together (a head, a run of neurons, a slab of
rows) rendered as one document: a header that says where the weights sit in the model and
what they do, then each tensor.  `plan` walks the hierarchy top-down and cuts at the highest
level that fits the per-chunk budget, so chunks are whole blocks when they can be, whole
components when blocks are too big, whole units when components are too big, and row slabs
only for single tensors that are too big on their own.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import addressing as A
from . import analysis as an
from . import hierarchy
from . import workspace as W
from .tokenizer import show
from .workspace import Ref

TOKENS_PER = {"q": 2.55, "g3": 5.3, "g2": 4.65, "sparse": 11.5}     # sparse: as rendered, with column labels
ROW_OVERHEAD = 5            # row label + newline
TENSOR_OVERHEAD = 90        # heading, shape, role line, scale
CHUNK_OVERHEAD = 450        # where-this-is header
SAFETY = 1.12
Q_LEVELS = 999
FORMATS = ("q", "g3", "g2", "sparse", "svd", "stats")
LADDER = ("g3", "q", "svd", "stats")      # auto: the most faithful format that fits


to_script = A.to_script


def est_tensor(shape: tuple[int, ...], fmt: str, k: int = 8, rank: int = 4) -> int:
    n = int(np.prod(shape)) if shape else 1
    rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    cols = shape[-1] if shape else 1
    if fmt in ("q", "g3", "g2"):
        t = n * TOKENS_PER[fmt] + rows * ROW_OVERHEAD
    elif fmt == "sparse":
        t = rows * (min(k, cols) * TOKENS_PER["sparse"] + ROW_OVERHEAD)
    elif fmt == "svd":
        r = min(rank, rows, cols)
        if rows <= 1:
            t = n * TOKENS_PER["g3"]
        elif rows + cols > 1024:               # factors too long to print: singular values and vocabulary readings only
            t = r * 420 + 200
        else:
            t = r * (rows + cols) * 5.6 + r * 260
    else:
        t = 60
    return int(math.ceil((t + TENSOR_OVERHEAD) * SAFETY))


def est_slices(ws, slices: list, fmt: str) -> int:
    return sum(est_tensor(A.slice_shape(ws.shape(n), i), fmt) for n, i in slices)


# ------------------------------------------------------------------------------ rendering
def _fmt_rows(m: np.ndarray, fmt: str, labels: list[str]) -> list[str]:
    if fmt == "q":
        return [f"{lab}: " + " ".join(map(str, row)) for lab, row in zip(labels, m.tolist())]
    spec = "%.3g" if fmt == "g3" else "%.2g"
    return [f"{lab}: " + " ".join(spec % x for x in row) for lab, row in zip(labels, m)]


def _row_labels(ws, role: str, start: int, n: int, prefix: str = "r") -> list[str]:
    if role == "token" and n <= 4096:
        return [f"{prefix}{i} {show(ws.tok.token_str(i))!r}" for i in range(start, start + n)]
    return [f"{prefix}{i}" for i in range(start, start + n)]


def render_tensor(ws, name: str, idx: tuple = (), fmt: str = "q", variant: str = "current", k: int = 8,
                  rank: int = 4, rows: tuple[int, int] | None = None) -> str:
    """One tensor (or slice) as text, with what is needed to write it back."""
    from .store import norm_index, region_shape, region_size
    shape_full = ws.shape(name)
    region = norm_index(idx, shape_full)
    kshape = region_shape(region)
    shape = list(kshape)
    R = int(np.prod(kshape[:-1])) if len(kshape) > 1 else 1
    C = kshape[-1] if kshape else 1
    rrole, crole = an.view_roles(ws, name, idx)
    roff, coff = an.axis_offset(name, idx, shape_full)
    r0, r1 = rows or (0, R)
    r1 = min(r1, R)
    part = f" rows {r0}:{r1} of {R}" if (r0, r1) != (0, R) else ""
    head = [f"### {to_script(name, idx)}   shape {shape}{part}   ({variant})",
            f"axes: rows = {rrole}, columns = {crole}.  {ws.struct.doc(name)}"]
    if len(kshape) >= 2 and ((r1 - r0) * C if len(kshape) == 2 else region_size(region)) > W.STREAM_ELEMS:
        return _render_streamed(ws, name, idx, region, kshape, fmt, variant, k, rank, rows, head)   # above one store read
    if len(kshape) == 2 and (r0, r1) != (0, R):          # a slab of rows: read only those rows
        kept = [d for d, x in enumerate(region) if isinstance(x, range)]
        sub_region = region[:kept[0]] + (region[kept[0]][r0:r1],) + region[kept[0] + 1:]
        m = None
        sub = ws.read(name, sub_region, variant)
        t = sub
    else:
        t = an.tensor_of(ws, name, idx, variant)
        m = an.as_matrix(t)
        sub = m[r0:r1]
    if float(sub.abs().max() if sub.numel() else 0) == 0:
        return "\n".join(head + ["all zero"])
    labels = _row_labels(ws, rrole, roff + r0, r1 - r0) if t.dim() <= 2 else [f"r{i}" for i in range(r0, r1)]
    if t.dim() == 3:                   # [head, a, b] flattened to rows: say which head each row belongs to
        per = t.shape[1]
        labels = [f"h{i // per}.r{i % per}" for i in range(r0, r1)]
    if fmt == "q":
        scale = float(sub.abs().max()) / Q_LEVELS
        q = torch.round(sub / scale).to(torch.int64).numpy()
        head.append(f"format q: weight = integer * {scale:.6g}   (integers in [-{Q_LEVELS}, {Q_LEVELS}])")
        return "\n".join(head + _fmt_rows(q, "q", labels))
    if fmt in ("g3", "g2"):
        head.append(f"format {fmt}: decimal values, {fmt[1]} significant digits")
        return "\n".join(head + _fmt_rows(sub.numpy(), fmt, labels))
    if fmt == "sparse":
        kk = min(k, C)
        head.append(f"format sparse: per row, the {kk} largest |weights| as column:value; row norm in brackets")
        return "\n".join(head + _sparse_lines(sub, labels, kk, _col_label(ws, crole, coff)))
    if fmt == "svd":
        if min(R, C) == 1 or m is None:
            return render_tensor(ws, name, idx, "g3", variant, rows=rows)
        sv = an.svd_view(ws, name + A.index_str(idx), k=rank, variant=variant, top=6)
        head.append(_svd_head(sv, rank))
        u, s, vh = torch.linalg.svd(m.double(), full_matrices=False) if min(R, C) <= 2048 and R * C <= 8_000_000 else (None, None, None)
        return "\n".join(head + _svd_lines(sv, rrole, crole, R, C, u, vh))
    st = an.stats(sub)
    head.append(f"format stats: mean {st['mean']:.4g} std {st['std']:.4g} min {st['min']:.4g} max {st['max']:.4g} "
                f"norm {st['norm']:.4g} zeros {st['zeros']}/{st['n']}")
    if R > 1 and R <= 4096:
        head.append("row norms: " + " ".join("%.3g" % x for x in sub.norm(dim=1).tolist()))
    return "\n".join(head)


def _col_label(ws, crole: str, coff: int):
    return (lambda j: f"{j + coff}") if crole != "token" else (lambda j: f"{j + coff}{show(ws.tok.token_str(j + coff))!r}")


def _sparse_lines(sub: torch.Tensor, labels: list[str], kk: int, clab) -> list[str]:
    top = sub.abs().topk(kk, dim=1).indices.sort(dim=1).values
    lines = []
    for i, lab in enumerate(labels):
        ent = " ".join(f"{clab(int(j))}:{float(sub[i, j]):.3g}" for j in top[i])
        lines.append(f"{lab} [{float(sub[i].norm()):.3g}]: {ent}")
    return lines


def _svd_head(sv: dict, rank: int) -> str:
    rk = f"of rank {sv['rank']} (effective rank {sv['effective_rank']:.1f})" if sv["rank"] is not None else "(randomised estimate)"
    return (f"format svd: W ≈ sum_i s_i * u_i v_i^T, top {min(rank, len(sv['singular_values']))} {rk}; singular values: "
            + " ".join(f"{s:.3g}" for s in sv["singular_values"][:16]))


def _svd_lines(sv: dict, rrole: str, crole: str, R: int, C: int, u, vh) -> list[str]:
    lines = []
    for c in sv["components"]:
        i = c["i"]
        lines.append(f"component {i}: s = {c['s']:.4g}  ({100 * (sv['energy'][i] - (sv['energy'][i - 1] if i else 0)):.1f}% of energy)")
        for side, role in (("left", rrole), ("right", crole)):
            d = c.get(side)
            if d and d.get("kind") == "resid":
                lines.append(f"  {side} ({role}): promotes {_toks(d['promotes']['top'])} | suppresses {_toks(d['promotes']['bottom'])}"
                             f" | embeddings along it {_toks(d['embeddings']['top'])}")
            elif d:
                lines.append(f"  {side} ({role}): + {_toks(d['top'])} | - {_toks(d['bottom'])}")
        if u is not None and R + C <= 1024:
            lines.append("  u: " + " ".join("%.3g" % x for x in u[:, i].tolist()))
            lines.append("  v: " + " ".join("%.3g" % x for x in vh[i].tolist()))
    return lines


def _row_slabs(ws, name: str, region: tuple, variant: str, R: int, r0: int, r1: int):
    """Rows r0:r1 of a view in its matrix form (as analysis.as_matrix flattens it), as (first row, [rows, C] tensor),
    read through Workspace.slabs in reads of at most STREAM_ELEMS values."""
    kept = [d for d, x in enumerate(region) if isinstance(x, range)]
    ax, first = kept[0], region[kept[0]]
    unit = R // len(first)                       # matrix rows per index of the first kept axis
    a0, a1 = r0 // unit, -(-r1 // unit)
    for pos, t in ws.slabs(name, region[:ax] + (first[a0:a1],) + region[ax + 1:], variant, max_elems=W.STREAM_ELEMS):
        m = an.as_matrix(t)
        g0 = (a0 + pos.start) * unit
        lo, hi = max(r0, g0), min(r1, g0 + m.shape[0])
        if hi > lo:
            yield lo, m[lo - g0:hi - g0]


STATS_PART = 1 << 20


def slab_stats(slabs, row_norms: bool = False) -> dict:
    """n, mean, std, min, max, norm, zeros and the largest |value| over [rows, C] slabs, without holding them together.

    Sums are taken in float64 slab by slab (mean and variance combined as Chan et al. 1979), so mean, std and norm can
    differ from analysis.stats of the whole (float32 reductions) in the last float32 digits; n, min, max, zeros and
    the row norms are the same. min, max and abs_max are NaN when any value is (as torch's are on one read). The
    float64 copies are of at most STATS_PART values at a time (8 MB each), not of the whole slab (a 16M-value slab
    would need several hundred MB of them)."""
    n, mean, m2, sq, lo, hi, amax, zeros, norms = 0, 0.0, 0.0, 0.0, None, None, None, 0, []
    for _, p in slabs:
        if not p.numel():
            continue
        for t in p.flatten().split(STATS_PART):
            t = t.double()
            nb, mb = t.numel(), float(t.mean())
            d, tot = mb - mean, n + nb
            mean += d * nb / tot
            m2 += float((t - mb).pow(2).sum()) + d * d * n * nb / tot
            n = tot
            sq += float(t.pow(2).sum())
        lo, hi, amax = (x if y is None else f(y, x) for x, y, f in (
            (p.min(), lo, torch.minimum), (p.max(), hi, torch.maximum), (p.abs().max(), amax, torch.maximum)))
        zeros += int((p == 0).sum())
        if row_norms:
            norms += p.norm(dim=1).tolist()
    if not n:
        return {"n": 0}
    return {"n": n, "mean": mean, "std": (m2 / (n - 1)) ** 0.5 if n > 1 else 0.0, "min": float(lo), "max": float(hi),
            "abs_max": float(amax), "norm": sq ** 0.5, "zeros": zeros, "row_norms": norms}


def _abs_max(slabs) -> float:
    """The largest |value| over slabs, NaN if any value is NaN (torch's max of one read), 0.0 for no values."""
    top = None
    for _, p in slabs:
        if p.numel():
            m = p.abs().max()
            top = m if top is None else torch.maximum(top, m)
    return 0.0 if top is None else float(top)


def _render_streamed(ws, name: str, idx: tuple, region: tuple, kshape: tuple, fmt: str, variant: str, k: int,
                     rank: int, rows, head: list[str]) -> str:
    """render_tensor for a view above one store read (STREAM_ELEMS values): the rows are read in slabs and never held
    together (a quantized store refuses the whole read, and the vocabulary matrices must never be assembled).

    As render_tensor, rows r0:r1 that are all zero give "all zero" whatever the format (found at the first non-zero
    slab, so usually after one read). q, g3, g2 and sparse give the same text as a whole read (q takes its scale from
    a first pass). stats is summed slab by slab (`slab_stats`), and like analysis.stats refuses a tensor holding inf
    or NaN. svd is analysis.svd_view's own, which streams above DENSE_ELEMS; the factors u and v are printed only for
    a matrix of rows + columns <= 1024, which is then read whole (in slabs)."""
    R, C = int(np.prod(kshape[:-1])), kshape[-1]
    r0, r1 = rows or (0, R)
    r1 = min(r1, R)
    rrole, crole = an.view_roles(ws, name, idx)
    roff, coff = an.axis_offset(name, idx, ws.shape(name))
    slabs = lambda: _row_slabs(ws, name, region, variant, R, r0, r1)      # noqa: E731
    if not any(bool(p.ne(0).any()) for _, p in slabs()):                 # a NaN is not zero, as in render_tensor
        return "\n".join(head + ["all zero"])
    if fmt == "svd":
        if min(R, C) == 1 or (len(kshape) == 2 and (r0, r1) != (0, R)):
            return render_tensor(ws, name, idx, "g3", variant, rows=rows)
        sv = an.svd_view(ws, name + A.index_str(idx), k=rank, variant=variant, top=6)
        u = vh = None
        if R + C <= 1024:
            u, _, vh = torch.linalg.svd(an.as_matrix(Ref(ws, name, region).whole(variant)).double(), full_matrices=False)
        return "\n".join(head + [_svd_head(sv, rank)] + _svd_lines(sv, rrole, crole, R, C, u, vh))
    if fmt not in ("q", "g3", "g2", "sparse"):
        st = slab_stats(slabs(), row_norms=1 < R <= 4096)
        lim = st["abs_max"] or 1.0                     # analysis.stats' histogram range: refused when not finite
        torch.histc(torch.zeros(1), bins=41, min=-lim, max=lim)
        head.append(f"format stats: mean {st['mean']:.4g} std {st['std']:.4g} min {st['min']:.4g} max {st['max']:.4g} "
                    f"norm {st['norm']:.4g} zeros {st['zeros']}/{st['n']}")
        if 1 < R <= 4096:
            head.append("row norms: " + " ".join("%.3g" % x for x in st["row_norms"]))
        return "\n".join(head)
    labels = _row_labels(ws, rrole, roff + r0, r1 - r0) if len(kshape) <= 2 else [f"r{i}" for i in range(r0, r1)]
    if len(kshape) == 3:
        per = kshape[1]
        labels = [f"h{i // per}.r{i % per}" for i in range(r0, r1)]
    lines = []
    if fmt == "q":
        scale = _abs_max(slabs()) / Q_LEVELS
        head.append(f"format q: weight = integer * {scale:.6g}   (integers in [-{Q_LEVELS}, {Q_LEVELS}])")
    elif fmt == "sparse":
        kk = min(k, C)
        head.append(f"format sparse: per row, the {kk} largest |weights| as column:value; row norm in brackets")
        clab = _col_label(ws, crole, coff)
    else:
        head.append(f"format {fmt}: decimal values, {fmt[1]} significant digits")
    for a, p in slabs():
        lab = labels[a - r0:a - r0 + p.shape[0]]
        if fmt == "q":
            lines += _fmt_rows(torch.round(p / scale).to(torch.int64).numpy(), "q", lab)
        else:
            lines += _sparse_lines(p, lab, kk, clab) if fmt == "sparse" else _fmt_rows(p.numpy(), fmt, lab)
    return "\n".join(head + lines)


def _toks(pairs: list, n: int = 6) -> str:
    return " ".join(f"{p[0]!r}({p[1]:.2g})" for p in pairs[:n])


def model_line(ws) -> str:
    return f"model {ws.name}: " + ws.struct.describe().splitlines()[0]


def where_header(ws, addr: str, context: str = "full") -> str:
    """Where a chunk sits in the model and what it does: the context the weights come from."""
    s = ws.struct
    addr = A.normalize(addr)
    lines = [f"address: {addr}"]
    if context == "none":
        return "\n".join(lines)
    parts = addr.split(".")
    need = ws.sem.vocab_bytes() if s.recognised else 0
    if need and len(parts) >= 5 and parts[0] == "blocks" and parts[3] in ("head", "neuron"):
        lines.append(f"(vocabulary readings omitted: they need the embedding / unembedding matrices, {need / 1e9:.1f} GB still on the Hub; "
                     f"`inspect` with fetch=true downloads them)")
        return "\n".join(lines)
    try:
        if s.recognised and len(parts) >= 5 and parts[0] == "blocks" and parts[3] == "head":
            l, h = int(parts[1]), int(parts[4].split("[")[0])
            c = an.head_card(ws, l, h, k=5)
            lines += [f"head {l}.{h}: reads the residual stream after {'the embeddings' if l == 0 else f'layer {l - 1}'} "
                      f"through its query (destination), key and value (source) matrices; writes (pattern @ value) through "
                      f"its output matrix." + (f" Keys and values are shared with the other heads of group {c['kv_group']}."
                                               if c.get("kv_group") is not None else ""),
                      f"  norms Q {c['norms']['Q']:.3g} K {c['norms']['K']:.3g} V {c['norms']['V']:.3g} O {c['norms']['O']:.3g}"]
            if "pos_offset_preference" in c:
                pref = c["pos_offset_preference"]
                best = sorted(range(len(pref)), key=lambda i: -pref[i])[:3]
                lines.append("  positional preference (score for attending `offset` tokens back, relative to an average earlier token): "
                             + ", ".join(f"-{i}: {pref[i]:+.2f}" for i in best))
            lines += [f"  OV copying: own token top-1 for {100 * c['copying']['top1']:.0f}% and top-10 for "
                      f"{100 * c['copying']['top10']:.0f}% of {c['copying']['sample']} sampled tokens",
                      "  OV examples (source token -> promoted): " + "; ".join(
                          f"{e['src']!r} -> " + " ".join(repr(o[0]) for o in e["out"][:3]) for e in c["ov_examples"][:4])]
            if context == "full":
                w = an.wiring(ws, 100000, layers=None if l in an.default_window(ws) else an.default_window(ws, l))
                nid = f"L{l}H{h}"
                ins = [e for e in w["edges"] if e["dst"] == nid][:5]
                outs = [e for e in w["edges"] if e["src"] == nid][:5]
                if ins:
                    lines.append("  strongest inputs (composition, 1.0 = chance): " +
                                 ", ".join(f"{e['src']}->{e['kind']} {e['score']:.2f}" for e in ins))
                if outs:
                    lines.append("  strongest outputs: " + ", ".join(f"->{e['dst']}.{e['kind']} {e['score']:.2f}" for e in outs))
        elif s.recognised and len(parts) >= 5 and parts[0] == "blocks" and parts[3] == "neuron":
            l, n = int(parts[1]), int(parts[4].split("[")[0])
            c = an.neuron_card(ws, l, n, k=6)
            rule = ("activation = act(x . gate) * (x . up)" if c["gated"] else "activation = act(x . read + bias)")
            lines += [f"neuron {l}.{n}: {rule}; adds activation * (its write vector) to the stream.",
                      f"  |write| {c['out_norm']:.3g}, cos(read, write) {c['cos_in_out']:+.2f} ({c['mechanism']})"]
            for r in c["reads"]:
                lines.append(f"  {r['kind']} read (|w| {r['norm']:.3g}" + (f", bias {r['bias']:.4g}" if r["bias"] is not None else "")
                             + f"): token embeddings along it {_toks(r['embeddings']['top'])}")
            lines += [f"  promotes: {_toks(c['writes']['promotes']['top'])}",
                      f"  suppresses: {_toks(c['writes']['promotes']['bottom'])}"]
        elif s.recognised and len(parts) == 3 and parts[0] == "blocks" and parts[2] == "mlp":
            l = int(parts[1])
            gated = "W_gate" in s.layers[l].mlp
            lines.append(f"MLP of layer {l}: x += W_out-weighted sum of neurons; neuron n's activation is "
                         + ("act(x . gate_n) * (x . up_n)" if gated else "act(x . in_n + b_n)")
                         + ". A neuron is one row/column across the read matrices, its bias, and its write vector.")
        elif s.recognised and len(parts) == 3 and parts[0] == "blocks" and parts[2] == "attn":
            lines.append(f"attention of layer {parts[1]}: per head, query (destination) and key (source) give the attention "
                         f"pattern over earlier positions; the value read at the attended positions is written back through the "
                         f"output matrix.")
        elif s.recognised and parts[0] == "blocks" and len(parts) == 2:
            lines.append(f"layer {parts[1]}: attention sublayer then MLP sublayer, each adding to the residual stream.")
    except Exception as e:      # context is best-effort; never block a read on it
        lines.append(f"(context unavailable: {type(e).__name__}: {e})")
    changed = sum(ws.changed(n, i)[0] for n, i in ws.resolve(addr) if ws.pieces(n))
    if changed:
        lines.append(f"edited: {changed} weights here differ from the base model")
    return "\n".join(lines)


def render(ws, addrs: list[str], fmt: str = "q", variant: str = "current", context: str = "full", title: str = "",
           rows: tuple[int, int] | None = None) -> str:
    """A chunk: header + every tensor slice the addresses resolve to."""
    out = [f"# {title or ', '.join(addrs)}", model_line(ws)]
    for a in addrs[:3]:
        out.append(where_header(ws, a, context))
    if len(addrs) > 3:
        out.append(f"(+ {len(addrs) - 3} more addresses: {', '.join(addrs[3:12])}{' …' if len(addrs) > 12 else ''})")
    out.append("To change any of this, write a weight script using the addresses in the headings, e.g. "
               "`<address>[row, col] = value` or `<address> *= 0.5`.")
    seen = set()
    for a in addrs:
        for name, idx in ws.resolve(a):
            key = (name, A.index_str(idx))
            if key in seen:
                continue
            seen.add(key)
            out.append(render_tensor(ws, name, idx, fmt, variant, rows=rows))
    return "\n\n".join(out)


def est_text(text: str) -> int:
    """Token estimate for already-rendered text (numbers dominate: ~1.5 characters per token; prose ~3.5)."""
    digits = sum(ch.isdigit() or ch in "-.: " for ch in text)
    return int((digits / 1.5 + (len(text) - digits) / 3.2) * 1.05)


# ------------------------------------------------------------------------------ planning
_partition = hierarchy.partition


def _split_rows(ws, sl: tuple, fmt: str, budget: int) -> list[dict]:
    name, idx = sl
    shape = A.slice_shape(ws.shape(name), idx)
    m_rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    cols = shape[-1] if shape else 1
    per_row = est_tensor((1, cols), fmt) - int(TENSOR_OVERHEAD * SAFETY)
    step = max(1, (budget - CHUNK_OVERHEAD - TENSOR_OVERHEAD) // max(1, per_row))
    out = []
    for r0 in range(0, m_rows, step):
        r1 = min(m_rows, r0 + step)
        out.append({"addrs": [A.slice_str(sl)], "rows": [r0, r1], "title": f"{A.slice_str(sl)} rows {r0}:{r1}",
                    "n_params": (r1 - r0) * cols, "est_tokens": est_tensor((r1 - r0, cols), fmt) + CHUNK_OVERHEAD})
    return out


def plan(ws, selection: list[str] | None = None, budget: int = 120_000, fmt: str = "auto",
         total_budget: int = 700_000) -> dict:
    """Cut a selection into chunks of at most `budget` estimated tokens each.

    fmt="auto" picks the most faithful format whose *total* fits `total_budget` (so that the whole
    selection can sit in one 1M-token context with room to work); otherwise the given format is used
    and the plan simply has as many chunks as it needs.
    """
    selection = selection or ["model"]
    budget = max(2_000, int(budget))
    all_slices = [s for a in selection for s in ws.resolve(a)]
    totals = {f: est_slices(ws, all_slices, f) for f in ("g3", "q", "svd", "stats")}
    if fmt == "auto":       # g3 only when it is cheap; q carries nearly the same information at half the tokens
        fmt = "g3" if totals["g3"] <= total_budget // 4 else next((f for f in LADDER[1:] if totals[f] <= total_budget), "stats")
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS} or 'auto'")
    chunks: list[dict] = []

    def cost(addr: str) -> int:
        return est_slices(ws, ws.resolve(addr), fmt) + CHUNK_OVERHEAD

    def walk(addr: str):
        if fmt in ("svd", "stats"):            # summaries are per tensor: never cut below a whole tensor (or the slice asked for)
            for sl in ws.resolve(addr):
                a = A.slice_str(sl)
                if a not in seen_summary:
                    seen_summary.add(a)
                    pending.append((a, cost(a)))
            return
        c = cost(addr)
        if c <= budget:
            pending.append((addr, c))
            return
        kids = _partition(ws, addr)
        if kids:
            flush()
            for k in kids:
                walk(k)
            flush()
            return
        flush()
        chunks.extend(_split_rows(ws, ws.resolve(addr)[0], fmt, budget))

    pending: list[tuple[str, int]] = []
    seen_summary: set[str] = set()

    def flush():
        """Merge consecutive siblings into one chunk while they fit."""
        cur: list[str] = []
        tot = 0
        for addr, c in pending:
            if cur and tot + c - CHUNK_OVERHEAD > budget:
                _emit(cur, tot)
                cur, tot = [], 0
            cur.append(addr)
            tot += c if not cur[:-1] else c - CHUNK_OVERHEAD
        if cur:
            _emit(cur, tot)
        pending.clear()

    def _emit(addrs: list[str], tot: int):
        chunks.append({"addrs": addrs, "title": _title(addrs), "est_tokens": tot,
                       "n_params": sum(A.count(ws, ws.resolve(a)) for a in addrs)})

    for a in selection:
        walk(a)
    flush()
    for i, c in enumerate(chunks):
        c["id"] = i
        c["format"] = fmt
    return {"selection": selection, "format": fmt, "budget": budget, "n_params": A.count(ws, all_slices),
            "est_tokens": sum(c["est_tokens"] for c in chunks), "chunks": chunks, "totals_by_format": totals,
            "fits_one_context": sum(c["est_tokens"] for c in chunks) <= total_budget}


def _title(addrs: list[str]) -> str:
    if len(addrs) == 1:
        return addrs[0]
    return f"{addrs[0]} … {addrs[-1]} ({len(addrs)} parts)"


def render_chunk(ws, chunk: dict, variant: str = "current", context: str = "full") -> str:
    rows = tuple(chunk["rows"]) if chunk.get("rows") else None
    return render(ws, chunk["addrs"], chunk["format"], variant, context, title=f"chunk {chunk['id']}: {chunk['title']}", rows=rows)
