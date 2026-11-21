"""Reading a model: views of weights, and views of what the weights do on a prompt.

Everything here is written against three things only: `ws.read` (one region of one tensor),
`ws.sem` (parts of the model in a canonical layout) and `ws.runner` (forward passes with
standard hook points).  So it is the same code for a 2-layer toy and a multi-gigabyte
checkpoint in any recognised architecture, and its cost is bounded by what it looks at:

Weight-only views (no forward pass, no model in memory)
    matrix_view      any 2-D slice pooled to a screen-sized grid; large tensors are streamed in slabs
    region_stats     exact for small regions, sampled (and labelled so) for large ones
    svd_view         exact for small matrices, randomised and streamed for large ones
    direction_tokens what a residual direction means: tokens that write it, tokens it promotes
    head_card        QK / OV circuits of a head (Elhage et al. 2021)
    neuron_card      what a neuron reads and writes (Geva et al. 2021/2022)
    wiring           composition between writers and later readers; sketched, and limited to a
                     window of layers, when the model is large

Activation views (forward passes through the runner)
    trace            predictions, attention patterns, logit lens, direct logit attribution
    patching         activation patching between a clean and a corrupted prompt
    ablation_scan    loss change when each head / MLP is removed
"""
from __future__ import annotations

from .runtime_events import job_scope
from .worker_jobs import worker_job
from .memory_guard import require_lease

import base64
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import addressing as A
from .store import norm_index, region_shape, region_size
from .tokenizer import show
from .workspace import Ref, STREAM_ELEMS

EXACT_ELEMS = 8_000_000           # regions up to this size get exact statistics
DENSE_ELEMS = 40_000_000          # matrices up to this size may be held whole for sorting / SVD


# ------------------------------------------------------------------------------ helpers
def tensor_of(ws, name: str, idx: tuple = (), variant: str = "current", extra=None) -> torch.Tensor:
    return ws.read(name, idx, variant, extra)


def as_matrix(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 0:
        return t.reshape(1, 1)
    if t.dim() == 1:
        return t[None]
    if t.dim() > 2:
        return t.reshape(-1, t.shape[-1])
    return t


def kept_roles(ws, name: str, idx: tuple) -> list[str]:
    roles = ws.struct.axis_roles(name)
    return [r for d, r in enumerate(roles) if d >= len(idx) or isinstance(idx[d], slice)]


def view_roles(ws, name: str, idx: tuple) -> tuple[str, str]:
    """Roles of the two displayed axes of tensor[idx] after as_matrix."""
    kept = kept_roles(ws, name, idx)
    if len(kept) == 0:
        return "scalar", "scalar"
    if len(kept) == 1:
        return "scalar", kept[0]
    if len(kept) > 2:
        return "×".join(kept[:-1]), kept[-1]
    return kept[0], kept[1]


def axis_offset(name: str, idx: tuple, shape: tuple[int, ...]) -> tuple[int, int]:
    """Index of the first displayed row / column in the tensor's own coordinates."""
    starts = [(i.indices(shape[d])[0] if isinstance(i, slice) else None) for d, i in enumerate(idx)]
    starts += [0] * (len(shape) - len(idx))
    kept = [s for s in starts if s is not None]
    if len(kept) == 1:
        return 0, kept[0]
    if len(kept) == 2:
        return kept[0], kept[1]
    return 0, (kept[-1] if kept else 0)


def cell_address(ws, addr: str, r: int, c: int) -> dict:
    """The tensor entry behind cell (r, c) of matrix_view(addr), as a weight-script address."""
    name, idx = ws.resolve(addr)[0]
    shape = ws.shape(name)
    region = norm_index(idx, shape)
    kept = [d for d, i in enumerate(region) if isinstance(i, range)]
    pos: dict[int, int] = {}
    if len(kept) == 1:
        pos[kept[0]] = c
    elif len(kept) >= 2:
        pos[kept[-1]] = c
        for d in reversed(kept[:-1]):
            n = len(region[d])
            pos[d] = r % n
            r //= n
    full = tuple(region[d][pos[d]] if d in pos else region[d] for d in range(len(shape)))
    return {"script": A.to_script(name, full), "param": name, "index": list(full),
            "current": float(ws.read(name, full)), "base": float(ws.read(name, full, "base"))}


def labels(ws, role: str, start: int, n: int) -> list[str]:
    if role == "token" and ws.tok is not None:
        return [show(ws.tok.token_str(i)) for i in range(start, start + n)]
    return [str(i) for i in range(start, start + n)]


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(np.asarray(a), dtype=np.float32).tobytes()).decode()


def _pool(m: torch.Tensor, fr: int, fc: int, how: str) -> torch.Tensor:
    """Pool an [R, C] matrix by integer factors (the last partial block is pooled over what is there)."""
    R, C = m.shape
    if fr == 1 and fc == 1:
        return m
    pr, pc = (-R) % fr, (-C) % fc
    x = F.pad(m, (0, pc, 0, pr), value=float("nan")).reshape((R + pr) // fr, fr, (C + pc) // fc, fc)
    x = x.permute(0, 2, 1, 3).reshape((R + pr) // fr, (C + pc) // fc, fr * fc)
    if how == "absmax":
        a = torch.nan_to_num(x, nan=0.0)
        return a.gather(-1, a.abs().argmax(-1, keepdim=True)).squeeze(-1)
    if how == "rms":
        return torch.nan_to_num(x, nan=0.0).pow(2).sum(-1).div((~x.isnan()).sum(-1).clamp(min=1)).sqrt()
    return torch.nanmean(x, -1)


class BandPool:
    """_pool of one band of rows (one row of the pooled grid) read in parts, for a band above one store read.

    `add` takes the band's rows in order, each part [rows, cols]. The blocks are _pool's. absmax gives _pool's value
    (the first largest |value| of each block); mean and rms are summed in float64, so they can differ from _pool's
    float32 reductions in the last float32 digits; inf and NaN give what _pool gives. `sample` holds the values
    t.flatten()[::max(1, t.numel() // 20_000)] of the whole band would give (`n_values` is the band's size)."""

    def __init__(self, cols: int, fc: int, how: str, n_values: int):
        self.nb, self.pc, self.fc, self.how = -(-cols // fc), (-cols) % fc, fc, how
        self.sum = torch.zeros(self.nb, dtype=torch.float64)
        self.count = torch.zeros(self.nb, dtype=torch.int64)
        self.best = None
        self.every, self.at, self.sample = max(1, n_values // 20_000), 0, []

    def add(self, t: torch.Tensor) -> None:
        h = t.shape[0]
        x = F.pad(t, (0, self.pc), value=float("nan")).reshape(h, self.nb, self.fc).permute(1, 0, 2).reshape(self.nb, h * self.fc)
        if self.how == "absmax":
            a = torch.nan_to_num(x, nan=0.0)
            v = a.gather(-1, a.abs().argmax(-1, keepdim=True)).squeeze(-1)
            self.best = v if self.best is None else torch.where(v.abs() > self.best.abs(), v, self.best)
        else:          # as _pool: NaN is missing; mean keeps inf (nanmean), rms squares in float32 after nan_to_num
            z = torch.nan_to_num(x, nan=0.0).pow(2) if self.how == "rms" else torch.nan_to_num(
                x, nan=0.0, posinf=float("inf"), neginf=float("-inf"))
            self.sum += z.double().sum(-1)
            self.count += (~x.isnan()).sum(-1)
        flat = t.flatten()
        self.sample.append(flat[(-self.at) % self.every::self.every])
        self.at += flat.numel()

    def result(self) -> torch.Tensor:
        """[1, blocks]"""
        if self.how == "absmax":
            return self.best[None]
        m = self.sum / self.count.clamp(min=1)
        return (m.sqrt() if self.how == "rms" else m).float()[None]


def robust_scale(a: torch.Tensor) -> float:
    a = a.abs().flatten()
    if a.numel() == 0:
        return 1.0
    if a.numel() > 200_000:
        a = a[torch.randint(0, a.numel(), (200_000,), generator=torch.Generator().manual_seed(0))]
    q = float(torch.quantile(a, 0.995))
    return q if q > 0 else (float(a.max()) or 1.0)


def _single(ws, addr: str) -> tuple[str, tuple]:
    slices = ws.resolve(addr)
    if len(slices) != 1:
        raise ValueError(f"{addr} is {len(slices)} tensors; pick one of them")
    return slices[0]


def matrix_view(ws, addr: str, variant: str = "current", r0: int = 0, r1: int | None = None, c0: int = 0,
                c1: int | None = None, max_h: int = 256, max_w: int = 384, pool: str = "mean",
                sort_rows: str = "", sort_cols: str = "") -> dict:
    name, idx = _single(ws, addr)
    shape = ws.shape(name)
    region = norm_index(idx, shape)
    kshape = region_shape(region)
    R = int(np.prod(kshape[:-1])) if len(kshape) > 1 else 1
    C = kshape[-1] if kshape else 1
    r1, c1 = min(r1 or R, R), min(c1 or C, C)
    fr, fc = max(1, -(-(r1 - r0) // max_h)), max(1, -(-(c1 - c0) // max_w))
    rperm = cperm = None
    sortable = R * C <= DENSE_ELEMS
    if len(kshape) != 2 or sortable:                      # small enough to hold: allows reordering
        full = as_matrix(Ref(ws, name, region).whole(variant))   # up to DENSE_ELEMS: above one quantized-store read
        rperm = _order(full, sort_rows, 0) if sortable else None
        cperm = _order(full, sort_cols, 1) if sortable else None
        m = full
        if rperm is not None:
            m = m[rperm]
        if cperm is not None:
            m = m[:, cperm]
        pooled = _pool(m[r0:r1, c0:c1], fr, fc, pool)
        scale = robust_scale(full)
    else:                                                 # stream the rows in slabs aligned to the pooling factor
        kept = [d for d, r in enumerate(region) if isinstance(r, range)]
        sub = list(region)
        sub[kept[0]] = region[kept[0]][r0:r1]
        sub[kept[1]] = region[kept[1]][c0:c1]
        cols = len(sub[kept[1]])
        step = max(fr, (STREAM_ELEMS // max(1, cols)) // fr * fr)
        parts, sample = [], []
        rows = sub[kept[0]]
        if len(rows) * 4 >= shape[kept[0]]:
            ws.store.fetch(name)
        per = max(1, STREAM_ELEMS // max(1, cols))       # rows per read when one band of fr rows is above one read
        for a in range(0, len(rows), step):
            piece = list(sub)
            piece[kept[0]] = rows[a:a + step]
            if fr * cols > STREAM_ELEMS:                 # a small max_h on a large matrix: pool the band as it is read
                band, bp = rows[a:a + step], BandPool(cols, fc, pool, len(rows[a:a + step]) * cols)
                for b in range(0, len(band), per):
                    piece[kept[0]] = band[b:b + per]
                    bp.add(ws.read(name, tuple(piece), variant))
                parts.append(bp.result())
                sample += bp.sample
                continue
            t = ws.read(name, tuple(piece), variant)
            parts.append(_pool(t, fr, fc, pool))
            sample.append(t.flatten()[:: max(1, t.numel() // 20_000)])
        pooled = torch.cat(parts, 0)
        scale = robust_scale(torch.cat(sample))
    rrole, crole = view_roles(ws, name, idx)
    roff, coff = axis_offset(name, idx, shape)
    out = {"addr": addr, "param": name, "variant": variant, "shape": [R, C], "region": [r0, r1, c0, c1],
           "grid": list(pooled.shape), "factor": [fr, fc], "data": b64(pooled.numpy()), "scale": scale,
           "row_role": rrole, "col_role": crole, "row_offset": roff, "col_offset": coff, "pool": pool, "sortable": sortable,
           "row_perm": rperm.tolist() if rperm is not None and R <= 4096 else None,
           "col_perm": cperm.tolist() if cperm is not None and C <= 4096 else None}
    exact = fr == 1 and fc == 1
    if exact and rperm is None and (r1 - r0) <= 512 and len(kshape) <= 2:
        out["row_labels"] = labels(ws, rrole, roff + r0, r1 - r0)
    if exact and cperm is None and (c1 - c0) <= 512:
        out["col_labels"] = labels(ws, crole, coff + c0, c1 - c0)
    return out


def _order(m: torch.Tensor, how: str, axis: int) -> torch.Tensor | None:
    if not how or m.shape[axis] > 16384:
        return None
    x = m if axis == 0 else m.T
    if how == "norm":
        return x.norm(dim=1).argsort(descending=True)
    if how == "svd":
        u, s, _ = torch.svd_lowrank(x, q=min(2, *x.shape))
        return u[:, 0].argsort()
    if how == "cluster":      # order by the angle of the two leading singular coordinates: groups similar rows
        u, s, _ = torch.svd_lowrank(x, q=min(3, *x.shape))
        if u.shape[1] < 2:
            return u[:, 0].argsort()
        return torch.atan2(u[:, 1] * s[1], u[:, 0] * s[0]).argsort()
    return None


def stats(t: torch.Tensor, bins: int = 41) -> dict:
    t = t.flatten().float()
    if t.numel() == 0:
        return {"n": 0}
    lim = float(t.abs().max()) or 1.0
    hist = torch.histc(t, bins=bins, min=-lim, max=lim)
    return {"n": t.numel(), "mean": float(t.mean()), "std": float(t.std()) if t.numel() > 1 else 0.0,
            "min": float(t.min()), "max": float(t.max()), "norm": float(t.norm()),
            "zeros": int((t == 0).sum()), "abs_mean": float(t.abs().mean()),
            "hist": hist.tolist(), "hist_lim": lim}


def region_stats(ws, addr: str) -> dict:
    """Distribution of the weights at an address. Exact up to EXACT_ELEMS weights; beyond that a sample of
    evenly spaced slabs is read (about a million weights) and the result says so."""
    slices = [(n, i) for n, i in ws.resolve(addr) if ws.editable(n)]
    total = A.count(ws, slices)
    changed, dsq = 0, 0.0
    for name, idx in slices:
        if ws.pieces(name):
            c, d = ws.changed(name, idx)
            changed, dsq = changed + c, dsq + d
    if total <= EXACT_ELEMS:
        cur = torch.cat([ws.read(n, i).flatten() for n, i in slices]) if slices else torch.zeros(0)
        out = {"current": stats(cur), "sampled": False}
    else:
        budget = 1_200_000
        parts = []
        for name, idx in slices:
            region = norm_index(idx, ws.shape(name))
            n = region_size(region)
            take = max(2_000, int(budget * n / total))
            kept = [d for d, r in enumerate(region) if isinstance(r, range)]
            if n <= take or not kept:
                parts.append(ws.read(name, region).flatten())
                continue
            rows = region[kept[0]]
            per = n // len(rows)
            n_rows = max(1, take // per)
            stride = max(1, len(rows) // n_rows)
            sub = region[:kept[0]] + (rows[::stride][:n_rows],) + region[kept[0] + 1:]
            parts.append(ws.read(name, sub).flatten())
        cur = torch.cat(parts)
        st = stats(cur)
        st["norm"] = st["norm"] * (total / max(1, cur.numel())) ** 0.5          # scaled up from the sample
        st["n"] = total
        out = {"current": st, "sampled": True, "sample": int(cur.numel())}
    out["delta"] = {"norm": dsq ** 0.5, "n": total}
    out["changed"] = changed
    return out


# ------------------------------------------------------------------------------ meaning of directions
def _top(ws, scores: torch.Tensor, k: int, role: str = "token") -> dict:
    k = min(k, scores.numel())
    hi = scores.topk(k)
    lo = (-scores).topk(k)
    lab = (lambda i: show(ws.tok.token_str(i))) if role == "token" and ws.tok is not None else str
    return {"top": [[lab(int(i)), round(float(v), 4), int(i)] for v, i in zip(hi.values, hi.indices)],
            "bottom": [[lab(int(i)), round(float(-v), 4), int(i)] for v, i in zip(lo.values, lo.indices)]}


def direction_tokens(ws, v: torch.Tensor, k: int = 12) -> dict:
    """A residual-stream direction read three ways."""
    return vocab_readings(ws, [], [v], k)[1][0]


def reader_tokens(ws, v: torch.Tensor, k: int = 12) -> dict:
    """A read direction: which token embeddings / positions excite it (direct path, earlier layers ignored)."""
    return vocab_readings(ws, [v], [], k)[0][0]


def vocab_readings(ws, reads: list[torch.Tensor], writes: list[torch.Tensor], k: int = 12) -> tuple[list[dict], list[dict]]:
    """reader_tokens of each read direction and direction_tokens of each write direction, with one pass over the
    embedding matrix (and one over the unembedding, none more if it is the same tensor) for all of them together:
    a card that read each direction on its own made four passes over a 248,320 x 5,120 vocabulary."""
    sem = ws.sem
    allv = [v.float() for v in reads] + [v.float() for v in writes]
    if not allv:
        return [], []
    nr = len(reads)
    emb, lgt = sem.vocab_scores(torch.stack(allv), torch.stack(allv[nr:]) if writes else None)
    ps = sem.pos_scores(torch.stack(allv, 1)) if sem.W_pos() is not None else None
    out = []
    for i, v in enumerate(allv):
        d = {"norm": float(v.norm())}
        if i >= nr:
            d["promotes"] = _top(ws, lgt[:, i - nr], k)                   # v . W_U: logits it raises / lowers
        d["embeddings"] = _top(ws, emb[:, i], k)                          # tokens whose embedding points along v
        if ps is not None:
            d["positions"] = _top(ws, ps[:, i], min(k, 6), "pos")
        out.append(d)
    return out[:nr], out[nr:]


def _streamed_svd(ref: Ref, variant: str, k: int = 64):
    """Randomised SVD of a 2-D view in two passes over its rows (Halko, Martinsson & Tropp 2011)."""
    m, n = ref.shape
    g = torch.Generator().manual_seed(0)
    omega = torch.randn(n, min(k + 8, n), generator=g)
    Y = ref.contract(omega, axis=1, variant=variant)                 # [m, k+8]
    Q, _ = torch.linalg.qr(Y)
    B = ref.contract(Q, axis=0, variant=variant).T                   # Q^T A  ->  [k+8, n]
    ub, s, vh = torch.linalg.svd(B.double(), full_matrices=False)
    return (Q.double() @ ub)[:, :k], s[:k], vh[:k]


def svd_view(ws, addr: str, k: int = 6, variant: str = "current", top: int = 8) -> dict:
    name, idx = _single(ws, addr)
    region = norm_index(idx, ws.shape(name))
    kshape = region_shape(region)
    if len(kshape) < 2:
        raise ValueError("svd needs a matrix")
    R, C = int(np.prod(kshape[:-1])), kshape[-1]
    approx = False
    fro = None
    if R * C <= DENSE_ELEMS and min(R, C) <= 2048:
        u, s, vh = torch.linalg.svd(as_matrix(Ref(ws, name, region).whole(variant)).double(), full_matrices=False)
    elif R * C <= DENSE_ELEMS:
        m = as_matrix(Ref(ws, name, region).whole(variant))
        u, s, v = torch.svd_lowrank(m, q=64)
        vh, approx, fro = v.T, True, float(m.double().pow(2).sum())
    else:
        if len(kshape) != 2:
            raise ValueError("this tensor is too large for an SVD of its flattened form; pick a 2-D slice")
        u, s, vh = _streamed_svd(Ref(ws, name, region), variant)
        approx = True
    rrole, crole = view_roles(ws, name, idx)
    tot = fro or float(s.pow(2).sum()) or 1.0
    p = s.pow(2) / tot
    out = {"addr": addr, "shape": [R, C], "singular_values": [float(x) for x in s[:256]], "approx": approx,
           "effective_rank": None if approx else float(torch.exp(-(p * (p + 1e-30).log()).sum())),
           "rank": None if approx else (int((s > s[0] * 1e-5).sum()) if s.numel() and s[0] > 0 else 0),
           "energy": [float(x) for x in (s.pow(2).cumsum(0) / tot)[:256]], "row_role": rrole, "col_role": crole,
           "components": []}
    recog = ws.struct.recognised
    for i in range(min(k, s.numel())):
        comp = {"i": i, "s": float(s[i])}
        for side, vec, role in (("left", u[:, i].float(), rrole), ("right", vh[i].float(), crole)):
            if role == "resid" and recog and vec.numel() == ws.struct.d_model:
                comp[side] = {"kind": "resid", **direction_tokens(ws, vec, top)}
            elif role == 'resid':
                # A sliced residual axis is not a complete residual direction.
                # Label local coordinates; don't multiply it by the full embedding.
                comp[side] = {"kind": "selected residual coordinates", **_top(ws, vec, top, 'coordinate')}
            elif "×" not in role and role != "scalar":
                comp[side] = {"kind": role, **_top(ws, vec, top, role)}
        out["components"].append(comp)
    return out


# ------------------------------------------------------------------------------ units
def _sample_tokens(ws, n: int = 512) -> torch.Tensor:
    V = ws.struct.d_vocab
    if V > 60_000:
        n = min(n, 192)
    if V <= n:
        return torch.arange(V)
    return torch.randperm(V, generator=torch.Generator().manual_seed(0))[:n]


def _attn_scale(ws) -> float:
    return float(ws.cfg.attn_scale) if ws.cfg is not None else float(ws.struct.d_head) ** 0.5


def head_card(ws, l: int, h: int, k: int = 8) -> dict:
    sem, s = ws.sem, ws.struct
    H = sem.head(l, h)
    WQ, WK, WV, WO = (H[m].read() for m in "QKVO")
    bq, bk, bv = (H[f"b_{m}"].read() if f"b_{m}" in H else 0.0 for m in "QKV")
    scale = _attn_scale(ws)
    out: dict = {"layer": l, "head": h, "addr": f"blocks.{l}.attn.head.{h}",
                 "norms": {m: float(x.norm()) for m, x in zip("QKVO", (WQ, WK, WV, WO))},
                 "kv_group": h * s.n_kv_heads // s.n_heads if s.n_kv_heads and s.n_kv_heads != s.n_heads else None}
    if s.pos is not None:                 # positional QK: how the score depends on (destination, source) position alone
        Wpos = sem.W_pos().read()
        pq = (Wpos @ WQ + bq) @ (Wpos @ WK + bk).T / scale
        n = pq.shape[0]
        offs = []
        for off in range(min(n, 24)):
            d = torch.diagonal(pq, -off)
            rest = torch.stack([pq[i, :i + 1].mean() for i in range(off, n)])
            offs.append(float((d - rest).mean()))
        out["pos_offset_preference"] = offs
        f = max(1, -(-n // 64))
        pooled = _pool(pq.tril(), f, f, "mean")
        out["pos_qk"] = {"grid": list(pooled.shape), "data": b64(pooled.numpy()), "scale": robust_scale(pq.tril())}
    # OV circuit on a token sample: source token -> promoted output tokens
    S = _sample_tokens(ws)
    ES = sem.embed_rows(S)
    ov = (ES @ WV + bv) @ WO                                        # [S, d_model]
    u, sv, vh = torch.linalg.svd(WV.double() @ WO.double(), full_matrices=False)
    nc = min(WO.shape[0], 3)
    # One pass over each vocabulary matrix for the OV sample and the singular directions together.
    emb, lgt = sem.vocab_scores(u[:, :nc].T.float(), torch.cat([ov, vh[:nc].float()]))
    lg = lgt[:, :len(S)].T                                          # [S, V]
    own = lg[torch.arange(len(S)), S]
    rank = (lg > own[:, None]).sum(1)
    out["copying"] = {"top1": float((rank == 0).float().mean()), "top10": float((rank < 10).float().mean()), "sample": len(S)}
    strength = ov.norm(dim=1)
    ex = []
    for i in strength.topk(min(k, len(S))).indices.tolist():
        t = lg[i].topk(4)
        ex.append({"src": show(ws.tok.token_str(int(S[i]))), "norm": float(strength[i]),
                   "out": [[show(ws.tok.token_str(int(j))), round(float(v), 3)] for v, j in zip(t.values, t.indices)]})
    out["ov_examples"] = ex
    qk = (ES @ WQ + bq) @ (ES @ WK + bk).T / scale                  # token QK on the sample
    qk = qk - qk.mean(1, keepdim=True)
    flat = qk.flatten().topk(min(k, qk.numel()))
    out["qk_examples"] = [{"query": show(ws.tok.token_str(int(S[i // len(S)]))),
                           "key": show(ws.tok.token_str(int(S[i % len(S)]))), "score": round(float(v), 3)}
                          for v, i in zip(flat.values, flat.indices.tolist())]
    out["ov_svd"] = {"singular_values": [float(x) for x in sv[:WO.shape[0]]], "components": [
        {"s": float(sv[i]), "reads": _top(ws, emb[:, i], 6), "writes": _top(ws, lgt[:, len(S) + i], 6)}
        for i in range(nc)]}
    return out


def neuron_card(ws, l: int, n: int, k: int = 10) -> dict:
    N = ws.sem.neuron(l, n)
    w_out = N["write"].read()
    ws_in = [r["w"].read() for r in N["reads"]]
    readings, (writes,) = vocab_readings(ws, ws_in, [w_out], k)        # one pass over each vocabulary matrix
    reads = []
    for r, w, tokens in zip(N["reads"], ws_in, readings):
        cos = float(F.cosine_similarity(w, w_out, dim=0)) if w.norm() > 0 and w_out.norm() > 0 else 0.0
        reads.append({"kind": r["kind"], "bias": float(r["b"].read()) if r["b"] is not None else None, "norm": float(w.norm()),
                      "cos_write": cos, "addr": r["w"].address(), **tokens})
    main = reads[-1] if reads else None                   # the value read (up) for gated MLPs, the only read otherwise
    cos = main["cos_write"] if main else 0.0
    mech = "amplifier" if cos > 0.1 else "damper" if cos < -0.1 else "router"
    return {"layer": l, "neuron": n, "addr": f"blocks.{l}.mlp.neuron.{n}", "bias": main["bias"] if main else None,
            "in_norm": main["norm"] if main else 0.0, "out_norm": float(w_out.norm()), "cos_in_out": cos, "mechanism": mech,
            "gated": len(reads) > 1, "reads": reads, "writes": writes, "write_addr": N["write"].address()}


def default_window(ws, focus: int | None = None) -> list[int]:
    """Layers the wiring graph covers by default: all of a small model, a three-layer window of a large one."""
    s = ws.struct
    if ws.n_params() <= 400_000_000 and s.d_model <= 1024:
        return list(range(s.n_layers))
    if s.d_model > 2048:
        return [max(0,min(s.n_layers-1,0 if focus is None else focus))]
    f = 1 if focus is None else focus
    lo = max(0, min(f - 1, s.n_layers - 3))
    return list(range(lo, min(s.n_layers, lo + 3)))


def wiring(ws, top: int = 400, layers: list[int] | None = None, cancel=None, progress=None) -> dict:
    """Composition between components, from weights alone.

    score(writer -> reader) = |W_write W_read|_F / (|W_write|_F |W_read|_F) * sqrt(d_model)
    so 1.0 is what two unrelated random matrices give; well above 1 means the reader is aligned
    with what the writer writes. Heads have three readers (Q, K, V); MLPs one per read matrix;
    the unembedding one. Normalisation between writer and reader is ignored.

    |W_w R|^2 = <W_w^T W_w, R R^T>, so each component is reduced once to a small Gram matrix and
    all pairs are one matrix product. When the Gram matrices of all writers would not fit in a
    few hundred megabytes (wide models, or many components) the residual space is first projected
    to r <= 1024 random dimensions. The inner products are then estimates (the projection's known
    bias is subtracted; the norms stay exact), noisier the smaller r is against d_model: see the
    README for how far they were from exact scores where both were computed.
    For large models only a window of layers is read.
    """
    s, sem = ws.struct, ws.sem
    layers = default_window(ws) if layers is None else sorted(set(int(l) for l in layers if 0 <= int(l) < s.n_layers))
    from .wiring_stream import Window, rank_for, writer_gram, reader_gram, head_writer, WORK_VALUES
    window = Window(cancel,progress)
    window.check()
    if not layers: raise ValueError('Select at least one layer for wiring')
    if len(layers)>3 and ws.n_params()>400_000_000:
        raise ValueError('Large-model wiring is limited to at most three layers per request')
    key = "wiring:" + ",".join(map(str, layers))
    if key in ws.calib:
        full = ws.calib[key]
        return {**full, "edges": full["edges"][:top]}
    d = s.d_model
    ends = len(layers) == s.n_layers and s.d_vocab * d <= 80_000_000     # embeddings only when the whole model is covered
    work = (sem.W_E().size+sem.W_U().size+(sem.W_pos().size if s.pos else 0)) if ends else 0
    for layer in layers:
        if s.has_heads(layer):
            for head_index in range(s.n_heads):
                h = sem.head(layer,head_index)
                work += h['O'].size+h['Q'].size+h['K'].size+2*h['V'].size
        if s.has_neurons(layer):
            ml = sem.mlp(layer)
            work += ml['W_out'].size+sum(ml[role].size for role in s.layers[layer].reads)
    if work>WORK_VALUES:raise ValueError('Wiring window exceeds 512M logical values; choose fewer layers before scanning weights')
    n_writers = sum((s.n_heads if s.has_heads(l) else 0) + (1 if s.has_neurons(l) else 0) for l in layers) + (2 if ends else 0)
    n_readers = sum((3*s.n_heads if s.has_heads(l) else 0)+len(s.layers[l].reads) for l in layers)+(1 if ends else 0)
    r, memory_bound = rank_for(d,n_writers,n_readers)
    sketch = r < d
    om = torch.randn(d,r,generator=torch.Generator().manual_seed(0))/r**.5 if sketch else None
    nodes, S, s_tr, t_tr, wid, rid, readers, order = [], [], [], [], [], [], [], {}

    def writer(nid, gram, tr):
        wid.append(nid)
        S.append(gram.float().reshape(-1))
        s_tr.append(float(tr))

    def reader(nid,kind,ref):
        gram,norm = reader_gram(ref,om,window)
        rid.append((nid,kind));readers.append(("gram",gram));t_tr.append(norm)

    if ends:
        nodes.append({"id": "embed", "kind": "embed", "layer": -1, "label": "embed", "addr": "embed"})
        writer("embed", *writer_gram(sem.W_E(),om,window))
        order["embed"] = -1.0
        if s.pos is not None:
            nodes.append({"id": "pos", "kind": "embed", "layer": -1, "label": "pos", "addr": "pos_embed"})
            writer("pos", *writer_gram(sem.W_pos(),om,window))
            order["pos"] = -1.0
    for l in layers:
        if s.has_heads(l):
            ch = sum(ws.changed(p.name)[0] for r_, p in s.layers[l].attn.items() if r_.startswith("W_"))
            for h in range(s.n_heads):
                nid = f"L{l}H{h}"
                head = sem.head(l,h)
                gram,tr,norm = head_writer(head['V'],head['O'],om,window)
                nodes.append({"id": nid, "kind": "head", "layer": l, "index": h, "label": f"{l}.{h}",
                              "addr": f"blocks.{l}.attn.head.{h}", "norm": norm, "edited": ch > 0})
                writer(nid,gram,tr)
                for kind in ('Q','K','V'):reader(nid,kind,head[kind])
                order[nid] = l
        if s.has_neurons(l):
            ml = sem.mlp(l)
            nid = f"L{l}MLP"
            ch = sum(ws.changed(p.name)[0] for r_, p in s.layers[l].mlp.items() if r_.startswith("W_"))
            gram,tr = writer_gram(ml["W_out"],om,window)
            nodes.append({"id": nid, "kind": "mlp", "layer": l, "label": f"mlp {l}", "addr": f"blocks.{l}.mlp",
                          "norm": tr**.5, "edited": ch > 0})
            writer(nid,gram,tr)
            for role in s.layers[l].reads:
                reader(nid, {"W_in": "in", "W_gate": "gate", "W_up": "in"}[role], ml[role])
            order[nid] = l + 0.5
    if ends:
        nodes.append({"id": "unembed", "kind": "unembed", "layer": s.n_layers, "label": "unembed", "addr": "unembed"})
        reader("unembed", "U", sem.W_U())
        order["unembed"] = s.n_layers + 1.0
    edges = []
    edge_count = 0
    if S and readers:
        Sm = torch.stack(S)                                                # [writers, r*r] float32
        st = torch.tensor(s_tr, dtype=torch.double)
        S.clear()
        step = 1                                  # readers in chunks: bounded memory
        for a in range(0, len(readers), step):
            part = readers[a:a + step]
            Tm = torch.stack([(P @ P.T if form == "factor" else P).reshape(-1) for form, P in part])
            tt = torch.tensor(t_tr[a:a + step], dtype=torch.double)
            sc2 = (Sm @ Tm.T).double() / (st[:, None] * tt[None]).clamp(min=1e-30) * d      # score squared
            if sketch:
                # With G = Omega Omega^T (entries of Omega ~ N(0, 1/r)):  E tr(S G T G) = (1 + 1/r) tr(ST) + tr(S) tr(T) / r.
                # The second term would lift every score by about sqrt(d / r); the norms are exact, so it can be removed.
                sc2 = (sc2 - d / r) / (1 + 1 / r)
            sc = sc2.clamp(min=0).sqrt()
            for j, (rn, kind) in enumerate(rid[a:a + step]):
                if t_tr[a + j] == 0:
                    continue
                for i, w in enumerate(wid):
                    if order[rn] > order[w] and s_tr[i] > 0 and not (s.tied and w == "embed" and rn == "unembed"):
                        edge_count += 1
                        edges.append({"src": w, "dst": rn, "kind": kind, "score": round(float(sc[i, j]), 4)})
            if len(edges)>8000:edges=sorted(edges,key=lambda e:-e['score'])[:4000]
            window.check()
        del Sm
    edges.sort(key=lambda e: -e["score"])
    edges = edges[:4000]
    out = {"nodes": nodes, "edges": edges, "n_edges": edge_count, "chance": 1.0, "layers": s.n_layers, "heads": s.n_heads,
           "window": layers, "sketched": sketch, "rank": r, "ends": ends,
           "bounds": {"slab_values": window.max_read, "read_values": window.values, "retained_bytes_bound": memory_bound, "planned_values": work, "edge_limit": 4000},
           "limitations": [f"Layer {l}: recurrent/hybrid attention composition is not represented; MLP only" for l in layers if not s.has_heads(l)]}
    ws.calib[key] = out
    return {**out, "edges": edges[:top]}


# ------------------------------------------------------------------------------ activations on a prompt
def _one_token(ws, s: str) -> int:
    ids = ws.tok.encode_token(s)
    if len(ids) != 1:
        raise ValueError(f"{s!r} is {len(ids)} tokens ({[ws.tok.token_str(i) for i in ids]}); a single token is needed")
    return ids[0]


def _hook_names(ws, kinds: list[str]) -> list[str]:
    r = ws.runner
    return [f"blocks.{l}.{k}" for l in range(ws.struct.n_layers) for k in kinds if r.available(f"blocks.{l}.{k}")]


def _head_out(ws, l: int, z: torch.Tensor) -> torch.Tensor:
    """Per-head contributions to the stream from mixed values z [..., head, d_head] -> [..., head, d_model]."""
    return torch.einsum("...he,hed->...hd", z, ws.sem.stack(l, "W_O"))


@job_scope('trace')
@worker_job('trace')  # the whole call runs in the model worker when the workspace uses one
def trace(ws, prompt: str, variant: str = "current", target: str | None = None, versus: str | None = None,
          k: int = 5, max_tokens: int = 64, extra=None) -> dict:
    s, r = ws.struct, ws.runner
    limit = min(r.max_ctx or 10 ** 9, max_tokens)
    ids = ws.encode(prompt)[-limit:]
    T, L = len(ids), s.n_layers
    names = _hook_names(ws, ["hook_resid_pre", "hook_resid_post", "hook_attn_out", "hook_mlp_out", "attn.hook_z", "mlp.hook_post"])
    names = [n for n in names if not n.endswith("hook_resid_pre") or n.startswith("blocks.0.")]
    pat_ok = L * s.n_heads * T * T <= 800_000
    names += [f"blocks.{l}.attn.hook_pattern" for l in range(L) if s.has_heads(l)] if pat_ok else []
    logits, cache = r.run(ids, variant, extra, names=names)
    logits = logits[0]
    probs = logits.softmax(-1)
    strs = [show(ws.tok.token_str(i)) for i in ids]
    out: dict = {"tokens": strs, "ids": ids, "variant": variant}
    tp = probs.topk(min(k, probs.shape[-1]), dim=-1)
    out["predictions"] = [[[show(ws.tok.token_str(int(j))), round(float(v), 4)] for v, j in zip(tp.values[t], tp.indices[t])]
                          for t in range(T)]
    pats = {}
    for l in range(L):
        if getattr(ws,'_runtime_client',None):require_lease(ws)
        p = cache.get(f"blocks.{l}.attn.hook_pattern")
        if p is not None:
            pats[l] = [[[round(float(x), 3) for x in row] for row in p[0, h]] for h in range(p.shape[1])]
    out["patterns"] = [pats.get(l) for l in range(L)]
    tgt = _one_token(ws, target) if target else int(logits[-1].argmax())
    alt = _one_token(ws, versus) if versus else None
    out["target"] = show(ws.tok.token_str(tgt))
    out["versus"] = show(ws.tok.token_str(alt)) if alt is not None else None
    # logit lens: decode the residual stream after every layer
    lens = []
    stages = [("embed", cache.get("blocks.0.hook_resid_pre"))] + [(f"L{l}", cache.get(f"blocks.{l}.hook_resid_post")) for l in range(L)]
    step = max(1, -(-L // 16))
    for i, (label, x) in enumerate(stages):
        if x is None or (i not in (0, len(stages) - 1) and (i - 1) % step != step - 1):
            continue
        lp = r.unembed(x[0], variant, extra).softmax(-1)
        top = lp.argmax(-1)
        lens.append({"layer": label, "top": [show(ws.tok.token_str(int(i_))) for i_ in top],
                     "top_p": [round(float(lp[t, top[t]]), 4) for t in range(T)],
                     "target_p": [round(float(lp[t, tgt]), 4) for t in range(T)]})
    out["lens"] = lens
    # direct logit attribution at the last position (the final norm's scale frozen at its actual value)
    x_final = cache[f"blocks.{L - 1}.hook_resid_post"][0, -1]
    gain, centre = r.final_gain(x_final, variant, extra)
    sem = ws.sem
    dirv = sem.unembed_col(tgt, variant=variant, extra=extra)
    if alt is not None:
        dirv = dirv - sem.unembed_col(alt, variant=variant, extra=extra)
    w = gain * dirv

    def contrib(x):
        if centre:
            x = x - x.mean(-1, keepdim=True)
        return x @ w

    dla = []
    if "blocks.0.hook_resid_pre" in cache:
        dla.append({"id": "embed", "label": "embed", "kind": "embed", "layer": -1, "value": float(contrib(cache["blocks.0.hook_resid_pre"][0, -1]))})
    mlp_layers = []
    for l in range(L):
        if getattr(ws,'_runtime_client',None):require_lease(ws)
        a = cache.get(f"blocks.{l}.hook_attn_out")
        z = cache.get(f"blocks.{l}.attn.hook_z")
        if z is not None and s.has_heads(l):
            per = contrib(_head_out(ws, l, z[0, -1]))
            for h in range(per.shape[0]):
                dla.append({"id": f"L{l}H{h}", "label": f"{l}.{h}", "kind": "head", "layer": l, "index": h, "value": float(per[h])})
            if a is not None:
                rest = float(contrib(a[0, -1])) - float(per.sum())
                if abs(rest) > 1e-6:
                    dla.append({"id": f"L{l}bO", "label": f"attn bias {l}", "kind": "bias", "layer": l, "value": rest})
        elif a is not None:
            dla.append({"id": f"L{l}ATTN", "label": f"attn {l}", "kind": "attn", "layer": l, "value": float(contrib(a[0, -1]))})
        m = cache.get(f"blocks.{l}.hook_mlp_out")
        if m is not None:
            v = float(contrib(m[0, -1]))
            dla.append({"id": f"L{l}MLP", "label": f"mlp {l}", "kind": "mlp", "layer": l, "value": v})
            mlp_layers.append((abs(v), l))
    out["dla"] = dla
    out["dla_total"] = float(logits[-1, tgt] - (logits[-1, alt] if alt is not None else 0))
    # neurons with the largest direct effect, in the MLPs that matter most
    neurons = []
    for _, l in sorted(mlp_layers, reverse=True)[:6]:
        post = cache.get(f"blocks.{l}.mlp.hook_post")
        if post is None or not s.has_neurons(l):
            continue
        W = ws.sem.mlp(l)["W_out"]
        wc = w - w.mean() if centre else w
        eff = post[0, -1] * W.contract(wc, axis=1, variant=variant, extra=extra)
        for v, i in zip(*eff.abs().topk(min(6, eff.numel()))):
            neurons.append({"addr": f"L{l}N{int(i)}", "layer": l, "neuron": int(i), "act": round(float(post[0, -1, i]), 4),
                            "effect": round(float(eff[i]), 4)})
    neurons.sort(key=lambda n: -abs(n["effect"]))
    out["neurons"] = neurons[:24]
    return out


@job_scope('patching')
@worker_job('patching')  # the whole call runs in the model worker when the workspace uses one
def patching(ws, clean: str, corrupt: str, target: str | None = None, versus: str | None = None,
             variant: str = "current", budget: int = 500) -> dict:
    """Run on `corrupt`, copy one activation from the `clean` run, measure how much of the clean answer returns.

    0 = no better than the corrupted run, 1 = fully restored (Meng et al. 2022; Heimersheim & Nanda 2024).
    Each cell is one forward pass; when layers x positions would exceed `budget` passes the scan is done
    per layer over all positions at once, and per-head patching is skipped."""
    s, r = ws.struct, ws.runner
    a, b = ws.encode(clean), ws.encode(corrupt)
    if len(a) != len(b):
        raise ValueError(f"clean ({len(a)} tokens) and corrupted ({len(b)} tokens) prompts must have the same length")
    if r.max_ctx and len(a) > r.max_ctx:
        raise ValueError("prompts are longer than the context")
    T, L = len(a), s.n_layers
    kinds = [("resid_pre", "hook_resid_pre"), ("attn_out", "hook_attn_out"), ("mlp_out", "hook_mlp_out")]
    kinds = [(k, h) for k, h in kinds if any(r.available(f"blocks.{l}.{h}") for l in range(L))]
    by_pos = len(kinds) * L * T <= budget
    do_heads = L * s.n_heads <= budget and all(s.has_heads(l) for l in range(L))
    names = [f"blocks.{l}.{h}" for l in range(L) for _, h in kinds if r.available(f"blocks.{l}.{h}")]
    names += [f"blocks.{l}.attn.hook_z" for l in range(L)] if do_heads else []
    lc, cc = r.run(a, variant, names=names)
    lb, _ = r.run(b, variant)
    lc, lb = lc[0, -1], lb[0, -1]
    tgt = _one_token(ws, target) if target else int(lc.argmax())
    alt = _one_token(ws, versus) if versus else int(lb.argmax())
    metric = lambda lg: float(lg[tgt] - lg[alt])  # noqa: E731
    m_clean, m_corr = metric(lc), metric(lb)
    span = (m_clean - m_corr) or 1.0

    def run(name, fn):
        return round((metric(r.run(b, variant, hooks={name: fn})[0][0, -1]) - m_corr) / span, 4)

    rows = []
    for kind, h in kinds:
        grid = []
        for l in range(L):
            name = f"blocks.{l}.{h}"
            if name not in cc:
                grid.append([None] * (T if by_pos else 1))
                continue
            if by_pos:
                def at(t, name=name):
                    def fn(x, _):
                        x = x.clone()
                        x[:, t] = cc[name][:, t].to(x.dtype)
                        return x
                    return fn
                grid.append([run(name, at(t)) for t in range(T)])
            else:
                grid.append([run(name, lambda x, _, name=name: cc[name].to(x.dtype))])
        rows.append({"kind": kind, "grid": grid})
    heads = None
    if do_heads:
        heads = []
        for l in range(L):
            name = f"blocks.{l}.attn.hook_z"
            row = []
            for h in range(s.n_heads):
                def fn(x, _, h=h, name=name):
                    x = x.clone()
                    x[:, :, h] = cc[name][:, :, h].to(x.dtype)
                    return x
                row.append(run(name, fn))
            heads.append(row)
    return {"tokens_clean": [show(ws.tok.token_str(i)) for i in a], "tokens_corrupt": [show(ws.tok.token_str(i)) for i in b],
            "target": show(ws.tok.token_str(tgt)), "versus": show(ws.tok.token_str(alt)),
            "clean": m_clean, "corrupt": m_corr, "by_position": rows, "heads": heads, "per_position": by_pos}


@job_scope('ablation_scan')
@worker_job('ablation_scan')  # the whole call runs in the model worker when the workspace uses one
def ablation_scan(ws, prompts: list[tuple[list[int], int]], variant: str = "current", budget: int = 1500) -> dict:
    """Mean change in loss on (context -> next token) pairs when each head / MLP is zero-ablated."""
    s, r = ws.struct, ws.runner
    L = s.n_layers
    do_heads = all(s.has_heads(l) for l in range(L))
    units = L * (s.n_heads if do_heads else 0) + L
    prompts = prompts[:max(1, budget // max(1, units))]
    if do_heads and units * len(prompts) > budget * 2:
        do_heads = False

    def loss(hooks=None):
        tot = 0.0
        for ids, tgt in prompts:
            lg = r.run(ids[-(r.max_ctx or len(ids)):], variant, hooks=hooks)[0][0, -1]
            tot += float(F.cross_entropy(lg[None], torch.tensor([tgt])))
        return tot / max(1, len(prompts))

    base = loss()
    heads = []
    if do_heads:
        for l in range(L):
            row = []
            for h in range(s.n_heads):
                def fn(x, _, h=h):
                    x = x.clone()
                    x[:, :, h] = 0
                    return x
                row.append(round(loss({f"blocks.{l}.attn.hook_z": fn}) - base, 4))
            heads.append(row)
    mlps = [round(loss({f"blocks.{l}.hook_mlp_out": lambda x, _: torch.zeros_like(x)}) - base, 4)
            if r.available(f"blocks.{l}.hook_mlp_out") else None for l in range(L)]
    return {"base_loss": base, "heads": heads, "mlps": mlps, "n": len(prompts)}


# ------------------------------------------------------------------------------ calibration corpus
def corpus(ws, n_seq: int = 48, length: int | None = None) -> list[list[int]]:
    """Token sequences for statistics (drift, covariances).

    Uses <workspace>/corpus.txt if present, else text sampled once from the base model itself and saved."""
    if "corpus" in ws.keep:
        return ws.keep["corpus"]
    r = ws.runner
    big = ws.kind != "wsp" or (ws.struct.d_model * ws.struct.n_layers > 2000)
    length = length or min(r.max_ctx or 64, 40 if big else 64)
    if big:
        n_seq = min(n_seq, 16)
    path, saved = os.path.join(ws.dir, "corpus.txt"), os.path.join(ws.dir, "corpus.sampled.json")
    if os.path.exists(path):
        ids = ws.tok.encode(open(path).read())
        seqs = [ids[i:i + length] for i in range(0, max(1, len(ids) - length), length)][:max(n_seq, 64)]
    elif os.path.exists(saved):
        seqs = json.load(open(saved))
    else:
        g = torch.Generator().manual_seed(0)
        V = ws.struct.d_vocab
        if ws.kind == "wsp":
            toks = torch.randint(0, V, (n_seq, 1), generator=g)
            for _ in range(length - 1):
                lg = r.run(toks, "base")[0][:, -1]
                if not torch.isfinite(lg).all() or float(lg.abs().max()) == 0:
                    nxt = torch.randint(0, V, (n_seq, 1), generator=g)
                else:
                    nxt = torch.multinomial(lg.softmax(-1), 1, generator=g)
                toks = torch.cat([toks, nxt], 1)
            seqs = toks.tolist()
        else:
            starts = torch.randint(0, V, (n_seq,), generator=g).tolist()     # random first tokens, as for the small models
            seqs = []
            for i in range(n_seq):
                first = [starts[i]]
                new = r.generate(first, "base", None, max_new_tokens=length - 1, temperature=1.0, top_k=0, seed=i)
                seqs.append((first + new + [first[0]] * length)[:length])
        with open(saved, "w") as f:
            json.dump(seqs, f)
    ws.keep["corpus"] = seqs
    return seqs
