"""Edits as compact deltas.

A patch does not store edited tensors. For each tensor it touches it stores *pieces*:

    block     a dense array on a rectangular region      (a neuron's row, a head's slice, a cell)
    lowrank   A @ B on a 2-D region                      (ROME, rank-one inserts, removing a direction)
    sparse    coordinates and values                      (scattered single weights)

so the size of an edit is the size of what was changed, never the size of the tensor it lives
in: a rank-one edit to a 16384 x 4096 matrix is 20,480 numbers, not 67 million.  Reading a
region of an edited tensor adds only the parts of the pieces that fall inside that region
(`add_to`), which is what lets the IDE show, test and export edited weights of models that
are never loaded whole.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .store import Region, norm_index, region_index, region_shape, region_size


@dataclass
class Piece:
    kind: str                        # block | lowrank | sparse
    region: Region                   # block / lowrank: where it applies; sparse: the whole tensor
    values: torch.Tensor | None = None     # block: region_shape(region); sparse: [nnz]
    A: torch.Tensor | None = None          # lowrank: [m, r]
    B: torch.Tensor | None = None          # lowrank: [r, n]
    coords: torch.Tensor | None = None     # sparse: [nnz, ndim] int64

    def numel(self) -> int:
        """Numbers stored."""
        if self.kind == "block":
            return self.values.numel()
        if self.kind == "lowrank":
            return self.A.numel() + self.B.numel()
        return self.values.numel() * (1 + self.coords.shape[1])

    def stats(self, rank: bool = True) -> dict:
        """`rank=False` skips the rank of a block (an SVD), which only a single-piece summary reports. A piece is
        never changed once made, so the answer is kept on it."""
        cache = self.__dict__.setdefault("_stats_kept", {})
        if rank not in cache:
            cache[rank] = self._stats(rank)
        return dict(cache[rank])

    def _stats(self, rank: bool) -> dict:
        if self.kind == "block":
            v = self.values
            return {"kind": "block", "changed": int((v != 0).sum()), "sq": float(v.double().pow(2).sum()),
                    "max_abs": float(v.abs().max()) if v.numel() else 0.0, "rank": _rank(v) if rank else None,
                    "stored": v.numel()}
        if self.kind == "lowrank":
            g = (self.A.double().T @ self.A.double()) * (self.B.double() @ self.B.double().T).T
            rows, cols = int((self.A != 0).any(1).sum()), int((self.B != 0).any(0).sum())     # entries that can be non-zero
            return {"kind": "lowrank", "changed": rows * cols, "sq": float(g.sum().clamp(min=0)),
                    "max_abs": float(self.A.abs().max() * self.B.abs().max()), "rank": int(self.A.shape[1]),
                    "stored": self.numel()}
        v = self.values
        return {"kind": "sparse", "changed": int((v != 0).sum()), "sq": float(v.double().pow(2).sum()),
                "max_abs": float(v.abs().max()) if v.numel() else 0.0, "rank": None, "stored": self.numel()}

    def dense(self) -> torch.Tensor:
        """The delta on its own region (block / lowrank)."""
        if self.kind == "block":
            return self.values
        if self.kind == "lowrank":
            return self.A @ self.B
        raise ValueError("a sparse piece has no dense region")

    def finite(self) -> bool:
        return all(bool(torch.isfinite(t).all()) for t in (self.values, self.A, self.B) if t is not None)

    def is_zero(self) -> bool:
        if self.kind == "lowrank":
            return not bool(self.A.any()) or not bool(self.B.any())
        return not bool(self.values.any())


def _rank(v: torch.Tensor) -> int | None:
    if v.dim() < 2 or v.numel() == 0 or not bool(v.any()):
        return None
    m = v.reshape(-1, v.shape[-1])
    if min(m.shape) > 512 or m.numel() > 4_000_000:
        return None
    s = torch.linalg.svdvals(m.double())
    return int((s > s[0] * 1e-5).sum())


def block(region: Region, values: torch.Tensor) -> Piece:
    values = values.detach().to(torch.float32)
    if tuple(values.shape) != region_shape(region):
        raise ValueError(f"block of shape {list(values.shape)} does not fit region {region_shape(region)}")
    return Piece("block", region, values=values.clone())


def lowrank(region: Region, A: torch.Tensor, B: torch.Tensor) -> Piece:
    A, B = A.detach().to(torch.float32), B.detach().to(torch.float32)
    if A.dim() == 1:
        A = A[:, None]
    if B.dim() == 1:
        B = B[None, :]
    if (A.shape[0], B.shape[1]) != region_shape(region) or A.shape[1] != B.shape[0]:
        raise ValueError(f"low-rank factors {list(A.shape)} x {list(B.shape)} do not fit region {region_shape(region)}")
    return Piece("lowrank", region, A=A.clone(), B=B.clone())


def _hits(p, q):
    """Positions (in the piece axis, in the request axis) of the indices two axis specs share."""
    pl = range(p, p + 1) if isinstance(p, int) else p
    ql = range(q, q + 1) if isinstance(q, int) else q
    if isinstance(pl, range) and isinstance(ql, range) and pl.step == 1 and ql.step == 1:
        lo, hi = max(pl.start, ql.start), min(pl.stop, ql.stop)
        if hi <= lo:
            return None
        return slice(lo - pl.start, hi - pl.start), slice(lo - ql.start, hi - ql.start)
    a, b = np.asarray(pl), np.asarray(ql)
    _, ia, ib = np.intersect1d(a, b, return_indices=True)
    if len(ia) == 0:
        return None
    return torch.from_numpy(ia), torch.from_numpy(ib)


def _ix(parts: list) -> tuple:
    """Per-axis slices / index vectors -> an index that takes their outer product."""
    n = len(parts)
    out = []
    for k, p in enumerate(parts):
        if isinstance(p, slice):            # mixing slices with index vectors would reorder axes: use vectors throughout
            p = torch.arange(p.start, p.stop)
        shape = [1] * n
        shape[k] = -1
        out.append(p.reshape(shape))
    return tuple(out)


def add_to(out: torch.Tensor, request: Region, piece: Piece, scale: float = 1.0) -> None:
    """out += scale * (piece restricted to the requested region).  `out` has shape region_shape(request)."""
    full = out.reshape([len(r) if isinstance(r, range) else 1 for r in request])
    if piece.kind == "sparse":
        c = piece.coords
        keep = torch.ones(c.shape[0], dtype=torch.bool)
        pos = []
        for d, q in enumerate(request):
            x = c[:, d]
            if isinstance(q, int):
                keep &= x == q
                pos.append(torch.zeros_like(x))
            else:
                rel = x - q.start
                keep &= (rel % q.step == 0) & (rel // q.step >= 0) & (rel // q.step < len(q))
                pos.append(rel // q.step)
        if bool(keep.any()):
            full.index_put_(tuple(p[keep] for p in pos), scale * piece.values[keep], accumulate=True)
        return
    hits = [_hits(p, q) for p, q in zip(piece.region, request)]
    if any(h is None for h in hits):
        return
    if piece.kind == "lowrank":
        kept = [d for d, r in enumerate(piece.region) if isinstance(r, range)]
        ra, ca = kept
        v = piece.A[hits[ra][0]] @ piece.B[:, hits[ca][0]]
        shape = [1] * len(request)
        shape[ra], shape[ca] = v.shape
        src = v.reshape(shape)
        src_ix = [slice(None)] * len(request)
    else:
        src = piece.values.reshape([len(r) if isinstance(r, range) else 1 for r in piece.region])
        src_ix = [h[0] for h in hits]
    dst_ix = [h[1] for h in hits]
    if all(isinstance(i, slice) for i in dst_ix) and all(isinstance(i, slice) for i in src_ix):
        full[tuple(dst_ix)] += scale * src[tuple(src_ix)]
    else:
        sub = src[_ix(src_ix)] if piece.kind == "block" else src
        ix = _ix(dst_ix)
        full[ix] = full[ix] + scale * sub


def intersects(piece: Piece, request: Region) -> bool:
    if piece.kind == "sparse":
        return True
    return all(_hits(p, q) is not None for p, q in zip(piece.region, request))


def changed_in(piece: Piece, request: Region) -> tuple[int, float]:
    """(number of changed weights, sum of squares) of a piece inside a region. Costs the piece, not the tensor."""
    if not intersects(piece, request):
        return 0, 0.0
    if piece.kind == "lowrank":
        kept = [d for d, r in enumerate(piece.region) if isinstance(r, range)]
        h = [_hits(p, q) for p, q in zip(piece.region, request)]
        A, B = piece.A[h[kept[0]][0]].double(), piece.B[:, h[kept[1]][0]].double()
        return int((A != 0).any(1).sum()) * int((B != 0).any(0).sum()), float(((A.T @ A) * (B @ B.T).T).sum().clamp(min=0))
    out = torch.zeros(region_shape(request)) if region_size(request) <= 4_000_000 else None
    if out is None:                                      # huge request: the piece's own totals are the answer or an upper bound
        s = piece.stats(rank=False)
        return s["changed"], s["sq"]
    add_to(out, request, piece)
    return int((out != 0).sum()), float(out.double().pow(2).sum())


def compact(pieces: list[Piece], shape: tuple[int, ...], dense_limit: int = 4_000_000) -> list[Piece]:
    """Tidy the pieces a script produced for one tensor.

    Small tensors: merge everything into one block or sparse piece. Large tensors: keep pieces as they
    are (they are already the size of the edit), dropping the ones that are exactly zero."""
    pieces = [p for p in pieces if not p.is_zero()]
    if not pieces:
        return []
    n = int(np.prod(shape)) if shape else 1
    if n <= dense_limit and (len(pieces) > 1 or pieces[0].kind == "block"):
        whole = norm_index((), shape)
        d = torch.zeros(shape)
        for p in pieces:
            add_to(d, whole, p)
        nz = d.nonzero()
        if nz.shape[0] == 0:
            return []
        if len(pieces) == 1 and pieces[0].kind == "lowrank":
            return [trim(pieces[0])]
        lo, hi = nz.min(0).values, nz.max(0).values
        region = tuple(range(int(a), int(b) + 1) for a, b in zip(lo, hi))
        if nz.shape[0] * (len(shape) + 1) < region_size(region):       # scattered: coordinates are cheaper than the box
            merged = Piece("sparse", whole, values=d[tuple(nz.T)].clone(), coords=nz.clone())
        else:
            merged = block(region, d[region_index(region)])
        if merged.numel() <= sum(p.numel() for p in pieces):
            return [merged]                                             # otherwise the pieces (e.g. low-rank ones) are the smaller form
    return [trim(p) for p in pieces]


def trim(piece: Piece) -> Piece:
    """Shrink a piece to the bounding box of what it actually changes (one pass over what it stores)."""
    if piece.kind == "lowrank":                       # rows of A / columns of B that are all zero change nothing
        kept = [d for d, r in enumerate(piece.region) if isinstance(r, range)]
        rn, cn = (piece.A != 0).any(1).nonzero().flatten(), (piece.B != 0).any(0).nonzero().flatten()
        if rn.numel() == 0 or cn.numel() == 0:
            return piece
        r0, r1, c0, c1 = int(rn[0]), int(rn[-1]) + 1, int(cn[0]), int(cn[-1]) + 1
        if (r0, r1, c0, c1) == (0, piece.A.shape[0], 0, piece.B.shape[1]):
            return piece
        region = list(piece.region)
        region[kept[0]], region[kept[1]] = region[kept[0]][r0:r1], region[kept[1]][c0:c1]
        return Piece("lowrank", tuple(region), A=piece.A[r0:r1].clone(), B=piece.B[:, c0:c1].clone())
    if piece.kind != "block" or piece.values.dim() == 0:
        return piece
    v = piece.values
    kept = [d for d, r in enumerate(piece.region) if isinstance(r, range)]
    region = list(piece.region)
    sl = []
    for k, d in enumerate(kept):
        other = tuple(i for i in range(v.dim()) if i != k)
        hit = (v != 0).any(dim=other) if other else (v != 0)
        nzi = hit.nonzero().flatten()
        if nzi.numel() == 0:
            return piece
        a, b = int(nzi[0]), int(nzi[-1]) + 1
        sl.append(slice(a, b))
        region[d] = region[d][a:b]
    if all(s.start == 0 and s.stop == n for s, n in zip(sl, v.shape)):
        return piece
    return Piece("block", tuple(region), values=v[tuple(sl)].clone())


# -------------------------------------------------------------------------------------- (de)serialisation
def _spec(region: Region) -> list:
    return [r if isinstance(r, int) else [r.start, r.stop, r.step] for r in region]


def _region(spec: list) -> Region:
    return tuple(r if isinstance(r, int) else range(*r) for r in spec)


def pack(deltas: dict[str, list[Piece]]) -> tuple[list[dict], dict[str, torch.Tensor]]:
    """-> (JSON-able index, tensors to save)"""
    index, tensors = [], {}
    k = 0
    for name, pieces in deltas.items():
        for p in pieces:
            e = {"tensor": name, "kind": p.kind, "region": _spec(p.region), "key": f"p{k}"}
            if p.kind == "block":
                tensors[f"p{k}.v"] = p.values.contiguous()
            elif p.kind == "lowrank":
                tensors[f"p{k}.A"], tensors[f"p{k}.B"] = p.A.contiguous(), p.B.contiguous()
            else:
                tensors[f"p{k}.v"], tensors[f"p{k}.c"] = p.values.contiguous(), p.coords.contiguous()
            index.append(e)
            k += 1
    return index, tensors


def unpack(index: list[dict], tensors: dict[str, torch.Tensor]) -> dict[str, list[Piece]]:
    out: dict[str, list[Piece]] = {}
    for e in index:
        k, region = e["key"], _region(e["region"])
        if e["kind"] == "block":
            p = Piece("block", region, values=tensors[f"{k}.v"])
        elif e["kind"] == "lowrank":
            p = Piece("lowrank", region, A=tensors[f"{k}.A"], B=tensors[f"{k}.B"])
        else:
            p = Piece("sparse", region, values=tensors[f"{k}.v"], coords=tensors[f"{k}.c"])
        out.setdefault(e["tensor"], []).append(p)
    return out


def summary(deltas: dict[str, list[Piece]]) -> list[dict]:
    """Per tensor: how much changed, how large, and in what form (what the UI and the programmer are told)."""
    out = []
    for name, pieces in deltas.items():
        st = [p.stats(rank=len(pieces) == 1) for p in pieces]     # the rank is reported for a single piece only
        ranks = [s["rank"] for s in st if s["rank"] is not None]
        out.append({"param": name, "changed": sum(s["changed"] for s in st), "norm": sum(s["sq"] for s in st) ** 0.5,
                    "max_abs": max(s["max_abs"] for s in st), "rank": ranks[0] if len(ranks) == 1 and len(st) == 1 else None,
                    "kinds": sorted({s["kind"] for s in st}), "stored": sum(s["stored"] for s in st), "pieces": len(st)})
    return out
