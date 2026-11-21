"""Before and after: the base weights, the current (edited) weights and their difference, for one tensor or slice.

Base and current are drawn on one colour scale so the picture shows what moved; the difference gets its own
scale, since an edit is usually much smaller than the weights it changes. On a quantized checkpoint "base" is
the reconstructed logical values of the displayed region only (what the IDE reads and edits), never the packed
integers, and nothing outside the region is decoded.

Reads go through the workspace (`ws.read` with variant base / delta), so they cost what one matrix view of the
same region costs: one store read for the base and the patch pieces for the difference. Large 2-D regions are
streamed in row slabs, as `analysis.matrix_view` does.
"""
from __future__ import annotations

import torch

from . import analysis as an
from .store import norm_index, region_shape
from .workspace import Ref, STREAM_ELEMS


class _Acc:
    """Exact norm / min / max over everything read, and an evenly spaced sample for the colour scale."""

    def __init__(self):
        self.sq, self.lo, self.hi, self.n = 0.0, float("inf"), float("-inf"), 0
        self.band = None                      # (min, max) of the band whose parts are being added
        self.sample: list[torch.Tensor] = []

    def add(self, t: torch.Tensor, at: int = 0, every: int | None = None, part: bool = False):
        """`at` / `every`: t is a part of a larger read, starting at flat position `at`, sampled every `every` values
        (the sample is then the one the whole read would give). `part`: min and max are kept for the whole band until
        `close`, so that a NaN anywhere in the band counts as it does on one read of the band."""
        if t.numel() == 0:
            return
        t = t.float()
        self.sq += float(t.double().pow(2).sum())
        if part:
            lo, hi = t.min(), t.max()
            self.band = (lo, hi) if self.band is None else (torch.minimum(self.band[0], lo), torch.maximum(self.band[1], hi))
        else:
            self.lo, self.hi = min(self.lo, float(t.min())), max(self.hi, float(t.max()))
        self.n += t.numel()
        every = every or max(1, t.numel() // 20_000)
        self.sample.append(t.flatten()[(-at) % every::every])

    def close(self):
        """The end of a band added in parts: its min and max count as one read's."""
        if self.band is not None:
            self.lo, self.hi = min(self.lo, float(self.band[0])), max(self.hi, float(self.band[1]))
            self.band = None

    def result(self) -> dict:
        if not self.n:
            return {"n": 0}
        return {"n": self.n, "norm": self.sq ** 0.5, "min": self.lo, "max": self.hi}

    def scale(self, nonzero: bool = False) -> float:
        if not self.sample:
            return 1.0
        s = torch.cat(self.sample)
        if nonzero:
            s = s[s != 0]
        return an.robust_scale(s) if s.numel() else 1.0


def matrix_compare(ws, addr: str, r0: int = 0, r1: int | None = None, c0: int = 0, c1: int | None = None,
                   max_h: int = 256, max_w: int = 384, pool: str = "mean") -> dict:
    name, idx = an._single(ws, addr)
    shape = ws.shape(name)
    region = norm_index(idx, shape)
    kshape = region_shape(region)
    R = 1
    for s in kshape[:-1]:
        R *= s
    C = kshape[-1] if kshape else 1
    r1, c1 = min(r1 or R, R), min(c1 or C, C)
    fr, fc = max(1, -(-(r1 - r0) // max_h)), max(1, -(-(c1 - c0) // max_w))
    acc = {"base": _Acc(), "current": _Acc(), "delta": _Acc()}
    grids = {"base": [], "current": [], "delta": []}
    changed = 0

    def take(base: torch.Tensor, delta: torch.Tensor, bands: dict | None = None, at: int = 0, every: int | None = None):
        """One read. With `bands`, the read is a part of a pooled band (an.BandPool per grid), starting at `at`."""
        nonlocal changed
        cur = base + delta
        for key, t in (("base", base), ("current", cur), ("delta", delta)):
            acc[key].add(t, at, every, part=bands is not None)
            if bands is None:
                grids[key].append(an._pool(t, fr, fc, "absmax" if key == "delta" else pool))
            else:
                bands[key].add(t)
        changed += int(delta.ne(0).sum())

    kept = [d for d, r in enumerate(region) if isinstance(r, range)]
    if len(kshape) != 2 or (r1 - r0) * (c1 - c0) <= STREAM_ELEMS:
        if len(kshape) == 2:
            sub = list(region)
            sub[kept[0]] = region[kept[0]][r0:r1]
            sub[kept[1]] = region[kept[1]][c0:c1]
            sub = tuple(sub)
            take(an.as_matrix(ws.read(name, sub, "base")), an.as_matrix(ws.read(name, sub, "delta")))
        else:
            b, d = an.as_matrix(Ref(ws, name, region).whole("base")), an.as_matrix(ws.read(name, region, "delta"))
            take(b[r0:r1, c0:c1], d[r0:r1, c0:c1])
    else:                                                     # stream the rows in slabs aligned to the pooling factor
        sub = list(region)
        sub[kept[0]] = region[kept[0]][r0:r1]
        sub[kept[1]] = region[kept[1]][c0:c1]
        cols = len(sub[kept[1]])
        step = max(fr, (STREAM_ELEMS // 2 // max(1, cols)) // fr * fr)
        rows = sub[kept[0]]
        per = max(1, STREAM_ELEMS // 2 // max(1, cols))       # rows per read when one band of fr rows is above that
        for a in range(0, len(rows), step):
            piece = list(sub)
            piece[kept[0]] = rows[a:a + step]
            if fr * cols > STREAM_ELEMS // 2:                 # a small max_h on a large matrix: pool the band as it is read
                band = rows[a:a + step]
                n = len(band) * cols
                bands = {key: an.BandPool(cols, fc, "absmax" if key == "delta" else pool, n) for key in grids}
                for b in range(0, len(band), per):
                    piece[kept[0]] = band[b:b + per]
                    p = tuple(piece)
                    take(ws.read(name, p, "base"), ws.read(name, p, "delta"), bands, b * cols, max(1, n // 20_000))
                for key in grids:
                    grids[key].append(bands[key].result())
                    acc[key].close()
                continue
            piece = tuple(piece)
            take(ws.read(name, piece, "base"), ws.read(name, piece, "delta"))
    pooled = {k: torch.cat(v, 0) for k, v in grids.items()}
    scale = max(acc["base"].scale(), acc["current"].scale())
    rrole, crole = an.view_roles(ws, name, idx)
    roff, coff = an.axis_offset(name, idx, shape)
    stats = {k: acc[k].result() for k in ("base", "current")}
    d = acc["delta"].result()
    stats["delta"] = {"n": d.get("n", 0), "norm": d.get("norm", 0.0), "changed": changed,
                      "max_abs": max(abs(d.get("min", 0.0)), abs(d.get("max", 0.0))) if d.get("n") else 0.0}
    out = {"addr": addr, "param": name, "shape": [R, C], "region": [r0, r1, c0, c1], "grid": list(pooled["base"].shape),
           "factor": [fr, fc], "pool": pool, "delta_pool": "absmax",
           "base": an.b64(pooled["base"].numpy()), "current": an.b64(pooled["current"].numpy()),
           "delta": an.b64(pooled["delta"].numpy()),
           "scale": scale, "delta_scale": acc["delta"].scale(nonzero=True), "stats": stats,
           "row_role": rrole, "col_role": crole, "row_offset": roff, "col_offset": coff,
           "dtype": ws.store.dtype(name), "edited": changed > 0}
    if fr == 1 and fc == 1 and (r1 - r0) <= 512 and len(kshape) <= 2:
        out["row_labels"] = an.labels(ws, rrole, roff + r0, r1 - r0)
    if fr == 1 and fc == 1 and (c1 - c0) <= 512:
        out["col_labels"] = an.labels(ws, crole, coff + c0, c1 - c0)
    return out
