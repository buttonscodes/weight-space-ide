"""Patch pieces applied to a layer's output on every forward pass, without a dense delta: out += x @ delta.T.

Used by wsp.gguf_runner (each packed matrix's edit, added to the ggml product) and by wsp.runner (the float32 side
path of a bfloat16 or float16 run, added to an nn.Linear's output in PEFT's order). wsp.gguf_runner imports every name
back, so `wsp.gguf_runner.DeltaOverlay` is this module's class.
"""
from __future__ import annotations

import torch


def _axis(r):
    """One axis of a piece region as an index of the activations: a slice (a view, nothing built per call)
    or, for a descending range, an index tensor built once."""
    if isinstance(r, int):
        return slice(r, r + 1)
    if r.step > 0:
        return slice(r.start, r.stop, r.step)
    return torch.as_tensor(list(r), dtype=torch.long)


def _take(x, ix):
    return x[..., ix] if isinstance(ix, slice) else x.index_select(-1, ix)


def _put(out, ix, change):
    if isinstance(ix, slice):
        out[..., ix] += change
    else:
        out.index_add_(-1, ix, change)


def delta_matmul(x, piece, scale, out):
    """Accumulate x @ delta.T without materializing a full low-rank/sparse delta."""
    if not scale:
        return
    if piece.kind == 'sparse':
        rows, cols = piece.coords.T.long()
        values = x[..., cols] * (piece.values * scale)
        out.index_add_(-1, rows, values)
        return
    r, c = piece.region
    rows, cols = _axis(r), _axis(c)
    selected = _take(x, cols)
    if piece.kind == 'lowrank':
        change = (selected @ piece.B.T) @ piece.A.T
    else:
        nr, nc = (1 if isinstance(r, int) else len(r)), (1 if isinstance(c, int) else len(c))
        change = selected @ piece.values.reshape(nr, nc).T
    _put(out, rows, change * scale)


GATHER_MIN = 4      # blocks spanning one whole axis from which one gathered matrix beats one product per piece


def _index(a):
    """One axis of a region (an int, or a range of any step in either direction) as a long index tensor."""
    if isinstance(a, int):
        return torch.tensor([a], dtype=torch.long)
    return torch.as_tensor(list(a), dtype=torch.long)


def _spans(a, n):
    return isinstance(a, range) and a.step == 1 and a.start == 0 and len(a) == n


def _owned(t, source):
    """Bytes of `t` when it is a new allocation, not a view of the stored piece."""
    if source is not None and t.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
        return 0
    return t.numel() * t.element_size()


def piece_key(pieces):
    """Identity of a list of (piece, strength): which pieces, at which strengths. Two lists holding the same
    pieces (as `Workspace.rebuild` makes for every tensor after any toggle) share one key; appending to a
    list, a new piece or a new strength changes it. Valid while the pieces are held (ids stay unique)."""
    return tuple((id(p), s) for p, s in pieces)


class DeltaOverlay:
    """The run-time form of one matrix's active edit pieces, for out += x @ delta.T on every forward pass.

    Built when a forward pass first needs it, kept while the same pieces stay active, and dropped when they
    change; the stored patch is never changed. Block pieces spanning the whole output axis (one neuron's or
    head's write, a column slab) are gathered into one [out, k] matrix with one column index, and block
    pieces spanning the whole input axis (a neuron's read row, a head's query rows) into one [k, in] matrix
    with one row index, so each group costs one gather and one matmul however many pieces it holds. Every
    other piece is applied on its own, with its indices prepared here. Everything is held in the output's
    dtype and on its device (`dtype`, `device`); the bytes this adds on top of the stored pieces (the
    gathered copies, scaled sparse values, converted copies) are `nbytes`."""

    def __init__(self, pieces, out_features, in_features=None, dtype=torch.float32, device='cpu'):
        self.dtype, self.device = dtype, torch.device(device)
        by_col, by_row, rest = [], [], []
        for piece, scale in pieces:
            if not scale:
                continue
            if piece.kind == 'block' and len(piece.region) == 2:
                r, c = piece.region
                if _spans(r, out_features):
                    by_col.append((piece, scale))
                    continue
                if in_features is not None and _spans(c, in_features):
                    by_row.append((piece, scale))
                    continue
            rest.append((piece, scale))
        if len(by_col) < GATHER_MIN:
            by_col, rest = [], by_col + rest
        if len(by_row) < GATHER_MIN:
            by_row, rest = [], by_row + rest
        self.nbytes = 0
        self.cols, self.W = self._gather(by_col, out_features, axis=1)
        self.rows, self.Wr = self._gather(by_row, in_features, axis=0)
        self.rest = [self._prepare(p, s) for p, s in rest]
        self.n_gathered, self.n_rest = len(by_col) + len(by_row), len(rest)

    def _gather(self, group, n, axis):
        """[n, k] (axis 1: columns of full-height blocks) or [k, n] (axis 0: rows of full-width blocks)."""
        if not group:
            return None, None
        widths = [p.values.numel() // n for p, _ in group]
        k = sum(widths)
        W = torch.empty((n, k) if axis == 1 else (k, n), dtype=self.dtype, device=self.device)
        index, at = [], 0
        for (p, s), w in zip(group, widths):
            index.append(_index(p.region[axis]))
            values = p.values.reshape((n, w) if axis == 1 else (w, n))
            dst = W[:, at:at + w] if axis == 1 else W[at:at + w]
            dst.copy_(values)
            if s != 1:
                dst.mul_(s)
            at += w
        index = torch.cat(index).to(self.device)
        self.nbytes += W.numel() * W.element_size() + index.numel() * index.element_size()
        return index, W

    def _prepare(self, piece, scale):
        to = dict(dtype=self.dtype, device=self.device)
        if piece.kind == 'sparse':
            rows, cols = piece.coords.T.long().to(self.device)
            values = (piece.values * scale).to(**to)
            self.nbytes += values.numel() * values.element_size()
            return ('sparse', rows, cols, values)
        r, c = piece.region
        rows, cols = _axis(r), _axis(c)
        rows, cols = [ix if isinstance(ix, slice) else ix.to(self.device) for ix in (rows, cols)]
        nr, nc = (1 if isinstance(r, int) else len(r)), (1 if isinstance(c, int) else len(c))
        # The factors and values are held contiguous: the product's last bits depend on the operands' memory layout
        # (BLAS takes another path for a transposed view), and the same piece is held as transposed views right after
        # a commit but as contiguous tensors once read back from its patch file, so a reopened workspace would
        # otherwise compute other bits from the same values (on a packed base, ggml's activation quantization at
        # the next matrix can turn that into a visible change of the logits).
        if piece.kind == 'lowrank':
            A, B = piece.A.to(**to).contiguous(), piece.B.to(**to).contiguous()
            self.nbytes += _owned(A, piece.A) + _owned(B, piece.B)
            return ('lowrank', rows, cols, (A, B), scale)
        values = piece.values.reshape(nr, nc).to(**to).contiguous()
        self.nbytes += _owned(values, piece.values)
        return ('block', rows, cols, values, scale)

    def apply(self, x, out):
        """out += x @ delta.T for every piece, computed in this overlay's dtype and added in `out`'s."""
        x = x.to(dtype=self.dtype, device=self.device)
        cast = (lambda t: t) if out.dtype == self.dtype else (lambda t: t.to(out.dtype))
        if self.W is not None:
            out += cast(x.index_select(-1, self.cols) @ self.W.T)
        if self.Wr is not None:
            out.index_add_(-1, self.rows, cast(x @ self.Wr.T))
        for kind, rows, cols, *rest in self.rest:
            if kind == 'sparse':
                out.index_add_(-1, rows, cast(x[..., cols] * rest[0]))
                continue
            values, scale = rest
            selected = _take(x, cols)
            change = (selected @ values[1].T) @ values[0].T if kind == 'lowrank' else selected @ values.T
            _put(out, rows, cast(change * scale))
