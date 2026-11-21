"""Edits that need more than arithmetic on weights.

rome          rank-one fact insertion (Meng et al. 2022) for any model the runner can hook;
              the result is a low-rank piece: two vectors, however large the matrix
checkout /    the file route: tensors as .npy files for a harness that prefers to compute its
commit_files  edits with its own tools; whatever it changes comes back as an ordinary patch
import_delta  another checkpoint with the same tensor names as a patch (a task vector, Ilharco et al. 2023)
"""
from __future__ import annotations

from .runtime_events import edit_preflight, job_scope

import os

import numpy as np
import torch
import torch.nn.functional as F

from . import addressing as A
from . import delta as D
from . import workspace as W
from .analysis import corpus
from .store import Store, norm_index


def _find(ids: list[int], sub: list[int]) -> int:
    for i in range(len(ids) - len(sub), -1, -1):
        if ids[i:i + len(sub)] == sub:
            return i + len(sub) - 1
    raise ValueError("the subject does not appear in the prompt (as tokens)")


def _keys(ws, layer: int) -> torch.Tensor:
    """MLP hidden activations at `layer` over the calibration corpus: [N, d_mlp] (cached per weights version)."""
    key = f"keys{layer}"
    if key not in ws.calib:
        name = f"blocks.{layer}.mlp.hook_post"
        seqs = corpus(ws)
        L = min(len(q) for q in seqs)
        toks = torch.tensor([q[:L] for q in seqs])
        parts = []
        for i in range(0, len(toks), 8):
            _, cache = ws.runner.run(toks[i:i + 8], "current", names=[name])
            parts.append(cache[name].reshape(-1, cache[name].shape[-1]))
        ws.calib[key] = torch.cat(parts).double()
    return ws.calib[key]


def whitened_key(ws, layer: int, k: torch.Tensor) -> torch.Tensor:
    """u = C^-1 k / (k^T C^-1 k) with C = E[k k^T] + ridge, over ordinary text.

    For wide MLPs C (d_mlp x d_mlp) is never formed: with N sampled keys K, the Woodbury identity gives
    (lam I + K^T K / N)^-1 k from an N x N solve."""
    K = _keys(ws, layer)
    N, m = K.shape
    k = k.double()
    lam = 1e-4 * float(K.pow(2).sum()) / (N * m) or 1e-8
    if m <= 4096:
        C = K.T @ K / N + lam * torch.eye(m, dtype=torch.double)
        u = torch.linalg.solve(C, k)
    else:
        G = K @ K.T + N * lam * torch.eye(N, dtype=torch.double)
        u = (k - K.T @ torch.linalg.solve(G, K @ k)) / lam
    return (u / (u @ k)).float()


def rome(ws, prompt: str, subject: str, target: str, layer: int | None = None, at: str = "auto", steps: int = 60,
         lr: float = 0.3, reg: float = 0.02, clamp: float = 8.0) -> dict:
    """Make `prompt` continue with `target` by a rank-one change to one MLP's output matrix.

    The MLP is treated as a linear associative memory (keys = hidden activations, values = what
    it writes).  k* is the hidden activation at the edited position; v* is found by gradient
    descent on the MLP *output* at that position so that the model predicts the target; then,
    with C = E[k k^T] over ordinary text,

        W_out += outer(u, v* - v)        u = C^-1 k* / (k*^T C^-1 k*)

    which changes the output for k* to v* while disturbing other keys as little as possible in
    the least-squares sense.  Only activations are differentiated, never the weights, and the
    edit is stored as the two vectors.

    `at` is the position whose key is edited: "subject" (the subject's last token, as in the
    paper, where large models keep the fact) or "last" (the predicting position, where very
    small models tend to do the lookup after attention has moved the subject there).  With
    layer=None / at="auto" every combination is tried (for models of up to 12 layers) and the
    one that reaches the target with the least drift on other text is returned; activation
    patching shows which one a model uses.
    """
    from .testsuite import drift
    from .worker_jobs import uses_worker
    if uses_worker(ws):
        # ROME differentiates through forward hooks; refused before any model load.
        raise RuntimeError('Packed GGUF parameter gradients are unsupported; use logical delta edits')
    s, r = ws.struct, ws.runner
    ids = ws.encode(prompt)
    tgt = ws.tok.encode_token(target)
    if not tgt:
        raise ValueError("empty target")
    sub_pos = _find(ids, ws.tok.encode_token(subject))
    full = ids + tgt[:-1]
    n_p = len(ids)
    usable = [l for l in range(s.n_layers) if s.has_neurons(l) and r.available(f"blocks.{l}.hook_mlp_out")
              and r.available(f"blocks.{l}.mlp.hook_post")]
    if not usable:
        raise ValueError("this model has no MLP the runner can hook, so ROME cannot be applied")
    if layer is None and len(usable) > 12:
        raise ValueError(f"this model has {len(usable)} MLP layers; give `layer` (use trace / activation patching to find "
                         f"where the fact is computed) instead of trying them all")
    layers = [layer] if layer is not None else usable
    if any(l not in usable for l in layers):
        raise ValueError(f"layer {layer} has no MLP that can be edited")
    places = {"subject": [("subject", sub_pos)], "last": [("last", n_p - 1)],
              "auto": [("subject", sub_pos), ("last", n_p - 1)]}[at]
    tgt_t = torch.tensor(tgt)
    best = None
    with r.lock:                        # gradients flow through the weights in memory: nobody may switch them meanwhile
        best = _rome_search(ws, r, s, layers, places, full, n_p, tgt, tgt_t, steps, lr, reg, clamp, drift)
    if best is None:
        raise ValueError("no layer / position to edit")
    return best[1]


def _rome_search(ws, r, s, layers, places, full, n_p, tgt, tgt_t, steps, lr, reg, clamp, drift):
    best = None
    for l, (where, pos) in [(l, pl) for l in layers for pl in places]:
        if where == "subject" and l == s.n_layers - 1 and pos != n_p - 1:
            continue                    # the last layer's output at an earlier position cannot reach the prediction
        hook_out, hook_post = f"blocks.{l}.hook_mlp_out", f"blocks.{l}.mlp.hook_post"
        base_logits, cache = r.run(full, "current", names=[hook_out, hook_post])
        base_logits = base_logits[0]
        k_star = cache[hook_post][0, pos]
        v0 = cache[hook_out][0, pos]
        delta = torch.zeros(s.d_model, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr * max(1.0, float(v0.norm()) / 4))
        limit = clamp * float(v0.norm()) if float(v0.norm()) > 0 else clamp

        def add(x, _):
            x = x.clone()
            x[:, pos] = x[:, pos] + delta.to(x.dtype)
            return x

        nll = torch.tensor(0.0)
        for _ in range(steps):
            lg = r.run(full, "current", hooks={hook_out: add}, grad=True)[0][0]
            nll = F.cross_entropy(lg[n_p - 1:n_p - 1 + len(tgt)], tgt_t)
            loss = nll + reg * (delta.norm() / (v0.norm() + 1e-6)) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                if float(delta.norm()) > limit:
                    delta.mul_(limit / delta.norm())
            if float(nll.detach()) < 0.02:
                break
        d = delta.detach()
        u = whitened_key(ws, l, k_star)
        ref = ws.sem.mlp(l)["W_out"]                     # canonical [neuron, d_model]
        piece = ref.lowrank_piece(u, d)
        extra = {ref.name: [piece]}
        lg = r.run(full, "current", extra)[0][0]
        p_after = float(lg[n_p - 1].softmax(-1)[tgt[0]])
        ok = bool((lg[n_p - 1:n_p - 1 + len(tgt)].argmax(-1) == tgt_t).all())
        dr = drift(ws, "current", extra)
        res = {"layer": l, "success": ok, "p_before": float(base_logits[n_p - 1].softmax(-1)[tgt[0]]),
               "p_after": p_after, "drift": dr, "delta_norm": float(u.norm() * d.norm()), "deltas": extra,
               "at": where, "pos": pos, "nll": float(nll.detach())}
        score = (ok, -dr["kl"] if ok else p_after)
        if best is None or score > best[0]:
            best = (score, res)
    return best


# ------------------------------------------------------------------------------ file route
PARTIAL = ".partial"        # a checked-out file still being written: <name>.npy.partial, renamed when complete


def checkout(ws, selection: list[str], directory: str, limit: int = 2_000_000_000) -> dict:
    """Write the tensors a selection touches as <directory>/<name>.npy (whole tensors, float32) and an index.

    Each file is written slab by slab (a quantized store refuses one read of a large tensor) with the bytes np.save
    of the whole tensor would write, and appears under its name only when complete. Every file is the current
    weights of one moment: the slabs all take the same patches, even if another edit is committed meanwhile."""
    os.makedirs(directory, exist_ok=True)
    for fn in os.listdir(directory):              # left by a checkout that was killed part-way
        if fn.endswith(".npy" + PARTIAL):
            os.remove(os.path.join(directory, fn))
    names = sorted({name for a in selection for name, _ in ws.resolve(a) if ws.editable(name)})
    size = sum(ws.store.numel(n) * 4 for n in names)
    if size > limit:
        raise ValueError(f"that selection is {size / 1e9:.1f} GB as float32 files; check out a narrower selection")
    lines = ["# Weights checked out for editing", "",
             "Each file is one whole tensor (float32). Edit them with any tool, then call `commit_files`.",
             "Only the entries inside the task scope may change.", "", "| file | shape | selected slices |", "|---|---|---|"]
    pieces = ws.piece_index()
    for name in names:
        _write_npy(os.path.join(directory, name + ".npy"), ws.shape(name),
                   (t.numpy() for _, _, t in _current_slabs(ws, name, pieces)))
        sl = sorted({A.slice_str(x) for a in selection for x in ws.resolve(a) if x[0] == name})
        lines.append(f"| {name}.npy | {list(ws.shape(name))} | {', '.join(sl[:6])}{' …' if len(sl) > 6 else ''} |")
    with open(os.path.join(directory, "INDEX.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return {"dir": directory, "files": [n + ".npy" for n in names]}


def _write_npy(path: str, shape: tuple, slabs) -> None:
    """A float32 .npy of `shape` from its slabs (C order, first axis), with np.save's bytes (format 1.0 header).

    Written to path + PARTIAL with ordinary writes (a full disk raises an error, where a memory-mapped write to a
    pre-sized file could kill the process), flushed and synced, then renamed to `path`. On any failure the partial
    file is removed, so the folder never holds a whole-looking file whose rows were not written."""
    tmp = path + PARTIAL
    try:
        with open(tmp, "wb") as f:
            np.lib.format.write_array_header_1_0(f, {"descr": np.lib.format.dtype_to_descr(np.dtype("<f4")),
                                                     "fortran_order": False, "shape": tuple(int(n) for n in shape)})
            for a in slabs:
                f.write(np.ascontiguousarray(a, dtype="<f4").reshape(-1).data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _current_slabs(ws, name: str, pieces: dict):
    """The current weights of one whole tensor as (rows, region, values) slabs of at most STREAM_ELEMS values: the
    base slab with the patch pieces of `pieces` (a Workspace.piece_index()) laid on, as Workspace.read does."""
    whole = norm_index((), ws.shape(name))
    for rows, t in ws.slabs(name, whole, "base", max_elems=W.STREAM_ELEMS):
        region = (whole[0][rows.start:rows.stop],) + whole[1:] if whole else whole
        for piece, scale in pieces.get(name, ()):
            D.add_to(t, region, piece, scale)
        yield rows, region, t


@job_scope('edit')
def read_files(ws, directory: str) -> dict[str, list[D.Piece]]:
    """Deltas implied by the .npy files in `directory` relative to the current weights, trimmed to what changed.

    Only complete files count (<name>.npy): a <name>.npy.partial left by a killed checkout is ignored. Every file is
    compared with the current weights of one moment (the patches when this starts)."""
    deltas: dict[str, list[D.Piece]] = {}
    names=[fn[:-4] for fn in os.listdir(directory) if fn.endswith(".npy")]
    if any(name not in ws.store for name in names):raise ValueError("Unknown tensor in edit files")
    edit_preflight(ws,sum(ws.store.numel(name) for name in names),scratch=max((ws.store.numel(name)*12 for name in names),default=0))
    current = ws.piece_index()
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".npy"):
            continue
        name = fn[:-4]
        if name not in ws.store:
            raise ValueError(f"{fn} does not correspond to a tensor")
        mapped = np.load(os.path.join(directory, fn), allow_pickle=False, mmap_mode="r")
        if tuple(mapped.shape) != tuple(ws.shape(name)):
            raise ValueError(f"{fn} has shape {list(mapped.shape)}, expected {list(ws.shape(name))}")
        if mapped.dtype.kind not in "fiu":raise ValueError(f"{fn} must contain numeric weights")
        # Shape/dtype are checked from the mapped header before materialization (one slab at a time).
        pieces = _slab_deltas(ws, name, current, lambda region, rows, f=mapped: torch.from_numpy(
            (f[rows.start:rows.stop] if f.ndim else f).astype(np.float32, copy=True)))
        if pieces:
            deltas[name] = D.compact(pieces, ws.shape(name))
    return deltas


def _slab_deltas(ws, name: str, current: dict, other) -> list[D.Piece]:
    """(other - current) for one whole tensor as block pieces: one per slab of at most STREAM_ELEMS values that differs.

    `current` is a Workspace.piece_index(): every slab is compared with the same patches. `other(region, rows)`
    gives the other values of one slab (its region of the tensor, its range of the first axis). A quantized store
    refuses one read of a large tensor. A tensor of up to STREAM_ELEMS values is a single slab, so it gives the same
    piece as the whole-tensor difference did; a larger one gives a piece per changed slab, with the same values."""
    out = []
    for rows, region, cur in _current_slabs(ws, name, current):
        d = other(region, rows) - cur
        if bool((d != 0).any()):
            out.append(D.block(region, d))
    return out


@job_scope('edit')
def import_delta(ws, checkpoint: str) -> dict[str, list[D.Piece]]:
    """current -> another checkpoint (safetensors, same tensor names) as deltas: a task vector.

    Tensors are compared one at a time; only those that differ are kept, trimmed to what differs."""
    from .store import checkpoint_files
    other = Store(checkpoint_files(checkpoint) if os.path.isdir(checkpoint) else [checkpoint])
    out: dict[str, list[D.Piece]] = {}
    names=[name for name in other.names() if name in ws.store and other.shape(name)==ws.shape(name) and ws.editable(name)]
    edit_preflight(ws,sum(ws.store.numel(name) for name in names),scratch=max((ws.store.numel(name)*12 for name in names),default=0))
    current = ws.piece_index()
    for name in names:
        if name in ws.store and other.shape(name) == ws.shape(name) and ws.editable(name):
            pieces = _slab_deltas(ws, name, current, lambda region, rows, n=name: other.read(n, region))
            if pieces:
                out[name] = D.compact(pieces, ws.shape(name))
    if not out:
        raise ValueError("the checkpoint has no tensors that differ from the current weights")
    return out
