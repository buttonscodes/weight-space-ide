"""The observed stage of the atlas: run the model on probe text and check the weights-only map against it.

Needs a model that can be run.  One pass through the runner's standard hook points records

    per neuron   mean, spread, the sign of its strong activations (`obs_peak`), and the positions where it fires
                 hardest in each direction
    per head     where it attends: previous token, first token, earlier copies of the current token, the token
                 after an earlier copy (induction), each against its chance level        (where patterns are exposed)
    the model    next-token loss on the probe text, and a direction on which the final normalised stream is
                 constant (what a token shift uses in place of a bias)

and then tests the claims of the weights stage: firing sign, token triggers, output concepts, same-token heads.

`obs_peak` is the sign to read a neuron's direction from.  The sign of the MEAN activation is not: a closed gate
still leaks a small value at most positions, often of the opposite sign, and that leak can decide the mean.

Probe text: <workspace>/corpus.txt if present, else the text bundled here (about five thousand tokens of prose,
dialogue, arithmetic, code and ten languages; small, so top contexts are examples, not a census), else, for
models with their own toy vocabulary, text sampled from the model.

Where the model runs.  In this process, windows go through the runner in batches of up to 32 and each batch is
reduced here.  A model that runs in the supervised model worker (a GGUF model) is run one window per job
(`observe_window`, see wsp/worker_jobs.py): the worker runs the window once, reduces every MLP layer's activations
inside the forward pass as it goes (nothing of a layer is kept once it is reduced, and no logits or captures are
sent), and returns only the window's reductions; the stage merges them here.  Both paths reduce with the same
functions (`mlp_sums`, `pattern_sums`, `pattern_context`, `loss_sum`, `final_stream`) and merge with the same ones
(`merge_mlp`, `merge_heads`), so the worker path gives what this process gives when it runs one window at a time.

Files are written only after every window has been run, each one atomically (a temporary name, then a rename); the
atlas records the stage only once all of them are written, and readers ignore observed files the record does not
list (see `Atlas.layer` / `Atlas.json`).
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np
import torch

from .build import ROLE, tau  # noqa: F401
from .tokens import token_table

K_TOP, K_BOT = 6, 3
RIDGE = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
LOSS_ROWS = 16          # in the model worker a window's loss is taken this many positions at a time (bounded float64 copies)
MLP_SUMS = ("s1", "s2", "np", "pm", "nm")
MLP_KEYS = MLP_SUMS + ("top_v", "top_i", "bot_v", "bot_i")
HEAD_KEYS = ("dist", "prev", "self", "first", "entropy", "same", "induct")
OBSERVED_FILES = ("validation.json", "heads_observed.json", "bias.json")          # read only when the stage is recorded
# The bias-direction fit keeps the final rows of at most this many positions, so the server's memory for it does not
# grow with the probe text: about 16 bytes per row and dimension while fitting (rows, float64 copies) plus 32 d_model^2
# (the normal matrix and its solve).  At d_model 5,120 that is 5,734 positions (the bundled text is about 5,000).
FIT_ROW_BYTES = 448 * 2 ** 20


def _probe_text(ws, corpus: str | None):
    """(text or None, where it came from): None means the text is sampled from the model itself."""
    from ..tokenizer import BPETokenizer
    local = os.path.join(ws.dir, "corpus.txt")
    if corpus:
        return open(corpus, encoding="utf-8").read(), corpus
    if os.path.exists(local):
        return open(local, encoding="utf-8").read(), "corpus.txt in the workspace"
    if isinstance(ws.tok, BPETokenizer):
        return open(os.path.join(os.path.dirname(__file__), "probe.txt"), encoding="utf-8").read(), "the bundled probe text"
    return None, "text sampled from the model itself"


def probe_windows(ws, corpus: str | None = None, tokens: int | None = None) -> tuple[torch.Tensor, str]:
    """Token windows [W, T] to run, and a description of where they came from."""
    r = ws.runner
    T = int(min(r.max_ctx or 128, 128))
    text, src = _probe_text(ws, corpus)
    if text is None:
        from .. import analysis
        seqs = analysis.corpus(ws, n_seq=64)
        T = min(len(s) for s in seqs)
        ids = torch.tensor([s[:T] for s in seqs])
    else:
        flat = ws.tok.encode(text)
        W = len(flat) // T
        if W < 2:
            raise ValueError(f"the probe text is {len(flat)} tokens, less than two windows of {T}")
        ids = torch.tensor(flat[:W * T]).view(W, T)
    if tokens:
        ids = ids[:max(2, int(tokens) // T)]
    return ids, src


def probe_plan(ws) -> dict | None:
    """What the stage would run, worked out without loading a model: {"tokens", "windows", "window", "source"}.  The
    window is 128 tokens unless a loaded runner (or the model) allows fewer.  None when the text would be sampled from
    the model (known only once it runs) or cannot be read."""
    try:
        text, src = _probe_text(ws, None)
        if text is None or ws.tok is None:
            return None
        ctx = getattr(getattr(ws, "_runner", None), "max_ctx", None) or getattr(ws.struct, "n_ctx", None) or 128
        T = int(min(ctx, 128))
        W = len(ws.tok.encode(text)) // T
        return {"tokens": W * T, "windows": W, "window": T, "source": src} if W >= 2 else None
    except Exception:                                                              # the status must always answer
        return None


def peak_sign(top_val: np.ndarray, bot_val: np.ndarray) -> np.ndarray:
    """+1 / -1 / 0: mean of the three largest activations against the three most negative; a sign when one side
    is more than twice the other."""
    tp = top_val[:, :3].astype(np.float32).clip(min=0).mean(1)
    tn = (-bot_val[:, :3].astype(np.float32)).clip(min=0).mean(1)
    return np.where(tp > 2 * tn, 1, np.where(tn > 2 * tp, -1, 0)).astype(np.int8)


class _Top:
    """Running top-k / bottom-k activations per neuron with the positions they occurred at."""

    def __init__(self, n: int):
        self.pv, self.pi = torch.full((K_TOP, n), -float("inf")), torch.zeros(K_TOP, n, dtype=torch.long)
        self.nv, self.ni = torch.full((K_BOT, n), float("inf")), torch.zeros(K_BOT, n, dtype=torch.long)

    def merge(self, tv: torch.Tensor, tpos: torch.Tensor, bv: torch.Tensor, bpos: torch.Tensor) -> None:
        """Fold in a batch's own top values tv (with their positions tpos) and bottom values bv (bpos)."""
        v, i = torch.cat([self.pv, tv]), torch.cat([self.pi, tpos])
        o = v.topk(K_TOP, dim=0).indices
        self.pv, self.pi = v.gather(0, o), i.gather(0, o)
        v, i = torch.cat([self.nv, bv]), torch.cat([self.ni, bpos])
        o = v.topk(K_BOT, dim=0, largest=False).indices
        self.nv, self.ni = v.gather(0, o), i.gather(0, o)


# ----------------------------------------------------------------------------------------- one batch, reduced
def mlp_sums(A: torch.Tensor) -> dict:
    """What one batch of a layer's activations A [positions, neurons] adds to its statistics: float32 sums over the
    positions (the running totals are float64), the number of positive values, and the batch's own top-K_TOP and
    bottom-K_BOT values with their rows in A."""
    t = A.topk(min(K_TOP, A.shape[0]), dim=0)
    b = A.topk(min(K_BOT, A.shape[0]), dim=0, largest=False)
    return {"s1": A.sum(0), "s2": A.pow(2).sum(0), "np": (A > 0).sum(0), "pm": A.clamp_min(0).sum(0),
            "nm": (-A).clamp_min(0).sum(0), "top_v": t.values, "top_i": t.indices, "bot_v": b.values, "bot_i": b.indices}


def pattern_context(tk: torch.Tensor) -> dict:
    """For windows tk [B, T]: where each position has an earlier copy of its token (eq), where it follows the token
    after an earlier copy (ind), the distances, and the counts the head statistics are normalised by."""
    T = tk.shape[1]
    tril = torch.ones(T, T).tril(-1).bool()
    ar = torch.arange(T)
    eq = (tk[:, :, None] == tk[:, None, :]) & tril                                    # an earlier copy of the same token
    ind = torch.zeros_like(eq)
    ind[:, :, 1:] = eq[:, :, :-1]                                                     # the token after an earlier copy
    ind &= torch.ones(T, T).tril(0).bool()
    has_eq, has_ind = eq.any(-1), ind.any(-1)
    npast = (ar + 1).float()
    counts = {"n": tk.shape[0] * T, "n_eq": float(has_eq.sum()), "n_ind": float(has_ind.sum()),
              "same_chance": float(((eq.sum(-1) / npast) * has_eq).sum()),
              "ind_chance": float(((ind.sum(-1) / npast) * has_ind).sum())}
    return {"eq": eq, "ind": ind, "has_eq": has_eq, "has_ind": has_ind, "dist": (ar[:, None] - ar[None, :]).clamp_min(0).float(),
            "counts": counts}


def pattern_sums(P: torch.Tensor, ctx: dict) -> dict:
    """Per-head sums of one layer's attention patterns P [B, head, dst, src] over the batch (see pattern_context)."""
    eq, ind, has_eq, has_ind, dist = ctx["eq"], ctx["ind"], ctx["has_eq"], ctx["has_ind"], ctx["dist"]
    return {"dist": (P * dist).sum(-1).sum((0, 2)), "prev": P.diagonal(offset=-1, dim1=-2, dim2=-1).sum((0, 2)),
            "self": P.diagonal(dim1=-2, dim2=-1).sum((0, 2)), "first": P[:, :, 1:, 0].sum((0, 2)),
            "entropy": -(P * (P + 1e-9).log()).sum(-1).sum((0, 2)),
            "same": ((P * eq[:, None]).sum(-1) * has_eq[:, None]).sum((0, 2)),
            "induct": ((P * ind[:, None]).sum(-1) * has_ind[:, None]).sum((0, 2))}


def loss_sum(logits: torch.Tensor, tk: torch.Tensor, rows: int | None = None) -> float:
    """Summed next-token loss of windows tk [B, T] under logits [B, T, vocabulary], in float64.  rows: take that many
    positions at a time, so the float64 copy is that many rows rather than the whole batch (the model worker)."""
    if rows is None:
        lp = torch.log_softmax(logits[:, :-1].double(), -1)
        return float(-lp.gather(-1, tk[:, 1:, None]).sum())
    total, n = 0.0, tk.shape[1] - 1
    for a in range(0, n, rows):
        lp = torch.log_softmax(logits[:, a:min(a + rows, n)].double(), -1)
        total += float(-lp.gather(-1, tk[:, a + 1:min(a + rows, n) + 1, None]).sum())
        del lp
    return total


def final_stream(ws, x: torch.Tensor, variant: str = "base", runner=None) -> torch.Tensor:
    """The residual stream as the unembedding sees it (final normalisation applied).  runner: the one to take the
    normalisation from (default the workspace's; in the model worker, the job's)."""
    r = ws.runner if runner is None else runner
    if hasattr(r, "_final"):
        r.use(variant)
        norm, _ = r._final()
        if norm is None:
            return x.float()
        with torch.no_grad():
            return norm(x.to(r.device, r.dtype)).float().cpu()
    cfg = r.cfg
    if not cfg.use_ln:
        return x.float()
    p = r.params(variant)
    x = x - x.mean(-1, keepdim=True)
    return x / (x.pow(2).mean(-1, keepdim=True) + cfg.ln_eps).sqrt() * p["ln_final.w"] + p["ln_final.b"]


def reduce_rest(ws, tk, logits, cache, pat, last, variant, runner=None, rows=None) -> dict:
    """Everything of one batch except the MLP layers: loss, the final normalised stream (position 0 left out) and
    per-head pattern sums; each capture is dropped from `cache` once it is reduced."""
    s = ws.struct
    out = {"loss": loss_sum(logits, tk, rows), "final": None, "heads": {}, "counts": None}
    if last in cache:
        out["final"] = final_stream(ws, cache.pop(last)[:, 1:], variant, runner).reshape(-1, s.d_model)
    ctx = None
    for l in pat:
        P = cache.pop(f"blocks.{l}.attn.hook_pattern", None)
        if P is None:
            continue
        if ctx is None:
            ctx = pattern_context(tk)
        out["heads"][l] = pattern_sums(P, ctx)
    out["counts"] = ctx["counts"] if ctx is not None else None
    return out


def reduce_batch(ws, tk, logits, cache, ls, pat, last, variant) -> dict:
    """One batch run in this process, reduced: {"loss", "mlp": {layer: mlp_sums}, "final", "heads", "counts"}."""
    s = ws.struct
    mlp = {}
    for l in ls:
        mlp[l] = mlp_sums(cache.pop(f"blocks.{l}.mlp.hook_post")[:, 1:].reshape(-1, s.mlp_width(l)).float())
    return {**reduce_rest(ws, tk, logits, cache, pat, last, variant), "mlp": mlp}


def merge_mlp(c: dict, r: dict, pos: torch.Tensor) -> None:
    """Add one batch's mlp_sums `r` to a layer's running totals `c`; pos: the probe position of each row of the batch."""
    for k in MLP_SUMS:
        c[k] += r[k].double()
    c["top"].merge(r["top_v"], pos[r["top_i"]], r["bot_v"], pos[r["bot_i"]])


def merge_heads(h: dict, sums: dict, counts: dict) -> None:
    for key in HEAD_KEYS:
        h[key] = h.get(key, 0) + sums[key]
    for key, value in counts.items():
        h[key] = h.get(key, 0) + value


# ----------------------------------------------------------------------------------------- the model worker's side
_NAME = re.compile(r"blocks\.(\d+)\.(mlp\.hook_post|attn\.hook_pattern|hook_resid_post)")


def observe_window(ws, r, ids: list, variant: str, names: list) -> dict:
    """Run one window once and reduce it next to the model: {"loss": repr of the float, "tensors": {...}}.

    names are the hook points to reduce: `blocks.L.mlp.hook_post` (per-neuron sums and the window's own top and
    bottom values, `L.s1` ... `L.bot_i`), `blocks.L.attn.hook_pattern` (per-head sums, `L.dist` ... `L.induct`, for the
    layers whose patterns the runner gives) and the last layer's `hook_resid_post` (`final`: the normalised stream,
    position 0 left out).  Each MLP layer is reduced by a hook inside the forward pass, so no layer's activations
    are held once it has been reduced; the logits give the loss and are dropped.  Nothing is kept after the job."""
    s = ws.struct
    if variant not in ("base", "current"):
        raise ValueError("observe_window variant must be base or current")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise ValueError("observe_window names must be a list of distinct hook names")
    mlp, pat, last = {}, [], None
    for n in names:
        m = _NAME.fullmatch(n)
        l = int(m.group(1)) if m else -1
        kind = m.group(2) if m else None
        if not 0 <= l < s.n_layers or (kind == "hook_resid_post" and l != s.n_layers - 1) \
                or (kind == "mlp.hook_post" and not s.has_neurons(l)) or (kind == "attn.hook_pattern" and not s.has_heads(l)):
            raise ValueError(f"observe_window does not reduce {n!r}")
        if not r.available(n):
            raise ValueError(f"the runner exposes no {n}")
        if kind == "mlp.hook_post":
            mlp[n] = l
        elif kind == "attn.hook_pattern":
            pat.append(l)
        else:
            last = n
    if not mlp:
        raise ValueError("observe_window needs at least one MLP layer (blocks.L.mlp.hook_post)")
    tk = torch.tensor([list(ids)])
    if tk.shape[1] < 2:
        raise ValueError("observe_window needs a window of at least two tokens")
    out: dict = {}

    def reduce(x, name):
        l = mlp[name]
        for k, v in mlp_sums(x[:, 1:].reshape(-1, s.mlp_width(l)).float()).items():
            out[f"{l}.{k}"] = v
        return None                                                                # the activation passes on unchanged

    keep = [n for n in names if n not in mlp]
    logits, cache = r.run(tk, variant, None, names=keep, hooks={n: reduce for n in mlp})
    rest = reduce_rest(ws, tk, logits, cache, pat, last, variant, runner=r, rows=LOSS_ROWS)
    del logits, cache
    missing = [l for l in mlp.values() if f"{l}.s1" not in out]
    if missing:
        raise RuntimeError(f"the runner did not show the activations of MLP layers {missing}")
    if rest["final"] is not None:
        out["final"] = rest["final"]
    for l, sums in rest["heads"].items():
        for k, v in sums.items():
            out[f"{l}.{k}"] = v
    return {"loss": repr(rest["loss"]), "tensors": out}


ADMISSION = ("forecast_exceeds_budget", "operation_exceeds_budget")


def _gb(b) -> str:
    return f"{b / 1e9:.1f} GB"


def stopped_words(record: dict, a: int, W: int) -> str:
    """The stage's words for a refusal that ended window a (0-based) of W: admission (how much was asked against what
    was free, and what to do), or a lost worker / revoked lease (what happened)."""
    where = f"the observed stage stopped at window {a + 1} of {W}"
    if record.get("code") in ADMISSION:
        m, limit = record.get("measured") or {}, record.get("limit")
        asked = m.get("new_bytes") or m.get("required")
        free = None
        if isinstance(m.get("available"), (int, float)):
            free = m["available"] - ((limit or {}).get("preferred", 0) if isinstance(limit, dict) else 0)
        asked_s = f"asking for {_gb(asked)} more" if isinstance(asked, (int, float)) else "asking for more memory"
        free_s = (f" with {_gb(max(0, free))} free above the reserve" if free is not None
                  else f" beyond the cap of {_gb(limit)}" if isinstance(limit, (int, float)) else "")
        return (f"{where}: memory admission refused it, {asked_s}{free_s} (the conservative forecast for a forward that keeps "
                "every layer's activations; the job itself holds far less). Close other software, lower the context setting "
                "or lower the preferred reserve in Settings, then build again; nothing of this stage was saved.")
    why = str(record.get("message") or "the model worker stopped").strip().rstrip(".")
    return f"{where}: {why}. Nothing of this stage was saved; the stages built before it are unchanged."


def _window(r, tk, a: int, W: int, variant: str, names: list, ls: list, pat: list) -> dict:
    """One window run in the model worker, as reduce_batch's result.  A refusal (admission, a lost worker) is raised
    again as a RuntimeRefusal with the same record and the stage's words as its message."""
    from ..runtime_events import RuntimeRefusal
    try:
        got = r.observe_window([int(i) for i in tk[0].tolist()], variant, names)
    except RuntimeRefusal as e:
        raise RuntimeRefusal({**e.record, "message": stopped_words(e.record, a, W)}, e.tokens) from e
    t = got["tensors"]
    want = [f"{l}.{k}" for l in ls for k in MLP_KEYS]
    if any(k not in t for k in want):
        raise RuntimeError(f"the model worker's reduction of window {a + 1} lacks MLP layers "
                           f"{sorted({int(k.split('.')[0]) for k in want if k not in t})}")
    heads = {l: {k: t[f"{l}.{k}"] for k in HEAD_KEYS} for l in pat if all(f"{l}.{k}" in t for k in HEAD_KEYS)}
    return {"loss": float(got["loss"]), "mlp": {l: {k: t[f"{l}.{k}"] for k in MLP_KEYS} for l in ls}, "final": t.get("final"),
            "heads": heads, "counts": pattern_context(tk)["counts"] if heads else None}


# ----------------------------------------------------------------------------------------- files
def _write(path: str, write) -> None:
    from .wiring import _write as atomic                                           # temporary name, fsync, rename
    try:
        atomic(path, write)
    except BaseException:
        try:
            os.unlink(path + ".tmp")                                               # never leave a half-written file
        except FileNotFoundError:
            pass
        raise


def _save_npz(path: str, **arrays) -> None:
    _write(path, lambda f: np.savez(f, **arrays))


def _save_npy(path: str, a: np.ndarray) -> None:
    _write(path, lambda f: np.save(f, a))


def _save_json(path: str, obj, indent: int | None = None) -> None:
    _write(path, lambda f: f.write(json.dumps(obj, indent=indent).encode()))


def build_observed(ws, adir: str, log=print, corpus: str | None = None, tokens: int | None = None, force: bool = False,
                   variant: str = "base", batch: int | None = None, before_write=None) -> dict:
    """Run the stage and write its files.  batch: at most this many windows per pass in this process (default 32,
    fewer for wide models); a model in the model worker is always run one window per job.  before_write, if given,
    is called once every window has run and before the first file is written (the atlas drops its record of an
    earlier observed stage there)."""
    from ..worker_jobs import uses_worker
    t0 = time.time()
    s, r = ws.struct, ws.runner
    ids, src = probe_windows(ws, corpus, tokens)
    W, T = ids.shape
    if T < 2:
        raise ValueError(f"windows of {T} token cannot be observed (position 0 has no context); the model runs with a "
                         f"context of {T}, so raise it to build the observed stage")
    ls = [l for l in range(s.n_layers) if s.has_neurons(l) and r.available(f"blocks.{l}.mlp.hook_post")]
    if not ls:
        raise ValueError(f"{ws.name}: the runner exposes no MLP activations (blocks.L.mlp.hook_post), so there is nothing to observe")
    pat = [l for l in range(s.n_layers) if s.has_heads(l) and r.available(f"blocks.{l}.attn.hook_pattern")]
    last = f"blocks.{s.n_layers - 1}.hook_resid_post"
    width = sum(s.mlp_width(l) for l in ls)
    remote = uses_worker(ws)
    if remote:
        B = 1
        log(f"  observed stage: {W} windows of {T} tokens from {src}, one at a time in the model worker")
    else:
        B = int(max(1, min(batch or 32, W, 600_000_000 // max(1, 4 * T * width))))
        log(f"  observed stage: {W} windows of {T} tokens from {src}, {B} at a time")
    acc = {l: {"s1": torch.zeros(s.mlp_width(l), dtype=torch.float64), "s2": torch.zeros(s.mlp_width(l), dtype=torch.float64),
               "np": torch.zeros(s.mlp_width(l), dtype=torch.float64), "pm": torch.zeros(s.mlp_width(l), dtype=torch.float64),
               "nm": torch.zeros(s.mlp_width(l), dtype=torch.float64), "top": _Top(s.mlp_width(l))} for l in ls}
    head = {l: {} for l in pat}
    ar = torch.arange(T)
    N, loss_sum_, loss_n, finals = 0, 0.0, 0, []
    fit_cap, fit_kept, fit_seen = max(2 * (T - 1), FIT_ROW_BYTES // (16 * s.d_model)), 0, 0
    names = [f"blocks.{l}.mlp.hook_post" for l in ls] + [f"blocks.{l}.attn.hook_pattern" for l in pat] + ([last] if r.available(last) else [])
    for a in range(0, W, B):
        tk = ids[a:a + B]
        bs = tk.shape[0]
        if remote:
            red = _window(r, tk, a, W, variant, names, ls, pat)
        else:
            logits, cache = r.run(tk, variant, names=names)
            red = reduce_batch(ws, tk, logits, cache, ls, pat, last, variant)
            del logits, cache
        loss_sum_ += red["loss"]
        loss_n += bs * (T - 1)
        pos = (torch.arange(a, a + bs)[:, None] * T + ar[None, 1:]).reshape(-1)          # position 0 has no context: left out
        for l in ls:
            merge_mlp(acc[l], red["mlp"][l], pos)
        N += bs * (T - 1)
        if red["final"] is not None:
            fit_seen += len(red["final"])
            if fit_kept < fit_cap:                                                  # the first fit_cap positions, both paths
                finals.append(red["final"][:fit_cap - fit_kept])
                fit_kept += len(finals[-1])
        for l, sums in red["heads"].items():
            merge_heads(head[l], sums, red["counts"])
        red = None
        if remote:
            log(f"    window {a + 1} of {W} ({time.time() - t0:.0f}s)")
        else:
            log(f"    windows {a}–{a + bs - 1} ({time.time() - t0:.0f}s)")
    if before_write is not None:
        before_write()
    _save_npy(os.path.join(adir, "probe_ids.npy"), ids.numpy().astype(np.int32))
    for l in ls:
        c = acc[l]
        mean = c["s1"] / N
        tv, bv = c["top"].pv.T.numpy(), c["top"].nv.T.numpy()
        _save_npz(os.path.join(adir, f"obs_{l:03d}.npz"),
                  obs_mean=mean.numpy().astype(np.float32), obs_std=(c["s2"] / N - mean ** 2).clamp_min(0).sqrt().numpy().astype(np.float32),
                  obs_frac_pos=(c["np"] / N).numpy().astype(np.float32),
                  obs_pos_share=(c["pm"] / (c["pm"] + c["nm"]).clamp_min(1e-12)).numpy().astype(np.float32),
                  obs_peak=peak_sign(tv, bv), obs_top_val=tv.astype(np.float32), obs_top_pos=c["top"].pi.T.numpy().astype(np.int32),
                  obs_bot_val=bv.astype(np.float32), obs_bot_pos=c["top"].ni.T.numpy().astype(np.int32))
    heads_obs = []
    for l in pat:
        h = head[l]
        if "n" not in h:
            continue
        for i in range(s.n_heads):
            same = float(h["same"][i]) / max(h["n_eq"], 1)
            indu = float(h["induct"][i]) / max(h["n_ind"], 1)
            sc, ic = h["same_chance"] / max(h["n_eq"], 1), h["ind_chance"] / max(h["n_ind"], 1)
            heads_obs.append({"layer": l, "head": i, "prev": round(float(h["prev"][i]) / (h["n"] - h["n"] / T), 4),
                              "first": round(float(h["first"][i]) / (h["n"] - h["n"] / T), 4), "self": round(float(h["self"][i]) / h["n"], 4),
                              "distance": round(float(h["dist"][i]) / h["n"], 2), "entropy": round(float(h["entropy"][i]) / h["n"], 3),
                              "same_x": round(same / sc, 2) if sc > 0 else None, "induction_x": round(indu / ic, 2) if ic > 0 else None})
    _save_json(os.path.join(adir, "heads_observed.json"), heads_obs)
    bias = None
    if finals:
        X = torch.cat(finals)
        finals = None
        if fit_seen > fit_kept:
            log(f"  the bias direction is fitted on the first {fit_kept:,} of {fit_seen:,} positions, which keeps the fit "
                f"near {_gb(16 * fit_kept * s.d_model + 32 * s.d_model ** 2)} of memory")
        win = torch.arange(len(X)) // (T - 1)
        fit = fit_bias_direction(X, win % 5 == 0)
        X = None
        _save_npy(os.path.join(adir, "bias_direction.npy"), fit.pop("w").numpy())
        bias = {**fit, "usable": bool(fit["test_sd"] < 0.25),
                "note": "w . x for the final normalised stream x; adding b*w to a token's unembedding row shifts its score by about b",
                **({"positions_seen": fit_seen} if fit_seen > fit_kept else {})}
        _save_json(os.path.join(adir, "bias.json"), bias, indent=1)
    else:
        for stale in ("bias_direction.npy", "bias.json"):                         # an earlier build's fit is not this one's
            if os.path.exists(os.path.join(adir, stale)):
                os.unlink(os.path.join(adir, stale))
        log(f"  the runner exposes no final residual stream ({last}), so no bias direction is fitted")
    out = {"windows": int(W), "window": int(T), "tokens": int(N), "source": src, "loss": round(loss_sum_ / max(1, loss_n), 4),
           "layers": ls, "patterns": bool(heads_obs), "seconds": round(time.time() - t0, 1)}
    _save_json(os.path.join(adir, "validation.json"), validate(ws, adir, out, bias), indent=1)
    log(f"  observed stage done in {time.time() - t0:.0f}s (loss {out['loss']:.3f} nats per token)")
    return out


def fit_bias_direction(X: torch.Tensor, test: torch.Tensor) -> dict:
    """Ridge fit of w . x = 1 over rows of X; held-out rows say how constant w . x really is."""
    Xtr, Xte = X[~test].double(), X[test].double()
    d = X.shape[1]
    M, mu = Xtr.T @ Xtr / len(Xtr), Xtr.mean(0)
    scale = float(M.diagonal().mean())
    best = None
    for lam in RIDGE:
        w = torch.linalg.lstsq(M + lam * max(scale, 1e-12) * torch.eye(d, dtype=torch.float64), mu[:, None]).solution[:, 0]
        pte = Xte @ w
        err = float((pte - 1).pow(2).mean().sqrt())
        if best is None or err < best[0]:
            best = (err, lam, w)
    _, lam, w = best
    pte = Xte @ w
    w = w / (float(pte.mean()) or 1.0)
    pte = Xte @ w
    return {"w": w.float(), "ridge": lam, "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)), "d_model": d,
            "test_mean": round(float(pte.mean()), 4), "test_sd": round(float(pte.std()), 4),
            "test_p01": round(float(torch.quantile(pte, 0.01)), 4), "test_p99": round(float(torch.quantile(pte, 0.99)), 4)}


def _ranks(a) -> np.ndarray:
    """1-based ranks, ties given the mean of the ranks they span."""
    a = np.asarray(a, dtype=np.float64)
    r = np.empty(len(a))
    r[a.argsort(kind="mergesort")] = np.arange(1, len(a) + 1)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=r) / counts)[inv]


def spearman(x, y) -> float:
    """Spearman's rank correlation (the Pearson correlation of the ranks), without scipy, which the IDE does not
    require.  Both inputs must vary.  NaN when either holds a NaN (as scipy's default)."""
    if np.isnan(np.asarray(x, dtype=np.float64)).any() or np.isnan(np.asarray(y, dtype=np.float64)).any():
        return float("nan")
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def _verdict(hit: float, chance: float, n: int) -> str:
    if n < 30:
        return "too few to tell"
    if chance <= 0:
        return "holds" if hit > 0.2 else "fails"
    r = hit / chance
    return "holds" if r >= 3 else "holds weakly" if r >= 1.5 else "fails"


def validate(ws, adir: str, run: dict, bias: dict | None) -> dict:
    """Every claim of the weights stage that the pass can test, with a verdict from fixed rules."""
    s = ws.struct
    ids = np.load(os.path.join(adir, "probe_ids.npy"))
    T = ids.shape[1]
    flat = ids.reshape(-1)
    tok2c = np.load(os.path.join(adir, "token_concept.npy"))
    cflat = tok2c[flat]
    freq = np.bincount(cflat[cflat >= 0], minlength=int(tok2c.max()) + 1) / max(1, len(cflat))
    per, tot = [], {k: [0, 0, 0.0, 0] for k in ("concept", "trigger")}
    agree = n_sign = peaked = n_all = 0
    for l in run["layers"]:
        A = {**np.load(os.path.join(adir, f"layer_{l:03d}.npz")), **np.load(os.path.join(adir, f"obs_{l:03d}.npz"))}
        pk, sw = A["obs_peak"], A["sign_w"]
        k = (pk != 0) & (sw != 0)
        a = int((pk[k] == sw[k]).sum())
        agree += a; n_sign += int(k.sum()); peaked += int((pk != 0).sum()); n_all += len(pk)
        rec = {"layer": l, "sign_agreement": round(a / max(1, int(k.sum())), 4), "sign_n": int(k.sum()),
               "peak_signed": round(float((pk != 0).mean()), 4)}
        specs = {"concept": [(A["obs_top_pos"][:, :4], 1, "out_pos", None), (A["obs_bot_pos"][:, :3], 1, "out_neg", None)],
                 "trigger": [(A["obs_top_pos"][:, :4], 0, "gate", "trig_E")] if "c_gate" in A else []}
        for key, parts in specs.items():
            h = n = nn = 0
            base = 0.0
            for idx, shift, src, need in parts:
                c = A[f"c_{src}"]
                pick = (c >= 0) & (A[f"n_{src}"] >= 3) & (A[f"p_{src}"] >= 0.5)
                if need:
                    pick &= A[need]
                pos = idx[pick].astype(np.int64) + shift
                ok = (pos % T != 0) if shift else np.ones_like(pos, dtype=bool)          # the next token must be in the same window
                hit = (cflat[np.minimum(pos, len(cflat) - 1)] == c[pick][:, None]) & ok
                h += int(hit.sum()); n += int(ok.sum()); base += float(freq[c[pick]].sum()) * idx.shape[1]; nn += int(pick.sum())
            rec[f"{key}_neurons"], rec[f"{key}_hit"], rec[f"{key}_chance"] = nn, round(h / max(1, n), 4), round(base / max(1, n), 4)
            t = tot[key]
            t[0] += h; t[1] += n; t[2] += base; t[3] += nn
        per.append(rec)
    checks = [{"claim": "Firing sign follows gate–value alignment", "n": n_sign,
               "value": round(agree / max(1, n_sign), 4), "chance": 0.5,
               "verdict": "too few to tell" if n_sign < 30 else "holds" if agree / n_sign >= 0.8 else "holds weakly" if agree / n_sign >= 0.6 else "fails",
               "text": f"The sign of a neuron's strong activations matches the sign predicted from its weights for {100 * agree / max(1, n_sign):.0f}% "
                       f"of the {n_sign:,} neurons that have both."}]
    for key, claim, what in (("trigger", "Gates open on specific current tokens", "the current token is in the concept its gate points at"),
                             ("concept", "Neurons promote a token concept", "the next token is in the concept its write direction points at")):
        h, n, base, nn = tot[key]
        hit, ch = h / max(1, n), base / max(1, n)
        checks.append({"claim": claim, "n": nn, "value": round(hit, 4), "chance": round(ch, 4), "verdict": _verdict(hit, ch, nn),
                       "text": f"At the strongest activations of {nn:,} such neurons, {what} {100 * hit:.1f}% of the time, against {100 * ch:.1f}% by chance."})
    hw_p, ho_p = os.path.join(adir, "heads.json"), os.path.join(adir, "heads_observed.json")
    mix = None
    if os.path.exists(hw_p) and os.path.exists(ho_p):
        hw = {(h["layer"], h["head"]): h for h in json.load(open(hw_p))}
        ho = {(h["layer"], h["head"]): h for h in json.load(open(ho_p)) if h["same_x"] is not None}
        keys = [k for k in ho if k in hw]
        if len(keys) >= 8:
            xs, ys = [hw[k]["same_token_z"] for k in keys], [ho[k]["same_x"] for k in keys]
            rho = spearman(xs, ys) if len(set(xs)) > 1 and len(set(ys)) > 1 else 0.0
            m = max(3, len(keys) // 12)
            tw = {k for k in sorted(keys, key=lambda k: -hw[k]["same_token_z"])[:m]}
            to = {k for k in sorted(keys, key=lambda k: -ho[k]["same_x"])[:m]}
            mix = {"heads": len(keys), "same_token_spearman": round(rho, 3), "top": m, "overlap": len(tw & to),
                   "prev_token_max": max(h["prev"] for h in ho.values()),
                   "induction_top": [[f"L{k[0]}H{k[1]}", ho[k]["induction_x"]] for k in sorted(keys, key=lambda k: -(ho[k]["induction_x"] or 0))[:5]]}
            ov, exp = len(tw & to), m * m / len(keys)
            corr = f"rank correlation {rho:.2f} over {len(keys)} heads"
            if rho != rho:                                                         # NaN: written as null, with a note
                mix["same_token_spearman"] = None
                mix["note"] = "the rank correlation is undefined: a head's weights or observed statistic is not a number"
                corr = f"rank correlation undefined over {len(keys)} heads: a head statistic is not a number"
            checks.append({"claim": "The weights say which heads attend to earlier copies of the current token", "n": len(keys),
                           "value": round(ov / m, 3), "chance": round(m / len(keys), 3), "verdict": _verdict(ov / m, m / len(keys), len(keys)),
                           "text": f"{ov} of the {m} heads the weights rank highest for matching a token with its earlier copies are among the "
                                   f"{m} that do so most on the probe text ({exp:.1f} expected by chance; {corr})."})
    return {"tokens": run["tokens"], "source": run["source"], "loss": run["loss"], "checks": checks, "per_layer": per,
            "peak_signed_share": round(peaked / max(1, n_all), 4), "mixers": mix, "bias": bias, "tau": tau(s.d_model)}


def unit_observed(atlas, l: int, n: int, span: int = 10) -> dict:
    """What one neuron did on the probe text: statistics and the contexts of its strongest activations."""
    A = atlas.layer(l)
    ids_p = os.path.join(atlas.dir, "probe_ids.npy")
    if "probe_ids" not in atlas._table:
        atlas._table["probe_ids"] = np.load(ids_p)
    ids = atlas._table["probe_ids"]
    T = ids.shape[1]
    flat = ids.reshape(-1)
    text = token_table(atlas.ws)["text"]

    def ctx(pos, val):
        w, t = divmod(int(pos), T)
        a = max(0, t - span)
        nxt = int(flat[pos + 1]) if t + 1 < T else None
        return {"value": round(float(val), 3), "before": [text[int(i)] for i in flat[w * T + a: pos]], "token": text[int(flat[pos])],
                "next": text[nxt] if nxt is not None else None}

    return {"peak": int(A["obs_peak"][n]), "mean": round(float(A["obs_mean"][n]), 4), "std": round(float(A["obs_std"][n]), 4),
            "frac_pos": round(float(A["obs_frac_pos"][n]), 3), "pos_share": round(float(A["obs_pos_share"][n]), 3),
            "sign_matches_weights": (int(A["obs_peak"][n]) == int(A["sign_w"][n])) if A["obs_peak"][n] and A["sign_w"][n] else None,
            "top": [ctx(p, v) for p, v in zip(A["obs_top_pos"][n][:5], A["obs_top_val"][n][:5]) if np.isfinite(v)],
            "bottom": [ctx(p, v) for p, v in zip(A["obs_bot_pos"][n][:3], A["obs_bot_val"][n][:3]) if np.isfinite(v) and v < 0]}
