"""The weights-only stage of the atlas: what every MLP neuron reads and writes, for any recognised model.

A neuron is the vectors it reads the residual stream with and the vector it writes back:

    gated MLP   act(g . n(x)) * (u . n(x)) * d      g = gate read, u = value read, d = write
    plain MLP   act(r . n(x) + b) * d               one read r (treated as g = u = r)

n(x) is the layer's normalisation; its gain (and, for LayerNorm, its centring) is folded into
the reads, and the final normalisation into the unembedding, so that every statement below is
about the residual stream itself.  Nothing is run.  For each neuron:

    geometry    cosines between its vectors, their norms, the firing threshold a bias implies
    writes      U_eff @ d over the whole vocabulary: top promoted / suppressed tokens as z-scores
    reads       E_hat @ g (which current tokens open it) and U_eff @ g (which predicted tokens do)
    classes     mechanism, vocabulary role and trigger type, each call made against the same statistic
                for random directions at its 99.9th percentile
    concepts    the cluster of the unembedding its outlier tokens fall in

All weights are reached through `ws.sem` / `Ref`, so this is the same code for every layout the
IDE recognises and for every store behind the workspace (safetensors on disk or on the Hub,
GGUF once that store is present).  Peak memory is one vocabulary matrix in the working dtype
plus one layer's MLP in float32.
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import numpy as np
import torch

from .tokens import CATS, CAT_INDEX, token_table

K_OUT, K_GATE, K_UP = 8, 6, 4
MECH = ["sharpener", "amplifier", "damper", "router"]
ROLE = ["diffuse", "few", "set_pos", "set_neg", "contrast"]
MECH_DOC = {
    "sharpener": "gate, value and write share an axis: it boosts a feature that is already present",
    "amplifier": "value and write share an axis, the gate reads something else: context sets the gain on a feature",
    "damper": "it writes against the direction it reads: an open gate erases that feature",
    "router": "it reads one direction and writes an unrelated one",
}
ROLE_DOC = {
    "diffuse": "its write direction is indistinguishable from a random direction in vocabulary space",
    "few": "one to seven tokens stand out at an end of its write direction",
    "set_pos": "a set of at least eight tokens stands out at the positive end",
    "set_neg": "a set of at least eight tokens stands out at the negative end",
    "contrast": "token sets stand out at both ends: an axis between two groups of tokens",
}
N_NULL = 2048
_PRE_MLP = re.compile(r"(post_attention|ln_?2|norm_?2|layer_norm_?2|layernorm_?2|ffn_norm|mlp_norm|post_attn|norm2)", re.I)
_OFFSET_TYPES = ("gemma", "qwen3_5")           # RMSNorm parameterised as (1 + weight)

_DTYPE: torch.dtype | None = None


def work_dtype() -> torch.dtype:
    """bfloat16 where the CPU multiplies it faster than float32 (AMX), else float32.  WSP_ATLAS_DTYPE overrides."""
    global _DTYPE
    if _DTYPE is None:
        env = os.environ.get("WSP_ATLAS_DTYPE")
        if env:
            _DTYPE = getattr(torch, env)
        else:
            a, b = torch.randn(256, 1024), torch.randn(1024, 4096)
            t = time.perf_counter(); a @ b; f32 = time.perf_counter() - t
            try:
                ab, bb = a.to(torch.bfloat16), b.to(torch.bfloat16)
                ab @ bb
                t = time.perf_counter(); ab @ bb; bf = time.perf_counter() - t
            except RuntimeError:
                bf = float("inf")
            _DTYPE = torch.bfloat16 if bf * 1.5 < f32 else torch.float32
    return _DTYPE


def tau(d_model: int) -> float:
    """|cosine| beyond which two directions are related: five standard deviations of a random pair."""
    return min(0.5, 5.0 / math.sqrt(max(d_model, 1)))                     # capped for toy widths, where 5 sd would exceed 1


# ----------------------------------------------------------------------------------------- normalisation
def norm_offset(ws) -> float:
    """1.0 for architectures whose norm gain is stored as (gain - 1), else 0.0."""
    if "atlas_norm_offset" in ws.keep:
        return ws.keep["atlas_norm_offset"]
    cfg = ws.hf_config or {}
    mt = " ".join(str(c.get("model_type") or "") for c in (cfg, cfg.get("text_config") or {}))
    # GGUF files of those families store the gain itself, so for them only the data decides
    off = 1.0 if ws.kind != "gguf" and any(k in mt for k in _OFFSET_TYPES) else 0.0
    if not off and ws.struct.norm == "rms" and ws.kind != "wsp":
        w = next((p for p in ws.struct.final_norm if p.role == "norm_w"), None)
        if w is not None:
            g = ws.read(w.name, (), "base")
            if float((g < 0).float().mean()) > 0.2 and float(g.median()) < 0.3:        # gains scattered around zero
                off = 1.0
    ws.keep["atlas_norm_offset"] = off
    return off


def _gain(ws, part, variant: str) -> torch.Tensor:
    return ws.read(part.name, (), variant) + norm_offset(ws)


def final_norm(ws, variant: str = "base") -> tuple[torch.Tensor | None, bool]:
    w = next((p for p in ws.struct.final_norm if p.role == "norm_w"), None)
    return (_gain(ws, w, variant) if w is not None else None), ws.struct.norm == "ln"


def mlp_norm(ws, l: int, variant: str = "base") -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
    """(gain, bias, centres?) of the normalisation in front of layer l's MLP."""
    L = ws.struct.layers[l]
    ws_ = [p for p in L.norms if p.role == "norm_w"]
    if not ws_:
        return None, None, False
    w = next((p for p in ws_ if _PRE_MLP.search(p.name[len(L.prefix):] if L.prefix and p.name.startswith(L.prefix) else p.name)), ws_[-1])
    stem = w.name.rsplit(".", 1)[0]
    b = next((p for p in L.norms if p.role == "norm_b" and p.name.rsplit(".", 1)[0] == stem), None)
    return _gain(ws, w, variant), (ws.read(b.name, (), variant) if b is not None else None), ws.struct.norm == "ln"


def fold(R: torch.Tensor, gain, centre: bool) -> torch.Tensor:
    """Read vectors [n, d] of a normalised stream -> read vectors of the stream itself (up to its scale)."""
    if gain is not None:
        R = R * gain
    if centre:
        R = R - R.mean(1, keepdim=True)
    return R


# ----------------------------------------------------------------------------------------- vocabulary matrices
def vocab_matrix(ws, which: str, variant: str = "base", log=None) -> dict:
    """The embedding ('E') or unembedding ('U') as a transposed matrix [d, V] in the working dtype.

    E: each token's row as the first normalisation presents it (unit RMS, centred for LayerNorm).
    U: each token's output direction with the final normalisation folded in.
    Also returns per-category mean rows [n_cat, d], the mean row, and the standard deviation of a unit random
    direction's projection (what "no better than random" looks like)."""
    s, sem = ws.struct, ws.sem
    tt = token_table(ws)
    V, d = s.d_vocab, s.d_model
    dt = work_dtype()
    centre = s.norm == "ln"
    gain = final_norm(ws, variant)[0] if which == "U" else None
    ref = sem.W_U() if which == "U" else sem.W_E()
    if not ws.store.local(ref.name):
        ws.store.fetch(ref.name)
    MT = torch.empty(d, V, dtype=dt)
    cat = torch.zeros(len(CATS), d, dtype=torch.float64)
    cnt = torch.zeros(len(CATS), dtype=torch.float64)
    cid = torch.from_numpy(tt["cat"].astype(np.int64))
    colsum = torch.zeros(d, dtype=torch.float64)
    sq = 0.0
    step = max(256, min(16384, 64_000_000 // max(d, 1)))
    for a in range(0, V, step):
        b = min(V, a + step)
        x = ref.sub((slice(None), slice(a, b))).whole(variant).T if which == "U" else ref.sub(slice(a, b)).whole(variant)
        x = x.float().contiguous()                                              # [chunk, d]
        if which == "E":
            if centre:
                x = x - x.mean(1, keepdim=True)
            x = x / x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        else:
            if gain is not None:
                x = x * gain
            if centre:
                x = x - x.mean(1, keepdim=True)
        MT[:, a:b] = x.T.to(dt)
        cat.index_add_(0, cid[a:b], x.double())
        cnt.index_add_(0, cid[a:b], torch.ones(b - a, dtype=torch.float64))
        colsum += x.sum(0).double()
        sq += float(x.double().pow(2).sum())
    mean = colsum / V
    sig = float(max((sq - V * float(mean.pow(2).sum())) / (V * d), 1e-30) ** 0.5)
    return {"MT": MT, "cat": (cat / cnt.clamp_min(1)[:, None]).float(), "mean": mean.float(), "sig_rand": sig}


def project(R: torch.Tensor, MT: torch.Tensor, k: int) -> dict:
    """Statistics of R @ MT per row.  R: [n, d] float32 directions, MT: [d, V].

    mean and std over the vocabulary, excess kurtosis, and the top-k / bottom-k token ids with their z-scores
    (value minus the row mean, in units of the row's standard deviation)."""
    n, V = R.shape[0], MT.shape[1]
    k = min(k, V)
    chunk = int(max(8, min(1024, 250_000_000 // max(V, 1))))
    mu, sd, ku = torch.empty(n), torch.empty(n), torch.empty(n)
    tp_i, tp_v = torch.empty(n, k, dtype=torch.int32), torch.empty(n, k)
    tn_i, tn_v = torch.empty(n, k, dtype=torch.int32), torch.empty(n, k)
    for a in range(0, n, chunk):
        x = (R[a:a + chunk].to(MT.dtype) @ MT).float()
        m = x.mean(1, keepdim=True)
        x -= m
        s = x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
        t, b = torch.topk(x, k, dim=1), torch.topk(x, k, dim=1, largest=False)
        sl = slice(a, a + x.shape[0])
        ku[sl] = (x / s).pow(4).mean(1) - 3.0
        mu[sl], sd[sl] = m.squeeze(1), s.squeeze(1)
        tp_i[sl], tp_v[sl] = t.indices.int(), t.values / s
        tn_i[sl], tn_v[sl] = b.indices.int(), b.values / s
    return {"mean": mu, "std": sd, "kurt": ku, "pos_ids": tp_i, "pos_z": tp_v, "neg_ids": tn_i, "neg_z": tn_v}


def calibrate(M: dict, d: int, seed: int = 0) -> dict:
    """What the top of a token list looks like for random directions: the thresholds every 'stands out' call uses."""
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(N_NULL, d, generator=g)
    R /= R.norm(dim=1, keepdim=True)
    p = project(R, M["MT"], K_OUT)
    top = torch.cat([p["pos_z"], -p["neg_z"]]).numpy()                     # both ends are exchangeable
    k = top.shape[1]
    cz = ((R @ M["cat"].T) - p["mean"][:, None]) / p["std"][:, None]
    return {"n": N_NULL, "k": k, "top1_q999": float(np.quantile(top[:, 0], 0.999)), "top1_median": float(np.median(top[:, 0])),
            "topk_q999": float(np.quantile(top[:, -1], 0.999)), "top6_q999": float(np.quantile(top[:, min(5, k - 1)], 0.999)),
            "kurt_q999": float(np.quantile(p["kurt"].numpy(), 0.999)), "sig_rand": M["sig_rand"],
            "cat_mean": [float(x) for x in cz.mean(0)], "cat_sd": [float(x) for x in cz.std(0).clamp_min(1e-6)]}


# ----------------------------------------------------------------------------------------- token concepts
def n_concepts(V: int) -> int:
    return int(max(4, min(256, round(math.sqrt(V) / 2))))


def kmeans(MT: torch.Tensor, mean: torch.Tensor, K: int, iters: int = 20, seed: int = 0, valid: np.ndarray | None = None,
           fit_rows: int = 60_000):
    """Spherical k-means of the centred vocabulary rows.  MT: [d, V].  Returns (assignment [V] with -1 for rows left
    out, unit centroids [K, d], cosine of every row to its centroid).  Large vocabularies are fitted on a random
    sample of rows and then assigned in one pass."""
    d, V = MT.shape
    ok = torch.ones(V, dtype=torch.bool) if valid is None else torch.from_numpy(valid)
    idx = torch.nonzero(ok).flatten()
    g = torch.Generator().manual_seed(seed)
    step = max(1024, min(32768, 96_000_000 // max(d, 1)))

    def unit(x):
        x = x.float() - mean
        return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)

    sample = idx[torch.randperm(len(idx), generator=g)[:fit_rows]].sort().values
    X = torch.cat([unit(MT[:, sample[a:a + step]].T) for a in range(0, len(sample), step)])        # [n, d] float32
    C = X[torch.randperm(len(X), generator=g)[:K]].clone()
    prev = torch.full((len(X),), -1, dtype=torch.long)
    for it in range(iters):
        v, j = (X @ C.T).max(1)
        acc = torch.zeros(K, d).index_add_(0, j, X)
        n = torch.bincount(j, minlength=K)
        empty = torch.nonzero(n == 0).flatten()
        if len(empty):                                                      # re-seed empty clusters on badly fitted rows
            acc[empty] = X[v.argsort()[:len(empty)]]
        C = acc / acc.norm(dim=1, keepdim=True).clamp_min(1e-12)
        moved = int((j != prev).sum())
        prev = j
        if moved < max(1, len(X) // 2000) and it >= 4:
            break
    assign = torch.full((V,), -1, dtype=torch.long)
    cos = torch.zeros(V)
    for a in range(0, V, step):
        b = min(V, a + step)
        v, j = (unit(MT[:, a:b].T) @ C.T).max(1)
        assign[a:b], cos[a:b] = torch.where(ok[a:b], j, torch.full_like(j, -1)), v
    return assign.numpy().astype(np.int16), C, cos.numpy()


def build_concepts(ws, M: dict, log) -> dict:
    """Cluster the unembedding into token concepts, then the concepts into groups.  Labels are the central tokens."""
    tt = token_table(ws)
    V = ws.struct.d_vocab
    valid = tt["cat"] != CAT_INDEX["unused"]
    K = min(n_concepts(int(valid.sum())), max(2, int(valid.sum()) // 2))
    assign, C, cos = kmeans(M["MT"], M["mean"], K, valid=valid)
    G = int(max(2, min(24, round(K / 10)))) if K >= 8 else 1
    if G > 1:
        g_assign, _, _ = kmeans(C.T.contiguous(), torch.zeros(C.shape[1]), G, iters=30, seed=1)
    else:
        g_assign = np.zeros(K, dtype=np.int16)
    nodes = []
    for c in range(K):
        m = np.nonzero(assign == c)[0]
        order = m[np.argsort(-cos[m])]
        cats = np.bincount(tt["cat"][m], minlength=len(CATS)) / max(1, len(m))
        top = np.argsort(-cats)[:3]
        nodes.append({"id": c, "group": int(g_assign[c]), "size": int(len(m)),
                      "cohesion": round(float(cos[m].mean()) if len(m) else 0.0, 3),
                      "tokens": [int(i) for i in order[:16]],
                      "cats": {CATS[i]: round(float(cats[i]), 2) for i in top if cats[i] > 0.05}})
    groups = []
    for gi in range(G):
        members = [n for n in nodes if n["group"] == gi]
        members.sort(key=lambda n: -n["size"])
        groups.append({"id": gi, "concepts": [n["id"] for n in members], "size": sum(n["size"] for n in members),
                       "tokens": [t for n in members[:4] for t in n["tokens"][:3]]})
    log(f"  concepts: {K} clusters of the unembedding in {G} groups (median cohesion "
        f"{float(np.median([n['cohesion'] for n in nodes])):.2f})")
    return {"assign": assign, "nodes": nodes, "groups": groups, "space": "centred unembedding rows with the final norm folded in"}


# ----------------------------------------------------------------------------------------- one layer
def layer_vectors(ws, l: int, variant: str = "base") -> dict:
    """The read and write vectors of layer l's neurons as [n, d] float32, reads with the layer's norm folded in."""
    s = ws.struct
    L = s.layers[l]
    m = ws.sem.mlp(l)
    for r in m.values():
        if not ws.store.local(r.name):
            ws.store.fetch(r.name)
    gain, nbias, centre = mlp_norm(ws, l, variant)
    D = m["W_out"].whole(variant).float().contiguous()
    out = {"D": D, "gated": "W_gate" in L.mlp}
    for role, key in (("W_gate", "G"), ("W_up", "U"), ("W_in", "G")):
        if role in L.mlp:
            raw = m[role].whole(variant).float().T.contiguous()
            b = m["b_" + role[2:]].read(variant).float() if "b_" + role[2:] in m else torch.zeros(raw.shape[0])
            if nbias is not None:
                b = b + raw @ nbias                                              # the norm's bias reaches the read as a constant
            out[key] = fold(raw, gain, centre)
            out["b" + key] = b
    return out


def layer_write(ws, l: int, variant: str = "base") -> torch.Tensor:
    """Only the write vectors of layer l's neurons, [n, d] float32: the same values as layer_vectors(...)["D"],
    without reading the gate and value matrices."""
    r = ws.sem.mlp(l)["W_out"]
    if not ws.store.local(r.name):
        ws.store.fetch(r.name)
    return r.whole(variant).float().contiguous()


def _timed_vectors(ws, l: int, variant: str, times: dict | None) -> dict:
    """layer_vectors, recording the first time each layer took to read (for the wiring stage's estimate)."""
    t = time.perf_counter()
    v = layer_vectors(ws, l, variant)
    if times is not None:
        times.setdefault(l, time.perf_counter() - t)
    return v


def read_record(times: dict) -> dict | None:
    if not times:
        return None
    return {"mlp": float(np.median(list(times.values()))), "layers": len(times),
            "note": "seconds to read one layer's MLP and fold its norm (median over layers, first read in this build)"}


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-20)


def _np(v, dtype=np.float32):
    return v.detach().float().numpy().astype(dtype) if isinstance(v, torch.Tensor) else np.asarray(v, dtype=dtype)


def layer_geometry(vec: dict) -> dict:
    D, G = vec["D"], vec["G"]
    Uv = vec.get("U", G)
    e = D * D
    tot = e.sum(1).clamp_min(1e-20)
    mx, ix = e.max(1)
    r = {"g_norm": G.norm(dim=1), "u_norm": Uv.norm(dim=1), "d_norm": D.norm(dim=1),
         "cos_gu": _cos(G, Uv) if vec["gated"] else torch.ones(D.shape[0]),
         "cos_gd": _cos(G, D), "cos_ud": _cos(Uv, D), "d_top_share": mx / tot,
         "bias": vec["bG"], "bias_z": vec["bG"] / G.norm(dim=1).clamp_min(1e-12)}
    out = {k: _np(v) for k, v in r.items()}
    out["d_top_dim"] = ix.numpy().astype(np.int32)
    return out


def _sig(prefix: str, p: dict, keep: int | None = None) -> dict:
    k = p["pos_ids"].shape[1] if keep is None else min(keep, p["pos_ids"].shape[1])
    return {f"{prefix}_pos_ids": p["pos_ids"][:, :k].numpy().astype(np.int32), f"{prefix}_pos_z": _np(p["pos_z"][:, :k], np.float16),
            f"{prefix}_neg_ids": p["neg_ids"][:, :k].numpy().astype(np.int32), f"{prefix}_neg_z": _np(p["neg_z"][:, :k], np.float16),
            f"{prefix}_std": _np(p["std"]), f"{prefix}_kurt": _np(p["kurt"])}


def layer_U(vec: dict, MU: dict) -> dict:
    """Everything about a layer that is read through the unembedding."""
    D, G = vec["D"], vec["G"]
    p = project(D, MU["MT"], K_OUT)
    r = _sig("out", p)
    r["out_vis"] = _np(p["std"] / (D.norm(dim=1).clamp_min(1e-20) * MU["sig_rand"]))
    r["out_cat_z"] = _np(((D @ MU["cat"].T) - p["mean"][:, None]) / p["std"][:, None], np.float16)
    r.update(_sig("gateU", project(G, MU["MT"], K_GATE)))
    return r


def layer_E(vec: dict, ME: dict) -> dict:
    """Everything about a layer that is read through the token embeddings."""
    G = vec["G"]
    p = project(G, ME["MT"], K_GATE)
    r = _sig("gate", p)
    r["gate_cat_z"] = _np(((G @ ME["cat"].T) - p["mean"][:, None]) / p["std"][:, None], np.float16)
    if vec["gated"]:
        r.update(_sig("up", project(vec["U"], ME["MT"], K_UP)))
    return r


# ----------------------------------------------------------------------------------------- classes
def concept_label(ids: np.ndarray, z: np.ndarray, thr: float, tok2c: np.ndarray):
    """The concept most of the outlier tokens (z > thr) at one end fall in: (concept or -1, outliers, purity)."""
    n, k = ids.shape
    c = tok2c[ids]
    wgt = np.where((z > thr) & (c >= 0), z, 0.0).astype(np.float32)
    nout = (wgt > 0).sum(1).astype(np.int8)
    best, purity, score = np.full(n, -1, dtype=np.int16), np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    for j in range(k):
        m = (wgt * (c == c[:, j:j + 1])).sum(1)
        upd = (m > score) & (wgt[:, j] > 0)
        best[upd], score[upd] = c[upd, j], m[upd]
    tot = wgt.sum(1)
    purity[tot > 0] = score[tot > 0] / tot[tot > 0]
    return best, nout, purity


def _cat_label(cat_z: np.ndarray, null: dict, sign: float):
    z = (cat_z.astype(np.float32) - np.array(null["cat_mean"], dtype=np.float32)) / np.array(null["cat_sd"], dtype=np.float32)
    z = z * sign
    z[:, CAT_INDEX["unused"]] = -np.inf
    return z.argmax(1).astype(np.int8), z.max(1).astype(np.float32)


def classify(A: dict, null: dict, d_model: int, tok2c: np.ndarray | None) -> dict:
    """Rule-based classes from the raw signatures of one layer."""
    t = tau(d_model)
    a, c = A["cos_gu"], A["cos_ud"]
    n = len(a)
    mech = np.full(n, MECH.index("router"), dtype=np.int8)
    mech[c < -t] = MECH.index("damper")
    mech[(c > t) & (np.abs(a) <= t)] = MECH.index("amplifier")
    mech[(c > t) & (np.abs(a) > t)] = MECH.index("sharpener")
    out = {"mech": mech, "sign_w": np.where(np.abs(a) > t, np.sign(a), 0).astype(np.int8)}
    U, E = null["U"], null["E"]
    op, on = A["out_pos_z"].astype(np.float32), -A["out_neg_z"].astype(np.float32)
    setP, setN = op[:, -1] > U["topk_q999"], on[:, -1] > U["topk_q999"]
    fewP, fewN = op[:, 0] > U["top1_q999"], on[:, 0] > U["top1_q999"]
    role = np.zeros(n, dtype=np.int8)
    role[fewP | fewN] = ROLE.index("few")
    role[setP & ~setN] = ROLE.index("set_pos")
    role[setN & ~setP] = ROLE.index("set_neg")
    role[setP & setN] = ROLE.index("contrast")
    out.update(role=role, n_out_pos=(op > U["top1_q999"]).sum(1).astype(np.int8), n_out_neg=(on > U["top1_q999"]).sum(1).astype(np.int8))
    gu = A["gateU_pos_z"].astype(np.float32)
    out["trig_U"] = gu[:, 0] > U["top1_q999"]
    ones = np.ones((n, 1), dtype=np.float32)
    out["out_cat_pos"], out["out_cat_pos_z"] = _cat_label(A["out_cat_z"], U, 1.0)
    out["out_cat_neg"], out["out_cat_neg_z"] = _cat_label(A["out_cat_z"], U, -1.0)
    del ones
    ends = [("out_pos", A["out_pos_ids"], op, U["top1_q999"]), ("out_neg", A["out_neg_ids"], on, U["top1_q999"]),
            ("gateU", A["gateU_pos_ids"], gu, U["top1_q999"])]
    if "gate_pos_z" in A:
        gp, gn = A["gate_pos_z"].astype(np.float32), -A["gate_neg_z"].astype(np.float32)
        out["trig_E"], out["inhib_E"] = gp[:, 0] > E["top1_q999"], gn[:, 0] > E["top1_q999"]
        out["gate_cat"], out["gate_cat_top_z"] = _cat_label(A["gate_cat_z"], E, 1.0)
        ends.append(("gate", A["gate_pos_ids"], gp, E["top1_q999"]))
    if tok2c is not None:
        for nm, ids, zz, thr in ends:
            out[f"c_{nm}"], out[f"n_{nm}"], out[f"p_{nm}"] = concept_label(ids, zz, thr, tok2c)
    return out


# ----------------------------------------------------------------------------------------- whole-model summaries
def residual_census(ws, variant: str = "base", times: dict | None = None) -> dict:
    """Which residual coordinates are written to or read from far more than the others."""
    s, sem = ws.struct, ws.sem
    d = s.d_model
    write, read = torch.zeros(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64)
    by_layer = []
    for l in range(s.n_layers):
        if not s.has_neurons(l):
            continue
        v = _timed_vectors(ws, l, variant, times)
        w = v["D"].double().pow(2).sum(0)
        write += w
        read += v["G"].double().pow(2).sum(0) + (v["U"].double().pow(2).sum(0) if "U" in v else 0)
        by_layer.append((l, (w / w.mean().clamp_min(1e-30)).float()))
    w, r = (write / write.mean().clamp_min(1e-30)).float(), (read / read.mean().clamp_min(1e-30)).float()
    g, _ = final_norm(ws, variant)
    top = w.argsort(descending=True)[:8].tolist()
    return {"note": "energy of MLP write / read vectors on each residual coordinate, relative to the average coordinate",
            "write_p99": float(torch.quantile(w, 0.99)), "write_max": float(w.max()),
            "privileged": [{"dim": int(j), "write": round(float(w[j]), 2), "read": round(float(r[j]), 2),
                            "final_gain": round(float(g[j]), 3) if g is not None else None,
                            "write_by_layer": {int(l): round(float(x[j]), 1) for l, x in by_layer}} for j in top],
            "sinks": [int(j) for j in top if float(w[j]) > 5 and float(r[j]) < 1]}


FEATURES = [("sharpener", "boost features already present"), ("amplifier", "let context set the gain on features"),
            ("damper", "write against what they read"), ("router", "read one thing and write another"),
            ("vocab", "write token-shaped directions"), ("contrast", "write an axis between two token sets"),
            ("trig_E", "open on specific current tokens"), ("trig_U", "open on specific predicted tokens")]


def layer_summary(l: int, kind: str, A: dict) -> dict:
    n = len(A["mech"])
    share = lambda m: round(float(np.mean(m)), 4)                                # noqa: E731
    rec = {"layer": l, "mixer": kind, "neurons": n,
           "mech": {m: int((A["mech"] == i).sum()) for i, m in enumerate(MECH)},
           "role": {m: int((A["role"] == i).sum()) for i, m in enumerate(ROLE)},
           "vocab": share(A["role"] != 0), "contrast": share(A["role"] == ROLE.index("contrast")),
           "trig_E": share(A["trig_E"]) if "trig_E" in A else None, "trig_U": share(A["trig_U"]),
           "sign_pos": int((A["sign_w"] > 0).sum()), "sign_neg": int((A["sign_w"] < 0).sum()),
           "median": {k: round(float(np.median(A[k])), 4) for k in ("cos_gu", "cos_gd", "cos_ud", "d_norm", "out_vis", "bias_z")},
           "abs_cos_gu_median": round(float(np.median(np.abs(A["cos_gu"]))), 4),
           "d_top_dim_mode": int(np.bincount(A["d_top_dim"]).argmax())}
    return rec


def _feat(x: dict) -> list[float]:
    n = max(1, x["neurons"])
    return [x["mech"][m] / n for m in MECH] + [x["vocab"], x["contrast"], x["trig_E"] or 0.0, x["trig_U"]]


def stages(layers: list[dict], kmax: int = 7) -> list[dict]:
    """Contiguous runs of layers with a shared weight-derived profile (least within-run variance, by dynamic
    programming), each described by the traits that set it apart from the rest of the stack."""
    n = len(layers)
    if n < 2:
        return [{"from": layers[0]["layer"], "to": layers[0]["layer"], "traits": [], "desc": ""}] if layers else []
    raw = np.array([_feat(x) for x in layers])
    X = (raw - raw.mean(0)) / raw.std(0).clip(1e-9)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            seg = X[i:j + 1]
            cost[i, j] = ((seg - seg.mean(0)) ** 2).sum()
    best, sse = {}, {}
    for K in range(1, min(kmax, n) + 1):
        dp = np.full((K + 1, n + 1), np.inf)
        arg = np.zeros((K + 1, n + 1), dtype=int)
        dp[0, 0] = 0
        for k in range(1, K + 1):
            for j in range(1, n + 1):
                for i in range(k - 1, j):
                    v = dp[k - 1, i] + cost[i, j - 1]
                    if v < dp[k, j]:
                        dp[k, j], arg[k, j] = v, i
        cuts, j = [], n
        for k in range(K, 0, -1):
            i = arg[k, j]
            cuts.append((int(i), int(j - 1)))
            j = i
        best[K], sse[K] = cuts[::-1], float(dp[K, n])
    K = 1
    while K + 1 in sse and sse[K] > 0 and (sse[K] - sse[K + 1]) / sse[1] > 0.06:   # stop when another cut explains < 6%
        K += 1
    out = []
    for a, b in best[K]:
        seg = raw[a:b + 1].mean(0)
        z = (seg - raw.mean(0)) / raw.std(0).clip(1e-9)
        order = [i for i in np.argsort(-z) if z[i] > 0.4][:2]
        traits = [{"feature": FEATURES[i][0], "share": round(float(seg[i]), 3), "all_layers": round(float(raw[:, i].mean()), 3)} for i in order]
        desc = "; ".join(f"{100 * seg[i]:.0f}% of neurons {FEATURES[i][1]} ({100 * raw[:, i].mean():.0f}% over all layers)" for i in order)
        out.append({"from": layers[a]["layer"], "to": layers[b]["layer"], "traits": traits,
                    "desc": desc or "close to the average layer on every measured trait"})
    return out


# ----------------------------------------------------------------------------------------- heads
def head_records(ws, log, variant: str = "base") -> list[dict]:
    """A compact weights-only record per attention head (layers whose mixer is recognised)."""
    from .. import analysis
    s, sem = ws.struct, ws.sem
    out = []
    S = analysis._sample_tokens(ws)
    ES = sem.embed_rows(S)
    ES = ES / ES.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
    for l in range(s.n_layers):
        if not s.has_heads(l):
            continue
        for h in range(s.n_heads):
            H = sem.head(l, h)
            WQ, WK, WV, WO = (H[m].read(variant) for m in "QKVO")
            qk = (ES @ WQ) @ (ES @ WK).T
            off = qk[~torch.eye(len(S), dtype=torch.bool)]
            same_z = float((qk.diagonal().mean() - off.mean()) / off.std().clamp_min(1e-12))
            ov = (ES @ WV) @ WO
            lg = sem.logit_effect(ov, variant)
            own = lg[torch.arange(len(S)), S]
            rank = (lg > own[:, None]).sum(1)
            out.append({"layer": l, "head": h, "norms": {m: round(float(x.norm()), 3) for m, x in zip("QKVO", (WQ, WK, WV, WO))},
                        "same_token_z": round(same_z, 2), "copy_top1": round(float((rank == 0).float().mean()), 4),
                        "copy_top10": round(float((rank < 10).float().mean()), 4)})
        log(f"  heads of layer {l} done")
    return out


# ----------------------------------------------------------------------------------------- the stage
def fetch_bytes(ws) -> int:
    """Bytes that a full weights build would have to download (0 when the checkpoint is on disk)."""
    s, st = ws.struct, ws.store
    names = {p.name for p in (s.embed, s.unembed) if p is not None}
    for l in range(s.n_layers):
        names |= {p.name for p in s.layers[l].mlp.values()}
    return sum(st.size_bytes(n) for n in names if not st.local(n))


def build_weights(ws, adir: str, log=print, variant: str = "base", force: bool = False, heads: bool = True) -> dict:
    s = ws.struct
    if not s.recognised or s.unembed is None:
        raise ValueError(f"{ws.name}: the architecture was not recognised, so there are no neurons to map "
                         "(the name tree and tensor-level tools still work)")
    ls = [l for l in range(s.n_layers) if s.has_neurons(l)]
    if not ls:
        raise ValueError(f"{ws.name}: no layer has a recognised MLP")
    os.makedirs(adir, exist_ok=True)
    t0 = time.time()
    times: dict = {}
    tt = token_table(ws)
    path = lambda l: os.path.join(adir, f"layer_{l:03d}.npz")                      # noqa: E731
    have = {l: dict(np.load(path(l))) for l in ls if os.path.exists(path(l)) and not force}
    full = not have                                                                 # every layer read in this build
    null_p, con_p = os.path.join(adir, "null.json"), os.path.join(adir, "concepts.json")
    null = json.load(open(null_p)) if os.path.exists(null_p) and not force else {}
    log(f"atlas of {ws.name}: {len(ls)} layers with neurons, {sum(s.mlp_width(l) for l in ls):,} neurons, "
        f"vocabulary {s.d_vocab:,}, working in {str(work_dtype()).split('.')[-1]}")

    # ---- pass 1: everything read through the unembedding
    need_U = "U" not in null or not os.path.exists(con_p) or any("out_pos_ids" not in have.get(l, {}) for l in ls)
    if need_U:
        MU = vocab_matrix(ws, "U", variant)
        log(f"  unembedding ready ({time.time() - t0:.0f}s)")
        if "U" not in null:
            null["U"] = calibrate(MU, s.d_model)
        if not os.path.exists(con_p) or force:
            con = build_concepts(ws, MU, log)
            np.save(os.path.join(adir, "token_concept.npy"), con.pop("assign"))
            json.dump(con, open(con_p, "w"))
        for l in ls:
            if "out_pos_ids" in have.get(l, {}):
                continue
            t1 = time.time()
            vec = _timed_vectors(ws, l, variant, times)
            A = {**layer_geometry(vec), **layer_U(vec, MU)}
            np.savez(path(l), **A)
            have[l] = A
            log(f"  layer {l}: writes and prediction triggers ({time.time() - t1:.0f}s)")
        del MU
    # ---- pass 2: everything read through the token embeddings
    need_E = "E" not in null or any("gate_pos_ids" not in have[l] for l in ls)
    if need_E and s.embed is not None:
        ME = vocab_matrix(ws, "E", variant)
        log(f"  embedding ready ({time.time() - t0:.0f}s)")
        if "E" not in null:
            null["E"] = calibrate(ME, s.d_model, seed=1)
        for l in ls:
            if "gate_pos_ids" in have[l]:
                continue
            t1 = time.time()
            have[l].update(layer_E(_timed_vectors(ws, l, variant, times), ME))
            np.savez(path(l), **have[l])
            log(f"  layer {l}: token triggers ({time.time() - t1:.0f}s)")
        del ME
    json.dump(null, open(null_p, "w"), indent=1)

    # ---- classes, summaries
    tok2c = np.load(os.path.join(adir, "token_concept.npy"))
    layers = []
    for l in ls:
        A = have[l]
        A.update(classify(A, null, s.d_model, tok2c))
        np.savez(path(l), **A)
        layers.append(layer_summary(l, "attention" if s.has_heads(l) else "other", A))
    json.dump({"layers": layers, "stages": stages(layers)}, open(os.path.join(adir, "layers.json"), "w"), indent=1)
    json.dump(residual_census(ws, variant, times), open(os.path.join(adir, "residual.json"), "w"), indent=1)
    hp = os.path.join(adir, "heads.json")
    if heads and (force or not os.path.exists(hp)) and any(s.has_heads(l) for l in range(s.n_layers)):
        json.dump(head_records(ws, lambda *_: None, variant), open(hp, "w"))
    log(f"  weights stage done in {time.time() - t0:.0f}s")
    return {"layers": ls, "neurons": int(sum(s.mlp_width(l) for l in ls)), "seconds": round(time.time() - t0, 1),
            "tau": tau(s.d_model), "dtype": str(work_dtype()).split(".")[-1], "norm_offset": norm_offset(ws),
            "n_unused": int((tt["cat"] == CAT_INDEX["unused"]).sum()), "read_s": read_record(times), "full": full}
