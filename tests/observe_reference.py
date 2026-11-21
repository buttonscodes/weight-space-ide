"""Frozen reference: wsp/atlas/observe.py as it was before the observed stage learned to run in the model worker
(commit 40ba391), kept verbatim except for these absolute imports.  tests/test_atlas_observed_worker.py builds the
observed stage with this and with the current code and requires the same files.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from wsp.atlas.build import ROLE, tau  # noqa: F401
from wsp.atlas.tokens import token_table

K_TOP, K_BOT = 6, 3
RIDGE = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def probe_windows(ws, corpus: str | None = None, tokens: int | None = None) -> tuple[torch.Tensor, str]:
    """Token windows [W, T] to run, and a description of where they came from."""
    from wsp.tokenizer import BPETokenizer
    r = ws.runner
    T = int(min(r.max_ctx or 128, 128))
    local = os.path.join(ws.dir, "corpus.txt")
    if corpus:
        text, src = open(corpus, encoding="utf-8").read(), corpus
    elif os.path.exists(local):
        text, src = open(local, encoding="utf-8").read(), "corpus.txt in the workspace"
    elif isinstance(ws.tok, BPETokenizer):
        text, src = open(os.path.join(os.path.dirname(__file__), "probe.txt"), encoding="utf-8").read(), "the bundled probe text"
    else:
        text, src = None, "text sampled from the model itself"
    if text is None:
        from wsp import analysis
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

    def update(self, A: torch.Tensor, pos: torch.Tensor) -> None:
        k = min(K_TOP, A.shape[0])
        t = A.topk(k, dim=0)
        v, i = torch.cat([self.pv, t.values]), torch.cat([self.pi, pos[t.indices]])
        o = v.topk(K_TOP, dim=0).indices
        self.pv, self.pi = v.gather(0, o), i.gather(0, o)
        k = min(K_BOT, A.shape[0])
        b = A.topk(k, dim=0, largest=False)
        v, i = torch.cat([self.nv, b.values]), torch.cat([self.ni, pos[b.indices]])
        o = v.topk(K_BOT, dim=0, largest=False).indices
        self.nv, self.ni = v.gather(0, o), i.gather(0, o)


def final_stream(ws, x: torch.Tensor, variant: str = "base") -> torch.Tensor:
    """The residual stream as the unembedding sees it (final normalisation applied)."""
    r = ws.runner
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


def build_observed(ws, adir: str, log=print, corpus: str | None = None, tokens: int | None = None, force: bool = False,
                   variant: str = "base") -> dict:
    t0 = time.time()
    s, r = ws.struct, ws.runner
    ids, src = probe_windows(ws, corpus, tokens)
    W, T = ids.shape
    ls = [l for l in range(s.n_layers) if s.has_neurons(l) and r.available(f"blocks.{l}.mlp.hook_post")]
    if not ls:
        raise ValueError(f"{ws.name}: the runner exposes no MLP activations (blocks.L.mlp.hook_post), so there is nothing to observe")
    pat = [l for l in range(s.n_layers) if s.has_heads(l) and r.available(f"blocks.{l}.attn.hook_pattern")]
    last = f"blocks.{s.n_layers - 1}.hook_resid_post"
    width = sum(s.mlp_width(l) for l in ls)
    B = int(max(1, min(32, W, 600_000_000 // max(1, 4 * T * width))))
    log(f"  observed stage: {W} windows of {T} tokens from {src}, {B} at a time")
    acc = {l: {"s1": torch.zeros(s.mlp_width(l), dtype=torch.float64), "s2": torch.zeros(s.mlp_width(l), dtype=torch.float64),
               "np": torch.zeros(s.mlp_width(l), dtype=torch.float64), "pm": torch.zeros(s.mlp_width(l), dtype=torch.float64),
               "nm": torch.zeros(s.mlp_width(l), dtype=torch.float64), "top": _Top(s.mlp_width(l))} for l in ls}
    head = {l: {} for l in pat}
    tril = torch.ones(T, T).tril(-1).bool()
    ar = torch.arange(T)
    dist = (ar[:, None] - ar[None, :]).clamp_min(0).float()
    N, loss_sum, loss_n, finals = 0, 0.0, 0, []
    names = [f"blocks.{l}.mlp.hook_post" for l in ls] + [f"blocks.{l}.attn.hook_pattern" for l in pat] + ([last] if r.available(last) else [])
    for a in range(0, W, B):
        tk = ids[a:a + B]
        bs = tk.shape[0]
        logits, cache = r.run(tk, variant, names=names)
        lp = torch.log_softmax(logits[:, :-1].double(), -1)
        loss_sum += float(-lp.gather(-1, tk[:, 1:, None]).sum())
        loss_n += bs * (T - 1)
        pos = (torch.arange(a, a + bs)[:, None] * T + ar[None, 1:]).reshape(-1)          # position 0 has no context: left out
        for l in ls:
            A = cache[f"blocks.{l}.mlp.hook_post"][:, 1:].reshape(-1, s.mlp_width(l)).float()
            c = acc[l]
            c["s1"] += A.sum(0).double(); c["s2"] += A.pow(2).sum(0).double(); c["np"] += (A > 0).sum(0).double()
            c["pm"] += A.clamp_min(0).sum(0).double(); c["nm"] += (-A).clamp_min(0).sum(0).double()
            c["top"].update(A, pos)
        N += bs * (T - 1)
        if last in cache:
            finals.append(final_stream(ws, cache[last][:, 1:], variant).reshape(-1, s.d_model))
        eq = (tk[:, :, None] == tk[:, None, :]) & tril                                    # an earlier copy of the same token
        ind = torch.zeros_like(eq)
        ind[:, :, 1:] = eq[:, :, :-1]                                                     # the token after an earlier copy
        ind &= torch.ones(T, T).tril(0).bool()
        has_eq, has_ind = eq.any(-1), ind.any(-1)
        npast = (ar + 1).float()
        for l in pat:
            P = cache.get(f"blocks.{l}.attn.hook_pattern")
            if P is None:
                continue
            h = head[l]
            for key, val in (("dist", (P * dist).sum(-1).sum((0, 2))), ("prev", P.diagonal(offset=-1, dim1=-2, dim2=-1).sum((0, 2))),
                             ("self", P.diagonal(dim1=-2, dim2=-1).sum((0, 2))), ("first", P[:, :, 1:, 0].sum((0, 2))),
                             ("entropy", -(P * (P + 1e-9).log()).sum(-1).sum((0, 2))),
                             ("same", ((P * eq[:, None]).sum(-1) * has_eq[:, None]).sum((0, 2))),
                             ("induct", ((P * ind[:, None]).sum(-1) * has_ind[:, None]).sum((0, 2)))):
                h[key] = h.get(key, 0) + val
            h["n"] = h.get("n", 0) + bs * T
            h["n_eq"] = h.get("n_eq", 0) + float(has_eq.sum())
            h["n_ind"] = h.get("n_ind", 0) + float(has_ind.sum())
            h["same_chance"] = h.get("same_chance", 0) + float(((eq.sum(-1) / npast) * has_eq).sum())
            h["ind_chance"] = h.get("ind_chance", 0) + float(((ind.sum(-1) / npast) * has_ind).sum())
        log(f"    windows {a}–{a + bs - 1} ({time.time() - t0:.0f}s)")
    np.save(os.path.join(adir, "probe_ids.npy"), ids.numpy().astype(np.int32))
    for l in ls:
        c = acc[l]
        mean = c["s1"] / N
        tv, bv = c["top"].pv.T.numpy(), c["top"].nv.T.numpy()
        np.savez(os.path.join(adir, f"obs_{l:03d}.npz"),
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
    json.dump(heads_obs, open(os.path.join(adir, "heads_observed.json"), "w"))
    bias = None
    if finals:
        X = torch.cat(finals)
        win = torch.arange(len(X)) // (T - 1)
        fit = fit_bias_direction(X, win % 5 == 0)
        np.save(os.path.join(adir, "bias_direction.npy"), fit.pop("w").numpy())
        bias = {**fit, "usable": bool(fit["test_sd"] < 0.25),
                "note": "w . x for the final normalised stream x; adding b*w to a token's unembedding row shifts its score by about b"}
        json.dump(bias, open(os.path.join(adir, "bias.json"), "w"), indent=1)
    out = {"windows": int(W), "window": int(T), "tokens": int(N), "source": src, "loss": round(loss_sum / max(1, loss_n), 4),
           "layers": ls, "patterns": bool(heads_obs), "seconds": round(time.time() - t0, 1)}
    json.dump(validate(ws, adir, out, bias), open(os.path.join(adir, "validation.json"), "w"), indent=1)
    log(f"  observed stage done in {time.time() - t0:.0f}s (loss {out['loss']:.3f} nats per token)")
    return out


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
            from scipy.stats import spearmanr
            xs, ys = [hw[k]["same_token_z"] for k in keys], [ho[k]["same_x"] for k in keys]
            rho = float(spearmanr(xs, ys).statistic) if len(set(xs)) > 1 and len(set(ys)) > 1 else 0.0
            m = max(3, len(keys) // 12)
            tw = {k for k in sorted(keys, key=lambda k: -hw[k]["same_token_z"])[:m]}
            to = {k for k in sorted(keys, key=lambda k: -ho[k]["same_x"])[:m]}
            mix = {"heads": len(keys), "same_token_spearman": round(rho, 3), "top": m, "overlap": len(tw & to),
                   "prev_token_max": max(h["prev"] for h in ho.values()),
                   "induction_top": [[f"L{k[0]}H{k[1]}", ho[k]["induction_x"]] for k in sorted(keys, key=lambda k: -(ho[k]["induction_x"] or 0))[:5]]}
            ov, exp = len(tw & to), m * m / len(keys)
            checks.append({"claim": "The weights say which heads attend to earlier copies of the current token", "n": len(keys),
                           "value": round(ov / m, 3), "chance": round(m / len(keys), 3), "verdict": _verdict(ov / m, m / len(keys), len(keys)),
                           "text": f"{ov} of the {m} heads the weights rank highest for matching a token with its earlier copies are among the "
                                   f"{m} that do so most on the probe text ({exp:.1f} expected by chance; rank correlation {rho:.2f} over {len(keys)} heads)."})
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
