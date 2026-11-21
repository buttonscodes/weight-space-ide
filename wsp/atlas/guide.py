"""Using the atlas while editing: which neurons push an answer in a prompt, and shifting tokens directly.

`attribute` runs one prompt and measures what every MLP neuron adds, through the final normalisation and the
unembedding, to the score of a token (or to its margin over another):

    effect_i = a_i * (d_i . v)        a_i the neuron's activation at the answer position, d_i its write vector,
                                      v the read-out direction of the target at that position

This is the direct path only; what a neuron does by feeding later layers is not counted.  The atlas says what a
neuron can push in general, this says which neurons are pushing here.  Each row is annotated from the atlas.

`shift_plan` says how to move a token's score by a fixed amount everywhere: through the unembedding bias where
the model has one, else along the direction on which the final stream is constant (fitted by the observed stage).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ..worker_jobs import worker_job


def _one(ws, s) -> int:
    if isinstance(s, int):
        return s
    ids = ws.tok.encode_token(s)
    if len(ids) != 1:
        raise ValueError(f"{s!r} is {len(ids)} tokens ({[ws.tok.token_str(i) for i in ids]}); give a single token (a leading space is part of it)")
    return ids[0]


@worker_job('attribute')  # runs in the model worker when the workspace uses one (it opens the operation)
def attribute(ws, prompt: str, target, versus=None, variant: str = "current", top: int = 20, group: dict | None = None) -> dict:
    from . import Atlas, select
    s, r, sem = ws.struct, ws.runner, ws.sem
    ids = ws.encode(prompt)
    if r.max_ctx:
        ids = ids[-r.max_ctx:]
    t, vs = _one(ws, target), (_one(ws, versus) if versus not in (None, "") else None)
    ls = [l for l in range(s.n_layers) if s.has_neurons(l) and r.available(f"blocks.{l}.mlp.hook_post")]
    last = f"blocks.{s.n_layers - 1}.hook_resid_post"
    logits, cache = r.run(ids, variant, names=[f"blocks.{l}.mlp.hook_post" for l in ls] + [last])
    lg = logits[0, -1]
    x = cache[last][0, -1]
    gain, centre = r.final_gain(x, variant)
    u = sem.unembed_col(t, variant=variant) - (sem.unembed_col(vs, variant=variant) if vs is not None else sem.W_U().whole(variant).mean(1)
                                              if sem.W_U().size <= 64_000_000 else 0.0)
    v = gain * u
    if centre:
        v = v - v.mean()
    eff, total = {}, 0.0
    for l in ls:
        a = cache[f"blocks.{l}.mlp.hook_post"][0, -1].float()
        proj = sem.mlp(l)["W_out"].contract(v, axis=1, variant=variant)
        eff[l] = (a * proj).numpy()
        total += float(eff[l].sum())
    A = Atlas.of(ws)
    flat = sorted(((float(e[i]), l, int(i)) for l, e in eff.items() for i in np.argsort(-np.abs(e))[:top]), key=lambda z: -abs(z[0]))[:top]
    rows = []
    for e, l, i in flat:
        row = {"layer": l, "neuron": i, "effect": round(e, 4)}
        if A.exists and l in A.layers:
            L = A.layer(l)
            row.update(mech=["sharpener", "amplifier", "damper", "router"][int(L["mech"][i])],
                       plus=A.text(L["out_pos_ids"][i][:4]), minus=A.text(L["out_neg_ids"][i][:4]))
        rows.append(row)
    out = {"prompt": prompt, "target": ws.tok.token_str(t), "versus": ws.tok.token_str(vs) if vs is not None else None,
           "margin": round(float(lg[t] - (lg[vs] if vs is not None else lg.mean())), 4), "mlp_total": round(total, 4),
           "by_layer": {int(l): round(float(e.sum()), 4) for l, e in eff.items()}, "top": rows}
    if group and A.exists:
        sel = select(A, group)
        g = np.array([eff[int(l)][int(i)] for l, i in zip(sel["layer"], sel["idx"]) if int(l) in eff])
        out["group"] = {"select": group, "neurons": int(len(g)), "total": round(float(g.sum()), 4),
                        "for": int((g > 0).sum()), "against": int((g < 0).sum())}
    return out


def top_script(res: dict, k: int = 5, scale: float = 0.0) -> str:
    """A weight script scaling the k neurons that push hardest for the target in an attribution result."""
    best = sorted((r for r in res["top"] if r["effect"] > 0), key=lambda r: -r["effect"])[:k]
    by: dict[int, list[int]] = {}
    for r in best:
        by.setdefault(r["layer"], []).append(r["neuron"])
    return "\n".join(f"scale_neurons({l}, {sorted(n)}, {scale:g})" for l, n in sorted(by.items()))


def shift_plan(ws) -> dict:
    """How a token's score can be shifted by a fixed amount in every context."""
    from . import atlas_dir
    s = ws.struct
    if s.unembed_bias is not None:
        return {"mode": "bias", "exact": True, "note": f"the model has a logit bias ({s.unembed_bias.name}); a shift is added to it"}
    from . import Atlas
    p = os.path.join(atlas_dir(ws), "bias.json")
    if not os.path.exists(p) or not Atlas.of(ws).has("observed"):                # only a fit the atlas records as finished
        return {"mode": None, "note": "this model has no logit bias. Run the atlas's observed stage: it fits a direction on which "
                                      "the final stream is constant, which a token shift can use instead"}
    import json
    b = json.load(open(p))
    if not b.get("usable"):
        return {"mode": None, "sd": b["test_sd"], "note": f"the fitted constant direction varies too much on held-out text "
                f"(±{b['test_sd']:.2f}; fitted on {b['train_rows']} positions for {b['d_model']} dimensions). Use more probe text"}
    return {"mode": "direction", "exact": False, "sd": b["test_sd"], "tied": bool(s.tied),
            "note": f"no logit bias; shifts go along a fitted direction on which the final stream is 1.00 ± {b['test_sd']:.2f} on held-out text"
                    + ("; the unembedding is tied to the embedding, so the token's input embedding moves too" if s.tied else "")}


def bias_direction(ws) -> torch.Tensor:
    from . import atlas_dir
    return torch.from_numpy(np.load(os.path.join(atlas_dir(ws), "bias_direction.npy"))).float()
