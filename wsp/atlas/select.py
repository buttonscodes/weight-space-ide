"""Choosing neurons from the atlas: the question "which units should this edit touch?" as a selector.

A selector is a dict; every condition given must hold.

    layers        [first, last] inclusive
    neurons       explicit units, "L26.4797" or [26, 4797]
    mech          sharpener | amplifier | damper | router            (one or a list)
    role          diffuse | few | set_pos | set_neg | contrast       (one or a list)
    trigger       token | predicted | none
    concept       token-concept id(s): neurons whose signature puts that concept at the matched end
    token         exact token string(s): neurons whose stored signature lists one of them
    word          like token, but every spelling of the word (case, leading space)
    end           where the concept / token must appear: write (default: either write end) | pos | neg | gate | predict
    effect        promote | suppress: keep the neurons that raise (lower) the target when they fire hard.
                  Uses the observed sign of strong firing where the observed stage ran, else the sign
                  predicted from weights; neurons with no clear sign are left out
    min_outliers  outlier tokens required at the matched end (default 2 for concepts)
    min_purity    share of the outlier mass that must be in the concept (default 0.5)
    where         extra conditions on any per-neuron column: [["d_top_dim", "==", 3994], ["out_vis", ">", 2]]
    top           keep only the N neurons with the strongest matched signature
"""
from __future__ import annotations

import re

import numpy as np

from .build import MECH, ROLE
from .tokens import token_table

OPS = {"==": np.equal, "!=": np.not_equal, ">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}
KEYS = {"layers", "neurons", "mech", "role", "trigger", "concept", "token", "word", "end", "effect", "min_outliers",
        "min_purity", "where", "top"}


def _list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def token_ids(ws, token=None, word=None) -> np.ndarray:
    text = token_table(ws)["text"]
    pick = np.zeros(len(text), dtype=bool)
    if token is not None:
        want = set(_list(token))
        hit = np.fromiter((t in want for t in text), bool, len(text))
        missing = want - {text[i] for i in np.nonzero(hit)[0]}
        if missing:
            raise ValueError(f"not in the vocabulary: {sorted(missing)!r} (a leading space is part of a token)")
        pick |= hit
    if word is not None:
        want = {w.strip().lower() for w in _list(word)}
        pick |= np.fromiter((t.strip().lower() in want for t in text), bool, len(text))
    return np.nonzero(pick)[0]


def fires(atlas) -> np.ndarray:
    """+1 / -1 / 0 per neuron: the sign it fires with when it fires hard."""
    try:
        return atlas.column("obs_peak").astype(np.int8)
    except KeyError:
        return atlas.column("sign_w").astype(np.int8)


def select(atlas, sel: dict) -> dict:
    """Resolve a selector.  Returns {"layer", "idx" (arrays), "strength", "by_layer", "n", "sign_source"}."""
    atlas.need()
    unknown = set(sel) - KEYS
    if unknown:
        raise ValueError(f"unknown selector key(s) {sorted(unknown)}; known: {sorted(KEYS)}")
    ws = atlas.ws
    L, I = atlas.column("layer"), atlas.column("idx")
    n = len(L)
    m = np.ones(n, dtype=bool)
    if "neurons" in sel:
        m[:] = False
        for u in _list(sel["neurons"]) if not (isinstance(sel["neurons"], (list, tuple)) and len(sel["neurons"]) == 2
                                               and all(isinstance(x, int) for x in sel["neurons"])) else [sel["neurons"]]:
            if isinstance(u, str):
                mm = re.fullmatch(r"[Ll](\d+)[.:Nn](\d+)", u.strip())
                if not mm:
                    raise ValueError(f"bad neuron id {u!r}; expected L<layer>.<index>")
                u = (int(mm.group(1)), int(mm.group(2)))
            m |= (L == int(u[0])) & (I == int(u[1]))
    if "layers" in sel:
        a, b = sel["layers"]
        m &= (L >= a) & (L <= b)
    for key, names in (("mech", MECH), ("role", ROLE)):
        if key in sel:
            want = _list(sel[key])
            bad = [w for w in want if w not in names]
            if bad:
                raise ValueError(f"unknown {key} {bad}; choose from {names}")
            m &= np.isin(atlas.column(key), [names.index(w) for w in want])
    for name, op, value in sel.get("where", []):
        if op not in OPS:
            raise ValueError(f"unknown comparison {op!r}")
        m &= OPS[op](atlas.column(name), value)
    if "trigger" in sel:
        tE, tU = atlas.column("trig_E"), atlas.column("trig_U")
        m &= {"token": tE, "predicted": tU, "none": ~tE & ~tU}[sel["trigger"]]

    end = sel.get("end", "write")
    ends = {"write": ["out_pos", "out_neg"], "pos": ["out_pos"], "neg": ["out_neg"], "gate": ["gate"], "predict": ["gateU"]}[end]
    zcol = {"out_pos": "out_pos_z", "out_neg": "out_neg_z", "gate": "gate_pos_z", "gateU": "gateU_pos_z"}
    icol = {"out_pos": "out_pos_ids", "out_neg": "out_neg_ids", "gate": "gate_pos_ids", "gateU": "gateU_pos_ids"}
    strength = np.zeros(n, dtype=np.float32)
    matched = np.zeros(n, dtype=np.int8)                          # +1: matched at the + end of the write, -1: at the - end
    target = any(k in sel for k in ("concept", "token", "word"))
    if target:
        hit = np.zeros(n, dtype=bool)
        ids = token_ids(ws, sel.get("token"), sel.get("word")) if ("token" in sel or "word" in sel) else None
        if ids is not None and not len(ids):
            raise ValueError(f"no vocabulary entry matches {sel.get('word', sel.get('token'))!r}")
        cids = np.array(sorted(int(c) for c in _list(sel["concept"]))) if "concept" in sel else None
        for e in ends:
            z = np.abs(atlas.column(zcol[e])[:, 0].astype(np.float32))
            k = np.zeros(n, dtype=bool)
            if cids is not None:
                k |= np.isin(atlas.column(f"c_{e}"), cids) & (atlas.column(f"n_{e}") >= sel.get("min_outliers", 2)) \
                    & (atlas.column(f"p_{e}") >= sel.get("min_purity", 0.5))
            if ids is not None:
                k |= np.isin(atlas.column(icol[e]), ids).any(1)
            better = k & (z > strength)
            strength[better] = z[better]
            matched[better] = -1 if e == "out_neg" else 1
            hit |= k
        m &= hit
    if "effect" in sel:
        if not target or end not in ("write", "pos", "neg"):
            raise ValueError("'effect' needs a concept, token or word matched at a write end")
        if sel["effect"] not in ("promote", "suppress"):
            raise ValueError("effect is promote or suppress")
        m &= fires(atlas) * matched == (1 if sel["effect"] == "promote" else -1)
    if not target:
        strength = np.maximum(np.abs(atlas.column("out_pos_z")[:, 0]), np.abs(atlas.column("out_neg_z")[:, 0])).astype(np.float32)
    if "top" in sel:
        keep = np.argsort(-np.where(m, strength, -1))[:int(sel["top"])]
        mm = np.zeros(n, dtype=bool)
        mm[keep] = True
        m &= mm
    idx = np.nonzero(m)[0]
    idx = idx[np.argsort(-strength[idx], kind="stable")]
    layers, counts = np.unique(L[idx], return_counts=True)
    return {"layer": L[idx], "idx": I[idx], "strength": strength[idx], "n": int(len(idx)),
            "by_layer": {int(a): int(b) for a, b in zip(layers, counts)},
            "sign_source": "observed" if atlas.has("observed") else "weights"}


def selection_script(res: dict, scale: float) -> str:
    """A weight script that scales what the selected neurons write."""
    lines = []
    for l in sorted(set(res["layer"].tolist())):
        ids = sorted(int(i) for i in res["idx"][res["layer"] == l])
        lines.append(f"scale_neurons({l}, {ids}, {scale:g})")
    return "\n".join(lines)
