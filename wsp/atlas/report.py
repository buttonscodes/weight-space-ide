"""The atlas as text: what the programmer model and the command line read."""
from __future__ import annotations

from ..tokenizer import show
from .build import MECH, MECH_DOC, ROLE, ROLE_DOC


def _toks(xs, n=6) -> str:
    return " ".join(repr(show(t["token"])) + (f"({t['z']:+.0f})" if t["outlier"] else "") for t in xs[:n])


def overview(A) -> str:
    if not A.exists:
        st = A.status()
        if not st["can_build"]:
            return "No atlas: this checkpoint's architecture was not recognised, so there are no neurons to map."
        return ("No atlas has been built for this model yet. Build one (Atlas tab, or `python -m wsp atlas --model NAME`) "
                "before editing: it says what every neuron reads and writes.")
    m, lt = A.meta, A.layer_table()
    st = m["stages"]
    out = [f"Atlas of {m['model']}: {st['weights']['neurons']:,} neurons in {len(st['weights']['layers'])} layers; stages built: {', '.join(st)}.",
           "It describes the BASE weights." + (f" Layers edited since: {A.stale_layers()}." if A.stale_layers() else ""),
           "", "Mechanism classes (from the angles between a neuron's gate read, value read and write vectors):"]
    out += [f"  {k}: {MECH_DOC[k]}" for k in MECH]
    out += ["Vocabulary roles (the write vector read through the unembedding, against random directions):"]
    out += [f"  {k}: {ROLE_DOC[k]}" for k in ROLE]
    out += ["", "Stages (runs of layers with a shared profile):"]
    out += [f"  L{s['from']}–L{s['to']}: {s['desc']}" for s in lt["stages"]]
    out += ["", "layer  sharp  ampl  damp  route  vocab-shaped  token-trig  pred-trig"]
    for x in lt["layers"]:
        n = max(1, x["neurons"])
        out.append(f"  L{x['layer']:<3d} " + " ".join(f"{100 * x['mech'][k] / n:5.1f}" for k in MECH)
                   + f"  {100 * x['vocab']:10.1f}  {100 * (x['trig_E'] or 0):9.1f}  {100 * x['trig_U']:8.1f}")
    v = A.json("validation.json")
    if v:
        out += ["", f"Checked against a forward pass over {v['tokens']:,} tokens ({v['source']}, loss {v['loss']:.2f} nats):"]
        out += [f"  [{c['verdict']}] {c['claim']}. {c['text']}" for c in v["checks"]]
    else:
        out += ["", "Not checked against a forward pass (the observed stage has not run): firing signs are predictions from weights."]
    res = A.json("residual.json") or {}
    if res.get("sinks"):
        out.append(f"Residual coordinates written far more than read (sinks): {res['sinks']}.")
    c = A.json("concepts.json")
    out += ["", f"{len(c['nodes'])} token concepts (clusters of the unembedding) in {len(c['groups'])} groups. Use action=concepts to list, "
            "action=search to find neurons, action=unit for one neuron, action=attribute to see which neurons push an answer in a prompt."]
    return "\n".join(out)


def unit(A, l: int, n: int) -> str:
    u = A.unit(l, n)
    g = u["geometry"]
    out = [f"L{l}N{n}  {u['mech']} ({MECH_DOC[u['mech']]}); {u['role']} ({ROLE_DOC[u['role']]})" + ("  [layer edited since the atlas was built]" if u["stale"] else ""),
           f"  cos(gate,value) {g['cos_gu']:+.2f}  cos(gate,write) {g['cos_gd']:+.2f}  cos(value,write) {g['cos_ud']:+.2f}  |write| {g['d_norm']:.2f}  "
           f"visibility {g['out_vis']:.2f}x random  threshold {g['bias_z']:+.2f}",
           f"  writes + end: {_toks(u['writes']['pos'])}", f"  writes - end: {_toks(u['writes']['neg'])}"]
    if u["gate_tokens"]:
        out.append(f"  gate opens on current tokens{' (stands out)' if u['trig_E'] else ''}: {_toks(u['gate_tokens']['pos'])}")
    out.append(f"  gate opens when predicting{' (stands out)' if u['trig_U'] else ''}: {_toks(u['gate_predicted']['pos'])}")
    for k, c in u["concepts"].items():
        out.append(f"  concept at {k}: C{c['id']} {c['name']!r} ({c['outliers']} outliers, purity {c['purity']:.2f})")
    for w in (u.get("wiring") or [])[:6]:
        out.append(f"  L{w['layer']}N{w['neuron']} {w['direction']} through its {w['via']} read (cos {w['cos']:+.2f})")
    o = u.get("observed")
    if o:
        sign = {1: "positive", -1: "negative", 0: "both signs"}[o["peak"]]
        out.append(f"  observed: fires hard {sign}" + ("" if o["sign_matches_weights"] is None else " (as the weights predict)" if o["sign_matches_weights"] else " (opposite to the weight prediction)")
                   + f"; mean {o['mean']:+.3f}, sd {o['std']:.3f}")
        for c in o["top"][:3]:
            out.append(f"    {c['value']:+.2f} at …{show(''.join(c['before'][-6:]))}[{show(c['token'])}] → {show(c['next'] or '')!r}")
    else:
        out.append(f"  not observed; sign predicted from weights: {'+' if u['sign_w'] > 0 else '-' if u['sign_w'] < 0 else 'either'}")
    return "\n".join(out)


def selection(A, sel: dict, res: dict, show_n: int = 12) -> str:
    out = [f"{res['n']:,} neurons match {sel}  (firing sign from {res['sign_source']})",
           "  by layer: " + " ".join(f"L{l}:{c}" for l, c in res["by_layer"].items())]
    for l, i in list(zip(res["layer"], res["idx"]))[:show_n]:
        u = A.unit(int(l), int(i))
        o = u.get("observed")
        out.append(f"  L{l}N{i} {u['mech']:9s} + {_toks(u['writes']['pos'], 4)}  |  - {_toks(u['writes']['neg'], 4)}"
                   + (f"  | fires {'+' if o['peak'] > 0 else '-' if o['peak'] < 0 else '±'}" if o else ""))
    return "\n".join(out)


def concepts(A, group: int | None = None) -> str:
    c = A.concepts()
    out = []
    for g in c["groups"]:
        if group is not None and g["id"] != group:
            continue
        out.append(f"group {g['id']} ({g['size']:,} tokens): " + " ".join(repr(show(t)) for t in g["text"][:8]))
        for n in c["nodes"]:
            if n["group"] == g["id"]:
                out.append(f"   C{n['id']:<4d} {n['size']:6,d} tokens, cohesion {n['cohesion']:.2f}: " + " ".join(repr(show(t)) for t in n["text"][:8]))
    return "\n".join(out)


def attribution(res: dict) -> str:
    out = [f"score of {res['target']!r}" + (f" minus {res['versus']!r}" if res["versus"] else " relative to the average token")
           + f": {res['margin']:+.2f} nats at the last position; all MLP neurons contribute {res['mlp_total']:+.2f} directly",
           "  by layer: " + " ".join(f"L{l}:{v:+.2f}" for l, v in res["by_layer"].items() if abs(v) >= 0.05), "  strongest neurons (direct effect):"]
    for r in res["top"]:
        out.append(f"    L{r['layer']}N{r['neuron']} {r['effect']:+.3f}" + (f"  {r['mech']:9s} + {' '.join(repr(show(t)) for t in r['plus'])}  |  - {' '.join(repr(show(t)) for t in r['minus'])}" if "mech" in r else ""))
    if "group" in res:
        g = res["group"]
        out.append(f"  the {g['neurons']:,} neurons selected by {g['select']} contribute {g['total']:+.3f} here ({g['for']} push for, {g['against']} against)")
    return "\n".join(out)
