"""The programmer model's tools.

One registry serves every transport: the stdio MCP server that a harness (Claude Code, Codex,
oh-my-pi, ...) launches, the HTTP MCP endpoint, and the IDE's own "tool console".  Each tool
takes JSON arguments and returns text sized for a context window: every reader reports an
estimated token count and refuses to return more than `max_tokens` without being asked.
"""
from __future__ import annotations

from .runtime_events import job_scope, RuntimeRefusal

import json
import os
import re

from . import addressing as A
from . import analysis as an
from . import chunks, edits, hierarchy, testsuite
from .tokenizer import show

TOOLS: dict[str, dict] = {}


def tool(name: str, description: str, properties: dict | None = None, required: list[str] | None = None):
    def deco(fn):
        TOOLS[name] = {"name": name, "description": description, "fn": fn,
                       "inputSchema": {"type": "object", "properties": properties or {}, "required": required or []}}
        return fn
    return deco


def specs() -> list[dict]:
    return [{k: t[k] for k in ("name", "description", "inputSchema")} for t in TOOLS.values()]


def call(sess, name: str, args: dict | None) -> str:
    if name not in TOOLS:
        raise KeyError(f"unknown tool {name!r}")
    if sess.ws is None:
        return "No workspace is open in the IDE."
    out = TOOLS[name]["fn"](sess, **(args or {}))
    return out if isinstance(out, str) else json.dumps(out, indent=1)


S = {"type": "string"}
I = {"type": "integer"}
B = {"type": "boolean"}


def _status(sess) -> str:
    ws = sess.ws
    lines = []
    if not ws.can_run:
        lines.append("this model cannot be run in this session; tests and drift are unavailable")
    t = sess.tests()
    if t:
        lines.append(f"tests `{t['name']}`: {t['passed']}/{t['total']} pass")
    d = sess.drift()
    if d["n"]:
        lim = (sess.task.get("preserve") or {}).get("max_kl")
        lines.append(f"drift from base on calibration text: mean KL {d['kl']} (max {d.get('kl_max')}), top-1 agreement "
                     f"{d['top1_agree']}" + (f"; limit {lim}" if lim is not None else ""))
    on = [p for p in ws.patches if p.enabled]
    lines.append(f"patches: {len(on)} enabled of {len(ws.patches)}")
    return "\n".join(lines)


@tool("overview", "The model, its hierarchy with parameter counts, every parameter's name / shape / role, the patches "
                  "applied so far, the task and the current test status. Call this first.")
def overview(sess) -> str:
    ws, st = sess.ws, sess.ws.struct
    out = [f"# {ws.name}", f"{ws.n_params():,} weights in {len(ws.names())} tensors"
           + (f" ({ws.meta['repo']})" if ws.meta.get("repo") else "") + ".", st.describe(), ws.notes().strip(), ""]
    if ws.cfg is not None:
        cfg = ws.cfg
        out += ["## Forward pass", "x = W_E[token] + W_pos[position]",
                "per block: x += sum_h (softmax_causal((ln1(x) W_Q[h] + b_Q[h]) . (ln1(x) W_K[h] + b_K[h]) / attn_scale) @ "
                "(ln1(x) W_V[h] + b_V[h])) W_O[h] + b_O" + ("" if cfg.attn_only else
                f";  x += {cfg.act_fn}(ln2(x) W_in + b_in) W_out + b_out"),
                "logits = ln_final(x) W_U + b_U" + ("" if cfg.use_ln else "   (this model has no LayerNorm: ln1 = ln2 = ln_final = identity)"),
                f"attn_scale = {cfg.attn_scale:.4g}", ""]
    out += ["## Hierarchy (address  params  |w|  Δ = edited)", hierarchy.outline(ws, 2 if st.n_layers <= 12 else 1), ""]
    if st.recognised:
        out.append("## Tensors and their layout (L = layer index)")
        seen = set()
        for name in ws.names():
            ps = st.parts.get(name)
            if not ps:
                continue
            generic = re.sub(r"(?<![A-Za-z_])(\d+)(?=\.)", "L", name, count=1) if any(name.startswith(L.prefix + ".") for L in st.layers) else name
            if generic in seen:
                continue
            seen.add(generic)
            out.append(f"{generic} {list(ws.shape(name))} {ws.store.dtype(name)}: {st.doc(name)}")
        out.append("Weight scripts reach these through views with a fixed layout whatever the checkpoint's: head(L, h).Q/.K/.V "
                   "[d_model, d_head], .O [d_head, d_model]; neuron(L, n); mlp(L).W_out [d_mlp, d_model]; or P(\"name\") natively.")
        if st.other:
            out.append(f"{len(st.other)} other tensors (outside the recognised decoder): list them with list_nodes raw:")
    else:
        out += ["## Tensors (by name; list more with list_nodes raw:<prefix>)"] + \
               [f"{n} {list(ws.shape(n))} {ws.store.dtype(n)}" for n in ws.names()[:60]]
    V = st.d_vocab
    if ws.tok is not None and 0 < V <= 120:
        out += ["", "## Vocabulary (id: token)", " ".join(f"{i}:{show(ws.tok.token_str(i))!r}" for i in range(V))]
    elif ws.tok is not None:
        out += ["", f"## Vocabulary: {V} tokens; use the `vocab` tool to look tokens up."]
    out += ["", "## Task", json.dumps(sess.task, indent=1), "", "## Status", _status(sess)]
    if ws.patches:
        out += ["", "## Patches"] + [f"{p.id} [{'on' if p.enabled else 'off'} x{p.strength:g}] {p.author}: {p.message}"
                                     for p in ws.patches[-20:]]
    if ws.groups:
        out += ["", "## Groups"] + [f"group:{k} = {v['members']}  {v.get('note', '')}" for k, v in ws.groups.items()]
    return "\n".join(out)


@tool("list_nodes", "List the children of a node in the hierarchy with parameter counts, norms and how much each has "
                    "been edited. Levels: model > section > block > component > unit group > unit (head / neuron) > tensor.",
      {"address": {**S, "description": "e.g. model, blocks.0, blocks.0.attn, blocks.0.mlp, blocks.0.mlp.neurons.0-63, L0H2, raw: (the checkpoint's own name tree)"},
       "depth": {**I, "description": "levels to descend (default 1)"}})
def list_nodes(sess, address: str = "model", depth: int = 1) -> str:
    ws = sess.ws
    lines: list[str] = []

    def walk(addr, d):
        for c in hierarchy.children(ws, addr):
            mark = f"  Δ{c['dnorm']:.3g}/{c['changed']} changed" if c["changed"] else ""
            shape = f" {c['shape']}" if "shape" in c else ""
            norm = f"  |w|={c['norm']:.4g}" if c.get("norm") is not None else ""
            lines.append(f"{'  ' * d}{c['addr']}{shape}  [{c['kind']}]  params={c['n_params']}{norm}{mark}")
            if d + 1 < depth and c["has_children"] and len(lines) < 400:
                walk(c["addr"], d + 1)

    walk(address, 0)
    return "\n".join(lines[:400]) or "(no children; this is a tensor: read it with read_weights)"


@tool("plan_chunks", "Split a selection of the model into chunks that each fit a token budget, cutting at the highest "
                     "level of the hierarchy that fits. Returns a manifest (chunk id, addresses, weights, estimated "
                     "tokens); read a chunk with read_chunk. format: auto | q (integers x scale, ~2.5 tokens/weight) | "
                     "g3 (3 significant digits, ~5.3) | g2 | sparse (top-8 per row) | svd (rank-4 factors + vocabulary "
                     "readings) | stats.",
      {"selection": {"type": "array", "items": S, "description": "addresses; default: the task scope"},
       "budget_tokens": {**I, "description": "per chunk, default 40000"}, "format": S})
def plan_chunks(sess, selection: list[str] | None = None, budget_tokens: int = 40_000, format: str = "auto") -> str:
    sel = selection or sess.task.get("scope") or ["model"]
    plan = chunks.plan(sess.ws, sel, budget_tokens, format)
    sess.plan = plan
    lines = [f"selection {sel}: {plan['n_params']} weights; format {plan['format']}; {len(plan['chunks'])} chunks, "
             f"~{plan['est_tokens']} tokens in total",
             "whole selection by format (estimated tokens): " + ", ".join(f"{k} {v}" for k, v in plan["totals_by_format"].items())]
    lines += [f"chunk {c['id']}: {c['title']}  weights={c['n_params']}  ~{c['est_tokens']} tokens" for c in plan["chunks"][:300]]
    if len(plan["chunks"]) > 300:
        lines.append(f"… {len(plan['chunks']) - 300} more chunks")
    return "\n".join(lines)


@tool("read_chunk", "Read one chunk from the last plan_chunks manifest: a header saying where the weights sit and what "
                    "they do, then the weights.",
      {"chunk_id": I, "variant": {**S, "description": "current (default) | base | delta"},
       "context": {**S, "description": "full (default) | brief | none"}}, ["chunk_id"])
def read_chunk(sess, chunk_id: int, variant: str = "current", context: str = "full") -> str:
    if not sess.plan:
        return "No chunk plan yet: call plan_chunks first."
    if not 0 <= chunk_id < len(sess.plan["chunks"]):
        return f"chunk_id must be between 0 and {len(sess.plan['chunks']) - 1}"
    return chunks.render_chunk(sess.ws, sess.plan["chunks"][chunk_id], variant, context)


@tool("read_weights", "Read the weights at an address as text. Refuses to return more than max_tokens (estimate) and "
                      "tells you how to narrow the request instead.",
      {"address": {**S, "description": "any address, e.g. blocks.0.attn.head.1, L0N12, blocks.0.mlp.W_in[:, 0:8], unembed.W_U[:, 17]"},
       "format": {**S, "description": "q (default) | g3 | g2 | sparse | svd | stats"},
       "variant": {**S, "description": "current (default) | base | delta"},
       "context": {**S, "description": "brief (default) | full | none"}, "max_tokens": I}, ["address"])
def read_weights(sess, address: str, format: str = "q", variant: str = "current", context: str = "brief",
                 max_tokens: int = 30_000) -> str:
    ws = sess.ws
    slices = ws.resolve(address)
    est = chunks.est_slices(ws, slices, format) + chunks.CHUNK_OVERHEAD
    if est > max_tokens:
        alt = {f: chunks.est_slices(ws, slices, f) for f in ("q", "sparse", "svd", "stats")}
        return (f"{address} is {A.count(ws, slices)} weights, about {est} tokens in format {format} (limit {max_tokens}). "
                f"Options: another format ({', '.join(f'{k} ~{v}' for k, v in alt.items())}); a narrower address or slice; "
                f"plan_chunks to split it; or raise max_tokens.")
    return chunks.render(ws, [address], format, variant, context)


def _no_wiring(ws, addr: str, win: list[int], w: dict, nid: str | None) -> str:
    """Why a wiring answer has no rows, and what to ask for instead (one line)."""
    s = ws.struct
    if not win:
        return "no rows: the wiring window has no layers."
    l = win[0]
    where = (f"the window is layer {l} alone" + (" (a model this wide gets a one-layer window)" if s.d_model > 2048 else "")
             if len(win) == 1 else f"the window is layers {win[0]}-{win[-1]}")
    if nid is not None and nid not in {n["id"] for n in w["nodes"]}:
        why = f"{nid} is not in the graph: layer {re.match(r'L(\d+)', nid).group(1)} has no such scored component"
    elif len(win) == 1 and not s.has_heads(l):
        why = f"{where}, and its attention has no heads the graph scores, so nothing in the window writes before its MLP reads"
    elif len(win) == 1:
        why = f"{where}, where only its heads can feed its MLP" + ("" if nid is None else f", and {nid} has no such edge")
    else:
        why = f"{where}, and no component in it writes before one that reads" + ("" if nid is None else f" {nid}")
    near = min((j for j in range(s.n_layers) if s.has_heads(j) and j != l), key=lambda j: (abs(j - l), j), default=None)
    m = re.search(r"\.mlp\.neuron\.(\d+)", addr)
    unit = f"L{l}N{m.group(1)}" if m else f"L{l}N<n>"
    alt = f"inspect blocks.{near}.attn.head.0 what=wiring for the nearest layer with heads; " if near is not None else ""
    return (f"no rows: {why}. Try {alt}for a neuron's partners in other layers, atlas action=unit address={unit} "
            f"(from the atlas wiring stage; atlas action=overview says whether it is built).")


@tool("inspect", "What a part of the model does, from its weights. what = card (a head's QK/OV circuits or a neuron's "
                 "read/write tokens) | svd (spectrum and singular vectors read through the vocabulary) | stats | "
                 "wiring (composition between components; give a head/MLP address to see only its edges) | "
                 "direction (read a residual direction given as a weight-script expression in `expr`).",
      {"address": S, "what": S, "expr": {**S, "description": "for what=direction, e.g. U(' cat') - U(' dog') or neuron(3, 5).w_out"},
       "k": I, "fetch": {**B, "description": "for a model whose weights are still on the Hub: allow downloading the embedding / "
                                             "unembedding matrices that vocabulary readings need"}})
def inspect(sess, address: str = "model", what: str = "auto", expr: str = "", k: int = 8, fetch: bool = False) -> str:
    ws = sess.ws
    need = ws.sem.vocab_bytes() if ws.struct.recognised else 0
    if need and not fetch and (what in ("auto", "card", "direction", "svd") or expr):
        if what in ("auto", "card") and not expr:
            what = "stats"
        else:
            return (f"This reading goes through the vocabulary, and the embedding / unembedding matrices ({need / 1e9:.1f} GB) are still "
                    f"on the Hub. Call again with fetch=true to download them, or use what=stats / read_weights.")
    if what == "direction" or expr:
        from .wscript import Interp
        it = Interp(ws)
        it.run(f"vdir = {expr} + 0")
        d = an.direction_tokens(ws, it.locals["vdir"], k)
        return (f"direction {expr}: norm {d['norm']:.4g}\n  promotes: {chunks._toks(d['promotes']['top'], k)}\n"
                f"  suppresses: {chunks._toks(d['promotes']['bottom'], k)}\n"
                f"  token embeddings along it: {chunks._toks(d['embeddings']['top'], k)}\n"
                f"  token embeddings against it: {chunks._toks(d['embeddings']['bottom'], k)}"
                + (f"\n  positions along it: {chunks._toks(d['positions']['top'], 6)}" if "positions" in d else ""))
    addr = A.normalize(address)
    parts = addr.split(".")
    kind = hierarchy.kind_of(ws, addr)
    if what == "auto":
        what = "card" if kind == "unit" else "svd" if kind == "tensor" and len(ws.resolve(addr)) == 1 else "stats"
    if what == "wiring":
        m = re.match(r"^blocks\.(\d+)(?:\.(?:attn\.head\.(\d+)|mlp))?", addr)
        focus = int(m.group(1)) if m else None
        win = an.default_window(ws, focus)
        if not win:
            return "no rows: the wiring window has no layers (this model has no layer the graph can score)."
        w = an.wiring(ws, 100000, layers=win)
        nid = None
        if m and "." in addr[7:]:
            nid = f"L{m.group(1)}H{m.group(2)}" if m.group(2) is not None else f"L{m.group(1)}MLP"
        es = [e for e in w["edges"] if nid is None or nid in (e["src"], e["dst"])][:40]
        note = "" if len(win) == ws.struct.n_layers else f" (layers {win[0]}-{win[-1]} only; give a block address to move the window)"
        head = ("composition score = |W_write W_read| / (|W_write| |W_read|) * sqrt(d_model); 1.0 is chance"
                + (f" (estimated through a {w['rank']}-dimensional random projection)" if w["sketched"] else "") + note)
        lines = [head] + [f"{e['src']} -> {e['dst']}.{e['kind']}  {e['score']:.2f}" for e in es]
        lines += [f"note: {x}" for x in w.get("limitations", [])]
        if not es:
            lines.append(_no_wiring(ws, addr, win, w, nid))
        return "\n".join(lines)
    if what == "card" and kind == "unit":
        return chunks.model_line(ws) + "\n" + chunks.where_header(ws, addr, "full")
    if what == "svd":
        sl = ws.resolve(addr)
        if len(sl) != 1:
            return f"{addr} is {len(sl)} tensors; svd needs one (e.g. {A.slice_str(sl[0])})"
        return chunks.render_tensor(ws, sl[0][0], sl[0][1], "svd")
    st = an.region_stats(ws, addr)
    c = st["current"]
    lines = [f"{addr}: {c['n']} weights, mean {c['mean']:.4g}, std {c['std']:.4g}, min {c['min']:.4g}, max {c['max']:.4g}, "
             f"norm {c['norm']:.4g}" + (f" (estimated from a sample of {st['sample']})" if st.get("sampled") else "")
             + f"; {st['changed']} differ from base (|Δ| {st['delta']['norm']:.4g})"]
    for name, idx in ws.resolve(addr)[:40]:
        lines.append(f"  {chunks.to_script(name, idx)} {list(A.slice_shape(ws.shape(name), idx))}")
    return "\n".join(lines)


@tool("vocab", "Tokens. Give `text` to see how it tokenises (ids and pieces), `query` to search the vocabulary for "
               "tokens containing a string, or `ids` to decode.",
      {"text": S, "query": S, "ids": {"type": "array", "items": I}})
def vocab(sess, text: str = "", query: str = "", ids: list[int] | None = None) -> str:
    tok = sess.ws.tok
    out = []
    if text:
        t = tok.encode_token(text)
        out.append(f"{text!r} -> {len(t)} token(s): " + " ".join(f"{i}:{show(tok.token_str(i))!r}" for i in t))
    if ids:
        out.append(" ".join(f"{i}:{show(tok.token_str(i))!r}" for i in ids))
    if query:
        hits = [i for i in range(tok.vocab_size) if query in tok.token_str(i)][:60]
        out.append(f"tokens containing {query!r}: " + (" ".join(f"{i}:{show(tok.token_str(i))!r}" for i in hits) or "none"))
    return "\n".join(out) or "give text, query or ids"


@tool("run", "Run the model on a prompt: top next-token predictions for the base and the edited model, and optionally a "
             "greedy continuation from each.",
      {"prompt": S, "max_new_tokens": {**I, "description": "0 (default) = next-token distribution only"},
       "top_k": I, "variant": {**S, "description": "both (default) | current | base"}}, ["prompt"])
def run(sess, prompt: str, max_new_tokens: int = 0, top_k: int = 8, variant: str = "both") -> str:
    ws = sess.ws
    if not ws.can_run:
        return "This model cannot be run in this session (weights not on disk, or no runnable config)."
    out = []
    for v in (["base", "current"] if variant == "both" else [variant]):
        lg = ws.runner.next_logits(ws.encode(prompt), v) if ws.kind == 'gguf' else ws.logits(prompt, v)[-1]
        p = lg.softmax(-1).topk(min(top_k, lg.numel()))
        line = f"{v:7s} next: " + "  ".join(f"{show(ws.tok.token_str(int(i)))!r} {float(x):.3f}" for x, i in zip(p.values, p.indices))
        out.append(line)
        if max_new_tokens:
            out.append(f"{v:7s} continues: {ws.complete(prompt, v, max_new_tokens=min(max_new_tokens, 200))!r}")
    return "\n".join(out)


@tool("trace", "Look inside one forward pass: logit lens after each layer, direct logit attribution of every head / "
               "MLP to the target token at the last position, the most effective neurons, and where each head attends "
               "from the last position. `target` defaults to the model's top prediction; `versus` makes it a logit difference.",
      {"prompt": S, "target": S, "versus": S, "variant": {**S, "description": "current (default) | base"}}, ["prompt"])
def trace(sess, prompt: str, target: str = "", versus: str = "", variant: str = "current") -> str:
    ws = sess.ws
    if not ws.can_run:
        return "This model cannot be run in this session (weights not on disk, or no runnable config)."
    t = an.trace(ws, prompt, variant, target or None, versus or None)
    T = len(t["tokens"])
    out = [f"tokens: " + " ".join(f"{i}:{s!r}" for i, s in enumerate(t["tokens"])),
           f"prediction at last position: " + "  ".join(f"{s!r} {p:.3f}" for s, p in t["predictions"][-1]),
           f"target {t['target']!r}" + (f" versus {t['versus']!r}" if t["versus"] else "") + f"; logit (difference) {t['dla_total']:.3f}",
           "logit lens (top token at the last position after each layer, and p(target)):"]
    out += [f"  {l['layer']:6s} {l['top'][-1]!r} p={l['top_p'][-1]:.3f}  p(target)={l['target_p'][-1]:.3f}" for l in t["lens"]]
    out.append("direct logit attribution at the last position (largest first):")
    for d in sorted(t["dla"], key=lambda d: -abs(d["value"]))[:16]:
        out.append(f"  {d['id']:8s} {d['value']:+.3f}")
    if t["neurons"]:
        out.append("neurons with the largest direct effect: " + ", ".join(
            f"{n['addr']} (act {n['act']:.2f}, effect {n['effect']:+.2f})" for n in t["neurons"][:12]))
    if any(p for p in t["patterns"]):
        out.append("attention from the last position (head: source positions with weight > 0.15):")
        shown = 0
        for l, layer in enumerate(t["patterns"]):
            for h, pat in enumerate(layer or []):
                srcs = [(j, w) for j, w in enumerate(pat[T - 1]) if w > 0.15]
                if srcs and shown < 80:
                    shown += 1
                    out.append(f"  L{l}H{h}: " + ", ".join(f"{j}:{t['tokens'][j]!r} {w:.2f}" for j, w in srcs))
    return "\n".join(out)


@tool("atlas", "The map of the model's units, built once from its weights (and, when the model can be run, checked "
               "against a forward pass). Consult it BEFORE editing: it says what each neuron reads and writes, which neurons "
               "carry a token or concept, and which ones push an answer in a given prompt. action = overview (default: classes, "
               "stages, per-layer table, how well the map held up) | unit (address L<layer>N<n>) | search (selector: neurons by "
               "word / token / concept / mech / role / layers / effect, strongest first) | concepts (token concepts; give `group`) "
               "| concept (id: its tokens and neurons) | tokens (query: find tokens and their concept) | attribute (prompt, target, "
               "optional versus and selector: direct effect of every neuron on the target's score at the last position) | "
               "shift (how a token's score can be shifted everywhere) | build (build or continue the atlas; stages optional).",
      {"action": S, "address": {**S, "description": "for unit: L<layer>N<n>"},
       "selector": {"type": "object", "description": "e.g. {\"word\": \"red\", \"layers\": [19, 31], \"effect\": \"promote\", \"top\": 20}"},
       "id": I, "group": I, "query": S, "prompt": S, "target": S, "versus": S, "top": I,
       "stages": {"type": "array", "items": S, "description": "for build: weights | wiring | observed"}})
def atlas_tool(sess, action: str = "overview", address: str = "", selector: dict | None = None, id: int | None = None,
               group: int | None = None, query: str = "", prompt: str = "", target: str = "", versus: str = "", top: int = 16,
               stages: list[str] | None = None) -> str:
    import re
    from . import atlas as at
    from .atlas import guide, report
    ws = sess.ws
    A = at.Atlas.of(ws)
    try:
        if action == "build":
            lines: list[str] = []
            # The atlas build lock first (refused at once if a build of this atlas is running anywhere), then the
            # workspace lock that the browser's build job holds too, so edits and Run wait for this build as they
            # wait for that one. Both are re-entrant for this thread; nothing that calls a tool holds either.
            with at.build_lock(ws), ws.lock:
                at.build(ws, stages=stages, log=lines.append)
            # the lines that matter whatever the build's length (the estimate, continuing or starting over), then the end
            head = [x for x in lines[:-12] if "neuron partners (" in x or "saved wiring progress" in x or "continuing" in x]
            return "\n".join(head + lines[-12:]) + "\n\n" + report.overview(at.Atlas.of(ws))
        if action == "overview" or not A.exists:
            return report.overview(A)
        if action == "unit":
            m = re.fullmatch(r"\s*(?:L(\d+)N(\d+)|blocks\.(\d+)\.mlp\.neuron\.(\d+))\s*", address)
            if not m:
                return "address must be L<layer>N<n>"
            l, n = (int(x) for x in m.groups() if x is not None)
            return report.unit(A, l, n)
        if action == "search":
            if not selector:
                return "give a selector, e.g. {\"word\": \"red\", \"effect\": \"promote\"}.\n" + at.select.__doc__
            return report.selection(A, selector, at.select(A, selector), top)
        if action == "concepts":
            return report.concepts(A, group)
        if action == "concept":
            c = A.concept(int(id), top)
            return (f"C{c['id']} ({c['size']:,} tokens): " + " ".join(repr(show(t)) for t in c["text"]) + "\n"
                    + "\n".join(f"  {k}: " + " ".join(f"L{l}:{n}" for l, n in v.items()) for k, v in c["by_layer"].items()) + "\n"
                    + "\n".join(f"  L{r['layer']}N{r['neuron']} {r['end']} (z {r['z']})" for r in c["neurons"]))
        if action == "tokens":
            return "\n".join(f"  {r['id']:7d} {show(r['token'])!r}  concept C{r['concept']}" for r in A.find_tokens(query)) or "no token matches"
        if action == "attribute":
            if not ws.can_run:
                return "This model cannot be run in this session, so attribution is not available."
            if not prompt or not target:
                return "attribute needs a prompt and a target token"
            return report.attribution(guide.attribute(ws, prompt, target, versus or None, top=top, group=selector))
        if action == "shift":
            return guide.shift_plan(ws)["note"]
    except (ValueError, KeyError, at.Busy) as e:
        return f"error: {e}"
    return f"unknown action {action!r}"


def _report(r: dict) -> str:
    if not r.get("ok"):
        return "ERROR: " + r.get("error", "unknown error") + ("\n" + r["output"] if r.get("output") else "")
    out = []
    if r.get("event"):out.append("RUNTIME: "+r["event"]["message"])
    if r.get("output"):
        out.append("output:\n" + r["output"])
    if not r.get("changed"):
        if r.get("committed"):out.append("COMMITTED as patch "+r["patch"])
        out.append(r.get("note", "Patch recorded; detailed change summary unavailable." if r.get("committed") else "no weights changed"))
        return "\n".join(out)
    out.append(("COMMITTED as patch " + r["patch"]) if r.get("committed") else "DRY RUN (nothing committed)")
    out.append("changed: " + "; ".join(
        f"{c['param']} {c['changed']}/{c['of']} (|Δ| {c['norm']}, max {c['max_abs']}"
        + (f", rank {c['rank']}" if c.get("rank") else "") + f"; stored as {'+'.join(c['kinds'])}, {c['stored']} numbers)" for c in r["changed"]))
    if r.get("not_run"):
        out.append("NOT TESTED: " + r["not_run"])
    if "rome" in r:
        x = r["rome"]
        out.append(f"rome: layer {x['layer']} at the {x['at']} position ({x['pos']}), p(target) {x['p_before']:.4f} -> "
                   f"{x['p_after']:.4f}, {'target is now the top prediction' if x['success'] else 'target is NOT yet the top prediction'}")
    if "tests" in r:
        t = r["tests"]
        out.append(f"tests: {t['before'][0]}/{t['before'][1]} -> {t['after'][0]}/{t['after'][1]}"
                   + (f"; newly passing {len(t['fixed'])}" if t["fixed"] else "")
                   + (f"; NEWLY FAILING ({len(t['broken'])}): {t['broken'][:15]}" if t["broken"] else ""))
        if t["after"][0] < t["after"][1]:
            out.append(t["summary"])
    d = r["drift"]
    if d.get("n"):
        out.append(f"drift: mean KL {d['kl']} (max {d.get('kl_max')}), top-1 agreement {d['top1_agree']}"
                   + (f"; limit {d['limit']} {'ok' if d['within_limit'] else 'EXCEEDED'}" if "limit" in d else ""))
    return "\n".join(out)


@tool("edit", "Apply a weight script to the current weights. The script is checked against the task scope, the test "
              "suite and the drift measurement are run on the result, and (unless dry_run) the change is committed as "
              "a patch. Returns what changed, print() output, test results before -> after, and drift. Scripts may "
              "also be used read-only (print only) to compute things about the weights.",
      {"script": S, "message": {**S, "description": "one line: what this edit does and why"},
       "dry_run": {**B, "description": "evaluate without committing (default false)"}}, ["script"])
def edit(sess, script: str, message: str = "", dry_run: bool = False) -> str:
    return _report(sess.apply(script, message, author="programmer", dry_run=dry_run))


@tool("rome", "Rank-one fact edit (ROME, Meng et al. 2022): change one MLP's W_out so that `prompt` continues with "
              "`target`, disturbing other inputs as little as possible. `subject` is the part of the prompt the fact is "
              "about. layer / at (subject | last) default to trying all and keeping the cleanest.",
      {"prompt": S, "subject": S, "target": S, "layer": I, "at": S, "message": S, "dry_run": B},
      ["prompt", "subject", "target"])
def rome(sess, prompt: str, subject: str, target: str, layer: int | None = None, at: str = "auto", message: str = "",
         dry_run: bool = False) -> str:
    return _report(sess.rome(prompt, subject, target, layer, message, "programmer", dry_run, at))


@tool("run_tests", "Run a test suite (default: the task's) on the edited or the base model. show = failures (default) | all | summary.",
      {"suite": S, "variant": S, "show": S})
def run_tests(sess, suite: str = "", variant: str = "current", show: str = "failures") -> str | dict:
    refusal = None
    try:
        r = sess.tests(suite or None, variant)
    except RuntimeRefusal as exc:
        refusal, r = exc.record, getattr(exc, "report", None)
    if r is None:
        if refusal:
            return dict(status="not_run", event=refusal, report=None, summary="NOT RUN: "+refusal["message"])
        return ("This model cannot be run in this session." if not sess.ws.can_run else
                "No test suite: " + json.dumps(testsuite.list_suites(sess.ws)))
    if show == "all":
        lines = [f"{r['name']}: {r['passed']}/{r['total']} pass" + (f"; {r['not_run']} NOT RUN" if r.get('not_run') else "")]
        for c in r["cases"][:300]:
            label = "NOT RUN" if c.get("status") == "not_run" or c.get("pass") is None else ("ok" if c["pass"] else "FAIL")
            lines.append(f"  {label} {c['name']!r} got {c.get('got')!r} value {c.get('value')}")
        summary = "\n".join(lines)
    else:
        summary = testsuite.summarize(r, 0 if show == "summary" else 25)
    if refusal or r.get("not_run"):
        refusal = refusal or next((c.get("event") for c in r["cases"] if c.get("status") == "not_run"), None)
        return dict(status="not_run", event=refusal, report=r, summary=summary)
    return summary


@tool("patches", "The patch stack. action = list (default) | show (the script / source of one patch) | revert (delete a "
                 "patch) | enable | disable | strength (scale a patch's delta by `value`; 0.5 halves it, -1 negates it).",
      {"action": S, "id": S, "value": {"type": "number"}})
def patches(sess, action: str = "list", id: str = "", value: float | None = None) -> str:
    ws = sess.ws
    if action == "list":
        if not ws.patches:
            return "no patches: the weights are the base model"
        return "\n".join(f"{p.id} [{'on' if p.enabled else 'off'} x{p.strength:g}] {p.author}: {p.message}  "
                         f"({sum(x['changed'] for x in p.stats())} weights in {', '.join(p.deltas)})" for p in ws.patches)
    if not id:
        return "give the patch id"
    if action == "show":
        return json.dumps(ws.patch(id).meta(), indent=1)
    if action == "revert":
        ws.delete_patch(id)
    elif action in ("enable", "disable"):
        ws.set_patch(id, enabled=action == "enable")
    elif action == "strength":
        ws.set_patch(id, strength=float(value if value is not None else 1.0))
    else:
        return f"unknown action {action!r}"
    sess.note("programmer", f"patch {action}", id)
    return f"{action} {id}: done\n" + _status(sess)


def _agent_dir(sess) -> str:
    d = os.path.join(sess.ws.dir, "agent", "weights")
    os.makedirs(d, exist_ok=True)
    return d


@tool("checkout_files", "File route: write the parameters a selection touches as .npy files (whole tensors, float32) "
                        "into the working directory's weights/ folder, for editing with your own tools (numpy). Then "
                        "call commit_files.", {"selection": {"type": "array", "items": S}})
def checkout_files(sess, selection: list[str] | None = None) -> str:
    r = edits.checkout(sess.ws, selection or sess.task.get("scope") or ["model"], _agent_dir(sess))
    return f"wrote {len(r['files'])} files to {r['dir']} (see INDEX.md): " + ", ".join(r["files"][:40])


@tool("commit_files", "File route: turn whatever changed in the checked-out .npy files into a patch (scope-checked, "
                      "tests and drift reported, like `edit`).", {"message": S, "dry_run": B})
@job_scope('edit')
def commit_files(sess, message: str = "", dry_run: bool = False) -> str:
    try:
        deltas = edits.read_files(sess.ws, _agent_dir(sess))
    except RuntimeRefusal:
        raise
    except Exception as e:
        return f"ERROR: {e}"
    return _report(sess.commit_deltas(deltas, message or "edit from files", "programmer", {"kind": "files"}, dry_run))


@tool("finish", "Report that you are done: a short account of what you changed, why it works, and anything left open.",
      {"summary": S}, ["summary"])
def finish(sess, summary: str) -> str:
    sess.note("programmer", "finish", summary)
    sess.publish({"type": "finish", "summary": summary})
    return "recorded.\n" + _status(sess)
