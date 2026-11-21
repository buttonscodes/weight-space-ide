"""Probes: small questions about neurons, answered where the model runs.

Each probe is a named model-worker job (``probe_reads``, ``probe_fires``, ``probe_contrast``,
``probe_next``) with an in-process equivalent: the same function, called directly when the
model runs in this process.  Reductions happen next to the model, so only the numbers asked
for come back (never a whole activation tensor).  Inputs are token ids; the server encodes
text and adds token strings.  Probes read activations; none of them writes a weight.

    probe_reads     activation of given neurons at every token, for base and/or current
                    (current steered when a steering wire form is given; values after the push)
    probe_fires     the neurons most active at one position (raw activation, not normalised)
    probe_contrast  mean over texts A of each neuron's max activation, minus the same over B
    probe_next      next-token distribution with and without steering: KL, top tokens, movers
"""
from __future__ import annotations

import json
import math
import os

import torch

from . import steering as S
from .runtime_events import job_scope
from .worker_jobs import _ids, worker_job

MAX_TOKENS = 256          # per text, reads / fires / contrast
MAX_NEXT_TOKENS = 512     # the next-token prompt
MAX_NEURONS = 16          # neurons per reads call
MAX_K = 200
MAX_TEXTS = 16            # per contrast set
MAX_TOP = 50
MOVER_FLOOR = 1e-4        # tokens below this probability in both distributions are not movers
VARIANTS = ('base', 'current')


class ProbeError(ValueError):
    pass


# ------------------------------------------------------------------ as a chat message
# A probe text can be read as the model sees a chat: one user turn rendered with the model's own chat
# template, generation prompt included, exactly as /api/chat renders a conversation (runner.chat_prompt).

def chat_template(ws):
    """Whether this model has its own chat template, found from its files without loading it -> (bool, reason).
    A GGUF's is its ``tokenizer.chat_template``, which opening the GGUF saved with the converted tokenizer in the
    workspace's ``model/``; the same files are looked for there."""
    if ws.kind == 'wsp':
        return False, 'this built-in model has no chat template'
    gguf = ws.kind == 'gguf'
    found = "the GGUF's own chat template" if gguf else "the model's own chat template"
    missing = 'this GGUF carries no chat template (tokenizer.chat_template)' if gguf else 'this model ships no chat template'
    d = getattr(ws, 'model_dir', None)
    if not d or not os.path.isdir(d):
        return False, missing if gguf else "the model's tokenizer files are not on this machine"
    if any(os.path.isfile(os.path.join(d, f)) for f in ('chat_template.jinja', 'chat_template.json')):
        return True, found
    try:
        with open(os.path.join(d, 'tokenizer_config.json')) as f:
            if json.load(f).get('chat_template'):
                return True, found
    except (OSError, ValueError, AttributeError):
        pass
    return False, missing


def check_chat_options(options):
    """The chat template options of a request (Chat, probes and direction measurements read as a chat message)."""
    options = options or {}
    if not isinstance(options, dict) or set(options) - {'enable_thinking', 'reasoning_effort', 'preserve_thinking'}:
        raise ValueError('Unsupported chat template options')
    if any(k in options and not isinstance(options[k], bool) for k in ('enable_thinking', 'preserve_thinking')):
        raise ValueError('Thinking switches must be booleans')
    if options.get('reasoning_effort', 'xhigh') not in ('xhigh', 'medium', 'low'):
        raise ValueError('Reasoning effort must be xhigh, medium or low')
    return options


def chat_encode(ws):
    """How a templated prompt is tokenized, as Chat tokenizes it: without the tokenizer's own special tokens
    for a BPE vocabulary (the template already holds them), else the vocabulary's plain encoding."""
    return (lambda t: ws.tok.encode(t, special=False)) if ws.tok.to_dict()["type"] == "bpe" else ws.tok.encode


def render_chat(ws, text, options, system=None):
    """After the runner exists: ``text`` as one user turn (after an optional system message), rendered by the
    runner's ``chat_prompt`` as /api/chat renders a conversation (generation prompt included) -> the rendering."""
    messages = ([{"role": "system", "content": system}] if system is not None else []) + [{"role": "user", "content": text}]
    rendered = ws.runner.chat_prompt(messages, **options)
    if rendered is None:
        raise ValueError("'as a chat message' needs the model's own chat template: this model has none")
    return rendered


def template_mask(encode, rendered, text, ids):
    """For each token of the rendered chat, True when it is the template's and not the text's -> list, or None
    when the boundary cannot be told (a token that spans the text and the template)."""
    start = rendered.rfind(text)
    if start < 0:
        return None
    head = encode(rendered[:start]) if start else []
    upto = encode(rendered[:start + len(text)])
    if list(ids[:len(head)]) != list(head) or list(ids[:len(upto)]) != list(upto):
        return None
    return [i < len(head) or i >= len(upto) for i in range(len(ids))]


def _text_ids(ids, limit=MAX_TOKENS, what='text'):
    ids = _ids(list(ids) if isinstance(ids, (list, tuple)) else ids)
    if not 1 <= len(ids) <= limit:
        raise ProbeError(f"{what} must have 1..{limit} tokens (it has {len(ids)})")
    return ids


def _int(value, lo, hi, what):
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ProbeError(f"{what} must be an integer from {lo} to {hi}")
    return value


def _variant(v):
    if v not in VARIANTS:
        raise ProbeError("variant must be 'base' or 'current'")
    return v


def _neuron_layers(ws):
    return [l for l in range(ws.struct.n_layers) if S.neurons_hookable(ws, l)]


def _post(l):
    return f'blocks.{l}.mlp.hook_post'


def _run(ws, ids, variant, hooks):
    with torch.no_grad():
        return ws.runner.run(ids, variant, None, hooks=hooks)[0]


# ------------------------------------------------------------------ input checks
# Each probe's arguments are checked by a function that needs only the workspace's structure and
# tokenizer, never the model: the server calls it before it loads anything, and the job calls it
# again where the model runs.
def check_reads(ws, ids, neurons, variants=('current',)):
    ids = _text_ids(ids)
    if not isinstance(neurons, (list, tuple)) or not 1 <= len(neurons) <= MAX_NEURONS:
        raise ProbeError(f"neurons must list 1..{MAX_NEURONS} neurons")
    pairs = []
    for item in neurons:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not all(type(v) is int for v in item):
            raise ProbeError("each neuron must be [layer, index]")
        S._neuron_ok(ws, item[0], item[1])
        pairs.append((item[0], item[1]))
    variants = list(variants)
    if not 1 <= len(variants) <= 2 or len(set(variants)) != len(variants):
        raise ProbeError("variants must be 'base', 'current' or both")
    for v in variants:
        _variant(v)
    return ids, pairs, variants


def check_position(position, n):
    """A position in a text of at most ``n`` tokens, before the text's own length is known."""
    return _int(position, -n, n - 1, 'position')


def check_fires(ws, ids, position=-1, k=20, sign='both', variant='current'):
    ids = _text_ids(ids)
    position = _int(position, -len(ids), len(ids) - 1, 'position')
    k = _int(k, 1, MAX_K, 'k')
    if sign not in ('+', '-', 'both'):
        raise ProbeError("sign must be '+', '-' or 'both'")
    return ids, position, k, sign, _variant(variant)


def check_contrast(ws, a, b, k=20, variant='current'):
    k = _int(k, 1, MAX_K, 'k')
    _variant(variant)
    sets = []
    for label, texts in (('a', a), ('b', b)):
        if not isinstance(texts, (list, tuple)) or not 1 <= len(texts) <= MAX_TEXTS:
            raise ProbeError(f"set {label.upper()} must have 1..{MAX_TEXTS} texts")
        sets.append([_text_ids(t, what=f'a text of set {label.upper()}') for t in texts])
    return sets, k, variant


def check_next(ws, ids, variant='current', top=10):
    ids = _text_ids(ids, MAX_NEXT_TOKENS, 'prompt')
    return ids, _variant(variant), _int(top, 1, MAX_TOP, 'top')


# ------------------------------------------------------------------ reads
@job_scope('probe_reads')
@worker_job('probe_reads')
def reads(ws, ids, neurons, variants=('current',), steer=None):
    """-> {'tokens': T, 'neurons': [[l, n]], 'values': {variant: [[a_t for each token] per neuron]}, 'steered': bool}"""
    ids, pairs, variants = check_reads(ws, ids, neurons, variants)
    wire = S.as_wire(ws, steer)
    by_layer = {}
    for k, (l, n) in enumerate(pairs):
        by_layer.setdefault(l, []).append((k, n))
    out = {}
    for v in variants:
        steer_hooks = S.hooks_for(ws, wire, v) if v == 'current' and not S.is_empty(wire) else {}
        got = [None] * len(pairs)
        hooks = {}
        for l, items in by_layer.items():
            def pick(x, name, items=items):
                cols = torch.tensor([n for _, n in items], dtype=torch.long, device=x.device)
                vals = x[0].index_select(-1, cols).float().cpu().T          # [neurons, T]
                for row, (k, _) in zip(vals, items):
                    got[k] = [float(a) for a in row.tolist()]
                return None
            hooks[_post(l)] = S.compose(steer_hooks.get(_post(l)), pick)
        for name, fn in steer_hooks.items():
            hooks.setdefault(name, fn)
        _run(ws, ids, v, hooks)
        out[v] = got
    return {'tokens': len(ids), 'neurons': [list(p) for p in pairs], 'values': out,
            'steered': 'current' in variants and not S.is_empty(wire)}


# ------------------------------------------------------------------ fires
@job_scope('probe_fires')
@worker_job('probe_fires')
def fires(ws, ids, position=-1, k=20, sign='both', variant='current'):
    """The k neurons with the largest activation at one position (default the last token)."""
    ids, position, k, sign, variant = check_fires(ws, ids, position, k, sign, variant)
    at = position % len(ids)
    best = []

    def rank(l):
        def hook(x, name):
            a = x[0, at].float().cpu()
            score = a if sign == '+' else (-a if sign == '-' else a.abs())
            top = score.topk(min(k, score.numel()))
            best.extend((float(s), l, int(i), float(a[i])) for s, i in zip(top.values.tolist(), top.indices.tolist()))
            return None
        return hook
    _run(ws, ids, variant, {_post(l): rank(l) for l in _neuron_layers(ws)})
    best.sort(key=lambda r: (-r[0], r[1], r[2]))
    return {'position': at, 'token': ids[at], 'sign': sign, 'variant': variant,
            'ranking': 'raw activation at this position (not normalised per neuron)',
            'neurons': [{'layer': l, 'neuron': n, 'addr': f'L{l}N{n}', 'value': v} for _, l, n, v in best[:k]]}


# ------------------------------------------------------------------ contrast
def _maxes(ws, ids, variant, layers):
    """{layer: per-neuron max activation over the text, position 0 excluded when there are more}"""
    out = {}

    def keep(l):
        def hook(x, name):
            a = x[0, 1:] if x.shape[1] > 1 else x[0]
            out[l] = a.float().amax(0).cpu().double()
            return None
        return hook
    _run(ws, ids, variant, {_post(l): keep(l) for l in layers})
    return out


@job_scope('probe_contrast')
@worker_job('probe_contrast')
def contrast(ws, a, b, k=20, variant='current'):
    """Neurons that fire more on texts A than on texts B (and the reverse): mean of per-text max."""
    sets, k, variant = check_contrast(ws, a, b, k, variant)
    layers = _neuron_layers(ws)
    means = []
    for texts in sets:
        total = {}
        for ids in texts:
            for l, m in _maxes(ws, ids, variant, layers).items():
                total[l] = m if l not in total else total[l] + m
        means.append({l: t / len(texts) for l, t in total.items()})
    rows = []
    for l in layers:
        if l in means[0] and l in means[1]:
            d = means[0][l] - means[1][l]
            rows.append((l, d, means[0][l], means[1][l]))
    if not rows:
        return {'a': len(sets[0]), 'b': len(sets[1]), 'variant': variant, 'up': [], 'down': []}
    diffs = torch.cat([d for _, d, _, _ in rows])
    owner = [(l, n) for l, d, _, _ in rows for n in range(d.numel())]
    av = torch.cat([x for _, _, x, _ in rows])
    bv = torch.cat([x for _, _, _, x in rows])

    def pick(order):
        out = []
        for i in order.tolist():
            l, n = owner[i]
            out.append({'layer': l, 'neuron': n, 'addr': f'L{l}N{n}', 'diff': float(diffs[i]), 'a': float(av[i]), 'b': float(bv[i])})
        return out
    kk = min(k, diffs.numel())
    up = diffs.topk(kk).indices
    down = (-diffs).topk(kk).indices
    return {'a': len(sets[0]), 'b': len(sets[1]), 'variant': variant,
            'measure': 'mean over texts of max activation (position 0 excluded); A minus B',
            'up': pick(up), 'down': pick(down)}


# ------------------------------------------------------------------ next
def _next_logits(ws, ids, variant, wire):
    r = ws.runner
    if hasattr(r, 'next_logits'):
        kw = {} if S.is_empty(wire) else {'steer': wire}
        with torch.no_grad():
            return r.next_logits(ids, variant, None, **kw).float().reshape(-1)
    hooks = S.hooks_for(ws, wire, variant) if not S.is_empty(wire) else None
    return _run(ws, ids, variant, hooks)[0, -1].float()


def compare(p_logits, q_logits, top=10):
    """Unsteered vs steered next-token distributions -> KL(p || q) in nats, tops and movers."""
    lp = torch.log_softmax(p_logits.double(), -1)
    lq = torch.log_softmax(q_logits.double(), -1)
    p, q = lp.exp(), lq.exp()
    mask = p > 0
    kl = float((p[mask] * (lp[mask] - lq[mask])).sum())
    if not math.isfinite(kl):
        kl = None                 # the steered model gives zero probability where the unsteered does not
    else:
        kl = max(0.0, kl)

    def tops(prob, logp):
        t = prob.topk(min(top, prob.numel()))
        return [{'id': int(i), 'p': float(v), 'logp': float(logp[i])} for v, i in zip(t.values, t.indices)]
    live = (p >= MOVER_FLOOR) | (q >= MOVER_FLOOR)
    delta = torch.where(live, lq - lp, torch.zeros_like(lp))
    delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    n = min(top, int(live.sum()))

    def movers(order):
        return [{'id': int(i), 'delta_logp': float(delta[i]), 'p': float(p[i]), 'q': float(q[i])}
                for i in order[:n].tolist() if live[i] and delta[i] != 0]
    rose = movers(torch.argsort(delta, descending=True))
    fell = movers(torch.argsort(delta))
    return {'kl': kl, 'top_unsteered': tops(p, lp), 'top_steered': tops(q, lq), 'rose': [m for m in rose if m['delta_logp'] > 0],
            'fell': [m for m in fell if m['delta_logp'] < 0]}


@job_scope('probe_next')
@worker_job('probe_next')
def next_token(ws, ids, variant='current', steer=None, top=10):
    """Next-token distribution of ``variant`` with and without ``steer``."""
    ids, variant, top = check_next(ws, ids, variant, top)
    wire = S.as_wire(ws, steer)
    if variant == 'base' and not S.is_empty(wire):
        raise S.SteeringError("the base model is never steered; steer the current variant")
    p = _next_logits(ws, ids, variant, None)
    q = p.clone() if S.is_empty(wire) else _next_logits(ws, ids, variant, wire)
    out = compare(p, q, top)
    out.update(tokens=len(ids), variant=variant, steered=not S.is_empty(wire), measure='KL(unsteered || steered), nats')
    return out
