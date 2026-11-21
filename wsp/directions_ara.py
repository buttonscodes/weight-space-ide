"""Arbitrary-rank ablation (ARA): fit a low-rank change to each output matrix from saved module inputs and outputs.

For each layer of a range and each of its two output matrices (``attn.o_proj``, ``mlp.down_proj``), a change
``delta_V = A^T @ B^T`` (``A [r, D_in]``, ``B [d, r]``; in the adapter convention ``delta_W = B @ A``) is fitted with
L-BFGS so that the matrix's outputs on the "good" prompts stay where they were and its outputs on the "bad" prompts
move towards the good prompts' outputs. The inputs and outputs are those a measurement with ``module_io`` saved
(wsp.directions). The method is "Arbitrary-Rank Ablation" (Weidmann 2026, the default method of the tool Heretic);
it was written here from a description of the method, none of that tool's code. Parameter names are that tool's,
so its parameters can be pasted in unchanged. The fit is sensitive to the last digit of its inputs, so the loss is
evaluated in the adapter layout (W = V^T) and in the order Heretic evaluates it: on the same numbers and the same
torch the two fits are then identical bit for bit (scripts/heretic_fit_check.py). Formulas and costs:
docs/DIRECTIONS.md.

Nothing here writes a weight: ``ablate`` hands each piece to ``add(view, A, B)``, the weight-script interpreter's
(``ablate_ara`` in wsp.wscript), so the edit goes through Session.apply like any other script.
"""
from __future__ import annotations

import json
import math

import torch

from . import directions as DR
from .runtime_events import edit_preflight

FIELDS = ('method', 'direction', 'start_layer_index', 'end_layer_index', 'preserve_good_behavior_weight',
          'steer_bad_behavior_weight', 'overcorrect_relative_weight', 'neighbor_count', 'preserve_row_magnitudes',
          'lora_rank', 'n_optimization_steps', 'learning_rate', 'max_iter', 'history_size', 'seed', 'expert_modules')
DEFAULTS = {'start_layer_index': 0, 'end_layer_index': None, 'preserve_good_behavior_weight': 1.0,
            'steer_bad_behavior_weight': 0.01, 'overcorrect_relative_weight': 0.0, 'neighbor_count': 1,
            'preserve_row_magnitudes': True, 'lora_rank': 50, 'n_optimization_steps': 5, 'learning_rate': 1.0,
            'max_iter': 20, 'history_size': 10, 'seed': 0}
MAX_LOSS_WEIGHT = 1000.0
MAX_OVERCORRECT = 10.0
MAX_NEIGHBORS = 64
MAX_LORA_RANK = 256
MAX_STEPS = 50
MAX_ITER = 100
MAX_HISTORY = 100
MAX_LEARNING_RATE = 10.0
MAX_SEED = 2 ** 63 - 1
MISMATCH_LIMIT = 0.05          # leaves room for a model that runs in bfloat16
MISMATCH_ADVICE = "the saved inputs and outputs do not match this matrix: measure again on these weights, or switch other patches off"
NOTHING_TO_FIT = "steer_bad_behavior_weight is 0: nothing to fit"
ROUTED_REFUSAL = ("layer {layer} {component}: ARA cannot fit routed experts yet (an expert's inputs exist only for the "
                  "prompts routed to it); pass expert_modules=\"shared\" to fit the shared expert only")


def _int(params, field, lo, hi):
    v = params.get(field, DEFAULTS[field])
    if isinstance(v, bool) or not isinstance(v, int) or v < lo or (hi is not None and v > hi):
        raise DR.DirectionError(f"{field}: must be an integer from {lo}" + (f" to {hi}" if hi is not None else " up"))
    return v


def _weight(params, field, hi):
    v = params.get(field, DEFAULTS[field])
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= hi:
        raise DR.DirectionError(f"{field}: must be a finite number from 0 to {hi:g}")
    return float(v)


def normalize(params) -> dict:
    """The ARA parameters checked without the model or the measurement -> a dict with every field."""
    if not isinstance(params, dict):
        raise DR.DirectionError("the ablation parameters must be an object")
    unknown = sorted(set(params) - set(FIELDS), key=str)
    if unknown:
        raise DR.DirectionError(f"{unknown[0]}: unknown field; the fields are {', '.join(FIELDS)}")
    if params.get('method', 'ara') != 'ara':
        raise DR.DirectionError("method: must be ara for these parameters")
    if 'direction' not in params:
        raise DR.DirectionError("direction: required (the name of a saved measurement)")
    try:
        name = DR.check_name(params['direction'])
    except DR.DirectionError as e:
        raise DR.DirectionError(f"direction: {e}") from None
    start = _int(params, 'start_layer_index', 0, None)
    end = params.get('end_layer_index', None)
    if end is not None:
        if isinstance(end, bool) or not isinstance(end, int) or end < 1:
            raise DR.DirectionError("end_layer_index: must be null (every layer from start_layer_index) or an integer "
                                    "from 1 up (exclusive)")
        if not start < end:
            raise DR.DirectionError(f"end_layer_index: must be above start_layer_index ({start}); the end is exclusive")
    p = {'method': 'ara', 'direction': name, 'start_layer_index': start, 'end_layer_index': end,
         'preserve_good_behavior_weight': _weight(params, 'preserve_good_behavior_weight', MAX_LOSS_WEIGHT),
         'steer_bad_behavior_weight': _weight(params, 'steer_bad_behavior_weight', MAX_LOSS_WEIGHT),
         'overcorrect_relative_weight': _weight(params, 'overcorrect_relative_weight', MAX_OVERCORRECT),
         'neighbor_count': _int(params, 'neighbor_count', 1, MAX_NEIGHBORS)}
    prm = params.get('preserve_row_magnitudes', DEFAULTS['preserve_row_magnitudes'])
    if not isinstance(prm, bool):
        raise DR.DirectionError("preserve_row_magnitudes: must be true or false")
    p['preserve_row_magnitudes'] = prm
    p['lora_rank'] = _int(params, 'lora_rank', 1, MAX_LORA_RANK)
    p['n_optimization_steps'] = _int(params, 'n_optimization_steps', 1, MAX_STEPS)
    lr = params.get('learning_rate', DEFAULTS['learning_rate'])
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or not 0 < lr <= MAX_LEARNING_RATE:
        raise DR.DirectionError(f"learning_rate: must be a number above 0 and at most {MAX_LEARNING_RATE:g}")
    p['learning_rate'] = float(lr)
    p['max_iter'] = _int(params, 'max_iter', 1, MAX_ITER)
    p['history_size'] = _int(params, 'history_size', 1, MAX_HISTORY)
    p['seed'] = _int(params, 'seed', 0, MAX_SEED)
    em = DR.expert_modules_of(params)
    if em != 'all':                 # only when not the default, so Heretic's parameters paste in and compare unchanged
        p['expert_modules'] = em
    return p


def scratch_bytes(d_in, d, n_good, n_bad, row, r, history, preserve) -> int:
    """What the fit of one component asks the edit admission for, in bytes (float32): the matrix whole (V [D_in, d]
    as read, then its transpose W; both exist while one is copied into the other, W alone after that) and, with
    preserve_row_magnitudes, the changed matrix, the product and their gradients during an evaluation (four more of
    that size); the layer's saved rows; P, the outputs and their gradients; the distance matrices; and the
    optimiser's state (2 history_size pairs of [r (D_in + d)] plus a few more of that size)."""
    n = n_good + n_bad
    mats = d_in * d * (5 if preserve else 2)
    rows = n * (row + 6 * d + 2 * r) + n_bad * n * 2
    lbfgs = (2 * history + 8) * r * (d_in + d)
    return 4 * (mats + rows + lbfgs)


def prepare(ws, params):
    """Parameters checked against the model and the saved measurement -> (params, rows, io) where rows is one row per
    layer, component and module of the range: {layer, component, module, views, d_in, stored, scratch, skipped}
    (+ private refs, saved layout entry and bias part), in fit order: layers ascending, COMPONENTS order, modules in
    source order (Heretic's). Routed experts in the range are refused unless ``expert_modules`` is "shared"."""
    p = normalize(params)
    DR.require_layout(ws)
    s = ws.struct
    n, d = s.n_layers, s.d_model
    em = p.get('expert_modules', 'all')
    if em == 'shared' and not DR.has_routed_experts(ws):
        raise DR.DirectionError(DR.NO_EXPERTS)
    start = p['start_layer_index']
    end = n if p['end_layer_index'] is None else p['end_layer_index']
    if not start < n:
        raise DR.DirectionError(f"start_layer_index: must be from 0 to {n - 1} (this model has {n} layers)")
    if end > n:
        raise DR.DirectionError(f"end_layer_index: must be at most {n} (this model has {n} layers; the end is exclusive)")
    name = p['direction']
    meta = DR._meta(ws, name)
    DR.matches(ws, meta, name)
    io = DR.load_io(ws, name, meta)
    counts = io['counts']
    most = min(counts['good'], counts['bad'], MAX_NEIGHBORS)
    if p['neighbor_count'] > most:
        raise DR.DirectionError(f"neighbor_count: must be at most {most} (the measurement has {counts['good']} good and "
                                f"{counts['bad']} bad prompts; at most {MAX_NEIGHBORS})")
    r = p['lora_rank']
    rows = []
    for l in range(start, end):
        L = io['layers'][l]
        saved = {(c.get('component'), c.get('module')): c for c in L['modules'] if isinstance(c, dict)}
        for comp in DR.COMPONENTS:
            mods = DR.modules(ws, l, comp, em)
            if not mods:
                rows.append({'layer': l, 'component': comp, 'module': None, 'views': 0, 'd_in': 0, 'stored': 0,
                             'scratch': 0, 'skipped': f"this layer has no {comp}"})
                continue
            if any(m['routed'] for m in mods):
                raise DR.DirectionError(ROUTED_REFUSAL.format(layer=l, component=comp))
            for m in mods:
                where = _where(l, comp, m, mods)
                row = {'layer': l, 'component': comp, 'module': m['path'], 'views': 0, 'd_in': 0, 'stored': 0,
                       'scratch': 0, 'skipped': None}
                rows.append(row)
                views = m['views']
                c = saved.get((comp, m['path'])) or {}
                if not c.get('captured'):
                    row['skipped'] = c.get('why') or "its inputs and outputs were not captured"
                    continue
                d_in = sum(v.shape[0] for v in views)
                if d_in != c['d_in']:
                    raise DR.DirectionError(f"{where}: the saved inputs have {c['d_in']} values, the matrix reads "
                                            f"{d_in}; measure it again with module_io")
                for v in views:
                    if not ws.editable(v.name):
                        raise DR.DirectionError(f"{where}: {v.name} is not a floating-point tensor and cannot be edited")
                if p['steer_bad_behavior_weight'] == 0:
                    row['skipped'] = NOTHING_TO_FIT
                    continue
                if r > min(d_in, d):
                    raise DR.DirectionError(f"lora_rank: {r} is above min(D_in, d_model) = {min(d_in, d)} of {where}")
                row.update(views=len(views), d_in=d_in, stored=r * (d_in + len(views) * d),
                           scratch=scratch_bytes(d_in, d, counts['good'], counts['bad'], L['row'], r, p['history_size'],
                                                 p['preserve_row_magnitudes']))
                row['_refs'] = views
                row['_io'] = c
                row['_bias'] = m['bias']
                row['_where'] = where
    return p, rows, io


def _where(layer, comp, module, mods) -> str:
    """How messages name a module: ``layer <l> <component>``, with the module path when the layer has several."""
    return f"layer {layer} {comp}" + (f" {module['path']}" if len(mods) > 1 else "")


def _public(row):
    return {k: v for k, v in row.items() if not k.startswith('_')}


def plan(ws, params) -> dict:
    """What ablate_ara will do, without reading a weight or running the model: per layer and component its views,
    the values its pieces will hold (r (D_in + views d)), why it is skipped, and the scratch memory its fit asks
    the admission for; plus totals."""
    p, rows, io = prepare(ws, params)
    fitted = [r for r in rows if r['skipped'] is None]
    n = ws.struct.n_layers
    return {'method': 'ara', 'direction': p['direction'], 'start_layer_index': p['start_layer_index'],
            'end_layer_index': n if p['end_layer_index'] is None else p['end_layer_index'], 'lora_rank': p['lora_rank'],
            'rows': [_public(r) for r in rows],
            'totals': {'layers': len({r['layer'] for r in fitted}), 'components': len(fitted),
                       'views': sum(r['views'] for r in fitted), 'stored': sum(r['stored'] for r in fitted),
                       'skipped': len(rows) - len(fitted), 'scratch': max([r['scratch'] for r in fitted], default=0)}}


def script_for(params) -> str:
    """The one-line weight script for these ARA parameters (numbers written with repr: they round-trip exactly)."""
    p = normalize(params)
    end = 'None' if p['end_layer_index'] is None else str(p['end_layer_index'])
    f = lambda k: repr(float(p[k]))  # noqa: E731
    return (f"ablate_ara({json.dumps(p['direction'])}, start_layer_index={p['start_layer_index']}, end_layer_index={end}, "
            f"preserve_good_behavior_weight={f('preserve_good_behavior_weight')}, "
            f"steer_bad_behavior_weight={f('steer_bad_behavior_weight')}, "
            f"overcorrect_relative_weight={f('overcorrect_relative_weight')}, neighbor_count={p['neighbor_count']}, "
            f"preserve_row_magnitudes={p['preserve_row_magnitudes']!r}, lora_rank={p['lora_rank']}, "
            f"n_optimization_steps={p['n_optimization_steps']}, learning_rate={f('learning_rate')}, "
            f"max_iter={p['max_iter']}, history_size={p['history_size']}, seed={p['seed']}"
            + (f", expert_modules={json.dumps(p['expert_modules'])}" if 'expert_modules' in p else '') + ")")


# ------------------------------------------------------------------ the loss and the fit of one component
def initial_values(g, r, d_in, d):
    """A [r, D_in] uniform in [-1/sqrt(D_in), 1/sqrt(D_in)] (kaiming_uniform_ with a = sqrt(5), from ``g``), B = 0."""
    A = torch.nn.init.kaiming_uniform_(torch.empty(r, d_in), a=math.sqrt(5), generator=g)
    return A, torch.zeros(d, r)


def column_scale(W, A, B, nrm):
    """scale[j] = nrm[j] / ||(W + B A)[j, :]||, 1 where nrm[j] = 0 (gradients flow through it). ``W`` [d, D_in] is
    V^T held contiguous, so row j of W is column j of V; ``nrm`` its row norms. The sum is taken over this layout,
    not over V + A^T B^T: the same value, but rounded as Heretic rounds it (see the module docstring)."""
    cn = torch.linalg.vector_norm(W + (B @ A), dim=1)
    ok = nrm > 0
    one = torch.ones_like(cn)
    return torch.where(ok, nrm / torch.where(ok, cn, one), one)


def knn(a, b, k):
    """Per row of a: the mean of its k smallest Euclidean distances to the rows of b."""
    return torch.cdist(a, b).topk(k, dim=1, largest=False).values.mean(1)


def loss(A, B, data, p):
    """preserve_good_behavior_weight * preserve + steer_bad_behavior_weight * steer (docs/DIRECTIONS.md), with
    out_new(X) = (P + (X @ A^T) @ B^T) * scale + bias and P = X @ W^T.

    The order below is Heretic's: both outputs first, the scale after them. A and B each receive their gradient
    along three paths (the two outputs and the scale), and the order the paths were built in is the order their
    sums are rounded in, so another order gives a fit that differs in the last digit and then, on a sensitive
    problem, in every digit (scripts/heretic_fit_check.py)."""
    out_g = data['P_g'] + (data['X_g'] @ A.T) @ B.T
    out_b = data['P_b'] + (data['X_b'] @ A.T) @ B.T
    if p['preserve_row_magnitudes']:
        scale = column_scale(data['W'], A, B, data['nrm'])
        out_g = out_g * scale
        out_b = out_b * scale
    out_g = out_g + data['bias']
    out_b = out_b + data['bias']
    k = p['neighbor_count']
    preserve = (out_g - data['Y_g']).pow(2).mean()
    steer = knn(out_b, data['Y_g'], k).mean() - p['overcorrect_relative_weight'] * knn(out_b, data['Y_b'], k).mean()
    return p['preserve_good_behavior_weight'] * preserve + p['steer_bad_behavior_weight'] * steer


def mismatch(P, bias, Y) -> float:
    """||P + bias - Y||_F / ||Y||_F (0 when both are 0)."""
    num = float(torch.linalg.vector_norm((P + bias - Y).double()))
    den = float(torch.linalg.vector_norm(Y.double()))
    return num / den if den > 0 else (0.0 if num == 0 else math.inf)


def prepare_data(V, bias, X_g, Y_g, X_b, Y_b) -> dict:
    """Everything the loss reads that does not change during the fit, in the adapter layout: W = V^T [d, D_in] held
    contiguous (V itself is not kept), P = X @ W^T once, and the row norms of W (the column norms of V)."""
    W = V.T.contiguous()
    return {'W': W, 'bias': bias, 'X_g': X_g, 'Y_g': Y_g, 'X_b': X_b, 'Y_b': Y_b, 'P_g': X_g @ W.T, 'P_b': X_b @ W.T,
            'nrm': torch.linalg.vector_norm(W, dim=1)}


def fit(data, A0, B0, p, where='') -> tuple[torch.Tensor, torch.Tensor, dict]:
    """L-BFGS on [A, B] from (A0, B0) -> (A, B, {loss_before, loss_after, scale_min, scale_max}), in the dtype of the
    inputs (``ablate`` passes float32 only; the tests also run it in float64), on the CPU."""
    def finite(x, what):
        ok = math.isfinite(x) if isinstance(x, float) else bool(torch.isfinite(x).all())
        if not ok:
            raise DR.DirectionError(f"{where}: the fit gave a non-finite {what}")
    with torch.enable_grad():
        A = A0.detach().clone().requires_grad_(True)
        B = B0.detach().clone().requires_grad_(True)
        with torch.no_grad():
            before = float(loss(A, B, data, p))
        finite(before, 'loss')
        opt = torch.optim.LBFGS([A, B], lr=p['learning_rate'], max_iter=p['max_iter'], history_size=p['history_size'],
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            value = loss(A, B, data, p)
            value.backward()
            return value
        for _ in range(p['n_optimization_steps']):
            got = opt.step(closure)
            finite(float(got.detach()) if isinstance(got, torch.Tensor) else float(got), 'loss')
        A, B = A.detach(), B.detach()
    with torch.no_grad():
        after = float(loss(A, B, data, p))
        scale = column_scale(data['W'], A, B, data['nrm']) if p['preserve_row_magnitudes'] else torch.ones(1)
    finite(after, 'loss')
    finite(A, 'result')
    finite(B, 'result')
    return A, B, {'loss_before': before, 'loss_after': after, 'scale_min': float(scale.min()), 'scale_max': float(scale.max())}


# ------------------------------------------------------------------ the whole call
def read_whole(ws, views, extra=None, stored=lambda: 0, scratch=0) -> torch.Tensor:
    """The component V [D_in, d] (rows of every view) whole in float32, read in row slabs of at most STREAM_ELEMS."""
    d = views[0].shape[1]
    V = torch.empty(sum(v.shape[0] for v in views), d, dtype=torch.float32)
    a = 0
    for view in views:
        for t in DR._slabs(ws, view, extra, stored, scratch):
            V[a:a + t.shape[0]] = t.to(torch.float32)
            a += t.shape[0]
    return V


def read_bias(ws, layer, comp, extra=None) -> torch.Tensor:
    """The bias of a component's first source (attn b_O, mlp b_out), float32 [d], or zeros."""
    group = DR.COMPONENTS[comp][0]
    part = (getattr(ws.struct.layers[layer], group, None) or {}).get(DR.BIASES[comp])
    return read_part_bias(ws, part, extra)


def read_part_bias(ws, part, extra=None) -> torch.Tensor:
    """A module's bias part read as float32 [d], or zeros when the module has none."""
    d = ws.struct.d_model
    if part is None:
        return torch.zeros(d)
    return ws.sem.ref(part).read('current', extra).to(torch.float32).reshape(d)


def ablate(ws, params, extra=None, add=None, stored=lambda: 0, progress=None) -> dict:
    """Fit every module of the layer range against the current weights plus ``extra`` and hand each piece to
    ``add(view, A, B)`` (delta = A @ B per view). -> report {components: [one per module], skipped: [...], stored}.
    One initial draw per fitted module, in fit order (Heretic's)."""
    p, rows, io = prepare(ws, params)
    work = [r for r in rows if r['skipped'] is None]
    name = p['direction']
    g = torch.Generator().manual_seed(p['seed'])          # once per call, consumed in fit order
    r = p['lora_rank']
    report, arrays = [], (None, None)
    for i, row in enumerate(work):
        if progress is not None:
            progress(i, len(work))
        l, c = row['layer'], row['_io']
        where = row['_where']
        edit_preflight(ws, stored(), scratch=row['scratch'])
        if arrays[0] != l:
            arrays = (l, DR.io_arrays(ws, name, io, l))
        good, bad = arrays[1]
        d_in, d = c['d_in'], ws.struct.d_model
        X_g, Y_g = good[:, c['x']:c['x'] + d_in], good[:, c['y']:c['y'] + d]
        X_b, Y_b = bad[:, c['x']:c['x'] + d_in], bad[:, c['y']:c['y'] + d]
        V = read_whole(ws, row['_refs'], extra, stored, row['scratch'])
        data = prepare_data(V, read_part_bias(ws, row['_bias'], extra), X_g, Y_g, X_b, Y_b)
        del V                                             # the fit reads its transpose, held in data
        mm = {'good': mismatch(data['P_g'], data['bias'], Y_g), 'bad': mismatch(data['P_b'], data['bias'], Y_b)}
        for label, m in mm.items():
            if not m <= MISMATCH_LIMIT:
                raise DR.DirectionError(f"{where}: mismatch {m:.4g} on the {label} prompts (above {MISMATCH_LIMIT:g}): "
                                        f"{MISMATCH_ADVICE}")
        A0, B0 = initial_values(g, r, d_in, d)
        A, B, info = fit(data, A0, B0, p, where)
        del data
        At, Bt = A.T.contiguous(), B.T.contiguous()       # delta_V = A^T @ B^T, as fitted (not rescaled by scale)
        n = 0
        if bool(At.any()) and bool(Bt.any()):
            a = 0
            for view in row['_refs']:
                piece = At[a:a + view.shape[0]].contiguous()
                a += view.shape[0]
                add(view, piece, Bt)
                n += piece.numel() + Bt.numel()
        report.append({**_public(row), 'stored': n, 'mismatch_good': mm['good'], 'mismatch_bad': mm['bad'], **info})
    return {'direction': name, 'lora_rank': r, 'components': report,
            'skipped': [_public(x) for x in rows if x['skipped'] is not None], 'stored': sum(x['stored'] for x in report)}
