"""The model-worker job table: one row per job that may cross the process boundary.

A row names the worker-side callable, the arguments it accepts (JSON values),
its result codec, whether it streams ``token`` frames, and whether it takes the
job's uncommitted ``extra``. ``extra`` is shipped with every job that uses it
and dropped when that job ends; nothing is cached across jobs.

``run`` returns only the captures it was asked for (no logits, no hooks, no
gradients) and ``logits`` returns only ``tail`` trailing rows when asked, so
transfers stay as small as the call site needs. ``observe_window`` (one window of
the atlas observed stage) returns only that window's reductions: no captures and
no logits (``wsp.atlas.observe.observe_window``). ``direction_residuals`` returns only the
last-token residual vector of each prompt at each layer (``wsp.directions.residuals``), and with ``module_io``
also the last-token input and output of each layer's attention and MLP output matrices, packed in the same tensor.
"""
from __future__ import annotations

import json
from functools import wraps
from inspect import signature
from typing import Callable, NamedTuple

WORKER_KINDS = {'gguf'}


class Job(NamedTuple):
    call: Callable          # worker side: (context, **arguments) -> value
    args: tuple             # accepted JSON arguments
    result: str             # key of RESULTS
    stream: bool = False    # sends one ``token`` frame per sampled token before its result
    extra: bool = False     # takes the job's uncommitted ``extra`` {tensor: [pieces]}


class JobContext(NamedTuple):
    ws: object
    runner: object
    extra: dict | None
    emit: Callable          # emit(token_id) -> sends a ``token`` frame
    make_runner: Callable | None = None  # load only


def _ids(ids):
    if not isinstance(ids, list) or not all(type(i) is int for i in ids):
        raise ValueError('ids must be a list of integer token ids')
    return ids


def _no_grad():
    import torch
    return torch.no_grad()


def hook_kinds() -> tuple:
    """The hook kinds the load job reports (``blocks.L.<kind>`` the runner has): wsp.runner's table, then attention
    patterns. Read when the job runs: importing wsp.runner here would load torch into every process that imports
    this table (the supervisor too)."""
    from .runner import HOOK_KINDS
    return tuple(HOOK_KINDS) + ('attn.hook_pattern',)


def _load(ctx):
    # The worker loop has already verified the base identity and synced patches.
    import os
    ws = ctx.ws
    ws._runner = runner = ctx.make_runner(ws)
    hooks = [f'blocks.{l}.{k}' for l in range(ws.struct.n_layers) for k in hook_kinds() if runner.available(f'blocks.{l}.{k}')]
    return dict(max_ctx=runner.max_ctx, available=hooks, threads=getattr(getattr(runner, 'kernel', None), 'threads', None),
                backend=os.environ.get('WSP_GGML_DEVICE', 'cpu'), pid=os.getpid())


def _run(ctx, ids, variant='current', names=()):
    names = [str(n) for n in names]
    with _no_grad():
        _, cache = ctx.runner.run(_ids(ids), variant, ctx.extra, names=names)
    return {n: cache[n] for n in names if n in cache}


def _logits(ctx, ids, variant='current', tail=None):
    with _no_grad():
        out = ctx.runner.logits(_ids(ids), variant, ctx.extra)
    return out[-int(tail):] if tail else out


def _observe_window(ctx, ids, variant='base', names=()):
    from .atlas.observe import observe_window
    with _no_grad():
        return observe_window(ctx.ws, ctx.runner, _ids(ids), variant, names)


def _steer(steer):
    # A steering wire form (wsp.steering) is passed on only when the job carries one.
    return {} if steer is None else {'steer': steer}


def _next_logits(ctx, ids, variant='current', steer=None):
    return ctx.runner.next_logits(_ids(ids), variant, ctx.extra, **_steer(steer))


def _generate(ctx, ids, variant='current', max_new_tokens=40, temperature=0., top_k=0, seed=None, stop_ids=(), steer=None):
    return [int(t) for t in ctx.runner.generate(_ids(ids), variant, ctx.extra, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, seed=seed, stop_ids=list(stop_ids), on_token=ctx.emit, **_steer(steer))]


def _trace(ctx, **arguments):
    from . import analysis
    return analysis.trace(ctx.ws, extra=ctx.extra, **arguments)


def _patching(ctx, **arguments):
    from . import analysis
    return analysis.patching(ctx.ws, **arguments)


def _ablation_scan(ctx, prompts, **arguments):
    from . import analysis
    return analysis.ablation_scan(ctx.ws, [(_ids(list(ids)), int(target)) for ids, target in prompts], **arguments)


def _attribute(ctx, **arguments):
    from .atlas.guide import attribute
    return attribute(ctx.ws, **arguments)


def _steer_calibrate(ctx, ids):
    from . import steering
    return steering.calibrate_job(ctx.ws, ids)


def _direction_residuals(ctx, prompts_ids, variant='base', module_io=False):
    from . import directions
    return directions.residuals(ctx.ws, prompts_ids, variant, module_io)


def _probe(name):
    def call(ctx, **arguments):
        from . import probes
        return getattr(probes, name)(ctx.ws, **arguments)
    return call


JOBS = {
    'load': Job(_load, (), 'json'),
    'run': Job(_run, ('ids', 'variant', 'names'), 'captures', extra=True),
    'logits': Job(_logits, ('ids', 'variant', 'tail'), 'tensor', extra=True),
    # One window of the atlas observed stage, reduced next to the model; committed weights only (no extra).
    'observe_window': Job(_observe_window, ('ids', 'variant', 'names'), 'reduced'),
    'next_logits': Job(_next_logits, ('ids', 'variant', 'steer'), 'tensor', extra=True),
    'generate': Job(_generate, ('ids', 'variant', 'max_new_tokens', 'temperature', 'top_k', 'seed', 'stop_ids', 'steer'), 'json',
                    stream=True, extra=True),
    'trace': Job(_trace, ('prompt', 'variant', 'target', 'versus', 'k', 'max_tokens'), 'json', extra=True),
    'patching': Job(_patching, ('clean', 'corrupt', 'target', 'versus', 'variant', 'budget'), 'json'),
    'ablation_scan': Job(_ablation_scan, ('prompts', 'variant', 'budget'), 'json'),
    'attribute': Job(_attribute, ('prompt', 'target', 'versus', 'variant', 'top', 'group'), 'json'),
    # Steering and probes (wsp.steering, wsp.probes): reduced JSON results only.
    'steer_calibrate': Job(_steer_calibrate, ('ids',), 'json'),
    'probe_reads': Job(_probe('reads'), ('ids', 'neurons', 'variants', 'steer'), 'json'),
    'probe_fires': Job(_probe('fires'), ('ids', 'position', 'k', 'sign', 'variant'), 'json'),
    'probe_contrast': Job(_probe('contrast'), ('a', 'b', 'k', 'variant'), 'json'),
    'probe_next': Job(_probe('next_token'), ('ids', 'variant', 'steer', 'top'), 'json'),
    # Behaviour directions (wsp.directions): the last-token residual stream at every layer of 1..8 prompts,
    # [prompts, n_layers + 1, d_model] float32; with module_io one packed row per prompt that also holds every
    # layer's output-matrix inputs and outputs (wsp.directions.io_layout); committed weights only (no extra).
    'direction_residuals': Job(_direction_residuals, ('prompts_ids', 'variant', 'module_io'), 'tensor'),
}


# ------------------------------------------------------------------ codecs
def _float32(name, t):
    from .worker_protocol import TransferUnsupported, dtype_name
    import torch
    if not isinstance(t, torch.Tensor):
        raise TransferUnsupported(f'{name} result is not a tensor')
    dtype_name(t, 'float32')
    return t


def strict_json(value):
    """A JSON result as plain Python values, or ValueError/TypeError; nothing is sent until it converts.

    numpy and torch scalars become Python numbers and arrays become lists. The value
    crosses as the frame's JSON body, so its size is bounded only by ``MAX_BODY``.
    Non-finite floats are refused, as the HTTP layer's strict JSON rendering refuses them.
    """
    import math
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Out of range float values are not JSON compliant')
        return float(value)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                key = strict_json(key)
                if not isinstance(key, (str, int, float, bool)):
                    raise TypeError(f'{type(key).__name__} is not a JSON object key')
            out[key if isinstance(key, str) else json.dumps(key)] = strict_json(item)
        return out
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    import numpy as np
    import torch
    if isinstance(value, np.generic):
        return strict_json(value.item())
    if isinstance(value, (np.ndarray, torch.Tensor)):
        items = value.tolist() if isinstance(value, np.ndarray) else value.detach().cpu().tolist()
        return strict_json(items)
    raise TypeError(f'{type(value).__name__} is not JSON serializable')


def _codec(name, t):
    from .worker_protocol import TransferUnsupported, dtype_name
    import torch
    if not isinstance(t, torch.Tensor):
        raise TransferUnsupported(f'{name} result is not a tensor')
    dtype_name(t)          # float32 values or int64 counts and positions; anything else is refused, never cast
    return t


def _reduced(f):
    loss, names = f['loss'], f['names']
    if not isinstance(loss, str) or not isinstance(names, list) or set(names) != set(f.tensors):
        from .worker_protocol import ProtocolError
        raise ProtocolError('Malformed reduced result')
    return {'loss': loss, 'tensors': {n: f.tensors[n] for n in names}}


RESULTS = {
    'tensor': (lambda v: ({}, {'value': _float32('value', v)}), lambda f: f.tensors['value']),
    'captures': (lambda v: ({'names': list(v)}, {n: _float32(n, t) for n, t in v.items()}),
                 lambda f: {n: f.tensors[n] for n in f['names']}),
    'json': (lambda v: ({'value': strict_json(v)}, {}), lambda f: f['value']),
    # {'loss': repr of a float (exact, and inf or nan too), 'tensors': {name: tensor}}
    'reduced': (lambda v: ({'loss': str(v['loss']), 'names': list(v['tensors'])}, {n: _codec(n, t) for n, t in v['tensors'].items()}),
                _reduced),
}


def _arguments(name, arguments):
    row = JOBS[name]
    unknown = set(arguments) - set(row.args)
    if unknown:
        raise ValueError(f'{name} job does not take {sorted(unknown)}')
    return row


def encode_job(name, arguments, extra=None):
    """Server side: -> (job frame fields, tensors). ``extra`` only for rows that take it."""
    from .worker_protocol import encode_extra
    row = _arguments(name, arguments)
    fields, tensors = {'name': name, 'args': dict(arguments)}, {}
    if extra:
        if not row.extra:
            raise ValueError(f'{name} job does not take extra')
        fields['extra'], tensors = encode_extra(extra)
    return fields, tensors


def decode_job(frame):
    """Worker side: -> (row, arguments, extra)."""
    from .worker_protocol import decode_extra
    name, arguments = frame.get('name'), frame.get('args', {})
    if name not in JOBS or not isinstance(arguments, dict):
        raise ValueError(f'Unknown worker job {name!r}')
    row = _arguments(name, arguments)
    extra = decode_extra(frame.get('extra'), frame.tensors) if row.extra else None
    return row, arguments, extra


def encode_result(name, value):
    return RESULTS[JOBS[name].result][0](value)


def decode_result(name, frame):
    return RESULTS[JOBS[name].result][1](frame)


# ------------------------------------------------------------------ the server-side fork
def uses_worker(ws):
    """True only for a supervised worker-mode runtime over a worker kind; never inside the worker."""
    from .runtime_supervisor import RuntimeClient
    client = getattr(ws, '_runtime_client', None)
    return getattr(ws, 'kind', None) in WORKER_KINDS and isinstance(client, RuntimeClient) and bool(getattr(client, 'worker', False))


def worker_job(name):
    """Forward a whole analysis call to the worker when the workspace uses one; else call it."""
    if name not in JOBS:
        raise KeyError(f'No worker job named {name!r}')
    def decorate(fn):
        sig = signature(fn)
        first = next(iter(sig.parameters))
        @wraps(fn)
        def wrapped(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            ws = bound.arguments[first]
            if not uses_worker(ws):
                return fn(*args, **kwargs)
            bound.apply_defaults()
            arguments = {k: v for k, v in bound.arguments.items() if k != first}
            # Opens the operation itself where no job_scope did; nests otherwise.
            with ws._runtime_client.operation(ws, name):
                return ws.runner.job(name, **arguments)
        return wrapped
    return decorate
