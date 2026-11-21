"""Behaviour directions: measure the residual stream on two sets of prompts, then remove the difference.

    measure      run "good" and "bad" prompts through the model (batch 1) and keep, for each set, the mean
                 residual-stream vector at the last token of every prompt, at every layer
    table        one unit direction per index: unit(bad - good), optionally made orthogonal to the good mean
    ablation     remove that direction from what the attention output matrix and the MLP output matrix of
                 chosen layers write, with a per-layer strength, as low-rank patch pieces (three variants)

A measurement is stored in the workspace as ``directions/<name>.safetensors`` (``good_mean`` and
``bad_mean``, float32 ``[n_layers + 1, d_model]``), ``<name>.prompts.json`` and ``<name>.json`` (written
last; its presence means the entry is complete).  Index ``k < n`` is the residual stream entering layer
``k`` (``blocks.k.hook_resid_pre``); index ``n`` is the stream leaving the last layer after the model's
final norm, the sequence ``transformers`` reports as hidden states.

With ``module_io`` a measurement also keeps, per prompt, layer and module, the last-token input and output of the
module's matrix (``<name>.io/``, one float32 ``.npy`` per layer and set, ``io.json`` written last); the fit of
wsp.directions_ara reads them. A component is one or more modules (COMPONENT_SOURCES): the attention output matrix or
a linear-attention mixer's output projection; the dense MLP's output matrix, or each routed expert's and the shared
expert's (docs/DIRECTIONS.md, "Hybrid and mixture-of-experts layouts").

The forward passes run where the model runs (the worker job ``direction_residuals``, or the same function
in this process); only the last-token vectors come back.  The ablation is the weight-script function
``ablate_direction`` (wsp.wscript): its pieces go through the interpreter's ``_add``, so the edit is
committed, scoped, tested and recorded by ``Session.apply`` like any other script.  Nothing here writes a
weight.  Formulas: docs/DIRECTIONS.md (directional ablation, Arditi et al. 2024; the projected and
norm-preserving variants, Lai 2025), implemented from those formulas.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid

import numpy as np

import torch

from . import workspace as W
from .runtime_events import edit_preflight, job_scope
from .worker_jobs import _ids, worker_job

VERSION = 1
DIRECTORY = 'directions'
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$', re.ASCII)
MAX_PROMPTS = 1024             # per set
MAX_CHARS = 20_000             # per text (and per system message / prefix)
MAX_TOKENS = 512               # per prompt, as encoded (as rendered with chat)
JOB_PROMPTS = 8                # prompts per direction_residuals job
CHUNK = 4                      # prompts the measurement sends per job
MAX_WEIGHT = 8.0               # |max_weight|, |min_weight|
MAX_RANK = 64                  # full_normalization_lora_rank
VARIANTS = ('base', 'current')
ROW_NORMALIZATIONS = ('none', 'pre', 'full')
# component -> its sources, in Heretic's module order (heretic/model.py at 64e9991, get_layer_modules). Each source
# yields the layer's modules of that component: its structure group and role, the hook points of the module's input
# and output (at the last token of each prompt; None: not captured) and its bias role. attn.o_proj is the softmax
# attention's output matrix, else a token mixer's output projection (Qwen3.5's linear_attn.out_proj); mlp.down_proj
# is the dense MLP's output matrix, then each routed expert's (one module per expert), then the shared expert's.
COMPONENT_SOURCES = {
    'attn.o_proj': ({'group': 'attn', 'role': 'W_O', 'hooks': ('attn.hook_z', 'hook_attn_out'), 'bias': 'b_O'},
                    {'group': 'mixer', 'role': 'W_O', 'hooks': ('mixer.hook_in', 'mixer.hook_out'), 'bias': 'b_O'}),
    'mlp.down_proj': ({'group': 'mlp', 'role': 'W_out', 'hooks': ('mlp.hook_post', 'hook_mlp_out'), 'bias': 'b_out'},
                      {'group': 'experts', 'role': 'W_out', 'hooks': None, 'bias': None, 'routed': True},
                      {'group': 'shared', 'role': 'W_out', 'hooks': ('shared.hook_in', 'shared.hook_out'), 'bias': None}),
}
# component name -> (structure group, role) of its first source: the layer's attention output matrix and its MLP
# output matrix (the values every earlier caller reads)
COMPONENTS = {c: (s[0]['group'], s[0]['role']) for c, s in COMPONENT_SOURCES.items()}
EXPERT_MODULES = ('all', 'shared')     # which mlp.down_proj modules of a mixture-of-experts layer an ablation changes
DISTRIBUTION_KEYS = ('max_weight', 'max_weight_position', 'min_weight', 'min_weight_distance')
PARAMS = ('direction', 'direction_index', 'weight_distributions', 'orthogonalize_direction', 'row_normalization',
          'full_normalization_lora_rank', 'expert_modules')
MEASURE_FIELDS = ('name', 'good', 'bad', 'chat', 'system', 'prefix', 'chat_options', 'variant',
                  'winsorization_quantile', 'replace', 'module_io')
METHODS = ('ablation', 'ara')          # "method" of the ablation parameters; absent means ablation
# component -> (hook point of its matrix's input, hook point of its output) of its first source
IO_HOOKS = {c: s[0]['hooks'] for c, s in COMPONENT_SOURCES.items()}
BIASES = {c: s[0]['bias'] for c, s in COMPONENT_SOURCES.items()}
ROUTED_WHY = "routed experts are not captured: ARA does not fit them yet"
EXPERT_MODULES_ERROR = 'expert_modules: must be "all" or "shared"'
NO_EXPERTS = 'expert_modules: "shared" changes nothing: this model has no routed experts'
IO_VERSION = 2                 # 2: io.json lists every module of a layer; a version 1 file is read as one per component
IO_FILE = 'io.json'
# A module_io job answers one float32 tensor of (n + 1) d + (the layout's width) values per prompt. The frame's
# payload length is a 32-bit field (wsp.worker_protocol), so a result must stay below 4 GiB; a job is kept at or
# below IO_JOB_BYTES (a quarter of that), with fewer prompts per job when a row is that large.
IO_JOB_BYTES = 1 << 30
DISK_RESERVE = 2 << 30         # module_io is refused unless this much stays free on the workspace's file system
LIMITS = {'prompts_per_set': MAX_PROMPTS, 'characters_per_text': MAX_CHARS, 'tokens_per_prompt': MAX_TOKENS,
          'prompts_per_job': JOB_PROMPTS, 'prompts_per_chunk': CHUNK, 'name': NAME.pattern,
          'max_abs_weight': MAX_WEIGHT, 'full_normalization_lora_rank': [1, MAX_RANK],
          'components': list(COMPONENTS), 'row_normalization': list(ROW_NORMALIZATIONS), 'variants': list(VARIANTS),
          'methods': list(METHODS), 'module_io_job_bytes': IO_JOB_BYTES, 'module_io_disk_reserve': DISK_RESERVE}


class DirectionError(ValueError):
    pass


class DirectionMissing(DirectionError):
    """No saved measurement of that name (the routes answer 404)."""


# ------------------------------------------------------------------ storage
def check_name(name) -> str:
    if not isinstance(name, str) or not NAME.match(name):
        raise DirectionError("name must be 1..64 characters from A-Z, a-z, 0-9, '.', '_' and '-', starting with a "
                             "letter or digit")
    if name.lower().endswith('.prompts'):
        # <name>.prompts.json is the prompt file of <name>; such a name would collide with it.
        raise DirectionError("name must not end in '.prompts' (that is the suffix of a measurement's prompt file)")
    if name.lower() == 'searches':
        # GET /api/directions/searches lists the automatic searches; a measurement of that name would be shadowed.
        raise DirectionError("name must not be 'searches' (the search routes use it)")
    return name


def _paths(ws, name):
    base = os.path.join(ws.dir, DIRECTORY, name)
    return {'meta': base + '.json', 'tensors': base + '.safetensors', 'prompts': base + '.prompts.json',
            'io': base + '.io'}


def exists(ws, name) -> bool:
    return os.path.isfile(_paths(ws, check_name(name))['meta'])


def _atomic(path, write):
    """write(temporary path), then os.replace onto ``path``; the temporary file never stays behind."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp'
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _write_json(path, obj):
    def write(tmp):
        with open(tmp, 'w') as f:
            json.dump(obj, f, indent=1)
    _atomic(path, write)


def save(ws, name, good_mean, bad_mean, meta, prompts, replace=False, io_dir=None):
    """Write one measurement: vectors, prompts, then the meta file (its presence means the entry is complete).
    ``io_dir``: a complete module input/output directory (its io.json written), moved in as ``<name>.io``; an old
    ``<name>.io`` is removed either way."""
    from safetensors.torch import save_file
    paths = _paths(ws, check_name(name))
    if os.path.exists(paths['meta']):
        if not replace:
            raise DirectionError(f"a direction named {name!r} exists; pass replace to measure it again")
        os.remove(paths['meta'])           # the old entry is incomplete (not listed) until the new one is written
    if os.path.isdir(paths['io']):
        shutil.rmtree(paths['io'])
    if io_dir is not None:
        os.replace(io_dir, paths['io'])
    tensors = {'good_mean': good_mean.detach().to(torch.float32).contiguous(),
               'bad_mean': bad_mean.detach().to(torch.float32).contiguous()}
    _atomic(paths['tensors'], lambda tmp: save_file(tensors, tmp))
    _write_json(paths['prompts'], prompts)
    _write_json(paths['meta'], meta)
    return meta


def _meta(ws, name):
    path = _paths(ws, name)['meta']
    if not os.path.isfile(path):
        raise DirectionMissing(f"no direction named {name!r}")
    try:
        with open(path) as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        raise DirectionError(f"direction {name!r}: its meta file cannot be read ({e})") from None
    if not isinstance(meta, dict) or meta.get('version') != VERSION or not isinstance(meta.get('n_layers'), int) \
            or not isinstance(meta.get('d_model'), int):
        raise DirectionError(f"direction {name!r}: its meta file is not a version {VERSION} direction measurement")
    if not isinstance(meta.get('name'), str) or meta['name'].lower() != name.lower():
        raise DirectionError(f"direction {name!r}: its meta file names {meta.get('name')!r}")
    return meta


def load(ws, name):
    """-> (meta, good_mean, bad_mean); each mean float32 [n_layers + 1, d_model]."""
    from safetensors.torch import load_file
    check_name(name)
    meta = _meta(ws, name)
    path = _paths(ws, name)['tensors']
    try:
        t = load_file(path)
        good, bad = t['good_mean'], t['bad_mean']
    except Exception as e:                  # a missing or damaged file
        raise DirectionError(f"direction {name!r}: its vectors cannot be read ({type(e).__name__}: {e})") from None
    shape = (meta['n_layers'] + 1, meta['d_model'])
    if tuple(good.shape) != shape or tuple(bad.shape) != shape or good.dtype != torch.float32 or bad.dtype != torch.float32:
        raise DirectionError(f"direction {name!r}: its vectors are not float32 {list(shape)}")
    return meta, good, bad


def prompts(ws, name) -> dict:
    check_name(name)
    _meta(ws, name)
    try:
        with open(_paths(ws, name)['prompts']) as f:
            out = json.load(f)
    except (OSError, ValueError) as e:
        raise DirectionError(f"direction {name!r}: its prompt file cannot be read ({e})") from None
    if not isinstance(out, dict) or not all(isinstance(out.get(k), list) for k in ('good', 'bad')):
        raise DirectionError(f"direction {name!r}: its prompt file does not hold the good and bad lists")
    return {'good': out['good'], 'bad': out['bad']}


def entries(ws) -> list[dict]:
    """Every saved measurement's meta, by name; a damaged entry is listed with ``error`` and the rest are kept."""
    d = os.path.join(ws.dir, DIRECTORY)
    out = []
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not fn.endswith('.json') or fn.endswith('.prompts.json'):
            continue
        name = fn[:-5]
        try:
            check_name(name)
            meta = _meta(ws, name)
            if not os.path.isfile(_paths(ws, name)['tensors']):
                raise DirectionError(f"direction {name!r}: its vectors file is missing")
            out.append(meta)
        except DirectionError as e:
            out.append({'name': name, 'error': str(e)})
    return out


def delete(ws, name) -> None:
    check_name(name)
    paths = _paths(ws, name)
    if not os.path.isfile(paths['meta']):
        raise DirectionMissing(f"no direction named {name!r}")
    os.remove(paths['meta'])                # first: from here on the entry is not listed
    for key in ('tensors', 'prompts'):
        if os.path.exists(paths[key]):
            os.remove(paths[key])
    if os.path.isdir(paths['io']):
        shutil.rmtree(paths['io'])


def matches(ws, meta, name):
    s = ws.struct
    if meta['n_layers'] != s.n_layers or meta['d_model'] != s.d_model:
        raise DirectionError(f"direction {name!r} was measured on a model with {meta['n_layers']} layers and d_model "
                             f"{meta['d_model']}; this model has {s.n_layers} layers and d_model {s.d_model}")


# ------------------------------------------------------------------ the layout: which modules a component has
def require_layout(ws) -> None:
    """Refuse directions on a workspace whose decoder layout is not recognised (no layer structure, e.g. a GGUF whose
    architecture has no validated mapping), before any model is loaded."""
    s = ws.struct
    if s.recognised:
        return
    reason = None
    if getattr(ws, 'kind', None) == 'gguf':
        try:
            reason = ws.store.capabilities()['generation']['reason']
        except Exception:                   # the reason is a courtesy; the refusal stands without it
            reason = None
    if not reason:
        reason = s.notes[-1] if s.notes else "the checkpoint's architecture was not recognised"
    raise DirectionError(f"directions need a recognised decoder layout; this model's tensors are available by raw name "
                         f"only ({reason})")


def _views(ws, layer, component, part):
    """A module's writer views ([d_in, d_model], the residual stream last), as Semantic.writers() gives them (one per
    head for the IDE's own 3-D attention output)."""
    r = ws.sem.ref(part)
    views = [r.sub(h) for h in range(r.shape[0])] if len(r.shape) == 3 and part.role == 'W_O' else [r]
    for v in views:
        if len(v.shape) != 2 or v.shape[1] != ws.struct.d_model:
            raise DirectionError(f"layer {layer} {component}: {part.name} gives a view of shape {list(v.shape)}, not "
                                 f"[d_in, {ws.struct.d_model}]")
    return views


def _module_path(L, part) -> str:
    """The module path relative to the layer: the tensor name without its layer prefix and ``.weight``."""
    path = part.name[:-len('.weight')] if part.name.endswith('.weight') else part.name
    return path[len(L.prefix) + 1:] if L.prefix and path.startswith(L.prefix + '.') else path


def routed_path(path) -> str:
    """``mlp.experts.3.down_proj`` -> ``mlp.experts.*.down_proj``: the name of a layer's routed experts together."""
    return re.sub(r'\.experts\.\d+\.', '.experts.*.', path, count=1)


def modules(ws, layer, component, expert_modules='all') -> list[dict]:
    """The modules of one component of a layer, in COMPONENT_SOURCES order (Heretic's): each
    {key, path, source, views, bias, hooks, routed}. ``key`` is the source's group (``experts.<e>`` for a routed
    expert), ``path`` the module path relative to the layer (``self_attn.o_proj``, ``linear_attn.out_proj``,
    ``mlp.experts.3.down_proj``), ``views`` its writer views, ``bias`` the bias Part or None. With
    ``expert_modules == "shared"`` routed experts are left out. From the structure alone (no weight read)."""
    L = ws.struct.layers[layer]
    out = []
    for src in COMPONENT_SOURCES[component]:
        if src.get('routed'):
            if expert_modules == 'shared':
                continue
            found = [(f"{src['group']}.{e}", ex.get(src['role']), None)
                     for e, ex in enumerate(getattr(L, src['group'], None) or [])]
        else:
            group = getattr(L, src['group'], None) or {}
            found = [(src['group'], group.get(src['role']), group.get(src['bias']) if src['bias'] else None)]
        for key, part, bias in found:
            if part is None:
                continue
            out.append({'key': key, 'path': _module_path(L, part), 'source': src['group'],
                        'views': _views(ws, layer, component, part), 'bias': bias, 'hooks': src['hooks'],
                        'routed': bool(src.get('routed'))})
    return out


def has_routed_experts(ws) -> bool:
    return any(getattr(L, 'experts', None) for L in ws.struct.layers)


# ------------------------------------------------------------------ module inputs and outputs (module_io)
_LEGACY_KEYS = ('d_in', 'captured', 'why', 'x', 'y', 'hooks')
_MODULE_KEYS = ('component', 'module', 'source', 'd_in', 'modules', 'captured', 'why', 'x', 'y')


def io_layout(ws, available=None) -> list[dict]:
    """Per layer, where each module's last-token input x [d_in] and output [d] sit in that layer's row:
    {layer, row, modules: [{component, module, source, d_in, captured, why, x, y, hooks}], components: {component:
    its first module's {d_in, captured, why, x, y, hooks}}}, modules in COMPONENTS order then source order
    (``modules``), x then output. A layer's routed experts are one entry, not captured (ARA does not fit them).
    From the structure alone (no weight read); ``available(hook name)`` (the runner's) marks a module whose hook
    point is missing as not captured, with the reason. Without it every module the layer has counts. For a model
    with one module per component the row is the one the version 1 layout gave."""
    s = ws.struct
    d = s.d_model
    out = []
    for l in range(s.n_layers):
        off, comps, entries = 0, {}, []
        for comp in COMPONENTS:
            mods = modules(ws, l, comp)
            if not mods:
                comps[comp] = {'d_in': 0, 'captured': False, 'why': f"this layer has no {comp}"}
                continue
            routed = [m for m in mods if m['routed']]
            mine = []
            for m in mods:
                d_in = sum(v.shape[0] for v in m['views'])
                if m['routed']:
                    if m is routed[0]:
                        mine.append({'component': comp, 'module': routed_path(m['path']), 'source': m['source'],
                                     'd_in': d_in, 'modules': len(routed), 'captured': False, 'why': ROUTED_WHY})
                    continue
                e = {'component': comp, 'module': m['path'], 'source': m['source'], 'd_in': d_in}
                hooks = [f'blocks.{l}.{h}' for h in m['hooks']]
                missing = [h for h in hooks if available is not None and not available(h)]
                if missing:
                    e.update(captured=False, why=f"hook point {missing[0]} is not available for this model")
                else:
                    e.update(captured=True, why=None, x=off, y=off + d_in, hooks=hooks)
                    off += d_in + d
                mine.append(e)
            comps[comp] = {k: mine[0][k] for k in _LEGACY_KEYS if k in mine[0]}
            entries += mine
        out.append({'layer': l, 'row': off, 'components': comps, 'modules': entries})
    return out


def io_width(layout) -> int:
    return sum(L['row'] for L in layout)


def io_bytes(layout, n_good, n_bad) -> int:
    """Bytes of the stored arrays (headers aside): 4 (N_good + N_bad) * sum over layers of the layer's row."""
    return 4 * (n_good + n_bad) * io_width(layout)


def io_job_prompts(ws, layout) -> int:
    """Prompts per module_io job: as many as keep one result at or below IO_JOB_BYTES, at most JOB_PROMPTS."""
    s = ws.struct
    row = 4 * ((s.n_layers + 1) * s.d_model + io_width(layout))
    if row > IO_JOB_BYTES:
        raise DirectionError(f"module_io: one prompt's inputs and outputs are {row:,} bytes, more than one worker "
                             f"result may carry ({IO_JOB_BYTES:,})")
    return max(1, min(JOB_PROMPTS, IO_JOB_BYTES // row))


def _free_bytes(path) -> int:
    return shutil.disk_usage(path).free


def check_disk(ws, n_good, n_bad) -> int:
    """Refuse module_io before any model is loaded when its arrays would not fit: the bytes from the structure (every
    component the model has; a hook point found missing later only makes it smaller) against the free space of
    the workspace's file system minus DISK_RESERVE. -> the bytes."""
    need = io_bytes(io_layout(ws), n_good, n_bad)
    free = _free_bytes(ws.dir)
    if need > free - DISK_RESERVE:
        raise DirectionError(f"module_io needs {need:,} bytes on disk; the workspace's file system has {free:,} bytes "
                             f"free, of which {DISK_RESERVE:,} stay free")
    return need


def _npy_header(path, rows, width):
    """Preallocate ``path`` as a float32 .npy [rows, width] (header, then the file extended to its full size)
    -> the offset of its data."""
    with open(path, 'wb') as f:
        np.lib.format.write_array_header_1_0(f, {'descr': np.dtype('<f4').str, 'fortran_order': False,
                                                 'shape': (rows, width)})
        start = f.tell()
        f.truncate(start + 4 * rows * width)
    return start


class _IoWriter:
    """The module_io files of one measurement, written into a temporary directory as chunks arrive (the server
    never holds more than one chunk), io.json last, then moved in place by ``save``."""

    def __init__(self, ws, name, layout, counts):
        self.ws, self.name, self.layout, self.counts = ws, name, layout, counts
        self.final = _paths(ws, name)['io']
        self.dir = f'{self.final}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp'
        self.start, self.files = {}, {}
        os.makedirs(self.dir)
        try:
            for L in layout:
                if L['row'] == 0:
                    continue
                self.files[L['layer']] = {label: f"{label}.L{L['layer']}.npy" for label in ('good', 'bad')}
                for label in ('good', 'bad'):
                    path = os.path.join(self.dir, self.files[L['layer']][label])
                    self.start[(label, L['layer'])] = _npy_header(path, counts[label], L['row'])
        except BaseException:
            self.discard()
            raise

    def write(self, label, first, packed):
        """``packed`` [prompts, width] (the job's result without the residual part) -> rows ``first..`` of each file."""
        a = 0
        arr = packed.detach().to(torch.float32).contiguous().numpy()
        for L in self.layout:
            w = L['row']
            if w == 0:
                continue
            path = os.path.join(self.dir, self.files[L['layer']][label])
            with open(path, 'r+b') as f:
                f.seek(self.start[(label, L['layer'])] + 4 * first * w)
                f.write(np.ascontiguousarray(arr[:, a:a + w]).astype('<f4', copy=False).tobytes())
            a += w

    def finish(self):
        layers = []
        for L in self.layout:
            comps = {}
            for comp, c in L['components'].items():
                comps[comp] = {k: c[k] for k in ('d_in', 'captured', 'why', 'x', 'y') if k in c}
            mods = [{k: e[k] for k in _MODULE_KEYS if k in e} for e in L['modules']]
            layers.append({'layer': L['layer'], 'row': L['row'], 'files': self.files.get(L['layer']), 'components': comps,
                           'modules': mods})
        total = io_bytes(self.layout, self.counts['good'], self.counts['bad'])
        info = {'version': IO_VERSION, 'name': self.name, 'n_layers': self.ws.struct.n_layers,
                'd_model': self.ws.struct.d_model, 'dtype': 'float32', 'counts': dict(self.counts), 'bytes': total,
                'layers': layers}
        _write_json(os.path.join(self.dir, IO_FILE), info)
        return {'bytes': total, 'layers': sum(1 for L in self.layout if L['row'] > 0)}

    def discard(self):
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)


def load_io(ws, name, meta=None) -> dict:
    """The io.json of a complete module_io measurement, checked against its meta and files, or refused."""
    meta = meta if meta is not None else _meta(ws, check_name(name))
    again = f"direction {name!r} has no complete module inputs and outputs: measure it again with module_io"
    if not meta.get('module_io'):
        raise DirectionError(again)
    d = _paths(ws, name)['io']
    try:
        with open(os.path.join(d, IO_FILE)) as f:
            info = json.load(f)
    except (OSError, ValueError):
        raise DirectionError(again) from None
    if not isinstance(info, dict) or info.get('version') not in (1, IO_VERSION) or info.get('counts') != meta.get('counts') \
            or info.get('n_layers') != meta['n_layers'] or info.get('d_model') != meta['d_model'] \
            or not isinstance(info.get('layers'), list) or len(info['layers']) != meta['n_layers']:
        raise DirectionError(again)
    for L in info['layers']:
        if not isinstance(L, dict) or (info['version'] == IO_VERSION and not isinstance(L.get('modules'), list)):
            raise DirectionError(again)
        for fn in (L.get('files') or {}).values():
            if not os.path.isfile(os.path.join(d, fn)):
                raise DirectionError(again)
    if info['version'] == 1:
        _modules_of_version_1(ws, info)
    return info


def _modules_of_version_1(ws, info):
    """A version 1 io.json holds one entry per component (the first source's module; its rows are those a version 2
    layout gives such a model): each layer gets the ``modules`` list of version 2, the module named from the
    structure (None when the layer has no module of that source now, so nothing is fitted from it)."""
    for L in info['layers']:
        mods = []
        for comp, c in (L.get('components') or {}).items():
            if comp not in COMPONENT_SOURCES or not isinstance(c, dict):
                continue
            legacy = COMPONENT_SOURCES[comp][0]['group']
            here = [m for m in modules(ws, L['layer'], comp) if m['source'] == legacy] \
                if isinstance(L.get('layer'), int) and 0 <= L['layer'] < ws.struct.n_layers else []
            mods.append({'component': comp, 'module': here[0]['path'] if here else None, 'source': legacy,
                         **{k: c[k] for k in ('d_in', 'captured', 'why', 'x', 'y') if k in c}})
        L['modules'] = mods


def io_arrays(ws, name, info, layer):
    """-> (good [N_good, row], bad [N_bad, row]) float32 of one layer, read from its two files."""
    L = info['layers'][layer]
    d = _paths(ws, name)['io']
    out = []
    for label in ('good', 'bad'):
        try:
            a = np.load(os.path.join(d, L['files'][label]), allow_pickle=False)
        except Exception as e:
            raise DirectionError(f"direction {name!r}: the module inputs and outputs of layer {layer} cannot be read "
                                 f"({type(e).__name__}: {e}); measure it again with module_io") from None
        if a.dtype != np.float32 or a.shape != (info['counts'][label], L['row']):
            raise DirectionError(f"direction {name!r}: the module inputs and outputs of layer {layer} are not float32 "
                                 f"{[info['counts'][label], L['row']]}; measure it again with module_io")
        out.append(torch.from_numpy(np.ascontiguousarray(a)))
    return out[0], out[1]


# ------------------------------------------------------------------ directions from the two means
def _unit_rows(x):
    n = x.norm(dim=-1, keepdim=True)
    ok = n > 0
    return torch.where(ok, x / torch.where(ok, n, torch.ones_like(n)), torch.zeros_like(x)), ok.squeeze(-1)


def table(good_mean, bad_mean, orthogonalize_direction=True) -> torch.Tensor:
    """One unit direction per index, float32 [n + 1, d]: r = unit(bad - good); with ``orthogonalize_direction``,
    r = unit(r - (r . g) g) with g = unit(good). A row whose norm is 0 at either step is all zeros (unusable)."""
    good, bad = good_mean.double(), bad_mean.double()
    r, ok = _unit_rows(bad - good)
    if orthogonalize_direction:
        g, _ = _unit_rows(good)
        r, ok2 = _unit_rows(r - (r * g).sum(-1, keepdim=True) * g)
        ok = ok & ok2
    return torch.where(ok[:, None], r, torch.zeros_like(r)).float()


def usable(tab) -> list[bool]:
    return [bool(x) for x in (tab.double().norm(dim=-1) > 0)]


def stats(good_mean, bad_mean, orthogonalize_direction=True) -> list[dict]:
    """Per index, from the two means only: their norms, the norm of their difference, their cosine, and whether
    a direction can be taken there (difference not 0, and after the projection when it is asked for)."""
    good, bad = good_mean.double(), bad_mean.double()
    gn, bn, dn = good.norm(dim=-1), bad.norm(dim=-1), (bad - good).norm(dim=-1)
    dot = (good * bad).sum(-1)
    ok = usable(table(good_mean, bad_mean, orthogonalize_direction))
    out = []
    for k in range(good.shape[0]):
        cos = float(dot[k] / (gn[k] * bn[k])) if gn[k] > 0 and bn[k] > 0 else None
        out.append({'index': k, 'good_norm': float(gn[k]), 'bad_norm': float(bn[k]), 'diff_norm': float(dn[k]),
                    'cosine': cos, 'usable': bool(dn[k] > 0) and ok[k]})
    return out


def _index(direction_index, n):
    """-> (i, f) for a number direction_index (p = x + 1, i = floor(p), f = p - i), refused outside the table."""
    p = float(direction_index) + 1.0
    i = math.floor(p)
    f = p - i
    if not (0 <= i and i + (1 if f > 0 else 0) <= n):
        raise DirectionError(f"direction_index {direction_index!r} is outside the measurement: it must be from -1 to "
                             f"{n - 1} (index k of the table is direction_index k - 1)")
    return i, f


def direction_at(tab, direction_index, layer, name='') -> torch.Tensor:
    """The unit direction a layer uses: ``table[layer + 1]`` when direction_index is None, else one direction for
    every layer, interpolated between table rows i and i + 1 (p = direction_index + 1, i = floor(p), f = p - i)."""
    n = tab.shape[0] - 1
    what = f"direction {name!r}" if name else "the direction"
    ok = usable(tab)
    if direction_index is None:
        k = layer + 1
        if not 0 <= k <= n:
            raise DirectionError(f"layer {layer} has no index in {what}")
        if not ok[k]:
            raise DirectionError(f"index {k} of {what} (used by layer {layer}) is unusable: the good and bad means do not "
                                 f"differ there (after the projection, when it is asked for)")
        return tab[k].clone()
    i, f = _index(direction_index, n)
    rows = [i] if f == 0 else [i, i + 1]
    for k in rows:
        if not ok[k]:
            raise DirectionError(f"index {k} of {what} (used by direction_index {direction_index!r}) is unusable: the "
                                 f"good and bad means do not differ there (after the projection, when it is asked for)")
    if f == 0:
        return tab[i].clone()
    mix = (1 - f) * tab[i].double() + f * tab[i + 1].double()
    nrm = mix.norm()
    if not nrm > 0:
        raise DirectionError(f"the directions at indices {i} and {i + 1} of {what} cancel at direction_index "
                             f"{direction_index!r}")
    return (mix / nrm).float()


def _describe_index(direction_index, layer, n):
    if direction_index is None:
        return f"index {layer + 1}"
    i, f = _index(direction_index, n)
    return f"index {i}" if f == 0 else f"between {i} and {i + 1} ({f:.6g})"


# ------------------------------------------------------------------ per-layer strength
def layer_weight(dist, layer):
    """-> (weight, None) or (None, why the layer is skipped)."""
    d = abs(layer - dist['max_weight_position'])
    if d > dist['min_weight_distance']:
        return None, (f"farther than min_weight_distance ({dist['min_weight_distance']:g}) from max_weight_position "
                      f"({dist['max_weight_position']:g})")
    w = dist['max_weight'] + (d / dist['min_weight_distance']) * (dist['min_weight'] - dist['max_weight'])
    if w == 0:
        return None, "weight 0"
    return w, None


# ------------------------------------------------------------------ parameters
def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DirectionError(f"{field}: must be a finite number")
    return float(value)


def normalize(params) -> dict:
    """The ablation parameters checked without the model or the measurement -> a dict with every field."""
    if not isinstance(params, dict):
        raise DirectionError("the ablation parameters must be an object")
    unknown = sorted(set(params) - set(PARAMS), key=str)
    if unknown:
        raise DirectionError(f"{unknown[0]}: unknown field; the fields are {', '.join(PARAMS)}")
    if 'direction' not in params:
        raise DirectionError("direction: required (the name of a saved measurement)")
    try:
        name = check_name(params['direction'])
    except DirectionError as e:
        raise DirectionError(f"direction: {e}") from None
    di = params.get('direction_index')
    if di is not None:
        di = _number(di, 'direction_index')
    wd = params.get('weight_distributions')
    if not isinstance(wd, dict) or not wd:
        raise DirectionError(f"weight_distributions: must be an object giving at least one of {', '.join(COMPONENTS)}")
    for comp in wd:
        if comp not in COMPONENTS:
            raise DirectionError(f"weight_distributions.{comp}: unknown component; the components are {', '.join(COMPONENTS)}")
    dists = {}
    for comp in COMPONENTS:                                 # a fixed order, whatever order they were given in
        if comp not in wd:
            continue
        d = wd[comp]
        field = f"weight_distributions.{comp}"
        if not isinstance(d, dict):
            raise DirectionError(f"{field}: must be an object with {', '.join(DISTRIBUTION_KEYS)}")
        extra = sorted(set(d) - set(DISTRIBUTION_KEYS), key=str)
        if extra:
            raise DirectionError(f"{field}.{extra[0]}: unknown field; the fields are {', '.join(DISTRIBUTION_KEYS)}")
        out = {}
        for k in DISTRIBUTION_KEYS:
            if k not in d:
                raise DirectionError(f"{field}.{k}: required")
            out[k] = _number(d[k], f"{field}.{k}")
        for k in ('max_weight', 'min_weight'):
            if abs(out[k]) > MAX_WEIGHT:
                raise DirectionError(f"{field}.{k}: must be from -{MAX_WEIGHT:g} to {MAX_WEIGHT:g}")
        if not out['min_weight_distance'] > 0:
            raise DirectionError(f"{field}.min_weight_distance: must be above 0")
        dists[comp] = out
    ortho = params.get('orthogonalize_direction', True)
    if not isinstance(ortho, bool):
        raise DirectionError("orthogonalize_direction: must be true or false")
    mode = params.get('row_normalization', 'none')
    if mode not in ROW_NORMALIZATIONS:
        raise DirectionError(f"row_normalization: must be one of {', '.join(ROW_NORMALIZATIONS)}")
    rank = params.get('full_normalization_lora_rank', 3)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= MAX_RANK:
        raise DirectionError(f"full_normalization_lora_rank: must be an integer from 1 to {MAX_RANK}")
    out = {'direction': name, 'direction_index': di, 'weight_distributions': dists, 'orthogonalize_direction': ortho,
           'row_normalization': mode, 'full_normalization_lora_rank': rank}
    em = expert_modules_of(params)
    if em != 'all':                 # only when not the default, so Heretic's parameters paste in and compare unchanged
        out['expert_modules'] = em
    return out


def expert_modules_of(params) -> str:
    """``expert_modules`` of ablation or ARA parameters, checked: "all" (the default) or "shared"."""
    em = params.get('expert_modules', 'all')
    if not isinstance(em, str) or em not in EXPERT_MODULES:
        raise DirectionError(EXPERT_MODULES_ERROR)
    return em


def component_views(ws, layer, component):
    """The writer views ([d_in, d_model], the residual stream last) of the first module of a component of a layer
    (its first source's, attn / mlp, when the layer has that one; else the first of its modules), as
    Semantic.writers() gives them (one per head for a 3-D attention output), or None when it has none."""
    mods = modules(ws, layer, component)
    if not mods:
        return None
    first = [m for m in mods if m['source'] == COMPONENT_SOURCES[component][0]['group']]
    return (first or mods)[0]['views']


def _stored_per_view(view, mode, rank, d):
    return view.shape[0] + d if mode != 'full' else (view.shape[0] + d) * min(rank, d)


def _row_groups(mods):
    """A layer's modules of one component -> the plan's rows: one per module, the routed experts together in one
    (at the place of the first)."""
    groups, routed = [], None
    for m in mods:
        if not m['routed']:
            groups.append([m])
        elif routed is None:
            routed = [m]
            groups.append(routed)
        else:
            routed.append(m)
    return groups


def prepare(ws, params):
    """Parameters checked against the model and the saved measurement -> (params, work, table) where work is one
    row per layer, component and module: {layer, component, module, weight, skipped, direction, views, stored}
    (+ private modules and vector). A layer's routed experts are one row: module ``mlp.experts.*.down_proj``,
    ``modules`` E, stored summed."""
    p = normalize(params)
    require_layout(ws)
    s = ws.struct
    n, d = s.n_layers, s.d_model
    em = p.get('expert_modules', 'all')
    if em == 'shared' and not has_routed_experts(ws):
        raise DirectionError(NO_EXPERTS)
    for comp, dist in p['weight_distributions'].items():
        if not -n <= dist['max_weight_position'] <= 2 * n:
            raise DirectionError(f"weight_distributions.{comp}.max_weight_position: must be from {-n} to {2 * n} "
                                 f"(this model has {n} layers)")
    if p['direction_index'] is not None:
        _index(p['direction_index'], n)
    meta, good, bad = load(ws, p['direction'])
    matches(ws, meta, p['direction'])
    tab = table(good, bad, p['orthogonalize_direction'])
    rows, present = [], {c: False for c in p['weight_distributions']}
    for l in range(n):
        for comp, dist in p['weight_distributions'].items():
            mods = modules(ws, l, comp, em)
            base = {'layer': l, 'component': comp, 'module': None, 'weight': None, 'skipped': None,
                    'direction': _describe_index(p['direction_index'], l, n), 'views': 0, 'stored': 0}
            if not mods:
                rows.append({**base, 'skipped': f"this layer has no {comp}"})
                continue
            present[comp] = True
            w, why = layer_weight(dist, l)
            for group in _row_groups(mods):
                row = dict(base)
                if group[0]['routed']:
                    row.update(module=routed_path(group[0]['path']), modules=len(group))
                else:
                    row['module'] = group[0]['path']
                if w is None:
                    row['skipped'] = why
                    rows.append(row)
                    continue
                views = [v for m in group for v in m['views']]
                for v in views:
                    if not ws.editable(v.name):
                        raise DirectionError(f"layer {l} {comp}: {v.name} is not a floating-point tensor and cannot be edited")
                row.update(weight=w, views=len(views),
                           stored=sum(_stored_per_view(v, p['row_normalization'], p['full_normalization_lora_rank'], d) for v in views))
                row['_modules'] = group
                row['_vector'] = direction_at(tab, p['direction_index'], l, p['direction'])
                rows.append(row)
    for comp, ok in present.items():
        if not ok:
            raise DirectionError(f"weight_distributions.{comp}: this model has no {comp} in any layer")
    return p, rows, tab


def _public(row):
    return {k: v for k, v in row.items() if not k.startswith('_')}


def method_of(params):
    """-> (method, parameters for that method): ``method`` "ablation" (the default when absent; the key is dropped)
    or "ara" (wsp.directions_ara)."""
    if not isinstance(params, dict):
        raise DirectionError("the ablation parameters must be an object")
    m = params.get('method', 'ablation')
    if not isinstance(m, str) or m not in METHODS:
        raise DirectionError(f"method: must be one of {', '.join(METHODS)}")
    return m, ({k: v for k, v in params.items() if k != 'method'} if m == 'ablation' else params)


def plan(ws, params) -> dict:
    """What an ablation will do, without reading a weight or running the model: one row per layer and component
    {layer, component, weight | None, skipped: reason | None, direction, views, stored} plus totals. With
    ``"method": "ara"``, wsp.directions_ara.plan."""
    m, params = method_of(params)
    if m == 'ara':
        from . import directions_ara
        return directions_ara.plan(ws, params)
    p, rows, _ = prepare(ws, params)
    written = [r for r in rows if r['skipped'] is None]
    totals = {'layers': len({r['layer'] for r in written}), 'components': len(written),
              'views': sum(r['views'] for r in written), 'stored': sum(r['stored'] for r in written),
              'skipped': len(rows) - len(written)}
    n_modules = sum(r.get('modules', 1) for r in written)
    if n_modules != len(written):          # a row of routed experts holds several modules; otherwise one row, one module
        totals['modules'] = n_modules
    return {'direction': p['direction'], 'row_normalization': p['row_normalization'], 'rows': [_public(r) for r in rows],
            'totals': totals}


def _num(x) -> str:
    return repr(float(x))                  # repr round-trips a float exactly


def script_for(params) -> str:
    """The one-line weight script for these ablation parameters (``"method": "ara"``: ablate_ara's)."""
    m, params = method_of(params)
    if m == 'ara':
        from . import directions_ara
        return directions_ara.script_for(params)
    p = normalize(params)
    wd = ', '.join('"%s": {%s}' % (c, ', '.join(f'"{k}": {_num(d[k])}' for k in DISTRIBUTION_KEYS))
                   for c, d in p['weight_distributions'].items())
    di = 'None' if p['direction_index'] is None else _num(p['direction_index'])
    em = f", expert_modules={json.dumps(p['expert_modules'])}" if 'expert_modules' in p else ''
    return (f"ablate_direction({json.dumps(p['direction'])}, direction_index={di}, weight_distributions={{{wd}}}, "
            f"orthogonalize_direction={p['orthogonalize_direction']!r}, row_normalization={json.dumps(p['row_normalization'])}, "
            f"full_normalization_lora_rank={p['full_normalization_lora_rank']}{em})")


# ------------------------------------------------------------------ the change to one component
def _slabs(ws, view, extra, stored, scratch=0):
    """The view [d_in, d] in row slabs of at most STREAM_ELEMS values (never whole above it)."""
    d_in, d = view.shape
    step = max(1, W.STREAM_ELEMS // max(1, d))
    for a in range(0, d_in, step):
        part = view.sub(slice(a, min(a + step, d_in)))
        edit_preflight(ws, stored(), scratch=part.size * 12 + scratch)
        yield part.read('current', extra)


def column_sq(ws, views, extra=None, stored=lambda: 0) -> torch.Tensor:
    """||V[:, j]||^2 over the whole component (every view's rows), float64 [d]."""
    d = views[0].shape[1]
    sq = torch.zeros(d, dtype=torch.float64)
    for view in views:
        for t in _slabs(ws, view, extra, stored):
            sq += t.double().pow(2).sum(0)
    return sq


def gram(ws, views, extra=None, stored=lambda: 0) -> torch.Tensor:
    """G = V^T V over the whole component, float64 [d, d], accumulated over row slabs of every view."""
    d = views[0].shape[1]
    G = torch.zeros(d, d, dtype=torch.float64)
    for view in views:
        for t in _slabs(ws, view, extra, stored, scratch=d * d * 8):
            t = t.double()
            G.addmm_(t.T, t)
    return G


def _safe_div(a, b, ok):
    return torch.where(ok, a / torch.where(ok, b, torch.ones_like(b)), torch.zeros_like(a))


def _contract(ws, view, M, extra, stored):
    edit_preflight(ws, stored(), scratch=min(view.size, W.STREAM_ELEMS) * 12 + view.shape[0] * M.reshape(M.shape[0], -1).shape[1] * 8)
    return view.contract(M, axis=1, extra=extra)


def change(ws, views, v, w, mode='none', rank=3, extra=None, stored=lambda: 0) -> tuple[list, dict]:
    """The change to one component V ([D_in, d], rows of every view) for unit direction v and strength w
    -> ([(view, A, B)], info): delta = A @ B per view (A [d_in, r], B [r, d], float32)."""
    d = views[0].shape[1]
    v = v.to(torch.float32)
    if mode == 'none':                     # delta = -w (V v) v^T
        return [(view, -w * _contract(ws, view, v, extra, stored), v) for view in views], {}
    if mode == 'pre':                      # delta = -w (V x) y^T, x = v / nrm, y = nrm * v
        nrm = column_sq(ws, views, extra, stored).sqrt()
        ok = nrm > 0
        v64 = v.double()
        x, y = _safe_div(v64, nrm, ok).float(), (nrm * v64).float()
        return [(view, -w * _contract(ws, view, x, extra, stored), y) for view in views], {}
    if mode != 'full':
        raise DirectionError(f"row_normalization: must be one of {', '.join(ROW_NORMALIZATIONS)}")
    edit_preflight(ws, stored(), scratch=4 * d * d * 8)          # G, then M, its eigenvectors and eigh's workspace
    G = gram(ws, views, extra, stored)
    nrm = G.diagonal().clamp_min(0).sqrt()
    ok = nrm > 0
    v64 = v.double()
    x = _safe_div(v64, nrm, ok)
    t = G @ x
    alpha = float(x @ t)
    c = _safe_div(t, nrm, ok)
    s = torch.sqrt(torch.clamp(1 - 2 * w * v64 * c + w * w * v64 * v64 * alpha, min=1e-24))
    g = torch.where(ok, 1 / s - 1, torch.zeros_like(s))
    u = torch.where(ok, nrm * w * v64 / s, torch.zeros_like(s))
    gt = g * t
    # M = diag(g) G diag(g) - (g t) u^T - u (g t)^T + alpha u u^T, built in G's memory (G is not needed again). It is
    # symmetric by construction; eigh reads its lower triangle, so no symmetrised copy is made.
    M = G
    del G
    M.mul_(g[:, None]).mul_(g[None, :])
    M.addr_(gt, u, alpha=-1).addr_(u, gt, alpha=-1).addr_(u, u, alpha=alpha)
    trace = float(M.diagonal().sum())
    if not trace > 0:
        return [], {'full_error': 0.0, 'trace': trace}
    r = min(int(rank), d)
    evals, evecs = torch.linalg.eigh(M)
    del M
    R = evecs[:, -r:].clone()
    # A zero column of V has g = u = 0, so its row and column of M are 0 and every eigenvector of a nonzero eigenvalue
    # has 0 there; eigh leaves rounding noise (~1e-19) instead. Exact zeros keep such a column exactly 0.
    R[~ok] = 0
    top = float(evals[-r:].sum())
    info = {'full_error': math.sqrt(max(0.0, 1 - top / trace)), 'trace': trace, 'rank': r}
    uR = (u @ R).float()
    mat = torch.cat([g[:, None] * R, x[:, None]], 1).float()      # [d, r + 1]
    B = R.T.float().contiguous()
    out = []
    for view in views:
        P = _contract(ws, view, mat, extra, stored)              # [d_in, r + 1]
        out.append((view, P[:, :r] - P[:, r:r + 1] * uR[None, :], B))
    return out, info


def ablate(ws, params, extra=None, add=None, stored=lambda: 0, progress=None) -> dict:
    """Run the ablation against the current weights plus ``extra`` and hand each piece to ``add(view, A, B)``
    (the interpreter's, so the edit goes through Session.apply). -> report {components: [...], stored}."""
    p, rows, _ = prepare(ws, params)
    work = [(row, m) for row in rows if row['skipped'] is None for m in row['_modules']]
    report = []
    for i, (row, m) in enumerate(work):
        if progress is not None:
            progress(i, len(work))
        # One module at a time: its own column norms (pre, full), the component's weight (Heretic's).
        pieces, info = change(ws, m['views'], row['_vector'], row['weight'], p['row_normalization'],
                              p['full_normalization_lora_rank'], extra, stored)
        n = 0
        for view, A, B in pieces:
            if bool(A.any()) and bool(B.any()):
                add(view, A, B)
                n += A.numel() + B.numel()
        entry = {k: v for k, v in _public(row).items() if k != 'modules'}
        entry.update(module=m['path'], views=len(m['views']))
        report.append({**entry, 'stored': n, **({'full_error': info['full_error']} if 'full_error' in info else {})})
    return {'direction': p['direction'], 'row_normalization': p['row_normalization'], 'components': report,
            'skipped': [_public(r) for r in rows if r['skipped'] is not None], 'stored': sum(r['stored'] for r in report)}


# ------------------------------------------------------------------ the measurement job (where the model runs)
@job_scope('direction_residuals')
@worker_job('direction_residuals')
def residuals(ws, prompts_ids, variant='base', module_io=False):
    """The residual stream at the last token of each prompt -> float32 [prompts, n_layers + 1, d_model]: index k < n
    entering layer k (``hook_resid_pre``), index n the last layer's ``hook_resid_post`` after the final norm. Each
    prompt runs alone (batch 1) on the committed weights of ``variant``.

    With ``module_io`` the same pass also keeps every layer's component inputs and outputs, and the answer is one
    packed row per prompt, float32 [prompts, (n + 1) d + width]: the residual vectors flattened (index-major), then
    layer by layer the parts ``io_layout(ws, runner.available)`` places there (per captured component its input,
    trailing dims flattened, then its output). Both sides derive that layout from the structure and the runner's
    hook points; at most ``io_job_prompts`` prompts per job."""
    if not isinstance(module_io, bool):
        raise DirectionError("module_io must be true or false")
    if variant not in VARIANTS:
        raise DirectionError("variant must be 'base' or 'current'")
    layout = None
    if module_io:
        layout = io_layout(ws, ws.runner.available)
        most = io_job_prompts(ws, layout)
        if not isinstance(prompts_ids, list) or not 1 <= len(prompts_ids) <= most:
            raise DirectionError(f"a direction_residuals job with module_io takes 1..{most} prompts on this model")
    if not isinstance(prompts_ids, list) or not 1 <= len(prompts_ids) <= JOB_PROMPTS:
        raise DirectionError(f"a direction_residuals job takes 1..{JOB_PROMPTS} prompts")
    r = ws.runner
    ctx = int(getattr(r, 'max_ctx', 0) or 0)
    limit = min(MAX_TOKENS, ctx) if ctx else MAX_TOKENS
    checked = []
    for i, ids in enumerate(prompts_ids):
        ids = _ids(list(ids) if isinstance(ids, (list, tuple)) else ids)
        if not 1 <= len(ids) <= limit:
            raise DirectionError(f"prompt {i + 1} must have 1..{limit} tokens (it has {len(ids)})")
        checked.append(ids)
    n, d = ws.struct.n_layers, ws.struct.d_model
    names = [f'blocks.{l}.hook_resid_pre' for l in range(n)] + [f'blocks.{n - 1}.hook_resid_post']
    for name in names:
        if not r.available(name):
            raise DirectionError(f"hook point {name} is not available for this model")
    parts = [(L['layer'], c['component'], c) for L in layout for c in L['modules'] if c['captured']] if module_io else []
    io_names = [h for _, _, c in parts for h in c['hooks']]
    out = []
    for ids in checked:
        with torch.no_grad():
            _, cache = r.run(ids, variant, None, names=names + io_names)
            missing = [nm for nm in names + io_names if nm not in cache]
            if missing:
                raise DirectionError(f"hook point {missing[0]} was not captured")
            rows = [cache[nm][0, -1].float().cpu() for nm in names[:-1]]
            rows.append(r.final_norm(cache[names[-1]][0, -1], variant).float().cpu())
            if not module_io:
                out.append(torch.stack(rows))
                continue
            packed = [torch.stack(rows).reshape(-1)]
            for l, comp, c in parts:
                x = cache[c['hooks'][0]][0, -1].reshape(-1).float().cpu()
                y = cache[c['hooks'][1]][0, -1].reshape(-1).float().cpu()
                if x.numel() != c['d_in'] or y.numel() != d:
                    raise DirectionError(f"layer {l} {comp}: hook points {c['hooks'][0]} and {c['hooks'][1]} give "
                                         f"{x.numel()} and {y.numel()} values; the matrix reads {c['d_in']} and writes {d}")
                packed += [x, y]
            out.append(torch.cat(packed))
    return torch.stack(out).to(torch.float32).contiguous()


# ------------------------------------------------------------------ the measurement (server process)
def _texts(value, label):
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_PROMPTS:
        raise DirectionError(f"{label} must list 1..{MAX_PROMPTS} prompts")
    for i, t in enumerate(value):
        if not isinstance(t, str) or not 1 <= len(t) <= MAX_CHARS:
            raise DirectionError(f"{label} prompt {i + 1} must be text of 1..{MAX_CHARS} characters")
    return value


def _prompt_value(value, label, field):
    """A prompt field (wsp.directions_datasets, 5.1) where ``_texts`` was called: a list without objects takes the
    old path, ``_texts(value, label)`` with its checks and messages (a list of texts); otherwise the checked items of
    ``directions_datasets.items`` (``(path, item)`` pairs). Only the shape is checked here: nothing is read."""
    from . import directions_datasets as DD
    try:
        items = DD.items(value, label, field)
    except DD.DatasetError as e:
        raise DirectionError(str(e)) from None
    return _texts(value, label) if items is None else items


def _prompt_set(ws, value, label, field, *, system, chat):
    """After system and chat are checked: ``_prompt_value``'s answer -> (the prompt set, its record). The old path
    is (the texts, None); items are resolved offline from files (``directions_datasets.resolve``) into the canonical
    set under the request's system message, and a prompt with a system message of its own is refused without chat."""
    if all(isinstance(t, str) for t in value):
        return value, None
    from . import directions_datasets as DD
    try:
        return DD.resolve(value, label, field, system=system, state=DD.state_dir(ws), allow_system=chat)
    except DD.DatasetError as e:
        raise DirectionError(str(e)) from None


def _optional_text(value, field):
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > MAX_CHARS:
        raise DirectionError(f"{field} must be text of at most {MAX_CHARS} characters")
    return value


def _encode(ws, encode, text, label, i):
    try:
        ids = [int(t) for t in encode(text)]
    except (ValueError, KeyError) as e:
        raise DirectionError(f"{label} prompt {i + 1}: {e}") from None
    return ids


def _token_limit(ws):
    n_ctx = int(getattr(ws.struct, 'n_ctx', 0) or 0)
    return min(MAX_TOKENS, n_ctx) if n_ctx else MAX_TOKENS


def _check_tokens(ids, limit, label, i, how=''):
    if not 1 <= len(ids) <= limit:
        raise DirectionError(f"{label} prompt {i + 1}{how} has {len(ids)} tokens; it must have 1..{limit}")


def check_measure(ws, name, good, bad, *, chat=False, system=None, prefix=None, chat_options=None, variant='base',
                  winsorization_quantile=1.0, replace=False, module_io=False):
    """Everything about a measurement that needs no model (the server calls it before loading anything; measure
    calls it again) -> the checked values, with each prompt's token ids when it is not read as a chat. With
    ``module_io``, the disk guard too."""
    from . import probes
    require_layout(ws)
    if ws.tok is None:
        raise DirectionError("this model has no tokenizer, so prompts cannot be encoded")
    check_name(name)
    if not isinstance(replace, bool):
        raise DirectionError("replace must be true or false")
    if not isinstance(module_io, bool):
        raise DirectionError("module_io must be true or false")
    if exists(ws, name) and not replace:
        raise DirectionError(f"a direction named {name!r} exists; pass replace to measure it again")
    good, bad = _prompt_value(good, 'good', 'good'), _prompt_value(bad, 'bad', 'bad')
    if not isinstance(chat, bool):
        raise DirectionError("chat must be true or false")
    if variant not in VARIANTS:
        raise DirectionError("variant must be 'base' or 'current'")
    q = winsorization_quantile
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or not 0 <= q <= 1:
        raise DirectionError("winsorization_quantile must be a number from 0 to 1 (1 is off)")
    system, prefix = _optional_text(system, 'system'), _optional_text(prefix, 'prefix')
    options = None
    if chat:
        options = probes.check_chat_options(chat_options)
        ok, reason = probes.chat_template(ws)
        if not ok:
            raise DirectionError(f"'as a chat message' needs the model's own chat template: {reason}")
        encode = probes.chat_encode(ws)
    else:
        if system is not None or prefix is not None:
            raise DirectionError("system and prefix apply to prompts read as a chat message; set chat to true or leave them out")
        if chat_options not in (None, {}):
            raise DirectionError("chat_options apply to prompts read as a chat message; set chat to true or leave them out")
        encode = ws.tok.encode
    from .directions_datasets import user_of
    good, good_rec = _prompt_set(ws, good, 'good', 'good', system=system, chat=chat)
    bad, bad_rec = _prompt_set(ws, bad, 'bad', 'bad', system=system, chat=chat)
    limit = _token_limit(ws)
    ids = {}
    for label, texts in (('good', good), ('bad', bad)):
        ids[label] = []
        for i, t in enumerate(texts):
            x = _encode(ws, encode, user_of(t), label, i)
            _check_tokens(x, limit, label, i)
            ids[label].append(x)
    if module_io:
        check_disk(ws, len(good), len(bad))
    return {'name': name, 'good': good, 'bad': bad, 'chat': chat, 'system': system, 'prefix': prefix,
            'chat_options': options, 'variant': variant, 'winsorization_quantile': float(q), 'replace': replace,
            'module_io': module_io, 'ids': None if chat else ids, 'datasets': {'good': good_rec, 'bad': bad_rec}}


def render_ids(ws, texts, label, *, chat, system=None, prefix=None, chat_options=None, rendering=None,
               _version=None) -> list[list[int]]:
    """Each text's token ids, checked against the token bound. Without ``chat``, the text as encoded. With ``chat``
    (after the runner exists): the text as one user turn (after the system message) rendered and tokenized as Heretic
    does (wsp.directions_rendering: the system role folded where the template has none, the date fixed, the prefix
    appended, the tokenizer's own special tokens added). A prompt is a text (the request's ``system``) or
    ``{"system", "user"}`` with its own system message (wsp.directions_datasets). ``rendering`` (a Renderer for
    ``chat_options``) is reused when given, so its system-role probe runs once; ``_version`` 1 renders as the IDE
    rendered before version 2, only to compare the two (``rendering_entry``)."""
    from . import directions_rendering as RD
    limit = _token_limit(ws)
    out = []
    if not chat:
        for i, text in enumerate(texts):
            x = _encode(ws, ws.tok.encode, text, label, i)
            _check_tokens(x, limit, label, i)
            out.append(x)
        return out
    r = rendering or RD.Renderer(ws, chat_options, _version or RD.VERSION)
    for i, text in enumerate(texts):
        user, own = (text, system) if isinstance(text, str) else (text["user"], text["system"])
        x = _encode(ws, r.encode, r.text(user, own, prefix), label, i)
        _check_tokens(x, limit, label, i, ' as a chat message')
        out.append(x)
    return out


def _rendered_ids(ws, checked, rendering=None):
    """After the runner exists: the good and bad prompts as chat messages (``render_ids``, one Renderer)."""
    from . import directions_rendering as RD
    r = rendering or RD.Renderer(ws, checked['chat_options'])
    return {label: render_ids(ws, checked[label], label, chat=True, system=checked['system'], prefix=checked['prefix'],
                              rendering=r) for label in ('good', 'bad')}


def rendering_entry(ws, checked, ids, rendering) -> dict:
    """A measurement's ``rendering``: the version its ids were rendered with (wsp.directions_rendering) and whether
    version 1, the IDE's rendering before, gives the same ids for these prompts on this model on any day (never when
    the template read the date). ``rendering`` is the Renderer that made ``ids``. Encoded prompts are the same in both."""
    from . import directions_rendering as RD
    same = True
    if checked['chat']:
        try:
            for label in ('good', 'bad'):
                if same:
                    same = render_ids(ws, checked[label], label, chat=True, system=checked['system'],
                                      prefix=checked['prefix'], chat_options=checked['chat_options'], _version=1) == ids[label]
        except Exception:                  # version 1 cannot render them (a template that raises on a system role)
            same = False
        same = same and not rendering.dated
    return {'version': RD.VERSION, 'same_as_version_1': bool(same)}


def winsorize(x, q):
    """Clamp each row's components to +-(the q quantile of their absolute values); q = 1 is off."""
    if q >= 1:
        return x
    t = torch.quantile(x.abs(), q, dim=-1, keepdim=True)
    return torch.maximum(torch.minimum(x, t), -t)


def _sha(texts):
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def runner_description(ws) -> dict | None:
    """The loaded runner's class, dtype and device, or None when none is loaded (never loads one); plus ``apply``, how
    patches are applied, only when that is not the dtype's default (wsp.runner.APPLY), so a run in the default mode
    is described exactly as before."""
    from .runner import nondefault_apply
    r = getattr(ws, '_runner', None)
    if r is None:
        return None
    out = {'class': type(r).__name__, 'dtype': str(getattr(r, 'dtype', None)), 'device': str(getattr(r, 'device', None))}
    mode = nondefault_apply(r)
    if mode is not None:
        out['apply'] = mode
    return out


def measurement_environment(ws) -> dict:
    """What a measurement records about how it ran: the library and sampler versions, Python, the thread count, the
    machine type and system (directions_search.environment; no host, user or path) and the runner."""
    from . import directions_search as DSR           # it imports this module at its top
    return {**DSR.environment(), 'runner': runner_description(ws)}


def measure(ws, name, good, bad, *, chat=False, system=None, prefix=None, chat_options=None, variant='base',
            winsorization_quantile=1.0, replace=False, module_io=False, progress=None, stopped=None):
    """Measure and save one direction -> the saved meta plus per-index statistics, or {'stopped': True, ...} with
    nothing saved when ``stopped()`` is true between chunks. ``progress(done, total)`` after every chunk. With
    ``module_io`` the same passes also fill ``<name>.io/`` (removed again on a stop or a failure)."""
    from .steering import model_identity
    from .directions_datasets import VERSION as DD_VERSION
    checked = check_measure(ws, name, good, bad, chat=chat, system=system, prefix=prefix, chat_options=chat_options,
                            variant=variant, winsorization_quantile=winsorization_quantile, replace=replace,
                            module_io=module_io)
    from . import directions_rendering as RD
    renderer = RD.Renderer(ws, checked['chat_options']) if chat else None
    ids = _rendered_ids(ws, checked, renderer) if chat else checked['ids']
    rendering = rendering_entry(ws, checked, ids, renderer)
    from .atlas import wiring
    # the patches the passes run on (the base when variant is 'base'); read before the passes start
    measured_on = wiring.patch_fingerprint(ws) if checked['variant'] == 'current' else []
    s = ws.struct
    n, d = s.n_layers, s.d_model
    q = checked['winsorization_quantile']
    total = len(ids['good']) + len(ids['bad'])
    counts = {'good': len(ids['good']), 'bad': len(ids['bad'])}
    done = 0
    started = time.monotonic()
    sums = {}
    io, chunk_size, width = None, CHUNK, (n + 1) * d
    if checked['module_io']:
        layout = io_layout(ws, ws.runner.available)
        chunk_size = min(CHUNK, io_job_prompts(ws, layout))
        width += io_width(layout)
        io = _IoWriter(ws, name, layout, counts)
    try:
        for label in ('good', 'bad'):
            acc = torch.zeros(n + 1, d, dtype=torch.float64)
            prompts_ids = ids[label]
            for a in range(0, len(prompts_ids), chunk_size):
                if stopped is not None and stopped():
                    if io is not None:
                        io.discard()
                    return {'stopped': True, 'name': name, 'done': done, 'total': total}
                chunk = prompts_ids[a:a + chunk_size]
                if io is None:
                    got = residuals(ws, chunk, checked['variant'])
                    want = (len(chunk), n + 1, d)
                else:
                    got = residuals(ws, chunk, checked['variant'], True)
                    want = (len(chunk), width)
                if not isinstance(got, torch.Tensor) or tuple(got.shape) != want:
                    raise DirectionError(f"the direction_residuals job answered {tuple(getattr(got, 'shape', ()))}, not {want}")
                if io is not None:
                    io.write(label, a, got[:, (n + 1) * d:])
                    got = got[:, :(n + 1) * d].reshape(len(chunk), n + 1, d)
                for k in range(len(chunk)):
                    acc += winsorize(got[k].float(), q).double()
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
            sums[label] = acc
        if stopped is not None and stopped():
            if io is not None:
                io.discard()
            return {'stopped': True, 'name': name, 'done': done, 'total': total}
        good_mean = (sums['good'] / len(ids['good'])).float()
        bad_mean = (sums['bad'] / len(ids['bad'])).float()
        meta = {'version': VERSION, 'name': name, 'created': int(time.time()), 'variant': checked['variant'],
                'n_layers': n, 'd_model': d, 'counts': counts,
                'tokens': {'good': sum(len(x) for x in ids['good']), 'bad': sum(len(x) for x in ids['bad'])},
                'options': {'chat': checked['chat'], 'system': checked['system'], 'prefix': checked['prefix'],
                            'chat_options': checked['chat_options'], 'winsorization_quantile': q},
                'prompts_sha256': {'good': _sha(checked['good']), 'bad': _sha(checked['bad'])},
                'model': model_identity(ws), 'module_io': io.finish() if io is not None else None,
                'datasets': {'version': DD_VERSION, **checked['datasets']},
                'environment': measurement_environment(ws), 'measured_on': measured_on, 'rendering': rendering}
        save(ws, name, good_mean, bad_mean, meta, {'good': checked['good'], 'bad': checked['bad']},
             replace=checked['replace'], io_dir=io.dir if io is not None else None)
    except BaseException:
        if io is not None:
            io.discard()
            if not os.path.isfile(_paths(ws, name)['meta']) and os.path.isdir(io.final):
                shutil.rmtree(io.final, ignore_errors=True)      # moved in, but the entry was never completed
        raise
    return {**meta, 'seconds': round(time.monotonic() - started, 3), 'stats': stats(good_mean, bad_mean)}


def measure_body(body) -> dict:
    """The measure arguments of a request body (``stream`` is the route's); an unknown field is refused."""
    if not isinstance(body, dict):
        raise DirectionError("the request must be an object")
    unknown = sorted(set(body) - set(MEASURE_FIELDS) - {'stream'}, key=str)
    if unknown:
        raise DirectionError(f"{unknown[0]}: unknown field; the fields are {', '.join(MEASURE_FIELDS)} and stream")
    for key in ('name', 'good', 'bad'):
        if key not in body:
            raise DirectionError(f"{key}: required")
    return {k: body[k] for k in MEASURE_FIELDS if k in body}
