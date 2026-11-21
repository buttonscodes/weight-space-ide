"""Versioned, declarative per-user settings. Importing this module never imports a model runtime."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .capacity import MAX_EXACT_INTEGER, DEFAULT_CONTEXT, DEFAULT_OUTPUT, NATIVE_MAX_THREADS

DEFAULTS = dict(root='workspaces', model='facts', host='127.0.0.1', port=8765,
                backend='auto', cpu_bridge=None, metal_bridge=None, threads=4,
                model_roots=[], context=DEFAULT_CONTEXT, new_tokens=DEFAULT_OUTPUT, resident_seconds=14400,
                floor_gib=None, preferred_gib=None, budget_gib=None, ceiling_gib=None, reservation_gib=None,
                swap_growth_mib=64, compression_growth_mib=1024, patch_limit_gib=None)
ENV = dict(root='WSP_ROOT', model='WSP_MODEL', backend='WSP_GGML_DEVICE',
           cpu_bridge='WSP_GGML_BRIDGE', metal_bridge='WSP_GGML_METAL_BRIDGE',
           threads='WSP_GGML_THREADS', context='WSP_MAX_CTX', new_tokens='WSP_MAX_NEW_TOKENS')
PATHS = {'root', 'cpu_bridge', 'metal_bridge'}
INTS = dict(port=(1,65535), threads=(1,MAX_EXACT_INTEGER), context=(1,MAX_EXACT_INTEGER), new_tokens=(1,MAX_EXACT_INTEGER),
            resident_seconds=(10,86400))
MEMORY = {'reservation_gib','floor_gib','preferred_gib','budget_gib','ceiling_gib','swap_growth_mib','compression_growth_mib','patch_limit_gib'}
# When a saved change takes effect (wsp.settings applies them; docs/SETTINGS.md describes them).
# The running server applies these as soon as they are saved, with or without a supervisor: the largest weight
# patch (wsp.edit_limit reads it) and the default output count (each Run and Chat request reads it).
SERVER_LIVE = {'patch_limit_gib', 'new_tokens'}
# The running supervisor applies these as soon as they are saved (memory policy and the idle clock). A running
# operation keeps the limits it was admitted with until it finishes. Without a supervisor: the next server start.
SUPERVISOR_LIVE = (MEMORY - {'patch_limit_gib'}) | {'resident_seconds'}
# Fixed when a model is loaded: the loaded model keeps them and the next model load uses the saved values.
# Without a supervisor: the next server start.
NEXT_LOAD = {'context', 'threads', 'backend'}
LIVE = SERVER_LIVE
# The smallest Largest weight patch accepted: 1 MiB of FP32 patch, about 262,000 weights (a few rows of a large
# matrix). A smaller value would refuse nearly every edit; Auto, not a tiny number, is the way to set no limit.
PATCH_LIMIT_MIN_GIB = 1/1024


def user_dir(kind='config', *, env=None, system=None):
    env = os.environ if env is None else env
    system = system or platform.system()
    home = Path(env.get('HOME', str(Path.home())))
    if env.get('XDG_'+kind.upper()+'_HOME'):
        return Path(env['XDG_'+kind.upper()+'_HOME']) / 'weight-space-ide'
    if system == 'Darwin':
        return home / 'Library' / ('Application Support' if kind == 'config' else 'Caches') / 'weight-space-ide'
    if system == 'Windows':
        return Path(env.get('LOCALAPPDATA', str(home / 'AppData' / 'Local'))) / 'weight-space-ide'
    return home / ('.config' if kind == 'config' else '.cache') / 'weight-space-ide'


def performance_core_count(system=None):
    """Metadata only; no native model or GPU runtime is imported.

    Apple Silicon exposes performance cores separately. If that key is absent,
    use the physical-core count (including Intel Macs). Elsewhere use the
    physical count; unknown counts use one. Discovery is cached for the process;
    each settings resolution still validates its effective thread setting.
    """
    return _discover_performance_cores(system or platform.system())


_DISCOVERED_CORES = {}


def _discover_performance_cores(system):
    # Only a definitive answer is cached. A transient failure (timeout, spawn
    # error, unparsable output) must pin neither the one-core fallback nor a
    # fallback that counts efficiency cores for the life of the process.
    if system in _DISCOVERED_CORES: return _DISCOVERED_CORES[system]
    definitive = True
    if system == 'Darwin':
        for key in ('hw.perflevel0.physicalcpu', 'hw.physicalcpu'):
            try:
                count = int(subprocess.check_output(['sysctl', '-n', key], text=True,
                    stderr=subprocess.DEVNULL, timeout=2).strip())
                if count > 0:
                    if definitive: _DISCOVERED_CORES[system] = count
                    return count
                definitive = False
            except subprocess.CalledProcessError as exc:
                # Exit status 1 means the key does not exist on this host (Intel
                # Macs): definitive. A probe killed by a signal is not an answer.
                if exc.returncode <= 0: definitive = False
            except (OSError, ValueError, subprocess.SubprocessError):
                definitive = False
    try:
        import psutil
        count = psutil.cpu_count(logical=False)
        if type(count) is int and count > 0:
            if definitive: _DISCOVERED_CORES[system] = count
            return count
    except ImportError:
        pass
    return 1


def runtime_thread_limit(system=None):
    return min(performance_core_count(system), NATIVE_MAX_THREADS)


def _validate(values, label, thread_limit=None, source=None):
    """`source`: the file `values` were read from, named with the fix when a saved value is refused at start."""
    if not isinstance(values, dict):
        raise ValueError(f'{label} must be an object')
    unknown = values.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f'{label}: unknown settings: {", ".join(sorted(unknown))}')
    for key, value in values.items():
        if key in INTS:
            lo, hi = INTS[key]
            if key == 'threads' and thread_limit is not None: hi = thread_limit
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'{label}.{key} must be an integer in {lo}..{hi}' + (' (detected performance/physical-core limit or pinned bridge maximum of 32; lower the saved thread count)' if key == 'threads' and thread_limit is not None else ''))
        elif key in MEMORY:
            if key.endswith('_mib') and value is None:
                raise ValueError(f'{label}.{key} requires a nonnegative number')
            if value is not None and (type(value) not in (int, float) or not 0 <= value < 1000000):
                raise ValueError(f'{label}.{key} must be a finite nonnegative number or null')
            if key == 'patch_limit_gib' and value is not None and not value >= PATCH_LIMIT_MIN_GIB:
                # An earlier build saved values below this minimum: such a file names the fix (not clamped silently).
                raise ValueError(f'{label}.patch_limit_gib is {value!r}; it must be at least {PATCH_LIMIT_MIN_GIB*1024:g} MiB '
                                 f'({PATCH_LIMIT_MIN_GIB:.6g} GiB, {int(PATCH_LIMIT_MIN_GIB*1024**3)//4:,} weights) or null (Auto)'
                                 + (f'. Edit {source}: set {label}.patch_limit_gib to null (Auto) or to at least '
                                    f'{PATCH_LIMIT_MIN_GIB:.6g}, then start again' if source else ''))
        elif key == 'model_roots':
            if not isinstance(value,list) or not all(isinstance(p,str) and p for p in value):
                raise ValueError(f'{label}.model_roots must be a list of paths')
        elif key in {'cpu_bridge','metal_bridge'} and value is None:
            continue
        elif not isinstance(value,str) or not value:
            raise ValueError(f'{label}.{key} must be a nonempty string')
    if values.get('backend', 'auto') not in {'auto','cpu','metal'}:
        raise ValueError('backend must be auto, cpu, or metal; CUDA/MLX inference is not implemented')


def _paths(values, directory):
    def resolve(value):
        p = Path(value).expanduser()
        return str((directory / p).resolve() if not p.is_absolute() else p)
    out = dict(values)
    for key in PATHS:
        if out.get(key): out[key] = resolve(out[key])
    if 'model_roots' in out: out['model_roots'] = [resolve(p) for p in out['model_roots']]
    # Repo IDs and bundled/workspace names remain names; explicit relative files are config-relative.
    if out.get('model','').endswith('.gguf') or out.get('model','').startswith(('./','../','~/','/')):
        out['model'] = resolve(out['model'])
    return out


def resolve(config=None, profile=None, overrides=None, *, env=None, system=None, document=None):
    env = os.environ if env is None else env
    system = system or platform.system()
    thread_limit = runtime_thread_limit(system)
    explicit = config or env.get('WSP_CONFIG')
    path = Path(explicit).expanduser() if explicit else user_dir(env=env,system=system) / 'config.json'
    profile = profile or env.get('WSP_PROFILE')
    data = {} if document is None else document
    if document is None:
        if path.exists():
            try: data = json.loads(path.read_text())
            except (OSError,ValueError) as exc: raise ValueError(f'Cannot read configuration {path}: {exc}') from exc
        elif explicit:
            raise ValueError(f'Configuration file does not exist: {path}')
    if data:
        if not isinstance(data,dict) or data.get('version') != 1 or type(data.get('version')) is not int:
            raise ValueError('Configuration requires integer version: 1')
        if data.keys() - {'version','defaults','profiles'}:
            raise ValueError('Unknown configuration keys; expected version, defaults, profiles')
        source = str(path) if document is None else None
        _validate(data.get('defaults',{}), 'defaults', source=source)
        if not isinstance(data.get('profiles',{}),dict): raise ValueError('profiles must be an object')
        for name, values in data.get('profiles',{}).items(): _validate(values, 'profile '+name, source=source)
    elif document is not None or path.exists():
        raise ValueError('Configuration requires integer version: 1')
    values, sources = dict(DEFAULTS, threads=min(DEFAULTS['threads'], thread_limit)), {key:'default' for key in DEFAULTS}
    def apply(v, source):
        values.update(v); sources.update({key:source for key in v})
    apply(_paths(data.get('defaults',{}),path.parent), 'config defaults')
    if profile:
        if profile not in data.get('profiles',{}): raise ValueError(f'Unknown profile: {profile}')
        apply(_paths(data['profiles'][profile],path.parent), 'profile '+profile)
    for key, name in ENV.items():
        if name in env:
            try: value = int(env[name]) if key in INTS else env[name]
            except ValueError as exc: raise ValueError(f'{name} requires an integer') from exc
            apply({key:value}, 'environment '+name)
    apply({k:v for k,v in (overrides or {}).items() if v is not None}, 'command line')
    notices = []
    source = sources['threads']
    # A saved (portable) thread count starts at this host's limit, visibly. Environment
    # and command-line values are explicit for this start and stay a start error.
    if (source == 'config defaults' or source.startswith('profile ')) and type(values['threads']) is int and values['threads'] > thread_limit:
        requested = values['threads']
        values['threads'] = thread_limit; sources['threads'] = f'{source} (clamped to host limit {thread_limit})'
        # Same record shape as runtime_events.event (not imported: it pulls in psutil).
        notices.append(dict(code='threads_clamped', action='warning',
            message=f'Saved thread count {requested} ({source}) exceeds this host\'s limit of {thread_limit} '
                    f'(detected performance/physical cores or the pinned bridge maximum of {NATIVE_MAX_THREADS}); using {thread_limit} threads.',
            measured=requested, limit=thread_limit,
            recommendations=[f'Lower the saved thread count to {thread_limit} or less in Settings to remove this notice.'], time=time.time()))
    _validate(values, 'effective configuration', thread_limit)
    return dict(values=values, sources=sources, config=str(path.resolve()), profile=profile, platform=system, thread_limit=thread_limit, notices=notices)


def bridge_status(path, kind):
    if not path: return dict(ok=False, reason=f'{kind} bridge is not configured')
    try:
        p = Path(path)
        if not p.is_file(): raise ValueError(f'library does not exist: {p}')
        build = json.loads(Path(str(p)+'.json').read_text())
        if build.get('abi') != 1 or build.get('ggml') != '0.29.0':
            raise ValueError('requires ggml 0.29.0 / ABI 1 manifest')
        dependency = build.get(kind+'_lib')
        if not isinstance(dependency,str) or not Path(dependency).is_file():
            raise ValueError(f'manifest {kind}_lib does not exist')
        return dict(ok=True, reason='ggml 0.29.0 / ABI 1 manifest and libraries present',
                    path=str(p), dependency=dependency)
    except (OSError,ValueError,TypeError) as exc:
        return dict(ok=False, reason=f'{kind} bridge: {exc}')


def backend_status(settings):
    v, system = settings['values'], settings['platform']
    cpu, metal = bridge_status(v['cpu_bridge'],'cpu'), bridge_status(v['metal_bridge'],'metal')
    selected = v['backend']
    reason = 'explicit backend'
    if selected == 'auto':
        selected = 'metal' if system == 'Darwin' and cpu['ok'] and metal['ok'] else 'cpu'
        reason = 'auto: configured compatible Metal and CPU bridges' if selected == 'metal' else 'auto: CPU path (Metal not configured/available)'
    blockers = []
    if system not in {'Darwin','Linux'}: blockers.append('Packed inference/watchdog supported only on macOS and Linux')
    if selected == 'metal' and system != 'Darwin': blockers.append('Metal requires macOS')
    if not cpu['ok']: blockers.append(cpu['reason'])
    if v['metal_bridge'] and not metal['ok']: blockers.append(metal['reason'])
    if selected == 'metal' and not metal['ok']: blockers.append(metal['reason'])
    return dict(selected=selected, reason=reason, ready=not blockers, blockers=list(dict.fromkeys(blockers)), cpu=cpu, metal=metal)


def doctor(settings, inspect_model=False):
    versions = {}
    for package in ('torch','transformers','gguf','numpy','psutil','uvicorn'):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    backend = backend_status(settings)
    blockers = list(backend['blockers'])
    for package, expected in [('gguf','0.19.0'),('transformers','5.22.2')]:
        if versions[package] != expected: blockers.append(f'{package} requires {expected}; install requirements-quantized.txt with this interpreter')
    if not versions['psutil']: blockers.append('Memory watchdog needs psutil; install requirements-quantized.txt')
    model = Path(settings['values']['model']).expanduser()
    model_info = dict(selection=str(model),exists=model.exists(),bytes=model.stat().st_size if model.is_file() else None,
                      inspected='path/stat only; no GGUF headers or payload imported')
    if inspect_model:
        try:
            if model.suffix.lower() != '.gguf' or not model.is_file(): raise ValueError('--inspect-model needs an existing GGUF file')
            from .memory_forecast import gguf_preview
            model_info.update(gguf_preview(model,settings['values']['context']))
            model_info['inspected'] = 'GGUF header/directory only; no tensor payload scan'
        except (ImportError,ValueError,OSError,KeyError) as exc:
            model_info['error'] = str(exc)
            blockers.append('Model preview: '+str(exc))
    return dict(**settings, model=model_info, interpreter=sys.executable, packages=versions, backend=backend,
                model_roots=[dict(path=p,exists=Path(p).is_dir()) for p in settings['values']['model_roots']],
                watchdog=dict(supported=settings['platform'] in {'Darwin','Linux'}, telemetry='checked at each explicit load, then every second'),
                ready=not blockers, blockers=blockers,
                remediation=[f'{sys.executable} -m pip install -r requirements-quantized.txt',
                             'Build pinned native bridges: docs/QUANTIZED.md; set cpu_bridge / metal_bridge in user config'],
                note='No weights loaded, no native ABI execution probe; manifests are checked, loader verifies ABI on explicit Run. Profiles do not authorize agent model loads.')


def check_backend(settings):
    """The backend check a server start makes (activate); the supervisor makes it again for a saved backend."""
    v = settings['values']; backend = backend_status(settings)
    # Explicit broken paths/devices fail early; absent optional configuration leaves other model paths usable.
    configured = v['cpu_bridge'] or v['metal_bridge'] or v['backend'] != 'auto'
    for kind in ('cpu','metal'):
        if v[kind+'_bridge'] and not backend[kind]['ok']:
            raise ValueError(backend[kind]['reason'])
    if configured and not backend['ready']: raise ValueError('; '.join(backend['blockers']))
    return backend


def activate(settings):
    """Apply resolved overrides only in the new server process; never modify shell/user environment."""
    v = settings['values']; backend = check_backend(settings)
    for root in v['model_roots']:
        if not Path(root).is_dir(): raise ValueError(f'Model root does not exist: {root}')
    for key, variable in ENV.items():
        if key in {'root','model'}: continue
        value = backend['selected'] if key == 'backend' else v[key]
        if value is not None: os.environ[variable] = str(value)
    return backend
