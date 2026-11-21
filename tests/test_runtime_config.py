"""Configuration and simulated-platform tests; never import or load model weights."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from wsp.runtime_config import resolve, doctor, activate, user_dir, backend_status
from wsp.runtime_supervisor import policy_for


@pytest.fixture(autouse=True)
def isolated_core_discovery_cache():
    from wsp.runtime_config import _DISCOVERED_CORES
    _DISCOVERED_CORES.clear()
    yield
    _DISCOVERED_CORES.clear()

def config(tmp_path, **data):
    p=tmp_path/'config.json';p.write_text(json.dumps(dict(version=1,**data)));return str(p)

def test_default_and_environment(tmp_path):
    env={'HOME':str(tmp_path),'WSP_GGML_DEVICE':'cpu','WSP_MAX_CTX':'128','API_TOKEN':'secret'}
    s=resolve(env=env,system='Linux')
    assert s['values']['model']=='facts' and s['values']['context']==128
    assert s['sources']['context']=='environment WSP_MAX_CTX'
    assert 'secret' not in json.dumps(doctor(s))
    assert str(tmp_path/'.config/weight-space-ide/config.json')==s['config']

def test_precedence_and_relative_paths(tmp_path):
    p=config(tmp_path, defaults={'context':256,'root':'./work'}, profiles={'small':{'context':128,'model':'./small.gguf','model_roots':['models']}})
    s=resolve(p,'small',{'context':64},env={'WSP_MAX_CTX':'512'})
    assert s['values']['context']==64 and s['sources']['context']=='command line'
    assert s['values']['root']==str(tmp_path/'work')
    assert s['values']['model']==str(tmp_path/'small.gguf')
    assert s['values']['model_roots']==[str(tmp_path/'models')]
    assert resolve(p,'small',env={})['values']['context']==128
    assert resolve(p,'small',env={'WSP_MAX_CTX':'512'})['values']['context']==512

@pytest.mark.parametrize('data',[{'version':2},{'version':True},{'version':1,'execute':'bad'},
    {'version':1,'defaults':{'bad':1}}, {'version':1,'defaults':{'context':0}},
    {'version':1,'defaults':{'context':True}}, {'version':1,'defaults':{'backend':'cuda'}},
    {'version':1,'defaults':{'budget_gib':float('nan')}}, {'version':1,'profiles':[]}])
def test_malformed(tmp_path,data):
    p=tmp_path/'bad.json';p.write_text(json.dumps(data))
    with pytest.raises(ValueError):resolve(str(p),env={})

def test_explicit_missing_and_unknown_profile(tmp_path):
    with pytest.raises(ValueError,match='does not exist'):resolve(str(tmp_path/'missing'),env={})
    with pytest.raises(ValueError,match='Unknown profile'):resolve(config(tmp_path),'absent',env={})

def bridges(tmp_path):
    paths={}
    for kind in ['cpu','metal']:
        p=tmp_path/(kind+'.dylib');p.touch()
        Path(str(p)+'.json').write_text(json.dumps({'abi':1,'ggml':'0.29.0',kind+'_lib':str(p)}))
        paths[kind+'_bridge']=str(p)
    return paths

@pytest.mark.parametrize('system,expected',[('Darwin','metal'),('Linux','cpu'),('Windows','cpu')])
def test_simulated_backend_selection(tmp_path,system,expected):
    s=resolve(config(tmp_path,defaults=bridges(tmp_path)),env={},system=system)
    b=backend_status(s)
    assert b['selected']==expected and b['ready']==(system!='Windows')

def test_explicit_metal_never_falls_back(tmp_path):
    s=resolve(config(tmp_path,defaults=dict(bridges(tmp_path),backend='metal')),env={},system='Linux')
    assert 'Metal requires macOS' in backend_status(s)['blockers']
    with pytest.raises(ValueError,match='Metal requires'):activate(s)
    s=resolve(config(tmp_path,defaults={'cpu_bridge':'absent'}),env={})
    with pytest.raises(ValueError,match='does not exist'):activate(s)

def test_metadata_checks_do_not_execute_library(tmp_path):
    p=config(tmp_path,defaults=bridges(tmp_path))
    assert backend_status(resolve(p,env={},system='Darwin'))['ready']

def test_xdg_and_platform_paths(tmp_path):
    env={'HOME':str(tmp_path),'XDG_CONFIG_HOME':str(tmp_path/'cfg')}
    assert user_dir(env=env,system='Darwin')==tmp_path/'cfg/weight-space-ide'
    assert 'Application Support' in str(user_dir(env={'HOME':str(tmp_path)},system='Darwin'))

def test_portable_memory_policy_and_small_machine():
    from wsp.memory_guard import GiB, GuardError
    s=resolve(env={'HOME':'/nonexistent'})['values']
    p=policy_for(s,8*GiB,6*GiB)
    assert p.floor<GiB and p.budget<5*GiB
    p.preflight({'available':6*GiB,'pressure':1},3*GiB)
    with pytest.raises(GuardError):p.preflight({'available':4*GiB,'pressure':1},3*GiB)
    with pytest.raises(ValueError):policy_for(dict(s,floor_gib=20,preferred_gib=25),8*GiB)

def test_cli_help_doctor_without_optional_imports(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('WSP_')}
    env.update(HOME=str(tmp_path),PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    for args in [('serve','--help'),('doctor',)]:
        r=subprocess.run([sys.executable,'-S','-m','wsp',*args],env=env,capture_output=True,text=True,timeout=10)
        assert r.returncode==0,r.stderr
        assert not list(tmp_path.rglob('workspace.json'))
        if args[0]=='doctor':
            d=json.loads(r.stdout);assert not d['ready'] and d['packages']['gguf'] is None
            assert 'requirements-quantized' in str(d['blockers'])

@pytest.mark.parametrize('performance,physical,expected', [(12,16,12),(None,16,16),(None,8,8),(None,2,2)])
def test_darwin_thread_limit_prefers_performance_cores(monkeypatch, performance, physical, expected):
    import wsp.runtime_config as rc
    calls=[]
    def sysctl(command, **kwargs):
        calls.append(command[-1])
        value={'hw.perflevel0.physicalcpu':performance,'hw.physicalcpu':physical}[command[-1]]
        if value is None: raise subprocess.CalledProcessError(1,command)
        return str(value)+'\n'
    monkeypatch.setattr(rc.subprocess,'check_output',sysctl)
    assert rc.performance_core_count('Darwin')==expected
    assert calls[0]=='hw.perflevel0.physicalcpu'
    if performance: assert len(calls)==1


def test_thread_limit_physical_fallback_and_unknown(monkeypatch):
    import psutil
    import wsp.runtime_config as rc
    seen=[]
    def count(*,logical):
        seen.append(logical)
        return 6
    monkeypatch.setattr(psutil,'cpu_count',count)
    assert rc.performance_core_count('Linux')==6 and seen==[False]
    monkeypatch.setattr(psutil,'cpu_count',lambda **_:None)
    assert rc.performance_core_count('Linux')==6  # Stable within this process.
    rc._DISCOVERED_CORES.clear()
    assert rc.performance_core_count('Linux')==1
    # The one-core fallback is never cached: discovery recovers without a restart.
    monkeypatch.setattr(psutil,'cpu_count',lambda **_:6)
    assert rc.performance_core_count('Linux')==6


def test_transient_performance_core_probe_failure_does_not_pin_the_fallback(monkeypatch):
    import wsp.runtime_config as rc
    answers={'hw.perflevel0.physicalcpu':subprocess.TimeoutExpired('sysctl',2),'hw.physicalcpu':'16\n'}
    def output(command,**_):
        value=answers[command[-1]]
        if isinstance(value,Exception):raise value
        return value
    monkeypatch.setattr(rc.subprocess,'check_output',output)
    # A timed-out performance-core probe yields the all-core fallback for this
    # call only; the next resolution asks again and gets the real answer.
    assert rc.performance_core_count('Darwin')==16 and 'Darwin' not in rc._DISCOVERED_CORES
    answers['hw.perflevel0.physicalcpu']='12\n'
    assert rc.performance_core_count('Darwin')==12 and rc._DISCOVERED_CORES['Darwin']==12
    # A probe killed by a signal, or one that prints zero, is not an answer either.
    for transient in (subprocess.CalledProcessError(-9,'sysctl'),'0\n'):
        rc._DISCOVERED_CORES.clear()
        answers['hw.perflevel0.physicalcpu']=transient
        assert rc.performance_core_count('Darwin')==16 and 'Darwin' not in rc._DISCOVERED_CORES
    # A host without the key (Intel Mac: sysctl exits 1) is definitive and is cached.
    rc._DISCOVERED_CORES.clear()
    answers['hw.perflevel0.physicalcpu']=subprocess.CalledProcessError(1,'sysctl')
    assert rc.performance_core_count('Darwin')==16 and rc._DISCOVERED_CORES['Darwin']==16


@pytest.mark.parametrize('cores,default',[(2,2),(12,4)])
def test_thread_default_fits_small_hosts_and_explicit_limit_is_actionable(tmp_path,monkeypatch,cores,default):
    import wsp.runtime_config as rc
    monkeypatch.setattr(rc,'performance_core_count',lambda system=None:cores)
    s=resolve(env={'HOME':str(tmp_path)},system='Linux')
    assert s['values']['threads']==default and s['thread_limit']==cores
    assert resolve(env={'HOME':str(tmp_path),'WSP_GGML_THREADS':str(cores)},system='Linux')['values']['threads']==cores
    with pytest.raises(ValueError,match='detected performance/physical-core limit'):
        resolve(env={'HOME':str(tmp_path),'WSP_GGML_THREADS':str(cores+1)},system='Linux')


def test_interactive_warning_defaults_do_not_change_acceptance_policy(tmp_path):
    from wsp.memory_guard import Policy
    s=resolve(env={'HOME':str(tmp_path)})
    assert s['values']['swap_growth_mib']==64
    assert s['values']['compression_growth_mib']==1024
    assert s['values']['resident_seconds']==14400
    # The outer model acceptance harness remains a separate strict policy.
    p=Policy()
    assert p.swap_growth==64*1024**2 and p.compression_growth==1024**3
    baseline={'swap_used':0,'swap_out':0,'compressed':0}
    sample=dict(baseline,available=100*1024**3,pressure=1,swap_used=65*1024**2)
    assert 'swap' in p.violation(sample,baseline).lower()


def test_inactive_portable_profile_does_not_enforce_this_host_cap(tmp_path,monkeypatch):
    import wsp.runtime_config as rc
    monkeypatch.setattr(rc,'performance_core_count',lambda system=None:8)
    p=config(tmp_path,defaults={'threads':12},profiles={'portable12':{'threads':12},'local4':{'threads':4}})
    s=resolve(p,'local4',env={},system='Darwin')
    assert s['values']['threads']==4 and s['thread_limit']==8 and s['notices']==[]
    # An active portable profile above this host's limit starts clamped, visibly.
    assert resolve(p,'portable12',env={},system='Darwin')['values']['threads']==8
    assert resolve(p,'portable12',{'threads':6},env={},system='Darwin')['values']['threads']==6
    assert resolve(env={'HOME':str(tmp_path)},system='Darwin')['values']['threads']==4


@pytest.mark.parametrize('saved,profile,source', [({'defaults':{'threads':12}},None,'config defaults'),
    ({'defaults':{'threads':2},'profiles':{'portable':{'threads':16}}},'portable','profile portable')])
def test_saved_threads_above_host_limit_start_clamped_with_notice(tmp_path, monkeypatch, saved, profile, source):
    import wsp.runtime_config as rc
    from wsp.settings import SettingsStore, preview
    monkeypatch.setattr(rc, 'performance_core_count', lambda system=None: 8)
    path = config(tmp_path, **saved)
    s = resolve(path, profile, env={}, system='Linux')
    requested = (saved.get('profiles') or {}).get(profile, saved['defaults'])['threads']
    assert s['values']['threads'] == 8 and s['sources']['threads'] == source+' (clamped to host limit 8)'
    [notice] = s['notices']
    assert {k: notice[k] for k in ('code','action','measured','limit')} == dict(code='threads_clamped', action='warning', measured=requested, limit=8)
    assert notice['message'] and notice['recommendations'] and notice['time']
    # The settings preview lists the same saved-value problem.
    machine = lambda: {'total': 64*1024**3, 'available': 40*1024**3, 'pressure': 1}
    assert notice['message'] in preview(s, telemetry=machine)['warnings']
    store = SettingsStore(s)
    assert store.resolve(json.loads(Path(path).read_text()))['notices'][0]['code'] == 'threads_clamped'
    # A launch thread override replaces the clamped value, so the next start has no notice.
    overridden = SettingsStore(resolve(path, profile, {'threads': 4}, env={}, system='Linux'))
    assert overridden.resolve(json.loads(Path(path).read_text()), effective=True)['notices'] == []


@pytest.mark.parametrize('env,overrides', [({'WSP_GGML_THREADS':'9'},None), ({},{'threads':9})])
def test_explicit_threads_above_host_limit_remain_a_start_error(tmp_path, monkeypatch, env, overrides):
    import wsp.runtime_config as rc
    monkeypatch.setattr(rc, 'performance_core_count', lambda system=None: 8)
    path = config(tmp_path, defaults={'threads': 12})
    with pytest.raises(ValueError, match='detected performance/physical-core limit'):
        resolve(path, overrides=overrides, env=env, system='Linux')


def test_managed_thread_limit_respects_pinned_bridge(tmp_path, monkeypatch):
    import wsp.runtime_config as rc
    monkeypatch.setattr(rc, 'performance_core_count', lambda system=None: 64)
    s = resolve(env={'HOME': str(tmp_path), 'WSP_GGML_THREADS': '32'}, system='Linux')
    assert s['thread_limit'] == 32 and s['values']['threads'] == 32
    with pytest.raises(ValueError, match='pinned bridge maximum of 32'):
        resolve(env={'HOME': str(tmp_path), 'WSP_GGML_THREADS': '33'}, system='Linux')


@pytest.mark.parametrize('managed,requested,expected', [(True, 10, 10), (True, 12, 12), (False, 12, 8)])
def test_actual_kernel_thread_setting_without_native_load(tmp_path, monkeypatch, managed, requested, expected):
    import wsp.packed as packed
    import wsp.runtime_config as rc
    from types import SimpleNamespace
    bridge = tmp_path/'fake.dylib'; bridge.touch()
    Path(str(bridge)+'.json').write_text(json.dumps(dict(ggml='0.29.0', abi=1, cpu_lib='fake')))
    class Function:
        def __init__(self, result): self.result = result
        def __call__(self, *args): return self.result
    lib = SimpleNamespace(wsp_abi=Function(1), wsp_init=Function(0),
                          wsp_supported=Function(1), wsp_matmul=Function(0))
    monkeypatch.setattr(packed.ctypes, 'CDLL', lambda path: lib)
    monkeypatch.setattr(rc, 'performance_core_count', lambda system=None: 12)
    monkeypatch.setenv('WSP_GGML_BRIDGE', str(bridge))
    monkeypatch.setenv('WSP_GGML_THREADS', str(requested))
    if managed: monkeypatch.setenv('WSP_RUNTIME_DIR', str(tmp_path))
    else: monkeypatch.delenv('WSP_RUNTIME_DIR', raising=False)
    assert packed.PackedKernel().threads == expected
    if managed:
        monkeypatch.setenv('WSP_GGML_THREADS', '13')
        with pytest.raises(ValueError, match='1..12'):
            packed.PackedKernel()


def test_runtime_status_reports_actual_kernel_threads(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from wsp.runtime_supervisor import RuntimeClient
    import wsp.runtime_config as rc
    monkeypatch.setattr(rc, 'performance_core_count', lambda system=None: 12)
    settings = resolve(env={'HOME': str(tmp_path)}, system='Linux')
    client = RuntimeClient(tmp_path, settings)
    ws = SimpleNamespace(kind='synthetic', _runner=None)
    assert client.status(ws)['actual_kernel_threads'] is None
    ws._runner = SimpleNamespace(kernel=SimpleNamespace(threads=12))
    assert client.status(ws)['actual_kernel_threads'] == 12


def test_core_discovery_cached_across_settings_resolution(tmp_path, monkeypatch):
    import wsp.runtime_config as rc
    from wsp.settings import SettingsStore
    calls = []
    monkeypatch.setattr(rc.platform, 'system', lambda: 'Darwin')
    def sysctl(command, **kwargs):
        calls.append(command[-1])
        return '8\n'
    monkeypatch.setattr(rc.subprocess, 'check_output', sysctl)
    assert rc.performance_core_count() == rc.performance_core_count('Darwin') == 8
    path = config(tmp_path, defaults={'threads': 4})
    settings = resolve(path, env={}, system='Darwin')
    store = SettingsStore(settings)
    for _ in range(3):
        assert store.snapshot()['active']['values']['threads'] == 4
        assert resolve(path, overrides={'threads': 8}, env={}, system='Darwin')['values']['threads'] == 8
    with pytest.raises(ValueError, match='detected performance/physical-core limit'):
        resolve(path, overrides={'threads': 9}, env={}, system='Darwin')
    assert calls == ['hw.perflevel0.physicalcpu']
