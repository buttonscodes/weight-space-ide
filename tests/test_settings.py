"""Settings persistence, metadata forecasts and API security; no models or inference."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from wsp.runtime_config import resolve
from wsp.settings import SettingsStore, SettingsConflict, preview, validate_interactive, GiB
from wsp.memory_forecast import estimate
from wsp.memory_guard import forecast
from wsp.session import Session
from wsp.server import create_app
URL='http://127.0.0.1:41881'

def active(tmp_path,profile=None,overrides=None):
    data={'version':1,'defaults':{'backend':'cpu','root':'./relative-workspaces'},
          'profiles':{'personal':{'context':64,'new_tokens':4,'model_roots':['./models']},
                      'untouched':{'model':'./other.gguf','threads':2}}}
    path=tmp_path/'config.json';path.write_text(json.dumps(data))
    return resolve(str(path),profile,overrides,env={},system='Linux')
def machine():return {'total':64*GiB,'available':40*GiB,'pressure':1}
def fake_ws():
    return SimpleNamespace(kind='gguf',runner_loaded=False,patches=[],
        struct=SimpleNamespace(d_model=64,n_layers=2,n_heads=4,d_vocab=128,n_ctx=4096),
        hf_config={'hidden_size':64,'num_hidden_layers':2,'num_attention_heads':4,'vocab_size':128},
        store=SimpleNamespace(nbytes=lambda:GiB),read=lambda *a:pytest.fail('settings read weights'))

def test_profile_save_preserves_paths_other_profiles_and_inheritance(tmp_path):
    settings=active(tmp_path,'personal');store=SettingsStore(settings)
    original=json.loads(store.path.read_text());before=copy.deepcopy(settings)
    after=store.save(store.snapshot()['revision'],{'new_tokens':32,'context':128})
    document=json.loads(store.path.read_text())
    assert document['defaults']==original['defaults']
    assert document['profiles']['untouched']==original['profiles']['untouched']
    assert document['profiles']['personal']==dict(original['profiles']['personal'],new_tokens=32,context=128)
    assert 'budget_gib' not in document['profiles']['personal']
    # The default output count applies when saved (the server reads it per request); without a supervisor
    # the context waits for the next start. The resolved settings handed to the store are never mutated.
    expected=copy.deepcopy(before);expected['values']['new_tokens']=32;expected['sources']['new_tokens']='profile personal'
    assert settings==before and store.active==expected
    assert after['applied']['now']==['new_tokens'] and after['applied']['next_start']==['context']
    assert after['restart_required'] and after['active']['values']['context']==64 and after['pending']=={'context':'next_start'}
    assert after['next_start']['values']['new_tokens']==32 and after['next_start']['values']['context']==128

def test_original_overrides_win_without_reading_activated_environment(tmp_path,monkeypatch):
    settings=active(tmp_path,'personal',{'new_tokens':16});monkeypatch.setenv('WSP_MAX_NEW_TOKENS','400')
    store=SettingsStore(settings);after=store.save(store.snapshot()['revision'],{'new_tokens':32})
    assert after['saved']['values']['new_tokens']==32 and after['next_start']['values']['new_tokens']==16
    assert after['next_start']['sources']['new_tokens']=='command line'
    assert after['overridden']==['new_tokens'] and not after['restart_required']

def test_atomic_revision_conflict_and_concurrent_saves(tmp_path):
    store=SettingsStore(active(tmp_path));rev=store.snapshot()['revision']
    def save(value):
        try:return store.save(rev,{'new_tokens':value})
        except SettingsConflict:return 'conflict'
    with ThreadPoolExecutor(2) as pool:results=list(pool.map(save,[80,96]))
    assert results.count('conflict')==1
    assert json.loads(store.path.read_text())['defaults']['new_tokens'] in (80,96)
    assert not list(tmp_path.glob('.*.tmp'))
    store.path.write_text(store.path.read_text()+' ')
    with pytest.raises(SettingsConflict):store.save(rev,{'threads':3})

def test_missing_file_create_and_symlink_special_file_refused(tmp_path):
    store=SettingsStore(active(tmp_path));store.path.unlink()
    store.save(store.snapshot()['revision'],{'threads':3})
    assert json.loads(store.path.read_text())['defaults']=={'threads':3}
    store.path.unlink();target=tmp_path/'target.json';target.write_text('{"version":1}');store.path.symlink_to(target)
    with pytest.raises(ValueError,match='regular'):store.read()
    store.path.unlink();store.path.mkdir()
    with pytest.raises(ValueError,match='regular'):store.read()

@pytest.mark.parametrize('changes',[{'context':True},{'context':0},{'context':2**53},{'new_tokens':2**53},
    {'budget_gib':float('nan')},{'floor_gib':float('inf')},{'root':'/other'},{'swap_growth_mib':None}])
def test_invalid_changes_never_write(tmp_path,changes):
    store=SettingsStore(active(tmp_path));before=store.path.read_bytes()
    with pytest.raises(ValueError):store.save(store.snapshot()['revision'],changes)
    assert store.path.read_bytes()==before

def test_policy_and_context_validation_even_without_available_telemetry(tmp_path):
    settings=active(tmp_path)
    for changes in ({'floor_gib':30,'preferred_gib':20},{'budget_gib':30,'ceiling_gib':20},
                    {'floor_gib':0},{'context':2**53},{'new_tokens':False}):
        candidate=copy.deepcopy(settings);candidate['values'].update(changes)
        with pytest.raises(ValueError):validate_interactive(candidate)
    candidate=copy.deepcopy(settings);candidate['values'].update(preferred_gib=40,budget_gib=40,ceiling_gib=50)
    with pytest.raises(ValueError):validate_interactive(candidate,64*GiB)

def test_forecast_quadratic_attention_and_preview_same_guard_context(tmp_path):
    small=estimate(GiB,128,64,2,128,heads=4);big=estimate(GiB,256,64,2,128,heads=4)
    assert big['eager_attention']==4*small['eager_attention']
    assert big['total']==sum(v for k,v in big.items() if k!='total')
    settings=active(tmp_path);settings['values']['context']=1024;w=fake_ws();w.struct.n_ctx=128
    result=preview(settings,w,telemetry=machine)
    assert result['effective_context']==128 and result['forecast']==forecast(w,1024,operation='interactive')
    assert result['incremental']==result['required']

def test_loaded_preview_no_double_count_and_separate_cold_headroom(tmp_path):
    settings=active(tmp_path);w=fake_ws();w.runner_loaded=True
    current=forecast(w,settings['values']['context'],operation='interactive')
    runtime=SimpleNamespace(settings=settings,status=lambda _:dict(reserved=True,forecast=current))
    same=preview(settings,w,runtime,telemetry=machine)
    assert same['resident'] and same['incremental']==0
    assert same['remaining']==40*GiB and same['cold_remaining']==40*GiB-same['required']
    draft=copy.deepcopy(settings);draft['values']['context']=16384
    changed=preview(draft,w,runtime,telemetry=machine)
    assert changed['incremental']==changed['required']-same['required']
    w.runner_loaded=False
    assert preview(draft,w,runtime,telemetry=machine)['incremental']==changed['required']

def test_unknown_telemetry_and_model_explicit(tmp_path):
    def unavailable():raise RuntimeError('not available')
    result=preview(active(tmp_path),telemetry=unavailable)
    assert result['incremental'] is None and result['remaining'] is None and result['machine']['error']=='not available'

@pytest.fixture
def api(tmp_path):
    settings=active(tmp_path,'personal');sess=Session(str(tmp_path/'ws'));sess.settings=settings
    app=create_app(sess,URL);app.state.settings_telemetry=machine
    client=TestClient(app,base_url=URL,raise_server_exceptions=False);snapshot=client.get('/api/settings').json()
    return sess,app,client,snapshot,{'X-WSP-Settings':snapshot['csrf'],'Origin':URL}

def test_empty_workspace_preview_save_and_csrf(api):
    sess,app,c,s,h=api
    assert sess.ws is None and s['saved']['values']['new_tokens']==4
    before=Path(s['config']).read_bytes();body={'revision':s['revision'],'changes':{'context':512,'new_tokens':128,'resident_seconds':14400}}
    p=c.post('/api/settings/preview',json=body,headers=h)
    assert p.status_code==200,p.text
    assert Path(s['config']).read_bytes()==before and c.post('/api/settings',json=body).status_code==403
    saved=c.post('/api/settings',json=body,headers=h)
    assert saved.status_code==200,saved.text
    assert saved.json()['restart_required'] and sess.settings['values']['context']==64
    assert c.post('/api/settings',json=body,headers=h).status_code==409
    assert c.get('/api/settings').headers['cache-control']=='no-store'

@pytest.mark.parametrize('headers',[{'Origin':'http://evil.test:41881'},{'Host':'evil.test:41881'},
    {'Origin':'https://127.0.0.1:41881'},{'Sec-Fetch-Site':'cross-site'}])
def test_cross_origin_rejected_unmanaged(api,headers):
    _,_,c,s,h=api
    assert c.post('/api/settings',json={'revision':s['revision'],'changes':{'threads':3}},headers={**h,**headers}).status_code==403

def test_bounded_payload_and_remote_peer_rejected(api):
    _,app,c,s,h=api
    assert c.post('/api/settings',content='x'*17000,headers=h).status_code==413
    remote=TestClient(app,base_url=URL,client=('203.0.113.8',1234))
    assert remote.get('/api/settings').status_code==403

def test_effective_launch_context_can_reduce_output_ceiling_per_request(api):
    _,app,c,s,h=api;app.state.settings_store.overrides['context']=64;app.state.settings_store.active['sources']['context']='command line'
    body={'revision':s['revision'],'changes':{'backend':'cpu','context':2048,'new_tokens':128}}
    p=c.post('/api/settings/preview',json=body,headers=h).json()
    assert p['preview']['effective_context']==64 and not p['preview']['errors']
    assert p['preview']['info']
    assert c.post('/api/settings',json=body,headers=h).status_code==200

def test_invalid_run_context_rejected_before_loading(api):
    sess,_,c,_,_=api;w=fake_ws();w.can_run=True;w.encode=lambda _:list(range(64));sess.ws=w
    result=c.post('/api/run',json={'prompt':'already full','max_new_tokens':4})
    assert result.status_code==400 and 'active context' in result.text


def test_unknown_model_minimum_reservation_warns_without_inventing_forecast(tmp_path):
    settings=active(tmp_path);settings['values'].update(reservation_gib=50,budget_gib=24,ceiling_gib=26,floor_gib=20,preferred_gib=25)
    result=preview(settings,telemetry=machine)
    assert result['incremental'] is None and result['required'] is None
    assert result['minimum_reservation']==50*GiB
    assert any('Minimum reservation alone exceeds' in w for w in result['warnings'])
    assert any('Minimum reservation alone leaves' in w for w in result['warnings'])


def test_operation_policy_preview_distinguishes_active_and_idle_caps(tmp_path):
    settings=active(tmp_path);w=fake_ws();w.runner_loaded=True
    current=forecast(w,settings['values']['context'],operation='interactive')
    reservation={'policy':{'budget':12*GiB,'ceiling':40*GiB,'floor':6*GiB,'preferred':10*GiB},
                 'operation_id':'op-1','admission':{'static_cap':40*GiB}}
    runtime=SimpleNamespace(settings=settings,status=lambda _:dict(reserved=True,forecast=current,reservation=reservation))
    result=preview(settings,w,runtime,telemetry=machine)
    assert result['active_operation'] and result['active_static_cap']==40*GiB
    assert result['active_policy']['budget']==12*GiB
    reservation['operation_id']=None
    assert not preview(settings,w,runtime,telemetry=machine)['active_operation']


def test_busy_runtime_status_without_forecast_gives_no_resident_credit(tmp_path):
    # status() answers busy (another thread holds the client lock) before any forecast is cached.
    import threading
    from wsp.runtime_supervisor import RuntimeClient
    settings=active(tmp_path);w=fake_ws();w.runner_loaded=True
    client=RuntimeClient(tmp_path,settings);client.reserved=True;client.binding='b'
    held,done=threading.Event(),threading.Event()
    def hold():
        with client.lock:held.set();done.wait(5)
    thread=threading.Thread(target=hold,daemon=True);thread.start();held.wait(5)
    try:
        status=client.status(w)
        assert status['busy'] and status['reserved'] and status['forecast'] is None
        result=preview(settings,w,client,telemetry=machine)
    finally:done.set();thread.join(5)
    assert result['resident'] and result['active_reservation'] is None
    assert result['incremental'] is None and result['remaining'] is None and result['required'] is not None
    assert any('forecast unknown while the runtime is busy' in m for m in result['warnings'])


def test_settings_page_builds_while_the_runtime_status_is_busy(api):
    sess,app,c,_,_=api;w=fake_ws();w.runner_loaded=True;sess.ws=w
    sess.runtime=SimpleNamespace(settings=sess.settings,status=lambda _:dict(reserved=True,busy=True,forecast=None,
        reservation={'state':'running','policy':None,'operation_id':'op'}))
    result=c.get('/api/settings')
    assert result.status_code==200,result.text
    assert result.json()['saved_preview']['active_reservation'] is None


def test_capacity_preset_uses_512_mib_minimum_without_changing_hard_cap(tmp_path):
    from wsp.settings import capacity_recommendation, RECOMMENDATION_BUFFER_MIN
    settings=active(tmp_path);settings['values'].update(floor_gib=1,preferred_gib=1)
    w=fake_ws();w.struct.n_ctx=131072;w.store.nbytes=lambda:GiB//2
    result=capacity_recommendation(settings,w,telemetry=lambda:dict(total=64*GiB,available=6*GiB,pressure=1))
    assert RECOMMENDATION_BUFFER_MIN==512*1024**2
    assert result['cap']==5*GiB and result['margin']==GiB//2
    assert result['target']==result['cap']-result['margin']
    assert result['changes'] is not None
    assert forecast(w,result['changes']['context'],operation='interactive')['total']<=result['target']
    assert 'preset-only slack' in result['note'] and 'not a new hard runtime limit' in result['note']


def test_capacity_buffer_cannot_be_silently_reduced(tmp_path):
    from wsp.settings import capacity_recommendation
    result=capacity_recommendation(active(tmp_path),fake_ws(),telemetry=machine,overrides={'budget_gib':.25})
    assert result['changes'] is None and result['status']=='blocked'
    assert result['margin']==GiB//2 and result['target']<0
