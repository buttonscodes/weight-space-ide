"""Exact resident Settings scenarios and configuration-only API; no model or tensor reads."""
import copy
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from test_settings import active, fake_ws, GiB
from wsp.settings import preview, interactive_defaults, SettingsStore
from wsp.session import Session
from wsp.server import create_app


def resident(tmp_path, monkeypatch, available=47.12, auto=False):
    s=active(tmp_path,'personal')
    s['values'].update(context=64,new_tokens=4,resident_seconds=900,floor_gib=20,preferred_gib=25,
        budget_gib=None if auto else 24,ceiling_gib=None if auto else 26,reservation_gib=23)
    w=fake_ws();w.runner_loaded=True;w.struct.n_ctx=32768
    monkeypatch.setattr('wsp.memory_guard.forecast',lambda *a,**k:dict(total=23*GiB))
    frozen=dict(budget=46*GiB if auto else 24*GiB,ceiling=103*GiB if auto else 26*GiB,
                preferred=25*GiB,floor=20*GiB)
    r=SimpleNamespace(settings=s,status=lambda _:dict(reserved=True,forecast={'total':23*GiB},reservation={'policy':frozen}))
    machine=lambda:dict(total=128*GiB,available=int(available*GiB),free=20*GiB,pressure=1)
    return s,w,r,machine


@pytest.mark.parametrize('available',[47.12,52])
@pytest.mark.parametrize('auto',[False,True])
def test_resident_available_is_not_charged_for_second_model(tmp_path,monkeypatch,available,auto):
    s,w,r,m=resident(tmp_path,monkeypatch,available,auto)
    out=preview(s,w,r,m)
    assert out['required']==out['active_reservation']==23*GiB
    assert out['incremental']==0 and out['remaining']==m()['available']
    assert out['cold_remaining']==m()['available']-23*GiB  # explicitly hypothetical
    assert out['post_unload_remaining'] is None and out['policy_hypothetical']==auto
    assert not out['warnings'] and out['info']
    assert out['active_policy']['budget']==(46 if auto else 24)*GiB
    preset=interactive_defaults(s,w,m,runtime=r)
    assert preset['context']==32768 and preset['new_tokens']==2048
    assert s['values']['context']==64 and s['values']['reservation_gib']==23


def test_known_caps_floor_and_nonresident_failures_remain_visible(tmp_path,monkeypatch):
    s,w,r,m=resident(tmp_path,monkeypatch)
    draft=copy.deepcopy(s);draft['values']['budget_gib']=22
    out=preview(draft,w,r,m)
    assert any('explicit configured budget' in x for x in out['warnings'])
    _,_,_,low=resident(tmp_path,monkeypatch,19)
    assert any('active reservation runtime floor' in x for x in preview(s,w,r,low)['warnings'])
    w.runner_loaded=False
    out=preview(s,w,r,m)
    assert out['incremental']==23*GiB and any('Cold-load' in x for x in out['warnings'])
    assert interactive_defaults(s,w,m,runtime=r) is None  # Weight floor cannot fit even at two tokens.


def test_unknown_resident_reservation_or_telemetry_never_invents_credit(tmp_path,monkeypatch):
    s,w,r,m=resident(tmp_path,monkeypatch)
    def unknown():raise ValueError('unknown telemetry')
    out=preview(s,w,r,unknown)
    assert out['remaining'] is None and not any('below' in x for x in out['warnings'])
    r.status=lambda _:dict(reserved=False)
    out=preview(s,w,r,m)
    assert out['resident'] and out['active_reservation'] is None and out['incremental'] is None
    assert out['post_unload_remaining'] is None


def test_runtime_summary_reads_saved_limits_and_retained_overrides_without_preview(tmp_path,monkeypatch):
    s=active(tmp_path,'personal',{'new_tokens':4})
    store=SettingsStore(s);store.save(store.snapshot()['revision'],{'context':8192,'new_tokens':2048})
    session=Session(str(tmp_path/'work'));session.settings=s
    monkeypatch.setattr('wsp.settings.preview',lambda *a,**k:pytest.fail('runtime summary computed memory preview'))
    client=TestClient(create_app(session,'http://127.0.0.1:41881'),base_url='http://127.0.0.1:41881')
    data=client.get('/api/runtime').json();summary=data['settings_summary']
    assert data['generation']['output_limit']==4
    assert summary['active']==dict(context=64,new_tokens=4)
    assert summary['saved']==dict(context=8192,new_tokens=2048)
    assert summary['next_start']==dict(context=8192,new_tokens=4)
    assert summary['restart_required'] and summary['overridden']==['new_tokens']


@pytest.mark.parametrize('kind,managed,expected_limit,expected_context', [
    ('hf', False, 4, None), ('hf', True, 4, None),
    ('gguf', False, 4, 64), ('gguf', True, None, 64),
])
def test_output_ceiling_removed_only_for_managed_gguf(tmp_path, kind, managed, expected_limit, expected_context):
    settings = active(tmp_path, 'personal')
    session = Session(str(tmp_path/'work')); session.settings = settings
    session.ws = SimpleNamespace(kind=kind, struct=SimpleNamespace(n_ctx=96),
                                 encode=lambda text: list(range(len(text.split()))))
    if managed:
        session.runtime = SimpleNamespace(settings=settings, status=lambda _: {'supervised': True})
    url = 'http://127.0.0.1:41881'
    client = TestClient(create_app(session, url), base_url=url, raise_server_exceptions=False)
    limits = client.get('/api/runtime').json()['generation']
    assert limits['output_limit'] == expected_limit
    assert limits['context_limit'] == expected_context
    assert limits['default_output'] == 4
    if kind == 'hf':
        # Oversized requests fail before touching a runner, even for histories
        # longer than a sliding-window model's n_ctx metadata.
        for length in (96, 200):
            prompt = ' '.join(['x'] * length)
            for route, body in (('/api/run', {'prompt': prompt}),
                                ('/api/chat', {'messages': [{'role': 'user', 'content': prompt}]})):
                response = client.post(route, json={**body, 'max_new_tokens': 5})
                assert response.status_code == 400
                assert '4' in response.json()['error']
