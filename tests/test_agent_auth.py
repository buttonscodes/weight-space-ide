"""Authentication protocol tests. Fakes only: no login, credentials or weights."""
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import agent_auth as auth


class FakeJob:
    states = {}
    created = []
    def __init__(self, provider, operation, context):
        self.provider, self.operation, self.context = provider, operation, context
        self.id = str(len(self.created)); self.created.append(self)
        self.retired = threading.Event(); self.updated = time.monotonic()
        self.data = {'provider': provider, 'status': 'checking', 'attempt_id': self.id}
    def start(self):
        self.data['status'] = self.states.get(self.provider, 'signed_out') if self.operation == 'status' else 'signing_in'
        if self.operation == 'status': self.retired.set()
    def snapshot(self): return dict(self.data)
    def cancel(self, *args): self.data['status'] = 'cancelled'; self.retired.set(); return self.snapshot()


@pytest.fixture
def manager(monkeypatch, tmp_path):
    FakeJob.states = {}; FakeJob.created = []
    manager = auth.AuthManager(SimpleNamespace(ws=SimpleNamespace(dir=str(tmp_path))), FakeJob)
    monkeypatch.setattr(manager, 'context', lambda provider: ('/fake/'+provider,str(tmp_path)))
    yield manager
    manager.shutdown()


def test_status_cached_and_start_forces_check_without_login(manager):
    assert manager.status('claude')['status'] == 'signed_out'
    assert manager.status('claude')['status'] == 'signed_out'
    assert len(FakeJob.created) == 1
    with pytest.raises(RuntimeError, match='Sign in'): manager.ensure_ready('claude')
    assert len(FakeJob.created) == 2
    FakeJob.states['claude'] = 'ready'
    manager.ensure_ready('claude')
    assert all(job.operation == 'status' for job in FakeJob.created)
    assert manager.login('claude')['status'] == 'ready'  # existing identity preserved
    assert all(job.operation == 'status' for job in FakeJob.created)


def test_context_change_retires_active_old_probe(manager, monkeypatch):
    manager.status('codex')
    previous = manager.jobs['codex']; previous.retired.clear()
    monkeypatch.setattr(manager, 'context', lambda provider: ('/fake/codex','/other/workspace'))
    FakeJob.states['codex'] = 'ready'
    assert manager.check('codex')['status'] == 'ready'
    assert previous.retired.is_set() and previous.data['status'] == 'cancelled'
    assert manager.jobs['codex'].context[1] == '/other/workspace'


def test_context_error_cannot_reuse_cached_ready(manager, monkeypatch):
    FakeJob.states['claude'] = 'ready'; manager.check('claude')
    def missing(provider): raise RuntimeError('Missing CLI')
    monkeypatch.setattr(manager, 'context', missing)
    with pytest.raises(RuntimeError, match='Sign in'): manager.ensure_ready('claude')


def test_explicit_login_is_restorable_locks_workspace_and_cleans_up(manager):
    state = manager.login('claude'); job = manager.jobs['claude']
    assert state['status'] == 'signing_in'
    assert manager.status('claude') == state
    assert manager.login('claude') == state
    with pytest.raises(RuntimeError, match='Finish or cancel'): manager.assert_no_login()
    with pytest.raises(RuntimeError, match='no longer'): manager.active_attempt('claude','wrong')
    manager.active_attempt('claude', job.id).cancel()
    manager.assert_no_login()
    with pytest.raises(RuntimeError, match='no longer'): manager.active_attempt('claude',job.id)
    manager.login('codex'); manager.shutdown()
    assert all(j.retired.is_set() for j in FakeJob.created)


def test_frames_and_invalid_codes_never_expose_secret(tmp_path):
    job = auth.AuthJob('claude', 'login', ('/fake',str(tmp_path)))
    job.accept({'status':'error','message':'secret-token'})
    assert 'secret-token' not in json.dumps(job.snapshot())
    job.accept({'status':'ready','method':'secret-token'})
    assert 'secret-token' not in json.dumps(job.snapshot())
    with pytest.raises(RuntimeError, match='Unsupported'): job.accept({'status':'signing_in','url':'https://claude.com.evil.test/oauth/authorize'})
    for code in ('secret\ncode',{},'', 'a'*4097):
        with pytest.raises(ValueError, match='Invalid sign-in code'): job.code(code)
    job.timer.cancel()


def test_auth_routes_origin_csrf_and_native_preflight(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from wsp import server
    from wsp.session import Session
    sess = Session('/not-a-model'); sess.ws=SimpleNamespace(dir=str(tmp_path)); sess.log=[]
    manager = auth.AuthManager(sess, FakeJob)
    FakeJob.states={}; FakeJob.created=[]
    monkeypatch.setattr(manager,'context',lambda provider:('/fake/'+provider,str(tmp_path)))
    monkeypatch.setattr(server,'AuthManager',lambda sess:manager)
    starts=[]
    monkeypatch.setattr(server.harness,'start',lambda *a: starts.append(a) or SimpleNamespace(meta=lambda:{'status':'running'}))
    with TestClient(server.create_app(sess,'http://127.0.0.1:8765'),base_url='http://127.0.0.1:8765',raise_server_exceptions=False) as client:
        url='/api/harness/auth?provider=claude'
        response=client.get(url)
        assert response.status_code==200 and response.headers['cache-control']=='no-store'
        token=response.json()['csrf_token']
        headers={'X-WSP-Auth':token,'Origin':'http://127.0.0.1:8765'}
        for bad in ({'Origin':'https://evil.test'},{'Host':'evil.test:8765'},{'Origin':'null'},{'Sec-Fetch-Site':'cross-site'},{'Host':'['}):
            assert client.get(url,headers=bad).status_code==403
        assert client.post('/api/harness/auth/login',json={'provider':'claude'}).status_code==403
        assert client.post('/api/harness/start',headers=headers,json={'harness':'claude'}).status_code==400
        assert not starts
        login=client.post('/api/harness/auth/login',headers=headers,json={'provider':'claude'}).json()
        assert login['status']=='signing_in'
        assert client.post('/api/open',json={'name':'never-load'}).status_code==400
        assert client.post('/api/task',json={'scope':[]}).status_code==400
        assert client.get(url).json()['attempt_id']==login['attempt_id']
        assert client.post('/api/harness/auth/cancel',headers=headers,json={'provider':'claude','attempt_id':login['attempt_id']}).json()['status']=='cancelled'
        FakeJob.states['claude']='ready'
        assert client.post('/api/harness/start',headers=headers,json={'harness':'claude'}).status_code==200
        assert client.post('/api/harness/start',headers=headers,json={'harness':'custom'}).status_code==200
        assert len(starts)==2


HELPER = r'''
import json, sys, time
for line in sys.stdin:
    m=json.loads(line)
    if m['method']=='start':
        if m['params']['executable']=='hang':
            continue
        print(json.dumps({'status':'signing_in','url':'https://claude.com/cai/oauth/authorize?code=true&redirect_uri=https%3A%2F%2Fplatform.claude.com%2Foauth%2Fcode%2Fcallback','needs_code':True}),flush=True)
    elif m['method']=='code':
        print(json.dumps({'id':m['id'],'ok':True}),flush=True)
        print(json.dumps({'status':'ready','method':'claude.ai'}),flush=True)
        break
    elif m['method']=='cancel':
        print(json.dumps({'status':'ready','method':'claude.ai'}),flush=True)
        break
'''


def fake_process(monkeypatch):
    original = auth.subprocess.Popen
    def spawn(cmd, **kwargs): return original([sys.executable,'-u','-c',HELPER],**kwargs)
    monkeypatch.setattr(auth.subprocess,'Popen',spawn)


def wait_for(predicate):
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        if predicate(): return
        time.sleep(.01)
    assert predicate()


def test_real_helper_pipe_code_ack_verification_and_cleanup(monkeypatch,tmp_path):
    fake_process(monkeypatch)
    job=auth.AuthJob('claude','login',('fake',str(tmp_path)))
    try:
        job.start(); wait_for(lambda:job.snapshot()['needs_code'])
        assert job.code('secret-code#state')['status'] in ('verifying','ready')
        assert job.retired.wait(3)
        assert job.snapshot()['status']=='ready' and job.proc.poll() is not None
        assert 'secret-code' not in json.dumps(job.snapshot())
        assert not job._acks and not job.snapshot().get('url')
    finally:
        if not job.retired.is_set(): job.cancel()


def test_timeout_and_cancel_ignore_late_success_and_retire_process(monkeypatch,tmp_path):
    fake_process(monkeypatch)
    monkeypatch.setattr(auth,'STATUS_TIMEOUT',.1)
    timed=auth.AuthJob('claude','status',('hang',str(tmp_path)))
    timed.start(); assert timed.retired.wait(4)
    assert timed.snapshot()['status']=='error' and 'timed out' in timed.snapshot()['message']
    assert timed.proc.poll() is not None
    job=auth.AuthJob('claude','login',('fake',str(tmp_path)))
    job.start(); wait_for(lambda:job.snapshot()['needs_code'])
    state=job.cancel()
    assert state['status']=='cancelled' and not state.get('url')
    assert job.retired.is_set() and job.proc.poll() is not None
    job.accept({'status':'ready','method':'claude.ai'})
    assert job.snapshot()['status']=='cancelled'
