"""Every Programmer route answers only the IDE's own page: the right Host, no foreign Origin, not cross-site,
and the per-process token (X-WSP-Auth) on anything that is not a read. Checked in both server modes: unmanaged
(sess.runtime is None) and supervised (sess.runtime set). Nothing here starts a bridge or contacts a provider:
the routes' actions are replaced by recorders, and a refused request must reach none of them.
"""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wsp import conversations, harness, native_harness as native

URL = 'http://127.0.0.1:8123'
TASK = {'title': 'T', 'instructions': 'I', 'scope': ['model'], 'suite': None, 'preserve': None}

GETS = ['/api/harness', '/api/harness?id=run-x', '/api/harness/conversations', '/api/harness/conversation?id=CID',
        '/api/harness/auth?provider=claude']
POSTS = [('/api/harness/start', {'harness': 'claude'}), ('/api/harness/prepare', {'harness': 'claude'}),
         ('/api/harness/send', {'id': 'run-x', 'text': 'hi'}), ('/api/harness/respond', {'id': 'run-x', 'request_id': '1', 'allow': True}),
         ('/api/harness/stop', {'id': 'run-x'}), ('/api/harness/close', {'id': 'run-x'}),
         ('/api/harness/resume', {'conversation': 'CID'}), ('/api/harness/conversation/release', {'conversation': 'CID', 'confirm': True}),
         ('/api/harness/auth/refresh', {'provider': 'claude'})]


class FakeRun:
    native = True
    id = 'run-x'
    status = 'idle'
    proc = None
    conversation_id = None
    _retired = threading.Event()

    def __init__(self, calls):
        self.calls = calls

    def meta(self):
        return {'id': self.id, 'status': self.status}

    def send(self, text):
        self.calls.append('send')
        return self.meta()

    def respond(self, *a):
        self.calls.append('respond')
        return self.meta()

    def interrupt(self):
        self.calls.append('stop')
        return self.meta()

    def close(self):
        self.calls.append('close')
        return self.meta()


@pytest.fixture(params=['unmanaged', 'supervised'])
def setup(request, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    from wsp.session import Session
    ws = tmp_path / 'ws'
    ws.mkdir()
    conv = conversations.Conversation.create(str(ws), provider='claude', native=True, mode='tools', model='', task=dict(TASK))
    conv.update(status='closed', session='s-1')
    conv.release()
    sess = Session(str(tmp_path / 'root'))
    sess.ws = SimpleNamespace(dir=str(ws), patches=[], name='w', lock=threading.RLock(), _runner=None, run_error=None, calib={}, keep={})
    sess.task = dict(TASK)
    sess.brief = lambda mode='tools': 'brief'
    calls = []
    run = FakeRun(calls)
    sess.runs = {run.id: run}
    monkeypatch.setattr(native, 'setup_error', lambda provider: None)
    monkeypatch.setattr(harness, 'start', lambda *a, **k: calls.append('start') or run)
    monkeypatch.setattr(harness, 'resume', lambda *a, **k: calls.append('resume') or SimpleNamespace(meta=run.meta, events=[]))
    monkeypatch.setattr(harness, 'prepare', lambda *a, **k: calls.append('prepare') or {'dir': '/x', 'mcp_config': '/x/.mcp.json', 'brief': 'b'})
    monkeypatch.setattr(conversations, 'mark_interrupted', lambda c: calls.append('release') or c.meta)
    monkeypatch.setattr(native, 'assert_available', lambda sess: None)
    app = create_app(sess, URL)
    if request.param == 'supervised':
        sess.runtime = SimpleNamespace()              # marks the server as managed; its own peer/origin checks also apply
    monkeypatch.setattr(app.state.agent_auth, 'ensure_ready', lambda provider: {'status': 'ready', 'method': 'claude.ai'})
    monkeypatch.setattr(app.state.agent_auth, 'assert_no_login', lambda: None)
    monkeypatch.setattr(app.state.agent_auth, 'status', lambda provider, refresh=False: calls.append('auth') or {'status': 'ready'})
    client = TestClient(app, base_url=URL, raise_server_exceptions=False)
    return client, app.state.agent_auth.csrf, calls, conv.id, request.param


BAD = {'foreign Host': {'Host': 'evil.test:8123'}, 'foreign Host port': {'Host': '127.0.0.1:9999'},
       'cross-site Origin': {'Origin': 'https://evil.test'}, 'null Origin': {'Origin': 'null'},
       'cross-site fetch': {'Sec-Fetch-Site': 'cross-site'}}


@pytest.mark.parametrize('bad', list(BAD))
def test_a_foreign_page_is_refused_on_every_programmer_route(setup, bad):
    client, token, calls, cid, mode = setup
    with client:
        for path in GETS:
            r = client.get(path.replace('CID', cid), headers=BAD[bad])
            assert r.status_code == 403, (mode, bad, path, r.status_code)
            assert r.headers['cache-control'] == 'no-store'
        for path, body in POSTS:
            body = {k: (cid if v == 'CID' else v) for k, v in body.items()}
            r = client.post(path, json=body, headers={**BAD[bad], 'X-WSP-Auth': token})   # even with the token
            assert r.status_code == 403, (mode, bad, path, r.status_code)
        assert calls == []                                 # (the app's own shutdown later closes the run)


def test_a_post_without_the_token_is_refused_and_does_nothing(setup):
    client, token, calls, cid, mode = setup
    with client:
        for path, body in POSTS:
            body = {k: (cid if v == 'CID' else v) for k, v in body.items()}
            for headers in ({}, {'X-WSP-Auth': ''}, {'X-WSP-Auth': token[:-1] + ('A' if token[-1] != 'A' else 'B')}):
                r = client.post(path, json=body, headers={'Origin': URL, **headers})
                assert r.status_code == 403, (mode, path, headers, r.status_code)
        assert calls == []


def test_the_ide_page_itself_gets_through(setup):
    client, token, calls, cid, mode = setup
    with client:
        listing = client.get('/api/harness/conversations', headers={'Origin': URL, 'Sec-Fetch-Site': 'same-origin'})
        assert listing.status_code == 200 and listing.json()['csrf_token'] == token
        assert client.get('/api/harness').json()['csrf_token'] == token
        assert client.get('/api/harness/conversation', params={'id': cid}).status_code == 200
        headers = {'Origin': URL, 'X-WSP-Auth': token}
        for path, body in POSTS:
            body = {k: (cid if v == 'CID' else v) for k, v in body.items()}
            r = client.post(path, json=body, headers=headers)
            assert r.status_code != 403, (mode, path, r.text)
        assert {'start', 'prepare', 'send', 'respond', 'stop', 'close', 'resume', 'release', 'auth'} <= set(calls)
