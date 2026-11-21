"""Agent lifecycle tests with a protocol fixture, no model weights or inference."""
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import harness, native_harness as native

FIXTURE = r'''
import json, sys
for line in sys.stdin:
    m=json.loads(line); p=m.get('params', {}); method=m['method']
    def emit(x): print(json.dumps(x), flush=True)
    if method=='start':
        emit({'event': {'kind':'init','session':'session-1'}})
        emit({'event': {'kind':'result','ok':True}})
        emit({'status':'idle'})
    elif method=='send':
        emit({'status':'running'})
        emit({'event': {'kind':'user','text':p['text']}})
        if p['text']=='approval':
            emit({'event': {'kind':'prompt','request_id':'1','title':'Write'}})
        else:
            emit({'event': {'kind':'result','ok': p['text']!='failure'}})
            emit({'status':'idle'})
    elif method=='respond':
        emit({'event': {'kind':'prompt_resolved','request_id':p['request_id']}})
        emit({'status':'idle'})
    elif method=='interrupt':
        emit({'event': {'kind':'prompt_resolved','request_id':'1'}})
        emit({'status':'idle'})
    elif method=='close':
        break
    if 'id' in m: emit({'id':m['id'],'result':{}})
'''


def wait_for(predicate):
    end = time.monotonic() + 3
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.01)
    assert predicate()


@pytest.fixture
def run(tmp_path):
    published = []
    sess = SimpleNamespace(ws=SimpleNamespace(dir=str(tmp_path)), publish=published.append, runs={}, lock=threading.RLock())
    prep = {"dir": str(tmp_path), "server": {}, "brief": "hello"}
    run = native.NativeRun(sess, "claude", prep, "tools", "", 0)
    run.cmd = [sys.executable, "-u", "-c", FIXTURE]
    sess.runs[run.id] = run
    run.start()
    wait_for(lambda: run.status == 'idle')
    yield run, sess, published
    run.close()


def test_followups_keep_session_and_turn_failure_does_not_kill_connection(run):
    r, sess, published = run
    pid = r.proc.pid
    r.send('one'); r.send('failure')
    assert r.status == 'idle' and r.last_result is False
    r.send('three')
    assert r.proc.pid == pid and r.session_id == 'session-1' and r.last_result is True
    assert [e['text'] for e in r.events if e['kind'] == 'user'] == ['one', 'failure', 'three']
    assert any(e['type'] == 'agent' for e in published)


def test_prompt_responses_and_interrupt_are_bidirectional(run):
    r, _, _ = run
    r.send('approval')
    assert r.pending['1']['title'] == 'Write'
    with pytest.raises(RuntimeError, match='Wait'):
        r.send('while waiting')
    with pytest.raises(ValueError, match='Invalid'):
        r.respond('1', 'yes')
    r.respond('1', False)
    assert not r.pending and r.status == 'idle'
    with pytest.raises(ValueError, match='no longer'):
        r.respond('1', True)
    r.send('approval'); r.interrupt()
    assert not r.pending and r.status == 'idle'


def test_idle_session_blocks_new_session_and_close_kills_process(run):
    r, sess, _ = run
    with pytest.raises(RuntimeError, match='Close'):
        native.assert_available(sess)
    r.close()
    assert r.proc.poll() is not None and r.status == 'closed'
    native.assert_available(sess)
    with pytest.raises(RuntimeError):
        r.send('after close')


def test_concurrent_send_is_reserved_before_writing(run, monkeypatch):
    r, _, _ = run
    entered, release = threading.Event(), threading.Event()
    def request(*args, **kwargs):
        entered.set(); release.wait(2)
    monkeypatch.setattr(r, 'request', request)
    sender = threading.Thread(target=lambda: r.send('one'))
    sender.start(); assert entered.wait(2)
    with pytest.raises(RuntimeError, match='Wait'):
        r.send('two')
    release.set(); sender.join()


def test_failed_state_is_not_overwritten_by_bridge_close(run):
    r, _, _ = run
    r._state('failed'); r._state('closed')
    with pytest.raises(RuntimeError, match='Close'):
        native.assert_available(r._sess)
    r.close()
    assert r.status == 'failed' and r.proc.poll() is not None


def test_missing_dependency_has_actionable_setup(monkeypatch):
    monkeypatch.setattr(native.shutil, 'which', lambda name: '/bin/' + name)
    monkeypatch.setattr(native, 'BRIDGE', Path('/nonexistent/bridge'))
    assert 'npm ci' in native.setup_error('claude')
    assert native.setup_error('codex') is None


def test_legacy_custom_command_and_native_prepare_route():
    prep = {'dir': '/tmp', 'brief': 'task', 'mcp_config': '/tmp/.mcp.json'}
    args, text = harness.command('custom', prep, 'http://local', custom='my-agent --file {brief}')
    assert args == ['my-agent', '--file', '/tmp/BRIEF.md'] and text == 'task'
    args, text = harness.command('claude', prep, 'http://local')
    assert args[-1].endswith('/bridge/index.mjs') and text is None


def test_api_exposes_pending_prompts_and_blocks_workspace_or_task_switch(run):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    from wsp.session import Session
    r, fake, _ = run
    sess = Session('/not-a-real-model-directory')
    sess.ws = fake.ws
    sess.runs = fake.runs
    sess.log = []
    app = create_app(sess, 'http://127.0.0.1:0')
    with TestClient(app, base_url='http://127.0.0.1:0', headers={'X-WSP-Auth': app.state.agent_auth.csrf},
                    raise_server_exceptions=False) as client:
        assert client.post('/api/open', json={'name': 'never-load-this'}).status_code == 400
        assert client.post('/api/task', json={'scope': ['model']}).status_code == 400
        response = client.post('/api/harness/send', json={'id': r.id, 'text': 'approval'})
        assert response.status_code == 200, response.text
        assert response.json()['pending'][0]['request_id'] == '1'
        prompt = client.get('/api/harness', params={'id': r.id}).json()
        assert prompt['session'] == 'session-1' and prompt['pending']
        assert client.post('/api/harness/respond', json={'id': r.id, 'request_id': '1', 'allow': False}).status_code == 200
    assert r.proc.poll() is not None and r.status == 'closed'  # application shutdown owns cleanup


def test_run_read_gives_server_time_and_events_from_an_index(run):
    """The status line times events with the server's clock; a reconnect fetches only what it missed."""
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    from wsp.session import Session
    r, fake, _ = run
    r.send('one')
    sess = Session('/not-a-real-model-directory')
    sess.ws, sess.runs, sess.log = fake.ws, fake.runs, []
    app = create_app(sess, 'http://127.0.0.1:0')
    with TestClient(app, base_url='http://127.0.0.1:0', headers={'X-WSP-Auth': app.state.agent_auth.csrf},
                    raise_server_exceptions=False) as client:
        before = time.time()
        full = client.get('/api/harness', params={'id': r.id}).json()
        assert before <= full['now'] <= time.time()
        assert [e['i'] for e in full['events']] == list(range(len(r.events)))
        tail = client.get('/api/harness', params={'id': r.id, 'since': 2}).json()
        assert tail['events'] == full['events'][2:] and tail['status'] == full['status']
        listing = client.get('/api/harness').json()
        assert isinstance(listing['now'], float) and listing['runs'][0]['id'] == r.id
        missing = client.get('/api/harness', params={'id': 'run-none'})
        assert missing.status_code == 404 and missing.json() == {'error': 'Unknown run'}
