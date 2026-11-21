"""No dollar cost for subscription runs; cost and budget kept where requests are billed per call.

Fakes only: a protocol fixture stands in for the bridge, a fake job for the sign-in check. No provider,
credential, model or network is involved.
"""
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import agent_auth, harness, native_harness as native

FIXTURE = r'''
import json, os, sys
source = os.environ.get('FAKE_KEY_SOURCE', 'none')
for line in sys.stdin:
    m = json.loads(line); method = m['method']
    def emit(x): print(json.dumps(x), flush=True)
    if method == 'start':
        emit({'event': {'kind': 'init', 'session': 'session-1', 'api_key_source': source}})
        emit({'event': {'kind': 'result', 'ok': True, 'cost_usd': 0.42, 'turns': 3, 'input_tokens': 1200, 'output_tokens': 80}})
        emit({'status': 'idle'})
    elif method == 'close':
        break
    if 'id' in m: emit({'id': m['id'], 'result': {}})
'''


def wait_for(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.01)
    assert predicate()


def started(tmp_path, monkeypatch, billing, key_source='none'):
    monkeypatch.setenv('FAKE_KEY_SOURCE', key_source)
    sess = SimpleNamespace(ws=SimpleNamespace(dir=str(tmp_path)), publish=lambda e: None, runs={}, lock=threading.RLock())
    run = native.NativeRun(sess, 'claude', {'dir': str(tmp_path), 'server': {}, 'brief': 'hello'}, 'tools', '', 0, billing)
    run.cmd = [sys.executable, '-u', '-c', FIXTURE]
    sess.runs[run.id] = run
    run.start()
    wait_for(lambda: run.status == 'idle')
    return run


@pytest.mark.parametrize('provider,method,billing', [
    ('claude', 'claude.ai', 'subscription'), ('claude', 'oauth_token', 'subscription'), ('codex', 'chatgpt', 'subscription'),
    ('claude', 'api_key', 'per_call'), ('claude', 'auth_token', 'per_call'), ('claude', 'bedrock', 'per_call'),
    ('claude', 'vertex', 'per_call'), ('claude', 'foundry', 'per_call'), ('claude', 'configured', 'per_call'),
    ('codex', 'apiKey', 'per_call'), ('codex', 'amazonBedrock', 'per_call'), ('codex', 'gcp', 'per_call'),
    ('codex', 'configured provider', 'per_call'), ('claude', None, 'per_call'), ('codex', 'claude.ai', 'per_call')])
def test_billing_follows_the_sign_in_method_and_unknown_methods_may_pay(provider, method, billing):
    assert native.billing_of(provider, method) == billing


def test_a_subscription_run_shows_no_cost_but_keeps_turns_and_tokens(tmp_path, monkeypatch):
    run = started(tmp_path, monkeypatch, 'subscription')
    try:
        result = next(e for e in run.events if e['kind'] == 'result')
        assert 'cost_usd' not in result
        assert (result['turns'], result['input_tokens'], result['output_tokens']) == (3, 1200, 80)
        assert run.meta()['billing'] == 'subscription' and run.meta()['budget'] is None
    finally:
        run.close()


def test_a_per_call_run_keeps_its_cost(tmp_path, monkeypatch):
    run = started(tmp_path, monkeypatch, 'per_call')
    try:
        assert next(e for e in run.events if e['kind'] == 'result')['cost_usd'] == 0.42
    finally:
        run.close()


def test_an_api_key_found_at_startup_makes_the_run_per_call_and_says_so(tmp_path, monkeypatch):
    run = started(tmp_path, monkeypatch, 'subscription', key_source='ANTHROPIC_API_KEY')
    try:
        assert run.billing == 'per_call'
        assert any(e['kind'] == 'raw' and 'ANTHROPIC_API_KEY' in e['text'] and 'billed per call' in e['text'] for e in run.events)
        assert next(e for e in run.events if e['kind'] == 'result')['cost_usd'] == 0.42
    finally:
        run.close()


def test_a_budget_is_refused_for_a_subscription_and_codex_and_kept_for_claude_billed_per_call(tmp_path, monkeypatch):
    sess = SimpleNamespace(ws=SimpleNamespace(dir=str(tmp_path)), publish=lambda e: None, runs={}, lock=threading.RLock())
    monkeypatch.setattr(native, 'setup_error', lambda provider: None)
    prep = {'dir': str(tmp_path), 'server': {}, 'brief': 'hello'}
    with pytest.raises(ValueError, match='uses your subscription'):
        native.start(sess, prep, 'claude', 'tools', '', 2.5, 'subscription')
    with pytest.raises(ValueError, match='Codex has no USD budget'):
        native.start(sess, prep, 'codex', 'tools', '', 2.5, 'per_call')
    assert sess.runs == {}
    started_runs = []
    monkeypatch.setattr(native.NativeRun, 'start', lambda self: started_runs.append(self))
    run = native.start(sess, prep, 'claude', 'tools', '', 2.5, 'per_call')
    assert run._config['budget'] == 2.5 and run.meta()['budget'] == 2.5 and started_runs == [run]


def test_the_sign_in_state_reports_billing_and_start_passes_it_on(tmp_path, monkeypatch):
    job = agent_auth.AuthJob('claude', 'status', ('/fake', str(tmp_path)))
    job.accept({'status': 'ready', 'method': 'claude.ai'})
    assert job.snapshot()['billing'] == 'subscription'
    job = agent_auth.AuthJob('claude', 'status', ('/fake', str(tmp_path)))
    job.accept({'status': 'ready', 'method': 'api_key'})
    assert job.snapshot()['billing'] == 'per_call'

    from fastapi.testclient import TestClient
    from wsp.server import create_app
    from wsp.session import Session
    sess = Session(str(tmp_path / 'root'))
    sess.ws = SimpleNamespace(dir=str(tmp_path / 'ws'), patches=[])
    seen = {}
    monkeypatch.setattr(harness, 'start', lambda *a: seen.setdefault('args', a) and SimpleNamespace(meta=lambda: {'id': 'run-x'}))
    app = create_app(sess, 'http://127.0.0.1:0')
    monkeypatch.setattr(app.state.agent_auth, 'ensure_ready', lambda provider: {'status': 'ready', 'method': 'claude.ai'})
    # The page's own origin and per-process token (the Programmer routes refuse anything else).
    with TestClient(app, base_url='http://127.0.0.1:0', headers={'X-WSP-Auth': app.state.agent_auth.csrf},
                    raise_server_exceptions=False) as client:
        r = client.post('/api/harness/start', json={'harness': 'claude'})
        assert r.status_code == 200, r.text
    assert seen['args'][-1] == 'subscription'
