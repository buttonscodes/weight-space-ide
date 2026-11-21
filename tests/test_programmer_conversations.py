"""Programmer conversations saved on disk: listed after a restart, read without a provider, resumed by session id.

A fake bridge process (a few lines of Python speaking the bridge protocol) stands in for Claude Code and Codex:
no provider, CLI, credential, network call or model is involved. A "restart" is a new Session and app on the
same root; a "server death" is a child process killed with SIGKILL in the middle of a turn.
"""
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wsp import conversations, harness, native_harness as native

FIXTURE = r'''
import json, sys
GONE, FLAKY, WRONG, CRASH, LEAKY = {'session-gone'}, {'session-flaky'}, {'session-wrong'}, {'session-crash'}, {'session-leaky'}
for line in sys.stdin:
    m = json.loads(line); p = m.get('params', {}); method = m['method']
    def emit(x): print(json.dumps(x), flush=True)
    if method == 'start':
        r = p.get('resume')
        if r in GONE:        # the provider's definite answer that the session does not exist
            emit({'event': {'kind': 'resume_failed', 'gone': True, 'text': 'Claude Code has no session %s for this directory.' % r}})
            emit({'status': 'failed'}); emit({'status': 'closed'}); break
        if r in LEAKY:       # dies with credentials in its stderr
            print('debug: account someone@example.test plan max', file=sys.stderr, flush=True)
            print('Error: 401 with key sk-ant-api03-abcdefghijklmnopqrstuvwx and Bearer eyJhbGciOiJIUzI1NiJ9.payload '
                  'session token 3f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a ' + 'z' * 400, file=sys.stderr, flush=True)
            sys.exit(1)
        if r in CRASH:       # the bridge dies at once, saying why only on stderr
            print('Error: no conversation found with session ID: %s' % r, file=sys.stderr, flush=True)
            sys.exit(1)
        if r in FLAKY:       # any other failure of the attempt (a timeout, an error the provider gave)
            print('provider diagnostic: connection reset', file=sys.stderr, flush=True)
            emit({'event': {'kind': 'resume_failed', 'gone': False, 'text': 'Claude Code could not check session %s: ETIMEDOUT' % r}})
            emit({'status': 'failed'}); emit({'status': 'closed'}); break
        emit({'event': {'kind': 'raw', 'text': 'START ' + json.dumps({'resume': r, 'mode': p.get('mode'), 'model': p.get('model'), 'budget': p.get('budget')})}})
        emit({'event': {'kind': 'init', 'session': 'session-other' if r in WRONG else (r or 'session-1')}})
        emit({'status': 'running'})
        emit({'event': {'kind': 'user', 'text': p['brief']}})
        emit({'event': {'kind': 'result', 'ok': True, 'cost_usd': 0.1}})
        emit({'status': 'idle'})
    elif method == 'send':
        emit({'status': 'running'}); emit({'event': {'kind': 'user', 'text': p['text']}})
        if p['text'] == 'hang':
            emit({'event': {'kind': 'tool_use', 'id': 't1', 'name': 'mcp__wsp__inspect', 'input': {'address': 'L1N1', 'what': 'card'}}})
        else:
            if p['text'] == 'big':
                emit({'event': {'kind': 'tool_result', 'id': 't0', 'text': 'x' * 200000}})
            emit({'event': {'kind': 'result', 'ok': True}}); emit({'status': 'idle'})
    elif method == 'close':
        break
    if 'id' in m: emit({'id': m['id'], 'result': {}})
'''

# Another IDE process holding a conversation: a child that takes the record's owner lock and keeps it.
HOLD = r'''
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR)
fcntl.flock(fd, fcntl.LOCK_EX)
print('HELD', flush=True)
time.sleep(60)
'''


def hold_elsewhere(conv):
    conv.release()                      # this test process created it, and so held it until now
    holder = subprocess.Popen([sys.executable, '-c', HOLD, str(Path(conv.path) / 'owner.lock')], stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == 'HELD'
    return holder


TASK = {'title': 'Paris to Rome', 'instructions': 'Make the model say Rome', 'scope': ['model'], 'suite': None, 'preserve': None}
PATCH = SimpleNamespace(id='0001-00042', enabled=True, strength=1.0, author='programmer', message='scale L3 neurons')


def wait_for(predicate, timeout=4):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.01)
    assert predicate()


@pytest.fixture
def fake_bridge(monkeypatch):
    """Every NativeRun talks to the fixture instead of node + the provider; Popen calls are recorded."""
    monkeypatch.setattr(native, 'setup_error', lambda provider: None)
    original = native.NativeRun.start
    started = []

    def start(self):
        self.cmd = [sys.executable, '-u', '-c', FIXTURE]
        started.append(self)
        original(self)
    monkeypatch.setattr(native.NativeRun, 'start', start)
    return started


def make_session(root, ws_dir, task=TASK):
    from wsp.session import Session
    sess = Session(str(root))
    sess.ws = SimpleNamespace(dir=str(ws_dir), patches=[PATCH], name='fixture', lock=threading.RLock(), _runner=None,
                              run_error=None, calib={}, keep={})
    sess.task = dict(task)
    sess.brief = lambda mode='tools': f'BRIEF for {mode}'
    return sess


def make_app(sess, monkeypatch, method='claude.ai', ready=True):
    """The app, opened as the IDE's own page does: its Host, and the per-process token on every POST."""
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    app = create_app(sess, 'http://127.0.0.1:0')

    def ensure_ready(provider):
        if not ready:
            raise RuntimeError('Sign in to Claude Code in the Programmer tab before starting.')
        return {'status': 'ready', 'method': method}
    monkeypatch.setattr(app.state.agent_auth, 'ensure_ready', ensure_ready)
    return TestClient(app, base_url='http://127.0.0.1:0', headers={'X-WSP-Auth': app.state.agent_auth.csrf},
                      raise_server_exceptions=False)


def new_conv(ws_dir, **fields):
    """A saved conversation in a workspace directory (which, as a real workspace's, exists)."""
    os.makedirs(ws_dir, exist_ok=True)
    return conversations.Conversation.create(str(ws_dir), **fields)


def close_all(sess):
    for r in list(sess.runs.values()):
        if getattr(r, 'native', False):
            r.close()


def files_under(path):
    out = {}
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            p = os.path.join(dirpath, name)
            out[p] = os.stat(p).st_mtime_ns
    return out


# ------------------------------------------------------------------------------------------------ saving and reading
def test_a_conversation_survives_a_restart_and_reading_it_contacts_no_provider(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        r = client.post('/api/harness/start', json={'harness': 'claude', 'mode': 'tools'})
        assert r.status_code == 200, r.text
        run = sess.runs[r.json()['id']]
        wait_for(lambda: run.status == 'idle')
        run.send('one more')
        assert run.conversation_id and r.json()['conversation'] == run.conversation_id
        listed = client.get('/api/harness/conversations').json()['conversations']
        assert [c['id'] for c in listed] == [run.conversation_id] and listed[0]['live']
        assert listed[0]['resume']['code'] == 'open'
        shown = list(run.events)
        run.close()
    cid = run.conversation_id

    # A new Session and app on the same root: the server restarted.
    sess2 = make_session(tmp_path / 'root', ws_dir)
    popen = []
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: popen.append(a) or (_ for _ in ()).throw(AssertionError('no process')))
    with make_app(sess2, monkeypatch) as client:
        listed = client.get('/api/harness/conversations').json()['conversations']
        assert len(listed) == 1
        c = listed[0]
        assert (c['id'], c['provider'], c['mode'], c['task'], c['status'], c['session']) == \
            (cid, 'claude', 'tools', 'Paris to Rome', 'closed', 'session-1')
        assert c['n_events'] == len(shown) and not c['live'] and c['resume'] == {'ok': True, 'code': None, 'reason': None}
        read = client.get('/api/harness/conversation', params={'id': cid}).json()
        assert [(e['kind'], e.get('text')) for e in read['events']] == [(e['kind'], e.get('text')) for e in shown]
        assert read['meta']['task_full'] == TASK and read['meta']['started'] <= read['meta']['ended']
        assert 'cost_usd' not in json.dumps(read['events'])        # a subscription run's saved events carry no cost
        assert client.get('/api/harness/conversation', params={'id': '../../etc'}).status_code == 404
    assert popen == [] and sess2.runs == {}


def test_files_are_private_bounded_and_outside_the_agent_directory(tmp_path, fake_bridge, monkeypatch):
    """The store is <workspace>/conversations, a sibling of the agent's working directory <workspace>/agent."""
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    (ws_dir / 'weights.bin').parent.mkdir(parents=True)
    (ws_dir / 'weights.bin').write_bytes(b'w')
    before = files_under(tmp_path)
    monkeypatch.setattr(conversations, 'MAX_EVENT_BYTES', 4096)
    run = harness.start(sess, 'http://x', 'claude', 'tools', billing='subscription')
    wait_for(lambda: run.status == 'idle')
    run.send('big')
    run.close()
    after = files_under(tmp_path)
    changed = [p for p in after if before.get(p) != after[p]]
    base = ws_dir / 'conversations'
    agent = [p for p in changed if p.startswith(str(ws_dir / 'agent') + os.sep)]
    saved = [p for p in changed if p.startswith(str(base) + os.sep)]
    assert sorted(os.path.basename(p) for p in agent) == ['.mcp.json', 'BRIEF.md']     # the agent's own inputs only
    assert saved and len(agent) + len(saved) == len(changed), changed
    assert not (ws_dir / 'agent' / 'conversations').exists()
    assert (base / '.gitignore').read_text().splitlines()[-1] == '*'               # never committed
    d = base / run.conversation_id
    assert stat.S_IMODE(os.stat(base).st_mode) == 0o700 and stat.S_IMODE(os.stat(d).st_mode) == 0o700
    for name in ('events.jsonl', 'meta.json', 'owner.lock'):
        assert stat.S_IMODE(os.stat(d / name).st_mode) == 0o600
    lines = (d / 'events.jsonl').read_bytes().splitlines()
    assert max(len(x) for x in lines) <= 4096
    big = next(json.loads(x) for x in lines if b'"t0"' in x)
    assert big['clipped'] and 'more characters not saved' in big['text']
    assert not list(base.glob('*/*.tmp'))


def test_event_count_and_size_limits_stop_saving_with_one_notice(tmp_path):
    conv = conversations.Conversation.create(str(tmp_path), provider='claude', native=True)
    old = conversations.MAX_EVENTS
    conversations.MAX_EVENTS = 5
    try:
        for i in range(12):
            conv.append({'kind': 'text', 'i': i, 't': i, 'text': f'line {i}'})
    finally:
        conversations.MAX_EVENTS = old
    events = conversations.Conversation.open(str(tmp_path), conv.id).events()
    assert [e['text'] for e in events][:4] == ['line 0', 'line 1', 'line 2', 'line 3']
    assert len(events) == 5 and 'limit reached' in events[-1]['text'] and conv.meta['truncated']
    conv2 = conversations.Conversation.create(str(tmp_path), provider='claude', native=True)
    old = conversations.MAX_FILE_BYTES
    conversations.MAX_FILE_BYTES = 3000
    try:
        for i in range(100):
            conv2.append({'kind': 'text', 'i': i, 't': i, 'text': 'y' * 100})
    finally:
        conversations.MAX_FILE_BYTES = old
    assert os.path.getsize(Path(conv2.path) / 'events.jsonl') <= 3000
    assert 'limit reached' in conv2.events()[-1]['text']


def test_a_partial_last_line_is_skipped_and_streamed_pieces_are_joined(tmp_path):
    conv = conversations.Conversation.create(str(tmp_path), provider='claude', native=True)
    for i, piece in enumerate(['Hel', 'lo', ' there']):
        conv.append({'kind': 'text_delta', 'id': 'm1:0', 'i': i, 't': i, 'text': piece})
    conv.append({'kind': 'tool_use', 'id': 'a', 'i': 3, 't': 3, 'name': 'overview', 'input': {}})
    with open(Path(conv.path) / 'events.jsonl', 'ab') as f:
        f.write(b'{"kind": "text", "text": "cut off by a cra')
    events = conversations.Conversation.open(str(tmp_path), conv.id).events()
    assert [(e['kind'], e.get('text'), e['i']) for e in events] == [('text_delta', 'Hello there', 0), ('tool_use', None, 1)]


def test_only_the_newest_conversations_are_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(conversations, 'KEEP', 3)
    made = []
    for k in range(5):
        c = conversations.Conversation.create(str(tmp_path), provider='claude', native=True)
        c.update(status='closed', started=1000 + k)
        c.release()                     # a record this process still holds is never pruned
        made.append(c.id)
    kept = sorted(n for n in os.listdir(conversations.root(str(tmp_path))) if n.startswith('conv-'))
    assert sorted(made[-3:]) == kept


# ------------------------------------------------------------------------------------------------ a server that died mid-turn
CHILD = r'''
import os, sys, threading, time
from types import SimpleNamespace
sys.path.insert(0, os.environ['WSP_ROOT'])
from wsp import native_harness as native
ws = SimpleNamespace(dir=os.environ['WS_DIR'], patches=[])
sess = SimpleNamespace(ws=ws, publish=lambda e: None, runs={}, lock=threading.RLock(),
                       task={'title': 'T', 'instructions': 'I', 'scope': ['model'], 'suite': None, 'preserve': None})
native.setup_error = lambda provider: None
original = native.NativeRun.start
def start(self):
    self.cmd = [sys.executable, '-u', '-c', os.environ['FIXTURE']]
    original(self)
native.NativeRun.start = start
agent = os.path.join(ws.dir, 'agent'); os.makedirs(agent, exist_ok=True)
run = native.start(sess, {'dir': agent, 'server': {}, 'brief': 'brief'}, 'claude', 'tools', '', 0, 'subscription')
while run.status != 'idle': time.sleep(.01)
run.send('hang')
print('READY', run.conversation_id, flush=True)
time.sleep(60)
'''


def test_a_conversation_cut_off_by_a_dead_server_is_recorded_as_interrupted(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    env = {**os.environ, 'WSP_ROOT': str(ROOT), 'WS_DIR': str(ws_dir), 'FIXTURE': FIXTURE}
    child = subprocess.Popen([sys.executable, '-u', '-c', CHILD], env=env, stdout=subprocess.PIPE, text=True,
                             start_new_session=True)
    try:
        line = child.stdout.readline().split()
        assert line[0] == 'READY', line
        cid = line[1]
    finally:
        os.killpg(child.pid, signal.SIGKILL)                 # the server and its bridge die at once
        child.wait(5)
    sess = make_session(tmp_path / 'root', ws_dir, task={'title': 'T', 'instructions': 'I', 'scope': ['model'], 'suite': None, 'preserve': None})
    with make_app(sess, monkeypatch) as client:
        c = client.get('/api/harness/conversations').json()['conversations'][0]
        assert c['id'] == cid and c['status'] == 'interrupted' and c['resume']['ok']
        read = client.get('/api/harness/conversation', params={'id': cid}).json()
        assert read['meta']['interrupted'] == 'The IDE server stopped while a turn was running.'
        kinds = [e['kind'] for e in read['events']]
        assert kinds[-2:] == ['user', 'tool_use'] and 'tool_result' not in kinds[kinds.index('tool_use'):]
        assert c['n_events'] == len(read['events'])
    saved = json.loads((ws_dir / 'conversations' / cid / 'meta.json').read_text())
    assert saved['status'] == 'interrupted'                    # recorded, not just shown


# ------------------------------------------------------------------------------------------------ resuming
def start_and_close(client, sess, monkeypatch, text=None):
    r = client.post('/api/harness/start', json={'harness': 'claude', 'mode': 'files', 'model': 'm-1'})
    assert r.status_code == 200, r.text
    run = sess.runs[r.json()['id']]
    wait_for(lambda: run.status == 'idle')
    if text:
        run.send(text)
    run.close()
    return run



def test_list_resume_and_start_work_beside_hundreds_of_malformed_records(tmp_path, fake_bridge, monkeypatch):
    """The store test's fuzz shapes (tests/test_conversation_store.py) beside a good conversation: the list route,
    a real Resume of the good one (fake bridge) and a new Start all work."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('store_fuzz', Path(__file__).with_name('test_conversation_store.py'))
    fuzz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fuzz)
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        first = start_and_close(client, sess, monkeypatch)
    monkeypatch.setattr(conversations, 'KEEP', 10_000)
    planted = 0
    for raw in fuzz.FUZZ_META:
        c = new_conv(ws_dir)
        (Path(c.path) / 'meta.json').write_bytes(raw)
        c.close(); planted += 1
    for field in fuzz.FIELDS:
        for value in fuzz.WRONG:
            c = new_conv(ws_dir)
            d = json.loads((Path(c.path) / 'meta.json').read_text())
            d[field] = value
            try:
                (Path(c.path) / 'meta.json').write_text(json.dumps(d))
            except (ValueError, OverflowError):
                pass
            c.close(); planted += 1
    for raw in fuzz.FUZZ_EVENTS:
        c = new_conv(ws_dir)
        c.update(status='running')
        (Path(c.path) / 'events.jsonl').write_bytes(raw)
        c.close(); planted += 1
    assert planted > 300
    sess2 = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess2, monkeypatch) as client:
        r = client.get('/api/harness/conversations')
        assert r.status_code == 200 and len(r.json()['conversations']) == planted + 1
        row = next(c for c in r.json()['conversations'] if c['id'] == first.conversation_id)
        assert row['resume']['ok'], row
        r = client.post('/api/harness/resume', json={'conversation': first.conversation_id})
        assert r.status_code == 200, r.text
        run = sess2.runs[r.json()['id']]
        wait_for(lambda: run.status == 'idle')
        run.close()
        wait_for(lambda: run._retired.is_set())
        r = client.post('/api/harness/start', json={'harness': 'claude'})
        assert r.status_code == 200, r.text
        run = sess2.runs[r.json()['id']]
        assert run.conversation_id
        wait_for(lambda: run.status == 'idle')
        run.close()

def test_resume_passes_the_recorded_session_and_tells_the_model_the_workspace_may_have_changed(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        first = start_and_close(client, sess, monkeypatch, 'look at L3')
    n_before = len(first.events)
    sess2 = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess2, monkeypatch) as client:
        r = client.post('/api/harness/resume', json={'conversation': first.conversation_id, 'text': 'carry on with L4'})
        assert r.status_code == 200, r.text
        body = r.json()
        run = sess2.runs[body['id']]
        wait_for(lambda: run.status == 'idle')
        assert body['conversation'] == first.conversation_id and body['resumed'] and body['mode'] == 'files' and body['model'] == 'm-1'
        start = next(e for e in run.events[n_before:] if e['kind'] == 'raw' and e['text'].startswith('START '))
        assert json.loads(start['text'][6:])['resume'] == 'session-1'
        first_message = [e for e in run.events if e['kind'] == 'user'][-1]['text']
        assert 'being resumed' in first_message and 'may have changed' in first_message
        assert '0001-00042 [on x1] programmer: scale L3 neurons' in first_message and first_message.endswith('carry on with L4')
        assert [e['kind'] for e in run.events[:n_before]] == [e['kind'] for e in first.events]
        assert any(e['kind'] == 'resumed' for e in run.events[n_before:])
        assert [e['i'] for e in body['events']] == list(range(len(body['events'])))
        # One conversation at a time: nothing else starts or resumes while it is open.
        busy = client.post('/api/harness/resume', json={'conversation': first.conversation_id})
        assert busy.status_code == 409 and busy.json()['refusal'] == 'open'
        assert client.post('/api/harness/start', json={'harness': 'claude'}).status_code == 400
        run.close()
    saved = conversations.Conversation.open(str(ws_dir), first.conversation_id)
    assert saved.meta['status'] == 'closed' and saved.meta['session'] == 'session-1' and len(saved.meta['resumes']) == 1
    assert len(saved.events()) == len(run.events)
    assert (saved.meta['pid'], saved.meta['boot']) == (os.getpid(), conversations.BOOT)   # the resuming process owns it now
    assert not conversations.held_here(saved.path)                                       # and let go when it closed


def test_a_session_the_provider_says_does_not_exist_is_gone_and_can_be_tried_again(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        first = start_and_close(client, sess, monkeypatch)
    conv = conversations.Conversation.open(str(ws_dir), first.conversation_id)
    conv.update(session='session-gone')
    sess2 = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess2, monkeypatch) as client:
        r = client.post('/api/harness/resume', json={'conversation': first.conversation_id})
        assert r.status_code == 200
        run = sess2.runs[r.json()['id']]
        wait_for(lambda: run.status == 'failed')
        # Read at once, with no wait for the run's thread to retire: the failed status is final, so the list
        # already says gone (never 'open' or resumable).
        listed = client.get('/api/harness/conversations').json()['conversations'][0]
        assert run.events[-1]['kind'] == 'resume_failed' and not any(e['kind'] == 'init' for e in run.events[len(first.events):])
        assert listed['status'] == 'closed' and listed['session'] == 'session-gone'
        assert listed['resume']['code'] == 'gone' and 'has no session session-gone' in listed['resume']['reason']
        assert listed['resume']['retry'] is True
        again = client.post('/api/harness/resume', json={'conversation': first.conversation_id})
        assert again.status_code == 409 and again.json()['refusal'] == 'gone'
        wait_for(lambda: run._retired.is_set())
        # Try again: the same refusal is lifted for one attempt, which the provider answers again.
        retry = client.post('/api/harness/resume', json={'conversation': first.conversation_id, 'retry': True})
        assert retry.status_code == 200, retry.text
        run2 = sess2.runs[retry.json()['id']]
        wait_for(lambda: run2.status == 'failed' and run2._retired.is_set())
    saved = conversations.Conversation.open(str(ws_dir), first.conversation_id)
    assert saved.meta['status'] == 'closed' and saved.meta['resume_error'] and saved.meta['last_resume_failure'] is None
    added = saved.events()[len(first.events):]
    assert [(e['kind'], e['gone']) for e in added] == [('resume_failed', True), ('resume_failed', True)]   # one per attempt


def test_any_other_resume_failure_is_one_failed_attempt_that_can_be_retried(tmp_path, fake_bridge, monkeypatch):
    """A timeout, an error or an ambiguous check is not "gone": the record keeps its status and session, says
    what failed (with the provider's own text), and Resume stays available."""
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        first = start_and_close(client, sess, monkeypatch)
    conversations.Conversation.open(str(ws_dir), first.conversation_id).update(session='session-flaky')
    before = conversations.Conversation.open(str(ws_dir), first.conversation_id).meta
    sess2 = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess2, monkeypatch) as client:
        run = sess2.runs[client.post('/api/harness/resume', json={'conversation': first.conversation_id}).json()['id']]
        wait_for(lambda: run.status == 'failed')
        listed = client.get('/api/harness/conversations').json()['conversations'][0]
        assert listed['resume']['ok'] and 'ETIMEDOUT' in listed['last_resume_failure']
        wait_for(lambda: run._retired.is_set())
    after = conversations.Conversation.open(str(ws_dir), first.conversation_id)
    assert after.meta['resume_error'] is None and 'ETIMEDOUT' in after.meta['last_resume_failure']
    assert {k: after.meta[k] for k in ('status', 'session', 'ended', 'started')} == \
        {k: before[k] for k in ('status', 'session', 'ended', 'started')}
    added = after.events()[len(first.events):]
    assert [e['kind'] for e in added] == ['resume_failed'] and added[0]['gone'] is False


def test_a_bridge_that_dies_during_a_resume_is_a_failed_attempt_with_the_provider_output(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        first = start_and_close(client, sess, monkeypatch)
    conversations.Conversation.open(str(ws_dir), first.conversation_id).update(session='session-crash')
    sess2 = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess2, monkeypatch) as client:
        run = sess2.runs[client.post('/api/harness/resume', json={'conversation': first.conversation_id}).json()['id']]
        wait_for(lambda: run._retired.is_set())
    meta = conversations.Conversation.open(str(ws_dir), first.conversation_id).meta
    assert meta['resume_error'] is None and 'Agent bridge exited (1)' in meta['last_resume_failure']
    assert 'Resume can be tried again' in meta['last_resume_failure'] and 'Start a new conversation' not in meta['last_resume_failure']
    assert 'Provider output: Error: no conversation found with session ID: session-crash' in meta['last_resume_failure']


def test_provider_output_saved_with_a_failure_is_one_short_redacted_line(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        first = start_and_close(client, sess, monkeypatch)
    conversations.Conversation.open(str(ws_dir), first.conversation_id).update(session='session-leaky')
    sess2 = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess2, monkeypatch) as client:
        run = sess2.runs[client.post('/api/harness/resume', json={'conversation': first.conversation_id}).json()['id']]
        wait_for(lambda: run._retired.is_set())
    saved = conversations.Conversation.open(str(ws_dir), first.conversation_id)
    text = saved.meta['last_resume_failure']
    provider = text.split('Provider output: ', 1)[1]
    assert provider.startswith('Error: 401 with key [redacted] and [redacted]') and len(provider) <= 301
    for secret in ('sk-ant-api03', 'eyJhbGciOiJIUzI1NiJ9', '3f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c', 'someone@example.test'):
        assert secret not in text and secret not in json.dumps(saved.events())


def test_redaction_of_provider_output():
    assert native.redact('a\nb\n  last line  \n') == 'last line'
    assert native.redact('Authorization: Bearer abc.def-ghi') == 'Authorization: [redacted]'
    assert native.redact('key sk-proj-AAAA1111bbbb') == 'key [redacted]'
    assert native.redact('hex ' + 'ab' * 20 + ' b64 ' + 'QUJD' * 10) == 'hex [redacted] b64 [redacted]'
    assert native.redact('session 1b2c3d4e-5f60-4789-a0b1-c2d3e4f5a6b7 not found') == 'session 1b2c3d4e-5f60-4789-a0b1-c2d3e4f5a6b7 not found'
    assert len(native.redact('x ' * 1000)) == 301


def test_a_status_is_saved_before_other_threads_can_see_it(tmp_path, fake_bridge, monkeypatch):
    """Whoever reads run.status finds that status already in the saved record."""
    seen = []
    original = conversations.Conversation.update
    def watch(self, **fields):
        run = next(iter(fake_bridge), None)
        if run is not None and 'status' in fields and 'session' in fields:
            seen.append((fields['status'], run.status))
        return original(self, **fields)
    monkeypatch.setattr(conversations.Conversation, 'update', watch)
    sess = make_session(tmp_path / 'root', tmp_path / 'ws')
    run = harness.start(sess, 'http://x', 'claude', 'tools', billing='subscription')
    wait_for(lambda: run.status == 'idle')
    run.send('again')
    run.close()
    # (status being saved, run.status at that moment): the status is still the previous one while it is saved.
    assert ('idle', 'running') in seen and ('closing', 'idle') in seen and ('closed', 'closing') in seen
    assert not [x for x in seen if x[0] == x[1] and x[0] in ('idle', 'closing', 'closed')]


def test_a_provider_that_starts_another_session_ends_the_resume(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        first = start_and_close(client, sess, monkeypatch)
    conversations.Conversation.open(str(ws_dir), first.conversation_id).update(session='session-wrong')
    sess2 = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess2, monkeypatch) as client:
        run = sess2.runs[client.post('/api/harness/resume', json={'conversation': first.conversation_id}).json()['id']]
        wait_for(lambda: run.status in ('failed', 'closed') and run._retired.is_set())
        failed = [e for e in run.events if e['kind'] == 'resume_failed']
        assert failed and 'session-other' in failed[0]['text'] and run.session_id is None
        assert 'may have been sent to that new session' in failed[0]['text']
        assert run.events[-1]['kind'] == 'resume_failed'          # nothing of the other session is shown after it
    saved = conversations.Conversation.open(str(ws_dir), first.conversation_id)
    assert saved.meta['session'] == 'session-wrong' and saved.meta['status'] == 'closed'
    assert saved.meta['resume_error'] is None and 'session-other' in saved.meta['last_resume_failure']   # retryable
    # What is kept of the attempt: exactly one resume_failed event. Not the resumed marker, the resume message,
    # nor the provider output that came before the other session's init.
    added = saved.events()[len(first.events):]
    assert [e['kind'] for e in added] == ['resume_failed'] and 'session-other' in added[0]['text']


@pytest.mark.parametrize('case,code,words', [
    ('legacy', 'legacy', 'This was a custom command run. Those run once and keep no session'), ('no_session', 'no_session', 'No Claude Code session id was recorded'),
    ('task', 'task', 'The task changed since this conversation started (instructions, scope)'),
    ('workspace', 'workspace', 'keeps no session for it'), ('elsewhere', 'elsewhere', 'Another IDE process')])
def test_each_refusal_says_why(tmp_path, fake_bridge, monkeypatch, case, code, words):
    ws_dir = tmp_path / 'ws'
    conv = new_conv(ws_dir, provider='claude', native=True, mode='tools', model='',
                                             task=dict(TASK), billing='subscription')
    conv.update(status='closed', session='session-1', ended=time.time())
    task = dict(TASK)
    if case == 'legacy':
        conv.update(provider='custom', native=False)
    elif case == 'no_session':
        conv.update(session=None)
    elif case == 'task':
        task.update(instructions='Something else', scope=['blocks.3'], title='renamed')
    elif case == 'workspace':
        conv.update(workspace='/somewhere/else/ws')
    if case == 'elsewhere':
        conv.update(status='running', pid=4242)
        sleeper = hold_elsewhere(conv)
    else:
        conv.release()
    sess = make_session(tmp_path / 'root', ws_dir, task=task)
    try:
        with make_app(sess, monkeypatch) as client:
            listed = client.get('/api/harness/conversations').json()['conversations'][0]
            assert listed['resume']['code'] == code and words in listed['resume']['reason']
            r = client.post('/api/harness/resume', json={'conversation': conv.id})
            assert r.status_code == 409 and r.json() == {'error': listed['resume']['reason'], 'refusal': code}
        assert sess.runs == {} and fake_bridge == []
    finally:
        if case == 'elsewhere':
            sleeper.kill(); sleeper.wait()
    if case == 'elsewhere':
        assert conversations.Conversation.open(str(ws_dir), conv.id).meta['status'] == 'running'   # its record left alone


def test_the_title_alone_changing_does_not_block_resume_and_the_sign_in_check_applies(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    conv = new_conv(ws_dir, provider='claude', native=True, mode='tools', model='', task=dict(TASK))
    conv.update(status='closed', session='session-1')
    conv.release()
    sess = make_session(tmp_path / 'root', ws_dir, task={**TASK, 'title': 'A new title'})
    with make_app(sess, monkeypatch, ready=False) as client:
        assert client.get('/api/harness/conversations').json()['conversations'][0]['resume']['ok']
        r = client.post('/api/harness/resume', json={'conversation': conv.id})
        assert r.status_code == 400 and 'Sign in' in r.json()['error']
    assert sess.runs == {} and fake_bridge == []


def test_a_per_call_resume_keeps_the_recorded_budget_and_a_subscription_one_drops_it(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    for method, expected in (('api_key', 2.5), ('claude.ai', 0)):
        conv = new_conv(ws_dir, provider='claude', native=True, mode='tools', model='',
                                                 task=dict(TASK), budget=2.5)
        conv.update(status='closed', session='session-1')
        conv.release()
        sess = make_session(tmp_path / 'root', ws_dir)
        with make_app(sess, monkeypatch, method=method) as client:
            run = sess.runs[client.post('/api/harness/resume', json={'conversation': conv.id}).json()['id']]
            wait_for(lambda: run.status == 'idle')
            start = next(e for e in run.events if e['kind'] == 'raw' and e['text'].startswith('START '))
            assert json.loads(start['text'][6:])['budget'] == expected
            resumed = next(e for e in run.events if e['kind'] == 'resumed')['text']
            assert ('budget starts again from zero' in resumed) == bool(expected)     # said where the budget is shown
            run.close()


def test_resuming_while_another_conversation_is_open_is_refused(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    old = new_conv(ws_dir, provider='claude', native=True, mode='tools', model='', task=dict(TASK))
    old.update(status='closed', session='session-1')
    old.release()
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        current = sess.runs[client.post('/api/harness/start', json={'harness': 'claude'}).json()['id']]
        wait_for(lambda: current.status == 'idle')
        listed = {c['id']: c for c in client.get('/api/harness/conversations').json()['conversations']}
        assert listed[old.id]['resume']['code'] == 'busy'
        r = client.post('/api/harness/resume', json={'conversation': old.id})
        assert r.status_code == 409 and r.json()['refusal'] == 'busy'
        assert len(sess.runs) == 1
        current.close()


def test_closing_during_a_turn_records_the_conversation_as_interrupted(tmp_path, fake_bridge, monkeypatch):
    sess = make_session(tmp_path / 'root', tmp_path / 'ws')
    run = harness.start(sess, 'http://x', 'claude', 'tools', billing='subscription')
    wait_for(lambda: run.status == 'idle')
    run.send('hang')
    assert run.status == 'running'
    run.close()
    meta = conversations.Conversation.open(str(tmp_path / 'ws'), run.conversation_id).meta
    assert meta['status'] == 'interrupted' and meta['interrupted'] == 'The conversation was closed while a turn was running.'
    run2 = harness.start(sess, 'http://x', 'claude', 'tools', billing='subscription')
    wait_for(lambda: run2.status == 'idle')
    run2.close()
    assert conversations.Conversation.open(str(tmp_path / 'ws'), run2.conversation_id).meta['status'] == 'closed'


def test_a_legacy_run_is_saved_and_can_be_read_but_not_resumed(tmp_path, monkeypatch):
    sess = make_session(tmp_path / 'root', tmp_path / 'ws')
    script = tmp_path / 'agent.py'
    script.write_text('import json\nprint(json.dumps({"type": "result", "result": "done", "total_cost_usd": 0.2, "num_turns": 1}))\n')
    run = harness.start(sess, 'http://x', 'custom', 'tools', custom=f'{sys.executable} {script}')
    wait_for(lambda: run.status == 'done')
    meta = conversations.Conversation.open(str(tmp_path / 'ws'), run.conversation_id).meta
    assert (meta['status'], meta['native'], meta['provider']) == ('done', False, 'custom')
    events = conversations.Conversation.open(str(tmp_path / 'ws'), run.conversation_id).events()
    assert events[-1]['kind'] == 'result' and events[-1]['cost_usd'] == 0.2
    assert native.resume_refusal(sess, meta, set())[0] == 'legacy'
    # The custom command line (it can carry tokens or private paths) is not saved; only the harness kind is.
    saved = (Path(conversations.root(str(tmp_path / 'ws'))) / run.conversation_id / 'meta.json').read_text()
    assert str(script) not in saved and 'command' not in json.loads(saved)
    assert run.meta()['command'].endswith(str(script))                          # the live run still shows it


# ------------------------------------------------------------------------------------------------ ordering
def test_a_failed_run_is_never_listed_as_open_in_the_moment_before_it_retires(tmp_path, fake_bridge, monkeypatch):
    """The failed status is final: the run no longer counts as live even before its thread has retired, and the
    last run event is published only after it retired."""
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        first = start_and_close(client, sess, monkeypatch)
    conversations.Conversation.open(str(ws_dir), first.conversation_id).update(session='session-gone')
    gate, reached = threading.Event(), threading.Event()
    original = native.NativeRun._kill_group
    def held_kill(self):
        reached.set()
        gate.wait(5)                     # the run has failed; its thread waits here, before it retires
        original(self)
    monkeypatch.setattr(native.NativeRun, '_kill_group', held_kill)
    sess2 = make_session(tmp_path / 'root', ws_dir)
    published = []
    with make_app(sess2, monkeypatch) as client:
        sess2.publish = lambda e: published.append((e, sess2.runs[e['run']['id']]._retired.is_set() if e.get('type') == 'run' else None))
        run = sess2.runs[client.post('/api/harness/resume', json={'conversation': first.conversation_id}).json()['id']]
        assert reached.wait(5) and run.status == 'failed' and not run._retired.is_set()
        assert run.conversation_id not in native.live_conversations(sess2)   # final: not live, even unretired
        # A listing in that window waits for the run to retire: it never shows the step in between (where the
        # retry check would say "busy" and Try again would be missing).
        answer = {}
        reader = threading.Thread(target=lambda: answer.setdefault('r', client.get('/api/harness/conversations').json()))
        reader.start()
        reader.join(0.5)
        assert reader.is_alive() and 'r' not in answer
        gate.set()
        reader.join(10)
        listed = answer['r']['conversations'][0]
        assert run._retired.is_set() and not listed['live']
        assert listed['resume']['code'] == 'gone' and listed['resume']['retry'] is True
        wait_for(lambda: run._retired.is_set() and published and published[-1][0].get('type') == 'run')
    last_run = [p for p in published if p[0].get('type') == 'run'][-1]
    assert last_run[1] is True and last_run[0]['run']['status'] == 'failed'


def test_a_legacy_run_finishing_is_never_recorded_as_interrupted(tmp_path, monkeypatch):
    """The final status is saved before the run stops counting as live; a listing in between sees it running."""
    sess = make_session(tmp_path / 'root', tmp_path / 'ws')
    gate, reached = threading.Event(), threading.Event()
    original = conversations.Conversation.update
    def slow_update(self, **fields):
        if fields.get('status') == 'done':
            reached.set()
            gate.wait(5)
        return original(self, **fields)
    monkeypatch.setattr(conversations.Conversation, 'update', slow_update)
    script = tmp_path / 'agent.py'
    script.write_text('print("hello")\n')
    run = harness.start(sess, 'http://x', 'custom', 'tools', custom=f'{sys.executable} {script}')
    assert reached.wait(5)
    live = native.live_conversations(sess)
    assert run.conversation_id in live and run.status == 'running'
    row = next(m for m in conversations.listing(str(tmp_path / 'ws'), live) if m['id'] == run.conversation_id)
    assert row['status'] == 'running'
    gate.set()
    wait_for(lambda: run.status == 'done')
    assert conversations.Conversation.open(str(tmp_path / 'ws'), run.conversation_id).meta['status'] == 'done'


# ------------------------------------------------------------------------------------------------ another IDE process
def test_mark_interrupted_is_refused_while_another_process_holds_the_conversation(tmp_path, fake_bridge, monkeypatch):
    """A held lock means another live process has the conversation open: the release is refused, naming the pid
    and what to do. Once that process is gone the same release works."""
    ws_dir = tmp_path / 'ws'
    conv = new_conv(ws_dir, provider='claude', native=True, mode='tools', model='', task=dict(TASK))
    conv.update(status='running', session='session-1', pid=4242)
    holder = hold_elsewhere(conv)
    sess = make_session(tmp_path / 'root', ws_dir)
    try:
        with make_app(sess, monkeypatch) as client:
            listed = client.get('/api/harness/conversations').json()['conversations'][0]
            assert listed['elsewhere'] and listed['owner_pid'] == 4242 and listed['holder'] == 'lock'
            assert listed['resume']['code'] == 'elsewhere' and 'release' not in listed['resume']
            assert 'Close it in that IDE window, or stop that process' in listed['resume']['reason']
            r = client.post('/api/harness/conversation/release', json={'conversation': conv.id, 'confirm': True})
            assert r.status_code == 409 and r.json()['refusal'] == 'elsewhere'
            assert 'pid 4242' in r.json()['error'] and 'Close the conversation in that IDE window' in r.json()['error']
            assert conversations.Conversation.open(str(ws_dir), conv.id).meta['status'] == 'running'   # nothing written
            assert client.post('/api/harness/resume', json={'conversation': conv.id}).json()['refusal'] == 'elsewhere'
    finally:
        holder.kill(); holder.wait()
    # The holder is gone: the release works (on a record that still says running).
    with make_app(sess, monkeypatch) as client:
        for unconfirmed in ({'conversation': conv.id}, {'conversation': conv.id, 'confirm': 'yes'}):
            r = client.post('/api/harness/conversation/release', json=unconfirmed)
            assert r.status_code == 400 and 'Confirm' in r.json()['error']
        assert conversations.Conversation.open(str(ws_dir), conv.id).meta['status'] == 'running'   # nothing written
        r = client.post('/api/harness/conversation/release', json={'conversation': conv.id, 'confirm': True})
        assert r.status_code == 200, r.text
        row = r.json()['conversation']
        assert row['status'] == 'interrupted' and row['resume']['ok']
        run = sess.runs[client.post('/api/harness/resume', json={'conversation': conv.id}).json()['id']]
        wait_for(lambda: run.status == 'idle')
        saved = conversations.Conversation.open(str(ws_dir), conv.id).meta
        assert saved['pid'] == os.getpid() and saved['status'] == 'idle' and conversations.held_here(conv.path)
        run.close()
    assert not conversations.held_here(conv.path)


def test_without_flock_the_tab_is_told_and_mark_interrupted_is_the_remedy(tmp_path, fake_bridge, monkeypatch):
    import errno
    monkeypatch.setattr(conversations.fcntl, 'flock', lambda fd, op: (_ for _ in ()).throw(OSError(errno.ENOTSUP, 'Operation not supported')))
    ws_dir = tmp_path / 'ws'
    sleeper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        conv = new_conv(ws_dir, provider='claude', native=True, mode='tools', model='', task=dict(TASK))
        conv.update(status='running', session='session-1', pid=sleeper.pid, boot='other-process')
        conv.close()
        sess = make_session(tmp_path / 'root', ws_dir)
        with make_app(sess, monkeypatch) as client:
            body = client.get('/api/harness/conversations').json()
            assert 'does not support file locks' in body['locking']
            row = body['conversations'][0]
            assert row['holder'] == 'pid' and row['resume']['release'] is True
            r = client.post('/api/harness/conversation/release', json={'conversation': conv.id, 'confirm': True})
            assert r.status_code == 200 and r.json()['conversation']['resume']['ok']
            run = sess.runs[client.post('/api/harness/resume', json={'conversation': conv.id}).json()['id']]
            wait_for(lambda: run.status == 'idle')
            run.close()
            started = client.post('/api/harness/start', json={'harness': 'claude'}).json()
            run2 = sess.runs[started['id']]
            assert run2.conversation_id                      # Start still saves without flock
            wait_for(lambda: run2.status == 'idle')
            run2.close()
    finally:
        sleeper.kill(); sleeper.wait()
        conversations._UNLOCKABLE.clear()


def test_a_resume_that_fails_to_start_gives_the_record_back(tmp_path, fake_bridge, monkeypatch):
    """A failure between taking the lock and the run owning it (a full disk at the claim's write, a thread that
    cannot start) releases the lock: the next Resume in this process works."""
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        first = start_and_close(client, sess, monkeypatch)
    sess2 = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess2, monkeypatch) as client:
        original_update = conversations.Conversation.update
        def full_disk(self, **fields):
            if 'pid' in fields:
                raise OSError(28, 'No space left on device')
            return original_update(self, **fields)
        monkeypatch.setattr(conversations.Conversation, 'update', full_disk)
        r = client.post('/api/harness/resume', json={'conversation': first.conversation_id})
        assert r.status_code == 400 and 'No space left' in r.json()['error']
        monkeypatch.setattr(conversations.Conversation, 'update', original_update)
        assert not conversations.held_here(first.conversation.path) and sess2.runs == {}
        original_start = native.NativeRun.start
        monkeypatch.setattr(native.NativeRun, 'start', lambda self: (_ for _ in ()).throw(RuntimeError("can't start new thread")))
        r = client.post('/api/harness/resume', json={'conversation': first.conversation_id})
        assert r.status_code == 400 and "can't start new thread" in r.json()['error']
        assert not conversations.held_here(first.conversation.path) and sess2.runs == {}
        conv = conversations.Conversation.open(str(ws_dir), first.conversation_id)
        with pytest.raises(RuntimeError, match="can't start new thread"):
            harness.resume(sess2, 'http://127.0.0.1:1/mcp', conv, '', 'per_call')
        assert not conversations.held_here(conv) and sess2.runs == {}     # given back by resume itself, record still open
        conv.close()
        r = client.post('/api/harness/start', json={'harness': 'claude'})    # a new conversation whose run cannot start
        assert r.status_code == 400 and "can't start new thread" in r.json()['error'] and sess2.runs == {}
        newest = conversations.listing(str(ws_dir), set())[0]
        assert newest['id'] != first.conversation_id
        assert not conversations.held_here(os.path.join(conversations.root(str(ws_dir)), newest['id']))
        monkeypatch.setattr(native.NativeRun, 'start', original_start)
        r = client.post('/api/harness/resume', json={'conversation': first.conversation_id})
        assert r.status_code == 200, r.text
        run = sess2.runs[r.json()['id']]
        wait_for(lambda: run.status == 'idle')
        run.close()


def test_a_legacy_run_gives_its_record_back_when_it_ends_or_cannot_start(tmp_path, monkeypatch):
    sess = make_session(tmp_path / 'root', tmp_path / 'ws')
    script = tmp_path / 'agent.py'
    script.write_text('print("hello")\n')
    run = harness.start(sess, 'http://x', 'custom', 'tools', custom=f'{sys.executable} {script}')
    wait_for(lambda: run.status == 'done')
    path = os.path.join(conversations.root(str(tmp_path / 'ws')), run.conversation_id)
    assert not conversations.held_here(path)
    class NoThread:
        def __init__(self, *a, **k):
            pass
        def start(self):
            raise RuntimeError("can't start new thread")
    monkeypatch.setattr(harness, 'threading', SimpleNamespace(Thread=NoThread, Lock=threading.Lock))
    with pytest.raises(RuntimeError):
        harness.start(sess, 'http://x', 'custom', 'tools', custom=f'{sys.executable} {script}')
    newest = conversations.listing(str(tmp_path / 'ws'), set())[0]
    assert newest['id'] != run.conversation_id
    assert not conversations.held_here(os.path.join(conversations.root(str(tmp_path / 'ws')), newest['id']))
    assert all(r.status != 'starting' for r in sess.runs.values())


def test_a_conversation_this_process_runs_cannot_be_marked_interrupted(tmp_path, fake_bridge, monkeypatch):
    sess = make_session(tmp_path / 'root', tmp_path / 'ws')
    with make_app(sess, monkeypatch) as client:
        run = sess.runs[client.post('/api/harness/start', json={'harness': 'claude'}).json()['id']]
        wait_for(lambda: run.status == 'idle')
        r = client.post('/api/harness/conversation/release', json={'conversation': run.conversation_id, 'confirm': True})
        assert r.status_code == 409 and 'close it instead' in r.json()['error']
        assert 'refusal' not in r.json()                   # not "another process": this one runs it
        assert conversations.Conversation.open(str(tmp_path / 'ws'), run.conversation_id).meta['status'] == 'idle'
        run.close()


def test_resume_refuses_at_the_click_when_another_process_took_the_conversation(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    conv = new_conv(ws_dir, provider='claude', native=True, mode='tools', model='', task=dict(TASK))
    conv.update(status='closed', session='session-1')
    sess = make_session(tmp_path / 'root', ws_dir)
    holder = None
    original = conversations.settle
    def settle_then_lose_the_race(c, live):
        nonlocal holder
        meta = original(c, live)
        holder = holder or hold_elsewhere(conv)       # another IDE process claims it right after the check
        return meta
    with make_app(sess, monkeypatch) as client:
        monkeypatch.setattr(conversations, 'settle', settle_then_lose_the_race)
        try:
            r = client.post('/api/harness/resume', json={'conversation': conv.id})
            assert r.status_code == 409 and r.json()['refusal'] == 'elsewhere'
            assert sess.runs == {} and fake_bridge == []
        finally:
            if holder:
                holder.kill(); holder.wait()


def test_a_conversation_id_cannot_name_a_path_outside_the_conversations_directory(tmp_path):
    base = tmp_path / 'ws' / 'conversations'
    real = 'conv-20260101-000000-abcdef12'
    (base / real).mkdir(parents=True)                          # so that a valid id followed by ../ resolves
    outside = tmp_path / 'ws' / 'evil'
    outside.mkdir()
    traversal = real + '/../../evil'                           # a valid conv-... prefix, then a way out
    for cid in ('../evil', traversal):
        (outside / 'meta.json').write_text(json.dumps({'id': cid, 'provider': 'claude', 'native': True, 'session': 's'}))
        assert os.path.exists(os.path.join(base, cid, 'meta.json'))   # the file such an id would name is there
        with pytest.raises(ValueError, match='Unknown conversation'):
            conversations.Conversation.open(str(tmp_path / 'ws'), cid)
    for cid in ('/etc', 'conv-1', '', None, real + '\n'):
        with pytest.raises(ValueError, match='Unknown conversation'):
            conversations.Conversation.open(str(tmp_path / 'ws'), cid)


def test_every_way_a_run_ends_gives_the_record_back(tmp_path, fake_bridge, monkeypatch):
    """The reviewer's probe E3, as a test: normal close, gone, a failed attempt, a dying bridge, a wrong session."""
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        first = start_and_close(client, sess, monkeypatch)
        path = first.conversation.path
        assert not conversations.held_here(path)
        for sid, gone in (('session-gone', True), ('session-flaky', False), ('session-crash', False), ('session-wrong', False)):
            conversations.Conversation.open(str(ws_dir), first.conversation_id).update(session=sid, resume_error=None)
            r = client.post('/api/harness/resume', json={'conversation': first.conversation_id})
            assert r.status_code == 200, (sid, r.text)
            run = sess.runs[r.json()['id']]
            wait_for(lambda: run._retired.is_set(), timeout=8)
            meta = conversations.Conversation.open(str(ws_dir), first.conversation_id).meta
            assert not conversations.held_here(path), sid
            assert bool(meta['resume_error']) is gone and (gone or meta['last_resume_failure']), sid
