"""The provider session id is saved the moment the provider gives it, so a server stop during a conversation's first
turn leaves a record that can be resumed; records saved before that change get the session their own events confirm.

A real record after eight minutes of a first turn: ``init`` with a session id in events.jsonl, while meta.json
still said ``session: null, n_events: 0`` (the id was written only at the next status change, the turn's end). After a
server stop such a record was refused as ``no_session``. No provider, CLI, model or network call is involved: a fake
bridge (a few lines of Python speaking the bridge protocol) stands in for the provider.
"""
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wsp import conversations, native_harness as native  # noqa: E402
from test_programmer_conversations import FIXTURE, TASK, fake_bridge, hold_elsewhere, make_app, make_session, new_conv, wait_for  # noqa: E402,F401

# A first turn that never ends, in the order the Claude bridge reports one: the adapter starts (idle), the brief is
# sent (running, user), then the turn's system/init with the session id, then a tool call that is still running.
FIRST_TURN = r'''
import json, sys
def emit(x): print(json.dumps(x), flush=True)
for line in sys.stdin:
    m = json.loads(line); p = m.get('params', {})
    if m['method'] == 'start':
        emit({'status': 'idle'})
        emit({'status': 'running'})
        emit({'event': {'kind': 'user', 'text': p['brief']}})
        emit({'event': {'kind': 'init', 'session': 'session-first', 'model': 'm', 'api_key_source': 'none'}})
        emit({'event': {'kind': 'thinking_delta', 'id': 'a:0', 'text': 'Looking '}})
        emit({'event': {'kind': 'thinking_delta', 'id': 'a:0', 'text': 'at L1.'}})
        emit({'event': {'kind': 'tool_use', 'id': 't1', 'name': 'mcp__wsp__inspect', 'input': {'address': 'L1N1'}}})
    elif m['method'] == 'close':
        break
    if 'id' in m: emit({'id': m['id'], 'result': {}})
'''

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
    self.cmd = [sys.executable, '-u', '-c', os.environ['FIRST_TURN']]
    original(self)
native.NativeRun.start = start
agent = os.path.join(ws.dir, 'agent'); os.makedirs(agent, exist_ok=True)
run = native.start(sess, {'dir': agent, 'server': {}, 'brief': 'brief'}, 'claude', 'tools', '', 0, 'subscription')
while not any(e.get('kind') == 'tool_use' for e in run.events): time.sleep(.01)
print('READY', run.conversation_id, run.status, flush=True)
time.sleep(60)
'''
T = {'title': 'T', 'instructions': 'I', 'scope': ['model'], 'suite': None, 'preserve': None}


def saved_meta(ws_dir, cid):
    return json.loads((Path(ws_dir) / 'conversations' / cid / 'meta.json').read_text())


def test_a_server_stopped_during_the_first_turn_leaves_a_record_that_lists_with_its_session_and_can_be_resumed(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    env = {**os.environ, 'WSP_ROOT': str(ROOT), 'WS_DIR': str(ws_dir), 'FIRST_TURN': FIRST_TURN}
    child = subprocess.Popen([sys.executable, '-u', '-c', CHILD], env=env, stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        line = child.stdout.readline().split()
        assert line[0] == 'READY', line
        cid, status = line[1], line[2]
        assert status == 'running'
        # Mid first turn, before any status change after init: the record already names the session.
        meta = saved_meta(ws_dir, cid)
        assert meta['session'] == 'session-first'
        assert meta['status'] == 'running'                           # the write at init did not touch the status
        assert meta['n_events'] == 2 and meta['lines'] == 2          # user and init, as saved at that moment
        assert meta['billing'] == 'subscription'
    finally:
        os.killpg(child.pid, signal.SIGKILL)                         # the server and its bridge die at once
        child.wait(5)
    sess = make_session(tmp_path / 'root', ws_dir, task=T)
    with make_app(sess, monkeypatch) as client:
        c = client.get('/api/harness/conversations').json()['conversations'][0]
        assert c['id'] == cid and c['session'] == 'session-first' and c['status'] == 'interrupted'
        assert c['resume'] == {'ok': True, 'code': None, 'reason': None}
        conv = conversations.Conversation.open(str(ws_dir), cid)
        try:
            assert native.resume_refusal(sess, conversations.settle(conv, set()), set()) is None
        finally:
            conv.close()


def test_the_session_is_written_at_init_without_waiting_for_the_turn_to_end(tmp_path, monkeypatch):
    """The same in this process, through the start route and the live run: after init, before the turn ends."""
    monkeypatch.setattr(native, 'setup_error', lambda provider: None)
    original = native.NativeRun.start

    def start(self):
        self.cmd = [sys.executable, '-u', '-c', FIRST_TURN]
        original(self)
    monkeypatch.setattr(native.NativeRun, 'start', start)
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        run = sess.runs[client.post('/api/harness/start', json={'harness': 'claude', 'mode': 'tools'}).json()['id']]
        wait_for(lambda: any(e.get('kind') == 'tool_use' for e in run.events))
        meta = saved_meta(ws_dir, run.conversation_id)
        assert (meta['session'], meta['status'], meta['n_events']) == ('session-first', 'running', 2)
        run.close()


def his_record(ws_dir, session='session-his', status='running', extra_events=()):
    """A record shaped like the real one: a first turn eight minutes in, meta.json without the session."""
    conv = new_conv(ws_dir, provider='claude', mode='tools', model='', run='run-old', native=True, task=dict(TASK),
                    billing='subscription')
    conv.update(status=status)
    for e in [{'kind': 'user', 'i': 0, 't': 1.0, 'text': 'BRIEF'},
              {'kind': 'init', 'i': 1, 't': 2.0, 'session': session, 'model': 'm'},
              {'kind': 'thinking_delta', 'i': 2, 't': 3.0, 'id': 'a:0', 'text': 'Look'},
              {'kind': 'thinking_delta', 'i': 3, 't': 3.5, 'id': 'a:0', 'text': 'ing.'},
              {'kind': 'tool_use', 'i': 4, 't': 4.0, 'id': 't1', 'name': 'mcp__wsp__inspect', 'input': {}},
              *extra_events]:
        conv.append(e)
    conv.meta = conversations.coerce({**conv.meta, 'session': None, 'n_events': 0, 'lines': 0}, conv.id)
    conv._write_meta()
    conv.close()                                                     # its process is gone: nobody holds it
    assert saved_meta(ws_dir, conv.id)['session'] is None
    return conv.id


def test_a_record_saved_without_its_session_lists_with_the_session_of_its_init(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    cid = his_record(ws_dir)
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        c = client.get('/api/harness/conversations').json()['conversations'][0]
        assert (c['id'], c['session'], c['status'], c['resume']['ok']) == (cid, 'session-his', 'interrupted', True)
        assert c['n_events'] == 4                                    # recounted from the saved events (pieces joined)
        read = client.get('/api/harness/conversation', params={'id': cid}).json()
        assert read['resume']['ok'] and read['meta']['session'] == 'session-his'
    saved = saved_meta(ws_dir, cid)
    assert saved['status'] == 'interrupted' and saved['session'] == 'session-his'   # settled with it, as it was cut off


def test_a_settled_record_without_its_session_is_resumed_with_the_session_of_its_init(tmp_path, fake_bridge, monkeypatch):
    """Already settled by an earlier build (interrupted, session null): listed and resumed with the init's session,
    which the first listing writes into the record (its own events confirm it), so it is looked up once."""
    ws_dir = tmp_path / 'ws'
    cid = his_record(ws_dir, status='interrupted')
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        c = client.get('/api/harness/conversations').json()['conversations'][0]
        assert c['session'] == 'session-his' and c['resume']['ok']
        saved = saved_meta(ws_dir, cid)
        assert (saved['session'], saved['status']) == ('session-his', 'interrupted')   # written once; status as it was
        r = client.post('/api/harness/resume', json={'conversation': cid})
        assert r.status_code == 200, r.text
        run = sess.runs[r.json()['id']]
        wait_for(lambda: run.status == 'idle')
        start = next(e for e in run.events if e.get('kind') == 'raw' and e['text'].startswith('START '))
        assert json.loads(start['text'][6:])['resume'] == 'session-his'
        assert saved_meta(ws_dir, cid)['session'] == 'session-his'   # written by the confirmed resume
        run.close()


@pytest.mark.parametrize('session,outcome', [('session-flaky', 'failed attempt'), ('session-wrong', 'another session')])
def test_an_unconfirmed_resume_of_such_a_record_leaves_it_alone_and_never_makes_another_session_resumable(tmp_path, fake_bridge, monkeypatch, session, outcome):
    ws_dir = tmp_path / 'ws'
    cid = his_record(ws_dir, session=session, status='interrupted')
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        client.get('/api/harness/conversations')                     # the listing records the session of its init
        before = saved_meta(ws_dir, cid)
        assert before['session'] == session
        r = client.post('/api/harness/resume', json={'conversation': cid})
        assert r.status_code == 200, r.text
        run = sess.runs[r.json()['id']]
        wait_for(lambda: run._retired.is_set())
        after = saved_meta(ws_dir, cid)
        assert (after['session'], after['status']) == (before['session'], before['status'])   # as they were before the attempt
        assert after['last_resume_failure']
        conv = conversations.Conversation.open(str(ws_dir), cid)
        try:
            events = conv.events()
        finally:
            conv.close()
        assert [e['kind'] for e in events].count('resume_failed') == 1
        assert events[-1]['kind'] == 'resume_failed' and events[-1]['session'] == session
        assert not any(e.get('session') == 'session-other' for e in events if e['kind'] == 'init')
        c = client.get('/api/harness/conversations').json()['conversations'][0]
        assert c['session'] == session and c['resume']['ok'], c        # still the conversation's own session, retryable


E = lambda kind, **kw: {'kind': kind, **kw}


@pytest.mark.parametrize('events,expected', [
    ([E('user'), E('init', session='S1'), E('text'), E('tool_use')], 'S1'),                       # the observed shape
    ([E('init', session='S1'), E('result'), E('user'), E('init', session='S1')], 'S1'),            # one per turn
    ([E('init', session='S1'), E('result'), E('user'), E('init', session='S2')], 'S2'),            # the last, as the run itself would have recorded it
    ([E('init', session='S2'), E('resume_failed', text='x')], None),                              # failed at that init
    ([E('init', session='S2'), E('raw', text='notice'), E('resume_failed')], None),               # notices aside
    ([E('init', session='S1'), E('result'), E('resumed', session='S1'), E('init', session='S2'), E('resume_failed')], 'S1'),
    ([E('init', session='S1'), E('result'), E('resumed', session='S1'), E('user'), E('init', session='S2')], 'S1'),   # not the resumed one
    ([E('init', session='S1'), E('resumed', session='S1'), E('user'), E('init', session='S1'), E('result')], 'S1'),
    ([E('init', session='S1'), E('resume_failed', session='S1')], 'S1'),                          # a failed retry of S1 itself
    ([E('init', session='S1'), E('tool_use'), E('resume_failed')], 'S1'),                         # a later failed attempt
    ([E('resume_failed', session='S9', text='S9')], None),                                         # never from a failure
    ([E('init', session=7), E('init', session=''), E('init', session='x' * 300), E('init')], None),
    ([], None),
    # a `resumed` marker without a usable session expects none in particular (an earlier marker's expectation ends)
    ([E('init', session='S1'), E('resumed', session='S1'), E('init', session='S1'), E('resumed'), E('init', session='S2')], 'S2'),
    ([E('init', session='S1'), E('resumed', session='S1'), E('init', session='S1'), E('resumed', session='--x'), E('init', session='S2')], 'S2'),
    # only a session id of the accepted form (conversations.SESSION_ID)
    ([E('init', session=' ')], None),
    ([E('init', session='--help')], None),
    ([E('init', session='-S1')], None),
    ([E('init', session='a b')], None),
    ([E('init', session='S1\n')], None),
    ([E('init', session='x' * 129)], None),
    ([E('init', session='S1'), E('result'), E('init', session='--resume')], 'S1'),
    ([E('init', session='x' * 128)], 'x' * 128),
    ([E('init', session='0199a213-81c0-7800-8aa1-bbab2a035a53')], '0199a213-81c0-7800-8aa1-bbab2a035a53'),
    ([E('init', session='thread_1.a:b')], 'thread_1.a:b'),
])
def test_which_init_is_trusted(events, expected):
    assert conversations.session_from_events(events) == expected


def test_no_session_is_taken_for_a_legacy_run_or_from_an_untrusted_init(tmp_path, monkeypatch, fake_bridge):
    ws_dir = tmp_path / 'ws'
    legacy = new_conv(ws_dir, provider='custom', mode='tools', run='r', native=False, task=dict(TASK))
    legacy.append({'kind': 'init', 'session': 'S-legacy'})
    legacy.update(status='done')
    legacy.close()
    bad = his_record(ws_dir, session='session-bad', status='interrupted')
    conv = conversations.Conversation.open(str(ws_dir), bad)
    conv.claim()
    conv.append({'kind': 'resume_failed', 'text': 'an older build saved the wrong init before this'})
    conv.close()
    # Rewrite so that the bad init is followed directly by the failure (an older or damaged record).
    path = Path(ws_dir) / 'conversations' / bad / 'events.jsonl'
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    path.write_text('\n'.join(json.dumps(e) for e in lines if e['kind'] in ('user', 'init', 'resume_failed')) + '\n')
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        rows = {c['id']: c for c in client.get('/api/harness/conversations').json()['conversations']}
        assert rows[legacy.id]['session'] is None and rows[legacy.id]['resume']['code'] == 'legacy'
        assert rows[bad]['session'] is None and rows[bad]['resume']['code'] == 'no_session'



# --------------------------------------------------------------------------------- read once, written once
def count_event_reads(monkeypatch):
    """How many times the store opens a conversation's events.jsonl."""
    opened = []
    original = conversations._open

    def counting(name, flags, dir_fd):
        if name == 'events.jsonl':
            opened.append(name)
        return original(name, flags, dir_fd)
    monkeypatch.setattr(conversations, '_open', counting)
    return opened


@pytest.mark.parametrize('status,session', [('interrupted', 'session-his'), ('closed', 'session-his'), ('running', 'session-his'),
                                            ('interrupted', None)])
def test_the_events_are_read_for_a_session_once_per_record_not_at_every_listing(tmp_path, monkeypatch, status, session):
    """A record saved without its session id: the first listing reads its events and writes what they confirm (the
    session, or that they confirm none); later listings, the transcript and a resume check do not read them for it."""
    ws_dir = tmp_path / 'ws'
    if session:
        cid = his_record(ws_dir, session=session, status=status)
    else:
        cid = his_record(ws_dir, status=status)
        path = Path(ws_dir) / 'conversations' / cid / 'events.jsonl'
        path.write_text('\n'.join(l for l in path.read_text().splitlines() if '"init"' not in l) + '\n')
    opened = count_event_reads(monkeypatch)
    first = conversations.listing(str(ws_dir), set())[0]
    assert opened, 'the first listing read the events'
    saved = saved_meta(ws_dir, cid)
    if session:
        assert first['session'] == saved['session'] == session
    else:
        assert first['session'] is None and saved['session'] is None and saved['session_scanned'] is True
    assert saved['status'] == ('interrupted' if status == 'running' else status)
    opened.clear()
    for _ in range(3):
        again = conversations.listing(str(ws_dir), set())[0]
        assert again['session'] == first['session'] and again['status'] == first['status']
    assert opened == [], 'a second listing does not read the events file'
    conv = conversations.Conversation.open(str(ws_dir), cid)
    try:
        assert conversations.settle(conv, set())['session'] == first['session']
    finally:
        conv.close()
    assert opened == []


def test_a_final_record_another_process_holds_is_not_written(tmp_path, monkeypatch):
    ws_dir = tmp_path / 'ws'
    cid = his_record(ws_dir, status='interrupted')
    conv = conversations.Conversation.open(str(ws_dir), cid)
    holder = hold_elsewhere(conv)
    try:
        before = (Path(ws_dir) / 'conversations' / cid / 'meta.json').read_bytes()
        row = conversations.listing(str(ws_dir), set())[0]
        assert row['session'] == 'session-his'                       # answered from its events, in memory only
        assert (Path(ws_dir) / 'conversations' / cid / 'meta.json').read_bytes() == before
    finally:
        holder.kill(); holder.wait(5); conv.close()
    conversations.listing(str(ws_dir), set())
    assert saved_meta(ws_dir, cid)['session'] == 'session-his'       # written once nobody holds it


def test_a_failure_to_read_the_events_leaves_the_record_and_its_meta_as_they_are(tmp_path, monkeypatch):
    ws_dir = tmp_path / 'ws'
    cid = his_record(ws_dir, status='interrupted')
    before = (Path(ws_dir) / 'conversations' / cid / 'meta.json').read_bytes()

    def broken(self):
        raise OSError(errno.EIO, 'Input/output error')
    monkeypatch.setattr(conversations.Conversation, 'events', broken)
    conv = conversations.Conversation.open(str(ws_dir), cid)
    try:
        meta = dict(conv.meta)
        assert conversations._with_session(conv, meta) == meta
        assert conversations.settle(conv, set()) == meta
    finally:
        conv.close()
    row = conversations.listing(str(ws_dir), set())[0]
    assert not row.get('unreadable') and row['session'] is None
    assert (Path(ws_dir) / 'conversations' / cid / 'meta.json').read_bytes() == before, 'not marked as looked at'
    monkeypatch.undo()
    assert conversations.listing(str(ws_dir), set())[0]['session'] == 'session-his'   # read again, and found, later


def test_marking_a_conversation_interrupted_answers_with_the_same_verdict_as_the_listing(tmp_path, fake_bridge, monkeypatch):
    ws_dir = tmp_path / 'ws'
    cid = his_record(ws_dir, status='running')                       # cut off, its record without the session
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        r = client.post('/api/harness/conversation/release', json={'conversation': cid, 'confirm': True})
        assert r.status_code == 200, r.text
        answer = r.json()['conversation']
        assert (answer['status'], answer['session'], answer['resume']['ok']) == ('interrupted', 'session-his', True), answer
        listed = client.get('/api/harness/conversations').json()['conversations'][0]
        assert listed['resume'] == answer['resume'] and listed['session'] == answer['session']
    assert saved_meta(ws_dir, cid)['session'] == 'session-his'


# --------------------------------------------------------------------------------- the write at init
SLOW_TURN = r"""
import json, sys, time
def emit(x): print(json.dumps(x), flush=True)
for line in sys.stdin:
    m = json.loads(line); p = m.get('params', {})
    if m['method'] == 'start':
        emit({'status': 'idle'}); emit({'status': 'running'})
        emit({'event': {'kind': 'user', 'text': p['brief']}})
        emit({'event': {'kind': 'init', 'session': 'session-first', 'model': 'm', 'api_key_source': 'none'}})
        emit({'event': {'kind': 'tool_use', 'id': 't1', 'name': 'mcp__wsp__inspect', 'input': {}}})
        time.sleep(0.5)
        emit({'event': {'kind': 'tool_result', 'id': 't1', 'text': 'ok'}})
        emit({'event': {'kind': 'result', 'ok': True}})
        emit({'status': 'idle'})
    elif m['method'] == 'send':
        emit({'status': 'running'}); emit({'event': {'kind': 'user', 'text': p['text']}})
        emit({'event': {'kind': 'init', 'session': 'session-first', 'model': 'm', 'api_key_source': p['text']}})
        emit({'event': {'kind': 'tool_use', 'id': 't2', 'name': 'mcp__wsp__inspect', 'input': {}}})
    elif m['method'] == 'close':
        break
    if 'id' in m: emit({'id': m['id'], 'result': {}})
"""


def slow_bridge(monkeypatch, script=SLOW_TURN):
    monkeypatch.setattr(native, 'setup_error', lambda provider: None)
    original = native.NativeRun.start

    def start(self):
        self.cmd = [sys.executable, '-u', '-c', script]
        original(self)
    monkeypatch.setattr(native.NativeRun, 'start', start)


@pytest.mark.parametrize('which', ['the write at init', 'every write from init on'])
def test_a_full_disk_at_init_does_not_end_the_run(tmp_path, monkeypatch, which):
    """The write at init goes through the run's _save: a failure is one notice in the log, and the turn goes on."""
    slow_bridge(monkeypatch)
    full = {'on': False}
    update, append = conversations.Conversation.update, conversations.Conversation.append

    def failing_update(self, **fields):
        at_init = 'session' in fields and 'status' not in fields
        if at_init:
            full['on'] = True
        if full['on'] and (at_init or which == 'every write from init on'):
            raise OSError(errno.ENOSPC, 'No space left on device')
        return update(self, **fields)

    def failing_append(self, event, sync=False):
        if event.get('kind') == 'init' and which == 'every write from init on':
            full['on'] = True
        if full['on'] and which == 'every write from init on':
            raise OSError(errno.ENOSPC, 'No space left on device')
        return append(self, event, sync)
    monkeypatch.setattr(conversations.Conversation, 'update', failing_update)
    monkeypatch.setattr(conversations.Conversation, 'append', failing_append)
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        run = sess.runs[client.post('/api/harness/start', json={'harness': 'claude', 'mode': 'tools'}).json()['id']]
        wait_for(lambda: run.status == 'idle' and any(e.get('kind') == 'result' for e in run.events), timeout=10)
        kinds = [e['kind'] for e in run.events]
        assert kinds.index('init') < kinds.index('tool_result') < kinds.index('result'), kinds
        notices = [e for e in run.events if e.get('kind') == 'raw' and 'Could not save this conversation' in e.get('text', '')]
        assert len(notices) == 1, notices
        assert 'No space left on device' in notices[0]['text']
        full['on'] = False
        run.close()
        wait_for(lambda: run._retired.is_set(), timeout=10)


def test_a_later_turns_init_saves_the_billing_when_it_changed(tmp_path, monkeypatch):
    """The session is the same, but the provider now says it uses an API key: billed per call. Saved at that init,
    as the session is, not at the turn's end."""
    slow_bridge(monkeypatch)
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        run = sess.runs[client.post('/api/harness/start', json={'harness': 'claude', 'mode': 'tools'}).json()['id']]
        wait_for(lambda: run.status == 'idle' and any(e.get('kind') == 'result' for e in run.events), timeout=10)
        assert saved_meta(ws_dir, run.conversation_id)['billing'] == 'subscription'
        assert client.post('/api/harness/send', json={'id': run.id, 'text': 'ANTHROPIC_API_KEY'}).status_code == 200
        wait_for(lambda: [e['kind'] for e in run.events].count('tool_use') == 2)
        wait_for(lambda: saved_meta(ws_dir, run.conversation_id)['billing'] == 'per_call')
        meta = saved_meta(ws_dir, run.conversation_id)
        assert (meta['session'], meta['status']) == ('session-first', 'running')   # mid-turn: the status untouched
        run.close()


# --------------------------------------------------------------------------------- the form of a session id
@pytest.mark.parametrize('value,kept', [
    ('session-1', True), ('0199a213-81c0-7800-8aa1-bbab2a035a53', True), ('S1', True), ('a.b_c:d-e', True), ('x' * 128, True),
    ('', False), (' ', False), ('--resume', False), ('-x', False), ('.hidden', False), ('a b', False), ('S1\n', False),
    ('x' * 129, False), ('sess/../x', False), ('séance', False), (7, False), (None, False),
])
def test_meta_json_keeps_only_a_session_id_of_the_accepted_form(value, kept):
    meta = conversations.coerce({'id': 'conv-x', 'session': value}, 'conv-x')
    assert meta['session'] == (value if kept else None)


def test_a_session_id_of_another_form_from_the_provider_is_never_saved_or_resumed(tmp_path, monkeypatch):
    slow_bridge(monkeypatch, SLOW_TURN.replace("'session': 'session-first'", "'session': '--dangerous'"))
    ws_dir = tmp_path / 'ws'
    sess = make_session(tmp_path / 'root', ws_dir)
    with make_app(sess, monkeypatch) as client:
        run = sess.runs[client.post('/api/harness/start', json={'harness': 'claude', 'mode': 'tools'}).json()['id']]
        wait_for(lambda: run.status == 'idle' and any(e.get('kind') == 'result' for e in run.events), timeout=10)
        assert saved_meta(ws_dir, run.conversation_id)['session'] is None
        run.close()
        wait_for(lambda: run._retired.is_set(), timeout=10)
        c = client.get('/api/harness/conversations').json()['conversations'][0]
        assert c['session'] is None and c['resume']['code'] == 'no_session'
        r = client.post('/api/harness/resume', json={'conversation': c['id']})
        assert r.status_code == 409 and r.json()['refusal'] == 'no_session'


def test_resume_itself_takes_the_session_from_the_events_when_the_record_has_none(tmp_path, fake_bridge):
    """The route settles the record first (which writes the session); native_harness.resume does not rely on it."""
    ws_dir = tmp_path / 'ws'
    cid = his_record(ws_dir, status='interrupted')
    sess = make_session(tmp_path / 'root', ws_dir)
    conv = conversations.Conversation.open(str(ws_dir), cid)
    assert conv.meta['session'] is None
    agent = ws_dir / 'agent'
    agent.mkdir(exist_ok=True)
    run = native.resume(sess, {'dir': str(agent), 'server': {}, 'brief': 'BRIEF'}, conv, '', 'subscription')
    try:
        wait_for(lambda: run.status == 'idle')
        start = next(e for e in run.events if e.get('kind') == 'raw' and e['text'].startswith('START '))
        assert json.loads(start['text'][6:])['resume'] == 'session-his'
    finally:
        run.close()
        wait_for(lambda: run._retired.is_set())
