"""The saved-conversation store does not trust its own files (wsp/conversations.py).

Links, other owners' entries and damaged records are refused one entry at a time: a bad entry is one
"unreadable" row and never stops the listing, Start or Resume. Ownership is an flock held by the IDE process
that runs the conversation; a record another live process holds is never written. No provider, bridge or
model is involved: a few child Python processes stand in for another IDE process.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wsp import conversations, native_harness as native

TASK = {'title': 'Paris to Rome', 'instructions': 'Make the model say Rome', 'scope': ['model'], 'suite': None, 'preserve': None}

HOLD = r'''
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR)
fcntl.flock(fd, fcntl.LOCK_EX)
print('HELD', flush=True)
sys.stdin.read()
'''


@pytest.fixture
def ws(tmp_path):
    d = tmp_path / 'ws'
    d.mkdir()
    return d


def make(ws, release=True, **fields):
    conv = conversations.Conversation.create(str(ws), provider='claude', native=True, mode='tools', model='',
                                             task=dict(TASK), **fields)
    if release:
        conv.release()
    return conv


class Holder:
    """Another IDE process holding a conversation's owner lock, until it is stopped."""

    def __init__(self, conv):
        self.proc = subprocess.Popen([sys.executable, '-c', HOLD, str(Path(conv.path) / 'owner.lock')],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert self.proc.stdout.readline().strip() == 'HELD'

    def stop(self):
        self.proc.stdin.close()
        self.proc.wait(5)


def rows(ws, live=frozenset()):
    return conversations.listing(str(ws), set(live))


def sess_for(ws, task=TASK):
    return SimpleNamespace(ws=SimpleNamespace(dir=str(ws), patches=[]), task=dict(task), runs={}, lock=threading.RLock())


# ------------------------------------------------------------------------------------------------ links and owners
def test_a_linked_store_root_is_refused_and_nothing_is_written_through_it(ws, tmp_path):
    target = tmp_path / 'elsewhere'
    target.mkdir()
    os.chmod(target, 0o755)
    os.symlink(target, ws / 'conversations')
    with pytest.raises(conversations.Unreadable, match='not a directory of this user'):
        conversations.Conversation.create(str(ws), provider='claude', native=True)
    assert os.listdir(target) == [] and (os.stat(target).st_mode & 0o777) == 0o755        # never chmod-ed through
    assert [r['unreadable'] for r in rows(ws)] == ['conversations is not a directory of this user (a link or another owner\'s entry)']
    # Start goes ahead without a record and says so.
    conv, problem = native.new_conversation(sess_for(ws), 'claude', 'tools', '', 'run-1', True)
    assert conv is None and 'is not being saved' in problem and 'not a directory of this user' in problem


def test_a_linked_conversation_directory_is_one_unreadable_row_and_is_never_pruned_through(ws, tmp_path, monkeypatch):
    good = make(ws)
    good.update(status='closed', session='s-1')
    victim = tmp_path / 'victim'
    victim.mkdir()
    (victim / 'precious').write_text('keep me')
    (victim / 'meta.json').write_text(json.dumps({'id': 'conv-20000101-000000-deadbeef', 'provider': 'claude'}))
    os.symlink(victim, Path(conversations.root(str(ws))) / 'conv-20000101-000000-deadbeef')
    listed = rows(ws)
    bad = [r for r in listed if r.get('unreadable')]
    assert [r['id'] for r in bad] == ['conv-20000101-000000-deadbeef'] and 'a link' in bad[0]['unreadable']
    assert [r['id'] for r in listed if not r.get('unreadable')] == [good.id]
    monkeypatch.setattr(conversations, 'KEEP', 1)
    for _ in range(3):
        make(ws).update(status='closed')
    assert (victim / 'precious').read_text() == 'keep me'
    with pytest.raises(conversations.Unreadable):
        conversations.Conversation.open(str(ws), 'conv-20000101-000000-deadbeef')


def test_linked_files_inside_a_conversation_are_refused(ws, tmp_path):
    conv = make(ws)
    target = tmp_path / 'target.txt'
    target.write_text('original')
    events = Path(conv.path) / 'events.jsonl'
    events.unlink()
    os.symlink(target, events)
    with pytest.raises(conversations.Unreadable, match='is a link'):   # O_NOFOLLOW: never a write to the target
        conv.append({'kind': 'text', 'text': 'x'})
    assert target.read_text() == 'original'
    meta = Path(conv.path) / 'meta.json'
    real = meta.read_text()
    meta.unlink()
    (tmp_path / 'other-meta.json').write_text(real)
    os.symlink(tmp_path / 'other-meta.json', meta)
    assert rows(ws)[0]['unreadable'] == 'meta.json is a link'
    # An update replaces the link itself (rename), never the file it points to.
    conv.update(status='closed')
    assert not os.path.islink(meta) and json.loads((tmp_path / 'other-meta.json').read_text()) == json.loads(real)


def test_a_fifo_planted_in_a_conversation_never_blocks_and_is_refused(ws):
    conv = make(ws)
    meta = Path(conv.path) / 'meta.json'
    meta.unlink()
    os.mkfifo(meta, 0o600)
    row = rows(ws)[0]                                       # returns at once: the open does not wait for a writer
    assert 'not a regular file' in row['unreadable']
    events = Path(conv.path) / 'events.jsonl'
    events.unlink()
    os.mkfifo(events, 0o600)
    with pytest.raises(conversations.Unreadable, match='not a regular file'):
        conv.append({'kind': 'text', 'text': 'x'})


# ------------------------------------------------------------------------------------------------ damaged records
def test_every_meta_field_is_checked_and_coerced(ws):
    conv = make(ws)
    raw = {'id': conv.id, 'status': 'exploded', 'started': 'yesterday', 'ended': float('nan'), 'session': 7,
           'n_events': -4, 'lines': True, 'provider': ['claude'], 'native': 'yes', 'mode': None, 'model': 12,
           'task': {'title': 3, 'instructions': 'Do it', 'scope': ['a', 5], 'suite': 9, 'preserve': {'max_kl': 'x', 'k': 1}},
           'workspace': {}, 'billing': 'free', 'budget': 'lots', 'pid': -1, 'boot': 5, 'resumes': [{'at': 'x', 'run': 2}, 'bad'],
           'truncated': 1, 'owner_override': 'true', 'resume_error': ['gone'], 'extra': None}
    (Path(conv.path) / 'meta.json').write_text(json.dumps(raw))
    m = conversations.Conversation.open(str(ws), conv.id).meta
    assert m['status'] == 'interrupted' and m['started'] is None and m['ended'] is None and m['session'] is None
    assert (m['n_events'], m['lines'], m['provider'], m['native'], m['mode'], m['model']) == (0, 0, 'unknown', False, 'tools', '')
    assert m['task'] == {'instructions': 'Do it', 'scope': ['a'], 'suite': None, 'preserve': {'k': 1}}
    assert (m['workspace'], m['billing'], m['budget'], m['pid'], m['boot']) == (None, None, None, None, None)
    assert m['resumes'] == [{'at': None, 'run': None}] and m['truncated'] is False and m['owner_override'] is False
    assert m['resume_error'] is None and m['extra'] == ''
    # A coerced legacy-shaped record is refused for the reason its coerced fields give, not by an exception.
    assert native.resume_refusal(sess_for(ws), m, set())[0] == 'legacy'
    # Integers past what a browser reads exactly (2**53) are dropped, as are numbers too big for a float.
    big = conversations.coerce({'id': conv.id, 'n_events': 2 ** 60, 'lines': 10 ** 400, 'pid': 10 ** 30,
                                'started': 10 ** 400, 'budget': 1e308 * 10}, conv.id)
    assert (big['n_events'], big['lines'], big['pid'], big['started'], big['budget']) == (0, 0, None, None, None)


@pytest.mark.parametrize('damage,words', [
    (b'{not json', 'not valid JSON'), (b'[1, 2]', 'does not describe'), (b'{"id": "conv-19990101-000000-00000000"}', 'does not describe'),
    (None, 'meta.json is missing'), (b'{"id": "X", "pad": "' + b'y' * (1024 * 1024) + b'"}', 'larger than 1 MiB')])
def test_a_damaged_record_is_one_row_and_never_stops_listing_start_or_resume(ws, damage, words, monkeypatch):
    good = make(ws)
    good.update(status='closed', session='s-1')
    bad = make(ws)
    meta = Path(bad.path) / 'meta.json'
    if damage is None:
        meta.unlink()
    else:
        meta.write_bytes(damage)
    listed = rows(ws)
    assert [r['id'] for r in listed if r.get('unreadable')] == [bad.id] and words in next(r for r in listed if r.get('unreadable'))['unreadable']
    assert [r['id'] for r in listed if not r.get('unreadable')] == [good.id]
    assert conversations.summary(next(r for r in listed if r.get('unreadable'))) == {'id': bad.id, 'unreadable': next(r for r in listed if r.get('unreadable'))['unreadable']}
    # Start still makes a record (pruning skips what it cannot read), and the good one is still resumable.
    monkeypatch.setattr(native, 'setup_error', lambda provider: None)
    conv, problem = native.new_conversation(sess_for(ws), 'claude', 'tools', '', 'run-1', True)
    assert conv is not None and problem is None
    conv.release()
    assert native.resume_refusal(sess_for(ws), conversations.Conversation.open(str(ws), good.id).meta, set()) is None


def test_a_torn_last_line_never_swallows_the_next_event(ws):
    conv = make(ws)
    conv.append({'kind': 'text', 'text': 'before'})
    with open(Path(conv.path) / 'events.jsonl', 'ab') as f:
        f.write(b'{"kind": "text", "text": "cut off by a cra')          # a crash in the middle of a write
    reopened = conversations.Conversation.open(str(ws), conv.id)
    reopened.append({'kind': 'text', 'text': 'after'})
    reopened.append({'kind': 'text', 'text': 'later'})
    assert [e['text'] for e in reopened.events()] == ['before', 'after', 'later']   # only the fragment is dropped
    raw = (Path(conv.path) / 'events.jsonl').read_bytes().splitlines()
    assert raw[1] == b'{"kind": "text", "text": "cut off by a cra' and len(raw) == 4


def test_the_listed_count_is_the_number_of_events_the_transcript_shows(ws):
    conv = make(ws)
    for piece in ('Hel', 'lo', ' there'):
        conv.append({'kind': 'text_delta', 'id': 'm1:0', 'text': piece})
    conv.append({'kind': 'tool_use', 'id': 'a', 'name': 'overview', 'input': {}})
    for piece in ('a', 'b'):
        conv.append({'kind': 'thinking_delta', 'id': 'm1:1', 'text': piece})
    conv.update(status='closed')
    listed = rows(ws)[0]
    shown = conversations.Conversation.open(str(ws), conv.id).events()
    assert listed['n_events'] == len(shown) == 3 and listed['lines'] == 6


@pytest.mark.parametrize('status,expected,words', [
    ('starting', 'interrupted', 'while a turn was running'), ('waiting', 'interrupted', 'while a turn was running'),
    ('closing', 'interrupted', 'while the conversation was closing'), ('idle', 'closed', 'waited for your message')])
def test_a_record_cut_off_by_a_server_stop_is_settled_once_and_recorded(ws, status, expected, words):
    conv = make(ws)
    conv.update(status=status)
    row = rows(ws)[0]
    assert row['status'] == expected and words in (row.get('interrupted') or row.get('closed_reason'))
    assert conversations.Conversation.open(str(ws), conv.id).meta['status'] == expected


# ------------------------------------------------------------------------------------------------ ownership
def test_claiming_records_this_process_and_a_record_held_elsewhere_is_never_written(ws):
    conv = make(ws, release=False)
    assert (conv.meta['pid'], conv.meta['boot']) == (os.getpid(), conversations.BOOT)
    conv.update(status='running', pid=99999, boot='another-boot')
    conv.release()
    holder = Holder(conv)
    try:
        before = (Path(conv.path) / 'meta.json').read_bytes()
        listed = rows(ws)[0]
        assert listed['elsewhere'] and listed['status'] == 'running'
        assert conversations.summary(listed)['owner_pid'] == 99999
        assert (Path(conv.path) / 'meta.json').read_bytes() == before                # settle left it alone
        with pytest.raises(conversations.Busy):
            conversations.Conversation.open(str(ws), conv.id).claim()
        refusal = native.resume_refusal(sess_for(ws), listed, set())
        assert refusal[0] == 'elsewhere' and 'pid 99999' in refusal[1]
    finally:
        holder.stop()
    # Its process is gone, so the kernel dropped the lock: the record is settled as cut off.
    assert rows(ws)[0]['status'] == 'interrupted'


def test_mark_interrupted_is_refused_while_another_process_holds_the_conversation(ws, monkeypatch):
    """A held flock means a live process has the conversation open: "it is not running" is checked, not taken on
    trust. Once that process is gone, a record that still says running can be marked."""
    monkeypatch.setattr(native, 'setup_error', lambda provider: None)
    conv = make(ws)
    conv.update(status='running', session='s-1', pid=99999)
    holder = Holder(conv)
    try:
        before = (Path(conv.path) / 'meta.json').read_bytes()
        with pytest.raises(conversations.Busy, match=r'pid 99999.*Close the conversation in that IDE window, or stop that process'):
            conversations.mark_interrupted(conversations.Conversation.open(str(ws), conv.id))
        assert (Path(conv.path) / 'meta.json').read_bytes() == before
        listed = rows(ws)[0]
        assert listed['holder'] == 'lock' and native.resume_refusal(sess_for(ws), listed, set())[0] == 'elsewhere'
    finally:
        holder.stop()
    opened = conversations.Conversation.open(str(ws), conv.id)
    meta = conversations.mark_interrupted(opened)
    assert meta['status'] == 'interrupted' and 'pid 99999' in meta['interrupted'] and not meta['owner_override']
    assert native.resume_refusal(sess_for(ws), rows(ws)[0], set()) is None


def test_one_process_cannot_hold_a_record_twice(ws):
    conv = make(ws, release=False)
    with pytest.raises(conversations.Busy, match='still finishing'):
        conversations.Conversation.open(str(ws), conv.id).claim()
    conv.claim()                                               # the holder itself: no-op
    for record in (conv, conversations.Conversation.open(str(ws), conv.id)):
        with pytest.raises(conversations.Busy, match='This IDE process is running this conversation; close it instead'):
            conversations.mark_interrupted(record)            # held here: close it, never mark it interrupted
    conv.release()
    other = conversations.Conversation.open(str(ws), conv.id)
    other.claim()
    assert conversations.held_here(other.path)
    conv.release()                                             # a former holder's release does not drop it
    assert conversations.held_here(other.path)
    other.release()


def test_pruning_never_deletes_a_conversation_that_is_running(ws, monkeypatch):
    oldest_live = make(ws)
    oldest_live.update(status='running', started=1)
    oldest_held = make(ws, release=False)
    oldest_held.update(status='running', started=2)
    oldest_elsewhere = make(ws)
    oldest_elsewhere.update(status='running', started=3)
    holder = Holder(oldest_elsewhere)
    try:
        doomed = make(ws)
        doomed.update(status='closed', started=4)
        monkeypatch.setattr(conversations, 'KEEP', 2)
        for k in range(3):
            make(ws, live_ids={oldest_live.id}).update(status='closed', started=10 + k)
        left = {r['id'] for r in rows(ws, {oldest_live.id})}
        assert {oldest_live.id, oldest_held.id, oldest_elsewhere.id} <= left and doomed.id not in left
    finally:
        holder.stop()
        oldest_held.release()


# ------------------------------------------------------------------------------------------------ task changes
@pytest.mark.parametrize('field,value', [('preserve', {'max_kl': 0.05}), ('suite', 'capitals'), ('scope', ['blocks.3']),
                                         ('instructions', 'Something else')])
def test_each_task_field_but_the_title_blocks_resume(ws, monkeypatch, field, value):
    monkeypatch.setattr(native, 'setup_error', lambda provider: None)
    conv = make(ws)
    conv.update(status='closed', session='s-1', workspace=os.path.realpath(ws))
    meta = conversations.Conversation.open(str(ws), conv.id).meta
    assert native.resume_refusal(sess_for(ws, {**TASK, 'title': 'renamed'}), meta, set()) is None
    refusal = native.resume_refusal(sess_for(ws, {**TASK, field: value}), meta, set())
    assert refusal[0] == 'task' and f'({field})' in refusal[1]


# ------------------------------------------------------------------------------------------------ round 2: damaged entries
def start_saves(ws):
    conv, problem = native.new_conversation(sess_for(ws), 'claude', 'tools', '', 'run-x', True)
    assert conv is not None and problem is None, problem
    conv.close()


@pytest.mark.parametrize('case', ['status_list', 'status_dict', 'deep_meta', 'deep_event'])
def test_an_entry_that_breaks_parsing_is_one_row_and_never_stops_listing_open_or_start(ws, case):
    """The reviewer's probe S1, as a test: unhashable fields and nesting too deep for the parser."""
    good = make(ws)
    good.update(status='closed', session='s')
    bad = make(ws)
    meta = Path(bad.path) / 'meta.json'
    d = json.loads(meta.read_text())
    if case in ('status_list', 'status_dict'):
        d['status'] = [] if case == 'status_list' else {'a': 1}
        meta.write_text(json.dumps(d))
    elif case == 'deep_meta':
        meta.write_text('[' * 200000)
    else:
        bad.update(status='running')                           # unsettled, so listing reads its events
        (Path(bad.path) / 'events.jsonl').write_text('[' * 30000 + ']' * 30000 + '\n' + json.dumps({'kind': 'text', 'text': 'kept'}) + '\n')
    listed = {r['id']: r for r in rows(ws)}
    assert listed[good.id]['status'] == 'closed'
    if case == 'deep_meta':
        assert listed[bad.id]['unreadable'] == 'meta.json is nested too deeply'
    else:
        assert listed[bad.id]['status'] == 'interrupted'       # coerced (a list status) or settled (running)
        events = conversations.Conversation.open(str(ws), bad.id).events()
        if case == 'deep_event':
            assert [e['text'] for e in events] == ['kept']     # the too-deep line is skipped, the next one kept
    start_saves(ws)



@pytest.mark.parametrize('where', ['read', 'settle'])
def test_an_unexpected_error_in_one_entry_is_one_row_and_never_stops_listing_open_or_start(ws, monkeypatch, where):
    """Every exception, not only the expected kinds, is caught per entry (reading its record, or settling it)."""
    good = make(ws)
    good.update(status='closed', session='s')
    bad = make(ws)
    bad.update(status='running')
    if where == 'read':
        real = conversations.coerce
        monkeypatch.setattr(conversations, 'coerce', lambda raw, cid: (_ for _ in ()).throw(RuntimeError('surprise'))
                            if cid == bad.id else real(raw, cid))
    else:
        real = conversations.Conversation.events
        monkeypatch.setattr(conversations.Conversation, 'events', lambda self: (_ for _ in ()).throw(RuntimeError('surprise'))
                            if self.id == bad.id else real(self))
    listed = {r['id']: r for r in rows(ws)}
    assert listed[good.id]['status'] == 'closed'
    assert 'surprise' in listed[bad.id]['unreadable']
    if where == 'read':
        with pytest.raises(conversations.Unreadable, match='surprise'):
            conversations.Conversation.open(str(ws), bad.id)
    start_saves(ws)
    monkeypatch.setattr(conversations, 'KEEP', 1)
    start_saves(ws)                                            # pruning past it still saves
    monkeypatch.setattr(conversations, 'prune', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('prune broke')))
    start_saves(ws)                                            # even a pruning failure never stops saving


def test_a_directory_of_another_owner_is_refused(ws, monkeypatch):
    """Simulated: this test cannot make a directory owned by another user, so this process's uid is changed."""
    conv = make(ws)
    real = os.getuid()
    monkeypatch.setattr(conversations.os, 'getuid', lambda: real + 1)
    with pytest.raises(conversations.Unreadable, match='not a directory of this user'):
        conversations.Conversation.open(str(ws), conv.id)
    listed = rows(ws)
    assert len(listed) == 1 and 'not a directory of this user' in listed[0]['unreadable']

def test_an_event_line_over_the_bound_is_skipped_without_being_parsed(ws, monkeypatch):
    conv = make(ws)
    conv.append({'kind': 'text', 'text': 'before'})
    with open(Path(conv.path) / 'events.jsonl', 'ab') as f:
        f.write(b'"' + b'x' * (conversations.MAX_EVENT_BYTES * 3) + b'"\n')
    conv.append({'kind': 'text', 'text': 'after'})
    parsed = []
    original = json.loads
    monkeypatch.setattr(conversations.json, 'loads', lambda b, *a, **k: parsed.append(len(b)) or original(b, *a, **k))
    assert [e['text'] for e in conversations.Conversation.open(str(ws), conv.id).events()] == ['before', 'after']
    assert max(parsed) <= conversations.MAX_EVENT_BYTES                # the long line never reached the parser



def test_the_line_reader_never_holds_a_long_line_and_stops_at_its_total(tmp_path):
    import tracemalloc
    path = tmp_path / 'lines'
    path.write_bytes(b'a\n' + b'x' * (4 << 20) + b'\nb\n' + b'y' * (4 << 20))   # 4 MiB lines, the last unterminated
    fd = os.open(path, os.O_RDONLY)
    try:
        tracemalloc.start()
        got = list(conversations._lines(fd, limit=100, total=10 << 20))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        os.close(fd)
    assert got == [b'a', None, b'b', None]
    assert peak < 1 << 20, peak                                     # never the whole 4 MiB line in memory
    path.write_bytes(b''.join(b'%04d\n' % i for i in range(1000)))   # 5 bytes a line
    fd = os.open(path, os.O_RDONLY)
    try:
        got = list(conversations._lines(fd, limit=100, total=50))
    finally:
        os.close(fd)
    assert got == [b'%04d' % i for i in range(10)]                  # reads at most `total` bytes

@pytest.mark.parametrize('kind', ['fifo', 'dir', 'symlink'])
def test_a_bad_lock_file_skips_its_entry_and_never_stops_saving(ws, tmp_path, monkeypatch, kind):
    """Probes S3 and S4: a FIFO, directory or link named owner.lock, in an entry old enough to be pruned."""
    monkeypatch.setattr(conversations, 'KEEP', 3)
    old = make(ws)
    old.update(status='running', started=1)
    lock = Path(old.path) / 'owner.lock'
    lock.unlink()
    {'fifo': lambda: os.mkfifo(lock), 'dir': lock.mkdir, 'symlink': lambda: os.symlink(tmp_path / 'nowhere', lock)}[kind]()
    for _ in range(3):
        make(ws).update(status='closed')
    for _ in range(2):
        start_saves(ws)                                         # saved every time; the bad entry is skipped
    assert old.path in [str(Path(conversations.root(str(ws))) / r['id']) for r in rows(ws)]
    assert len(rows(ws)) == 4                                   # and the others past the limit were still pruned
    row = next(r for r in rows(ws) if r['id'] == old.id)
    assert 'owner.lock' in row['unreadable']
    with pytest.raises(conversations.Unreadable):
        conversations.Conversation.open(str(ws), old.id).claim()


def test_a_planted_gitignore_link_is_never_written_through(ws, tmp_path):
    (ws / 'conversations').mkdir()
    target = tmp_path / 'target'
    target.write_text('KEEP\n')
    os.symlink(target, ws / 'conversations' / '.gitignore')
    make(ws)
    assert target.read_text() == 'KEEP\n'


FUZZ_META = [
    b'', b'{', b'null', b'[]', b'"text"', b'12', b'\xff\xfe\x00binary\x00', b'{"id": "', '{"id": "é"}'.encode(),
    b'[' * 10000 + b']' * 10000, b'{"a":' * 10000 + b'1' + b'}' * 10000,
]
FIELDS = ['status', 'started', 'ended', 'session', 'n_events', 'lines', 'provider', 'native', 'mode', 'model', 'task',
          'workspace', 'billing', 'budget', 'pid', 'boot', 'run', 'extra', 'resume_error', 'resume_error_at',
          'last_resume_failure', 'last_resume_failure_at', 'interrupted', 'closed_reason', 'resumes', 'truncated',
          'owner_override']
WRONG = [None, True, -1, 10 ** 400, 1e308, [], [1, [2]], {}, {'x': [1]}, 'x' * 300000, [[[]]]]
FUZZ_EVENTS = [b'', b'\n\n\n', b'{', b'\xff\xfe binary \x00\n', b'[]\n', b'{"kind": 5}\n', b'{"text": "no kind"}\n',
               b'[' * 20000 + b']' * 20000 + b'\n', b'{"kind": "text_delta", "id": [], "text": {}}\n',
               b'{"kind": "text", "text": "ok"}', b'{"kind": "text"' + b' ' * 200000 + b'}\n', b'1e999999\n']


def test_fuzz_malformed_records_never_stop_listing_open_start_or_resume(ws, monkeypatch):
    """A few dozen malformed meta.json and events.jsonl shapes: listing, opening, Start and the resume check
    of a good conversation all keep working, and every bad entry is one row."""
    monkeypatch.setattr(native, 'setup_error', lambda provider: None)
    monkeypatch.setattr(conversations, 'KEEP', 10_000)              # keep them all for the check
    good = make(ws)
    good.update(status='closed', session='s-1')
    shapes = []
    for raw in FUZZ_META:
        shapes.append(('meta', raw))
    for field in FIELDS:
        for value in WRONG:
            shapes.append(('field', (field, value)))
    for raw in FUZZ_EVENTS:
        shapes.append(('events', raw))
    made, readable = [], set()
    for what, data in shapes:
        c = make(ws)
        if what == 'meta':
            (Path(c.path) / 'meta.json').write_bytes(data)
        elif what == 'field':
            d = json.loads((Path(c.path) / 'meta.json').read_text())
            d[data[0]] = data[1]
            try:
                text = json.dumps(d)
            except (ValueError, OverflowError):
                continue
            (Path(c.path) / 'meta.json').write_text(text)
            readable.add(c.id)                                      # a wrong field is coerced, never a reason to refuse
        else:
            c.update(status='running')
            (Path(c.path) / 'events.jsonl').write_bytes(data)
        made.append(c.id)
    assert len(made) > 300
    listed = {r['id']: r for r in rows(ws)}
    assert set(made) | {good.id} == set(listed)
    for cid in made:
        row = listed[cid]
        assert row.get('unreadable') or row['status'] in conversations.STATUSES
        if cid in readable:
            assert not row.get('unreadable'), row
        summary = conversations.summary(row)
        json.dumps(summary)                                           # every row can be sent to the page
        if not row.get('unreadable'):
            conv = conversations.Conversation.open(str(ws), cid)
            json.dumps(conv.events())
            native.resume_refusal(sess_for(ws), row, set())             # never raises
            conv.close()
    start_saves(ws)
    assert native.resume_refusal(sess_for(ws), listed[good.id], set()) is None
    opened = conversations.Conversation.open(str(ws), good.id)
    opened.claim()                                                    # what Resume does first
    opened.close()
    monkeypatch.setattr(conversations, 'KEEP', 1)
    start_saves(ws)                                                   # pruning through all of them still saves


# ------------------------------------------------------------------------------------------------ round 2: links
def test_a_hard_link_to_a_file_elsewhere_is_refused(ws, tmp_path):
    """Probe S2: an events.jsonl or meta.json hard-linked to an outside file is never written."""
    conv = make(ws)
    outside = tmp_path / 'outside.txt'
    outside.write_text('ORIG\n')
    ev = Path(conv.path) / 'events.jsonl'
    ev.unlink()
    os.link(outside, ev)
    with pytest.raises(conversations.Unreadable, match='another link'):
        conversations.Conversation.open(str(ws), conv.id).append({'kind': 'text', 'text': 'via hard link'})
    assert outside.read_text() == 'ORIG\n'
    meta = Path(conv.path) / 'meta.json'
    outside_meta = tmp_path / 'outside-meta.json'
    os.link(meta, outside_meta)
    assert 'another link' in rows(ws)[0]['unreadable']


def test_a_directory_swapped_for_a_link_after_opening_never_redirects_a_write(ws, tmp_path):
    conv = make(ws)
    opened = conversations.Conversation.open(str(ws), conv.id)
    other = tmp_path / 'other'
    other.mkdir()
    (other / 'events.jsonl').write_text('')
    moved = conv.path + '.moved'
    os.rename(conv.path, moved)
    os.symlink(other, conv.path)
    opened.append({'kind': 'text', 'text': 'after swap'})
    opened.update(status='closed')
    assert (other / 'events.jsonl').read_text() == '' and not (other / 'meta.json').exists()
    assert 'after swap' in (Path(moved) / 'events.jsonl').read_text()          # it went to the directory it opened
    # And a fresh open of the swapped name is refused.
    assert 'not a directory of this user' in next(r for r in rows(ws) if r['id'] == conv.id)['unreadable']


# ------------------------------------------------------------------------------------------------ round 2: ownership
def test_a_failed_claim_gives_the_lock_back(ws, monkeypatch):
    """Probe S7: the record write after taking the lock fails (a full disk): the lock is released."""
    conv = make(ws)
    opened = conversations.Conversation.open(str(ws), conv.id)
    original = conversations.Conversation.update
    monkeypatch.setattr(conversations.Conversation, 'update', lambda self, **f: (_ for _ in ()).throw(OSError(28, 'No space left on device')))
    with pytest.raises(OSError, match='No space left'):
        opened.claim()
    monkeypatch.setattr(conversations.Conversation, 'update', original)
    assert not conversations.held_here(conv.path) and not conversations.locked_elsewhere(conv.path)
    again = conversations.Conversation.open(str(ws), conv.id)
    again.claim()
    assert conversations.held_here(conv.path)
    again.close()


def test_a_settle_in_another_thread_makes_a_claim_wait_not_fail(ws):
    """Probe S6: while a listing's settle holds the lock on its own descriptor, a claim in this process waits for
    it instead of reporting "another IDE process"."""
    conv = make(ws)
    conv.update(status='running')                               # so a listing settles it (and takes the lock)
    entered, done = threading.Event(), threading.Event()
    original = conversations.Conversation.update

    def slow_update(self, **fields):
        if fields.get('status') == 'interrupted':
            entered.set()
            time.sleep(0.3)                                     # the settle holds the lock meanwhile
        return original(self, **fields)
    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    mp.setattr(conversations.Conversation, 'update', slow_update)
    t = threading.Thread(target=lambda: (rows(ws), done.set()))
    t.start()
    assert entered.wait(5)
    mp.undo()
    claimer = conversations.Conversation.open(str(ws), conv.id)
    claimer.claim()                                             # waits for the settle, then takes the lock
    assert done.is_set() and conversations.held_here(conv.path)
    t.join()
    claimer.close()


@pytest.fixture
def no_flock(monkeypatch):
    """A filesystem without flock: every flock call fails with ENOTSUP (probe F1)."""
    import errno
    import fcntl as real
    def nolock(fd, op):
        raise OSError(errno.ENOTSUP, 'Operation not supported')
    monkeypatch.setattr(conversations.fcntl, 'flock', nolock)
    yield
    conversations._UNLOCKABLE.clear()


def test_without_flock_the_pid_check_decides_and_mark_interrupted_is_the_remedy(ws, no_flock, monkeypatch):
    monkeypatch.setattr(native, 'setup_error', lambda provider: None)
    idle = make(ws)
    idle.update(status='closed', session='s-1')
    sleeper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        held = make(ws)
        held.update(status='running', session='s-2', pid=sleeper.pid, boot='other')
        dead = make(ws)
        dead.update(status='running', session='s-3', pid=2 ** 22 + 12345, boot='gone')
        reused = make(ws)
        reused.update(status='running', session='s-4', pid=os.getpid(), boot='an-earlier-process')
        listed = {r['id']: r for r in rows(ws)}
        assert not any(r.get('unreadable') for r in listed.values())
        assert listed[reused.id]['status'] == 'interrupted'             # this pid, another process before: not held
        assert listed[idle.id]['status'] == 'closed'
        assert listed[held.id]['elsewhere'] and listed[held.id]['holder'] == 'pid'
        assert listed[dead.id]['status'] == 'interrupted'               # its recorded process is gone
        assert 'does not support file locks' in conversations.locking_note(str(ws))
        refusal = native.resume_refusal(sess_for(ws), listed[held.id], set())
        assert refusal[0] == 'elsewhere' and 'no file locks' in refusal[1]
        with pytest.raises(conversations.Busy):
            conversations.Conversation.open(str(ws), held.id).claim()
        start_saves(ws)                                                  # Start still saves
        monkeypatch.setattr(conversations, 'KEEP', 1)
        start_saves(ws)                                                  # pruning hard: the pid-held one is kept
        assert held.id in {r['id'] for r in rows(ws)}
        meta = conversations.mark_interrupted(conversations.Conversation.open(str(ws), held.id))
        assert meta['status'] == 'interrupted' and meta['owner_override']
        again = conversations.Conversation.open(str(ws), held.id)
        again.claim()                                                    # what Resume does next
        assert again.meta['pid'] == os.getpid()
        again.close()
    finally:
        sleeper.kill(); sleeper.wait()


def test_with_flock_there_is_no_locking_note(ws):
    make(ws)
    rows(ws)
    assert conversations.locking_note(str(ws)) is None
