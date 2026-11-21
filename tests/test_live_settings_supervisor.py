"""The supervisor's settings action, through its real mailbox. Model-free: an inert server child, synthetic memory."""
from contextlib import contextmanager
import copy
import json
import os
import signal
import threading
import time

import pytest

import test_runtime_supervisor as T
from wsp import runtime_supervisor as rs
from wsp.memory_guard import GiB
from wsp.runtime_events import RuntimeRefusal
from wsp.runtime_supervisor import RuntimeClient, descriptor, write
from wsp.settings import SettingsOutOfStep, SettingsStore
from test_runtime_supervisor import fake_ws, wait

# Memory settings come from the user's configuration, as on a normal start; only the backend is a launch override.
CONFIG = dict(version=1, defaults=dict(floor_gib=20, preferred_gib=25, resident_seconds=30), profiles={})


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    original = T.supervisor_script
    def script(path, *, inert=False):
        out = original(path, inert=inert)
        text = out.read_text()
        launch = "s=resolve(env=dict(HOME=os.environ['HOME']),overrides=dict(backend='cpu',floor_gib=20,preferred_gib=25,resident_seconds=30))"
        assert launch in text
        out.write_text(text.replace(launch, "s=resolve(os.environ['HOME']+'/config.json',env=dict(HOME=os.environ['HOME']),overrides=dict(backend='cpu'))"))
        return out
    monkeypatch.setattr(T, 'supervisor_script', script)
    (tmp_path/'config.json').write_text(json.dumps(CONFIG))
    with T.inert_supervisor(tmp_path) as running:
        yield running


def rows(folder):
    return [json.loads(line) for line in (folder/'watchdog.jsonl').read_text().splitlines()]


def running_settings(folder):
    return json.loads((folder/'settings.json').read_text())


def change(request, folder, n, changes=None, *, values=None, restore=False, check=False, deadline=None, sources=None, confirmed=None):
    """confirmed=True: confirm the effect a check of the same change judges now (its fingerprint); a string is sent as is."""
    if confirmed is True:
        confirmed = (change(request, folder, f'{n}-check', changes, values=values, check=True, sources=sources)['effect'] or {}).get('judged')
    s = running_settings(folder)
    body = dict(values=values if values is not None else dict(s['values'], **(changes or {})), sources=sources or s['sources'],
                notices=s['notices'], restore=restore, check=check, confirmed=confirmed,
                deadline=time.time()+5 if deadline is None else deadline)
    reply = request(n, body, action='settings')
    assert reply['ok'], reply                                   # a settings answer is never a runtime refusal
    return reply['settings']


@contextmanager
def gpu_telemetry(folder):
    # A loaded runner needs fresh allocation telemetry, which a real model worker writes.
    stop = threading.Event()
    def beat():
        while not stop.is_set():
            write(folder/'watchdog.jsonl.gpu.json', dict(time=time.time(), active=True, backend='cpu'));stop.wait(.3)
    beat_thread = threading.Thread(target=beat, daemon=True); beat_thread.start()
    try:yield
    finally:stop.set();beat_thread.join(2)


WORKSPACES = {}


def cold(tmp_path, total, context=8192):
    # One fake model file per test: writing it again would change its identity (stat).
    ws = WORKSPACES.get(tmp_path) or WORKSPACES.setdefault(tmp_path, fake_ws(tmp_path))
    value = descriptor(ws, context, backend='cpu')
    value['forecast'] = dict(value['forecast'], total=total)
    return value


def test_the_owners_case_lower_reserve_then_the_same_load_is_admitted(tmp_path, supervisor):
    proc, folder, request = supervisor
    value = cold(tmp_path, 50*GiB)                              # 71 GiB available - 25 GiB reserve = 46 GiB budget
    refused = request('cold', value)
    assert not refused['ok'] and refused['event']['code'] == 'forecast_exceeds_budget'
    answer = change(request, folder, 'lower', dict(preferred_gib=20, floor_gib=15))
    assert answer['accepted'] and answer['changed'] == ['floor_gib', 'preferred_gib'] and answer['effect'] is None
    assert running_settings(folder)['values']['preferred_gib'] == 20
    changed = [r for r in rows(folder) if r['event'] == 'settings_changed']
    assert len(changed) == 1 and changed[0]['old'] == dict(floor_gib=20, preferred_gib=25) and changed[0]['new'] == dict(floor_gib=15, preferred_gib=20)
    again = request('again', value)                             # the same request, no restart
    assert again['ok'] and again['reserved']
    assert json.loads((folder/'status.json').read_text())['policy']['preferred'] == 20*GiB
    assert request('release', action='release')['ok'] and proc.poll() is None


def test_refused_settings_change_nothing_anywhere(tmp_path, supervisor):
    proc, folder, request = supervisor
    before = (folder/'settings.json').read_bytes(); status = json.loads((folder/'status.json').read_text())
    s = running_settings(folder)
    # Only the source changes, so nothing but the override rule can refuse it.
    source = change(request, folder, 'source', sources=dict(s['sources'], backend='config defaults'))
    assert not source['accepted'] and 'launch override' in source['event']['message'], source
    cases = [
        ('order', dict(floor_gib=30), 'settings_invalid', 'floor_gib <= preferred_gib'),
        ('negative', dict(resident_seconds=-1), 'settings_invalid', 'resident_seconds'),
        ('total', dict(preferred_gib=100, floor_gib=20, budget_gib=40, ceiling_gib=40), 'settings_invalid', 'physical RAM'),
        ('fixed', dict(port=1), 'settings_invalid', 'only at server start: port'),
        ('override', dict(backend='metal'), 'settings_invalid', 'launch override'),
    ]
    for n, changes, code, text in cases:
        answer = change(request, folder, n, changes)
        assert not answer['accepted'] and answer['event']['code'] == code and text in answer['event']['message'], (n, answer)
    expired = change(request, folder, 'expired', dict(preferred_gib=20), deadline=time.time()-1)
    assert not expired['accepted'] and expired['event']['code'] == 'settings_expired'
    missing = change(request, folder, 'shape', values={k: v for k, v in s['values'].items() if k != 'context'})
    assert not missing['accepted'] and 'does not match' in missing['event']['message']
    assert (folder/'settings.json').read_bytes() == before
    assert not [r for r in rows(folder) if r['event'] in ('settings_changed', 'runtime_event')]
    assert json.loads((folder/'status.json').read_text())['event'] == status['event']   # a refused save notifies nothing
    # A refused save leaves admission exactly as it was.
    assert not request('cold', cold(tmp_path, 50*GiB))['ok']


def test_a_check_changes_nothing_and_a_restore_only_goes_back(tmp_path, supervisor):
    proc, folder, request = supervisor
    before = (folder/'settings.json').read_bytes()
    checked = change(request, folder, 'check', dict(preferred_gib=20, floor_gib=15), check=True)
    assert (folder/'settings.json').read_bytes() == before and not [r for r in rows(folder) if r['event'] == 'settings_changed']
    assert checked['accepted'] and checked['check']
    original = running_settings(folder)['values']
    sideways = change(request, folder, 'sideways', dict(resident_seconds=50), restore=True)
    assert not sideways['accepted'] and 'restore must return' in sideways['event']['message']
    # A restore is not validated, so it cannot be a way past validation either.
    invalid = change(request, folder, 'invalid-restore', dict(floor_gib=30), restore=True)
    assert not invalid['accepted']
    assert change(request, folder, 'apply', dict(resident_seconds=50))['accepted']
    back = change(request, folder, 'back', values=original, restore=True)
    assert back['accepted'] and running_settings(folder)['values'] == original
    assert change(request, folder, 'noop', values=original, restore=True)['accepted']
    restored = [r for r in rows(folder) if r['event'] == 'settings_changed']
    assert [r['restore'] for r in restored] == [False, True, True]


def test_a_running_operation_keeps_the_limits_it_was_admitted_with(tmp_path, supervisor):
    proc, folder, request = supervisor
    with gpu_telemetry(folder):
        loaded = dict(cold(tmp_path, GiB), runner=True)
        assert request('loaded', loaded)['ok']
        op = dict(loaded, operation_id='op', working={'kv': GiB}, job='generate')
        assert request('begin', op, action='begin')['ok']
        answer = change(request, folder, 'raise', dict(preferred_gib=60))
        assert answer['accepted'] and answer['running'] and answer['effect']['kind'] == 'after_operation'
        assert [r['operation_id'] for r in rows(folder) if r['event'] == 'settings_changed'] == ['op']
        # 71 GiB available - 15 GiB new: inside the admitted 25 GiB reserve, not the saved 60 GiB.
        grown = request('extend', dict(op, working={'kv': 15*GiB}), action='extend')
        assert grown['ok'], grown
        assert request('finish', dict(operation_id='op', actual=loaded, success=True), action='finish')['ok']
        later = request('later', dict(loaded, operation_id='op2', working={'kv': 15*GiB}, job='generate'), action='begin')
        assert not later['ok'] and later['event']['code'] == 'forecast_exceeds_budget'       # the saved limit applies now
        assert json.loads((folder/'status.json').read_text())['watch']['preferred'] == 60*GiB
        assert request('release', action='release')['ok']


def test_a_loaded_model_keeps_its_context_until_it_is_unloaded(tmp_path, supervisor):
    proc, folder, request = supervisor
    with gpu_telemetry(folder):
        loaded = dict(cold(tmp_path, GiB), runner=True)
        assert request('bind', loaded)['ok']
        assert request('load', dict(loaded, operation_id='load', working={}, job='generate'), action='begin')['ok']
        assert request('loaded', dict(operation_id='load', actual=loaded, success=True), action='finish')['ok']
        answer = change(request, folder, 'context', dict(context=4096))
        assert answer['accepted'] and answer['loaded'] and answer['load']['context'] == 8192
        status = json.loads((folder/'status.json').read_text())
        assert status['load']['context'] == 8192 and status['next_load']['context'] == 4096
        # Work on the loaded model is not refused because the saved context changed ...
        assert request('use', dict(loaded, operation_id='use', working={'kv': GiB}, job='generate'), action='begin')['ok']
        assert request('used', dict(operation_id='use', actual=loaded, success=True), action='finish')['ok']
        smaller = dict(cold(tmp_path, GiB, 4096), runner=True)
        mixed = request('mixed', dict(smaller, operation_id='mixed', working={}, job='generate'), action='begin')
        assert not mixed['ok'] and mixed['event']['code'] == 'configuration_mismatch'
        # ... and the next load uses the new one.
        assert request('release', action='release')['ok']
        stale = request('stale', cold(tmp_path, GiB))
        assert not stale['ok'] and stale['event']['code'] == 'configuration_mismatch'
        assert request('fresh', cold(tmp_path, GiB, 4096))['ok']
        assert request('release2', action='release')['ok']


def test_the_idle_watch_does_not_treat_a_saved_context_as_a_changed_model(tmp_path, supervisor):
    # A reservation bound without an operation has no pin: the watch's own check is of the model file only.
    proc, folder, request = supervisor
    with gpu_telemetry(folder):
        assert request('bind', dict(cold(tmp_path, GiB), runner=True))['ok']
        assert change(request, folder, 'context', dict(context=4096))['accepted']
        mark = len(rows(folder))
        wait(lambda: sum(r['event'] == 'sample' for r in rows(folder)[mark:]) >= 3, 8)
        assert not [r for r in rows(folder)[mark:] if r['event'] == 'runtime_event']
        assert json.loads((folder/'status.json').read_text())['state'] == 'idle'
        assert request('release', action='release')['ok']


def test_a_shorter_idle_timeout_unloads_on_the_idle_clock_and_the_save_says_so(tmp_path, supervisor):
    proc, folder, request = supervisor
    with gpu_telemetry(folder):
        assert request('bind', dict(cold(tmp_path, GiB), runner=True))['ok']
        time.sleep(11)                                              # idle since start; the shortest timeout is 10 s
        mark = len(rows(folder))
        wait(lambda: any(r['event'] == 'sample' for r in rows(folder)[mark:]), 5)
        assert not (folder/'control.json').exists()                 # 30 s: nothing to do yet
        asked = change(request, folder, 'ask', dict(resident_seconds=10))
        assert not asked['accepted'] and asked['event']['code'] == 'settings_unconfirmed' and asked['effect']['code'] == 'idle_expired'
        assert not (folder/'control.json').exists()                 # refused unconfirmed: nothing changed
        answer = change(request, folder, 'short', dict(resident_seconds=10), confirmed=True)
        assert answer['accepted'] and answer['effect']['kind'] == 'unload' and answer['effect']['code'] == 'idle_expired'
        assert 'unloads the loaded model' in answer['effect']['message']
        control = wait(lambda: json.loads((folder/'control.json').read_text()), 5)
        assert control['action'] == 'unload' and control['event']['code'] == 'idle_expired'
        assert control['event']['limit'] == 10
        assert request('release', action='release')['ok']


def test_the_check_warns_when_new_values_would_unload_the_loaded_model(tmp_path, supervisor):
    proc, folder, request = supervisor
    with gpu_telemetry(folder):
        assert request('bind', dict(cold(tmp_path, GiB), runner=True))['ok']
        before = (folder/'settings.json').read_bytes()
        effect = change(request, folder, 'check', dict(floor_gib=75, preferred_gib=80), check=True)['effect']
        assert effect['kind'] == 'unload' and effect['code'] == 'headroom_floor' and effect['seconds'] == 3
        assert 'if this lasts 3 s' in effect['message']
        assert change(request, folder, 'fine', dict(floor_gib=15, preferred_gib=20), check=True)['effect']['kind'] == 'none'
        assert (folder/'settings.json').read_bytes() == before
        assert request('release', action='release')['ok']


def test_a_load_spawns_its_worker_with_the_settings_it_began_with(tmp_path, supervisor):
    # A save between a load's begin and its worker spawn changes the next load, not this one.
    proc, folder, request = supervisor
    load = dict(cold(tmp_path, GiB), runner=True, worker=True, operation_id='load', working={}, job='load')
    assert request('begin', load, action='begin')['ok']
    assert change(request, folder, 'threads', dict(threads=1, context=4096))['accepted']
    spawned = request('spawn', dict(operation_id='load', workspace_dir=str(tmp_path)), action='spawn_worker')
    assert spawned['ok'], spawned
    row = [r for r in rows(folder) if r['event'] == 'worker_spawned'][-1]
    assert row['load']['context'] == 8192 and row['load']['threads'] != 1 and row['load']['selected'] == 'cpu'
    status = json.loads((folder/'status.json').read_text())
    assert status['load']['context'] == 8192 and status['next_load']['threads'] == 1 and status['next_load']['context'] == 4096
    assert request('release', action='release')['ok']


def test_new_memory_limits_drive_the_idle_watch_of_a_loaded_model(tmp_path, supervisor):
    # The save does not unload; the watch's next samples do, through the debounced ladder, with the new floor.
    proc, folder, request = supervisor
    with gpu_telemetry(folder):
        assert request('bind', dict(cold(tmp_path, GiB), runner=True))['ok']
        answer = change(request, folder, 'raise', dict(floor_gib=75, preferred_gib=80), confirmed=True)
        assert answer['accepted'] and answer['effect']['code'] == 'headroom_floor'
        assert not (folder/'control.json').exists()                     # nothing unloaded by the save itself
        control = wait(lambda: json.loads((folder/'control.json').read_text()), 10)
        assert control['action'] == 'unload' and control['event']['code'] == 'headroom_floor' and control['event']['limit'] == 75*GiB
        assert request('release', action='release')['ok']


def test_an_operation_that_leaves_no_model_loaded_leaves_no_pin(tmp_path, supervisor):
    proc, folder, request = supervisor
    metadata = cold(tmp_path, GiB)
    assert request('begin', dict(metadata, operation_id='scan', working={}, job='analysis'), action='begin')['ok']
    assert json.loads((folder/'status.json').read_text())['load']['context'] == 8192
    assert request('done', dict(operation_id='scan', actual=metadata, success=True), action='finish')['ok']
    assert json.loads((folder/'status.json').read_text())['load'] is None
    assert change(request, folder, 'context', dict(context=4096))['accepted']
    smaller = cold(tmp_path, GiB, 4096)
    nxt = request('next', dict(smaller, operation_id='next', working={}, job='analysis'), action='begin')
    assert nxt['ok'], nxt
    assert request('next-done', dict(operation_id='next', actual=smaller, success=True), action='finish')['ok']
    assert request('release', action='release')['ok']


# ------------------------------------------------------------------ late answers, confirmations, refusals (review)
STALLS = '''
import sys as _sys, time as _time
def _delay(name):
    try:return float(json.loads((Path(os.environ['HOME'])/'sample.json').read_text()).get(name, 0))
    except (OSError, ValueError):return 0
_orig_snapshot = m.snapshot
def _slow_snapshot():
    # effect_sleep: a slow memory sample while the effect of a settings change is judged (before its commit).
    f = _sys._getframe()
    while f is not None:
        if f.f_code.co_name == 'settings_effect':
            _time.sleep(_delay('effect_sleep'))
            if _delay('effect_fail'):raise OSError('vm_stat did not answer')
            break
        f = f.f_back
    return _orig_snapshot()
m.snapshot = _slow_snapshot
_orig_write = r.write
def _slow_write(path, value):
    # commit_sleep: a commit (the runtime directory's settings.json) that stalls after the deadline check passed.
    if Path(path).name == 'settings.json':_time.sleep(_delay('commit_sleep'))
    return _orig_write(path, value)
r.write = _slow_write
'''
LAUNCH = "s=resolve(env=dict(HOME=os.environ['HOME']),overrides=dict(backend='cpu',floor_gib=20,preferred_gib=25,resident_seconds=30))"
INERT_CHILD = "original([sys.executable,'-c','import time;time.sleep(60)'],**kw)"


@contextmanager
def custom_supervisor(tmp_path, monkeypatch, *, overrides="dict(backend='cpu')", stalls=False, child=None, config=CONFIG):
    original = T.supervisor_script
    def script(path, *, inert=False):
        out = original(path, inert=inert)
        text = out.read_text()
        assert LAUNCH in text and INERT_CHILD in text
        text = text.replace(LAUNCH, f"s=resolve(os.environ['HOME']+'/config.json',env=dict(HOME=os.environ['HOME']),overrides={overrides})")
        if stalls:text = text.replace("m.snapshot=snapshot\n", "m.snapshot=snapshot\n"+STALLS)
        if child is not None:text = text.replace(INERT_CHILD, f"original([sys.executable,'-c',{child!r}],**kw)")
        out.write_text(text)
        return out
    monkeypatch.setattr(T, 'supervisor_script', script)
    (tmp_path/'config.json').write_text(json.dumps(config))
    with T.inert_supervisor(tmp_path) as running:
        yield running


def stall(tmp_path, **delays):
    sample = json.loads((tmp_path/'sample.json').read_text()); sample.update(delays)
    write(tmp_path/'sample.json', sample)


def three(folder, client, store):
    """floor_gib in the supervisor, the server's client, the store's active settings and the config file."""
    saved = json.loads(store.path.read_text())['defaults']['floor_gib']
    return running_settings(folder)['values']['floor_gib'], client.settings['values']['floor_gib'], store.active['values']['floor_gib'], saved


def test_a_reply_that_arrives_late_with_a_refusal_is_not_applied_anywhere(tmp_path, monkeypatch):
    # The review's probe: the effect's memory sample stalls past the deadline. The supervisor now checks the
    # deadline again just before it commits, so it refuses; the client, without an answer in time, asks what the
    # supervisor runs and reports "not applied" because it is true.
    monkeypatch.setattr(rs, 'REQUEST_TIMEOUT_SECONDS', 3)                 # the client waits 3 s; its deadline is 2 s
    with custom_supervisor(tmp_path, monkeypatch, stalls=True) as (proc, folder, request), gpu_telemetry(folder):
        assert request('bind', dict(cold(tmp_path, GiB), runner=True))['ok']
        client = RuntimeClient(folder, copy.deepcopy(running_settings(folder))); store = SettingsStore(client.settings)
        before = store.path.read_bytes()
        stall(tmp_path, effect_sleep=3.5)
        with pytest.raises(RuntimeRefusal, match='still runs the previous settings, so these were not applied') as caught:
            store.save(store.snapshot()['revision'], dict(floor_gib=5, preferred_gib=6), runtime=client)
        assert caught.value.record['code'] == 'watchdog_unavailable'
        stall(tmp_path, effect_sleep=0)
        assert three(folder, client, store) == (20, 20, 20, 20) and store.path.read_bytes() == before
        assert store.out_of_step is None
        assert not [r for r in rows(folder) if r['event'] == 'settings_changed']
        assert request('release', action='release')['ok']


def test_a_commit_whose_reply_is_late_is_found_and_saved_everywhere(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, 'REQUEST_TIMEOUT_SECONDS', 3)
    with custom_supervisor(tmp_path, monkeypatch, stalls=True) as (proc, folder, request), gpu_telemetry(folder):
        assert request('bind', dict(cold(tmp_path, GiB), runner=True))['ok']
        client = RuntimeClient(folder, copy.deepcopy(running_settings(folder))); store = SettingsStore(client.settings)
        stall(tmp_path, commit_sleep=3.5)                                   # committed at 3.5 s, after the client gave up
        started = time.monotonic()
        saved = store.save(store.snapshot()['revision'], dict(floor_gib=5, preferred_gib=6), runtime=client)
        assert time.monotonic()-started > 3
        stall(tmp_path, commit_sleep=0)
        assert saved['applied']['late'] and saved['applied']['now'] == ['floor_gib', 'preferred_gib']
        assert three(folder, client, store) == (5, 5, 5, 5) and store.out_of_step is None
        assert [r['new'] for r in rows(folder) if r['event'] == 'settings_changed'] == [dict(floor_gib=5, preferred_gib=6)]
        assert request('release', action='release')['ok']


def test_a_commit_not_known_either_way_is_said_so_until_the_next_accepted_save(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, 'REQUEST_TIMEOUT_SECONDS', 3)
    with custom_supervisor(tmp_path, monkeypatch, stalls=True) as (proc, folder, request), gpu_telemetry(folder):
        assert request('bind', dict(cold(tmp_path, GiB), runner=True))['ok']
        client = RuntimeClient(folder, copy.deepcopy(running_settings(folder))); store = SettingsStore(client.settings)
        before = store.path.read_bytes(); rev = store.snapshot()['revision']
        stall(tmp_path, commit_sleep=7)                                     # past the request and the question (3 + 3 s)
        with pytest.raises(SettingsOutOfStep) as caught:
            store.save(rev, dict(floor_gib=5, preferred_gib=6), runtime=client)
        text = str(caught.value)
        assert 'may be running the unsaved values until the next accepted save or a restart' in text and 'nothing was' not in text.lower()
        # Not known: the file and both server copies keep the last known values, and the page says it does not know.
        assert three(folder, client, store)[1:] == (20, 20, 20) and store.path.read_bytes() == before
        assert store.snapshot(client)['out_of_step'] == text
        wait(lambda: running_settings(folder)['values']['floor_gib'] == 5, 8)   # it did commit, late
        stall(tmp_path, commit_sleep=0)
        # The next accepted save is sent although it changes nothing here, and makes all of them equal again.
        after = store.save(rev, {}, runtime=client)
        assert three(folder, client, store) == (20, 20, 20, 20) and after['out_of_step'] is None and store.out_of_step is None
        assert [r['new'] for r in rows(folder) if r['event'] == 'settings_changed'] == [dict(floor_gib=5, preferred_gib=6), dict(floor_gib=20, preferred_gib=25)]
        assert request('release', action='release')['ok']


@pytest.mark.parametrize('deadline', [float('nan'), float('inf'), float('-inf'), 'later', None])
def test_a_deadline_must_be_finite_and_within_one_request_timeout(tmp_path, supervisor, deadline):
    proc, folder, request = supervisor
    before = (folder/'settings.json').read_bytes()
    body = dict(values=dict(running_settings(folder)['values'], preferred_gib=20), sources=running_settings(folder)['sources'],
                notices=running_settings(folder)['notices'], restore=False, check=False)
    if deadline is not None:body['deadline'] = deadline
    answer = request('odd', body, action='settings')['settings']
    assert not answer['accepted'] and answer['event']['code'] == 'settings_expired'
    far = change(request, folder, 'far', dict(preferred_gib=20), deadline=time.time()+rs.REQUEST_TIMEOUT_SECONDS+60)
    assert not far['accepted'] and far['event']['code'] == 'settings_expired'
    # Checked on arrival too: a stale check is refused before any work (a check never reaches the commit).
    stale = change(request, folder, 'stale', dict(preferred_gib=20), check=True, deadline=time.time()-1)
    assert not stale['accepted'] and stale['event']['code'] == 'settings_expired'
    assert (folder/'settings.json').read_bytes() == before


def test_a_saved_backend_is_checked_by_the_supervisor_as_at_start(tmp_path, monkeypatch):
    # The backend comes from the saved configuration (no launch override), so only the backend check can refuse it.
    with custom_supervisor(tmp_path, monkeypatch, overrides='dict()') as (proc, folder, request):
        assert running_settings(folder)['values']['backend'] == 'auto'
        before = (folder/'settings.json').read_bytes()
        answer = change(request, folder, 'cpu', dict(backend='cpu'))
        assert not answer['accepted'] and answer['event']['code'] == 'settings_invalid' and 'cpu bridge is not configured' in answer['event']['message']
        assert (folder/'settings.json').read_bytes() == before
        assert change(request, folder, 'same', dict(resident_seconds=60))['accepted']   # the backend unchanged: not checked


def test_advice_inside_an_operation_uses_the_limits_it_was_admitted_with(tmp_path, supervisor):
    proc, folder, request = supervisor
    with gpu_telemetry(folder):
        loaded = dict(cold(tmp_path, GiB), runner=True)
        assert request('loaded', loaded)['ok']
        assert request('begin', dict(loaded, operation_id='op', working={'kv': GiB}, job='generate'), action='begin')['ok']
        assert change(request, folder, 'raise', dict(preferred_gib=60))['accepted']
        edit = dict(loaded, working={'activations': 64*1024**2})
        inside = request('inside', edit, action='advise')['advice']
        assert inside['preferred'] == 25*GiB, inside                       # the admitted reserve, not the saved 60 GiB
        assert request('finish', dict(operation_id='op', actual=loaded, success=True), action='finish')['ok']
        after = request('after', edit, action='advise')['advice']
        assert after['preferred'] == 60*GiB
        assert request('release', action='release')['ok']


def test_a_settings_save_is_refused_while_the_supervisor_shuts_down(tmp_path, monkeypatch):
    # Deterministic: the inert server child ignores the shutdown signal until the test releases it, so the
    # supervisor stays in its shutdown for as long as the test needs. The signal is sent only once the child has
    # said it ignores it (the supervisor's start row can come before the child's interpreter is up).
    release, ready = tmp_path/'release-child', tmp_path/'child-ready'
    child = ("import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
             f"open({str(ready)!r},'w').close();"
             f"[time.sleep(.05) for _ in iter(lambda:os.path.exists({str(release)!r}),True)]")
    with custom_supervisor(tmp_path, monkeypatch, child=child) as (proc, folder, request):
        before = (folder/'settings.json').read_bytes()
        try:
            wait(ready.exists, 15)
            proc.send_signal(signal.SIGTERM)
            wait(lambda: [r for r in rows(folder) if r['event'] == 'server_signal' and r['reason'] == 'user_shutdown'], 5)
            answer = change(request, folder, 'stopping', dict(preferred_gib=20))
            assert not answer['accepted'] and answer['event']['code'] == 'settings_unavailable' and 'shutting down' in answer['event']['message']
            # Asking what it runs still has an answer: it changes nothing.
            held = request('held', dict(query=True), action='settings')['settings']
            assert held['query'] and held['values']['preferred_gib'] == 25
            assert (folder/'settings.json').read_bytes() == before and proc.poll() is None
        finally:release.touch()
        proc.wait(timeout=10)


def test_a_save_that_unloads_the_loaded_model_is_refused_unless_confirmed(tmp_path, supervisor):
    proc, folder, request = supervisor
    with gpu_telemetry(folder):
        assert request('bind', dict(cold(tmp_path, GiB), runner=True))['ok']
        before = (folder/'settings.json').read_bytes()
        asked = change(request, folder, 'ask', dict(floor_gib=75, preferred_gib=80))
        assert not asked['accepted'] and asked['event']['code'] == 'settings_unconfirmed'
        assert asked['effect']['kind'] == 'unload' and asked['effect']['confirm'] and 'unloads the loaded model' in asked['event']['message']
        assert (folder/'settings.json').read_bytes() == before and not [r for r in rows(folder) if r['event'] == 'settings_changed']
        assert not (folder/'control.json').exists()
        # A change that unloads nothing needs no confirmation.
        assert change(request, folder, 'fine', dict(floor_gib=15, preferred_gib=20))['accepted']
        assert change(request, folder, 'yes', dict(floor_gib=75, preferred_gib=80), confirmed=True)['accepted']
        assert [r['confirmed'] for r in rows(folder) if r['event'] == 'settings_changed'] == [False, True]
        # The model or the running operation changed since the confirmation: judged again, refused again.
        stale = asked['effect']['judged']
        assert change(request, folder, 'back', dict(floor_gib=20, preferred_gib=25))['accepted']
        assert request('begin', dict(cold(tmp_path, GiB), runner=True, operation_id='op', working={}, job='generate'), action='begin')['ok']
        again = change(request, folder, 'stale', dict(floor_gib=75, preferred_gib=80), confirmed=stale)
        assert not again['accepted'] and again['event']['code'] == 'settings_unconfirmed' and 'changed since you confirmed' in again['event']['message']
        assert again['effect']['kind'] == 'after_operation' and again['effect']['judged'] != stale
        assert running_settings(folder)['values']['floor_gib'] == 20
        assert request('done', dict(operation_id='op', actual=dict(cold(tmp_path, GiB), runner=True), success=True), action='finish')['ok']
        assert request('release', action='release')['ok']


def test_an_idle_reservation_without_a_model_needs_no_confirmation(tmp_path, supervisor):
    proc, folder, request = supervisor
    assert request('bind', cold(tmp_path, GiB))['ok']                      # metadata only: no model loaded
    answer = change(request, folder, 'raise', dict(floor_gib=75, preferred_gib=80))
    assert answer['accepted'] and answer['effect']['kind'] == 'unload' and not answer['effect']['confirm']
    assert request('release', action='release')['ok']


def test_an_unload_that_follows_the_running_operation_is_judged_and_needs_a_confirmation(tmp_path, supervisor):
    proc, folder, request = supervisor
    with gpu_telemetry(folder):
        loaded = dict(cold(tmp_path, GiB), runner=True)
        assert request('loaded', loaded)['ok']
        assert request('begin', dict(loaded, operation_id='op', working={'kv': GiB}, job='generate'), action='begin')['ok']
        checked = change(request, folder, 'check', dict(floor_gib=75, preferred_gib=80), check=True)['effect']
        assert checked['kind'] == 'after_operation' and checked['unload'] and checked['code'] == 'headroom_floor' and checked['confirm']
        assert 'apply when it finishes' in checked['message'] and 'would then unload the loaded model' in checked['message']
        asked = change(request, folder, 'ask', dict(floor_gib=75, preferred_gib=80))
        assert not asked['accepted'] and asked['event']['code'] == 'settings_unconfirmed'
        assert change(request, folder, 'yes', dict(floor_gib=75, preferred_gib=80), confirmed=True)['accepted']
        # A change that leaves the idle state safe is told it applies afterwards, with nothing to confirm.
        calm = change(request, folder, 'calm', dict(floor_gib=15, preferred_gib=20))
        assert calm['accepted'] and calm['effect']['kind'] == 'after_operation' and not calm['effect']['confirm'] and not calm['effect'].get('unload')
        assert request('finish', dict(operation_id='op', actual=loaded, success=True), action='finish')['ok']
        # A confirmation shown during one operation does not cover the next one.
        assert request('begin2', dict(loaded, operation_id='op2', working={'kv': GiB}, job='generate'), action='begin')['ok']
        later = change(request, folder, 'later', dict(floor_gib=75, preferred_gib=80), confirmed=checked['judged'])
        assert not later['accepted'] and later['event']['code'] == 'settings_unconfirmed' and later['effect']['judged'] != checked['judged']
        assert later['effect']['kind'] == checked['kind'] == 'after_operation' and running_settings(folder)['values']['floor_gib'] == 15
        assert request('finish2', dict(operation_id='op2', actual=loaded, success=True), action='finish')['ok']
        assert request('release', action='release')['ok']


def test_a_save_whose_effect_cannot_be_sampled_while_a_model_is_loaded_asks_first(tmp_path, monkeypatch):
    with custom_supervisor(tmp_path, monkeypatch, stalls=True) as (proc, folder, request), gpu_telemetry(folder):
        assert request('bind', dict(cold(tmp_path, GiB), runner=True))['ok']
        # A confirmation of one judged cause (the floor) does not cover another (not known), same model, no operation.
        floor = change(request, folder, 'floor', dict(floor_gib=75, preferred_gib=80), check=True)['effect']
        stall(tmp_path, effect_fail=1)
        other = change(request, folder, 'other', dict(floor_gib=75, preferred_gib=80), confirmed=floor['judged'])
        assert not other['accepted'] and other['effect']['kind'] == 'unknown' and other['effect']['judged'] != floor['judged']
        asked = change(request, folder, 'ask', dict(floor_gib=15, preferred_gib=20))
        assert not asked['accepted'] and asked['event']['code'] == 'settings_unconfirmed'
        assert asked['effect']['kind'] == 'unknown' and asked['effect']['confirm'] and 'could not be sampled' in asked['effect']['message']
        assert change(request, folder, 'yes', dict(floor_gib=15, preferred_gib=20), confirmed=True)['accepted']
        stall(tmp_path, effect_fail=0)
        assert request('release', action='release')['ok']



def test_a_confirmation_for_one_loaded_model_does_not_unload_another(tmp_path, supervisor):
    proc, folder, request = supervisor
    with gpu_telemetry(folder):
        assert request('bind', dict(cold(tmp_path, GiB), runner=True))['ok']
        first = change(request, folder, 'ask', dict(floor_gib=75, preferred_gib=80))
        assert not first['accepted'] and first['effect']['confirm']
        assert request('release', action='release')['ok']
        (tmp_path/'other').mkdir(); other = fake_ws(tmp_path/'other')     # another model file
        value = descriptor(other, 8192, backend='cpu'); value['forecast'] = dict(value['forecast'], total=GiB)
        assert request('bind-other', dict(value, runner=True))['ok']
        again = change(request, folder, 'stale', dict(floor_gib=75, preferred_gib=80), confirmed=first['effect']['judged'])
        assert not again['accepted'] and again['event']['code'] == 'settings_unconfirmed' and again['effect']['judged'] != first['effect']['judged']
        assert running_settings(folder)['values']['floor_gib'] == 20 and not (folder/'control.json').exists()
        assert change(request, folder, 'yes', dict(floor_gib=75, preferred_gib=80), confirmed=again['effect']['judged'])['accepted']
        assert request('release', action='release')['ok']


def test_only_a_change_of_what_the_supervisor_applies_is_judged(tmp_path, monkeypatch):
    # Memory cannot be sampled while a model is loaded, so any judged change would ask for a confirmation.
    with custom_supervisor(tmp_path, monkeypatch, stalls=True) as (proc, folder, request), gpu_telemetry(folder):
        assert request('bind', dict(cold(tmp_path, GiB), runner=True))['ok']
        stall(tmp_path, effect_fail=1)
        for n, changes in (('tokens', dict(new_tokens=7)), ('patch', dict(patch_limit_gib=2)), ('context', dict(context=4096)), ('same', {})):
            answer = change(request, folder, n, changes)
            assert answer['accepted'] and answer['effect'] is None, (n, answer)
        # The resend of a store that is out of step: the values it already runs need nothing; values it does not
        # run (it held the unsaved ones) are judged like any change.
        held = running_settings(folder)['values']
        assert change(request, folder, 'resend', values=held)['accepted']
        judged = change(request, folder, 'differs', values=dict(held, floor_gib=15, preferred_gib=20))
        assert not judged['accepted'] and judged['effect']['kind'] == 'unknown' and judged['effect']['confirm']
        stall(tmp_path, effect_fail=0)
        assert request('release', action='release')['ok']


def test_an_existing_unload_cause_does_not_hide_a_new_one(tmp_path, supervisor):
    proc, folder, request = supervisor
    with gpu_telemetry(folder):
        assert request('bind', dict(cold(tmp_path, GiB), runner=True))['ok']
        stall(tmp_path, pressure=2)                                       # memory pressure, whatever the settings
        try:
            floor = change(request, folder, 'floor', dict(floor_gib=75, preferred_gib=80), check=True)['effect']
            assert floor['kind'] == 'unload' and floor['code'] == 'headroom_floor' and floor['causes'] == ['headroom_floor'] and floor['confirm']
            calm = change(request, folder, 'calm', dict(floor_gib=15, preferred_gib=20), check=True)['effect']
            assert calm['kind'] == 'none' and not calm['confirm']                  # pressure alone is not the save's doing
        finally:stall(tmp_path, pressure=1)
        # A floor cause the old values already have does not hide a budget cause the new ones add.
        stall(tmp_path, available=18*GiB)                                 # below the floor of 20 GiB
        try:
            budget = change(request, folder, 'budget', dict(budget_gib=0.0005), check=True)['effect']
            assert budget['kind'] == 'unload' and budget['code'] == 'runtime_budget' and budget['causes'] == ['runtime_budget'] and budget['confirm']
        finally:stall(tmp_path, available=71*GiB)
        assert request('release', action='release')['ok']


def test_idle_expiry_is_judged_beside_pressure_and_never_during_an_operation(tmp_path, supervisor):
    proc, folder, request = supervisor
    with gpu_telemetry(folder):
        loaded = dict(cold(tmp_path, GiB), runner=True)
        assert request('bind', loaded)['ok']
        time.sleep(11)                                                    # idle since start; the shortest timeout is 10 s
        stall(tmp_path, pressure=2)
        try:
            expiry = change(request, folder, 'short', dict(resident_seconds=10), check=True)['effect']
            assert expiry['kind'] == 'unload' and expiry['code'] == 'idle_expired' and expiry['confirm']
            # Two new causes: the expiry, which unloads at the next sample, is the one named, with its timing.
            both = change(request, folder, 'both', dict(resident_seconds=10, floor_gib=75, preferred_gib=80), check=True)['effect']
            assert both['code'] == 'idle_expired' and both['seconds'] == 1 and set(both['causes']) == {'idle_expired', 'headroom_floor'}
        finally:stall(tmp_path, pressure=1)
        # During an operation the idle clock restarts at its finish: a shorter timeout unloads nothing then.
        assert request('begin', dict(loaded, operation_id='op', working={}, job='generate'), action='begin')['ok']
        during = change(request, folder, 'during', dict(resident_seconds=10))
        assert during['accepted'] and during['effect']['kind'] == 'after_operation' and not during['effect']['confirm']
        assert request('finish', dict(operation_id='op', actual=loaded, success=True), action='finish')['ok']
        assert request('release', action='release')['ok']


def test_a_restore_is_never_held_back_for_a_confirmation(tmp_path, supervisor):
    proc, folder, request = supervisor
    with gpu_telemetry(folder):
        assert request('bind', dict(cold(tmp_path, GiB), runner=True))['ok']
        original = running_settings(folder)['values']
        assert change(request, folder, 'lower', dict(floor_gib=15, preferred_gib=20))['accepted']
        stall(tmp_path, available=18*GiB)                                 # between the new floor 15 and the old 20
        try:
            check = change(request, folder, 'check', values=original, check=True)['effect']
            assert check['kind'] == 'unload' and check['code'] == 'headroom_floor' and check['confirm']   # an apply would ask
            back = change(request, folder, 'back', values=original, restore=True)
            assert back['accepted'] and running_settings(folder)['values'] == original
        finally:stall(tmp_path, available=71*GiB)
        assert request('release', action='release')['ok']
