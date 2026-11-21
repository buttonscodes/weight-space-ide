"""Live settings without processes: the save order, failure paths, classes and the client side. No models."""
import copy
from contextlib import contextmanager
import json
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from wsp import runtime_supervisor as R
from wsp.memory_guard import GiB
from wsp.runtime_config import NEXT_LOAD, SERVER_LIVE, SUPERVISOR_LIVE, resolve
from wsp.runtime_events import RuntimeRefusal, event, unavailable
from wsp.runtime_policy import admission
from wsp.runtime_supervisor import RuntimeClient, SettingsUnsettled, idle_watch_policy, load_config, unsettled, write
from wsp.server import create_app
from wsp.session import Session
from wsp.settings import EDITABLE, SettingsNeedConfirmation, SettingsOutOfStep, SettingsStore, SettingsConflict, takes_effect
from test_runtime_supervisor import fake_ws

URL = 'http://127.0.0.1:41893'


def settings_at(tmp_path, defaults=None, overrides=None):
    path = tmp_path/'config.json'
    path.write_text(json.dumps({'version': 1, 'defaults': {'context': 128, 'floor_gib': 20, 'preferred_gib': 25, **(defaults or {})}, 'profiles': {}}))
    return resolve(str(path), overrides=overrides, env={}, system='Linux')


class Supervisor:
    """The client side of a supervisor that accepts, refuses, or fails a restore, recording every call.
    ``effect``: what the change would do; one with ``confirm`` is refused unless the call is confirmed, as the
    supervisor refuses it (settings_unconfirmed). ``restore_fails`` may be an exception to raise for the restore."""
    def __init__(self, settings, refuse=None, restore_fails=False, effect=None):
        self.settings = copy.deepcopy(settings); self.calls = []; self.refuse = refuse; self.restore_fails = restore_fails
        self.loaded = None; self.effect = effect

    def update_settings(self, settings, *, restore=False, confirmed=None):
        self.calls.append(('restore' if restore else 'apply', copy.deepcopy(settings['values']), confirmed))
        if restore and self.restore_fails:
            raise self.restore_fails if isinstance(self.restore_fails, Exception) else unavailable('The memory watchdog did not answer.')
        if not restore and self.refuse is not None: raise self.refuse
        if not restore and (self.effect or {}).get('confirm') and confirmed != self.effect.get('judged'):
            refusal = RuntimeRefusal(event('settings_unconfirmed', 'refused', self.effect['message']+' Nothing was changed; confirm to save these settings.'))
            refusal.effect = self.effect
            raise refusal
        self.settings = copy.deepcopy(settings)
        return dict(accepted=True, changed=[], effect=self.effect, running=False, loaded=bool(self.loaded))

    def loaded_config(self, ws=None): return self.loaded

    def config(self): return self.loaded or load_config(self.settings, 'cpu')


def test_classes_cover_every_editable_setting_once():
    assert SERVER_LIVE | SUPERVISOR_LIVE | NEXT_LOAD == EDITABLE
    assert not SERVER_LIVE & SUPERVISOR_LIVE and not SERVER_LIVE & NEXT_LOAD and not SUPERVISOR_LIVE & NEXT_LOAD
    assert {k: takes_effect(k, True) for k in ('patch_limit_gib', 'new_tokens', 'floor_gib', 'resident_seconds', 'context', 'threads', 'backend')} == dict(
        patch_limit_gib='saved', new_tokens='saved', floor_gib='saved', resident_seconds='saved', context='next_load', threads='next_load', backend='next_load')
    assert {takes_effect(k, False) for k in SUPERVISOR_LIVE | NEXT_LOAD} == {'next_start'}


def test_supervisor_applies_first_then_the_file_and_all_three_end_equal(tmp_path):
    settings = settings_at(tmp_path); store = SettingsStore(settings); runtime = Supervisor(settings)
    seen = {}
    original = store._write
    def write_file(raw, expected):
        seen['supervisor_floor_when_written'] = runtime.settings['values']['floor_gib']
        return original(raw, expected)
    store._write = write_file
    after = store.save(store.snapshot()['revision'], {'floor_gib': 15, 'preferred_gib': 20, 'context': 96, 'new_tokens': 7}, runtime=runtime)
    assert seen['supervisor_floor_when_written'] == 15                           # the supervisor accepted before the file was written
    assert json.loads(store.path.read_text())['defaults'] == dict(context=96, floor_gib=15, preferred_gib=20, new_tokens=7)
    assert runtime.settings == store.active and [c[0] for c in runtime.calls] == ['apply']
    assert store.active['values']['floor_gib'] == 15 and store.active['values']['context'] == 96
    applied = after['applied']
    assert applied['now'] == ['floor_gib', 'new_tokens', 'preferred_gib'] and applied['next_load'] == ['context']
    assert applied['next_start'] == [] and applied['overridden'] == [] and applied['supervised'] is True
    assert not after['restart_required'] and after['changed'] == [] and after['takes_effect']['context'] == 'next_load'


def test_a_refused_or_unanswered_change_writes_and_changes_nothing(tmp_path):
    settings = settings_at(tmp_path)
    for refusal in (RuntimeRefusal(event('settings_invalid', 'refused', 'Not applied: Memory policy requires 0 < floor_gib <= preferred_gib')),
                    unavailable('The memory watchdog did not answer within 8 s; the settings were not applied.')):
        store = SettingsStore(settings); runtime = Supervisor(settings, refuse=refusal)
        before = store.path.read_bytes(); active = copy.deepcopy(store.active)
        with pytest.raises(RuntimeRefusal):
            store.save(store.snapshot()['revision'], {'floor_gib': 30}, runtime=runtime)
        assert store.path.read_bytes() == before and store.active == active and runtime.settings == settings
        assert [c[0] for c in runtime.calls] == ['apply']                        # nothing to restore: nothing was applied


def test_an_unreachable_supervisor_writes_nothing_and_says_it_is_not_known_what_it_runs(tmp_path, monkeypatch):
    # A real client with nobody answering its mailbox: a stalled supervisor might still have applied the change, so
    # the save is not reported as "not applied"; the file is not written and the page says the state is not known.
    monkeypatch.setattr(R, 'REQUEST_TIMEOUT_SECONDS', .4)
    settings = settings_at(tmp_path); client = RuntimeClient(tmp_path/'mailbox', copy.deepcopy(settings))
    (tmp_path/'mailbox').mkdir(); store = SettingsStore(client.settings)
    before = store.path.read_bytes()
    with pytest.raises(SettingsOutOfStep) as caught:
        store.save(store.snapshot()['revision'], {'preferred_gib': 20, 'floor_gib': 15}, runtime=client)
    assert 'may be running the unsaved values until the next accepted save or a restart' in str(caught.value)
    assert isinstance(caught.value.__cause__, SettingsUnsettled) and caught.value.__cause__.held is None
    assert store.path.read_bytes() == before and client.settings == settings and store.active == settings
    assert store.out_of_step == str(caught.value)
    assert client.last_event is None                                              # a settings refusal is not an operation's event
    # The second request asked what the supervisor runs; the first carried a deadline it checks before committing.
    request = json.loads((tmp_path/'mailbox'/'request.json').read_text())
    assert request['action'] == 'settings' and request['descriptor'] == dict(query=True)


def test_a_failed_file_write_restores_the_supervisor(tmp_path):
    settings = settings_at(tmp_path); store = SettingsStore(settings); runtime = Supervisor(settings)
    before = store.path.read_bytes()
    def fail(raw, expected): raise OSError('disk full')
    store._write = fail
    with pytest.raises(OSError, match='disk full'):
        store.save(store.snapshot()['revision'], {'floor_gib': 15, 'preferred_gib': 20}, runtime=runtime)
    assert [c[0] for c in runtime.calls] == ['apply', 'restore'] and runtime.calls[1][1] == settings['values']
    assert runtime.settings == settings and store.active == settings and store.path.read_bytes() == before


def test_a_file_changed_while_saving_restores_the_supervisor(tmp_path):
    settings = settings_at(tmp_path); store = SettingsStore(settings); runtime = Supervisor(settings)
    original = store._write
    def edited_meanwhile(raw, expected):
        store.path.write_text(store.path.read_text().replace('"context": 128', '"context": 100'))
        return original(raw, expected)
    store._write = edited_meanwhile
    with pytest.raises(SettingsConflict):
        store.save(store.snapshot()['revision'], {'resident_seconds': 60}, runtime=runtime)
    assert [c[0] for c in runtime.calls] == ['apply', 'restore'] and runtime.settings == settings and store.active == settings


def test_a_failed_restore_says_the_supervisor_runs_the_unsaved_values(tmp_path):
    settings = settings_at(tmp_path); store = SettingsStore(settings); runtime = Supervisor(settings, restore_fails=True)
    def fail(raw, expected): raise OSError('read-only file system')
    store._write = fail
    with pytest.raises(SettingsOutOfStep, match='uses the unsaved values until the next accepted save or a server restart'):
        store.save(store.snapshot()['revision'], {'resident_seconds': 60}, runtime=runtime)
    assert runtime.settings['values']['resident_seconds'] == 60 and store.active == runtime.settings
    assert json.loads(store.path.read_text())['defaults'].get('resident_seconds') is None
    # The page says so until an accepted save, which is sent even when it changes nothing.
    view = store.snapshot(runtime)
    assert 'uses the unsaved values' in view['out_of_step']
    store._write = SettingsStore._write.__get__(store)
    store.save(view['revision'], {}, runtime=runtime)
    assert [c[0] for c in runtime.calls] == ['apply', 'restore', 'apply'] and runtime.calls[-1][1]['resident_seconds'] == settings['values']['resident_seconds']
    assert store.out_of_step is None and store.snapshot(runtime)['out_of_step'] is None and runtime.settings == store.active


def test_an_unsettled_restore_says_it_is_not_known_what_the_supervisor_runs(tmp_path):
    settings = settings_at(tmp_path); store = SettingsStore(settings)
    runtime = Supervisor(settings, restore_fails=unsettled(None, 'it may be running the unsaved values until the next accepted save or a restart.'))
    def fail(raw, expected): raise OSError('read-only file system')
    store._write = fail
    with pytest.raises(SettingsOutOfStep, match='may be running the unsaved values until the next accepted save or a restart'):
        store.save(store.snapshot()['revision'], {'resident_seconds': 60}, runtime=runtime)
    assert store.active == runtime.settings and 'may be running' in store.snapshot(runtime)['out_of_step']


def test_an_unsettled_change_writes_nothing_and_the_page_says_it_is_not_known(tmp_path):
    settings = settings_at(tmp_path); store = SettingsStore(settings)
    runtime = Supervisor(settings, refuse=unsettled(None, 'The memory watchdog did not answer within 8 s, nor when asked again what it runs: '
                                                                 'it may be running the unsaved values until the next accepted save or a restart.'))
    before = store.path.read_bytes()
    with pytest.raises(SettingsOutOfStep) as caught:
        store.save(store.snapshot()['revision'], {'floor_gib': 15, 'preferred_gib': 20}, runtime=runtime)
    assert 'may be running the unsaved values until the next accepted save or a restart' in str(caught.value)
    assert 'nothing was written' not in str(caught.value).lower()
    assert store.path.read_bytes() == before and store.active == runtime.settings == settings
    assert store.snapshot(runtime)['out_of_step'] == str(caught.value)
    runtime.refuse = None
    after = store.save(store.snapshot()['revision'], {}, runtime=runtime)                 # sent although nothing changed
    assert [c[0] for c in runtime.calls] == ['apply', 'apply'] and after['out_of_step'] is None


def test_a_save_that_unloads_the_loaded_model_needs_a_confirmation_bound_to_its_candidate(tmp_path):
    settings = settings_at(tmp_path); store = SettingsStore(settings)
    effect = dict(kind='unload', code='idle_expired', confirm=True, loaded=True, judged='j1',
                  message='With these values the memory watchdog unloads the loaded model within about a second.')
    runtime = Supervisor(settings, effect=effect)
    before = store.path.read_bytes(); rev = store.snapshot()['revision']
    with pytest.raises(SettingsNeedConfirmation) as caught:
        store.save(rev, {'resident_seconds': 10}, runtime=runtime)
    assert caught.value.effect == effect and 'unloads the loaded model' in str(caught.value) and runtime.calls[-1][2] is None
    assert store.path.read_bytes() == before and store.active == settings and runtime.settings == settings
    token = caught.value.token
    assert token.endswith('.j1')
    with pytest.raises(SettingsNeedConfirmation) as other:                               # another draft: not authorised
        store.save(rev, {'resident_seconds': 11}, runtime=runtime, confirm=token)
    assert other.value.token != token and runtime.settings == settings and runtime.calls[-1][2] is None
    with pytest.raises(SettingsNeedConfirmation):                                        # a forged fingerprint: not authorised
        store.save(rev, {'resident_seconds': 10}, runtime=runtime, confirm=token[:-2]+'j9')
    assert runtime.calls[-1][2] is None
    # The file revision is part of it: the same values after another write of the file need a new confirmation.
    store.path.write_text(json.dumps(json.loads(store.path.read_text()), indent=4))
    rev2 = store.snapshot()['revision']
    assert rev2 != rev and store.updated(store.resolve(store.read()[0], effective=True), True) == store.active
    with pytest.raises(SettingsNeedConfirmation) as moved:
        store.save(rev2, {'resident_seconds': 10}, runtime=runtime, confirm=token)
    assert moved.value.token != token and runtime.calls[-1][2] is None
    # The model or the running operation changed since: the supervisor judges a new fingerprint and asks again.
    runtime.effect = dict(effect, judged='j2')
    with pytest.raises(SettingsNeedConfirmation) as changed:
        store.save(rev2, {'resident_seconds': 10}, runtime=runtime, confirm=moved.value.token)
    assert runtime.calls[-1][2] == 'j1' and changed.value.token.endswith('.j2') and runtime.settings == settings
    after = store.save(rev2, {'resident_seconds': 10}, runtime=runtime, confirm=changed.value.token)
    assert runtime.calls[-1][2] == 'j2' and runtime.settings['values']['resident_seconds'] == 10 and store.active == runtime.settings
    assert after['applied']['effect']['judged'] == 'j2' and json.loads(store.path.read_text())['defaults']['resident_seconds'] == 10


def test_a_launch_override_still_wins_and_is_reported(tmp_path):
    settings = settings_at(tmp_path, overrides={'floor_gib': 20, 'context': 128}); store = SettingsStore(settings); runtime = Supervisor(settings)
    after = store.save(store.snapshot()['revision'], {'floor_gib': 10, 'context': 64}, runtime=runtime)
    assert json.loads(store.path.read_text())['defaults']['floor_gib'] == 10       # saved for a start without the override
    assert store.active['values']['floor_gib'] == 20 and store.active['sources']['floor_gib'] == 'command line'
    assert store.active['values']['context'] == 128 and runtime.calls == []          # nothing the server runs changed
    assert after['applied']['overridden'] == ['context', 'floor_gib'] and after['applied']['now'] == []
    assert after['takes_effect']['floor_gib'] == 'override' and after['overridden'] == ['context', 'floor_gib']


def test_without_a_supervisor_memory_settings_wait_for_a_start(tmp_path):
    settings = settings_at(tmp_path); store = SettingsStore(settings)
    after = store.save(store.snapshot()['revision'], {'floor_gib': 15, 'preferred_gib': 20, 'new_tokens': 9, 'patch_limit_gib': 2, 'threads': 1}, runtime=None)
    assert after['applied']['now'] == ['new_tokens', 'patch_limit_gib'] and after['applied']['next_start'] == ['floor_gib', 'preferred_gib', 'threads']
    assert store.active['values']['floor_gib'] == 20 and store.active['values']['new_tokens'] == 9
    assert after['restart_required'] and after['pending'] == {'floor_gib': 'next_start', 'preferred_gib': 'next_start', 'threads': 'next_start'}
    assert 'no memory supervisor' in after['note'] and after['supervised'] is False and after['next_load'] == []
    # A runtime that cannot take settings (an embedded test double) is treated the same way.
    assert store.describe(*store.read(), SimpleNamespace(status=lambda ws: {}))['supervised'] is False


def test_a_hand_edit_of_the_file_is_pending_until_the_next_save_when_supervised(tmp_path):
    settings = settings_at(tmp_path); store = SettingsStore(settings); runtime = Supervisor(settings)
    store.path.write_text(store.path.read_text().replace('"floor_gib": 20', '"floor_gib": 18'))
    view = store.snapshot(runtime)
    assert view['pending'] == {'floor_gib': 'next_save'} and not view['restart_required']
    after = store.save(view['revision'], {'resident_seconds': 60}, runtime=runtime)
    assert runtime.settings['values']['floor_gib'] == 18 and after['applied']['now'] == ['floor_gib', 'resident_seconds']


@pytest.mark.parametrize('values', [dict(), dict(budget_gib=40), dict(ceiling_gib=50), dict(budget_gib=30, ceiling_gib=35),
                                    dict(ceiling_gib=120), dict(floor_gib=5, preferred_gib=40), dict(budget_gib=100, ceiling_gib=110)])
def test_the_idle_watch_cap_is_admissions_static_cap(tmp_path, values):
    settings = settings_at(tmp_path)
    settings['values'].update(values)
    sample = dict(total=128*GiB, available=71*GiB, pressure=1, tree_footprint=GiB, tree_rss=GiB)
    policy, decision = admission(settings, dict(forecast=dict(total=1), runner=False), sample)
    watch = idle_watch_policy(settings['values'], sample['total'])
    assert watch.budget == watch.ceiling == decision['static_cap']
    assert (watch.floor, watch.preferred, watch.swap_growth, watch.compression_growth) == (policy.floor, policy.preferred, policy.swap_growth, policy.compression_growth)


# ------------------------------------------------------------------ the client
def client_with(tmp_path, reply):
    client = RuntimeClient(tmp_path, settings_at(tmp_path))
    client._forecast_cache = ('stale', {}); client.last_event = dict(code='earlier')
    calls = []
    def request(action, value=None, timeout=None):
        calls.append((action, value))
        return reply(action, value)
    client._request = request
    return client, calls


def test_client_takes_the_settings_only_when_the_supervisor_accepts(tmp_path):
    new = settings_at(tmp_path, {'context': 96, 'floor_gib': 15, 'preferred_gib': 20})
    client, calls = client_with(tmp_path, lambda a, v: dict(ok=True, settings=dict(accepted=True, selected='cpu', changed=['context'])))
    assert client.update_settings(new)['changed'] == ['context']
    assert client.settings == new and client._forecast_cache is None and client._settings_generation == 1
    assert client.config()['context'] == 96 and calls[0][0] == 'settings' and calls[0][1]['restore'] is False
    refused = event('settings_invalid', 'refused', 'Not applied: nope')
    client, _ = client_with(tmp_path, lambda a, v: dict(ok=True, settings=dict(accepted=False, event=refused)))
    before = copy.deepcopy(client.settings)
    with pytest.raises(RuntimeRefusal, match='nope'):client.update_settings(new)
    assert client.settings == before and client._settings_generation == 0 and client.last_event == dict(code='earlier')


def reconciling_client(tmp_path, held):
    """A client whose settings request times out; asked again, the supervisor answers ``held(new, previous)``."""
    client = RuntimeClient(tmp_path, settings_at(tmp_path)); client.last_event = dict(code='earlier')
    calls = []
    def request(action, value=None, timeout=None):
        calls.append((action, value, timeout))
        if not value.get('query'):
            refusal = unavailable('The memory watchdog did not respond; this operation was not started.'); refusal.timed_out = True
            raise refusal
        answer = held(calls[0][1], client.settings)
        if answer is None:
            refusal = unavailable('The memory watchdog did not respond; this operation was not started.'); refusal.timed_out = True
            raise refusal
        return dict(ok=True, settings=dict(query=True, selected='cpu', running=False, loaded=True, load=None,
                                           last=dict(changed=['floor_gib'], effect=dict(kind='none')), **answer))
    client._request = request
    return client, calls


def test_a_late_answer_is_reconciled_by_asking_what_the_supervisor_runs(tmp_path):
    new = settings_at(tmp_path, {'floor_gib': 15, 'preferred_gib': 20})
    picked = lambda s: dict(values=s['values'], sources=s['sources'], notices=list(s.get('notices', ())))  # noqa: E731
    # Committed, answer late: the change took effect, and the client says so.
    client, calls = reconciling_client(tmp_path, lambda sent, previous: picked(sent))
    result = client.update_settings(new)
    assert result['accepted'] and result['late'] and result['changed'] == ['floor_gib'] and client.settings == new
    assert [c[0] for c in calls] == ['settings', 'settings'] and calls[1][1] == dict(query=True) and client.last_event == dict(code='earlier')
    # Still the previous settings: not applied, and it says it cannot apply later.
    client, _ = reconciling_client(tmp_path, lambda sent, previous: picked(previous))
    before = copy.deepcopy(client.settings)
    with pytest.raises(RuntimeRefusal, match='still runs the previous settings, so these were not applied and cannot apply later') as caught:
        client.update_settings(new)
    assert not isinstance(caught.value, SettingsUnsettled) and caught.value.record['code'] == 'watchdog_unavailable'
    assert client.settings == before and client._settings_generation == 0 and client.last_event == dict(code='earlier')
    # No answer to the question either: not known.
    client, _ = reconciling_client(tmp_path, lambda sent, previous: None)
    with pytest.raises(SettingsUnsettled, match='may be running the unsaved values until the next accepted save or a restart') as caught:
        client.update_settings(new)
    assert caught.value.held is None and client.settings == before and client._settings_generation == 1
    # Other settings: the client takes what the supervisor runs.
    other = settings_at(tmp_path, {'floor_gib': 10, 'preferred_gib': 12})
    client, _ = reconciling_client(tmp_path, lambda sent, previous: picked(other))
    with pytest.raises(SettingsUnsettled) as caught:
        client.update_settings(new)
    assert caught.value.held['values'] == other['values'] and client.settings['values'] == other['values']


def test_the_settings_request_carries_a_confirmation_and_a_refusal_keeps_its_effect(tmp_path):
    effect = dict(kind='unload', confirm=True, loaded=True, judged='j1', message='unloads')
    client, calls = client_with(tmp_path, lambda a, v: dict(ok=True, settings=dict(accepted=False, effect=effect,
                                                                                  event=event('settings_unconfirmed', 'refused', 'unloads'))))
    with pytest.raises(RuntimeRefusal) as caught:client.update_settings(settings_at(tmp_path, {'resident_seconds': 10}))
    assert caught.value.effect == effect and calls[0][1]['confirmed'] is None
    client, calls = client_with(tmp_path, lambda a, v: dict(ok=True, settings=dict(accepted=True, selected='cpu')))
    client.update_settings(settings_at(tmp_path, {'resident_seconds': 10}), confirmed='j1')
    client.update_settings(settings_at(tmp_path, {'resident_seconds': 11}), confirmed=True)   # only a fingerprint is sent
    assert calls[0][1]['confirmed'] == 'j1' and calls[1][1]['confirmed'] is None


def test_client_settings_wait_a_bounded_time_for_a_starting_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'SETTINGS_LOCK_SECONDS', .2)
    client, calls = client_with(tmp_path, lambda a, v: dict(ok=True, settings=dict(accepted=True)))
    held = threading.Event(); done = threading.Event()
    def hold():
        with client.lock:held.set();done.wait(5)
    threading.Thread(target=hold, daemon=True).start(); held.wait(5)
    try:
        with pytest.raises(RuntimeRefusal) as caught:client.update_settings(settings_at(tmp_path, {'context': 96}))
        assert caught.value.record['code'] == 'settings_busy' and calls == []
    finally:done.set()


def test_a_cold_start_is_described_and_begun_before_or_after_a_settings_change_never_between(tmp_path, monkeypatch):
    # update_settings needs the client lock, which the cold start holds from describing the operation to its grant.
    ws = fake_ws(tmp_path); ws.version = 0; ws.patch_stored = 0
    client = RuntimeClient(tmp_path, settings_at(tmp_path))
    began = threading.Event(); release = threading.Event(); order = []
    def request(action, value=None, timeout=None):
        if action == 'begin':
            order.append(('begin', value['configured_context'])); began.set(); release.wait(5)
        elif action == 'settings':order.append(('settings', value['values']['context']))
        return dict(ok=True, reserved=True, settings=dict(accepted=True, selected='cpu'))
    monkeypatch.setattr(client, '_request', request)
    monkeypatch.setattr(client, 'require', lambda ws, *a, **k: None)
    monkeypatch.setattr(client, '_check', lambda binding: None)
    def run():
        with client.operation(ws, 'generate', tokens=4):order.append(('ran', client.config()['context']))
    worker = threading.Thread(target=run, daemon=True); worker.start(); began.wait(5)
    saver = threading.Thread(target=lambda: client.update_settings(settings_at(tmp_path, {'context': 96})), daemon=True); saver.start()
    time.sleep(.3)
    assert order == [('begin', 128)]                                             # the save waits for the grant
    release.set(); worker.join(5); saver.join(5)
    assert order[:2] == [('begin', 128), ('ran', 128)] or order[:2] == [('begin', 128), ('settings', 96)]
    assert ('settings', 96) in order and ('ran', 128) in order                    # the running operation kept its pin
    assert client.config()['context'] == 96 and client._pinned is None            # no model loaded: the pin ended with it


def test_an_operation_sized_by_its_pinned_load_settings_is_sized_under_the_client_lock(tmp_path, monkeypatch):
    ws = fake_ws(tmp_path); ws.version = 0; ws.patch_stored = 0
    client = RuntimeClient(tmp_path, settings_at(tmp_path))
    monkeypatch.setattr(client, '_request', lambda action, value=None, timeout=None: dict(ok=True, reserved=True))
    monkeypatch.setattr(client, 'require', lambda ws, *a, **k: None)
    monkeypatch.setattr(client, '_check', lambda binding: None)
    described = []; original = client._value
    def value(ws, **kw):
        described.append(kw['tokens']); return original(ws, **kw)
    monkeypatch.setattr(client, '_value', value)
    seen = []
    def size(load):
        seen.append((load, client.lock._is_owned())); return 77
    client.settings = settings_at(tmp_path, {'context': 96})                     # saved after the route's check
    with client.operation(ws, 'completion', tokens=5, size=size):pass
    assert seen == [(dict(context=96, threads=4, backend='auto', selected='cpu'), True)] and described == [77]
    # Without size, the tokens given are used.
    with client.operation(ws, 'completion', tokens=5):pass
    assert described == [77, 5]
    # A loaded model keeps its pin: a later saved context does not size work on it.
    with client.operation(ws, 'load'):ws._runner = SimpleNamespace(lost=None, lock=threading.RLock(), release=lambda: None)
    client.settings = settings_at(tmp_path, {'context': 64})
    with client.operation(ws, 'completion', tokens=5, size=size):pass
    assert seen[-1][0]['context'] == 96
    # Nested inside a running operation: sized the same way, under the lock, before the extension is described.
    with client.operation(ws, 'generate', tokens=5):
        with client.operation(ws, 'completion', tokens=5, size=size):pass
    assert seen[-1] == (dict(context=96, threads=4, backend='auto', selected='cpu'), True) and described[-1] == 77


def test_an_advisory_answer_across_a_settings_change_is_dropped(tmp_path, monkeypatch):
    ws = fake_ws(tmp_path); ws.version = 0; ws.patch_stored = 0
    client = RuntimeClient(tmp_path, settings_at(tmp_path))
    def supervisor():
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            try:request = json.loads((tmp_path/'request.json').read_text())
            except (FileNotFoundError, ValueError):time.sleep(.01);continue
            client._settings_generation += 1                                     # a save lands while advise waits
            write(tmp_path/'reply.json', dict(id=request['id'], ok=True, advice=dict(largest=5, event=None)))
            return
    threading.Thread(target=supervisor, daemon=True).start()
    advice = client._advise(ws, R.SCRATCH_ADMISSION_QUANTUM, time.monotonic()+1.5)
    assert advice['largest'] is None and 'settings changed' in advice['reason']


def test_pins_follow_the_loaded_model(tmp_path, monkeypatch):
    ws = fake_ws(tmp_path); ws.version = 0; ws.patch_stored = 0
    client = RuntimeClient(tmp_path, settings_at(tmp_path))
    monkeypatch.setattr(client, '_request', lambda action, value=None, timeout=None: dict(ok=True, reserved=True, settings=dict(accepted=True, selected='cpu')))
    monkeypatch.setattr(client, 'require', lambda ws, *a, **k: None)
    monkeypatch.setattr(client, '_check', lambda binding: None)
    with client.operation(ws, 'load'):ws._runner = SimpleNamespace(lost=None, lock=threading.RLock(), release=lambda: None)
    assert client.loaded_config(ws)['context'] == 128
    client.update_settings(settings_at(tmp_path, {'context': 96}))
    assert client.config()['context'] == 128 and client.status(ws)['next_load']['context'] == 96   # loaded: the pin holds
    client.release()
    assert client.config()['context'] == 96 and client._pinned is None
    # A runner dropped without release (a lost model worker) leaves its pin; the next cold start is a new load.
    with client.operation(ws, 'load'):ws._runner = SimpleNamespace(lost=None, lock=threading.RLock(), release=lambda: None)
    assert client.loaded_config(ws)['context'] == 96
    client.update_settings(settings_at(tmp_path, {'context': 80}))
    ws._runner = None
    sent = []
    def record(action, value=None, timeout=None):
        sent.append((action, value));return dict(ok=True, reserved=True)
    monkeypatch.setattr(client, '_request', record)
    with client.operation(ws, 'load'):pass
    assert [v['configured_context'] for a, v in sent if a == 'begin'] == [80]
    # A refused begin with no model loaded leaves no pin behind.
    def refuse(action, value=None, timeout=None):raise RuntimeRefusal(event('forecast_exceeds_budget', 'refused', 'no'))
    monkeypatch.setattr(client, '_request', refuse); ws._runner = None
    with pytest.raises(RuntimeRefusal):
        with client.operation(ws, 'load'):pass
    assert client._pinned is None


# ------------------------------------------------------------------ the routes of an embedded server
class EmbeddedRuntime(Supervisor):
    def __init__(self, settings, **kwargs):
        super().__init__(settings, **kwargs); self.checks = []
    def status(self, ws=None): return dict(supervised=True, limits=self.settings['values'])
    def check_settings(self, settings):
        self.checks.append(copy.deepcopy(settings['values']))
        return dict(accepted=True, check=True, effect=self.effect)
    def release(self): pass


def embedded(tmp_path, runtime_kwargs=None):
    settings = settings_at(tmp_path)
    sess = Session(str(tmp_path/'work')); sess.settings = settings
    sess.runtime = EmbeddedRuntime(settings, **(runtime_kwargs or {}))
    app = create_app(sess, URL); app.state.settings_telemetry = lambda: dict(total=128*GiB, available=71*GiB, free=10*GiB, pressure=1)
    return sess, app


def test_save_route_reports_what_happened_and_refusals_say_not_saved(tmp_path):
    sess, app = embedded(tmp_path)
    with TestClient(app, base_url=URL) as c:
        view = c.get('/api/settings').json(); headers = {'X-WSP-Settings': view['csrf'], 'Origin': URL}
        assert view['supervised'] and view['takes_effect']['preferred_gib'] == 'saved' and 'Saving applies settings to this server' in view['note']
        saved = c.post('/api/settings', json={'revision': view['revision'], 'changes': {'preferred_gib': 20, 'floor_gib': 15, 'threads': 1}}, headers=headers)
        assert saved.status_code == 200, saved.text
        body = saved.json()
        assert body['applied']['now'] == ['floor_gib', 'preferred_gib'] and body['applied']['next_load'] == ['threads']
        assert sess.runtime.settings['values']['preferred_gib'] == 20 and app.state.settings_store.active == sess.runtime.settings
        sess.runtime.refuse = RuntimeRefusal(event('settings_invalid', 'refused', 'Not applied: bad'))
        before = app.state.settings_store.path.read_bytes()
        refused = c.post('/api/settings', json={'revision': body['revision'], 'changes': {'floor_gib': 19}}, headers=headers)
        assert refused.status_code == 409 and refused.json()['detail'] == 'Not saved: Not applied: bad'
        sess.runtime.refuse = unavailable('The memory watchdog did not answer within 8 s; the settings were not applied.')
        missing = c.post('/api/settings', json={'revision': body['revision'], 'changes': {'floor_gib': 19}}, headers=headers)
        assert missing.status_code == 503 and missing.json()['detail'].startswith('Not saved: The memory watchdog did not answer')
        assert app.state.settings_store.path.read_bytes() == before and sess.runtime.settings['values']['floor_gib'] == 15


def test_preview_asks_the_supervisor_and_warns_before_an_unloading_save(tmp_path):
    sess, app = embedded(tmp_path)
    sess.runtime.effect = dict(kind='unload', code='headroom_floor', seconds=3,
                               message='With these values the memory watchdog unloads the loaded model if this lasts 3 s. Available memory is below the runtime floor.')
    with TestClient(app, base_url=URL) as c:
        view = c.get('/api/settings').json(); headers = {'X-WSP-Settings': view['csrf'], 'Origin': URL}
        assert sess.runtime.checks == []                                          # nothing to check without a draft
        out = c.post('/api/settings/preview', json={'revision': view['revision'], 'changes': {'floor_gib': 75, 'preferred_gib': 75}}, headers=headers).json()
        assert sess.runtime.checks[-1]['floor_gib'] == 75 and out['watchdog_check']['accepted'] is True
        assert any('unloads the loaded model' in w for w in out['preview']['warnings'])
        # Output tokens and the patch limit need no supervisor check.
        c.post('/api/settings/preview', json={'revision': view['revision'], 'changes': {'new_tokens': 5}}, headers=headers)
        assert len(sess.runtime.checks) == 1


def test_a_settings_save_does_not_wait_behind_a_running_runtime_request(tmp_path):
    sess, app = embedded(tmp_path)
    entered = threading.Event(); finish = threading.Event()
    def slow_unload():entered.set();finish.wait(10)
    sess.unload = slow_unload
    with TestClient(app, base_url=URL) as c:
        view = c.get('/api/settings').json(); headers = {'X-WSP-Settings': view['csrf'], 'Origin': URL}
        holder = threading.Thread(target=lambda: c.post('/api/runtime/unload', json={}, headers={'Origin': URL}), daemon=True); holder.start()
        assert entered.wait(5)
        try:
            started = time.monotonic()
            saved = c.post('/api/settings', json={'revision': view['revision'], 'changes': {'resident_seconds': 60}}, headers=headers)
            assert saved.status_code == 200 and time.monotonic()-started < 3
            assert sess.runtime.settings['values']['resident_seconds'] == 60
        finally:finish.set();holder.join(5)


def test_a_save_that_would_unload_the_loaded_model_is_refused_until_it_is_confirmed(tmp_path):
    effect = dict(kind='unload', code='idle_expired', seconds=1, confirm=True, loaded=True, judged='j1',
                  message='With these values the memory watchdog unloads the loaded model within about a second.')
    sess, app = embedded(tmp_path, dict(effect=effect))
    with TestClient(app, base_url=URL) as c:
        view = c.get('/api/settings').json(); headers = {'X-WSP-Settings': view['csrf'], 'Origin': URL}
        before = app.state.settings_store.path.read_bytes()
        body = {'revision': view['revision'], 'changes': {'resident_seconds': 10}}
        asked = c.post('/api/settings', json=body, headers=headers)
        assert asked.status_code == 409, asked.text
        detail = asked.json()['detail']
        assert detail['effect'] == effect and detail['confirm'] and 'unloads the loaded model' in detail['message'] and 'confirm' in detail['message']
        assert app.state.settings_store.path.read_bytes() == before and sess.runtime.settings['values']['resident_seconds'] == 14400
        # A confirmation of this candidate does not authorise another one.
        other = c.post('/api/settings', json={**body, 'changes': {'resident_seconds': 11}, 'confirm': detail['confirm']}, headers=headers)
        assert other.status_code == 409 and other.json()['detail']['confirm'] != detail['confirm']
        assert sess.runtime.settings['values']['resident_seconds'] == 14400
        assert c.post('/api/settings', json={**body, 'confirm': 5}, headers=headers).status_code == 400
        assert c.post('/api/settings/preview', json={**body, 'confirm': detail['confirm']}, headers=headers).status_code == 400
        saved = c.post('/api/settings', json={**body, 'confirm': detail['confirm']}, headers=headers)
        assert saved.status_code == 200, saved.text
        assert sess.runtime.calls[-1][2] == 'j1' and sess.runtime.settings['values']['resident_seconds'] == 10
        assert saved.json()['applied']['effect'] == effect


def test_the_preview_says_saving_asks_first(tmp_path):
    sess, app = embedded(tmp_path)
    sess.runtime.effect = dict(kind='after_operation', unload=True, code='headroom_floor', confirm=True, loaded=True,
                               message='A model operation is running. When it finishes, the memory watchdog would then unload the loaded model.')
    with TestClient(app, base_url=URL) as c:
        view = c.get('/api/settings').json(); headers = {'X-WSP-Settings': view['csrf'], 'Origin': URL}
        out = c.post('/api/settings/preview', json={'revision': view['revision'], 'changes': {'floor_gib': 75, 'preferred_gib': 75}}, headers=headers).json()
        assert any('would then unload the loaded model' in w and 'Saving asks you to confirm first' in w for w in out['preview']['warnings'])


def test_the_runtime_status_does_not_wait_for_a_save_in_flight_and_is_never_torn(tmp_path):
    sess, app = embedded(tmp_path)
    store = app.state.settings_store
    entered = threading.Event(); finish = threading.Event()
    with TestClient(app, base_url=URL) as c:
        assert c.get('/api/runtime').json()['settings_summary']['saved']['context'] == 128
        def in_flight():
            with store.lock:
                # As a save does: the file is replaced while the lock is held, the active settings only after.
                store.path.write_text(store.path.read_text().replace('"context": 128', '"context": 96'))
                entered.set(); finish.wait(10)
        holder = threading.Thread(target=in_flight, daemon=True); holder.start()
        assert entered.wait(5)
        try:
            started = time.monotonic()
            summary = c.get('/api/runtime').json()['settings_summary']
            assert time.monotonic()-started < 2
            # The state as it was when the lock was last released: the old file with the old active settings.
            assert summary['saved']['context'] == summary['active']['context'] == 128 and summary['pending'] == {}
        finally:finish.set();holder.join(5)
        assert c.get('/api/runtime').json()['settings_summary']['saved']['context'] == 96


# ------------------------------------------------------------------ Run and Chat sized by the pinned load settings
class PinnedRuntime(EmbeddedRuntime):
    """Loaded with ``pin``; ``pin_at_begin`` (when set) is what a cold begin pins, as if a save landed meanwhile."""
    def __init__(self, settings, pin, pin_at_begin=None):
        super().__init__(settings); self.pin = pin; self.pin_at_begin = pin_at_begin; self.tokens = []
    def config(self): return self.pin
    @contextmanager
    def operation(self, ws, name, *, tokens=0, size=None, **kwargs):
        # As RuntimeClient.operation: size, given the pinned load settings, replaces the tokens computed before.
        if self.pin_at_begin is not None:self.pin = self.pin_at_begin
        self.tokens.append(size(dict(self.pin)) if size is not None else tokens)
        yield


def gguf_session(tmp_path, runtime_for):
    settings = settings_at(tmp_path, {'context': 64})
    sess = Session(str(tmp_path/'work')); sess.settings = settings
    sess.runtime = runtime_for(settings)
    calls = []
    def generate(ids, variant, extra=None, **kw):
        calls.append((variant, kw['max_new_tokens'])); return [9]
    tok = SimpleNamespace(token_str=str, to_dict=lambda: {'mode': 'word'}, encode=lambda *a, **k: [1, 2], decode=lambda ids: ' '.join(map(str, ids)))
    runner = SimpleNamespace(next_logits=lambda *a: torch.zeros(3), chat_prompt=lambda *a, **k: 'whole chat', generate=generate)
    sess.ws = SimpleNamespace(kind='gguf', lock=threading.RLock(), _runtime_client=sess.runtime, struct=SimpleNamespace(n_ctx=512), can_run=True,
                              tok=tok, runner=runner, encode=lambda _: [1, 2], complete=lambda p, v, **kw: tok.decode(generate([1, 2], v, **kw)))
    app = create_app(sess, URL)
    return sess, app, calls


@pytest.mark.parametrize('route,body', [('/api/run', {'prompt': 'test'}), ('/api/chat', {'messages': [{'role': 'user', 'content': 'test'}]})])
def test_run_and_chat_use_the_loaded_models_context_not_the_saved_one(tmp_path, route, body):
    pin = dict(context=32, threads=4, backend='cpu', selected='cpu')
    sess, app, calls = gguf_session(tmp_path, lambda s: PinnedRuntime(s, pin))
    with TestClient(app, base_url=URL) as c:
        assert app.state.settings_store.active['values']['context'] == 64
        assert c.get('/api/runtime').json()['generation']['context_limit'] == 32
        out = c.post(route, json={**body, 'max_new_tokens': 1000}).json()
    assert out['generation']['context'] == 32 and out['generation']['effective_output'] == 30
    assert set(calls) == {('current', 30), ('base', 30)} and sess.runtime.tokens == [32]


@pytest.mark.parametrize('route,body', [('/api/run', {'prompt': 'test'}), ('/api/chat', {'messages': [{'role': 'user', 'content': 'test'}]})])
def test_a_context_saved_between_the_check_and_the_begin_sizes_the_request_by_the_pinned_one(tmp_path, route, body):
    # Checked with the next-load context 64; a save pins 40 before the operation begins.
    before = dict(context=64, threads=4, backend='cpu', selected='cpu'); pinned = dict(before, context=40)
    sess, app, calls = gguf_session(tmp_path, lambda s: PinnedRuntime(s, before, pin_at_begin=pinned))
    with TestClient(app, base_url=URL) as c:
        out = c.post(route, json=body).json()                                      # the default output follows too
        # The default output is the pinned context's (39, not 63) and the scope is sized for it: 2 + 38 tokens.
        assert out['generation']['context'] == 40 and out['generation']['requested_output'] == 39 and out['generation']['effective_output'] == 38
        assert sess.runtime.tokens == [40]
        sess.runtime.pin = before
        out = c.post(route, json={**body, 'max_new_tokens': 1000}).json()
    assert out['generation']['context'] == 40 and out['generation']['effective_output'] == 38 and sess.runtime.tokens[-1] == 40
    assert ('current', 38) in calls


def test_after_a_save_a_status_read_during_the_next_save_shows_that_save(tmp_path):
    settings = settings_at(tmp_path); store = SettingsStore(settings); runtime = Supervisor(settings)
    rev = store.snapshot()['revision']                                          # the consistent state before the save
    store.save(rev, {'context': 96}, runtime=runtime)
    entered = threading.Event(); finish = threading.Event()
    def next_save():
        with store.lock:entered.set();finish.wait(10)
    holder = threading.Thread(target=next_save, daemon=True); holder.start()
    assert entered.wait(5)
    try:
        view = store.snapshot(runtime)                                          # served without the lock
        assert view['revision'] != rev and view['saved']['values']['context'] == view['active']['values']['context'] == 96
    finally:finish.set();holder.join(5)
