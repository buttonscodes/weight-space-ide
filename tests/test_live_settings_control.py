"""Live settings end to end: the real settings route, server child, supervisor and model worker spawn (scalar fakes).

No checkpoint, model constructor, native library or tensor. Each test has its own temp root and an OS-assigned port.
"""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

import test_runtime_control as C
from test_runtime_control import alive, log_rows, read, wait, worker_load
from wsp.memory_guard import GiB

# The memory settings are the user's saved configuration, not launch overrides: only the backend is fixed.
LAUNCH = "overrides=dict(backend='cpu',context=128,floor_gib=20,preferred_gib=25,resident_seconds=30)"
SAVED = dict(defaults=dict(context=128, floor_gib=20, preferred_gib=25, resident_seconds=30), profiles={})


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    assert LAUNCH in C.SUPERVISOR
    monkeypatch.setattr(C, 'SUPERVISOR', C.SUPERVISOR.replace(LAUNCH, "overrides=dict(backend='cpu')"))
    with C.control_fixture(tmp_path, config=SAVED) as f:
        yield f


def http(f, method, path, body=None, headers=None):
    url = f"http://127.0.0.1:{read(f.root/'listener.json')['port']}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url+path, data=data, method=method,
                                     headers={'Content-Type': 'application/json', 'Origin': url, **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b'{}')


def settings_page(f):
    status, view = http(f, 'GET', '/api/settings')
    assert status == 200, view
    return view


def save(f, changes, view=None):
    view = view or settings_page(f)
    return http(f, 'POST', '/api/settings', dict(revision=view['revision'], changes=changes), {'X-WSP-Settings': view['csrf']})


def supervisor_settings(f):
    return read(f.folder/'settings.json')['values']


def test_the_owners_case_lower_the_reserve_and_load_again_without_a_restart(fixture):
    f = fixture
    server = f.heartbeat()['pid']
    f.telemetry(available=24*GiB)                                   # less free memory than the 25 GiB reserve
    refused = f.command('worker-load', seconds=20)
    assert not refused['ok'] and refused['event']['code'] == 'forecast_exceeds_budget'
    view = settings_page(f)
    assert view['supervised'] and view['takes_effect']['preferred_gib'] == 'saved'
    status, saved = save(f, dict(preferred_gib=20, floor_gib=15), view)
    assert status == 200, saved
    assert saved['applied']['now'] == ['floor_gib', 'preferred_gib'] and saved['applied']['next_start'] == []
    assert not saved['restart_required'] and saved['pending'] == {}
    # The supervisor, the server's client and the settings file agree.
    assert supervisor_settings(f)['preferred_gib'] == 20 and supervisor_settings(f)['floor_gib'] == 15
    assert saved['active']['values']['preferred_gib'] == 20
    assert f.api('/api/runtime')['limits']['preferred_gib'] == 20
    worker_load(f)                                                 # the same load, admitted
    assert f.heartbeat()['pid'] == server and f.process.poll() is None
    assert read(f.folder/'status.json')['policy']['preferred'] == 20*GiB
    assert [r['event'] for r in log_rows(f)].count('settings_changed') == 1


def test_load_settings_reach_the_next_worker_and_the_loaded_model_keeps_its_own(fixture):
    f = fixture
    first = worker_load(f)
    assert read(f.root/f"worker-{first['pid']}.json")['env']['WSP_MAX_CTX'] == '128'
    status, saved = save(f, dict(context=64, threads=1))
    assert status == 200, saved
    assert saved['applied']['next_load'] == ['context', 'threads'] and saved['applied']['now'] == []
    assert saved['applied']['loaded']['context'] == 128
    runtime = f.api('/api/runtime')
    assert runtime['loaded_config']['context'] == 128 and runtime['next_load']['context'] == 64 and runtime['next_load']['threads'] == 1
    # Work on the loaded model is admitted with the settings it was loaded with.
    assert f.command('probe')['ok']
    begins = [r for r in log_rows(f) if r['event'] == 'begin']
    assert begins[-1]['descriptor']['configured_context'] == 128
    # Unloaded (the worker stops, the server releases): the next load uses the saved settings, everywhere at once.
    assert f.command('worker-shutdown')['ok']
    wait(lambda: 'worker_stopped' in [r['event'] for r in log_rows(f)], 8)
    assert not alive(first['pid'], first['create_time'])
    assert f.api('/api/runtime')['loaded_config']['context'] == 128    # still reserved, so still the loaded settings
    assert f.command('release-keep')['ok']
    assert f.api('/api/runtime')['loaded_config'] is None
    second = worker_load(f)
    env = read(f.root/f"worker-{second['pid']}.json")['env']
    assert env['WSP_MAX_CTX'] == '64' and env['WSP_GGML_THREADS'] == '1' and env['WSP_GGML_DEVICE'] == 'cpu'
    load = [r for r in log_rows(f) if r['event'] == 'begin'][-1]['descriptor']
    assert load['configured_context'] == 64 and load['context'] == 64
    assert load['forecast'] != begins[0]['descriptor']['forecast']           # the forecast is for the new context
    assert f.api('/api/runtime')['loaded_config']['context'] == 64


def test_a_save_during_a_load_applies_after_it_and_the_load_keeps_its_settings(fixture):
    f = fixture
    n = f.send('worker-load', hold=True)
    wait(lambda: f.heartbeat()['operation_id'], 20)
    wait(lambda: [r for r in log_rows(f) if r['event'] == 'spawn_worker'], 20)
    started = time.monotonic()
    status, saved = save(f, dict(preferred_gib=30, floor_gib=20, context=64))
    assert status == 200, saved
    assert time.monotonic()-started < 8                             # the load is not waited for
    assert saved['applied']['running'] and saved['applied']['effect']['kind'] == 'after_operation'
    assert saved['applied']['now'] == ['preferred_gib'] and saved['applied']['next_load'] == ['context']
    spawn = [r for r in log_rows(f) if r['event'] == 'spawn_worker'][-1]
    assert wait(lambda: read(f.root/f"worker-{spawn['worker']['pid']}.json"))['env']['WSP_MAX_CTX'] == '128'
    assert read(f.folder/'status.json')['policy']['preferred'] == 25*GiB  # the running load's admitted limits
    (f.root/'release-hang').touch()
    assert f.response(n, 20)['ok']
    wait(lambda: read(f.folder/'status.json')['state'] == 'idle')
    status_file = read(f.folder/'status.json')
    assert status_file['watch']['preferred'] == 30*GiB and status_file['load']['context'] == 128
    assert status_file['next_load']['context'] == 64
    changed = [r for r in log_rows(f) if r['event'] == 'settings_changed']
    assert len(changed) == 1 and changed[0]['operation_id'] and changed[0]['pinned']['context'] == 128


def test_two_saves_close_together_one_wins_and_the_other_changes_nothing(fixture):
    f = fixture
    view = settings_page(f)
    results = {}
    def post(name, changes):results[name] = save(f, changes, view)
    threads = [threading.Thread(target=post, args=('a', dict(preferred_gib=22, floor_gib=18))),
               threading.Thread(target=post, args=('b', dict(preferred_gib=24, floor_gib=19)))]
    for t in threads:t.start()
    for t in threads:t.join(30)
    codes = sorted(status for status, _ in results.values())
    assert codes == [200, 409], results
    winner = next(name for name, (status, _) in results.items() if status == 200)
    expected = dict(a=(22, 18), b=(24, 19))[winner]
    assert (supervisor_settings(f)['preferred_gib'], supervisor_settings(f)['floor_gib']) == expected
    page = settings_page(f)
    assert (page['active']['values']['preferred_gib'], page['saved']['values']['preferred_gib']) == (expected[0], expected[0])
    assert [r['event'] for r in log_rows(f)].count('settings_changed') == 1


def test_an_invalid_save_through_the_route_writes_nothing_and_changes_no_process(fixture):
    # The route validates against the server's machine; the supervisor refusal itself is in test_live_settings_supervisor.
    f = fixture
    view = settings_page(f)
    before = (f.folder/'settings.json').read_bytes()
    with open(view['config'], 'rb') as file:saved_before = file.read()
    status, body = save(f, dict(preferred_gib=4096, floor_gib=20), view)
    assert status == 400 and 'physical RAM' in body['error'], body
    assert (f.folder/'settings.json').read_bytes() == before
    with open(view['config'], 'rb') as file:assert file.read() == saved_before
    assert settings_page(f)['active']['values']['preferred_gib'] == 25
    assert 'settings_changed' not in [r['event'] for r in log_rows(f)]


def test_a_supervisor_that_cannot_record_the_settings_applies_nothing(fixture):
    f = fixture
    view = settings_page(f)
    with open(view['config'], 'rb') as file:saved_before = file.read()
    before = (f.folder/'settings.json').read_bytes()
    f.telemetry(deny_names=['settings.json'])
    status, body = save(f, dict(preferred_gib=20, floor_gib=15), view)
    assert status == 409 and body['detail'].startswith('Not saved: The memory watchdog could not record the settings'), body
    assert (f.folder/'settings.json').read_bytes() == before
    with open(view['config'], 'rb') as file:assert file.read() == saved_before
    assert settings_page(f)['active']['values']['preferred_gib'] == 25
    f.telemetry(deny_names=[])
    status, body = save(f, dict(preferred_gib=20, floor_gib=15), view)
    assert status == 200 and supervisor_settings(f)['preferred_gib'] == 20


def test_a_settings_answer_that_is_not_delivered_is_undone(fixture):
    f = fixture
    view = settings_page(f)
    with open(view['config'], 'rb') as file:saved_before = file.read()
    f.telemetry(deny_reply_once='settings')
    status, body = save(f, dict(preferred_gib=20, floor_gib=15), view)
    assert status == 409 and body['detail'].startswith('Not saved:'), body
    assert supervisor_settings(f)['preferred_gib'] == 25                 # the supervisor went back
    events = [r['event'] for r in log_rows(f)]
    assert events.index('settings_changed') < events.index('settings_restored')
    with open(view['config'], 'rb') as file:assert file.read() == saved_before
    assert settings_page(f)['active']['values']['preferred_gib'] == 25


def test_a_save_that_would_unload_the_loaded_model_asks_first_through_the_real_route(fixture):
    f = fixture
    worker_load(f)
    view = settings_page(f)
    with open(view['config'], 'rb') as file:saved_before = file.read()
    status, body = save(f, dict(floor_gib=75, preferred_gib=80), view)    # above the 71 GiB available
    assert status == 409, body
    detail = body['detail']
    assert detail['effect']['code'] == 'headroom_floor' and detail['effect']['confirm'] and 'unloads the loaded model' in detail['message']
    assert supervisor_settings(f)['floor_gib'] == 20 and settings_page(f)['active']['values']['floor_gib'] == 20
    with open(view['config'], 'rb') as file:assert file.read() == saved_before
    assert 'settings_changed' not in [r['event'] for r in log_rows(f)] and f.heartbeat()['resident']
    status, body = http(f, 'POST', '/api/settings', dict(revision=view['revision'], changes=dict(floor_gib=75, preferred_gib=80),
                                                       confirm=detail['confirm']), {'X-WSP-Settings': view['csrf']})
    assert status == 200, body
    assert supervisor_settings(f)['floor_gib'] == 75 and body['active']['values']['floor_gib'] == 75
    assert body['applied']['effect']['code'] == 'headroom_floor'
    changed = [r for r in log_rows(f) if r['event'] == 'settings_changed']
    assert len(changed) == 1 and changed[0]['confirmed'] is True
