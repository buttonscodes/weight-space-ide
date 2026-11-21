"""Stopping a Run or Chat request: the registry, the callbacks, the routes and the full worker stack. No model.

The route tests use fake runners in two styles: one lets ``GenerationStopped`` leave
``generate`` (as ``FunctionalRunner`` and an in-process ``GGUFRunner`` do), the other
returns early with the tokens so far (as ``HFRunner`` and ``RemoteRunner`` do). A tiny
random in-memory model checks the real ``FunctionalRunner``. The full-stack tests run the
supervisor, the server and the stub model worker of ``test_remote_runner``.
"""
from contextlib import contextmanager
import json
import threading
import time
from threading import RLock
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from test_settings import active
from test_token_stream import Runtime, apply
from wsp import generation_stop
from wsp.generation_stop import GenerationStopped, Stop, Stops
from wsp.runner import HFRunner, TokenCallback
from wsp.server import create_app
from wsp.session import Session
from wsp.token_stream import TextStream

URL = 'http://127.0.0.1:41882'


# ---------------------------------------------------------------- registry
def test_a_running_request_is_found_and_a_finished_one_is_not():
    stops = Stops()
    with stops.open('a') as stop:
        assert stops.get('a') is stop and not stop.is_set()
        assert stops.stop('a') is True and stop.is_set()
        assert stops.stop('a') is True                      # a second stop is not an error
    assert stops.get('a') is None and stops.stop('a') is False
    with stops.open('a') as again:                          # a finished id is not remembered as stopped
        assert not again.is_set()
    with stops.open(None) as nothing:
        assert nothing is None


def test_a_stop_that_arrives_before_its_request_applies_when_it_arrives(monkeypatch):
    stops = Stops()
    assert stops.stop('early') is False
    with stops.open('early') as stop:
        assert stop.is_set()
    with stops.open('early') as again:                      # used once: the next request with that id runs
        assert not again.is_set()
    # However long the request waits behind another operation, the stop still applies.
    assert stops.stop('queued') is False
    later = time.monotonic()+24*3600
    monkeypatch.setattr(time, 'monotonic', lambda: later)
    with stops.open('queued') as stop:
        assert stop.is_set()


def test_a_stream_id_in_use_cannot_be_registered_twice():
    stops = Stops()
    with stops.open('a'):
        with pytest.raises(ValueError, match='already used'):
            with stops.open('a'):
                pass
        assert stops.get('a') is not None
    assert stops.get('a') is None


def test_remembered_ids_are_bounded():
    stops = Stops()
    for n in range(generation_stop.REMEMBERED+50):
        stops.stop(f'x{n}')
        with stops.open(f'y{n}'):
            pass
    assert len(stops.early) == len(stops.finished) == generation_stop.REMEMBERED


# ---------------------------------------------------------------- callbacks
def test_the_live_text_callback_raises_only_once_stop_is_set_and_records_every_token():
    events, stop = [], Stop()
    live = TextStream(events.append, 's', 'current', lambda ids: ''.join(chr(i) for i in ids), stop=stop, interval=0)
    live(97)
    assert not live.stopped
    stop.set()
    with pytest.raises(GenerationStopped):
        live(98)
    with pytest.raises(GenerationStopped):
        live(99)
    assert live.stopped and live.ids == [97, 98, 99]
    live.end('stopped')
    assert events[-1] == dict(type='generation', stream='s', variant='current', state='stopped', tokens=3)
    assert apply(events) == 'abc'


def test_without_a_stop_the_live_text_callback_never_raises():
    live = TextStream(lambda e: None, 's', 'current', str, interval=0)
    live(1)
    assert not live.stopped


def test_stop_files_name_their_job_and_never_raise(tmp_path):
    assert generation_stop.requested(tmp_path, 'job1') is False
    assert generation_stop.request(tmp_path, 'job1') is True
    assert generation_stop.requested(tmp_path, 'job1') and not generation_stop.requested(tmp_path, 'job2')
    generation_stop.clear(tmp_path, 'job2')                 # another job's request stays
    assert generation_stop.requested(tmp_path, 'job1')
    generation_stop.clear(tmp_path, 'job1')
    assert not (tmp_path/generation_stop.FILE).exists() and list(tmp_path.iterdir()) == []
    assert generation_stop.request(tmp_path/'missing', 'job') is False and generation_stop.request(None, 'job') is False
    assert generation_stop.requested(None, 'job') is False
    (tmp_path/generation_stop.FILE).write_text('not json')
    assert generation_stop.requested(tmp_path, 'job') is False
    generation_stop.clear(None, 'job')


def test_the_transformers_callback_turns_a_stop_into_a_stopping_criterion():
    seen = []
    def on_token(token):
        seen.append(token)
        if token == 8:
            raise GenerationStopped()
    callback = TokenCallback(on_token)
    callback.put(torch.tensor([[1, 2]]))                    # the prompt
    ids = torch.tensor([[1, 2, 7]])
    callback.put(torch.tensor([7]))
    assert callback(ids, None).tolist() == [False]
    callback.put(torch.tensor([8]))                         # raises inside: caught, not passed to transformers
    done = callback(ids, None)
    assert done.dtype == torch.bool and done.tolist() == [True] and seen == [7, 8]


def test_the_stopping_criterion_merges_with_transformers_own_criteria():
    from transformers import GenerationMixin
    from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList
    callback = TokenCallback(lambda token: None)
    merged = GenerationMixin._merge_criteria_processor_list(None, StoppingCriteriaList([MaxLengthCriteria(50)]),
                                                            StoppingCriteriaList([callback]))
    ids = torch.ones(1, 5, dtype=torch.long)
    assert merged(ids, None).tolist() == [False]
    callback.stopped = True
    assert merged(ids, None).tolist() == [True]


def fake_transformers_generate(seen, order='criteria_first'):
    """The decoding loop of transformers' ``_sample``: a criterion check and a streamer put per token."""
    def generate(input_ids, attention_mask=None, pad_token_id=0, max_new_tokens=8, streamer=None, stopping_criteria=None, **kw):
        seen.update(kw, streamer=streamer, stopping_criteria=stopping_criteria)
        if streamer is not None:
            streamer.put(input_ids)
        ids = input_ids
        for step in range(max_new_tokens):
            token = torch.tensor([30+step])
            ids = torch.cat([ids, token[:, None]], -1)
            if order == 'put_first' and streamer is not None:
                streamer.put(token)
            done = stopping_criteria(ids, None) if stopping_criteria is not None else torch.tensor([False])
            if order == 'criteria_first' and streamer is not None:
                streamer.put(token)
            if done.all():
                break
        return ids
    return generate


@pytest.mark.parametrize('order', ['criteria_first', 'put_first'])
def test_the_transformers_runner_ends_generate_normally_after_a_stop(order):
    """transformers 5 checks the criteria before putting the token; older versions put first."""
    seen, stop = {}, Stop()
    config = SimpleNamespace(pad_token_id=0, eos_token_id=0, num_beams=1)
    runner = SimpleNamespace(lock=RLock(), use=lambda *a: None, device='cpu',
                             model=SimpleNamespace(generation_config=config, generate=fake_transformers_generate(seen, order)))
    live = TextStream(lambda e: None, 's', 'current', str, stop=stop, interval=0)
    def on_token(token):
        if token == 32:
            stop.set()
        live(token)
    out = HFRunner.generate(runner, [1, 2], max_new_tokens=8, on_token=on_token)
    # The stop is seen on token 32; generation ends there or one token later, and returns normally.
    assert out[:3] == [30, 31, 32] and len(out) <= 4 and live.ids == out and live.stopped
    assert isinstance(seen['streamer'], TokenCallback) and list(seen['stopping_criteria']) == [seen['streamer']]


def test_the_transformers_runner_without_a_callback_gets_no_criterion():
    seen = {}
    config = SimpleNamespace(pad_token_id=0, eos_token_id=0, num_beams=1)
    runner = SimpleNamespace(lock=RLock(), use=lambda *a: None, device='cpu',
                             model=SimpleNamespace(generation_config=config, generate=fake_transformers_generate(seen)))
    assert HFRunner.generate(runner, [1, 2], max_new_tokens=3) == [30, 31, 32]
    assert seen['streamer'] is None and seen['stopping_criteria'] is None


# ---------------------------------------------------------------- routes, fake runners
EDITED, BASE = [10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]


def fixture(tmp_path, style='raises', hold=None):
    """Words 'w<id>'. ``hold=(variant, n)`` pauses that variant before its token n until ``gate.release``.
    ``style``: on a stop, ``raises`` lets GenerationStopped leave generate, ``returns`` returns the tokens so
    far, ``quiet`` returns them before calling the callback again (as the model worker does when a lease
    check during the prompt ends the job)."""
    sess = Session(str(tmp_path/'work'));runtime = Runtime(active(tmp_path));sess.runtime = runtime
    calls, queue = [], sess.subscribe()
    gate = SimpleNamespace(reached=threading.Event(), release=threading.Event())
    def generate(ids, variant, extra=None, **kw):
        calls.append(variant)
        on, made = kw.get('on_token'), []
        for n, token in enumerate(EDITED if variant == 'current' else BASE):
            if hold == (variant, n):
                gate.reached.set()
                assert gate.release.wait(10)
            if style == 'quiet' and on is not None and on.stop is not None and on.stop.is_set():
                return made             # like the model worker stopped while it read the prompt: no token, no exception
            made.append(token)
            if on is not None:
                try:
                    on(token)
                except GenerationStopped:
                    if style == 'raises':
                        raise
                    return made         # an early return: generation ends after the token on which stop was seen
        return made
    decode = lambda ids: ' '.join(f'w{i}' for i in ids)
    tok = SimpleNamespace(token_str=str, to_dict=lambda: {'mode': 'word'}, encode=lambda *a, **k: [1, 2], decode=decode)
    runner = SimpleNamespace(next_logits=lambda *a: torch.zeros(3), chat_prompt=lambda *a, **k: 'whole chat', generate=generate)
    ws = SimpleNamespace(kind='gguf', lock=RLock(), _runtime_client=runtime, struct=SimpleNamespace(n_ctx=512), can_run=True,
                         tok=tok, runner=runner, encode=lambda _: [1, 2], complete=lambda p, v, **kw: decode(generate([1, 2], v, **kw)))
    sess.ws = ws
    client = TestClient(create_app(sess, URL), base_url=URL, raise_server_exceptions=False)
    def drain():
        out = []
        while not queue.empty():
            e = queue.get_nowait()
            if e['type'] == 'generation':out.append(e)
        return out
    return SimpleNamespace(client=client, calls=calls, gate=gate, drain=drain, runtime=runtime, ws=ws, sess=sess)


ROUTES = [('/api/run', {'prompt': 'test'}), ('/api/chat', {'messages': [{'role': 'user', 'content': 'test'}]})]


def text_of(route, data, variant):
    return data[variant].get('text') if route == '/api/run' else data[variant]


def status_of(route, data, variant):
    return data[variant]['status'] if route == '/api/run' else data['variants'][variant]


def tokens_of(route, data, variant):
    return data[variant]['tokens'] if route == '/api/run' else data['tokens'][variant]


def in_background(fn):
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault('value', fn()), daemon=True)
    thread.start()
    return thread, out


def stop(client, stream):
    response = client.post('/api/generation/stop', json={'stream': stream})
    assert response.status_code == 200, response.text
    return response.json()


def assert_no_runtime_record(f, data):
    assert 'interrupted' not in data and 'event' not in data
    assert f.runtime.last is None


@pytest.mark.parametrize('style', ['raises', 'returns'])
@pytest.mark.parametrize('route,body', ROUTES)
def test_a_stop_during_the_edited_reply_ends_it_and_never_runs_the_base_reply(tmp_path, route, body, style, monkeypatch):
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    f = fixture(tmp_path, style, hold=('current', 2))
    thread, out = in_background(lambda: f.client.post(route, json={**body, 'max_new_tokens': 6, 'stream': 'stop-1'}))
    assert f.gate.reached.wait(10)
    # Answered at once, while the request still holds the supervised server's request lock.
    assert stop(f.client, 'stop-1') == {'ok': True, 'found': True}
    assert thread.is_alive()
    f.gate.release.set()
    thread.join(10)
    response = out['value']
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['stopped'] is True and f.calls == ['current']
    # The token being generated when the stop arrived is the last one.
    assert status_of(route, data, 'current') == 'stopped' and text_of(route, data, 'current') == 'w10 w11 w12'
    assert tokens_of(route, data, 'current') == 3
    assert status_of(route, data, 'base') == 'not_run'
    assert_no_runtime_record(f, data)
    events = f.drain()
    assert {e['variant'] for e in events} == {'current'}
    assert events[-1] == {**events[-1], 'state': 'stopped', 'tokens': 3} and apply(events) == 'w10 w11 w12'
    assert stop(f.client, 'stop-1') == {'ok': True, 'found': False}


@pytest.mark.parametrize('style', ['raises', 'returns'])
@pytest.mark.parametrize('route,body', ROUTES)
def test_a_stop_during_the_base_reply_keeps_the_edited_reply_complete(tmp_path, route, body, style, monkeypatch):
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    f = fixture(tmp_path, style, hold=('base', 1))
    thread, out = in_background(lambda: f.client.post(route, json={**body, 'max_new_tokens': 6, 'stream': 'stop-2'}))
    assert f.gate.reached.wait(10)
    assert stop(f.client, 'stop-2')['found'] is True
    f.gate.release.set()
    thread.join(10)
    data = out['value'].json()
    assert data['stopped'] is True and f.calls == ['current', 'base']
    assert status_of(route, data, 'current') == 'complete' and text_of(route, data, 'current') == 'w10 w11 w12 w13 w14 w15'
    assert status_of(route, data, 'base') == 'stopped' and text_of(route, data, 'base') == 'w20 w21'
    assert tokens_of(route, data, 'base') == 2
    assert_no_runtime_record(f, data)
    events = f.drain()
    finals = [(e['variant'], e['state']) for e in events if e['state'] != 'started']
    assert finals == [('current', 'complete'), ('base', 'stopped')]
    assert apply([e for e in events if e['variant'] == 'base']) == 'w20 w21'


@pytest.mark.parametrize('route,body', ROUTES)
def test_a_stop_while_the_request_waits_for_the_model_lock_generates_nothing(tmp_path, route, body):
    f = fixture(tmp_path)
    with f.ws.lock:                                         # another operation holds the model
        thread, out = in_background(lambda: f.client.post(route, json={**body, 'max_new_tokens': 6, 'stream': 'stop-3'}))
        deadline = time.monotonic()+10
        while f.client.app.state.generation_stops.get('stop-3') is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert stop(f.client, 'stop-3') == {'ok': True, 'found': True}
    thread.join(10)
    data = out['value'].json()
    assert data['stopped'] is True and f.calls == [] and f.drain() == []
    assert status_of(route, data, 'current') == status_of(route, data, 'base') == 'not_run'
    assert_no_runtime_record(f, data)


@pytest.mark.parametrize('route,body', ROUTES)
def test_a_stop_sent_before_its_request_arrives_still_applies(tmp_path, route, body):
    f = fixture(tmp_path)
    assert stop(f.client, 'stop-4') == {'ok': True, 'found': False}
    data = f.client.post(route, json={**body, 'max_new_tokens': 6, 'stream': 'stop-4'}).json()
    assert data['stopped'] is True and f.calls == [] and f.drain() == []
    assert status_of(route, data, 'current') == status_of(route, data, 'base') == 'not_run'


@pytest.mark.parametrize('route,body', ROUTES)
def test_a_stop_for_a_request_queued_longer_than_ten_minutes_still_applies(tmp_path, route, body, monkeypatch):
    f = fixture(tmp_path)
    assert stop(f.client, 'queued') == {'ok': True, 'found': False}   # the browser shows "Stopping..."
    later = time.monotonic()+3600                                      # the operation ahead ran for an hour
    monkeypatch.setattr(time, 'monotonic', lambda: later)
    data = f.client.post(route, json={**body, 'max_new_tokens': 6, 'stream': 'queued'}).json()
    assert data['stopped'] is True and f.calls == []
    assert status_of(route, data, 'current') == status_of(route, data, 'base') == 'not_run'


def test_a_stop_while_the_prompt_is_scored_for_the_next_token_display_generates_nothing(tmp_path, monkeypatch):
    """/api/run scores the prompt (a full prompt read for GGUF) before it generates: a stop then skips generation."""
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    f = fixture(tmp_path)
    reached, release = threading.Event(), threading.Event()
    def slow_next(*a):
        reached.set()
        assert release.wait(10)
        return torch.zeros(3)
    f.ws.runner.next_logits = slow_next
    thread, out = in_background(lambda: f.client.post('/api/run', json={'prompt': 'p', 'max_new_tokens': 6, 'stream': 'pre'}))
    assert reached.wait(10)
    assert stop(f.client, 'pre')['found'] is True
    release.set()
    thread.join(10)
    data = out['value'].json()
    assert data['stopped'] is True and f.calls == []
    assert data['current']['status'] == 'stopped' and data['current']['tokens'] == 0 and data['current']['text'] == ''
    assert data['current']['next'] and data['base'] == {'next': [], 'status': 'not_run'}
    assert_no_runtime_record(f, data)
    assert [(e['variant'], e['state'], e.get('tokens')) for e in f.drain() if e['state'] != 'started'] == [('current', 'stopped', 0)]


@pytest.mark.parametrize('route,body', ROUTES)
def test_a_runner_that_returns_early_with_no_token_after_a_stop_is_reported_stopped(tmp_path, route, body, monkeypatch):
    """The model worker stopped at a lease check while reading the prompt: an ordinary empty result."""
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    f = fixture(tmp_path, 'quiet', hold=('current', 0))
    thread, out = in_background(lambda: f.client.post(route, json={**body, 'max_new_tokens': 6, 'stream': 'quiet'}))
    assert f.gate.reached.wait(10)
    assert stop(f.client, 'quiet')['found'] is True
    f.gate.release.set()
    thread.join(10)
    data = out['value'].json()
    assert data['stopped'] is True and f.calls == ['current']
    assert status_of(route, data, 'current') == 'stopped' and tokens_of(route, data, 'current') == 0
    assert text_of(route, data, 'current') == '' and status_of(route, data, 'base') == 'not_run'
    assert_no_runtime_record(f, data)
    assert [(e['variant'], e['state'], e.get('tokens')) for e in f.drain() if e['state'] != 'started'] == [('current', 'stopped', 0)]


class RevokedAtExit:
    """A runtime whose operation ends with a refusal: the lease was revoked while the stopped request ran."""
    def __init__(self, settings):
        self.settings, self.last = settings, None

    @contextmanager
    def operation(self, *a, **k):
        from wsp.runtime_events import RuntimeRefusal, event
        yield
        self.last = event('lease_revoked', 'abort', 'Lease revoked while the request ran', None, None, ['Retry.'])
        raise RuntimeRefusal(self.last)

    def status(self, ws):
        return dict(supervised=True, event=self.last, limits=self.settings['values'])

    def dismiss(self):
        self.last = None


@pytest.mark.parametrize('route,body', ROUTES)
def test_a_stopped_request_that_was_also_interrupted_reports_both(tmp_path, route, body, monkeypatch):
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    f = fixture(tmp_path, hold=('current', 2))
    f.ws._runtime_client = f.sess.runtime = RevokedAtExit(f.runtime.settings)
    thread, out = in_background(lambda: f.client.post(route, json={**body, 'max_new_tokens': 6, 'stream': 'both'}))
    assert f.gate.reached.wait(10)
    assert stop(f.client, 'both')['found'] is True
    f.gate.release.set()
    thread.join(10)
    data = out['value'].json()
    assert data['stopped'] is True and data['interrupted'] is True and data['event']['code'] == 'lease_revoked'
    assert status_of(route, data, 'current') == 'stopped'


@pytest.mark.parametrize('route,body', ROUTES)
def test_a_stop_after_the_request_finished_is_not_found_and_changes_nothing(tmp_path, route, body, monkeypatch):
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    f = fixture(tmp_path)
    data = f.client.post(route, json={**body, 'max_new_tokens': 6, 'stream': 'stop-5'}).json()
    assert stop(f.client, 'stop-5') == {'ok': True, 'found': False}
    assert 'stopped' not in data and f.calls == ['current', 'base']
    assert text_of(route, data, 'base') == 'w20 w21 w22 w23 w24 w25'
    assert [e['state'] for e in f.drain() if e['state'] != 'started'] == ['complete', 'complete']
    # The finished id is not remembered as stopped, and the next request runs in full.
    again = f.client.post(route, json={**body, 'max_new_tokens': 6, 'stream': 'stop-6'}).json()
    assert 'stopped' not in again and f.calls == ['current', 'base']*2


@pytest.mark.parametrize('route,body', ROUTES)
def test_a_request_without_a_stream_id_is_unaffected_by_stops(tmp_path, route, body):
    f = fixture(tmp_path)
    stop(f.client, 'someone-else')
    data = f.client.post(route, json={**body, 'max_new_tokens': 6}).json()
    assert 'stopped' not in data and (route == '/api/run' or 'tokens' not in data)
    assert status_of(route, data, 'current') == status_of(route, data, 'base') == 'complete'
    assert f.calls == ['current', 'base'] and f.drain() == []


def test_the_stop_route_refuses_a_missing_or_bad_id_with_a_clean_answer(tmp_path):
    f = fixture(tmp_path)
    for body in ({}, {'stream': None}, {'stream': 'not valid'}, {'stream': 7}):
        response = f.client.post('/api/generation/stop', json=body)
        assert response.status_code == 400 and 'stream' in response.json()['error']


@pytest.mark.parametrize('route,body', ROUTES)
def test_a_stream_id_already_running_is_refused(tmp_path, route, body):
    f = fixture(tmp_path, hold=('current', 0))
    f.sess.runtime = None       # a supervised server would hold the second request behind the first
    thread, out = in_background(lambda: f.client.post(route, json={**body, 'max_new_tokens': 6, 'stream': 'twice'}))
    assert f.gate.reached.wait(10)
    second = f.client.post('/api/run', json={'prompt': 'test', 'max_new_tokens': 6, 'stream': 'twice'})
    assert second.status_code == 400 and 'already used' in second.json()['error']
    f.gate.release.set()
    thread.join(10)
    assert out['value'].status_code == 200 and 'stopped' not in out['value'].json()


# ---------------------------------------------------------------- the model worker's lease checks
def worker_lease(tmp_path, *, ok=True, stoppable=True, job='job-1'):
    import os
    from wsp.worker_lease import WorkerLease
    (tmp_path/'lease.json').write_text(json.dumps(dict(ok=ok, binding='b', worker={'pid': os.getpid()}, time=time.time(),
                                                       event=None if ok else {'code': 'lease_revoked', 'message': 'revoked'})))
    lease, ws = WorkerLease(tmp_path), SimpleNamespace(version=1)
    lease.begin(job, binding='b', version=1, stoppable=stoppable)
    return lease, ws


def test_a_generate_job_is_stopped_at_its_lease_checks_while_it_reads_the_prompt(tmp_path):
    lease, ws = worker_lease(tmp_path)
    lease.require(ws)
    lease.layer_checkpoint(ws)                                # no stop asked: nothing happens
    assert generation_stop.request(tmp_path, 'job-1')
    with pytest.raises(GenerationStopped):
        lease.require(ws)
    lease._last_lease_check = None
    with pytest.raises(GenerationStopped):
        lease.layer_checkpoint(ws)
    lease.end()
    assert lease.stoppable is False


def test_lease_checks_ignore_a_stop_for_another_job_or_a_job_that_is_not_generate(tmp_path):
    lease, ws = worker_lease(tmp_path)
    generation_stop.request(tmp_path, 'another-job')
    lease.require(ws)
    lease, ws = worker_lease(tmp_path, stoppable=False)
    generation_stop.request(tmp_path, 'job-1')
    lease.require(ws)
    lease._last_lease_check = None
    lease.layer_checkpoint(ws)


def test_a_revoked_lease_wins_over_a_stop(tmp_path):
    from wsp.runtime_events import RuntimeRefusal
    lease, ws = worker_lease(tmp_path, ok=False)
    generation_stop.request(tmp_path, 'job-1')
    with pytest.raises(RuntimeRefusal):
        lease.require(ws)


# ---------------------------------------------------------------- routes, a real FunctionalRunner on a tiny random model
@pytest.fixture()
def tiny(tmp_path):
    from wsp.model import Config, init_params
    from wsp.tokenizer import VocabTokenizer
    from wsp.workspace import Workspace
    tok = VocabTokenizer(list('abcdefghijklmnopqrstuvwxyz'), 'char')
    cfg = Config(d_vocab=26, n_ctx=32, d_model=16, n_layers=1, n_heads=2, d_mlp=24, name='tiny')
    Workspace.create(str(tmp_path/'tiny'), cfg, tok, init_params(cfg, seed=1, std=0.3))
    sess = Session(str(tmp_path))
    sess.open('tiny')
    return sess


@pytest.mark.parametrize('route,body', [('/api/run', {'prompt': 'abc'}),
                                        ('/api/chat', {'messages': [{'role': 'user', 'content': 'abc'}], 'force_template': True,
                                                       'template': {'user': '', 'assistant': '', 'sep': ''}})])
def test_a_real_functional_runner_stops_after_the_token_on_which_stop_arrived(tiny, route, body, monkeypatch):
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    client = TestClient(create_app(tiny, URL), base_url=URL)
    stops, publish = client.app.state.generation_stops, tiny.publish
    def press_stop_on_the_second_token(item):
        publish(item)
        if item.get('type') == 'generation' and item.get('variant') == 'current' and item.get('tokens') == 2:
            stops.stop('tiny-1')
    tiny.publish = press_stop_on_the_second_token
    data = client.post(route, json={**body, 'max_new_tokens': 10, 'temperature': 0, 'stream': 'tiny-1'}).json()
    assert data['stopped'] is True
    assert status_of(route, data, 'current') == 'stopped' and tokens_of(route, data, 'current') == 2
    assert status_of(route, data, 'base') == 'not_run'
    # The same model answers the next request in full.
    tiny.publish = publish
    again = client.post(route, json={**body, 'max_new_tokens': 10, 'temperature': 0, 'stream': 'tiny-2'}).json()
    assert 'stopped' not in again
    assert status_of(route, again, 'current') == status_of(route, again, 'base') == 'complete'
    full = again['current']['text'] if route == '/api/run' else again['current']
    assert full.startswith(text_of(route, data, 'current')) and len(full) > len(text_of(route, data, 'current'))


# ---------------------------------------------------------------- full stack: supervisor, server, stub model worker
LOSS_EVENTS = {'worker_signal', 'worker_lost', 'worker_exited', 'worker_killed', 'worker_reaped', 'worker_stopped',
               'worker_start_failed', 'server_signal', 'abort'}


def chat_body(n, stream):
    return dict(messages=[{'role': 'user', 'content': 'ab'}], force_template=True, template={'user': '', 'assistant': '', 'sep': ' '},
                max_new_tokens=n, temperature=0, stream=stream)


@contextmanager
def full_stack(tmp_path):
    from test_remote_runner import http, run_body, stack, workers
    with stack(tmp_path) as f:
        assert http(f, '/api/run', run_body(2))[0] == 200     # loads the model in its worker
        [pid] = workers(f)
        f.pid, f.rows = pid, len(rows(f))
        f.loads = loads(f)
        yield f


def rows(f):
    from test_runtime_control import log_rows
    return log_rows(f)


def loads(f):
    return [r for r in rows(f) if r['event'] == 'begin' and r['descriptor'].get('job') == 'load']


def assert_same_worker_and_no_loss(f):
    """A user stop is not a memory emergency or a lost worker: nothing anywhere says so."""
    from test_remote_runner import behave, http, run_body, workers
    from test_runtime_control import alive, server_signals, wait
    new = rows(f)[f.rows:]
    assert not [r for r in new if r['event'] in LOSS_EVENTS or r.get('action') == 'abort'], new
    assert not any(isinstance(r.get('event'), dict) or (r['event'] == 'finish' and r.get('descriptor', {}).get('event'))
                   for r in new), new
    beat = wait(lambda: (h := f.heartbeat())['operation_id'] is None and h)
    assert beat['lost'] is None and beat['event'] is None and beat['runner_pid'] == f.pid and beat['run_error'] is None
    assert f.api('/api/runtime').get('event') is None
    assert not (f.folder/generation_stop.FILE).exists()
    # The next request runs at once in the same worker, with no new load.
    behave(f)
    started = time.monotonic()
    status, result = http(f, '/api/run', run_body(3))
    assert status == 200 and 'stopped' not in result and not result.get('interrupted'), result
    assert result['current']['text'] == result['base']['text'] == 'cde'
    assert time.monotonic()-started < 5
    assert workers(f) == [f.pid] and alive(f.pid) and loads(f) == f.loads
    assert f.process.poll() is None and server_signals(f) == []


def test_full_stack_a_stop_during_the_edited_reply_ends_it_in_the_worker_and_keeps_the_model_loaded(tmp_path):
    from test_remote_runner import background, behave, http, live_events, run_body, shown
    from test_runtime_control import read, wait
    with full_stack(tmp_path) as f:
        seen, close = live_events(f)
        try:
            behave(f, slow_after=3, slow_variant='current', slow_seconds=.2)
            thread, out = background(lambda: http(f, '/api/run', {**run_body(40), 'stream': 'full-1'}))
            assert wait(lambda: read(f.root/f'slow-{f.pid}.json'), 20)['emitted'] == [3, 4, 5]
            started = time.monotonic()
            status, answer = http(f, '/api/generation/stop', {'stream': 'full-1'}, timeout=5)
            assert status == 200 and answer == {'ok': True, 'found': True}
            assert time.monotonic()-started < 2                 # not held behind the running request
            thread.join(20)
            status, result = out['value']
            assert status == 200 and result['stopped'] is True, result
            current = result['current']
            # The worker ends the job within a token or two of the stop (40 were asked for).
            assert current['status'] == 'stopped' and current['text'].startswith('cde') and 4 <= current['tokens'] <= 7, current
            assert len(current['text']) == current['tokens']
            assert result['base'] == {'next': [], 'status': 'not_run'}
            assert 'interrupted' not in result and 'event' not in result
            last = wait(lambda: seen and seen[-1]['state'] == 'stopped' and seen[-1])
            assert last['variant'] == 'current' and last['tokens'] == current['tokens']
            assert shown(seen, 'current') == current['text'] and {e['variant'] for e in seen} == {'current'}
        finally:
            close()
        assert_same_worker_and_no_loss(f)


def test_full_stack_a_stop_during_the_base_reply_keeps_the_edited_reply(tmp_path):
    from test_remote_runner import background, behave, http, live_events, shown
    from test_runtime_control import read, wait
    with full_stack(tmp_path) as f:
        seen, close = live_events(f)
        try:
            behave(f, slow_after=2, slow_variant='base', slow_seconds=.2)
            thread, out = background(lambda: http(f, '/api/chat', chat_body(30, 'full-2')))
            assert wait(lambda: read(f.root/f'slow-{f.pid}.json'), 20)['variant'] == 'base'
            assert http(f, '/api/generation/stop', {'stream': 'full-2'}, timeout=5) == (200, {'ok': True, 'found': True})
            thread.join(20)
            status, result = out['value']
            assert status == 200 and result['stopped'] is True, result
            assert result['variants'] == {'current': 'complete', 'base': 'stopped'}
            assert len(result['current']) == 30 and result['current'].startswith('abcd')
            assert result['base'].startswith('ab') and 3 <= result['tokens']['base'] <= 6 and len(result['base']) == result['tokens']['base']
            wait(lambda: [e['state'] for e in seen if e['state'] != 'started'] == ['complete', 'stopped'])
            assert shown(seen, 'current') == result['current'] and shown(seen, 'base') == result['base']
        finally:
            close()
        assert_same_worker_and_no_loss(f)


@pytest.mark.parametrize('route', ['/api/run', '/api/chat'])
def test_full_stack_a_stop_while_the_worker_reads_a_long_prompt_ends_the_job_with_no_token(tmp_path, route):
    from test_remote_runner import background, behave, http, live_events, run_body
    from test_runtime_control import read, wait
    with full_stack(tmp_path) as f:
        seen, close = live_events(f)
        try:
            behave(f, prefill_seconds=60, prefill_variant='current')
            body = {**run_body(20), 'stream': 'prompt-1'} if route == '/api/run' else chat_body(20, 'prompt-1')
            thread, out = background(lambda: http(f, route, body))
            wait(lambda: read(f.root/f'prefill-{f.pid}.json'), 20)
            started = time.monotonic()
            assert http(f, '/api/generation/stop', {'stream': 'prompt-1'}, timeout=5) == (200, {'ok': True, 'found': True})
            thread.join(30)
            status, result = out['value']
            # Ended while the prompt was read (a 60 s read), seen at the next quiet second and lease check.
            assert time.monotonic()-started < 5
            assert status == 200 and result['stopped'] is True, result
            assert 'interrupted' not in result and 'event' not in result
            if route == '/api/run':
                assert result['current']['status'] == 'stopped' and result['current']['tokens'] == 0 and result['current']['text'] == ''
                assert result['base'] == {'next': [], 'status': 'not_run'}
            else:
                assert result['variants'] == {'current': 'stopped', 'base': 'not_run'} and result['tokens'] == {'current': 0}
                assert result['current'] == ''
            last = wait(lambda: seen and seen[-1]['state'] == 'stopped' and seen[-1])
            assert last['variant'] == 'current' and last['tokens'] == 0
        finally:
            close()
        assert_same_worker_and_no_loss(f)


def test_full_stack_a_request_stopped_while_it_waits_behind_another_generates_nothing(tmp_path):
    from test_remote_runner import background, behave, http, live_events, run_body
    from test_runtime_control import read, wait
    with full_stack(tmp_path) as f:
        seen, close = live_events(f)
        try:
            behave(f, slow_after=0, slow_variant='current', slow_seconds=.1)
            first, first_out = background(lambda: http(f, '/api/run', {**run_body(12), 'stream': 'ahead'}))
            wait(lambda: read(f.root/f'slow-{f.pid}.json'), 20)
            second, second_out = background(lambda: http(f, '/api/run', {**run_body(12), 'stream': 'behind'}))
            time.sleep(.3)
            # The waiting request has not reached its route: not found, but the stop is kept for it.
            assert http(f, '/api/generation/stop', {'stream': 'behind'}, timeout=5) == (200, {'ok': True, 'found': False})
            first.join(30);second.join(30)
            assert first_out['value'][0] == 200 and 'stopped' not in first_out['value'][1]
            status, result = second_out['value']
            assert status == 200 and result['stopped'] is True, result
            assert result['current']['status'] == result['base']['status'] == 'not_run'
            assert [e for e in seen if e['stream'] == 'behind'] == []
        finally:
            close()
        assert_same_worker_and_no_loss(f)


def test_full_stack_a_stop_after_the_reply_finished_is_not_found_and_the_result_stands(tmp_path):
    from test_remote_runner import http, run_body
    with full_stack(tmp_path) as f:
        status, result = http(f, '/api/run', {**run_body(4), 'stream': 'done-1'})
        assert status == 200 and 'stopped' not in result and result['current']['text'] == 'cdef'
        assert http(f, '/api/generation/stop', {'stream': 'done-1'}) == (200, {'ok': True, 'found': False})
        assert_same_worker_and_no_loss(f)
