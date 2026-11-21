"""Live generation text: the token callback, its events and the Run/Chat routes. No model."""
from contextlib import contextmanager
from threading import RLock
import time
from types import SimpleNamespace
import pytest
import torch
from fastapi.testclient import TestClient
from test_settings import active
from wsp.runner import HFRunner, TokenCallback
from wsp.runtime_events import RuntimeRefusal, event
from wsp.server import create_app
from wsp.session import Session
from wsp.token_stream import TextStream, stream_id


def letters(ids):
    return ''.join(chr(i) for i in ids)


def apply(events):
    """What a browser shows after these events, in JavaScript's UTF-16 string units."""
    text = ''
    for e in events:
        if 'text' in e:
            units = text.encode('utf-16-le')
            assert e['keep']*2 <= len(units)
            text = units[:e['keep']*2].decode('utf-16-le')+e['text']
    return text


class Timers:
    """Stands in for threading.Timer: the test decides when a pending timer fires."""
    def __init__(self):
        self.pending = []

    def __call__(self, wait, fn):
        timer = SimpleNamespace(wait=wait, fn=fn, daemon=False, start=lambda: self.pending.append(timer),
                                cancel=lambda: timer in self.pending and self.pending.remove(timer))
        return timer

    def fire(self):
        [timer] = self.pending
        self.pending.remove(timer)
        timer.fn()


def stream(render=letters, **kw):
    events = []
    kw.setdefault('timer', Timers())
    return TextStream(events.append, 's1', 'current', render, **kw), events


def test_each_token_appends_and_the_end_names_the_state():
    live, events = stream(interval=0)
    live.begin()
    for c in 'hey':
        live(ord(c))
    live.end('complete')
    assert all(e['type'] == 'generation' and e['stream'] == 's1' and e['variant'] == 'current' for e in events)
    assert [(e['state'], e.get('keep'), e.get('text')) for e in events] == [
        ('started', 0, ''), ('started', 0, 'h'), ('started', 1, 'e'), ('started', 2, 'y'), ('complete', None, None)]
    assert events[-1]['tokens'] == 3 and apply(events) == 'hey'


def test_events_are_throttled_and_the_end_sends_the_rest():
    now, timers = [0.0], Timers()
    live, events = stream(interval=0.05, clock=lambda: now[0], timer=timers)
    for c in 'abcd':        # one burst inside the interval: only the first token is sent at once
        now[0] += 0.01
        live(ord(c))
    assert [e['text'] for e in events] == ['a']
    # One timer covers the burst and is due when the interval ends.
    assert len(timers.pending) == 1 and timers.pending[0].wait == pytest.approx(0.04)
    now[0] = 0.07
    live(ord('e'))          # a token after the interval is sent at once, with what was held
    assert [e['text'] for e in events] == ['a', 'bcde']
    timers.fire()           # the burst's timer finds nothing new
    assert [e['text'] for e in events] == ['a', 'bcde']
    live(ord('f'))
    live.end('complete')
    assert timers.pending == []
    assert apply(events) == 'abcdef' and events[-1] == dict(type='generation', stream='s1', variant='current', state='complete', tokens=6)


def test_held_tokens_are_shown_when_the_interval_ends_even_if_the_model_stalls():
    now, timers = [0.0], Timers()
    live, events = stream(interval=0.05, clock=lambda: now[0], timer=timers)
    for c in 'cde':
        live(ord(c))
    assert apply(events) == 'c'
    now[0] = 0.05
    timers.fire()           # no further token arrives
    assert apply(events) == 'cde' and events[-1]['tokens'] == 3 and timers.pending == []
    # With the real timer: the rest of a burst appears without another token.
    live, events = stream(interval=0.02, timer=__import__('threading').Timer)
    for c in 'cde':
        live(ord(c))
    deadline = time.monotonic()+5
    while apply(events) != 'cde' and time.monotonic() < deadline:
        time.sleep(0.005)
    assert apply(events) == 'cde'
    live.end('interrupted')


def test_nothing_is_published_after_the_end():
    now, timers = [0.0], Timers()
    live, events = stream(interval=0.05, clock=lambda: now[0], timer=timers)
    live(ord('a')); live(ord('b'))
    [timer] = timers.pending
    live.end('complete')
    assert timers.pending == [] and events[-1]['state'] == 'complete' and apply(events) == 'ab'
    count = len(events)
    timer.fn()              # a timer that had already started running
    live(ord('c'))
    assert len(events) == count
    # close() alone also stops live text; the end still reports every token.
    live, events = stream(interval=0)
    live(ord('a')); live.close(); live(ord('b'))
    assert apply(events) == 'a'
    live.end('complete')
    assert apply(events) == 'ab' and events[-1]['tokens'] == 2


def test_a_failure_in_the_timer_is_contained():
    timers, calls = Timers(), []
    def render(ids):
        calls.append(len(ids))
        if len(calls) == 2:
            raise RuntimeError('decode failed')
        return letters(ids)
    now = [0.0]
    live, events = stream(render, interval=0.05, clock=lambda: now[0], timer=timers)
    live(ord('a')); live(ord('b'))
    timers.fire()
    assert live.failed and events[-1].get('lost') is True
    live.end('complete')
    assert events[-1]['state'] == 'complete' and events[-1]['lost'] is True and events[-1]['tokens'] == 2


def test_offsets_count_utf16_units():
    live, events = stream(interval=0)
    for c in 'a\U0001F600b':
        live(ord(c))
    assert [e['keep'] for e in events] == [0, 1, 3]   # the emoji is two units in a JavaScript string
    assert apply(events) == 'a\U0001F600b'


def test_an_incomplete_character_is_held_back_until_its_last_token():
    data = 'é'.encode()                                       # two bytes, one token each
    live, events = stream(lambda ids: bytes(ids).decode('utf-8', 'replace'), interval=0)
    live(ord('a'))
    live(data[0])
    assert apply(events) == 'a' and len(events) == 1
    live(data[1])
    assert apply(events) == 'aé'
    # At the end nothing is held back: the final text is exactly what the response carries.
    live(data[0])
    live.end('interrupted')
    assert apply(events) == 'aé�' and events[-1]['state'] == 'interrupted'


def test_a_completed_stop_string_retracts_what_was_shown():
    render = lambda ids: letters(ids).split('User:')[0].strip('\n')
    live, events = stream(render, interval=0)
    for c in 'Hi\nUser: next':
        live(ord(c))
    live.end('complete')
    shown = [apply(events[:n+1]) for n in range(len(events))]
    assert 'Hi\nUser' in shown and shown[-1] == 'Hi'
    retract = next(e for e in events if e.get('text') == '' and e.get('keep') == 2)
    assert retract['state'] == 'started'


def test_the_end_can_take_the_tokens_or_the_text_of_the_response():
    live, events = stream(interval=0)
    live(ord('a'))
    live.end('interrupted', [ord('a'), ord('b')])
    assert apply(events) == 'ab' and events[-1]['tokens'] == 2
    live, events = stream(interval=0)
    live(ord('a'))
    live.end('complete', text='a!')
    assert apply(events) == 'a!'


def test_the_callback_never_raises_and_says_the_live_text_is_lost():
    def render(ids):
        if len(ids) > 1:
            raise RuntimeError('decode failed')
        return letters(ids)
    live, events = stream(render, interval=0)
    for c in 'abc':
        live(ord(c))                      # raising here would be handled as a lost model worker
    live.end('complete')
    assert [e.get('text') for e in events] == ['a', None, None]
    assert events[1]['lost'] is True and events[2] == dict(type='generation', stream='s1', variant='current', state='complete', tokens=3, lost=True)

    def broken(_):
        raise OSError('no subscribers table')
    live = TextStream(broken, 's1', 'current', letters, interval=0)
    live.begin(); live(1); live.end('complete')
    assert live.failed


def test_stream_ids_are_validated():
    assert stream_id({}) is None and stream_id({'stream': 'a-B_9'}) == 'a-B_9'
    assert stream_id({'stream': None}) is None and stream_id({'stream': False}) is None
    for bad in ('', 'a b', 'x'*65, 5, ['a'], 'a\n', True):
        with pytest.raises(ValueError):
            stream_id({'stream': bad})


def test_the_transformers_streamer_skips_the_prompt():
    seen = []
    put = TokenCallback(seen.append)
    put.put(torch.tensor([[1, 2, 3]]))    # generate hands over the prompt first
    put.put(torch.tensor([7]))
    put.put(torch.tensor([8, 9]))
    put.end()
    assert seen == [7, 8, 9]


def test_the_transformers_runner_streams_only_without_beam_search():
    def run(beams):
        seen = {}
        def generate(**kw):
            seen.update(kw)
            return torch.tensor([[1, 2, 7]])
        config = SimpleNamespace(pad_token_id=0, eos_token_id=0, num_beams=beams)
        runner = SimpleNamespace(lock=RLock(), use=lambda *a: None, device='cpu', model=SimpleNamespace(generation_config=config, generate=generate))
        assert HFRunner.generate(runner, [1, 2], on_token=lambda token: None) == [7]
        return seen
    assert isinstance(run(1)['streamer'], TokenCallback) and isinstance(run(None)['streamer'], TokenCallback)
    assert 'streamer' not in run(4)       # transformers raises for a streamer with beam search


# ---------------------------------------------------------------- routes
class Runtime:
    def __init__(self, settings):self.settings = settings;self.last = None
    @contextmanager
    def operation(self, *args, **kwargs):
        try:yield
        except RuntimeRefusal as exc:self.last = exc.record;raise
    def status(self, ws):return dict(supervised=True, event=self.last, limits=self.settings['values'])
    def dismiss(self):self.last = None


def fixture(tmp_path, fail=None, error=None):
    """Words 'w<id>'; the edited model says 10 11 12, the base model 20 21 22."""
    sess = Session(str(tmp_path/'work'));runtime = Runtime(active(tmp_path));sess.runtime = runtime
    calls, seen = [], []
    queue = sess.subscribe()
    def generate(ids, variant, extra=None, **kw):
        calls.append((variant, 'on_token' in kw))
        out = [10, 11, 12] if variant == 'current' else [20, 21, 22]
        for n, token in enumerate(out):
            if fail == variant and n == 2:
                raise RuntimeRefusal(event('runtime_budget', 'abort', 'Synthetic stop', 4, 3, ['Retry.']), tokens=out[:2])
            if error == variant and n == 2:
                raise ValueError('Synthetic failure')
            if kw.get('on_token'):
                kw['on_token'](token)
                seen.append((variant, token, queue.qsize()))
        return out
    decode = lambda ids: ' '.join(f'w{i}' for i in ids)
    tok = SimpleNamespace(token_str=str, to_dict=lambda: {'mode': 'word'}, encode=lambda *a, **k: [1, 2], decode=decode)
    runner = SimpleNamespace(next_logits=lambda *a: torch.zeros(3), chat_prompt=lambda *a, **k: 'whole chat', generate=generate)
    ws = SimpleNamespace(kind='gguf', lock=RLock(), _runtime_client=runtime, struct=SimpleNamespace(n_ctx=512), can_run=True,
                         tok=tok, runner=runner, encode=lambda _: [1, 2], complete=lambda p, v, **kw: decode(generate([1, 2], v, **kw)))
    sess.ws = ws
    client = TestClient(create_app(sess, 'http://127.0.0.1:41881'), base_url='http://127.0.0.1:41881', raise_server_exceptions=True)
    client.ws = ws
    def drain():
        out = []
        while not queue.empty():
            e = queue.get_nowait()
            if e['type'] == 'generation':out.append(e)
        return out
    return client, calls, seen, drain


ROUTES = [('/api/run', {'prompt': 'test'}), ('/api/chat', {'messages': [{'role': 'user', 'content': 'test'}]})]
text_of = lambda route, data, variant: data[variant]['text'] if route == '/api/run' else data[variant]


@pytest.mark.parametrize('route,body', ROUTES)
def test_a_request_with_a_stream_id_publishes_the_edited_text_first_and_as_it_is_generated(tmp_path, route, body, monkeypatch):
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    client, calls, seen, drain = fixture(tmp_path)
    response = client.post(route, json={**body, 'max_new_tokens': 3, 'stream': 'req-1'})
    assert response.status_code == 200, response.text
    data = response.json()
    assert calls == [('current', True), ('base', True)]
    # Each token's event was already queued for the browser when the runner produced the next one.
    sizes = [size for _, _, size in seen]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    events = drain()
    assert all(e['stream'] == 'req-1' for e in events)
    order = [e['variant'] for e in events]
    assert order == sorted(order, key=['current', 'base'].index) and set(order) == {'current', 'base'}
    for variant, words in (('current', 'w10 w11 w12'), ('base', 'w20 w21 w22')):
        mine = [e for e in events if e['variant'] == variant]
        assert mine[0] == {**mine[0], 'state': 'started', 'keep': 0, 'text': '', 'tokens': 0}
        assert mine[-1]['state'] == 'complete' and mine[-1]['tokens'] == 3
        assert apply(mine) == words == text_of(route, data, variant)
        assert [apply(mine[:n+1]) for n in range(len(mine))][1:4] == ['w10', 'w10 w11', 'w10 w11 w12'] if variant == 'current' else True


@pytest.mark.parametrize('route,body', ROUTES)
def test_without_a_stream_id_nothing_is_published_and_the_runner_gets_no_callback(tmp_path, route, body):
    client, calls, seen, drain = fixture(tmp_path)
    response = client.post(route, json={**body, 'max_new_tokens': 3})
    assert response.status_code == 200 and calls == [('current', False), ('base', False)]
    assert drain() == [] and seen == []
    assert text_of(route, response.json(), 'current') == 'w10 w11 w12' and text_of(route, response.json(), 'base') == 'w20 w21 w22'


@pytest.mark.parametrize('route,body', ROUTES)
@pytest.mark.parametrize('fail', ['current', 'base'])
def test_an_interrupted_variant_ends_its_stream_with_the_partial_text(tmp_path, route, body, fail, monkeypatch):
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    client, calls, seen, drain = fixture(tmp_path, fail)
    response = client.post(route, json={**body, 'max_new_tokens': 3, 'stream': 'req-2'})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['interrupted'] and data['event']['code'] == 'runtime_budget'
    assert [c[0] for c in calls] == (['current'] if fail == 'current' else ['current', 'base'])
    events = drain()
    mine = [e for e in events if e['variant'] == fail]
    partial = 'w10 w11' if fail == 'current' else 'w20 w21'
    assert mine[-1]['state'] == 'interrupted' and apply(mine) == partial == text_of(route, data, fail)
    other = [e for e in events if e['variant'] != fail]
    assert (other == []) if fail == 'current' else (other[-1]['state'] == 'complete' and apply(other) == 'w10 w11 w12')
    if route == '/api/chat':
        assert data['variants'] == ({'current': 'interrupted', 'base': 'not_run'} if fail == 'current' else {'current': 'complete', 'base': 'interrupted'})
    else:
        assert data['base']['status'] == ('not_run' if fail == 'current' else 'interrupted')


@pytest.mark.parametrize('route,body', ROUTES)
def test_a_bad_stream_id_is_refused_before_anything_runs(tmp_path, route, body):
    client, calls, seen, drain = fixture(tmp_path)
    client = TestClient(client.app, base_url='http://127.0.0.1:41881', raise_server_exceptions=False)
    response = client.post(route, json={**body, 'max_new_tokens': 3, 'stream': 'not valid'})
    assert response.status_code == 400, response.text
    assert calls == [] and drain() == []


def test_a_zero_output_run_has_no_stream(tmp_path):
    client, calls, seen, drain = fixture(tmp_path)
    response = client.post('/api/run', json={'prompt': 'test', 'max_new_tokens': 0, 'stream': 'req-3'})
    assert response.status_code == 200 and drain() == [] and calls == []


def test_chat_live_text_applies_the_stop_string_like_the_response(tmp_path, monkeypatch):
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    client, calls, seen, drain = fixture(tmp_path)
    words = {10: 'Hi', 11: '\nUser:', 12: ' more', 20: 'Yo', 21: '\nUser:', 22: ' x'}
    tok = client.ws.tok
    tok.decode = lambda ids: ''.join(words[i] for i in ids)
    tok.to_dict = lambda: {'mode': 'word', 'type': 'word'}
    # Untemplated chat: the reply is cut where the next user turn would start.
    response = client.post('/api/chat', json={'messages': [{'role': 'user', 'content': 'test'}], 'force_template': True,
                                              'max_new_tokens': 3, 'stream': 'req-4'})
    assert response.status_code == 200, response.text
    data = response.json()
    events = drain()
    for variant, word in (('current', 'Hi'), ('base', 'Yo')):
        mine = [e for e in events if e['variant'] == variant]
        shown = [apply(mine[:n+1]) for n in range(len(mine))]
        assert data[variant] == word == shown[-1] and not any('User' in s for s in shown)


@pytest.mark.parametrize('route,body', ROUTES)
def test_a_request_that_fails_with_an_error_ends_its_stream(tmp_path, route, body, monkeypatch):
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    client, calls, seen, drain = fixture(tmp_path, error='current')
    client = TestClient(client.app, base_url='http://127.0.0.1:41881', raise_server_exceptions=False)
    response = client.post(route, json={**body, 'max_new_tokens': 3, 'stream': 'req-5'})
    assert response.status_code == 400 and 'Synthetic failure' in response.text
    events = drain()
    # A client following the events is told the variant stopped; the response carries the error.
    assert {e['variant'] for e in events} == {'current'} and [e['state'] for e in events][-1] == 'failed'
    assert apply(events) == 'w10 w11' and events[-1]['tokens'] == 2
