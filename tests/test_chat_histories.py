"""/api/chat with a conversation per model (``messages``, ``base_messages``) and ``variants``. No checkpoint.

Most tests run a fake GGUF-style runner whose chat template is visible in the ids each variant receives
(one id per character of the rendered prompt) under a fake runtime that records its operation scopes; the
active context is 64 tokens. Refusals and steering run the tiny random model of tests/test_steering.py,
with ``make_runner`` replaced where a test asserts that nothing was built.
"""
from contextlib import contextmanager
import threading
from threading import RLock
from types import SimpleNamespace

import pytest
import torch

from fastapi.testclient import TestClient
from test_settings import active
from test_token_stream import apply
from test_steering_runs import CHAT, client as model_client, greedy_tokens, pushed_params, sess  # noqa: F401 (fixture)
from wsp.runtime_events import RuntimeRefusal, event
from wsp.server import create_app
from wsp.session import Session

URL = 'http://127.0.0.1:41884'
EDITED, BASE = [10, 11, 12, 13], [20, 21, 22, 23]
ASK = [{'role': 'user', 'content': 'hi'}]
EDITED_CHAT = [*ASK, {'role': 'assistant', 'content': 'edited said'}, {'role': 'user', 'content': 'again'}]
BASE_CHAT = [*ASK, {'role': 'assistant', 'content': 'base say'}, {'role': 'user', 'content': 'again'}]


def template(messages, **options):
    return ''.join(f"<{m['role']}>{m['content']}" for m in messages) + '<assistant>'


def ids_of(messages):
    return [ord(c) for c in template(messages)]


class Runtime:
    """Records each operation scope's token count; a refusal inside it is remembered as the last event."""
    def __init__(self, settings):
        self.settings, self.last, self.scopes = settings, None, []

    @contextmanager
    def operation(self, ws, name, **kwargs):
        self.scopes.append((name, kwargs.get('tokens')))
        try:
            yield
        except RuntimeRefusal as exc:
            self.last = exc.record
            raise

    def status(self, ws):
        return dict(supervised=True, event=self.last, limits=self.settings['values'])

    def dismiss(self):
        self.last = None


def fixture(tmp_path, hold=None, fail=None, templated=True, kind='gguf'):
    """``hold=(variant, n)`` pauses that variant before its token n until ``gate.release``; ``fail=variant``
    interrupts that variant with a runtime refusal after two tokens. ``templated=False``: the model has no chat
    template, so the plain turn template renders the prompt."""
    sess = Session(str(tmp_path / 'work'))
    runtime = Runtime(active(tmp_path))
    sess.runtime = runtime
    calls, rendered, queue = [], [], sess.subscribe()
    gate = SimpleNamespace(reached=threading.Event(), release=threading.Event())

    def generate(ids, variant, extra=None, **kw):
        calls.append((variant, list(ids), {k: v for k, v in kw.items() if k != 'on_token'}))
        on, made = kw.get('on_token'), []
        for n, token in enumerate((EDITED if variant == 'current' else BASE)[:kw['max_new_tokens']]):
            if hold == (variant, n):
                gate.reached.set()
                assert gate.release.wait(10)
            if fail == variant and n == 2:
                raise RuntimeRefusal(event('runtime_budget', 'abort', 'Synthetic stop', 4, 3, ['Retry.']), tokens=made)
            made.append(token)
            if on is not None:
                on(token)               # GenerationStopped leaves generate, as FunctionalRunner does
        return made

    def chat_prompt(messages, **options):
        rendered.append([m.get('content') for m in messages])
        return template(messages) if templated else None
    decode = lambda ids: ' '.join(f'w{i}' for i in ids)
    encode = lambda text, **k: [ord(c) for c in text]
    tok = SimpleNamespace(token_str=str, to_dict=lambda: {'mode': 'word', 'type': 'bpe'}, encode=encode, decode=decode)
    runner = SimpleNamespace(next_logits=lambda *a: torch.zeros(3), chat_prompt=chat_prompt, generate=generate)
    ws = SimpleNamespace(kind=kind, lock=RLock(), _runtime_client=runtime, struct=SimpleNamespace(n_ctx=64), can_run=True,
                         tok=tok, runner=runner, encode=encode)
    sess.ws = ws
    app = create_app(sess, URL)
    client = TestClient(app, base_url=URL, raise_server_exceptions=False)

    def drain():
        out = []
        while not queue.empty():
            e = queue.get_nowait()
            if e['type'] == 'generation':
                out.append(e)
        return out
    return SimpleNamespace(client=client, calls=calls, rendered=rendered, gate=gate, drain=drain, runtime=runtime,
                           stops=app.state.generation_stops)


def in_background(fn):
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault('value', fn()), daemon=True)
    thread.start()
    return thread, out


def post(f, body):
    r = f.client.post('/api/chat', json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ requests without the new fields
def test_a_request_without_the_new_fields_answers_exactly_as_before(tmp_path):
    f = fixture(tmp_path)
    data = post(f, {'messages': EDITED_CHAT, 'max_new_tokens': 3})
    n = len(ids_of(EDITED_CHAT))
    assert data == {'prompt': template(EDITED_CHAT), 'chat_template': True,
                    'generation': {'requested_output': 3, 'effective_output': 3, 'prompt_tokens': n, 'context': 64},
                    'base': 'w20 w21 w22', 'current': 'w10 w11 w12', 'variants': {'base': 'complete', 'current': 'complete'}}
    # Both models answer the one conversation from the same ids, rendered once, under one scope sized for it.
    assert [(v, ids) for v, ids, _ in f.calls] == [('current', ids_of(EDITED_CHAT)), ('base', ids_of(EDITED_CHAT))]
    assert len(f.rendered) == 1 and f.runtime.scopes == [('chat', n + 3)]


# ------------------------------------------------------------------ a conversation per model
def test_each_model_answers_its_own_conversation(tmp_path):
    f = fixture(tmp_path)
    data = post(f, {'messages': EDITED_CHAT, 'base_messages': BASE_CHAT, 'max_new_tokens': 3, 'stream': 'two'})
    assert [(v, ids) for v, ids, _ in f.calls] == [('current', ids_of(EDITED_CHAT)), ('base', ids_of(BASE_CHAT))]
    assert data['current'] == 'w10 w11 w12' and data['base'] == 'w20 w21 w22'
    assert data['variants'] == {'base': 'complete', 'current': 'complete'}
    # The top-level prompt and generation are the edited model's; each model's own are reported beside them.
    assert data['prompt'] == template(EDITED_CHAT) and data['generation']['prompt_tokens'] == len(ids_of(EDITED_CHAT))
    assert data['prompts'] == {'current': template(EDITED_CHAT), 'base': template(BASE_CHAT)}
    assert data['generations'] == {
        'current': {'requested_output': 3, 'effective_output': 3, 'prompt_tokens': len(ids_of(EDITED_CHAT)), 'context': 64},
        'base': {'requested_output': 3, 'effective_output': 3, 'prompt_tokens': len(ids_of(BASE_CHAT)), 'context': 64}}
    # The scope covers the longer of the two sequences (the edited conversation is the longer one here).
    assert len(ids_of(EDITED_CHAT)) > len(ids_of(BASE_CHAT))
    assert f.runtime.scopes == [('chat', len(ids_of(EDITED_CHAT)) + 3)]
    events = f.drain()
    assert [(e['variant'], e['state']) for e in events if e['state'] != 'started'] == [('current', 'complete'), ('base', 'complete')]
    assert apply([e for e in events if e['variant'] == 'base']) == 'w20 w21 w22'


def test_the_scope_covers_the_longer_conversation_whichever_model_has_it(tmp_path):
    f = fixture(tmp_path)
    longer = [{'role': 'user', 'content': 'a longer question'}]
    assert len(ids_of(longer)) > len(ids_of(ASK))
    post(f, {'messages': ASK, 'base_messages': longer, 'max_new_tokens': 3})
    assert f.runtime.scopes == [('chat', len(ids_of(longer)) + 3)]


def test_equal_conversations_render_once(tmp_path):
    f = fixture(tmp_path)
    data = post(f, {'messages': EDITED_CHAT, 'base_messages': [dict(m) for m in EDITED_CHAT], 'max_new_tokens': 2})
    assert len(f.rendered) == 1
    assert [ids for _, ids, _ in f.calls] == [ids_of(EDITED_CHAT)] * 2
    assert data['prompts'] == {'current': template(EDITED_CHAT), 'base': template(EDITED_CHAT)}


@pytest.mark.parametrize('variants,body,ran', [
    (['base'], {'base_messages': BASE_CHAT}, 'base'),
    (['current'], {'messages': EDITED_CHAT}, 'current'),
])
def test_one_requested_model_runs_alone_and_is_the_only_one_reported(tmp_path, variants, body, ran):
    f = fixture(tmp_path)
    data = post(f, {**body, 'variants': variants, 'max_new_tokens': 3, 'stream': 'one'})
    conversation = BASE_CHAT if ran == 'base' else EDITED_CHAT
    assert [(v, ids) for v, ids, _ in f.calls] == [(ran, ids_of(conversation))]
    # The model not asked for has no status: it is not one "not run after an interruption".
    assert data['variants'] == {ran: 'complete'}
    other = 'current' if ran == 'base' else 'base'
    assert data[ran] == ('w20 w21 w22' if ran == 'base' else 'w10 w11 w12') and data[other] == ''
    assert data['prompt'] == template(conversation) and data['prompts'] == {ran: template(conversation)}
    assert set(data['generations']) == {ran} and data['generation'] == data['generations'][ran]
    assert f.runtime.scopes == [('chat', len(ids_of(conversation)) + 3)]
    assert {e['variant'] for e in f.drain()} == {ran}          # stream events only for the model that ran


def test_variants_in_either_order_run_the_edited_model_first(tmp_path):
    f = fixture(tmp_path)
    data = post(f, {'messages': EDITED_CHAT, 'base_messages': BASE_CHAT, 'variants': ['base', 'current'], 'max_new_tokens': 2})
    assert [v for v, _, _ in f.calls] == ['current', 'base'] and data['variants'] == {'base': 'complete', 'current': 'complete'}


# ------------------------------------------------------------------ refused before the model is touched
@pytest.mark.parametrize('extra,message', [
    ({'variants': []}, 'variants must list'),
    ({'variants': 'base'}, 'variants must list'),
    ({'variants': ['edited']}, 'variants must list'),
    ({'variants': ['current', 'current']}, 'each at most once'),
    ({'variants': [['base']]}, 'variants must list'),
    ({'base_messages': 'hello'}, 'base_messages must be a list of turns'),
    ({'base_messages': {'role': 'user', 'content': 'x'}}, 'base_messages must be a list of turns'),
    ({'base_messages': ['hello']}, 'base_messages must be a list of turns'),
    ({'messages': 'hello'}, 'messages must be a list of turns'),
    ({'messages': None}, 'messages must be a list of turns'),
    # CHAT sets force_template, so the plain turn template is certain and what it needs is checked first.
    ({'base_messages': [{'role': 'user'}]}, 'base_messages[0]: force_template is set, so each turn needs a role and its content as text'),
    ({'base_messages': [{'content': 'x'}]}, 'base_messages[0]: force_template is set'),
    ({'base_messages': [*ASK, {'role': 'user', 'content': 5}]}, 'base_messages[1]: force_template is set'),
    ({'messages': [{'role': 'user', 'content': None}]}, 'messages[0]: force_template is set'),
    ({'template': 'x'}, 'template must be an object of turn prefixes'),
    ({'template': {'user': 5}}, 'template prefixes (system, user, assistant) and sep must be text'),
    ({'variants': ['current'], 'base_messages': ASK}, 'runs the edited model only'),
    ({'variants': ['base']}, 'messages is the edited model\'s conversation, and this request runs the base model only'),
    ({'variants': ['base'], 'messages': None}, 'send its conversation as base_messages'),
    ({'variants': ['base'], 'messages': None, 'base_messages': ASK, 'steer': [{'addr': 'L1N3', 'mode': 'add', 'dose': 1}]},
     'steering applies to the edited model only, and this request runs the base model only'),
    ({'variants': ['base'], 'messages': None, 'base_messages': ASK, 'steer': 'junk'}, 'steering applies to the edited model only'),
    ({'base_messages': ASK, 'steer': [{'addr': 'L0H1', 'mode': 'add', 'dose': 1}]}, 'add pushes neurons only'),
])
def test_a_malformed_request_is_refused_before_the_runner_is_built(sess, extra, message, monkeypatch):
    import wsp.runner
    built = []

    def refuse(w):
        built.append(w.name)
        raise AssertionError('the runner was built for a request that should have been refused first')
    monkeypatch.setattr(wsp.runner, 'make_runner', refuse)
    assert sess.ws._runner is None
    body = {**CHAT, **extra}
    if body.get('messages') is None:
        body.pop('messages')
    r = model_client(sess).post('/api/chat', json=body)
    assert r.status_code == 400, (extra, r.text)
    assert message in r.json()['error'], (extra, r.json()['error'])
    assert built == [] and sess.ws._runner is None


@pytest.mark.parametrize('fields', [
    {'steer': [], 'recalibrate': False},
    {'p50': {}},
    {'recalibrate': 'false'},
    {'steer': None, 'p50': 'junk', 'recalibrate': True},
])
def test_without_steering_p50_and_recalibrate_are_ignored_on_a_base_only_request_as_on_any(tmp_path, fields):
    # Only a steering request (a non-empty steer) is refused on a base-only request; the shaping fields of a
    # push are ignored without one, exactly as on a request that runs both models.
    f = fixture(tmp_path)
    data = post(f, {'variants': ['base'], 'base_messages': ASK, 'max_new_tokens': 2, **fields})
    assert data['variants'] == {'base': 'complete'} and 'steering' not in data
    data = post(f, {'messages': ASK, 'base_messages': BASE_CHAT, 'max_new_tokens': 2, **fields})
    assert data['variants'] == {'base': 'complete', 'current': 'complete'} and 'steering' not in data
    assert [kw.get('steer') for _, _, kw in f.calls] == [None, None, None]


# ------------------------------------------------------------------ shapes accepted before stay accepted
@pytest.mark.parametrize('templated,prompt', [(True, '<assistant>'), (False, 'Assistant:')])
def test_an_empty_conversation_is_answered_from_the_generation_prompt_as_before(tmp_path, templated, prompt):
    f = fixture(tmp_path, templated=templated)
    data = post(f, {'messages': [], 'max_new_tokens': 2})
    assert data['prompt'] == prompt and data['variants'] == {'base': 'complete', 'current': 'complete'}
    assert 'prompts' not in data                       # still the old response, key for key
    data = post(f, {'messages': ASK, 'base_messages': [], 'max_new_tokens': 2})
    assert data['prompts']['base'] == prompt and data['variants']['base'] == 'complete'
    data = post(f, {'variants': ['base'], 'base_messages': [], 'max_new_tokens': 2})
    assert data['prompt'] == prompt and data['variants'] == {'base': 'complete'}


@pytest.mark.parametrize('templated,prompt', [(True, '<None>hi<assistant>'), (False, 'Assistant: hi\nAssistant:')])
def test_any_role_value_is_accepted_as_before(tmp_path, templated, prompt):
    # The plain turn template renders any role other than "user" with the assistant prefix, as it always did.
    f = fixture(tmp_path, templated=templated)
    data = post(f, {'messages': [{'role': None, 'content': 'hi'}], 'max_new_tokens': 2})
    assert data['prompt'] == prompt and 'prompts' not in data
    data = post(f, {'messages': ASK, 'base_messages': [{'role': None, 'content': 'hi'}], 'max_new_tokens': 2})
    assert data['prompts']['base'] == prompt


@pytest.mark.parametrize('body,message', [
    ({'messages': [{'role': 'user', 'content': 5}]}, 'messages[0]: this model has no chat template, so each turn needs a role and its content as text'),
    ({'messages': [{'role': 'user', 'content': None}]}, 'messages[0]: this model has no chat template'),
    ({'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}]}, 'messages[0]: this model has no chat template'),
    ({'messages': [{'content': 'hi'}]}, 'messages[0]: this model has no chat template'),
    ({'messages': ASK, 'base_messages': [*ASK, {'role': 'assistant', 'content': None}]}, 'base_messages[1]: this model has no chat template'),
    ({'variants': ['base'], 'base_messages': [{'role': 'user', 'content': 5}]}, 'base_messages[0]: this model has no chat template'),
    ({'messages': ASK, 'template': 'x'}, 'template must be an object of turn prefixes'),
    ({'messages': ASK, 'template': {'sep': None}}, 'template prefixes (system, user, assistant) and sep must be text'),
])
def test_the_plain_turn_template_refuses_what_it_cannot_join_with_a_message_not_a_type_error(tmp_path, body, message):
    # Whether the plain template is used is known only after the model's own template is asked (a templated model
    # may take content parts), so this is after the runner, but still before any scope opens or token is generated.
    f = fixture(tmp_path, templated=False, kind='hf')
    r = f.client.post('/api/chat', json={**body, 'max_new_tokens': 2})
    assert r.status_code == 400, r.text
    assert message in r.json()['error'] and 'TypeError' not in r.json()['error'], r.json()
    assert f.calls == [] and f.runtime.scopes == []


def test_a_model_with_its_own_chat_template_still_ignores_the_plain_template_and_takes_content_parts(tmp_path):
    f = fixture(tmp_path, templated=True, kind='hf')
    parts = [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}]
    data = post(f, {'messages': parts, 'base_messages': parts, 'template': 'x', 'max_new_tokens': 2})
    assert data['variants'] == {'base': 'complete', 'current': 'complete'} and data['chat_template'] is True


# ------------------------------------------------------------------ each variant from the request's settings
def test_each_variant_generates_with_the_request_seed_and_sampling(tmp_path):
    f = fixture(tmp_path)
    post(f, {'messages': EDITED_CHAT, 'base_messages': BASE_CHAT, 'max_new_tokens': 2, 'seed': 7, 'temperature': 0.3, 'top_k': 5})
    post(f, {'variants': ['base'], 'base_messages': BASE_CHAT, 'max_new_tokens': 2, 'seed': 9})
    settings = [(v, kw['seed'], kw['temperature'], kw['top_k']) for v, _, kw in f.calls]
    assert settings == [('current', 7, 0.3, 5), ('base', 7, 0.3, 5), ('base', 9, 0.7, 40)]


def test_gguf_text_only_applies_to_the_base_conversation_too(tmp_path):
    f = fixture(tmp_path)
    for body in ({'messages': ASK, 'base_messages': [{'role': 'user', 'content': [{'type': 'image'}]}]},
                 {'variants': ['base'], 'base_messages': [{'role': 'user', 'content': [{'type': 'image'}]}]}):
        r = f.client.post('/api/chat', json=body)
        assert r.status_code == 400 and 'text-only' in r.json()['error']
    assert f.calls == [] and f.rendered == [] and f.runtime.scopes == []


def test_a_base_conversation_too_long_for_the_context_fails_before_the_edited_reply_runs(tmp_path):
    f = fixture(tmp_path)
    long = [{'role': 'user', 'content': 'x' * 60}]                 # 60 + template > the 64-token context
    r = f.client.post('/api/chat', json={'messages': ASK, 'base_messages': long, 'max_new_tokens': 3})
    assert r.status_code == 400
    error = r.json()['error']
    assert "The base model's conversation: Prompt/history uses" in error and 'No history was truncated' in error
    assert f.calls == [] and f.runtime.scopes == []               # nothing generated, no scope opened
    # The same conversation on its own (one prompt) keeps today's message.
    r = f.client.post('/api/chat', json={'variants': ['base'], 'base_messages': long, 'max_new_tokens': 3})
    assert r.status_code == 400 and r.json()['error'].startswith('ValueError: Prompt/history uses')
    assert f.calls == []


def test_each_model_fits_its_own_conversation_to_the_context(tmp_path):
    f = fixture(tmp_path)
    near = [{'role': 'user', 'content': 'x' * 40}]                 # 40 + 17 template characters: 7 left
    data = post(f, {'messages': ASK, 'base_messages': near, 'max_new_tokens': 20})
    assert data['generations']['current']['effective_output'] == 20
    assert data['generations']['base']['effective_output'] == 64 - len(ids_of(near))
    assert [kw['max_new_tokens'] for _, _, kw in f.calls] == [20, 64 - len(ids_of(near))]


# ------------------------------------------------------------------ Stop and interruptions
def test_a_stop_during_a_base_only_reply_reports_only_the_base_model(tmp_path, monkeypatch):
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    f = fixture(tmp_path, hold=('base', 2))
    thread, out = in_background(lambda: f.client.post('/api/chat', json={'variants': ['base'], 'base_messages': BASE_CHAT,
                                                                          'max_new_tokens': 4, 'stream': 'b-stop'}))
    assert f.gate.reached.wait(10)
    assert f.client.post('/api/generation/stop', json={'stream': 'b-stop'}).json() == {'ok': True, 'found': True}
    f.gate.release.set()
    thread.join(10)
    data = out['value'].json()
    assert data['stopped'] is True and data['variants'] == {'base': 'stopped'} and data['tokens'] == {'base': 3}
    assert data['base'] == 'w20 w21 w22' and data['current'] == '' and [v for v, _, _ in f.calls] == ['base']
    events = f.drain()
    assert {e['variant'] for e in events} == {'base'} and events[-1]['state'] == 'stopped'


def test_a_stop_during_an_edited_only_reply_reports_only_the_edited_model(tmp_path, monkeypatch):
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    f = fixture(tmp_path, hold=('current', 1))
    thread, out = in_background(lambda: f.client.post('/api/chat', json={'variants': ['current'], 'messages': EDITED_CHAT,
                                                                          'max_new_tokens': 4, 'stream': 'c-stop'}))
    assert f.gate.reached.wait(10)
    f.client.post('/api/generation/stop', json={'stream': 'c-stop'})
    f.gate.release.set()
    thread.join(10)
    data = out['value'].json()
    assert data['stopped'] is True and data['variants'] == {'current': 'stopped'} and data['tokens'] == {'current': 2}
    assert {e['variant'] for e in f.drain()} == {'current'}


def test_a_stop_before_a_base_only_request_starts_runs_nothing(tmp_path):
    f = fixture(tmp_path)
    f.client.post('/api/generation/stop', json={'stream': 'early'})
    data = post(f, {'variants': ['base'], 'base_messages': ASK, 'max_new_tokens': 4, 'stream': 'early'})
    assert data['stopped'] is True and data['variants'] == {'base': 'not_run'} and f.calls == []


def test_an_interrupted_base_reply_with_its_own_conversation_keeps_the_edited_reply(tmp_path):
    f = fixture(tmp_path, fail='base')
    data = post(f, {'messages': EDITED_CHAT, 'base_messages': BASE_CHAT, 'max_new_tokens': 4})
    assert data['interrupted'] is True and data['event']['code'] == 'runtime_budget'
    assert data['variants'] == {'base': 'interrupted', 'current': 'complete'}
    assert data['current'] == 'w10 w11 w12 w13' and data['base'] == 'w20 w21'
    assert [(v, ids) for v, ids, _ in f.calls] == [('current', ids_of(EDITED_CHAT)), ('base', ids_of(BASE_CHAT))]


def test_a_stop_during_the_edited_reply_with_two_conversations_leaves_the_base_model_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr('wsp.token_stream.INTERVAL', 0)
    f = fixture(tmp_path, hold=('current', 1))
    thread, out = in_background(lambda: f.client.post('/api/chat', json={'messages': EDITED_CHAT, 'base_messages': BASE_CHAT,
                                                                          'max_new_tokens': 4, 'stream': 'two-stop'}))
    assert f.gate.reached.wait(10)
    f.client.post('/api/generation/stop', json={'stream': 'two-stop'})
    f.gate.release.set()
    thread.join(10)
    data = out['value'].json()
    assert data['stopped'] is True and data['variants'] == {'current': 'stopped', 'base': 'not_run'}
    assert data['tokens'] == {'current': 2} and data['current'] == 'w10 w11' and data['base'] == ''
    assert [(v, ids) for v, ids, _ in f.calls] == [('current', ids_of(EDITED_CHAT))]      # nothing for the base model
    assert data['prompts'] == {'current': template(EDITED_CHAT), 'base': template(BASE_CHAT)}
    assert {e['variant'] for e in f.drain()} == {'current'}


def test_an_interrupted_edited_reply_with_two_conversations_leaves_the_base_model_not_run(tmp_path):
    f = fixture(tmp_path, fail='current')
    data = post(f, {'messages': EDITED_CHAT, 'base_messages': BASE_CHAT, 'max_new_tokens': 4, 'stream': 'two-fail'})
    assert data['interrupted'] is True and data['event']['code'] == 'runtime_budget'
    assert data['variants'] == {'current': 'interrupted', 'base': 'not_run'}
    assert data['current'] == 'w10 w11' and data['base'] == ''
    assert [(v, ids) for v, ids, _ in f.calls] == [('current', ids_of(EDITED_CHAT))]
    assert f.runtime.scopes == [('chat', len(ids_of(EDITED_CHAT)) + 4)]
    assert {e['variant'] for e in f.drain()} == {'current'}


# ------------------------------------------------------------------ steering
def test_steering_with_two_conversations_still_reaches_only_the_edited_model(sess):
    ws, c = sess.ws, model_client(sess)
    real, calls = ws.runner.generate, []

    def generate(ids, variant='current', extra=None, **kw):
        calls.append((variant, list(ids), 'steer' in kw))
        return real(ids, variant, extra, **kw)
    ws.runner.generate = generate
    n, dose = 6, 6.0
    data = c.post('/api/chat', json={**CHAT, 'messages': [{'role': 'user', 'content': 'abc'}],
                                     'base_messages': [{'role': 'user', 'content': 'bca'}], 'max_new_tokens': n,
                                     'temperature': 0, 'steer': [{'addr': 'L1N3', 'mode': 'add', 'dose': dose, 'unit': 'raw'}]}).json()
    assert calls == [('current', ws.encode('abc'), True), ('base', ws.encode('bca'), False)]
    assert data['current'] == ws.tok.decode(greedy_tokens(pushed_params(ws, 1, 3, dose), ws.cfg, ws.encode('abc'), n)).strip('\n')
    assert data['base'] == ws.tok.decode(greedy_tokens(ws.runner.params('base'), ws.cfg, ws.encode('bca'), n)).strip('\n')
    assert data['steering']['variant'] == 'current' and data['steering']['resolved'] is True
    assert data['prompts'] == {'current': 'abc', 'base': 'bca'}


def test_a_base_only_request_does_no_steering_work(sess, monkeypatch):
    from wsp import steering as S
    touched = []
    for name in ('prepare', 'describe', 'resolve', 'calibration'):
        monkeypatch.setattr(S, name, lambda *a, _n=name, **k: touched.append(_n))
    calls = []
    real = sess.ws.runner.generate

    def generate(ids, variant='current', extra=None, **kw):
        calls.append((variant, 'steer' in kw))
        return real(ids, variant, extra, **kw)
    sess.ws.runner.generate = generate
    data = model_client(sess).post('/api/chat', json={**{k: v for k, v in CHAT.items() if k != 'messages'}, 'variants': ['base'],
                                                      'base_messages': [{'role': 'user', 'content': 'abc'}], 'max_new_tokens': 3}).json()
    assert data['variants'] == {'base': 'complete'} and 'steering' not in data
    assert calls == [('base', False)] and touched == []
