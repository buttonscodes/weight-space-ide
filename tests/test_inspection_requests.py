"""Inspection contracts with fake workspaces only: no model, weights, training or inference."""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp.inspection import InspectionRequests, InspectionBusy
from wsp import server, hierarchy
from wsp.session import Session


class FakeStore(dict):
    def dtype(self, name): return 'F32'
    def local(self, name): return True


class FakeWorkspace:
    def __init__(self):
        self.version = 0
        self.lock = RLock()
        self.groups = {'chosen': {'members': ['blocks.0.mlp.neuron.0'], 'note': ''}}
        self.calib, self.keep, self.patches = {}, {}, []
        self.runner_loaded, self.can_run, self.run_error = True, True, None
        self.kind, self.name, self.dir = 'gguf', 'fake', '/nonexistent-inspection-fixture'
        self.store = FakeStore(W_in=(64, 8), W_out=(8, 64), vector=(8,))
        self.struct = SimpleNamespace(recognised=True, d_vocab=0, doc=lambda _: 'fake tensor',
            layers=[SimpleNamespace(mlp={'W_out': SimpleNamespace(name='W_out')},
                                    attn={'W_O': SimpleNamespace(name='W_out')})])
        self.sem = SimpleNamespace(vocab_bytes=lambda: 0)
        self.reads = 0
    def read(self, *args):
        self.reads += 1
        raise AssertionError('passive inspection must never read weights')
    def shape(self, name): return self.store[name]
    def pieces(self, name): return []
    def touched(self): return []
    def editable(self, name): return True
    def group_members(self, name): return self.groups[name]['members']
    def resolve(self, addr):
        if addr == 'group:chosen': return sum((self.resolve(a) for a in self.group_members('chosen')), [])
        if '.neuron.' in addr:
            n = int(addr.rsplit('.', 1)[1])
            return [('W_in', (n, slice(None))), ('W_out', (slice(None), n))]
        if '.head.' in addr: return [('W_out', (slice(None), 0))]
        if addr == 'vector': return [('vector', ())]
        return [('W_in', ()), ('W_out', ())]


@pytest.fixture
def env(tmp_path, monkeypatch):
    sess = Session(str(tmp_path))
    sess.ws = FakeWorkspace()
    monkeypatch.setattr(server.an, 'kept_roles', lambda *a: ['resid'])
    # Fail loudly if state/status accidentally evaluates a model.
    monkeypatch.setattr(sess, 'tests', lambda *a, **kw: pytest.fail('status ran tests'))
    monkeypatch.setattr(sess, 'drift', lambda *a, **kw: pytest.fail('status ran inference'))
    client = TestClient(server.create_app(sess, 'http://127.0.0.1:0'), raise_server_exceptions=False)
    return sess, client


def scope(client):
    data = client.get('/api/status').json()
    return {'workspace_id': data['workspace_id'], 'version': data['version']}


def test_passive_metadata_neuron_group_tree_never_reads(env, monkeypatch):
    sess, client = env
    for name in ('neuron_card', 'head_card', 'direction_tokens', 'region_stats'):
        monkeypatch.setattr(server.an, name, lambda *a: pytest.fail('implicit analysis'))
    for addr in ('L0N0', 'group:chosen', 'L0H0', 'vector'):
        out = client.get('/api/node', params={'addr': addr, 'fetch': 1})
        assert out.status_code == 200, out.text
        assert out.json()['stats']['deferred']
        assert not any(k in out.json() for k in ('neuron', 'head', 'direction'))
    out = client.get('/api/tree', params={'addr': 'group:chosen'})
    assert out.status_code == 200, out.text
    assert out.json()['children'][0]['norm'] is None
    assert sess.ws.reads == 0
    assert client.get('/api/inspection-capabilities').json()['passive_metadata']
    status = client.get('/api/status').json()
    assert status['tests'] is None and status['drift']['kl'] is None


def test_explicit_analysis_shared_retry_and_scope(env, monkeypatch):
    sess, client = env
    started, release = Event(), Event()
    calls = []
    def compute(w, l, n):
        calls.append((w, n)); started.set()
        assert release.wait(5)
        return {'neuron': n}
    monkeypatch.setattr(server.an, 'neuron_card', compute)
    params = dict(addr='L0N0', analysis=1, **scope(client))
    with ThreadPoolExecutor(3) as pool:
        first = pool.submit(client.get, '/api/node', params=params)
        assert started.wait(3)
        second = pool.submit(client.get, '/api/node', params=params)
        # Passive metadata/status remain responsive while the workspace weight lock is held.
        passive = pool.submit(client.get, '/api/node', params={'addr': 'L0N1'})
        assert passive.result(2).status_code == 200
        assert client.get('/api/status').status_code == 200
        release.set()
        assert first.result(3).status_code == second.result(3).status_code == 200
    assert len(calls) == 1
    assert client.get('/api/node', params=params).status_code == 200
    assert len(calls) == 1
    sess.ws.version += 1
    assert client.get('/api/node', params=params).status_code == 409
    assert client.get('/api/node', params=dict(addr='L0N0', analysis=1, **scope(client))).status_code == 200
    assert len(calls) == 2
    previous = scope(client)
    sess.ws = FakeWorkspace()  # same name and version zero, new identity
    assert client.get('/api/node', params=dict(addr='L0N0', analysis=1, **previous)).status_code == 409
    assert client.get('/api/node', params=dict(addr='L0N0', analysis=1, **scope(client))).status_code == 200
    assert len(calls) == 3


def test_failed_analysis_retry_is_explicit(env, monkeypatch):
    sess, client = env
    calls = []
    def compute(*a):
        calls.append(1)
        if len(calls) == 1: raise ValueError('decode failed')
        return {'ok': True}
    monkeypatch.setattr(server.an, 'neuron_card', compute)
    params = dict(addr='L0N0', analysis=1, **scope(client))
    assert client.get('/api/node', params=params).status_code == 400
    assert client.get('/api/node', params={'addr': 'L0N0'}).status_code == 200
    assert len(calls) == 1
    assert client.get('/api/node', params=params).json()['neuron']['ok']
    assert len(calls) == 2


def test_remote_fetch_is_separate_from_analysis_consent(env, monkeypatch):
    sess, client = env
    sess.ws.sem.vocab_bytes = lambda: 1234
    calls = []
    monkeypatch.setattr(server.an, 'neuron_card', lambda *a: calls.append(1) or {})
    assert client.get('/api/node', params={'addr': 'L0N0', 'fetch': 1}).json()['needs_fetch'] == 1234
    assert calls == []
    assert client.get('/api/node', params={'addr': 'L0N0', 'analysis': 1, **scope(client)}).status_code == 409
    assert client.get('/api/node', params={'addr': 'L0N0', 'analysis': 1, 'fetch': 1, **scope(client)}).status_code == 200
    assert calls == [1]


def test_queued_old_version_rejected_before_compute(env, monkeypatch):
    sess, client = env
    entered = Event()
    calls = []
    monkeypatch.setattr(server.an, 'neuron_card', lambda *a: calls.append(1) or {})
    params = dict(addr='L0N0', analysis=1, **scope(client))
    # A lock adapter signals when the handler has reached the weight lock.
    class Gate:
        def __enter__(self): entered.set(); sess.ws.real_lock.acquire()
        def __exit__(self, *a): sess.ws.real_lock.release()
    sess.ws.real_lock = RLock()
    sess.ws.lock = Gate()
    with ThreadPoolExecutor(1) as pool:
        with sess.ws.real_lock:
            future = pool.submit(client.get, '/api/node', params=params)
            assert entered.wait(3)
            sess.ws.version += 1
        assert future.result(3).status_code == 409
    assert calls == []


def test_group_membership_invalidates_metadata(env):
    sess, client = env
    old = scope(client)
    assert client.get('/api/tree', params={'addr': 'group:chosen', **old}).json()['children'][0]['addr'].endswith('.0')
    sess.ws.groups['chosen']['members'] = ['blocks.0.mlp.neuron.1']
    assert client.get('/api/tree', params={'addr': 'group:chosen', **old}).status_code == 409
    assert client.get('/api/tree', params={'addr': 'group:chosen', **scope(client)}).json()['children'][0]['addr'].endswith('.1')


def test_singleflight_pending_limit_completed_eviction_and_failure():
    cache = InspectionRequests(limit=1, max_pending=1)
    entered, release = Event(), Event()
    def calculate(): entered.set(); assert release.wait(3); return 42
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(cache.run, 'a', calculate)
        assert entered.wait(2)
        second = pool.submit(cache.run, 'a', lambda: pytest.fail('duplicate'))
        with pytest.raises(InspectionBusy): cache.run('b', lambda: 99)
        release.set()
        assert first.result(2) == second.result(2) == 42
    cache.run('b', lambda: 99)
    assert list(cache.completed) == ['b']
    with pytest.raises(ValueError): cache.run('c', lambda: (_ for _ in ()).throw(ValueError('failed')))
    assert cache.pending == {}
    assert cache.run('c', lambda: 100) == 100


def test_explicit_stats_svd_and_variant_keys(env, monkeypatch):
    sess, client = env
    calls = []
    monkeypatch.setattr(server.an, 'region_stats', lambda *a: calls.append('stats') or {'current': {'norm': 2}})
    monkeypatch.setattr(server.an, 'svd_view', lambda w, addr, k, variant: calls.append(variant) or {'rank': 2})
    assert client.get('/api/node', params={'addr': 'L0N0'}).status_code == 200
    assert calls == []
    for _ in range(2):
        assert client.get('/api/node', params={'addr': 'L0N0', 'statistics': 1, **scope(client)}).status_code == 200
        assert client.get('/api/svd', params={'addr': 'W_in', **scope(client)}).status_code == 200
    assert calls == ['stats', 'current']
    assert client.get('/api/svd', params={'addr': 'W_in', 'variant': 'base', **scope(client)}).status_code == 200
    assert calls == ['stats', 'current', 'base']


def test_workspace_switch_during_analysis_discards_old_result(env, monkeypatch):
    sess, client = env
    started, release = Event(), Event()
    def calculate(*a): started.set(); assert release.wait(3); return {'ok': True}
    monkeypatch.setattr(server.an, 'neuron_card', calculate)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(client.get, '/api/node', params={'addr': 'L0N0', 'analysis': 1, **scope(client)})
        assert started.wait(2)
        sess.ws = FakeWorkspace()
        release.set()
        assert future.result(2).status_code == 409


def test_ablation_suite_edits_invalidate_and_hold_session_lifecycle(env, monkeypatch):
    sess, client = env
    sess.task['suite'] = 's'
    sess.ws.runner = object()  # inert sentinel, no model
    sess.ws.encode = lambda p: [1]
    sess.ws.tok = SimpleNamespace(encode_token=lambda t: [ord(t)])
    suites = {'s': {'cases': [{'type': 'next_token', 'prompt': 'p', 'expect': 'a'}]}}
    monkeypatch.setattr(server.testsuite, 'load_suite', lambda w, n: suites[n])
    monkeypatch.setattr(server.testsuite, 'save_suite', lambda w, n, v: suites.__setitem__(n, v))
    monkeypatch.setattr(server.testsuite, 'list_suites', lambda w: [])
    calls = []
    def scan(w, pairs, variant):
        assert sess.lock._is_owned() and w.lock._is_owned()
        calls.append(pairs[0][1]); return {'target': pairs[0][1]}
    monkeypatch.setattr(server.an, 'ablation_scan', scan)
    assert client.post('/api/ablation', json=scope(client)).json()['target'] == ord('a')
    assert client.post('/api/ablation', json=scope(client)).json()['target'] == ord('a')
    old = scope(client)
    assert client.post('/api/suite', json={'name': 's', 'suite': {'cases': [{'type': 'next_token', 'prompt': 'p', 'expect': 'b'}]}}).status_code == 200
    assert scope(client)['workspace_id'] != old['workspace_id']
    assert client.post('/api/ablation', json=old).status_code == 409
    assert client.post('/api/ablation', json=scope(client)).json()['target'] == ord('b')
    assert calls == [ord('a'), ord('b')]


def test_legacy_expensive_requests_fail_closed(env, monkeypatch):
    sess, client = env
    for name in ('region_stats', 'neuron_card', 'matrix_view', 'svd_view', 'wiring'):
        monkeypatch.setattr(server.an, name, lambda *a: pytest.fail('legacy client started work'))
    for path, params in [('/api/node', {'addr': 'L0N0', 'analysis': 1}),
                         ('/api/node', {'addr': 'L0N0', 'statistics': 1}),
                         ('/api/matrix', {'addr': 'W_in'}), ('/api/svd', {'addr': 'W_in'})]:
        assert client.get(path, params=params).status_code == 428
    assert client.get('/api/node', params={'addr': 'L0N0'}).status_code == 200


def test_asset_graph_is_fingerprinted_and_snapshotted(env, monkeypatch, tmp_path):
    import re
    sess, client = env
    for route in ('/', '/index.html'):
        page = client.get(route)
        assert page.headers['cache-control'] == 'no-store'
        entry = re.search(r'src="(/assets/[^\"]+/js/app.js)"', page.text).group(1)
        response = client.get(entry)
        assert response.status_code == 200 and 'immutable' in response.headers['cache-control']
        weights = entry.rsplit('/', 1)[0] + '/weights.js'
        assert client.get(weights).status_code == 200
        assert 'Calculate vocabulary readings' in client.get(weights).text
    # Build a tiny alternate asset graph and then change disk contents: server keeps its snapshot.
    (tmp_path / 'js').mkdir()
    (tmp_path / 'js' / 'app.js').write_text('first')
    (tmp_path / 'index.html').write_text('<script src="js/app.js?inspection=1"></script>')
    monkeypatch.setattr(server, 'WEB', str(tmp_path))
    second = TestClient(server.create_app(sess, 'http://127.0.0.1:0'))
    entry = re.search(r'src="([^\"]+)"', second.get('/').text).group(1)
    (tmp_path / 'js' / 'app.js').write_text('second')
    assert second.get(entry).text == 'first'
    third = TestClient(server.create_app(sess, 'http://127.0.0.1:0'))
    new_entry = re.search(r'src="([^\"]+)"', third.get('/').text).group(1)
    assert new_entry != entry and third.get(new_entry).text == 'second'


def test_bounded_wiring_shared_scoped_cancel_and_retry(env, monkeypatch):
    sess, client = env
    started, release = Event(), Event()
    calls=[]
    def calculate(w,top,layers,cancel,progress):
        calls.append(tuple(layers));started.set()
        assert release.wait(3)
        if cancel.is_set():raise ValueError('Wiring analysis cancelled')
        return {'window':layers,'bounds':{'slab_values':64}}
    monkeypatch.setattr(server.an,'wiring',calculate)
    params=dict(layers='0',top=4000,**scope(client))
    assert client.get('/api/wiring',params={'layers':'0'}).status_code==428
    with ThreadPoolExecutor(2) as pool:
        first=pool.submit(client.get,'/api/wiring',params=params)
        assert started.wait(2)
        second=pool.submit(client.get,'/api/wiring',params=params)
        release.set()
        assert first.result(3).status_code==second.result(3).status_code==200
    assert calls==[(0,)]
    # New revision cannot reuse the completed result; stale cancel cannot target it.
    sess.ws.version+=1
    assert client.post('/api/wiring/cancel',json=params).status_code==409
    params=dict(params,**scope(client));started.clear();release.clear()
    with ThreadPoolExecutor(1) as pool:
        future=pool.submit(client.get,'/api/wiring',params=params)
        assert started.wait(2)
        assert client.post('/api/wiring/cancel',json=params).json()=={'cancelled':True}
        release.set();response=future.result(3);assert response.status_code==400
        assert 'cancelled' in response.text
    assert client.get('/api/wiring',params=params).status_code==200
    assert calls==[(0,),(0,),(0,)]  # explicit retry, cancelled result not cached
