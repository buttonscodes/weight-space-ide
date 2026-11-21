"""Atlas integration guards: fake workspaces, no model execution."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time
from test_inspection_requests import env, scope
from wsp import server
from wsp.atlas import guide


def test_build_atomic_scope_defaults_and_workspace_lock(env, monkeypatch, tmp_path):
    sess, client = env
    monkeypatch.setattr(sess.ws, 'dir', str(tmp_path))       # the build job takes the atlas build lock in <dir>/atlas
    started, release = Event(), Event()
    calls = []
    def build(w, **kw):
        calls.append(kw); started.set()
        assert release.wait(5)
        kw['log']('finished')
    monkeypatch.setattr(server.at, 'build', build)
    params = scope(client)
    assert client.post('/api/atlas/build', json={}).status_code == 428
    with ThreadPoolExecutor(4) as pool:
        responses = list(pool.map(lambda _: client.post('/api/atlas/build', json=params), range(4)))
        assert all(r.status_code == 200 for r in responses)
        assert started.wait(2) and len(calls) == 1
        assert calls[0]['stages'] == ['weights']
        assert not sess.ws.lock.acquire(blocking=False)
        assert client.post('/api/atlas/build', json={**params, 'stages':['observed']}).status_code == 409
        release.set()
    for _ in range(100):
        if not sess.atlas_job['running']: break
        time.sleep(.01)
    assert not sess.atlas_job['running'] and sess.atlas_job['error'] is None
    assert sess.atlas_job['workspace_id'] == params['workspace_id']
    sess.ws.version += 1
    assert client.post('/api/atlas/build', json=params).status_code == 409
    assert len(calls) == 1 and sess.ws.reads == 0


def test_attribute_explicit_shared_retry_and_stale(env, monkeypatch):
    sess, client = env
    started, release = Event(), Event()
    calls = []
    def compute(*args):
        calls.append(args); started.set()
        assert release.wait(5)
        return {'top': []}
    monkeypatch.setattr(guide, 'attribute', compute)
    monkeypatch.setattr(guide, 'top_script', lambda *args: '')
    params = dict(prompt='x', target='y', **scope(client))
    assert client.post('/api/atlas/attribute', json={'prompt':'x','target':'y'}).status_code == 428
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(client.post, '/api/atlas/attribute', json=params)
        assert started.wait(2)
        second = pool.submit(client.post, '/api/atlas/attribute', json=params)
        release.set()
        assert first.result().status_code == second.result().status_code == 200
    assert len(calls) == 1
    sess.ws.version += 1
    assert client.post('/api/atlas/attribute', json=params).status_code == 409
    monkeypatch.setattr(guide, 'attribute', lambda *args: (_ for _ in ()).throw(RuntimeError('retry me')))
    params = {**params, **scope(client)}
    assert client.post('/api/atlas/attribute', json=params).status_code == 400
    monkeypatch.setattr(guide, 'attribute', lambda *args: {'top': []})
    assert client.post('/api/atlas/attribute', json=params).status_code == 200
    assert sess.ws.reads == 0


def test_the_observed_stage_is_accepted_for_a_worker_model_and_refused_only_when_the_model_cannot_run(env, monkeypatch, tmp_path):
    """A GGUF model runs in the model worker, where the observed stage now runs one window per job: the status offers
    it and the build route claims a job with it. A model that cannot be run at all is still refused in plain words
    before any job or build starts."""
    from wsp import atlas, worker_jobs
    sess, client = env
    monkeypatch.setattr(sess.ws, 'dir', str(tmp_path))
    calls = []
    monkeypatch.setattr(server.at, 'build', lambda w, **kw: calls.append(kw))
    params = scope(client)
    monkeypatch.setattr(sess.ws, 'can_run', True, raising=False)
    assert atlas.observed_unavailable(sess.ws) is None                    # a model run in this process
    monkeypatch.setattr(worker_jobs, 'uses_worker', lambda ws: True)
    assert atlas.observed_unavailable(sess.ws) is None                    # and one run in the model worker
    assert client.post('/api/atlas/build', json={**params, 'stages': ['weights', 'observed']}).status_code == 200
    for _ in range(100):
        if not sess.atlas_job['running']: break
        time.sleep(.01)
    assert len(calls) == 1 and calls[0]['stages'] == ['observed', 'weights'] and sess.atlas_job['error'] is None
    # A model that cannot be run at all gets its own sentence and a 422 before a job is claimed.
    monkeypatch.setattr(worker_jobs, 'uses_worker', lambda ws: False)
    monkeypatch.setattr(sess.ws, 'can_run', False, raising=False)
    reason = atlas.observed_unavailable(sess.ws)
    assert reason.startswith('This model cannot be run here')
    r = client.post('/api/atlas/build', json={**params, 'stages': ['observed']})
    assert r.status_code == 422 and r.json()['detail'] == reason + ' Untick Observed and build again.'
    assert len(calls) == 1 and not sess.atlas_job['running']
    # The worker case is not a way around it: a worker model that cannot be run is refused the same way.
    monkeypatch.setattr(worker_jobs, 'uses_worker', lambda ws: True)
    assert client.post('/api/atlas/build', json={**params, 'stages': ['weights', 'observed']}).status_code == 422
    assert len(calls) == 1
