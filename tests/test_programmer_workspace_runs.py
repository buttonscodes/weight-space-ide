"""The server keeps its programmer runs across a workspace switch. Each run's meta names the workspace it ran in, and
the runs listing names the open one, so the Programmer tab shows only the open workspace's runs (web/js/work.js
lastRunHere). A fake bridge stands in for the provider; no CLI, model or network call is involved.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wsp import harness  # noqa: E402
from test_programmer_conversations import PATCH, fake_bridge, make_app, make_session, wait_for  # noqa: E402,F401


def test_every_run_names_its_workspace_and_the_listing_names_the_open_one(tmp_path, fake_bridge, monkeypatch):
    a, b = tmp_path / 'ws-a', tmp_path / 'ws-b'
    os.makedirs(a); os.makedirs(b)
    sess = make_session(tmp_path / 'root', a)
    with make_app(sess, monkeypatch) as client:
        native = client.post('/api/harness/start', json={'harness': 'claude', 'mode': 'tools'}).json()
        run = sess.runs[native['id']]
        wait_for(lambda: run.status == 'idle')
        assert native['workspace'] == os.path.realpath(a)
        assert client.post('/api/harness/close', json={'id': run.id}).status_code == 200
        wait_for(lambda: run._retired.is_set())
        script = tmp_path / 'agent.py'
        script.write_text('print("done")\n')
        legacy = harness.start(sess, 'http://x', 'custom', 'tools', custom=f'{sys.executable} {script}')
        wait_for(lambda: legacy.status in ('done', 'failed'))
        assert legacy.meta()['workspace'] == os.path.realpath(a)
        listed = client.get('/api/harness').json()
        assert listed['workspace'] == os.path.realpath(a)
        assert [r['workspace'] for r in listed['runs']] == [os.path.realpath(a)] * 2
        # Another workspace opened: the runs stay, the listing names the new one.
        sess.ws = SimpleNamespace(dir=str(b), patches=[PATCH], name='other', lock=threading.RLock(), _runner=None,
                                  run_error=None, calib={}, keep={})
        listed = client.get('/api/harness').json()
        assert listed['workspace'] == os.path.realpath(b)
        assert all(r['workspace'] != listed['workspace'] for r in listed['runs'])
