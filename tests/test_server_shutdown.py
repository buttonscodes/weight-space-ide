"""Real Uvicorn/socket/signal regressions: empty Session, no weights or providers."""
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

import pytest

from wsp.lifecycle import Lifecycle

CHILD = r'''
import json,os,socket,time
from pathlib import Path
from types import SimpleNamespace
from wsp.server import create_app
from wsp.session import Session
from wsp.lifecycle import serve, IDEServer
root=Path(os.environ['PROBE_ROOT'])
sess=Session(str(root/'empty'))  # deliberately never open/build/train/load
app=create_app(sess,'http://127.0.0.1:0')
records=[]
def close(name):
    records.append(name+' start');(root/(name+'-started')).touch()
    time.sleep(float(os.environ.get('CLOSE_DELAY','0')))
    if name=='exports' and os.environ.get('CLOSE_FAULT'):raise RuntimeError('intentional cleanup fault')
    records.append(name+' done')
app.state.exports.shutdown=lambda:close('exports')
app.state.agent_auth.shutdown=lambda:close('auth')
sess.runs['fake']=SimpleNamespace(id='fake',native=True,proc=None,close=lambda:close('native'))
@app.get('/probe')
def probe():return {'subscribers':len(sess.subscribers),'records':records}
app.router.routes.insert(0,app.router.routes.pop())  # before static catch-all
def report():
    (root/'result.json').write_text(json.dumps({'subscribers':len(sess.subscribers),'records':records,
        'completed':app.state.ide_lifecycle.completed,'errors':app.state.ide_lifecycle.errors}))
original_cleanup=app.state.ide_lifecycle.cleanup
async def cleanup():
    await original_cleanup()
    report()  # before Uvicorn replays SIGTERM, which intentionally exits by signal
app.state.ide_lifecycle.cleanup=cleanup
original_signal=IDEServer.handle_exit
def handle(self,sig,frame):
    original_signal(self,sig,frame)
    (root/'signal-seen').touch()
IDEServer.handle_exit=handle
sock=socket.socket(fileno=int(os.environ['PROBE_FD']))
try:
    serve(app,host='127.0.0.1',port=0,log_level='info',sockets=[sock])
finally:
    (root/'result.json').write_text(json.dumps({'subscribers':len(sess.subscribers),'records':records,
        'completed':app.state.ide_lifecycle.completed,'errors':app.state.ide_lifecycle.errors}))
'''


def wait_until(fn, seconds=10):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        try:
            value=fn()
            if value:return value
        except (OSError,ValueError):pass
        time.sleep(.02)
    raise AssertionError('condition timed out')


@pytest.fixture
def server_process(tmp_path):
    children=[]
    def start(delay=0, fault=False):
        sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen();port=sock.getsockname()[1]
        log=(tmp_path/'server.log').open('w')
        env={**os.environ,'PROBE_ROOT':str(tmp_path),'PROBE_FD':str(sock.fileno()),
             'CLOSE_DELAY':str(delay),'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
        if fault:env['CLOSE_FAULT']='1'
        child=subprocess.Popen([sys.executable,'-u','-c',CHILD],env=env,pass_fds=[sock.fileno()],stdout=log,stderr=log,start_new_session=True)
        children.append((child,log));sock.close()
        url=f'http://127.0.0.1:{port}'
        def probe():
            with urllib.request.urlopen(url+'/probe',timeout=.3) as r:return json.load(r)
        wait_until(probe)
        return child,url,probe
    yield start
    for child,log in children:
        if child.poll() is None:child.kill()
        child.wait(timeout=5);log.close()


@pytest.mark.parametrize('connection',['none','open','closed'])
@pytest.mark.parametrize('sig',[signal.SIGINT,signal.SIGTERM])
def test_first_signal_drains_sse_and_cleans_resources(server_process,tmp_path,connection,sig):
    child,url,probe=server_process()
    stream=None
    if connection!='none':
        stream=urllib.request.urlopen(url+'/api/events',timeout=3)
        assert stream.readline()==b'retry: 1500\n'
        wait_until(lambda:probe()['subscribers']==1)
        if connection=='closed':
            stream.close();stream=None
            wait_until(lambda:probe()['subscribers']==0,2)
    started=time.monotonic();child.send_signal(sig);child.wait(timeout=5)
    elapsed=time.monotonic()-started
    (tmp_path/'timing.json').write_text(json.dumps({'connection':connection,'signal':int(sig),'seconds':elapsed,'pid':child.pid,'returncode':child.returncode}))
    if stream:stream.close()
    report=json.loads((tmp_path/'result.json').read_text())
    assert elapsed<3 and report['subscribers']==0 and report['completed']
    assert sorted(report['records'])==sorted(n+' '+phase for n in ('exports','auth','native') for phase in ('start','done'))
    assert report['errors']==[]
    logs=(tmp_path/'server.log').read_text()
    assert 'Traceback' not in logs and 'CancelledError' not in logs and 'KeyboardInterrupt' not in logs
    assert child.returncode in (0,-signal.SIGTERM)


@pytest.mark.parametrize('second_phase',['drain','cleanup'])
def test_double_interrupt_still_completes_cleanup(server_process,tmp_path,second_phase):
    child,url,probe=server_process(delay=.4)
    stream=urllib.request.urlopen(url+'/api/events',timeout=3);stream.readline()
    started=time.monotonic();child.send_signal(signal.SIGINT)
    marker='native-started' if second_phase=='cleanup' else 'signal-seen'
    wait_until(lambda:(tmp_path/marker).exists(),3)
    child.send_signal(signal.SIGINT);child.wait(timeout=5);stream.close()
    result=json.loads((tmp_path/'result.json').read_text())
    assert time.monotonic()-started<3 and result['completed'] and result['subscribers']==0
    assert all(n+' done' in result['records'] for n in ('exports','auth','native'))
    assert result['errors']==[] and child.returncode==0
    assert 'Traceback' not in (tmp_path/'server.log').read_text()


def test_cleanup_fault_is_visible_and_other_resources_still_close(server_process,tmp_path):
    child,_,_=server_process(fault=True)
    child.send_signal(signal.SIGINT);child.wait(timeout=5)
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['completed'] and result['errors']
    assert 'auth done' in result['records'] and 'native done' in result['records']
    assert child.returncode!=0
    assert 'intentional cleanup fault' in (tmp_path/'server.log').read_text()


def test_cleanup_timeout_kills_owned_child_and_preserves_unrelated_sentinel():
    owned=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    try:
        lifecycle=Lifecycle(lambda:[('blocked provider',lambda:time.sleep(.5),lambda:[owned])],timeout=.05)
        asyncio.run(lifecycle.cleanup())
        assert lifecycle.completed and any('TimeoutError' in e for e in lifecycle.errors)
        assert owned.poll() is not None and sentinel.poll() is None
    finally:
        for p in (owned,sentinel):
            if p.poll() is None:p.kill()
            p.wait(timeout=5)


def test_export_shutdown_cancels_and_joins_inflight_worker():
    import threading
    from wsp.export_jobs import ExportJobs
    jobs=ExportJobs();cancel=threading.Event();exited=threading.Event()
    def work():cancel.wait(2);exited.set()
    worker=threading.Thread(target=work);worker.start()
    jobs.jobs['fixture']={'cancel':cancel,'thread':worker}
    jobs.shutdown()
    assert cancel.is_set() and exited.is_set() and not worker.is_alive()


def test_unexpected_startup_keyboard_interrupt_is_not_suppressed(monkeypatch):
    from wsp import lifecycle
    app=SimpleNamespace(state=SimpleNamespace(ide_lifecycle=Lifecycle(lambda:[])))
    def interrupt(self,**kwargs):raise KeyboardInterrupt()
    monkeypatch.setattr(lifecycle.IDEServer,'run',interrupt)
    with pytest.raises(KeyboardInterrupt):lifecycle.serve(app)


def test_stopped_legacy_process_is_still_owned_and_reaped(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    from wsp.session import Session
    from wsp import harness
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    try:
        sess=Session(str(tmp_path/'empty'))
        sess.runs['stopped']=SimpleNamespace(id='stopped',native=False,status='stopped',proc=child)
        monkeypatch.setattr(harness,'stop',lambda run:None)  # failed earlier stop
        app=create_app(sess,'http://127.0.0.1:0')
        with TestClient(app):pass
        assert child.poll() is not None and app.state.ide_lifecycle.errors==[]
    finally:
        if child.poll() is None:child.kill()
        child.wait(timeout=5)
