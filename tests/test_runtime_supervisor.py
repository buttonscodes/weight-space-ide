"""Model-free runtime transport; the separately named HF test needs model-test consent."""
import json
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import urllib.request

import psutil
import pytest
from wsp.runtime_config import resolve
from wsp.runtime_supervisor import RuntimeClient, descriptor, fingerprint, write
from wsp.runtime_events import RuntimeRefusal
from wsp.runtime_policy import admission, observations


def wait(fn, seconds=15):
    until=time.monotonic()+seconds
    while time.monotonic()<until:
        try:
            value=fn()
            if value:return value
        except (OSError,ValueError):pass
        time.sleep(.025)
    raise AssertionError('deadline expired')


def supervisor_script(tmp_path, *, inert=False):
    """Owned no-fork child; synthetic metrics avoid platform enumeration privileges."""
    script=tmp_path/'supervisor-fixture.py'
    script.write_text("""
import json, os, sys, psutil
from pathlib import Path
from wsp import runtime_supervisor as r, memory_guard as m
from wsp.runtime_config import resolve
# This fixture child never forks. Production descendant discovery is tested
# separately; only this isolated interpreter gets synthetic process metrics.
psutil.Process.children=lambda self,recursive=False: []
m.process_memory=lambda pid: dict(tree_rss=1024**2,tree_footprint=1024**2,pids=[pid],identities={pid:psutil.Process(pid).create_time()})
def snapshot():
    path=Path(os.environ['HOME'])/'sample.json'
    return json.loads(path.read_text())
m.snapshot=snapshot
""" + ("""
original=r.subprocess.Popen
r.subprocess.Popen=lambda command,**kw: original([sys.executable,'-c','import time;time.sleep(60)'],**kw) if command[:3]==[sys.executable,'-m','wsp'] else original(command,**kw)
s=resolve(env=dict(HOME=os.environ['HOME']),overrides=dict(backend='cpu',floor_gib=20,preferred_gib=25,resident_seconds=30))
raise SystemExit(r.supervise(s,empty=True))
""" if inert else """
from wsp.__main__ import main
main()
"""))
    write(tmp_path/'sample.json',dict(available=71*1024**3,total=128*1024**3,pressure=1,swap_used=0,swap_out=0,compressed=0))
    return script


@contextmanager
def inert_supervisor(tmp_path):
    script=supervisor_script(tmp_path,inert=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('WSP_')}
    env.update(HOME=str(tmp_path),XDG_CACHE_HOME=str(tmp_path/'cache'),PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    with (tmp_path/'log').open('w') as log:
        proc=subprocess.Popen([sys.executable,str(script)],env=env,stdout=log,stderr=log,start_new_session=True)
        owned=None
        try:
            folder=wait(lambda:next((tmp_path/'cache').rglob('settings.json'),None)).parent
            # The log can exist before its first line is written: wait for the line, not just the file.
            start=wait(lambda:(lines:=(folder/'watchdog.jsonl').read_text().splitlines()) and json.loads(lines[0]))
            child=psutil.Process(start['pid']);owned=(child.pid,child.create_time())
            def request(n,value=None,action='bind'):
                write(folder/'request.json',dict(id=n,action=action,descriptor=value))
                def reply():
                    data=json.loads((folder/'reply.json').read_text())
                    return data if data.get('id')==n else None
                return wait(reply,5)
            yield proc,folder,request
        finally:
            if proc.poll() is None:proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=8)
            if owned:
                try:
                    child=psutil.Process(owned[0])
                    if child.create_time()==owned[1] and child.status()!=psutil.STATUS_ZOMBIE:
                        child.terminate();child.wait(timeout=3)
                except psutil.NoSuchProcess:pass
            rows=[json.loads(line) for line in (folder/'watchdog.jsonl').read_text().splitlines()]
            assert rows[-1]['remaining']==[] and rows[-1]['errors']==[],rows[-1]
            assert proc.returncode==0,(tmp_path/'log').read_text()


@pytest.mark.parametrize('connection',['none','open','closed'])
@pytest.mark.parametrize('sig',[signal.SIGINT,signal.SIGTERM])
def test_regular_serve_supervisor_shutdown(tmp_path,connection,sig):
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    env={k:v for k,v in os.environ.items() if not k.startswith('WSP_')}
    env.update(HOME=str(tmp_path),XDG_CONFIG_HOME=str(tmp_path/'config'),XDG_CACHE_HOME=str(tmp_path/'cache'),
               PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    stream=None
    with (tmp_path/'server.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,str(supervisor_script(tmp_path)),'serve','--empty','--root',str(tmp_path/'empty'),
                               '--port',str(port)],env=env,stdout=log,stderr=log,start_new_session=True)
        known=[]
        try:
            def request():
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/runtime',timeout=.3) as r:return json.load(r)
            result=wait(request)
            assert result['supervised'] and not result['reserved']
            runtime_log=next((tmp_path/'cache').rglob('watchdog.jsonl'))
            started=json.loads(runtime_log.read_text().splitlines()[0])
            child=psutil.Process(started['pid']);known=[(child.pid,child.create_time())]
            if connection!='none':
                stream=urllib.request.urlopen(f'http://127.0.0.1:{port}/api/events',timeout=2)
                assert stream.readline()==b'retry: 1500\n'
                if connection=='closed':stream.close();stream=None
            if connection == 'open' and sig == signal.SIGINT:
                # Exercise real supervisor sampling with a metadata-only fake
                # reservation: no runner/tensor is ever constructed.
                runtime_dir=next((tmp_path/'cache').rglob('settings.json')).parent
                settings=json.loads((runtime_dir/'settings.json').read_text())
                v=descriptor(fake_ws(tmp_path),8192)
                write(runtime_dir/'request.json',dict(id='probe',action='bind',descriptor=v))
                def sampled():
                    return any(json.loads(line)['event']=='sample' for line in (runtime_dir/'watchdog.jsonl').read_text().splitlines())
                wait(sampled,5)
                write(runtime_dir/'request.json',dict(id='release',action='release'))
                wait(lambda:json.loads((runtime_dir/'reply.json').read_text()).get('id')=='release')
            started=time.monotonic();proc.send_signal(sig)
            if connection == 'open' and sig == signal.SIGINT:
                time.sleep(.05)
                if proc.poll() is None:proc.send_signal(sig)
            proc.wait(timeout=6)
            assert time.monotonic()-started<6
            assert proc.returncode==0,(tmp_path/'server.log').read_text()
            logs=list((tmp_path/'cache').rglob('watchdog.jsonl'));assert len(logs)==1
            records=[json.loads(x) for x in logs[0].read_text().splitlines()]
            if connection != 'open' or sig != signal.SIGINT:
                assert not any(r['event']=='sample' for r in records) # idle does not sample machine/load
            assert records[-1]['remaining']==[] and records[-1]['errors']==[]
            for pid,born in known:
                assert not psutil.pid_exists(pid) or psutil.Process(pid).create_time()!=born
            assert not list(tmp_path.rglob('workspace.json'))
            assert not (tmp_path/'config').exists()
        finally:
            if stream:stream.close()
            if proc.poll() is None:proc.send_signal(signal.SIGTERM)
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                from wsp.memory_guard import stop_group
                stop_group(proc,dict(known+[(proc.pid,psutil.Process(proc.pid).create_time())]))


def fake_ws(tmp_path):
    path=tmp_path/'fixture.gguf';path.write_bytes(b'not model weights')
    return SimpleNamespace(lock=threading.RLock(),store=SimpleNamespace(path=str(path),nbytes=lambda:17),meta={'base_sha256':'known'},
        n_params=lambda:32,kind='gguf',hf_config={'hidden_size':8,'num_hidden_layers':1,'vocab_size':32},
        struct=SimpleNamespace(d_model=8,n_layers=1,d_vocab=32),patches=[],_runner=None)


def test_binding_tracks_model_size_and_patches(tmp_path):
    ws=fake_ws(tmp_path);a=descriptor(ws,64)
    ws.store.nbytes=lambda:1000
    ws.patches=[SimpleNamespace(meta=lambda:{'stored':256})]
    b=descriptor(ws,64)
    assert fingerprint(a)!=fingerprint(b)
    assert b['forecast']['patches']==3072
    assert fingerprint(b)!=fingerprint(descriptor(ws,128))
    Path(ws.store.path).write_bytes(b'different')
    assert descriptor(ws,64)['stat']!=a['stat']


def test_client_refuses_expired_wrong_binding_and_unsafe(tmp_path,monkeypatch):
    ws=fake_ws(tmp_path);client=RuntimeClient(tmp_path,resolve(env={'HOME':str(tmp_path)}))
    monkeypatch.setenv('WSP_GGML_DEVICE','cpu')
    value=descriptor(ws,8192);client.binding=fingerprint(value)
    for lease in [dict(ok=True,time=time.time()-10,pid=os.getpid(),binding=client.binding),
                  dict(ok=False,time=time.time(),pid=os.getpid(),binding=client.binding),
                  dict(ok=True,time=time.time(),pid=os.getpid(),binding='other'),
                  dict(ok=True,time=time.time(),pid=0,binding=client.binding)]:
        write(tmp_path/'lease.json',lease)
        with pytest.raises(RuntimeRefusal) as exc:client.require(ws)
        assert exc.value.record['code'] in {'watchdog_unavailable','lease_revoked'}


def test_identity_is_deferred_and_guarded(tmp_path,monkeypatch):
    ws=fake_ws(tmp_path);ws._metadata_only=True;ws.dir=str(tmp_path);ws.meta['base_sha256']=None
    client=RuntimeClient(tmp_path,resolve(env={'HOME':str(tmp_path)}));requests=[]
    def respond(action,value=None):
        requests.append((action,value))
        write(tmp_path/'lease.json',dict(ok=True,time=time.time(),pid=os.getpid(),binding=fingerprint(value)))
    monkeypatch.setattr(client,'_request',respond)
    client.require(ws,loading=True)
    assert not ws._metadata_only and ws.meta['base_sha256']
    assert requests[0][1]['runner'] is False and requests[-1][1]['runner'] is True
    assert json.loads((tmp_path/'workspace.json').read_text())==ws.meta


def test_fresh_admission_refuses_growth_and_reports_live_pressure_without_fatal_bind(tmp_path):
    from wsp.memory_guard import GiB
    ws=fake_ws(tmp_path);settings=resolve(env={'HOME':str(tmp_path)})
    value=descriptor(ws,8192)
    sample=dict(total=32*GiB,available=28*GiB,pressure=1,swap_used=0,swap_out=0,compressed=0,tree_footprint=GiB)
    policy,decision=admission(settings,value,sample)
    assert decision['required']==value['forecast']['total']
    for unsafe in [dict(sample,available=1),dict(sample,pressure=None)]:
        with pytest.raises(RuntimeRefusal):admission(settings,value,unsafe)
    growth=dict(value,forecast=dict(total=30*GiB,patches=GiB))
    with pytest.raises(RuntimeRefusal):admission(settings,growth,sample,value)
    issue,warnings=observations(policy,dict(sample,pressure=2),sample)
    assert issue['code']=='memory_pressure' and issue['action']=='abort'
    issue,warnings=observations(policy,dict(sample,swap_used=65*1024**2,compressed=2*GiB),sample)
    assert issue is None and {x['code'] for x in warnings}=={'swap_growth','compression_growth'}


def test_regular_session_keeps_hf_path_independent(tmp_path):
    from test_hf import build
    from wsp.session import Session
    from wsp import delta as D
    import torch
    build('llama',str(tmp_path/'tiny'))
    s=Session(str(tmp_path/'work'));s.runtime=SimpleNamespace(release=lambda:None)
    w=s.open(str(tmp_path/'tiny'),log=lambda _:None)
    assert w.kind=='hf' and not hasattr(w,'_runtime_client')
    ref=w.sem.mlp(0)['W_out'].sub((slice(0,1),slice(0,1)))
    assert w.commit({ref.name:[ref.piece(torch.ones(1,1))]},'ordinary HF edit')


def test_missing_optional_gguf_is_actionable(tmp_path,monkeypatch):
    from wsp.gguf_store import GGUFStore
    monkeypatch.setitem(sys.modules,'gguf',None)
    with pytest.raises(RuntimeError,match='requirements-quantized.txt'):GGUFStore(str(tmp_path/'missing.gguf'))


def test_runner_construction_serialized(tmp_path,monkeypatch):
    import wsp.runner
    from wsp.workspace import Workspace
    ws=Workspace(str(tmp_path),None,SimpleNamespace(),None,{'kind':'gguf'})
    calls=[];instance=object()
    def construct(w):
        calls.append(w);time.sleep(.03);return instance
    monkeypatch.setattr(wsp.runner,'make_runner',construct)
    values=[]
    threads=[threading.Thread(target=lambda:values.append(ws.runner)) for _ in range(2)]
    for t in threads:t.start()
    for t in threads:t.join()
    assert values==[instance,instance] and calls==[ws]


def test_status_uses_cached_gguf_evidence_without_inference(tmp_path):
    from wsp.server import create_app
    from wsp.session import Session
    from fastapi.testclient import TestClient
    sess=Session(str(tmp_path));sess.task['suite']='sample'
    cached={'name':'sample','passed':1,'total':1}
    drift={'n':1,'kl':.01,'top1_agree':1.0}
    class CachedWorkspace: pass
    sess.ws=CachedWorkspace()
    sess.ws.__dict__.update(groups={},kind='gguf',runner_loaded=True,calib={'tests:sample:current':cached,'drift':drift},
        keep={'tests:sample:base':cached},patches=[],version=2,can_run=True,run_error=None,touched=lambda:[],_runner=None,
        lock=threading.RLock())
    sess.tests=lambda *a,**k:pytest.fail('status must not run tests')
    sess.drift=lambda *a,**k:pytest.fail('status must not run drift')
    with TestClient(create_app(sess,'http://testserver')) as c:
        response=c.get('/api/status');assert response.status_code==200
        assert response.json()['tests']['passed']==1 and response.json()['drift']==drift


def test_verified_source_replacement_cannot_reuse_reservation(tmp_path,monkeypatch):
    ws=fake_ws(tmp_path);ws._runtime_stat=descriptor(ws,8192)['stat']
    client=RuntimeClient(tmp_path,resolve(env={'HOME':str(tmp_path)}))
    monkeypatch.setattr(client,'_request',lambda *a,**k:pytest.fail('changed source must not reserve'))
    Path(ws.store.path).write_bytes(b'replacement')
    with pytest.raises(RuntimeRefusal,match='source changed'):client.require(ws,loading=True)


@pytest.mark.parametrize('stage',['before_native','after_native','after_thread','cleanup_failure'])
def test_failed_runner_cleans_partial_resources_before_releasing_reservation(tmp_path,monkeypatch,stage):
    from wsp.gguf_runner import GGUFRunner
    from wsp.workspace import Workspace
    ws=Workspace(str(tmp_path),None,SimpleNamespace(),None,{'kind':'gguf'})
    events=[]
    ws._runtime_client=SimpleNamespace(release=lambda:events.append('released'),operation=lambda *a,**k:nullcontext())
    class Kernel:
        def close(self):
            events.append('closed')
            if stage=='cleanup_failure':raise RuntimeError('native close failed')
    def initialize(r,w):
        events.append('reserved')
        if stage!='before_native':r.kernel=Kernel()
        if stage=='after_thread':
            r._telemetry_thread=threading.Thread(target=lambda:r._telemetry_stop.wait(10))
            r._telemetry_thread.start()
        raise RuntimeError('synthetic construction failure')
    monkeypatch.setattr(GGUFRunner,'_initialize',initialize)
    with pytest.raises(RuntimeError,match='construction fail'):ws.runner
    assert ws._runner is None and 'construction fail' in ws.run_error
    if stage=='cleanup_failure':
        assert events==['reserved','closed']
        assert ws._failed_runner.kernel is not None
        with pytest.raises(RuntimeError,match='cleanup failed'):ws.runner
        assert events==['reserved','closed']  # no retry or false idle
    else:
        assert events==(['reserved'] if stage=='before_native' else ['reserved','closed'])+['released']
        assert not hasattr(ws,'_failed_runner')


def test_idle_missing_gpu_telemetry_blocks_work_without_stopping_server_and_recovers(tmp_path):
    with inert_supervisor(tmp_path) as (proc,folder,request):
        value=descriptor(fake_ws(tmp_path),8192);value['backend']='cpu'
        assert request('metadata',value)['ok']
        loaded=dict(value,runner=True)
        assert request('loaded',loaded)['ok']
        wait(lambda:json.loads((folder/'status.json').read_text()).get('state')=='blocked',8)
        assert proc.poll() is None
        denied=request('stale_begin',dict(loaded,operation_id='blocked-op',working={}),action='begin')
        assert not denied['ok'] and denied['event']['code']=='telemetry_unavailable'
        # Fresh evidence, same resident and baseline: no unload/restart required.
        write(folder/'watchdog.jsonl.gpu.json',dict(time=time.time(),active=True,backend='cpu'))
        begun=dict(loaded,operation_id='retry-op',working={'kv':1024},job='generate')
        assert request('retry',begun,action='begin')['ok']
        assert request('finish',dict(operation_id='retry-op',actual=loaded,success=True),action='finish')['ok']
        rows=[json.loads(x) for x in (folder/'watchdog.jsonl').read_text().splitlines()]
        admitted=[x for x in rows if x['event'] in ('bind','begin')]
        assert len({json.dumps(x['baseline'],sort_keys=True) for x in admitted})==1
        assert request('release',action='release')['ok']
        assert not (folder/'lease.json').exists() and proc.poll() is None


def test_abandon_after_a_begin_timeout_waits_only_its_own_short_bound(tmp_path):
    # Nobody answers the mailbox: the best-effort abandon gives up after
    # ABANDON_TIMEOUT_SECONDS, not the 8 s request timeout, and never raises.
    from wsp import runtime_supervisor as R
    ws=fake_ws(tmp_path);client=RuntimeClient(tmp_path,resolve(env={'HOME':str(tmp_path)}))
    assert R.ABANDON_TIMEOUT_SECONDS==2<R.REQUEST_TIMEOUT_SECONDS
    start=time.monotonic()
    client._abandon(ws,'operation')
    assert 1.9<=time.monotonic()-start<3
    request=json.loads((tmp_path/'request.json').read_text())
    assert request['action']=='finish' and request['descriptor']['operation_id']=='operation'
    assert request['descriptor']['abandoned'] is True and request['descriptor']['success'] is False


def test_status_answers_while_a_mailbox_request_holds_the_client_lock(tmp_path, monkeypatch):
    # A release waiting up to 8 s on the supervisor must not freeze /api/runtime and /api/state.
    ws=fake_ws(tmp_path);client=RuntimeClient(tmp_path,resolve(env={'HOME':str(tmp_path)}))
    client.reserved=True;client.binding='bound';client.last_event=dict(code='old')
    client._forecast_cache=('key',dict(total=123))
    record=dict(code='memory_pressure',action='abort',message='pressure')
    write(tmp_path/'status.json',dict(state='reserved',event=record,warnings=[]))
    entered=threading.Event();unblock=threading.Event()
    def blocked(action,value=None,timeout=None):
        entered.set();unblock.wait(8)
        return dict(ok=True)
    monkeypatch.setattr(client,'_request',blocked)
    holder=threading.Thread(target=client.release,daemon=True);holder.start()
    try:
        assert entered.wait(5)
        def recompute(ws):raise AssertionError('the forecast is not recomputed without the lock')
        monkeypatch.setattr(client,'_forecast',recompute)
        out={};started=time.monotonic()
        reader=threading.Thread(target=lambda:out.setdefault('value',client.status(ws)),daemon=True);reader.start()
        reader.join(.5)
        assert not reader.is_alive() and time.monotonic()-started<.5
        status=out['value']
        assert status['busy'] is True and status['reserved'] is True and status['event']==record
        assert status['reservation']['state']=='reserved' and status['forecast']==dict(total=123)
    finally:
        unblock.set();holder.join(5)
    monkeypatch.setattr(client,'_forecast',lambda ws:dict(total=7))
    status=client.status(ws)
    assert status['busy'] is False and status['reserved'] is False and status['forecast']==dict(total=7)
