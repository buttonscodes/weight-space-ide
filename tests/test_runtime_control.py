"""Real supervisor/client/control-thread wiring with scalar fakes and owned processes.

The listener uses a private OS-assigned port. No checkpoint parser, model
constructor, native library or tensor is used. ``WORKER`` is the fake server
(the supervisor's child); ``MODEL_WORKER`` is the fake model worker the
supervisor spawns, built on the real ``serve_worker`` with a scalar runner.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from types import SimpleNamespace

import psutil
import pytest
from wsp.runtime_supervisor import write
from wsp.memory_guard import GiB

WORKER = r'''
import json, os, threading, time
from pathlib import Path
from types import SimpleNamespace
from wsp import runtime_supervisor, lifecycle
from wsp.runtime_supervisor import RuntimeClient, write
from wsp.runtime_events import RuntimeRefusal
if os.environ.get('FIXTURE_REQUEST_TIMEOUT'):runtime_supervisor.REQUEST_TIMEOUT_SECONDS=float(os.environ['FIXTURE_REQUEST_TIMEOUT'])
root=Path(os.environ['HOME']);directory=Path(os.environ['WSP_RUNTIME_DIR'])
settings=json.loads((directory/'settings.json').read_text())
client=RuntimeClient(directory,settings)
source=root/'dummy.gguf';source.write_bytes(b'scalar fixture only')
write(root/'server-env.json',{k:v for k,v in os.environ.items() if k.startswith('WSP_')})
# Every SIGTERM/SIGINT this server receives is recorded before the normal shutdown.
handle_exit=lifecycle.IDEServer.handle_exit
def recording_exit(self,sig,frame):
    with (root/'server-signals.txt').open('a') as f:f.write(f'{int(sig)}\n')
    handle_exit(self,sig,frame)
lifecycle.IDEServer.handle_exit=recording_exit
released=[];serial=0;worker_gpu=False;kept=[]
class FakeRunner:
    def __init__(self,number):self.number=number;self.lock=threading.RLock()
    def release(self):released.append(self.number)
class RemoteFake:
    """The server's handle on a model worker: its connection after the load job."""
    def __init__(self,number,channel,pid):self.number=number;self.channel=channel;self.pid=pid;self.lock=threading.RLock()
    def release(self):
        released.append(self.number)
        try:self.channel.send('shutdown')
        except Exception:pass
        self.channel.close()
def attach(spawned):
    import socket as so
    from wsp.worker_protocol import Channel
    from wsp.worker_jobs import encode_job
    end=time.monotonic()+30
    while True:
        s=so.socket(so.AF_UNIX,so.SOCK_STREAM)
        try:s.connect(spawned['socket']);break
        except (FileNotFoundError,ConnectionRefusedError):
            s.close()
            if time.monotonic()>end:raise
            time.sleep(.05)
    s.settimeout(30)
    channel=Channel(s);channel.send('hello',{'token':spawned['token']})
    fields,tensors=encode_job('load',{})
    st=source.stat()
    fields.update(version=0,manifest=[],binding=client.binding,operation_id=client.operation_id,atlas=None,
                  descriptor=dict(path=str(source.resolve()),stat=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns],base_hash='fixture'))
    job=os.urandom(8).hex();channel.send('job',fields,tensors,job=job)
    frame=channel.recv(job=job)
    if frame.type!='result':raise RuntimeError(f'worker load failed: {frame.type} {frame.header}')
    return channel
def wait_release(limit):
    end=time.monotonic()+limit
    while not (root/'release-hang').exists() and time.monotonic()<end:time.sleep(.02)
    try:(root/'release-hang').unlink()
    except FileNotFoundError:pass
ws=SimpleNamespace(lock=threading.RLock(),kind='gguf',version=0,_runner=None,run_error=None,
    patches=[],patch_stored=0,calib={},keep={},_cache={},_cache_bytes=0,
    store=SimpleNamespace(path=str(source),nbytes=lambda:19),meta={'base_sha256':'fixture'},n_params=lambda:32,
    struct=SimpleNamespace(n_ctx=128,d_model=8,n_layers=1,n_heads=2,d_vocab=32),
    hf_config={'hidden_size':8,'num_hidden_layers':1,'num_attention_heads':2,'vocab_size':32})
ws._runtime_client=client
stop=threading.Event()
def background():
    while not stop.wait(.025):
        # With a model worker, the worker publishes the GPU telemetry, not the server.
        if not worker_gpu:write(directory/'watchdog.jsonl.gpu.json',dict(time=time.time(),active=False,backend='cpu'))
        write(root/'heartbeat.json',dict(time=time.time(),pid=os.getpid(),resident=ws._runner is not None,
            runner_id=ws._runner.number if ws._runner else None,released=list(released),reserved=client.reserved,
            operation_id=client.operation_id,event=client.last_event))
threading.Thread(target=background,daemon=True).start()
def commands():
    global serial, worker_gpu
    handled=None
    while True:
        try:command=json.loads((root/'command.json').read_text())
        except FileNotFoundError:time.sleep(.02);continue
        if command['id']==handled:time.sleep(.02);continue
        handled=command['id'];out={'id':handled,'ok':True}
        try:
            if command['action']=='forecast':out['forecast']=client._forecast(ws)
            elif command['action']=='outer-load':
                with client.operation(ws,'analysis'):
                    with client.operation(ws,'load'):
                        serial+=1;ws._runner=FakeRunner(serial)
            elif command['action']=='load':
                worker_gpu=False
                with client.operation(ws,'load'):
                    serial+=1;ws._runner=FakeRunner(serial)
            elif command['action']=='worker-load':
                if command.get('replace') and isinstance(ws._runner,RemoteFake):
                    # What B4's RemoteRunner does with a lost worker: drop it, keep no binding.
                    ws._runner.channel.close();ws._runner=None;client.binding=None
                worker_gpu=True;client.worker=True
                try:
                    with client.operation(ws,'load'):
                        if command.get('revoke_first'):
                            sample=json.loads((root/'sample.json').read_text());sample['pressure']=2
                            write(root/'sample.json',sample);end=time.monotonic()+6
                            while json.loads((directory/'lease.json').read_text()).get('ok'):
                                if time.monotonic()>end:raise RuntimeError('lease was not revoked')
                                time.sleep(.02)
                            sample['pressure']=1;write(root/'sample.json',sample)
                        spawned=client.spawn_worker(command.get('operation_id',client.operation_id),str(root))
                        out['spawn']={k:spawned[k] for k in ('socket','pid','create_time')}
                        if command.get('twice'):
                            try:client.spawn_worker(client.operation_id,str(root))
                            except RuntimeRefusal as exc:out['twice']=exc.record['code']
                        channel=attach(spawned)
                        serial+=1;ws._runner=RemoteFake(serial,channel,spawned['pid'])
                        if command.get('hold'):wait_release(40)
                finally:client.worker=False
            elif command['action']=='rollback-hang':
                # A load whose spawned worker is rolled back, then a model that holds the workspace.
                worker_gpu=True;client.worker=True
                try:
                    with client.operation(ws,'load'):
                        try:client.spawn_worker(client.operation_id,str(root))
                        except RuntimeRefusal as exc:out['spawn_refused']=exc.record['code']
                        wait_release(command.get('duration',40))
                finally:client.worker=False
            elif command['action']=='spawn-in':
                with client.operation(ws,command['job'],tokens=4):client.spawn_worker(client.operation_id,str(root))
            elif command['action']=='worker-shutdown':
                ws._runner.channel.send('shutdown')
            elif command['action']=='release-keep':
                # A server that releases while its worker still runs (the connection stays open).
                with ws.lock:kept.append(ws._runner);ws._runner=None;client.release()
            elif command['action']=='raw':
                client._request(command['request'])
            elif command['action']=='hang-op':
                # An admitted operation that never reaches a lease checkpoint.
                with client.operation(ws,'generate',tokens=4):wait_release(command.get('duration',40))
            elif command['action']=='run':
                produced=0
                with client.operation(ws,'generate',tokens=32):
                    end=time.monotonic()+command.get('duration',5)
                    try:
                        while time.monotonic()<end:
                            client.require(ws);produced+=1;time.sleep(.05)
                    except RuntimeRefusal:
                        if command.get('recover'):
                            sample=json.loads((root/'sample.json').read_text());sample['pressure']=1
                            write(root/'sample.json',sample)
                        raise
                out['produced']=produced
            elif command['action']=='probe':
                with client.operation(ws,'generate',tokens=4):client.require(ws)
            elif command['action']=='stale-cycle':
                # The real control thread can observe the old command while blocked
                # on this workspace lock. Release must invalidate it before reload.
                with ws.lock:
                    sample=json.loads((root/'sample.json').read_text());sample['pressure']=2
                    write(root/'sample.json',sample)
                    end=time.monotonic()+6
                    while not (directory/'control.json').exists():
                        if time.monotonic()>end:raise RuntimeError('no unload command')
                        time.sleep(.02)
                    out['old_control']=json.loads((directory/'control.json').read_text())['id']
                    time.sleep(.15)
                    sample['pressure']=1;write(root/'sample.json',sample)
                    ws._runner.release();ws._runner=None;client.release()
                    with client.operation(ws,'load'):
                        serial+=1;ws._runner=FakeRunner(serial)
                    out['new_runner']=serial
            elif command['action']=='hang':
                # An unresponsive model: holds the workspace with no lease checkpoint.
                with ws.lock:wait_release(command.get('duration',40))
            else:raise ValueError('unknown fixture command')
        except RuntimeRefusal as exc:
            out.update(ok=False,event=exc.record,produced=locals().get('produced',0))
        except Exception as exc:out.update(ok=False,error=repr(exc))
        write(root/'reply.json',out)
# The HTTP layer and lifecycle are production implementations. Session.ws stays
# empty; the separate scalar workspace above exercises only runtime ownership.
import socket
from wsp.server import create_app
from wsp.session import Session
from wsp.lifecycle import serve
sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(128)
port=sock.getsockname()[1];url=f'http://127.0.0.1:{port}'
session=Session(str(root/'app-work'));session.settings=settings;session.runtime=client
app=create_app(session,url)
write(root/'listener.json',{'port':port})
threading.Thread(target=commands,daemon=True).start()
try:serve(app,sockets=[sock],host='127.0.0.1',port=port,log_level='error')
finally:stop.set();sock.close()

'''

SUPERVISOR = r'''
import json, os, sys, time, psutil
from pathlib import Path
from wsp import runtime_supervisor as r, runtime_policy as policy, memory_guard as memory
from wsp.runtime_config import resolve
root=Path(os.environ['HOME'])
# Test-only short cooperative bounds. Sampling still uses the production 1 Hz.
policy.ABORT_GRACE_SECONDS=.3;policy.UNLOAD_GRACE_SECONDS=1;policy.GPU_STALE_GRACE_SECONDS=1
# The fake server and model worker never fork. Avoid privileged OS child enumeration in tests.
psutil.Process.children=lambda self,recursive=False:[]
ramp=[0]
def measured(pid):
    control=json.loads((root/'sample.json').read_text())
    if control.get('raise_supervisor'):raise RuntimeError('injected telemetry failure')
    process=psutil.Process(pid)
    # *_rss: a separate RSS figure (pages mapped in both processes); default RSS == footprint.
    if 'model_worker.py' in ' '.join(process.cmdline()):
        value=control.get('worker_measured',1024**2)
        # worker_ramp: every sample of the worker tree grows by this many bytes.
        if control.get('worker_ramp'):ramp[0]+=control['worker_ramp'];value+=ramp[0]
        rss=control.get('worker_measured_rss',value)
    else:value=control.get('measured',1024**2);rss=control.get('measured_rss',value)
    return dict(tree_rss=rss,tree_footprint=value,pids=[pid],identities={pid:process.create_time()})
memory.process_memory=measured
# supervisor_fault: a fault in supervision itself (not in memory sampling) after the samples were taken.
original_observations=policy.observations
def observations(*args,**kwargs):
    if json.loads((root/'sample.json').read_text()).get('supervisor_fault'):raise RuntimeError('injected supervisor fault')
    return original_observations(*args,**kwargs)
policy.observations=observations
memory.snapshot=lambda:json.loads((root/'sample.json').read_text())
# server_send_signal_raises: the signal to the fake server is sent, then send_signal raises.
original_send_signal=r.subprocess.Popen.send_signal
def send_signal(self,sig):
    original_send_signal(self,sig)
    if self.args[:2]==[sys.executable,str(root/'worker.py')] and json.loads((root/'sample.json').read_text()).get('server_send_signal_raises'):
        raise OSError('injected send_signal failure')
r.subprocess.Popen.send_signal=send_signal
original_write=r.write
once=set()
def write_with_fault(path,value):
    control=json.loads((root/'sample.json').read_text())
    if control.get('deny_persistence') and Path(path).name in {'lease.json','status.json','last-failure.json','FAILED.json'}:
        raise OSError('injected persistence failure')
    if Path(path).name in control.get('deny_names',()):raise OSError('injected persistence failure')
    if Path(path).name=='reply.json':
        # Reply faults scoped to the request being answered (still in the mailbox).
        action=json.loads((Path(path).parent/'request.json').read_text())['action']
        if action in control.get('deny_reply_actions',()):raise OSError('injected reply failure')
        if action==control.get('deny_reply_once') and 'deny' not in once:
            once.add('deny');raise OSError('injected reply failure')
        if action==control.get('delay_reply_once',{}).get('action') and 'delay' not in once:
            once.add('delay');time.sleep(control['delay_reply_once']['seconds'])
    return original_write(path,value)
r.write=write_with_fault
original=r.subprocess.Popen
def popen(command,**kw):
    # The model worker command runs the fake model worker; the server command the fake server.
    if command[:4]==[sys.executable,'-m','wsp','worker']:
        if json.loads((root/'sample.json').read_text()).get('worker_popen_fails'):raise OSError('injected spawn failure')
        return original([sys.executable,str(root/'model_worker.py'),*command[4:]],**kw)
    if command[:3]==[sys.executable,'-m','wsp']:return original([sys.executable,str(root/'worker.py')],**kw)
    return original(command,**kw)
r.subprocess.Popen=popen
settings=resolve(env={'HOME':str(root)},overrides=dict(backend='cpu',context=128,floor_gib=20,preferred_gib=25,resident_seconds=30))
os.environ['WSP_MAX_CTX']='128'  # what activate() exports before supervise()
# The production default (require_worker=True) unless a test uses the in-process fake runner.
raise SystemExit(r.supervise(settings,empty=True,**({'require_worker':False} if os.environ.get('FIXTURE_REQUIRE_WORKER')=='0' else {})))
'''

MODEL_WORKER = r'''
import argparse, json, os, signal, socket, threading, time
from pathlib import Path
from types import SimpleNamespace
root=Path(os.environ['HOME'])
parser=argparse.ArgumentParser();parser.add_argument('--runtime-dir');parser.add_argument('--socket');parser.add_argument('--workspace')
args=parser.parse_args()
control=lambda:json.loads((root/'sample.json').read_text())  # noqa: E731
if control().get('worker_ignore_term'):signal.signal(signal.SIGTERM,signal.SIG_IGN)
from wsp.worker import serve_worker, watch_parent
from wsp.memory_guard import require_lease
from wsp.runtime_supervisor import write
from wsp.workspace import Workspace
watch_parent()
token=os.environ.pop('WSP_WORKER_TOKEN',None)
write(root/f'worker-{os.getpid()}.json',dict(env={k:v for k,v in os.environ.items() if k.startswith('WSP_')},token=bool(token),argv=vars(args)))
gpu=Path(args.runtime_dir)/'watchdog.jsonl.gpu.json'
def telemetry():
    while True:
        try:
            if not control().get('worker_gpu_stale'):write(gpu,dict(time=time.time(),active=False,backend='cpu',pid=os.getpid()))
        except Exception:pass
        time.sleep(.025)
threading.Thread(target=telemetry,daemon=True).start()
class FakeRunner:
    """Scalar runner: the real lease check at construction, no weights."""
    def __init__(self,ws):
        self.ws,self.lock,self.max_ctx,self.kernel=ws,threading.RLock(),64,SimpleNamespace(threads=1)
        require_lease(ws,loading=True)
    def available(self,name):return False
    def release(self):pass
store=SimpleNamespace(path=str(root/'dummy.gguf'),names=lambda:[],shape=lambda n:(4,),nbytes=lambda:19)
struct=SimpleNamespace(n_ctx=128,n_layers=1,d_model=8,n_heads=2,d_vocab=32)
def open_workspace(directory):
    (Path(directory)/'patches').mkdir(exist_ok=True)
    ws=Workspace(directory,store,struct,None,{'kind':'gguf','base_sha256':'fixture','name':'scalar'});ws.rebuild()
    return ws
listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
previous=os.umask(0o177)
try:listener.bind(args.socket)
finally:os.umask(previous)
listener.listen(1)
status=serve_worker(listener,args.workspace,token,FakeRunner,runtime_dir=args.runtime_dir,open_workspace=open_workspace)
try:os.unlink(args.socket)
except FileNotFoundError:pass
raise SystemExit(status)
'''


def wait(fn, seconds=6):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        try:
            result=fn()
            if result:return result
        except (OSError,ValueError,KeyError):pass
        time.sleep(.02)
    raise AssertionError('fixture deadline expired')


def read(path):return json.loads(path.read_text())


def spawned_workers(folder):
    if folder is None or not (folder/'watchdog.jsonl').exists():return []
    rows=[]
    for line in (folder/'watchdog.jsonl').read_text().splitlines():
        try:row=json.loads(line)
        except ValueError:continue
        if row.get('event')=='worker_spawned':rows.append(row)
    return rows


def alive(pid, create_time=None):
    try:
        process=psutil.Process(pid)
        return process.status()!=psutil.STATUS_ZOMBIE and (create_time is None or process.create_time()==create_time)
    except psutil.NoSuchProcess:return False


@contextmanager
def control_fixture(tmp_path, request_timeout=None, config=None, require_worker=True, server=WORKER, model_worker=MODEL_WORKER):
    # require_worker=False: the test loads the in-process FakeRunner (load/outer-load/stale-cycle).
    # server/model_worker: other fake server and model worker scripts (B4 runs a real Session and RemoteRunner).
    (tmp_path/'worker.py').write_text(server);(tmp_path/'supervisor.py').write_text(SUPERVISOR)
    (tmp_path/'model_worker.py').write_text(model_worker)
    if config is not None:
        from wsp.runtime_config import user_dir
        saved=user_dir(env={'HOME':str(tmp_path)})/'config.json';saved.parent.mkdir(parents=True)
        saved.write_text(json.dumps(dict(version=1,**config)))
    write(tmp_path/'sample.json',dict(total=128*GiB,available=71*GiB,pressure=1,swap_used=0,swap_out=0,compressed=0))
    env={k:v for k,v in os.environ.items() if not k.startswith('WSP_') and k!='FIXTURE_REQUEST_TIMEOUT'}
    env.update(HOME=str(tmp_path),XDG_CACHE_HOME=str(tmp_path/'cache'),XDG_CONFIG_HOME=str(tmp_path/'config'),
               PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    env.pop('FIXTURE_REQUIRE_WORKER',None)
    if request_timeout:env['FIXTURE_REQUEST_TIMEOUT']=str(request_timeout)
    if not require_worker:env['FIXTURE_REQUIRE_WORKER']='0'
    owned=None;folder=None
    with (tmp_path/'process.log').open('w') as log:
        process=subprocess.Popen([sys.executable,str(tmp_path/'supervisor.py')],env=env,stdout=log,stderr=log,start_new_session=True)
        try:
            beat=wait(lambda:read(tmp_path/'heartbeat.json'))
            child=psutil.Process(beat['pid']);owned=(child.pid,child.create_time())
            folder=wait(lambda:next((tmp_path/'cache').rglob('settings.json'),None)).parent
            port=wait(lambda:read(tmp_path/'listener.json'))['port']
            def api(path):
                with urllib.request.urlopen(f'http://127.0.0.1:{port}'+path,timeout=2) as response:return json.load(response)
            wait(lambda:api('/api/state'))
            counter=0
            def send(action,**kwargs):
                nonlocal counter
                counter+=1;write(tmp_path/'command.json',dict(id=counter,action=action,**kwargs));return counter
            def response(n,seconds=6):return wait(lambda:read(tmp_path/'reply.json') if read(tmp_path/'reply.json')['id']==n else None,seconds)
            def command(action,seconds=6,**kwargs):return response(send(action,**kwargs),seconds)
            def telemetry(**kwargs):
                value=read(tmp_path/'sample.json');value.update(kwargs);write(tmp_path/'sample.json',value)
            yield SimpleNamespace(root=tmp_path,folder=folder,process=process,send=send,response=response,
                command=command,telemetry=telemetry,api=api,heartbeat=lambda:read(tmp_path/'heartbeat.json'))
        finally:
            if process.poll() is None:process.send_signal(signal.SIGTERM)
            try:process.wait(timeout=12)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=2)
            if owned:
                try:
                    child=psutil.Process(owned[0])
                    if child.create_time()==owned[1] and child.status()!=psutil.STATUS_ZOMBIE:
                        child.terminate()
                        try:child.wait(timeout=12)
                        except psutil.TimeoutExpired:child.kill();child.wait(timeout=2)
                except psutil.NoSuchProcess:pass
                assert not psutil.pid_exists(owned[0]) or psutil.Process(owned[0]).create_time()!=owned[1] or psutil.Process(owned[0]).status()==psutil.STATUS_ZOMBIE
            # No fake model worker outlives its supervisor (the supervisor stops it at teardown).
            leaked=[]
            for spawned in spawned_workers(folder):
                if alive(spawned['pid'],spawned['create_time']):
                    leaked.append(spawned['pid'])
                    try:psutil.Process(spawned['pid']).kill()
                    except psutil.NoSuchProcess:pass
            assert leaked==[]


def test_cooperative_abort_healthy_finish_keeps_runner_and_retry(tmp_path):
    with control_fixture(tmp_path,require_worker=False) as f:
        assert f.command('load')['ok']
        wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id']);runner=f.heartbeat()['runner_id']
        command=f.send('run',recover=True)
        wait(lambda:f.heartbeat()['operation_id'])
        assert f.api('/api/state')['open'] is None
        assert f.api('/api/runtime')['operation_id']
        f.telemetry(pressure=2)
        result=f.response(command)
        assert not result['ok'] and result['event']['code']=='memory_pressure' and result['produced']>0
        assert f.process.poll() is None
        wait(lambda:not f.heartbeat()['operation_id'])
        assert f.heartbeat()['resident'] and f.heartbeat()['runner_id']==runner and f.heartbeat()['released']==[]
        assert read(f.folder/'lease.json')['ok']
        assert f.api('/api/state')['open'] is None
        assert f.api('/api/runtime')['event']['code']=='memory_pressure'
        assert f.command('probe')['ok']
        assert f.heartbeat()['runner_id']==runner


def test_persistent_pressure_unloads_through_real_control_thread_but_worker_lives(tmp_path):
    with control_fixture(tmp_path,require_worker=False) as f:
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        runner=f.heartbeat()['runner_id'];command=f.send('run')
        wait(lambda:f.heartbeat()['operation_id'])
        assert f.api('/api/state')['open'] is None
        assert f.api('/api/runtime')['operation_id']
        f.telemetry(pressure=2)
        result=f.response(command)
        assert not result['ok'] and result['event']['code']=='memory_pressure'
        wait(lambda:not f.heartbeat()['resident'] and not f.heartbeat()['reserved'])
        assert f.heartbeat()['released']==[runner] and f.process.poll() is None
        status=read(f.folder/'status.json')
        assert status['forecast'] is None and status['operation_id'] is None
        assert not (f.folder/'lease.json').exists()
        assert f.api('/api/state')['open'] is None
        assert not f.api('/api/runtime')['reserved']
        before=f.heartbeat()['time'];wait(lambda:f.heartbeat()['time']>before)
        f.telemetry(pressure=1)
        assert f.command('load')['ok']
        assert f.command('probe')['ok']


def test_supervisor_exception_revokes_lease_persists_event_and_worker_stays_alive(tmp_path):
    with control_fixture(tmp_path,require_worker=False) as f:
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        f.telemetry(supervisor_fault=True)
        wait(lambda:(read(f.folder/'lease.json').get('event') or {}).get('code')=='watchdog_unavailable',seconds=7)
        assert f.process.poll() is None
        before=f.heartbeat()['time'];wait(lambda:f.heartbeat()['time']>before)
        assert f.heartbeat()['resident']
        lease=read(f.folder/'lease.json')
        assert not lease['ok'] and lease['event']['code']=='watchdog_unavailable'
        assert read(f.folder.parent/'last-failure.json')['code']=='watchdog_unavailable'
        assert f.api('/api/state')['open'] is None
        assert f.api('/api/runtime')['event']['code']=='watchdog_unavailable'
        result=f.command('probe')
        assert not result['ok'] and result['event']['code']=='watchdog_unavailable'
        # A terminally failed watchdog retains signal/child ownership until user shutdown.
        f.process.send_signal(signal.SIGTERM)
        assert f.process.wait(timeout=8)==1
        final=read(f.folder/'FAILED.json')
        assert not final['server_left_running'] and final['remaining']==[] and final['errors']==[]


def test_delayed_old_unload_command_does_not_unload_reloaded_runner(tmp_path):
    with control_fixture(tmp_path,require_worker=False) as f:
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        first=f.heartbeat()['runner_id']
        result=f.command('stale-cycle')
        assert result['ok'],result
        assert result['old_control'] and result['new_runner']!=first
        wait(lambda:f.heartbeat()['runner_id']==result['new_runner'])
        before=f.heartbeat()['time'];wait(lambda:f.heartbeat()['time']>before+.4)
        assert f.heartbeat()['resident'] and f.heartbeat()['released']==[first]
        assert not (f.folder/'control.json').exists()
        assert f.api('/api/state')['open'] is None
        assert f.api('/api/runtime')['reserved']
        assert f.command('probe')['ok'] and f.process.poll() is None



def test_nested_cold_load_preserves_small_already_admitted_slack(tmp_path):
    with control_fixture(tmp_path,require_worker=False) as f:
        required=f.command('forecast')['forecast']['total']
        f.telemetry(available=25*GiB+required+128*1024**2,measured=GiB)
        result=f.command('outer-load')
        assert result['ok'],result
        wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        rows=[json.loads(line) for line in (f.folder/'watchdog.jsonl').read_text().splitlines()]
        begin=next(row for row in rows if row['event']=='begin')
        extension=next(row for row in rows if row['event']=='extend')
        assert begin['policy']==extension['policy']
        assert begin['policy']['budget']==required+128*1024**2
        assert f.api('/api/state')['open'] is None and f.api('/api/runtime')['reserved']


def test_transient_sampling_failure_blocks_then_recovers_same_server(tmp_path):
    with control_fixture(tmp_path,require_worker=False) as f:
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        runner=f.heartbeat()['runner_id']
        f.telemetry(raise_supervisor=True)
        wait(lambda:(read(f.folder/'lease.json').get('event') or {}).get('code')=='telemetry_unavailable')
        assert not read(f.folder/'lease.json')['ok'] and f.process.poll() is None
        assert f.api('/api/state')['open'] is None
        f.telemetry(raise_supervisor=False)
        wait(lambda:read(f.folder/'lease.json')['ok'])
        assert f.command('probe')['ok'] and f.heartbeat()['runner_id']==runner
        assert not (f.folder/'FAILED.json').exists()


def assert_server_kept(f):
    """Step 4 without a live worker: recorded, nothing signalled, the IDE still answers."""
    wait(lambda:'kill_worker_skipped' in [r['event'] for r in log_rows(f)],seconds=15)
    before=f.heartbeat()['time'];wait(lambda:f.heartbeat()['time']>before+1)
    rows=log_rows(f)
    assert [r['event'] for r in rows].count('kill_worker_skipped')==1
    assert 'worker_signal' not in [r['event'] for r in rows]
    assert f.process.poll() is None and server_signals(f)==[]
    assert f.api('/api/state')['open'] is None
    assert not (f.folder/'FAILED.json').exists()
    text=(f.folder/'watchdog.jsonl').read_text()
    assert 'sampling_error' not in text and 'watchdog_unavailable' not in text


def test_last_resort_without_a_worker_survives_receipt_write_failure(tmp_path):
    # B5: an unresponsive in-process model at step 4 no longer stops the server;
    # the skipped step stays a recorded no-op even when its receipt cannot be written.
    with control_fixture(tmp_path,require_worker=False) as f:
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        f.telemetry(deny_names=['last-failure.json'])
        f.send('hang');time.sleep(.3)
        f.telemetry(pressure=2)
        assert_server_kept(f)
        # The model answers again: the real control thread completes the unload, and a fresh load is admitted.
        (f.root/'release-hang').touch();f.telemetry(pressure=1)
        wait(lambda:not f.heartbeat()['resident'],seconds=10)
        assert f.command('load')['ok'] and f.process.poll() is None and server_signals(f)==[]


def test_last_resort_without_a_worker_survives_lease_and_status_write_failure(tmp_path):
    # Evidence storage failing during the ladder is not a sampling fault, and the
    # never-delivered unload does not turn into a server stop.
    with control_fixture(tmp_path,require_worker=False) as f:
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        f.send('hang');time.sleep(.3)
        f.telemetry(pressure=2,deny_names=['lease.json','status.json','control.json'])
        assert_server_kept(f)
        (f.root/'release-hang').touch();f.telemetry(pressure=1,deny_names=[])
        # The unload command was never written, so the server drops the runner and releases on its own.
        assert f.command('release-keep')['ok']
        assert f.command('load')['ok'] and f.process.poll() is None and server_signals(f)==[]


def test_spawn_rollback_then_pressure_never_signals_the_server(tmp_path):
    # A rolled-back spawn leaves the running load with no live worker; the ladder's
    # step 4 must then be the recorded no-op, never a stop of the server.
    with control_fixture(tmp_path) as f:
        f.telemetry(deny_reply_once='spawn_worker')
        n=f.send('rollback-hang')
        wait(lambda:spawned_workers(f.folder),seconds=10)
        spawned=spawned_workers(f.folder)[0]
        wait(lambda:not alive(spawned['pid'],spawned['create_time']))
        # The load is still running (its reservation is the runner's) with no live worker.
        wait(lambda:f.heartbeat()['operation_id'] and read(f.folder/'status.json')['operation_id'])
        f.telemetry(pressure=2)
        wait(lambda:'kill_worker_skipped' in [r['event'] for r in log_rows(f)],seconds=15)
        before=f.heartbeat()['time'];wait(lambda:f.heartbeat()['time']>before+1)
        rows=log_rows(f)
        assert [r['reason'] for r in rows if r['event']=='worker_signal']==['spawn_rollback']
        assert f.process.poll() is None and server_signals(f)==[]
        assert f.api('/api/state')['open'] is None and not (f.folder/'FAILED.json').exists()
        (f.root/'release-hang').touch();f.telemetry(pressure=1)
        result=f.response(n,seconds=10)
        assert result['spawn_refused']=='runtime_request_invalid'
        assert f.command('release-keep')['ok']
        spawn=worker_load(f)
        assert alive(spawn['pid']) and f.process.poll() is None and server_signals(f)==[]


def test_admission_that_cannot_publish_its_lease_is_not_granted(tmp_path):
    # A begin whose lease write fails is refused and must leave no phantom
    # running operation behind: the next begin succeeds once writes recover.
    with control_fixture(tmp_path,require_worker=False) as f:
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        f.telemetry(deny_names=['lease.json'])
        assert not f.command('probe')['ok']
        f.telemetry(deny_names=[])
        assert f.command('probe')['ok']
        assert f.heartbeat()['resident'] and 'watchdog_unavailable' not in (f.folder/'watchdog.jsonl').read_text()


def log_rows(f):return [json.loads(line) for line in (f.folder/'watchdog.jsonl').read_text().splitlines()]


def test_begin_grant_whose_reply_is_not_delivered_is_rolled_back(tmp_path):
    # The lease was published but the grant reply was not: the delivered refusal
    # must not leave a phantom running operation that refuses every later begin.
    with control_fixture(tmp_path,require_worker=False) as f:
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        f.telemetry(deny_reply_once='begin')
        result=f.command('probe')
        assert not result['ok'] and result['event']['code']=='runtime_request_invalid'
        result=f.command('probe')
        assert result['ok'],result
        log=(f.folder/'watchdog.jsonl').read_text()
        assert 'operation_busy' not in log and 'watchdog_unavailable' not in log and f.process.poll() is None


def test_undeliverable_error_reply_does_not_end_supervision(tmp_path):
    with control_fixture(tmp_path,request_timeout=1.5,require_worker=False) as f:
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        f.telemetry(deny_reply_actions=['begin'])
        result=f.command('probe')
        # Neither the grant nor the refusal reached the client; it gave up on the reply.
        assert not result['ok'] and result['event']['code']=='watchdog_unavailable'
        assert f.process.poll() is None
        f.telemetry(deny_reply_actions=[])
        result=f.command('probe')
        assert result['ok'],result
        log=(f.folder/'watchdog.jsonl').read_text()
        assert 'watchdog_unavailable' not in log and 'operation_busy' not in log
        assert not (f.folder/'FAILED.json').exists() and f.heartbeat()['resident']


def test_begin_timeout_abandons_a_late_grant(tmp_path):
    with control_fixture(tmp_path,request_timeout=2,require_worker=False) as f:
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        f.telemetry(delay_reply_once=dict(action='begin',seconds=3))
        result=f.command('probe')
        assert not result['ok'] and result['event']['code']=='watchdog_unavailable'
        abandoned=lambda:[r['descriptor'] for r in log_rows(f) if r['event']=='finish' and r['descriptor'].get('abandoned')]
        abandon=wait(abandoned)
        late=[r for r in log_rows(f) if r['event']=='begin'][-1]['descriptor']['operation_id']
        assert [a['operation_id'] for a in abandon]==[late] and abandon[0]['success'] is False
        assert read(f.folder/'status.json')['operation_id'] is None
        result=f.command('probe')
        assert result['ok'],result
        assert 'operation_busy' not in (f.folder/'watchdog.jsonl').read_text()


def test_clamped_saved_threads_are_a_visible_runtime_warning(tmp_path):
    with control_fixture(tmp_path,config={'defaults':{'threads':999}},require_worker=False) as f:
        settings=read(f.folder/'settings.json');limit=settings['thread_limit']
        assert settings['values']['threads']==limit
        assert settings['sources']['threads']==f'config defaults (clamped to host limit {limit})'
        warnings=f.api('/api/runtime')['warnings']
        assert [w['code'] for w in warnings]==['threads_clamped']
        assert warnings[0]['action']=='warning' and warnings[0]['measured']==999 and warnings[0]['limit']==limit
        # Sampled warnings are re-keyed every second; the start-up notice stays.
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        before=f.heartbeat()['time'];wait(lambda:f.heartbeat()['time']>before+1.2)
        assert 'threads_clamped' in {w['code'] for w in f.api('/api/runtime')['warnings']}


def test_finish_during_sampling_failure_does_not_renew_lease(tmp_path):
    with control_fixture(tmp_path,require_worker=False) as f:
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        n=f.send('run',duration=5);wait(lambda:f.heartbeat()['operation_id'])
        f.telemetry(raise_supervisor=True)
        assert f.response(n)['event']['code']=='telemetry_unavailable'
        seen=[];end=time.monotonic()+.9
        while time.monotonic()<end:
            try:seen.append(read(f.folder/'lease.json')['ok'])
            except (OSError,ValueError):pass
            time.sleep(.01)
        assert seen and not any(seen)
        assert f.api('/api/state')['open'] is None


def test_idle_debounce_and_keyed_warnings_preserve_actionable_event(tmp_path):
    with control_fixture(tmp_path,require_worker=False) as f:
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        runner=f.heartbeat()['runner_id']
        f.telemetry(pressure=2)
        wait(lambda:(read(f.folder/'lease.json').get('event') or {}).get('code')=='memory_pressure')
        assert not read(f.folder/'lease.json')['ok'] and f.heartbeat()['resident']
        f.telemetry(pressure=1,swap_used=65*1024**2,compressed=GiB+1)
        wait(lambda:len(f.api('/api/runtime')['warnings'])==2)
        before=f.heartbeat()['time'];wait(lambda:f.heartbeat()['time']>before+2.1)
        runtime=f.api('/api/runtime')
        assert runtime['event']['code']=='memory_pressure'
        assert {r['code'] for r in runtime['warnings']}=={'swap_growth','compression_growth'}
        assert f.heartbeat()['runner_id']==runner and f.heartbeat()['released']==[]
        rows=[json.loads(line) for line in (f.folder/'watchdog.jsonl').read_text().splitlines()]
        codes=[r['record']['code'] for r in rows if r['event']=='runtime_warning']
        assert codes.count('swap_growth')==codes.count('compression_growth')==1
        f.telemetry(swap_used=0,compressed=0)
        wait(lambda:not f.api('/api/runtime')['warnings'])
        assert f.api('/api/runtime')['event']['code']=='memory_pressure'


def test_failed_watchdog_persistence_still_owns_server_shutdown(tmp_path):
    with control_fixture(tmp_path,require_worker=False) as f:
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        f.telemetry(supervisor_fault=True,deny_persistence=True)
        wait(lambda:'The watchdog failed:' in (tmp_path/'process.log').read_text())
        assert f.process.poll() is None
        assert f.api('/api/state')['open'] is None
        # Publishing revocation failed; the old lease must expire rather than renew.
        time.sleep(4.1)
        result=f.command('probe')
        assert not result['ok'] and result['event']['code']=='watchdog_unavailable'
        f.process.send_signal(signal.SIGTERM)
        assert f.process.wait(timeout=12)==1
        assert (tmp_path/'process.log').read_text().count('The watchdog failed:')==1


# ---------------------------------------------------------------- model worker (B3)
def server_signals(f):
    path=f.root/'server-signals.txt'
    return path.read_text().split() if path.exists() else []


def worker_load(f, **kwargs):
    result=f.command('worker-load',seconds=20,**kwargs)
    assert result['ok'],result
    wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
    return result['spawn']


def assert_worker_lost(f, spawn, code='worker_killed'):
    """The worker is gone, the loss is visible, and the server was never signalled."""
    wait(lambda:not alive(spawn['pid'],spawn['create_time']),seconds=8)
    lease=wait(lambda:(l:=read(f.folder/'lease.json'))['event'] and l['event']['code']==code and l)
    assert lease['ok'] is False and lease['event']['action']=='worker_killed'
    assert lease['worker']['pid']==spawn['pid']
    assert 'The IDE remains available.' in lease['event']['message']
    assert lease['event']['recommendations'][-1]=='Run again to load the model through fresh admission.'
    control=wait(lambda:(c:=read(f.folder/'control.json'))['action']=='worker_lost' and c)
    assert control['worker_pid']==spawn['pid'] and control['event']['code']==code
    status=wait(lambda:(s:=read(f.folder/'status.json'))['event']['code']==code and s)
    assert status['state']=='worker_killed' and status['worker']['pid']==spawn['pid']
    assert f.process.poll() is None and server_signals(f)==[]
    before=f.heartbeat()['time'];wait(lambda:f.heartbeat()['time']>before)
    assert f.api('/api/state')['open'] is None
    assert f.api('/api/runtime')['event']['code']==code
    return lease


def assert_fresh_admission(f, old):
    spawn=worker_load(f,replace=True)
    assert spawn['pid']!=old['pid'] and alive(spawn['pid'])
    lease=wait(lambda:(l:=read(f.folder/'lease.json'))['ok'] and l)
    assert lease['worker']['pid']==spawn['pid']
    assert not (f.folder/'control.json').exists()
    assert 'lost_reservation_released' in [r['event'] for r in log_rows(f)]
    assert server_signals(f)==[]
    return spawn


def test_worker_spawns_with_server_environment_and_sums_both_trees(tmp_path):
    with control_fixture(tmp_path) as f:
        f.telemetry(measured=3*1024**2,worker_measured=5*1024**2)
        spawn=worker_load(f)
        assert alive(spawn['pid'],spawn['create_time'])
        lease=read(f.folder/'lease.json')
        assert lease['ok'] and lease['worker']=={'pid':spawn['pid'],'create_time':spawn['create_time']}
        assert read(f.folder/'status.json')['worker']['pid']==spawn['pid']
        rows=log_rows(f)
        assert [r['pid'] for r in rows if r['event']=='worker_spawned']==[spawn['pid']]
        assert [r['worker']['pid'] for r in rows if r['event']=='spawn_worker']==[spawn['pid']]
        sample=wait(lambda:[r for r in log_rows(f) if r['event']=='sample' and r['worker_tree']['pids']][-1:])[0]
        assert sample['server_tree']['tree_footprint']==3*1024**2 and sample['worker_tree']['tree_footprint']==5*1024**2
        assert sample['tree_footprint']==8*1024**2 and spawn['pid'] in sample['pids']
        # The worker runs with the server's environment plus a token it removed from its own.
        worker=read(f.root/f"worker-{spawn['pid']}.json");server=read(f.root/'server-env.json')
        assert worker['token'] and 'WSP_WORKER_TOKEN' not in worker['env'] and 'WSP_WORKER_TOKEN' not in server
        assert worker['env']['WSP_MAX_CTX']=='128'
        assert worker['env']['WSP_RUNTIME_DIR']==server['WSP_RUNTIME_DIR']==str(f.folder)
        assert worker['argv']['runtime_dir']==str(f.folder) and worker['argv']['workspace']==str(f.root)
        assert Path(spawn['socket']).name=='worker.sock' and len(spawn['socket'].encode())<=100
        socket_dir=Path(spawn['socket']).parent
        if socket_dir!=f.folder:
            assert socket_dir.parent==Path('/tmp') and socket_dir.name.startswith('wsp-')
            assert socket_dir.stat().st_mode&0o777==0o700
        # Admission and budget see the sum: neither tree alone exceeds the idle static cap.
        f.telemetry(measured=60*GiB,worker_measured=50*GiB)
        lease=wait(lambda:(l:=read(f.folder/'lease.json'))['event'] and l['event']['code']=='runtime_budget' and l)
        assert lease['event']['measured']==110*GiB
    assert not socket_dir.exists() or socket_dir==f.folder


def worker_samples(f):
    return [r for r in log_rows(f) if r['event']=='sample' and r['worker_tree']['pids']]


def test_rss_shared_by_server_and_worker_is_counted_once(tmp_path):
    # The worker maps the same GGUF pages the server's page cache holds: both RSS figures
    # contain them. Total = server footprint + max(worker footprint, worker RSS).
    with control_fixture(tmp_path) as f:
        # The fixture's budget is 46 GiB: 31 GiB fits, the 60 GiB RSS sum would not.
        f.telemetry(measured=GiB,measured_rss=30*GiB,worker_measured=5*GiB,worker_measured_rss=30*GiB)
        spawn=worker_load(f)
        sample=wait(lambda:worker_samples(f)[2:3])[0]
        assert sample['tree_footprint']==sample['tree_rss']==31*GiB
        assert sample['server_tree']=={'tree_rss':30*GiB,'tree_footprint':GiB,'pids':sample['server_tree']['pids']}
        assert sample['worker_tree']['tree_rss']==30*GiB and sample['worker_tree']['tree_footprint']==5*GiB
        lease=read(f.folder/'lease.json')
        assert lease['ok'] and lease['event'] is None and lease['worker']['pid']==spawn['pid']
        # The worker's own RSS growth still counts in full (a 30 GiB step is also runaway growth).
        f.telemetry(worker_measured_rss=60*GiB)
        event=wait(lambda:(l:=read(f.folder/'lease.json'))['event'])
        assert (event['code'],event['measured']) in (('runtime_budget',61*GiB),('runaway_growth',30*GiB)),event
        assert server_signals(f)==[]


def test_without_a_worker_the_server_tree_is_measured_as_before(tmp_path):
    with control_fixture(tmp_path,require_worker=False) as f:
        f.telemetry(measured=2*GiB,measured_rss=3*GiB)
        assert f.command('load')['ok']
        sample=wait(lambda:[r for r in log_rows(f) if r['event']=='sample'][2:3])[0]
        assert sample['tree_rss']==3*GiB and sample['tree_footprint']==2*GiB and sample['worker_tree']['pids']==[]
        assert sample['server_tree']['tree_rss']==3*GiB and sample['server_tree']['tree_footprint']==2*GiB
        lease=read(f.folder/'lease.json');assert lease['ok'] and lease['event'] is None


def sampled(f, total, count=2):
    """``count`` further samples whose combined total is ``total`` (each loop iteration samples once)."""
    seen=len(log_rows(f))
    wait(lambda:len([r for r in log_rows(f)[seen:] if r['event']=='sample' and r['tree_footprint']==total])>=count,seconds=8)


def running_admission(f):
    return wait(lambda:(s:=read(f.folder/'status.json'))['operation_id'] and s['admission'])


def test_a_worker_hosted_servers_base_is_outside_the_operation_forecast(tmp_path):
    # Rehearsal: the server's pre-model footprint (Python + torch) was charged against the model's
    # forecast and a loaded chat the forecast admits was aborted with runtime_budget.
    with control_fixture(tmp_path) as f:
        forecast=f.command('forecast')['forecast'];base=3*GiB;MiB=1024**2
        f.telemetry(measured=base,worker_measured=MiB)
        worker_load(f)
        n=f.send('run',duration=15);admitted=running_admission(f)
        assert admitted['outside']==base and admitted['policy']['budget']==base+forecast['total']
        # PR A's cap, max(F, measured + new + weights), is below the total sampled next.
        assert max(forecast['total'],base+MiB+admitted['new_bytes']+forecast['weights'])<base+forecast['total']-64*MiB
        # Above the forecast total, below server base + forecast: the operation continues.
        f.telemetry(worker_measured=forecast['total']-64*MiB);sampled(f,base+forecast['total']-64*MiB)
        assert read(f.folder/'status.json')['event'] is None and f.heartbeat()['operation_id']
        assert 'runtime_budget' not in [r.get('record',{}).get('code') for r in log_rows(f)]
        # Above server base + forecast: aborted with that limit (server growth still counts).
        f.telemetry(worker_measured=forecast['total']+256*MiB)
        result=f.response(n,seconds=10)
        assert not result['ok'] and result['event']['code']=='runtime_budget',result
        assert result['event']['limit']==base+forecast['total'] and result['event']['measured']==base+forecast['total']+256*MiB
        assert server_signals(f)==[]


def test_the_server_base_is_fixed_at_the_reservations_first_admission(tmp_path):
    # Server growth after the first admission is model memory: it reduces a later operation's room.
    with control_fixture(tmp_path) as f:
        forecast=f.command('forecast')['forecast'];base=3*GiB;grown=base+512*1024**2;MiB=1024**2
        f.telemetry(measured=base,worker_measured=MiB)
        worker_load(f)
        f.telemetry(measured=grown);sampled(f,grown+MiB)
        n=f.send('run',duration=15);admitted=running_admission(f)
        assert admitted['outside']==base and admitted['policy']['budget']==base+forecast['total']
        # Within grown + forecast but above first base + forecast: aborted at the first-admission limit.
        f.telemetry(worker_measured=forecast['total']-256*MiB)
        result=f.response(n,seconds=10)
        assert not result['ok'] and result['event']['code']=='runtime_budget' and result['event']['limit']==base+forecast['total'],result


def test_a_worker_hosted_cold_load_may_pass_available_minus_preferred_by_the_server_base(tmp_path):
    with control_fixture(tmp_path) as f:
        forecast=f.command('forecast')['forecast']['total'];base=GiB;MiB=1024**2
        # Auto budget = available - preferred (25 GiB) = forecast + 128 MiB; available excludes the server.
        f.telemetry(measured=base,worker_measured=MiB,available=25*GiB+forecast+128*MiB)
        n=f.send('worker-load',hold=True)
        admitted=running_admission(f)
        assert admitted['outside']==base and admitted['policy']['budget']==base+forecast+128*MiB
        wait(lambda:read(f.folder/'status.json')['worker'])
        # The worker reaches its forecast: whole tree base + forecast > available - preferred, not aborted.
        f.telemetry(worker_measured=forecast);sampled(f,base+forecast)
        assert read(f.folder/'status.json')['event'] is None and read(f.folder/'lease.json')['ok']
        (f.root/'release-hang').touch()
        assert f.response(n,seconds=20)['ok']


def test_an_in_process_runner_keeps_the_whole_tree_against_its_forecast(tmp_path):
    with control_fixture(tmp_path,require_worker=False) as f:
        forecast=f.command('forecast')['forecast'];base=3*GiB
        f.telemetry(measured=base)
        assert f.command('load')['ok'];wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        n=f.send('run',duration=15);admitted=running_admission(f)
        limit=max(forecast['total'],base+admitted['new_bytes']+forecast['weights'])
        assert admitted['outside']==0 and admitted['policy']['budget']==limit
        f.telemetry(measured=limit+64*1024**2)
        result=f.response(n,seconds=10)
        assert not result['ok'] and result['event']['code']=='runtime_budget' and result['event']['limit']==limit,result


def test_idle_memory_over_the_forecast_only_by_the_server_base_is_not_a_warning(tmp_path):
    with control_fixture(tmp_path) as f:
        forecast=f.command('forecast')['forecast']['total'];base=forecast+512*1024**2;MiB=1024**2
        assert forecast+128*MiB<4*GiB  # the worker step below stays under runaway growth
        f.telemetry(measured=base,worker_measured=MiB)
        worker_load(f);sampled(f,base+MiB,count=3)
        assert 'idle_above_forecast' not in {w['code'] for w in f.api('/api/runtime')['warnings']}
        # The model itself above its forecast still warns; the base is reported, the excess is the model's.
        f.telemetry(worker_measured=forecast+128*MiB)
        warning=wait(lambda:next((w for w in f.api('/api/runtime')['warnings'] if w['code']=='idle_above_forecast'),None))
        assert warning['measured']=={'total':base+forecast+128*MiB,'outside':base,'excess':128*MiB}
        assert warning['limit']==forecast and read(f.folder/'lease.json')['ok']


def kill_rows(f):
    return [r for r in log_rows(f) if r['event']=='worker_signal' and r['reason']=='kill_worker']


def server_signal_rows(f):
    # The supervisor's own record of every _signal_server call (reason, signal or stop_group, server pid).
    rows=[r for r in log_rows(f) if r['event']=='server_signal']
    assert all(r['pid']==f.heartbeat()['pid'] and isinstance(r['time'],(int,float)) and r['errors']==[] for r in rows),rows
    return [(r['reason'],r['signal']) for r in rows]


def test_unload_timeout_kills_only_the_worker_and_supervision_continues(tmp_path):
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        f.send('hang');time.sleep(.3)
        f.telemetry(pressure=2)
        lease=assert_worker_lost(f,spawn)
        assert lease['event']['cause']['code']=='memory_pressure'
        assert lease['event']['measured']['trigger']=='memory_pressure'
        assert [r['signal'] for r in kill_rows(f)]==[int(signal.SIGKILL)]
        wait(lambda:'worker_reaped' in [r['event'] for r in log_rows(f)])
        rows=log_rows(f)
        assert server_signal_rows(f)==[]
        assert any(r['event']=='sample' and r.get('action')=='unload' for r in rows)
        assert read(f.folder.parent/'last-failure.json')['code']=='worker_killed'
        assert not (f.folder/'FAILED.json').exists()
        (f.root/'release-hang').touch();wait(lambda:not (f.root/'release-hang').exists())
        # A later invalid request is logged but does not replace the visible loss.
        f.telemetry(pressure=1)
        reply=f.command('raw',request='no_such_action')
        assert not reply['ok'] and reply['event']['code']=='runtime_request_invalid'
        assert read(f.folder/'status.json')['event']['code']=='worker_killed'
        assert f.api('/api/runtime')['event']['code']=='worker_killed'
        assert_fresh_admission(f,spawn)
        assert f.process.poll() is None


def test_abort_timeout_of_an_operation_kills_the_worker(tmp_path):
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        n=f.send('hang-op');wait(lambda:f.heartbeat()['operation_id'])
        f.telemetry(pressure=2)
        lease=assert_worker_lost(f,spawn)
        assert lease['event']['cause']['code']=='memory_pressure' and lease['operation_id']
        # running stays until the operation's own finish; the finish keeps the loss visible.
        assert read(f.folder/'status.json')['operation_id']==lease['operation_id']
        (f.root/'release-hang').touch()
        result=f.response(n)
        assert not result['ok'] and result['event']['code']=='worker_killed'
        status=wait(lambda:(s:=read(f.folder/'status.json'))['operation_id'] is None and s)
        assert status['event']['code']=='worker_killed' and status['state']=='worker_killed'
        assert not read(f.folder/'lease.json')['ok']
        f.telemetry(pressure=1)
        # An extension of the lost reservation is refused with the loss itself.
        assert_fresh_admission(f,spawn)


def test_continued_runaway_growth_kills_the_worker_without_unload(tmp_path):
    with control_fixture(tmp_path) as f:
        n=f.send('worker-load',hold=True)
        wait(lambda:spawned_workers(f.folder) and f.heartbeat()['resident'],seconds=20)
        spawn=spawned_workers(f.folder)[0]
        before=len(log_rows(f));wait(lambda:len([r for r in log_rows(f)[before:] if r['event']=='sample'])>=1)
        f.telemetry(worker_ramp=5*GiB)
        lease=assert_worker_lost(f,spawn)
        assert lease['event']['cause']['code']=='runaway_growth'
        actions=[r.get('action') for r in log_rows(f) if r['event']=='sample' and r.get('operation_id')]
        assert 'abort' in actions and 'unload' not in actions and 'kill_worker' in actions
        f.telemetry(worker_ramp=0)
        (f.root/'release-hang').touch()
        result=f.response(n)
        assert not result['ok'] and result['event']['code']=='worker_killed'
        assert_fresh_admission(f,spawn)


def test_stale_worker_gpu_telemetry_abort_that_does_not_stop_kills_the_worker(tmp_path):
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        f.send('hang-op');wait(lambda:f.heartbeat()['operation_id'])
        f.telemetry(worker_gpu_stale=True)
        lease=assert_worker_lost(f,spawn)
        assert lease['event']['cause']['code']=='gpu_telemetry_stale'
        # The dead worker's telemetry is removed, never kept as evidence.
        assert not (f.folder/'watchdog.jsonl.gpu.json').exists() or read(f.folder/'watchdog.jsonl.gpu.json').get('pid')!=spawn['pid']
        f.telemetry(worker_gpu_stale=False)
        (f.root/'release-hang').touch()
        assert_fresh_admission(f,spawn)


def test_worker_kill_survives_every_evidence_write_failing(tmp_path):
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        f.send('hang');time.sleep(.3)
        log=f.folder/'watchdog.jsonl'
        f.telemetry(pressure=2,deny_names=['lease.json','status.json','control.json','last-failure.json','FAILED.json'])
        log.chmod(0o400)
        try:
            wait(lambda:not alive(spawn['pid'],spawn['create_time']),seconds=15)
            assert f.process.poll() is None and server_signals(f)==[]
            before=f.heartbeat()['time'];wait(lambda:f.heartbeat()['time']>before)
            assert f.api('/api/state')['open'] is None
        finally:log.chmod(0o600)
        f.telemetry(deny_names=[])
        # Once storage recovers the lost reservation is re-published on the next samples.
        assert_worker_lost(f,spawn)
        text=log.read_text()
        assert 'sampling_error' not in text and 'watchdog_unavailable' not in text
        (f.root/'release-hang').touch()
        f.telemetry(pressure=1)
        assert_fresh_admission(f,spawn)


def test_worker_kill_survives_receipt_write_failure(tmp_path):
    # Worker-path counterpart of test_last_resort_without_a_worker_survives_receipt_write_failure.
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        f.telemetry(deny_names=['last-failure.json'])
        f.send('hang');time.sleep(.3)
        f.telemetry(pressure=2)
        assert_worker_lost(f,spawn)
        text=(f.folder/'watchdog.jsonl').read_text()
        assert 'sampling_error' not in text and 'watchdog_unavailable' not in text
        assert not (f.folder/'FAILED.json').exists() and f.process.poll() is None
        (f.root/'release-hang').touch()


def test_worker_kill_survives_lease_and_status_write_failure(tmp_path):
    # Worker-path counterpart of test_last_resort_without_a_worker_survives_lease_and_status_write_failure.
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        f.send('hang');time.sleep(.3)
        f.telemetry(pressure=2,deny_names=['lease.json','status.json','control.json'])
        wait(lambda:kill_rows(f),seconds=15)
        wait(lambda:not alive(spawn['pid'],spawn['create_time']))
        time.sleep(1.2)
        assert f.process.poll() is None and server_signals(f)==[]
        text=(f.folder/'watchdog.jsonl').read_text()
        assert 'sampling_error' not in text and 'watchdog_unavailable' not in text
        assert read(f.folder.parent/'last-failure.json')['code']=='worker_killed'
        f.telemetry(deny_names=[])
        assert_worker_lost(f,spawn)
        (f.root/'release-hang').touch()


def test_worker_killed_from_outside_is_a_visible_worker_exit(tmp_path):
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        os.kill(spawn['pid'],signal.SIGKILL)
        lease=assert_worker_lost(f,spawn,code='worker_exited')
        assert lease['event']['measured']['returncode']==-signal.SIGKILL
        assert 'return code -9' in lease['event']['message']
        rows=log_rows(f)
        assert [r['reason'] for r in rows if r['event']=='worker_signal']==['worker_exited']
        assert [r['returncode'] for r in rows if r['event']=='worker_reaped']==[-signal.SIGKILL]
        assert server_signal_rows(f)==[]
        # Extending the lost reservation is refused with the loss record.
        assert not f.command('probe')['ok']
        assert_fresh_admission(f,spawn)


def test_worker_exit_zero_is_an_expected_stop(tmp_path):
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        assert f.command('worker-shutdown')['ok']
        wait(lambda:'worker_stopped' in [r['event'] for r in log_rows(f)])
        assert not alive(spawn['pid'],spawn['create_time'])
        before=len(log_rows(f));wait(lambda:len([r for r in log_rows(f)[before:] if r['event']=='sample'])>=2)
        text=(f.folder/'watchdog.jsonl').read_text()
        assert 'worker_exited' not in text and 'worker_killed' not in text and 'worker_signal' not in text
        assert read(f.folder/'lease.json')['ok'] and not (f.folder/'control.json').exists()
        assert read(f.folder/'status.json')['event'] is None
        # The reservation without a running worker expects no runner telemetry.
        assert 'telemetry' not in json.dumps(read(f.folder/'lease.json'))


def test_spawn_worker_refusals(tmp_path):
    with control_fixture(tmp_path) as f:
        result=f.command('worker-load',seconds=20,twice=True)
        assert result['ok'] and result['twice']=='worker_busy'
        wait(lambda:f.heartbeat()['resident'] and not f.heartbeat()['operation_id'])
        assert len(spawned_workers(f.folder))==1
        result=f.command('spawn-in',job='generate')
        assert not result['ok'] and result['event']['code']=='runtime_request_invalid'
        assert 'inside the running load' in result['event']['message']
        assert len(spawned_workers(f.folder))==1
        assert f.command('worker-shutdown')['ok']
        wait(lambda:'worker_stopped' in [r['event'] for r in log_rows(f)])
        result=f.command('worker-load',seconds=20,operation_id='not-this-operation')
        assert not result['ok'] and result['event']['code']=='runtime_request_invalid'
        f.telemetry(worker_popen_fails=True)
        result=f.command('worker-load',seconds=20)
        assert not result['ok'] and result['event']['code']=='worker_start_failed'
        f.telemetry(worker_popen_fails=False)
        assert len(spawned_workers(f.folder))==1 and f.process.poll() is None


def test_spawn_worker_is_refused_on_a_revoked_lease(tmp_path):
    with control_fixture(tmp_path) as f:
        result=f.command('worker-load',seconds=20,revoke_first=True)
        assert not result['ok'] and result['event']['code']=='memory_pressure'
        assert spawned_workers(f.folder)==[]
        assert f.process.poll() is None and server_signals(f)==[]


def test_undelivered_spawn_reply_rolls_back_and_kills_the_worker(tmp_path):
    with control_fixture(tmp_path) as f:
        f.telemetry(deny_reply_once='spawn_worker')
        result=f.command('worker-load',seconds=20)
        assert not result['ok'] and result['event']['code']=='runtime_request_invalid'
        spawned=spawned_workers(f.folder)
        assert len(spawned)==1
        wait(lambda:not alive(spawned[0]['pid'],spawned[0]['create_time']))
        rows=log_rows(f)
        assert [r['reason'] for r in rows if r['event']=='worker_signal']==['spawn_rollback']
        assert 'worker_exited' not in [r['event'] for r in rows]
        # The rolled-back worker is not the reservation's; the next load admits a new one.
        spawn=worker_load(f)
        assert spawn['pid']!=spawned[0]['pid'] and read(f.folder/'lease.json')['worker']['pid']==spawn['pid']
        assert server_signals(f)==[]


def test_release_reaps_a_worker_that_does_not_stop(tmp_path):
    with control_fixture(tmp_path) as f:
        f.telemetry(worker_ignore_term=True)
        spawn=worker_load(f)
        assert f.command('release-keep')['ok']
        released=time.monotonic()
        assert not (f.folder/'lease.json').exists()
        # The mailbox stays responsive while the reap deadlines run in the loop.
        assert f.command('forecast')['ok'] and f.command('raw',request='dismiss')['ok']
        assert time.monotonic()-released<4 and alive(spawn['pid'])
        wait(lambda:not alive(spawn['pid'],spawn['create_time']),seconds=16)
        wait(lambda:'worker_reaped' in [r['event'] for r in log_rows(f)])
        rows=log_rows(f)
        assert [(r['reason'],r['signal']) for r in rows if r['event']=='worker_signal']==[
            ('release_reap',int(signal.SIGTERM)),('release_reap',int(signal.SIGKILL))]
        assert [r['returncode'] for r in rows if r['event']=='worker_reaped']==[-signal.SIGKILL]
        assert 'worker_exited' not in [r['event'] for r in rows] and server_signals(f)==[]
        assert read(f.folder/'status.json')['event'] is None


def test_supervisor_fault_stops_the_worker_and_keeps_the_server(tmp_path):
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        f.telemetry(supervisor_fault=True)
        wait(lambda:(read(f.folder/'lease.json').get('event') or {}).get('code')=='watchdog_unavailable',seconds=8)
        wait(lambda:not alive(spawn['pid'],spawn['create_time']),seconds=6)
        rows=log_rows(f)
        assert ('watchdog_failed',int(signal.SIGTERM)) in [(r['reason'],r['signal']) for r in rows if r['event']=='worker_signal']
        assert f.process.poll() is None and server_signals(f)==[]
        before=f.heartbeat()['time'];wait(lambda:f.heartbeat()['time']>before)
        assert f.api('/api/state')['open'] is None
        f.process.send_signal(signal.SIGTERM)
        assert f.process.wait(timeout=12)==1
        assert server_signals(f)==[str(int(signal.SIGTERM))]
        assert server_signal_rows(f)==[('user_shutdown',int(signal.SIGTERM)),('user_shutdown','stop_group')]


def test_teardown_stops_the_worker_before_the_server(tmp_path):
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        f.process.send_signal(signal.SIGTERM)
        assert f.process.wait(timeout=12)==0
        assert not alive(spawn['pid'],spawn['create_time'])
        # Forwarded SIGTERM, then the teardown's verified group stop; the worker group went first.
        assert server_signal_rows(f)==[('user_shutdown',int(signal.SIGTERM)),('user_shutdown','stop_group')]
        rows=log_rows(f);order=[(r['event'],r.get('reason')) for r in rows if r['event'] in ('worker_signal','server_signal')]
        assert order.index(('worker_signal','teardown'))<order.index(('server_signal','user_shutdown'),1)
        socket_dir=Path(spawn['socket']).parent
        assert socket_dir==f.folder or not socket_dir.exists()


# ---------------------------------------------------------------- recoverable sampling failure (F5)
def sampling_errors(f):
    return [r for r in log_rows(f) if r['event']=='sampling_error']


def assert_supervision_kept(f):
    text=(f.folder/'watchdog.jsonl').read_text()
    assert 'watchdog_unavailable' not in text and not (f.folder/'FAILED.json').exists()
    assert f.process.poll() is None and server_signals(f)==[] and server_signal_rows(f)==[]
    assert f.api('/api/state')['open'] is None


def test_a_short_sampling_failure_blocks_then_admits_without_a_kill(tmp_path):
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        f.telemetry(raise_supervisor=True)
        wait(lambda:(read(f.folder/'lease.json').get('event') or {}).get('code')=='telemetry_unavailable')
        result=f.command('probe')
        assert not result['ok'] and result['event']['code']=='telemetry_unavailable'
        f.telemetry(raise_supervisor=False)
        wait(lambda:read(f.folder/'lease.json')['ok'])
        assert max(r['duration'] for r in sampling_errors(f))<3 and all(r['measuring'] for r in sampling_errors(f))
        assert f.command('probe')['ok']
        assert alive(spawn['pid'],spawn['create_time']) and len(spawned_workers(f.folder))==1
        assert 'worker_signal' not in [r['event'] for r in log_rows(f)]
        assert_supervision_kept(f)


def test_sampling_failure_past_grace_stops_only_the_worker_then_recovers(tmp_path):
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        f.telemetry(raise_supervisor=True)
        # Past the 3 s grace the unmeasured worker gets TERM (KILL 10 s later); supervision goes on.
        wait(lambda:not alive(spawn['pid'],spawn['create_time']),seconds=10)
        lease=wait(lambda:(l:=read(f.folder/'lease.json'))['event'] and l['event']['code']=='worker_killed' and l)
        assert lease['ok'] is False and lease['event']['cause']['code']=='telemetry_unavailable'
        assert lease['event']['measured']['trigger']=='telemetry_unavailable'
        assert 'Memory sampling failed for 3 seconds' in lease['event']['message']
        control=wait(lambda:(c:=read(f.folder/'control.json'))['action']=='worker_lost' and c)
        assert control['worker_pid']==spawn['pid'] and control['event']['code']=='worker_killed'
        wait(lambda:'worker_reaped' in [r['event'] for r in log_rows(f)])
        assert [(r['reason'],r['signal']) for r in log_rows(f) if r['event']=='worker_signal'][0]==('sampling_failed',int(signal.SIGTERM))
        assert {r['reason'] for r in log_rows(f) if r['event']=='worker_signal'}=={'sampling_failed'}
        # Still failing: sampling is retried every second and every admission is refused.
        before=len(sampling_errors(f));wait(lambda:len(sampling_errors(f))>=before+2,seconds=4)
        result=f.command('worker-load',seconds=20,replace=True)
        assert not result['ok'] and (result['event'].get('cause') or {}).get('code')=='telemetry_unavailable'
        assert len(spawned_workers(f.folder))==1 and not read(f.folder/'lease.json')['ok']
        assert_supervision_kept(f)
        before=len(log_rows(f));f.telemetry(raise_supervisor=False)
        # Recovered: the next sample re-publishes the loss (A2), then a cold load is admitted afresh.
        wait(lambda:any(r['event']=='sample' and r.get('action')=='lost' for r in log_rows(f)[before:]))
        assert_worker_lost(f,spawn)
        assert_fresh_admission(f,spawn)
        assert len(spawned_workers(f.folder))==2
        assert_supervision_kept(f)


def test_sampling_failure_past_grace_without_a_worker_blocks_then_recovers(tmp_path):
    with control_fixture(tmp_path) as f:
        f.telemetry(raise_supervisor=True)
        wait(lambda:any(r['duration']>=3.5 for r in sampling_errors(f)),seconds=8)
        result=f.command('worker-load',seconds=20)
        assert not result['ok'] and result['event']['code']=='telemetry_unavailable'
        assert spawned_workers(f.folder)==[] and 'worker_signal' not in [r['event'] for r in log_rows(f)]
        assert_supervision_kept(f)
        f.telemetry(raise_supervisor=False)
        time.sleep(1.2)  # the next 1 Hz sample succeeds and ends the failure
        before=len(sampling_errors(f));time.sleep(1.1);assert len(sampling_errors(f))==before
        spawn=worker_load(f)
        assert alive(spawn['pid']) and read(f.folder/'lease.json')['ok']
        assert_supervision_kept(f)


def runtime_event_code(f):
    return ((f.api('/api/runtime').get('event') or {}).get('code'),
            (read(f.folder/'status.json').get('event') or {}).get('code'), read(f.folder/'status.json')['state'])


def test_a_failed_server_signal_row_records_the_error(tmp_path):
    with control_fixture(tmp_path) as f:
        f.telemetry(server_send_signal_raises=True)
        f.process.send_signal(signal.SIGTERM)
        f.process.wait(timeout=20)
        rows=[r for r in log_rows(f) if r['event']=='server_signal']
        assert (rows[0]['reason'],rows[0]['signal'])==('user_shutdown',int(signal.SIGTERM))
        assert any('injected send_signal failure' in e for e in rows[0]['errors']),rows


def test_recovered_sampling_without_a_reservation_clears_its_message(tmp_path):
    with control_fixture(tmp_path) as f:
        f.telemetry(raise_supervisor=True)
        wait(lambda:read(f.folder/'status.json')['state']=='blocked')
        assert runtime_event_code(f)==('telemetry_unavailable','telemetry_unavailable','blocked')
        assert read(f.folder/'lease.json')['ok'] is False
        f.telemetry(raise_supervisor=False)
        # The first good sample clears the failure's message and publishes the idle state; no request needed.
        wait(lambda:read(f.folder/'status.json')['state']=='idle',seconds=4)
        assert runtime_event_code(f)==(None,None,'idle') and not (f.folder/'lease.json').exists()
        assert_supervision_kept(f)


def test_recovered_short_sampling_failure_with_a_resident_worker_clears_its_message(tmp_path):
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        f.telemetry(raise_supervisor=True)
        wait(lambda:(read(f.folder/'status.json').get('event') or {}).get('code')=='telemetry_unavailable')
        f.telemetry(raise_supervisor=False)
        wait(lambda:read(f.folder/'lease.json')['ok'])
        assert max(r['duration'] for r in sampling_errors(f))<3
        # Idle, with no later operation: the recovered failure is no longer shown.
        wait(lambda:runtime_event_code(f)==(None,None,'idle'),seconds=4)
        assert alive(spawn['pid'],spawn['create_time']) and 'worker_signal' not in [r['event'] for r in log_rows(f)]
        assert_supervision_kept(f)


def test_a_worker_loss_caused_by_a_sampling_failure_survives_recovery(tmp_path):
    with control_fixture(tmp_path) as f:
        spawn=worker_load(f)
        f.telemetry(raise_supervisor=True)
        wait(lambda:not alive(spawn['pid'],spawn['create_time']),seconds=10)
        before=len(log_rows(f));f.telemetry(raise_supervisor=False)
        wait(lambda:len([r for r in log_rows(f)[before:] if r['event']=='sample' and r.get('action')=='lost'])>=2,seconds=5)
        # A2: the loss stays the record (until dismissed or the next successful operation), with its cause.
        status=read(f.folder/'status.json')
        assert status['state']=='worker_killed' and status['event']['code']=='worker_killed'
        assert status['event']['cause']['code']=='telemetry_unavailable'
        assert f.api('/api/runtime')['event']['code']=='worker_killed'
        assert not read(f.folder/'lease.json')['ok'] and read(f.folder/'control.json')['action']=='worker_lost'
        assert_supervision_kept(f)


def test_require_worker_refuses_in_process_runner_and_step_four_without_worker_is_a_noop(tmp_path):
    with control_fixture(tmp_path) as f:
        result=f.command('load')
        assert not result['ok'] and result['event']['code']=='in_process_runner_unsupported'
        spawn=worker_load(f)
        assert f.command('worker-shutdown')['ok']
        wait(lambda:'worker_stopped' in [r['event'] for r in log_rows(f)])
        f.send('hang');time.sleep(.3)
        f.telemetry(pressure=2)
        wait(lambda:'kill_worker_skipped' in [r['event'] for r in log_rows(f)],seconds=12)
        before=f.heartbeat()['time'];wait(lambda:f.heartbeat()['time']>before+1)
        rows=log_rows(f)
        assert [r['event'] for r in rows].count('kill_worker_skipped')==1
        assert 'worker_signal' not in [r['event'] for r in rows] and server_signals(f)==[]
        assert f.process.poll() is None and not read(f.folder/'lease.json')['ok']
        assert not alive(spawn['pid'])
        (f.root/'release-hang').touch()


def test_runtime_sample_counts_only_the_live_worker_telemetry(tmp_path, monkeypatch):
    from wsp import memory_guard
    from wsp.memory_guard import GuardError
    from wsp.runtime_supervisor import runtime_sample
    monkeypatch.setattr(memory_guard,'snapshot',lambda:{'pressure':1})
    gpu=tmp_path/'watchdog.jsonl.gpu.json'
    write(gpu,dict(time=time.time(),active=False,backend='cpu',pid=111))
    assert runtime_sample(gpu,True,0,10,pid=111)['gpu']['pid']==111
    # Another process's (for example a dead worker's) fresh file is not this worker's telemetry.
    with pytest.raises(GuardError,match='missing'):runtime_sample(gpu,True,0,10,pid=222)
    assert 'gpu' not in runtime_sample(gpu,True,0,3,pid=222)
