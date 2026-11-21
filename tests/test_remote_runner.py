"""B4: the server's RemoteRunner over a real supervisor, server and model worker; no model.

The full-stack tests reuse ``test_runtime_control.control_fixture`` with two other
scripts: ``SERVER`` is the supervisor's child and runs a real gguf-kind
``Workspace``, ``Session``, ``RuntimeClient(worker=True)`` and HTTP app over a
scalar store; ``MODEL_WORKER`` is the real ``serve_worker`` with a scalar
``FakeRunner``. Both import ``fixture_common`` (written next to them). Nothing
opens a checkpoint, native library or model; every listener and file is private
to the pytest temporary directory.
"""
import hashlib
import json
import os
import signal
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from wsp.runtime_supervisor import write
from test_runtime_control import alive, control_fixture, log_rows, read, server_signals, spawned_workers, wait

FIXTURE_COMMON = r'''
"""Shared by the fake server and the fake model worker: a scalar gguf workspace and runner."""
import json, os, threading, time
from pathlib import Path
from types import SimpleNamespace
import torch
from wsp import arch
from wsp.tokenizer import VocabTokenizer
from wsp.workspace import Workspace
from wsp.worker_jobs import JOBS, Job, uses_worker, worker_job
root=Path(os.environ['HOME'])
VOCAB=list(' abcdefghijklmnz');V=len(VOCAB)
HOOK='blocks.0.hook_resid_post'
class FakeStore:
    remote=False
    def __init__(self,path):self.path=str(path)
    def names(self):return ['w']
    def shape(self,name):return (4,)
    def __contains__(self,name):return name=='w'
    def nbytes(self):return 19
    def total(self):return 4
    def numel(self,name):return 4
    def is_float(self,name):return True
    def dtype(self,name):return 'F32'
    def capabilities(self):
        yes=dict(supported=True,reason='scalar fixture')
        return dict(format='gguf',architecture='fixture',tensor_types=['F32'],representation='scalar fixture',
                    import_read=yes,logical_edit=yes,generation=yes,trace_intervention=yes,adapter_export=yes,merged_export=yes)
def workspace(directory,patches=True):
    directory=Path(directory);(directory/'patches').mkdir(parents=True,exist_ok=True);(directory/'tests').mkdir(exist_ok=True)
    struct=arch.Structure(family='generic',runner='none',n_layers=1,n_heads=2,n_kv_heads=2,d_model=8,d_head=4,d_mlp=8,d_vocab=V,n_ctx=128)
    ws=Workspace(str(directory),FakeStore(root/'dummy.gguf'),struct,VocabTokenizer(VOCAB),{'kind':'gguf','base_sha256':'fixture','name':'scalar'})
    if patches:
        for fn in sorted(os.listdir(directory/'patches')):
            if fn.endswith('.json'):ws.patches.append(Workspace._read_patch(str(directory/'patches'),fn,{'w':(4,)}))
    ws.rebuild()
    return ws
def behavior():
    try:return json.loads((root/'behavior.json').read_text())
    except (OSError,ValueError):return {}
class FakeRunner:
    """Scalar: logits favour the next symbol; generate counts up; no weights."""
    def __init__(self,ws):
        b=behavior()
        if b.get('load_exit'):os._exit(3)
        if b.get('load_error'):raise ValueError('the fixture model is unreadable')
        from wsp.memory_guard import require_lease
        self.ws,self.lock,self.max_ctx,self.kernel=ws,threading.RLock(),64,SimpleNamespace(threads=2)
        require_lease(ws,loading=True)
    def available(self,name):return name==HOOK
    def release(self):pass
    def _hang(self,method,emitted=()):
        from wsp.runtime_supervisor import write
        write(root/f'hanging-{os.getpid()}.json',dict(method=method,emitted=list(emitted)))
        while True:time.sleep(.05)  # no lease checkpoint: only a kill ends this
    def logits(self,ids,variant='current',extra=None):
        b=behavior()
        if b.get('hang_token') is not None and b['hang_token'] in ids:self._hang('logits')
        out=torch.zeros(len(ids),V)
        for t,i in enumerate(ids):out[t,(int(i)+1)%V]=4.
        return out
    def next_logits(self,ids,variant='current',extra=None):return self.logits(ids,variant,extra)[-1].clone()
    def generate(self,ids,variant='current',extra=None,max_new_tokens=40,temperature=0.,top_k=0,seed=None,stop_ids=(),on_token=None):
        b=behavior();out=[];last=int(ids[-1])
        if b.get('prefill_seconds') and variant==b.get('prefill_variant','current'):
            # A long prompt read: lease checks between chunks and layers, as in GGUFRunner._prefill.
            from wsp.runtime_supervisor import write
            write(root/f'prefill-{os.getpid()}.json',dict(variant=variant))
            ends=time.monotonic()+b['prefill_seconds']
            while time.monotonic()<ends:
                self.ws._runtime_client.require(self.ws)
                for _ in range(4):self.ws._runtime_client.layer_checkpoint(self.ws);time.sleep(.02)
        for n in range(max_new_tokens):
            if b.get('hang_after') is not None and n>=b['hang_after'] and variant==b.get('hang_variant','base'):self._hang('generate',out)
            if b.get('slow_after') is not None and n>=b['slow_after'] and variant==b.get('slow_variant','current'):
                # Keeps generating, slowly: lease checks and tokens continue, so a stop can end it.
                if n==b['slow_after']:
                    from wsp.runtime_supervisor import write
                    write(root/f'slow-{os.getpid()}.json',dict(variant=variant,emitted=list(out)))
                time.sleep(b.get('slow_seconds',.2))
            self.ws._runtime_client.require(self.ws)
            last=(last+1)%V;out.append(last)
            if on_token is not None:on_token(last)
            time.sleep(.01)
        return out
    def run(self,ids,variant='current',extra=None,names=(),hooks=None,grad=False):
        cache={n:torch.full((1,len(ids),8),float(len(ids))) for n in names if self.available(n)}
        return self.logits(ids,variant,extra)[None],cache
def _probe(ctx,tokens=8,stored=0):
    # Worker side of a named job: admission requests cross back to the server as callbacks.
    lease=ctx.ws._runtime_client
    with lease.operation(ctx.ws,'forward',tokens=int(tokens),fresh=True):pass
    if stored:lease.preflight_edit(ctx.ws,int(stored))
    return dict(pid=os.getpid(),version=ctx.ws.version,patches=[p.id for p in ctx.ws.patches],uses_worker=uses_worker(ctx.ws))
JOBS['fake_probe']=Job(_probe,('tokens','stored'),'json')
@worker_job('fake_probe')
def probe(ws,tokens=8,stored=0):
    raise AssertionError('the fake probe runs only in the model worker')
def big_value(n):
    # Trace-shaped: 32 layers x 32 heads x 27 x 27 attention patterns (about 5 MB of JSON).
    pats=[[[round(((h*31+i*27+j)%97)/97,3) for j in range(27)] for i in range(27)] for h in range(32)]
    return dict(patterns=[pats for _ in range(32)],tokens=['x']*27,received=n)
def _big(ctx,ids=()):return big_value(len(ids))
JOBS['fake_big']=Job(_big,('ids',),'json')
@worker_job('fake_big')
def big(ws,ids=()):
    raise AssertionError('the fake big job runs only in the model worker')
'''

SERVER = r'''
import json, os, socket, threading, time
os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
import torch
from wsp import lifecycle, delta as D
from wsp.runtime_supervisor import RuntimeClient, write
from wsp.runtime_events import RuntimeRefusal
root=Path(os.environ['HOME']);directory=Path(os.environ['WSP_RUNTIME_DIR'])
settings=json.loads((directory/'settings.json').read_text())
client=RuntimeClient(directory,settings,worker=True)
(root/'dummy.gguf').write_bytes(b'scalar fixture only')
handle_exit=lifecycle.IDEServer.handle_exit
def recording_exit(self,sig,frame):
    with (root/'server-signals.txt').open('a') as f:f.write(f'{int(sig)}\n')
    handle_exit(self,sig,frame)
lifecycle.IDEServer.handle_exit=recording_exit
# self-loss: the server's reader fails on its own (no supervisor record) while the worker lives.
from wsp import remote_runner
from wsp.worker_protocol import ProtocolError
quiet_reader=remote_runner.RemoteRunner._idle
def idle(self):
    try:(root/'self-loss').unlink()
    except FileNotFoundError:return quiet_reader(self)
    raise ProtocolError('injected reader failure')
remote_runner.RemoteRunner._idle=idle
import fixture_common as fx
from wsp.workspace import Patch
from wsp.session import DEFAULT_TASK, Session
wdir=root/'ws'
if not (wdir/'patches').exists():
    seed=fx.workspace(wdir)
    for n,value in enumerate((.25,-.5)):
        seed._save_patch(Patch(f'000{n+1}-fixture','fixture patch','test',{'kind':'manual'},
                               {'w':[D.block((range(0,4),),torch.full((4,),value))]},created=1000.+n))
    (wdir/'corpus.txt').write_text('abcdefghijklmn '*12)
    cases=[dict(name='first',type='next_token',prompt='ab',expect='c'),dict(name='second',type='next_token',prompt='zb',expect='c'),
           dict(name='third',type='preserve',prompt='cd',max_kl=.05)]
    (wdir/'tests'/'probe.json').write_text(json.dumps(dict(name='probe',description='scalar',cases=cases)))
ws=fx.workspace(wdir)
session=Session(str(root/'app-work'));session.settings=settings;session.runtime=client
ws._runtime_client=client;client.attach(ws);ws.listeners.append(session.publish)
session.ws=ws;session.task={**DEFAULT_TASK,'suite':None}
stop=threading.Event()
def background():
    while not stop.wait(.025):
        r=ws._runner
        write(root/'heartbeat.json',dict(time=time.time(),pid=os.getpid(),runner=type(r).__name__ if r is not None else None,
            runner_pid=getattr(r,'pid',None),lost=getattr(r,'lost',None),runner_loaded=ws.runner_loaded,reserved=client.reserved,
            operation_id=client.operation_id,event=client.last_event,run_error=ws.run_error,patches=[p.id for p in ws.patches]))
threading.Thread(target=background,daemon=True).start()
def commands():
    handled=None
    while True:
        try:command=json.loads((root/'command.json').read_text())
        except (FileNotFoundError,ValueError):time.sleep(.02);continue
        if command['id']==handled:time.sleep(.02);continue
        handled=command['id'];out={'id':handled,'ok':True};action=command['action']
        try:
            if action=='calls':
                ids=[1,2,3];r=ws.runner
                out['available']=[r.available(fx.HOOK),r.available('blocks.0.hook_mlp_out')]
                out['full']=list(r.logits(ids,'base').shape);out['tail']=list(r.logits(ids,'base',tail=1).shape)
                logits,captures=r.run(ids,'current',names=[fx.HOOK,'blocks.0.hook_mlp_out'])
                out['run_names']=[logits is None,{k:list(v.shape) for k,v in captures.items()}]
                logits,captures=r.run(ids,'current');out['run_plain']=[list(logits.shape),captures]
                out['next']=int(r.next_logits(ids,'current').argmax())
                out.update(max_ctx=r.max_ctx,threads=r.kernel.threads,pid=r.pid)
            elif action=='probe':
                if command.get('fail_preflight'):
                    def failing(*args,**kwargs):raise OSError('injected callback handler failure')
                    client.preflight_edit=failing
                try:out['probe']=fx.probe(ws,tokens=command.get('tokens',8),stored=command.get('stored',0))
                finally:client.__dict__.pop('preflight_edit',None)
            elif action=='big':
                # About 2 MB of arguments out and 5 MB of result back, both above the 1 MiB header cap.
                n=command.get('ids',300000)
                out['equal']=fx.big(ws,ids=list(range(n)))==fx.big_value(n);out['pid']=ws.runner.pid
            elif action=='unsendable':
                r=ws.runner
                try:fx.big(ws,ids=[object()])
                except RuntimeRefusal as exc:out['event']=exc.record
                out.update(lost=r.lost,same=ws.runner is r,pid=r.pid,probe=fx.probe(ws))
            elif action=='drift':out['drift']=session.drift()
            elif action=='edit':
                session.task={**session.task,'suite':command.get('suite')}
                try:
                    out['report']=session.commit_deltas({'w':[D.block((range(0,4),),torch.full((4,),.125))]},'candidate','test',{'kind':'manual'})
                finally:session.task={**session.task,'suite':None}
            else:raise ValueError('unknown fixture command')
        except RuntimeRefusal as exc:
            out.update(ok=False,event=exc.record,tokens=[int(t) for t in exc.tokens],committed=bool(getattr(exc,'committed',None)))
        except Exception as exc:out.update(ok=False,error=f'{type(exc).__name__}: {exc}')
        write(root/'reply.json',out)
from wsp.server import create_app
from wsp.lifecycle import serve
sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(128)
port=sock.getsockname()[1];url=f'http://127.0.0.1:{port}'
app=create_app(session,url)
write(root/'listener.json',{'port':port})
threading.Thread(target=commands,daemon=True).start()
try:serve(app,sockets=[sock],host='127.0.0.1',port=port,log_level='error')
finally:stop.set();sock.close()
'''

MODEL_WORKER = r'''
import argparse, os, socket, threading, time
from pathlib import Path
root=Path(os.environ['HOME'])
parser=argparse.ArgumentParser();parser.add_argument('--runtime-dir');parser.add_argument('--socket');parser.add_argument('--workspace')
args=parser.parse_args()
from wsp.worker import serve_worker, watch_parent
from wsp.runtime_supervisor import write
watch_parent()
token=os.environ.pop('WSP_WORKER_TOKEN',None)
gpu=Path(args.runtime_dir)/'watchdog.jsonl.gpu.json'
def telemetry():
    while True:
        try:write(gpu,dict(time=time.time(),active=False,backend='cpu',pid=os.getpid()))
        except Exception:pass
        time.sleep(.025)
threading.Thread(target=telemetry,daemon=True).start()
import fixture_common as fx
listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
previous=os.umask(0o177)
try:listener.bind(args.socket)
finally:os.umask(previous)
listener.listen(1)
status=serve_worker(listener,args.workspace,token,fx.FakeRunner,runtime_dir=args.runtime_dir,
                    open_workspace=lambda directory:fx.workspace(directory,patches=False))
try:os.unlink(args.socket)
except FileNotFoundError:pass
raise SystemExit(status)
'''

Z, A = 15, 1   # token ids of 'z' (only in the second suite case) and 'a' (in the drift corpus)


@contextmanager
def stack(tmp_path):
    (tmp_path/'fixture_common.py').write_text(FIXTURE_COMMON)
    with control_fixture(tmp_path, server=SERVER, model_worker=MODEL_WORKER) as f:
        yield f


def http(f, path, body=None, timeout=60):
    port = read(f.root/'listener.json')['port']
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{port}'+path, data=data, method='GET' if body is None else 'POST',
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def background(fn):
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault('value', fn()), daemon=True)
    thread.start()
    return thread, out


def behave(f, **values):
    write(f.root/'behavior.json', values)


def workers(f):
    return [w['pid'] for w in spawned_workers(f.folder)]


def hanging(f, pid, seconds=20):
    return wait(lambda: read(f.root/f'hanging-{pid}.json'), seconds)


def patch_hashes(f):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((f.root/'ws'/'patches').iterdir())}


def events(f, name):
    return [r for r in log_rows(f) if r['event'] == name]


def run_body(n, prompt='ab'):
    return dict(prompt=prompt, max_new_tokens=n, temperature=0, top_k=0, seed=0, top_k_show=3)


def assert_loss_visible(f, code):
    assert f.process.poll() is None and server_signals(f) == []
    status, state = http(f, '/api/state')
    assert status == 200 and state['open']['name'] == 'scalar'
    assert state['status']['runner_loaded'] is False and state['status']['run_error'] is None
    assert f.api('/api/runtime')['event']['code'] == code
    return state


def assert_new_worker(f, old_pid, begins_before):
    behave(f)
    status, result = http(f, '/api/run', run_body(3))
    assert status == 200 and not result.get('interrupted'), result
    assert result['base']['text'] == 'cde' and result['current']['text'] == 'cde'
    pids = workers(f)
    assert pids[-1] != old_pid and alive(pids[-1]) and not alive(old_pid)
    assert len(events(f, 'begin')) > begins_before
    assert wait(lambda: (h := f.heartbeat())['runner'] == 'RemoteRunner' and h['runner_pid'] == pids[-1] and h['lost'] is None)
    return pids[-1]


# ---------------------------------------------------------------- full stack: happy path
def test_supervised_gguf_runs_every_call_class_in_the_worker(tmp_path):
    with stack(tmp_path) as f:
        assert workers(f) == []
        status, result = http(f, '/api/run', run_body(4))
        assert status == 200 and not result.get('interrupted'), result
        # next_logits picks 'c' after "ab"; generate counts up from the last token.
        assert result['base']['next'][0][0] == 'c' and result['current']['next'][0][0] == 'c'
        assert result['base']['text'] == 'cdef' and result['current']['text'] == 'cdef'
        assert result['base']['status'] == result['current']['status'] == 'complete'
        [pid] = workers(f)
        beat = wait(lambda: (h := f.heartbeat())['runner'] == 'RemoteRunner' and h)
        assert beat['runner_pid'] == pid and beat['runner_loaded'] and beat['lost'] is None
        # One admitted worker-mode load spawned it; the run itself was a separate operation.
        loads = [r['descriptor'] for r in events(f, 'begin') if r['descriptor'].get('job') == 'load']
        assert len(loads) == 1 and loads[0]['worker'] is True and loads[0]['runner'] is True
        assert [r['descriptor']['operation_id'] for r in events(f, 'spawn_worker')] == [loads[0]['operation_id']]
        # The model never ran in the server: it holds a RemoteRunner and its pid is not the worker's.
        assert beat['pid'] != pid
        status, chat = http(f, '/api/chat', dict(messages=[{'role': 'user', 'content': 'ab'}], force_template=True,
                                                 template={'user': '', 'assistant': '', 'sep': ' '}, max_new_tokens=4, temperature=0))
        assert status == 200 and not chat.get('interrupted'), chat
        assert chat['base'] == chat['current'] == 'abcd' and chat['variants'] == {'base': 'complete', 'current': 'complete'}
        out = f.command('calls', seconds=20)
        assert out['ok'], out
        assert out['available'] == [True, False] and out['max_ctx'] == 64 and out['threads'] == 2 and out['pid'] == pid
        # tail crosses one row; run(names) returns only the requested captures the worker has.
        assert out['full'] == [3, 16] and out['tail'] == [1, 16]
        assert out['run_names'] == [True, {'blocks.0.hook_resid_post': [1, 3, 8]}]
        assert out['run_plain'] == [[1, 3, 16], {}]
        assert out['next'] == 4
        # A named job whose in-job admission requests are answered inline by the server.
        extends = len(events(f, 'extend'))
        out = f.command('probe', seconds=20, tokens=8, stored=4)
        assert out['ok'], out
        probe = out['probe']
        assert probe['pid'] == pid and probe['uses_worker'] is False
        assert probe['patches'] == ['0001-fixture', '0002-fixture'] and probe['version'] >= 1
        jobs = [r['descriptor'].get('job') for r in events(f, 'extend')[extends:]]
        assert 'forward' in jobs and 'edit' in jobs, jobs
        assert workers(f) == [pid] and alive(pid)
        assert f.process.poll() is None and server_signals(f) == []


def test_results_and_arguments_above_the_header_cap_cross_and_an_unsendable_request_is_no_loss(tmp_path):
    with stack(tmp_path) as f:
        out = f.command('big', seconds=60)
        assert out['ok'] and out['equal'] is True, out
        [pid] = workers(f)
        assert out['pid'] == pid
        # An argument the server cannot encode is refused before anything is sent: no loss,
        # the same worker stays loaded and answers the next job.
        out = f.command('unsendable', seconds=30)
        assert out['ok'], out
        assert out['event']['code'] == 'worker_request_unsendable' and out['event']['action'] == 'refused'
        assert out['lost'] is None and out['same'] is True and out['pid'] == pid and out['probe']['pid'] == pid
        assert workers(f) == [pid] and alive(pid) and events(f, 'worker_signal') == [] and f.heartbeat()['lost'] is None
        assert f.process.poll() is None and server_signals(f) == []


# ---------------------------------------------------------------- full stack: losses
def test_sigkill_mid_generate_is_an_interrupted_run_and_the_next_run_uses_a_new_worker(tmp_path):
    with stack(tmp_path) as f:
        hashes = patch_hashes(f)
        status, result = http(f, '/api/run', run_body(2))
        assert status == 200 and not result.get('interrupted')
        [pid] = workers(f)
        begins = len(events(f, 'begin'))
        behave(f, hang_after=3, hang_variant='current')
        thread, out = background(lambda: http(f, '/api/run', run_body(8)))
        assert hanging(f, pid)['emitted'] == [3, 4, 5]
        os.kill(pid, signal.SIGKILL)
        thread.join(30)
        status, result = out['value']
        assert status == 200 and result['interrupted'] is True
        assert result['event']['code'] == 'worker_exited' and result['event']['measured']['worker_pid'] == pid
        # The edited model runs first: it keeps the partial text and the base model is not started.
        assert result['current']['status'] == 'interrupted' and result['current']['text'] == 'cde'
        assert result['current']['event']['code'] == 'worker_exited'
        assert result['base']['status'] == 'not_run'
        assert_loss_visible(f, 'worker_exited')
        assert patch_hashes(f) == hashes
        assert_new_worker(f, pid, begins)
        assert patch_hashes(f) == hashes


def test_ladder_kill_mid_generate_reports_worker_killed(tmp_path):
    with stack(tmp_path) as f:
        hashes = patch_hashes(f)
        assert http(f, '/api/run', run_body(2))[0] == 200
        [pid] = workers(f)
        begins = len(events(f, 'begin'))
        behave(f, hang_after=2, hang_variant='current')
        thread, out = background(lambda: http(f, '/api/run', run_body(8)))
        hanging(f, pid)
        f.telemetry(pressure=2)
        thread.join(30)
        status, result = out['value']
        assert status == 200 and result['interrupted'] is True, result
        assert result['event']['code'] == 'worker_killed' and result['event']['cause']['code'] == 'memory_pressure'
        assert result['current']['status'] == 'interrupted' and result['current']['text'] == 'cd'
        assert result['base']['status'] == 'not_run'
        wait(lambda: not alive(pid))
        text = (f.folder/'watchdog.jsonl').read_text()
        assert 'watchdog_unavailable' not in text
        f.telemetry(pressure=1)
        assert_loss_visible(f, 'worker_killed')
        assert patch_hashes(f) == hashes
        assert_new_worker(f, pid, begins)


def live_events(f):
    """Subscribe to the server's event stream as a browser does. -> (generation events so far, close)"""
    port = read(f.root/'listener.json')['port']
    response = urllib.request.urlopen(f'http://127.0.0.1:{port}/api/events', timeout=60)
    assert response.readline().startswith(b'retry:')      # subscribed before anything is requested
    seen = []
    def pump():
        try:
            for line in response:
                if line.startswith(b'data: '):
                    item = json.loads(line[6:])
                    if item.get('type') == 'generation':seen.append(item)
        except (OSError, ValueError, AttributeError):       # closed under the reader at the end of the test
            pass
    threading.Thread(target=pump, daemon=True).start()
    return seen, response.close


def shown(events, variant):
    text = ''
    for e in events:
        if e['variant'] == variant and 'text' in e:
            assert e['keep'] <= len(text)
            text = text[:e['keep']]+e['text']
    return text


def test_generated_text_reaches_the_browser_while_the_request_is_still_running(tmp_path):
    with stack(tmp_path) as f:
        assert http(f, '/api/run', run_body(2))[0] == 200
        [pid] = workers(f)
        seen, close = live_events(f)
        try:
            # The worker emits three tokens of the edited model's text and then stalls.
            behave(f, hang_after=3, hang_variant='current')
            thread, out = background(lambda: http(f, '/api/run', {**run_body(8), 'stream': 'live-1'}))
            assert hanging(f, pid)['emitted'] == [3, 4, 5]
            # Everything generated before the stall is shown, with no further token to push it out.
            wait(lambda: shown(seen, 'current') == 'cde')
            assert thread.is_alive() and 'value' not in out          # nothing has been returned yet
            assert all(e['stream'] == 'live-1' and e['variant'] == 'current' and e['state'] == 'started' for e in seen)
            os.kill(pid, signal.SIGKILL)
            thread.join(30)
            status, result = out['value']
            assert status == 200 and result['interrupted'] is True and result['current']['text'] == 'cde'
            last = wait(lambda: seen[-1]['state'] == 'interrupted' and seen[-1])
            assert last['variant'] == 'current' and shown(seen, 'current') == 'cde'
            # A complete request streams the edited text, then the base text, and matches its response.
            behave(f)
            del seen[:]
            status, result = http(f, '/api/chat', dict(messages=[{'role': 'user', 'content': 'ab'}], force_template=True,
                                                       template={'user': '', 'assistant': '', 'sep': ' '}, max_new_tokens=4,
                                                       temperature=0, stream='live-2'))
            assert status == 200 and not result.get('interrupted'), result
            wait(lambda: [e['state'] for e in seen if e['state'] != 'started'] == ['complete', 'complete'])
            assert [e['variant'] for e in seen] == sorted((e['variant'] for e in seen), key=['current', 'base'].index)
            assert shown(seen, 'current') == result['current'] == 'abcd' and shown(seen, 'base') == result['base'] == 'abcd'
            assert workers(f)[-1] != pid and f.process.poll() is None and server_signals(f) == []
        finally:
            close()


def test_suite_with_a_mid_way_kill_keeps_finished_cases_and_never_fails_the_rest(tmp_path):
    with stack(tmp_path) as f:
        status, report = http(f, '/api/tests?suite=probe')
        assert status == 200 and report['passed'] == 3 and report['score'] == 1.0, report
        [pid] = workers(f)
        behave(f, hang_token=Z)
        thread, out = background(lambda: http(f, '/api/tests?suite=probe&variant=base'))
        hanging(f, pid)
        os.kill(pid, signal.SIGKILL)
        thread.join(30)
        status, body = out['value']
        assert status == 409 and body['status'] == 'not_run' and body['event']['code'] == 'worker_exited'
        report = body['report']
        assert report['score'] is None and report['not_run'] == 2 and report['passed'] == 1
        first, second, third = report['cases']
        assert first['pass'] is True and first['got'] == 'c'
        assert second['status'] == third['status'] == 'not_run' and second['pass'] is None and third['pass'] is None
        assert all(c['pass'] is not False for c in report['cases'])
        assert_loss_visible(f, 'worker_exited')
        assert_new_worker(f, pid, 0)


def test_drift_after_a_loss_is_not_run_not_zero(tmp_path):
    with stack(tmp_path) as f:
        assert http(f, '/api/run', run_body(1))[0] == 200
        [pid] = workers(f)
        behave(f, hang_token=A)
        thread, out = background(lambda: http(f, '/api/load', {}))
        hanging(f, pid)
        os.kill(pid, signal.SIGKILL)
        thread.join(30)
        status, body = out['value']
        assert status == 409 and body['status'] == 'not_run' and body['event']['code'] == 'worker_exited', body
        assert 'kl' not in json.dumps(body)
        state = assert_loss_visible(f, 'worker_exited')
        assert state['status']['drift']['kl'] is None
        # Session.drift itself: the loss is the refusal, never a zero-drift result.
        new = assert_new_worker(f, pid, 0)
        behave(f, hang_token=A)
        n = f.send('drift')
        hanging(f, new)
        os.kill(new, signal.SIGKILL)
        result = f.response(n, seconds=20)
        assert not result['ok'] and result['event']['code'] == 'worker_exited' and 'drift' not in result
        assert f.heartbeat()['run_error'] is None


def test_candidate_edit_whose_tests_lost_the_worker_is_not_committed(tmp_path):
    with stack(tmp_path) as f:
        hashes = patch_hashes(f)
        assert http(f, '/api/run', run_body(1))[0] == 200
        [pid] = workers(f)
        behave(f, hang_token=Z)
        n = f.send('edit', suite='probe')
        hanging(f, pid, seconds=40)
        os.kill(pid, signal.SIGKILL)
        result = f.response(n, seconds=20)
        assert not result['ok'] and result['event']['code'] == 'worker_exited' and not result['committed'], result
        assert 'report' not in result
        assert patch_hashes(f) == hashes
        assert f.heartbeat()['patches'] == ['0001-fixture', '0002-fixture']
        assert_loss_visible(f, 'worker_exited')
        # The same edit with a healthy worker commits through the same path.
        behave(f)
        result = f.command('edit', seconds=40, suite='probe')
        assert result['ok'] and result['report']['committed'], result
        assert len(patch_hashes(f)) == len(hashes)+2
        assert workers(f)[-1] != pid


def test_load_transport_failure_leaves_run_error_unset_but_a_load_error_frame_sets_it(tmp_path):
    with stack(tmp_path) as f:
        behave(f, load_exit=True)
        status, body = http(f, '/api/run', run_body(1))
        assert status == 409 and body['status'] == 'not_run' and body['event']['code'] == 'worker_exited', body
        [pid] = workers(f)
        wait(lambda: not alive(pid))
        assert f.heartbeat()['run_error'] is None and f.heartbeat()['runner'] is None
        status, state = http(f, '/api/state')
        assert state['status']['run_error'] is None and state['status']['can_run'] is True
        assert f.process.poll() is None and server_signals(f) == []
        # A deterministic construction failure, answered by an error frame, is remembered.
        behave(f, load_error=True)
        status, body = http(f, '/api/run', run_body(1))
        assert status == 400 and 'the fixture model is unreadable' in body['error'], body
        second = workers(f)[-1]
        assert second != pid
        wait(lambda: not alive(second))
        assert wait(lambda: 'the fixture model is unreadable' in (f.heartbeat()['run_error'] or ''))
        status, state = http(f, '/api/state')
        assert 'the fixture model is unreadable' in state['status']['run_error'] and state['status']['can_run'] is False
        # The worker answered its failed load and stopped normally: not a loss.
        rows = log_rows(f)
        later = rows[max(i for i, r in enumerate(rows) if r['event'] == 'worker_spawned'):]
        time.sleep(.3)
        later = log_rows(f)[len(rows)-len(later):]
        assert 'release' in [r['event'] for r in later]
        assert not [r for r in later if r['event'] == 'worker_signal'] and 'worker_exited' not in json.dumps(later)
        # Logged like any requested stop: one worker_stopped row (status 0), never worker_reaped.
        stopped = wait(lambda: [r for r in log_rows(f) if r['event'] in ('worker_stopped', 'worker_reaped') and r['pid'] == second])
        time.sleep(.3)
        stopped = [r for r in log_rows(f) if r['event'] in ('worker_stopped', 'worker_reaped') and r['pid'] == second]
        assert [(r['event'], r['returncode'], r['killed']) for r in stopped] == [('worker_stopped', 0, False)], stopped


def test_callback_handler_failure_loses_the_runner_and_the_next_load_is_in_sync(tmp_path):
    with stack(tmp_path) as f:
        assert http(f, '/api/run', run_body(1))[0] == 200
        [pid] = workers(f)
        result = f.command('probe', seconds=20, tokens=8, stored=4, fail_preflight=True)
        assert not result['ok'] and result['event']['code'] in ('worker_exited', 'worker_killed'), result
        wait(lambda: not alive(pid))
        reaped = wait(lambda: [r for r in log_rows(f) if r['event'] == 'worker_reaped'])
        assert reaped[-1]['returncode'] == 70   # EXIT_LOST: the server closed the connection mid-callback
        assert f.heartbeat()['run_error'] is None
        assert f.process.poll() is None and server_signals(f) == []
        # The next operation loads a new worker; its jobs match the committed patches.
        result = f.command('probe', seconds=20, tokens=8, stored=4)
        assert result['ok'], result
        assert result['probe']['pid'] != pid and result['probe']['patches'] == ['0001-fixture', '0002-fixture']
        assert http(f, '/api/run', run_body(3))[1]['base']['text'] == 'cde'


def test_idle_worker_loss_drops_the_runner_releases_and_reloads(tmp_path):
    with stack(tmp_path) as f:
        assert http(f, '/api/run', run_body(1))[0] == 200
        [pid] = workers(f)
        wait(lambda: not f.heartbeat()['operation_id'])
        releases = len(events(f, 'release'))
        os.kill(pid, signal.SIGKILL)
        # The control thread drops the lost runner while idle and releases the reservation.
        wait(lambda: f.heartbeat()['runner'] is None and not f.heartbeat()['reserved'], seconds=10)
        wait(lambda: len(events(f, 'release')) > releases)
        assert f.heartbeat()['runner_loaded'] is False and f.heartbeat()['run_error'] is None
        assert not (f.folder/'lease.json').exists() or not read(f.folder/'lease.json').get('ok')
        assert_loss_visible(f, 'worker_exited')
        assert_new_worker(f, pid, 0)


def test_a_loss_only_the_server_saw_is_released_and_the_next_load_reaps_that_worker(tmp_path):
    with stack(tmp_path) as f:
        assert http(f, '/api/run', run_body(1))[0] == 200
        [pid] = workers(f)
        begins = len(events(f, 'begin'))
        behave(f, hang_after=1)
        (f.root/'self-loss').touch()
        status, result = http(f, '/api/run', run_body(5))
        assert result.get('interrupted') or status != 200, result
        hanging(f, pid)
        assert alive(pid) and not [r for r in log_rows(f) if r['event'] in ('worker_reaped', 'worker_stopped')]
        # The supervisor saw nothing; the server releases so the supervisor owns the reap.
        wait(lambda: f.heartbeat()['runner'] is None and not f.heartbeat()['reserved'], seconds=10)
        assert alive(pid)  # still hanging: the release_reap deadline (5 s) has not passed
        # The next load does not wait for that deadline: the old worker is SIGKILLed at once.
        new = assert_new_worker(f, pid, begins)
        reaps = [r for r in log_rows(f) if r['event'] == 'worker_signal' and r['pid'] == pid]
        assert [(r['reason'], r['signal']) for r in reaps] == [('release_reap', int(signal.SIGKILL))], reaps
        assert 'worker_busy' in json.dumps(log_rows(f))
        assert wait(lambda: [r for r in log_rows(f) if r['event'] == 'worker_reaped' and r['pid'] == pid])
        assert alive(new) and f.process.poll() is None and server_signals(f) == []


# ---------------------------------------------------------------- units
def test_only_supervised_worker_mode_gguf_constructs_a_remote_runner(tmp_path, monkeypatch):
    from wsp import runner as R, remote_runner, worker_jobs
    from wsp.runtime_supervisor import RuntimeClient
    from wsp.worker_lease import WorkerLease
    built = []
    class Sentinel:
        def __init__(self, ws):built.append(ws.kind)
    monkeypatch.setattr(remote_runner, 'RemoteRunner', Sentinel)
    import wsp.gguf_runner as G
    monkeypatch.setattr(G, 'GGUFRunner', lambda ws: 'in-process')
    worker = RuntimeClient(tmp_path, {'values': {}}, worker=True)
    plain = RuntimeClient(tmp_path, {'values': {}})
    hf = SimpleNamespace(kind='hf', name='hf', struct=SimpleNamespace(runner='none'), _runtime_client=worker)
    assert not worker_jobs.uses_worker(hf)
    with pytest.raises(RuntimeError, match='no config.json'):
        R.make_runner(hf)
    # PR A fallback: a supervised client without worker mode keeps the in-process runner.
    assert R.make_runner(SimpleNamespace(kind='gguf', _runtime_client=plain)) == 'in-process'
    assert built == []
    # Inside the model worker the client is the WorkerLease: never a RemoteRunner there.
    inside = SimpleNamespace(kind='gguf', _runtime_client=WorkerLease(tmp_path))
    assert not worker_jobs.uses_worker(inside) and R.make_runner(inside) == 'in-process'
    assert built == []
    assert isinstance(R.make_runner(SimpleNamespace(kind='gguf', _runtime_client=worker)), Sentinel) and built == ['gguf']


class _Client:
    def __init__(self):self.names = [];self.operation_id = None
    @contextmanager
    def operation(self, ws, name, **kwargs):
        self.names.append((name, kwargs.get('tokens', 0)))
        yield


class _JobRunner:
    def __init__(self):self.calls = []
    def job(self, name, **arguments):
        self.calls.append((name, arguments))
        return {'job': name}


@pytest.mark.parametrize('name', ['trace', 'patching', 'ablation_scan', 'attribute'])
def test_whole_analysis_calls_cross_as_named_jobs(monkeypatch, name):
    from wsp import analysis, worker_jobs
    from wsp.atlas import guide
    from wsp.worker_jobs import JOBS, encode_job
    monkeypatch.setattr(worker_jobs, 'uses_worker', lambda ws: True)
    runner = _JobRunner()
    ws = SimpleNamespace(lock=threading.RLock(), _runtime_client=_Client(), runner=runner, encode=lambda text: list(range(len(text))))
    calls = {'trace': lambda: analysis.trace(ws, 'abcd', max_tokens=3),
             'patching': lambda: analysis.patching(ws, 'abc', 'abd', 'x'),
             'ablation_scan': lambda: analysis.ablation_scan(ws, [([1, 2], 3)]),
             'attribute': lambda: guide.attribute(ws, 'abc', 'x')}
    assert calls[name]() == {'job': name}
    [(job, arguments)] = runner.calls
    assert job == name
    extra = arguments.pop('extra', None)
    assert set(arguments) == set(JOBS[name].args) and extra is None
    encode_job(name, arguments)          # every forwarded argument is one the worker row accepts
    json.dumps(arguments)                # and crosses as JSON
    names = [n for n, _ in ws._runtime_client.names]
    assert names[-1] == name
    if name == 'trace':assert ws._runtime_client.names[0] == ('trace', 3)
    if name == 'patching':assert ws._runtime_client.names[0] == ('patching', 3)


def test_rome_refuses_before_any_load_and_the_observed_stage_loads_only_when_asked(tmp_path, monkeypatch):
    from wsp import atlas, edits, worker_jobs
    monkeypatch.setattr(worker_jobs, 'uses_worker', lambda ws: True)
    class Unloadable:
        kind, name, version, patches, can_run, dir = 'gguf', 'scalar', 0, [], True, str(tmp_path)
        struct = SimpleNamespace(family='generic'); keep = {}
        @property
        def runner(self):raise AssertionError('a model load was requested')
    with pytest.raises(RuntimeError, match='Packed GGUF parameter gradients are unsupported'):
        edits.rome(Unloadable(), 'a b', 'a', 'b')
    (tmp_path/'atlas').mkdir()
    meta = {'version': atlas.VERSION, 'stages': {'weights': {'layers': [0]}}}
    (tmp_path/'atlas'/'meta.json').write_text(json.dumps(meta))
    monkeypatch.setattr(atlas, 'fetch_bytes', lambda ws: 0)
    # An explicit observed request runs the stage on a worker model: it asks for the model (here, the load is refused).
    with pytest.raises(AssertionError, match='a model load was requested'):
        atlas.build(Unloadable(), stages=['observed'], log=lambda *a: None)
    assert json.loads((tmp_path/'atlas'/'meta.json').read_text())['stages'] == meta['stages']    # the weights stage stays


def test_default_atlas_stages_skip_observed_for_a_worker_model(tmp_path, monkeypatch):
    from wsp import atlas, worker_jobs
    from wsp.atlas import observe
    monkeypatch.setattr(worker_jobs, 'uses_worker', lambda ws: True)
    monkeypatch.setattr(atlas, 'fetch_bytes', lambda ws: 0)
    monkeypatch.setattr(atlas, 'wiring_cost', lambda ws: float('inf'))
    built, observed = [], []
    monkeypatch.setattr(atlas, 'build_weights', lambda ws, adir, log, variant, force: built.append(variant) or {'layers': [0]})
    monkeypatch.setattr(observe, 'build_observed', lambda ws, adir, log, **kw: observed.append(kw['variant']) or {'layers': [0]})
    class Worker:
        kind, name, version, patches, can_run, dir = 'gguf', 'scalar', 0, [], True, str(tmp_path)
        struct = SimpleNamespace(family='generic'); keep = {}
        @property
        def runner(self):raise AssertionError('a model load was requested')
    lines = []
    meta = atlas.build(Worker(), log=lines.append)
    assert built == ['base'] and list(meta['stages']) == ['weights'] and observed == []    # no model was loaded
    assert meta['skipped'] == {'observed': atlas.OBSERVED_NOT_BY_DEFAULT}
    assert any(line.strip() == 'observed stage skipped: not run by default on a model that runs in the model worker, because it '
               'loads and runs the model; ask for the observed stage explicitly to build it' for line in lines)
    assert 'skipped' not in json.loads((tmp_path/'atlas'/'meta.json').read_text())
    # Asked for explicitly, it runs, after the weights stage.
    built.clear()
    meta = atlas.build(Worker(), stages=['weights', 'observed'], log=lines.append, force=True)
    assert built == ['base'] and observed == ['base'] and list(meta['stages']) == ['weights', 'observed'] and 'skipped' not in meta
    # A worker model that cannot be run is not reported as skipped by default: the stage is not available at all.
    class Unrunnable(Worker):
        can_run, run_error = False, 'no runner'
    meta = atlas.build(Unrunnable(), log=lines.append, force=True)
    assert 'skipped' not in meta and list(meta['stages']) == ['weights'] and observed == ['base']


def test_lost_runner_serves_only_its_own_operation_and_is_not_loaded():
    from wsp import runtime_supervisor
    from wsp.workspace import Workspace
    lost = SimpleNamespace(lost={'code': 'worker_exited'}, lost_in='op-1')
    dropped = []
    class Client:
        operation_id = 'op-1'
        def drop_lost(self, ws):dropped.append(ws._runner);ws._runner = 'fresh'
    ws = SimpleNamespace(lock=threading.RLock(), _runner=lost, _runtime_client=Client(), kind='gguf', _failed_runner=None)
    # Inside the operation that lost it: the lost runner refuses at once (no reload mid-operation).
    assert Workspace.runner.fget(ws) is lost and dropped == []
    assert Workspace.runner_loaded.fget(ws) is False and runtime_supervisor.runner_loaded(ws) is False
    ws._runtime_client.operation_id = 'op-2'
    assert Workspace.runner.fget(ws) == 'fresh' and dropped == [lost]
    assert Workspace.runner_loaded.fget(ws) is True


def test_lost_remote_runner_refuses_remote_calls_at_once():
    from wsp.remote_runner import RemoteRunner
    from wsp.runtime_events import RuntimeRefusal
    record = {'code': 'worker_exited', 'action': 'worker_killed', 'message': 'lost', 'recommendations': []}
    r = object.__new__(RemoteRunner)
    r.lock, r.channel, r.lost, r.lost_in, r.max_ctx, r.pid = threading.RLock(), None, record, 'op', 64, 1
    r.ws = SimpleNamespace(_runtime_client=None)
    for call in (lambda: r.job('trace', prompt='x'), lambda: r._call('generate', {'ids': [1]})):
        with pytest.raises(RuntimeRefusal) as exc:
            call()
        assert exc.value.record is record
    r.release()   # nothing to shut down


def test_runtime_child_server_runs_gguf_in_the_worker(tmp_path, monkeypatch):
    import wsp.__main__ as main
    from wsp import lifecycle, runtime_config, runtime_supervisor, server
    directory = tmp_path/'runtime';directory.mkdir()
    (directory/'settings.json').write_text(json.dumps({'platform': 'Darwin', 'values': {
        'host': '127.0.0.1', 'port': 1, 'model_roots': [], 'context': 128, 'new_tokens': 4}}))
    monkeypatch.setenv('WSP_RUNTIME_DIR', str(directory.resolve()))
    made = []
    class Client:
        def __init__(self, directory, settings, worker=False):made.append(worker)
    monkeypatch.setattr(runtime_supervisor, 'RuntimeClient', Client)
    monkeypatch.setattr(runtime_config, 'activate', lambda settings: 'cpu')
    monkeypatch.setattr(server, 'create_app', lambda sess, url: 'app')
    served = []
    monkeypatch.setattr(lifecycle, 'serve', lambda app, **kwargs: served.append(app))
    main.main(['serve', '--empty', '--runtime-child', str(directory), '--root', str(tmp_path/'root')])
    assert made == [True] and served == ['app']


def test_spawn_retries_worker_busy_in_short_steps_checking_the_lease(tmp_path, monkeypatch):
    from wsp import remote_runner
    from wsp.runtime_events import RuntimeRefusal, event
    assert .05 <= remote_runner.SPAWN_RETRY_SECONDS <= .1
    busy = lambda: RuntimeRefusal(event('worker_busy', 'refused', 'The previous model worker is stopping; retry shortly.'))  # noqa: E731
    class Client:
        operation_id, binding = 'op', 'binding'
        def __init__(self, busy_times, check=None):
            self.busy_times, self.calls, self.checks, self.check = busy_times, [], [], check
        def spawn_worker(self, operation_id, directory):
            self.calls.append((time.monotonic(), operation_id, directory))
            if len(self.calls) <= self.busy_times:raise busy()
            return {'pid': 1, 'socket': 's', 'token': 't'}
        def _check(self, binding):
            self.checks.append(binding)
            if self.check:self.check(len(self.checks))
    runner = remote_runner.RemoteRunner.__new__(remote_runner.RemoteRunner)
    runner.ws = SimpleNamespace(dir=str(tmp_path))
    client = Client(3)
    assert runner._spawn(client)['pid'] == 1
    assert [c[1:] for c in client.calls] == [('op', str(tmp_path))]*4 and client.checks == ['binding']*3
    steps = [b[0]-a[0] for a, b in zip(client.calls, client.calls[1:])]
    assert all(.05 <= s < .5 for s in steps), steps
    # A revoked lease ends the retries with its own refusal.
    def revoked(n):
        if n == 2:raise RuntimeRefusal(event('lease_revoked', 'abort', 'The operation lease was revoked.'))
    client = Client(10, revoked)
    with pytest.raises(RuntimeRefusal) as exc:runner._spawn(client)
    assert exc.value.record['code'] == 'lease_revoked' and len(client.calls) == 2
    # Past WORKER_START_SECONDS the worker_busy refusal is the answer; any other refusal is never retried.
    monkeypatch.setattr(remote_runner, 'WORKER_START_SECONDS', .3)
    client = Client(1000)
    with pytest.raises(RuntimeRefusal) as exc:runner._spawn(client)
    assert exc.value.record['code'] == 'worker_busy' and 2 <= len(client.calls) <= 6
    class Other(Client):
        def spawn_worker(self, operation_id, directory):
            self.calls.append(1);raise RuntimeRefusal(event('worker_start_failed', 'refused', 'no'))
    client = Other(0)
    with pytest.raises(RuntimeRefusal):runner._spawn(client)
    assert client.calls == [1] and client.checks == []
