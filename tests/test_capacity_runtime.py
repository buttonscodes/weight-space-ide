"""Synthetic kernels, metadata and fake forwards only; never construct a model."""
import ctypes
import mmap
import os
from threading import RLock
from types import SimpleNamespace
import gguf
import numpy as np
import pytest
import torch
from wsp.packed import PackedKernel, MetalKernel
from wsp.capacity import PREFILL_CHUNK
from wsp import gguf_runner as gr
from wsp.memory_forecast import estimate
from wsp.memory_guard import GiB, GuardError
from wsp.runtime_config import resolve
from wsp.runtime_supervisor import RuntimeClient, descriptor, fingerprint, policy_for, runtime_sample, write
from wsp.settings import interactive_defaults
from wsp.runtime_events import RuntimeRefusal
from wsp.runtime_policy import admission, observations
from test_runtime_supervisor import fake_ws


def raw_matrix():
    w=np.arange(17*64,dtype=np.float32).reshape(17,64)/1000
    return w,w.view(np.uint8).reshape(17,256)


@pytest.mark.parametrize('kind',['cpu','metal'])
def test_chunked_calls_preserve_rows_and_only_retry_scratch(kind):
    w,raw=raw_matrix();x=torch.arange(8201*64,dtype=torch.float32).reshape(8201,64)/10000;calls=[]
    def compute(*args):
        q,ptr,rows,cols,xptr,count,yptr=args if kind=='metal' else args[:-1]
        calls.append(count)
        if count>127:return -12
        a=np.ctypeslib.as_array((ctypes.c_float*(count*cols)).from_address(xptr)).reshape(count,cols)
        out=np.ctypeslib.as_array((ctypes.c_float*(count*rows)).from_address(yptr)).reshape(count,rows)
        out[:]=a@w.T
        return 0
    cls=PackedKernel if kind=='cpu' else MetalKernel;k=cls.__new__(cls)
    k.supports=lambda _:True;k.threads=1;k.handle=1
    k.lib=SimpleNamespace(wsp_matmul=compute);k.metal=SimpleNamespace(wsp_metal_matmul=lambda handle,*a:compute(*a))
    out=k.matmul(raw,gguf.GGMLQuantizationType.F32,w.shape,x)
    torch.testing.assert_close(out,x@torch.from_numpy(w).T,rtol=1e-5,atol=.001)
    assert max(calls)<={'cpu':2048,'metal':512}[kind] and min(calls)<=127
    k.lib.wsp_matmul=lambda *a:-14;k.metal.wsp_metal_matmul=lambda *a:-14
    with pytest.raises(RuntimeError,match='-14'):k.matmul(raw,gguf.GGMLQuantizationType.F32,w.shape,x[:1])


@pytest.mark.parametrize('kind',['cpu','metal'])
def test_installed_abi1_synthetic_long_rows(kind):
    if os.environ.get('WSP_SYNTHETIC_NATIVE')!='1':pytest.skip('opt-in tiny synthetic native kernel check')
    # Binary-exact bounded values isolate chunk/offset correctness from native
    # mixed-precision accumulation differences (this is not a model quality test).
    w=((np.arange(17*64,dtype=np.float32)%17)-8).reshape(17,64)/8
    raw=w.view(np.uint8).reshape(17,256)
    backing=mmap.mmap(-1,16384);storage=np.frombuffer(backing,dtype=np.uint8)
    storage[:raw.size]=raw.ravel();mapped=storage[:raw.size].reshape(raw.shape)
    store=SimpleNamespace(reader=SimpleNamespace(data=storage),where={'synthetic':SimpleNamespace(n_bytes=raw.nbytes)})
    k=PackedKernel() if kind=='cpu' else MetalKernel(store)
    try:
        for count in (513,2049,8201):
            x=((torch.arange(count*64,dtype=torch.float32)%31)-15).reshape(count,64)/16
            out=k.matmul(mapped,gguf.GGMLQuantizationType.F32,w.shape,x)
            torch.testing.assert_close(out,x@torch.from_numpy(w).T,rtol=0,atol=0)
        # Separate chunk correctness from native floating-point behavior on ramps.
        ramp, ramp_raw = raw_matrix()
        storage[:ramp_raw.size] = ramp_raw.ravel()
        x = torch.arange(513*64,dtype=torch.float32).reshape(513,64)/10000
        whole = k.matmul(mapped,gguf.GGMLQuantizationType.F32,w.shape,x)
        step = 512 if kind == 'metal' else 2048
        reference = torch.cat([k.matmul(mapped,gguf.GGMLQuantizationType.F32,w.shape,x[i:i+step]) for i in range(0,len(x),step)])
        torch.testing.assert_close(whole,reference,rtol=0,atol=0)
    finally:
        if kind=='metal':k.close()
        del mapped,storage,store,k
        import gc
        gc.collect()
        backing.close()


class FakeForward:
    """Cumulative scalar history produces fake logits; not a neural model."""
    def __init__(self,fail_at=None):self.calls=[];self.fail_at=fail_at
    def __call__(self,input_ids,past_key_values,**kw):
        if self.fail_at==len(self.calls):raise RuntimeError('synthetic forward failure')
        past=[] if past_key_values is None else past_key_values
        self.calls.append((list(input_ids[0]),id(past),kw));past.extend(input_ids[0].tolist())
        logits=torch.zeros(1,1,11);logits[0,0,sum(past)%11]=10
        return SimpleNamespace(logits=logits,past_key_values=past)


def fake_runner(monkeypatch,managed=True):
    r=gr.GGUFRunner.__new__(gr.GGUFRunner);r.lock=RLock();r.device=torch.device('cpu');r.max_ctx=32768
    from contextlib import nullcontext
    r.ws=SimpleNamespace(lock=RLock(),_runtime_client=SimpleNamespace(operation=lambda *a,**kw:nullcontext()) if managed else None,hf_config={})
    r.use=lambda *a:None;r.forward_options=lambda _:{};r.model=FakeForward()
    monkeypatch.setattr(gr,'require_lease',lambda *a,**kw:None)
    monkeypatch.delenv('WSP_MAX_NEW_TOKENS',raising=False)
    return r


@pytest.mark.parametrize('count',[1,PREFILL_CHUNK,PREFILL_CHUNK+1,2*PREFILL_CHUNK+17])
def test_prefill_preserves_history_positions_and_cache(count,monkeypatch):
    r=fake_runner(monkeypatch);ids=[2]*count
    assert r.generate(ids,max_new_tokens=2)==[(2*count)%11,(2*count+(2*count)%11)%11]
    calls=r.model.calls
    assert [int(t) for c in calls[:-1] for t in c[0]]==ids
    assert max(len(c[0]) for c in calls)<=PREFILL_CHUNK and len({c[1] for c in calls})==1
    assert all(c[2]['logits_to_keep']==1 for c in calls)
    assert r.last_generation['prompt_tokens']==count
    r.model=FakeForward();assert r.next_logits(ids,'base').argmax().item()==(2*count)%11
    assert len(r.model.calls)==(count+PREFILL_CHUNK-1)//PREFILL_CHUNK


def test_larger_output_error_retry_and_legacy_limits(monkeypatch):
    r=fake_runner(monkeypatch);assert len(r.generate([1],max_new_tokens=513))==513
    r.model=FakeForward(fail_at=1)
    with pytest.raises(RuntimeError,match='synthetic'):r.generate([1]*(PREFILL_CHUNK+1),max_new_tokens=2)
    r.model=FakeForward();assert r.generate([3],variant='base',max_new_tokens=1)==[3]
    r.max_ctx=2
    with pytest.raises(ValueError,match='No history was truncated'):r.generate([1,2],max_new_tokens=1)
    legacy=fake_runner(monkeypatch,False)
    with pytest.raises(ValueError,match='64'):legacy.generate([1],max_new_tokens=65)


def test_forecast_geometry_and_transients():
    kw=dict(weights=18*GiB,context=8192,hidden=4096,layers=64,vocab=150000,heads=32,
            kv_heads=4,head_dim=128,attention_layers=16,intermediate=14336,hybrid=(32,128,128,8,4))
    interactive=estimate(**kw,operation='interactive');trace=estimate(**kw,operation='trace')
    assert interactive['kv']==2*16*8192*4*128*4
    assert interactive['kv_append']==2*8192*4*128*4
    assert interactive['eager_attention']==3*32*128*8192*4 and interactive['logits']==150000*4
    assert interactive['recurrent']==3*48*32*128*128*4
    assert trace['retained_attention']==16*32*8192*8192*4 and trace['total']>interactive['total']
    for f in (interactive,trace):assert f['total']==sum(v for k,v in f.items() if k!='total')


def test_available_budget_and_explicit_migration_preserve_reserves(tmp_path):
    s=resolve(env={'HOME':str(tmp_path)},system='Linux')
    s['values'].update(floor_gib=20,preferred_gib=25,budget_gib=24,ceiling_gib=26,reservation_gib=16)
    machine=lambda:dict(total=128*GiB,available=71*GiB,free=24*GiB,pressure=1)
    preset=interactive_defaults(s,telemetry=machine)
    assert preset==dict(context=8192,new_tokens=2048,resident_seconds=14400,budget_gib=None,ceiling_gib=None,reservation_gib=None)
    p=policy_for(dict(s['values'],**preset),128*GiB,71*GiB)
    assert p.budget==46*GiB and p.preferred==25*GiB and p.floor==20*GiB
    assert policy_for(s['values'],128*GiB,71*GiB).budget==24*GiB and s['values']['budget_gib']==24


def test_trace_working_forecast_ends_with_job_and_next_generation_uses_own_buffers(tmp_path,monkeypatch):
    import time
    ws=fake_ws(tmp_path);ws.struct.n_ctx=32768;ws.version=0
    client=RuntimeClient(tmp_path,resolve(env={'HOME':str(tmp_path)}));requests=[]
    monkeypatch.setattr(client,'attach',lambda ws:None)
    def respond(action,value=None):
        requests.append((action,value))
        chosen=value.get('actual') if action=='finish' else value
        write(tmp_path/'lease.json',dict(ok=True,time=time.time(),pid=os.getpid(),binding=fingerprint(chosen)))
        return {'reserved':True}
    monkeypatch.setattr(client,'_request',respond);monkeypatch.setenv('WSP_GGML_DEVICE','cpu')
    with client.operation(ws,'forward',tokens=2048,retained=True):pass
    with client.operation(ws,'generate',tokens=64):pass
    begun=[value for action,value in requests if action=='begin']
    assert 'retained_attention' in begun[0]['working']
    assert 'retained_attention' not in begun[1]['working']
    assert begun[1]['working']['total']<begun[0]['working']['total']
    assert begun[0]['forecast']==begun[1]['forecast']  # configured-context static forecast
    assert all('working' not in value['actual'] for action,value in requests if action=='finish')


def test_fresh_admission_checks_pressure_and_gpu_evidence(tmp_path,monkeypatch):
    import time
    ws=fake_ws(tmp_path);value=descriptor(ws,8192)
    settings=resolve(env={'HOME':str(tmp_path)})
    current=dict(total=128*GiB,available=71*GiB,pressure=1,tree_footprint=GiB,swap_used=0,swap_out=0,compressed=0)
    assert admission(settings,value,current)[0].budget>value['forecast']['total']
    with pytest.raises(RuntimeRefusal):admission(settings,value,dict(current,pressure=2))
    monkeypatch.setattr('wsp.memory_guard.snapshot',lambda:current)
    for gpu in [dict(time=0,active=True,backend='metal'),dict(time=time.time(),active=True,backend='metal',metal_active=True)]:
        write(tmp_path/'gpu.json',gpu)
        with pytest.raises(GuardError):runtime_sample(tmp_path/'gpu.json',True,0,10)
    write(tmp_path/'gpu.json',dict(time=time.time(),active=True,backend='metal',metal_active=True,metal_allocated=GiB,metal_recommended=100*GiB))
    assert runtime_sample(tmp_path/'gpu.json',True,0,10)['gpu']['metal_allocated']==GiB


def test_server_run_uses_bounded_next_logits_and_managed_output_limit(tmp_path):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    from wsp.session import Session
    s=Session(str(tmp_path/'work'));s.settings=resolve(env={'HOME':str(tmp_path)})
    calls=[]
    def next_logits(ids,variant):calls.append(('next',len(ids),variant));return torch.zeros(7)
    tok=SimpleNamespace(token_str=str,to_dict=lambda:{'mode':'word'})
    s.ws=SimpleNamespace(kind='gguf',struct=SimpleNamespace(n_ctx=32768),can_run=True,tok=tok,
        encode=lambda _:list(range(4097)),runner=SimpleNamespace(next_logits=next_logits),
        logits=lambda *a:pytest.fail('full-sequence logits called'),
        complete=lambda prompt,variant,**kw:calls.append(('complete',kw['max_new_tokens'],variant)) or 'done')
    app=create_app(s,'http://127.0.0.1:41881')
    s.runtime=SimpleNamespace()  # Mark this fake route as managed, without constructing a runner.
    c=TestClient(app,base_url='http://127.0.0.1:41881',raise_server_exceptions=False)
    response=c.post('/api/run',json={'prompt':'long','max_new_tokens':2048})
    assert response.status_code==200,response.text
    assert calls==[('next',4097,'current'),('complete',2048,'current'),('next',4097,'base'),('complete',2048,'base')]
    larger=c.post('/api/run',json={'prompt':'long','max_new_tokens':8192})
    assert larger.status_code==200,larger.text
    assert larger.json()['generation']['effective_output']==8192-4097


def test_supervisor_refused_extension_preserves_operation_lease_budget_and_baseline(tmp_path):
    import json
    from test_runtime_supervisor import inert_supervisor
    with inert_supervisor(tmp_path) as (proc,folder,request):
        value=descriptor(fake_ws(tmp_path),8192);value['backend']='cpu'
        assert request('initial',value)['ok']
        initial_binding=json.loads((folder/'lease.json').read_text())['binding']
        for key,bad in [('context',1),('backend','metal')]:
            denied=request('bad-'+key,dict(value,**{key:bad}))
            assert not denied['ok'] and denied['event']['code']=='configuration_mismatch'
            assert json.loads((folder/'lease.json').read_text())['binding']==initial_binding
        other=tmp_path/'other';other.mkdir()
        replacement=descriptor(fake_ws(other),8192);replacement['backend']='cpu'
        denied=request('different-source',replacement)
        assert not denied['ok'] and denied['event']['code']=='model_changed'
        assert json.loads((folder/'lease.json').read_text())['binding']==initial_binding
        begun=dict(value,runner=True,operation_id='trace-op',job='trace',working={'kv':1024})
        assert request('begin',begun,action='begin')['ok']
        original=json.loads((folder/'lease.json').read_text())
        policy=json.loads((folder/'status.json').read_text())['policy']
        too_big=dict(begun,working={'retained_attention':1024*GiB})
        refusal=request('extension',too_big,action='extend')
        assert not refusal['ok'] and refusal['event']['action']=='refused'
        assert proc.poll() is None
        assert json.loads((folder/'lease.json').read_text())['binding']==original['binding']
        assert json.loads((folder/'status.json').read_text())['policy']==policy
        # A smaller prospective extension retries against current measurements.
        small=dict(begun,working={'retained_attention':1024**2})
        assert request('smaller',small,action='extend')['ok']
        assert json.loads((folder/'status.json').read_text())['policy']==policy
        assert request('finish',dict(operation_id='trace-op',actual=dict(value,runner=True),success=True),action='finish')['ok']
        rows=[json.loads(line) for line in (folder/'watchdog.jsonl').read_text().splitlines()]
        admitted=[r for r in rows if r['event'] in ('bind','begin','extend')]
        assert len({json.dumps(r['baseline'],sort_keys=True) for r in admitted})==1
        assert request('release',action='release')['ok'] and proc.poll() is None
