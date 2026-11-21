"""Reviewer regressions: scalar metadata, fake forwards and fake transport only."""
import copy
import os
import time
from types import SimpleNamespace
import pytest
import torch
from fastapi.testclient import TestClient
from test_settings import active, fake_ws, GiB
from test_runtime_supervisor import fake_ws as binding_ws
from test_capacity_runtime import fake_runner
from wsp.capacity import output_count
from wsp.settings import capacity_recommendation, preview, validate_interactive
from wsp.memory_guard import forecast
from wsp.memory_forecast import estimate
from wsp.runtime_supervisor import RuntimeClient, descriptor, fingerprint, write, policy_for
from wsp.runtime_policy import admission
from wsp.runtime_events import RuntimeRefusal, event


def machine(available=60):
    return lambda:dict(total=128*GiB,available=int(available*GiB),pressure=1)


def test_output_ceiling_preserves_full_prompt_and_zero_scoring():
    assert output_count(2048,7500,8192)==692
    assert output_count(2048,8191,8192)==1
    assert output_count(0,8192,8192,allow_zero=True)==0
    for count,context in [(8192,8192),(8193,8192)]:
        with pytest.raises(ValueError,match='No history was truncated'):output_count(1,count,context)
    for n in (True,1.5,float('nan'),2**53):
        with pytest.raises(ValueError):output_count(n,1,8192)


def test_generator_clamps_before_fake_prefill(monkeypatch):
    r=fake_runner(monkeypatch);r.max_ctx=4
    assert len(r.generate([1,2,3],max_new_tokens=2048))==1
    assert r.last_generation['requested_output']==2048 and r.last_generation['effective_output']==1
    assert len(r.model.calls)==1 and len(r.model.calls[0][0])==3


def test_server_remaining_space_for_completion_chat_and_zero_mode(tmp_path):
    from wsp.server import create_app
    from wsp.session import Session
    s=Session(str(tmp_path/'work'));s.settings=active(tmp_path)
    s.settings['values'].update(context=131072,new_tokens=65536)
    calls=[];size=[131071]
    tok=SimpleNamespace(token_str=str,to_dict=lambda:{'mode':'word'},encode=lambda *a,**k:list(range(size[0])),decode=lambda _: 'reply')
    runner=SimpleNamespace(next_logits=lambda ids,v:torch.zeros(2),chat_prompt=lambda *a,**k:'whole conversation',
        generate=lambda ids,v,extra,**kw:calls.append((len(ids),kw['max_new_tokens'])) or [1])
    s.ws=SimpleNamespace(kind='gguf',struct=SimpleNamespace(n_ctx=131072),can_run=True,tok=tok,runner=runner,
        encode=lambda _:list(range(size[0])),complete=lambda p,v,**kw:calls.append((size[0],kw['max_new_tokens'])) or 'reply')
    app=create_app(s,'http://127.0.0.1:41881');s.runtime=SimpleNamespace()
    client=TestClient(app,base_url='http://127.0.0.1:41881',raise_server_exceptions=False)
    for path,body in [('/api/run',{'prompt':'full'}),('/api/chat',{'messages':[{'role':'user','content':'full'}]})]:
        calls.clear();response=client.post(path,json={**body,'max_new_tokens':65536})
        assert response.status_code==200,response.text
        assert response.json()['generation']['effective_output']==1
        assert calls==[(131071,1),(131071,1)]
    size[0]=131072;calls.clear()
    assert client.post('/api/chat',json={'messages':[{'role':'user','content':'full'}]}).status_code==400
    assert not calls
    response=client.post('/api/run',json={'prompt':'full','max_new_tokens':0})
    assert response.status_code==200 and response.json()['generation']['effective_output']==0 and not calls


def test_upward_preset_exact_forecast_boundary_and_no_fit(tmp_path,monkeypatch):
    s=active(tmp_path);w=fake_ws();w.struct.n_ctx=131072
    s['values'].update(floor_gib=20,preferred_gib=25)
    # The preset uses max(512 MiB, 5% of cap); assert the exact scalar boundary.
    monkeypatch.setattr('wsp.memory_guard.forecast',lambda ws,n,**kw:dict(total=20*GiB+n*1024**2))
    r=capacity_recommendation(s,w,machine())
    assert r['changes']['context']==(r['target']-20*GiB)//1024**2 and r['status']=='estimated'
    assert r['changes']['new_tokens']<r['changes']['context']
    assert capacity_recommendation(s,w,machine(44))['changes'] is None
    limited=capacity_recommendation(s,w,machine(),overrides={'budget_gib':24,'ceiling_gib':26})
    assert limited['changes']['context']==(limited['target']-20*GiB)//1024**2
    assert capacity_recommendation(s,w,machine(),overrides={'context':64})['changes']['context']==64
    w.struct.n_ctx=1
    assert capacity_recommendation(s,w,machine())['changes'] is None


@pytest.mark.parametrize('known',[False,True])
def test_resident_physical_impossibility_known_even_without_credit(tmp_path,monkeypatch,known):
    s=active(tmp_path);s['values'].update(floor_gib=20,preferred_gib=25)
    w=fake_ws();w.runner_loaded=True;w.struct.n_ctx=131072
    monkeypatch.setattr('wsp.memory_guard.forecast',lambda ws,n,**kw:dict(total=110*GiB+n))
    rt=SimpleNamespace(settings=s,status=lambda _:dict(reserved=known,forecast={'total':23*GiB},reservation={}))
    p=preview(s,w,rt,machine(47.12))
    assert any('physical RAM' in message for message in p['warnings'])
    assert p['incremental'] is None if not known else p['incremental']>0
    assert capacity_recommendation(s,w,machine(47.12),rt)['changes'] is None


def test_resident_recommendation_is_physical_bound_not_cold_admission(tmp_path,monkeypatch):
    s=active(tmp_path);s['values'].update(floor_gib=20,preferred_gib=25)
    w=fake_ws();w.runner_loaded=True;w.struct.n_ctx=131072
    monkeypatch.setattr('wsp.memory_guard.forecast',lambda ws,n,**kw:dict(total=20*GiB+n*1024**2))
    r=capacity_recommendation(s,w,machine(47.12))
    assert r['changes']['context']==(r['target']-20*GiB)//1024**2 and r['status']=='uncertain'
    assert 'unknown' in r['note'] and 'upper bound' in r['note']


def test_effective_context_shared_and_large_config_valid(tmp_path):
    s=active(tmp_path);s['values'].update(context=131072,new_tokens=65536)
    validate_interactive(s)
    w=binding_ws(tmp_path);w.struct.n_ctx=4096
    d=descriptor(w,131072)
    assert d['configured_context']==131072 and d['context']==d['model_context']==4096
    assert d['forecast']==forecast(w,4096,operation='interactive')==forecast(w,131072,operation='interactive')
    p=policy_for(s['values'],128*GiB,60*GiB)
    assert admission(s,d,machine()())[1]['required']==d['forecast']['total']


def test_plain_full_forward_keeps_logits_without_retained_trace(monkeypatch):
    from wsp.gguf_runner import HFRunner
    r=fake_runner(monkeypatch);reservations=[]
    from contextlib import nullcontext
    r.ws._runtime_client=SimpleNamespace(operation=lambda ws,name,**kw:reservations.append((kw['tokens'],kw['retained'])) or nullcontext())
    expected=torch.zeros(1,3,7)
    monkeypatch.setattr(HFRunner,'run',lambda *a,**k:(expected,{}))
    assert r.run([1,2,3])[0].shape==(1,3,7)
    r.run([1,2,3],names=['residual'])
    assert reservations==[(3,False),(3,True)]
    args=dict(weights=GiB,context=2048,hidden=128,layers=16,vocab=32000,heads=8,kv_heads=2)
    f=estimate(**args,operation='forward');t=estimate(**args,operation='trace')
    assert f['eager_attention']==t['eager_attention'] and f['logits']==2048*32000*4
    assert 'retained_attention' not in f and 'retained_activations' not in f and f['total']<t['total']


def test_patch_storage_invalidation_is_scoped_and_rejected_prospective_growth_keeps_lease(tmp_path,monkeypatch):
    import wsp.memory_guard as guard
    ws=binding_ws(tmp_path);ws.struct.n_ctx=8192;ws.version=0;ws.patch_stored=0
    c=RuntimeClient(tmp_path,active(tmp_path));requests=[];estimates=[];real=guard.forecast
    monkeypatch.setattr(c,'attach',lambda ws:None)
    monkeypatch.setattr(guard,'forecast',lambda *a,**kw:estimates.append(kw.get('operation')) or real(*a,**kw))
    def respond(action,value=None):
        requests.append((action,value))
        if action=='extend' and value['forecast']['patches']>GiB:
            raise RuntimeRefusal(event('forecast_exceeds_budget','refused','prospective patch too large'))
        chosen=value.get('actual') if action=='finish' else value
        write(tmp_path/'lease.json',dict(ok=True,time=time.time(),pid=os.getpid(),binding=fingerprint(chosen)))
        return {'reserved':True}
    monkeypatch.setattr(c,'_request',respond)
    with c.operation(ws,'edit'):
        calls=len(estimates);before=c.binding
        for _ in range(10):c.require(ws)
        assert len(estimates)==calls
        with pytest.raises(RuntimeRefusal):c.preflight_edit(ws,2**30)
        assert c.binding==before and ws.patch_stored==0 and ws.version==0
        ws.patch_stored=256;ws.patches=[SimpleNamespace(stored=256)];ws.version=1;c.mutated(ws)
    assert len(estimates)>calls
    assert requests[-1][1]['actual']['forecast']['patches']==3072


def test_optional_gguf_geometry_defaults_without_reading_tensors(tmp_path,monkeypatch):
    import gguf
    from wsp.memory_forecast import gguf_preview
    path=tmp_path/'metadata.gguf';path.write_bytes(b'scalar fixture, not a checkpoint')
    values={'general.architecture':'llama','llama.embedding_length':64,'llama.block_count':2,
            'llama.attention.head_count':4,'llama.context_length':4096}
    fields={k:SimpleNamespace(contents=lambda v=v:v) for k,v in values.items()}
    fields['tokenizer.ggml.tokens']=SimpleNamespace(data=list(range(128)))
    tensor=SimpleNamespace(n_elements=100,n_bytes=400,tensor_type=SimpleNamespace(name='F32'),name='synthetic')
    monkeypatch.setattr(gguf,'GGUFReader',lambda *a,**k:SimpleNamespace(fields=fields,tensors=[tensor]))
    result=gguf_preview(path,131072)
    assert result['effective_context']==4096
    assert result['forecast']==estimate(400,4096,64,2,128,heads=4,operation='interactive')


def test_status_forecast_cache_is_serialized_with_admission(tmp_path,monkeypatch):
    import threading
    import wsp.runtime_supervisor as supervisor
    a=binding_ws(tmp_path)
    other=tmp_path/'other';other.mkdir();b=binding_ws(other)
    client=RuntimeClient(tmp_path,active(tmp_path));entered=threading.Event();release=threading.Event();received=[]
    def estimate(ws,*args,**kwargs):
        if ws is a:
            entered.set();assert release.wait(3)
        return {'total':100 if ws is a else 200}
    monkeypatch.setattr(supervisor,'reservation_forecast',estimate)
    def respond(action,value=None):
        received.append(value['forecast']['total'])
        write(tmp_path/'lease.json',dict(ok=True,time=time.time(),pid=os.getpid(),binding=fingerprint(value)))
    monkeypatch.setattr(client,'_request',respond)
    failures=[]
    def run(fn):
        try:fn()
        except BaseException as exc:failures.append(exc)
    status=threading.Thread(target=lambda:run(lambda:client.status(a)))
    admission=threading.Thread(target=lambda:run(lambda:client.require(b)))
    status.start();assert entered.wait(3)
    try:
        # This proves the status calculation and its cached result are protected
        # throughout, not just the assignment of the cache tuple.
        acquired=client.lock.acquire(blocking=False)
        if acquired:client.lock.release()
        assert not acquired
        admission.start()
    finally:release.set();status.join(3)
    admission.join(3)
    assert not failures and not status.is_alive() and not admission.is_alive()
    assert received==[200]


@pytest.mark.parametrize('prompt_tokens',[96,200])
def test_non_gguf_sliding_window_keeps_total_prompt_and_output(tmp_path,prompt_tokens):
    from wsp.server import create_app
    from wsp.session import Session
    session=Session(str(tmp_path/'work'));session.settings=active(tmp_path)
    calls=[]
    tok=SimpleNamespace(token_str=str,to_dict=lambda:{'mode':'word'},
        encode=lambda *a,**k:[1]*prompt_tokens,decode=lambda _: 'reply')
    runner=SimpleNamespace(chat_prompt=lambda *a,**k:'whole conversation',
        generate=lambda ids,v,extra,**kw:calls.append((len(ids),kw['max_new_tokens'])) or [1])
    session.ws=SimpleNamespace(kind='hf',struct=SimpleNamespace(n_ctx=96),can_run=True,tok=tok,runner=runner,
        encode=lambda _:[1]*prompt_tokens,logits=lambda *a:torch.zeros(prompt_tokens,2),
        complete=lambda p,v,**kw:calls.append((prompt_tokens,kw['max_new_tokens'])) or 'reply')
    client=TestClient(create_app(session,'http://127.0.0.1:41881'),base_url='http://127.0.0.1:41881',raise_server_exceptions=False)
    generation=client.get('/api/runtime').json()['generation']
    assert generation['context_limit'] is None and generation['default_output']==2048
    for path,body in [('/api/run',{'prompt':'full'}),('/api/chat',{'messages':[{'role':'user','content':'full'}]})]:
        calls.clear();response=client.post(path,json={**body,'max_new_tokens':200})
        assert response.status_code==200,response.text
        assert response.json()['generation']['effective_output']==200
        assert response.json()['generation']['context'] is None
        assert calls==[(prompt_tokens,200),(prompt_tokens,200)]


@pytest.mark.parametrize('resident',[False,True])
def test_recommendation_buffer_leaves_room_for_jitter_and_patch_growth(tmp_path,resident):
    from wsp.settings import RECOMMENDATION_BUFFER_MIN, RECOMMENDATION_BUFFER_FRACTION
    import math
    settings=active(tmp_path);w=binding_ws(tmp_path)
    w.runner_loaded=resident;w.struct.n_ctx=131072
    w.hf_config=dict(hidden_size=4096,num_hidden_layers=32,num_attention_heads=32,
        num_key_value_heads=8,vocab_size=128256,intermediate_size=14336)
    w.store.nbytes=lambda:int(4.9*GiB)
    sample=dict(machine()(),swap_used=0,swap_out=0,compressed=0,tree_footprint=0)
    recommendation=capacity_recommendation(settings,w,lambda:sample)
    changes=recommendation['changes'];assert changes is not None
    values=dict(settings['values'],**changes)
    policy=policy_for(values,sample['total'],sample['total'] if resident else sample['available'])
    cap=min(policy.budget,policy.ceiling)
    margin=max(RECOMMENDATION_BUFFER_MIN,math.ceil(cap*RECOMMENDATION_BUFFER_FRACTION))
    assert recommendation['cap']==cap and recommendation['margin']==margin
    assert recommendation['target']==cap-margin
    proposed=forecast(w,changes['context'],operation='interactive')['total']
    assert proposed<=cap-margin
    assert 'not a new hard runtime limit' in recommendation['note']
    if changes['context']<w.struct.n_ctx:
        assert forecast(w,changes['context']+1,operation='interactive')['total']>cap-margin
    if resident:return  # Physical upper bound does not promise post-unload availability.
    # A fresh Auto admission still passes after availability drops by half the buffer.
    dipped=dict(sample,available=sample['available']-margin//2)
    policy_for(values,dipped['total'],dipped['available']).preflight(dipped,proposed)
    configured=copy.deepcopy(settings);configured['values'].update(changes)
    initial=descriptor(w,changes['context'])
    assert admission(configured,initial,sample)[1]['required']==proposed
    # Scalar equivalent of one dense 4096x4096 FP32 patch: 64 MiB storage,
    # 192 MiB conservative patch allowance. No matrix is allocated here.
    w.patches=[SimpleNamespace(meta=lambda:{'stored':4096**2})]
    updated=descriptor(w,changes['context'])
    assert updated['forecast']['total']-initial['forecast']['total']==192*1024**2
    resident_sample=dict(sample,available=sample['available']-proposed,tree_footprint=proposed)
    loaded=dict(initial,runner=True)
    assert admission(configured,updated,resident_sample,loaded,sample)[1]['new_bytes']==192*1024**2


def test_cap_smaller_than_recommendation_buffer_has_no_preset(tmp_path):
    settings=active(tmp_path);w=fake_ws()
    result=capacity_recommendation(settings,w,machine(),overrides={'budget_gib':.25,'ceiling_gib':2})
    assert result['changes'] is None and result['status']=='blocked'
    assert result['cap']==GiB//4 and result['margin']==GiB//2 and result['target']==-GiB//4
    assert 'cannot accommodate the recommendation buffer' in result['note']
