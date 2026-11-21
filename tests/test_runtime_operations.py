"""Operation admission and client lifecycle using scalar fakes; no models/native calls."""
import copy
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from wsp.memory_guard import GiB, Policy
from wsp.runtime_config import DEFAULTS
from wsp.runtime_events import RuntimeRefusal, event
from wsp.runtime_policy import (admission, measured_total, observations, Escalation,
    OPERATION_COMPONENTS, ABORT_GRACE_SECONDS, UNLOAD_GRACE_SECONDS, GPU_STALE_GRACE_SECONDS)
from wsp.runtime_supervisor import RuntimeClient, write


def settings(**changes):
    return dict(values=dict(DEFAULTS, backend='cpu', floor_gib=20, preferred_gib=25,
                            **changes), platform='Linux')


def sample(available=70, measured=0):
    return dict(total=128*GiB, available=int(available*GiB), pressure=1,
                tree_rss=int(measured*GiB), tree_footprint=int(measured*GiB),
                swap_used=0, swap_out=0, compressed=0)


def descriptor(total=40, weights=23, patches=0, working=None):
    return dict(runner=True, forecast=dict(total=int(total*GiB), weights=int(weights*GiB),
                patches=int(patches*GiB)), working=working or {})


def test_loaded_repeated_operations_do_not_recharge_resident_private_or_file_pages():
    active=descriptor();request=descriptor(working={'kv':5*GiB})
    baseline=sample(available=70)
    # Availability fell by 20 GiB, much more than the preset margin. Charging the
    # full 40 GiB again would reject, but every new 5 GiB operation is admissible.
    for _ in range(5):
        policy,decision=admission(settings(),request,sample(available=50,measured=30),active,baseline)
        assert decision['new_bytes']==5*GiB and decision['available']==50*GiB
        assert policy.budget==58*GiB  # max(F40, measured30 + new5 + refaultW23)
        active=copy.deepcopy(request)
    with pytest.raises(RuntimeRefusal):
        admission(settings(),request,sample(available=29,measured=90),active,baseline)


def test_a_worker_hosted_servers_base_is_outside_the_loaded_forecast_cap():
    # Server base B=4 GiB outside the forecast F=40 GiB.
    active=descriptor();request=descriptor(working={'kv':GiB})
    # measured 5 (server 4 + worker 1) + new 1 + weights 23 = 29 < F=40 < B+F=44.
    policy,decision=admission(settings(),request,sample(measured=5),active,outside=4*GiB)
    assert policy.budget==44*GiB and decision['outside']==4*GiB and decision['required']==40*GiB
    # Without it (in-process runner) the cap is PR A's: max(F, measured+new+weights).
    policy,decision=admission(settings(),request,sample(measured=5),active)
    assert policy.budget==40*GiB and decision['outside']==0
    # The larger measured term still wins, and static caps still bound the result.
    assert admission(settings(),request,sample(measured=30),active,outside=4*GiB)[0].budget==54*GiB
    assert admission(settings(budget_gib=42),request,sample(measured=5),active,outside=4*GiB)[0].budget==42*GiB
    with pytest.raises(RuntimeRefusal):admission(settings(budget_gib=39),request,sample(measured=5),active,outside=4*GiB)


def test_a_worker_hosted_cold_loads_running_limit_excludes_the_server_base():
    # Auto budget: available 70 - preferred 25 = 45 GiB, and available already excludes the server.
    request=descriptor(total=40)
    plain,decision=admission(settings(),request,sample(measured=4))
    policy,hosted=admission(settings(),request,sample(measured=4),outside=4*GiB)
    assert plain.budget==45*GiB and policy.budget==49*GiB and hosted['outside']==4*GiB
    assert policy==plain.__class__(**{**decision['policy'],'budget':49*GiB}) and hosted['new_bytes']==decision['new_bytes']==40*GiB
    # Admission itself is unchanged: the checks use the unmodified budget and availability.
    for kwargs in ({},{'outside':4*GiB}):
        with pytest.raises(RuntimeRefusal):admission(settings(),descriptor(total=46),sample(measured=4),**kwargs)
    # The static caps still bound the whole application: an explicit budget, the ceiling, total - preferred.
    assert admission(settings(budget_gib=42),request,sample(measured=4),outside=4*GiB)[0].budget==42*GiB
    assert admission(settings(ceiling_gib=47),request,sample(measured=4),outside=4*GiB)[0].budget==47*GiB
    assert admission(settings(),request,sample(available=120,measured=4),outside=10*GiB)[0].budget==103*GiB


def test_all_request_components_and_only_positive_patch_growth_are_charged():
    expected={'kv','kv_append','activations','logits','eager_attention','hybrid_workspace',
              'retained_activations','retained_attention','recurrent','convolution'}
    assert set(OPERATION_COMPONENTS)==expected
    working={name:(i+1)*1024**2 for i,name in enumerate(sorted(expected))}
    working.update(weights=100*GiB,runtime=100*GiB,temporary=100*GiB,patches=100*GiB,total=1000*GiB)
    active=descriptor(total=41,patches=1)
    request=descriptor(total=42,patches=2,working=working)
    _,decision=admission(settings(),request,sample(),active)
    assert decision['new_bytes']==sum(working[k] for k in expected)+GiB
    _,smaller=admission(settings(),descriptor(total=40,patches=0,working=working),sample(),active)
    assert smaller['new_bytes']==sum(working[k] for k in expected)


def test_cold_and_loaded_refusal_retry_use_new_available_sample_without_mutating_state():
    cfg=settings();active=descriptor();before=copy.deepcopy(active)
    request=descriptor(working={'kv':5*GiB})
    with pytest.raises(RuntimeRefusal) as exc:admission(cfg,request,sample(available=29),active)
    assert exc.value.record['code']=='forecast_exceeds_budget'
    assert exc.value.record['action']=='refused' and active==before
    assert admission(cfg,request,sample(available=31),active)[1]['new_bytes']==5*GiB
    with pytest.raises(RuntimeRefusal):admission(cfg,request,sample(available=64))
    assert admission(cfg,request,sample(available=66))[1]['new_bytes']==40*GiB


@pytest.mark.parametrize('caps,total,measured', [({'budget_gib':39},40,0),
    ({'ceiling_gib':39},40,0), ({},104,0), ({'budget_gib':50},40,51)])
def test_known_static_caps_and_already_unsafe_measured_memory_refuse(caps,total,measured):
    with pytest.raises(RuntimeRefusal):
        admission(settings(**caps),descriptor(total=total),sample(measured=measured),descriptor())


def test_process_bound_allows_refaulted_weights_but_never_exceeds_static_caps():
    active=descriptor();request=descriptor(working={'kv':5*GiB})
    policy,decision=admission(settings(budget_gib=55),request,sample(measured=30),active)
    assert policy.budget==55*GiB and decision['static_cap']==55*GiB
    assert decision['available']==70*GiB and decision['new_bytes']==5*GiB
    assert measured_total(dict(tree_rss=1,tree_footprint=2,gpu={'metal_allocated':3,'mps_driver':4}))==4
    assert measured_total(dict(tree_rss=5,tree_footprint=2,gpu={'metal_allocated':None,'mps_driver':None}))==5


def test_normal_pressure_growth_warns_but_strict_acceptance_policy_still_aborts():
    policy=Policy(budget=100*GiB,ceiling=103*GiB)
    baseline=sample();current=sample();current.update(swap_used=65*1024**2,compressed=2*GiB)
    issue,warnings=observations(policy,current,baseline)
    assert issue is None and {x['code'] for x in warnings}=={'swap_growth','compression_growth'}
    assert all(x['action']=='warning' and x['recommendations'] for x in warnings)
    assert policy.violation(current,baseline) is not None
    current['pressure']=2
    issue,warnings=observations(policy,current,baseline)
    assert issue['code']=='memory_pressure' and issue['action']=='abort'


def test_abort_deadline_stays_latched_after_symptom_clears_or_gpu_goes_stale():
    issue=event('memory_pressure','abort','pressure');ladder=Escalation()
    assert ladder.observe(issue,True,0)=='abort'
    assert ladder.observe(None,True,ABORT_GRACE_SECONDS-1)=='abort'
    assert ladder.observe(event('gpu_telemetry_stale','abort','stale'),True,ABORT_GRACE_SECONDS,gpu_stale=True)=='unload'
    assert ladder.observe(None,False,ABORT_GRACE_SECONDS+UNLOAD_GRACE_SECONDS)=='kill_worker'
    ladder.reset();assert ladder.observe(None,False,100) is None


def test_idle_unload_requires_ack_even_after_pressure_recovers():
    ladder=Escalation();issue=event('headroom_floor','unload','floor')
    assert ladder.observe(issue,False,0)=='blocked'
    assert ladder.observe(issue,False,3)=='unload'
    assert ladder.observe(None,False,3+UNLOAD_GRACE_SECONDS)=='kill_worker'


def test_gpu_grace_blocks_new_work_then_aborts_running_work():
    issue=event('gpu_telemetry_stale','abort','stale');ladder=Escalation()
    assert ladder.observe(issue,False,0,gpu_stale=True)=='blocked'
    assert ladder.observe(None,False,1) is None
    assert ladder.observe(issue,True,2,gpu_stale=True)=='blocked'
    assert ladder.observe(issue,True,2+GPU_STALE_GRACE_SECONDS,gpu_stale=True)=='abort'


def test_continued_runaway_growth_escalates_to_kill_worker():
    issue=event('runaway_growth','abort','runaway');ladder=Escalation()
    assert ladder.observe(issue,True,0)=='abort'
    assert ladder.observe(issue,True,1)=='kill_worker'


def test_admission_records_the_admitted_footprint_envelope():
    baseline=sample(measured=1)
    _,cold=admission(settings(),dict(descriptor(total=40),runner=True),sample(measured=1))
    assert cold['envelope']==GiB+40*GiB  # pre-model tree + the whole cold forecast
    _,reserved=admission(settings(reservation_gib=42),descriptor(total=40),sample(measured=1))
    assert reserved['envelope']==GiB+42*GiB
    working={'kv':3*GiB,'activations':GiB,'weights':100*GiB}  # only operation components count
    _,loaded=admission(settings(),descriptor(total=41,patches=1,working=working),sample(measured=30),descriptor(total=40),baseline)
    assert loaded['new_bytes']==5*GiB and loaded['envelope']==GiB+41*GiB+4*GiB


def test_runaway_growth_is_bounded_by_the_admitted_envelope():
    policy=Policy(floor=GiB,preferred=2*GiB,budget=100*GiB,ceiling=100*GiB)
    baseline=sample(measured=1);previous=sample(measured=10);grown=sample(measured=15)
    # Unchanged without an envelope (no running operation): 5 GiB in one sample is runaway.
    assert observations(policy,grown,baseline,previous)[0]['code']=='runaway_growth'
    # A load faulting in admitted weights below its envelope is not runaway.
    assert observations(policy,grown,baseline,previous,envelope=20*GiB)[0] is None
    assert observations(policy,grown,baseline,previous,envelope=15*GiB)[0] is None
    issue=observations(policy,grown,baseline,previous,envelope=12*GiB)[0]
    assert issue['code']=='runaway_growth' and issue['action']=='abort' and issue['measured']==5*GiB
    # The budget check still precedes it, envelope or not.
    over=sample(measured=101)
    assert observations(policy,over,baseline,grown,envelope=200*GiB)[0]['code']=='runtime_budget'
    # Growth that continues above the envelope still escalates to step 4 (kill_worker).
    ladder=Escalation();envelope=12*GiB;samples=[sample(measured=m) for m in (10,15,20)]
    actions=[ladder.observe(observations(policy,now,baseline,before,envelope=envelope)[0],True,t)
             for t,(before,now) in enumerate(zip(samples,samples[1:]))]
    assert actions==['abort','kill_worker']


def test_begin_timeout_sends_abandon_finish(tmp_path,monkeypatch):
    import wsp.runtime_supervisor as rs
    client,ws,calls,_=client_fixture(tmp_path,monkeypatch)
    timeouts=[]
    def request(action,value=None,timeout=None):
        calls.append((action,copy.deepcopy(value)));timeouts.append(timeout)
        if action=='begin':
            refusal=rs.unavailable('no reply');refusal.timed_out=True;raise refusal
    monkeypatch.setattr(client,'_request',request)
    with pytest.raises(RuntimeRefusal):
        with client.operation(ws,'generate',tokens=32):pytest.fail('timed-out begin executed')
    begin=next(v for a,v in calls if a=='begin');finish=calls[-1]
    assert finish[0]=='finish' and finish[1]['operation_id']==begin['operation_id']
    assert timeouts[-1]==rs.ABANDON_TIMEOUT_SECONDS  # best-effort, on its own short bound
    assert finish[1]['abandoned'] is True and finish[1]['success'] is False and finish[1]['actual']
    assert client.operation_id is None
    # A refusal the supervisor delivered is not a timeout: nothing to abandon.
    calls.clear()
    def refuse(action,value=None):
        calls.append((action,value))
        if action=='begin':raise RuntimeRefusal(event('forecast_exceeds_budget','refused','no room'))
    monkeypatch.setattr(client,'_request',refuse)
    with pytest.raises(RuntimeRefusal):
        with client.operation(ws,'generate',tokens=32):pass
    assert [a for a,_ in calls if a=='finish']==[]


def test_request_timeout_is_marked_for_abandon(tmp_path,monkeypatch):
    import wsp.runtime_supervisor as rs
    monkeypatch.setattr(rs,'REQUEST_TIMEOUT_SECONDS',.05)
    with pytest.raises(RuntimeRefusal) as exc:RuntimeClient(tmp_path,settings())._request('begin',{})
    assert exc.value.timed_out and exc.value.record['code']=='watchdog_unavailable'


def test_constant_token_check_never_forecasts_or_scans_patches(tmp_path,monkeypatch):
    import wsp.runtime_supervisor as rs
    client=RuntimeClient(tmp_path,settings());client.operation_id='op';client.operation_version=3;client.binding='valid'
    write(tmp_path/'lease.json',dict(ok=True,binding='valid',pid=os.getpid(),time=time.time()))
    def forbidden(*args,**kwargs):raise AssertionError('token checkpoint inspected model/patch metadata')
    monkeypatch.setattr(client,'_forecast',forbidden);monkeypatch.setattr(rs,'descriptor',forbidden)
    class WS:
        version=3
        @property
        def patches(self):return forbidden()
        @property
        def store(self):return forbidden()
    for _ in range(100):client.require(WS())
    write(tmp_path/'lease.json',dict(ok=False,event=event('headroom_floor','abort','too low')))
    with pytest.raises(RuntimeRefusal,match='too low'):client.require(WS())


def client_fixture(tmp_path,monkeypatch):
    import wsp.runtime_supervisor as rs
    import wsp.memory_guard as mg
    ws=SimpleNamespace(lock=threading.RLock(),version=3,kind='gguf',_runner=object(),_metadata_only=False)
    f=dict(total=40*GiB,weights=23*GiB,patches=0)
    client=RuntimeClient(tmp_path,settings());calls=[]
    def fake_descriptor(ws,context,loading=False,*args,forecast_value=None,**kwargs):
        return dict(path='scalar-only',stat=[1,2,3,4],context=context,workspace_version=ws.version,
                    runner=loading or ws._runner is not None,forecast=dict(forecast_value or f))
    monkeypatch.setattr(rs,'descriptor',fake_descriptor)
    monkeypatch.setattr(mg,'forecast',lambda ws,tokens,operation='interactive':{'kv':tokens*1024})
    monkeypatch.setattr(client,'attach',lambda ws:None)
    monkeypatch.setattr(client,'_forecast',lambda ws:dict(f))
    monkeypatch.setattr(client,'_check',lambda binding:None)
    monkeypatch.setattr(client,'_request',lambda action,value=None:calls.append((action,copy.deepcopy(value))))
    return client,ws,calls,f


def test_finish_preserves_resident_ownership_for_later_release(tmp_path,monkeypatch):
    client,ws,calls,_=client_fixture(tmp_path,monkeypatch)
    with client.operation(ws,'generate',tokens=32):assert client.operation_id and client.reserved
    assert client.operation_id is None and client.reserved and client.binding
    client.release()
    assert [x[0] for x in calls]==['bind','begin','finish','release']
    assert not client.reserved and client.binding is None


def test_nested_forward_has_one_owner_and_one_finish(tmp_path,monkeypatch):
    client,ws,calls,_=client_fixture(tmp_path,monkeypatch)
    with client.operation(ws,'trace',tokens=32,retained=True):
        owner=client.operation_id
        with client.operation(ws,'forward',tokens=16,retained=True):assert client.operation_id==owner
        assert client.operation_id==owner
    actions=[x[0] for x in calls]
    assert actions.count('begin')==actions.count('finish')==1 and actions.count('extend')==0


def test_refused_extension_preserves_patch_baseline_and_finish_reconciles_candidate(tmp_path,monkeypatch):
    client,ws,calls,_=client_fixture(tmp_path,monkeypatch)
    with client.operation(ws,'edit'):
        client.preflight_edit(ws,3);client.preflight_edit(ws,4,cumulative=True)
        assert client._prospective_stored==7 and calls[-1][1]['forecast']['patches']==7*12
        before=client.binding;request=client._request
        def refuse(action,value=None):
            if action=='extend':raise RuntimeRefusal(event('forecast_exceeds_budget','refused','no room'))
            return request(action,value)
        monkeypatch.setattr(client,'_request',refuse)
        with pytest.raises(RuntimeRefusal):client.preflight_edit(ws,10,cumulative=True)
        assert client._prospective_stored==7 and client.binding==before
    finish=next(value for action,value in calls if action=='finish')
    assert finish['actual']['forecast']['patches']==0 and client._prospective_stored==0


def test_mailbox_serializes_simultaneous_requests_by_nonce(tmp_path):
    client=RuntimeClient(tmp_path,settings());stop=threading.Event();seen=[]
    def supervisor():
        handled=None
        while not stop.wait(.001):
            try:request=json.loads((tmp_path/'request.json').read_text())
            except FileNotFoundError:continue
            if request['id']==handled:continue
            handled=request['id'];seen.append(request['action']);time.sleep(.04)
            write(tmp_path/'reply.json',dict(id=handled,ok=True))
    worker=threading.Thread(target=supervisor);worker.start()
    try:
        with ThreadPoolExecutor(2) as pool:answers=list(pool.map(client._request,['begin','dismiss']))
        assert len(answers)==2 and set(seen)=={'begin','dismiss'}
        assert answers[0]['id']!=answers[1]['id']
    finally:stop.set();worker.join(timeout=1)
    assert not worker.is_alive()


def test_cold_refusal_does_not_poison_workspace_runner_retry(monkeypatch):
    from wsp.workspace import Workspace
    import wsp.runner as runner_module
    ws=object.__new__(Workspace);ws.lock=threading.RLock();ws._runner=None;ws.run_error=None
    calls=[];successful=object()
    def fake_make_runner(actual):
        calls.append(actual)
        if len(calls)==1:raise RuntimeRefusal(event('forecast_exceeds_budget','refused','free memory and retry'))
        return successful
    monkeypatch.setattr(runner_module,'make_runner',fake_make_runner)
    with pytest.raises(RuntimeRefusal):_ = ws.runner
    assert ws.run_error is None and ws._runner is None
    assert ws.runner is successful and len(calls)==2


def test_revoked_begin_lease_still_finishes_and_releases_operation_ownership(tmp_path,monkeypatch):
    client,ws,calls,_=client_fixture(tmp_path,monkeypatch)
    def check(binding):
        if client.operation_id:
            raise RuntimeRefusal(event('headroom_floor','abort','revoked before first allocation'))
    monkeypatch.setattr(client,'_check',check)
    with pytest.raises(RuntimeRefusal,match='before first allocation'):
        with client.operation(ws,'generate',tokens=32):pytest.fail('revoked operation executed')
    assert [x[0] for x in calls].count('finish')==1
    assert client.operation_id is None and client.operation_owner is None


def test_committed_candidate_moves_to_actual_storage_without_double_count(tmp_path,monkeypatch):
    client,ws,calls,f=client_fixture(tmp_path,monkeypatch)
    with client.operation(ws,'edit'):
        client.preflight_edit(ws,7)
        f['patches']=7*12;f['total']+=7*12;ws.version+=1
        client.mutated(ws)
        assert client._prospective_stored==0 and client.operation_version==ws.version
        client.require(ws)
    actual=next(value['actual'] for action,value in calls if action=='finish')
    assert actual['forecast']['patches']==7*12
    with client.operation(ws,'edit'):client.preflight_edit(ws,2)
    latest=next(value for action,value in reversed(calls) if action=='extend')
    assert latest['forecast']['patches']==9*12


def test_generation_interruption_preserves_tokens_from_actual_decode_loop(monkeypatch):
    import torch
    import wsp.gguf_runner as gguf_runner
    # Bypass construction entirely: only the generation loop runs on scalar logits.
    r=object.__new__(gguf_runner.GGUFRunner)
    r.lock=threading.RLock();r.max_ctx=32;r.device='cpu'
    r.ws=SimpleNamespace(hf_config={})
    r.use=lambda *args:None
    r.forward_options=lambda tokens:{}
    r._prefill=lambda ids:(torch.tensor([0.,1.]),object())
    calls=[]
    def forward(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(logits=torch.tensor([[[0.,1.]]]),past_key_values=object())
    r.model=forward
    checks=[]
    def checkpoint(ws):
        checks.append(None)
        if len(checks)==3:raise RuntimeRefusal(event('headroom_floor','abort','synthetic interruption'))
    monkeypatch.setattr(gguf_runner,'require_lease',checkpoint)
    with pytest.raises(RuntimeRefusal) as exc:r.generate([2,3],max_new_tokens=3)
    assert exc.value.tokens==[1] and exc.value.record['code']=='headroom_floor'
    assert len(calls)==1 and len(checks)==3
    assert r.last_generation['prompt_tokens']==2
    assert r.last_generation['cache_scope']=='single generation call; discarded'


def refusing_workspace():
    from contextlib import nullcontext
    calls=[]
    class Store:
        def __contains__(self,name):return name=='weight'
        def numel(self,name):return 4
    def refuse(ws,stored,scratch=0,**kwargs):
        calls.append((stored,scratch))
        raise RuntimeRefusal(event('forecast_exceeds_budget','refused','edit admission refused'))
    def read(*args,**kwargs):raise AssertionError('read before edit admission')
    client=SimpleNamespace(preflight_edit=refuse,operation=lambda *a,**k:nullcontext())
    ws=SimpleNamespace(_runtime_client=client,lock=threading.RLock(),store=Store(),
                       shape=lambda name:(2,2),editable=lambda name:True,read=read,
                       patches=[],version=0)
    return ws,calls


def test_interpreter_write_refuses_before_first_weight_read_or_delta_allocation(monkeypatch):
    from wsp.workspace import Ref
    from wsp.wscript import Interp
    import wsp.wscript as script_module
    ws,calls=refusing_workspace();interp=object.__new__(Interp);interp.ws=ws;interp.extra={}
    def allocation(*a,**k):pytest.fail('tensor allocation before edit admission')
    monkeypatch.setattr(script_module.torch,'as_tensor',allocation)
    with pytest.raises(RuntimeRefusal):interp.write(Ref(ws,'weight'),object())
    assert calls==[(4,48)] and interp.extra=={} and ws.patches==[] and ws.version==0


@pytest.mark.parametrize('kind',['block','lowrank'])
def test_reference_clone_refuses_before_clone_or_layout_conversion(monkeypatch,kind):
    from wsp.workspace import Ref
    import wsp.workspace as workspace_module
    ws,calls=refusing_workspace();ref=Ref(ws,'weight')
    def clone(*a,**k):pytest.fail('delta clone before edit admission')
    monkeypatch.setattr(workspace_module.D,'block',clone)
    monkeypatch.setattr(workspace_module.D,'lowrank',clone)
    class Factor:
        def numel(self):return 2
        def dim(self):return clone()
        def expand(self,*a):return clone()
    with pytest.raises(RuntimeRefusal):
        if kind=='block':ref.piece(Factor())
        else:ref.lowrank_piece(Factor(),Factor())
    assert calls==[(4,0)] and ws.patches==[]


def test_edit_files_refuse_before_array_read_and_leave_files_unchanged(tmp_path,monkeypatch):
    import wsp.edits as edits
    ws,calls=refusing_workspace();path=tmp_path/'weight.npy'
    path.write_bytes(b'not loaded: refusal precedes payload access')
    before=path.read_bytes()
    def load(*a,**k):pytest.fail('file array materialized before admission')
    monkeypatch.setattr(edits.np,'load',load)
    with pytest.raises(RuntimeRefusal):edits.read_files(ws,str(tmp_path))
    assert calls==[(4,48)] and ws.patches==[] and ws.version==0
    assert list(tmp_path.iterdir())==[path] and path.read_bytes()==before


def test_nested_cold_load_has_one_operation_and_refusal_is_retryable(tmp_path,monkeypatch):
    from wsp.workspace import Workspace
    import wsp.runner as runner_module
    client,_,calls,_=client_fixture(tmp_path,monkeypatch)
    ws=object.__new__(Workspace);ws.lock=threading.RLock();ws._runner=None;ws.run_error=None
    ws._runtime_client=client;ws.version=3;ws._metadata_only=False
    attempts=[];successful=object()
    def build(actual):
        attempts.append(actual)
        if len(attempts)==1:raise RuntimeRefusal(event('forecast_exceeds_budget','refused','transient cold refusal'))
        return successful
    monkeypatch.setattr(runner_module,'make_runner',build)
    with pytest.raises(RuntimeRefusal):
        with client.operation(ws,'generate',tokens=32):_ = ws.runner
    assert ws.run_error is None and ws._runner is None and client.operation_id is None
    with client.operation(ws,'generate',tokens=32):assert ws.runner is successful
    actions=[action for action,_ in calls]
    assert actions.count('begin')==actions.count('finish')==2
    assert actions.count('extend')==2 and len(attempts)==2


@pytest.mark.parametrize('code,healthy,elapsed,accepted',[
    ('headroom_floor',True,1,True),('headroom_floor',False,1,False),
    ('headroom_floor',True,ABORT_GRACE_SECONDS,False),('model_changed',True,1,False),
    ('runaway_growth',True,1,False)])
def test_only_timely_healthy_acknowledged_abort_keeps_model(code,healthy,elapsed,accepted):
    ladder=Escalation();ladder.observe(event(code,'abort','test'),True,0)
    assert ladder.acknowledge_finish(healthy,elapsed) is accepted
    assert (ladder.action is None) is accepted


@pytest.mark.parametrize('running',[True,False])
def test_model_change_does_not_wait_for_gpu_grace(running):
    ladder=Escalation()
    assert ladder.observe(event('model_changed','abort','source changed'),running,0,gpu_stale=True)==('abort' if running else 'unload')


def test_missing_source_is_typed_before_any_reservation(tmp_path,monkeypatch):
    client=RuntimeClient(tmp_path,settings())
    ws=SimpleNamespace(store=SimpleNamespace(path=str(tmp_path/'missing.gguf')))
    monkeypatch.setattr(client,'_forecast',lambda ws:{'total':1})
    monkeypatch.setattr(client,'_request',lambda *a,**k:pytest.fail('missing source must not reach admission'))
    with pytest.raises(RuntimeRefusal) as exc:client.require(ws)
    assert exc.value.record['code']=='model_changed' and exc.value.record['action']=='refused'


@pytest.mark.parametrize('expression',['where(c,a,b)','maximum(a,b)','minimum(a,b)','a < b','a @ b'])
def test_script_broadcast_refuses_before_materialized_result(monkeypatch,expression):
    import torch
    import wsp.wscript as module
    st=SimpleNamespace(d_vocab=1,n_ctx=1,d_model=1,n_heads=1,d_head=1,d_mlp=1,n_layers=1)
    ws=SimpleNamespace(struct=st,sem=object(),_runtime_client=object(),names=lambda:[])
    calls=[]
    def refuse(ws,stored=0,scratch=0,**kw):
        calls.append(scratch)
        raise RuntimeRefusal(event('forecast_exceeds_budget','refused','before result allocation'))
    monkeypatch.setattr(module,'edit_preflight',refuse)
    interp=module.Interp(ws)
    interp.locals.update(c=torch.empty((4096,1),dtype=torch.bool,device='meta'),
        a=torch.empty((4096,1),device='meta'),b=torch.empty((1,4096),device='meta'))
    with pytest.raises(RuntimeRefusal):interp.run('x = '+expression)
    assert calls==[4096**2*12] and 'x' not in interp.locals


def test_script_array_counts_before_conversion(monkeypatch):
    import wsp.wscript as module
    st=SimpleNamespace(d_vocab=1,n_ctx=1,d_model=1,n_heads=1,d_head=1,d_mlp=1,n_layers=1)
    ws=SimpleNamespace(struct=st,sem=object(),_runtime_client=object(),names=lambda:[])
    def refuse(*a,**kw):raise RuntimeRefusal(event('forecast_exceeds_budget','refused','before conversion'))
    monkeypatch.setattr(module,'edit_preflight',refuse)
    interp=module.Interp(ws)
    with pytest.raises(RuntimeRefusal):interp.run('x = tensor([[1,2],[3,4]])')
    assert 'x' not in interp.locals


def test_late_revocation_preserves_actual_durable_commit_result(tmp_path,monkeypatch):
    import torch
    from wsp.workspace import Workspace
    from wsp.delta import block
    from wsp.runtime_events import job_scope
    client,_,calls,_=client_fixture(tmp_path,monkeypatch)
    class Store:
        def __contains__(self,name):return name=='weight'
        def is_float(self,name):return True
        def shape(self,name):return (2,2)
    ws=Workspace(str(tmp_path/'workspace'),Store(),SimpleNamespace(),None,{'kind':'gguf'})
    ws._runtime_client=client
    def check(binding):
        if ws.patches:raise RuntimeRefusal(event('headroom_floor','abort','Late synthetic revoke'))
    monkeypatch.setattr(client,'_check',check)
    @job_scope('edit')
    def commit_and_report(ws):
        patch=ws.commit({'weight':[block((range(2),range(2)),torch.ones(2,2))]},'synthetic')
        report=dict(ok=True,committed=True,patch=patch.id,changed=[{'param':'weight'}])
        client.record_commit(ws,patch.id,report)
        return report
    result=commit_and_report(ws)
    assert result['committed'] and result['outcome']=='committed_interrupted'
    assert result['changed']==[{'param':'weight'}] and result['patch']==ws.patches[0].id
    assert 'already committed' in result['event']['message'] and client.last_event==result['event']
    assert len(list((tmp_path/'workspace'/'patches').iterdir()))==2
    assert len(ws.patches)==1 and ws.patch_stored==4
    assert next(v for a,v in calls if a=='finish')['success'] is False



def test_nested_cold_load_does_not_charge_existing_runtime_twice():
    from wsp.runtime_policy import extend_admission
    cfg=settings();current=sample(available=65.125,measured=1)
    initial=descriptor(total=40);initial['runner']=False
    policy,first=admission(cfg,initial,current)
    assert policy.budget==int(40.125*GiB)
    nested=descriptor(total=40)
    _,next_decision=admission(cfg,nested,current,initial)
    assert next_decision['loaded'] is False and next_decision['new_bytes']==40*GiB
    assert extend_admission(policy,next_decision,current)==policy
    # A fresh explicit extension may enlarge the cap, without double-charging M.
    _,larger=admission(cfg,descriptor(total=40.25),sample(available=70,measured=1),initial)
    assert extend_admission(policy,larger,sample(available=70,measured=1)).budget==int(40.25*GiB)


def test_consumable_scratch_grants_amortize_small_loop_and_recheck_retained_allocations(tmp_path,monkeypatch):
    from wsp.runtime_supervisor import SCRATCH_ADMISSION_QUANTUM
    client,ws,calls,_=client_fixture(tmp_path,monkeypatch)
    with client.operation(ws,'edit'):
        for _ in range(402):client.preflight_edit(ws,0,scratch=96)
        assert sum(a=='extend' for a,v in calls)==1
        assert client._scratch_remaining==SCRATCH_ADMISSION_QUANTUM-402*96
        # These scalar claims represent retained12MiB materializations, no tensors.
        for _ in range(10):client.preflight_edit(ws,0,scratch=12*1024**2)
        assert sum(a=='extend' for a,v in calls)==2
        before=client._scratch_remaining
        def refuse(*args,**kwargs):raise RuntimeRefusal(event('forecast_exceeds_budget','refused','no grant'))
        monkeypatch.setattr(client,'_request',refuse)
        with pytest.raises(RuntimeRefusal):client.preflight_edit(ws,0,scratch=SCRATCH_ADMISSION_QUANTUM+1)
        assert client._scratch_remaining==before
    assert client._scratch_remaining==0


def test_repeated_forward_admission_is_mode_and_version_scoped(tmp_path,monkeypatch):
    client,ws,calls,_=client_fixture(tmp_path,monkeypatch)
    with client.operation(ws,'generate',tokens=32):
        for _ in range(100):
            with client.operation(ws,'next',tokens=16):pass
        assert sum(a=='extend' for a,v in calls)==0
        for _ in range(100):
            with client.operation(ws,'forward',tokens=16):pass
        assert sum(a=='extend' for a,v in calls)==1
        with client.operation(ws,'forward',tokens=16,retained=True):pass
        assert sum(a=='extend' for a,v in calls)==2
        ws.version+=1;client.mutated(ws)
        with client.operation(ws,'forward',tokens=16,retained=True):pass
        assert sum(a=='extend' for a,v in calls)==3
    assert client._forward_allowances=={}


def test_name_capturing_forward_is_always_freshly_admitted(tmp_path,monkeypatch):
    client,ws,calls,_=client_fixture(tmp_path,monkeypatch)
    extends=lambda:sum(a=='extend' for a,v in calls)
    with client.operation(ws,'generate',tokens=32):
        # Hook-only sweeps of one size reuse a single admission.
        for _ in range(5):
            with client.operation(ws,'forward',tokens=16,retained=True):pass
        assert extends()==1
        # Captures by name can accumulate, so each one is admitted again.
        for _ in range(3):
            with client.operation(ws,'forward',tokens=16,retained=True,fresh=True):pass
        assert extends()==4


def test_admitted_edit_growth_invalidates_earlier_forward_admission(tmp_path,monkeypatch):
    from wsp.memory_guard import GiB
    client,ws,calls,_=client_fixture(tmp_path,monkeypatch)
    extends=lambda:sum(a=='extend' for a,v in calls)
    with client.operation(ws,'edit'):
        with client.operation(ws,'forward',tokens=4096):pass
        with client.operation(ws,'forward',tokens=4096):pass
        assert extends()==1
        client.preflight_edit(ws,int(2*GiB));grown=extends()
        with client.operation(ws,'forward',tokens=4096):pass
        assert extends()==grown+1                   # checked again after the growth
        for _ in range(20):client.preflight_edit(ws,0,scratch=60*1024**2)
        granted=extends()
        with client.operation(ws,'forward',tokens=4096):pass
        assert extends()==granted+1                 # and again after new scratch grants
        value=[v for a,v in calls if a=='extend'][-1]
        assert value['working']['activations']>=client._scratch_remaining>0


def test_script_logits_returns_a_charged_copy_not_a_view(tmp_path,monkeypatch):
    import torch
    from wsp.wscript import Interp
    client,ws,calls,_=client_fixture(tmp_path,monkeypatch)
    ws.struct=SimpleNamespace(d_vocab=1000,n_ctx=512,d_model=1,n_heads=1,d_head=1,d_mlp=1,n_layers=1)
    ws.sem=object();ws.names=lambda:[];ws._runtime_client=client
    ws.encode=lambda prompt:[1]*512
    ws.runner=SimpleNamespace(max_ctx=512,logits=lambda ids,variant,extra,tail=None:torch.zeros(len(ids),1000)[-(tail or len(ids)):])
    with client.operation(ws,'edit'):
        interp=Interp(ws)
        interp.run('kept = [logits("x") for i in range(20)]')
        kept=interp.locals['kept']
        assert all(v.shape==(1000,) and v.untyped_storage().nbytes()==1000*4 for v in kept)
        assert client._scratch_allowance-client._scratch_remaining>=20*1000*12 or sum(a=='extend' for a,v in calls)>1


def test_idle_flapping_condition_still_unloads():
    ladder=Escalation();issue=event('memory_pressure','abort','unsafe')
    assert ladder.observe(issue,False,0)=='blocked'
    assert ladder.observe(None,False,1) is None
    assert ladder.observe(issue,False,2)=='blocked'
    assert ladder.observe(None,False,3) is None
    assert ladder.observe(issue,False,4)=='unload'
    steady=Escalation()
    assert steady.observe(issue,False,0)=='blocked'
    for t in (1,2,3,4):assert steady.observe(None,False,t) is None
    assert steady.observe(issue,False,5)=='blocked'  # stayed clear long enough: debounce restarts
    started=Escalation()
    assert started.observe(issue,False,0)=='blocked'
    assert started.observe(None,True,1) is None        # an operation started: no carried-over debounce
    assert started.observe(issue,False,2)=='blocked'
    assert started.observe(issue,False,4.9)=='blocked'


def test_job_scope_marks_only_name_captures_fresh(monkeypatch):
    import threading
    from contextlib import contextmanager
    from types import SimpleNamespace
    from wsp.runtime_events import job_scope
    seen=[]
    class Client:
        @contextmanager
        def operation(self,ws,name,**kw):seen.append(kw);yield
    ws=SimpleNamespace(lock=threading.RLock(),_runtime_client=Client())
    class Runner:
        max_ctx=64
        def __init__(self):self.ws=ws
        @job_scope('forward',runner=True)
        def run(self,ids,names=None,hooks=None):return len(ids)
    r=Runner()
    r.run([1,2,3]);r.run([1,2,3],hooks={'x':None});r.run([1,2,3],names=['x'])
    assert [(k['retained'],k['fresh']) for k in seen]==[(False,False),(True,False),(True,True)]


def test_explicit_trace_extension_matches_direct_admission():
    from wsp.runtime_policy import extend_admission
    active=descriptor(total=8,weights=4.9)
    current=sample(available=60,measured=7.5)
    initial,_=admission(settings(),active,current,active)
    forward=descriptor(total=8,weights=4.9,working={'retained_attention':6*GiB})
    direct,decision=admission(settings(),forward,current,active)
    extended=extend_admission(initial,decision,current,weights=forward['forecast']['weights'])
    assert extended.budget==direct.budget==int(7.5*GiB)+6*GiB+int(4.9*GiB)
    assert extended.budget>initial.budget
    with pytest.raises(RuntimeRefusal) as exc:
        extend_admission(initial,decision,sample(available=30),weights=forward['forecast']['weights'])
    assert exc.value.record['code']=='operation_exceeds_budget'


@pytest.mark.parametrize('code',['memory_pressure','headroom_floor'])
def test_idle_transient_unsafe_blocks_then_recovers_without_unload(code):
    ladder=Escalation();issue=event(code,'abort','unsafe')
    assert ladder.observe(issue,False,10)=='blocked'
    assert ladder.observe(issue,False,12.9)=='blocked'
    assert ladder.observe(None,False,13) is None
    assert ladder.observe(issue,False,20)=='blocked'
    assert ladder.observe(issue,False,23)=='unload'
    immediate=Escalation()
    assert immediate.observe(event('model_changed','abort','changed'),False,0)=='unload'


def test_layer_cache_is_bounded_but_token_checks_are_unconditional(tmp_path,monkeypatch):
    import wsp.runtime_supervisor as rs
    client=RuntimeClient(tmp_path,settings());client.operation_id='op';client.operation_version=3;client.binding='valid'
    ws=SimpleNamespace(version=3);clock=[100.]
    monkeypatch.setattr(rs.time,'monotonic',lambda:clock[0])
    write(tmp_path/'lease.json',dict(ok=True,binding='valid',pid=os.getpid(),time=time.time()))
    client.require(ws)
    write(tmp_path/'lease.json',dict(ok=False,event=event('headroom_floor','abort','revoked')))
    clock[0]+=.049
    client.layer_checkpoint(ws)  # bounded extra layer delay is explicitly allowed
    with pytest.raises(RuntimeRefusal):client.require(ws)  # next token cannot reuse it
    with pytest.raises(RuntimeRefusal):client.layer_checkpoint(ws)  # failed read invalidates
    write(tmp_path/'lease.json',dict(ok=True,binding='valid',pid=os.getpid(),time=time.time()))
    client.require(ws)
    write(tmp_path/'lease.json',dict(ok=False,event=event('headroom_floor','abort','revoked')))
    clock[0]+=.051
    with pytest.raises(RuntimeRefusal):client.layer_checkpoint(ws)


def test_real_interpreter_small_arithmetic_loop_uses_one_scratch_grant(tmp_path,monkeypatch):
    from wsp.wscript import Interp
    client,ws,calls,_=client_fixture(tmp_path,monkeypatch)
    ws.struct=SimpleNamespace(d_vocab=1,n_ctx=1,d_model=1,n_heads=1,d_head=1,d_mlp=1,n_layers=1)
    ws.sem=object();ws.names=lambda:[];ws._runtime_client=client
    with client.operation(ws,'edit'):
        interp=Interp(ws)
        interp.run('v = ones(8)\nacc = zeros(8)\nfor i in range(100):\n    acc = acc + v * 0.5 - v / 3')
        assert interp.locals['acc'].shape==(8,)
    assert sum(action=='extend' for action,value in calls)==1
