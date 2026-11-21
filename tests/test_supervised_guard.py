"""Decision-gate tests use fake workspace metadata; never load a large model."""
import json
import os
import time
from types import SimpleNamespace

import pytest
from wsp import memory_guard as g


@pytest.fixture
def permit(tmp_path):
    p = dict(supervised=True,approval_message='synthetic test approval',base_sha256=g.SUPERVISED_SHA256,
             parameters=g.SUPERVISED_PARAMETERS,expires=time.time()+120,context=64,new_tokens=4,
             budget_gib=24,ceiling_gib=26,forecast_gib=23,timeout=600,allow_derived_exports=True)
    path=tmp_path/'permit.json';path.write_text(json.dumps(p))
    return p,path


@pytest.mark.parametrize('key,value',[('supervised',False),('base_sha256','other'),('parameters',27),
    ('expires',0),('context',129),('new_tokens',9),('budget_gib',27),('forecast_gib',25),
    ('timeout',901),('approval_message',''),('allow_derived_exports','true')])
def test_permit_fails_closed(permit,key,value):
    p,path=permit;p[key]=value;path.write_text(json.dumps(p))
    with pytest.raises(g.GuardError,match='permit invalid'):g.supervised_permit(path)


def test_permit_exact_base_and_explicit_derived_identity(permit,tmp_path):
    p,path=permit
    ws=SimpleNamespace(kind='gguf',n_params=lambda:g.SUPERVISED_PARAMETERS,
        meta={'base_sha256':g.SUPERVISED_SHA256},store=SimpleNamespace(path=tmp_path/'export.gguf'))
    assert g.permit_matches_workspace(p,ws)
    ws.meta['base_sha256']='derived'
    assert not g.permit_matches_workspace(p,ws)
    sidecar=tmp_path/'export.gguf.json'
    sidecar.write_text(json.dumps(dict(format='standalone-gguf-same-scheme',base_sha256=g.SUPERVISED_SHA256,artifact_sha256='derived')))
    assert g.permit_matches_workspace(p,ws)
    p['allow_derived_exports']=False
    assert not g.permit_matches_workspace(p,ws)


def test_lease_gate_reservation_and_no_double_subtraction(permit,tmp_path,monkeypatch):
    p,path=permit;_,digest=g.supervised_permit(path)
    lease=tmp_path/'lease.json'
    lease.write_text(json.dumps(dict(time=time.time(),pgid=os.getpgrp(),ok=True,
        forecast_bytes=23*g.GiB,supervised_permit_sha256=digest)))
    monkeypatch.setenv('WSP_GUARD_LEASE',str(lease))
    ws=SimpleNamespace(kind='gguf',n_params=lambda:g.SUPERVISED_PARAMETERS,meta={'base_sha256':g.SUPERVISED_SHA256})
    with pytest.raises(g.GuardError,match='below 9B'):g.require_lease(ws)
    monkeypatch.setenv('WSP_SUPERVISED_PERMIT',str(path))
    monkeypatch.setattr(g,'forecast',lambda ws,context:dict(total=23*g.GiB))
    monkeypatch.setattr(g,'snapshot',lambda:dict(available=30*g.GiB,pressure=1))
    g.require_lease(ws) # active allocation already reflected in current headroom
    with pytest.raises(g.GuardError,match='25 GiB'):g.require_lease(ws,loading=True)
    monkeypatch.setattr(g,'snapshot',lambda:dict(available=50*g.GiB,pressure=1))
    g.require_lease(ws,loading=True)
    monkeypatch.setattr(g,'forecast',lambda ws,context:dict(total=24*g.GiB))
    with pytest.raises(g.GuardError,match='reservation'):g.require_lease(ws)
    p['context']=63;path.write_text(json.dumps(p))
    with pytest.raises(g.GuardError,match='does not cover'):g.require_lease(ws)


def test_mismatched_invocation_refuses_before_spawn(permit,tmp_path):
    _,path=permit
    with pytest.raises(g.GuardError,match='forecast/timeout'):
        g.supervise(['never-run'],tmp_path/'log.jsonl',22*g.GiB,600,permit_path=path)


def test_supervised_policy_stops_any_swap_or_compression_growth(permit):
    p,_=permit;policy=g.supervised_policy(p)
    baseline=dict(available=50*g.GiB,pressure=1,tree_rss=0,tree_footprint=0,swap_used=0,swap_out=0,compressed=100)
    assert policy.violation({**baseline,'compressed':101},baseline)=='material compression growth'
    assert policy.violation({**baseline,'swap_used':1},baseline)=='material swap growth'
    assert policy.floor==20*g.GiB and policy.preferred==25*g.GiB


def test_external_stage_deadline_is_fail_closed():
    g.check_stage(dict(name='smoke',deadline=60),59)
    with pytest.raises(g.GuardError,match='Stage wall-clock timeout'):
        g.check_stage(dict(name='smoke',deadline=60),61)
    for bad in (float('nan'),float('inf'),'60'):
        with pytest.raises(g.GuardError,match='Malformed'):
            g.check_stage(dict(name='smoke',deadline=bad),1)


def test_refused_preflight_records_snapshot_and_never_spawns(tmp_path,monkeypatch):
    sample=dict(available=30*g.GiB,pressure=1)
    monkeypatch.setattr(g,'snapshot',lambda:sample)
    monkeypatch.setattr(g.subprocess,'Popen',lambda *a,**k:pytest.fail('refused preflight spawned a child'))
    log=tmp_path/'guard.jsonl'
    with pytest.raises(g.GuardError,match='25 GiB'):
        g.supervise(['never-run'],log,8*g.GiB)
    event=json.loads(log.read_text())
    assert event['baseline']==sample and event['child_spawned'] is False
    with pytest.raises(g.GuardError,match='latched'):
        g.supervise(['never-run'],log,8*g.GiB)


def test_mixed_candidate_requires_explicit_exact_hash(permit,tmp_path):
    p,path=permit;digest='a'*64
    ws=SimpleNamespace(kind='gguf',n_params=lambda:g.SUPERVISED_PARAMETERS,
        meta={'base_sha256':digest},store=SimpleNamespace(path=tmp_path/'mixed.gguf'))
    (tmp_path/'mixed.gguf.json').write_text(json.dumps(dict(format='standalone-gguf-mixed',
        base_sha256=g.SUPERVISED_SHA256,artifact_sha256=digest)))
    assert not g.permit_matches_workspace(p,ws)
    p['derived_sha256']=[digest];assert g.permit_matches_workspace(p,ws)
    for invalid in ('a'*64,['not a hash'],[None]):
        p['derived_sha256']=invalid;path.write_text(json.dumps(p))
        with pytest.raises(g.GuardError,match='derived_sha256'):g.supervised_permit(path)


def test_cleanup_never_signals_reused_pid_or_departed_group(monkeypatch):
    known={123:1.0};sent=[]
    p=SimpleNamespace(create_time=lambda:2.0,terminate=lambda:sent.append('term'),kill=lambda:sent.append('kill'))
    monkeypatch.setattr(g.psutil,'Process',lambda pid:p)
    monkeypatch.setattr(g.os,'killpg',lambda *a:sent.append('group'))
    child=SimpleNamespace(pid=123,wait=lambda **k:0)
    errors=[]
    assert g.stop_group(child,known,errors)==[]
    assert sent==[] and errors==[]


def test_unknown_cleanup_identity_fails_closed(monkeypatch):
    def denied(pid):raise g.psutil.AccessDenied(pid)
    monkeypatch.setattr(g.psutil,'Process',denied)
    errors=[]
    assert g.owned_remaining({123:1.0},errors)==[123]
    assert errors[0]['operation']=='verify'


def test_unexpected_cleanup_error_preserves_teardown_and_latch(tmp_path,monkeypatch):
    import sys
    sample=dict(available=64*g.GiB,pressure=1,swap_used=0,swap_out=0,compressed=0)
    monkeypatch.setattr(g,'snapshot',lambda:sample)
    real_stop=g.stop_group
    def stop_then_error(child,known,errors):
        assert real_stop(child,known,errors)==[]
        raise PermissionError('synthetic cleanup bookkeeping failure')
    monkeypatch.setattr(g,'stop_group',stop_then_error)
    log=tmp_path/'guard.jsonl'
    with pytest.raises(g.GuardError,match='cleanup error'):
        g.supervise([sys.executable,'-c','pass'],log,g.GiB,timeout=10)
    last=json.loads(log.read_text().splitlines()[-1])
    assert last['event']=='teardown' and last['cleanup_complete'] and last['remaining']==[]
    assert last['cleanup_errors'][-1]['operation']=='stop_group'
    assert json.loads((tmp_path/'GUARD_FAILED.json').read_text())==last
