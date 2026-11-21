import importlib.util
import json
from pathlib import Path

import pytest
from wsp.memory_guard import GuardError


@pytest.fixture
def flow(monkeypatch,tmp_path):
    scripts=Path(__file__).resolve().parents[1]/'scripts';monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('mixed_workflow',scripts/'validate_mixed_export.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    workspace=tmp_path/'workspace';workspace.mkdir()
    ws=type('WS',(),{'n_params':lambda self:1000,'meta':{'base_sha256':'base'}})()
    monkeypatch.setattr(mod.Workspace,'load',lambda _,**kw:ws)
    monkeypatch.setattr(mod,'preview_export',lambda ws,p:{'output_bytes':1024,'increase_bytes':512})
    monkeypatch.setattr(mod,'forecast',lambda ws,c:{'total':mod.GiB})
    args=type('Args',(),dict(workspace=str(workspace),out=str(tmp_path/'out'),precisions='Q8_0,F16,F32',
        total_seconds=1200,phase_seconds=60,max_disk_gib=1,permit=None))()
    return mod,args


def test_adaptive_reject_then_promote_after_teardown(flow):
    mod,a=flow;calls=[]
    def supervisor(command,log,*args,**kwargs):
        phase=command[command.index('--phase')+1];p=command[-1] if '--precision' in command else None
        calls.append((phase,p));log.parent.mkdir(parents=True)
        end={'event':'teardown','remaining':[],'reason':None,'returncode':3 if p=='Q8_0' and phase=='candidate' else 0}
        log.write_text(json.dumps(end)+'\n')
        if phase=='candidate':
            (Path(a.out)/f'{p}.gguf.validation.json').write_text(json.dumps({'validation_status':'FAILED' if p=='Q8_0' else 'VERIFIED'}))
            if p=='Q8_0':raise GuardError('semantic exit3')
    mod.run(a,supervisor)
    assert calls==[('baseline',None),('export','Q8_0'),('candidate','Q8_0'),('export','F16'),('candidate','F16')]
    assert json.loads((Path(a.out)/'result.json').read_text())['selected']=='F16'


@pytest.mark.parametrize('reason,remaining,code',[('pressure warning',[],3),(None,[123],3),(None,[],1)])
def test_safety_runtime_or_owned_child_stops_search(flow,reason,remaining,code):
    mod,a=flow;calls=[]
    def supervisor(command,log,*args,**kwargs):
        calls.append(command);log.parent.mkdir(parents=True)
        log.write_text(json.dumps({'event':'teardown','reason':reason,'remaining':remaining,'returncode':code})+'\n')
        raise GuardError('stop')
    with pytest.raises(GuardError):mod.run(a,supervisor)
    assert len(calls)==1 and (Path(a.out)/'failure.json').exists()


@pytest.mark.parametrize('precisions',['F32,Q8_0','Q8_0,Q8_0','Q4_K','SAME,Q8_0,F16,F32,F32'])
def test_candidate_order_and_count_bounded(flow,precisions):
    mod,a=flow;a.precisions=precisions
    with pytest.raises(ValueError,match='increasing candidates'):mod.run(a,lambda *a,**k:pytest.fail('spawned'))


def test_disk_bound_before_any_load(flow):
    mod,a=flow;a.max_disk_gib=1e-9
    with pytest.raises(ValueError,match='disk limit'):mod.run(a,lambda *a,**k:pytest.fail('spawned'))
