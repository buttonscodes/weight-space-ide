import importlib.util
from pathlib import Path
import sys
import json
import threading

import pytest
import torch
from wsp.export_quality import Criteria, evaluate, select_verified
from wsp.export_jobs import ExportJobs


def evidence(gain=1.):
    base=[torch.tensor([2.,0.,-1.]),torch.tensor([0.,2.,1.])]
    overlay=[base[0]+torch.tensor([gain,0.,0.]),base[1].clone()]
    return base,overlay


def test_strict_gain_not_coarse_correctness():
    base,overlay=evidence();candidate=[base[0]+torch.tensor([.02,0.,0.]),base[1]]
    r=evaluate(base,overlay,candidate,0,1,criteria=Criteria(),chat_prompt_identical=True)
    assert base[0].argmax()==overlay[0].argmax()==candidate[0].argmax()
    assert r['validation_status']=='FAILED' and not r['checks']['gain_preserved']
    assert r['required_gain']==.9
    good=evaluate(base,overlay,overlay,0,1,criteria=Criteria(),chat_prompt_identical=True)
    assert select_verified([r,good]) is good
    with pytest.raises(ValueError,match='companion'):select_verified([r])


@pytest.mark.parametrize('gain,direction',[(0,1),(-1,-1)])
def test_zero_and_negative_direction(gain,direction):
    b,o=evidence(gain)
    r=evaluate(b,o,o,0,1,criteria=Criteria(direction=direction),chat_prompt_identical=True)
    assert r['validation_status']=='VERIFIED'
    if not gain:assert r['retention_ratio'] is None
    else:
        with pytest.raises(ValueError,match='direction'):evaluate(b,o,o,0,1,criteria=Criteria(),chat_prompt_identical=True)


def test_chat_top1_drift_and_logit_gates():
    b,o=evidence();c=[o[0],torch.tensor([3.,2.,1.])]
    r=evaluate(b,o,c,0,1,criteria=Criteria(),chat_prompt_identical=False)
    assert not r['checks']['unrelated_top1_preserved'] and not r['checks']['chat_prompt_identical']
    assert not r['checks']['preservation_drift'] and not r['checks']['max_logit_error']
    with pytest.raises(ValueError):Criteria(max_logit_error=float('nan'))


def test_job_status_never_equates_structure_with_behavior(tmp_path,monkeypatch):
    import wsp.export_jobs as jobs
    path=tmp_path/'output.gguf';path.write_bytes(b'tiny')
    monkeypatch.setattr(jobs,'preview_export',lambda *a:{'changed_tensor_count':1})
    monkeypatch.setattr(jobs,'export_mixed',lambda *a,**kw:{'path':str(path),'artifact_sha256':'abc','structural_status':'PASSED'})
    ws=type('WS',(),{'dir':str(tmp_path)})()
    m=ExportJobs();state=m.start(ws);m.jobs[state['id']]['thread'].join()
    assert m.get(state['id'])['validation_status']=='UNVALIDATED'
    Path(str(path)+'.validation.json').write_text(json.dumps({'artifact_sha256':'abc','validation_status':'FAILED'}))
    assert m.get(state['id'])['validation_status']=='FAILED'
    Path(str(path)+'.validation.json').write_text(json.dumps({'artifact_sha256':'other','validation_status':'VERIFIED'}))
    assert m.get(state['id'])['validation_status']=='UNVALIDATED'
    def fail(*a,**kw):raise ValueError('disk exhausted')
    monkeypatch.setattr(jobs,'export_mixed',fail)
    state=m.start(ws);m.jobs[state['id']]['thread'].join()
    assert m.get(state['id'])['status']=='FAILED'
    m.shutdown()
