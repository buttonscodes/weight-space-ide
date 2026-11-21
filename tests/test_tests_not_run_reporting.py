"""Scalar suite reports across Session -> tools -> MCP; no runner or weights."""
import copy
import json
from types import SimpleNamespace
import pytest
from wsp.session import Session
from wsp import testsuite, tools
from wsp.mcp import call_tool
from wsp.runtime_events import RuntimeRefusal, event


def report():
    record=event('forecast_exceeds_budget','refused','Need fresh headroom.',30,25,['Free memory and retry.'])
    return dict(name='demo',passed=1,total=2,score=None,not_run=1,cases=[
        {'name':'completed','pass':True,'value':1,'got':'yes'},
        {'name':'blocked','pass':None,'status':'not_run','event':record,'value':None}])


def test_mcp_not_run_preserves_event_and_cases_without_caching_refusal(tmp_path,monkeypatch):
    session=Session(str(tmp_path));session.task['suite']='demo'
    session.ws=SimpleNamespace(can_run=True,run_error=None,calib={},keep={})
    calls=[];blocked=report()
    def run(*args,**kwargs):
        calls.append(None)
        if len(calls)==1:return copy.deepcopy(blocked)
        result=copy.deepcopy(blocked);result.update(passed=2,score=1.,not_run=0)
        result['cases'][1]={'name':'blocked','pass':True,'value':1,'got':'yes'}
        return result
    monkeypatch.setattr(testsuite,'run_suite',run)
    text,error=call_tool(session,'run_tests',{'show':'all'})
    assert not error
    payload=json.loads(text)
    assert payload['status']=='not_run' and payload['event']==blocked['cases'][1]['event']
    assert payload['report']['cases'][1]['pass'] is None
    assert 'NOT RUN' in payload['summary'] and 'FAIL' not in payload['summary']
    assert session.ws.calib=={}  # A transient refusal must not poison retry.
    text,error=call_tool(session,'run_tests',{'show':'all'})
    assert not error and '2/2 pass' in text and 'NOT RUN' not in text
    assert len(calls)==2 and session.ws.calib['tests:demo:current']['passed']==2


@pytest.mark.parametrize('show',['all','failures','summary'])
def test_tool_cold_refusal_keeps_structured_reason_without_fabricated_case_failures(show):
    record=report()['cases'][1]['event']
    def refuse(*a,**k):raise RuntimeRefusal(record)
    session=SimpleNamespace(tests=refuse,ws=SimpleNamespace(can_run=True))
    payload=tools.run_tests(session,show=show)
    assert payload['status']=='not_run' and payload['event']==record and payload['report'] is None
    assert payload['summary'].startswith('NOT RUN:')


def test_show_all_distinguishes_real_failure_from_not_run():
    result=report();result['cases'][0]['pass']=False;result['passed']=0
    session=SimpleNamespace(tests=lambda *a:result)
    payload=tools.run_tests(session,show='all')
    assert "FAIL 'completed'" in payload['summary']
    assert "NOT RUN 'blocked'" in payload['summary']
    assert "FAIL 'blocked'" not in payload['summary']


def test_mcp_edit_refusal_retains_typed_event_without_false_commit(monkeypatch):
    record=report()['cases'][1]['event']
    def refuse(*a,**kw):raise RuntimeRefusal(record)
    monkeypatch.setattr(tools,'call',refuse)
    session=SimpleNamespace(publish=lambda _:None)
    text,error=call_tool(session,'edit',{'script':'synthetic'})
    payload=json.loads(text)
    assert error and payload['status']=='not_run' and payload['event']==record
    assert 'committed' not in payload


def test_mcp_late_revoke_preserves_committed_patch_report(monkeypatch):
    record=report()['cases'][1]['event']
    committed=dict(ok=True,committed=True,patch='0001-test',outcome='committed_interrupted',event=record)
    def late(*a,**kw):
        exc=RuntimeRefusal(record);exc.committed=['0001-test'];exc.report=committed
        raise exc
    monkeypatch.setattr(tools,'call',late)
    text,error=call_tool(SimpleNamespace(publish=lambda _:None),'edit',{})
    assert not error and json.loads(text)==committed
