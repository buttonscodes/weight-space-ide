"""Real HTTP handlers, scalar workspace and fake runner; never model construction."""
from contextlib import contextmanager
from threading import RLock, Event, Thread
from types import SimpleNamespace
import pytest
import torch
from fastapi.testclient import TestClient
from test_settings import active
from wsp.runtime_events import RuntimeRefusal,event
from wsp.server import create_app
from wsp.session import Session


class Runtime:
    def __init__(self,settings):self.settings=settings;self.last=None;self.finishes=[]
    @contextmanager
    def operation(self,*args,**kwargs):
        try:yield
        except RuntimeRefusal as exc:self.last=exc.record;self.finishes.append('interrupted');raise
        else:self.last=None;self.finishes.append('complete')
    def status(self,ws):return dict(supervised=True,event=self.last,limits=self.settings['values'])
    def dismiss(self):self.last=None


def fixture(tmp_path, failure_variant='base'):
    sess=Session(str(tmp_path/'work'));runtime=Runtime(active(tmp_path));sess.runtime=runtime
    calls=[];failing=[True]
    def generate(ids,variant,extra=None,**kw):
        calls.append((variant,kw['max_new_tokens']))
        if failing[0] and variant==failure_variant:
            raise RuntimeRefusal(event('runtime_budget','abort','Synthetic stop',4,3,['Retry after freeing memory.']),tokens=[7,8])
        return [9]
    tok=SimpleNamespace(token_str=str,to_dict=lambda:{'mode':'word'},encode=lambda *a,**k:[1,2],decode=lambda ids:' '.join(map(str,ids)))
    runner=SimpleNamespace(next_logits=lambda *a:torch.zeros(3),chat_prompt=lambda *a,**k:'whole chat',generate=generate)
    ws=SimpleNamespace(kind='gguf',lock=RLock(),_runtime_client=runtime,struct=SimpleNamespace(n_ctx=512),can_run=True,
        tok=tok,runner=runner,encode=lambda _:[1,2],complete=lambda p,v,**kw:tok.decode(generate([1,2],v,**kw)))
    sess.ws=ws
    client=TestClient(create_app(sess,'http://127.0.0.1:41881'),base_url='http://127.0.0.1:41881',raise_server_exceptions=True)
    return client,runtime,calls,failing


@pytest.mark.parametrize('route,body',[('/api/run',{'prompt':'test'}),('/api/chat',{'messages':[{'role':'user','content':'test'}]})])
@pytest.mark.parametrize('variant',['base','current'])
def test_partial_output_stops_variants_and_persists_failure_until_retry(tmp_path,route,body,variant):
    client,rt,calls,failing=fixture(tmp_path,variant)
    limits=client.get('/api/runtime').json()['generation']
    assert limits['output_limit'] is None
    response=client.post(route,json={**body,'max_new_tokens':128})
    assert response.status_code==200,response.text
    data=response.json();assert data['interrupted'] and data['event']['code']=='runtime_budget'
    assert data[variant]['text']=='7 8' if route=='/api/run' else data[variant]=='7 8'
    # The edited model runs first; whichever variant is interrupted, nothing after it starts.
    assert [c[0] for c in calls]==(['current'] if variant=='current' else ['current','base'])
    assert rt.finishes==['interrupted'] and client.get('/api/runtime').json()['event']['code']=='runtime_budget'
    assert client.post('/api/runtime/dismiss',json={}).status_code==200
    assert client.get('/api/runtime').json()['event'] is None
    failing[0]=False;calls.clear()
    response=client.post(route,json={**body,'max_new_tokens':128})
    assert response.status_code==200 and not response.json().get('interrupted')
    assert calls==[('current',128),('base',128)] and rt.finishes[-1]=='complete'


@pytest.mark.parametrize('route,body',[('/api/run',{'prompt':'test'}),('/api/chat',{'messages':[{'role':'user','content':'test'}]})])
def test_begin_refusal_is_not_run_and_normal_exception_middleware(tmp_path,route,body):
    client,rt,calls,_=fixture(tmp_path)
    @contextmanager
    def refuse(*args,**kwargs):
        raise RuntimeRefusal(event('forecast_exceeds_budget','refused','No room before allocation'))
        yield
    rt.operation=refuse
    # Expected refusals must not escape Starlette as an application crash.
    response=client.post(route,json={**body,'max_new_tokens':128})
    assert response.status_code==409 and response.json()['status']=='not_run'
    assert response.json()['event']['action']=='refused' and not calls
