"""Dense-reference and lazy-large-shape Systems reductions, no downloaded model loads."""
import threading
from types import SimpleNamespace
import pytest
import torch
from wsp.wiring_stream import Window, writer_gram, reader_gram, head_writer, rank_for, MEMORY_BYTES


class Ref:
    def __init__(self, shape, data=None, index=None, calls=None, cancel=None):
        self.full_shape=shape;self.data=data;self.index=index or (slice(None),slice(None));self.calls=[] if calls is None else calls;self.cancel=cancel
    @property
    def shape(self):return tuple(len(range(n)[s]) for n,s in zip(self.full_shape,self.index))
    @property
    def size(self):return self.shape[0]*self.shape[1]
    def sub(self,index):return Ref(self.full_shape,self.data,index,self.calls,self.cancel)
    def read(self,cache=True):
        assert not cache
        assert self.size<=262144, 'oversized logical read'
        self.calls.append(self.size)
        if self.cancel:self.cancel.set()
        return self.data[self.index].clone() if self.data is not None else torch.zeros(self.shape)


@pytest.mark.parametrize('rank',[None,7])
def test_dense_and_same_seed_projected_grams(rank,monkeypatch):
    import wsp.wiring_stream as S
    monkeypatch.setattr(S,'SLAB_VALUES',64)
    g=torch.Generator().manual_seed(123);d=16
    w=torch.randn(45,d,generator=g);v=torch.randn(d,5,generator=g);o=torch.randn(5,d,generator=g)
    om=torch.randn(d,rank,generator=torch.Generator().manual_seed(0))/rank**.5 if rank else None
    p=w@om if om is not None else w
    a,n=writer_gram(Ref(w.shape,w),om,Window())
    torch.testing.assert_close(a,p.T@p);assert n==pytest.approx(float(w.double().square().sum()))
    a,n=reader_gram(Ref(w.T.shape,w.T),om,Window())
    torch.testing.assert_close(a,p.T@p);assert n==pytest.approx(float(w.double().square().sum()))
    a,n,norm=head_writer(Ref(v.shape,v),Ref(o.shape,o),om,Window())
    dense=v@o;projected=dense@om if om is not None else dense
    torch.testing.assert_close(a,projected.T@projected,rtol=3e-5,atol=1e-4)
    assert n==pytest.approx(float(dense.square().sum()),rel=2e-6)
    assert norm==pytest.approx(float(o.norm()))


def test_large_lazy_mlp_cancels_after_bounded_first_read():
    cancel=threading.Event();ref=Ref((17408,5120),cancel=cancel)
    r,bound=rank_for(5120,25,74)
    assert bound<=MEMORY_BYTES
    om=torch.zeros(5120,r)
    with pytest.raises(ValueError,match='cancelled'):writer_gram(ref,om,Window(cancel))
    assert len(ref.calls)==1 and max(ref.calls)<=262144
    assert sum(ref.calls)<ref.size


def test_resource_bound_and_deadline(monkeypatch):
    for d,nw,nr in [(5120,25,74),(1024,64,192),(16384,384,1152)]:
        r,bytes_=rank_for(d,nw,nr);assert 0<r<=1024 and bytes_<=MEMORY_BYTES
    with pytest.raises(ValueError,match='bound'):Window(seconds=-1).read(Ref((1,1)))


def test_native_permutations_and_patch_overlay(tmp_path,monkeypatch):
    import gguf,numpy as np
    from wsp import arch,delta as D
    from wsp.gguf_store import GGUFStore
    from wsp.workspace import Workspace,Ref as WRef
    data=np.random.default_rng(5).normal(size=(32,32)).astype('float32')
    path=tmp_path/'layout.gguf';writer=gguf.GGUFWriter(str(path),'llama')
    writer.add_head_count(4);writer.add_tensor('blk.0.attn_q.weight',data)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    store=GGUFStore(path);name=store.names()[0]
    ws=Workspace(str(tmp_path),store,SimpleNamespace(),None,dict(kind='gguf'))
    region=(range(2,6),range(3,8));piece=D.block(region,torch.full((4,5),.25))
    ws._pieces={name:[(piece,.5)]}
    ref=WRef(ws,name,perm=(1,0));dense=ref.read(cache=False)
    import wsp.wiring_stream as S
    monkeypatch.setattr(S,'SLAB_VALUES',64)
    om=torch.randn(32,7,generator=torch.Generator().manual_seed(0))/7**.5
    a,n=writer_gram(ref,om,Window())
    torch.testing.assert_close(a,(dense@om).T@(dense@om),rtol=2e-5,atol=1e-4)
    assert n==pytest.approx(float(dense.double().square().sum()))
    assert ws._cache_bytes==0


def test_reject_whole_model_window_before_any_read():
    from wsp.analysis import wiring
    s=SimpleNamespace(n_layers=64,d_model=5120,d_vocab=248320,n_heads=24,
                      has_heads=lambda l:False,has_neurons=lambda l:True,
                      layers=[SimpleNamespace(reads=['W_in','W_gate'])]*64)
    ref=Ref((17408,5120));sem=SimpleNamespace(mlp=lambda l:{'W_out':ref,'W_in':ref,'W_gate':ref})
    ws=SimpleNamespace(struct=s,sem=sem,n_params=lambda:27_000_000_000,calib={})
    with pytest.raises(ValueError,match='three layers'):wiring(ws,layers=list(range(64)))
    with pytest.raises(ValueError,match='before scanning'):wiring(ws,layers=[0,1,2])
    assert not ref.calls


def test_dense_wiring_scores_cache_invalidation_and_api(tmp_path,monkeypatch):
    from test_hf import build
    from wsp.session import Session
    from wsp.server import create_app
    from wsp import analysis,delta as D
    from fastapi.testclient import TestClient
    build('llama',str(tmp_path/'tiny'))  # tiny random fixture, no pretrained weights
    sess=Session(str(tmp_path/'ws'));ws=sess.open(str(tmp_path/'tiny'),log=lambda _:None)
    result=analysis.wiring(ws,top=4000)
    # Independently multiply the actual semantic writer and reader, rather than reusing Gram helpers.
    for edge in result['edges']:
        if edge['src']=='L0MLP' and edge['dst']=='L1MLP':
            W=ws.sem.mlp(0)['W_out'].read();role='W_gate' if edge['kind']=='gate' else 'W_up'
            R=ws.sem.mlp(1)[role].read()
            score=float((W@R).norm()/(W.norm()*R.norm())*ws.struct.d_model**.5)
            assert edge['score']==pytest.approx(score,abs=1e-4)
    cache_keys=[k for k in ws.calib if k.startswith('wiring:')];assert cache_keys
    ref=ws.sem.mlp(0)['W_out'];piece=ref.sub((slice(0,2),slice(0,2))).piece(torch.ones(2,2))
    patch=ws.commit({ref.name:[piece]},'fixture patch')
    assert not any(k.startswith('wiring:') for k in ws.calib)
    changed=analysis.wiring(ws,top=4000)
    assert changed['edges']!=result['edges']
    ws.set_patch(patch.id,strength=0)
    assert analysis.wiring(ws,top=4000)['edges']==result['edges']
    ws.set_patch(patch.id,strength=1);ws.delete_patch(patch.id)
    with TestClient(create_app(sess,'http://testserver')) as c:
        scope=c.get('/api/status').json()
        reply=c.get('/api/wiring?layers=0,1',params={k:scope[k] for k in ('workspace_id','version')});assert reply.status_code==200
        assert reply.json()['bounds']['slab_values']<=262144
