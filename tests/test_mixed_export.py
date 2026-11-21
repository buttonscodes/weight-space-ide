import json
import os
import threading

import gguf
import numpy as np
import pytest
import torch

from wsp import arch, delta as D, gguf_mixed as M
from wsp.gguf_store import GGUFStore, file_hash
from wsp.workspace import Workspace
from test_hybrid_grouped import hybrid


@pytest.fixture
def simple(tmp_path,monkeypatch):
    if not os.environ.get('WSP_GGML_BRIDGE'):pytest.skip('native bridge required')
    path=tmp_path/'source.gguf';w=gguf.GGUFWriter(str(path),'qwen2')
    w.add_file_type(2);w.add_string('tokenizer.chat_template','{{ messages }}');w.add_array('test.array',['one','two'])
    rng=np.random.default_rng(17)
    for n,q,shape in [('blk.0.ffn_down.weight',gguf.GGMLQuantizationType.Q4_0,(32,64)),
                      ('token_embd.weight',gguf.GGMLQuantizationType.Q4_0,(64,32)),
                      ('output_norm.weight',gguf.GGMLQuantizationType.F32,(32,))]:
        a=rng.normal(size=shape).astype(np.float32)
        w.add_tensor(n,gguf.quantize(a,q),raw_dtype=q)
    w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
    store=GGUFStore(path);ws=Workspace(str(tmp_path),store,arch.detect({n:store.shape(n) for n in store.names()},{}),None,
        dict(kind='gguf',gguf_path=str(path),base_sha256=file_hash(path)))
    monkeypatch.setattr(M,'require_lease',lambda ws:None)
    return ws


@pytest.mark.parametrize('precision',['Q8_0','F16','F32'])
def test_mixed_directory_promotion_and_untouched(simple,tmp_path,monkeypatch,precision):
    ws=simple;n='model.layers.0.mlp.down_proj.weight'
    ws.commit({n:[D.block((0,range(64)),torch.full((64,),.001))]},'fine edit')
    disabled=ws.commit({'model.embed_tokens.weight':[D.block((0,range(32)),torch.ones(32))]},'disabled')
    ws.set_patch(disabled.id,strength=0)
    original=ws.read(n);before=file_hash(ws.store.path)
    read=ws.store.read;slabs=[]
    def bounded(name,index):
        value=read(name,index);slabs.append(value.numel()*4);assert value.ndim<=1
        return value
    monkeypatch.setattr(ws.store,'read',lambda *a:pytest.fail('preview read weights'))
    preview=M.preview_export(ws,precision)
    assert preview['changed_tensor_count']==1
    monkeypatch.setattr(ws.store,'read',bounded)
    report=M.export_mixed(ws,tmp_path/(precision+'.gguf'),precision)
    merged=GGUFStore(report['path']);assert os.path.getsize(report['path'])==preview['output_bytes']
    assert report['validation_status']=='UNVALIDATED' and report['structural_status']=='PASSED'
    assert merged.dtype(n)==precision
    assert max(slabs)<=M.SLAB
    for name,t in ws.store.where.items():
        if name!=n:assert t.data.tobytes()==merged.where[name].data.tobytes()
        assert merged.where[name].data_offset%merged.reader.alignment==0
    assert 'general.file_type' not in merged.reader.fields
    assert merged.reader.fields['test.array'].contents()==['one','two']
    assert merged.reader.fields['tokenizer.chat_template'].contents()=='{{ messages }}'
    torch.testing.assert_close(merged.read(n),original,atol={'Q8_0':.03,'F16':.002,'F32':0}[precision],rtol=0)
    assert file_hash(ws.store.path)==before
    with pytest.raises(ValueError,match='new path'):M.export_mixed(ws,report['path'],precision)


@pytest.mark.parametrize('precision',['Q8_0','F16','F32'])
def test_grouped_mixed_all_inverse_maps(hybrid,tmp_path,monkeypatch,precision):
    ws,_,_=hybrid;monkeypatch.setattr(M,'require_lease',lambda ws:None)
    names=[n for n in ws.names() if n.startswith('model.layers.0.')]
    for n in names:
        shape=ws.shape(n)
        if len(shape)==2:
            # Include grouped V rows and output columns, not just Q/K prefix.
            region=(range(shape[0]-2,shape[0]),range(shape[1]-16,shape[1]))
            piece=D.lowrank(region,torch.ones(2,1)*.013,torch.ones(1,16))
        else:
            from wsp.store import norm_index
            piece=D.block(norm_index((),shape),torch.full(shape,.007))
        p=ws.commit({n:[piece]},'mapped edit');ws.set_patch(p.id,strength=.7)
    expected={n:ws.read(n) for n in names}
    result=M.export_mixed(ws,tmp_path/(precision+'.gguf'),precision);other=GGUFStore(result['path'])
    for n in names:
        torch.testing.assert_close(other.read(n),expected[n],atol={'Q8_0':.01,'F16':.001,'F32':1e-6}[precision],rtol=1e-5)
    for n in ws.names():
        if n not in names:assert ws.store.where[n].data.tobytes()==other.where[n].data.tobytes()


def test_failure_cleanup_zero_patch_and_precision(simple,tmp_path,monkeypatch):
    ws=simple;n='model.layers.0.mlp.down_proj.weight'
    ws.commit({n:[D.block((0,range(64)),torch.zeros(64))]},'zero')
    assert M.preview_export(ws)['changed_tensor_count']==0
    with pytest.raises(ValueError,match='Precision'):M.preview_export(ws,'Q4_0')
    event=threading.Event();event.set()
    with pytest.raises(M.ExportCancelled):M.export_mixed(ws,tmp_path/'cancel.gguf',cancel=event)
    assert not list(tmp_path.glob('*.part'))
    monkeypatch.setattr(M.shutil,'disk_usage',lambda path:type('Disk',(),{'free':0})())
    with pytest.raises(ValueError,match='Insufficient disk'):M.export_mixed(ws,tmp_path/'disk.gguf')
    assert not (tmp_path/'disk.gguf').exists()


def test_cancel_during_copy_and_higher_precision(simple,tmp_path):
    ws=simple;n='model.norm.weight';ws.commit({n:[D.block((range(32),),torch.full((32,),.001))]},'norm')
    assert M.preview_export(ws,'Q8_0')['tensors'][0]['after']=='F32'
    event=threading.Event()
    with pytest.raises(M.ExportCancelled):
        M.export_mixed(ws,tmp_path/'cancel.gguf',progress=lambda _:event.set(),cancel=event)
    assert not (tmp_path/'cancel.gguf').exists() and not list(tmp_path.glob('.*.part*'))


def test_multi_piece_scaled_disabled_and_race(simple,tmp_path,monkeypatch):
    ws=simple;n='model.layers.0.mlp.down_proj.weight'
    for scale in (.3,-.5):
        p=ws.commit({n:[D.block((0,range(64)),torch.full((64,),.013))]},'composed');ws.set_patch(p.id,strength=scale)
    expected=ws.read(n);target=tmp_path/'race.gguf';real=M.os.link
    def race(src,dst):
        if Path(dst)==target:target.write_bytes(b'other writer')
        return real(src,dst)
    from pathlib import Path
    monkeypatch.setattr(M.os,'link',race)
    with pytest.raises(FileExistsError):M.export_mixed(ws,target,'F32')
    assert target.read_bytes()==b'other writer' and not Path(str(target)+'.json').exists()
    assert not list(tmp_path.glob('.*.part*'))
    monkeypatch.setattr(M.os,'link',real)
    result=M.export_mixed(ws,tmp_path/'composed.gguf','F32')
    torch.testing.assert_close(GGUFStore(result['path']).read(n),expected,atol=0,rtol=0)


def test_metadata_preview_load_never_hashes_payload_or_loads_runner(hybrid,monkeypatch):
    ws,_,_=hybrid
    import wsp.gguf_store as store
    from wsp.memory_guard import require_lease, GuardError
    monkeypatch.setattr(store,'file_hash',lambda *a:pytest.fail('preview hashed tensor payload'))
    view=Workspace.load(ws.dir,metadata_only=True)
    assert M.preview_export(view)['changed_tensor_count']==0
    with pytest.raises(GuardError,match='Metadata-only'):
        require_lease(view,loading=True)


def test_export_api_preview_and_job_do_not_infer(simple,tmp_path,monkeypatch):
    import time
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    from wsp.session import Session
    ws=simple;n='model.layers.0.mlp.down_proj.weight'
    ws.commit({n:[D.block((0,range(64)),torch.full((64,),.01))]},'api')
    monkeypatch.setattr(Workspace,'runner',property(lambda _:pytest.fail('export loaded inference')))
    sess=Session(str(tmp_path));sess.ws=ws
    with TestClient(create_app(sess,'http://testserver'),raise_server_exceptions=False) as client:
        p=client.post('/api/gguf/export/preview',json={'precision':'F16'})
        assert p.status_code==200 and p.json()['changed_tensor_count']==1
        assert client.post('/api/gguf/export/preview',json={'precision':'bad'}).status_code==400
        r=client.post('/api/gguf/export',json={'precision':'F16'});assert r.status_code==200
        ident=r.json()['id']
        for _ in range(100):
            state=client.get('/api/gguf/export/'+ident).json()
            if state['status']!='RUNNING':break
            time.sleep(.01)
        assert state['status']=='COMPLETE' and state['validation_status']=='UNVALIDATED'
        assert client.get('/api/gguf/export/'+ident+'/download').content[:4]==b'GGUF'


def test_wide_matrix_uses_one_bounded_row(tmp_path,monkeypatch):
    if not os.environ.get('WSP_GGML_BRIDGE'):pytest.skip('native bridge required')
    width=M.SLAB//4;path=tmp_path/'wide.gguf';writer=gguf.GGUFWriter(str(path),'qwen2')
    writer.add_tensor('blk.0.ffn_down.weight',gguf.quantize(np.ones((2,width),np.float32),gguf.GGMLQuantizationType.Q4_0),raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    store=GGUFStore(path);ws=Workspace(str(tmp_path),store,arch.detect({n:store.shape(n) for n in store.names()},{}),None,dict(kind='gguf',base_sha256=file_hash(path)))
    n=ws.names()[0];ws.commit({n:[D.block((1,0),torch.tensor(.125))]},'wide single cell')
    monkeypatch.setattr(M,'require_lease',lambda ws:None);read=store.read;sizes=[]
    def limited(name,index):
        value=read(name,index);sizes.append(value.numel()*4);assert value.ndim==1;return value
    monkeypatch.setattr(store,'read',limited)
    result=M.export_mixed(ws,tmp_path/'wide-f32.gguf','F32')
    assert sizes==[M.SLAB,M.SLAB] and GGUFStore(result['path']).read(n,(1,0)).item()==1.125


def test_source_change_and_float_overflow_publish_nothing(simple,tmp_path):
    ws=simple;n='model.layers.0.mlp.down_proj.weight'
    ws.commit({n:[D.block((0,0),torch.tensor(100000.))]},'overflow')
    with pytest.raises(ValueError,match='overflow'):M.export_mixed(ws,tmp_path/'overflow.gguf','F16')
    assert not (tmp_path/'overflow.gguf').exists()
    def mutate(_):
        with open(ws.store.path,'ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='Base identity changed'):M.export_mixed(ws,tmp_path/'changed.gguf','F32',progress=mutate)
    assert not (tmp_path/'changed.gguf').exists() and not list(tmp_path.glob('.*.part*'))


def test_precision_order_never_downgrades():
    q=gguf.GGMLQuantizationType
    assert M._type(q.F32,'F16')==q.F32
    assert M._type(q.F16,'Q8_0')==q.F16
    assert M._type(q.BF16,'F16')==q.F32
    assert M._type(q.BF16,'Q8_0')==q.BF16
