"""Synthetic converter-layout fixtures; no downloaded weights or network."""
import json
import os
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

from wsp import arch, delta as D, gguf_runner, gguf_export
from wsp.gguf_store import GGUFStore, file_hash
from wsp.workspace import Workspace
from wsp.store import norm_index


@pytest.fixture(params=[(2,6),(16,48)])
def hybrid(request,tmp_path,monkeypatch):
    if not os.environ.get('WSP_GGML_BRIDGE'):
        pytest.skip('Pinned native bridge required')
    kh,vh = request.param
    cfg = Qwen3_5TextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
        num_attention_heads=6,num_key_value_heads=1,head_dim=16,max_position_embeddings=64,
        linear_key_head_dim=16,linear_value_head_dim=16,linear_num_key_heads=kh,linear_num_value_heads=vh,
        linear_conv_kernel_dim=4,full_attention_interval=4,tie_word_embeddings=False,
        rope_parameters={'rope_theta':10000.,'partial_rotary_factor':.5,'mrope_section':[2,1,1]})
    torch.manual_seed(713)
    ref = Qwen3_5ForCausalLM(cfg).eval()
    path = tmp_path/'tiny.gguf'; writer = gguf.GGUFWriter(str(path),'qwen35')
    for key,value in {'ssm.group_count':kh,'ssm.time_step_rank':vh,'ssm.state_size':16,'ssm.inner_size':vh*16}.items():
        writer.add_uint32('qwen35.'+key,value)
    mapper = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35,4)
    state, originals = {}, {}
    leaves = {'A_log':'ssm_a','dt_bias':'ssm_dt.bias','in_proj_qkv.weight':'attn_qkv.weight',
              'in_proj_z.weight':'attn_gate.weight','in_proj_a.weight':'ssm_alpha.weight',
              'in_proj_b.weight':'ssm_beta.weight','conv1d.weight':'ssm_conv1d.weight',
              'norm.weight':'ssm_norm.weight','out_proj.weight':'ssm_out.weight'}
    # Independent explicit head listing, matching pinned upstream grouped->tiled
    # converter, rather than reusing the production inverse mapping.
    def order(dim):
        return [((k*(vh//kh)+v)*dim+d) for v in range(vh//kh) for k in range(kh) for d in range(dim)]
    for name,value in ref.state_dict().items():
        a = value.numpy().copy()
        if name.endswith('.A_log'):
            a = np.linspace(-.4,.5,vh,dtype=np.float32)
        if name.endswith('.dt_bias'):
            a = np.linspace(-.2,.3,vh,dtype=np.float32)
        mapped = mapper.get_name(name,try_suffixes=('.weight','.bias'))
        axis = None; perm = None
        if '.linear_attn.' in name:
            prefix,leaf = name.split('.linear_attn.')
            mapped = prefix.replace('model.layers.','blk.')+'.'+leaves[leaf]
            if leaf in ('in_proj_qkv.weight','conv1d.weight'):
                axis,perm = 0,list(range(2*kh*16))+[2*kh*16+i for i in order(16)]
            elif leaf == 'in_proj_z.weight': axis,perm = 0,order(16)
            elif leaf in ('in_proj_a.weight','in_proj_b.weight','A_log','dt_bias'): axis,perm = 0,order(1)
            elif leaf == 'out_proj.weight': axis,perm = 1,order(16)
        originals[name] = a.copy()
        if perm is not None: a = np.take(a,perm,axis=axis)
        if name.endswith('.A_log'): a = -np.exp(a)
        elif name.endswith('norm.weight') and '.linear_attn.norm.' not in name: a = a+1
        elif '.conv1d.' in name: a = a[:,0,:]
        qtype = gguf.GGMLQuantizationType.Q4_0 if value.ndim==2 else gguf.GGMLQuantizationType.F32
        packed = gguf.quantize(a,qtype)
        writer.add_tensor(mapped,packed,raw_dtype=qtype)
        decoded = gguf.dequantize(packed,qtype)
        if name.endswith('.A_log'): decoded = np.log(-decoded)
        elif name.endswith('norm.weight') and '.linear_attn.norm.' not in name: decoded = decoded-1
        elif '.conv1d.' in name: decoded = decoded[:,None,:]
        if perm is not None: decoded = np.take(decoded,np.argsort(perm),axis=axis)
        state[name] = torch.from_numpy(decoded.copy())
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    ref.load_state_dict(state)
    store = GGUFStore(path)
    meta = dict(kind='gguf',gguf_path=str(path),base_sha256=file_hash(path),model_dir='model')
    ws = Workspace(str(tmp_path),store,arch.detect({n:store.shape(n) for n in store.names()},cfg.to_dict()),None,meta,hf_config=cfg.to_dict())
    (tmp_path/'model').mkdir();(tmp_path/'model/config.json').write_text(json.dumps(cfg.to_dict()))
    (tmp_path/'workspace.json').write_text(json.dumps(meta));ws.model_dir=str(tmp_path/'model')
    monkeypatch.setattr(gguf_runner,'require_lease',lambda ws,**kw:None)  # synthetic <300k params only
    monkeypatch.setattr(gguf_export,'require_lease',lambda ws:None)
    return ws,ref,state


def test_grouped_dense_logits_and_all_logical_tensors(hybrid):
    ws,ref,state = hybrid
    for name,value in state.items():
        torch.testing.assert_close(ws.read(name),value,atol=0,rtol=0)
    # Semantic head addressing must select Q rows, excluding interleaved gates.
    query='model.layers.3.self_attn.q_proj.weight'
    for head in range(6):
        index=dict(ws.resolve(f'L3H{head}'))[query]
        torch.testing.assert_close(ws.read(query,index),state[query][head*32:head*32+16],atol=0,rtol=0)
    runner = gguf_runner.GGUFRunner(ws)
    try:
        ids = torch.tensor([[3,7,9,4]])
        with torch.no_grad(): expected = ref(ids,use_cache=False).logits
        actual,cache = runner.run(ids,'base',names=['blocks.0.hook_resid_post','blocks.3.attn.hook_pattern'])
        torch.testing.assert_close(actual,expected,atol=.015,rtol=.03)
        assert cache['blocks.0.hook_resid_post'].shape == (1,4,32)
        assert cache['blocks.3.attn.hook_pattern'].shape == (1,6,4,4)
    finally: runner.release()


def test_grouped_cache_matches_uncached_and_is_ephemeral(hybrid):
    ws,ref,_ = hybrid
    runner = gguf_runner.GGUFRunner(ws)
    try:
        ids=[3,7,9,4,8]; past=None
        runner.use('base')
        with torch.no_grad():
            for i,token in enumerate(ids):
                output=runner.model(input_ids=torch.tensor([[token]]),past_key_values=past,use_cache=True)
                past=output.past_key_values
                expected=ref(torch.tensor([ids[:i+1]]),use_cache=False).logits[:,-1:]
                torch.testing.assert_close(output.logits,expected,atol=.015,rtol=.03)
            recurrent=past.get_recurrent_state(0,state_idx=0)
            assert recurrent.shape == (1,ws.hf_config['linear_num_value_heads'],16,16)
            assert past.get_conv_state(0,state_idx=0).shape[-1] == 4
        sizes=[]
        hook=runner.model.register_forward_pre_hook(lambda m,a,k:sizes.append(k['input_ids'].shape[-1]),with_kwargs=True)
        before=runner.generate(ids[:3],'base',max_new_tokens=3)
        assert sizes==[3,1,1]
        patch=ws.commit({'model.layers.0.linear_attn.out_proj.weight':[D.block((0,range(ws.shape('model.layers.0.linear_attn.out_proj.weight')[1])),torch.ones(ws.shape('model.layers.0.linear_attn.out_proj.weight')[1])*.2)]},'cache isolation')
        runner.generate(ids[:3],'current',max_new_tokens=3)
        ws.set_patch(patch.id,strength=.3);runner.generate(ids[:3],'current',max_new_tokens=3)
        ws.set_patch(patch.id,enabled=False)
        assert runner.generate(ids[:3],'current',max_new_tokens=3)==before
        assert runner.generate(ids[:3],'base',max_new_tokens=3)==before
        assert sizes==[3,1,1]*5
        hook.remove()
        assert runner.last_generation['decode_steps']==2
        assert not hasattr(runner,'_decode_cache')
    finally: runner.release()


def test_grouped_inverse_export_all_transforms(hybrid,tmp_path):
    ws,_,_ = hybrid
    names=['model.layers.0.linear_attn.'+leaf for leaf in ('in_proj_qkv.weight','in_proj_z.weight',
        'in_proj_a.weight','in_proj_b.weight','out_proj.weight','conv1d.weight','A_log','dt_bias','norm.weight')]
    names += ['model.layers.0.input_layernorm.weight','model.layers.3.self_attn.q_proj.weight']
    for name in names:
        shape=ws.shape(name)
        if len(shape)==2:
            region=(range(min(3,shape[0])),range(min(32,shape[1])))
            if name.endswith('in_proj_qkv.weight'):
                start=2*ws.hf_config['linear_num_key_heads']*16
                region=(range(start,start+3),range(32))  # value rows, not unchanged Q/K prefix
            piece=D.lowrank(region,torch.ones(len(region[0]),1)*.02,torch.ones(1,len(region[1])))
        else:
            region=norm_index((),shape);piece=D.block(region,torch.ones(shape)*.01)
        ws.commit({name:[piece]},'logical transform '+name)
    before=file_hash(ws.store.path)
    report=gguf_export.export_merged(ws,tmp_path/'merged.gguf')
    merged=GGUFStore(report['path'])
    for name in names:
        torch.testing.assert_close(merged.read(name),ws.read(name),atol=.02,rtol=.02)
    assert file_hash(ws.store.path)==before
    assert report['changed_rows_max_weight_error']<.02
