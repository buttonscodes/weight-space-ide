"""Packed math and logical addressing checked against independent GGUF decoding."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
gguf = pytest.importorskip('gguf')
from wsp import delta as D
from wsp.gguf_runner import delta_matmul
from wsp.gguf_store import GGUFStore, logical_name
from wsp.memory_guard import Policy, GuardError, GiB, require_lease
from wsp.packed import PackedKernel
from wsp.store import norm_index


@pytest.fixture(params=['F32','F16','BF16','Q4_0','Q4_1','Q5_0','Q5_1','Q8_0','Q2_K','Q3_K','Q4_K','Q5_K','Q6_K'])
def matrix(request, tmp_path):
    rng = np.random.default_rng(718)
    original = rng.normal(0, .1, (32, 256)).astype(np.float32)
    qtype = getattr(gguf.GGMLQuantizationType, request.param)
    if request.param.endswith('_K'):
        if not os.environ.get('WSP_GGML_BRIDGE'):
            pytest.skip('K fixture quantization needs the pinned upstream ggml library')
        import ctypes as C
        kernel = PackedKernel()
        quantize = kernel.lib.ggml_quantize_chunk
        quantize.argtypes = [C.c_int,C.c_void_p,C.c_void_p,C.c_int64,C.c_int64,C.c_int64,C.c_void_p]
        quantize.restype = C.c_size_t
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        packed = np.empty((32,256//block*size),dtype=np.uint8)
        assert quantize(int(qtype),original.ctypes.data,packed.ctypes.data,0,32,256,None) == packed.nbytes
    else:
        packed = gguf.quantize(original, qtype)
    path = tmp_path / 'tiny.gguf'
    writer = gguf.GGUFWriter(str(path), 'qwen2')
    writer.add_tensor('blk.0.attn_q.weight', packed, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return GGUFStore(str(path)), gguf.dequantize(packed, qtype), original


def test_decode_regions(matrix):
    store, dense, original = matrix
    name = 'model.layers.0.self_attn.q_proj.weight'
    assert store.shape(name) == dense.shape
    for idx in [(3, slice(31, 71, 3)), (slice(2, 20, 3), 17), (slice(None, None, -1), slice(None, None, -3)), ()]:
        actual = store.read(name, norm_index(idx, dense.shape))
        np.testing.assert_array_equal(actual.numpy(), dense[idx])
    assert not store.native(name).flags.writeable
    # Quantization error is distinct from overlay/export error.
    assert np.max(np.abs(dense-original)) < .16


def test_packed_matmul_overlay(matrix):
    if not os.environ.get('WSP_GGML_BRIDGE'):
        pytest.skip('Build the optional pinned ggml bridge to run native math validation')
    store, dense, _ = matrix
    name = store.names()[0]
    kernel = PackedKernel()
    x = torch.randn(1, 5, 256, generator=torch.Generator().manual_seed(19))
    raw = store.native(name).view(np.uint8).reshape(32, -1)
    actual = kernel.matmul(raw, store.where[name].tensor_type, (32, 256), x)
    want = x @ torch.from_numpy(dense).T
    # ggml packs activations to Q8 for quantized dot products: account for that
    # independently of GGUF weight reconstruction / FP32 delta accuracy.
    torch.testing.assert_close(actual, want, atol=.04, rtol=.02)
    base = actual.clone()
    pieces = [D.lowrank((range(3, 19, 2), range(7, 60, 3)), torch.randn(8, 2), torch.randn(2, 18)),
              D.block((5, range(2, 10)), torch.randn(8)),
              D.Piece('sparse', norm_index((), (32, 256)), values=torch.tensor([.2, -.4, .7]),
                      coords=torch.tensor([[1, 15], [1, 15], [9, 71]]))]
    delta = torch.zeros(32, 256)
    for piece in pieces:
        delta_matmul(x, piece, -.7, actual)
        # independent reconstruction, including repeated sparse coordinates
        if piece.kind == 'sparse':
            delta.index_put_(tuple(piece.coords.T), piece.values * -.7, accumulate=True)
        else:
            D.add_to(delta, norm_index((), delta.shape), piece, -.7)
    torch.testing.assert_close(actual-base, x@delta.T, atol=2e-5, rtol=2e-5)


def test_layout_names_and_rotary(tmp_path):
    assert logical_name('blk.2.ffn_down.weight', 'qwen2') == 'model.layers.2.mlp.down_proj.weight'
    assert logical_name('blk.2.ffn_down.weight', 'unknown') == 'blk.2.ffn_down.weight'
    original = np.arange(32*32, dtype=np.float32).reshape(32, 32)
    stored = original.reshape(4, 2, 4, 32).swapaxes(1, 2).reshape(32, 32)
    p = tmp_path/'llama.gguf'
    w = gguf.GGUFWriter(str(p), 'llama')
    w.add_head_count(4)
    w.add_head_count_kv(4)
    w.add_tensor('blk.0.attn_q.weight', stored)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    np.testing.assert_array_equal(GGUFStore(str(p)).read('model.layers.0.self_attn.q_proj.weight').numpy(), original)


def test_guard_fail_closed():
    p = Policy()
    baseline = dict(available=64*GiB, pressure=1, swap_used=0, swap_out=0, compressed=0,
                    tree_rss=0, tree_footprint=0)
    p.preflight(baseline, 6*GiB)
    for bad in [dict(baseline, available=None), dict(baseline, pressure=2), dict(baseline, available=29*GiB)]:
        with pytest.raises(GuardError):
            p.preflight(bad, 6*GiB)
    for change in [dict(available=19*GiB), dict(pressure=2), dict(swap_used=128*1024**2),
                   dict(compressed=2*GiB), dict(tree_footprint=17*GiB)]:
        assert p.violation({**baseline, **change}, baseline)


def test_missing_guard_refuses_before_load(monkeypatch):
    monkeypatch.delenv('WSP_GUARD_LEASE', raising=False)
    with pytest.raises(GuardError, match='watchdog'):
        require_lease(None)


def test_kernel_missing_has_no_fallback(monkeypatch):
    monkeypatch.delenv('WSP_GGML_BRIDGE', raising=False)
    with pytest.raises(RuntimeError, match='dequantization is disabled'):
        PackedKernel()


def test_packed_safetensors_refuse_raw_integer_cast(tmp_path):
    from wsp.store import Store,write_safetensors
    p = tmp_path/'packed.safetensors'
    write_safetensors(str(p),[('weight','U32',(2,4))],lambda _:np.ones((2,4),dtype=np.uint32).tobytes())
    s = Store([str(p)])
    assert not s.is_float('weight')
    with pytest.raises(ValueError,match='raw integers are not weights'):
        s.read('weight')
    p = tmp_path/'scaled.safetensors'
    write_safetensors(str(p),[('weight','F32',(2,4))],lambda _:np.ones((2,4),dtype=np.float32).tobytes())
    s = Store([str(p)]);s.quantized_config={'quant_method':'unknown'}
    assert not s.is_float('weight')
    with pytest.raises(ValueError,match='format backend'):
        s.read('weight')


def test_hybrid_unvalidated_head_layout_stays_raw(tmp_path):
    p=tmp_path/'unsupported.gguf';w=gguf.GGUFWriter(str(p),'qwen35')
    w.add_uint32('qwen35.ssm.group_count',2);w.add_uint32('qwen35.ssm.time_step_rank',3)
    w.add_tensor('blk.0.attn_gate.weight',np.zeros((32,32),dtype=np.float32))
    w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
    store=GGUFStore(str(p))
    assert not store.layout_supported and store.names()==['blk.0.attn_gate.weight']
    assert not store.capabilities()['generation']['supported']


@pytest.mark.parametrize('level', ['1', '2', '4', '0', 'garbage', None])
def test_macos_pressure_parser(monkeypatch, level):
    from wsp import memory_guard as g
    import subprocess
    from types import SimpleNamespace
    monkeypatch.setattr(g.psutil,'virtual_memory',lambda:SimpleNamespace(available=64*GiB,free=GiB,total=128*GiB))
    monkeypatch.setattr(g.psutil,'swap_memory',lambda:SimpleNamespace(used=0,sout=0))
    monkeypatch.setattr(g.platform, 'system', lambda:'Darwin')
    raw = '''Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100000.
Pages inactive: 4000000.
Pages speculative: 1.
File-backed pages: 3000000.
Pages purgeable: 2000.
Pages occupied by compressor: 10.
Swapouts: 0.
'''
    def command(*args):
        if args[0].endswith('vm_stat'):
            return raw
        if args[-1] == 'vm.memory_pressure':
            return '10221'  # pageout activity is not a WARN notification
        if level is None:
            raise subprocess.CalledProcessError(1,args,stderr='permission denied')
        return level
    monkeypatch.setattr(g, '_command', command)
    if level in ('1','2','4'):
        s = g.snapshot()
        assert s['pressure'] == int(level)
        assert s['vm_pressure'] == 10221
        assert bool(Policy().violation({**s,'tree_footprint':0}, s)) == (level != '1')
    else:
        with pytest.raises((GuardError,ValueError,subprocess.CalledProcessError)):
            g.snapshot()


@pytest.mark.parametrize('escape_group',[False,True])
@pytest.mark.parametrize('deny_group',[False,True])
def test_watchdog_owned_tree_and_unrelated_sentinel(monkeypatch, tmp_path, escape_group, deny_group):
    import subprocess
    import psutil
    from wsp import memory_guard as g
    baseline = dict(time=0, available=64*GiB, pressure=1, swap_used=0, swap_out=0, compressed=0)
    calls = 0
    def sample():
        nonlocal calls
        calls += 1
        return {**baseline, 'pressure':2 if calls >= 4 else 1}
    monkeypatch.setattr(g, 'snapshot', sample)
    if deny_group:
        def denied(*args):raise PermissionError('synthetic killpg denial')
        monkeypatch.setattr(g.os,'killpg',denied)
    pidfile = tmp_path/'child.pid'
    command = [sys.executable, '-c',
        f'import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session={escape_group}); '
        f'pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)']
    sentinel = subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
    try:
        with pytest.raises(GuardError,match='pressure warning'):
            g.supervise(command,tmp_path/'guard.jsonl',GiB,timeout=10)
        assert sentinel.poll() is None
        child = int(pidfile.read_text())
        assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
        records = [json.loads(x) for x in (tmp_path/'guard.jsonl').read_text().splitlines()]
        assert records[-1]['remaining'] == []
        assert records[-1]['cleanup_complete'] is True
        assert bool(records[-1]['cleanup_errors']) is deny_group
        assert (tmp_path/'GUARD_FAILED.json').exists()
        with pytest.raises(GuardError,match='latched'):
            g.supervise(command,tmp_path/'again.jsonl',GiB,timeout=10)
    finally:
        sentinel.terminate(); sentinel.wait(timeout=5)


@pytest.mark.parametrize('family',['qwen2','llama','qwen35'])
def test_tiny_packed_skeleton_and_hooks(monkeypatch, tmp_path, family):
    if not os.environ.get('WSP_GGML_BRIDGE'):
        pytest.skip('native bridge unavailable')
    from transformers import Qwen2Config, Qwen2ForCausalLM, LlamaConfig, LlamaForCausalLM, Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from wsp.workspace import Workspace
    from wsp import arch, gguf_runner as r
    config_cls, model_cls = (Qwen2Config,Qwen2ForCausalLM) if family == 'qwen2' else (LlamaConfig,LlamaForCausalLM)
    cfg = config_cls(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=False)
    if family == 'qwen35':
        cfg = Qwen3_5TextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
              num_attention_heads=2,num_key_value_heads=2,head_dim=16,max_position_embeddings=64,
              linear_key_head_dim=16,linear_value_head_dim=16,linear_num_key_heads=2,linear_num_value_heads=2,
              linear_conv_kernel_dim=4,full_attention_interval=4,tie_word_embeddings=False,
              rope_parameters={'rope_theta':10000.,'partial_rotary_factor':.25,'mrope_section':[1,1,0]})
        model_cls = Qwen3_5ForCausalLM
    torch.manual_seed(731)
    ref = model_cls(cfg).eval()
    mapping = gguf.get_tensor_name_map({'qwen2':gguf.MODEL_ARCH.QWEN2,'llama':gguf.MODEL_ARCH.LLAMA,'qwen35':gguf.MODEL_ARCH.QWEN35}[family],cfg.num_hidden_layers)
    p = tmp_path/'tiny-qwen.gguf'
    w = gguf.GGUFWriter(str(p),family)
    w.add_head_count(4); w.add_head_count_kv(2)
    if family == 'qwen35':
        w.add_uint32('qwen35.ssm.group_count',cfg.linear_num_key_heads)
        w.add_uint32('qwen35.ssm.time_step_rank',cfg.linear_num_value_heads)
    state = {}
    for name, value in ref.state_dict().items():
        qtype = gguf.GGMLQuantizationType.Q4_0 if value.ndim == 2 else gguf.GGMLQuantizationType.F32
        original = value.detach().numpy().astype(np.float32)
        if family == 'qwen35':
            if name.endswith('.A_log'):
                original = -np.exp(original)
            elif name.endswith('norm.weight') and '.linear_attn.norm.' not in name:
                original = original+1
            elif '.conv1d.' in name:
                original = original.squeeze(1)
        heads = 4 if '.q_proj.' in name else 2 if '.k_proj.' in name else None
        if family == 'llama' and heads:
            original = original.reshape(heads,2,-1,32).swapaxes(1,2).reshape(value.shape)
        packed = gguf.quantize(original,qtype)
        mapped = mapping.get_name(name,try_suffixes=('.weight','.bias'))
        if family == 'qwen35' and '.linear_attn.' in name:
            prefix,leaf = name.split('.linear_attn.')
            converted = {'A_log':'ssm_a','dt_bias':'ssm_dt.bias','in_proj_qkv.weight':'attn_qkv.weight',
                         'in_proj_z.weight':'attn_gate.weight','in_proj_a.weight':'ssm_alpha.weight',
                         'in_proj_b.weight':'ssm_beta.weight','conv1d.weight':'ssm_conv1d.weight',
                         'norm.weight':'ssm_norm.weight','out_proj.weight':'ssm_out.weight'}
            mapped = prefix.replace('model.layers.','blk.')+'.'+converted[leaf]
        assert mapped is not None,name
        w.add_tensor(mapped,packed,raw_dtype=qtype)
        decoded = gguf.dequantize(packed,qtype)
        if family == 'qwen35':
            if name.endswith('.A_log'):
                decoded = np.log(-decoded)
            elif name.endswith('norm.weight') and '.linear_attn.norm.' not in name:
                decoded = decoded-1
            elif '.conv1d.' in name:
                decoded = decoded[:,None,:]
        if family == 'llama' and heads:
            decoded = decoded.reshape(heads,-1,2,32).swapaxes(1,2).reshape(value.shape)
        state[name] = torch.from_numpy(decoded.copy())
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    ref.load_state_dict(state)
    store = GGUFStore(str(p))
    shapes = {n:store.shape(n) for n in store.names()}
    ws = Workspace(str(tmp_path),store,arch.detect(shapes,cfg.to_dict()),None,dict(kind='gguf'),hf_config=cfg.to_dict())
    monkeypatch.setattr(r,'require_lease',lambda ws,**kw:None)  # synthetic 10k-parameter fixture only
    # Construction MUST NOT read any matrix. Packed embedding may read selected rows at runtime.
    original_read = store.read
    def limited(name, region=None):
        assert len(store.shape(name)) == 1 or '.conv1d.' in name or region is not None, 'full matrix decode'
        return original_read(name,region)
    monkeypatch.setattr(store,'read',limited)
    runner = r.GGUFRunner(ws)
    ids = [3,7,9]
    with torch.no_grad():
        expected = ref(torch.tensor([ids])).logits
    result, cache = runner.run(ids,'base',names=['blocks.0.hook_resid_post'])
    torch.testing.assert_close(result,expected,atol=.01,rtol=.02)
    internal = 'model.layers.0.mlp.down_proj.weight'
    probe = D.Piece('sparse',norm_index((),ws.shape(internal)),coords=torch.tensor([[0,1],[2,3]]),values=torch.tensor([1.,-1.]))
    changed = runner.run(ids,'current',extra={internal:[probe]})[0]
    assert float((changed-result).abs().max()) > 1e-6
    assert cache['blocks.0.hook_resid_post'].shape == (1,3,32)
    assert all(p.ndim <= 1 or ('.conv1d.' in n and p.numel()<=65536) for n,p in runner.model.named_parameters())
    from wsp.analysis import svd_view
    layer = 3 if family == 'qwen35' else 0
    view = svd_view(ws,f'model.layers.{layer}.self_attn.q_proj.weight[0:8,0:16]',k=2)
    assert view['components'][0]['right']['kind'] == 'selected residual coordinates'
    runner.release()
    monkeypatch.setattr(store,'read',original_read)
    # Save/reopen both compact patch kinds, then a streamed same-scheme merge.
    from wsp.gguf_store import file_hash
    from wsp.companion import export_bundle, import_bundle
    from wsp import gguf_export
    ws.meta.update(model_dir='model',gguf_path=str(p),base_sha256=file_hash(p))
    (tmp_path/'model').mkdir()
    (tmp_path/'model/config.json').write_text(json.dumps(cfg.to_dict()))
    (tmp_path/'workspace.json').write_text(json.dumps(ws.meta))
    name = f'model.layers.{layer}.self_attn.q_proj.weight'
    piece = D.lowrank((range(2,5),range(4,10)),torch.ones(3,1)*.02,torch.ones(1,6))
    patch = ws.commit({name:[piece]},'rank one')
    ws.set_patch(patch.id,strength=.5)
    disabled = ws.commit({name:[D.block((0,range(32)),torch.ones(32))]},'disabled')
    ws.set_patch(disabled.id,enabled=False)
    desired = ws.read(name).clone()
    bundle = export_bundle(ws,tmp_path/'patches-export')['path']
    reopened = import_bundle(str(tmp_path/'reloaded'),bundle)
    torch.testing.assert_close(reopened.read(name),desired,atol=0,rtol=0)
    assert reopened.patches[0].strength == .5 and not reopened.patches[1].enabled
    import zipfile
    bad = tmp_path/'tampered.zip'
    with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(bad,'w') as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            dst.writestr(item.filename,data+b' ' if item.filename=='workspace.json' else data)
    with pytest.raises(ValueError,match='checksum mismatch'):
        import_bundle(str(tmp_path/'bad'),bad)
    monkeypatch.setattr(gguf_export,'require_lease',lambda ws:None) # only this synthetic tiny fixture
    report = gguf_export.export_merged(ws,tmp_path/'merged.gguf')
    merged = GGUFStore(report['path'])
    diff = (merged.read(name)-desired).abs()
    assert float(diff.max()) < .02
    assert report['rows_requantized'] == 3
    # Standard mixed export exercises the independent Qwen/Llama Q-head fixture,
    # including the Llama RoPE permutation, with the same enabled/scaled edits.
    from wsp import gguf_mixed
    monkeypatch.setattr(gguf_mixed,'require_lease',lambda ws:None)
    for precision in ('Q8_0','F16','F32'):
        result=gguf_mixed.export_mixed(ws,tmp_path/(precision+'.gguf'),precision)
        promoted=GGUFStore(result['path'])
        torch.testing.assert_close(promoted.read(name),desired,atol=.003 if precision!='F32' else 1e-6,rtol=.01)
        for untouched in store.names():
            if untouched!=name:assert store.where[untouched].data.tobytes()==promoted.where[untouched].data.tobytes()
    assert file_hash(p) == ws.meta['base_sha256']
    with pytest.raises(ValueError,match='new path'):
        gguf_export.export_merged(ws,p)


def test_preserve_limit_is_explicitly_advisory(tmp_path,monkeypatch):
    from wsp import zoo
    from wsp.session import Session
    zoo.build_blank_modadd(str(tmp_path/'blank'))
    session = Session(str(tmp_path)); session.open('blank')
    session.task['preserve'] = {'max_kl':.001}
    monkeypatch.setattr(session,'drift',lambda extra=None:dict(kl=.5,n=4,top1_agree=0.))
    name = next(n for n in session.ws.names() if len(session.ws.shape(n))==2)
    region = norm_index((0,0),session.ws.shape(name))
    report = session.commit_deltas({name:[D.block(region,torch.tensor(.1))]},'advisory check','test',{})
    assert report['committed'] and report['drift']['within_limit'] is False
    assert 'advisory' in report['drift']['enforcement']


def test_metal_matrix_types(matrix):
    if not os.environ.get('WSP_GGML_METAL_BRIDGE'):
        pytest.skip('Optional pinned Metal bridge not built')
    from wsp.packed import MetalKernel
    store,dense,_ = matrix
    name = store.names()[0]
    kernel = MetalKernel(store)
    try:
        x = torch.randn(1,3,256,generator=torch.Generator().manual_seed(725))
        actual = kernel.matmul(store.native(name).view(np.uint8).reshape(32,-1),store.where[name].tensor_type,(32,256),x)
        # Upstream Metal MMA uses half fragments; separate kernel arithmetic error
        # from weight reconstruction (CPU uses Q8 activation packing instead).
        torch.testing.assert_close(actual,x@torch.from_numpy(dense).T,atol=.01,rtol=.003)
        assert kernel.metrics()['metal_allocated'] > 0
        assert kernel.metrics()['metal_mapped_bytes'] == store.reader.data.nbytes
    finally:
        kernel.close()
    assert kernel.metrics()['metal_mapped_bytes'] == 0
