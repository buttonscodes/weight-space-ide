"""Review regressions: conservative telemetry and portable tokenizer behavior."""
import hashlib
import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import gguf
from transformers import AutoTokenizer, Qwen2Tokenizer

from wsp.companion import export_bundle, import_bundle, safe_member
from wsp.gguf_store import file_hash
from wsp.memory_guard import GiB, GuardError, Policy, macos_vm_stat


@pytest.mark.parametrize('body,reason',[
    ({'messages':[{'role':'user','content':[]} ]},'text-only'),
    ({'messages':[{'role':'user','content':'Hi'}],'chat_options':{'enable_thinking':'no'}},'booleans'),
    ({'messages':[{'role':'user','content':'Hi'}],'chat_options':{'reasoning_effort':'invalid'}},'Reasoning effort'),
    ({'messages':[{'role':'user','content':'Hi'}],'chat_options':{'unknown':True}},'Unsupported chat'),
])
def test_chat_rejects_unsupported_before_runner_load(tmp_path,body,reason):
    from fastapi.testclient import TestClient
    from wsp.session import Session
    from wsp.server import create_app
    session=Session(str(tmp_path))
    # No runner/store attributes: validation must finish before touching them.
    session.ws=SimpleNamespace(kind='gguf')
    response=TestClient(create_app(session,'http://testserver'),raise_server_exceptions=False).post('/api/chat',json=body)
    assert response.status_code==400 and reason in response.text


def vm_stat(**changes):
    counters = {'Pages free': 2*GiB//16384, 'Pages inactive': 60*GiB//16384,
                'Pages speculative': GiB//16384, 'File-backed pages': 8*GiB//16384,
                'Pages purgeable': 4*GiB//16384, 'Pages occupied by compressor': 0, 'Swapouts': 0}
    counters.update(changes)
    return 'Mach Virtual Memory Statistics: (page size of 16384 bytes)\n' + ''.join(
        f'{key}: {value}.\n' for key,value in counters.items() if value is not None)


def test_anonymous_inactive_is_not_headroom():
    sample = macos_vm_stat(vm_stat(), 128*GiB)
    assert sample['available'] == 10*GiB  # no speculative/purgeable double credit
    assert sample['purgeable'] == 4*GiB
    old = {**sample, 'available': 62*GiB, 'pressure': 1}
    Policy().preflight(old, 6*GiB)  # the old free+inactive estimate incorrectly passed
    with pytest.raises(GuardError, match='25 GiB'):
        Policy().preflight({**sample, 'pressure': 1}, 6*GiB)
    assert Policy().violation({**sample, 'pressure': 1}, sample) == 'available headroom below 20 GiB'


@pytest.mark.parametrize('counter', ['Pages free','Pages inactive','Pages speculative','File-backed pages',
                                    'Pages purgeable','Pages occupied by compressor','Swapouts'])
@pytest.mark.parametrize('value', [None, 'garbage', '-1'])
def test_macos_required_counters_fail_closed(counter, value):
    with pytest.raises(GuardError):
        macos_vm_stat(vm_stat(**{counter:value}),128*GiB)


@pytest.mark.parametrize('raw', [vm_stat()+'Pages free: 1.\n', vm_stat()+'Pages free: garbage\n', vm_stat().replace('16384 bytes','unknown bytes'),
                               vm_stat(**{'File-backed pages': 129*GiB//16384})])
def test_macos_ambiguous_or_impossible_counters(raw):
    with pytest.raises(GuardError):
        macos_vm_stat(raw,128*GiB)


@pytest.mark.parametrize('named', [False, True])
def test_companion_chat_template_roundtrip(tmp_path, named):
    model_dir = tmp_path/'workspace/model'
    model_dir.mkdir(parents=True)
    template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    templates = {'default':template, 'tool_use':template+'TOOLS', 'review.v2':template+'REVIEW'} if named else template
    tok = Qwen2Tokenizer(vocab={c:i for i,c in enumerate('abcdefghijklmnopqrstuvwxyz :')}, merges=[], chat_template=templates)
    tok.save_pretrained(model_dir)
    assert 'chat_template' not in json.loads((model_dir/'tokenizer_config.json').read_text())
    (model_dir/'config.json').write_text(json.dumps({'model_type':'qwen2','hidden_size':32,'num_attention_heads':4,
                                                   'num_key_value_heads':2,'num_hidden_layers':1,'vocab_size':2}))
    base = tmp_path/'tiny.gguf'
    writer = gguf.GGUFWriter(str(base),'qwen2')
    writer.add_tensor('blk.0.attn_q.weight',np.zeros((32,32),dtype=np.float32))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    meta = dict(kind='gguf',base_sha256=file_hash(base),gguf_path=str(base),model_dir='model')
    (model_dir.parent/'workspace.json').write_text(json.dumps(meta))
    ws = SimpleNamespace(dir=str(model_dir.parent), store=SimpleNamespace(path=str(base)),meta=meta)
    bundle = export_bundle(ws,tmp_path/'export')['path']
    restored = import_bundle(tmp_path/'restored',bundle)
    reloaded = AutoTokenizer.from_pretrained(restored.model_dir,local_files_only=True)
    assert reloaded.chat_template == tok.chat_template
    messages = [{'role':'user','content':'hello'}]
    for name in templates if named else [None]:
        for tokenize in (False, True):
            kwargs = dict(chat_template=name,tokenize=tokenize,add_generation_prompt=True)
            assert reloaded.apply_chat_template(messages,**kwargs) == tok.apply_chat_template(messages,**kwargs)
    with zipfile.ZipFile(bundle) as src:
        manifest = json.loads(src.read('manifest.json'))
        for path in ('model/chat_template.jinja', 'model/additional_chat_templates/tool_use.jinja') if named else ('model/chat_template.jinja',):
            assert manifest['files'][path] == hashlib.sha256(src.read(path)).hexdigest()
        with zipfile.ZipFile(tmp_path/'tampered.zip','w') as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                dst.writestr(item.filename,data+b'changed' if item.filename.endswith('.jinja') else data)
    with pytest.raises(ValueError,match='checksum mismatch'):
        import_bundle(tmp_path/'bad',tmp_path/'tampered.zip')
    (model_dir/'chat_template.jinja').unlink()
    (model_dir/'chat_template.jinja').symlink_to(tmp_path/'tiny.gguf')
    with pytest.raises(ValueError,match='Symlink'):
        export_bundle(ws,tmp_path/'symlink')


@pytest.mark.parametrize('name', ['model/chat_templates/../../escape.jinja','/model/chat_template.jinja',
                                'model/./chat_template.jinja','model//chat_template.jinja',
                                'model/chat_templates/nested/escape.jinja','model/chat_templates/evil.py',
                                'model\\chat_templates\\evil.jinja','patches/chat_template.jinja'])
def test_template_paths_are_narrow(name):
    assert not safe_member(name)


def test_qwen38_source_normalization_profile():
    from tokenizers import Tokenizer, models
    from wsp.gguf_store import tokenizer_profile
    tok=Tokenizer(models.WordLevel({'[UNK]':0,'café':1},unk_token='[UNK]'))
    store=SimpleNamespace(architecture='qwen35',field=lambda key: {'general.name':'Qwen3.8-27B','tokenizer.ggml.pre':'qwen35'}.get(key))
    config={}
    assert tok.encode('cafe\u0301').ids == [0]
    tokenizer_profile(store,tok,config)
    assert tok.encode('cafe\u0301').ids == tok.encode('café').ids == [1]
    assert 'NFC' in config['wsp_tokenizer_profile']
    store.field=lambda key: 'unknown'
    other=Tokenizer(models.WordLevel({'[UNK]':0,'café':1},unk_token='[UNK]'))
    tokenizer_profile(store,other,{})
    assert other.normalizer is None


def test_metal_driver_allocation_is_guarded():
    sample=dict(available=40*GiB,pressure=1,swap_used=0,swap_out=0,compressed=0,tree_rss=GiB,tree_footprint=GiB)
    assert Policy().violation({**sample,'gpu':{'active':True,'backend':'metal'}},sample)=='Metal allocation telemetry missing'
    gpu=dict(active=True,backend='metal',metal_allocated=17*GiB,metal_recommended=100*GiB)
    assert Policy().violation({**sample,'gpu':gpu},sample)=='process-tree memory exceeds budget'
