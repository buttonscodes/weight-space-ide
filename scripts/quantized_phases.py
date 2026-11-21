"""Sequential acceptance with process exit between original/companion/merged loads.

Small models need no permit. A >=9B model requires an operator-created supervised
permit; this script never creates approval. Use only with explicit model consent.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import torch
from fastapi.testclient import TestClient
from wsp import delta as D, testsuite
from wsp.companion import import_bundle, export_bundle
from wsp.gguf_store import GGUFStore, file_hash, open_gguf
from wsp.gguf_export import export_merged
from wsp.memory_guard import GiB, GuardError, supervise, supervised_permit, SUPERVISED_SHA256, stage
from wsp.server import create_app
from wsp.session import Session
from wsp.store import norm_index
from wsp.workspace import Workspace

PROMPTS = ['The capital of France is', 'The capital of Germany is', 'Water freezes at', 'A small bird flies over']
MESSAGES = [{'role':'user','content':'Say hello.'}]


def write(path, data):
    path.write_text(json.dumps(data,indent=2))


def close(w):
    if w is not None and w._runner is not None:
        w._runner.release(); w._runner=None
    gc.collect()


def generate(w, prompt, variant='current'):
    ids=w.encode(prompt);started=time.monotonic()
    tokens=w.runner.generate(ids,variant,max_new_tokens=4,seed=0)
    elapsed=time.monotonic()-started
    return dict(ids=tokens,text=w.tok.decode(tokens),seconds=elapsed,
                tokens_per_second=len(tokens)/elapsed,**w.runner.last_generation)


def logits(w,variant='current'):
    return [w.logits(p,variant)[-1].detach().clone() for p in PROMPTS]


def metrics(values, base, overlay, target, contrast):
    kl=lambda a,b:max(0.,float(torch.nn.functional.kl_div(a.log_softmax(-1),b.log_softmax(-1),log_target=True,reduction='sum')))
    margin=lambda a:float(a[target]-a[contrast])
    return dict(target_margin=margin(values[0]),target_gain_vs_base=margin(values[0])-margin(base[0]),
        target_gain_overlay=margin(overlay[0])-margin(base[0]),
        preservation_kl=[kl(a,b) for a,b in zip(values[1:],base[1:])],
        unrelated_top1_preserved=[int(a.argmax())==int(b.argmax()) for a,b in zip(values[1:],base[1:])],
        kl_vs_overlay=[kl(a,b) for a,b in zip(values,overlay)],
        max_logit_error_vs_overlay=max(float((a-b).abs().max()) for a,b in zip(values,overlay)))


def original(a,out):
    s=Session(str(out/'workspaces'));client=TestClient(create_app(s,'http://testserver'))
    def post(route,body):
        r=client.post('/api/'+route,json=body);assert r.status_code==200,(route,r.text)
        return r.json()
    post('open',{'name':a.model});w=s.ws
    report=dict(phase='original',model=a.model,artifact_sha256=w.meta['base_sha256'],parameters=w.n_params(),
        tensor_types=w.summary()['capabilities']['tensor_types'],seed=0,batch=1,
        context=int(os.environ['WSP_MAX_CTX']),generation_limit=4,
        versions={p:importlib.metadata.version(p) for p in ('torch','transformers','gguf','numpy')})
    try:
        assert not w.runner_loaded
        name=f'model.layers.{w.struct.n_layers-1}.mlp.down_proj.weight'
        assert w.read(name,(slice(0,2),slice(0,32))).shape==(2,32)
        Path(w.dir,'corpus.txt').write_text(('The capital of France is Paris. Water freezes at zero degrees Celsius. '
                                         'A small bird flies over a green tree. The sum of one and two is three. ')*4)
        stage('first four-token smoke',60)
        report['completion_base']=generate(w,PROMPTS[0],'base')
        write(out/'first-smoke.json',report['completion_base'])
        stage('chat and edit acceptance')
        # Stop an impractical initial path before the longer edit/export phase.
        if report['completion_base']['seconds']>60:
            raise RuntimeError('Four-token smoke exceeded60s; interactive performance remains incomplete')
        chat=w.runner.chat_prompt(MESSAGES,enable_thinking=False)
        report['chat_base']=generate(w,chat,'base')
        report['chat_prompt']=chat
        report['chat_template_sha256']={p.relative_to(w.model_dir).as_posix():file_hash(p) for p in Path(w.model_dir).rglob('*.jinja')}
        report['residency']=w.runner.residency
        assert all(p.ndim<=1 or ('.conv1d.' in n and p.numel()<=65536) for n,p in w.runner.model.named_parameters())
        base=logits(w,'base');target=w.tok.encode(' Paris')[-1];contrast=w.tok.encode(' London')[-1]
        layer=w.struct.n_layers-1;hook=f'blocks.{layer}.mlp.hook_post'
        acts=[]
        for prompt in PROMPTS:
            _,cache=w.runner.run(w.encode(prompt),'base',names=[hook,f'blocks.{layer}.hook_resid_post'])
            acts.append(cache[hook][0,-1])
            if len(acts)==1:rms=float(cache[f'blocks.{layer}.hook_resid_post'][0,-1].square().mean().sqrt())
        head='lm_head.weight' if 'lm_head.weight' in w.store else 'model.embed_tokens.weight'
        direction=w.read(head,(target,slice(None)),'base')-w.read(head,(contrast,slice(None)),'base')
        direction/=direction.norm()
        q=torch.linalg.qr(torch.stack(acts[1:3]).T,mode='reduced').Q
        key=acts[0]-q@(q.T@acts[0]);key/=torch.dot(key,acts[0]).clamp_min(1e-8)
        region=norm_index((),w.shape(name))
        suite=dict(name='supervised-locality',cases=[dict(name='France target',type='next_token',prompt=PROMPTS[0],expect=' Paris'),
            *[dict(name=f'preserve{i}',type='preserve',prompt=p,max_kl=.01) for i,p in enumerate(PROMPTS[1:])]])
        testsuite.save_suite(w,'supervised-locality',suite)
        s.set_task(dict(title='Bounded factual reinforcement',scope=[name],suite='supervised-locality',preserve={'max_kl':.01}))
        rejected=s.apply(f'P("{head}")[0,0] += 0.001',message='Scope rejection probe')
        assert not rejected['ok'] and not w.patches
        rank=D.lowrank(region,direction*.3*rms,key)
        commits=[s.commit_deltas({name:[rank]},'Rank-one factual reinforcement','acceptance',{'kind':'lowrank'})]
        assert commits[-1]['committed']
        rank_value=w.logits(PROMPTS[0])[-1].detach().clone()
        cols=acts[0].abs().topk(3).indices;rows=direction.abs().topk(3).indices
        sparse=D.Piece('sparse',region,coords=torch.stack((rows,cols),dim=1),values=.03*rms*direction[rows].sign()/acts[0][cols])
        commits.append(s.commit_deltas({name:[sparse]},'Sparse internal probe','acceptance',{'kind':'sparse'}))
        assert commits[-1]['committed']
        sparse_value=w.logits(PROMPTS[0])[-1].detach().clone()
        assert float((sparse_value-rank_value).abs().max())>1e-6
        row=int(rows[0]);col=int(cols[0]);amount=float(.01*rms*direction[row].sign()/acts[0][col])
        commits.append(post('edit',dict(script=f'P("{name}")[{row}:{row+1}, {col}:{col+1}] += {amount!r}',message='Block internal probe',enforce_scope=True)))
        assert commits[-1]['committed']
        overlay=logits(w)
        assert float((overlay[0]-sparse_value).abs().max())>1e-6
        for c in commits:post('patch',dict(action='set',id=c['patch'],enabled=False))
        for actual,expected in zip(logits(w),base):torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
        assert generate(w,PROMPTS[0])['ids']==report['completion_base']['ids']
        for c in commits:post('patch',dict(action='set',id=c['patch'],enabled=True))
        post('patch',dict(action='set',id=commits[0]['patch'],strength=.5))
        assert float((w.logits(PROMPTS[0])[-1]-overlay[0]).abs().max())>1e-6
        post('patch',dict(action='set',id=commits[0]['patch'],strength=1.))
        for actual,expected in zip(logits(w),overlay):torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
        _,cache=w.runner.run(w.encode(PROMPTS[0]),names=['blocks.0.hook_resid_post','blocks.3.attn.hook_pattern'])
        assert cache['blocks.0.hook_resid_post'].shape[-1]==w.struct.d_model
        changed=w.runner.run(w.encode(PROMPTS[0]),'base',hooks={'blocks.0.hook_mlp_out':lambda x,_:torch.zeros_like(x)})[0][0,-1]
        assert float((changed-base[0]).abs().max())>.001
        report.update(commits=commits,task=s.task,scope_rejection=rejected,suite=testsuite.run_suite(w,suite),
            behavior=metrics(overlay,base,overlay,target,contrast),completion_overlay=generate(w,PROMPTS[0]),
            chat_overlay=generate(w,chat),workspace=w.dir,undo_redo=True,scaling=True,trace_intervention=True,
            qualification='Factual reinforcement/locality probe, not demonstrated new knowledge; max_kl is advisory')
        torch.save(dict(base=base,overlay=overlay,target=target,contrast=contrast),out/'expected.pt')
        report['companion']=export_bundle(w,out/'edited')['path']
        close(w)
        report['merged']=export_merged(w,out/'edited-standalone.gguf')
        assert file_hash(a.model)==report['artifact_sha256']
        report['immutable_base']=True
        write(out/'original.json',report)
    finally:close(w)


def reload_phase(a,out):
    original_report=json.loads((out/'original.json').read_text())
    expected=torch.load(out/'expected.pt',weights_only=True)
    report=dict(phase=a.phase)
    w=None
    try:
        if a.phase=='companion':
            w=Workspace.load(original_report['workspace'])
            for actual,want in zip(logits(w),expected['overlay']):torch.testing.assert_close(actual,want,atol=1e-6,rtol=0)
            close(w);w=None;gc.collect()
            w=import_bundle(str(out/'companion-reload'),original_report['companion'])
        else:w=open_gguf(str(out/'merged-reload'),original_report['merged']['path'])
        values=logits(w)
        report['behavior']=metrics(values,**expected)
        report['completion']=generate(w,PROMPTS[0])
        chat=w.runner.chat_prompt(MESSAGES,enable_thinking=False)
        report['chat_prompt_identical']=chat==original_report['chat_prompt']
        assert report['chat_prompt_identical']
        report['chat']=generate(w,chat)
        target_case=testsuite.run_case(w,dict(name='France target',type='next_token',prompt=PROMPTS[0],expect=' Paris'),'current')
        cases=[target_case,*[dict(name=f'preserve{i}',type='preserve',prompt=prompt,value=kl,max_kl=.01,
            top1_preserved=top,**{'pass':kl<=.01}) for i,(prompt,kl,top) in enumerate(zip(PROMPTS[1:],
            report['behavior']['preservation_kl'],report['behavior']['unrelated_top1_preserved']))]]
        report['suite']=dict(name='supervised-locality',cases=cases,passed=sum(c['pass'] for c in cases),total=len(cases),
            comparison_base='original immutable GGUF logits, not exported checkpoint logits')
        if a.phase=='companion':
            assert report['behavior']['max_logit_error_vs_overlay']==0
            assert report['completion']['ids']==original_report['completion_overlay']['ids']
            assert report['chat']['ids']==original_report['chat_overlay']['ids']
            assert len(w.patches)==3 and len(w.meta.get('base_sha256',''))==64
            assert json.loads(Path(w.dir,'task.json').read_text())==original_report['task']
            assert all(p.meta()['source'] and 'drift' in p.meta()['info'] for p in w.patches)
            report['saved_reopen_exact']=True
        else:
            report['qualification']='Requantized standalone; rounding may erase target gains, no undo; preservation uses original base logits'
            report['retains_minimum_target_gain']=report['behavior']['target_gain_vs_base']>=.1
        write(out/(a.phase+'.json'),report)
        if a.phase=='merged' and not report['retains_minimum_target_gain']:
            raise RuntimeError('Standalone required target gain FAILED; use companion, do not treat exit/file validity as equivalence')
    finally:close(w)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    p.add_argument('--permit');p.add_argument('--phase',choices=['original','companion','merged'])
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(0)
    if a.phase:
        if a.phase=='original':original(a,out)
        else:reload_phase(a,out)
        return
    if list(out.iterdir()):raise GuardError('Choose a fresh evidence directory; preserve previous attempts')
    s=GGUFStore(a.model);count=sum(s.numel(n) for n in s.names());del s;gc.collect()
    if count>=9_000_000_000:
        if not a.permit:raise GuardError('READY_FOR_SUPERVISED_27B approval is required before starting')
        approval,_=supervised_permit(a.permit)
        if file_hash(a.model)!=SUPERVISED_SHA256:raise GuardError('Unapproved large artifact')
        forecast,timeout=approval['forecast_gib']*GiB,approval['timeout']
    else:
        if a.permit:raise GuardError('Use the unchanged small-model policy without a large permit')
        forecast,timeout=8*GiB,240
    # No parallel workers, no retries: teardown success is required before next load.
    for phase in ('original','companion','merged'):
        command=[sys.executable,__file__,'--model',a.model,'--out',str(out),'--phase',phase]
        supervise(command,out/phase/'watchdog.jsonl',int(forecast),timeout,permit_path=a.permit)
    write(out/'completed.json',dict(phases=['original','companion','merged'],all_owned_processes_exited=True))


if __name__=='__main__':main()
