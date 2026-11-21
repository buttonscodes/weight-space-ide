"""Real-model acceptance, only runnable under wsp.memory_guard. Artifacts stay outside Git."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import torch
from fastapi.testclient import TestClient

from wsp.gguf_store import file_hash
from wsp.server import create_app
from wsp.session import Session
from wsp.workspace import Workspace
from wsp import delta as D, testsuite
from wsp.store import norm_index


def bounded_behavior(session, out, merge=False):
    """Bounded rank-one factual reinforcement, with held-out preservation probes.

    This is a locality experiment, not a claim to learn a new fact. Three fixed
    magnitudes at most; failures are evidence, not hidden by a success assertion.
    """
    w = session.ws
    for p in w.patches:
        w.set_patch(p.id, enabled=False)
    prompts = ['The capital of France is', 'The capital of Germany is',
               'Water freezes at', 'The sum of one and two is', 'A small bird flies over']
    layer = w.struct.n_layers-1
    hook = f'blocks.{layer}.mlp.hook_post'
    base, acts = [], []
    for prompt in prompts:
        logits, cache = w.runner.run(w.encode(prompt),'base',names=[hook,f'blocks.{layer}.hook_resid_post'])
        base.append(logits[0,-1].clone())
        acts.append(cache[hook][0,-1].clone())
        if len(base)==1:
            residual_rms = float(cache[f'blocks.{layer}.hook_resid_post'][0,-1].square().mean().sqrt())
    head = 'lm_head.weight' if 'lm_head.weight' in w.store else 'model.embed_tokens.weight'
    target, contrast = w.tok.encode(' Paris')[-1], w.tok.encode(' London')[-1]
    direction = w.read(head,(target,slice(None)),'base')-w.read(head,(contrast,slice(None)),'base')
    direction /= direction.norm()
    # Orthogonalize the target key against three calibration prompts. The final
    # unrelated prompt is held out, so preservation is not only fitted behavior.
    keys = torch.stack(acts[1:4]).T
    q = torch.linalg.qr(keys,mode='reduced').Q
    key = acts[0]-q@(q.T@acts[0])
    key /= torch.dot(key,acts[0]).clamp_min(1e-8)
    name = f'model.layers.{layer}.mlp.down_proj.weight'
    region = norm_index((),w.shape(name))
    trials, selected = [], None
    for magnitude in (.03,.1,.3):
        piece = D.lowrank(region,direction*magnitude*residual_rms,key)
        results = [w.runner.run(w.encode(p),'current',extra={name:[piece]})[0][0,-1] for p in prompts]
        kls = [float(torch.nn.functional.kl_div(x.log_softmax(-1),b.log_softmax(-1),log_target=True,reduction='sum')) for x,b in zip(results,base)]
        gain = float((results[0][target]-results[0][contrast])-(base[0][target]-base[0][contrast]))
        record = dict(magnitude=magnitude,target_margin_gain=gain,preservation_kl=kls[1:],
                      unrelated_top1_preserved=[int(x.argmax())==int(b.argmax()) for x,b in zip(results[1:],base[1:])])
        trials.append(record)
        selected = piece
        if gain >= .1 and max(kls[1:]) <= .01:
            break
    session.task['preserve'] = {'max_kl':.01}
    rank = session.commit_deltas({name:[selected]},'Bounded factual reinforcement rank-one experiment','acceptance',{'kind':'lowrank'})
    assert rank['committed'] and any(c['kinds']==['lowrank'] for c in rank['changed'])
    rank_logits = [w.logits(p).detach() for p in prompts]
    cols = acts[0].abs().topk(3).indices
    rows = direction.abs().topk(3).indices
    sparse = D.Piece('sparse',region,coords=torch.stack((rows,cols),dim=1),
                     values=.03*residual_rms*direction[rows].sign()/acts[0][cols])
    sp = session.commit_deltas({name:[sparse]},'Sparse internal-layer interoperability probe','acceptance',{'kind':'sparse'})
    assert sp['committed']
    combined = w.logits(prompts[0]).detach()
    sparse_change = float((combined-rank_logits[0]).abs().max())
    assert sparse_change > 0
    w.set_patch(sp['patch'],enabled=False)
    torch.testing.assert_close(w.logits(prompts[0]),rank_logits[0],atol=1e-6,rtol=0)
    w.set_patch(sp['patch'],enabled=True)
    suite = dict(name='bounded-locality',cases=[
        dict(name='France target',type='next_token',prompt=prompts[0],expect=' Paris'),
        *[dict(name=f'preserve {i}',type='preserve',prompt=p,max_kl=.01) for i,p in enumerate(prompts[1:])]])
    result = testsuite.run_suite(w,suite)
    testsuite.save_suite(w,'bounded-locality',suite)
    overlay = [w.logits(p).detach()[-1].clone() for p in prompts]
    from wsp.companion import export_bundle, import_bundle
    bundle = export_bundle(w,out/'internal-overlays')['path']
    w.runner.release(); w._runner=None; gc.collect()
    reloaded = import_bundle(str(out/'behavior-reload'),bundle)
    torch.testing.assert_close(reloaded.logits(prompts[0]),combined,atol=1e-6,rtol=0)
    reloaded.runner.release(); reloaded._runner=None; gc.collect()
    merged_evidence = None
    if merge:
        from wsp.gguf_export import export_merged
        from wsp.gguf_store import open_gguf
        merged_evidence = export_merged(w,out/'internal-standalone.gguf')
        standalone = open_gguf(str(out/'internal-standalone-reload'),merged_evidence['path'])
        exported = [standalone.logits(p,'base')[-1].detach().clone() for p in prompts]
        cases = [testsuite.run_case(standalone,suite['cases'][0],'base')]
        for case,x,b in zip(suite['cases'][1:],exported[1:],base[1:]):
            kl = max(0.,float(torch.nn.functional.kl_div(x.log_softmax(-1),b.log_softmax(-1),log_target=True,reduction='sum')))
            cases.append({**case,'value':kl,'pass':kl<=case['max_kl'],'top1_preserved':int(x.argmax())==int(b.argmax())})
        merged_evidence.update(suite=dict(name=suite['name'],cases=cases,passed=sum(c['pass'] for c in cases),total=len(cases)),
            comparison_base='original immutable GGUF logits, not the exported checkpoint itself',
            target_margin_base=float(base[0][target]-base[0][contrast]),
            target_margin_overlay=float(overlay[0][target]-overlay[0][contrast]),
            target_margin_exported=float(exported[0][target]-exported[0][contrast]),
            retains_minimum_target_gain=float((exported[0][target]-exported[0][contrast])-(base[0][target]-base[0][contrast]))>=.1,
            max_last_logit_error_vs_overlay=max(float((x-y).abs().max()) for x,y in zip(exported,overlay)),
            kl_vs_overlay=[max(0.,float(torch.nn.functional.kl_div(x.log_softmax(-1),y.log_softmax(-1),log_target=True,reduction='sum'))) for x,y in zip(exported,overlay)],
            edited_tensor_type=w.store.dtype(name))
        standalone.runner.release(); standalone._runner=None; gc.collect()
    return dict(intent='reinforce an existing fact; not demonstrated knowledge acquisition',
                residual_rms=residual_rms,
                trials=trials,rank_commit=rank,sparse_commit=sp,sparse_max_logit_change=sparse_change,
                suite=result,held_out_prompt=prompts[-1],combined_companion_reload_error=0.,
                locality_pass=trials[-1]['target_margin_gain']>=.1 and result['passed']==result['total'],
                merged_export=merged_evidence)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--behavior', action='store_true')
    ap.add_argument('--merged', action='store_true')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    s = Session(str(out/'workspaces'))
    client = TestClient(create_app(s, 'http://testserver'), raise_server_exceptions=False)
    def post(route, body):
        response = client.post('/api/'+route, json=body)
        if response.status_code != 200:
            raise AssertionError((route, response.status_code, response.text))
        return response.json()
    post('open', dict(name=args.model))
    w = s.ws
    assert w.kind == 'gguf' and 0 < w.n_params() < 9_000_000_000
    assert not w.runner_loaded
    before_hash = file_hash(args.model)
    report = dict(repo=args.repo, revision=args.revision, artifact_sha256=before_hash,
                  logical_stored_parameters=w.n_params(), tensor_types=w.summary()['capabilities']['tensor_types'],
                  versions={p:importlib.metadata.version(p) for p in ('torch','transformers','gguf','numpy')},
                  context=512, generation_limit=8, seed=0, batch=1)
    # A real, explicit bounded calibration corpus; no surprise sampled generation.
    Path(w.dir, 'corpus.txt').write_text(('The capital of France is Paris. Water freezes at zero degrees Celsius. '
                                       'A small bird flies over a green tree. The sum of one and two is three. ')*4)
    prompt = 'The capital of France is'
    name = 'lm_head.weight' if 'lm_head.weight' in w.store else 'model.embed_tokens.weight'
    read = w.read(name, (slice(0,2), slice(0,32)), 'base')
    assert read.shape == (2,32) and torch.isfinite(read).all()
    t = time.monotonic()
    base = w.logits(prompt, 'base').detach()
    report['first_logits_seconds'] = time.monotonic()-t
    report['residency'] = w.runner.residency
    chat_messages = [{'role':'user','content':'Say hello.'}]
    chat_prompt = w.runner.chat_prompt(chat_messages)
    assert chat_prompt, 'Acceptance requires the source chat template'
    report['chat_portability'] = dict(prompt_sha256=hashlib.sha256(chat_prompt.encode()).hexdigest(),
        template_files={p.relative_to(w.model_dir).as_posix():file_hash(p)
                        for p in Path(w.model_dir).rglob('*.jinja')})
    # No dequantized matrix parameters may exist, including embedding/head.
    assert all(p.ndim <= 1 or ('.conv1d.' in n and p.numel()<=65536) for n,p in w.runner.model.named_parameters())
    top = int(base[-1].argmax())
    target = w.tok.encode(' London')[-1]
    if target == top:
        target = w.tok.encode(' Rome')[-1]
    report.update(base_top_id=top, target_id=target, base_target_logit=float(base[-1,target]))
    script = f'P("{name}")[{target}, :] += 4 * P("{name}")[{top}, :]'
    edit = post('edit', dict(script=script, message='Acceptance: promote selected city token', enforce_scope=True))
    assert edit['ok'] and edit['committed'], edit
    current = w.logits(prompt).detach()
    assert float(current[-1,target]-base[-1,target]) > 1
    assert int(current[-1].argmax()) == target
    report.update(edited_target_logit=float(current[-1,target]), edited_top_id=int(current[-1].argmax()),
                  max_logit_change=float((current-base).abs().max()), edit=edit)
    t = time.monotonic()
    run = post('run', dict(prompt=prompt, max_new_tokens=8, seed=0))
    report['api_run_seconds'] = time.monotonic()-t
    report['generation_benchmarks'] = {}
    for variant in ('base','current'):
        started = time.monotonic()
        tokens = w.runner.generate(w.encode(prompt),variant,max_new_tokens=8,seed=0)
        seconds = time.monotonic()-started
        report['generation_benchmarks'][variant] = dict(tokens=len(tokens),seconds=seconds,tokens_per_second=len(tokens)/seconds,
                                                       **getattr(w.runner,'last_generation',{}))
    report['mechanics_edit_qualification'] = 'Intentionally destructive plumbing test, not localized knowledge editing'
    report['run'] = run
    assert run['base']['next'][0][0] != run['current']['next'][0][0]
    for enabled, expected in [(False,base),(True,current)]:
        post('patch', dict(action='set', id=edit['patch'], enabled=enabled))
        torch.testing.assert_close(w.logits(prompt), expected, atol=1e-6, rtol=0)
    w.set_patch(edit['patch'], strength=.5)
    middle = w.logits(prompt)
    assert float((middle-base).abs().max()) > 0
    w.set_patch(edit['patch'], strength=1.)
    # Check existing intervention hooks still affect the same mapped neurons.
    _, cache = w.runner.run(w.encode(prompt), names=['blocks.0.hook_resid_post'])
    assert cache['blocks.0.hook_resid_post'].shape[-1] == w.struct.d_model
    intervention = w.runner.run(w.encode(prompt), 'base', hooks={'blocks.0.hook_mlp_out':lambda x,_:torch.zeros_like(x)})[0][0]
    assert float((intervention-base).abs().max()) > .001
    response = client.get('/api/export')
    assert response.status_code == 200, response.text
    bundle = out/'edited.wsppatch.zip'; bundle.write_bytes(response.content)
    report['companion'] = str(bundle)
    report['merged_export'] = 'unsupported; companion carries FP32 patches and original-base SHA-256'
    original_dir = w.dir
    w.runner.release(); w._runner = None; gc.collect()
    reopened = Workspace.load(original_dir)
    torch.testing.assert_close(reopened.logits(prompt), current, atol=1e-6, rtol=0)
    reopened.runner.release(); reopened._runner = None; del reopened; gc.collect()
    post('open', dict(name=str(bundle)))
    assert s.ws.runner.chat_prompt(chat_messages) == chat_prompt
    report['chat_portability']['companion_prompt_identical'] = True
    torch.testing.assert_close(s.ws.logits(prompt), current, atol=1e-6, rtol=0)
    torch.testing.assert_close(s.ws.logits(prompt,'base'), base, atol=1e-6, rtol=0)
    if args.behavior:
        report['bounded_behavior'] = bounded_behavior(s,out,merge=args.merged)
    if s.ws._runner:
        s.ws.runner.release(); s.ws._runner = None; gc.collect()
    if args.merged:
        from wsp.gguf_export import export_merged
        from wsp.gguf_store import open_gguf
        merged = export_merged(w,out/'edited-standalone.gguf')
        standalone = open_gguf(str(out/'standalone-reload'),merged['path'])
        logits = standalone.logits(prompt,'base').detach()
        assert torch.isfinite(logits).all()
        merged.update(max_logit_error=float((logits-current).abs().max()),
                      next_token_preserved=int(logits[-1].argmax())==int(current[-1].argmax()),
                      last_token_kl=float(torch.nn.functional.kl_div(logits[-1].log_softmax(-1),current[-1].log_softmax(-1),
                                          log_target=True,reduction='sum')))
        report['merged_export'] = merged
        standalone.runner.release(); standalone._runner=None; gc.collect()
    assert file_hash(args.model) == before_hash
    report.update(immutable_base=True, undo_redo=True, patch_scaling=True, save_reopen=True,
                  companion_reload=True, companion_logit_error=0., trace_intervention=True, runner_released=True)
    torch.save(dict(base=base,current=current), out/'logits.pt')
    (out/'evidence.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
