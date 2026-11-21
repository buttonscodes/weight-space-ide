"""Explicit consent-gated, bounded standalone validation and small-model promotion.

Run only with authorization to load the selected model. Each baseline, export and
reload is a separate watchdog-owned process. Failed quality candidates are kept;
a safety/runtime failure stops the entire workflow. Large models require an exact
fresh supervised permit and ONE preselected candidate (no automatic large search).
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from wsp.export_quality import Criteria, evaluate, select_verified
from wsp.gguf_mixed import preview_export, export_mixed
from wsp.gguf_store import open_gguf, file_hash
from wsp.memory_guard import GiB, GuardError, Policy, forecast, supervise, stage, supervised_permit
from wsp.workspace import Workspace
from quantized_phases import PROMPTS, MESSAGES, logits, generate, close


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2)


def child(a):
    root=Path(a.out);plan=json.loads((root/'plan.json').read_text());criteria=Criteria(**plan['criteria'])
    os.environ['WSP_MAX_CTX']='64';os.environ['WSP_MAX_NEW_TOKENS']='4'
    torch.set_num_threads(4);torch.manual_seed(0)
    w=None
    try:
        if a.phase=='baseline':
            w=Workspace.load(plan['workspace'])
            stage('first four-token smoke',60)
            completion=generate(w,PROMPTS[0],'base');stage('fixed baseline suite')
            base=logits(w,'base');overlay=logits(w)
            target=w.tok.encode(' Paris')[-1];contrast=w.tok.encode(' London')[-1]
            chat=w.runner.chat_prompt(MESSAGES,enable_thinking=False)
            if not isinstance(chat,str):raise ValueError('Chat template missing; restore original GGUF tokenizer assets before validation')
            self_check=evaluate(base,overlay,overlay,target,contrast,criteria=criteria,chat_prompt_identical=True)
            if self_check['validation_status']!='VERIFIED':raise ValueError('Overlay itself fails predeclared quality criteria')
            torch.save(dict(base=base,overlay=overlay,target=target,contrast=contrast),root/'baseline.pt')
            write(root/'baseline.json',dict(artifact_sha256=w.meta['base_sha256'],completion=completion,
                chat_prompt=chat,chat=generate(w,chat),overlay_completion=generate(w,PROMPTS[0]),
                quality=self_check,residency=w.runner.residency))
        elif a.phase=='export':
            w=Workspace.load(plan['workspace']);assert not w.runner_loaded
            result=export_mixed(w,root/f'{a.precision}.gguf',a.precision)
            assert not w.runner_loaded
            write(root/f'{a.precision}.export.json',result)
        else:
            original=json.loads((root/'baseline.json').read_text())
            expected=torch.load(root/'baseline.pt',weights_only=True)
            candidate=Path(a.candidate_file) if a.candidate_file else root/f'{a.precision}.gguf'
            w=open_gguf(str(root/f'{a.precision}-workspace'),str(candidate))
            stage('first four-token smoke',60)
            completion=generate(w,PROMPTS[0]);stage('fixed candidate suite')
            values=logits(w);chat=w.runner.chat_prompt(MESSAGES,enable_thinking=False)
            result=evaluate(candidate=values,**expected,criteria=criteria,chat_prompt_identical=chat==original['chat_prompt'])
            result.update(artifact_sha256=w.meta['base_sha256'],base_sha256=original['artifact_sha256'],
                precision=a.precision,completion=completion,chat=generate(w,chat),
                plan_sha256=file_hash(root/'plan.json'),baseline_sha256=file_hash(root/'baseline.pt'))
            write(str(candidate)+'.validation.json',result)
            if result['validation_status']!='VERIFIED':return 3  # semantic failure, never structural success
        return 0
    finally:close(w)


def run(a, supervisor=supervise):
    out=Path(a.out)
    if out.exists():raise ValueError('Choose a fresh evidence directory; never overwrite a previous attempt')
    precisions=a.precisions.split(',')
    order=['SAME','Q8_0','F16','F32']
    if not precisions or len(precisions)>4 or any(p not in order for p in precisions) or sorted(set(precisions),key=order.index)!=precisions:
        raise ValueError('Use at most four unique increasing candidates: SAME,Q8_0,F16,F32')
    if not 0<a.total_seconds<=2400 or not 0<a.phase_seconds<=600 or not 0<a.max_disk_gib<=100:
        raise ValueError('Invalid candidate time/disk bounds')
    criteria=Criteria();w=Workspace.load(a.workspace,metadata_only=True)
    large=w.n_params()>=9_000_000_000
    if large and (not a.permit or len(precisions)!=1):raise GuardError('Large validation needs fresh consent permit and ONE justified candidate')
    if a.permit and not large:raise GuardError('Small models use unchanged budgets, without a large permit')
    plans={p:preview_export(w,p) for p in precisions}
    total_bytes=sum(x['output_bytes'] for x in plans.values())
    if total_bytes>a.max_disk_gib*GiB:raise ValueError('Candidate disk limit exceeded before export')
    if shutil.disk_usage(out.parent).free<total_bytes+GiB:raise ValueError('Insufficient disk for bounded candidate sequence')
    forecast_bytes=forecast(w,64 if large else 512)['total']
    largest=max(x['increase_bytes'] for x in plans.values())
    reservation=math.ceil((forecast_bytes+max(0,largest))/GiB)*GiB
    if reservation>16*GiB and not large:raise GuardError('Candidate forecast exceeds small-model16GiB budget')
    permit=None
    if large:
        permit,_=supervised_permit(a.permit)
        if reservation>permit['forecast_gib']*GiB:raise GuardError('Fresh permit reservation is below promoted forecast')
        reservation=int(permit['forecast_gib']*GiB)
    out.mkdir();shutil.copytree(a.workspace,out/'source-workspace')
    plan=dict(workspace=str(out/'source-workspace'),parameters=w.n_params(),base_sha256=w.meta['base_sha256'],
              **criteria.record(),prompts=PROMPTS,messages=MESSAGES,target=' Paris',contrast=' London',
              context=64,new_tokens=4,seed=0,precisions=precisions,previews=plans,
              reservation_bytes=reservation,max_disk_bytes=int(a.max_disk_gib*GiB),total_seconds=a.total_seconds)
    write(out/'plan.json',plan);del w;gc.collect();started=time.monotonic()
    def phase(name,precision=None):
        remaining=a.total_seconds-(time.monotonic()-started)
        if remaining<a.phase_seconds:raise GuardError('Total deadline cannot accommodate another bounded phase')
        command=[sys.executable,__file__,'--out',str(out),'--phase',name]
        if precision:command+=['--precision',precision]
        log=out/(f'{precision}-{name}' if precision else name)/'watchdog.jsonl'
        try:supervisor(command,log,reservation,a.phase_seconds,policy=Policy(swap_growth=0,compression_growth=0),permit_path=a.permit)
        except GuardError:
            # Only an explicitly recorded semantic rejection may advance SMALL
            # adaptive search. Never retry a safety/runtime/preflight failure.
            rows=[json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
            end=rows[-1] if rows else {}
            result_path=out/f'{precision}.gguf.validation.json'
            if (not large and name=='candidate' and end.get('event')=='teardown' and end.get('reason') is None
                    and end.get('returncode')==3 and end.get('remaining')==[] and result_path.exists()
                    and json.loads(result_path.read_text()).get('validation_status')=='FAILED'):
                return
            raise
    reports=[]
    try:
        phase('baseline')
        for p in precisions:
            phase('export',p);phase('candidate',p)
            report=json.loads((out/f'{p}.gguf.validation.json').read_text());reports.append(report)
            if report['validation_status']=='VERIFIED':
                write(out/'result.json',dict(status='VERIFIED',selected=p,candidates=reports,all_owned_processes_exited=True));return
        select_verified(reports)
    except Exception as e:
        write(out/'failure.json',dict(status='FAILED',reason=str(e),candidates=reports,recommendation='Use exact companion; do not weaken thresholds'))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace');p.add_argument('--out',required=True)
    p.add_argument('--precisions',default='Q8_0,F16,F32');p.add_argument('--max-disk-gib',type=float,default=8)
    p.add_argument('--total-seconds',type=int,default=1800);p.add_argument('--phase-seconds',type=int,default=180)
    p.add_argument('--permit');p.add_argument('--phase',choices=['baseline','export','candidate']);p.add_argument('--precision');p.add_argument('--candidate-file',help='Explicit existing standalone artifact for guarded candidate phase; no export/rewrite')
    a=p.parse_args()
    if a.phase:return child(a)
    if not a.workspace:p.error('--workspace required')
    run(a);return 0


if __name__=='__main__':sys.exit(main())
