"""Single background export per IDE; never starts inference or validation."""
import json
from pathlib import Path
import secrets
import threading

def preview_export(*args, **kwargs):
    from .gguf_mixed import preview_export as implementation
    return implementation(*args, **kwargs)


def export_mixed(*args, **kwargs):
    from .gguf_mixed import export_mixed as implementation
    return implementation(*args, **kwargs)


class ExportJobs:
    def __init__(self):
        self.lock=threading.Lock();self.jobs={}

    def start(self,ws,precision='F16'):
        from .gguf_mixed import ExportCancelled
        preview=preview_export(ws,precision)
        with self.lock:
            if any(j['status']=='RUNNING' for j in self.jobs.values()):raise ValueError('An export is already running')
            if len(self.jobs)>=32:raise ValueError('Export job limit reached; reopen the isolated server after retaining reports')
            ident=secrets.token_hex(12);cancel=threading.Event()
            job=dict(id=ident,status='RUNNING',validation_status='UNVALIDATED',preview=preview,cancel=cancel)
            self.jobs[ident]=job
        def update(**values):
            with self.lock:job.update(values)
        def run():
            try:
                out=Path(ws.dir)/'exports'/f'edited-{precision.lower()}-{ident}.gguf'
                result=export_mixed(ws,out,precision,cancel=cancel,progress=lambda v:update(progress=v))
                update(status='COMPLETE',report=result)
            except ExportCancelled as e:update(status='CANCELLED',error=str(e))
            except Exception as e:update(status='FAILED',error=str(e))
        thread=threading.Thread(target=run,name='gguf-export',daemon=True);job['thread']=thread;thread.start()
        return self.get(ident)

    def get(self,ident):
        with self.lock:
            if ident not in self.jobs:raise ValueError('Unknown export job')
            result={k:v for k,v in self.jobs[ident].items() if k not in ('cancel','thread')}
        if result.get('report'):
            path=Path(result['report']['path']+'.validation.json')
            if path.exists():
                evidence=json.loads(path.read_text())
                if evidence.get('artifact_sha256')==result['report']['artifact_sha256']:
                    result['validation_status']=evidence.get('validation_status','UNVALIDATED')
                    result['validation']=evidence
        return result

    def cancel(self,ident):
        with self.lock:
            if ident not in self.jobs:raise ValueError('Unknown export job')
            self.jobs[ident]['cancel'].set()
        return self.get(ident)

    def shutdown(self):
        with self.lock:jobs=list(self.jobs.values())
        for j in jobs:j['cancel'].set()
        for j in jobs:j['thread'].join(timeout=5)
        if any(j['thread'].is_alive() for j in jobs):
            raise RuntimeError('Export cancellation did not finish within5s; partial output is not published')
