"""Structured interactive-runtime outcomes, separate from legacy acceptance guards."""
from __future__ import annotations
import time
from .memory_guard import GuardError


def event(code, action, message, measured=None, limit=None, recommendations=()):
    return dict(code=code, action=action, message=message, measured=measured,
                limit=limit, recommendations=list(recommendations), time=time.time())


class RuntimeRefusal(GuardError):
    """No new allocation was authorized, or a running lease was revoked."""
    def __init__(self, record, tokens=None):
        self.record = record
        self.tokens = list(tokens or ())
        super().__init__(record['message'])


def unavailable(message='The memory watchdog is unavailable; this operation was not run.'):
    return RuntimeRefusal(event('watchdog_unavailable','refused',message,
        recommendations=['Check the server terminal and watchdog log.', 'Restore supervision before retrying.']))


# one forward pass (probe_next: two) over the longest text; probe_contrast runs each text in turn
STEERING_JOBS=('steer_calibrate','probe_reads','probe_fires','probe_contrast','probe_next')


def job_scope(name, *, runner=False):
    """Keep one lease over a public job and all its nested runner calls."""
    from functools import wraps
    from inspect import signature
    def decorate(fn):
        sig=signature(fn)
        @wraps(fn)
        def wrapped(*args, **kwargs):
            owner=args[0]
            ws=owner.ws if hasattr(owner,'ws') else owner
            client=getattr(ws,'_runtime_client',None)
            if client is None:return fn(*args,**kwargs)
            values=sig.bind_partial(*args,**kwargs);values.apply_defaults();values=values.arguments
            tokens=0;retained=False;fresh=False
            if runner:
                ids=values['ids']
                tokens=int(ids.shape[-1]) if hasattr(ids,'shape') else len(ids)
                if name=='generate':
                    from .capacity import output_count
                    tokens+=output_count(values['max_new_tokens'],tokens,owner.max_ctx)
                retained=bool(values.get('names') or values.get('hooks'))
                fresh=bool(values.get('names'))
            if name=='trace':
                tokens=min(len(ws.encode(values['prompt'])),values['max_tokens']);retained=True
            elif name=='patching':
                tokens=len(ws.encode(values['clean']));retained=True
            elif name in STEERING_JOBS:
                # Steering and probe jobs take token ids; a malformed argument is left for the job's
                # own check to refuse (with tokens 0 it still refuses before anything runs).
                try:
                    texts=list(values['a'])+list(values['b']) if name=='probe_contrast' else [values['ids']]
                    tokens=max(int(t.shape[-1]) if hasattr(t,'shape') else len(t) for t in texts)
                except (TypeError,ValueError,KeyError):
                    tokens=0
                retained=name!='probe_next' or values.get('steer') is not None
            elif name=='direction_residuals':
                # One forward pass per prompt, in turn, holding the named residual captures: sized by the longest.
                try:
                    tokens=max(int(t.shape[-1]) if hasattr(t,'shape') else len(t) for t in values['prompts_ids'])
                except (TypeError,ValueError,KeyError):
                    tokens=0
                retained=True
            result=None
            try:
                with ws.lock, client.operation(ws,name,tokens=tokens,retained=retained,fresh=fresh):
                    result=fn(*args,**kwargs)
            except RuntimeRefusal as exc:
                if getattr(exc,'committed',None) and isinstance(result,dict):
                    return {**result,**exc.report}
                raise
            return result
        return wrapped
    return decorate


def edit_preflight(ws, stored=0, scratch=0, *, cumulative=False):
    client=getattr(ws,'_runtime_client',None)
    if client is not None:client.preflight_edit(ws,stored,scratch,cumulative=cumulative)
