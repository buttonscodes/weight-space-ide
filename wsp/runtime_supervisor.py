"""Regular-serve external watchdog. Idle until an explicit GGUF operation requests a reservation.

This is application resource ownership, not an agent consent permit. Acceptance
harness leases/large-test permits remain a separate interface in memory_guard.
"""
from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .runtime_config import SUPERVISOR_LIVE, backend_status, check_backend, user_dir
from .memory_guard import GuardError
from .runtime_events import RuntimeRefusal, event, unavailable

SCRATCH_ADMISSION_QUANTUM = 64 * 1024**2
# Memory admission's charge per patch weight: the FP32 delta plus room to build and gather it (memory_forecast's patches*3).
PATCH_ADMISSION_BYTES = 12
ADVISE_SECONDS = 2.0            # an advisory answer, or the lack of one, is reused this long for the same workspace state
ADVISE_TIMEOUT_SECONDS = 1.5    # an advisory call waits at most this long in all (locks and reply), then has no answer
LAYER_CHECK_INTERVAL_SECONDS = .05
TELEMETRY_FAILURE_GRACE_SECONDS = 3
REQUEST_TIMEOUT_SECONDS = 8
# status() waits at most this long for the client lock, which a mailbox request can hold for 8 s.
STATUS_LOCK_SECONDS = .2
# The best-effort finish of a begin that timed out: short, so a late grant cannot hold the caller.
ABANDON_TIMEOUT_SECONDS = 2
# A saved settings change (RuntimeClient.update_settings) waits at most this long for the client lock, which a
# starting operation holds while it describes and begins its admission, then is refused with nothing changed.
SETTINGS_LOCK_SECONDS = 10
# A dry-run check of draft settings (the Settings preview) waits at most this long in all, then has no answer.
SETTINGS_CHECK_SECONDS = 1.5
# A settings request carries a deadline this far inside the client's wait. The supervisor checks it again
# immediately before it commits (after the slow work), so a reply normally arrives before the client stops waiting.
# A reply can still be late (a stalled process); the client then asks what the supervisor holds (held_settings).
SETTINGS_MARGIN_SECONDS = 1
# The effect of a saved change while an operation runs (settings_effect).
AFTER_OPERATION = ('A model operation is running. It keeps the memory limits it was admitted with; '
                   'the new limits and idle timeout apply when it finishes.')
# Idle conditions the escalation ladder debounces (runtime_policy.IDLE_UNSAFE_PERSIST_SECONDS) before an unload.
IDLE_DEBOUNCED = ('memory_pressure','headroom_floor')
GRANT_ACTIONS = ('bind','upgrade','begin','extend','spawn_worker')
COLD_ACTIONS = ('bind','upgrade','begin')
# AF_UNIX sun_path is 104 bytes on macOS; longer runtime paths use a private /tmp directory.
WORKER_SOCKET_LIMIT = 100
RELEASE_REAP_SECONDS = 5
WATCHDOG_FAILED_KILL_SECONDS = 10
WORKER_LOST_MESSAGE = 'The model worker was stopped; the model is unloaded. The IDE remains available.'
FRESH_ADMISSION = 'Run again to load the model through fresh admission.'
# Every signal the supervisor sends goes through _signal_server or _signal_worker
# with one of these literal reasons (tests/test_signal_sites.py pins each call
# site). Memory faults only ever reach the worker; the server is signalled only
# for a user-requested shutdown or to sweep the group of a server that already exited.
SIGNAL_SITES = {
    'user_shutdown': 'server',
    'shutdown_forced': 'server',
    'server_exited_cleanup': 'server',
    'kill_worker': 'worker',
    'worker_exited': 'worker',
    'release_reap': 'worker',
    'spawn_rollback': 'worker',
    'watchdog_failed': 'worker',
    'sampling_failed': 'worker',
    'teardown': 'worker',
}


class SettingsUnsettled(RuntimeRefusal):
    """A settings request whose outcome is not the settings asked for nor the previous ones: ``held`` is what the
    supervisor runs (settings, adopted by the client), or None when that is not known. Made by unsettled()."""
    held=None


def unsettled(held, message):
    # A factory, not an __init__ override: tests/test_signal_sites.py reads dunder access in this module as reflection.
    refusal=SettingsUnsettled(event('settings_unsettled','refused',message,
        recommendations=['Save again to apply known values, or restart the server.']))
    refusal.held=held
    return refusal


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def policy_for(values, total, available=None):
    from .memory_guard import Policy, GiB
    def size(key, fraction, minimum=0):
        v = values[key]
        return int(v*GiB) if v is not None else max(minimum, int(total*fraction))
    floor = size('floor_gib', .10, 512*1024**2)
    preferred = size('preferred_gib', .15, 768*1024**2)
    # Auto budgets are measured at admission; operation limits remain fixed until finish.
    ceiling = int(values['ceiling_gib']*GiB) if values['ceiling_gib'] is not None else total-preferred
    budget = int(values['budget_gib']*GiB) if values['budget_gib'] is not None else min(ceiling, max(1, (total if available is None else available)-preferred))
    if not 0 < floor <= preferred or not 0 < budget <= ceiling or preferred+budget > total:
        raise ValueError('Memory policy requires 0 < floor <= preferred, 0 < budget <= ceiling, and preferred + budget <= physical RAM')
    return Policy(floor=floor, preferred=preferred, budget=budget, ceiling=ceiling,
                  swap_growth=int(values['swap_growth_mib']*1024**2),
                  compression_growth=int(values['compression_growth_mib']*1024**2))


def idle_watch_policy(values, total):
    """The limits the idle watch checks a reserved model against: the memory policy of ``values`` at physical
    RAM ``total``, with admission's static total cap (runtime_policy.admission) as both budget and ceiling."""
    from dataclasses import replace
    from .memory_guard import GiB
    policy=policy_for(values,total)
    cap=min([*(int(values[k]*GiB) for k in ('budget_gib','ceiling_gib') if values[k] is not None),total-policy.preferred])
    return replace(policy,budget=cap,ceiling=cap)


def load_config(settings, selected):
    """The settings a model load is fixed with: the loaded model keeps them until it is unloaded."""
    v=settings['values']
    return dict(context=v['context'],threads=v['threads'],backend=v['backend'],selected=selected)


def descriptor(ws, context, loading=False, forecast_value=None, backend=None):
    from .capacity import context_limit
    p = Path(ws.store.path).resolve()
    try:st=p.stat()
    except OSError as exc:
        raise RuntimeRefusal(event('model_changed','refused','The model source is missing or unreadable; unload and reopen it.',recommendations=['Restore the model file, then unload and reopen the workspace.'])) from exc
    model_context = getattr(ws.struct, 'n_ctx', None)
    effective = context_limit(context, model_context)
    chosen = forecast_value if forecast_value is not None else reservation_forecast(ws,effective)
    return dict(path=str(p), stat=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns],
                base_hash=ws.meta.get("base_sha256"), params=ws.n_params(),
                backend=backend if backend is not None else os.environ.get('WSP_GGML_DEVICE','cpu'),
                configured_context=context, model_context=model_context, context=effective,
                workspace_version=getattr(ws,'version',0), runner=loading or runner_loaded(ws), forecast=chosen)


def runner_loaded(ws):
    # A runner whose model worker was lost (RemoteRunner.lost) holds no model.
    runner=getattr(ws,'_runner',None)
    return runner is not None and getattr(runner,'lost',None) is None


def reservation_forecast(ws, context):
    from .memory_guard import forecast
    from .capacity import context_limit
    return forecast(ws,context_limit(context,getattr(ws.struct,'n_ctx',None)),operation='interactive')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


class RuntimeClient:
    def __init__(self, directory, settings, worker=False):
        # worker: this server runs its runner in a supervisor-spawned model worker
        # (``wsp.worker_jobs.uses_worker``); its load descriptors carry worker: true.
        self.directory=Path(directory);self.settings=settings;self.worker=bool(worker)
        self.lock=threading.RLock();self.request_lock=threading.RLock()
        self.operation_lock=threading.RLock()
        self.binding=None;self.reserved=False;self.error=None;self.last_event=None
        self.operation_id=self.operation_owner=self.operation_version=None
        self._forecast_cache=None
        self._workspace=None;self._control_thread=None
        self._prospective_stored=0
        self._scratch_allowance=0;self._scratch_remaining=0;self._forward_allowances={};self._last_lease_check=None
        self._committed=[];self._commit_report=None
        self._advice=None;self._advice_lock=threading.Lock()     # advise_edit's last answer, or lack of one
        # The backend activate() selected for this server; after a saved change, the one the supervisor selected.
        self._selected=os.environ.get('WSP_GGML_DEVICE','cpu')
        # The load settings (load_config) fixed while an operation runs or a model is loaded; None otherwise.
        # The supervisor keeps the same pin, so descriptors for the loaded model match it after a saved change.
        self._pinned=None
        self._settings_generation=0     # counts applied settings changes; an advisory answer across one is dropped

    def config(self):
        """Load settings for the next descriptor: the pinned ones, else the saved ones."""
        return self._pinned or load_config(self.settings,self._selected)

    def loaded_config(self, ws=None):
        """The settings the loaded model was loaded with, or None when no model is loaded."""
        ws=ws if ws is not None else self._workspace
        return dict(self._pinned) if self._pinned is not None and ws is not None and runner_loaded(ws) else None

    def _describe(self, ws, loading=False, forecast_value=None):
        cfg=self.config()
        return descriptor(ws,cfg['context'],loading,forecast_value=forecast_value,backend=cfg['selected'])

    def _request(self, action, value=None, timeout=None):
        # Single mailbox, including requests from the independent unload thread.
        with self.request_lock:
            nonce=os.urandom(16).hex()
            write(self.directory/'request.json',dict(id=nonce,action=action,descriptor=value,
                  previous_binding=self.binding if action=='upgrade' else None))
            deadline=time.monotonic()+(REQUEST_TIMEOUT_SECONDS if timeout is None else timeout)
            while time.monotonic()<deadline:
                try:
                    reply=json.loads((self.directory/'reply.json').read_text())
                    if reply['id']==nonce:
                        if not reply['ok']:
                            record=reply.get('event') or event('admission_refused','refused',reply['reason'])
                            self.last_event=record
                            raise RuntimeRefusal(record)
                        return reply
                except FileNotFoundError:pass
                time.sleep(.025)
            refusal=unavailable('The memory watchdog did not respond; this operation was not started.')
            refusal.timed_out=True
            raise refusal

    def _abandon(self, ws, operation_id):
        # A begin the supervisor granted after this client gave up must not stay
        # running and refuse every later begin with operation_busy.
        try:actual=self._describe(ws,forecast_value=self._forecast(ws))
        except Exception:actual=None
        try:self._request('finish',dict(operation_id=operation_id,event=None,actual=actual,success=False,abandoned=True),
                          timeout=ABANDON_TIMEOUT_SECONDS)
        except Exception:pass

    def _check(self, binding):
        # Constant-size lease only: no patch scan, forecast or model descriptor.
        try:
            lease=json.loads((self.directory/'lease.json').read_text())
            if not lease.get('ok'):
                raise RuntimeRefusal(lease.get('event') or event('lease_revoked','abort','The operation lease was revoked.'))
            if lease['binding']!=binding or lease['pid']!=os.getpid() or not 0<=time.time()-lease['time']<4:
                raise unavailable()
            self._last_lease_check=(binding,time.monotonic())
        except RuntimeRefusal as exc:
            self._last_lease_check=None;self.last_event=exc.record;raise
        except (OSError,ValueError,KeyError) as exc:
            self._last_lease_check=None
            refusal=unavailable();self.last_event=refusal.record
            raise refusal from exc

    def layer_checkpoint(self, ws):
        """Only extra native-layer checks may reuse a <=50ms successful lease read."""
        with self.lock:
            if self.operation_id is None or getattr(ws,'version',0)!=self.operation_version:
                return self.require(ws)
            previous=self._last_lease_check
            if previous and previous[0]==self.binding and 0<=time.monotonic()-previous[1]<LAYER_CHECK_INTERVAL_SECONDS:
                return
            self._check(self.binding)

    @staticmethod
    def _forward_kind(name, retained):
        return 'trace' if retained else ('forward' if name=='forward' else 'interactive')

    def attach(self, ws):
        self._workspace=ws
        if self._control_thread is None:
            self._control_thread=threading.Thread(target=self._controls,daemon=True,name='runtime-unload')
            self._control_thread.start()

    def drop_lost(self, ws):
        """Drop a runner whose model worker was lost, unless it is serving its own operation.

        Outside any operation the lost reservation is released too, so the next
        load is admitted afresh (and the supervisor reaps a worker still running).
        -> whether a runner was dropped."""
        with ws.lock:
            runner=getattr(ws,'_runner',None)
            if runner is None or getattr(runner,'lost',None) is None:return False
            if self.operation_id is not None and self.operation_id==runner.lost_in:return False
            with runner.lock:runner.release()
            ws._runner=None
            if self.operation_id is None:self.release()
            return True

    def _worker_lost(self, ws, command):
        # control.json worker_lost: idle, so the runner it names is dropped and the reservation released.
        from .remote_runner import RemoteRunner
        runner=ws._runner
        if isinstance(runner,RemoteRunner) and (runner.lost is not None or runner.pid==command.get('worker_pid')):
            if isinstance(command.get('event'),dict):runner.mark_lost(command['event'])
            else:runner.mark_lost(event('worker_exited','worker_killed',WORKER_LOST_MESSAGE,recommendations=[FRESH_ADMISSION]))
            self.drop_lost(ws)
            self.last_event=command.get('event') or runner.lost

    def _drop_idle_lost(self):
        # A loss the RemoteRunner detected itself (no supervisor record): once its
        # operation has ended, release, so the supervisor reaps that worker now.
        ws=self._workspace
        runner=getattr(ws,'_runner',None) if ws is not None else None
        if runner is None or getattr(runner,'lost',None) is None or self.operation_id is not None:return
        if ws.lock.acquire(timeout=.1):
            try:
                if self.operation_id is None:self.drop_lost(ws)
            finally:ws.lock.release()

    def _controls(self):
        handled=None
        while True:
            try:self._drop_idle_lost()
            except Exception as exc:
                self.last_event=event('unload_failed','unload',str(exc),recommendations=['Inspect the watchdog log.'])
            try:
                command=json.loads((self.directory/'control.json').read_text())
                if command.get('id')!=handled and command.get('action')=='worker_lost':
                    ws=self._workspace
                    if ws is not None and ws.lock.acquire(timeout=.1):
                        try:
                            if self.operation_id is None:
                                current=json.loads((self.directory/'control.json').read_text())
                                if current.get('id')==command['id']:
                                    self._worker_lost(ws,current)
                                    handled=command['id']
                        finally:ws.lock.release()
                if command.get('id')!=handled and command.get('action')=='unload':
                    ws=self._workspace
                    if ws is not None and ws.lock.acquire(timeout=.1):
                        try:
                            if self.operation_id is None:
                                # Confirm the command still exists after waiting for work.
                                current=json.loads((self.directory/'control.json').read_text())
                                if current['id']==command['id']:
                                    failed=getattr(ws,'_failed_runner',None)
                                    if failed is not None:
                                        with failed.lock:failed._dispose()
                                        del ws._failed_runner
                                    runner=ws._runner
                                    if runner is not None:
                                        with runner.lock:runner.release()
                                        ws._runner=None
                                    ws.run_error=None
                                    ws.calib.clear();ws.keep.clear()
                                    if hasattr(ws,'_cache'):ws._cache.clear();ws._cache_bytes=0
                                    self.release()
                                    handled=command['id']
                                    self.last_event=command.get('event')
                                    write(self.directory/'control-reply.json',dict(id=handled,ok=True))
                        finally:ws.lock.release()
            except FileNotFoundError:pass
            except Exception as exc:
                self.last_event=event('unload_failed','unload',str(exc),recommendations=['Inspect the watchdog log.'])
            time.sleep(.1)

    def _forecast(self, ws):
        with self.lock:
            context=self.config()['context']
            key=(id(ws),getattr(ws,'version',0),context,getattr(ws.struct,'n_ctx',None),
                 ws.store.nbytes(),getattr(ws,'patch_stored',None))
            if self._forecast_cache is None or self._forecast_cache[0]!=key:
                self._forecast_cache=(key,reservation_forecast(ws,context))
            return self._forecast_cache[1]

    def _value(self, ws, *, name=None, tokens=0, retained=False, loading=False, stored=0, scratch=0):
        from .memory_guard import forecast
        value=self._describe(ws,loading,forecast_value=self._forecast(ws))
        value['forecast']=dict(value['forecast'])
        value['forecast']['patches']+=int(stored)*PATCH_ADMISSION_BYTES
        value['forecast']['total']+=int(stored)*PATCH_ADMISSION_BYTES
        if tokens:
            value['working']=forecast(ws,tokens,operation='trace' if retained else ('forward' if name=='forward' else 'interactive'))
        else:value['working']={}
        if scratch:value['working']={**value['working'],'activations':value['working'].get('activations',0)+int(scratch)}
        if self.operation_id:value.update(operation_id=self.operation_id,job=name or 'edit')
        if loading and self.worker:value['worker']=True
        return value

    def spawn_worker(self, operation_id, workspace_dir):
        """Inside the running load: ask the supervisor to start the model worker.

        -> the reply with ``socket``, ``pid``, ``create_time`` and ``token``. The
        supervisor does not wait for the worker; the caller connects and retries.
        """
        return self._request('spawn_worker',dict(operation_id=operation_id,workspace_dir=str(workspace_dir)))

    def _adopt(self, value):
        self.binding=fingerprint(value);self.reserved=True;self.error=None
        self._operation_descriptor=value
        self._check(self.binding)

    def _extend(self, ws, *, name, tokens=0, retained=False, loading=False, fresh=False):
        self.require(ws)
        kind=self._forward_kind(name,retained)
        # A forward that captures activations by name can accumulate across calls,
        # so it is never served from an earlier admission of the same size.
        if not loading and not fresh and tokens<=self._forward_allowances.get(kind,0):return
        value=self._value(ws,name=name,tokens=tokens,retained=retained,loading=loading,stored=self._prospective_stored,scratch=self._scratch_remaining)
        self._request('extend',value);self._adopt(value)
        if tokens:self._forward_allowances[kind]=max(tokens,self._forward_allowances.get(kind,0))

    @contextmanager
    def operation(self, ws, name, *, tokens=0, retained=False, extra_stored=0, fresh=False, size=None):
        # size: a function of the load settings the operation is pinned to (load_config) -> its tokens, replacing
        # ``tokens``; called under the client lock once they are pinned, so a request sized by the context it runs
        # with (the Run and Chat routes) cannot be sized by a context a save replaced between its check and begin.
        with ws.lock, self.operation_lock:
            if self.operation_id is not None:
                if self.operation_owner!=threading.get_ident():raise RuntimeError('Operation owner mismatch')
                if size is not None:
                    with self.lock:tokens=size(self.config())
                if tokens or name=='load':self._extend(ws,name=name,tokens=tokens,retained=retained,loading=name=='load',fresh=fresh)
                if extra_stored:self.preflight_edit(ws,extra_stored)
                yield
                return
            self.attach(ws)
            # Under the client lock from describing the operation to its grant: a saved settings change
            # (update_settings) lands before or after, never between them.
            with self.lock:
                self.drop_lost(ws)  # a lost model worker's reservation is released before fresh admission
                # With no model loaded this is a new load: it takes the saved load settings (a lost worker's
                # runner is dropped without release, so an old pin may remain).
                pinned=self._pinned is None or not runner_loaded(ws)
                if pinned:self._pinned=load_config(self.settings,self._selected)
                # Not begun (any failure, re-raised as it is): a pin taken here for a new load goes with it.
                # A flag and finally, not an exit-exception handler (tests/test_signal_sites.py allows one site).
                begun=False
                try:
                    if size is not None:tokens=size(dict(self._pinned))
                    self.require(ws)  # source identity verification before model allocation
                    value=self._value(ws,name=name,tokens=tokens,retained=retained,loading=name=='load',stored=extra_stored)
                    operation_id=os.urandom(16).hex();value.update(operation_id=operation_id,job=name)
                    try:self._request('begin',value)
                    except RuntimeRefusal as exc:
                        if getattr(exc,'timed_out',False):self._abandon(ws,operation_id)
                        raise
                    begun=True
                finally:
                    if not begun and pinned and not runner_loaded(ws):self._pinned=None
                self.operation_id=operation_id;self.operation_owner=threading.get_ident()
                self.operation_version=getattr(ws,'version',0);self._prospective_stored=extra_stored
                self._committed=[];self._commit_report=None
                self._scratch_allowance=0;self._scratch_remaining=0;self._forward_allowances={}
                if tokens:self._forward_allowances[self._forward_kind(name,retained)]=tokens
                self._last_lease_check=None
                self.binding=fingerprint(value);self.reserved=True;self._operation_descriptor=value
            failure=None;success=False
            try:
                self._check(self.binding)
                yield
                self._check(self.binding)
                success=True
            except RuntimeRefusal as exc:
                if self._committed:
                    exc.committed=list(self._committed)
                    exc.record={**exc.record,'message':exc.record['message']+' Patch '+', '.join(self._committed)+' was already committed; inspect it before reapplying.',
                        'recommendations':['Inspect the committed patch before retrying this edit.',*exc.record.get('recommendations',[])]}
                    exc.args=(exc.record['message'],)
                    exc.report={**(self._commit_report or {}),'ok':True,'committed':True,'patch':self._committed[-1],
                        'patches':list(self._committed),'outcome':'committed_interrupted','event':exc.record}
                failure=exc.record;self.last_event=failure;raise
            finally:
                try:
                    actual=self._describe(ws,forecast_value=self._forecast(ws))
                    reply=self._request('finish',dict(operation_id=operation_id,event=failure,actual=actual,success=success))
                    self.reserved=reply.get('reserved',True) if reply else True
                    self.binding=fingerprint(actual) if self.reserved else None
                    if success:self.last_event=None
                except Exception as exc:
                    self.last_event=exc.record if isinstance(exc,RuntimeRefusal) else unavailable(str(exc)).record
                finally:
                    self.operation_id=self.operation_owner=self.operation_version=None
                    # The supervisor drops its pin at a finish that leaves no model loaded.
                    if not runner_loaded(ws):self._pinned=None
                    self._prospective_stored=0
                    self._scratch_allowance=0;self._scratch_remaining=0;self._forward_allowances={};self._last_lease_check=None

    def preflight_edit(self, ws, stored, scratch=0, *, cumulative=False):
        """Stored is the total candidate size, or a new increment with cumulative=True."""
        with self.lock:
            self.require(ws)
            proposed=self._prospective_stored+int(stored) if cumulative else max(self._prospective_stored,int(stored))
            scratch=int(scratch)
            if self.operation_id and proposed<=self._prospective_stored and scratch<=self._scratch_remaining:
                self._scratch_remaining-=scratch
                return
            allowance=max(scratch,SCRATCH_ADMISSION_QUANTUM) if scratch>self._scratch_remaining else self._scratch_remaining
            value=self._value(ws,name='edit',stored=proposed,scratch=allowance)
            self._request('extend' if self.operation_id else 'bind',value)
            # Forwards admitted before this growth were checked against memory
            # that the growth now occupies; they must be admitted again.
            self._forward_allowances={}
            self._adopt(value);self._prospective_stored=proposed;self._scratch_allowance=allowance
            self._scratch_remaining=allowance-scratch

    def advise_edit(self, ws, scratch=0):
        """Advisory: the largest new patch (weights) memory admission would grant this workspace now, as
        ``preflight_edit`` would ask for it. Binds, reserves and records nothing, and never raises.

        Returns ``dict(largest, event, available, preferred, scratch, bounds)``; ``event`` is admission's refusal
        one weight past ``largest``. ``largest`` is None when there is no answer, with ``reason`` saying why.
        One call waits at most ADVISE_TIMEOUT_SECONDS in all (this client's lock, describing the edit, the mailbox,
        the reply). Describing the edit cannot be interrupted: if it alone uses up the time, nothing is asked and
        the call has no answer. An answer, or the lack of one, is reused for ADVISE_SECONDS for the same
        workspace state and scratch.
        The real ``preflight_edit`` calls remain the final word."""
        allowance=max(int(scratch),SCRATCH_ADMISSION_QUANTUM)
        key=(id(ws),getattr(ws,'version',0),allowance,getattr(ws,'patch_stored',None))
        with self._advice_lock:
            cached=self._advice
            if cached and cached[0]==key and time.monotonic()-cached[1]<ADVISE_SECONDS:return cached[2]
        advice=self._advise(ws,allowance,time.monotonic()+ADVISE_TIMEOUT_SECONDS)
        with self._advice_lock:self._advice=(key,time.monotonic(),advice)
        return advice

    def _advise(self, ws, allowance, deadline):
        def none(reason):return dict(largest=None,reason=reason)
        started=time.monotonic()
        if not self.lock.acquire(timeout=max(0,deadline-time.monotonic())):
            return none('the runtime client is busy with another model operation')
        generation=self._settings_generation
        try:value=self._value(ws,name='edit',stored=0,scratch=allowance)
        except Exception as exc:return none(f'the edit could not be described for memory admission: {type(exc).__name__}: {exc}')
        finally:self.lock.release()
        value.pop('operation_id',None);value.pop('job',None)
        if time.monotonic()>=deadline:
            # A request sent with no time left would only go unanswered; it is not sent.
            return none(f'describing the edit for memory admission took {time.monotonic()-started:.1f} s, '
                        f'past the {ADVISE_TIMEOUT_SECONDS:g} s limit, so the memory watchdog was not asked')
        if not self.request_lock.acquire(timeout=max(0,deadline-time.monotonic())):
            return none('the memory watchdog mailbox is busy with another request')
        try:
            nonce=os.urandom(16).hex()
            try:write(self.directory/'request.json',dict(id=nonce,action='advise',descriptor=value,previous_binding=None))
            except OSError as exc:return none(f'the memory watchdog mailbox could not be written: {type(exc).__name__}: {exc}')
            while time.monotonic()<deadline:
                try:
                    reply=json.loads((self.directory/'reply.json').read_text())
                    if reply.get('id')==nonce:
                        if generation!=self._settings_generation:
                            # Described with the settings before a saved change: judged against the wrong ones.
                            return none('the settings changed while the memory watchdog was asked')
                        if not reply.get('ok'):
                            # Not kept as last_event: an advisory refusal is not this client's operation.
                            return none(str((reply.get('event') or {}).get('message') or reply.get('reason') or 'the memory watchdog refused the request'))
                        advice=reply.get('advice')
                        if not isinstance(advice,dict):return none('the memory watchdog gave no advice')
                        if advice.get('largest') is None:
                            return dict(advice,reason=str((advice.get('event') or {}).get('message') or 'the memory watchdog has no answer'))
                        return advice
                except (FileNotFoundError,ValueError):pass
                time.sleep(.025)
            return none(f'the memory watchdog did not answer within {time.monotonic()-started:.1f} s')
        finally:self.request_lock.release()

    def update_settings(self, settings, *, restore=False, confirmed=None):
        """Apply saved settings to the supervisor, then to this client. The supervisor validates them itself and
        rewrites its settings.json; ``restore`` returns it to the settings before its last change (a save whose
        file write failed); ``confirmed``: the fingerprint (``effect['judged']``) of the unloading effect the person
        confirmed. Without it, or when the effect judged now has another fingerprint, the supervisor refuses such a
        change (``settings_unconfirmed``, with the effect it judged now on the exception).
        -> the supervisor's answer: ``changed``, ``effect``, ``running``, ``loaded`` (``late``: it answered only
        when asked again). Raises RuntimeRefusal (refused, or no answer and the supervisor still holds the
        previous settings) with nothing changed here or in the supervisor; SettingsUnsettled when the supervisor
        holds other settings (adopted here) or its settings are not known. A refusal is not this client's
        operation and is not kept as ``last_event``."""
        if not self.lock.acquire(timeout=SETTINGS_LOCK_SECONDS):
            raise RuntimeRefusal(event('settings_busy','refused',f'A model operation was starting for more than {SETTINGS_LOCK_SECONDS} s; settings were not applied.',
                recommendations=['Save again when the operation has started.']))
        try:
            payload=dict(values=settings['values'],sources=settings['sources'],notices=list(settings.get('notices',())),
                         restore=bool(restore),check=False,confirmed=confirmed if isinstance(confirmed,str) else None,
                         deadline=time.time()+REQUEST_TIMEOUT_SECONDS-SETTINGS_MARGIN_SECONDS)
            event_before=self.last_event
            try:reply=self._request('settings',payload);result=reply.get('settings')
            except RuntimeRefusal as exc:
                self.last_event=event_before
                if not getattr(exc,'timed_out',False):raise
                # No answer in time is not "not applied": a stalled supervisor may still have committed it.
                result=self._reconcile(settings,exc)
            if not isinstance(result,dict) or not result.get('accepted'):
                refusal=RuntimeRefusal((result or {}).get('event') or event('settings_refused','refused','The memory watchdog did not accept the settings.'))
                refusal.effect=(result or {}).get('effect')
                raise refusal
            self._adopt_settings(settings,result.get('selected'))
            return result
        finally:self.lock.release()

    def _adopt_settings(self, settings, selected):
        self.settings=copy.deepcopy(settings)
        if isinstance(selected,str):self._selected=selected
        self._forecast_cache=None;self._settings_generation+=1
        with self._advice_lock:self._advice=None

    def held_settings(self):
        """What the supervisor runs now, asked through the mailbox, or None without an answer. The mailbox holds
        one request and the supervisor handles them in turn, so this answer comes after an earlier request is
        finished, or after it was overwritten unseen and can never run: it cannot be stale. The reply is an atomic
        replace matched by its id: it cannot be torn. (The runtime directory's settings.json is also replaced
        atomically, but it is written before the change takes effect, so read alone it can be stale.)"""
        event_before=self.last_event
        try:reply=self._request('settings',dict(query=True),timeout=REQUEST_TIMEOUT_SECONDS)
        except RuntimeRefusal:return None
        finally:self.last_event=event_before
        held=reply.get('settings')
        if not isinstance(held,dict) or not all(isinstance(held.get(k),dict) for k in ('values','sources')):return None
        return held

    def _reconcile(self, settings, timeout):
        # After a settings request got no answer in time: ask what the supervisor holds, and say exactly that.
        def same(held, wanted):
            return (held['values']==wanted['values'] and held['sources']==wanted['sources']
                    and list(held.get('notices') or [])==list(wanted.get('notices') or []))
        held=self.held_settings()
        if held is not None and same(held,settings):
            # Committed, its answer late: the request took effect.
            last=held.get('last') or {}
            return dict(accepted=True,late=True,changed=last.get('changed'),effect=last.get('effect'),selected=held.get('selected'),
                        running=held.get('running'),loaded=held.get('loaded'),load=held.get('load'))
        if held is not None and same(held,self.settings):
            refusal=unavailable(f'The memory watchdog did not answer within {REQUEST_TIMEOUT_SECONDS} s. Asked again, it still runs the '
                                'previous settings, so these were not applied and cannot apply later.')
            refusal.timed_out=True
            raise refusal from timeout
        self._settings_generation+=1;self._forecast_cache=None    # an advisory answer across this is dropped
        with self._advice_lock:self._advice=None
        if held is None:
            raise unsettled(None,f'The memory watchdog did not answer within {REQUEST_TIMEOUT_SECONDS} s, nor when asked again what it '
                                    'runs: it may be running the unsaved values until the next accepted save or a restart.') from timeout
        adopted=dict(copy.deepcopy(self.settings),values=copy.deepcopy(held['values']),sources=copy.deepcopy(held['sources']),
                     notices=copy.deepcopy(list(held.get('notices') or [])))
        self._adopt_settings(adopted,held.get('selected'))
        raise unsettled(adopted,'The memory watchdog answered late and runs settings that are neither the previous nor the new '
                                'ones; this server now shows those until the next accepted save or a restart.') from timeout

    def check_settings(self, settings):
        """Advisory dry run of draft settings in the supervisor: whether it would accept them and what the idle
        watch would do with them now. Changes nothing, never raises, waits at most SETTINGS_CHECK_SECONDS in all.
        -> the supervisor's answer, or ``dict(accepted=None, reason=...)`` when there is none."""
        deadline=time.monotonic()+SETTINGS_CHECK_SECONDS
        def none(reason):return dict(accepted=None,reason=reason)
        if not self.request_lock.acquire(timeout=SETTINGS_CHECK_SECONDS):
            return none('the memory watchdog mailbox is busy with another request')
        try:
            nonce=os.urandom(16).hex()
            payload=dict(values=settings['values'],sources=settings['sources'],notices=list(settings.get('notices',())),
                         restore=False,check=True,deadline=time.time()+SETTINGS_CHECK_SECONDS)
            try:write(self.directory/'request.json',dict(id=nonce,action='settings',descriptor=payload,previous_binding=None))
            except OSError as exc:return none(f'the memory watchdog mailbox could not be written: {type(exc).__name__}: {exc}')
            while time.monotonic()<deadline:
                try:
                    reply=json.loads((self.directory/'reply.json').read_text())
                    if reply.get('id')==nonce:
                        result=reply.get('settings')
                        if not reply.get('ok') or not isinstance(result,dict):
                            return none(str((reply.get('event') or {}).get('message') or reply.get('reason') or 'the memory watchdog gave no answer'))
                        return result
                except (FileNotFoundError,ValueError):pass
                time.sleep(.025)
            return none(f'the memory watchdog did not answer within {SETTINGS_CHECK_SECONDS:g} s')
        finally:self.request_lock.release()

    def record_commit(self, ws, patch, report=None):
        with self.lock:
            if self.operation_id and self.operation_owner==threading.get_ident():
                if patch not in self._committed:self._committed.append(patch)
                if report is not None:self._commit_report=dict(report)

    def mutated(self, ws):
        # Workspace mutations happen under ws.lock and invalidate scalar metadata.
        with self.lock:
            self._forecast_cache=None;self._forward_allowances={};self._last_lease_check=None
            self._scratch_allowance=0;self._scratch_remaining=0
            if self.operation_id and self.operation_owner==threading.get_ident():
                self.operation_version=ws.version
                # A committed candidate is now in actual storage, never count twice.
                self._prospective_stored=0

    def require(self, ws, loading=False, *, preallocation=False):
        with self.lock:
            if self.operation_id is not None:
                if getattr(ws,'version',0)!=self.operation_version:
                    raise RuntimeRefusal(event('workspace_changed','abort','Workspace changed during the operation; retry on the current version.'))
                self._check(self.binding)
                return
            unverified=getattr(ws,'_metadata_only',False)
            value=self._describe(ws,loading and not unverified,forecast_value=self._forecast(ws))
            if getattr(ws,'_runtime_stat',value['stat'])!=value['stat']:
                raise RuntimeRefusal(event('model_changed','refused','GGUF source changed; unload and reopen the model.'))
            binding=fingerprint(value)
            # Fresh operation admission is always begin; bare require is metadata/
            # identity reservation and is not a token hot path outside an operation.
            if self.binding!=binding or loading:
                self._request('bind',value);self._adopt(value)
            self._check(binding)
            if unverified:
                digest=hashlib.sha256()
                with open(value['path'],'rb') as f:
                    while block:=f.read(8*1024**2):self._check(binding);digest.update(block)
                if ws.meta.get('base_sha256') not in (None,digest.hexdigest()):
                    raise RuntimeRefusal(event('model_changed','refused','GGUF source digest changed; refusing saved patches.'))
                if self._describe(ws)['stat']!=value['stat']:
                    raise RuntimeRefusal(event('model_changed','refused','GGUF changed during identity verification.'))
                ws.meta['base_sha256']=digest.hexdigest();write(Path(ws.dir)/'workspace.json',ws.meta)
                ws._runtime_stat=value['stat'];ws._metadata_only=False
                return self.require(ws,loading=loading)

    def release(self):
        with self.lock:
            if self.reserved or self.binding is not None:
                try:stopping=json.loads((self.directory/'status.json').read_text()).get('state')=='stopping'
                except FileNotFoundError:stopping=False
                if not stopping:self._request('release')
            self.binding=None;self.reserved=False;self.error=None
            self.operation_id=self.operation_owner=self.operation_version=None
            self._pinned=None
            self._forecast_cache=None;self._prospective_stored=0
            self._scratch_allowance=0;self._scratch_remaining=0;self._forward_allowances={};self._last_lease_check=None

    def dismiss(self):
        self._request('dismiss');self.last_event=None

    def status(self, ws=None):
        """Answers promptly: while another thread holds the lock (a mailbox request can
        wait 8 s) the same plain reads are made without it, the cached forecast is reused
        instead of recomputed, and ``busy`` is true."""
        locked=self.lock.acquire(timeout=STATUS_LOCK_SECONDS)
        try:
            result=dict(backend=backend_status(self.settings),limits=self.settings['values'],supervised=True,
                loaded_config=self.loaded_config(ws),next_load=load_config(self.settings,self._selected),
                reserved=self.reserved,error=self.error,event=self.last_event,operation_id=self.operation_id,busy=not locked,
                note='Each operation uses fresh admission; profile selection is not agent load consent.')
            result['actual_kernel_threads']=getattr(getattr(getattr(ws,'_runner',None),'kernel',None),'threads',None)
            if ws is not None and ws.kind=='gguf':
                cached=self._forecast_cache
                result['forecast']=self._forecast(ws) if locked else (cached[1] if cached else None)
            try:result['reservation']=json.loads((self.directory/'status.json').read_text())
            except FileNotFoundError:result['reservation']=dict(state='idle')
            result['warnings']=result['reservation'].get('warnings',[])
            result['event']=result['reservation'].get('event') or result['event']
            if not result['event']:
                try:result['event']=json.loads((self.directory.parent/'last-failure.json').read_text())
                except FileNotFoundError:pass
            return result
        finally:
            if locked:self.lock.release()


def runtime_sample(gpu_path, runner, runner_since, now, pid=None):
    """Same fresh CPU/GPU evidence for watchdog sampling and refused upgrades.

    With a live model worker (``pid``) only telemetry that process wrote counts;
    any other file is treated as not yet written.
    """
    from .memory_guard import snapshot, GuardError
    sample = snapshot()
    gpu = json.loads(gpu_path.read_text()) if gpu_path.exists() else None
    if gpu is not None and pid is not None and gpu.get('pid') != pid:
        gpu = None
    if gpu is not None:
        if gpu.get('error') or time.time()-gpu.get('time',0) > 3:
            raise GuardError('GPU telemetry lost/stale')
        if gpu.get('active') and gpu.get('backend')=='metal' and (type(gpu.get('metal_allocated')) is not int or type(gpu.get('metal_recommended')) is not int):
            raise GuardError('Active Metal allocation telemetry is incomplete')
        sample['gpu'] = gpu
    elif runner and runner_since is not None and now-runner_since > 5:
        raise GuardError('Runner allocation telemetry missing')
    return sample


def supervise(settings, *, empty=False, require_worker=True):
    """Cooperative operation abort/unload; step 4 stops only the model worker.

    The supervisor spawns, samples and kills the model worker. A kill or an
    unexpected worker exit loses the reservation visibly (lease, control and
    status) and supervision continues; nothing signals the server for a memory
    reason, and no memory or model fault ends the loop. Memory sampling that fails
    blocks operations until it recovers; after 3 s a live worker is stopped and its
    reservation lost, and the loop goes on retrying. A supervisor exception
    revokes the lease, stops the worker and leaves the app alive. The server is
    signalled only for a user-requested shutdown (``user_shutdown``,
    ``shutdown_forced``) and to sweep the group of a server that already exited
    (``server_exited_cleanup``).

    ``require_worker`` (the default) refuses a runner load whose descriptor lacks
    ``worker: true`` with ``in_process_runner_unsupported``. Tests that use an
    in-process fake runner pass ``require_worker=False``; either way step 4
    without a live worker is a recorded no-op (``kill_worker_skipped``).
    """
    import psutil
    from dataclasses import asdict, replace
    from . import runtime_policy as rp
    from .memory_guard import snapshot, process_memory, stop_group, kill_group, owned_remaining
    from .runtime_policy import admission, extend_admission, observations, Escalation, measured_total
    from .settings import EDITABLE, validate_interactive
    cache=user_dir('cache')/'runtime';cache.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='serve-',dir=cache));directory.chmod(0o700)
    write(directory/'settings.json',settings)
    # The model worker gets this environment, its token and the load settings of its load (worker_environment):
    # activate() already applied the resolved overrides (WSP_MAX_CTX, threads, backend) here.
    env=dict(os.environ,WSP_RUNTIME_DIR=str(directory),WSP_GUARD_TELEMETRY=str(directory/'watchdog.jsonl'))
    for key in ('WSP_GUARD_LEASE','WSP_SUPERVISED_PERMIT','WSP_WORKER_TOKEN'):env.pop(key,None)
    command=[sys.executable,'-m','wsp','serve','--runtime-child',str(directory)]
    if empty:command.append('--empty')
    child=None;known={};handlers={}
    stopping=None;forced=False;supervisor_failed=False
    # worker: the model worker process until it is reaped. worker_identity: the
    # reservation's worker {pid, create_time}, kept (lease, status) after a loss.
    worker=None;worker_identity=None;socket_dir=None
    lost=None;lost_control=False;kill_skipped=False
    empty_tree=dict(tree_rss=0,tree_footprint=0,pids=[],identities={})
    def _signal_server(reason, sig=None, errors=None):
        # The only server signal site. sig None: stop the verified group (TERM, wait, KILL).
        # Every call leaves a best-effort server_signal row, written even if the signal fails.
        if SIGNAL_SITES.get(reason)!='server':raise ValueError(f'Not a server signal site: {reason}')
        errors=errors if errors is not None else []
        try:
            if sig is None:return stop_group(child,known,errors)
            if child is not None and child.poll() is None:child.send_signal(sig)
            return []
        except Exception as exc:
            errors.append(repr(exc));raise  # the row carries the failure; the caller still sees it
        finally:
            evidence(lambda:record(dict(event='server_signal',reason=reason,signal='stop_group' if sig is None else int(sig),
                                        pid=child.pid if child is not None else None,errors=list(errors))))
    def _signal_worker(reason, sig=None, errors=None):
        # The only worker signal site: the verified worker group, never the server.
        if SIGNAL_SITES.get(reason)!='worker':raise ValueError(f'Not a worker signal site: {reason}')
        if worker is None:return []
        errors=errors if errors is not None else []
        w=worker
        if sig is None:remaining=stop_group(w['proc'],w['known'],errors)
        else:kill_group(w['proc'],w['known'],errors,sig);remaining=[]
        evidence(lambda:record(dict(event='worker_signal',reason=reason,signal='stop_group' if sig is None else int(sig),
                                    pid=w['pid'],errors=list(errors))))
        return remaining
    def forward(sig,frame):
        nonlocal stopping
        if stopping is None:stopping=time.monotonic()
        _signal_server('user_shutdown',sig)
    active=baseline=previous=policy=decision=None
    # op_settings: the settings the running operation was admitted with; every admission and advice inside it
    # uses them until it finishes, whatever is saved meanwhile. pinned: the load settings (load_config) fixed
    # while an operation runs or a model is loaded; the client pins the same. settings_before: the settings
    # before the last saved change, the only ones a restore may return to. last_change: what it changed.
    op_settings=pinned=settings_before=last_change=None;published='idle'
    running=None;idle_since=time.monotonic();runner_since=None;last_request=None
    latest=None;reason=None;ladder=Escalation();warning_records={}
    sampling_failed_since=unavailable_event=None
    log=directory/'watchdog.jsonl';gpu_path=Path(str(log)+'.gpu.json')
    last_failure=cache/'last-failure.json'
    def record(value):
        with log.open('a') as f:f.write(json.dumps({'time':time.time(),**value})+'\n')
    def config():
        return pinned or load_config(settings,backend_status(settings)['selected'])
    def watch_policy(total):
        # What the idle watch checks against: the current saved values, deliberately (a running operation is
        # watched with the policy it was admitted with). None when they cannot be applied to this total.
        try:return idle_watch_policy(settings['values'],total)
        except (ValueError,KeyError,TypeError):return None
    def publish(state):
        nonlocal published
        published=state
        watch=watch_policy(baseline['total']) if baseline and not running else None
        write(directory/'status.json',dict(state=state,event=latest,operation_id=running,
            forecast=active['forecast'] if active else None,policy=asdict(policy) if policy else None,
            admission=decision,idle_timeout=settings['values']['resident_seconds'],worker=worker_identity,
            watch=asdict(watch) if watch else None,load=pinned,next_load=load_config(settings,backend_status(settings)['selected']),
            warnings=[*settings.get('notices',()),*warning_records.values()]))
    def lease(ok,record=None):
        write(directory/'lease.json',dict(ok=ok,time=time.time(),pid=child.pid,
            binding=fingerprint(active) if active else None,operation_id=running,worker=worker_identity,event=record))
    def notify(value):
        nonlocal latest
        latest=value;record(dict(event='runtime_event',record=value));publish(value['action'])
    def update_warnings(values):
        current={value['code']:value for value in values}
        for code,value in current.items():
            if code not in warning_records:record(dict(event='runtime_warning',record=value))
        for code in warning_records.keys()-current.keys():record(dict(event='runtime_warning_cleared',code=code))
        warning_records.clear();warning_records.update(current)
    evidence_lost=False
    def evidence(*steps):
        # Evidence storage is not telemetry. A failed write here must not become
        # a sampling fault that halts the ladder before step 4; a lease that
        # cannot be rewritten expires closed after four seconds.
        nonlocal evidence_lost
        for step in steps:
            try:step()
            except OSError as exc:
                if not evidence_lost:
                    evidence_lost=True
                    try:print(f'Runtime evidence write failed: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
                    except (OSError,ValueError):pass
    def request_unload(outcome):
        control=directory/'control.json'
        try:existing=json.loads(control.read_text())
        except (OSError,ValueError):existing={}
        if existing.get('event',{}).get('time')!=outcome['time']:
            write(control,dict(id=os.urandom(12).hex(),action='unload',event=outcome))

    def identity(value,configuration=True):
        # configuration: the descriptor must carry the load settings in force (the pinned ones, else the saved
        # ones). The watch's own check of the reserved model is of its file only: a saved change is not a fault.
        from .capacity import context_limit
        p=Path(value['path'])
        try:st=p.stat()
        except OSError as exc:
            raise RuntimeRefusal(event('model_changed','refused','The model source is missing or unreadable; unload and reopen it.',recommendations=['Restore the source and reopen the workspace.'])) from exc
        if value['stat']!=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns]:
            raise RuntimeRefusal(event('model_changed','refused','The model file changed; reopen it before retrying.',recommendations=['Unload and reopen the model.']))
        cfg=config() if configuration else None
        if cfg and (value.get('configured_context',value['context'])!=cfg['context'] or value['context']!=context_limit(cfg['context'],value.get('model_context')) or value['backend']!=cfg['selected']):
            raise RuntimeRefusal(event('configuration_mismatch','refused','Operation differs from the active runtime configuration.'))
        if value['params']<=0 or value['forecast']['total']<st.st_size:raise ValueError('Invalid model descriptor')
        if active and (value['path']!=active['path'] or value['stat']!=active['stat']):
            raise RuntimeRefusal(event('model_changed','refused','Unload the previous model before switching artifacts.'))

    def outside_bytes(value,active,baseline,current,worker_identity):
        # A worker-hosted model's forecast excludes the server's own footprint at the
        # reservation's first admission (server tree only: a dying worker never inflates it).
        hosted=worker_identity is not None or bool(value.get('worker') or (active or {}).get('worker'))
        return (baseline or current).get('server_tree',{}).get('tree_footprint',0) if hosted else 0
    def advise(value,now):
        # Advisory (RuntimeClient.advise_edit): the largest new patch a grant of this edit descriptor would
        # admit now. The grant path's checks up to admission on a fresh sample, then runtime_policy.largest_patch.
        # Nothing is assigned, leased, published, notified or logged. A lost reservation is judged as the cold
        # request that would release it first. None: no answer (the caller falls back and preflight decides).
        if stopping is not None:return dict(largest=None,event=unavailable('Server is shutting down.').record)
        if sampling_failed_since is not None:return dict(largest=None,event=lost or unavailable_event)
        try:
            if lost is not None:state=(None,None,None)
            else:
                identity(value)
                if ladder.action is not None:return dict(largest=0,event={**ladder.record,'action':ladder.action})
                state=(active,baseline,worker_identity)
            act,base,owner=state
            current={**(sample_runtime(now) if act else snapshot()),**measure()}
            limits=op_settings if running and op_settings is not None else settings   # as the grant would admit it
            largest,refused=rp.largest_patch(limits,value,current,act,base,outside=outside_bytes(value,act,base,current,owner),
                                             per_weight=PATCH_ADMISSION_BYTES)
            vals=limits['values'];pol=policy_for(vals,current['total'],current['available'])
            # bounds: what the edit_limit refusal needs to name the bound admission stops at (text only).
            return dict(largest=largest,event=refused,available=current['available'],preferred=pol.preferred,
                        scratch=sum((value.get('working') or {}).values()),
                        bounds=dict(budget=pol.budget,budget_gib=vals['budget_gib'],ceiling_gib=vals['ceiling_gib'],
                                    loaded=bool(act and act.get('runner')),total=current['total'],measured=rp.measured_total(current)))
        except RuntimeRefusal as exc:return dict(largest=0,event=exc.record)
        except Exception as exc:return dict(largest=None,event=event('advice_unavailable','refused',f'{type(exc).__name__}: {exc}'))

    # ---------------------------------------------------------------- saved settings
    def unload_causes(values,current,idle_for):
        # Every unload cause of the idle watch with ``values`` on this sample, each on its own: observations()
        # reports only the first, so each cause it reports is set aside on a copy and it is asked again (its own
        # conditions and thresholds, nothing restated here). Memory pressure does not depend on the settings.
        policy=idle_watch_policy(values,current['total']);probe=dict(current);found={}
        while True:
            issue,_=observations(policy,probe,baseline)
            if issue is None or issue['code'] in found:break
            found[issue['code']]=issue
            if issue['code']=='memory_pressure':probe['pressure']=1
            elif issue['code']=='headroom_floor':probe['available']=policy.floor
            else:break      # runtime_budget is the last cause an idle sample without a previous one can have
        if idle_for is not None and idle_for>=values['resident_seconds']:found['idle_expired']=event('idle_expired','unload',
            f"It has been idle {idle_for:.0f} s, longer than the idle timeout of {values['resident_seconds']} s.")
        return found
    def judged(effect):
        # What a confirmation of this effect is for: the unload it names, the reservation and worker it would
        # unload, and the operation it waits for. A confirmation is accepted only while this is unchanged.
        if not effect or not effect.get('confirm'):return None
        body=[effect.get('kind'),effect.get('code'),bool(effect.get('unknown')),running,worker_identity,
              (active or {}).get('path'),(active or {}).get('stat'),bool((active or {}).get('runner'))]
        return hashlib.sha256(json.dumps(body,sort_keys=True,default=str).encode()).hexdigest()[:24]
    def settings_effect(old,new,now):
        # What the watch does with the new values, judged on a fresh sample against the old ones. Advisory:
        # the save itself never aborts, unloads or kills anything; the next samples apply the ladder as always.
        # While an operation runs, the new values are judged against the idle state that follows it (the same
        # sample; the idle clock restarts at its finish, so only memory conditions apply then).
        # An unload cause counts when the new values produce it and the old ones do not, whatever else is present.
        # confirm: the effect unloads (or may unload) a loaded model, so a save asks first (apply_settings);
        # judged: the fingerprint a confirmation must carry back.
        effect=settings_effect_of(old,new,now)
        if effect is not None:effect['judged']=judged(effect)
        return effect
    def settings_effect_of(old,new,now):
        if active is None or lost is not None:
            return None if running is None else dict(kind='after_operation',confirm=False,loaded=False,message=AFTER_OPERATION)
        loaded=bool(active['runner'])
        what='the loaded model' if loaded else 'the idle memory reservation'
        try:
            current={**sample_runtime(now),**measure()}
            # Idle expiry only while idle with a model loaded: during an operation the idle clock restarts at its finish.
            idle_for=now-idle_since if running is None and loaded else None
            before,after=unload_causes(old,current,idle_for),unload_causes(new,current,idle_for)
        except Exception as exc:
            sampled=f'Memory could not be sampled to check the idle watch: {type(exc).__name__}: {exc}'
            if running is not None:
                return dict(kind='after_operation',unknown=True,confirm=loaded,loaded=loaded,
                            message=f'{AFTER_OPERATION} What the new values do after it is not known. {sampled}')
            return dict(kind='unknown',confirm=loaded,loaded=loaded,message=sampled)
        added=[issue for code,issue in after.items() if code not in before]
        if running is not None:
            if not added:return dict(kind='after_operation',confirm=False,loaded=loaded,message=AFTER_OPERATION)
            cause=added[0]
            when=f'if this lasts {rp.IDLE_UNSAFE_PERSIST_SECONDS} s' if cause['code'] in IDLE_DEBOUNCED else 'within about a second'
            return dict(kind='after_operation',unload=True,code=cause['code'],causes=[i['code'] for i in added],confirm=loaded,loaded=loaded,
                        message=f'{AFTER_OPERATION} When it finishes, the memory watchdog would then unload {what} {when}, '
                                f'if memory stays as it is now. {cause["message"]}')
        idle=dict(idle_seconds=round(now-idle_since,1),loaded=loaded)
        if not added:return dict(kind='none',confirm=False,**idle)
        # Expiry first: it unloads at the next sample, a debounced memory cause only if it lasts.
        cause=next((i for i in added if i['code']=='idle_expired'),added[0])
        if cause['code'] in IDLE_DEBOUNCED:
            when,seconds=f'if this lasts {rp.IDLE_UNSAFE_PERSIST_SECONDS} s',rp.IDLE_UNSAFE_PERSIST_SECONDS
        else:when,seconds='within about a second',1
        return dict(kind='unload',code=cause['code'],causes=[i['code'] for i in added],seconds=seconds,confirm=loaded,
                    message=f'With these values the memory watchdog unloads {what} {when}. {cause["message"]}',**idle)
    def held_settings():
        # What this process runs now (the client's question after a reply it did not get in time). Read-only.
        return dict(query=True,values=settings['values'],sources=settings['sources'],notices=settings.get('notices',[]),
                    selected=backend_status(settings)['selected'],running=running is not None,
                    loaded=bool(active and active.get('runner')),load=pinned,last=last_change)
    def in_time(deadline):
        # A finite deadline, at most one request timeout ahead, not yet passed.
        clock=time.time()
        return (type(deadline) in (int,float) and math.isfinite(deadline)
                and clock<=deadline<=clock+REQUEST_TIMEOUT_SECONDS)
    def apply_settings(value,now):
        # Settings saved in the IDE (RuntimeClient.update_settings / check_settings). Validated here as at start;
        # accepted values replace this process's settings, then settings.json. Never notified or refused through
        # the admission path: the answer is in the reply. Nothing is changed unless the answer is accepted.
        # The slow work (validation, the memory sample for the effect) is done first; the deadline is checked
        # again immediately before the commit (settings.json, then these values), with nothing slow between.
        # The settings_changed row and the status are written after the reply (the main loop).
        nonlocal settings_before,last_change
        def refuse(code,message,recommendations=(),effect=None):
            return dict(accepted=False,effect=effect,event=event(code,'refused',message,recommendations=list(recommendations)))
        if isinstance(value,dict) and value.get('query') is True:return held_settings()
        if stopping is not None:return refuse('settings_unavailable','The server is shutting down; settings were not applied.')
        if not isinstance(value,dict):return refuse('settings_invalid','Settings request has no settings.')
        expired=lambda:refuse('settings_expired','The settings request reached the memory watchdog after the IDE stopped waiting; nothing was changed.',  # noqa: E731
                              ['Save again.'])
        if not in_time(value.get('deadline')):return expired()
        values,sources,notices=value.get('values'),value.get('sources'),value.get('notices')
        if (not isinstance(values,dict) or values.keys()!=settings['values'].keys() or not isinstance(sources,dict)
                or sources.keys()!=settings['sources'].keys() or not all(isinstance(v,str) for v in sources.values())
                or not isinstance(notices,list) or not all(isinstance(n,dict) and isinstance(n.get('message'),str) for n in notices)):
            return refuse('settings_invalid','Settings request does not match the running settings.')
        changed=sorted(k for k in values if values[k]!=settings['values'][k])
        fixed=[k for k in changed if k not in EDITABLE]+[k for k in sources if k not in EDITABLE and sources[k]!=settings['sources'][k]]
        if fixed:return refuse('settings_invalid','These settings change only at server start: '+', '.join(sorted(set(fixed)))+'.')
        # A command-line or environment value holds for this whole start (the settings page never sends one).
        held=sorted({k for k in changed+[k for k in sources if sources[k]!=settings['sources'][k]]
                     if settings['sources'][k]=='command line' or settings['sources'][k].startswith('environment ')})
        if held:return refuse('settings_invalid','These settings are held by a launch override until the server restarts without it: '+', '.join(held)+'.')
        candidate=dict(settings,values=dict(values),sources=dict(sources),notices=list(notices))
        restore=bool(value.get('restore'))
        if restore:
            # Only back to the settings before the last change (or a no-op): a restore is not a way around validation.
            if values!=settings['values'] and (settings_before is None or values!=settings_before['values']):
                return refuse('settings_invalid','A restore must return to the settings in effect before the last change.')
        else:
            try:
                total=(baseline or {}).get('total') or snapshot()['total']
                validate_interactive(candidate,total)
                if values['backend']!=settings['values']['backend']:check_backend(candidate)
            except (ValueError,GuardError,OSError,KeyError,TypeError) as exc:
                return refuse('settings_invalid',f'Not applied: {exc}',['Correct the value and save again.'])
        selected=backend_status(candidate)['selected']
        # Only values this process applies can change what the watch does: a save that changes none of them
        # (output tokens, the patch limit, load settings, or a resend of what it already runs) is not judged.
        effect=settings_effect(settings['values'],values,now) if any(k in SUPERVISOR_LIVE for k in changed) else None
        result=dict(accepted=True,changed=changed,effect=effect,selected=selected,running=running is not None,
                    loaded=bool(active and active.get('runner')),load=pinned)
        if value.get('check'):return dict(result,check=True)
        # Saving never by itself unloads a loaded model without the person having confirmed that effect first,
        # judged now: a confirmation is for the effect it was shown (its fingerprint), not for whatever is loaded
        # or running when it arrives. A restore returns to values that were in effect and is never held back.
        if not restore and (effect or {}).get('confirm') and value.get('confirmed')!=effect['judged']:
            again=value.get('confirmed') is not None
            return refuse('settings_unconfirmed',effect['message']+(' The model or the running operation changed since you confirmed.' if again else '')
                          +' Nothing was changed; confirm to save these settings.',['Confirm the save, or change the values.'],effect=effect)
        old=copy.deepcopy(settings)
        # The point of commitment. Nothing slow may run between this check and the in-memory replace below.
        if not in_time(value.get('deadline')):return expired()
        try:write(directory/'settings.json',candidate)
        except OSError as exc:
            return refuse('settings_write_failed',f'The memory watchdog could not record the settings ({type(exc).__name__}: {exc}); nothing was changed.')
        settings_before=old
        settings['values']=candidate['values'];settings['sources']=candidate['sources'];settings['notices']=candidate['notices']
        last_change=dict(changed=changed,effect=effect,restore=restore)
        log=dict(event='settings_changed',changed=changed,old={k:old['values'][k] for k in changed},new={k:values[k] for k in changed},
                 restore=restore,confirmed=(effect or {}).get('confirm') is True and not restore,operation_id=running,pinned=pinned,effect=effect)
        return dict(result,old=old,log=log)
    def settings_logged(log):
        # After the reply: the change's row and the published status (which names the new watch and next load).
        evidence(lambda:record(log),lambda:publish(published))
    def undo_settings(old,log):
        # The reply carrying an accepted change was not delivered: the client keeps the old settings, so must we.
        nonlocal settings_before,last_change
        settings['values']=old['values'];settings['sources']=old['sources'];settings['notices']=old['notices'];settings_before=None;last_change=None
        evidence(lambda:write(directory/'settings.json',settings),lambda:record(log),
                 lambda:record(dict(event='settings_restored',reason='reply undelivered')),lambda:publish(published))

    # ---------------------------------------------------------------- the model worker
    def live_worker():
        # The reservation's own worker while it runs and is not being killed.
        w=worker
        return w if w is not None and w['owned'] and not w['killed'] and w['proc'].returncode is None else None
    def worker_tree():
        w=live_worker()
        if w is None:return empty_tree
        try:tree=process_memory(w['pid'])
        except (psutil.Error,GuardError):
            if w['proc'].poll() is not None:return empty_tree  # exited between polls; reaped next iteration
            raise
        w['known'].update(tree['identities'])
        return tree
    def measure():
        # B2: the server tree plus the worker tree (its own session, so never inside the server's).
        # With a live worker the total is the server footprint plus max(worker footprint,
        # worker RSS): mapped pages resident in both processes are counted once. Without
        # one it is PR A's server tree (measured_total takes max(footprint, RSS)).
        # Both tree figures carry the total; the raw trees stay in server_tree/worker_tree.
        server=process_memory(child.pid);known.update(server['identities'])
        model=worker_tree()
        part=lambda t:dict(tree_rss=t['tree_rss'],tree_footprint=t['tree_footprint'],pids=list(t['pids']))  # noqa: E731
        if model['pids']:
            total=server['tree_footprint']+max(model['tree_footprint'],model['tree_rss'])
            rss=footprint=total
        else:rss,footprint=server['tree_rss'],server['tree_footprint']
        return dict(tree_rss=rss,tree_footprint=footprint,
                    pids=[*server['pids'],*model['pids']],identities={**server['identities'],**model['identities']},
                    server_tree=part(server),worker_tree=part(model))
    def sample_runtime(now):
        # GPU telemetry follows the live worker; a reservation whose worker has stopped expects none.
        w=live_worker()
        if w is not None:return runtime_sample(gpu_path,active['runner'],runner_since,now,w['pid'])
        return runtime_sample(gpu_path,bool(active['runner']) and worker_identity is None,runner_since,now)
    def worker_socket():
        nonlocal socket_dir
        path=directory/'worker.sock'
        if len(str(path).encode())<=WORKER_SOCKET_LIMIT:return path
        if socket_dir is None:
            socket_dir=Path(tempfile.mkdtemp(prefix='wsp-',dir='/tmp'));socket_dir.chmod(0o700)
        return socket_dir/'worker.sock'
    def start_worker(workspace_dir):
        nonlocal worker,worker_identity
        path=worker_socket()
        path.unlink(missing_ok=True)  # a killed worker leaves its socket behind
        token=os.urandom(32).hex()
        argv=[sys.executable,'-m','wsp','worker','--runtime-dir',str(directory),'--socket',str(path),'--workspace',workspace_dir]
        failed=lambda detail:RuntimeRefusal(event('worker_start_failed','refused',f'The model worker could not be started: {detail}',  # noqa: E731
            recommendations=['Inspect the server terminal, then retry.']))
        cfg=config()  # the running load's pinned settings
        try:proc=subprocess.Popen(argv,env=dict(env,WSP_WORKER_TOKEN=token,WSP_MAX_CTX=str(cfg['context']),WSP_GGML_THREADS=str(cfg['threads']),
                                                WSP_GGML_DEVICE=cfg['selected'],WSP_MAX_NEW_TOKENS=str(settings['values']['new_tokens'])),
                                  start_new_session=True)
        except (OSError,ValueError) as exc:raise failed(f'{type(exc).__name__}: {exc}') from exc
        try:born=psutil.Process(proc.pid).create_time()
        except psutil.Error as exc:
            # Gone before its identity was read: nothing verified to signal; the loop reaps it.
            worker=dict(proc=proc,pid=proc.pid,create_time=None,known={},owned=False,killed=True,stop=None)
            raise failed('it exited during start-up') from exc
        worker=dict(proc=proc,pid=proc.pid,create_time=born,known={proc.pid:born},owned=True,killed=False,stop=None)
        worker_identity=dict(pid=proc.pid,create_time=born)
        evidence(lambda:record(dict(event='worker_spawned',pid=proc.pid,create_time=born,socket=str(path),operation_id=running,load=cfg)))
        return dict(socket=str(path),pid=proc.pid,create_time=born,token=token)
    def grant_state():
        return (active,baseline,policy,decision,running,runner_since,previous,worker_identity,op_settings,pinned)
    def rollback(snapshot,spawned=None):
        # A grant the client never received is not granted, and its worker is killed.
        nonlocal active,baseline,policy,decision,running,runner_since,previous,worker_identity,op_settings,pinned
        active,baseline,policy,decision,running,runner_since,previous,worker_identity,op_settings,pinned=snapshot
        if spawned is not None and worker is spawned:
            spawned['owned']=False;spawned['killed']=True
            _signal_worker('spawn_rollback',signal.SIGKILL)
    def loss_record(code,trigger,**measured):
        trigger=trigger or {}
        message=WORKER_LOST_MESSAGE
        if 'returncode' in measured:message+=f" The worker exited with return code {measured['returncode']}."
        if trigger.get('message'):message+=f" Cause: {trigger['message']}"
        recommendations=[r for r in trigger.get('recommendations',()) if r!=FRESH_ADMISSION]+[FRESH_ADMISSION]
        value=event(code,'worker_killed',message,dict(trigger=trigger.get('code'),**measured),
                    dict(abort_seconds=rp.ABORT_GRACE_SECONDS,unload_seconds=rp.UNLOAD_GRACE_SECONDS),recommendations)
        value['cause']=trigger or None
        return value
    def lose(value,sample_row=None):
        # A2 steps 1-3: decided first, best-effort evidence, then the lease (kept, ok: false).
        nonlocal lost,lost_control
        lost=value;lost_control=False
        steps=([lambda:record(sample_row)] if sample_row else [])+[lambda:notify(value),lambda:write(last_failure,value),lambda:lease(False,value)]
        for persist in steps:
            try:persist()
            except Exception:pass
    def report_lost():
        # control.json worker_lost stays until the server's release for this reservation.
        nonlocal lost_control
        try:
            write(directory/'control.json',dict(id=os.urandom(12).hex(),action='worker_lost',
                worker_pid=(worker_identity or {}).get('pid'),event=lost))
            lost_control=True
        except Exception:pass
    def kill_worker(sample):
        w=worker;w['killed']=True  # decide first
        lose(loss_record('worker_killed',ladder.record,worker_pid=w['pid']),dict(event='sample',**sample,operation_id=running,action='kill_worker'))
        _signal_worker('kill_worker',signal.SIGKILL)
        w['proc'].poll()  # reaped now if already gone, else by the next loop iteration
        report_lost();ladder.reset()
    def stop_unmeasured(now,trigger):
        # F5: model memory that cannot be measured is not acceptable. The worker gets TERM now and
        # KILL after 10 s (loop deadlines) and the reservation is lost (A2); supervision goes on.
        w=worker;w['killed']=True  # decide first
        w['stop']=dict(reason='sampling_failed',term=now,kill=now+WATCHDOG_FAILED_KILL_SECONDS,sent=None)
        lose(loss_record('worker_killed',trigger,worker_pid=w['pid']))
        report_lost();ladder.reset()
    def poll_worker(now):
        nonlocal worker
        if worker is None:return
        w=worker;code=w['proc'].poll()
        if code is None:
            stop=w['stop']
            if stop is not None:
                due=None
                if stop['sent'] is None and now>=stop['term']:due=signal.SIGTERM
                elif stop['sent']==signal.SIGTERM and now>=stop['kill']:due=signal.SIGKILL
                if due is not None:
                    stop['sent']=due
                    if stop['reason']=='release_reap':_signal_worker('release_reap',due)
                    elif stop['reason']=='sampling_failed':_signal_worker('sampling_failed',due)
                    else:_signal_worker('watchdog_failed',due)
            return
        # A3: status 0 is an expected stop (a requested shutdown or an answered failed load).
        unexpected=code!=0 and w['owned'] and not w['killed'] and lost is None and stopping is None and not supervisor_failed
        if unexpected:
            lose(loss_record('worker_exited',ladder.record if ladder.action else None,worker_pid=w['pid'],returncode=code))
            _signal_worker('worker_exited',signal.SIGKILL)  # sweep what it left in its group
        worker=None
        evidence(lambda:record(dict(event='worker_stopped' if code==0 else 'worker_reaped',pid=w['pid'],returncode=code,
                                    killed=w['killed'],owned=w['owned'])))
        try:
            # A dead worker's telemetry is no evidence for anything.
            if json.loads(gpu_path.read_text()).get('pid')==w['pid']:gpu_path.unlink(missing_ok=True)
        except (OSError,ValueError):pass
        if unexpected:report_lost();ladder.reset()
    def reset_reservation(now):
        nonlocal active,baseline,previous,policy,decision,running,runner_since,latest,lost,lost_control,worker_identity,kill_skipped,op_settings,pinned
        old=latest
        active=baseline=previous=policy=decision=None;running=None;runner_since=None;ladder.reset()
        op_settings=pinned=None
        warning_records.clear();lost=None;lost_control=False;worker_identity=None;kill_skipped=False
        if worker is not None:
            # Release reap: a worker still alive 5 s after release gets TERM, then KILL 5 s later.
            worker['owned']=False
            if worker['stop'] is None and not worker['killed']:
                worker['stop']=dict(reason='release_reap',term=now+RELEASE_REAP_SECONDS,kill=now+2*RELEASE_REAP_SECONDS,sent=None)
        # The release itself is done; storage faults below must not
        # refuse it and leave the client bound to a model it unloaded.
        evidence(lambda:(directory/'lease.json').unlink(missing_ok=True),lambda:gpu_path.unlink(missing_ok=True),
                 lambda:(directory/'control.json').unlink(missing_ok=True))
        if old and old['action'] in ('abort','unload'):
            latest={**old,'action':'unloaded','message':old['message']+' Model unloaded; the IDE remains available.'}
            evidence(lambda:record(dict(event='runtime_event',record=latest)))
        evidence(lambda:publish('idle'))

    next_sample=time.monotonic()
    # Install forwarding before spawn; failures here cannot orphan a child.
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,forward)
        child=subprocess.Popen(command,env=env,start_new_session=True)
        if stopping is not None:_signal_server('user_shutdown',signal.SIGTERM)
    except BaseException:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        raise
    try:
        try:known[child.pid]=psutil.Process(child.pid).create_time()
        except psutil.Error:
            # Stopped before start-up finished (early signal): nothing to supervise.
            if child.poll() is None:raise
        print(f'Runtime supervisor: {log}',flush=True)
        record(dict(event='start',pid=child.pid,settings=settings,models_loaded=False,require_worker=require_worker))
        # Start-up notices (for example a clamped saved thread count) are visible before any admission.
        evidence(lambda:publish('idle'))
        while child.poll() is None:
            now=time.monotonic()
            if stopping is not None and now-stopping>30:
                forced=True;reason=event('shutdown_timeout','shutdown','Requested shutdown exceeded 30 seconds.');break
            poll_worker(now)
            if now>=next_sample:
                # measuring: a failure here is a sampling failure (recoverable, F5); one after it is a supervisor fault.
                measuring=True
                try:
                    tree=measure()
                    if active is not None and lost is None:
                        issue=None;gpu_stale=False
                        try:sample={**sample_runtime(now),**tree}
                        except GuardError as exc:
                            sample={**snapshot(),**tree};gpu_stale=True
                            issue=event('gpu_telemetry_stale','abort',str(exc),recommendations=['Wait for telemetry to recover before retrying.', 'Unload if telemetry remains unavailable.'])
                    measuring=False
                    # The first good sample after a sampling failure ends its message (never a loss it caused, A2).
                    recovered=sampling_failed_since is not None and latest is unavailable_event
                    if recovered:latest=None
                    if active is not None and lost is None:
                        try:identity(active,configuration=False)
                        except (RuntimeRefusal,OSError) as exc:
                            issue=event('model_changed','abort','Reserved model file changed or disappeared.',recommendations=['Unload and reopen the model.'])
                        static_cap=decision['static_cap']
                        # Running: the limits the operation was admitted with. Idle: the saved values now.
                        watched=policy if running else (watch_policy(sample['total']) or replace(policy,budget=static_cap,ceiling=static_cap))
                        unsafe,warnings=observations(watched,sample,baseline,previous,envelope=decision.get('envelope') if running else None)
                        issue=unsafe or issue
                        # The forecast bounds the model: a worker-hosted model's server base is outside it.
                        outside=(decision or {}).get('outside',0);model_total=measured_total(sample)-outside
                        if not running and model_total>active['forecast']['total'] and not issue:
                            warning=event('idle_above_forecast','warning','Idle memory exceeds its forecast; admission remains available within static caps.',
                                dict(total=measured_total(sample),outside=outside,excess=model_total-active['forecast']['total']),active['forecast']['total'],['Unload the model to release retained allocations.'])
                            warnings.append(warning)
                        evidence(lambda:update_warnings(warnings))
                        expired=not running and active['runner'] and now-idle_since>=settings['values']['resident_seconds']
                        if expired and issue is None:issue=event('idle_expired','unload','Model idle timeout expired; unloading the model.',now-idle_since,settings['values']['resident_seconds'],[FRESH_ADMISSION])
                        action=ladder.observe(issue,bool(running),now,idle_expired=expired,gpu_stale=bool(issue and issue['code']=='gpu_telemetry_stale'))
                        if action=='kill_worker' and live_worker() is not None:
                            # Step 4: only the worker is killed; supervision and the server go on.
                            kill_worker(sample);previous=None
                        else:
                            if action=='kill_worker' and not kill_skipped:
                                # B2: with no live worker step 4 is a recorded no-op; nothing is signalled,
                                # whether the worker was never spawned, rolled back or already gone.
                                kill_skipped=True
                                evidence(lambda:record(dict(event='kill_worker_skipped',reason='no live model worker',record=ladder.record,operation_id=running)))
                            evidence(lambda:record(dict(event='sample',**sample,operation_id=running,action=action)));previous=sample
                            if action in ('abort','unload'):
                                outcome={**ladder.record,'action':action}
                                if not latest or latest.get('code')!=outcome['code'] or latest.get('action')!=action:evidence(lambda:notify(outcome))
                                evidence(lambda:lease(False,outcome))
                                if action=='unload':evidence(lambda:request_unload(outcome))
                            elif action=='kill_worker':evidence(lambda:lease(False,{**ladder.record,'action':'unload'}))
                            elif action=='blocked' and not running:
                                blocked={**issue,'action':'blocked'}
                                if not latest or latest.get('code')!=blocked['code'] or latest.get('action')!='blocked':evidence(lambda:notify(blocked))
                                evidence(lambda:lease(False,blocked))
                            else:evidence(lambda:lease(True))
                            evidence(lambda:publish('unload' if action=='kill_worker' else (action or ('running' if running else 'idle'))))
                    elif lost is not None:
                        # A2: the lost reservation stays visible until the server releases it.
                        evidence(lambda:record(dict(event='sample',**tree,operation_id=running,action='lost')),
                                 lambda:lease(False,lost),lambda:publish(lost['action']))
                        if not lost_control:report_lost()
                    elif recovered:
                        # No reservation: the failure's lease goes and the idle state is published again.
                        evidence(lambda:(directory/'lease.json').unlink(missing_ok=True),lambda:publish('running' if running else 'idle'))
                    sampling_failed_since=None
                except Exception as exc:
                    if sampling_failed_since is None:
                        sampling_failed_since=now
                        unavailable_event=event('telemetry_unavailable','blocked',f'Memory sampling failed; retrying: {type(exc).__name__}: {exc}',
                            recommendations=['Wait for memory telemetry to recover before retrying.'])
                        # A worker loss stays the visible record (A2).
                        if lost is None:evidence(lambda:notify(unavailable_event))
                    previous=None
                    evidence(lambda:lease(False,lost or latest),lambda:record(dict(event='sampling_error',error=repr(exc),duration=now-sampling_failed_since,measuring=measuring)))
                    if now-sampling_failed_since>=TELEMETRY_FAILURE_GRACE_SECONDS:
                        if not measuring:
                            raise RuntimeError(f'Memory sampling failed for {TELEMETRY_FAILURE_GRACE_SECONDS} seconds: {exc}') from exc
                        # A sampling failure stays recoverable: operations stay blocked, sampling is retried
                        # every second and the mailbox answered; a live worker cannot stay unmeasured.
                        if live_worker() is not None:
                            stop_unmeasured(now,event('telemetry_unavailable','blocked',f'Memory sampling failed for {TELEMETRY_FAILURE_GRACE_SECONDS} seconds: {type(exc).__name__}: {exc}',
                                recommendations=['Wait for memory telemetry to recover before retrying.']))
                next_sample=now+1
            try:request=json.loads((directory/'request.json').read_text())
            except FileNotFoundError:request=None
            if request and request['id']!=last_request:
                last_request=request['id'];action=request['action'];value=request.get('descriptor')
                granted=None;spawned=None;reply_extra={};applied=changes_log=None
                try:
                    # advise, settings: answered even while stopping (they say so themselves); a refusal here would notify.
                    if stopping is not None and action not in ('release','finish','advise','settings'):raise unavailable('Server is shutting down.')
                    if action=='release':
                        reset_reservation(now)
                    elif action=='dismiss':
                        latest=None;evidence(lambda:last_failure.unlink(missing_ok=True),lambda:publish('running' if running else 'idle'))
                    elif action=='finish' and value.get('abandoned') and value.get('operation_id')!=running:
                        pass  # The timed-out begin was never granted, or its grant was already rolled back.
                    elif action=='finish' and lost is not None:
                        # A2: finish ends the operation; the loss stays the record and the lease stays revoked.
                        if running and value.get('operation_id')!=running:raise ValueError('Operation finish does not match active operation')
                        running=None;idle_since=now;op_settings=None
                        evidence(lambda:publish(lost['action']))
                    elif action=='finish':
                        if running and value.get('operation_id')!=running:raise ValueError('Operation finish does not match active operation')
                        running=None;idle_since=now
                        if value.get('event'):latest=value['event']
                        elif ladder.action is None and value.get('success'):
                            latest=None;evidence(lambda:last_failure.unlink(missing_ok=True))
                        if active:
                            actual=value.get('actual')
                            if actual:
                                identity(actual);active=actual
                            else:
                                active=dict(active);active.pop('operation_id',None);active.pop('working',None)
                            if ladder.action=='abort':
                                try:
                                    healthy={**sample_runtime(now),**measure()}
                                    # Healthy for what follows: the idle watch with the saved values.
                                    idle_policy=watch_policy(healthy['total']) or replace(policy,budget=decision['static_cap'],ceiling=decision['static_cap'])
                                    issue,_=observations(idle_policy,healthy,baseline)
                                    identity(active)
                                    ladder.acknowledge_finish(issue is None,now)
                                except (GuardError,OSError):pass
                            evidence(lambda:lease(ladder.action is None and sampling_failed_since is None,latest))
                        # The operation's admitted limits end with it; the pin ends unless a model stays loaded.
                        op_settings=None
                        if not (active and active.get('runner')):pinned=None
                        evidence(lambda:publish('idle'))
                    elif action=='advise':
                        reply_extra=dict(advice=advise(value,now))
                    elif action=='settings':
                        try:answer=apply_settings(value,now)
                        except Exception as exc:
                            answer=dict(accepted=False,event=event('settings_invalid','refused',f'Not applied: {type(exc).__name__}: {exc}'))
                        applied=answer.pop('old',None);changes_log=answer.pop('log',None)
                        reply_extra=dict(settings=answer)
                    elif action in GRANT_ACTIONS:
                        # While sampling fails nothing is admitted and a lost reservation is not released (A2).
                        if sampling_failed_since is not None:raise RuntimeRefusal(lost or unavailable_event)
                        if lost is not None:
                            if action not in COLD_ACTIONS:raise RuntimeRefusal(lost)
                            # The next cold request releases the lost reservation, then is admitted afresh.
                            evidence(lambda:record(dict(event='lost_reservation_released',record=lost,operation_id=running)))
                            reset_reservation(now)
                        if action=='spawn_worker':
                            if active is None:raise ValueError('No reservation to start a model worker for')
                            identity(active)
                            if ladder.action is not None:raise RuntimeRefusal({**ladder.record,'action':ladder.action})
                            if not running or value.get('operation_id')!=running or active.get('job')!='load':
                                raise ValueError('A model worker starts only inside the running load operation')
                            poll_worker(now)  # an exited worker is reaped and logged first
                            if worker is not None:
                                w=worker
                                if w['owned'] and not w['killed']:
                                    # This reservation's own worker is still running.
                                    raise RuntimeRefusal(event('worker_busy','refused','A model worker is already running; retry once it has stopped.',
                                        recommendations=['Wait a few seconds and retry.']))
                                # A released or lost worker never delays a new load past its reap:
                                # SIGKILL now (once), and the new worker starts as soon as it is gone.
                                if not w['killed'] and (w['stop'] is None or w['stop']['sent']!=signal.SIGKILL):
                                    w['stop']=dict(reason='release_reap',term=now,kill=now,sent=signal.SIGKILL)
                                    _signal_worker('release_reap',signal.SIGKILL)
                                raise RuntimeRefusal(event('worker_busy','refused','The previous model worker is stopping; retry shortly.',
                                    recommendations=['Retry in a moment.']))
                            if not isinstance(value.get('workspace_dir'),str):raise ValueError('spawn_worker needs the workspace directory')
                            granted=grant_state()
                            reply_extra=start_worker(value['workspace_dir']);spawned=worker
                            # The worker's footprint starts a new runaway baseline and its own telemetry clock.
                            previous=None;runner_since=now
                            try:lease(True)
                            except OSError:
                                rollback(granted,spawned)
                                raise
                            evidence(lambda:publish('running'))
                        else:
                            identity(value)
                            if ladder.action is not None:
                                raise RuntimeRefusal({**ladder.record,'action':ladder.action})
                            if require_worker and value.get('job')=='load' and value.get('runner') and not value.get('worker'):
                                raise RuntimeRefusal(event('in_process_runner_unsupported','refused',
                                    'Supervised model runners run only in the model worker; this in-process load was refused.',
                                    recommendations=['Restart the IDE; if this persists, inspect the server configuration.']))
                            if action=='begin' and running:raise RuntimeRefusal(event('operation_busy','refused','Another model operation is running.'))
                            if action=='extend' and value.get('operation_id')!=running:raise ValueError('Extension does not match active operation')
                            current={**(sample_runtime(now) if active else snapshot()),**measure()}
                            outside=outside_bytes(value,active,baseline,current,worker_identity)
                            # Inside a running operation: the settings it was admitted with, not later saves.
                            limits=op_settings if running and op_settings is not None else settings
                            new_policy,new_decision=admission(limits,value,current,active,baseline,outside=outside)
                            if action=='extend':
                                new_policy=extend_admission(policy,new_decision,current,weights=value['forecast'].get('weights',0),edit=value.get('job')=='edit')
                                new_decision['policy']=asdict(new_policy)
                                # Like the cap, the running operation's envelope only grows by extension.
                                if running and decision:new_decision['envelope']=max(new_decision['envelope'],decision.get('envelope',0))
                            granted=grant_state()
                            if baseline is None:baseline=current
                            if value['runner'] and (active is None or not active['runner']):runner_since=now
                            active=value;policy=new_policy;decision=new_decision
                            if action=='begin':
                                running=value['operation_id'];op_settings=copy.deepcopy(settings)
                                if pinned is None:pinned=config()
                            try:lease(True)
                            except OSError:
                                # An admission whose lease cannot be published is not granted.
                                rollback(granted)
                                raise
                            evidence(lambda:publish('running' if running else 'reserved'))
                    else:raise ValueError('Unknown runtime request')
                    try:write(directory/'reply.json',dict(id=last_request,ok=True,reserved=active is not None,**reply_extra))
                    except OSError:
                        # A grant the client never received is not granted: otherwise an
                        # undelivered begin leaves a phantom operation behind (operation_busy),
                        # and an undelivered spawn leaves a worker nobody connects to.
                        # release, finish and dismiss are already done and are not undone.
                        if granted is not None:
                            rollback(granted,spawned)
                            evidence(lambda:lease(True) if active else (directory/'lease.json').unlink(missing_ok=True),
                                     lambda:publish('running' if running else ('reserved' if active else 'idle')))
                        if applied is not None:undo_settings(applied,changes_log)
                        raise
                    # The reply is already delivered: a log fault must not turn it into a refusal.
                    # An advisory answer is not an admission and is not logged; a settings change logs its own row.
                    if applied is not None:settings_logged(changes_log)
                    if action not in ('advise','settings'):evidence(lambda:record(dict(event=action,descriptor=value,baseline=baseline,policy=asdict(policy) if policy else None,
                                                **({'worker':worker_identity} if action=='spawn_worker' else {}))))
                except (RuntimeRefusal,GuardError) as exc:
                    refusal=exc.record if isinstance(exc,RuntimeRefusal) else event('telemetry_unavailable','refused',str(exc),recommendations=['Wait for telemetry to recover and retry.'])
                    evidence(lambda:notify(refusal),
                             lambda:write(directory/'reply.json',dict(id=last_request,ok=False,reason=refusal['message'],event=refusal)))
                except Exception as exc:
                    # A reply that cannot be delivered is lost evidence, never the end of supervision.
                    refusal=event('runtime_request_invalid','refused',f'{type(exc).__name__}: {exc}',recommendations=['Inspect the runtime log before retrying.'])
                    if action=='advise':
                        # An advisory never notifies or changes the published status, even when its reply
                        # cannot be written: the fault is logged and the client, without an answer, falls back.
                        evidence(lambda:record(dict(event='advice_undelivered',id=last_request,error=refusal['message'])),
                                 lambda:write(directory/'reply.json',dict(id=last_request,ok=False,reason=refusal['message'],event=refusal)))
                    else:
                        # A2: a worker loss stays the visible record until dismissed or the next successful operation.
                        keep=bool(latest and latest.get('action')=='worker_killed')
                        evidence((lambda:record(dict(event='runtime_event',record=refusal))) if keep else (lambda:notify(refusal)),
                                 lambda:write(directory/'reply.json',dict(id=last_request,ok=False,reason=refusal['message'],event=refusal)))
            time.sleep(.025)
    except Exception as exc:
        supervisor_failed=True;reason=event('watchdog_unavailable','blocked',f'The watchdog failed: {type(exc).__name__}: {exc}',recommendations=['Restore the watchdog before retrying.'])
        latest=reason
        try:print(reason['message'],file=sys.stderr,flush=True)
        except (OSError,ValueError):pass
        # Failure of evidence storage must never abandon child/signal ownership.
        # If lease publication also fails, its old timestamp expires closed.
        for persist in (lambda:notify(reason),lambda:lease(False,reason),lambda:write(last_failure,reason)):
            try:persist()
            except Exception:pass
        # Unsupervised model memory is not acceptable: the worker gets TERM now and
        # KILL after 10 s, as loop deadlines. The server stays up.
        if worker is not None and not worker['killed']:
            worker['owned']=False
            worker['stop']=dict(reason='watchdog_failed',term=time.monotonic(),kill=time.monotonic()+WATCHDOG_FAILED_KILL_SECONDS,sent=None)
        # Keep shutdown ownership after a terminal watchdog fault. No model work
        # is admitted, but release/finish and user signals remain responsive.
        while child.poll() is None:
            now=time.monotonic()
            if stopping is not None and now-stopping>30:forced=True;break
            try:poll_worker(now)
            except Exception as error:
                try:print(f'Model worker stop failed: {type(error).__name__}: {error}',file=sys.stderr,flush=True)
                except (OSError,ValueError):pass
            try:
                request=json.loads((directory/'request.json').read_text())
                if request['id']!=last_request:
                    last_request=request['id'];action=request['action']
                    if action in ('release','finish','dismiss'):
                        if action=='release':active=None;running=None;warning_records.clear()
                        elif action=='finish':running=None
                        publish('blocked')
                        write(directory/'reply.json',dict(id=last_request,ok=True,reserved=active is not None))
                    else:
                        write(directory/'reply.json',dict(id=last_request,ok=False,reason=reason['message'],event=reason))
            except (OSError,ValueError,KeyError):pass
            time.sleep(.05)
    finally:
        errors=[];remaining=[]
        # The worker group is stopped before the server group. One that already
        # exited is reaped and logged like any other exit, not signalled.
        try:poll_worker(time.monotonic())
        except Exception as exc:errors.append(repr(exc))
        if worker is not None:
            try:remaining+=_signal_worker('teardown',errors=errors)
            except Exception as exc:errors.append(repr(exc));remaining+=owned_remaining(worker['known'],errors)
        # The server group is signalled only for a user-requested shutdown or after the server exited.
        if forced or stopping is not None or child.poll() is not None:
            try:
                if forced:remaining+=_signal_server('shutdown_forced',errors=errors)
                elif stopping is not None:remaining+=_signal_server('user_shutdown',errors=errors)
                else:remaining+=_signal_server('server_exited_cleanup',errors=errors)
            except Exception as exc:errors.append(repr(exc));remaining+=owned_remaining(known,errors)
        if socket_dir is not None:shutil.rmtree(socket_dir,ignore_errors=True)
        outcome=dict(event='teardown',reason=reason,returncode=child.returncode,remaining=remaining,errors=errors,server_left_running=supervisor_failed and child.poll() is None)
        try:record(outcome)
        except OSError:pass
        for sig,handler in handlers.items():signal.signal(sig,handler)
        if reason or errors or remaining:
            try:write(directory/'FAILED.json',outcome)
            except OSError:pass
    return 1 if reason or errors or remaining or child.returncode not in (0,-signal.SIGTERM) else 0
