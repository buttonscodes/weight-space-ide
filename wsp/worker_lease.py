"""The model worker's stand-in for RuntimeClient: local lease checks, admission by callback.

The server's RuntimeClient keeps the only supervisor mailbox. Inside the worker,
the unchanged runner and analysis code still calls ``job_scope``,
``require_lease``, ``layer_checkpoint`` and ``edit_preflight``; this object
answers them. Checks read the constant-size ``lease.json`` (never patches, the
forecast or the model); extensions go back to the server as ``callback`` frames
on the job's connection and are answered by the operation owner's thread.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from . import generation_stop
from .runtime_events import RuntimeRefusal, event, unavailable

LAYER_CHECK_INTERVAL_SECONDS = .05
LEASE_MAX_AGE_SECONDS = 4


class WorkerLease:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = threading.RLock()
        self.binding = self.version = self.operation_id = self.job = None
        self.last_event = None
        self._callback = None
        self._last_lease_check = None
        self._rebuilding = False
        self.stoppable = False

    # ------------------------------------------------------------------ job scope (set by the worker loop)
    def begin(self, job, *, binding, version, operation_id=None, callback=None, stoppable=False):
        """``stoppable``: a ``generate`` job, which a person's stop may end at a lease check."""
        with self.lock:
            self.job, self.binding, self.version, self.operation_id = job, binding, version, operation_id
            self._callback = callback
            self._last_lease_check = None
            self.stoppable = bool(stoppable)

    def end(self):
        with self.lock:
            self.job = self.binding = self.version = self.operation_id = self._callback = None
            self._last_lease_check = None
            self.stoppable = False

    @property
    def reserved(self):
        return self.job is not None

    @contextmanager
    def rebuilding(self):
        """The worker's own manifest sync may rebuild; no other mutation is allowed."""
        self._rebuilding = True
        try:
            yield
        finally:
            self._rebuilding = False

    # ------------------------------------------------------------------ checks
    def _check(self, binding):
        # Constant-size lease only: no patch scan, forecast or model descriptor.
        try:
            lease = json.loads((self.directory/'lease.json').read_text())
            if not lease.get('ok'):
                raise RuntimeRefusal(lease.get('event') or event('lease_revoked', 'abort', 'The operation lease was revoked.'))
            worker = lease.get('worker')
            # A lease that names no worker (or another one) is simply not this worker's lease.
            if (binding is None or lease.get('binding') != binding or not isinstance(worker, dict)
                    or worker.get('pid') != os.getpid() or not 0 <= time.time()-lease['time'] < LEASE_MAX_AGE_SECONDS):
                raise unavailable()
            self._last_lease_check = (binding, time.monotonic())
        except RuntimeRefusal as exc:
            self._last_lease_check = None; self.last_event = exc.record; raise
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self._last_lease_check = None
            refusal = unavailable(); self.last_event = refusal.record
            raise refusal from exc

    def _active(self):
        if self.job is None:
            raise unavailable('No model-worker job is active; this operation was not run.')

    def require(self, ws, loading=False, *, preallocation=False):
        with self.lock:
            self._active()
            if getattr(ws, 'version', 0) != self.version:
                raise RuntimeRefusal(event('workspace_changed', 'abort', 'Workspace changed during the operation; retry on the current version.'))
            self._check(self.binding)
            self._stopped()

    def _stopped(self):
        """After a good lease read: a stop for this generate job ends it here, during the prompt too.

        The lease checks run between prompt chunks and, at most every 50 ms, between
        layers, so a stop does not wait for the whole prompt to be read. The job
        then answers with the tokens emitted so far (``Worker._job``)."""
        if self.stoppable and generation_stop.requested(self.directory, self.job):
            raise generation_stop.GenerationStopped()

    def layer_checkpoint(self, ws):
        """Only extra native-layer checks may reuse a <=50ms successful lease read."""
        with self.lock:
            if self.job is None or getattr(ws, 'version', 0) != self.version:
                return self.require(ws)
            previous = self._last_lease_check
            if previous and previous[0] == self.binding and 0 <= time.monotonic()-previous[1] < LAYER_CHECK_INTERVAL_SECONDS:
                return
            self._check(self.binding)
            self._stopped()

    # ------------------------------------------------------------------ admission by callback
    def _call(self, kind, **fields):
        with self.lock:
            self._active()
            reply = self._callback(kind, fields)
            if not reply.get('ok'):
                record = reply.get('record')
                if not isinstance(record, dict) or not isinstance(record.get('message'), str):
                    record = event('admission_refused', 'refused', 'The server refused the worker request.')
                self.last_event = record
                raise RuntimeRefusal(record)
            # An extension re-binds the server's reservation; adopt it and re-check.
            self.binding = reply.get('binding', self.binding)
            self._last_lease_check = None
            self._check(self.binding)
            return reply

    @contextmanager
    def operation(self, ws, name, *, tokens=0, retained=False, extra_stored=0, fresh=False):
        with ws.lock:
            self._active()
            if name == 'load':
                raise RuntimeError('worker cannot load a runner outside its load job')
            if tokens or extra_stored:
                self._call('operation', name=name, tokens=int(tokens), retained=bool(retained),
                           extra_stored=int(extra_stored), fresh=bool(fresh))
            yield

    def preflight_edit(self, ws, stored, scratch=0, *, cumulative=False):
        self._call('preflight', stored=int(stored), scratch=int(scratch), cumulative=bool(cumulative))

    # ------------------------------------------------------------------ calls the worker never makes
    def release(self):
        """The server owns the reservation; a failed runner construction releases nothing here."""

    def mutated(self, ws):
        if not self._rebuilding:
            raise RuntimeError('worker cannot mutate the workspace')
        self._last_lease_check = None

    def record_commit(self, ws, patch, report=None):
        raise RuntimeError('worker cannot commit patches')

    def dismiss(self):
        raise RuntimeError('worker cannot dismiss runtime events')

    def bind(self, *args, **kwargs):
        raise RuntimeError('worker cannot bind a reservation')
