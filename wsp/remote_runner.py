"""The server's runner for a supervised model worker: every forward pass runs in that process.

``make_runner`` returns a ``RemoteRunner`` when ``worker_jobs.uses_worker(ws)``.
It is constructed inside the server's admitted ``load`` operation: it asks the
supervisor to spawn the worker, connects, authenticates and sends the ``load``
job. Its methods mirror ``GGUFRunner`` (``run`` with named captures only,
``logits`` with an optional ``tail``, ``next_logits``, ``generate``) and open the
same ``job_scope`` names; whole analysis calls cross as named jobs (``job``), and
one window of the atlas observed stage crosses as ``observe_window``.
``extra`` is shipped with every job that uses it; nothing is cached.

The worker's admission requests (``callback`` frames) are answered inline on the
operation owner's thread, through the real ``RuntimeClient``.

Any transport failure (connect timeout, EOF, reset, broken pipe, a bad, oversize
or foreign frame, a loss the supervisor reports while a job is waiting, or any
exception while handling the job, including in a callback) marks the runner
lost, closes the socket and raises ``RuntimeRefusal`` with the tokens streamed so
far. Its record is the supervisor's loss record when there is one. Later remote
calls raise that refusal at once; tokenizer methods keep working. A job that
cannot even be encoded was never sent: it is refused with
``worker_request_unsendable`` and the worker and connection stay in use. A loss never
sets ``ws.run_error``: only an ``error`` frame answering ``load`` (a
deterministic construction failure) becomes a plain exception.

The one exception from ``on_token`` that is not a loss is
``generation_stop.GenerationStopped`` (the person stopped the request): the
runner asks the worker to end the job and keeps serving its frames until the
job's ordinary ``result`` (the tokens generated so far) arrives. While the worker
reads the prompt no token arrives, so the frame loop also looks at the callback's
``stop`` each quiet second and asks then.
"""
from __future__ import annotations

import errno
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from . import generation_stop
from .generation_stop import GenerationStopped
from .directions import DirectionError
from .probes import ProbeError
from .runner import HFRunner, Runner
from .runtime_events import RuntimeRefusal, event, job_scope, unavailable
from .steering import SteeringError
from .worker_protocol import Channel, ConnectionLost, ProtocolError, encode, manifest

WORKER_START_SECONDS = 20
CONNECT_RETRY_SECONDS = .05
SPAWN_RETRY_SECONDS = .075     # worker_busy while the supervisor reaps a released worker: retry in 50-100 ms steps
SHUTDOWN_SECONDS = 2
LOSS_RECORD_SECONDS = 2        # how long to wait for the supervisor's own loss record after EOF
LOSS_CODES = ('worker_killed', 'worker_exited')
# error frames of non-load jobs keep their type when it is one of these
ERRORS = {e.__name__: e for e in (ValueError, KeyError, IndexError, TypeError, RuntimeError, NotImplementedError,
                                  ZeroDivisionError, AssertionError, SteeringError, ProbeError, DirectionError)}


class WorkerLost(ConnectionLost):
    """The supervisor reports this runner's worker lost while a job waits on it."""
    def __init__(self, record):
        super().__init__(record['message'])
        self.record = record


class _Answered(Exception):
    """A terminal frame that is not a loss (``refusal`` or ``error``); the stream stays in sync."""
    def __init__(self, exc):
        super().__init__(str(exc))
        self.exc = exc


def atlas_stamp(ws):
    """[mtime_ns, size] of the atlas meta file, or None; a change drops the worker's cached Atlas."""
    from .atlas import atlas_dir
    try:
        st = (Path(atlas_dir(ws))/'meta.json').stat()
    except OSError:
        return None
    return [st.st_mtime_ns, st.st_size]


class TransferMeter:
    """What carrying a candidate's pieces to the model worker cost over a run of jobs (one search trial): the jobs
    that carried pieces, their bytes, the server's encode and send time and the worker's decode and drop time."""

    def __init__(self):
        self.jobs, self.bytes = 0, 0
        self.encode_seconds = self.send_seconds = self.worker_seconds = 0.0

    def as_dict(self) -> dict:
        seconds = self.encode_seconds + self.send_seconds + self.worker_seconds
        return {"jobs": self.jobs, "bytes": self.bytes, "bytes_per_job": self.bytes // self.jobs if self.jobs else 0,
                "encode_seconds": self.encode_seconds, "send_seconds": self.send_seconds,
                "worker_seconds": self.worker_seconds, "seconds": seconds}


class RemoteRunner(Runner):
    # Tokenizer-only and borrowed unchanged: reads the checkpoint's tokenizer files, never the model.
    chat_prompt = HFRunner.chat_prompt
    hf_tokenizer = HFRunner.hf_tokenizer
    meter = None    # a TransferMeter while a search trial measures its transport (class attribute: fixtures skip __init__)

    def __init__(self, ws):
        self.ws, self.lock = ws, threading.RLock()
        self.channel = None
        self.lost = self.lost_in = None
        self.pid = None
        self._tok = None
        self.max_ctx, self.hooks, self.backend = None, frozenset(), None
        self.kernel = SimpleNamespace(threads=None)
        client = ws._runtime_client
        try:
            spawned = self._spawn(client)
            self.pid = spawned['pid']
            self.channel = Channel(self._connect(spawned['socket']))
            info = self._call('load', {}, hello=spawned['token'])
            self.max_ctx = int(info['max_ctx'])
            self.hooks = frozenset(info.get('available') or ())
            self.kernel = SimpleNamespace(threads=info.get('threads'))
            self.backend, self.pid = info.get('backend'), info.get('pid', self.pid)
        except BaseException:
            # Like GGUFRunner: a failed construction releases the reservation it was admitted under.
            self._close()
            try:
                client.release()
            except Exception:
                pass  # the next cold request releases a reservation that is still held
            raise

    # ------------------------------------------------------------------ connection
    def _spawn(self, client):
        """spawn_worker, retried on ``worker_busy`` (a released or lost worker being reaped)
        within WORKER_START_SECONDS; the lease is checked before every retry."""
        deadline = time.monotonic()+WORKER_START_SECONDS
        while True:
            try:
                return client.spawn_worker(client.operation_id, self.ws.dir)
            except RuntimeRefusal as exc:
                if exc.record.get('code') != 'worker_busy' or time.monotonic()+SPAWN_RETRY_SECONDS > deadline:
                    raise
            time.sleep(SPAWN_RETRY_SECONDS)
            client._check(client.binding)

    def _connect(self, path):
        deadline = time.monotonic()+WORKER_START_SECONDS
        while True:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.connect(path)
                return s
            except OSError as exc:
                s.close()
                if exc.errno not in (errno.ENOENT, errno.ECONNREFUSED):
                    raise self._lose(exc, ()) from exc
                record = self._supervisor_loss() or self._lease_refusal()
                if record is not None:
                    raise self._lose(WorkerLost(record), ()) from exc
                if time.monotonic() > deadline:
                    raise self._lose(ConnectionLost(f'The model worker did not accept a connection within {WORKER_START_SECONDS} s'), ()) from exc
                time.sleep(CONNECT_RETRY_SECONDS)

    def _close(self):
        channel, self.channel = self.channel, None
        if channel is not None:
            channel.close()

    def release(self):
        """Ask the worker to shut down, wait up to SHUTDOWN_SECONDS for its EOF, close."""
        with self.lock:
            channel, self.channel = self.channel, None
            if channel is None:
                return
            try:
                if self.lost is None and channel.broken is None:
                    channel.send('shutdown')
                    channel.sock.settimeout(SHUTDOWN_SECONDS)
                    deadline = time.monotonic()+SHUTDOWN_SECONDS
                    while time.monotonic() < deadline and channel.sock.recv(4096):
                        pass
            except (OSError, ProtocolError):
                pass
            finally:
                channel.close()

    def use(self, variant='current', extra=None):
        raise RuntimeError('The model runs in the model worker; this server holds no weights to switch')

    # ------------------------------------------------------------------ loss
    def _read(self, name):
        try:
            return json.loads((self.ws._runtime_client.directory/name).read_text())
        except (OSError, ValueError, AttributeError):
            return None

    def _supervisor_loss(self):
        """The supervisor's loss record naming this runner's worker, from control.json or lease.json."""
        if self.pid is None:
            return None
        control = self._read('control.json')
        if isinstance(control, dict) and control.get('action') == 'worker_lost' and control.get('worker_pid') == self.pid \
                and isinstance(control.get('event'), dict):
            return control['event']
        lease = self._read('lease.json')
        record = lease.get('event') if isinstance(lease, dict) and not lease.get('ok') else None
        if isinstance(record, dict) and record.get('code') in LOSS_CODES and (lease.get('worker') or {}).get('pid') == self.pid:
            return record
        return None

    def _lease_refusal(self):
        # While the worker starts, a revoked lease (other than a loss) ends the wait as well.
        lease = self._read('lease.json')
        if isinstance(lease, dict) and not lease.get('ok') and isinstance(lease.get('event'), dict):
            return lease['event']
        return None

    def _runtime_directory(self):
        return getattr(getattr(self.ws, '_runtime_client', None), 'directory', None)

    def _idle(self):
        record = self._supervisor_loss()
        if record is not None:
            raise WorkerLost(record)

    def _loss_record(self, exc):
        if isinstance(exc, WorkerLost):
            return exc.record
        deadline = time.monotonic()+LOSS_RECORD_SECONDS
        while self.pid is not None:
            record = self._supervisor_loss()
            if record is not None or time.monotonic() > deadline:
                if record is not None:
                    return record
                break
            time.sleep(.05)
        from .runtime_supervisor import FRESH_ADMISSION, WORKER_LOST_MESSAGE
        detail = f'{type(exc).__name__}: {exc}'
        return event('worker_exited', 'worker_killed', f'{WORKER_LOST_MESSAGE} Cause: {detail}',
                     dict(worker_pid=self.pid, transport=detail), None, [FRESH_ADMISSION])

    def _lose(self, exc, tokens):
        # Close first: a worker still waiting on this connection exits, so the supervisor's
        # own loss record (with its return code) can appear before the wait gives up.
        self._close()
        record = self._loss_record(exc)
        client = self.ws._runtime_client
        self.lost, self.lost_in = record, client.operation_id
        client.last_event = record
        return RuntimeRefusal(record, tokens)

    def mark_lost(self, record):
        """The control thread saw this worker lost while no job was waiting on it."""
        with self.lock:
            if self.lost is None:
                self.lost, self.lost_in = record, None
            self._close()

    # ------------------------------------------------------------------ one job
    def _header(self, loading):
        ws, client = self.ws, self.ws._runtime_client
        fields = dict(version=ws.version, manifest=manifest(ws.patches), binding=client.binding,
                      operation_id=client.operation_id, atlas=atlas_stamp(ws))
        if loading:
            path = Path(ws.store.path).resolve()
            stat = getattr(ws, '_runtime_stat', None)
            if stat is None:
                st = path.stat()
                stat = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]
            fields['descriptor'] = dict(path=str(path), stat=list(stat), base_hash=ws.meta.get('base_sha256'))
        return fields

    def _call(self, name, arguments, extra=None, *, on_token=None, hello=None):
        """Send one job and serve its frames until its terminal frame; see the module docstring."""
        from .worker_jobs import decode_result, encode_job
        with self.lock:
            if self.lost is not None:
                raise RuntimeRefusal(self.lost)
            if self.channel is None:
                raise unavailable('The model worker connection is closed; this operation was not run.')
            # Validation and encoding failures here are not losses: nothing has been sent.
            meter = self.meter if extra else None
            if meter is not None:
                t0 = time.monotonic()
            fields, tensors = encode_job(name, arguments, extra)
            fields.update(self._header(name == 'load'))
            job = os.urandom(8).hex()
            try:
                frames = ([encode('hello', {'token': hello})] if hello is not None else [])+[encode('job', fields, tensors, job=job)]
            except RuntimeRefusal:
                raise
            except Exception as exc:
                raise RuntimeRefusal(event('worker_request_unsendable', 'refused',
                    f'The {name} request could not be encoded for the model worker and was not sent: {type(exc).__name__}: {exc}',
                    recommendations=['Retry with smaller or plain JSON arguments.'])) from exc
            if meter is not None:
                t1 = time.monotonic()
                carried = sum(t.numel() * t.element_size() for t in tensors.values())
            tensors = fields = None
            streamed = []
            stopping = warned = False
            halt = getattr(on_token, 'stop', None)  # a live-text callback's Stop (token_stream.TextStream)

            def ask_stop():
                # The person stopped the request: not a loss. The worker ends the job at its next
                # token or lease check and answers with a result.
                nonlocal stopping, warned
                if stopping:
                    return
                stopping = generation_stop.request(self._runtime_directory(), job)
                if not stopping and not warned:
                    warned = True
                    print(f'Stop not sent to the model worker: {generation_stop.FILE} could not be written '
                          'in the runtime directory; the reply runs to its end.', file=sys.stderr, flush=True)

            def idle():
                self._idle()
                if halt is not None and halt.is_set():
                    ask_stop()  # no token arrives while the worker reads the prompt
            try:
                for encoded in frames:
                    self.channel.send_encoded(encoded)
                frames = None
                if meter is not None:
                    t2 = time.monotonic()
                    meter.jobs += 1
                    meter.bytes += carried
                    meter.encode_seconds += t1 - t0
                    meter.send_seconds += t2 - t1
                while True:
                    frame = self.channel.recv(job=job, idle=idle)
                    kind = frame.type
                    if kind == 'token':
                        for token in frame['ids']:
                            streamed.append(int(token))
                            if on_token is not None:
                                try:
                                    on_token(int(token))
                                except GenerationStopped:
                                    ask_stop()
                    elif kind == 'callback':
                        self.channel.send('callback_reply', self._answer(frame), job=job)
                    elif kind == 'result':
                        if meter is not None:
                            meter.worker_seconds += float(frame.get('extra_seconds') or 0.0)
                        return decode_result(name, frame)
                    elif kind == 'refusal':
                        record = frame['record']
                        if not isinstance(record, dict) or not isinstance(record.get('message'), str):
                            raise ProtocolError('Malformed refusal frame')
                        tokens = frame.get('tokens')
                        raise _Answered(RuntimeRefusal(record, tokens if isinstance(tokens, list) and tokens else streamed))
                    elif kind == 'error':
                        exception, message = str(frame.get('exception')), str(frame.get('message'))
                        if name == 'load':
                            raise _Answered(RuntimeError(f'{exception}: {message}'))
                        raise _Answered(ERRORS.get(exception, RuntimeError)(message if exception in ERRORS else f'{exception}: {message}'))
                    else:
                        raise ProtocolError(f'Unexpected {kind} frame during job {name}')
            except _Answered as answered:
                if name == 'load':
                    self._close()  # the worker exits after answering a failed load
                raise answered.exc from None
            except BaseException as exc:
                refusal = self._lose(exc, streamed)
                if not isinstance(exc, Exception):
                    raise
                raise refusal from exc
            finally:
                if stopping:
                    generation_stop.clear(self._runtime_directory(), job)

    def _answer(self, frame):
        """A worker admission request, on this (the operation owner's) thread. -> callback_reply fields."""
        ws, client = self.ws, self.ws._runtime_client
        kind = frame.get('kind')
        try:
            if client.operation_id is None:
                raise unavailable('No operation is active for this model-worker request; it was not admitted.')
            if kind == 'operation':
                with client.operation(ws, str(frame['name']), tokens=int(frame.get('tokens', 0)), retained=bool(frame.get('retained')),
                                      extra_stored=int(frame.get('extra_stored', 0)), fresh=bool(frame.get('fresh'))):
                    pass
            elif kind == 'preflight':
                client.preflight_edit(ws, int(frame['stored']), int(frame.get('scratch', 0)), cumulative=bool(frame.get('cumulative')))
            else:
                raise ProtocolError(f'Unknown callback kind {kind!r}')
        except RuntimeRefusal as exc:
            return {'ok': False, 'record': exc.record}
        return {'ok': True, 'binding': client.binding}

    # ------------------------------------------------------------------ GGUFRunner's interface
    def _ids(self, ids):
        t = torch.as_tensor(ids)
        if t.ndim not in (1, 2) or (t.ndim == 2 and t.shape[0] != 1) or not 0 < t.shape[-1] <= self.max_ctx:
            raise ValueError(f'Guarded GGUF runs require batch 1 and 1..{self.max_ctx} tokens')
        return [int(i) for i in t.reshape(-1).tolist()]

    def available(self, name):
        return name in self.hooks

    @job_scope('forward', runner=True)
    def run(self, ids, variant='current', extra=None, names=(), hooks=None, grad=False):
        """Named captures only (no logits) when ``names`` is given; else (logits[None], {})."""
        if hooks is not None:
            raise RuntimeError('Forward hooks cannot cross the model-worker boundary; this needs a named worker job')
        if grad:
            raise RuntimeError('Packed GGUF parameter gradients are unsupported; use logical delta edits')
        flat = self._ids(ids)
        if names:
            return None, self._call('run', dict(ids=flat, variant=variant, names=[str(n) for n in names]), extra)
        return self._call('logits', dict(ids=flat, variant=variant), extra)[None], {}

    @job_scope('observe_window', runner=True)
    def observe_window(self, ids, variant='base', names=()):
        """One window of the atlas observed stage, run once and reduced in the worker (``wsp.atlas.observe``):
        -> {'loss', 'tensors'}, no captures and no logits. Admitted as a forward that holds the named hook
        points (``names``), as ``run`` is; committed weights only."""
        return self._call('observe_window', dict(ids=self._ids(ids), variant=variant, names=[str(n) for n in names]))

    def logits(self, ids, variant='current', extra=None, tail=None):
        ids = list(ids)[-self.max_ctx:] if self.max_ctx else list(ids)
        return self._logits(ids, variant, extra, tail)

    @job_scope('forward', runner=True)
    def _logits(self, ids, variant='current', extra=None, tail=None):
        arguments = dict(ids=self._ids(ids), variant=variant)
        if tail:
            arguments['tail'] = int(tail)
        return self._call('logits', arguments, extra)

    @job_scope('next', runner=True)
    def next_logits(self, ids, variant='current', extra=None, steer=None):
        if not 0 < len(ids) <= self.max_ctx:
            raise ValueError(f'Next-token scoring needs 1..{self.max_ctx} prompt tokens')
        arguments = dict(ids=[int(i) for i in ids], variant=variant)
        if steer is not None:
            from .steering import as_wire
            arguments['steer'] = as_wire(self.ws, steer)  # the worker resolves the wire form where the weights are
        return self._call('next_logits', arguments, extra)

    def first_token_logits(self, ids, variant='current', extra=None):
        """The next token's raw logits after ``ids`` (wsp.directions_scores): the ``next_logits`` job, its extra attached."""
        return self.next_logits(ids, variant, extra)

    @job_scope('generate', runner=True)
    def generate(self, ids, variant='current', extra=None, max_new_tokens=40, temperature=0., top_k=0, seed=None, stop_ids=(), on_token=None,
                 steer=None):
        if not len(ids):
            raise ValueError('Generation requires a nonempty prompt')
        arguments = dict(ids=[int(i) for i in ids], variant=variant, max_new_tokens=int(max_new_tokens), temperature=float(temperature),
                         top_k=int(top_k), seed=None if seed is None else int(seed), stop_ids=[int(t) for t in stop_ids])
        if steer is not None:
            from .steering import as_wire
            arguments['steer'] = as_wire(self.ws, steer)  # the worker resolves the wire form where the weights are
        return self._call('generate', arguments, extra, on_token=on_token)

    def final_norm(self, x, variant='current', extra=None):
        # Like unembed and final_gain, which this runner does not have: the final norm is applied next to the
        # model, inside the direction_residuals job (wsp.directions), never across the boundary per vector.
        raise RuntimeError('The final norm runs in the model worker (job direction_residuals); '
                           'it is not applied on the server side of a supervised model')

    def job(self, name, **arguments):
        """A whole named worker job (``worker_jobs.worker_job``); its ``extra`` travels with it."""
        extra = arguments.pop('extra', None)
        return self._call(name, arguments, extra)
