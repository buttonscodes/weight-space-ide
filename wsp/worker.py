"""The model worker: one connection, one read-only view of one workspace, one runner.

Started by the runtime supervisor inside an admitted load (``python -m wsp
worker``). It accepts exactly one connection, which must authenticate with
``hello{token}`` first, then runs jobs from ``worker_jobs.JOBS`` until the
server asks it to shut down. Before every job it syncs the committed patches to
the manifest in the job header (by version; any mismatch refuses the job with
``workspace_out_of_sync``). It writes only its ``.gpu.json`` telemetry and its
socket in the runtime directory, never the workspace, patches or the GGUF.
In a ``generate`` job it reads the server's ``stop.json`` (see
``generation_stop``) after each token it emits and at each lease check, which
also runs while the prompt is read: a stop asked for by the person ends the job
with an ordinary result holding the tokens so far, possibly none.

Exit status: 0 only after a requested ``shutdown`` or after a failed ``load``
has been answered with its terminal frame; anything else is non-zero.

This module imports only the standard library at the top, so the process can
publish idle GPU telemetry before torch is imported.
"""
from __future__ import annotations

import hmac
import json
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

ACCEPT_SECONDS = 30
PARENT_POLL_SECONDS = .5
IDLE_TELEMETRY_SECONDS = 1
EXIT_OK, EXIT_LOST, EXIT_PROTOCOL, EXIT_AUTH, EXIT_NO_CONNECTION = 0, 70, 71, 72, 73


# ------------------------------------------------------------------ process-level pieces
def telemetry_path(runtime_dir):
    """The same file GGUFRunner._telemetry writes and the supervisor samples."""
    os.environ.setdefault('WSP_GUARD_TELEMETRY', str(Path(runtime_dir)/'watchdog.jsonl'))
    return Path(os.environ['WSP_GUARD_TELEMETRY']+'.gpu.json')


class IdleTelemetry:
    """``active: false`` GPU telemetry from process start until the runner's own telemetry takes over.

    It keeps running through runner construction (a load can take longer than the
    supervisor's 3 s staleness limit) and is stopped only once the runner exists.
    ``stop`` returns only when no idle write is in flight or can follow, so the
    runner's next write is the last word; both writers record this pid.
    """

    def __init__(self, runtime_dir, interval=IDLE_TELEMETRY_SECONDS):
        self.path, self.interval = telemetry_path(runtime_dir), interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.write()
        self._thread = threading.Thread(target=self._loop, daemon=True, name='worker-idle-telemetry')
        self._thread.start()

    def write(self):
        with self._lock:
            if self._stop.is_set():
                return
            temporary = self.path.with_name(self.path.name+'.idle.tmp')
            temporary.write_text(json.dumps(dict(time=time.time(), active=False, backend=os.environ.get('WSP_GGML_DEVICE', 'cpu'),
                                                 pid=os.getpid(), mps_current=None, mps_driver=None, mps_recommended=None)))
            temporary.replace(self.path)

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:self.write()
            except OSError:pass

    def stop(self):
        with self._lock:
            self._stop.set()
        self._thread.join(timeout=2)


def watch_parent(interval=PARENT_POLL_SECONDS, exit=os._exit):
    """A worker whose supervisor died must not keep a model loaded."""
    original = os.getppid()
    if original == 1:
        exit(EXIT_LOST)
    def loop():
        while True:
            time.sleep(interval)
            if os.getppid() != original:
                exit(EXIT_LOST)
    thread = threading.Thread(target=loop, daemon=True, name='worker-parent')
    thread.start()
    return thread


def main(runtime_dir, socket_path, workspace):
    """``python -m wsp worker``: telemetry first, then bind, listen (backlog 1) and serve."""
    telemetry = IdleTelemetry(runtime_dir)
    watch_parent()
    token = os.environ.pop('WSP_WORKER_TOKEN', None)
    if not token:
        print('Model worker requires WSP_WORKER_TOKEN', file=sys.stderr, flush=True)
        return EXIT_AUTH
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        previous = os.umask(0o177)
        try:listener.bind(socket_path)
        finally:os.umask(previous)
        listener.listen(1)
        return serve_worker(listener, workspace, token, runtime_dir=runtime_dir, telemetry=telemetry)
    finally:
        listener.close()
        try:os.unlink(socket_path)
        except FileNotFoundError:pass


# ------------------------------------------------------------------ serving
def _gguf_runner(ws):
    from .gguf_runner import GGUFRunner
    return GGUFRunner(ws)


def _open_workspace(directory):
    from .workspace import Workspace
    return Workspace.load(directory, metadata_only=True)


def _accept_one(listener, timeout):
    listener.settimeout(timeout)
    try:
        connection, _ = listener.accept()
    except (socket.timeout, TimeoutError):
        return None
    finally:
        listener.close()
    connection.settimeout(None)
    return connection


def serve_worker(sock, ws_dir, token, make_runner=None, *, runtime_dir=None, open_workspace=None,
                 accept_timeout=ACCEPT_SECONDS, telemetry=None):
    """Serve one connection. ``sock`` is a listening socket (one connection is accepted,
    within ``accept_timeout``) or an already connected one. Returns the exit status."""
    from .worker_protocol import Channel
    try:
        sock.getpeername()
        connection = sock
    except OSError:
        connection = _accept_one(sock, accept_timeout)
        if connection is None:
            return EXIT_NO_CONNECTION
    channel = Channel(connection)
    try:
        return _Worker(channel, ws_dir, runtime_dir or os.environ['WSP_RUNTIME_DIR'], make_runner or _gguf_runner,
                       open_workspace or _open_workspace, telemetry).serve(token)
    finally:
        channel.close()


class OutOfSync(Exception):
    pass


_UNSET = object()


class _Worker:
    def __init__(self, channel, ws_dir, runtime_dir, make_runner, open_workspace, telemetry):
        from .worker_lease import WorkerLease
        self.channel, self.ws_dir = channel, os.path.abspath(ws_dir)
        self.make_runner, self.open_workspace, self.telemetry = make_runner, open_workspace, telemetry
        self.lease = WorkerLease(runtime_dir)
        self.ws = None
        self.synced = None
        self.atlas = _UNSET
        self._shapes = None

    def _lost(self):
        from .worker_protocol import ConnectionLost
        return EXIT_LOST if isinstance(self.channel.broken, ConnectionLost) else EXIT_PROTOCOL

    def serve(self, token):
        from .worker_protocol import ProtocolError
        try:
            # Nothing is allocated for an unauthenticated peer: no payload, no tensors.
            hello = self.channel.recv(hello=True)
        except ProtocolError:
            return EXIT_AUTH
        offered = hello.get('token') if hello.type == 'hello' else None
        if not isinstance(offered, str) or not hmac.compare_digest(offered.encode(), str(token).encode()):
            return EXIT_AUTH
        while True:
            try:
                frame = self.channel.recv()
            except ProtocolError:
                return self._lost()
            if frame.type == 'shutdown':
                self._release()
                return EXIT_OK
            if frame.type != 'job':
                return EXIT_PROTOCOL
            status = self._job(frame)
            if status is not None:
                return status

    def _release(self):
        runner = getattr(self.ws, '_runner', None)
        if runner is not None:
            try:
                with runner.lock:runner.release()
            except Exception as exc:
                print(f'Model worker release failed: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
            self.ws._runner = None

    def _job(self, frame):
        """Run one job; -> exit status, or None to keep serving."""
        from . import generation_stop
        from .runtime_events import RuntimeRefusal, event
        from .worker_jobs import JobContext, decode_job, encode_result
        from .worker_protocol import ProtocolError, encode
        job, name = frame.job, frame.get('name')
        loading = name == 'load' and self.ws is None
        reply = None
        try:
            if name == 'load' and not loading:
                raise RuntimeError('The model worker already has a loaded runner')
            if name != 'load' and getattr(self.ws, '_runner', None) is None:
                raise RuntimeError('The model worker has no loaded runner')
            started = time.perf_counter()
            row, arguments, extra = decode_job(frame)
            extra_seconds = time.perf_counter() - started   # with the drop below: what carrying the pieces cost here
            carried = row.extra and extra is not None
            frame.tensors.clear()  # the decoded extra owns them now; nothing else may outlive the job
            version = frame.get('version')
            if type(version) is not int:
                raise ValueError('Worker job needs an integer workspace version')
            self.lease.begin(job, binding=frame.get('binding'), version=version, operation_id=frame.get('operation_id'),
                             callback=lambda kind, fields: self._callback(job, kind, fields), stoppable=name == 'generate')
            if loading:
                self._open(frame.get('descriptor'))
            with self.ws.lock:
                try:
                    self._sync(version, frame.get('manifest'))
                except OutOfSync as exc:
                    raise RuntimeRefusal(event('workspace_out_of_sync', 'refused',
                        f'The model worker could not match the committed patches ({exc}); this job was not run.',
                        recommendations=['Retry the operation.', 'Unload and reload the model if this persists.'])) from None
                if self.atlas != frame.get('atlas'):
                    self.ws.keep.pop('atlas', None)
                    self.atlas = frame.get('atlas')
                emitted = []
                def emit(token):
                    self.channel.send('token', {'ids': [int(token)]}, job=job)
                    emitted.append(int(token))
                    if generation_stop.requested(self.lease.directory, job):
                        raise generation_stop.GenerationStopped()
                try:
                    value = row.call(JobContext(self.ws, self.ws._runner, extra, emit, self.make_runner), **arguments)
                except generation_stop.GenerationStopped as exc:
                    # Stopped by the person, after a token (emit) or at a lease check during the
                    # prompt: the result is what was generated so far. The runner's key-value
                    # cache is a local of generate and goes with the cleared frames.
                    traceback.clear_frames(exc.__traceback__)
                    value = emitted
                finally:
                    started = time.perf_counter()
                    self._drop_extra(extra)
                    extra_seconds += time.perf_counter() - started
                    extra = None
                if loading:
                    self._hand_over_telemetry()
                reply = ('result',) + encode_result(name, value)
                if carried:
                    reply = (reply[0], {**reply[1], 'extra_seconds': extra_seconds}, reply[2])
                value = None
                # Validated before anything is sent: a result that cannot cross (for example an
                # oversize tensor table) is this job's error frame; the stream stays in sync.
                try:
                    reply = (reply[0], encode(reply[0], reply[1], reply[2], job=job))
                except ProtocolError as exc:
                    raise ValueError(f'The {name} result cannot be sent: {exc}') from None
        except ProtocolError:
            return self._lost()
        except RuntimeRefusal as exc:
            reply = ('refusal', {'record': exc.record, 'tokens': [int(t) for t in exc.tokens]}, {})
        except Exception as exc:
            reply = ('error', {'exception': type(exc).__name__, 'message': str(exc)}, {})
        finally:
            self.lease.end()
        if self.channel.broken is not None:
            return self._lost()
        try:
            encoded = reply[1] if reply[0] == 'result' else self._encode_answer(reply, job)
            self.channel.send_encoded(encoded)
        except ProtocolError:
            return self._lost()
        if loading and reply[0] != 'result':
            return EXIT_OK  # a failed load is answered once; the server spawns afresh
        return None

    @staticmethod
    def _encode_answer(reply, job):
        """A refusal or error frame; one that cannot be encoded becomes a short error."""
        from .worker_protocol import ProtocolError, encode
        try:
            return encode(reply[0], reply[1], reply[2], job=job)
        except (ProtocolError, TypeError, ValueError) as exc:
            return encode('error', {'exception': type(exc).__name__, 'message': f'The {reply[0]} frame could not be encoded'}, job=job)

    def _hand_over_telemetry(self):
        """The runner exists: its telemetry takes over from the idle writer, with no gap.

        The idle writer is stopped (no write of it can follow), then the runner
        republishes at once, so an idle write racing the runner's first one never
        stands. A runner that publishes nothing keeps the idle writer."""
        runner = getattr(self.ws, '_runner', None)
        publish = getattr(runner, '_telemetry', None)
        if self.telemetry is None or not callable(publish) or not os.environ.get('WSP_GUARD_TELEMETRY'):
            return
        self.telemetry.stop()
        try:
            publish()
        except Exception as exc:
            print(f'Model worker telemetry hand-over failed: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)

    def _callback(self, job, kind, fields):
        from .worker_protocol import ProtocolError
        self.channel.send('callback', {**fields, 'kind': kind}, job=job)
        reply = self.channel.recv(job=job)
        if reply.type != 'callback_reply':
            self.channel.broken = ProtocolError(f'Expected callback_reply, got {reply.type}')
            raise self.channel.broken
        return reply.header

    def _drop_extra(self, extra):
        # Runners remember the last overlay; a shipped extra must not outlive its job.
        runner = getattr(self.ws, '_runner', None)
        if extra is not None and runner is not None and hasattr(runner, 'use'):
            with runner.lock:runner.use('base')

    def _open(self, descriptor):
        from .runtime_events import RuntimeRefusal, event
        changed = RuntimeRefusal(event('model_changed', 'refused', 'The model file differs from the admitted one; the worker did not load it.',
                                       recommendations=['Unload and reopen the model.']))
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get('stat'), list):
            raise ValueError('Load job needs the admitted model descriptor')
        ws = self.open_workspace(self.ws_dir)
        try:
            path = Path(ws.store.path).resolve()
            st = path.stat()
        except OSError:
            raise changed from None
        if ([st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns] != descriptor['stat']
                or descriptor.get('path', str(path)) != str(path) or ws.meta.get('base_sha256') != descriptor.get('base_hash')):
            raise changed
        # The server verified the digest for this stat before admission; no rehash here.
        ws._runtime_stat, ws._metadata_only = descriptor['stat'], False
        ws._runtime_client = self.lease
        self.ws = ws

    def _shapes_of(self):
        if self._shapes is None:
            self._shapes = {n: self.ws.store.shape(n) for n in self.ws.names()}
        return self._shapes

    def _sync(self, version, rows):
        from .worker_protocol import parse_manifest
        from .workspace import Workspace
        ws = self.ws
        try:
            rows = parse_manifest(rows)
        except ValueError as exc:
            raise OutOfSync(str(exc)) from None
        if self.synced == (version, rows) and ws.version == version:
            return
        first, self.synced = self.synced is None, None
        loaded = {p.id: p for p in ws.patches}
        pdir = os.path.join(ws.dir, 'patches')
        plan = []
        for pid, created, enabled, strength, stored in rows:
            p = loaded.get(pid)
            if p is None or p.created != created:  # unseen, or an id reused by a newer patch
                try:
                    p = Workspace._read_patch(pdir, pid+'.json', self._shapes_of())
                except Exception as exc:
                    raise OutOfSync(f'patch {pid} is unreadable: {type(exc).__name__}') from None
                if p.id != pid:
                    raise OutOfSync(f'patch file {pid} names {p.id}')
            if p.created != created:
                raise OutOfSync(f'patch {pid} was created at {p.created}, the server has {created}')
            if p.stored != stored:
                raise OutOfSync(f'patch {pid} stores {p.stored} numbers, the server has {stored}')
            plan.append((p, enabled, float(strength)))
        # Nothing is mutated until every row has been verified.
        changed = [(id(p), p.enabled, p.strength) for p in ws.patches] != [(id(p), e, s) for p, e, s in plan]
        for p, e, s in plan:
            p.enabled, p.strength = e, s
        ws.patches = [p for p, _, _ in plan]
        if changed or first:
            with self.lease.rebuilding():
                ws.rebuild()
        ws.version = version
        self.synced = (version, rows)
