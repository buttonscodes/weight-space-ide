"""Model-worker protocol, lease and job loop with scalar fakes; no model, checkpoint or native code.

The worker runs `serve_worker` on one end of a socketpair in a thread; the test
plays the server. The workspace is a real `Workspace` over a fake store, so the
patch files, `rebuild` and versions are the production ones.
"""
import gc
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import torch

from wsp import delta as D
from wsp.memory_guard import require_lease
from wsp.runtime_events import RuntimeRefusal, event, job_scope
from wsp.runtime_supervisor import RuntimeClient, write
from wsp.workspace import Patch, Workspace
from wsp import worker as W
from wsp.worker import serve_worker
from wsp.worker_jobs import JOBS, WORKER_KINDS, decode_result, encode_job, encode_result, uses_worker, worker_job
from wsp.worker_lease import WorkerLease
from wsp.worker_protocol import (Channel, ConnectionLost, JobMismatch, MAX_BODY, MAX_HEADER, PREFIX, PROTOCOL_KEYS,
                                 ProtocolError, RECV_CHUNK, TransferUnsupported, encode, encode_extra, manifest, read_frame)

APP = Path(__file__).resolve().parents[1]
TOKEN = 'a5'*32


# ------------------------------------------------------------------ fixtures
class FakeRunner:
    """Scalar stand-in for GGUFRunner: real job_scope/require_lease calls, no tensors of weights."""
    on_step = None

    def __init__(self, ws):
        self.ws, self.lock = ws, threading.RLock()
        self.max_ctx, self.kernel = 64, SimpleNamespace(threads=3)
        self._variant, self._extra, self.released, self.seen = 'base', None, False, []
        require_lease(ws, loading=True)

    def available(self, name):
        return name.endswith('hook_resid_post')

    def use(self, variant='current', extra=None):
        self._variant, self._extra = variant, extra

    def _value(self, extra):
        patches = sum(p.strength for p in self.ws.patches if p.enabled)
        return float(patches)+sum(float(p.values.sum()) for ps in (extra or {}).values() for p in ps)

    @job_scope('forward', runner=True)
    def run(self, ids, variant='current', extra=None, names=(), hooks=None, grad=False):
        require_lease(self.ws)
        self.use(variant, extra)
        self.seen.append(weakref.ref(next(iter(extra.values()))[0].values) if extra else None)
        logits = torch.full((1, len(ids), 4), self._value(extra))
        return logits, {n: torch.full((1, len(ids), 2), float(i)) for i, n in enumerate(names)}

    def logits(self, ids, variant='current', extra=None):
        return self.run(ids, variant, extra)[0][0]

    @job_scope('next', runner=True)
    def next_logits(self, ids, variant='current', extra=None):
        require_lease(self.ws)
        return torch.full((4,), self._value(extra))

    @job_scope('generate', runner=True)
    def generate(self, ids, variant='current', extra=None, max_new_tokens=4, temperature=0., top_k=0, seed=None,
                 stop_ids=(), on_token=None):
        self.use(variant, extra)
        out = []
        try:
            for step in range(max_new_tokens):
                if self.on_step:self.on_step(step)
                require_lease(self.ws)
                out.append(10+step)
                if on_token:on_token(10+step)
        except RuntimeRefusal as exc:
            exc.tokens = list(out)
            raise
        return out

    def release(self):
        self.released = True


class Lease:
    def __init__(self, directory):
        self.directory, self.binding, self.revoked = directory, 'bound', None

    def write(self, **changes):
        value = dict(ok=True, time=time.time(), pid=os.getpid()+1, binding=self.binding, worker=dict(pid=os.getpid(), create_time=0.))
        value.update(changes)
        write(self.directory/'lease.json', value)

    def revoke(self, record):
        self.revoked = record
        self.write(ok=False, event=record)


class Harness:
    def __init__(self, tmp_path, make_runner=None, token=TOKEN, telemetry=None):
        self.root = tmp_path; self.runtime = tmp_path/'runtime'; self.runtime.mkdir()
        self.dir = tmp_path/'ws'; (self.dir/'patches').mkdir(parents=True)
        self.source = self.dir/'dummy.gguf'; self.source.write_bytes(b'scalar fixture only')
        self.struct = SimpleNamespace(n_ctx=128, n_layers=2, d_model=8, n_heads=2, d_vocab=4)
        self.store = SimpleNamespace(path=str(self.source), names=lambda: [], shape=lambda n: (4,), nbytes=lambda: 19)
        self.meta = {'kind': 'gguf', 'base_sha256': 'fixture', 'name': 'scalar'}
        self.ws = Workspace(str(self.dir), self.store, self.struct, None, dict(self.meta))  # the server's view
        self.ws.rebuild()
        self.lease = Lease(self.runtime); self.lease.write()
        self.runners = []
        def default_runner(ws):
            runner = FakeRunner(ws); self.runners.append(runner); return runner
        self.make_runner = make_runner or default_runner
        server, worker = socket.socketpair()
        server.settimeout(10)
        self.channel = Channel(server)
        self.status = {}
        def target():
            self.status['value'] = serve_worker(worker, str(self.dir), token, self.make_runner,
                                                runtime_dir=self.runtime, open_workspace=self.open_workspace,
                                                telemetry=telemetry(self.runtime) if telemetry else None)
        self.thread = threading.Thread(target=target, daemon=True); self.thread.start()

    def open_workspace(self, directory):
        # Workspace.load's patch loop without a GGUF parser: same files, same _read_patch.
        ws = Workspace(directory, self.store, self.struct, None, dict(self.meta))
        pdir = os.path.join(directory, 'patches')
        for fn in sorted(os.listdir(pdir)):
            if fn.endswith('.json'):ws.patches.append(Workspace._read_patch(pdir, fn, {}))
        ws.rebuild()
        return ws

    def descriptor(self, **changes):
        st = self.source.stat()
        return {**dict(path=str(self.source.resolve()), stat=[st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns], base_hash='fixture'), **changes}

    def hello(self, token=TOKEN):
        self.channel.send('hello', {'token': token})

    def call(self, name, args=None, *, extra=None, rows=None, atlas=None, on_callback=None, job=None, descriptor=None,
             unchecked=False):
        job = job or os.urandom(8).hex()
        if self.lease.revoked is None:self.lease.write()
        # unchecked: bypass the server-side encoder to exercise the worker's own validation
        fields, tensors = ({'name': name, 'args': dict(args or {})}, {}) if unchecked else encode_job(name, args or {}, extra)
        fields.update(version=self.ws.version, manifest=manifest(self.ws.patches) if rows is None else rows,
                      binding=self.lease.binding, operation_id='op', atlas=atlas)
        if name == 'load':fields['descriptor'] = descriptor or self.descriptor()
        self.channel.send('job', fields, tensors, job=job)
        tokens, callbacks = [], []
        while True:
            frame = self.channel.recv(job=job)
            if frame.type == 'token':tokens += frame['ids']
            elif frame.type == 'callback':
                callbacks.append(frame.header)
                reply = on_callback(frame.header) if on_callback else {'ok': True, 'binding': self.lease.binding}
                self.channel.send('callback_reply', reply, job=job)
            else:
                return frame, tokens, callbacks

    def load(self):
        self.hello()
        frame, _, _ = self.call('load')
        assert frame.type == 'result', frame.header
        return frame

    def finish(self, timeout=5):
        self.thread.join(timeout)
        assert not self.thread.is_alive()
        return self.status['value']

    def shutdown(self):
        self.channel.send('shutdown')
        return self.finish()

    @property
    def worker_ws(self):
        return self.runners[-1].ws


def add_patch(ws, pid, value=1., created=None):
    p = Patch(pid, 'm', 'test', {'kind': 'manual'}, {'w': [D.block((range(4),), torch.full((4,), value))]}, created=created)
    ws.patches.append(p); ws._save_patch(p); ws.rebuild()
    return p


def state(ws):
    return [(p.id, p.created, p.enabled, p.strength, p.stored) for p in ws.patches]


# ------------------------------------------------------------------ framing and codecs
def test_frames_round_trip_tensors_once_with_job_id():
    a, b = socket.socketpair()
    try:
        sent = {'f': torch.arange(6, dtype=torch.float32).reshape(2, 3).T, 'ids': torch.tensor([3, -1, 7]),
                'empty': torch.empty(0, 5), 'scalar': torch.tensor(2.5)}
        Channel(a).send('result', {'value': {'x': [1, 2]}, 'nan': float('nan')}, sent, job='job-1')
        frame = Channel(b).recv(job='job-1')
        assert frame.type == 'result' and frame.job == 'job-1' and frame['value'] == {'x': [1, 2]}
        assert set(frame.tensors) == set(sent)
        for name, t in sent.items():
            assert frame.tensors[name].dtype == t.dtype and torch.equal(frame.tensors[name], t.contiguous())
        Channel(a).send('shutdown')
        assert Channel(b).recv().type == 'shutdown'
    finally:a.close();b.close()


@pytest.mark.parametrize('dtype', [torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.uint8, torch.bool])
def test_other_dtypes_are_refused_not_cast_and_nothing_is_sent(dtype):
    a, b = socket.socketpair()
    try:
        with pytest.raises(TransferUnsupported) as exc:
            Channel(a).send('result', {}, {'x': torch.zeros(3, dtype=dtype)}, job='j')
        assert isinstance(exc.value, RuntimeRefusal) and exc.value.record['code'] == 'transfer_unsupported'
        b.setblocking(False)
        with pytest.raises(BlockingIOError):b.recv(1)
    finally:a.close();b.close()


def test_extra_codec_requires_float32_values_and_int64_coordinates():
    sparse = D.Piece('sparse', (range(4),), values=torch.ones(2), coords=torch.tensor([[0], [3]]))
    header, tensors = encode_extra({'w': [sparse, D.block((range(2),), torch.ones(2))]})
    assert {t.dtype for k, t in tensors.items() if k.endswith('.c')} == {torch.int64}
    with pytest.raises(TransferUnsupported):
        encode_extra({'w': [D.Piece('block', (range(2),), values=torch.ones(2, dtype=torch.float64))]})
    with pytest.raises(TransferUnsupported):
        encode_extra({'w': [D.Piece('sparse', (range(4),), values=torch.ones(2), coords=torch.tensor([[0], [3]], dtype=torch.int32))]})


def raw_frame(header, payload=b'', header_len=None):
    data = json.dumps(header).encode()
    return PREFIX.pack(len(data) if header_len is None else header_len, len(payload))+data+payload


@pytest.mark.parametrize('data,message', [
    (PREFIX.pack(MAX_HEADER+1, 0), 'exceeds'),
    (raw_frame({'type': 'result', 'job': 'j', 'tensors': [dict(name='x', dtype='float32', shape=[2], offset=0, nbytes=4)]}, b'\0'*4), 'nbytes'),
    (raw_frame({'type': 'result', 'job': 'j', 'tensors': [dict(name='x', dtype='float64', shape=[1], offset=0, nbytes=8)]}, b'\0'*8), 'dtype'),
    (raw_frame({'type': 'result', 'job': 'j', 'tensors': [dict(name='x', dtype='float32', shape=[1], offset=0, nbytes=4)]}, b'\0'*8), 'payload length'),
    (raw_frame({'type': 'result', 'tensors': []}), 'job'),
    (raw_frame({'type': 'teleport', 'job': 'j'}), 'type'),
    (PREFIX.pack(3, 0)+b'{x}', 'JSON'),
    (raw_frame({'type': 'result', 'job': 'j', 'body': 3}, b'{x}'), 'body is not JSON'),
    (raw_frame({'type': 'result', 'job': 'j', 'body': 2}, b'[]'), 'JSON object'),
    (raw_frame({'type': 'result', 'job': 'j', 'body': -1}), 'non-negative'),
    (raw_frame({'type': 'result', 'job': 'j', 'body': 5}, b'{}'), 'payload length'),
])
def test_bad_frames_are_protocol_errors(data, message):
    a, b = socket.socketpair()
    try:
        a.sendall(data)
        with pytest.raises(ProtocolError, match=message):read_frame(b)
    finally:a.close();b.close()


def utf16(value):return json.dumps(value).encode('utf-16')


@pytest.mark.parametrize('data,hello,message', [
    # json.loads on bytes would detect UTF-16/32; frames are UTF-8 only.
    (PREFIX.pack(len(utf16({'type': 'result', 'job': 'j'})), 0)+utf16({'type': 'result', 'job': 'j'}), False, 'header is not JSON'),
    (PREFIX.pack(len(json.dumps({'type': 'hello'}).encode('utf-32')), 0)+json.dumps({'type': 'hello'}).encode('utf-32'), True, 'header is not JSON'),
    (raw_frame({'type': 'result', 'job': 'j', 'body': len(utf16({'a': 1}))}, utf16({'a': 1})), False, 'body is not JSON'),
    (raw_frame({'type': 'result', 'job': 'j', 'body': 5}, b'"\xed\xa0\x80"'), False, 'body is not JSON'),   # a UTF-8-encoded surrogate
])
def test_frames_are_strict_utf8(data, hello, message):
    _refused(data, hello, message)


def _nested(depth=20_000):
    return b'['*depth+b']'*depth


@pytest.mark.parametrize('data,message', [
    # Deeply nested JSON raises RecursionError inside json.loads: a bad frame, never an escaping exception.
    # (A hello header is at most HELLO_MAX_HEADER bytes, too shallow to reach the parser's limit.)
    (PREFIX.pack(len(_nested()), 0)+_nested(), 'header is not JSON'),
    (raw_frame({'type': 'result', 'job': 'j', 'body': len(_nested())}, _nested()), 'body is not JSON'),
])
def test_deeply_nested_json_is_a_bad_frame(data, message):
    with pytest.raises(RecursionError):json.loads(_nested())   # the premise, on this interpreter
    _refused(data, False, message)


def _refused(data, hello, message):
    a, b = socket.socketpair()
    def send():                      # from a thread: a frame larger than the socket buffer needs a reader
        try:a.sendall(data)
        except OSError:pass
    sender = threading.Thread(target=send, daemon=True); sender.start()
    try:
        b.settimeout(10)
        with pytest.raises(ProtocolError, match=message):read_frame(b, hello=hello)
    finally:
        b.close();sender.join(5);a.close()


def test_oversize_header_and_payload_limits_and_eof_mid_frame():
    # JSON fields travel in the body, so only a tensor table can outgrow the header.
    with pytest.raises(ProtocolError, match='exceeds'):
        encode('result', {}, {f't{i}': torch.zeros(0) for i in range(MAX_HEADER//40)}, job='j')
    a, b = socket.socketpair()
    try:
        # A declared body over MAX_BODY is refused from the header alone (the sender never sends it).
        header = json.dumps({'type': 'result', 'job': 'j', 'body': MAX_BODY+1}).encode()
        a.sendall(PREFIX.pack(len(header), MAX_BODY+1)+header)
        b.settimeout(10)  # a reader that waits for the body instead fails here, not hangs
        with pytest.raises(ProtocolError, match='body of .* exceeds'):Channel(b).recv()
        b.settimeout(None)
        a.close();b.close();a, b = socket.socketpair()  # a refused frame leaves the stream unusable
        a.sendall(raw_frame({'type': 'result', 'job': 'j', 'tensors': [dict(name='x', dtype='float32', shape=[4], offset=0, nbytes=16)]}, b'\0'*16)[:-3])
        a.close()
        with pytest.raises(ConnectionLost):read_frame(b)
    finally:a.close();b.close()


def trace_shaped(layers=32, heads=32, tokens=27):
    torch.manual_seed(0)
    pattern = torch.rand(heads, tokens, tokens).softmax(-1)
    pats = [[[round(float(x), 3) for x in row] for row in pattern[h]] for h in range(heads)]
    return {'tokens': ['x']*tokens, 'patterns': [pats for _ in range(layers)], 'top': [[1, .5]]*tokens}


def test_a_trace_sized_result_crosses_in_the_body_not_the_header():
    # 32 layers x 32 heads x 27 x 27 attention patterns: about 5 MB of JSON, five times MAX_HEADER.
    value = trace_shaped()
    fields, tensors = encode_result('trace', value)
    head, views = encode('result', fields, tensors, job='j')
    header_len, payload_len = PREFIX.unpack(head[:PREFIX.size])
    assert header_len < 200 and payload_len > 4*MAX_HEADER
    a, b = socket.socketpair()
    try:
        sender = threading.Thread(target=Channel(a).send, args=('result', fields, tensors), kwargs={'job': 'j'})
        sender.start()
        frame = Channel(b).recv(job='j')
        sender.join(10)
        assert frame.type == 'result' and decode_result('trace', frame) == value
        # A large JSON job argument crosses the same way.
        args = {'prompts': ['y'*1000]*(2*MAX_HEADER//1000)}
        fields, tensors = encode_job('ablation_scan', args)
        sender = threading.Thread(target=Channel(a).send, args=('job', fields, tensors), kwargs={'job': 'k'})
        sender.start()
        frame = Channel(b).recv(job='k')
        sender.join(10)
        assert frame['name'] == 'ablation_scan' and frame['args']['prompts'] == args['prompts']
    finally:a.close();b.close()


def test_the_body_cap_is_enforced_when_encoding_and_tensors_are_not_capped(monkeypatch):
    import wsp.worker_protocol as P
    assert MAX_BODY == 256*1024**2
    monkeypatch.setattr(P, 'MAX_BODY', 64)
    with pytest.raises(ProtocolError, match='body of .* exceeds 64'):encode('result', {'value': 'x'*64}, job='j')
    encode('result', {'v': 'x'*40}, job='j')
    # No hard-coded transfer limit on tensors (B3): only the 32-bit length field bounds them.
    encode('result', {}, {'x': torch.zeros(64)}, job='j')


@pytest.mark.parametrize('key', PROTOCOL_KEYS)
def test_a_field_named_like_a_protocol_key_is_refused_not_dropped(key):
    with pytest.raises(ProtocolError, match='protocol keys'):encode('callback', {'kind': 'operation', key: 1}, job='j')
    # A body that carries one is a corrupt frame.
    body = json.dumps({key: 'x'}).encode()
    a, b = socket.socketpair()
    try:
        a.sendall(raw_frame({'type': 'result', 'job': 'j', 'body': len(body)}, body))
        with pytest.raises(ProtocolError, match='protocol key'):read_frame(b)
    finally:a.close();b.close()


def test_payloads_larger_than_a_receive_chunk_cross_intact():
    sent = {'big': torch.arange(RECV_CHUNK//4*3+7, dtype=torch.float32).reshape(-1, 1),
            'ids': torch.arange(RECV_CHUNK//8+3), 'small': torch.ones(3)}
    value = 'y'*(2*RECV_CHUNK+3)
    a, b = socket.socketpair()
    try:
        sender = threading.Thread(target=Channel(a).send, args=('result', {'value': value}, sent), kwargs={'job': 'j'})
        sender.start()
        frame = Channel(b).recv(job='j')
        sender.join(10)
        assert frame['value'] == value
        for name, t in sent.items():
            assert frame.tensors[name].dtype == t.dtype and torch.equal(frame.tensors[name], t)
        frame.tensors['big'] += 1  # a received tensor is an ordinary writable tensor
        assert float(frame.tensors['big'][0, 0]) == 1.
    finally:a.close();b.close()


def test_a_declared_length_allocates_nothing_before_the_data_arrives(monkeypatch):
    # A peer that declares a 256 MiB body or a 3 GiB tensor and sends nothing costs about one
    # RECV_CHUNK: nothing is allocated (or zero-filled) for the declared length.
    import tracemalloc
    import wsp.worker_protocol as P
    declared, real_empty = [], torch.empty
    def counting_empty(*args, **kwargs):
        if args and args[0] != ():declared.append(args)  # () is the itemsize probe in _entries
        return real_empty(*args, **kwargs)
    monkeypatch.setattr(P.torch, 'empty', counting_empty)
    tensor = 3*1024**3
    for header, payload_len in (({'type': 'result', 'job': 'j', 'body': MAX_BODY}, MAX_BODY),
                                ({'type': 'result', 'job': 'j', 'tensors': [dict(name='x', dtype='float32', shape=[tensor//4],
                                                                                   offset=0, nbytes=tensor)]}, tensor)):
        a, b = socket.socketpair()
        raw, ended = json.dumps(header).encode(), []
        def reader():
            try:read_frame(b)
            except ConnectionLost as exc:ended.append(exc)
        tracemalloc.start()
        try:
            a.sendall(PREFIX.pack(len(raw), payload_len)+raw)
            thread = threading.Thread(target=reader, daemon=True);thread.start()
            time.sleep(.5)
            _, peak = tracemalloc.get_traced_memory()
            a.close();thread.join(5)
        finally:tracemalloc.stop();a.close();b.close()
        assert ended and peak < 4*RECV_CHUNK, (header, peak)
    assert declared == []


def test_a_result_over_the_body_cap_is_that_jobs_error_and_the_worker_stays_up(tmp_path, monkeypatch):
    import wsp.worker_protocol as P
    from wsp import worker_jobs
    monkeypatch.setattr(P, 'MAX_BODY', 8192)
    calls = []
    def call(ctx, **arguments):
        calls.append(1)
        return {'x': 'z'*10000} if len(calls) == 1 else {'ok': 1}
    monkeypatch.setitem(worker_jobs.JOBS, 'patching', worker_jobs.JOBS['patching']._replace(call=call))
    h = Harness(tmp_path); h.load()
    frame, _, _ = h.call('patching')
    assert frame.type == 'error' and 'exceeds 8192' in frame['message'], frame.header
    frame, _, _ = h.call('patching')
    assert frame.type == 'result' and decode_result('patching', frame) == {'ok': 1}
    assert h.thread.is_alive() and h.shutdown() == W.EXIT_OK


def test_a_request_over_the_body_cap_is_unsendable_and_no_loss(monkeypatch):
    import wsp.worker_protocol as P
    from wsp import remote_runner as RR
    monkeypatch.setattr(P, 'MAX_BODY', 4096)
    monkeypatch.setattr(RR, 'atlas_stamp', lambda ws: None)
    a, b = socket.socketpair()
    try:
        runner = object.__new__(RR.RemoteRunner)
        runner.ws = SimpleNamespace(version=1, patches=[], _runtime_client=SimpleNamespace(binding='b', operation_id='op'))
        runner.lock, runner.channel, runner.lost, runner.lost_in = threading.RLock(), Channel(a), None, None
        def sent():raise AssertionError('the oversize request was sent and is waiting for an answer')
        runner._idle = sent  # a sent request would wait here (and fail) instead of hanging the test
        with pytest.raises(RuntimeRefusal) as exc:runner._call('patching', {'clean': 'z'*5000})
        assert exc.value.record['code'] == 'worker_request_unsendable' and 'exceeds 4096' in exc.value.record['message']
        assert runner.lost is None and runner.channel is not None and runner.channel.broken is None
        b.setblocking(False)
        with pytest.raises(BlockingIOError):b.recv(1)  # nothing was sent
    finally:a.close();b.close()


def test_job_id_mismatch_is_detected_by_the_channel():
    a, b = socket.socketpair()
    try:
        Channel(a).send('result', {}, job='other')
        receiver = Channel(b)
        with pytest.raises(JobMismatch):receiver.recv(job='mine')
        assert isinstance(receiver.broken, JobMismatch)
    finally:a.close();b.close()


def test_job_table_and_argument_codec():
    assert set(JOBS) == {'load', 'run', 'logits', 'next_logits', 'generate', 'trace', 'patching', 'ablation_scan', 'attribute',
                         'steer_calibrate', 'probe_reads', 'probe_fires', 'probe_contrast', 'probe_next', 'observe_window',
                         'direction_residuals'}
    assert [n for n, row in JOBS.items() if row.stream] == ['generate']
    assert WORKER_KINDS == {'gguf'} and 'logits' in JOBS and 'tail' in JOBS['logits'].args
    with pytest.raises(ValueError, match='does not take'):encode_job('run', {'ids': [1], 'hooks': {}})
    with pytest.raises(ValueError, match='extra'):encode_job('patching', {}, extra={'w': [D.block((range(2),), torch.ones(2))]})


# ------------------------------------------------------------------ the job loop
def test_token_authentication_closes_and_exits_nonzero(tmp_path):
    h = Harness(tmp_path)
    h.hello('wrong')
    assert h.finish() == W.EXIT_AUTH
    with pytest.raises(ConnectionLost):h.channel.recv()


def test_first_frame_must_be_hello(tmp_path):
    h = Harness(tmp_path)
    # The worker refuses on the header and closes; the body, sent second, may meet the closed socket.
    try:h.channel.send('job', {'name': 'load', 'args': {}}, job='j')
    except ConnectionLost:pass
    assert h.finish() == W.EXIT_AUTH and not h.runners


def test_hello_with_a_payload_or_tensors_is_refused_before_any_allocation(tmp_path, monkeypatch):
    # An unauthenticated peer must not make the worker allocate: the declared 64 MiB
    # tensor is never sent, and nothing is read or allocated for it.
    import wsp.worker_protocol as P
    allocated = []
    real_empty = torch.empty
    def counting_empty(*args, **kwargs):
        allocated.append(args)
        return real_empty(*args, **kwargs)
    monkeypatch.setattr(P.torch, 'empty', counting_empty)
    tensor = [dict(name='x', dtype='float32', shape=[16*1024**2], offset=0, nbytes=64*1024**2)]
    for i, data in enumerate((PREFIX.pack(len(json.dumps({'type': 'hello', 'token': TOKEN, 'tensors': tensor}).encode()), 64*1024**2)
                 + json.dumps({'type': 'hello', 'token': TOKEN, 'tensors': tensor}).encode(),           # declared payload
                 raw_frame({'type': 'hello', 'token': TOKEN, 'tensors': []}),                           # a tensors field at all
                 raw_frame({'type': 'hello', 'token': TOKEN, 'body': 0}),                               # a body field at all
                 raw_frame({'type': 'job', 'job': 'j', 'name': 'load'}),                                # not a hello
                 PREFIX.pack(P.HELLO_MAX_HEADER+1, 0))):                                                # an oversize header
        (tmp_path/str(i)).mkdir()
        h = Harness(tmp_path/str(i))
        h.channel.sock.sendall(data)
        assert h.finish() == W.EXIT_AUTH and not h.runners
        with pytest.raises(ConnectionLost):h.channel.recv()
        assert allocated == []


def test_slow_runner_construction_keeps_idle_telemetry_fresh_and_hands_over(tmp_path, monkeypatch):
    # A load slower than the supervisor's 3 s staleness limit: the idle writer keeps
    # going until the runner exists, then stops for good and the runner's write stands.
    from wsp import memory_guard
    from wsp.runtime_supervisor import runtime_sample
    monkeypatch.setattr(memory_guard, 'snapshot', lambda: {'pressure': 1})
    monkeypatch.setenv('WSP_GUARD_TELEMETRY', str(tmp_path/'runtime'/'watchdog.jsonl'))
    gpu = tmp_path/'runtime'/'watchdog.jsonl.gpu.json'
    class Slow(FakeRunner):
        def __init__(self, ws):
            time.sleep(4.2)
            super().__init__(ws)
        def _telemetry(self):
            write(gpu, dict(time=time.time(), active=True, backend='cpu', pid=os.getpid(), writer='runner'))
    h = Harness(tmp_path, make_runner=Slow, telemetry=lambda runtime: W.IdleTelemetry(runtime, interval=.1))
    failures, done = [], threading.Event()
    def heartbeat():  # the supervisor keeps the lease fresh during a slow load
        while not done.wait(.3):h.lease.write()
    threading.Thread(target=heartbeat, daemon=True).start()
    def load():
        try:h.load()
        except BaseException as exc:failures.append(exc)
    loaded = threading.Thread(target=load, daemon=True)
    start = time.monotonic()
    loaded.start()
    samples = 0
    while loaded.is_alive():
        # Raises 'GPU telemetry lost/stale' once the file is older than 3 s.
        sample = runtime_sample(gpu, True, start, time.monotonic(), pid=os.getpid())
        samples += 'gpu' in sample
        time.sleep(.05)
    done.set()
    assert not failures and time.monotonic()-start > 4 and samples > 60
    assert read_json(gpu)['writer'] == 'runner'
    time.sleep(.5)  # five idle intervals: no idle write follows the hand-over
    assert read_json(gpu)['writer'] == 'runner' and read_json(gpu)['active'] is True
    assert h.shutdown() == W.EXIT_OK


def test_idle_telemetry_stop_admits_no_later_write(tmp_path, monkeypatch):
    monkeypatch.setenv('WSP_GUARD_TELEMETRY', str(tmp_path/'watchdog.jsonl'))
    t = W.IdleTelemetry(tmp_path, interval=.01)
    time.sleep(.05)
    t.stop()
    (tmp_path/'watchdog.jsonl.gpu.json').write_text('{"writer": "runner"}')
    time.sleep(.1)
    t.write()
    assert read_json(tmp_path/'watchdog.jsonl.gpu.json') == {'writer': 'runner'}


def read_json(path):
    return json.loads(Path(path).read_text())


def test_load_checks_admitted_identity_then_reports_runner_facts(tmp_path):
    h = Harness(tmp_path)
    frame = h.load()
    info = decode_result('load', frame)
    assert info['max_ctx'] == 64 and info['threads'] == 3 and info['pid'] == os.getpid()
    assert info['available'] == ['blocks.0.hook_resid_post', 'blocks.1.hook_resid_post']
    ws = h.worker_ws
    assert ws is not h.ws and isinstance(ws._runtime_client, WorkerLease) and ws.version == h.ws.version
    assert h.shutdown() == W.EXIT_OK and h.runners[-1].released


@pytest.mark.parametrize('change', [dict(stat=[0, 0, 0, 0]), dict(base_hash='other'), dict(path='/elsewhere')])
def test_changed_base_refuses_load_before_construction_and_exits_cleanly(tmp_path, change):
    h = Harness(tmp_path)
    h.hello()
    frame, _, _ = h.call('load', descriptor=h.descriptor(**change))
    assert frame.type == 'refusal' and frame['record']['code'] == 'model_changed'
    assert not h.runners and h.finish() == W.EXIT_OK


def test_load_construction_error_is_terminal_error_frame_then_status_zero(tmp_path):
    def failing(ws):raise RuntimeError('No validated packed runtime mapping for scalar')
    h = Harness(tmp_path, make_runner=failing)
    h.hello()
    frame, _, _ = h.call('load')
    assert frame.type == 'error' and frame['exception'] == 'RuntimeError' and 'packed runtime mapping' in frame['message']
    assert h.finish() == W.EXIT_OK


def test_eof_during_a_job_exits_nonzero(tmp_path):
    h = Harness(tmp_path); h.load()
    h.channel.send('job', {**encode_job('generate', {'ids': [1, 2]})[0], 'version': h.ws.version,
                           'manifest': [], 'binding': 'bound'}, job='g')
    assert h.channel.recv(job='g').type == 'callback'
    h.channel.close()
    assert h.finish() == W.EXIT_LOST


def test_eof_between_jobs_is_not_a_clean_exit(tmp_path):
    h = Harness(tmp_path); h.load()
    h.channel.close()
    assert h.finish() == W.EXIT_LOST


def test_callback_reply_for_another_job_closes_the_connection(tmp_path):
    h = Harness(tmp_path); h.load()
    h.channel.send('job', {**encode_job('generate', {'ids': [1]})[0], 'version': h.ws.version,
                           'manifest': [], 'binding': 'bound'}, job='g')
    assert h.channel.recv(job='g').type == 'callback'
    h.channel.send('callback_reply', {'ok': True, 'binding': 'bound'}, job='not-g')
    assert h.finish() == W.EXIT_PROTOCOL
    with pytest.raises(ConnectionLost):h.channel.recv()


def test_requested_jobs_stream_and_return_only_requested_results(tmp_path):
    h = Harness(tmp_path); h.load()
    frame, tokens, callbacks = h.call('generate', {'ids': [1, 2, 3], 'max_new_tokens': 3})
    assert frame.type == 'result' and decode_result('generate', frame) == tokens == [10, 11, 12]
    assert callbacks[0]['kind'] == 'operation' and callbacks[0]['name'] == 'generate' and callbacks[0]['tokens'] == 6
    frame, _, _ = h.call('run', {'ids': [1, 2], 'names': ['blocks.1.hook_resid_post']})
    assert set(frame.tensors) == {'blocks.1.hook_resid_post'} and frame['names'] == ['blocks.1.hook_resid_post']
    frame, _, _ = h.call('logits', {'ids': [1, 2, 3], 'tail': 1})
    assert decode_result('logits', frame).shape == (1, 4)
    frame, _, _ = h.call('next_logits', {'ids': [1]})
    assert decode_result('next_logits', frame).shape == (4,)
    with pytest.raises(ValueError, match='does not take'):encode_job('trace', {'nope': 1})
    frame, _, _ = h.call('trace', {'nope': 1}, unchecked=True)
    assert frame.type == 'error' and 'does not take' in frame['message']
    frame, _, _ = h.call('logits', {'ids': [1]})  # an error frame does not end the worker
    assert frame.type == 'result'
    assert h.shutdown() == W.EXIT_OK


def test_callback_adopts_new_binding_and_refusal_raises_inside_the_job(tmp_path):
    h = Harness(tmp_path); h.load()
    def rebind(header):
        h.lease.binding = 'extended'; h.lease.write()
        return {'ok': True, 'binding': 'extended'}
    frame, _, callbacks = h.call('logits', {'ids': [1, 2]}, on_callback=rebind)
    assert frame.type == 'result' and callbacks[0]['tokens'] == 2
    record = event('forecast_exceeds_budget', 'refused', 'No room for this forward.')
    frame, _, _ = h.call('logits', {'ids': [1, 2, 3]}, on_callback=lambda header: {'ok': False, 'record': record})
    assert frame.type == 'refusal' and frame['record'] == record and frame['tokens'] == []
    assert h.worker_ws._runtime_client.last_event == record
    assert h.call('logits', {'ids': [1]})[0].type == 'result'


def test_revoked_lease_answers_with_refusal_carrying_tokens_so_far(tmp_path):
    h = Harness(tmp_path); h.load()
    record = event('headroom_floor', 'abort', 'Memory headroom fell below the floor.')
    h.runners[-1].on_step = lambda step: step == 2 and h.lease.revoke(record)
    frame, tokens, _ = h.call('generate', {'ids': [1], 'max_new_tokens': 5})
    assert frame.type == 'refusal' and frame['record'] == record
    assert tokens == frame['tokens'] == [10, 11]


# ------------------------------------------------------------------ committed patches
def test_manifest_sync_add_reorder_strength_drop_and_reused_id(tmp_path):
    h = Harness(tmp_path); h.load()
    def logits():
        frame, _, _ = h.call('next_logits', {'ids': [1]})
        assert frame.type == 'result', frame.header
        assert state(h.worker_ws) == state(h.ws) and h.worker_ws.version == h.ws.version
        return float(decode_result('next_logits', frame)[0])
    assert logits() == 0
    a = add_patch(h.ws, '0001-a', created=100.5); b = add_patch(h.ws, '0002-b', created=101.25)
    assert logits() == 2
    h.ws.patches.reverse(); h.ws.rebuild()
    assert logits() == 2 and [p.id for p in h.worker_ws.patches] == ['0002-b', '0001-a']
    a.strength = .5; h.ws._save_patch(a, deltas=False); h.ws.rebuild()
    assert logits() == 1.5
    b.enabled = False; h.ws._save_patch(b, deltas=False); h.ws.rebuild()
    assert logits() == .5
    h.ws.delete_patch('0002-b')
    assert logits() == .5 and [p.id for p in h.worker_ws.patches] == ['0001-a']
    # The same id, reused by a newer patch within the same second, is read again.
    cached = h.worker_ws.patches[0]
    h.ws.delete_patch('0001-a'); add_patch(h.ws, '0001-a', value=2., created=100.75)
    assert logits() == 1 and h.worker_ws.patches[0] is not cached
    assert torch.equal(h.worker_ws.pieces('w')[0][0].values, torch.full((4,), 2.))


def test_unchanged_version_and_manifest_skip_the_rebuild(tmp_path):
    h = Harness(tmp_path); h.load(); add_patch(h.ws, '0001-a')
    h.call('next_logits', {'ids': [1]})
    rebuilt = []
    original = h.worker_ws.rebuild
    h.worker_ws.rebuild = lambda: (rebuilt.append(None), original())
    h.call('next_logits', {'ids': [1]})
    assert not rebuilt
    h.ws.rebuild()  # a new version with the same rows only renumbers the worker
    h.call('next_logits', {'ids': [1]})
    assert not rebuilt and h.worker_ws.version == h.ws.version


@pytest.mark.parametrize('corrupt', ['stored', 'created', 'file', 'missing', 'malformed'])
def test_any_patch_mismatch_refuses_the_job_as_out_of_sync(tmp_path, corrupt):
    h = Harness(tmp_path); h.load()
    add_patch(h.ws, '0001-a', created=50.)
    h.call('next_logits', {'ids': [1]})
    before = state(h.worker_ws)
    add_patch(h.ws, '0002-b', created=60.)
    rows = manifest(h.ws.patches)
    pdir = h.dir/'patches'
    if corrupt == 'stored':rows[1][4] += 1
    elif corrupt == 'created':rows[1][1] = 61.
    elif corrupt == 'file':
        from safetensors.torch import save_file
        save_file({'p0.v': torch.ones(5)}, str(pdir/'0002-b.safetensors'))  # stores 5, the server has 4
    elif corrupt == 'missing':(pdir/'0002-b.safetensors').unlink()
    else:rows[1] = ['0002-b', 60., 'yes', 1., 4]
    frame, _, _ = h.call('next_logits', {'ids': [1]}, rows=rows)
    assert frame.type == 'refusal' and frame['record']['code'] == 'workspace_out_of_sync'
    assert state(h.worker_ws) == before  # nothing applied from a partial sync
    if corrupt in ('stored', 'created', 'malformed'):
        assert h.call('next_logits', {'ids': [1]})[0].type == 'result'  # the worker keeps serving
        assert state(h.worker_ws) == state(h.ws)
    assert h.shutdown() == W.EXIT_OK


def test_load_reads_patch_files_through_the_refactored_reader(tmp_path):
    ws = Workspace(str(tmp_path), SimpleNamespace(path='x'), SimpleNamespace(), None, {'kind': 'gguf'})
    p = Patch('0001-a', 'm', 'a', {'kind': 'manual'}, {'w': [D.block((range(2),), torch.tensor([1., 2.]))]}, strength=.5, created=7.)
    ws._save_patch(p)
    read = Workspace._read_patch(os.path.join(str(tmp_path), 'patches'), '0001-a.json', {})
    assert (read.id, read.created, read.strength, read.stored) == ('0001-a', 7., .5, 2)
    from safetensors.torch import save_file
    legacy = tmp_path/'patches'/'0002-l'
    save_file({'w': torch.ones(2, 2)}, str(legacy)+'.safetensors')
    Path(str(legacy)+'.json').write_text(json.dumps(dict(id='0002-l', message='', author='', source={}, enabled=True, strength=1., created=8.)))
    read = Workspace._read_patch(str(tmp_path/'patches'), '0002-l.json', {'w': (2, 2)})
    assert read.deltas['w'][0].kind == 'block' and read.stored == 4


# ------------------------------------------------------------------ extra and atlas
def test_extra_is_shipped_with_each_job_and_not_retained(tmp_path):
    h = Harness(tmp_path); h.load()
    extra = {'w': [D.block((range(2),), torch.tensor([1.5, 2.5]))]}
    frame, _, _ = h.call('logits', {'ids': [1, 2]}, extra=extra)
    assert float(decode_result('logits', frame)[0, 0]) == 4.
    runner = h.runners[-1]
    gc.collect()
    assert runner.seen[-1] is not None and runner.seen[-1]() is None and runner._extra is None
    frame, _, _ = h.call('logits', {'ids': [1, 2]})  # nothing cached: no extra means none
    assert float(decode_result('logits', frame)[0, 0]) == 0. and runner.seen[-1] is None


def test_atlas_stamp_change_drops_the_cached_atlas(tmp_path):
    h = Harness(tmp_path); h.load()
    h.call('next_logits', {'ids': [1]}, atlas='stamp-1')
    cached = object(); h.worker_ws.keep['atlas'] = cached
    h.call('next_logits', {'ids': [1]}, atlas='stamp-1')
    assert h.worker_ws.keep['atlas'] is cached
    h.call('next_logits', {'ids': [1]}, atlas='stamp-2')
    assert 'atlas' not in h.worker_ws.keep


# ------------------------------------------------------------------ WorkerLease
def active_lease(directory, **lease):
    """The job is bound to 'valid'; ``lease`` overrides only what the supervisor wrote."""
    write(Path(directory)/'lease.json', {**dict(ok=True, binding='valid', pid=1, time=time.time(), worker=dict(pid=os.getpid())), **lease})
    client = WorkerLease(directory)
    client.begin('job', binding='valid', version=3)
    return client


@pytest.mark.parametrize('patches', [0, 200])
def test_worker_lease_check_never_reads_patches_or_the_model(tmp_path, monkeypatch, patches):
    client = active_lease(tmp_path)
    def forbidden(*args, **kwargs):raise AssertionError('worker lease check inspected model/patch metadata')
    synthetic = [Patch(f'{i:04d}', '', '', {}, {}) for i in range(patches)]
    class WS:
        version = 3; lock = threading.RLock(); committed = synthetic
        @property
        def patches(self):return forbidden()
        @property
        def store(self):return forbidden()
    reads = []
    original = Path.read_text
    def counted(self, *args, **kwargs):
        reads.append(self.name);return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', counted)
    for _ in range(100):client.require(WS())
    for _ in range(100):client.layer_checkpoint(WS())
    monkeypatch.undo()
    assert set(reads) == {'lease.json'} and 100 <= len(reads) <= 200  # layer checks may reuse a 50ms read
    write(tmp_path/'lease.json', dict(ok=False, event=event('headroom_floor', 'abort', 'too low')))
    with pytest.raises(RuntimeRefusal, match='too low'):client.require(WS())
    stale = active_lease(tmp_path)
    stale.version = 4
    with pytest.raises(RuntimeRefusal) as exc:stale.require(WS())
    assert exc.value.record['code'] == 'workspace_changed'


@pytest.mark.parametrize('worker', ['absent', None, [], 'pid', {'pid': 1}])
def test_lease_without_this_worker_is_the_normal_refusal_not_a_type_error(tmp_path, worker):
    write(tmp_path/'lease.json', dict(ok=True, binding='valid', pid=os.getpid(), time=time.time(),
          **({} if worker == 'absent' else {'worker': worker})))
    client = WorkerLease(tmp_path); client.begin('job', binding='valid', version=3)
    with pytest.raises(RuntimeRefusal) as exc:client.require(SimpleNamespace(version=3))
    assert exc.value.record['code'] == 'watchdog_unavailable'


@pytest.mark.parametrize('lease', [dict(binding='other'), dict(time=time.time()-5), dict(time='now')])
def test_wrong_binding_or_stale_lease_refuses(tmp_path, lease):
    client = active_lease(tmp_path, **lease)
    with pytest.raises(RuntimeRefusal):client.require(SimpleNamespace(version=3))
    (tmp_path/'lease.json').write_text('[1, 2]')
    with pytest.raises(RuntimeRefusal):client.require(SimpleNamespace(version=3))


def test_worker_lease_refuses_outside_a_job_and_forbids_server_only_calls(tmp_path):
    client = active_lease(tmp_path); client.end()
    with pytest.raises(RuntimeRefusal):client.require(SimpleNamespace(version=3))
    assert client.release() is None
    for call in (lambda: client.record_commit(None, 'p'), client.dismiss, client.bind, lambda: client.mutated(None)):
        with pytest.raises(RuntimeError, match='worker cannot'):call()
    with client.rebuilding():client.mutated(None)
    client.begin('job', binding='valid', version=3)
    ws = SimpleNamespace(version=3, lock=threading.RLock())
    with pytest.raises(RuntimeError, match='worker cannot load'):
        with client.operation(ws, 'load'):pass
    with client.operation(ws, 'analysis'):pass  # no tokens: no callback needed


# ------------------------------------------------------------------ the server-side fork (not wired yet)
def test_uses_worker_only_for_a_worker_mode_runtime_client(tmp_path):
    client = RuntimeClient(tmp_path, {'values': {}})
    assert not uses_worker(SimpleNamespace(kind='gguf', _runtime_client=None))
    assert not uses_worker(SimpleNamespace(kind='gguf', _runtime_client=client))  # worker=False today
    client.worker = True
    assert uses_worker(SimpleNamespace(kind='gguf', _runtime_client=client))
    assert not uses_worker(SimpleNamespace(kind='hf', _runtime_client=client))
    assert not uses_worker(SimpleNamespace(kind='gguf', _runtime_client=WorkerLease(tmp_path)))


def test_worker_job_forwards_whole_call_inside_an_operation(tmp_path):
    from contextlib import contextmanager
    calls = []
    @worker_job('attribute')
    def attribute(ws, prompt, target, versus=None, variant='current', top=20, group=None):
        return 'in-process'
    assert attribute(SimpleNamespace(kind='wsp'), 'p', 't') == 'in-process'
    client = RuntimeClient(tmp_path, {'values': {}}); client.worker = True
    @contextmanager
    def operation(ws, name, **kwargs):
        calls.append(name);yield
    client.operation = operation
    ws = SimpleNamespace(kind='gguf', _runtime_client=client,
                         runner=SimpleNamespace(job=lambda name, **arguments: (name, arguments)))
    assert attribute(ws, 'p', target='t', top=3) == ('attribute', dict(prompt='p', target='t', versus=None, variant='current', top=3, group=None))
    assert calls == ['attribute']
    with pytest.raises(KeyError):worker_job('transfer')


def test_gguf_generate_streams_each_token_before_a_revocation(monkeypatch):
    import wsp.gguf_runner as gguf_runner
    r = object.__new__(gguf_runner.GGUFRunner)
    r.lock = threading.RLock(); r.max_ctx = 32; r.device = 'cpu'; r.ws = SimpleNamespace(hf_config={})
    r.use = lambda *args: None; r.forward_options = lambda tokens: {}
    r._prefill = lambda ids: (torch.tensor([0., 1.]), object())
    r.model = lambda **kwargs: SimpleNamespace(logits=torch.tensor([[[1., 0.]]]), past_key_values=object())
    checks = []
    def checkpoint(ws):
        checks.append(None)
        if len(checks) == 4:raise RuntimeRefusal(event('headroom_floor', 'abort', 'synthetic interruption'))
    monkeypatch.setattr(gguf_runner, 'require_lease', checkpoint)
    streamed = []
    with pytest.raises(RuntimeRefusal) as exc:r.generate([2, 3], max_new_tokens=5, on_token=streamed.append)
    assert streamed == exc.value.tokens == [1, 0]


# ------------------------------------------------------------------ the process entry point
@pytest.fixture
def short_dir():
    # AF_UNIX paths are limited to ~104 bytes; pytest temp paths on macOS exceed it.
    directory = Path(tempfile.mkdtemp(prefix='wsp-t-', dir='/tmp'))
    yield directory
    for path in sorted(directory.rglob('*'), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    directory.rmdir()


def test_listener_accepts_one_connection_or_times_out(short_dir, tmp_path):
    path = str(short_dir/'s')
    listener = socket.socket(socket.AF_UNIX); listener.bind(path); listener.listen(1)
    assert serve_worker(listener, str(tmp_path), TOKEN, runtime_dir=tmp_path, accept_timeout=.2) == W.EXIT_NO_CONNECTION
    os.unlink(path)
    listener = socket.socket(socket.AF_UNIX); listener.bind(path); listener.listen(1)
    status = {}
    thread = threading.Thread(target=lambda: status.update(value=serve_worker(listener, str(tmp_path), TOKEN, runtime_dir=tmp_path)))
    thread.start()
    client = socket.socket(socket.AF_UNIX); client.connect(path)
    channel = Channel(client); channel.send('hello', {'token': TOKEN}); channel.send('shutdown')
    thread.join(5)
    assert status['value'] == W.EXIT_OK and listener.fileno() == -1
    client.close()


def test_worker_module_publishes_idle_telemetry_before_importing_torch(tmp_path):
    code = ('import sys, json; from wsp.worker import IdleTelemetry; t = IdleTelemetry(sys.argv[1], interval=.05);'
            'print(json.dumps(["torch" in sys.modules, json.loads(t.path.read_text())])); t.stop()')
    env = {k: v for k, v in os.environ.items() if k != 'WSP_GUARD_TELEMETRY'}
    out = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=APP, env=env, capture_output=True, text=True, timeout=60)
    imported, value = json.loads(out.stdout)
    assert not imported and value['active'] is False and time.time()-value['time'] < 30
    assert (tmp_path/'watchdog.jsonl.gpu.json').exists()


def test_worker_subcommand_serves_one_authenticated_connection(short_dir, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ('WSP_GUARD_TELEMETRY', 'WSP_WORKER_TOKEN')}
    command = [sys.executable, '-m', 'wsp', 'worker', '--runtime-dir', str(short_dir), '--socket', str(short_dir/'w.sock'),
               '--workspace', str(tmp_path)]
    assert subprocess.run(command, cwd=APP, env=env, capture_output=True, timeout=60).returncode == W.EXIT_AUTH
    process = subprocess.Popen(command, cwd=APP, env={**env, 'WSP_WORKER_TOKEN': TOKEN})
    try:
        deadline = time.monotonic()+30
        while not (short_dir/'w.sock').exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(.02)
        assert json.loads((short_dir/'watchdog.jsonl.gpu.json').read_text())['active'] is False
        client = socket.socket(socket.AF_UNIX); client.connect(str(short_dir/'w.sock'))
        channel = Channel(client); channel.send('hello', {'token': TOKEN}); channel.send('shutdown')
        assert process.wait(30) == W.EXIT_OK and not (short_dir/'w.sock').exists()
        client.close()
    finally:
        if process.poll() is None:process.kill();process.wait()


def test_parent_watcher_exits_a_worker_whose_supervisor_died():
    child = 'import time; from wsp.worker import watch_parent; watch_parent(.05); time.sleep(30)'
    parent = ('import subprocess, sys, time; p = subprocess.Popen([sys.executable, "-c", sys.argv[1]]);'
              'print(p.pid, flush=True); time.sleep(.5)')
    out = subprocess.run([sys.executable, '-c', parent, child], cwd=APP, capture_output=True, text=True, timeout=60)
    pid = int(out.stdout.split()[0])
    deadline = time.monotonic()+10
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:break
        except psutil.NoSuchProcess:break
        time.sleep(.05)
    else:
        psutil.Process(pid).kill()
        pytest.fail('worker outlived its parent')


# ------------------------------------------------------------------ strict JSON results
def test_json_results_cross_as_plain_numbers_and_lists(tmp_path, monkeypatch):
    import numpy as np
    from wsp import worker_jobs
    value = {'np': np.float32(1.5), 'int': np.int64(7), 'flag': np.bool_(True), 'torch': torch.tensor(2.5),
             'array': np.arange(3), 'tensor': torch.tensor([[1., 2.]]), 'tuple': (1, np.float64(.25)), 3: 'key'}
    monkeypatch.setitem(worker_jobs.JOBS, 'patching', worker_jobs.JOBS['patching']._replace(call=lambda ctx, **a: value))
    h = Harness(tmp_path); h.load()
    frame, _, _ = h.call('patching')
    assert frame.type == 'result', frame.header
    assert decode_result('patching', frame) == {'np': 1.5, 'int': 7, 'flag': True, 'torch': 2.5, 'array': [0, 1, 2],
                                                'tensor': [[1., 2.]], 'tuple': [1, .25], '3': 'key'}
    assert h.shutdown() == W.EXIT_OK


def test_large_json_results_and_arguments_cross_the_worker_end_to_end(tmp_path, monkeypatch):
    # Results and arguments far above MAX_HEADER go in the frame body, not the header.
    from wsp import worker_jobs
    value, seen = trace_shaped(), []
    monkeypatch.setitem(worker_jobs.JOBS, 'trace', worker_jobs.JOBS['trace']._replace(call=lambda ctx, **a: value))
    def patching(ctx, **arguments):
        seen.append(arguments['clean'])
        return {'length': len(arguments['clean'])}
    monkeypatch.setitem(worker_jobs.JOBS, 'patching', worker_jobs.JOBS['patching']._replace(call=patching))
    h = Harness(tmp_path); h.load()
    frame, _, _ = h.call('trace', {'prompt': 'x'})
    assert frame.type == 'result', frame.header
    assert decode_result('trace', frame) == value
    clean = 'z'*(2*MAX_HEADER)
    frame, _, _ = h.call('patching', {'clean': clean})
    assert frame.type == 'result' and decode_result('patching', frame) == {'length': len(clean)} and seen == [clean]
    assert h.thread.is_alive() and h.shutdown() == W.EXIT_OK


@pytest.mark.parametrize('bad', ['nan', 'inf', 'object'])
def test_a_result_that_cannot_be_encoded_is_that_jobs_error_and_the_worker_stays_in_sync(tmp_path, monkeypatch, bad):
    import numpy as np
    from wsp import worker_jobs
    values = {'nan': {'kl': float('nan')}, 'inf': [np.float32('inf')], 'object': {'x': object()}}
    calls = []
    def call(ctx, **arguments):
        calls.append(1)
        return values[bad] if len(calls) == 1 else {'ok': np.float32(1)}
    monkeypatch.setitem(worker_jobs.JOBS, 'patching', worker_jobs.JOBS['patching']._replace(call=call))
    h = Harness(tmp_path); h.load()
    frame, _, _ = h.call('patching')
    assert frame.type == 'error', frame.header
    if bad in ('nan', 'inf'):
        assert frame['exception'] == 'ValueError' and 'not JSON compliant' in frame['message']
    # Nothing of the failed result was sent: the next job on the same stream is answered.
    frame, _, _ = h.call('patching')
    assert frame.type == 'result' and decode_result('patching', frame) == {'ok': 1.0}
    assert h.thread.is_alive() and h.shutdown() == W.EXIT_OK
