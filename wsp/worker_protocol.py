"""Wire protocol between the IDE server and its model worker. No pickle.

A frame is ``struct.pack('>II', header_len, payload_len) + header + payload``:
the header is UTF-8 JSON (at most ``MAX_HEADER`` bytes) naming the frame
``type`` and, after ``hello``, the ``job`` it belongs to; ``tensors`` lists
``{name, dtype, shape, offset, nbytes}`` for the raw payload that follows.
After ``hello`` every other field (job arguments, JSON results, records) travels
as a UTF-8 JSON object after the tensors, ``body`` bytes long, so a large result
or argument list never meets the header limit; ``MAX_BODY`` (256 MiB) bounds it
on both sides (``encode`` refuses to build such a frame, ``read_frame`` refuses
to read one). A field named like a protocol key (``PROTOCOL_KEYS``) is refused,
never dropped. Tensor payloads have no cap beyond the 32-bit length field.
The unauthenticated ``hello`` is header-only (``HELLO_MAX_HEADER``, no payload).
Only the dtypes in ``CODEC_DTYPES`` cross: FP32 values and results, int64
coordinates and token ids. Anything else is refused, never cast; integer GGUF
storage therefore never crosses. Each side copies a tensor once (``sendall`` of
a memoryview over the contiguous tensor; the receiver's tensor shares the buffer
it received into). The receiver never allocates a declared length ahead of the
data: anything over ``RECV_CHUNK`` grows as bytes arrive.

Sockets stay blocking. A reader that must notice a loss nobody reports on the
socket passes ``idle``: it is called after every ``IDLE_SECONDS`` without data
and may raise (the server's ``RemoteRunner`` re-reads the supervisor's
control and lease state there). There is no wall-clock limit on a job.
"""
from __future__ import annotations

import json
import math
import select
import struct
import threading
from dataclasses import dataclass, field

import torch

from .runtime_events import RuntimeRefusal, event

MAX_HEADER = 1024**2
# The JSON body: far above any real result (trace patterns are bounded near 10 MB by
# ``pat_ok``); it exists to stop a corrupt length from allocating gigabytes. Tensor
# payloads have no cap beyond the 32-bit length field (no hard-coded transfer limit).
MAX_BODY = 256 * 1024**2
# Receive step for anything larger: memory grows with the bytes that arrived, never
# with a length the peer declared.
RECV_CHUNK = 1024**2
# Before authentication only a hello is read: no payload, no tensors, a small header.
HELLO_MAX_HEADER = 4096
PREFIX = struct.Struct('>II')
CODEC_DTYPES = {'float32': torch.float32, 'int64': torch.int64}
PROTOCOL_KEYS = ('type', 'job', 'tensors', 'body')
FRAMES = ('hello', 'job', 'callback', 'callback_reply', 'token', 'result', 'refusal', 'error', 'shutdown')
JOB_OPTIONAL = {'shutdown'}  # every frame after hello names its job; only shutdown may name none
IDLE_SECONDS = 1


class ProtocolError(Exception):
    """The stream can no longer be trusted: the connection must be closed."""


class ConnectionLost(ProtocolError):
    """EOF, reset or broken pipe."""


class JobMismatch(ProtocolError):
    """A frame names a different job than the one in progress."""


class TransferUnsupported(RuntimeRefusal):
    """A value that this protocol does not carry; nothing was sent."""
    def __init__(self, message):
        super().__init__(event('transfer_unsupported', 'refused', message,
            recommendations=['Only float32 values and int64 coordinates or token ids cross the model-worker boundary.']))


def dtype_name(t, expected=None):
    for name, dtype in CODEC_DTYPES.items():
        if t.dtype == dtype and (expected is None or name == expected):
            return name
    raise TransferUnsupported(f'{t.dtype} cannot cross the model-worker boundary'
                              + (f' (expected {expected})' if expected else ''))


@dataclass
class Frame:
    header: dict
    tensors: dict = field(default_factory=dict)

    @property
    def type(self):
        return self.header['type']

    @property
    def job(self):
        return self.header.get('job')

    def __getitem__(self, key):
        return self.header[key]

    def get(self, key, default=None):
        return self.header.get(key, default)


def encode(kind, fields=None, tensors=None, *, job=None):
    """-> (prefix + header bytes, payload views). Validates everything before anything is sent.

    ``fields`` go in the header only for ``hello``; otherwise they are the JSON body."""
    if kind not in FRAMES:
        raise ValueError(f'Unknown frame type {kind!r}')
    header, body = {'type': kind}, b''
    if kind == 'hello':
        header = {**(fields or {}), 'type': kind}
    else:
        header['job'] = job
        fields = fields or {}
        clash = sorted(k for k in fields if k in PROTOCOL_KEYS)
        if clash:
            raise ProtocolError(f'Frame fields {clash} are protocol keys and cannot be sent')
        if fields:
            body = json.dumps(fields).encode()
            if len(body) > MAX_BODY:
                raise ProtocolError(f'Frame body of {len(body)} bytes exceeds {MAX_BODY}')
    entries, views, offset = [], [], 0
    for name, t in (tensors or {}).items():
        if not isinstance(t, torch.Tensor):
            raise TransferUnsupported(f'{name} is not a tensor')
        dtype = dtype_name(t)
        t = t.detach()
        if t.device.type != 'cpu':
            raise TransferUnsupported(f'{name} is on {t.device}; only CPU tensors cross')
        shape = list(t.shape)
        t = t.contiguous().reshape(-1)
        nbytes = t.numel() * t.element_size()
        entries.append(dict(name=str(name), dtype=dtype, shape=shape, offset=offset, nbytes=nbytes))
        if nbytes:
            views.append(memoryview(t.numpy()).cast('B'))
        offset += nbytes
    if entries:
        header['tensors'] = entries
    if body:
        header['body'] = len(body)
        views.append(memoryview(body))
    raw = json.dumps(header).encode()
    if len(raw) > MAX_HEADER:
        raise ProtocolError(f'Frame header of {len(raw)} bytes exceeds {MAX_HEADER}')
    return PREFIX.pack(len(raw), offset + len(body)) + raw, views


def _wait_readable(sock, idle, interval=IDLE_SECONDS):
    """Block until ``sock`` has data (or EOF/error); ``idle()`` after each quiet ``interval``."""
    if hasattr(select, 'poll'):
        poller = select.poll()
        poller.register(sock, select.POLLIN | select.POLLPRI | select.POLLERR | select.POLLHUP)
        ready = lambda: poller.poll(interval * 1000)  # noqa: E731
    else:
        ready = lambda: select.select([sock], [], [sock], interval)[0]  # noqa: E731
    while True:
        try:
            if ready():
                return
        except (OSError, ValueError) as exc:
            raise ConnectionLost(f'{type(exc).__name__}: {exc}') from exc
        idle()


def _recv_some(sock, view, idle=None):
    """At least one byte into ``view``; -> how many."""
    if idle is not None:
        _wait_readable(sock, idle)
    try:
        n = sock.recv_into(view)
    except OSError as exc:
        raise ConnectionLost(f'{type(exc).__name__}: {exc}') from exc
    if not n:
        raise ConnectionLost('Connection closed')
    return n


def _recv_into(sock, view, idle=None):
    while len(view):
        view = view[_recv_some(sock, view, idle):]


def _recv_exact(sock, n, idle=None):
    buffer = bytearray(n)
    _recv_into(sock, memoryview(buffer), idle)
    return bytes(buffer)


def _recv_growing(sock, n, idle=None):
    """``n`` bytes as a bytearray that grows only as they arrive: a peer that declares
    gigabytes and sends nothing costs one ``RECV_CHUNK``, never the declared length."""
    if n <= RECV_CHUNK:
        buffer = bytearray(n)
        _recv_into(sock, memoryview(buffer), idle)
        return buffer
    buffer, scratch = bytearray(), memoryview(bytearray(RECV_CHUNK))
    while len(buffer) < n:
        buffer += scratch[:_recv_some(sock, scratch[:min(RECV_CHUNK, n - len(buffer))], idle)]
    return buffer


def _entries(header, payload_len):
    """-> (tensor entries, body length); the body follows the tensors and ends the payload."""
    body = header.get('body', 0)
    if type(body) is not int or body < 0:
        raise ProtocolError('Frame body length must be a non-negative integer')
    if body > MAX_BODY:
        raise ProtocolError(f'Frame body of {body} bytes exceeds {MAX_BODY}')
    entries = header.get('tensors', [])
    if not isinstance(entries, list):
        raise ProtocolError('Frame tensors must be a list')
    offset, names = 0, set()
    for e in entries:
        if not isinstance(e, dict) or not isinstance(e.get('name'), str) or e['name'] in names:
            raise ProtocolError('Frame tensor entries need distinct names')
        names.add(e['name'])
        if e.get('dtype') not in CODEC_DTYPES:
            raise ProtocolError(f"Unsupported tensor dtype {e.get('dtype')!r}")
        shape = e.get('shape')
        if not isinstance(shape, list) or not all(type(d) is int and d >= 0 for d in shape):
            raise ProtocolError(f"Bad tensor shape for {e['name']}")
        itemsize = torch.empty((), dtype=CODEC_DTYPES[e['dtype']]).element_size()
        if type(e.get('nbytes')) is not int or e['nbytes'] != math.prod(shape) * itemsize:
            raise ProtocolError(f"Tensor {e['name']} nbytes does not match its shape and dtype")
        if e.get('offset') != offset:
            raise ProtocolError(f"Tensor {e['name']} is not contiguous in the payload")
        offset += e['nbytes']
    if offset + body != payload_len:
        raise ProtocolError('Frame payload length does not match its tensors and body')
    return entries, body


def read_frame(sock, *, idle=None, hello=False):
    """One frame. ``hello``: the unauthenticated first frame, which must be a ``hello`` with
    no payload and no tensors; anything else is refused before any allocation. A body over
    ``MAX_BODY`` is refused before it is read; body and tensors are received as they arrive
    (``_recv_growing``), so a declared length allocates nothing ahead of the data."""
    prefix = _recv_exact(sock, PREFIX.size, idle)
    header_len, payload_len = PREFIX.unpack(prefix)
    if hello and (payload_len or header_len > HELLO_MAX_HEADER):
        raise ProtocolError('The first frame must be a hello without payload')
    if header_len > MAX_HEADER:
        raise ProtocolError(f'Frame header of {header_len} bytes exceeds {MAX_HEADER}')
    try:
        header = json.loads(str(_recv_growing(sock, header_len, idle), 'utf-8'))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:  # RecursionError: deeply nested JSON
        raise ProtocolError(f'Frame header is not JSON: {exc}') from exc
    if not isinstance(header, dict) or header.get('type') not in FRAMES:
        raise ProtocolError('Frame has no known type')
    if hello and (header['type'] != 'hello' or 'tensors' in header or 'body' in header):
        raise ProtocolError('The first frame must be a hello without tensors')
    if header['type'] != 'hello':
        if 'job' not in header:
            raise ProtocolError(f"{header['type']} frame does not name its job")
        if header['type'] not in JOB_OPTIONAL and not (isinstance(header['job'], str) and header['job']):
            raise ProtocolError(f"{header['type']} frame needs a job id")
    tensors = {}
    entries, body = _entries(header, payload_len)
    for e in entries:
        dtype = CODEC_DTYPES[e['dtype']]
        # The tensor shares the received buffer (frombuffer copies nothing). Up to RECV_CHUNK the socket
        # writes straight into it; a larger one is copied twice (socket -> scratch chunk -> growing
        # buffer), and growing the bytearray can also move it.
        tensors[e['name']] = (torch.frombuffer(_recv_growing(sock, e['nbytes'], idle), dtype=dtype).reshape(e['shape'])
                              if e['nbytes'] else torch.empty(e['shape'], dtype=dtype))
    header.pop('tensors', None)
    if header.pop('body', None):
        try:
            # Strict UTF-8 (json.loads on bytes would also accept UTF-16/32); the str is the one
            # decoded copy json.loads would make itself, with no bytes() copy before it.
            fields = json.loads(str(_recv_growing(sock, body, idle), 'utf-8'))
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise ProtocolError(f'Frame body is not JSON: {exc}') from exc
        if not isinstance(fields, dict):
            raise ProtocolError('Frame body must be a JSON object')
        if any(k in PROTOCOL_KEYS for k in fields):
            raise ProtocolError('Frame body repeats a protocol key')
        header = {**fields, **header}
    return Frame(header, tensors)


class Channel:
    """One connected stream socket; sends are serialized, reads belong to one thread."""

    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()
        self.broken = None

    def send(self, kind, fields=None, tensors=None, *, job=None):
        self.send_encoded(encode(kind, fields, tensors, job=job))

    def send_encoded(self, encoded):
        """Send the ``encode`` result of a frame validated beforehand."""
        head, views = encoded
        with self.lock:
            try:
                self.sock.sendall(head)
                for view in views:
                    self.sock.sendall(view)
            except OSError as exc:
                self.broken = ConnectionLost(f'{type(exc).__name__}: {exc}')
                raise self.broken from exc

    def recv(self, *, job=None, idle=None, hello=False):
        """Next frame; with ``job``, a frame for any other job is a protocol violation.
        ``idle`` is called after each quiet ``IDLE_SECONDS``; a ProtocolError it raises breaks the channel.
        ``hello``: the unauthenticated first frame (see ``read_frame``)."""
        try:
            frame = read_frame(self.sock, idle=idle, hello=hello)
        except ProtocolError as exc:
            self.broken = exc
            raise
        if job is not None and frame.job != job:
            self.broken = JobMismatch(f'{frame.type} frame for job {frame.job!r} during job {job!r}')
            raise self.broken
        return frame

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


# ------------------------------------------------------------------ manifest and extra codecs
def manifest(patches):
    """Server order; ``created`` distinguishes a reused id within one second."""
    return [[p.id, p.created, bool(p.enabled), p.strength, int(p.stored)] for p in patches]


def parse_manifest(rows):
    out = []
    if not isinstance(rows, list):
        raise ValueError('Patch manifest must be a list')
    for row in rows:
        if not isinstance(row, list) or len(row) != 5:
            raise ValueError('Patch manifest rows are [id, created, enabled, strength, stored]')
        pid, created, enabled, strength, stored = row
        if (not isinstance(pid, str) or not pid or '/' in pid or pid.startswith('.') or type(enabled) is not bool
                or type(stored) is not int or not all(type(x) in (int, float) for x in (created, strength))):
            raise ValueError(f'Malformed patch manifest row {row!r}')
        out.append((pid, created, enabled, strength, stored))
    if len({r[0] for r in out}) != len(out):
        raise ValueError('Patch manifest repeats an id')
    return out


EXTRA_PREFIX = 'extra.'


def encode_extra(extra):
    """{tensor: [pieces]} -> (header value, tensors). Coordinates are int64, values float32."""
    from . import delta as D
    index, tensors = D.pack(extra)
    for key, t in tensors.items():
        dtype_name(t, 'int64' if key.endswith('.c') else 'float32')
    return {'index': index}, {EXTRA_PREFIX + k: t for k, t in tensors.items()}


def decode_extra(value, tensors):
    from . import delta as D
    if value is None:
        return None
    raw = {k[len(EXTRA_PREFIX):]: t for k, t in tensors.items() if k.startswith(EXTRA_PREFIX)}
    for key, t in raw.items():
        if t.dtype != CODEC_DTYPES['int64' if key.endswith('.c') else 'float32']:
            raise ValueError(f'extra tensor {key} has dtype {t.dtype}')
    try:
        return D.unpack(value['index'], raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Malformed extra: {exc}') from exc
