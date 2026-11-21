"""Programmer conversations kept on disk, so they survive a server restart.

Each conversation lives in ``<workspace>/conversations/<id>/``, a sibling of the agent's working
directory ``<workspace>/agent`` (outside it: a sandbox that confines writes to the working directory, such as
Codex's workspace-write, cannot write there; any process of this user can still read it, see docs/PROGRAMMER.md):

    events.jsonl   one JSON event per line, appended as events arrive (the tab's own events)
    meta.json      provider, provider session id, mode, model, task, times, status; replaced atomically
    owner.lock     held (flock) by the IDE process that runs the conversation, for as long as it runs it

Directories are 0700 and files 0600. The workspace root is ignored by Git (``workspaces/``), as for the
workspace's patches/ and atlas/; ``<workspace>/conversations/.gitignore`` (``*``) also covers a root kept
elsewhere. Reading a conversation never contacts a provider.

The store does not trust its own files. The workspace directory, the store root and each conversation
directory are opened with O_DIRECTORY | O_NOFOLLOW and must be directories owned by this user; the descriptor
is kept, and every file is opened, created, renamed and removed relative to it (``dir_fd``), so a directory
swapped for a link after the check cannot redirect a write. Every file open uses O_NOFOLLOW and O_NONBLOCK
and must give a regular file owned by this user with a single link (a hard link to a file elsewhere is
refused). Every field read from meta.json is type-checked and coerced; meta.json and event lines are size-
bounded before they are parsed. Any failure while reading one entry turns that entry into one unreadable row
(or, for an event line, a skipped line): it never stops listing, opening, Start or Resume.

Ownership: the process running a conversation holds an exclusive flock on its owner.lock. Where the
filesystem does not support flock (detected per filesystem, reported by ``locking_note``), the recorded pid
is the fallback, with the person's confirmed "mark interrupted" as the remedy.

Bounds (see docs/PROGRAMMER.md): ``MAX_EVENT_BYTES`` per saved event (long strings clipped, the event marked
``clipped``); ``MAX_EVENTS`` lines or ``MAX_FILE_BYTES`` per conversation (then one notice, later events
live only); the newest ``KEEP`` conversations per workspace (older ones nobody runs are deleted when a new
conversation starts).

A crash can leave a torn last line; the next append starts on a new line and reading skips the fragment.
A conversation whose meta.json says a turn was in progress, which no live IDE process holds, was cut off by
a server stop: it is recorded as ``interrupted`` the first time it is listed. The custom command line of a
legacy run is never saved (it can carry tokens); only the harness kind is.
"""
from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import re
import secrets
import shutil
import stat
import threading
import time

MAX_EVENT_BYTES = 64 * 1024
MAX_EVENTS = 50_000
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_META_BYTES = 1024 * 1024
KEEP = 50
ROOT_NAME = "conversations"
LIVE = ("starting", "running", "waiting", "interrupting", "closing")   # a turn (or startup/shutdown) was in progress
FINAL = ("closed", "failed", "interrupted", "done", "stopped")
STATUSES = frozenset(LIVE) | frozenset(FINAL) | {"idle"}
ID = re.compile(r"conv-[0-9]{8}-[0-9]{6}-[0-9a-f]{8}")
DIR_MODE, FILE_MODE = 0o700, 0o600
NOFOLLOW = os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
NONBLOCK = getattr(os, "O_NONBLOCK", 0)     # a FIFO planted in the store must not block the open; no effect on files
DIRFLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | NOFOLLOW
NO_LOCK_ERRNOS = {getattr(errno, n) for n in ("ENOTSUP", "EOPNOTSUPP", "ENOLCK") if hasattr(errno, n)}
BOOT = secrets.token_hex(8)        # this IDE process, recorded with the pid (for display; the lock decides)
_HELD: dict[tuple, tuple] = {}     # (st_dev, st_ino) of a conversation directory -> (lock fd, the Conversation holding it)
_HELD_LOCK = threading.Lock()
_KEY_LOCKS: dict[tuple, threading.Lock] = {}   # in-process exclusion of settle / claim / probe on one conversation
_UNLOCKABLE: set[int] = set()      # st_dev of filesystems on which flock is not supported


class Unreadable(ValueError):
    """A conversation entry that cannot be used (missing, damaged, or not a plain private file/directory)."""


class Busy(RuntimeError):
    """Another live IDE process (or a run of this one that is still finishing) holds this conversation."""


def root(ws_dir: str) -> str:
    return os.path.join(ws_dir, ROOT_NAME)


# ------------------------------------------------------------------ safe file primitives
def _open_dir(name: str, dir_fd: int | None = None) -> int:
    """A descriptor of a directory owned by this user, never through a link."""
    try:
        fd = os.open(name, DIRFLAGS, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise Unreadable(f"{os.path.basename(name)} is not a directory of this user (a link or another owner's entry)") from None
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
            raise Unreadable(f"{os.path.basename(name)} is not a directory of this user (a link or another owner's entry)")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open(name: str, flags: int, dir_fd: int) -> int:
    """A regular file of this user with one link, opened relative to its directory, never through a link."""
    try:
        fd = os.open(name, flags | NOFOLLOW | NONBLOCK, FILE_MODE, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise Unreadable(f"{name} is a link") from None
        if exc.errno in (errno.EISDIR, errno.ENXIO):
            raise Unreadable(f"{name} is not a regular file of this user") from None
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
            raise Unreadable(f"{name} is not a regular file of this user")
        if st.st_nlink != 1:
            raise Unreadable(f"{name} has another link (a hard link to a file elsewhere)")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _write_json_atomic(dir_fd: int, name: str, data: dict) -> None:
    tmp = f"{name}.{secrets.token_hex(4)}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, FILE_MODE, dir_fd=dir_fd)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)   # replaces a planted link itself, never its target
    except BaseException:
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
        raise


def _open_root(ws_dir: str, create: bool) -> int | None:
    """The store root's descriptor (created when `create`), or None when there is no store yet."""
    ws_fd = _open_dir(ws_dir)
    try:
        if create:
            try:
                os.mkdir(ROOT_NAME, DIR_MODE, dir_fd=ws_fd)
            except FileExistsError:
                pass
        try:
            fd = _open_dir(ROOT_NAME, ws_fd)
        except FileNotFoundError:
            return None
    finally:
        os.close(ws_fd)
    if create:
        os.fchmod(fd, DIR_MODE)                         # through the descriptor: never a link's target
        try:
            gfd = os.open(".gitignore", os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, FILE_MODE, dir_fd=fd)
        except OSError:
            pass                                        # already there (or planted): never written through
        else:
            with os.fdopen(gfd, "w") as f:
                f.write("# Saved programmer conversations: generated, private, never committed.\n*\n")
    return fd


def _lines(fd: int, limit: int = MAX_EVENT_BYTES + 1024, total: int = MAX_FILE_BYTES + MAX_EVENT_BYTES):
    """The file's lines (bytes, without the newline), reading at most `total` bytes. A line longer than `limit`
    is never held in memory or parsed: it is yielded as None."""
    buf, over, read = b"", False, 0
    while read < total:
        chunk = os.read(fd, min(1 << 16, total - read))
        if not chunk:
            break
        read += len(chunk)
        buf += chunk
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                if len(buf) > limit:
                    over, buf = True, b""
                break
            line, buf = buf[:nl], buf[nl + 1:]
            yield None if over or len(line) > limit else line
            over = False
    if buf or over:
        yield None if over or len(buf) > limit else buf


# ------------------------------------------------------------------ meta.json, checked and coerced
def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except (OverflowError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _int(v, default=0):
    return v if isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 2 ** 53 else default


def _str(v, limit=20_000):
    return v[:limit] if isinstance(v, str) else None


def _one_of(v, allowed, default=None):
    return v if isinstance(v, str) and v in allowed else default


def _task(v) -> dict:
    if not isinstance(v, dict):
        return {}
    out = {}
    for k in ("title", "instructions"):
        if isinstance(v.get(k), str):
            out[k] = v[k][:100_000]
    if isinstance(v.get("scope"), list):
        out["scope"] = [x[:4096] for x in v["scope"][:10_000] if isinstance(x, str)]
    if "suite" in v:
        out["suite"] = v["suite"][:4096] if isinstance(v["suite"], str) else None
    if "preserve" in v:
        p = v["preserve"]
        out["preserve"] = ({k[:256]: _num(x) for k, x in list(p.items())[:100] if isinstance(k, str) and _num(x) is not None}
                           if isinstance(p, dict) else None)
    return out


# A provider session id as this code accepts one, from meta.json or from a saved `init`: letters, digits, `.`, `_`, `:`
# and `-`, starting with a letter or digit, at most 128 characters. Claude Code session ids and Codex thread ids are
# UUIDs; the tests' fixtures are words such as `session-1`. A resume hands the id to the provider (the Claude SDK
# passes it to the CLI's --resume), so a string a command line could take for an option, or a blank one, is never one.
SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def session_or_none(v) -> str | None:
    return v if isinstance(v, str) and SESSION_ID.fullmatch(v) else None


def coerce(raw, cid: str) -> dict:
    """The meta fields this code uses, each of the expected type; anything else is dropped or defaulted."""
    raw = raw if isinstance(raw, dict) else {}
    resumes = raw.get("resumes")
    resumes = [{"at": _num(r.get("at")), "run": _str(r.get("run"), 64)} for r in resumes[-100:] if isinstance(r, dict)] \
        if isinstance(resumes, list) else []
    return {
        "id": cid, "version": 2, "started": _num(raw.get("started")), "ended": _num(raw.get("ended")),
        "status": _one_of(raw.get("status"), STATUSES, "interrupted"),
        "session": session_or_none(raw.get("session")), "session_scanned": raw.get("session_scanned") is True,
        "n_events": _int(raw.get("n_events")), "lines": _int(raw.get("lines")),
        "provider": _str(raw.get("provider"), 32) or "unknown", "native": raw.get("native") is True,
        "mode": _str(raw.get("mode"), 16) or "tools", "model": _str(raw.get("model"), 256) or "", "task": _task(raw.get("task")),
        "workspace": _str(raw.get("workspace"), 4096), "billing": _one_of(raw.get("billing"), ("subscription", "per_call")),
        "budget": _num(raw.get("budget")), "pid": _int(raw.get("pid"), None), "boot": _str(raw.get("boot"), 64),
        "run": _str(raw.get("run"), 64), "extra": _str(raw.get("extra"), 100_000) or "",
        "resume_error": _str(raw.get("resume_error")), "resume_error_at": _num(raw.get("resume_error_at")),
        "last_resume_failure": _str(raw.get("last_resume_failure")), "last_resume_failure_at": _num(raw.get("last_resume_failure_at")),
        "interrupted": _str(raw.get("interrupted"), 1000), "closed_reason": _str(raw.get("closed_reason"), 1000),
        "resumes": resumes, "truncated": raw.get("truncated") is True, "owner_override": raw.get("owner_override") is True,
    }


def _read_meta(dir_fd: int, cid: str) -> dict:
    try:
        fd = _open("meta.json", os.O_RDONLY, dir_fd)
    except FileNotFoundError:
        raise Unreadable("meta.json is missing") from None
    except Unreadable:
        raise
    except OSError as exc:
        raise Unreadable(f"meta.json cannot be opened ({exc.strerror or exc})") from None
    with os.fdopen(fd, "rb") as f:
        data = f.read(MAX_META_BYTES + 1)
    if len(data) > MAX_META_BYTES:
        raise Unreadable("meta.json is larger than 1 MiB")
    try:
        raw = json.loads(data)
    except RecursionError:
        raise Unreadable("meta.json is nested too deeply") from None
    except Exception:
        raise Unreadable("meta.json is not valid JSON") from None
    if not isinstance(raw, dict) or raw.get("id") != cid:
        raise Unreadable("meta.json does not describe this conversation")
    return coerce(raw, cid)


# ------------------------------------------------------------------ ownership
def _key_lock(key) -> threading.Lock:
    with _HELD_LOCK:
        return _KEY_LOCKS.setdefault(key, threading.Lock())


def _flock_try(fd: int, op: int, dev: int):
    """True: taken; False: another process holds it; None: this filesystem does not support flock."""
    try:
        fcntl.flock(fd, op | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    except OSError as exc:
        if exc.errno in NO_LOCK_ERRNOS:
            _UNLOCKABLE.add(dev)
            return None
        raise


def _pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OverflowError:
        return False
    return True


def _pid_elsewhere(meta: dict) -> bool:
    """The no-flock fallback: the record says a live process other than this one has it open. (This process's
    own pid means an earlier process whose pid was reused: this one would hold it in _HELD if it ran it. Another
    process's boot token cannot be checked, so a reused pid there looks alive: the person's confirmed "mark
    interrupted" is the remedy.)"""
    if meta.get("status") not in LIVE + ("idle",):
        return False
    pid = meta.get("pid")
    if pid == os.getpid():
        return False
    return _pid_alive(pid)


def _key(x):
    if isinstance(x, Conversation):
        return x.key
    st = os.stat(x, follow_symlinks=False)
    return (st.st_dev, st.st_ino)


def held_here(x) -> bool:
    """True when this process holds the conversation (a Conversation or its directory path)."""
    try:
        key = _key(x)
    except OSError:
        return False
    with _HELD_LOCK:
        return key in _HELD


def locking_note(ws_dir: str) -> str | None:
    """A sentence for the tab when this workspace's store is on a filesystem without flock."""
    try:
        dev = os.stat(root(ws_dir), follow_symlinks=False).st_dev
    except OSError:
        return None
    if dev in _UNLOCKABLE:
        return ("The filesystem of this workspace does not support file locks, so saved conversations cannot be "
                "protected against two IDE processes using the same one. The recorded process id is checked "
                "instead; if it is wrong, mark the conversation interrupted.")
    return None


class Conversation:
    """One conversation's files, reached only through its directory descriptor. Appends are serialised; meta is
    replaced atomically. close() when done (a run closes its record when it ends)."""

    def __init__(self, path: str, meta: dict, dir_fd: int):
        self._dfd = None                               # set last: a failed __init__ leaves the caller to close dir_fd
        self.path = path
        self.id = meta["id"]
        self.meta = meta
        st = os.fstat(dir_fd)
        self.key, self.dev = (st.st_dev, st.st_ino), st.st_dev
        self._lock = threading.Lock()
        self._full = bool(meta.get("truncated"))
        try:
            self._bytes = os.stat("events.jsonl", dir_fd=dir_fd, follow_symlinks=False).st_size
        except FileNotFoundError:
            self._bytes = 0
        self._lines = meta.get("lines") or meta.get("n_events") or 0
        self._shown = meta.get("n_events") or 0
        self._last_stream = None                       # (kind, id) of the last saved streamed piece
        self._dfd = dir_fd

    @classmethod
    def _at(cls, root_fd: int, name: str, path: str) -> "Conversation":
        dfd = _open_dir(name, root_fd)
        try:
            return cls(path, _read_meta(dfd, name), dfd)
        except BaseException:
            os.close(dfd)
            raise

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -------------------------------------------------------------- create / open
    @classmethod
    def create(cls, ws_dir: str, live_ids: set[str] | None = None, **fields) -> "Conversation":
        """A new conversation, already claimed by this process. Other conversations' files never stop it."""
        try:
            prune(ws_dir, KEEP - 1, live_ids)
        except Exception:
            pass                                        # pruning is housekeeping; never a reason not to save
        root_fd = _open_root(ws_dir, create=True)
        try:
            now = time.time()
            cid = time.strftime("conv-%Y%m%d-%H%M%S-", time.localtime(now)) + secrets.token_hex(4)
            os.mkdir(cid, DIR_MODE, dir_fd=root_fd)
            dfd = _open_dir(cid, root_fd)
        finally:
            os.close(root_fd)
        try:
            os.fchmod(dfd, DIR_MODE)
            os.close(_open("events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, dfd))
            meta = coerce({"id": cid, "started": now, "status": "starting", "workspace": os.path.realpath(ws_dir), **fields}, cid)
            conv = cls(os.path.join(root(ws_dir), cid), meta, dfd)
        except BaseException:
            os.close(dfd)
            raise
        conv.claim()
        return conv

    @classmethod
    def open(cls, ws_dir: str, cid: str) -> "Conversation":
        if not isinstance(cid, str) or not ID.fullmatch(cid):
            raise Unreadable("Unknown conversation")
        try:
            root_fd = _open_root(ws_dir, create=False)
        except FileNotFoundError:
            root_fd = None
        if root_fd is None:
            raise Unreadable("Unknown conversation")
        try:
            return cls._at(root_fd, cid, os.path.join(root(ws_dir), cid))
        except FileNotFoundError:
            raise Unreadable("Unknown conversation") from None
        except Unreadable:
            raise
        except Exception as exc:
            raise Unreadable(f"{type(exc).__name__}: {exc}") from None
        finally:
            os.close(root_fd)

    @property
    def _dir(self) -> int:
        """The directory descriptor; never None (a None dir_fd would mean the current directory)."""
        if self._dfd is None:
            raise Unreadable("This conversation record is closed.")
        return self._dfd

    # -------------------------------------------------------------- ownership
    def _lock_fd(self, create: bool) -> int:
        return _open("owner.lock", (os.O_RDWR | os.O_CREAT) if create else os.O_RDONLY, self._dir)

    def claim(self) -> None:
        """Hold the owner lock for this process and record it (pid, boot token). Busy when another live process
        holds it, or a run of this process is still finishing with it."""
        with _key_lock(self.key):
            with _HELD_LOCK:
                held = _HELD.get(self.key)
                if held is not None:
                    if held[1] is self:
                        return
                    raise Busy("This IDE process is still finishing a run of this conversation; try again in a moment.")
            fd = self._lock_fd(create=True)
            try:
                got = _flock_try(fd, fcntl.LOCK_EX, self.dev)
                if got is False or (got is None and _pid_elsewhere(self.meta)):
                    pid = self.meta.get("pid")
                    raise Busy(f"Another IDE process on this machine{f' (pid {pid})' if pid else ''} is running this conversation.")
            except BaseException:
                os.close(fd)
                raise
            with _HELD_LOCK:
                _HELD[self.key] = (fd, self)
        try:
            self.update(pid=os.getpid(), boot=BOOT, owner_override=False)
        except BaseException:
            self.release()                              # a record this process could not write is not this process's
            raise

    def release(self) -> None:
        with _HELD_LOCK:
            held = _HELD.get(self.key)
            if held is None or held[1] is not self:
                return
            del _HELD[self.key]
        os.close(held[0])

    def close(self) -> None:
        self.release()
        fd, self._dfd = self._dfd, None
        if fd is not None:
            os.close(fd)

    # -------------------------------------------------------------- writing
    def _write_meta(self) -> None:
        _write_json_atomic(self._dir, "meta.json", self.meta)

    def update(self, **fields) -> None:
        with self._lock:
            self.meta = coerce({**self.meta, **fields, "n_events": self._shown, "lines": self._lines}, self.id)
            self._write_meta()

    def append(self, event: dict, sync: bool = False) -> None:
        """Append one event. Past the limits one notice is saved and later events are dropped."""
        with self._lock:
            if self._full:
                return
            saved = clip_event(event)
            line = (json.dumps(saved, default=str) + "\n").encode()
            if self._lines + 1 >= MAX_EVENTS or self._bytes + len(line) > MAX_FILE_BYTES - 1024:
                self._full = True
                line = (json.dumps({"kind": "raw", "i": event.get("i"), "t": event.get("t", time.time()),
                                    "text": f"Saved transcript limit reached ({MAX_EVENTS:,} events or "
                                            f"{MAX_FILE_BYTES // 2**20} MiB); later events were shown live but not saved."})
                        + "\n").encode()
            fd = _open("events.jsonl", os.O_RDWR | os.O_APPEND | os.O_CREAT, self._dir)   # read: the torn-tail check
            try:
                size = os.fstat(fd).st_size
                if size and os.pread(fd, 1, size - 1) != b"\n":
                    os.write(fd, b"\n")                 # a torn last line (crash mid-append) stays a line of its own
                    size += 1
                os.write(fd, line)
                if sync or self._full:
                    os.fsync(fd)
                self._bytes = size + len(line)
            finally:
                os.close(fd)
            self._lines += 1
            key = (saved.get("kind"), saved.get("id"))
            if not (key[0] in ("text_delta", "thinking_delta") and key == self._last_stream and isinstance(saved.get("text"), str)):
                self._shown += 1                        # streamed pieces of one block are one event when shown
            self._last_stream = key if key[0] in ("text_delta", "thinking_delta") else None
            if self._full:
                self.meta = coerce({**self.meta, "truncated": True, "n_events": self._shown, "lines": self._lines}, self.id)
                self._write_meta()

    # -------------------------------------------------------------- reading
    def events(self) -> list[dict]:
        """Saved events, renumbered; consecutive streamed pieces of one block are joined into one. A line that is
        too long, not JSON, nested too deeply or not an event is skipped."""
        out: list[dict] = []
        try:
            fd = _open("events.jsonl", os.O_RDONLY, self._dir)
        except FileNotFoundError:
            return out
        try:
            for raw in _lines(fd):
                if raw is None:
                    continue
                try:
                    ev = json.loads(raw)
                except Exception:                       # a torn fragment, binary, or nesting too deep
                    continue
                if not isinstance(ev, dict) or not isinstance(ev.get("kind"), str):
                    continue
                last = out[-1] if out else None
                if (last is not None and ev["kind"] in ("text_delta", "thinking_delta") and last.get("kind") == ev["kind"]
                        and last.get("id") == ev.get("id") and isinstance(ev.get("text"), str) and isinstance(last.get("text"), str)):
                    last["text"] = last["text"] + ev["text"]
                    last["t"] = ev.get("t", last.get("t"))
                    continue
                out.append(ev)
        finally:
            os.close(fd)
        for i, ev in enumerate(out):
            ev["i"] = i
        return out

    def raw_lines(self) -> int:
        try:
            fd = _open("events.jsonl", os.O_RDONLY, self._dir)
        except FileNotFoundError:
            return 0
        try:
            return sum(1 for _ in _lines(fd))
        finally:
            os.close(fd)

    def last_write(self):
        try:
            return os.stat("events.jsonl", dir_fd=self._dir, follow_symlinks=False).st_mtime
        except (OSError, Unreadable):
            return self.meta.get("started")


def clip_event(event: dict, limit: int | None = None) -> dict:
    """The event as saved: unchanged when it fits, else with its long strings clipped to fit."""
    limit = limit or MAX_EVENT_BYTES
    line = json.dumps(event, default=str)
    if len(line.encode()) <= limit:
        return event
    out = dict(event, clipped=True)
    budget = max(256, limit // 2)

    def clip(value):
        if isinstance(value, str):
            return value if len(value) <= budget else value[:budget] + f"… [{len(value) - budget} more characters not saved]"
        if isinstance(value, dict):
            return {k: clip(v) for k, v in list(value.items())[:200]}
        if isinstance(value, list):
            return [clip(v) for v in value[:200]]
        return value

    for _ in range(8):
        out = {k: (v if k in ("kind", "i", "t", "id") else clip(v)) for k, v in out.items()}
        if len(json.dumps(out, default=str).encode()) <= limit:
            return out
        budget //= 4
    return {k: event[k] for k in ("kind", "i", "t", "id") if k in event} | {"clipped": True, "text": "[event too large to save]"}


# ------------------------------------------------------------------ holders, settling, listing, pruning
def holder(conv: Conversation) -> str | None:
    """'lock' when another live process holds the conversation's lock, 'pid' when (on a filesystem without
    flock) its record names another live process, None otherwise. A bad lock file raises Unreadable."""
    with _key_lock(conv.key):
        if held_here(conv):
            return None
        try:
            fd = conv._lock_fd(create=False)
        except FileNotFoundError:
            return "pid" if conv.dev in _UNLOCKABLE and _pid_elsewhere(conv.meta) else None
        try:
            got = _flock_try(fd, fcntl.LOCK_SH, conv.dev)
        finally:
            os.close(fd)                                # also drops the probe's shared lock
        if got is False:
            return "lock"
        if got is None and _pid_elsewhere(conv.meta):
            return "pid"
        return None


def locked_elsewhere(x) -> bool:
    """True when another live IDE process holds the conversation (a Conversation, or its directory path)."""
    if isinstance(x, Conversation):
        return holder(x) is not None
    parent, name = os.path.split(os.path.normpath(x))
    pfd = _open_dir(parent)
    try:
        conv = Conversation._at(pfd, name, x)
    finally:
        os.close(pfd)
    try:
        return holder(conv) is not None
    finally:
        conv.close()


def _settled_fields(conv: Conversation) -> dict:
    meta, last = conv.meta, conv.last_write()
    if meta["status"] == "closing":
        return dict(status="interrupted", ended=last, interrupted="The IDE server stopped while the conversation was closing.")
    if meta["status"] in LIVE:
        return dict(status="interrupted", ended=last, interrupted="The IDE server stopped while a turn was running.")
    return dict(status="closed", ended=last, closed_reason="The IDE server stopped while the conversation waited for your message.")


def _session_id(v) -> bool:
    return session_or_none(v) is not None


def session_from_events(events) -> str | None:
    """The provider session id a conversation's own saved events confirm: the session of its last trusted `init`.

    For records saved before the session id was written to meta.json at `init` (until then it was written at the
    next status change, the end of the first turn). An `init` is not trusted when
      - it follows a `resumed` marker that names another session (a fresh session started in place of the resumed
        one), or
      - the next event after it (notices aside) is a `resume_failed` that does not name that same session (an
        attempt that failed at that very init). A failed attempt to resume the right session names it, so it does
        not make that session untrusted.
    No version of the store ever saved such an `init`: each turned a wrong-session init into a `resume_failed`
    before saving anything. The two exclusions matter only for a damaged or hand-edited record.
    A `resume_failed` never carries a session that could be taken from it: only `init` events give one, and only a
    session id of the accepted form (SESSION_ID)."""
    trusted: list[str] = []
    expected = None                                 # the session a `resumed` marker says this part continues
    just_init = None                                # the last init, while nothing but notices followed it
    for ev in events:
        kind = ev.get("kind") if isinstance(ev, dict) else None
        if kind == "raw":
            continue
        if kind == "resumed":
            expected = ev.get("session") if _session_id(ev.get("session")) else None
        elif kind == "init":
            s = ev.get("session")
            if _session_id(s) and (expected is None or s == expected):
                trusted.append(s)
                just_init = s
                continue
        elif kind == "resume_failed" and just_init is not None and ev.get("session") != just_init:
            trusted.pop()
        just_init = None
    return trusted[-1] if trusted else None


def _needs_session(meta: dict) -> bool:
    """A native record with no session id whose saved events have not been looked at for one yet."""
    return bool(meta.get("native")) and not meta.get("session") and not meta.get("session_scanned")


def _session_fields(meta: dict, events) -> dict:
    """What a record without a session id gets written once its saved events have been read: the session they
    confirm (session_from_events), or `session_scanned` when they confirm none, so that they are read for it once
    per record and not at every listing. Empty for a record that needs nothing (see _needs_session)."""
    if not _needs_session(meta):
        return {}
    session = session_from_events(events)
    return {"session": session} if session else {"session_scanned": True}


def _with_session(conv: Conversation, meta: dict, events=None) -> dict:
    """`meta` with _session_fields applied, not written (for a record this process may not write). Reading the
    events is the store's bounded reader; any failure leaves `meta` as it is."""
    if not _needs_session(meta):
        return meta
    try:
        return {**meta, **_session_fields(meta, conv.events() if events is None else events)}
    except Exception:
        return meta


def _record_session(conv: Conversation, meta: dict) -> dict:
    """A final record without a session id: its events are read once for it and the answer written (under the
    owner lock, so that no process claims it meanwhile). A record another process holds is never written: its
    meta is returned with the answer, unsaved."""
    with _key_lock(conv.key):
        if held_here(conv):
            return conv.meta
        fd = conv._lock_fd(create=True)
        try:
            got = _flock_try(fd, fcntl.LOCK_EX, conv.dev)
            if got is False or (got is None and _pid_elsewhere(meta)):
                return _with_session(conv, meta)
            try:
                fields = _session_fields(meta, conv.events())
            except Exception:
                return meta                             # unreadable events: the record as it is
            try:
                conv.update(**fields)
            except OSError:
                return {**meta, **fields}
            return conv.meta
        finally:
            os.close(fd)


def settle(conv: Conversation, live_ids: set[str]) -> dict:
    """The conversation's meta, with a run cut off by a server stop recorded as interrupted.

    A native record with no session id has its saved events read for one once (see _session_fields): what they
    confirm is written with it, so a later listing does not read them again.

    Never writes to a record that a live IDE process (this one or another) holds."""
    meta = conv.meta
    if conv.id in live_ids or held_here(conv):
        return meta
    if meta["status"] in FINAL:
        return _record_session(conv, meta) if _needs_session(meta) else meta
    with _key_lock(conv.key):                            # a claim in another thread waits for this, and vice versa
        if held_here(conv):
            return conv.meta
        fd = conv._lock_fd(create=True)                  # hold the lock while writing, so no process claims it meanwhile
        try:
            got = _flock_try(fd, fcntl.LOCK_EX, conv.dev)
            if got is False:
                return {**meta, "elsewhere": True, "holder": "lock"}     # another IDE process runs it: leave it alone
            if got is None and _pid_elsewhere(meta):
                return {**meta, "elsewhere": True, "holder": "pid"}
            events = conv.events()
            conv._shown, conv._lines = len(events), conv.raw_lines()   # the meta's counts predate the stop
            fields = {**_settled_fields(conv), **_session_fields(meta, events)}   # its own events confirm the session
            try:
                conv.update(**fields)
            except OSError:
                return {**meta, **fields, "n_events": conv._shown}
            return conv.meta
        finally:
            os.close(fd)


def mark_interrupted(conv: Conversation) -> dict:
    """The person's confirmed "It is not running; mark interrupted".

    Refused (Busy) while another live process holds the conversation's lock: a held flock means that process
    has it open. Where the filesystem has no flock, the recorded pid cannot be verified the same way, and the
    person's confirmation is taken as the remedy: the record is marked interrupted with `owner_override`."""
    with _key_lock(conv.key):
        if held_here(conv):
            raise Busy("This IDE process is running this conversation; close it instead.")
        fd = conv._lock_fd(create=True)
        try:
            got = _flock_try(fd, fcntl.LOCK_EX, conv.dev)
            if got is False:
                pid = conv.meta.get("pid")
                raise Busy(f"An IDE process{f' (pid {pid})' if pid else ''} still has this conversation open, so it is "
                           "running there. Close the conversation in that IDE window, or stop that process, then try again.")
            events = conv.events()
            conv._shown, conv._lines = len(events), conv.raw_lines()
            pid = conv.meta.get("pid")
            # The session its events confirm, as a listing would record it: the verdict on this answer is the same.
            fields = {"owner_override": got is None, "ended": conv.meta.get("ended") or conv.last_write(),
                      **_session_fields(conv.meta, events)}
            if conv.meta["status"] not in FINAL:
                fields.update(status="interrupted", interrupted=(
                    f"Marked interrupted by you: the IDE process recorded as running it{f' (pid {pid})' if pid else ''} "
                    "was not running it."))
            conv.update(**fields)
            return conv.meta
        finally:
            os.close(fd)


def _entries(ws_dir: str):
    """[(name, Conversation | None, reason)] for each entry of the store, bad ones included. Close the
    Conversations when done."""
    try:
        root_fd = _open_root(ws_dir, create=False)
    except FileNotFoundError:
        return []
    except Exception as exc:
        return [("conversations", None, str(exc) or type(exc).__name__)]
    if root_fd is None:
        return []
    out = []
    try:
        names = sorted(n for n in os.listdir(root_fd) if ID.fullmatch(n))
        for name in names:
            try:
                out.append((name, Conversation._at(root_fd, name, os.path.join(root(ws_dir), name)), None))
            except Exception as exc:
                out.append((name, None, str(exc) or type(exc).__name__))
    finally:
        os.close(root_fd)
    return out


def prune(ws_dir: str, keep: int = KEEP, live_ids: set[str] | None = None) -> None:
    """Delete the oldest conversations beyond `keep`, except one that is running here or elsewhere, or whose
    files cannot be checked (a bad entry or lock file is skipped, never deleted and never a reason to fail)."""
    entries = _entries(ws_dir)
    try:
        found = [(c.meta.get("started") or 0, n, c) for n, c, _ in entries if c is not None]
        found.sort(key=lambda x: (x[0], x[1]), reverse=True)
        doomed = []
        for _, name, conv in found[keep:]:
            try:
                if (live_ids and name in live_ids) or held_here(conv) or holder(conv) is not None:
                    continue
            except Exception:
                continue
            doomed.append(name)
    finally:
        for _, c, _ in entries:
            if c is not None:
                c.close()
    if not doomed:
        return
    root_fd = _open_root(ws_dir, create=False)
    if root_fd is None:
        return
    try:
        for name in doomed:
            try:
                os.close(_open_dir(name, root_fd))       # a real directory of this user, checked again
                shutil.rmtree(name, dir_fd=root_fd)      # relative to the root's descriptor; never follows a link
            except Exception:
                continue
    finally:
        os.close(root_fd)


def first_line(task: dict | None) -> str:
    task = task or {}
    for value in (task.get("title"), task.get("instructions")):
        if isinstance(value, str) and value.strip():
            return value.strip().splitlines()[0][:200]
    return "(no title or instructions)"


def listing(ws_dir: str, live_ids: set[str]) -> list[dict]:
    """Settled metadata of the workspace's conversations, newest first; a damaged entry is an `unreadable` row."""
    out = []
    for name, conv, reason in _entries(ws_dir):
        if conv is None:
            out.append({"id": name, "unreadable": reason, "started": None})
            continue
        try:
            out.append(settle(conv, live_ids))
        except Exception as exc:
            out.append({"id": name, "unreadable": str(exc) or type(exc).__name__, "started": None})
        finally:
            conv.close()
    out.sort(key=lambda m: m.get("started") or 0, reverse=True)
    return out


def summary(meta: dict, live: bool = False) -> dict:
    if meta.get("unreadable"):
        return {"id": meta["id"], "unreadable": meta["unreadable"]}
    return {"id": meta["id"], "started": meta.get("started"), "ended": meta.get("ended"), "provider": meta.get("provider"),
            "mode": meta.get("mode"), "model": meta.get("model") or "", "status": meta.get("status"),
            "n_events": meta.get("n_events", 0), "task": first_line(meta.get("task")), "session": meta.get("session"),
            "native": bool(meta.get("native")), "live": live, "elsewhere": bool(meta.get("elsewhere")),
            "holder": meta.get("holder") if meta.get("elsewhere") else None,
            "owner_pid": meta.get("pid") if meta.get("elsewhere") else None,
            "run": meta.get("run"), "truncated": bool(meta.get("truncated")), "billing": meta.get("billing"),
            "budget": meta.get("budget"), "last_resume_failure": meta.get("last_resume_failure")}
