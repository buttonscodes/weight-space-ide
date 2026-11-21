"""The wiring stage of the atlas: for every neuron, the later neurons whose read direction best matches its write
direction, and the earlier neurons it reads from, by |cosine|.

The search visits the layers as writers in order.  Writer layer a's pass compares its write vectors with the read
vectors of every later layer, which finishes two things:

    down table of layer a        (who reads what a writes)          final after a's own pass
    up table of layer a + 1      (whom a + 1 reads)                 final once every earlier layer has been a writer

The up tables of the layers after that are still being merged: they are the carry state.  After every pass the
finished layer's results, the carry state and a progress marker are written (each to a temporary file, then
`os.replace`), so a stopped build loses at most the layer in progress and continues from the last finished one with
the same arrays, bit for bit.

On disk, inside <atlas>/wiring_partial/ while the stage is unfinished:

    progress.json          the marker: which build this is (`identity`), how many writer passes are done, the name
                           of the carry file, and "searching" or "finishing"
    carry_NNN.npz          the carry state after NNN passes (the name changes every pass, so the marker never names a
                           file that is half written)
    wiring_LLL.npz         final results of layer LLL, in the same form as the finished atlas

When every pass is done the marker says "finishing", the layer files are moved to <atlas>/wiring_LLL.npz, the caller
records the stage in meta.json, and the marker and the directory are removed.  The wiring is complete when, and
only when, meta.json lists the stage, there is no marker, and every layer file is present (`complete`): an atlas
built before this format existed has exactly that form.

Saved progress is used only when `assess` says so: the same identity (model, weights, variant, enabled patches,
number type, torch version, search settings) and every saved file opening with the expected arrays.  The build, the
estimate and the Atlas tab all ask `assess`, so what the tab offers is what the build does.

One build per atlas directory at a time, across threads and processes: `build_lock` holds an exclusive `flock` on
<atlas>/build.lock for the whole build; a second build is refused at once (`Busy`), and the lock goes with the process
if it dies.

The time estimate comes from timing one step of the same search on random matrices on this machine, in the working
number type and with the thread count the build uses, once per process (`step_timing`).
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import threading
import time

try:
    import fcntl
except ImportError:                                   # not on POSIX: the lock then covers this process only
    fcntl = None

import numpy as np
import torch

from .build import layer_vectors, layer_write, work_dtype

FORMAT = 1
KEEP = 2
PARTIAL = "wiring_partial"
MARKER = "progress.json"
PROBE = 4096                        # largest side of the timing run's matrices
LOCK = "build.lock"
UNREADABLE = "the saved progress is unreadable"


# ----------------------------------------------------------------------------------------- shape of the search
def neuron_layers(ws) -> list[int]:
    s = ws.struct
    return [l for l in range(s.n_layers) if s.has_neurons(l)]


def _kinds(s, l) -> tuple:
    return ("gate", "up") if "W_gate" in s.layers[l].mlp else ("gate",)


def _pass_flops(s, ls: list[int], a: int) -> float:
    lw = ls[a]
    return sum(2.0 * s.mlp_width(lw) * s.mlp_width(lr) * s.d_model * len(_kinds(s, lr)) for lr in ls[a + 1:])


def wiring_cost(ws) -> float:
    """Floating-point operations of the whole search, counted as two per multiply-add (the unit of AUTO_WIRING_FLOPS)."""
    s = ws.struct
    ls = neuron_layers(ws)
    return sum(_pass_flops(s, ls, a) for a in range(len(ls) - 1))


# ----------------------------------------------------------------------------------------- one step
def _blank(n: int, keep: int):
    return torch.zeros(n, keep), torch.full((n, keep), -1, dtype=torch.int32), torch.full((n, keep), -1, dtype=torch.int32)


def _merge(cur, v, i, layer: int, keep: int):
    cv, cl, ci = torch.cat([cur[0], v], 1), torch.cat([cur[1], torch.full_like(i, layer)], 1), torch.cat([cur[2], i], 1)
    o = cv.abs().topk(keep, dim=1).indices
    return cv.gather(1, o), cl.gather(1, o), ci.gather(1, o)


def _product(R: torch.Tensor, DT: torch.Tensor) -> torch.Tensor:
    return (R @ DT).float()                                                        # [readers, writers]


def _select(C: torch.Tensor, down, up, lr: int, lw: int, keep: int):
    """Merge one reader layer's cosines into the writer's down table and the reader's up table."""
    ab = C.abs()
    i0 = ab.argmax(0)
    down = _merge(down, C.gather(0, i0[None]).T, i0[:, None].int(), lr, keep)
    i1 = ab.argmax(1)
    up = _merge(up, C.gather(1, i1[:, None]), i1[:, None].int(), lw, keep)
    return down, up


def _unit(M: torch.Tensor, dt) -> torch.Tensor:
    return (M / M.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(dt)


# ----------------------------------------------------------------------------------------- files
# Every change to the progress files goes through these three, so a test can stop the stage at each of them.
def _replace(src: str, dst: str) -> None:
    os.replace(src, dst)


def _unlink(path: str) -> None:
    os.unlink(path)


def _rmtree(path: str) -> None:
    shutil.rmtree(path)


def _write(path: str, write) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    _replace(tmp, path)


def save_npz(path: str, **arrays) -> None:
    _write(path, lambda f: np.savez(f, **arrays))


def save_json(path: str, obj) -> None:
    _write(path, lambda f: f.write(json.dumps(obj, indent=1).encode()))


def final_path(adir: str, l: int) -> str:
    return os.path.join(adir, f"wiring_{l:03d}.npz")


def partial_dir(adir: str) -> str:
    return os.path.join(adir, PARTIAL)


def marker_path(adir: str) -> str:
    return os.path.join(adir, PARTIAL, MARKER)


def complete(adir: str, meta: dict) -> bool:
    """The one definition of a finished wiring stage, used by the build, `Atlas.status()` and the readers."""
    stages = (meta or {}).get("stages") or {}
    if "wiring" not in stages or "weights" not in stages or os.path.exists(marker_path(adir)):
        return False
    return all(os.path.exists(final_path(adir, l)) for l in stages["weights"].get("layers") or [])


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def read_progress(adir: str) -> dict | None:
    """The saved marker, or None.  A marker that cannot be read, or is not of the form the build writes, is
    {"unreadable": True}: still "not complete", and never an exception."""
    p = marker_path(adir)
    if not os.path.lexists(p):
        return None
    try:
        with open(p) as f:
            m = json.load(f)
    except (OSError, ValueError, UnicodeDecodeError):
        return {"unreadable": True}
    ok = (isinstance(m, dict) and isinstance(m.get("identity"), dict) and isinstance(m.get("layers"), list)
          and m["layers"] and all(_int(l) for l in m["layers"]) and _int(m.get("done"))
          and 0 <= m["done"] <= max(0, len(m["layers"]) - 1) and m.get("state") in ("searching", "finishing")
          and (m.get("carry") is None or (isinstance(m["carry"], str) and os.path.basename(m["carry"]) == m["carry"])))
    return m if ok else {"unreadable": True}


def layers_done(passes: int, n: int) -> int:
    """Layers whose results are final after `passes` writer passes (the last pass finishes two)."""
    return n if n and passes >= n - 1 else passes


def cleanup(adir: str) -> None:
    """Remove the marker (from then on the stage is complete), then the rest of the progress directory."""
    if os.path.exists(marker_path(adir)):
        _unlink(marker_path(adir))
    if os.path.isdir(partial_dir(adir)):
        _rmtree(partial_dir(adir))


# ----------------------------------------------------------------------------------------- whose progress it is
def patch_fingerprint(ws) -> list:
    """The enabled patches as they change the weights, the same in every process: [id, strength, created, stored].
    A patch's pieces never change under its id (a commit makes a new patch; `set_patch` changes only enabled and
    strength), but an id can be given again after a deletion (it is the patch count and the time modulo 100,000 s),
    so the commit time and the number of stored values are kept too.  Not `ws.version`: that counts changes in this
    process only and starts again at every load."""
    return [[p.id, float(p.strength), float(p.created), int(p.stored)] for p in ws.patches if p.enabled]


def identity(ws, variant: str, weights_source, keep: int = KEEP) -> dict:
    s = ws.struct
    ls = neuron_layers(ws)
    out = {"format": FORMAT, "model": ws.name, "weights": weights_source, "variant": variant, "layers": ls,
           "widths": [s.mlp_width(l) for l in ls], "d_model": s.d_model, "dtype": str(work_dtype()).split(".")[-1],
           "torch": torch.__version__, "keep": keep}
    if variant != "base":
        out["patches"] = patch_fingerprint(ws)
    return out


def mismatch(saved: dict, want: dict) -> str | None:
    """Why saved progress does not belong to this build (None when it does)."""
    if not saved:
        return "the saved progress is unreadable"
    for key, why in (("format", lambda a, b: "it was saved in another format"),
                     ("model", lambda a, b: f"it was saved for {a}, not {b}"),
                     ("variant", lambda a, b: f"it was saved for the {a} weights and this build maps the {b} weights"),
                     ("patches", lambda a, b: "the patches have changed since"),
                     ("weights", lambda a, b: "the weights stage has been rebuilt from other weights since"),
                     ("dtype", lambda a, b: f"it was computed in {a} and this build works in {b}"),
                     ("torch", lambda a, b: f"it was computed with torch {a} and this build uses torch {b}"),
                     ("keep", lambda a, b: "it keeps a different number of partners"),
                     ("layers", lambda a, b: "the layers with neurons differ"),
                     ("widths", lambda a, b: "the layer widths differ"),
                     ("d_model", lambda a, b: "the model width differs")):
        if saved.get(key) != want.get(key):
            return why(saved.get(key), want.get(key))
    return None


LAYER_DTYPES = {"cos": np.float16, "layer": np.int16, "idx": np.int32}
CARRY_DTYPES = {"cos": np.float32, "layer": np.int32, "idx": np.int32}


def _arrays(path: str, expect: dict) -> dict | None:
    """The arrays of an .npz file if it opens and holds exactly the expected names, dtypes and shapes, else None."""
    try:
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(expect):
                return None
            out = {k: z[k] for k in z.files}
    except Exception:                                    # truncated, garbage, not a zip, a directory, ...
        return None
    for k, (dtype, shape) in expect.items():
        if out[k].dtype != dtype or out[k].shape != shape:
            return None
    return out


def layer_file_ok(path: str, width: int, keep: int = KEEP) -> bool:
    """A finished layer file that opens and has every array of the final format."""
    return _arrays(path, {f"w_{d}_{k}_{x}": (np.dtype(t), (width, keep)) for d in ("down", "up") for k in ("gate", "up")
                          for x, t in LAYER_DTYPES.items()}) is not None


def assess(adir: str, want: dict) -> dict | None:
    """Whether this build can continue the saved progress: the one decision the build, the estimate and the Atlas tab
    all use.  None when nothing is saved; otherwise {"usable", "reason", "done", "total", "next_layer", "variant",
    "state"}, with the loaded carry state under "_state" when usable.  Never raises on damaged files."""
    m = read_progress(adir)
    if m is None:
        return None
    ls, n = want["layers"], len(want["layers"])
    out = {"usable": False, "reason": UNREADABLE, "done": 0, "total": n, "next_layer": None, "variant": None, "state": None}
    if m.get("unreadable"):
        return out
    saved_ls, done = m["layers"], m["done"]
    out.update(done=layers_done(done, len(saved_ls)), total=len(saved_ls), state=m["state"],
               next_layer=saved_ls[done] if done < len(saved_ls) - 1 else None, variant=m["identity"].get("variant"))
    why = mismatch(m["identity"], want)
    if why:
        out["reason"] = why
        return out
    keep, pdir = want["keep"], partial_dir(adir)
    width = dict(zip(ls, want["widths"]))
    in_partial = lambda l: os.path.join(pdir, os.path.basename(final_path(adir, l)))      # noqa: E731
    if m["state"] == "finishing":
        # every layer either still in the progress directory or already moved into place, and each one whole
        if all(layer_file_ok(in_partial(l), width[l], keep) or (not os.path.lexists(in_partial(l))
               and layer_file_ok(final_path(adir, l), width[l], keep)) for l in ls):
            out.update(usable=True, reason=None, _state={"done": done, "state": "finishing"})
        else:
            out["reason"] = "some finished layers are missing or damaged"
        return out
    finished = ls[:done] + ([ls[-1]] if n > 1 and done == n - 1 else [])
    if not all(layer_file_ok(in_partial(l), width[l], keep) for l in finished):
        out["reason"] = "some finished layers are missing or damaged"
        return out
    carry = None
    if done:
        expect = {f"up_{l}_{k}_{x}": (np.dtype(t), (width[l], keep)) for l in ls[done:] for k in ("gate", "up")
                  for x, t in CARRY_DTYPES.items()}
        expect["done"] = (np.dtype(np.int64), ())
        carry = _arrays(os.path.join(pdir, m.get("carry") or "-"), expect) if m.get("carry") else None
        if carry is None or int(carry["done"]) != done:
            out["reason"] = "the saved search state is missing or damaged"
            return out
    out.update(usable=True, reason=None, _state={"done": done, "state": "searching", "carry": carry, "carry_name": m.get("carry")})
    return out


def public(a: dict | None) -> dict | None:
    """`assess` without the loaded arrays, for status and the estimate."""
    return None if a is None else {k: v for k, v in a.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------------------- one build at a time
class Busy(RuntimeError):
    pass


BUSY = ("a build of this atlas is already running{where}; wait for it to finish, then build again "
        "(it continues where it stopped)")
_HELD: dict = {}                                       # atlas directory -> [thread, depth, fd]
_HELD_LOCK = threading.Lock()
_sleep = time.sleep
# A status probe in another process holds a shared flock for a few microseconds (see build_running); a build that
# meets one tries again for up to RETRIES x RETRY_S before it reports Busy.  A running build holds it for hours.
RETRIES, RETRY_S = 10, 0.01
WHERE = {"this IDE": " in this IDE", "another process": " in another process"}


def _would_block(e: OSError) -> bool:
    import errno
    return e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES)


def build_running(adir: str) -> str | None:
    """Whether a build of this atlas holds its lock now: "this IDE", "another process" or None.  Never waits, never
    creates the lock file, and cannot leave the lock held: a shared flock taken and dropped at once (the build
    retries briefly if it meets one, so the probe never makes a real build see Busy)."""
    key = os.path.realpath(adir)
    with _HELD_LOCK:                                   # no build of this process takes or drops the lock meanwhile
        if key in _HELD:
            return "this IDE"
        if fcntl is None:
            return None
        try:
            fd = os.open(os.path.join(adir, LOCK), os.O_RDONLY)
        except OSError:
            return None                                # no lock file: no build has run here
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError as e:
                return "another process" if _would_block(e) else None
            fcntl.flock(fd, fcntl.LOCK_UN)
            return None
        finally:
            os.close(fd)


@contextlib.contextmanager
def build_lock(adir: str):
    """Exclusive, for one atlas directory, across threads and processes; refused at once (Busy), never waited for.
    The same thread may take it again (the server and the MCP tool take it before the workspace lock, and
    `atlas.build` takes it too).  An `flock` on <atlas>/build.lock: the system releases it when the process ends,
    however it ends, so a crash leaves nothing to clean up."""
    key = os.path.realpath(adir)
    me = threading.get_ident()
    with _HELD_LOCK:
        held = _HELD.get(key)
        if held is not None and held[0] != me:
            raise Busy(BUSY.format(where=WHERE["this IDE"]))
        if held is not None:
            held[1] += 1
    if held is not None:
        try:
            yield
        finally:
            with _HELD_LOCK:
                held[1] -= 1
        return
    for attempt in range(RETRIES + 1):
        with _HELD_LOCK:                                 # held while taking the flock (never waits): no thread of
            if key in _HELD:                             # this process can be told "another process" by mistake
                raise Busy(BUSY.format(where=WHERE["this IDE"]))
            os.makedirs(adir, exist_ok=True)
            fd = os.open(os.path.join(adir, LOCK), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                os.close(fd)
                if not _would_block(e):
                    raise
                if attempt == RETRIES:
                    raise Busy(BUSY.format(where=WHERE["another process"])) from None
                fd = None
            if fd is not None:
                _HELD[key] = [me, 1, fd]
                break
        _sleep(RETRY_S)                                  # perhaps only a status probe; a build holds it for hours
    try:
        yield
    finally:
        with _HELD_LOCK:
            _HELD.pop(key, None)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


# ----------------------------------------------------------------------------------------- the stage
def _layer_arrays(down: dict, up: dict, l: int) -> dict:
    r = {}
    for d_, tab in (("down", down), ("up", up)):
        for k in ("gate", "up"):
            v, ll, ii = tab[(l, k)]
            r[f"w_{d_}_{k}_cos"], r[f"w_{d_}_{k}_layer"], r[f"w_{d_}_{k}_idx"] = \
                v.numpy().astype(np.float16), ll.numpy().astype(np.int16), ii.numpy().astype(np.int32)
    return r


def build_wiring(ws, adir: str, log, variant: str = "base", keep: int = KEEP, *, weights_source=None, restart: bool = False,
                 progress=None) -> dict:
    """Run (or continue) the search.  Leaves the stage "finishing": the caller records it in meta.json and then calls
    `cleanup`.  progress, if given, is called with a dict after every layer: phase, layers done, time left."""
    s = ws.struct
    dt = work_dtype()
    ls = neuron_layers(ws)
    n = len(ls)
    pdir = partial_dir(adir)
    want = identity(ws, variant, weights_source, keep)
    saved = None if restart else assess(adir, want)
    state = saved.get("_state") if saved and saved["usable"] else None
    if restart and read_progress(adir) is not None:
        log("  saved wiring progress discarded: a rebuild was requested; starting over")
    elif saved and not saved["usable"]:
        log(f"  saved wiring progress discarded: {saved['reason']}; starting over")
    note = lambda **p: progress and progress({"stage": "wiring", "total": n, **p})      # noqa: E731
    resumed = None
    if state is None:                                                                   # a fresh start
        if os.path.isdir(pdir) and not os.path.islink(pdir):
            _rmtree(pdir)
        elif os.path.lexists(pdir):
            _unlink(pdir)
        os.makedirs(pdir)
        save_json(marker_path(adir), {"identity": want, "layers": ls, "done": 0, "carry": None, "state": "searching", "at": time.time()})
        for l in ls:                                                                    # results of an earlier build are now stale
            if os.path.exists(final_path(adir, l)):
                _unlink(final_path(adir, l))
        state = {"done": 0, "state": "searching", "carry": None, "carry_name": None}
    elif state["done"]:
        resumed = ls[state["done"]] if state["done"] < n - 1 else None
        log(f"  continuing from layer {resumed} of {n} ({state['done']} layers already done)" if resumed is not None
            else f"  continuing: all {n} layers were already done")
    done = state["done"]

    down = {(l, k): _blank(s.mlp_width(l), keep) for l in ls for k in ("gate", "up")}
    up = {(l, k): _blank(s.mlp_width(l), keep) for l in ls for k in ("gate", "up")}
    if state.get("carry"):
        c = state["carry"]
        for l in ls[done:]:
            for k in ("gate", "up"):
                up[(l, k)] = tuple(torch.from_numpy(np.array(c[f"up_{l}_{k}_{x}"])) for x in ("cos", "layer", "idx"))

    if state["state"] == "searching":
        reads = {}
        need = ls[done + 1:] if n > 1 else []
        for i, l in enumerate(need):                                                    # only layers still to be read
            v = layer_vectors(ws, l, variant)
            reads[l] = {"gate": _unit(v["G"], dt), **({"up": _unit(v["U"], dt)} if "U" in v else {})}
            del v
            note(phase="reading", done=layers_done(done, n), read=i + 1, read_total=len(need), layer=l, remaining_s=None,
                 resumed_from=resumed)
            log(f"  wiring: read vectors of layer {l} ready ({i + 1} of {len(need)})")
        left = sum(_pass_flops(s, ls, a) for a in range(done, n - 1))
        flops_run, t_run = 0.0, 0.0
        carry_name = state.get("carry_name")
        for a in range(done, n - 1):
            t0 = time.perf_counter()
            lw = ls[a]
            DT = _unit(layer_write(ws, lw, variant), dt).T.contiguous()
            for lr in ls[a + 1:]:
                for k in _kinds(s, lr):
                    down[(lw, k)], up[(lr, k)] = _select(_product(reads[lr][k], DT), down[(lw, k)], up[(lr, k)], lr, lw, keep)
            del DT
            # checkpoint: the finished layer, then the carry state, then the marker that names it
            save_npz(os.path.join(pdir, os.path.basename(final_path(adir, lw))), **_layer_arrays(down, up, lw))
            if a == n - 2:
                save_npz(os.path.join(pdir, os.path.basename(final_path(adir, ls[-1]))), **_layer_arrays(down, up, ls[-1]))
            new_carry = f"carry_{a + 1:03d}.npz"
            save_npz(os.path.join(pdir, new_carry), done=np.array(a + 1),
                     **{f"up_{l}_{k}_{x}": t.numpy() for l in ls[a + 1:] for k in ("gate", "up")
                        for x, t in zip(("cos", "layer", "idx"), up[(l, k)])})
            save_json(marker_path(adir), {"identity": want, "layers": ls, "done": a + 1, "carry": new_carry,
                                          "state": "searching", "at": time.time()})
            if carry_name and os.path.exists(os.path.join(pdir, carry_name)):
                _unlink(os.path.join(pdir, carry_name))
            carry_name = new_carry
            f =_pass_flops(s, ls, a)
            flops_run += f
            t_run += time.perf_counter() - t0
            left -= f
            rem = t_run / flops_run * left if flops_run > 0 else None
            note(phase="searching", done=layers_done(a + 1, n), layer=lw, remaining_s=rem, resumed_from=resumed)
            log(f"  partners of layer {lw} done ({layers_done(a + 1, n)} of {n} layers" +
                (f", {duration(rem)} left)" if rem is not None and a < n - 2 else ")"))
        if n == 1:                                                                      # one layer: nothing to compare
            save_npz(os.path.join(pdir, os.path.basename(final_path(adir, ls[0]))), **_layer_arrays(down, up, ls[0]))
        save_json(marker_path(adir), {"identity": want, "layers": ls, "done": max(0, n - 1), "carry": carry_name,
                                      "state": "finishing", "at": time.time()})
    for l in ls:                                                                        # finishing: move into place
        src = os.path.join(pdir, os.path.basename(final_path(adir, l)))
        if os.path.exists(src):
            _replace(src, final_path(adir, l))
    missing = [l for l in ls if not os.path.exists(final_path(adir, l))]
    if missing:
        raise RuntimeError(f"wiring results of layer(s) {missing} are missing after the search; build the wiring stage again")
    note(phase="finishing", done=n, remaining_s=0.0, resumed_from=resumed)
    return {"dtype": want["dtype"], "resumed_from": resumed}


# ----------------------------------------------------------------------------------------- the estimate
_TIMINGS: dict = {}
_TIMING_LOCK = threading.Lock()


def step_timing(n_r: int, n_w: int, d: int, dtype=None, threads: int | None = None) -> tuple[float, float]:
    """Seconds for one step of the search on random [n_r, d] x [d, n_w] matrices: (product, selection on its result).
    Measured with the build's own step code, best of two, once per process for each shape, type and thread count."""
    dtype = dtype or work_dtype()
    threads = threads or torch.get_num_threads()
    key = (n_r, n_w, d, str(dtype), threads)
    with _TIMING_LOCK:
        if key not in _TIMINGS:
            g = torch.Generator().manual_seed(0)                 # a small random block, tiled: filling is cheap
            R = (torch.rand(64, d, generator=g) - 0.5).repeat((n_r + 63) // 64, 1)[:n_r].contiguous().to(dtype)
            DT = (torch.rand(d, 64, generator=g) - 0.5).repeat(1, (n_w + 63) // 64)[:, :n_w].contiguous().to(dtype)
            tp, ts = float("inf"), float("inf")
            for _ in range(2):
                t0 = time.perf_counter()
                C = _product(R, DT)
                t1 = time.perf_counter()
                _select(C, _blank(n_w, KEEP), _blank(n_r, KEEP), 1, 0, KEEP)
                t2 = time.perf_counter()
                tp, ts = min(tp, t1 - t0), min(ts, t2 - t1)
                del C
            _TIMINGS[key] = (tp, ts)
        return _TIMINGS[key]


def _mlp_numel_share(s, l: int) -> float:
    """The write matrix's share of a layer's MLP matrices (a third for a gated MLP, half for a plain one)."""
    return 1.0 / len([r for r in ("W_gate", "W_up", "W_in", "W_out") if r in s.layers[l].mlp])


_RECORDED = object()


def estimate(ws, meta: dict | None, adir: str, variant: str = "base", timing=None, restart: bool = False,
             source=_RECORDED) -> dict:
    """How long the stage takes on this machine, what it holds in memory, and what waits for it.

    timing(n_r, n_w, d, dtype, threads) -> (product_s, selection_s) for one step at the probe shape; by default a
    short measurement (`step_timing`).  Search time scales the product by multiply-adds and the selection by the
    number of cosines.  Reading time comes from the weights stage's record of how long one layer's MLP took to read
    (`read_s`), or is reported as not known.  Saved progress that this build would continue is taken into account
    unless restart (a rebuild) is set; `source` is the weights-stage source the build will record (by default the one
    in meta), so the estimate decides about saved progress exactly as the build will (`assess`)."""
    s = ws.struct
    ls = neuron_layers(ws)
    n = len(ls)
    dt = work_dtype()
    threads = torch.get_num_threads()
    meta = meta or {}
    weights = (meta.get("stages") or {}).get("weights") or {}
    state = saved = None
    if weights and not restart:
        saved = assess(adir, identity(ws, variant, weights.get("source") if source is _RECORDED else source))
        state = saved.get("_state") if saved and saved["usable"] else None
    done = state["done"] if state else 0
    finishing = bool(state and state["state"] == "finishing")
    widest = max((s.mlp_width(l) for l in ls), default=0)
    probe = (min(widest, PROBE), min(widest, PROBE), s.d_model)
    search = 0.0
    if n > 1 and not finishing:
        tp, ts = (timing or step_timing)(*probe, dt, threads)
        pf, pe = 2.0 * probe[0] * probe[1] * probe[2], float(probe[0] * probe[1])
        for a in range(done, n - 1):
            nw = s.mlp_width(ls[a])
            for lr in ls[a + 1:]:
                nr = s.mlp_width(lr)
                search += len(_kinds(s, lr)) * (tp * 2.0 * nr * nw * s.d_model / pf + ts * nr * nw / pe)
    rec = weights.get("read_s") or {}
    per_layer = rec.get("mlp")
    readers = ls[done + 1:] if not finishing else []
    writers = ls[done:n - 1] if not finishing else []
    read = None if per_layer is None else per_layer * (len(readers) + sum(_mlp_numel_share(s, l) for l in writers))
    read_note = None if read is not None else (
        "the weights stage has not been built yet; it records how long one layer takes to read" if not weights else
        "the weights stage was built before reading times were recorded; building it again records them")
    size = torch.empty(0, dtype=dt).element_size()
    d = s.d_model
    hold = sum(s.mlp_width(l) * d * size * len(_kinds(s, l)) for l in readers)
    tables = sum(s.mlp_width(l) * KEEP * 12 * 2 * 2 for l in ls[done:])
    loading = max((s.mlp_width(l) * d * 4 * (1 + 2 * len(_kinds(s, l))) for l in readers), default=0)
    step = 0
    for a in range(done, n - 1):
        nw = s.mlp_width(ls[a])
        for lr in ls[a + 1:]:
            nr = s.mlp_width(lr)
            # write vectors as read, normalised and transposed (seen held together at the peak), then the product,
            # its float32 copy when the product is in another type, and its absolute value
            step = max(step, nw * d * (8 + size + (size if size != 4 else 0)) + nr * nw * (8 + (size if size != 4 else 0)))
    waits = ["Run", "Chat", "edits"] if ws.can_run else ["edits"]
    return {"search_s": search, "read_s": read, "read_known": read is not None, "read_note": read_note,
            "total_s": search + read if read is not None else None,
            "memory_bytes": int(hold + tables + max(loading, step)), "read_vectors_bytes": int(hold),
            "dtype": str(dt).split(".")[-1], "threads": threads, "waits": waits,
            "continuing": ({"done": layers_done(done, n), "total": n, "next_layer": ls[done] if done < n - 1 else None}
                           if state and (done or finishing) else None),
            "saved": public(saved) if not restart else None,
            "probe": list(probe), "flops": sum(_pass_flops(s, ls, a) for a in range(done, n - 1))}


def duration(seconds) -> str:
    """'about 1 h 50 min', 'about 12 min', 'about 40 s', 'under a second'."""
    if seconds is None:
        return "an unknown time"
    s = float(seconds)
    r = lambda x: int(math.floor(x + 0.5))              # halves up, as the browser's Math.round  # noqa: E731
    if s < 1:
        return "under a second"
    if s < 90:
        return f"about {r(s)} s"
    if r(s / 60) < 60:
        return f"about {r(s / 60)} min"
    m = r(s / 300.0) * 5
    h, m = divmod(m, 60)
    return f"about {h} h" + (f" {m} min" if m else "")


def summary(est: dict) -> str:
    """The estimate in one line, for the CLI / MCP log."""
    if est["read_known"]:
        t = f"estimate: {duration(est['total_s'])} on this machine, including reading the weights"
    else:
        t = f"estimate: {duration(est['search_s'])} on this machine, plus reading the weights (not yet known)"
    waits = est["waits"]
    who = ", ".join(waits[:-1]) + " and " + waits[-1] if len(waits) > 1 else waits[0]
    return f"{t}; holds about {est['memory_bytes'] / 1e9:.1f} GB while it runs; {who} wait until it finishes"
