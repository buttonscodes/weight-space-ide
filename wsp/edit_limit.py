"""How large one weight patch may be, and what a large one costs. No tensors are read here.

The limit is the smallest of these constraints, each stated with its numbers when it refuses:

- **setting**: the "Largest weight patch" setting (``patch_limit_gib``, GiB of FP32 patch: 4 bytes per weight,
  the size of the patch file). Auto (``None``) sets no limit of its own. It never overrides memory. It applies at
  every size.
- **memory**: under the runtime supervisor, the largest patch its admission would grant now (the ``advise``
  request: the same ``runtime_policy.admission`` over the descriptor ``preflight_edit`` sends, binding and
  reserving nothing). Without one, the machine's available memory (``memory_guard.snapshot``) less the
  admission headroom reserve and the edit's read scratch, at 12 bytes per weight, the figure memory admission
  charges. ``watchdog`` stands in for memory when the supervisor admits no edit for another reason (the model
  file changed, a configuration mismatch, its memory ladder).
- **disk**: free space where the patch is written, less 1 GiB, at 4 bytes per weight.

Memory and disk are measured only for an edit of more than ``SMALL_EDIT`` weights, and never refuse one of that
size or smaller: such a patch is charged less than the 64 MiB read scratch every edit is already admitted with,
and before this limit existed nothing but its own preflight checked it.

``fallback`` (the old fixed 60,000,000 weights) applies only where the old limit applied, to planning a selection
edit, and only when memory cannot be measured and nothing is set. A script is bounded by the setting, a memory
measurement and disk, and by nothing else (see ``wscript.Interp._add``).

Under the supervisor the real ``edit_preflight`` calls stay the final word on memory: this module only says
beforehand what they allow.
"""
from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass, field

GiB = 1024 ** 3
PATCH_BYTES = 4                 # an FP32 delta per weight: the patch file and the pieces in memory
MEMORY_BYTES = 12               # memory admission's charge per patch weight (runtime_supervisor.PATCH_ADMISSION_BYTES)
CONFIRM_ABOVE = 60_000_000      # the browser asks before a selection edit larger than this (the old fixed limit)
FALLBACK = 60_000_000           # selection planning only, when memory cannot be measured and nothing is set
DISK_RESERVE = GiB              # free disk left after the patch file is written
TIMED_MIN = 1_000_000           # an edit this large or larger is timed for the next estimate
SMALL_EDIT = 1_000_000          # memory and disk are measured only above this (12 MB, below the 64 MiB read scratch)
SNAPSHOT_SECONDS = 2.0          # a machine memory reading is reused this long (it runs vm_stat and sysctl)
MEASURED = ("memory", "watchdog", "disk")       # the constraints SMALL_EDIT exempts

telemetry = None                # tests replace this; default memory_guard.snapshot
_machine: tuple[float, dict] | None = None
_machine_lock = threading.Lock()

SETTING = "Largest weight patch"
RESERVE = "Admission headroom reserve"


@dataclass
class Limit:
    weights: int                         # the largest patch, in weights, these constraints allow
    binding: str                         # setting | memory | watchdog | disk | fallback | none
    detail: dict = field(default_factory=dict)
    found: dict = field(default_factory=dict)          # every constraint checked: {name: (weights, detail)}

    def exceeded(self, written: int) -> "Limit | None":
        """The constraint a patch of `written` weights breaks (the tightest; the setting on a tie), or None.
        Memory and disk never refuse a patch of SMALL_EDIT weights or fewer."""
        if written <= self.weights:
            return None
        over = {k: v for k, v in self.found.items() if written > v[0] and not (k in MEASURED and written <= SMALL_EDIT)}
        if not over:
            return None
        k = min(over, key=lambda k: (over[k][0], k != "setting"))
        return Limit(weights=over[k][0], binding=k, detail=over[k][1], found=self.found)


def gib(n_bytes: float) -> str:
    v = n_bytes / GiB
    if v >= 0.95:
        return f"{v:.1f} GiB"
    return f"{n_bytes / 1024**2:.0f} MiB" if n_bytes >= 1024**2 else "less than 1 MiB"


def setting(ws) -> float | None:
    values = getattr(ws, "edit_settings", None) or {}
    v = values.get("patch_limit_gib")
    return float(v) if v is not None else None


def setting_weights(gib_value: float) -> int:
    return int(gib_value * GiB) // PATCH_BYTES


def _values(ws) -> dict:
    from .runtime_config import DEFAULTS
    return {**DEFAULTS, **(getattr(ws, "edit_settings", None) or {})}


def machine() -> dict:
    """The machine's memory now (cached SNAPSHOT_SECONDS). Raises when it cannot be measured."""
    global _machine
    from . import memory_guard
    with _machine_lock:
        now = time.monotonic()
        if _machine is None or now - _machine[0] > SNAPSHOT_SECONDS:
            _machine = (now, (telemetry or memory_guard.snapshot)())
        return _machine[1]


def forget_machine() -> None:
    global _machine
    with _machine_lock:
        _machine = None


def _scratch(scratch: int) -> int:
    from .runtime_supervisor import SCRATCH_ADMISSION_QUANTUM
    return max(int(scratch), SCRATCH_ADMISSION_QUANTUM)


def _memory(ws, scratch: int):
    """(name, weights, detail): what memory allows now. name is "memory", or "watchdog" when the supervisor
    admits no edit for a reason other than memory; weights is None when it cannot be told (detail["reason"])."""
    client = getattr(ws, "_runtime_client", None)
    if client is not None:
        advise = getattr(client, "advise_edit", None)
        advice = advise(ws, scratch) if advise is not None else None
        if not advice or advice.get("largest") is None:
            reason = (advice or {}).get("reason") or "the runtime client gives no advice"
            return "memory", None, dict(source="runtime", reason=reason)
        detail = dict(source="runtime", **{k: advice[k] for k in ("available", "preferred", "scratch", "event", "bounds")
                                           if k in advice})
        code = (advice.get("event") or {}).get("code")
        name = "watchdog" if advice["largest"] == 0 and code not in ("forecast_exceeds_budget", "memory_pressure") else "memory"
        return name, int(advice["largest"]), detail
    try:
        from .runtime_supervisor import policy_for
        snap = machine()
        policy = policy_for(_values(ws), snap["total"], snap["available"])
    except Exception as exc:                        # unsupported OS, missing telemetry, inconsistent settings
        return "memory", None, dict(source="machine", reason=f"memory telemetry failed: {type(exc).__name__}: {exc}")
    allowance = _scratch(scratch)
    # In-process nothing else admits the edit: the measured room is the limit, whatever the pressure level
    # (macOS reports level 2 on many healthy machines; the supervisor's own admission does refuse under it).
    detail = dict(source="machine", available=snap["available"], preferred=policy.preferred, scratch=allowance,
                  pressure=snap.get("pressure"))
    return "memory", max(0, snap["available"] - policy.preferred - allowance) // MEMORY_BYTES, detail


def _disk(ws):
    directory = getattr(ws, "dir", None)
    if not directory:
        return None, {}
    try:
        free = shutil.disk_usage(directory).free
    except OSError:
        return None, {}
    return max(0, free - DISK_RESERVE) // PATCH_BYTES, dict(free=free, reserve=DISK_RESERVE)


def allowed(ws, written: int, *, scratch: int = 0, memory: bool = True, fallback: bool = False) -> Limit:
    """The largest patch an edit of `written` weights may store now.

    The setting applies at every size; memory (`memory=True`) and disk are measured only when `written` is more
    than SMALL_EDIT. `scratch`: the bytes the edit reads at once (as preflight counts them). `fallback`: when
    memory cannot be measured and nothing is set, apply the old fixed FALLBACK (selection planning only)."""
    found: dict[str, tuple[int, dict]] = {}
    s = setting(ws)
    if s is not None:
        found["setting"] = (setting_weights(s), dict(gib=s))
    if written > SMALL_EDIT:
        if memory:
            name, w, detail = _memory(ws, scratch)
            if w is not None:
                found[name] = (w, detail)
            elif fallback and s is None:
                found["fallback"] = (FALLBACK, detail)
        w, d = _disk(ws)
        if w is not None:
            found["disk"] = (int(w), d)
    if not found:
        return Limit(weights=1 << 62, binding="none")
    binding = min(found, key=lambda k: (found[k][0], k != "setting"))
    return Limit(weights=found[binding][0], binding=binding, detail=found[binding][1], found=found)


def over_setting(ws, weights: int) -> Limit | None:
    """The setting's Limit when a patch of `weights` is larger than it allows, else None (no reads, no memory)."""
    s = setting(ws)
    if s is None or weights <= setting_weights(s):
        return None
    n = setting_weights(s)
    return Limit(weights=n, binding="setting", detail=dict(gib=s), found={"setting": (n, dict(gib=s))})


# ---------------------------------------------------------------------- refusals

SMALLER = "Select fewer neurons or a smaller slice"
NOT_SETTING = f"Raising {SETTING} in Settings does not override memory."


def _cap_source(cap: int, bounds: dict, *, freeing: bool = False) -> tuple[str, str]:
    """(what the static cap is, how to raise it). `freeing`: freeing memory does help (do not say otherwise)."""
    note = "" if freeing else " (freeing memory does not help)"
    for key, label in (("budget_gib", "Memory budget"), ("ceiling_gib", "Memory ceiling")):
        v = bounds.get(key)
        if v is not None and int(v * GiB) == cap:
            return f"the {label} setting", f"raise {label} in Settings{note}"
    return (f"physical memory less the {RESERVE}",
            f"lower the {RESERVE}{note}" if freeing else f"use a smaller model or lower the {RESERVE}{note}")


def _then(n: int, remedy: str) -> str:
    """The remedy, after "select a smaller edit" only when a smaller edit can fit (n > 0)."""
    return f"{SMALLER}, or {remedy}" if n > 0 else remedy[:1].upper() + remedy[1:]


def _allows(n: int) -> str:
    return f"memory admission allows {n:,} weights now" if n > 0 else "memory admission allows no edit now"


def _admission(n: int, d: dict) -> str:
    """Why the supervisor's memory admission stops at n weights, from its refusal one weight past n
    (runtime_policy.admission's forecast_exceeds_budget: measured required/new_bytes/available, limit
    total_cap/preferred) and the bounds the advice reports (with the measured memory of the server's process
    tree, the first thing admission compares with the cap)."""
    event = d.get("event") or {}
    m, lim, b = event.get("measured") or {}, event.get("limit") or {}, d.get("bounds") or {}
    one = "one weight more" if n > 0 else "even one weight"

    def other() -> str:
        message = str(event.get("message") or "no reason given").rstrip(".")
        if n > 0:
            return f"{_allows(n)} ({message}). {SMALLER}. {NOT_SETTING}"
        tips = event.get("recommendations") or []
        return f"{_allows(n)} ({message})." + (f" {tips[0].rstrip('.')}." if tips else "") + f" {NOT_SETTING}"
    try:
        available, new, required = m["available"], m["new_bytes"], m["required"]
        preferred, cap = lim["preferred"], lim["total_cap"]
    except (KeyError, TypeError):
        return other()
    measured = b.get("measured")
    if measured is not None and measured > cap:
        # Admission refuses every operation while the measured memory is over the cap: no edit size helps.
        what, remedy = _cap_source(cap, b, freeing=True)
        return (f"memory admission allows no edit now: the memory measured for the server and its model processes, "
                f"{gib(measured)}, is already over {gib(cap)}, {what}, and admission refuses every operation while "
                f"it is, whatever its size. Unload the model to release the memory it holds (restart the server if "
                f"no model is loaded), or {remedy}. {NOT_SETTING}")
    if available - new < preferred:
        what = "the patch and this edit's working memory" if b.get("loaded") else "loading the model with the patch"
        return (f"{_allows(n)}: {one} would need {gib(new)} of new memory "
                f"({what}, {MEMORY_BYTES} bytes per patch weight), with {gib(available)} available and "
                f"{gib(preferred)} kept free by the {RESERVE}. {_then(n, 'free memory on the machine')}. {NOT_SETTING}")
    if required > cap:
        what, remedy = _cap_source(cap, b)
        return (f"{_allows(n)}: the model's whole memory forecast with the patch "
                f"({MEMORY_BYTES} bytes per patch weight) must stay within {gib(cap)}, {what}, and {one} "
                f"would exceed it. {_then(n, remedy)}. {NOT_SETTING}")
    budget = b.get("budget")
    if budget is not None and not b.get("loaded") and required > budget:
        if b.get("budget_gib") is not None:
            what, remedy = "the Memory budget setting", "raise Memory budget in Settings (freeing memory does not help)"
        else:
            what, remedy = f"the available memory less the {RESERVE}", "free memory on the machine"
        return (f"{_allows(n)}: before the model is loaded, its whole memory forecast "
                f"with the patch must fit the memory budget of {gib(budget)}, {what}, and {one} would "
                f"exceed it. {_then(n, remedy)}. {NOT_SETTING}")
    return other()


def refusal(weights: int, limit: Limit, steer: str | None = None) -> str:
    """The plain refusal: the size, the constraint that binds with its numbers, and what can be done."""
    head = f"This is {weights:,} weights ({gib(weights * PATCH_BYTES)} as FP32 edits), too large for a weight patch: "
    d, n = limit.detail, limit.weights
    event = d.get("event") or {}
    if limit.binding == "setting":
        why = f"the {SETTING} setting allows {n:,} weights ({d['gib']:g} GiB). {SMALLER}, or raise {SETTING} in Settings."
    elif limit.binding == "memory" and event.get("code") == "memory_pressure":
        why = (f"memory pressure on the machine is not normal, and the memory watchdog admits no edit until it is. "
               f"Free memory on the machine and retry. {NOT_SETTING}")
    elif limit.binding == "memory" and d.get("source") == "runtime":
        why = _admission(n, d)
    elif limit.binding == "memory":
        why = (f"memory allows {n:,} weights now: {gib(d['available'])} is available on this machine, "
               f"{gib(d['preferred'])} is kept free by the {RESERVE} and {gib(d['scratch'])} is set aside for reading "
               f"the weights, at an estimated {MEMORY_BYTES} bytes per patch weight while a patch is built and kept "
               f"(the runtime supervisor's admission charge; the edit runs here without the supervisor, so no memory "
               f"admission checks it). {_then(n, 'free memory on the machine')}. {NOT_SETTING}")
    elif limit.binding == "watchdog":
        tips = event.get("recommendations") or []
        why = (f"the memory watchdog admits no edit now: {str(event.get('message') or 'no reason given').rstrip('.')}."
               + (f" {tips[0].rstrip('.')}." if tips else ""))
    elif limit.binding == "disk":
        why = (f"free disk space allows {n:,} weights ({gib(d['free'])} free, {gib(d['reserve'])} kept free). "
               f"{SMALLER}, or free disk space.")
    else:
        checked = ("the memory watchdog still checks memory when the edit runs" if d.get("source") == "runtime"
                   else "nothing else checks memory when the edit runs here, so set it no higher than this machine holds")
        why = (f"memory could not be measured for this edit ({str(d.get('reason') or 'no reason given').rstrip('.')}), "
               f"so the old fixed limit of {n:,} weights applies. {SMALLER}, or set {SETTING} in Settings to allow "
               f"more ({checked}).")
    return head + why + (" " + steer if steer else "")


# ---------------------------------------------------------------------- cost and time

def record_timing(ws, stored: int, seconds: float) -> None:
    """Remember how long building a patch of `stored` weights took in this open workspace (in memory only)."""
    if stored >= TIMED_MIN and seconds > 0:
        ws.edit_timing = dict(stored=int(stored), seconds=float(seconds), time=time.time())


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def worker_hosted(ws) -> bool:
    """The model runs in the supervisor's model worker (a GGUF workspace under a worker-hosting client)."""
    client = getattr(ws, "_runtime_client", None)
    return getattr(ws, "kind", None) == "gguf" and bool(getattr(client, "worker", False))


def cost(ws, weights: int) -> dict:
    """What a selection edit of `weights` costs, and the confirmation the browser shows above CONFIRM_ABOVE."""
    timing = getattr(ws, "edit_timing", None)
    estimate = timing["seconds"] * weights / timing["stored"] if timing else None
    gguf = getattr(ws, "kind", None) == "gguf"
    hosted = worker_hosted(ws)
    if getattr(ws, "_runtime_client", None) is not None:
        memory = (f"Memory admission counts about {gib(weights * MEMORY_BYTES)} for it ({MEMORY_BYTES} bytes per "
                  "weight) while it is built and kept")
    else:
        memory = (f"In memory it is estimated at about {gib(weights * MEMORY_BYTES)} ({MEMORY_BYTES} bytes per weight, "
                  "the runtime supervisor's admission charge) while it is built and kept; the edit runs here without "
                  "the supervisor, so no memory admission checks it")
    lines = [f"Apply an edit of {weights:,} weights?",
             f"Patch: about {gib(weights * PATCH_BYTES)} on disk (FP32 edits). {memory}; "
             "the real use on this model has not been measured."]
    if hosted:
        lines.append(f"The model runs in the model worker, so the patch is held twice, by the server and by the "
                     f"worker: at least {gib(weights * PATCH_BYTES * 2)} while it is enabled.")
    if estimate is not None:
        lines.append(f"Time to build it: about {_duration(estimate)}. This is an estimate from the last large edit "
                     f"since this workspace was opened ({timing['stored']:,} weights in {timing['seconds']:.1f} s); "
                     "tests and drift checks run after it.")
    else:
        lines.append("Time: no estimate; no edit of a million weights or more has been timed since this workspace "
                     "was opened (timings are kept only while it stays open).")
    lines.append("Progress and the time left are shown while it runs. A running edit cannot be stopped; "
                 "nothing is changed until it is committed.")
    if gguf:
        lines.append("Generation is slower while this patch is enabled: on every forward pass the runner adds the "
                     "patch's contribution to the output of each matrix it changes (the quantized weights are never "
                     "changed). How much slower has not been measured.")
    return dict(weights=int(weights), disk_bytes=int(weights) * PATCH_BYTES, memory_bytes=int(weights) * MEMORY_BYTES,
                confirm_above=CONFIRM_ABOVE, confirm=weights > CONFIRM_ABOVE, gguf=gguf, worker_hosted=hosted,
                cancellable=False, estimate_seconds=estimate, basis=dict(timing) if timing else None,
                message="\n\n".join(lines))
