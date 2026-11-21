"""Test harness only: pace the arrival of tensors on the Apple GPU, so that a guarded process grows slowly.

Use: put this directory on PYTHONPATH and set LOAD_GIB_PER_S (1.5 was used) for a model process started under
``python -m wsp.memory_guard`` with the model on ``mps``; scripts/load_pacing_check.py checks it without a model.

The memory guard ends a run whose process tree grows by more than 4 GiB between two samples (one second apart). A
4B model in bfloat16 put on `mps` at full speed grows by about 5 GiB a second, so the guard ends it during the
load. Nothing of the guard is changed: with LOAD_GIB_PER_S set, each tensor that arrives on `mps` by one of the two
paths below is followed by a pause that keeps the average rate of arrival at or below that many GiB per second.
The calls themselves are the libraries' own, with the same arguments; no value, dtype or order of operations
changes.

  1. `Tensor.to` from another device onto `mps` (a model loaded on the CPU and then moved: the IDE's runner);
  2. `transformers.core_model_loading._materialize_copy` returning a tensor on `mps` (a model loaded with
     `device_map = "mps"`: `transformers` opens the weight files with safetensors' `pread` backend on `mps`, which
     reads each tensor straight into GPU memory without a `Tensor.to`: Heretic's loader).

Installed when those modules are first imported by a process that has this directory on PYTHONPATH; a process that
never imports torch (the guard itself, a shell) is not affected. Without LOAD_GIB_PER_S it does nothing.
"""
import importlib.abc
import importlib.util
import os
import sys
import threading
import time

_RATE = float(os.environ.get("LOAD_GIB_PER_S", "0") or 0) * 2 ** 30
_LOCK = threading.Lock()
STATE = {"free_at": 0.0, "bytes": 0, "paused": 0.0, "by_to": 0, "by_materialize": 0}


def _pace(n, key):
    with _LOCK:
        now = time.monotonic()
        wait = max(0.0, STATE["free_at"] - now)
        STATE["free_at"] = max(now, STATE["free_at"]) + n / _RATE
        STATE["bytes"] += n
        STATE[key] += n
        STATE["paused"] += wait
    if wait > 0:
        time.sleep(wait)


def _install_torch(torch):
    original = torch.Tensor.to

    def to(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        if out.device.type == "mps" and self.device.type != "mps":
            _pace(out.numel() * out.element_size(), "by_to")
        return out

    torch.Tensor.to = to
    torch._load_pacing = STATE                      # read by a check script; not used by either tool


def _install_loading(module):
    original = module._materialize_copy

    def _materialize_copy(tensor, device=None, dtype=None):
        before = STATE["bytes"]
        out = original(tensor, device, dtype)
        on = getattr(out, "device", None)
        if on is not None and on.type == "mps" and STATE["bytes"] == before:      # not already paced by path 1
            _pace(out.numel() * out.element_size(), "by_materialize")
        return out

    module._materialize_copy = _materialize_copy


_TARGETS = {"torch": _install_torch, "transformers.core_model_loading": _install_loading}


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        install = _TARGETS.pop(name, None)           # from here on this hook does not answer for `name`
        if install is None:
            return None
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            install(module)

        spec.loader.exec_module = exec_module
        return spec


if _RATE > 0:
    sys.meta_path.insert(0, _Hook())
