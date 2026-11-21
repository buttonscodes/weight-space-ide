"""scripts/load_pacing/sitecustomize.py, model-free: what is paced, by how much, and that it is off unless asked for."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "load_pacing" / "sitecustomize.py"


def load(monkeypatch, rate):
    if rate is None:
        monkeypatch.delenv("LOAD_GIB_PER_S", raising=False)
    else:
        monkeypatch.setenv("LOAD_GIB_PER_S", rate)
    before = list(sys.meta_path)
    spec = importlib.util.spec_from_file_location("load_pacing_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    added = [f for f in sys.meta_path if f not in before]
    for f in added:
        sys.meta_path.remove(f)                     # the test process keeps its own import machinery
    return module, added


class FakeTensor:
    def __init__(self, device, n=0):
        self.device, self.n = types.SimpleNamespace(type=device), n

    def numel(self):
        return self.n

    def element_size(self):
        return 2

    def to(self, device=None, **kw):
        return FakeTensor(device or self.device.type, self.n)


@pytest.mark.parametrize("rate", [None, "", "0"])
def test_without_a_rate_nothing_is_installed(monkeypatch, rate):
    module, added = load(monkeypatch, rate)
    assert module._RATE == 0 and added == []


def test_with_a_rate_one_import_hook_waits_for_torch_and_the_loader(monkeypatch):
    module, added = load(monkeypatch, "1.5")
    assert module._RATE == 1.5 * 2 ** 30 and len(added) == 1
    assert set(module._TARGETS) == {"torch", "transformers.core_model_loading"}
    assert added[0].find_spec("json", None) is None and set(module._TARGETS) == {"torch", "transformers.core_model_loading"}


def test_only_arrivals_on_mps_are_paced_and_the_pause_keeps_the_rate(monkeypatch):
    module, _ = load(monkeypatch, "1")              # 1 GiB a second
    now, slept = [100.0], []
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    fake = types.SimpleNamespace(Tensor=type("Tensor", (FakeTensor,), {}))
    module._install_torch(fake)
    half = 2 ** 28                                  # elements of 2 bytes: half a GiB
    cpu = fake.Tensor("cpu", half)
    assert cpu.to("mps").device.type == "mps" and slept == []          # the first arrival is not held back
    cpu.to("mps")
    assert slept == [0.5]                           # the second waits for the first's half second
    cpu.to("cpu"), fake.Tensor("mps", half).to("mps"), fake.Tensor("mps", half).to("cpu")
    assert slept == [0.5] and module.STATE["bytes"] == 2 ** 30 and module.STATE["by_to"] == 2 ** 30
    now[0] += 10.0                                  # idle: no debt is carried, and no credit either
    cpu.to("mps"), cpu.to("mps")
    assert slept == [0.5, 0.5]


def test_a_tensor_read_straight_onto_mps_is_paced_once(monkeypatch):
    module, _ = load(monkeypatch, "1")
    now, slept = [0.0], []
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    fake = types.SimpleNamespace(Tensor=type("Tensor", (FakeTensor,), {}))
    module._install_torch(fake)
    half = 2 ** 28

    def materialize(tensor, device=None, dtype=None):                  # safetensors put it on mps; .to is mps to mps
        return tensor.to(device)

    loading = types.SimpleNamespace(_materialize_copy=materialize)
    module._install_loading(loading)
    loading._materialize_copy(fake.Tensor("mps", half), "mps")         # read straight onto mps: counted here
    loading._materialize_copy(fake.Tensor("cpu", half), "mps")         # moved by Tensor.to: counted there, not twice
    loading._materialize_copy(fake.Tensor("cpu", half), "cpu")         # not on mps: not counted
    assert module.STATE["by_materialize"] == 2 ** 29 and module.STATE["by_to"] == 2 ** 29
    assert module.STATE["bytes"] == 2 ** 30 and slept == [0.5]
