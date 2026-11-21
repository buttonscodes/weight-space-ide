"""The replica of Heretic's randomized rank-r approximation (B-S4) on the device Heretic's run was made on: model-free,
scripts/formats_shared.py only. When its bound is asserted (same torch AND same device), and that the device
argument reaches Heretic's lines without changing the CPU arithmetic."""
import importlib.util
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / "scripts" / "formats_shared.py"
_SPEC = importlib.util.spec_from_file_location("formats_shared_replica_under_test", _PATH)
FS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FS)

RUN = {"torch": "2.14.1"}
LINE = re.compile(r"^(PASS |FAIL |     )(\S+) (.*)$")
MPS = torch.backends.mps.is_available()
needs_mps = pytest.mark.skipif(not MPS, reason="torch.backends.mps.is_available() is False")


# ------------------------------------------------------------------ the rule
def test_same_torch_and_device_is_asserted():
    applies, why = FS.replica_applies("2.14.1", RUN, "mps", "mps:0")
    assert applies is True and "ran on mps" in why
    assert FS.replica_applies("2.14.1+cpu", RUN, "cpu", "cpu")[0] is True
    assert FS.replica_applies("2.14.1", RUN, "cpu", None)[0] is True          # no recorded device is the CPU


def test_other_device_is_reported_with_the_reason():
    applies, why = FS.replica_applies("2.14.1", RUN, "cpu", "mps:0")
    assert applies is False
    assert "ran on cpu" in why and "run on mps" in why
    assert "torch" not in why                                                 # the torch matches: only the device
    applies, why = FS.replica_applies("2.14.1", RUN, "mps", None)
    assert applies is False and "run on cpu" in why


def test_other_torch_is_reported_with_the_reason():
    applies, why = FS.replica_applies("2.15.0", RUN, "mps", "mps:0")
    assert applies is False
    assert "2.15.0" in why and "2.14.1" in why
    assert "ran on" not in why                                                # the device matches: only the torch


def test_both_differ_names_both():
    applies, why = FS.replica_applies("2.15.0", RUN, "cpu", "mps:0")
    assert applies is False and "2.15.0" in why and "run on mps" in why
    assert FS.replica_applies("2.14.1", {}, "cpu", "cpu")[0] is False         # no recorded torch


@pytest.mark.parametrize("recorded,available,device", [
    ("mps:0", True, "mps"), ("mps", True, "mps"), ("mps:0", False, "cpu"), ("cpu", True, "cpu"),
    (None, True, "cpu"), ("cuda:0", True, "cpu")])
def test_replica_device(recorded, available, device):
    assert FS.replica_device(recorded, available) == device


def test_replica_device_asks_torch(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert FS.replica_device("mps:0") == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert FS.replica_device("mps:0") == "mps"


def _rows(worst):
    good = {"exact_truncation_vs_ide": 5e-7, "replica_vs_heretic": worst, "replica_vs_ide": 4e-4,
            "ide_vs_heretic": 4e-4, "tail": 0.9, "sigma_r_over_sigma_1": 0.02,
            "replica_of_heretic_formula_vs_heretic": 2e-6, "replica_bit_equal_heretic": False,
            "module": "m", "tensor": "t"}
    return [dict(good), dict(good, replica_vs_heretic=worst / 2)]


BOUND = {"replica_vs_heretic": 1e-5}


def _check(capsys, worst, this_torch, device, run_device):
    rep = FS.Report()
    numbers = FS.replica_check(rep, "run", _rows(worst), BOUND, this_torch, RUN, device, run_device)
    lines = [l for l in capsys.readouterr().out.splitlines() if l]
    assert all(LINE.match(l) for l in lines), lines
    return rep, numbers, lines


@pytest.mark.parametrize("worst,passes", [(9e-6, True), (1.2e-5, False)])
def test_check_asserts_the_bound_when_both_match(capsys, worst, passes):
    rep, numbers, lines = _check(capsys, worst, "2.14.1", "mps", "mps:0")
    entry = rep.labels["run"][FS.REPLICA_STAGE]
    assert entry["pass"] is passes and rep.ok() is passes
    assert numbers["device"] == "mps" and numbers["run_device"] == "mps"
    stage_lines = [l for l in lines if FS.REPLICA_STAGE in l]
    assert stage_lines and stage_lines[0].startswith("PASS " if passes else "FAIL ")
    assert '"device": "mps"' in stage_lines[0] and '"run_device": "mps"' in stage_lines[0]
    assert rep.labels["run"][FS.REPLICA_INPUT]["pass"] is True                # the premise, always asserted


@pytest.mark.parametrize("this_torch,device,run_device,named", [
    ("2.14.1", "cpu", "mps:0", "ran on cpu"), ("2.15.0", "mps", "mps:0", "2.15.0")])
def test_check_reports_when_either_differs(capsys, this_torch, device, run_device, named):
    rep, numbers, lines = _check(capsys, 1.2e-5, this_torch, device, run_device)
    entry = rep.labels["run"][FS.REPLICA_STAGE]
    assert "pass" not in entry and named in entry["note"]
    assert rep.ok() is True                                                   # reported: no failed assertion
    assert numbers["device"] == FS.device_kind(device) and numbers["run_device"] == "mps"
    words = [l for l in lines if l.startswith("     ") and "reported only" in l and named in l]
    assert words and f"replica on {FS.device_kind(device)}, run on mps" in words[0]
    assert "pass" in rep.labels["run"][FS.REPLICA_INPUT]                      # the premise is still asserted


def test_check_defaults_are_a_cpu_run(capsys):
    rep = FS.Report()
    numbers = FS.replica_check(rep, "run", _rows(1.2e-5), BOUND, "2.14.1", RUN)
    capsys.readouterr()
    assert rep.labels["run"][FS.REPLICA_STAGE]["pass"] is False               # asserted on the CPU for a CPU run
    assert numbers["device"] == "cpu" and numbers["run_device"] == "cpu"


# ------------------------------------------------------------------ the arithmetic
def _reference_lowrank(D, r, seed, device="cpu"):
    """Heretic's lines (abliteration.py, the "full" branch), literally, on ``device``; factors back on the CPU."""
    W = D.to(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)
    U = U[:, :r]
    S = S[:r]
    Vh = Vh[:, :r].T
    sqrt_S = torch.sqrt(S)
    lora_B = U @ torch.diag(sqrt_S)
    lora_A = torch.diag(sqrt_S) @ Vh
    return lora_B.cpu(), lora_A.cpu()


def _reference_full_change(W, v, weight, device="cpu"):
    v = v.to(device)
    W = W.to(device).to(torch.float32)
    W_org = W
    W_row_norms = torch.linalg.vector_norm(W, dim=1, keepdim=True)
    W = F.normalize(W, p=2, dim=1)
    lora_A = (v @ W).view(1, -1)
    lora_B = (-weight * v).view(-1, 1)
    W = W + lora_B @ lora_A
    W = F.normalize(W, p=2, dim=1)
    W = W * W_row_norms
    return (W - W_org).cpu()


def _inputs(d_out=48, d_in=40, seed=7):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d_out, d_in, generator=g, dtype=torch.float32)
    v = torch.randn(d_out, generator=g, dtype=torch.float32)
    return W, v / v.norm()


def test_lowrank_on_the_cpu_is_heretic_lines_bit_for_bit():
    W, v = _inputs()
    D = FS.full_change(W, v, 0.8).to(torch.float32)
    want_b, want_a = _reference_lowrank(D, 3, 12345)
    for args in ((), ("cpu",), (torch.device("cpu"),)):
        b, a = FS.heretic_lowrank(D, 3, 12345, *args)
        assert b.device.type == "cpu" and a.device.type == "cpu"
        assert torch.equal(b, want_b) and torch.equal(a, want_a)


def test_full_change_on_the_cpu_is_heretic_lines_bit_for_bit():
    W, v = _inputs()
    want = _reference_full_change(W.to(torch.bfloat16), v, 0.8)
    for args in ((), ("cpu",)):
        assert torch.equal(FS.heretic_full_change(W.to(torch.bfloat16), v, 0.8, *args), want)


def test_replica_row_with_cpu_is_unchanged():
    W, v = _inputs()
    r, seed = 3, 12345
    ide, _ = FS.exact_lowrank(FS.full_change(W, v, 0.8), r)
    ide = ide.to(torch.float32)
    hb, ha = _reference_lowrank(FS.heretic_full_change(W, v, 0.8), r, seed)
    default = FS.replica_row(W, v, 0.8, r, seed, hb @ ha, ide)
    assert default == FS.replica_row(W, v, 0.8, r, seed, hb @ ha, ide, "cpu")
    assert default["replica_of_heretic_formula_vs_heretic"] == 0.0            # the reference is Heretic's lines


def test_replica_rows_passes_the_device(monkeypatch):
    from wsp import directions as DR
    W, v = _inputs()
    monkeypatch.setattr(DR, "table", lambda good, bad, orth: None)
    monkeypatch.setattr(DR, "layer_weight", lambda dist, layer: (0.8, None))
    monkeypatch.setattr(DR, "direction_at", lambda tab, index, layer: v)
    seen = []
    monkeypatch.setattr(FS, "replica_row", lambda *a: seen.append(a[-1]) or {"replica": True})
    name = "model.layers.0.self_attn.o_proj"
    base = types.SimpleNamespace(keys=lambda: [f"{name}.weight"], get_tensor=lambda t: W)
    adapter = types.SimpleNamespace(config={"r": 3}, modules=[name], delta=lambda m: torch.zeros_like(W))
    dump = {"modifier_settings": {"Abliteration": {}}, "exported_trial_index": 0, "seed": 1, "layers": 1,
            "trials": [{"index": 0, "parameters": {"weight_distributions": {"attn.o_proj": {}}}}]}
    for device in ("mps", "cpu"):
        rows = FS.replica_rows(base, {}, dump, None, None, adapter, lambda *a: torch.zeros_like(W), device=device)
        assert rows[0]["replica"] is True and seen[-1] == device
    FS.replica_rows(base, {}, dump, None, None, adapter, lambda *a: torch.zeros_like(W))
    assert seen[-1] == "cpu"


def test_script_refuses_another_device(tmp_path):
    out = subprocess.run([sys.executable, str(_ROOT / "scripts" / "heretic_replica_check.py"), str(tmp_path),
                          str(tmp_path / "o.json"), f"full={tmp_path}", "device=cuda"],
                         cwd=_ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode != 0 and "device=cuda: the device is cpu or mps" in out.stderr


@needs_mps
def test_lowrank_on_mps_runs_heretic_lines_there():
    W, v = _inputs(96, 80)
    D = FS.full_change(W, v, 0.8).to(torch.float32)
    want_b, want_a = _reference_lowrank(D, 3, 12345, "mps")
    b, a = FS.heretic_lowrank(D, 3, 12345, "mps")
    assert b.device.type == "cpu" and a.device.type == "cpu"                  # compared on the CPU
    assert torch.equal(b, want_b) and torch.equal(a, want_a)
    cpu_b, cpu_a = FS.heretic_lowrank(D, 3, 12345)
    assert not torch.equal(b @ a, cpu_b @ cpu_a)                              # the device's own stream and rounding


@needs_mps
def test_full_change_on_mps_runs_heretic_lines_there():
    W, v = _inputs(96, 80)
    got = FS.heretic_full_change(W, v, 0.8, "mps")
    assert got.device.type == "cpu"
    assert torch.equal(got, _reference_full_change(W, v, 0.8, "mps"))


@needs_mps
def test_replica_row_on_mps_reproduces_heretic_lines_there():
    W, v = _inputs(96, 80)
    r, seed = 3, 12345
    ide, _ = FS.exact_lowrank(FS.full_change(W, v, 0.8), r)
    hb, ha = _reference_lowrank(_reference_full_change(W, v, 0.8, "mps"), r, seed, "mps")
    row = FS.replica_row(W, v, 0.8, r, seed, hb @ ha, ide.to(torch.float32), "mps")
    assert row["replica_of_heretic_formula_vs_heretic"] == 0.0                # Heretic's lines on mps, reproduced
    # R: the IDE's D (float64 on the CPU) in float32, Heretic's low-rank lines on mps
    rb, ra = _reference_lowrank(FS.full_change(W, v, 0.8).to(torch.float32), r, seed, "mps")
    row = FS.replica_row(W, v, 0.8, r, seed, rb @ ra, ide.to(torch.float32), "mps")
    assert row["replica_vs_heretic"] == 0.0 and row["replica_bit_equal_heretic"] is True
