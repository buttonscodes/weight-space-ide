"""The size limit of one weight patch (wsp.edit_limit): the Largest weight patch setting, memory and disk.

Model-free. Quantized-looking workspaces are tests/selection_fixtures.QuantStore (in memory, fixed values); the
runtime supervisor is the inert one of tests/test_runtime_supervisor.py (synthetic memory, a sleeping child, no
model worker). Thresholds are lowered here instead of allocating: nothing of 60,000,000 weights is built.

No outcome depends on this host: free disk and the machine memory reading are synthetic in every test (the
autouse fixture gives plenty of both; a test that needs less sets its own)."""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from fastapi.testclient import TestClient  # noqa: E402

from selection_fixtures import quant_workspace  # noqa: E402
from wsp import delta as D  # noqa: E402
from wsp import edit_limit as EL  # noqa: E402
from wsp import runtime_config  # noqa: E402
from wsp import selections as S  # noqa: E402
from wsp import wscript as W  # noqa: E402
import wsp.runtime_supervisor as R  # noqa: E402
from wsp.runtime_config import DEFAULTS, _validate, resolve  # noqa: E402
from wsp.runtime_events import RuntimeRefusal  # noqa: E402
from wsp.runtime_policy import admission, largest_patch, measured_total  # noqa: E402
from wsp.runtime_supervisor import (PATCH_ADMISSION_BYTES, SCRATCH_ADMISSION_QUANTUM, RuntimeClient,  # noqa: E402
                                    descriptor, policy_for)
from wsp.server import create_app  # noqa: E402
from wsp.session import Session  # noqa: E402
from wsp.settings import SettingsStore, validate_interactive  # noqa: E402

GiB = 1024 ** 3
MiB = 1024 ** 2
DOWN = "model.layers.0.mlp.down_proj.weight"       # 16 x 24 = 384 weights in the fixture
URL = "http://127.0.0.1:41881"


def config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"version": 1, "defaults": {"backend": "cpu"}, "profiles": {}}))
    return str(path)


def gib_for(weights):
    return weights * EL.PATCH_BYTES / GiB


def machine(available, total=64 * GiB, pressure=1, calls=None):
    """Synthetic memory_guard.snapshot(); `calls` (a list) counts the readings."""
    def snap():
        if calls is not None:
            calls.append(1)
        return dict(available=int(available), total=int(total), pressure=pressure)
    return snap


def disk(free, calls=None):
    """Synthetic free disk space for edit_limit (shutil.disk_usage)."""
    def usage(path):
        if calls is not None:
            calls.append(path)
        return SimpleNamespace(total=1 << 50, used=0, free=int(free))
    return SimpleNamespace(disk_usage=usage)


@pytest.fixture(autouse=True)
def pinned_machine(monkeypatch):
    EL.forget_machine()
    monkeypatch.setattr(EL, "shutil", disk(1 << 50))
    EL.telemetry = machine(512 * GiB, total=1024 * GiB)
    yield
    EL.forget_machine()
    EL.telemetry = None


def room_for(weights, total=64 * GiB, scratch=0):
    """Available memory that leaves exactly `weights` patch weights after the preferred reserve and scratch."""
    preferred = policy_for(DEFAULTS, total, total).preferred
    return preferred + max(scratch, SCRATCH_ADMISSION_QUANTUM) + weights * EL.MEMORY_BYTES


@pytest.fixture()
def quant(tmp_path):
    d = tmp_path / "quant"
    d.mkdir()
    s = Session(str(tmp_path))
    s.ws = quant_workspace(str(d))
    return s


class Advisor:
    """A runtime client that only advises: counts the calls and returns `advice`."""
    def __init__(self, advice):
        self.advice, self.calls, self.worker = advice, 0, False

    def advise_edit(self, ws, scratch=0):
        self.calls += 1
        return self.advice


# ------------------------------------------------------------------ the setting

def test_the_setting_defaults_to_auto_and_is_validated_like_the_memory_settings(tmp_path):
    assert DEFAULTS["patch_limit_gib"] is None
    settings = resolve(config(tmp_path), env={}, system="Linux")
    assert settings["values"]["patch_limit_gib"] is None and settings["sources"]["patch_limit_gib"] == "default"
    _validate({"patch_limit_gib": 2.5}, "settings")
    _validate({"patch_limit_gib": 1 / 1024}, "settings")                 # the smallest accepted: 1 MiB
    for bad in (0, -1, "2", 2_000_000, 1e-300, 1e-4, 1 / 1024 - 1e-12):
        with pytest.raises(ValueError, match="patch_limit_gib"):
            _validate({"patch_limit_gib": bad}, "settings")
    with pytest.raises(ValueError, match=r"at least 1 MiB \(0.000976562 GiB, 262,144 weights\) or null \(Auto\)"):
        _validate({"patch_limit_gib": 1e-300}, "settings")
    explicit = resolve(config(tmp_path), overrides={"patch_limit_gib": 3}, env={}, system="Linux")
    assert explicit["values"]["patch_limit_gib"] == 3 and explicit["sources"]["patch_limit_gib"] == "command line"
    values = dict(settings, values=dict(settings["values"], patch_limit_gib=0))
    with pytest.raises(ValueError):
        validate_interactive(values)


@pytest.mark.parametrize("section", ["defaults", "profile"])
def test_a_value_saved_below_the_minimum_by_an_earlier_build_is_refused_at_start_with_the_fix(tmp_path, section):
    """An earlier build saved values below 1 MiB. At start such a file is refused, naming the file, the key, the
    value, the minimum and the fix; it is not raised or cleared silently. Save and flags still refuse."""
    path = tmp_path / "config.json"
    values = {"backend": "cpu", "patch_limit_gib": 0.0001}
    document = ({"version": 1, "defaults": values, "profiles": {}} if section == "defaults"
                else {"version": 1, "defaults": {"backend": "cpu"}, "profiles": {"small": values}})
    path.write_text(json.dumps(document))
    label = "defaults" if section == "defaults" else "profile small"
    with pytest.raises(ValueError) as refused:
        resolve(str(path), "small" if section == "profile" else None, env={}, system="Linux")
    message = str(refused.value)
    assert message.startswith(f"{label}.patch_limit_gib is 0.0001; it must be at least 1 MiB (0.000976562 GiB, "
                              "262,144 weights) or null (Auto)")
    assert message.endswith(f"Edit {path}: set {label}.patch_limit_gib to null (Auto) or to at least 0.000976562, "
                            "then start again")
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError) as flag:                              # a flag names no file
        resolve(config(other), overrides={"patch_limit_gib": 0.0001}, env={}, system="Linux")
    assert "patch_limit_gib is 0.0001" in str(flag.value) and "Edit " not in str(flag.value)


def test_a_saved_setting_persists_and_applies_without_a_restart(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"version": 1, "defaults": {"backend": "cpu"}, "profiles": {}}))
    store = SettingsStore(resolve(str(path), env={}, system="Linux"))
    with pytest.raises(ValueError):
        store.save(store.snapshot()["revision"], {"patch_limit_gib": -1})
    after = store.save(store.snapshot()["revision"], {"patch_limit_gib": 2})
    assert json.loads(path.read_text())["defaults"]["patch_limit_gib"] == 2
    assert after["active"]["values"]["patch_limit_gib"] == 2 and not after["restart_required"]
    assert "patch_limit_gib" in after["live"] and "patch_limit_gib" not in after["changed"]
    assert resolve(str(path), env={}, system="Linux")["values"]["patch_limit_gib"] == 2       # read again at start
    # Without a supervisor, a setting that waits for a restart still does
    assert store.save(after["revision"], {"context": 99})["restart_required"]
    back = store.save(store.snapshot()["revision"], {"patch_limit_gib": None})
    assert back["active"]["values"]["patch_limit_gib"] is None


def test_a_command_line_setting_still_wins_after_a_save(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"version": 1, "defaults": {"backend": "cpu"}, "profiles": {}}))
    store = SettingsStore(resolve(str(path), overrides={"patch_limit_gib": 3}, env={}, system="Linux"))
    after = store.save(store.snapshot()["revision"], {"patch_limit_gib": 2})
    assert json.loads(path.read_text())["defaults"]["patch_limit_gib"] == 2               # saved for the next start
    assert after["active"]["values"]["patch_limit_gib"] == 3                            # the launch override wins now
    assert after["active"]["sources"]["patch_limit_gib"] == "command line"
    assert store.active["values"]["patch_limit_gib"] == 3


def test_settings_shows_the_setting_and_a_save_reaches_the_edits_at_once(tmp_path, quant, monkeypatch):
    monkeypatch.setattr(runtime_config, "PATCH_LIMIT_MIN_GIB", gib_for(1))     # a 100-weight setting for the fixture
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"version": 1, "defaults": {"backend": "cpu"}, "profiles": {}}))
    quant.settings = resolve(str(path), env={}, system="Linux")
    app = create_app(quant, URL)
    app.state.settings_telemetry = machine(40 * GiB)
    c = TestClient(app, base_url=URL, raise_server_exceptions=False)
    view = c.get("/api/settings").json()
    assert "patch_limit_gib" in view["editable"] and view["active"]["values"]["patch_limit_gib"] is None
    assert quant.ws.edit_settings["patch_limit_gib"] is None
    headers = {"X-WSP-Settings": view["csrf"], "Origin": URL}
    r = c.post("/api/settings", json={"revision": view["revision"], "changes": {"patch_limit_gib": gib_for(100)}}, headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["active"]["values"]["patch_limit_gib"] == gib_for(100) and not r.json()["restart_required"]
    assert EL.setting(quant.ws) == gib_for(100)
    refused = c.post("/api/selection/script", json={"addr": DOWN, "action": "scale", "factor": 2})
    assert refused.status_code == 400 and "Largest weight patch setting allows 100 weights" in refused.json()["error"]
    bad = c.post("/api/settings", json={"revision": r.json()["revision"], "changes": {"patch_limit_gib": 0}}, headers=headers)
    assert bad.status_code == 400 and EL.setting(quant.ws) == gib_for(100)


def test_a_workspace_opened_after_the_setting_was_set_reads_it(tmp_path, monkeypatch):
    """Switching workspace keeps the setting (Session.open): the new workspace is limited at once."""
    import wsp.session as SS
    s = Session(str(tmp_path))
    s.set_edit_settings({**DEFAULTS, "patch_limit_gib": gib_for(100)})
    target = quant_workspace(str(tmp_path / "next"))
    monkeypatch.setattr(SS, "is_workspace", lambda path: True)
    monkeypatch.setattr(SS, "Workspace", SimpleNamespace(load=lambda *a, **k: target))
    s.open(str(tmp_path / "next"))
    assert s.ws is target and EL.setting(target) == gib_for(100)
    assert "Largest weight patch setting allows 100 weights" in S.actions(target, DOWN)["scale"]["reason"]


def test_wsp_apply_reads_the_setting_from_the_configuration_and_the_command_line(tmp_path):
    import argparse
    from wsp.__main__ import _apply_settings
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"version": 1, "defaults": {"backend": "cpu", "patch_limit_gib": 2}, "profiles": {}}))
    ap = argparse.ArgumentParser()
    args = lambda **kw: SimpleNamespace(**{"config": str(path), "profile": None, "patch_limit_gib": None, **kw})  # noqa: E731
    assert _apply_settings(args(), ap)["patch_limit_gib"] == 2
    assert _apply_settings(args(patch_limit_gib=0.5), ap)["patch_limit_gib"] == 0.5
    with pytest.raises(SystemExit):                                        # an explicit bad value is an error
        _apply_settings(args(patch_limit_gib=1e-300), ap)


def test_wsp_apply_refuses_past_the_setting_it_was_given(tmp_path, monkeypatch, capsys):
    import wsp.session as SS
    from wsp.__main__ import main
    monkeypatch.setattr(runtime_config, "PATCH_LIMIT_MIN_GIB", gib_for(1))
    target = quant_workspace(str(tmp_path / "q"))
    def fake_open(self, name, log=print, pull=False):          # no model is built or loaded
        self.ws = target
        target.edit_settings = self.edit_settings
        return target
    monkeypatch.setattr(SS.Session, "open", fake_open)
    script = tmp_path / "edit.wsp"
    script.write_text(f'P("{DOWN}")[:, :] *= 2')
    argv = ["apply", str(script), "--root", str(tmp_path / "root"), "--config", config(tmp_path)]
    main(argv + ["--patch-limit-gib", str(gib_for(100))])
    out = capsys.readouterr().out
    assert out.startswith("ERROR: ") and "Largest weight patch setting allows 100 weights" in out and not target.patches
    main(argv)                                                   # the config's Auto
    assert "COMMITTED" in capsys.readouterr().out and len(target.patches) == 1


# wsp apply without flags (adopted from the round-2 review's test_review_apply_cli.py; HOME and XDG_CONFIG_HOME
# are temporary, so the default configuration read is the test's own)

def apply_without_flags(tmp_path, monkeypatch, content=None):
    import wsp.session as SS
    monkeypatch.setattr(runtime_config, "PATCH_LIMIT_MIN_GIB", gib_for(1))
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in [k for k in os.environ if k.startswith("WSP_")]:
        monkeypatch.delenv(name)
    folder = tmp_path / "xdg" / "weight-space-ide"
    folder.mkdir(parents=True)
    if content is not None:
        (folder / "config.json").write_text(content)
    target = quant_workspace(str(tmp_path / "q"))
    def fake_open(self, name, log=print, pull=False):          # no model is built or loaded
        self.ws = target
        target.edit_settings = self.edit_settings
        return target
    monkeypatch.setattr(SS.Session, "open", fake_open)
    script = tmp_path / "edit.wsp"
    script.write_text(f'P("{DOWN}")[:, :] *= 2')
    return target, ["apply", str(script), "--root", str(tmp_path / "root")], folder / "config.json"


@pytest.mark.parametrize("content", ["{not json", "", json.dumps([1]), json.dumps({"version": 2}),
                                     json.dumps({"version": 1, "defaults": {"patch_limit_gib": None}})])
def test_wsp_apply_without_flags_or_a_readable_limit_commits_as_before(tmp_path, monkeypatch, capsys, content):
    from wsp.__main__ import main
    target, argv, _ = apply_without_flags(tmp_path, monkeypatch, content)
    main(argv)
    out = capsys.readouterr()
    if content != json.dumps({"version": 1, "defaults": {"patch_limit_gib": None}}):
        assert "settings not read" in out.err and "Largest weight patch is Auto" in out.err
    assert "COMMITTED" in out.out and len(target.patches) == 1


def test_wsp_apply_with_a_default_configuration_that_is_a_directory_commits(tmp_path, monkeypatch, capsys):
    from wsp.__main__ import main
    target, argv, path = apply_without_flags(tmp_path, monkeypatch)
    path.mkdir()
    main(argv)
    out = capsys.readouterr()
    assert "COMMITTED" in out.out and len(target.patches) == 1 and "Largest weight patch is Auto" in out.err


def test_wsp_apply_honours_a_saved_setting_without_flags(tmp_path, monkeypatch, capsys):
    from wsp.__main__ import main
    target, argv, _ = apply_without_flags(tmp_path, monkeypatch,
                                          json.dumps({"version": 1, "defaults": {"patch_limit_gib": gib_for(100)}}))
    main(argv)
    out = capsys.readouterr().out
    assert out.startswith("ERROR") and "Largest weight patch setting allows 100 weights" in out and not target.patches


def test_wsp_apply_keeps_a_saved_setting_when_an_unrelated_variable_is_bad(tmp_path, monkeypatch, capsys):
    from wsp.__main__ import main
    target, argv, path = apply_without_flags(tmp_path, monkeypatch,
                                             json.dumps({"version": 1, "defaults": {"patch_limit_gib": gib_for(100)}}))
    monkeypatch.setenv("WSP_GGML_THREADS", "abc")
    main(argv)
    out = capsys.readouterr()
    assert "WSP_GGML_THREADS requires an integer" in out.err and f"Largest weight patch of {gib_for(100):g} GiB" in out.err
    assert out.out.startswith("ERROR") and "Largest weight patch setting allows 100 weights" in out.out
    assert not target.patches                                         # the saved limit still refuses the edit


@pytest.mark.parametrize("saved", [-1, "abc", 0])
def test_wsp_apply_refuses_when_a_saved_setting_cannot_be_read(tmp_path, monkeypatch, capsys, saved):
    from wsp.__main__ import main
    target, argv, path = apply_without_flags(tmp_path, monkeypatch,
                                             json.dumps({"version": 1, "defaults": {"patch_limit_gib": saved}}))
    with pytest.raises(SystemExit):
        main(argv)
    err = capsys.readouterr().err
    assert "patch_limit_gib" in err and "a saved setting is not dropped" in err and str(path) in err
    assert not target.patches


def test_wsp_apply_refuses_a_saved_setting_in_a_configuration_of_another_version(tmp_path, monkeypatch, capsys):
    from wsp.__main__ import main
    target, argv, _ = apply_without_flags(tmp_path, monkeypatch,
                                          json.dumps({"version": 2, "defaults": {"patch_limit_gib": 5}}))
    with pytest.raises(SystemExit):
        main(argv)
    assert "Configuration requires integer version: 1" in capsys.readouterr().err and not target.patches


@pytest.mark.parametrize("variable", ["WSP_CONFIG", "WSP_PROFILE"])
def test_wsp_apply_treats_the_configuration_variables_as_explicit(tmp_path, monkeypatch, capsys, variable):
    from wsp.__main__ import main
    target, argv, _ = apply_without_flags(tmp_path, monkeypatch)
    monkeypatch.setenv(variable, str(tmp_path / "nope.json") if variable == "WSP_CONFIG" else "nope")
    with pytest.raises(SystemExit):
        main(argv)
    err = capsys.readouterr().err
    assert ("Configuration file does not exist" if variable == "WSP_CONFIG" else "Unknown profile: nope") in err
    assert f"given by {variable}" in err and not target.patches


@pytest.mark.parametrize("flag", [["--patch-limit-gib", "nan"], ["--patch-limit-gib", "inf"], ["--patch-limit-gib", "-1"],
                                  ["--patch-limit-gib", "abc"], ["--profile", "nope"], ["--config", "/nonexistent.json"]])
def test_wsp_apply_refuses_bad_flags(tmp_path, monkeypatch, capsys, flag):
    from wsp.__main__ import main
    target, argv, _ = apply_without_flags(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as stopped:
        main(argv + flag)
    assert stopped.value.code == 2 and not target.patches


def test_wsp_apply_dry_run_reports_the_limit_and_commits_nothing(tmp_path, monkeypatch, capsys):
    from wsp.__main__ import main
    target, argv, _ = apply_without_flags(tmp_path, monkeypatch)
    main(argv + ["--dry-run", "--patch-limit-gib", str(gib_for(100))])
    out = capsys.readouterr().out
    assert out.startswith("ERROR") and "Largest weight patch setting allows 100 weights" in out and not target.patches


# ------------------------------------------------------------------ actions() and script() under each limit

def test_a_small_setting_refuses_and_names_the_setting(quant):
    ws = quant.ws
    ws.edit_settings = {"patch_limit_gib": gib_for(100)}
    reason = S.actions(ws, DOWN)["scale"]["reason"]
    assert not S.actions(ws, DOWN)["scale"]["enabled"]
    assert reason.startswith("This is 384 weights (less than 1 MiB as FP32 edits), too large for a weight patch: ")
    assert "Largest weight patch setting allows 100 weights" in reason and "raise Largest weight patch in Settings" in reason
    assert "memory" not in reason
    with pytest.raises(S.Refusal, match="Largest weight patch setting allows 100 weights"):
        S.script(ws, DOWN, "scale", 2)
    neurons = S.actions(ws, "blocks.0.mlp")["scale"]["reason"]
    assert "Run tab's steering panel" in neurons                        # the steering line stays for an MLP


def test_a_large_setting_allows_and_the_script_says_what_it_costs(quant):
    ws = quant.ws
    ws.edit_settings = {"patch_limit_gib": 10}
    assert S.actions(ws, DOWN)["scale"]["enabled"]
    r = S.script(ws, DOWN, "scale", 2)
    assert r["weights"] == 384 and r["cost"]["weights"] == 384 and r["cost"]["disk_bytes"] == 384 * 4
    assert r["cost"]["memory_bytes"] == 384 * 12 and r["cost"]["confirm"] is False and r["cost"]["cancellable"] is False


def test_automatic_follows_the_machines_memory_and_a_refusal_names_memory(quant, monkeypatch):
    ws = quant.ws
    monkeypatch.setattr(EL, "SMALL_EDIT", 0)                            # measure memory at this tiny size
    EL.telemetry = machine(room_for(1000))
    assert S.actions(ws, DOWN)["scale"]["enabled"]
    assert EL.allowed(ws, 384).binding == "memory" and EL.allowed(ws, 384).weights == 1000
    EL.forget_machine()
    avail = room_for(200)
    EL.telemetry = machine(avail)
    reason = S.actions(ws, DOWN)["scale"]["reason"]
    preferred = policy_for(DEFAULTS, 64 * GiB, 64 * GiB).preferred
    assert (f"memory allows 200 weights now: {EL.gib(avail)} is available on this machine, {EL.gib(preferred)} is "
            f"kept free by the Admission headroom reserve and 64 MiB is set aside for reading the weights, at an "
            f"estimated 12 bytes per patch weight while a patch is built and kept (the runtime supervisor's admission "
            f"charge; the edit runs here without the supervisor, so no memory admission checks it). Select fewer "
            f"neurons or a smaller slice, or free memory on the machine.") in reason
    assert "does not override memory" in reason and "setting allows" not in reason
    ws.edit_settings = {"patch_limit_gib": 10}                          # an explicit setting never overrides memory
    with pytest.raises(S.Refusal, match="memory allows 200 weights now"):
        S.script(ws, DOWN, "scale", 2)


def test_in_process_memory_pressure_alone_does_not_zero_the_limit(quant, monkeypatch):
    """Nothing else admits an in-process edit: the measured room is the limit at every pressure level."""
    monkeypatch.setattr(EL, "SMALL_EDIT", 0)
    EL.telemetry = machine(room_for(10 ** 6), pressure=2)
    assert S.actions(quant.ws, DOWN)["scale"]["enabled"]
    assert EL.allowed(quant.ws, 384).weights == 10 ** 6
    out = quant.apply(f'P("{DOWN}")[:, :] *= 2', "agent")
    assert out["ok"] and out["committed"], out


def test_memory_and_disk_are_not_measured_for_a_small_edit(quant, monkeypatch):
    readings, usages = [], []
    EL.telemetry = machine(0, calls=readings)                           # no memory at all: still never asked
    monkeypatch.setattr(EL, "shutil", disk(0, calls=usages))
    assert S.actions(quant.ws, DOWN)["scale"]["enabled"]
    assert S.script(quant.ws, DOWN, "scale", 2)["weights"] == 384
    out = quant.apply(f'P("{DOWN}")[:, :] *= 2', "console")
    assert out["ok"] and out["committed"], out
    assert readings == [] and usages == []


def test_without_a_memory_reading_selection_planning_uses_the_old_figure(quant, monkeypatch):
    ws = quant.ws
    monkeypatch.setattr(EL, "SMALL_EDIT", 0)
    monkeypatch.setattr(EL, "FALLBACK", 100)                            # stands for the old 60,000,000
    def unavailable():
        raise RuntimeError("no telemetry on this OS")
    EL.telemetry = unavailable
    reason = S.actions(ws, DOWN)["scale"]["reason"]
    assert ("memory could not be measured for this edit (memory telemetry failed: RuntimeError: no telemetry on "
            "this OS), so the old fixed limit of 100 weights applies") in reason
    assert "nothing else checks memory when the edit runs here, so set it no higher than this machine holds" in reason
    assert "still checks memory" not in reason
    with pytest.raises(S.Refusal, match="old fixed limit of 100 weights"):
        S.script(ws, DOWN, "scale", 2)
    # A script never had a total cap: without a memory reading it is bounded by the setting and disk only
    out = quant.apply(f'P("{DOWN}")[:, :] *= 2', "agent")
    assert out["ok"] and out["committed"], out
    ws.edit_settings = {"patch_limit_gib": gib_for(1000)}
    assert S.actions(ws, DOWN)["scale"]["enabled"]


def test_an_edit_above_the_old_figure_is_allowed_when_the_limit_allows_it(quant, monkeypatch):
    """CONFIRM_ABOVE stands for the old fixed 60,000,000: a selection above it is enabled, asks first, and its
    generated script commits through Session.apply as one patch."""
    ws = quant.ws
    monkeypatch.setattr(EL, "SMALL_EDIT", 0)
    monkeypatch.setattr(EL, "CONFIRM_ABOVE", 100)
    EL.telemetry = machine(room_for(10 ** 6))
    acts = S.actions(ws, DOWN)
    assert acts["scale"]["enabled"] and "you are asked to confirm first" in acts["scale"]["reason"]
    r = S.script(ws, DOWN, "scale", 2)
    assert r["cost"]["confirm"] is True and r["cost"]["confirm_above"] == 100
    out = quant.apply(r["script"], "large")
    assert out["ok"] and out["committed"] and out["stored"] == 384


def test_the_confirmation_starts_one_weight_above_the_old_figure(quant, monkeypatch):
    assert EL.cost(quant.ws, EL.CONFIRM_ABOVE)["confirm"] is False
    assert EL.cost(quant.ws, EL.CONFIRM_ABOVE + 1)["confirm"] is True
    monkeypatch.setattr(EL, "CONFIRM_ABOVE", 384)
    assert "confirm first" not in S.actions(quant.ws, DOWN)["scale"]["reason"]
    monkeypatch.setattr(EL, "CONFIRM_ABOVE", 383)
    assert "confirm first" in S.actions(quant.ws, DOWN)["scale"]["reason"]


def test_revert_is_limited_and_asks_like_any_edit(quant, monkeypatch):
    ws = quant.ws
    assert quant.apply(f'P("{DOWN}")[:, :] *= 2', "edit")["committed"]
    assert S.actions(ws, DOWN)["revert"]["enabled"]
    monkeypatch.setattr(EL, "CONFIRM_ABOVE", 100)
    revert = S.actions(ws, DOWN)["revert"]
    assert revert["enabled"] and "you are asked to confirm first). " + S.REVERT_NOTE in revert["reason"]
    ws.edit_settings = {"patch_limit_gib": gib_for(100)}
    revert = S.actions(ws, DOWN)["revert"]
    assert not revert["enabled"] and "Largest weight patch setting allows 100 weights" in revert["reason"]


def test_a_tie_between_the_setting_and_memory_names_the_setting(quant, monkeypatch):
    ws = quant.ws
    monkeypatch.setattr(EL, "SMALL_EDIT", 0)
    EL.telemetry = machine(room_for(300))
    ws.edit_settings = {"patch_limit_gib": gib_for(300)}
    limit = EL.allowed(ws, 384, scratch=384 * 12)
    assert limit.found["memory"][0] == limit.found["setting"][0] == 300
    assert limit.binding == "setting" and limit.exceeded(384).binding == "setting"
    assert "Largest weight patch setting allows 300 weights" in S.actions(ws, DOWN)["scale"]["reason"]


def test_one_view_asks_the_memory_watchdog_once(quant, monkeypatch):
    """Scale and revert share one advisory: opening a node costs at most one supervisor round trip."""
    ws = quant.ws
    assert quant.apply(f'P("{DOWN}")[:, :] *= 2', "edit")["committed"]
    monkeypatch.setattr(EL, "SMALL_EDIT", 0)
    client = Advisor(dict(largest=10 ** 6, event=None, available=40 * GiB, preferred=25 * GiB, scratch=64 * MiB))
    ws._runtime_client = client
    try:
        acts = S.actions(ws, DOWN)
        assert acts["scale"]["enabled"] and acts["revert"]["enabled"] and client.calls == 1
    finally:
        ws._runtime_client = None


def test_a_limit_measured_for_a_larger_edit_never_refuses_a_small_one(quant, monkeypatch):
    """A node view measures once for the larger of scale and revert; checked against the smaller, a small edit
    is still exempt from memory and disk (Limit.exceeded)."""
    monkeypatch.setattr(EL, "shutil", disk(0))
    limit = EL.allowed(quant.ws, EL.SMALL_EDIT + 1)
    assert limit.binding == "disk" and limit.weights == 0
    assert limit.exceeded(EL.SMALL_EDIT) is None and limit.exceeded(EL.SMALL_EDIT + 1).binding == "disk"


def test_memory_and_disk_are_measured_only_past_exactly_small_edit(quant, monkeypatch):
    assert EL.SMALL_EDIT == 1_000_000
    readings, usages = [], []
    EL.telemetry = machine(512 * GiB, total=1024 * GiB, calls=readings)
    monkeypatch.setattr(EL, "shutil", disk(1 << 50, calls=usages))
    limit = EL.allowed(quant.ws, EL.SMALL_EDIT)                         # exactly 1,000,000: not measured
    assert limit.binding == "none" and readings == [] and usages == []
    limit = EL.allowed(quant.ws, EL.SMALL_EDIT + 1)
    assert readings and usages and set(limit.found) == {"memory", "disk"}


def test_a_watchdog_that_admits_no_edit_never_refuses_a_small_one(quant, monkeypatch):
    changed = dict(code="model_changed", action="refused", message="The model file changed; reopen it before retrying.")
    quant.ws._runtime_client = Advisor(dict(largest=0, event=changed))
    try:
        limit = EL.allowed(quant.ws, EL.SMALL_EDIT + 1)
    finally:
        quant.ws._runtime_client = None
    assert limit.binding == "watchdog" and limit.weights == 0
    assert limit.exceeded(EL.SMALL_EDIT) is None and limit.exceeded(EL.SMALL_EDIT + 1).binding == "watchdog"


def test_one_view_measures_for_the_larger_of_scale_and_revert(quant, monkeypatch):
    """The shared limit is asked for the larger edit. Revert is not larger than scale in this fixture, so its
    plan is made larger here; the limit must still be measured for it and refuse it."""
    ws = quant.ws
    assert quant.apply(f'P("{DOWN}")[:, :] *= 2', "edit")["committed"]
    real_plan, real_limit, asked = S.plan, S._limit, []
    def plan(ws, addr, action, sel=None):
        p = real_plan(ws, addr, action, sel)
        return dict(p, written=p["written"] + 1000) if action == "revert" else p
    monkeypatch.setattr(S, "plan", plan)
    monkeypatch.setattr(S, "_limit", lambda ws, written: (asked.append(written), real_limit(ws, written))[1])
    monkeypatch.setattr(EL, "SMALL_EDIT", 1000)                         # scale (384) exempt, revert (1,384) measured
    EL.telemetry = machine(room_for(500, scratch=1384 * 12))
    acts = S.actions(ws, DOWN)
    assert asked == [1384]
    assert acts["scale"]["enabled"] and not acts["revert"]["enabled"]
    assert "memory allows 500 weights now" in acts["revert"]["reason"]


def test_only_a_gguf_workspace_runs_in_the_model_worker():
    worker = SimpleNamespace(worker=True)
    assert EL.worker_hosted(SimpleNamespace(kind="gguf", _runtime_client=worker))
    assert not EL.worker_hosted(SimpleNamespace(kind="hf", _runtime_client=worker))
    assert not EL.worker_hosted(SimpleNamespace(_runtime_client=worker))
    assert not EL.worker_hosted(SimpleNamespace(kind="gguf", _runtime_client=SimpleNamespace(worker=False)))
    assert not EL.worker_hosted(SimpleNamespace(kind="gguf"))
    c = EL.cost(SimpleNamespace(kind="hf", _runtime_client=worker), 1000)
    assert c["worker_hosted"] is False and "held twice" not in c["message"]


def test_the_generated_script_of_an_allowed_selection_is_not_refused_by_script_guards(quant, monkeypatch):
    """The wscript guards bound one read (MAX_VIEW) and one constructed tensor (MAX_ELEMS), never the whole patch.
    Generated scripts read at most selections.WINDOW at once; with all three lowered far below the selection,
    the script still applies and no read is larger than WINDOW."""
    assert S.WINDOW <= W.MAX_VIEW                                       # the production invariant
    ws = quant.ws
    monkeypatch.setattr(S, "WINDOW", 48)
    monkeypatch.setattr(W, "MAX_VIEW", 48)
    monkeypatch.setattr(W, "MAX_ELEMS", 6)
    for addr in (DOWN, "blocks.1.mlp", f"{DOWN}[:, 2:20]"):
        r = S.script(ws, addr, "scale", 2)
        assert r["weights"] > 48
        ws.store.reads.clear()
        out = quant.apply(r["script"], addr)
        assert out["ok"] and out["committed"], out
        assert max(k for _, k in ws.store.reads) <= 48


# ------------------------------------------------------------------ disk

def test_a_small_edit_is_not_refused_on_a_nearly_full_disk(quant, monkeypatch):
    """At or below SMALL_EDIT weights disk is not measured, as before the limit existed: buttons, the
    generated script and the console agree."""
    ws = quant.ws
    monkeypatch.setattr(EL, "shutil", disk(500 * MiB))
    assert S.actions(ws, DOWN)["scale"]["enabled"]
    assert S.script(ws, DOWN, "scale", 2)["weights"] == 384
    out = quant.apply(f'P("{DOWN}")[:, :] *= 2', "console")
    assert out["ok"] and out["committed"], out


def test_free_disk_space_less_the_reserve_bounds_a_larger_edit(quant, monkeypatch):
    ws = quant.ws
    monkeypatch.setattr(EL, "SMALL_EDIT", 0)
    monkeypatch.setattr(EL, "shutil", disk(500 * MiB))
    reason = S.actions(ws, DOWN)["scale"]["reason"]
    assert "free disk space allows 0 weights (500 MiB free, 1.0 GiB kept free)" in reason and "free disk space." in reason
    with pytest.raises(S.Refusal, match="free disk space allows 0 weights"):
        S.script(ws, DOWN, "scale", 2)
    out = quant.apply(f'P("{DOWN}")[:, :] *= 2', "console")
    assert not out["ok"] and "free disk space allows 0 weights (500 MiB free, 1.0 GiB kept free)" in out["error"]
    monkeypatch.setattr(EL, "shutil", disk(EL.DISK_RESERVE + 200 * EL.PATCH_BYTES))     # room for exactly 200
    limit = EL.allowed(ws, 384)
    assert limit.binding == "disk" and limit.weights == 200
    assert "free disk space allows 200 weights" in S.actions(ws, DOWN)["scale"]["reason"]
    monkeypatch.setattr(EL, "shutil", disk(EL.DISK_RESERVE + 384 * EL.PATCH_BYTES))
    assert S.actions(ws, DOWN)["scale"]["enabled"]
    out = quant.apply(f'P("{DOWN}")[:, :] *= 2', "console")
    assert out["ok"] and out["committed"], out


# ------------------------------------------------------------------ the guard every edit path shares

def test_a_script_larger_than_the_setting_is_refused_while_it_runs(quant):
    """An agent or console script has no dialog: the limit is its guard (wscript Interp._add), and nothing is
    committed."""
    ws = quant.ws
    ws.edit_settings = {"patch_limit_gib": gib_for(100)}
    before = len(ws.patches)
    out = quant.apply(f'P("{DOWN}")[:, :] *= 2', "agent")
    assert not out["ok"] and "Largest weight patch setting allows 100 weights" in out["error"]
    assert len(ws.patches) == before
    small = quant.apply(f'P("{DOWN}")[0, 0:50] *= 2', "agent")      # 24 columns: a 24-weight patch
    assert small["ok"] and small["committed"]


def test_the_interpreter_itself_stops_at_the_setting(quant):
    """Below SMALL_EDIT the guard still checks a setting smaller than it (Interp._unchecked)."""
    quant.ws.edit_settings = {"patch_limit_gib": gib_for(100)}
    with pytest.raises(W.ScriptError, match="Largest weight patch setting allows 100 weights"):
        W.run_script(quant.ws, f'P("{DOWN}")[:, :] *= 2')


def test_overlapping_writes_are_compacted_before_the_guard_refuses(quant):
    quant.ws.edit_settings = {"patch_limit_gib": gib_for(400)}
    out = quant.apply(f'P("{DOWN}")[:, :] *= 2\nP("{DOWN}")[:, :] *= 3', "twice")    # 768 written, 384 stored
    assert out["ok"] and out["committed"] and out["stored"] == 384, out


def test_the_script_guard_measures_memory_only_in_process_and_once_past_small_edits(quant, monkeypatch):
    ws = quant.ws
    readings = []
    EL.telemetry = machine(room_for(10 ** 6), calls=readings)
    monkeypatch.setattr(EL, "SMALL_EDIT", 1000)
    interp = W.Interp(ws)
    assert "memory" not in interp._limit(10).found and readings == []
    first = interp._limit(2000)
    assert first.found["memory"][0] == 10 ** 6 and len(readings) == 1
    assert interp._limit(3000) is first and len(readings) == 1         # measured once
    client = Advisor(dict(largest=5))
    ws._runtime_client = client
    try:
        limit = W.Interp(ws)._limit(2000)
        assert "memory" not in limit.found and "disk" in limit.found and client.calls == 0
        assert EL.allowed(ws, 2000, memory=False).found.keys() == {"disk"}
    finally:
        ws._runtime_client = None


def test_commit_deltas_applies_the_setting_to_every_edit_path(quant):
    ws = quant.ws
    ws.edit_settings = {"patch_limit_gib": gib_for(100)}
    piece = D.block((range(0, 16), range(0, 24)), torch.ones(16, 24))
    out = quant.commit_deltas({DOWN: [piece]}, "files", "programmer", {"kind": "files"})
    assert not out["ok"] and "Largest weight patch setting allows 100 weights" in out["error"]
    assert not ws.patches


def test_commit_deltas_counts_overlapping_pieces_as_the_patch_stores_them(quant):
    """Agent-file pieces are committed as they come (not compacted): two overlapping full pieces are stored, and
    counted, as 768 weights, which is the size the patch really has."""
    ws = quant.ws
    pieces = lambda: [D.block((range(0, 16), range(0, 24)), torch.full((16, 24), v)) for v in (1.0, 2.0)]  # noqa: E731
    dry = quant.commit_deltas({DOWN: pieces()}, "files", "programmer", {"kind": "files"}, dry_run=True)
    assert dry["ok"] and dry["stored"] == 768
    ws.edit_settings = {"patch_limit_gib": gib_for(767)}
    out = quant.commit_deltas({DOWN: pieces()}, "files", "programmer", {"kind": "files"})
    assert not out["ok"] and "This is 768 weights" in out["error"] and "allows 767 weights" in out["error"]
    ws.edit_settings = {"patch_limit_gib": gib_for(768)}
    out = quant.commit_deltas({DOWN: pieces()}, "files", "programmer", {"kind": "files"})
    assert out["ok"] and out["committed"] and ws.patches[-1].stored == 768


def test_a_script_is_refused_when_the_machines_memory_does_not_allow_it(quant, monkeypatch):
    monkeypatch.setattr(EL, "SMALL_EDIT", 0)
    EL.telemetry = machine(room_for(100))
    out = quant.apply(f'P("{DOWN}")[:, :] *= 2', "agent")
    assert not out["ok"] and "memory allows 100 weights now" in out["error"]


# ------------------------------------------------------------------ cost, time and the dialog text

def test_the_cost_states_sizes_time_and_gguf_slowdown(quant):
    ws = quant.ws
    c = EL.cost(ws, 383_273_184)
    assert c["confirm"] and c["cancellable"] is False and c["estimate_seconds"] is None and c["gguf"]
    assert "1.4 GiB on disk" in c["message"] and "Memory admission counts" not in c["message"]      # in-process
    assert ("In memory it is estimated at about 4.3 GiB (12 bytes per weight, the runtime supervisor's admission "
            "charge) while it is built and kept; the edit runs here without the supervisor, so no memory admission "
            "checks it; the real use on this model has not been measured.") in c["message"]
    assert c["worker_hosted"] is False and "held twice" not in c["message"]
    assert "Time: no estimate" in c["message"] and "since this workspace was opened" in c["message"]
    assert "cannot be stopped" in c["message"]
    assert "Generation is slower while this patch is enabled" in c["message"] and "has not been measured" in c["message"]
    assert "adds the patch's contribution to the output of each matrix" in c["message"]
    assert "the quantized weights are never changed" in c["message"]
    EL.record_timing(ws, 500, 1.0)                                      # too small to time
    assert ws.edit_timing is None
    EL.record_timing(ws, 10_000_000, 4.0)
    c = EL.cost(ws, 100_000_000)
    assert c["estimate_seconds"] == pytest.approx(40.0) and "about 40 s" in c["message"] and "estimate" in c["message"]
    assert "10,000,000 weights in 4.0 s" in c["message"] and "since this workspace was opened" in c["message"]


def test_the_cost_says_a_worker_hosted_patch_is_held_twice(quant):
    ws = quant.ws
    ws._runtime_client = SimpleNamespace(worker=True)
    try:
        c = EL.cost(ws, 383_273_184)
        assert c["worker_hosted"] is True
        assert "held twice, by the server and by the worker: at least 2.9 GiB" in c["message"]
        assert ("Memory admission counts about 4.3 GiB for it (12 bytes per weight) while it is built and kept; the "
                "real use on this model has not been measured.") in c["message"]                # under the supervisor
        assert "estimated" not in c["message"]
    finally:
        ws._runtime_client = None


def test_apply_times_large_edits_for_the_next_estimate(quant, monkeypatch):
    monkeypatch.setattr(EL, "TIMED_MIN", 100)
    out = quant.apply(f'P("{DOWN}")[:, :] *= 2', "timed")
    assert out["committed"] and quant.ws.edit_timing["stored"] == 384 and quant.ws.edit_timing["seconds"] > 0


def test_progress_events_carry_the_weights_written(quant, monkeypatch):
    seen = []
    quant.ws.listeners.append(lambda ev: seen.append((ev["type"], ev)))
    monkeypatch.setattr(W, "PROGRESS_EVERY", 0)
    quant.apply("\n".join(f'P("{DOWN}")[{i}, :] *= 2' for i in range(4)), "steps", request="r1")
    steps = [kw for kind, kw in seen if kind == "edit_progress"]
    assert steps and all("stored" in s for s in steps) and steps[-1]["stored"] == 72 and steps[-1]["request"] == "r1"


# ------------------------------------------------------------------ refusal texts under the supervisor

def advice_from_admission(tmp_path, available, *, budget_gib=None, ceiling_gib=None, tree=4 * GiB):
    """What the supervisor's advise() answers for a cold edit (no model loaded), from the real admission.
    `tree`: the memory measured for the server's process tree."""
    overrides = dict(backend="cpu", floor_gib=20, preferred_gib=25)
    if budget_gib is not None:
        overrides["budget_gib"] = budget_gib
    if ceiling_gib is not None:
        overrides["ceiling_gib"] = ceiling_gib
    settings = resolve(config(tmp_path), env={}, system="Linux", overrides=overrides)
    value = dict(forecast=dict(total=3 * GiB, patches=0, weights=GiB), working={"activations": SCRATCH_ADMISSION_QUANTUM},
                 runner=False)
    sample = dict(available=available, total=128 * GiB, pressure=1, tree_footprint=tree, tree_rss=tree)
    n, refused = largest_patch(settings, value, sample, None, None, per_weight=PATCH_ADMISSION_BYTES)
    vals = settings["values"]
    pol = policy_for(vals, sample["total"], sample["available"])
    return dict(largest=n, event=refused, available=available, preferred=pol.preferred, scratch=SCRATCH_ADMISSION_QUANTUM,
                bounds=dict(budget=pol.budget, budget_gib=vals["budget_gib"], ceiling_gib=vals["ceiling_gib"],
                            loaded=False, total=sample["total"], measured=measured_total(sample)))


def runtime_reason(quant, monkeypatch, advice):
    monkeypatch.setattr(EL, "SMALL_EDIT", 0)
    quant.ws._runtime_client = Advisor(advice)
    try:
        return S.actions(quant.ws, DOWN)["scale"]["reason"], EL.allowed(quant.ws, 384, fallback=True)
    finally:
        quant.ws._runtime_client = None


def test_a_memory_budget_refusal_names_the_budget_and_the_remedy_that_works(tmp_path, quant, monkeypatch):
    advice = advice_from_admission(tmp_path, 100 * GiB, budget_gib=30)     # the reviewer's case
    n = advice["largest"]
    assert n > 0 and advice["event"]["code"] == "forecast_exceeds_budget"
    _, limit = runtime_reason(quant, monkeypatch, advice)
    assert limit.binding == "memory" and limit.weights == n
    reason = EL.refusal(n + 1, limit)
    assert f"memory admission allows {n:,} weights now" in reason
    assert "must stay within 30.0 GiB, the Memory budget setting, and one weight more would exceed it" in reason
    assert "raise Memory budget in Settings (freeing memory does not help)" in reason
    assert "free memory on the machine" not in reason


def test_an_available_memory_refusal_gives_numbers_that_add_up(tmp_path, quant, monkeypatch):
    advice = advice_from_admission(tmp_path, 40 * GiB)
    n = advice["largest"]
    measured = advice["event"]["measured"]
    _, limit = runtime_reason(quant, monkeypatch, advice)
    reason = EL.refusal(n + 1, limit)
    assert f"memory admission allows {n:,} weights now" in reason
    assert (f"one weight more would need {EL.gib(measured['new_bytes'])} of new memory (loading the model with the "
            f"patch, 12 bytes per patch weight), with {EL.gib(advice['available'])} available and 25.0 GiB kept free "
            "by the Admission headroom reserve") in reason
    assert measured["available"] - measured["new_bytes"] < 25 * GiB <= measured["available"] - (measured["new_bytes"] - 12)
    assert "or free memory on the machine" in reason


def test_a_memory_ceiling_refusal_names_the_ceiling(tmp_path, quant, monkeypatch):
    advice = advice_from_admission(tmp_path, 100 * GiB, ceiling_gib=30)
    n = advice["largest"]
    assert n > 0 and advice["event"]["code"] == "forecast_exceeds_budget"
    _, limit = runtime_reason(quant, monkeypatch, advice)
    reason = EL.refusal(n + 1, limit)
    assert "must stay within 30.0 GiB, the Memory ceiling setting, and one weight more would exceed it" in reason
    assert "raise Memory ceiling in Settings (freeing memory does not help)" in reason and "Memory budget" not in reason


def test_measured_memory_over_the_cap_admits_no_edit_and_says_why(tmp_path, quant, monkeypatch):
    """Admission refuses every operation while the measured memory is over the cap: no smaller selection helps."""
    advice = advice_from_admission(tmp_path, 100 * GiB, budget_gib=30, tree=35 * GiB)
    assert advice["largest"] == 0 and advice["event"]["code"] == "forecast_exceeds_budget"
    reason, limit = runtime_reason(quant, monkeypatch, advice)
    assert limit.binding == "memory" and limit.weights == 0
    assert ("memory admission allows no edit now: the memory measured for the server and its model processes, "
            "35.0 GiB, is already over 30.0 GiB, the Memory budget setting, and admission refuses every operation "
            "while it is, whatever its size. Unload the model to release the memory it holds (restart the server if "
            "no model is loaded), or raise Memory budget in Settings.") in reason
    assert "Select fewer" not in reason and "freeing memory does not help" not in reason
    physical = advice_from_admission(tmp_path, 100 * GiB, tree=120 * GiB)          # no setting: 128 GiB less the reserve
    reason, _ = runtime_reason(quant, monkeypatch, physical)
    assert "is already over 103.0 GiB, physical memory less the Admission headroom reserve" in reason
    assert reason.endswith("or lower the Admission headroom reserve. Raising Largest weight patch in Settings does "
                           "not override memory.")


def test_when_even_one_weight_does_not_fit_no_smaller_selection_is_suggested(tmp_path, quant, monkeypatch):
    advice = advice_from_admission(tmp_path, 26 * GiB)                  # 1 GiB over the 25 GiB reserve, 3 GiB to load
    assert advice["largest"] == 0 and advice["event"]["code"] == "forecast_exceeds_budget"
    reason, limit = runtime_reason(quant, monkeypatch, advice)
    assert limit.binding == "memory"
    assert "memory admission allows no edit now: even one weight would need 3.0 GiB of new memory" in reason
    assert "Select fewer" not in reason and "kept free by the Admission headroom reserve. Free memory on the machine." in reason
    EL.forget_machine()
    EL.telemetry = machine(room_for(0) - 1)                             # in-process, no room at all
    monkeypatch.setattr(EL, "SMALL_EDIT", 0)
    reason = EL.refusal(384, EL.allowed(quant.ws, 384))
    assert "memory allows 0 weights now" in reason and "Select fewer" not in reason
    assert "no memory admission checks it). Free memory on the machine." in reason


def test_a_watchdog_that_admits_no_edit_for_another_reason_says_so(quant, monkeypatch):
    event = dict(code="model_changed", action="refused", message="The model file changed; reopen it before retrying.",
                 recommendations=["Unload and reopen the model."])
    reason, limit = runtime_reason(quant, monkeypatch, dict(largest=0, event=event))
    assert limit.binding == "watchdog"
    assert ("the memory watchdog admits no edit now: The model file changed; reopen it before retrying. "
            "Unload and reopen the model.") in reason
    assert "memory admission allows" not in reason
    pressure = dict(code="memory_pressure", action="refused", message="Memory pressure is not normal.")
    reason, limit = runtime_reason(quant, monkeypatch, dict(largest=0, event=pressure))
    assert limit.binding == "memory" and "memory pressure on the machine is not normal" in reason


def test_an_admission_refusal_without_its_figures_at_zero_gives_the_watchdogs_advice(quant, monkeypatch):
    bare = dict(code="forecast_exceeds_budget", action="refused",
                message="The operation forecast exceeds current memory admission bounds.",
                recommendations=["Free memory and retry without restarting.", "Reduce the edit."])
    reason, limit = runtime_reason(quant, monkeypatch, dict(largest=0, event=bare))
    assert limit.binding == "memory" and limit.weights == 0
    assert ("memory admission allows no edit now (The operation forecast exceeds current memory admission bounds). "
            "Free memory and retry without restarting. Raising Largest weight patch") in reason
    assert "Select fewer" not in reason
    reason, _ = runtime_reason(quant, monkeypatch, dict(largest=100, event=bare))
    assert "memory admission allows 100 weights now (The operation forecast" in reason and "Select fewer" in reason


def test_without_an_answer_planning_falls_back_and_says_why(quant, monkeypatch):
    monkeypatch.setattr(EL, "FALLBACK", 100)
    reason, limit = runtime_reason(quant, monkeypatch, dict(largest=None, reason="the memory watchdog did not answer within 1.5 s"))
    assert limit.binding == "fallback"
    assert ("memory could not be measured for this edit (the memory watchdog did not answer within 1.5 s), so the old "
            "fixed limit of 100 weights applies") in reason
    assert "the memory watchdog still checks memory when the edit runs" in reason


# ------------------------------------------------------------------ the client's advisory call

def advising_client(tmp_path):
    from test_runtime_supervisor import fake_ws
    client = RuntimeClient(tmp_path, resolve(env={"HOME": str(tmp_path)}))          # nobody answers this mailbox
    ws = fake_ws(tmp_path)
    ws.dir, ws.edit_settings = str(tmp_path), None
    return client, ws


def test_no_answer_is_bounded_remembered_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ADVISE_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(EL, "SMALL_EDIT", 0)
    client, ws = advising_client(tmp_path)
    started = time.monotonic()
    got = client.advise_edit(ws, 0)
    elapsed = time.monotonic() - started
    waited = re.fullmatch(r"the memory watchdog did not answer within (\d+\.\d) s", got["reason"] or "")
    assert got["largest"] is None and waited and 0.2 <= float(waited[1]) <= round(elapsed, 1) + 0.05
    assert elapsed < 0.6 and client.last_event is None
    (tmp_path / "request.json").unlink()
    started = time.monotonic()
    assert client.advise_edit(ws, 0) is got and time.monotonic() - started < 0.1      # the lack of an answer is reused
    assert not (tmp_path / "request.json").exists()
    ws._runtime_client = client
    limit = EL.allowed(ws, 1, fallback=True)
    assert limit.binding == "fallback" and limit.weights == EL.FALLBACK
    assert got["reason"] in EL.refusal(EL.FALLBACK + 1, limit)
    assert "memory" not in EL.allowed(ws, 1).found and "fallback" not in EL.allowed(ws, 1).found
    assert not (tmp_path / "reply.json").exists()


def test_one_deadline_covers_both_locks_and_the_reply(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ADVISE_TIMEOUT_SECONDS", 0.6)
    monkeypatch.setattr(R, "ADVISE_SECONDS", 0)
    client, ws = advising_client(tmp_path)

    def hold(lock, seconds):
        held = threading.Event()
        def run():
            with lock:
                held.set()
                time.sleep(seconds)
        t = threading.Thread(target=run)
        t.start()
        held.wait()
        return t
    for lock, reason in ((client.lock, "the runtime client is busy with another model operation"),
                         (client.request_lock, "the memory watchdog mailbox is busy with another request")):
        t = hold(lock, 1.5)
        started = time.monotonic()
        got = client.advise_edit(ws, 0)
        assert got == dict(largest=None, reason=reason) and time.monotonic() - started < 0.9
        t.join()
    t = hold(client.lock, 0.4)                       # 0.4 s of the 0.6 s go waiting for the lock, the rest for a reply
    started = time.monotonic()
    got = client.advise_edit(ws, 0)
    elapsed = time.monotonic() - started
    t.join()
    assert "did not answer" in got["reason"] and 0.5 < elapsed < 0.85


def test_the_wait_named_is_the_time_waited(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ADVISE_TIMEOUT_SECONDS", 0.6)
    client, ws = advising_client(tmp_path)
    got = client.advise_edit(ws, 0)
    assert re.fullmatch(r"the memory watchdog did not answer within 0\.[67] s", got["reason"])
    got = client._advise(ws, SCRATCH_ADMISSION_QUANTUM, time.monotonic() + 0.3)        # a shorter deadline
    assert re.fullmatch(r"the memory watchdog did not answer within 0\.[34] s", got["reason"])


def test_the_advisory_descriptor_carries_no_operation(tmp_path, monkeypatch):
    """Inside an operation `_value` adds its id and job; an advisory request must not claim the operation."""
    monkeypatch.setattr(R, "ADVISE_TIMEOUT_SECONDS", 0.2)
    client, ws = advising_client(tmp_path)
    client.operation_id = "op-1"
    assert client._value(ws, name="edit", stored=0, scratch=SCRATCH_ADMISSION_QUANTUM)["operation_id"] == "op-1"
    client.advise_edit(ws, 0)
    sent = json.loads((tmp_path / "request.json").read_text())
    assert sent["action"] == "advise" and "operation_id" not in sent["descriptor"] and "job" not in sent["descriptor"]


def test_a_slow_edit_description_uses_up_the_deadline_and_nothing_is_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ADVISE_TIMEOUT_SECONDS", 0.2)
    client, ws = advising_client(tmp_path)
    described = client._value
    monkeypatch.setattr(client, "_value", lambda *a, **k: (time.sleep(0.5), described(*a, **k))[1])
    started = time.monotonic()
    got = client.advise_edit(ws, 0)
    elapsed = time.monotonic() - started
    assert got["largest"] is None and 0.5 <= elapsed < 0.8               # describing cannot be interrupted
    assert re.fullmatch(r"describing the edit for memory admission took 0\.[56] s, past the 0\.2 s limit, "
                        r"so the memory watchdog was not asked", got["reason"])
    assert not (tmp_path / "request.json").exists()                     # no request sent with no time left


def test_an_answer_without_a_largest_gives_the_watchdogs_reason(tmp_path):
    client, ws = advising_client(tmp_path)
    unavailable = dict(code="telemetry_unavailable", action="blocked", message="Memory sampling failed; retrying.")
    def answer():
        def request():
            try:
                return json.loads((tmp_path / "request.json").read_text())
            except (OSError, ValueError):
                return None
        got = until(request, 5)
        R.write(tmp_path / "reply.json", dict(id=got["id"], ok=True, reserved=False, advice=dict(largest=None, event=unavailable)))
    responder = threading.Thread(target=answer)
    responder.start()
    got = client.advise_edit(ws, 0)
    responder.join()
    assert got["largest"] is None and got["reason"] == "Memory sampling failed; retrying." and got["event"] == unavailable
    assert client.last_event is None


def test_a_mailbox_that_cannot_be_written_is_no_answer_not_an_error(tmp_path, monkeypatch):
    client, ws = advising_client(tmp_path)
    def broken(path, value):
        raise OSError("disk full")
    monkeypatch.setattr(R, "write", broken)
    got = client.advise_edit(ws, 0)
    assert got == dict(largest=None, reason="the memory watchdog mailbox could not be written: OSError: disk full")


def test_an_advisory_refusal_is_not_kept_as_the_clients_event(tmp_path):
    client, ws = advising_client(tmp_path)
    stop = threading.Event()
    def responder():
        while not stop.is_set():
            try:
                request = json.loads((tmp_path / "request.json").read_text())
            except (FileNotFoundError, ValueError):
                time.sleep(.01)
                continue
            R.write(tmp_path / "reply.json", dict(id=request["id"], ok=False,
                                                  event=dict(code="shutdown", message="Server is shutting down.")))
            return
    t = threading.Thread(target=responder)
    t.start()
    try:
        got = client.advise_edit(ws, 0)
    finally:
        stop.set()
        t.join()
    assert got == dict(largest=None, reason="Server is shutting down.") and client.last_event is None


def test_the_advice_is_reused_only_for_the_same_workspace_state(tmp_path, monkeypatch):
    client, ws = advising_client(tmp_path)
    calls = []
    monkeypatch.setattr(client, "_advise", lambda ws, allowance, deadline: calls.append(allowance) or dict(largest=5))
    client.advise_edit(ws, 0)
    client.advise_edit(ws, 0)
    client.advise_edit(ws, 1000)                                         # below the quantum: the same allowance
    assert len(calls) == 1
    ws.version = 7                                                       # a commit
    client.advise_edit(ws, 0)
    assert len(calls) == 2
    client.advise_edit(ws, 256 * MiB)                                    # a larger read scratch
    assert len(calls) == 3 and calls[-1] == 256 * MiB
    ws.patch_stored = 12                                                 # a patch enabled or disabled
    client.advise_edit(ws, 256 * MiB)
    assert len(calls) == 4
    monkeypatch.setattr(R, "ADVISE_SECONDS", 0)
    client.advise_edit(ws, 256 * MiB)
    assert len(calls) == 5


# ------------------------------------------------------------------ the advisory admission check

def test_largest_patch_is_the_edge_of_admission(tmp_path):
    settings = resolve(config(tmp_path), env={}, system="Linux", overrides=dict(backend="cpu", floor_gib=20, preferred_gib=25))
    value = dict(forecast=dict(total=3 * GiB, patches=0, weights=GiB), working={"activations": SCRATCH_ADMISSION_QUANTUM}, runner=True)
    active = dict(value, working={}, runner=True)
    sample = dict(available=71 * GiB, total=128 * GiB, pressure=1, tree_footprint=4 * GiB, tree_rss=4 * GiB)
    n, refused = largest_patch(settings, value, sample, active, None, per_weight=PATCH_ADMISSION_BYTES)
    grow = lambda k: dict(value, forecast=dict(value["forecast"], total=3 * GiB + 12 * k, patches=12 * k))  # noqa: E731
    admission(settings, grow(n), sample, active)
    with pytest.raises(RuntimeRefusal):
        admission(settings, grow(n + 1), sample, active)
    assert refused["code"] == "forecast_exceeds_budget"
    assert n == (71 * GiB - 25 * GiB - SCRATCH_ADMISSION_QUANTUM) // 12
    assert largest_patch(settings, value, dict(sample, pressure=2), active)[0] == 0


def edit_of(value, scratch=SCRATCH_ADMISSION_QUANTUM):
    return dict(value, working={"activations": scratch})


def model(tmp_path, name):
    from test_runtime_supervisor import fake_ws
    folder = tmp_path / name
    folder.mkdir()
    ws = fake_ws(folder)
    ws.dir, ws.edit_settings = str(folder), None
    value = descriptor(ws, 8192)
    value["backend"] = "cpu"
    return ws, value


def rows(folder):
    return [json.loads(x) for x in (folder / "watchdog.jsonl").read_text().splitlines()]


def until(check, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        got = check()
        if got:
            return got
        time.sleep(.05)
    raise AssertionError("deadline expired")


def test_the_supervisor_advises_without_binding_and_agrees_with_a_real_bind(tmp_path, monkeypatch):
    from test_runtime_supervisor import inert_supervisor
    monkeypatch.setattr(EL, "SMALL_EDIT", 0)
    with inert_supervisor(tmp_path) as (proc, folder, request):
        ws, value = model(tmp_path, "a")
        edit = edit_of(value)
        cold = request("advise-cold", edit, action="advise")
        assert cold["ok"] and cold["advice"]["largest"] > 0
        assert cold["advice"]["bounds"] == dict(budget=cold["advice"]["bounds"]["budget"], budget_gib=None, ceiling_gib=None,
                                                loaded=False, total=128 * GiB, measured=MiB)   # the inert tree
        assert not (folder / "lease.json").exists()                     # nothing was bound
        assert request("metadata", value)["ok"]
        lease = json.loads((folder / "lease.json").read_text())
        status = json.loads((folder / "status.json").read_text())
        advice = request("advise", edit, action="advise")["advice"]
        n = advice["largest"]
        assert n == cold["advice"]["largest"] == (71 * GiB - 25 * GiB - value["forecast"]["total"]) // 12
        assert advice["available"] == 71 * GiB and advice["preferred"] == 25 * GiB
        assert advice["event"]["code"] == "forecast_exceeds_budget"
        after = json.loads((folder / "status.json").read_text())
        keys = ("state", "event", "operation_id", "forecast", "admission")
        assert {k: after.get(k) for k in keys} == {k: status.get(k) for k in keys} and json.loads((folder / "lease.json").read_text())["binding"] == lease["binding"]
        assert not [r for r in rows(folder) if r.get("event") == "advise" or (r.get("event") == "runtime_event" and "advise" in json.dumps(r))]

        def grow(k):
            f = dict(edit["forecast"], patches=edit["forecast"]["patches"] + 12 * k, total=edit["forecast"]["total"] + 12 * k)
            return dict(edit, forecast=f)
        over = request("over", grow(n + 1))
        assert not over["ok"] and over["event"]["code"] == "forecast_exceeds_budget"
        assert request("edge", grow(n))["ok"]                           # the real admission grants exactly n

        # Through the client: the same answer, and the client keeps no binding or event from it
        client = RuntimeClient(folder, json.loads((folder / "settings.json").read_text()))
        assert request("back", value)["ok"]
        got = client.advise_edit(ws, 0)
        assert got["largest"] == n and client.binding is None and client.last_event is None and not client.reserved
        ws._runtime_client = client
        assert EL.allowed(ws, 1).found["memory"][0] == n
        assert request("release", action="release")["ok"]
    assert proc.returncode == 0


def test_the_supervisors_advice_checks_what_a_grant_checks_first(tmp_path, monkeypatch):
    """advise() binds nothing, checks the model's identity, answers 'no' while sampling fails and while the
    memory ladder acts, as a grant would; none of it changes the reservation."""
    from test_runtime_supervisor import inert_supervisor, write
    monkeypatch.setattr(EL, "SMALL_EDIT", 0)
    with inert_supervisor(tmp_path) as (proc, folder, request):
        ws_a, a = model(tmp_path, "a")
        ws_b, b = model(tmp_path, "b")
        assert request("advise-a", edit_of(a), action="advise")["advice"]["largest"] > 0
        assert request("bind-b", b)["ok"]                              # the advice for a bound nothing

        # Identity: another model while b is reserved, or b's file changed since its descriptor
        other = request("advise-other", edit_of(a), action="advise")["advice"]
        assert other["largest"] == 0 and other["event"]["code"] == "model_changed"
        stale = edit_of(dict(b, stat=[*b["stat"][:3], b["stat"][3] - 1]))
        changed = request("advise-stale", stale, action="advise")["advice"]
        assert changed["largest"] == 0 and changed["event"]["code"] == "model_changed"
        client = RuntimeClient(folder, json.loads((folder / "settings.json").read_text()))
        ws_a._runtime_client = client
        limit = EL.allowed(ws_a, 1)
        assert limit.binding == "watchdog" and limit.weights == 0
        assert "the memory watchdog admits no edit now: Unload the previous model before switching artifacts." in EL.refusal(384, limit)
        assert request("still-b", edit_of(b), action="advise")["advice"]["largest"] > 0

        # Sampling failure: no answer, with the watchdog's own event
        (tmp_path / "sample.json").write_text("not json")
        until(lambda: any(r.get("event") == "sampling_error" for r in rows(folder)))
        failing = request("advise-failing", edit_of(b), action="advise")["advice"]
        assert failing["largest"] is None and failing["event"]["code"] == "telemetry_unavailable"
        write(tmp_path / "sample.json", dict(available=71 * GiB, total=128 * GiB, pressure=1, swap_used=0, swap_out=0, compressed=0))
        n = [0]
        def answered():
            n[0] += 1
            return request(f"advise-recovered-{n[0]}", edit_of(b), action="advise")["advice"]["largest"]
        assert until(answered) > 0

        # The memory ladder: idle below the headroom floor for longer than the persistence window -> unload
        write(tmp_path / "sample.json", dict(available=10 * GiB, total=128 * GiB, pressure=1, swap_used=0, swap_out=0, compressed=0))
        until(lambda: json.loads((folder / "status.json").read_text()).get("state") == "unload", 15)
        write(tmp_path / "sample.json", dict(available=71 * GiB, total=128 * GiB, pressure=1, swap_used=0, swap_out=0, compressed=0))
        laddered = request("advise-ladder", edit_of(b), action="advise")["advice"]
        assert laddered["largest"] == 0 and laddered["event"]["action"] == "unload"
        assert laddered["event"]["code"] == "headroom_floor"
        assert request("release", action="release")["ok"]
    assert proc.returncode == 0


def test_the_advice_counts_the_edits_read_scratch_on_a_loaded_model(tmp_path):
    from test_runtime_supervisor import inert_supervisor, write
    with inert_supervisor(tmp_path) as (proc, folder, request):
        ws, value = model(tmp_path, "a")
        ws._runner = SimpleNamespace(lost=None)                         # the model is loaded (no tensors)
        value = dict(value, runner=True)
        gpu = folder / "watchdog.jsonl.gpu.json"
        stop = threading.Event()
        def fresh():
            while not stop.is_set():
                write(gpu, dict(time=time.time(), active=False, backend="cpu"))
                time.sleep(.3)
        t = threading.Thread(target=fresh)
        t.start()
        try:
            bound = request("load", value)
            assert bound["ok"], bound
            client = RuntimeClient(folder, json.loads((folder / "settings.json").read_text()))
            small = client.advise_edit(ws, 0)
            large = client.advise_edit(ws, 256 * MiB)
            assert small["bounds"]["loaded"] is True and small["largest"] > 0
            assert small["scratch"] == SCRATCH_ADMISSION_QUANTUM and large["scratch"] == 256 * MiB
            assert abs((small["largest"] - large["largest"]) - (256 * MiB - SCRATCH_ADMISSION_QUANTUM) // 12) <= 1
            assert request("release", action="release")["ok"]
        finally:
            stop.set()
            t.join()
    assert proc.returncode == 0


def test_the_supervisor_still_answers_advice_while_it_stops(tmp_path):
    """A stopping supervisor says it is shutting down instead of refusing (a refusal would notify the server)."""
    from test_runtime_supervisor import supervisor_script
    flag, ready = tmp_path / "child-may-exit", tmp_path / "child-ready"
    child = ("import os,signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nsignal.signal(signal.SIGINT,signal.SIG_IGN)\n"
             f"open({str(ready)!r},'w').close()\nend=time.time()+40\nwhile not os.path.exists({str(flag)!r}) and time.time()<end:time.sleep(.05)\n")
    script = supervisor_script(tmp_path, inert=True)
    text = script.read_text()
    assert text.count("'import time;time.sleep(60)'") == 1
    script.write_text(text.replace("'import time;time.sleep(60)'", repr(child)))
    env = {k: v for k, v in os.environ.items() if not k.startswith("WSP_")}
    env.update(HOME=str(tmp_path), XDG_CACHE_HOME=str(tmp_path / "cache"), PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    with (tmp_path / "log").open("w") as log:
        proc = subprocess.Popen([sys.executable, str(script)], env=env, stdout=log, stderr=log, start_new_session=True)
        owned = None
        try:
            folder = until(lambda: next((tmp_path / "cache").rglob("settings.json"), None)).parent
            start = until(lambda: (folder / "watchdog.jsonl").exists() and rows(folder)[0])
            owned = psutil.Process(start["pid"])
            _, value = model(tmp_path, "a")
            until(ready.exists)                                     # the child ignores SIGTERM from here on
            proc.send_signal(signal.SIGTERM)
            until(lambda: any(r.get("event") == "server_signal" for r in rows(folder)))
            R.write(folder / "request.json", dict(id="advise-stopping", action="advise", descriptor=edit_of(value)))
            reply = until(lambda: (lambda d: d if d.get("id") == "advise-stopping" else None)(
                json.loads((folder / "reply.json").read_text())) if (folder / "reply.json").exists() else None)
            assert reply["ok"] and reply["advice"]["largest"] is None
            assert reply["advice"]["event"]["message"] == "Server is shutting down."
            assert not [r for r in rows(folder) if r.get("event") == "runtime_event"]           # nothing notified
            R.write(folder / "request.json", dict(id="bind-stopping", action="bind", descriptor=value))
            refused = until(lambda: (lambda d: d if d.get("id") == "bind-stopping" else None)(
                json.loads((folder / "reply.json").read_text())))
            assert not refused["ok"]                                                         # a grant is refused
            flag.touch()
            proc.wait(timeout=10)
        finally:
            flag.touch()
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            if owned is not None:
                try:
                    owned.kill()
                except psutil.NoSuchProcess:
                    pass


# ------------------------------------------------------------------ the supervisor under faults and load
# (adopted from the round-2 review's test_review_reply_fault.py and test_review_concurrency.py)

def test_an_advice_reply_that_cannot_be_written_changes_no_status(tmp_path):
    from test_runtime_supervisor import inert_supervisor
    with inert_supervisor(tmp_path) as (proc, folder, request):
        ws, value = model(tmp_path, "a")
        assert request("advise-ok", edit_of(value), action="advise")["ok"]
        status = lambda: json.loads((folder / "status.json").read_text())   # noqa: E731
        before = until(lambda: (folder / "status.json").exists() and status())
        (folder / "reply.json").unlink()
        (folder / "reply.json").mkdir()                                     # no reply can be written
        R.write(folder / "request.json", dict(id="advise-fault", action="advise", descriptor=edit_of(value)))
        lost = until(lambda: [r for r in rows(folder) if r.get("event") == "advice_undelivered"], 5)
        assert lost[0]["id"] == "advise-fault" and "reply.json" in lost[0]["error"]
        time.sleep(0.3)
        assert not [r for r in rows(folder) if r.get("event") == "runtime_event"]
        assert status()["state"] == before["state"] and status()["event"] == before["event"]
        # A grant whose reply cannot be written still takes the grant path: rolled back and notified.
        R.write(folder / "request.json", dict(id="bind-fault", action="bind", descriptor=value))
        notified = until(lambda: [r for r in rows(folder) if r.get("event") == "runtime_event"], 5)
        assert notified[0]["record"]["code"] == "runtime_request_invalid"
        assert until(lambda: status()["state"] == "refused", 5)
        (folder / "reply.json").rmdir()
        assert request("dismiss", action="dismiss")["ok"]
        assert request("advise-after", edit_of(value), action="advise")["advice"]["largest"] > 0
    assert proc.returncode == 0


def test_grants_and_advice_interleave_without_crossing(tmp_path, monkeypatch):
    """Real grant exchanges (bind, release) on one thread and advice on two others, one giving up almost at once
    so abandoned advise requests pile up: no grant sees an advisory reply, nothing advisory is logged or kept."""
    from test_runtime_supervisor import inert_supervisor
    monkeypatch.setattr(R, "ADVISE_SECONDS", 0)
    with inert_supervisor(tmp_path) as (proc, folder, request):
        ws, value = model(tmp_path, "a")
        client = RuntimeClient(folder, json.loads((folder / "settings.json").read_text()))
        stop = time.monotonic() + 12
        grants, advices, errors = [], [], []
        def granter():
            while time.monotonic() < stop:
                try:
                    grants.append(client._request("bind", value))
                    grants.append(client._request("release"))
                except Exception as exc:
                    errors.append(repr(exc))
        def adviser(timeout):
            while time.monotonic() < stop:
                advices.append(client._advise(ws, SCRATCH_ADMISSION_QUANTUM, time.monotonic() + timeout))
        threads = [threading.Thread(target=granter), threading.Thread(target=adviser, args=(0.03,)),
                   threading.Thread(target=adviser, args=(1.5,))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors and grants and all(g.get("ok") and "advice" not in g for g in grants)
        answered = [a for a in advices if a.get("largest") is not None]
        assert answered and all(isinstance(a["largest"], int) and "reserved" not in a for a in answered)
        log = rows(folder)
        assert not [r for r in log if r.get("event") in ("advise", "runtime_event", "advice_undelivered")]
        assert client.last_event is None and client.binding is None
    assert proc.returncode == 0
