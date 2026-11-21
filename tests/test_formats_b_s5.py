"""When B-S5 (the IDE's ARA fit from Heretic's captures bit-equal to Heretic's) is asserted: model-free, the rule
in scripts/formats_shared.py only (standard library at its top, so no torch is imported here)."""
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "formats_shared.py"
_SPEC = importlib.util.spec_from_file_location("formats_shared_under_test", _PATH)
FS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FS)

RUN = {"torch": "2.14.1"}


@pytest.mark.parametrize("recorded,kind", [("mps:0", "mps"), ("mps", "mps"), ("cpu", "cpu"), ("cuda:1", "cuda"),
                                           (None, "cpu"), ("", "cpu"), (" MPS:0 ", "mps")])
def test_device_kind(recorded, kind):
    assert FS.device_kind(recorded) == kind


def test_same_torch_and_device_is_asserted():
    applies, why = FS.b_s5_applies("2.14.1", RUN, "mps", "mps:0")
    assert applies is True
    assert "mps" in why
    assert FS.b_s5_applies("2.14.1+cpu", RUN, "cpu", "cpu")[0] is True


def test_missing_run_device_is_cpu():
    assert FS.b_s5_applies("2.14.1", RUN, "cpu", None)[0] is True
    applies, why = FS.b_s5_applies("2.14.1", RUN, "mps", None)
    assert applies is False and "run fitted on cpu" in why


def test_old_call_still_works_for_a_cpu_run():
    assert FS.b_s5_applies("2.14.1", RUN) == FS.b_s5_applies("2.14.1", RUN, "cpu", None)
    assert FS.b_s5_applies("2.14.1", RUN)[0] is True
    assert FS.b_s5_applies("2.15.0", RUN)[0] is False


def test_other_device_is_reported_with_the_reason():
    applies, why = FS.b_s5_applies("2.14.1", RUN, "cpu", "mps:0")
    assert applies is False
    assert "fits on cpu" in why and "fitted on mps" in why
    assert "torch" not in why                                    # the torch matches: only the device is named


def test_other_torch_is_reported_with_the_reason():
    applies, why = FS.b_s5_applies("2.15.0", RUN, "mps", "mps:0")
    assert applies is False
    assert "2.15.0" in why and "2.14.1" in why
    assert "fits on" not in why                                  # the device matches: only the torch is named


def test_both_differ_names_both():
    applies, why = FS.b_s5_applies("2.15.0", RUN, "cpu", "mps:0")
    assert applies is False
    assert "2.15.0" in why and "fitted on mps" in why


def test_no_recorded_torch_is_reported():
    applies, why = FS.b_s5_applies("2.14.1", {}, "cpu", "cpu")
    assert applies is False and "no torch version" in why
    assert FS.b_s5_applies("2.14.1", None, "cpu", "cpu")[0] is False


def test_torch_matches_run_is_unchanged():
    assert FS.torch_matches_run("2.14.1+cpu", {"torch": "2.14.1"})[0] is True
    assert FS.torch_matches_run("2.15.0", {"torch": "2.14.1"})[0] is False


def test_device_matches_run():
    assert FS.device_matches_run("mps", "mps:0")[0] is True
    assert FS.device_matches_run("cpu", "mps:0")[0] is False
    assert FS.device_matches_run("mps", "cpu")[0] is False
