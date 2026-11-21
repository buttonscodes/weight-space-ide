"""scripts/heretic_moe_probe.py: its IDE side and its comparison, model-free (the tiny MoE fixture of layout_fixtures).

The Heretic side needs Heretic's interpreter and is not run here. What Heretic at 64e9991 lists on the fixture under
transformers 5.18.0 (docs/DIRECTIONS.md, "Mixture-of-experts models in Heretic") is written out below as the reference.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
sys.path.insert(0, os.path.dirname(__file__))
from layout_fixtures import EXPERTS, N_LAYERS, tiny_qwen35  # noqa: E402

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "heretic_moe_probe.py"
_SPEC = importlib.util.spec_from_file_location("heretic_moe_probe_under_test", _PATH)
PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(PROBE)

ATTN = ["linear_attn.out_proj"] * 3 + ["self_attn.o_proj"]          # full_attention_interval 4: layer 3 is attention


def heretic_layers(prefix="model.language_model.layers", experts=False):
    """Heretic's get_layer_modules on the fixture, as its run printed it: no routed experts under transformers 5.x."""
    out = []
    for l in range(N_LAYERS):
        mlp = [f"{prefix}.{l}.mlp.experts.{e}.down_proj" for e in range(EXPERTS)] if experts else []
        out.append({"modules": {"attn.o_proj": [f"{prefix}.{l}.{ATTN[l]}"],
                                "mlp.down_proj": mlp + [f"{prefix}.{l}.mlp.shared_expert.down_proj"]}})
    return out


@pytest.fixture(scope="module")
def moe(tmp_path_factory):
    base = tmp_path_factory.mktemp("probe")
    return base, tiny_qwen35(base / "moe", moe=True)


@pytest.fixture(scope="module")
def ide_json(moe):
    base, model = moe
    out = base / "ide.json"
    cwd = os.getcwd()
    os.chdir(Path(__file__).resolve().parents[1])
    try:
        PROBE.ide(model, base / "root", out)
    finally:
        os.chdir(cwd)
    return out


def test_the_ide_side_lists_every_module_in_heretic_order(ide_json):
    got = json.loads(Path(ide_json).read_text())["layers"]
    for l in range(N_LAYERS):
        shared = f"model.layers.{l}.mlp.shared_expert.down_proj"
        experts = [f"model.layers.{l}.mlp.experts.{e}.down_proj" for e in range(EXPERTS)]
        assert got["all"][l] == {"attn.o_proj": [f"model.layers.{l}.{ATTN[l]}"], "mlp.down_proj": experts + [shared]}
        assert got["shared"][l] == {"attn.o_proj": [f"model.layers.{l}.{ATTN[l]}"], "mlp.down_proj": [shared]}


def test_heretic_as_run_matches_shared_only(ide_json, tmp_path, capsys):
    h = tmp_path / "heretic.json"
    h.write_text(json.dumps({"layers": heretic_layers()}))
    out = tmp_path / "compare.json"
    assert PROBE.compare(h, ide_json, out) == 1
    res = json.loads(out.read_text())
    assert res["checks"]["shared"]["pass"] is True and res["checks"]["shared"]["differences"] == []
    assert res["checks"]["all"]["pass"] is False and res["pass"] is False
    assert (res["checks"]["all"]["modules_heretic"], res["checks"]["all"]["modules_ide"]) == (8, 24)
    assert [d["layer"] for d in res["checks"]["all"]["differences"]] == list(range(N_LAYERS))
    assert {d["component"] for d in res["checks"]["all"]["differences"]} == {"mlp.down_proj"}
    printed = capsys.readouterr().out
    assert "FAIL  (i) same modules in the same order, expert_modules='all'" in printed
    assert "PASS  (i) same modules in the same order, expert_modules='shared'" in printed


def test_heretic_listing_experts_would_match_all_and_order_counts(ide_json, tmp_path):
    h = tmp_path / "heretic.json"
    h.write_text(json.dumps({"layers": heretic_layers(experts=True)}))
    out = tmp_path / "compare.json"
    assert PROBE.compare(h, ide_json, out) == 1                      # "shared" now differs
    res = json.loads(out.read_text())
    assert res["checks"]["all"]["pass"] is True and res["checks"]["shared"]["pass"] is False
    swapped = heretic_layers(experts=True)
    m = swapped[2]["modules"]["mlp.down_proj"]
    m[0], m[1] = m[1], m[0]                                          # the same modules, another order
    h.write_text(json.dumps({"layers": swapped}))
    PROBE.compare(h, ide_json, out)
    res = json.loads(out.read_text())
    assert res["checks"]["all"]["pass"] is False
    assert [(d["layer"], d["component"]) for d in res["checks"]["all"]["differences"]] == [(2, "mlp.down_proj")]


def test_a_component_only_one_side_lists_fails(ide_json, tmp_path):
    layers = heretic_layers()
    del layers[0]["modules"]["mlp.down_proj"]                        # as Heretic lists a Qwen3-MoE layer
    h = tmp_path / "heretic.json"
    h.write_text(json.dumps({"layers": layers}))
    out = tmp_path / "compare.json"
    assert PROBE.compare(h, ide_json, out) == 1
    res = json.loads(out.read_text())
    assert res["checks"]["shared"]["pass"] is False
    assert res["checks"]["shared"]["differences"] == [
        {"layer": 0, "component": "mlp.down_proj", "heretic": [], "ide": ["model.layers.0.mlp.shared_expert.down_proj"]}]


def test_a_missing_layer_fails(ide_json, tmp_path):
    h = tmp_path / "heretic.json"
    h.write_text(json.dumps({"layers": heretic_layers()[:-1]}))
    out = tmp_path / "compare.json"
    assert PROBE.compare(h, ide_json, out) == 1
    assert json.loads(out.read_text())["checks"]["shared"]["pass"] is False


def test_the_comparison_scripts_take_expert_modules():
    for name in ("heretic_compare", "heretic_compare_ara"):
        spec = importlib.util.spec_from_file_location(f"{name}_under_test", _PATH.parent / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        a, runs = mod.parse(["snap", "root", "out.json", "x=run"])
        assert a.expert_modules == "all" and runs == {"x": "run"}
        a, _ = mod.parse(["--expert-modules", "shared", "snap", "root", "out.json", "x=run"])
        assert a.expert_modules == "shared"
        with pytest.raises(SystemExit):
            mod.parse(["--expert-modules", "none", "snap", "root", "out.json", "x=run"])


def _compare_module():
    spec = importlib.util.spec_from_file_location("heretic_compare_helpers", _PATH.parent / "heretic_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_ablation_comparison_passes_expert_modules_to_the_ide_edit():
    HC = _compare_module()
    trial = {"direction_index": 1.5, "weight_distributions": {}}
    assert HC.ide_edit_params(trial, "full", True, "all") == {
        "direction": "oracle", "direction_index": 1.5, "weight_distributions": {}, "orthogonalize_direction": True,
        "row_normalization": "full", "full_normalization_lora_rank": 3}
    assert HC.ide_edit_params(trial, "none", True, "shared")["expert_modules"] == "shared"


def test_the_ablation_comparison_counts_the_ide_modules_of_its_expert_modules(moe):
    from wsp.session import Session
    base, model = moe
    ws = Session(str(base / "root-c41")).open(model, log=lambda *a, **k: None)
    HC = _compare_module()
    pre = "model.language_model.layers"
    attn = {f"{pre}.{l}.{ATTN[l]}.weight" for l in range(N_LAYERS)}
    shared = {f"{pre}.{l}.mlp.shared_expert.down_proj.weight" for l in range(N_LAYERS)}
    experts = {f"{pre}.{l}.mlp.experts.{e}.down_proj.weight" for l in range(N_LAYERS) for e in range(EXPERTS)}
    assert HC.ide_module_tensors(ws, "shared") == attn | shared
    assert HC.ide_module_tensors(ws, "all") == HC.ide_module_tensors(ws) == attn | shared | experts
