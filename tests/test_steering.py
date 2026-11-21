"""Steering and probes on a tiny random model (Config + init_params, no training), in process.

Every expectation is computed independently: from `wsp.model.forward` over a copy of the
parameters, or from captures of an unsteered run.  No checkpoint is loaded.
"""
import hashlib
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from wsp import probes as P
from wsp import steering as S
from wsp.model import Config, forward, init_params
from wsp.server import create_app
from wsp.session import Session
from wsp.tokenizer import VocabTokenizer
from wsp.workspace import Workspace

CFG = dict(d_vocab=26, n_ctx=16, d_model=24, n_layers=2, n_heads=2, d_mlp=40, name="rnd")


def make_ws(directory, params=None):
    cfg = Config(**CFG)
    tok = VocabTokenizer(list("abcdefghijklmnopqrstuvwxyz"), "char")
    return Workspace.create(str(directory), cfg, tok, params if params is not None else init_params(cfg, seed=1, std=0.3))


@pytest.fixture()
def ws(tmp_path):
    return make_ws(tmp_path / "rnd")


@pytest.fixture()
def sess(tmp_path):
    make_ws(tmp_path / "rnd")
    s = Session(str(tmp_path))
    s.open("rnd")
    return s


def post(name, l):
    return f"blocks.{l}.mlp.hook_post"


def capture(ws, ids, names, variant="current", hooks=None):
    _, cache = ws.runner.run(ids, variant, names=list(names), hooks=hooks)
    return cache


def independent_p50(ws, variant="base"):
    ids = S.reference_ids(ws)
    cache = {}
    names = [f"blocks.{l}.hook_resid_post" for l in range(ws.cfg.n_layers)]
    forward(ws.runner.params(variant), ws.cfg, torch.tensor(ids), cache=cache, names=names)
    return {str(l): float(cache[n][0, 1:].norm(dim=-1).median()) for l, n in enumerate(names)}


# ------------------------------------------------------------------ the push itself
def test_neutral_steering_is_exactly_the_unsteered_model(ws):
    r, ids = ws.runner, ws.encode("abcab")
    spec = [{"addr": "L1N3", "mode": "add", "dose": 0, "unit": "raw"}, {"addr": "L0N2", "mode": "add", "dose": 0},
            {"addr": "L1N5", "mode": "mul", "dose": 1}, {"addr": "blocks.0.mlp", "mode": "mul", "dose": 1},
            {"addr": "L0H1", "mode": "mul", "dose": 1}]
    wire, info = S.prepare(ws, spec)
    hooks = S.hooks_for(ws, wire)
    assert set(hooks) == {"blocks.0.mlp.hook_post", "blocks.1.mlp.hook_post", "blocks.0.attn.hook_z"}  # the path still runs
    assert torch.equal(r.run(ids, hooks=hooks)[0], r.run(ids)[0])
    assert r.generate(ids, max_new_tokens=8, steer=wire) == r.generate(ids, max_new_tokens=8)
    assert r.generate(ids, max_new_tokens=8, temperature=1.0, seed=3, steer=spec) == r.generate(ids, max_new_tokens=8, temperature=1.0, seed=3)
    # An empty specification is no hooks at all.
    assert S.hooks_for(ws, []) == {} and S.hooks_for(ws, None) == {}
    assert r.generate(ids, max_new_tokens=6, steer=[]) == r.generate(ids, max_new_tokens=6)


@pytest.mark.parametrize("spec,layer", [
    ([{"addr": "L0N7", "mode": "add", "dose": 0.5}], 0),
    ([{"addr": "L1N3", "mode": "add", "dose": 1.5, "unit": "raw"}, {"addr": "L1N9", "mode": "add", "dose": -0.25}], 1),
])
def test_an_add_push_changes_hook_post_by_exactly_its_constant_at_every_position(ws, spec, layer):
    r, prompt = ws.runner, ws.encode("hello")
    wire, _ = S.prepare(ws, spec)
    c = S.resolve(ws, wire)[layer].c
    pushed_neurons = sorted({int(e["addr"][3:]) for e in spec})
    assert c.nonzero().flatten().tolist() == pushed_neurons
    new = r.generate(prompt, max_new_tokens=8, steer=wire)
    full = prompt + new
    name = post("", layer)
    plain = capture(ws, full, [name])[name]
    pushed = capture(ws, full, [name], hooks=S.hooks_for(ws, wire))[name]
    assert torch.equal(pushed, plain + c)                      # every position: prompt and generated
    # Generation applied the push at every step: each new token is the steered model's argmax.
    logits = r.run(full, hooks=S.hooks_for(ws, wire))[0][0]
    assert [int(logits[len(prompt) - 1 + i].argmax()) for i in range(len(new))] == new


def test_a_large_push_changes_what_is_generated(ws):
    r, ids = ws.runner, ws.encode("abc")
    plain = r.generate(ids, max_new_tokens=10)
    assert any(r.generate(ids, max_new_tokens=10, steer=[{"addr": f"L1N{n}", "mode": "add", "dose": 40, "unit": "raw"}]) != plain
               for n in range(8))


def test_the_resid_unit_is_dose_times_p50_over_the_output_column_norm(ws):
    wire, info = S.prepare(ws, [{"addr": "L1 N3", "mode": "add", "dose": 0.75}])   # neuron-explorer's spaced form
    p50 = independent_p50(ws)
    assert wire["p50"] == {"1": pytest.approx(p50["1"], rel=1e-6)}
    norm = float(ws.runner.params()["blocks.1.mlp.W_out"][3].double().norm())
    c = S.resolve(ws, wire)[1].c
    assert float(c[3]) == pytest.approx(0.75 * p50["1"] / norm, rel=1e-6) and int(torch.count_nonzero(c)) == 1
    d = S.describe(ws, wire, info)["entries"][0]
    assert d["norm"] == pytest.approx(norm) and d["value"] == pytest.approx(0.75 * p50["1"] / norm, rel=1e-6)


def test_calibration_excludes_position_zero_and_takes_the_lower_middle_value(ws, monkeypatch):
    """Residual norms are set per position, so each mistake gives a different number:
    layer 0  [100, 1, 2, 3, 4]    -> 2  (with position 0: 3; mean of the middle: 2.5; upper middle: 3)
    layer 1  [0.1, 5, 6, 7, 8, 9] -> 7  (with position 0: 6)"""
    norms = {0: [100.0, 1.0, 2.0, 3.0, 4.0], 1: [0.1, 5.0, 6.0, 7.0, 8.0, 9.0]}
    seen = []

    class Fake:
        lost = None

        def available(self, name):
            return True

        def run(self, ids, variant, extra, hooks):
            seen.append(variant)
            for l, values in norms.items():
                x = torch.zeros(1, len(values), ws.cfg.d_model)
                x[0, :, l] = torch.tensor(values)                 # a residual whose norm at position t is values[t]
                assert hooks[f"blocks.{l}.hook_resid_post"](x, f"blocks.{l}.hook_resid_post") is None
            return torch.zeros(1), {}
    monkeypatch.setattr(ws, "_runner", Fake())
    out = S.calibrate_job(ws, [1, 2, 3, 4, 5])
    assert out == {"p50": {"0": 2.0, "1": 7.0}, "tokens": 5} and seen == ["base"]


def test_a_malformed_wire_form_is_always_a_steering_error(ws):
    """Layer keys are decimal ASCII without leading zeros; nothing but SteeringError leaves check_wire."""
    ok = {"add_resid": [[3], [0.5]]}
    for layers, p50, message in [
        ({"01": {"mul": [[3], [1.0]]}}, {}, "does not exist"),
        ({"١": {"mul": [[3], [1.0]]}}, {}, "does not exist"),          # Arabic-Indic one: int() would accept it
        ({"¹": {"mul": [[3], [1.0]]}}, {}, "does not exist"),          # superscript one: isdigit() would accept it
        ({" 1": {"mul": [[3], [1.0]]}}, {}, "does not exist"),
        ({"+1": {"mul": [[3], [1.0]]}}, {}, "does not exist"),
        ({"1": {"mul": [[], []]}}, {}, "is empty"),
        ({"1": {"add_resid": [[], []]}}, {}, "is empty"),
        ({"1": ok}, {"01": 2.0}, "is not a layer"),
        ({"1": ok}, {"1": "2"}, "must be a number"),
        ({"1": ok}, {"1": True}, "must be a number"),
        ({"1": ok}, {"1": 10 ** 400}, "too large"),
        ({"1": ok}, {"1": float("inf")}, "finite number above 0"),
        ({"1": {"mul": [[3], [10 ** 400]]}}, {}, "at most"),
        ({"1": {"mul": [["3"], [1.0]]}}, {}, "equal length"),
        ({"1": {"mul": [[True], [1.0]]}}, {}, "equal length"),
        ({"1": {"mul": [[3], [1.0], [2.0]]}}, {}, "equal length"),
        ({"1": {"mul": {"3": 1.0}}}, {}, "equal length"),
        ({"1": {"mlp_mul": "2"}}, {}, "must be a number"),
        ({"1": []}, {}, "must be an object"),
    ]:
        with pytest.raises(S.SteeringError, match=message):
            S.check_wire(ws, {"version": 1, "layers": layers, "p50": p50})
    for wire in [None, [], {"version": 1, "layers": []}, {"version": 1, "layers": {}, "p50": []},
                 {"version": 1, "layers": {}, "calibration": 7}, {"version": 1, "layers": {1: {}}},
                 {"version": 1, "layers": {}, "p50": {1: 2.0}}]:
        with pytest.raises(S.SteeringError):
            S.check_wire(ws, wire)
    # Anything else that goes wrong inside the check comes back as a SteeringError too.
    class Odd(dict):
        def items(self):
            raise KeyError("odd")
    with pytest.raises(S.SteeringError, match="malformed"):
        S.check_wire(ws, {"version": 1, "layers": Odd()})
    assert S.check_wire(ws, {"version": 1, "layers": {"0": {"mul": [[3], [1.0]]}, "1": ok}, "p50": {"1": 2.0}})


def test_mul_zero_equals_zeroing_the_neurons_output_column_in_a_copy(ws):
    r, ids = ws.runner, ws.encode("abcdefg")
    p = {k: v.clone() for k, v in r.params().items()}
    p["blocks.0.mlp.W_out"][7] = 0
    expect = forward(p, ws.cfg, torch.tensor(ids))
    got = r.run(ids, hooks=S.hooks_for(ws, [{"addr": "L0N7", "mode": "mul", "dose": 0}]))[0]
    assert torch.allclose(got, expect, rtol=0, atol=1e-6) and not torch.allclose(got, r.run(ids)[0], atol=1e-4)
    assert ws.patches == [] and float(r.params()["blocks.0.mlp.W_out"][7].abs().sum()) > 0   # no weight changed


def test_mlp_and_head_multipliers(ws):
    r, ids = ws.runner, ws.encode("abcdefg")
    t = torch.tensor(ids)

    def with_params(change):
        p = {k: v.clone() for k, v in r.params().items()}
        change(p)
        return forward(p, ws.cfg, t)

    def steered(spec):
        return r.run(ids, hooks=S.hooks_for(ws, spec))[0]
    cases = [
        ([{"addr": "blocks.1.mlp", "mode": "mul", "dose": 0}], lambda p: p["blocks.1.mlp.W_out"].zero_()),
        ([{"addr": "blocks.0.mlp", "mode": "mul", "dose": 0.5}], lambda p: p["blocks.0.mlp.W_out"].mul_(0.5)),
        # whole-MLP and neuron multipliers compose: 0.5 * 2 leaves neuron 3 as it was
        ([{"addr": "blocks.0.mlp", "mode": "mul", "dose": 0.5}, {"addr": "L0N3", "mode": "mul", "dose": 2}],
         lambda p: (p["blocks.0.mlp.W_out"].mul_(0.5), p["blocks.0.mlp.W_out"][3].mul_(2))),
        ([{"addr": "L0H1", "mode": "mul", "dose": 0}], lambda p: p["blocks.0.attn.W_O"][1].zero_()),
        ([{"addr": "blocks.1.attn.head.0", "mode": "mul", "dose": 3}], lambda p: p["blocks.1.attn.W_O"][0].mul_(3)),
        # repeated multipliers multiply
        ([{"addr": "L1N4", "mode": "mul", "dose": 2}, {"addr": "L1N4", "mode": "mul", "dose": 3}],
         lambda p: p["blocks.1.mlp.W_out"][4].mul_(6)),
    ]
    for spec, change in cases:
        assert torch.allclose(steered(spec), with_params(change), rtol=0, atol=1e-5), spec
    wire, _ = S.normalize(ws, [{"addr": "L1N4", "mode": "add", "dose": 1, "unit": "raw"}, {"addr": "L1N4", "mode": "add", "dose": 2, "unit": "raw"},
                               {"addr": "L1N4", "mode": "mul", "dose": 2}, {"addr": "L0H1", "mode": "mul", "dose": 0.5},
                               {"addr": "L0H1", "mode": "mul", "dose": 4}])
    assert wire["layers"] == {"0": {"head_mul": [[1], [2.0]]}, "1": {"add_raw": [[4], [3.0]], "mul": [[4], [2.0]]}}
    # multiply first, then add: x * 2 + 3 on neuron 4
    name = "blocks.1.mlp.hook_post"
    plain = capture(ws, ids, [name])[name]
    pushed = capture(ws, ids, [name], hooks=S.hooks_for(ws, {"version": 1, "layers": {"1": wire["layers"]["1"]}}))[name]
    expect = plain.clone()
    expect[..., 4] = plain[..., 4] * 2 + 3
    assert torch.equal(pushed, expect)


def test_groups_expand_to_their_neurons(ws):
    from wsp import selections
    ws.set_group("outer", ["L0N1"])
    ws.set_group("inner", ["L1N9", "group:outer"])
    # set_group now refuses a cycle (selections branch); one written by an older version is still read, each group once
    ws.groups["outer"] = {"members": ["L0N1", "blocks.1.mlp.neurons.2-4", "L0H1", "blocks.0.mlp", "group:inner"], "note": ""}
    wire, entries = S.normalize(ws, [{"addr": "group:outer", "mode": "add", "dose": 0.5, "unit": "raw"},
                                     {"addr": "group:inner", "mode": "mul", "dose": 2}])
    assert wire["layers"] == {"0": {"add_raw": [[1], [0.5]], "mul": [[1], [2.0]]},
                              "1": {"add_raw": [[2, 3, 4, 9], [0.5] * 4], "mul": [[2, 3, 4, 9], [2.0] * 4]}}
    assert entries[0]["neurons"] == 5 and entries[0]["layers"] == {"0": 1, "1": 4}
    assert entries[0]["targets"] == ["L0N1", "L1N2", "L1N3", "L1N4", "L1N9"]
    # the same neurons as the selections module's neurons_of (contract C2)
    assert S.group_neurons(ws, "outer")[0] == selections.neurons_of(ws, "outer")
    # what a group holds that is not pushed is reported, not dropped silently: here a head and the MLP's tensors
    for e in entries:                                   # inner reaches outer, so both hold the head and the MLP
        assert e["skipped"]["heads"] == 1 and e["skipped"]["slices"] > 0 and e["skipped"]["unresolved"] == 0
        assert "members" not in e["skipped"]
    ws.set_group("heads", ["L0H1", "blocks.0.mlp"])
    with pytest.raises(S.SteeringError, match=r"steer entry 1 \(group:heads\): group 'heads' has no neurons to push "
                                              r"\(it holds 1 head, \d+ tensor slices\)"):
        S.normalize(ws, [{"addr": "group:heads", "mode": "add", "dose": 1}])
    with pytest.raises(S.SteeringError, match=r"steer entry 1 \(group:nope\): no group named 'nope'"):
        S.normalize(ws, [{"addr": "group:nope", "mode": "add", "dose": 1}])


def test_a_group_member_that_no_longer_resolves_is_counted_in_the_report(ws):
    """A deleted inner selection: the outer one still steers its other neurons, and the report says what was left out."""
    ws.set_group("gone", ["L1N5"])
    ws.set_group("sel", ["L0N2", "blocks.1.mlp.neurons.6-7", "group:gone"])
    ws.delete_group("gone")
    wire, entries = S.normalize(ws, [{"addr": "group:sel", "mode": "add", "dose": 1.5, "unit": "raw"}])
    assert wire["layers"] == {"0": {"add_raw": [[2], [1.5]]}, "1": {"add_raw": [[6, 7], [1.5, 1.5]]}}
    e = entries[0]
    assert e["neurons"] == 3
    sk = e["skipped"]
    assert (sk["unresolved"], sk["heads"], sk["slices"]) == (1, 0, 0)
    assert [m["member"] for m in sk["members"]] == ["group:gone"] and "gone" in sk["members"][0]["reason"]
    assert S.skipped_text(sk) == "1 member that no longer resolves"
    # describe (the /api/steer/resolve and Run/Chat report) carries it too
    d = S.describe(ws, wire, {"entries": entries, "calibration": {}})
    assert d["entries"][0]["skipped"]["unresolved"] == 1
    # nothing skipped: no key at all
    ws.set_group("plain", ["L0N2", "L1N3"])
    assert "skipped" not in S.normalize(ws, [{"addr": "group:plain", "mode": "mul", "dose": 0}])[1][0]


def fake_hf(types_=None, o_proj=True):
    def layer(l, attn):
        mods = {"layer": f"model.layers.{l}", "attn": f"model.layers.{l}.{attn}", "mlp_out": f"model.layers.{l}.mlp.down_proj"}
        if o_proj and attn == "self_attn":
            mods["o_proj"] = f"model.layers.{l}.self_attn.o_proj"
        return SimpleNamespace(modules=mods)
    struct = SimpleNamespace(n_layers=3, n_heads=4, layers=[layer(0, "self_attn"), layer(1, "linear_attn"), layer(2, "self_attn")],
                             has_heads=lambda l: True, has_neurons=lambda l: True, mlp_width=lambda l: 8)
    return SimpleNamespace(kind="hf", struct=struct, hf_config={"layer_types": types_} if types_ else {}, resolve=lambda a: [])


def test_head_targets_need_a_hook_z_and_are_refused_on_linear_attention():
    w = fake_hf()
    assert S.target(w, "L0H3") == ("head", 0, 3) and S.target(w, "L1N2") == ("neurons", [(1, 2)])
    with pytest.raises(S.SteeringError, match="layer 1 uses linear attention"):
        S.normalize(w, [{"addr": "L1H0", "mode": "mul", "dose": 0}])
    with pytest.raises(S.SteeringError, match="layer 2 uses linear attention"):
        S.target(fake_hf(["full_attention", "full_attention", "linear_attention"]), "L2H0")
    with pytest.raises(S.SteeringError, match=r"steer entry 1 \(L0H1\): layer 0 has no attn.hook_z hook point"):
        S.normalize(fake_hf(o_proj=False), [{"addr": "L0H1", "mode": "mul", "dose": 0}])
    with pytest.raises(S.SteeringError, match="linear attention"):
        S.check_wire(w, {"version": 1, "layers": {"1": {"head_mul": [[0], [0.0]]}}})


@pytest.mark.parametrize("spec,message", [
    ({"addr": "L0N1"}, "steer must be a list"),
    (["L0N1"], r"steer entry 1: must be an object"),
    ([{"mode": "add", "dose": 1}], r"steer entry 1: addr must be a nonempty address"),
    ([{"addr": "blocks.9.wat", "mode": "add", "dose": 1}], r"steer entry 1 \(blocks.9.wat\): unknown address"),
    ([{"addr": "embed", "mode": "mul", "dose": 1}], r"steer entry 1 \(embed\): cannot be steered"),
    ([{"addr": "L0N1", "mode": "mul", "dose": 1}, {"addr": "L0H1", "mode": "add", "dose": 1}],
     r"steer entry 2 \(L0H1\): add pushes neurons only"),
    ([{"addr": "blocks.0.mlp", "mode": "add", "dose": 1}], r"\(blocks.0.mlp\): add pushes neurons only"),
    ([{"addr": "L0N1", "mode": "add", "dose": float("nan")}], r"\(L0N1\): dose must be a finite number"),
    ([{"addr": "L0N1", "mode": "add", "dose": float("inf")}], r"\(L0N1\): dose must be a finite number"),
    ([{"addr": "L0N1", "mode": "mul", "dose": "2"}], r"\(L0N1\): dose must be a number"),
    ([{"addr": "L0N1", "mode": "mul", "dose": True}], r"\(L0N1\): dose must be a number"),
    ([{"addr": "L0N1", "mode": "mul", "dose": 1e7}], r"\(L0N1\): dose 1e\+07 is larger than"),
    ([{"addr": "L0N1", "mode": "add"}], r"\(L0N1\): dose is missing"),
    ([{"addr": "L0N1", "mode": "push", "dose": 1}], r"\(L0N1\): mode must be 'add' or 'mul'"),
    ([{"addr": "L0N1", "mode": "mul", "dose": 1, "unit": "raw"}], r"\(L0N1\): unit applies to add only"),
    ([{"addr": "L0N1", "mode": "add", "dose": 1, "unit": "sigma"}], r"\(L0N1\): unit must be 'resid' or 'raw'"),
    ([{"addr": "L0N1", "mode": "add", "dose": 1, "layer": 3}], r"\(L0N1\): unknown field layer"),
    ([{"addr": "L0N40", "mode": "add", "dose": 1}], r"\(L0N40\): neuron 40 does not exist \(layer 0 has 40\)"),
    ([{"addr": "L5N1", "mode": "add", "dose": 1}], r"\(L5N1\): layer 5 does not exist \(the model has 2\)"),
    ([{"addr": "L0H2", "mode": "mul", "dose": 1}], r"\(L0H2\): head 2 does not exist"),
    ([{"addr": "blocks.0.mlp.neurons.5-2", "mode": "mul", "dose": 1}], r"neuron range 5-2 is empty"),
    ([{"addr": "L0N1", "mode": "mul", "dose": 1}] * 257, r"steer has 257 entries; at most 256"),
])
def test_every_validation_error_names_the_entry(ws, spec, message):
    with pytest.raises(S.SteeringError, match=message):
        S.normalize(ws, spec)


def test_a_zero_output_column_cannot_take_a_resid_dose(tmp_path):
    cfg = Config(**CFG)
    p = init_params(cfg, seed=1, std=0.3)
    p["blocks.0.mlp.W_out"][5] = 0
    w = make_ws(tmp_path / "z", p)
    wire, info = S.prepare(w, [{"addr": "L0N5", "mode": "add", "dose": 0.5}])
    with pytest.raises(S.SteeringError, match=r"L0N5: its output weights have norm 0, so a resid dose has no size; use unit raw"):
        S.hooks_for(w, wire)
    with pytest.raises(S.SteeringError, match=r"steer entry 1 \(L0N5\): L0N5 has output weights of norm 0"):
        S.describe(w, wire, info)
    assert S.hooks_for(w, [{"addr": "L0N5", "mode": "add", "dose": 0.5, "unit": "raw"}])     # raw is fine


def test_only_the_current_variant_is_steered(ws):
    r, ids = ws.runner, ws.encode("abc")
    with pytest.raises(S.SteeringError, match="the base model is never steered"):
        S.hooks_for(ws, [{"addr": "L0N1", "mode": "mul", "dose": 0}], "base")
    with pytest.raises(S.SteeringError, match="the base model is never steered"):
        r.generate(ids, "base", max_new_tokens=2, steer=[{"addr": "L0N1", "mode": "mul", "dose": 0}])
    assert r.generate(ids, "base", max_new_tokens=3, steer=[]) == r.generate(ids, "base", max_new_tokens=3)


def test_the_wire_form_is_checked_where_it_is_applied(ws):
    good = {"version": 1, "layers": {"1": {"add_resid": [[3], [0.5]]}}, "p50": {"1": 2.0}}
    assert S.check_wire(ws, good) is good
    for bad, message in [
        ({"version": 2, "layers": {}}, "version 1"),
        ({"version": 1, "layers": {}, "hooks": 1}, "unknown fields"),
        ({"version": 1, "layers": {"7": {}}}, "layer '7' does not exist"),
        ({"version": 1, "layers": {"1": {"add_resid": [[3], [0.5]]}}}, "resid push but no p50"),
        ({"version": 1, "layers": {"1": {"add_resid": [[3], [0.5]]}}, "p50": {"1": 0.0}}, "finite number above 0"),
        ({"version": 1, "layers": {"1": {"mul": [[3, 3], [1.0, 2.0]]}}}, "repeats an index"),
        ({"version": 1, "layers": {"1": {"mul": [[40], [1.0]]}}}, "out of range"),
        ({"version": 1, "layers": {"1": {"mul": [[3], [float("nan")]]}}}, "finite number"),
        ({"version": 1, "layers": {"1": {"mul": [[3], []]}}}, "equal length"),
        ({"version": 1, "layers": {"0": {"head_mul": [[2], [1.0]]}}}, "head index out of range"),
    ]:
        with pytest.raises(S.SteeringError, match=message):
            S.check_wire(ws, bad)


# ------------------------------------------------------------------ calibration
def test_calibration_is_cached_invalidated_and_measured_on_base(sess, monkeypatch):
    ws = sess.ws
    calls = []
    real = S.calibrate_job
    monkeypatch.setattr(S, "calibrate_job", lambda w, ids: calls.append(len(ids)) or real(w, ids))
    # unit raw and mul need no calibration
    _, info = S.prepare(ws, [{"addr": "L0N1", "mode": "add", "dose": 1, "unit": "raw"}, {"addr": "L0N2", "mode": "mul", "dose": 0}])
    assert info["calibration"] == {"needed": False} and not calls
    assert not os.path.exists(os.path.join(ws.dir, S.CALIBRATION_FILE))
    # not calibrated and not allowed to run: reported, nothing measured, the wire cannot be applied
    wire, info = S.prepare(ws, [{"addr": "L0N1", "mode": "add", "dose": 1}], calibrate=False)
    assert info["calibration"]["needed"] and not info["calibration"]["cached"] and not info["calibration"]["ran"] and "p50" not in wire
    with pytest.raises(S.SteeringError, match="resid push but no p50"):
        S.hooks_for(ws, wire)
    # an edit makes current differ from base; the calibration is of base
    sess.apply('n = neuron(0, 5)\nn.w_out = 5 * unit(U("d"))', message="test edit")
    base, current = independent_p50(ws, "base"), independent_p50(ws, "current")
    assert base["1"] != pytest.approx(current["1"], rel=1e-3)
    events = []
    wire, info = S.prepare(ws, [{"addr": "L0N5", "mode": "add", "dose": 0.5}, {"addr": "L1N2", "mode": "add", "dose": 1}],
                           on_event=events.append)
    assert info["calibration"]["ran"] and not info["calibration"]["cached"] and calls == [ws.cfg.n_ctx]
    assert [e["state"] for e in events] == ["calibrating", "calibrated"]
    assert wire["p50"] == {"0": pytest.approx(base["0"], rel=1e-6), "1": pytest.approx(base["1"], rel=1e-6)}
    # ...while the column norm is that of the variant run: neuron 0.5 was just given an output of norm 5
    assert float(S.resolve(ws, wire)[0].c[5]) == pytest.approx(0.5 * base["0"] / 5, rel=1e-5)
    data = json.load(open(os.path.join(ws.dir, S.CALIBRATION_FILE)))
    assert list(data) == [info["calibration"]["key"]] and data[wire["calibration"]]["tokens"] == ws.cfg.n_ctx
    # cached: no second forward pass
    again, info2 = S.prepare(ws, [{"addr": "L1N2", "mode": "add", "dose": 1}])
    assert info2["calibration"]["cached"] and not info2["calibration"]["ran"] and len(calls) == 1
    assert again["p50"] == {"1": wire["p50"]["1"]}
    # a different reference text, or a different base file, is measured again
    monkeypatch.setattr(S, "REFERENCE_TEXT", S.REFERENCE_TEXT + " One more sentence.")
    assert S.prepare(ws, [{"addr": "L1N2", "mode": "add", "dose": 1}])[1]["calibration"]["ran"] and len(calls) == 2
    path = os.path.join(ws.dir, "base.safetensors")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    _, info3 = S.prepare(ws, [{"addr": "L1N2", "mode": "add", "dose": 1}])
    assert info3["calibration"]["ran"] and len(calls) == 3 and len(json.load(open(os.path.join(ws.dir, S.CALIBRATION_FILE)))) == 3
    # an unreadable cache is measured again
    open(os.path.join(ws.dir, S.CALIBRATION_FILE), "w").write("{not json")
    assert S.prepare(ws, [{"addr": "L1N2", "mode": "add", "dose": 1}])[1]["calibration"]["ran"] and len(calls) == 4


def test_column_norms_read_runs_of_nearby_neurons_and_cache_per_version(sess, monkeypatch):
    ws = sess.ws
    reads = []
    real = ws.read
    monkeypatch.setattr(ws, "read", lambda name, index=(), variant="current", extra=None, cache=True:
                        reads.append((name, index)) or real(name, index, variant, extra, cache))
    norms, cost = S.column_norms(ws, 1, [9, 2, 3, 39])
    assert reads == [("blocks.1.mlp.W_out", (range(2, 40), range(0, 24)))]                # one run 2..39, nothing else
    assert cost["reads"] == 1 and cost["elements"] == 38 * 24
    W = ws.runner.params()["blocks.1.mlp.W_out"].double()
    assert torch.allclose(norms, W[[9, 2, 3, 39]].norm(dim=1))
    monkeypatch.setattr(S, "READ_GAP", 4)
    _, cost = S.column_norms(ws, 0, [0, 1, 20, 21])
    assert cost["reads"] == 2 and cost["elements"] == 4 * 24
    _, cost = S.column_norms(ws, 0, [0, 21])
    assert cost["reads"] == 0 and cost["cached"] == 2
    sess.apply('n = neuron(0, 0)\nn.w_out = 3 * unit(U("a"))', message="edit")          # a new version clears the cache
    norms, cost = S.column_norms(ws, 0, [0])
    assert cost["reads"] == 1 and float(norms[0]) == pytest.approx(3.0, rel=1e-5)


NEURON_EXPLORER_REFERENCE_SHA256 = "afb86522116b9ccd10233826503dad0992dd0283f04ee2a15dabf5f0e9e1ce54"


def test_the_reference_text_is_neuron_explorers_tokenized_as_it_does():
    # The sha256 of REFERENCE_TEXT in neuron-explorer explorer/runtime.py (read from that file when this was written).
    assert hashlib.sha256(S.REFERENCE_TEXT.encode()).hexdigest() == NEURON_EXPLORER_REFERENCE_SHA256
    assert "MIT" in S.REFERENCE_SOURCE and "Copyright (c) 2026 lyramakesmusic" in S.REFERENCE_SOURCE
    # A Hugging Face tokenizer gets its own special tokens (transformers' default add_special_tokens=True), even
    # when the workspace encodes prompts without them; nothing is cut when the context is long enough.
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from wsp.tokenizer import BPETokenizer
    words = sorted(set(S.REFERENCE_TEXT.replace("(", " ").split()))
    tk = Tokenizer(models.WordLevel({"[BOS]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(words)}}, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.post_processor = processors.TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 0)])
    tok = BPETokenizer(tk.to_str(), special=False)
    assert tok.encode("The history")[0] != 0
    ids, how = S.reference_tokens(SimpleNamespace(tok=tok, struct=SimpleNamespace(n_ctx=4096)))
    assert ids == tk.encode(S.REFERENCE_TEXT, add_special_tokens=True).ids and ids[0] == 0
    assert how == {"special_tokens": True, "truncated_to": None, "fallback": None, "exact": True}
    # A toy vocabulary is not exact, and says why.
    ids, how = S.reference_tokens(SimpleNamespace(tok=VocabTokenizer(list("abc"), "char"), struct=SimpleNamespace(n_ctx=8)))
    assert len(ids) == 8 and how == {"special_tokens": False, "truncated_to": 8, "fallback": "characters the vocabulary can spell",
                                     "exact": False}


def test_the_calibration_key_changes_with_the_text_and_the_tokens(ws, monkeypatch):
    ids = S.reference_ids(ws)
    key = S.calibration_key(ws, ids)
    assert S.calibration_key(ws, ids[:-1]) != key                   # same text, other tokens (another tokenizer)
    monkeypatch.setattr(S, "REFERENCE_TEXT", S.REFERENCE_TEXT + " x")
    assert S.calibration_key(ws, ids) != key                         # same tokens, other text


def test_a_given_p50_is_used_as_is_and_that_layer_is_not_measured(sess, monkeypatch):
    ws = sess.ws
    calls = []
    real = S.calibrate_job
    monkeypatch.setattr(S, "calibrate_job", lambda w, ids: calls.append(1) or real(w, ids))
    norm = float(ws.runner.params()["blocks.1.mlp.W_out"][3].double().norm())
    spec = [{"addr": "L1N3", "mode": "add", "dose": 0.75}, {"addr": "L0N2", "mode": "add", "dose": 1, "unit": "raw"}]
    wire, info = S.prepare(ws, spec, p50={1: 3.5, "0": 9.0})          # a layer that is not pushed is ignored
    assert wire["p50"] == {"1": 3.5} and "calibration" not in wire and not calls
    assert info["calibration"] == {"needed": False, "given": [1]}
    assert info["entries"][0]["source"] == "given" and "source" not in info["entries"][1]
    assert float(S.resolve(ws, wire)[1].c[3]) == pytest.approx(0.75 * 3.5 / norm, rel=1e-6)
    assert not os.path.exists(os.path.join(ws.dir, S.CALIBRATION_FILE))
    # one layer given, the other measured
    wire, info = S.prepare(ws, [{"addr": "L0N2", "mode": "add", "dose": 1}, spec[0]], p50={"1": 3.5})
    assert calls == [1] and info["calibration"]["ran"] and info["calibration"]["given"] == [1]
    assert wire["p50"] == {"0": pytest.approx(independent_p50(ws)["0"], rel=1e-6), "1": 3.5} and wire["calibration"]
    assert [e["source"] for e in info["entries"]] == ["measured", "given"]
    for bad, message in [([], "must be an object"), ({"7": 1.0}, "does not exist"), ({"01": 1.0}, "is not a layer"),
                         ({"١": 1.0}, "is not a layer"), ({True: 1.0}, "is not a layer"), ({"1": 0}, "above 0"),
                         ({"1": -2.0}, "above 0"), ({"1": float("nan")}, "above 0"), ({"1": float("inf")}, "above 0"),
                         ({"1": 2e9}, "at most"), ({"1": "2"}, "must be a number"), ({"1": None}, "must be a number"),
                         ({"1": 10 ** 400}, "too large"), ({1: 2.0, "1": 2.0}, "twice")]:
        with pytest.raises(S.SteeringError, match=message):
            S.prepare(ws, spec, p50=bad)


def test_recalibrate_measures_again_and_an_unusable_measurement_is_never_cached(sess, monkeypatch):
    ws = sess.ws
    path = os.path.join(ws.dir, S.CALIBRATION_FILE)
    real = S.calibrate_job
    calls = []
    monkeypatch.setattr(S, "calibrate_job", lambda w, ids: calls.append(1) or real(w, ids))
    spec = [{"addr": "L1N3", "mode": "add", "dose": 1}]
    _, info = S.prepare(ws, spec)
    assert info["calibration"]["ran"] and info["calibration"]["stored"] and calls == [1]
    _, info = S.prepare(ws, spec)
    assert info["calibration"]["cached"] and calls == [1]
    _, info = S.prepare(ws, spec, recalibrate=True)
    assert info["calibration"]["ran"] and info["calibration"]["recalibrated"] and not info["calibration"]["cached"] and calls == [1, 1]
    # A measurement with a zero (or non-finite) layer is not written; a push on that layer is refused by name.
    os.remove(path)
    for bad in (0.0, float("nan"), -1.0, float("inf")):
        monkeypatch.setattr(S, "calibrate_job", lambda w, ids, bad=bad: {"p50": {"0": 2.0, "1": bad}, "tokens": len(ids)})
        events = []
        wire, info = S.prepare(ws, [{"addr": "L0N3", "mode": "add", "dose": 1}], on_event=events.append)
        assert wire["p50"] == {"0": 2.0} and info["calibration"]["stored"] is False and info["calibration"]["unusable"] == [1]
        assert {"state": "calibration_unusable", "layers": [1]} in events and not os.path.exists(path)
        with pytest.raises(S.SteeringError, match="layer 1 is .*nothing was cached. Give p50 for layer 1"):
            S.prepare(ws, spec)
        assert not os.path.exists(path)
        assert S.prepare(ws, spec, p50={"1": 2.5})[0]["p50"] == {"1": 2.5}      # the way out the message names
    # A cache entry holding an unusable value (written by hand, or by an older version) is not a hit.
    monkeypatch.setattr(S, "calibrate_job", lambda w, ids: calls.append(1) or real(w, ids))
    key = S.calibration_key(ws, S.reference_ids(ws))
    ws.write_json(S.CALIBRATION_FILE, {key: {"p50": {"0": 1.0, "1": 0.0}, "tokens": 3, "time": 1}})
    _, info = S.prepare(ws, spec)
    assert info["calibration"]["ran"] and calls == [1, 1, 1] and json.load(open(path))[key]["p50"]["1"] > 0


def test_plan_reads_never_decodes_a_block_twice_and_bounds_each_read():
    import random
    rng = random.Random(4)
    for unit, d, window in [(256, 5120, 4_000_000), (32, 64, 4096), (256, 8, 100), (16, 8, 1000)]:
        width = 64 * unit
        for _ in range(30):
            idx = sorted(rng.sample(range(width), rng.randint(1, 60)))
            plan = S.plan_reads(idx, d, unit, window, gap=4)
            assert [n for _, _, ns in plan for n in ns] == idx                         # every neuron once, in order
            blocks = [set(range(a // unit, (b - 1) // unit + 1)) for a, b, _ in plan]
            needed = {n // unit for n in idx}
            assert set().union(*blocks) == needed and sum(map(len, blocks)) == len(needed)   # only needed blocks, each once
            assert all((b - a) * d <= max(window, unit * d) for a, b, _ in plan)
            assert all(a == ns[0] and b == ns[-1] + 1 for a, b, ns in plan)
    # Unquantized storage: neurons close together share a read, far apart do not.
    assert S.plan_reads([0, 1, 20, 21], 24, 1, gap=4) == [(0, 2, [0, 1]), (20, 22, [20, 21])]
    assert S.plan_reads([0, 3, 7], 24, 1, gap=4) == [(0, 8, [0, 3, 7])]
    # Quantized: consecutive needed blocks share a read; a gap of one unneeded block splits it.
    assert S.plan_reads([0, 300, 900], 8, 256) == [(0, 301, [0, 300]), (900, 901, [900])]


def test_a_slow_column_norm_read_publishes_progress(sess, monkeypatch):
    ws = sess.ws
    monkeypatch.setattr(S, "PROGRESS_AFTER", 0.0)
    monkeypatch.setattr(S, "PROGRESS_EVERY", 0.0)
    monkeypatch.setattr(S, "READ_GAP", 0)                              # one read per neuron
    events = []
    wire = {"version": 1, "layers": {"0": {"add_resid": [[1, 5, 9], [1.0, 1.0, 1.0]]}, "1": {"add_resid": [[2], [1.0]]}},
            "p50": {"0": 2.0, "1": 2.0}}
    S.resolve(ws, wire, on_event=events.append)
    reading = [e for e in events if e["state"] == "reading"]
    assert [(e["layer"], e["done"], e["total"]) for e in reading] == [(0, 1, 4), (0, 2, 4), (0, 3, 4), (1, 4, 4)]
    assert events[-1]["state"] == "read" and events[-1]["done"] == 4 and events[-1]["total"] == 4
    assert all(isinstance(e["seconds"], float) for e in events)
    # cached norms count as done at once; a fast read publishes nothing; a failing callback never stops the read
    events.clear()
    monkeypatch.setattr(S, "PROGRESS_AFTER", 60.0)
    S.resolve(ws, wire, on_event=events.append)
    assert events == []
    monkeypatch.setattr(S, "PROGRESS_AFTER", 0.0)

    def broken(e):
        raise RuntimeError("the browser went away")
    ws.calib.clear()
    assert float(S.resolve(ws, wire, on_event=broken)[0].c[5]) > 0
    # the resolve route publishes the same events on the session's event stream
    published = []
    monkeypatch.setattr(sess, "publish", lambda e: published.append(e))
    ws.calib.clear()
    r = client(sess).post("/api/steer/resolve", json={"steer": [{"addr": "L0N1", "mode": "add", "dose": 1},
                                                                {"addr": "L0N9", "mode": "add", "dose": 1}],
                                                      "p50": {"0": 2.0}})
    assert r.status_code == 200, r.text
    steering = [e for e in published if e.get("type") == "steering"]
    assert [e["state"] for e in steering] == ["reading", "reading", "read"] and steering[0]["workspace"] == ws.name


# ------------------------------------------------------------------ probes
def test_reads_against_captures(sess):
    ws = sess.ws
    sess.apply('n = neuron(1, 7)\nn.w_in = 4 * unit(E("c"))', message="edit")
    ids = ws.encode("abcabc")
    names = ["blocks.0.mlp.hook_post", "blocks.1.mlp.hook_post"]
    base, current = capture(ws, ids, names, "base"), capture(ws, ids, names, "current")
    out = P.reads(ws, ids, [[1, 7], [0, 3]], ["base", "current"])
    assert out["tokens"] == 6 and out["neurons"] == [[1, 7], [0, 3]] and not out["steered"]
    assert out["values"]["base"][0] == pytest.approx(base[names[1]][0, :, 7].tolist())
    assert out["values"]["current"][0] == pytest.approx(current[names[1]][0, :, 7].tolist())
    assert out["values"]["current"][1] == pytest.approx(current[names[0]][0, :, 3].tolist())
    assert out["values"]["base"][0] != pytest.approx(out["values"]["current"][0])
    wire, _ = S.prepare(ws, [{"addr": "L1N7", "mode": "add", "dose": 2, "unit": "raw"}, {"addr": "L0N3", "mode": "mul", "dose": 0}])
    out = P.reads(ws, ids, [[1, 7], [0, 3]], ["base", "current"], wire)
    pushed = capture(ws, ids, names, hooks=S.hooks_for(ws, wire))
    assert out["steered"] and out["values"]["current"][0] == pytest.approx(pushed[names[1]][0, :, 7].tolist())   # after the push
    assert out["values"]["current"][1] == [0.0] * 6
    assert out["values"]["base"][0] == pytest.approx(base[names[1]][0, :, 7].tolist())                          # base never steered


def test_fires_against_captures(ws):
    ids = ws.encode("abcdef")
    names = ["blocks.0.mlp.hook_post", "blocks.1.mlp.hook_post"]
    cap = capture(ws, ids, names)
    for position, sign in ((-1, "both"), (2, "+"), (0, "-")):
        rows = [(l, n, float(cap[names[l]][0, position, n])) for l in range(2) for n in range(40)]
        key = {"both": lambda r: -abs(r[2]), "+": lambda r: -r[2], "-": lambda r: r[2]}[sign]
        expect = sorted(rows, key=key)[:5]
        out = P.fires(ws, ids, position, 5, sign)
        assert out["position"] == position % 6 and out["token"] == ids[position]
        assert [(r["layer"], r["neuron"]) for r in out["neurons"]] == [(l, n) for l, n, _ in expect]
        assert [r["value"] for r in out["neurons"]] == pytest.approx([v for _, _, v in expect])


def test_contrast_against_captures(ws):
    a = [ws.encode("aaab"), ws.encode("abab")]
    b = [ws.encode("zzzy"), ws.encode("q")]
    names = ["blocks.0.mlp.hook_post", "blocks.1.mlp.hook_post"]

    def maxes(ids):
        cap = capture(ws, ids, names)
        return torch.cat([(cap[n][0, 1:] if len(ids) > 1 else cap[n][0]).amax(0) for n in names]).double()
    diff = sum(maxes(t) for t in a) / 2 - sum(maxes(t) for t in b) / 2
    out = P.contrast(ws, a, b, 4)
    assert [r["layer"] * 40 + r["neuron"] for r in out["up"]] == diff.topk(4).indices.tolist()
    assert [r["layer"] * 40 + r["neuron"] for r in out["down"]] == (-diff).topk(4).indices.tolist()
    assert [r["diff"] for r in out["up"]] == pytest.approx(diff.topk(4).values.tolist())
    assert all(r["diff"] == pytest.approx(r["a"] - r["b"]) for r in out["up"] + out["down"])


def test_next_token_kl_and_movers(ws):
    ids = ws.encode("abcab")
    out = P.next_token(ws, ids, "current", None, 5)
    assert out["kl"] == 0 and out["top_unsteered"] == out["top_steered"] and out["rose"] == [] and out["fell"] == []
    assert P.next_token(ws, ids, "current", S.prepare(ws, [])[0], 5)["kl"] == 0
    spec = [{"addr": "L1N3", "mode": "add", "dose": 6, "unit": "raw"}, {"addr": "L0H0", "mode": "mul", "dose": 0}]
    wire, _ = S.prepare(ws, spec)
    out = P.next_token(ws, ids, "current", wire, 5)
    p_logits = ws.runner.run(ids)[0][0, -1].double()
    q_logits = ws.runner.run(ids, hooks=S.hooks_for(ws, wire))[0][0, -1].double()
    lp, lq = p_logits.log_softmax(-1), q_logits.log_softmax(-1)
    assert out["kl"] == pytest.approx(float((lp.exp() * (lp - lq)).sum()), rel=1e-9) and out["kl"] > 0
    assert [t["id"] for t in out["top_unsteered"]] == lp.topk(5).indices.tolist()
    assert [t["id"] for t in out["top_steered"]] == lq.topk(5).indices.tolist()
    delta = lq - lp
    assert [t["id"] for t in out["rose"]] == [i for i in delta.argsort(descending=True)[:5].tolist() if delta[i] > 0]
    assert all(t["delta_logp"] > 0 for t in out["rose"]) and all(t["delta_logp"] < 0 for t in out["fell"])
    with pytest.raises(S.SteeringError, match="never steered"):
        P.next_token(ws, ids, "base", wire)


def test_probe_inputs_are_bounded(ws):
    ids = ws.encode("abc")
    for call, message in [
        (lambda: P.reads(ws, ids, [[0, n] for n in range(17)]), "1..16 neurons"),
        (lambda: P.reads(ws, ids, [[0, 40]]), "neuron 40 does not exist"),
        (lambda: P.reads(ws, ids, [[0, 1]], ["delta"]), "variant must be"),
        (lambda: P.reads(ws, [], [[0, 1]]), "1..256 tokens"),
        (lambda: P.reads(ws, ids, [[0, 1]], ["current"], [{"addr": "L0N1", "mode": "mul", "dose": 0}]), None),
        (lambda: P.fires(ws, ids, 3), "position must be"),
        (lambda: P.fires(ws, ids, k=0), "k must be"),
        (lambda: P.fires(ws, ids, k=201), "k must be"),
        (lambda: P.fires(ws, ids, sign="up"), "sign must be"),
        (lambda: P.contrast(ws, [], [ids]), "set A must have"),
        (lambda: P.contrast(ws, [ids] * 17, [ids]), "set A must have"),
        (lambda: P.next_token(ws, ids, top=51), "top must be"),
        (lambda: P.next_token(ws, list(range(26)) * 20), "1..512 tokens"),
    ]:
        if message is None:
            call()       # a C3 list is prepared in process
            continue
        with pytest.raises(ValueError, match=message):
            call()


# ------------------------------------------------------------------ routes
def client(sess):
    return TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)


def test_steering_and_probe_routes(sess):
    c = client(sess)
    spec = [{"addr": "L1N3", "mode": "add", "dose": 0.75}, {"addr": "L0N2", "mode": "mul", "dose": 2},
            {"addr": "blocks.0.mlp", "mode": "mul", "dose": 0.5}, {"addr": "L0H1", "mode": "mul", "dose": 0}]
    r = c.post("/api/steer/resolve", json={"steer": spec, "calibrate": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["calibration"]["needed"] and not body["calibration"]["ran"] and body["entries"][0]["value"] is None
    r = c.post("/api/steer/resolve", json={"steer": spec})
    body = r.json()
    assert r.status_code == 200 and body["calibration"]["ran"] and body["layers"] == 2 and body["neurons"] == 2
    assert body["heads"] == 1 and body["mlps"] == 1
    e = body["entries"][0]
    assert e["value"] == pytest.approx(0.75 * e["p50"]["1"] / e["norm"]) and body["wire"]["p50"] == {"1": e["p50"]["1"]}
    assert c.post("/api/steer/resolve", json={"steer": spec}).json()["calibration"]["cached"]
    bad = c.post("/api/steer/resolve", json={"steer": [{"addr": "L0H1", "mode": "add", "dose": 1}]})
    assert bad.status_code == 400 and bad.json()["error"].startswith("SteeringError: steer entry 1 (L0H1): add pushes neurons only")
    assert c.post("/api/steer/resolve", json={"steer": spec, "variant": "base"}).status_code == 400

    reads = c.post("/api/probe/reads", json={"prompt": "abcab", "neurons": ["L1N3", "blocks.0.mlp.neurons.1-2"],
                                            "variants": ["base", "current"], "steer": spec})
    assert reads.status_code == 200, reads.text
    rb = reads.json()
    assert rb["addrs"] == ["L1N3", "L0N1", "L0N2"] and rb["token_strs"] == list("abcab") and rb["steered"]
    assert rb["calibration"]["cached"]
    assert len(rb["values"]["base"]) == 3 and len(rb["values"]["base"][0]) == 5
    assert c.post("/api/probe/reads", json={"prompt": "abc", "neurons": ["L0H1"]}).status_code == 400
    assert c.post("/api/probe/reads", json={"prompt": "abc", "neurons": ["L0N1"], "variants": ["base"], "steer": spec}).status_code == 400
    assert c.post("/api/probe/reads", json={"prompt": "abc", "neurons": ["blocks.0.mlp.neurons.0-16"]}).status_code == 400

    fires = c.post("/api/probe/fires", json={"prompt": "abcab", "k": 3})
    assert fires.status_code == 200 and len(fires.json()["neurons"]) == 3 and fires.json()["token_str"] == "b"
    assert c.post("/api/probe/fires", json={"prompt": "a" * 300}).status_code == 400

    con = c.post("/api/probe/contrast", json={"a": ["aaab", "abab"], "b": ["zzzy"], "k": 2})
    assert con.status_code == 200 and len(con.json()["up"]) == 2 and len(con.json()["down"]) == 2
    assert c.post("/api/probe/contrast", json={"a": [], "b": ["z"]}).status_code == 400

    nxt = c.post("/api/probe/next", json={"prompt": "abcab", "steer": spec, "top": 3})
    nb = nxt.json()
    assert nxt.status_code == 200 and nb["steered"] and nb["kl"] > 0 and len(nb["top_steered"]) == 3
    assert all(isinstance(t["token"], str) for t in nb["top_unsteered"] + nb["top_steered"] + nb["rose"] + nb["fell"])
    assert c.post("/api/probe/next", json={"prompt": "abcab", "steer": []}).json()["kl"] == 0
    assert c.post("/api/probe/next", json={"prompt": "abcab", "steer": spec, "variant": "base"}).status_code == 400
    # nothing above wrote a weight or recorded a patch
    assert sess.ws.patches == []


def test_steering_routes_take_p50_and_recalibrate(sess, monkeypatch):
    c = client(sess)
    calls = []
    real = S.calibrate_job
    monkeypatch.setattr(S, "calibrate_job", lambda w, ids: calls.append(1) or real(w, ids))
    spec = [{"addr": "L1N3", "mode": "add", "dose": 0.75}]
    body = c.post("/api/steer/resolve", json={"steer": spec, "p50": {"1": 3.5}}).json()
    assert body["wire"]["p50"] == {"1": 3.5} and body["entries"][0]["source"] == "given" and not calls
    assert body["calibration"] == {"needed": False, "given": [1]}
    assert body["entries"][0]["value"] == pytest.approx(0.75 * 3.5 / body["entries"][0]["norm"])
    assert c.post("/api/steer/resolve", json={"steer": spec}).json()["calibration"]["ran"] and calls == [1]
    assert c.post("/api/steer/resolve", json={"steer": spec}).json()["calibration"]["cached"] and calls == [1]
    again = c.post("/api/steer/resolve", json={"steer": spec, "recalibrate": True}).json()
    assert again["calibration"]["ran"] and again["calibration"]["recalibrated"] and calls == [1, 1]
    nxt = c.post("/api/probe/next", json={"prompt": "abcab", "steer": spec, "p50": {"1": 3.5}, "top": 3}).json()
    assert nxt["steered"] and nxt["calibration"]["given"] == [1] and calls == [1, 1]
    reads = c.post("/api/probe/reads", json={"prompt": "abc", "neurons": ["L1N3"], "steer": spec, "recalibrate": True}).json()
    assert reads["calibration"]["recalibrated"] and calls == [1, 1, 1]
    wire = {"version": 1, "layers": {"1": {"add_resid": [[3], [0.75]]}}, "p50": {"1": 3.5}}
    assert reads["values"]["current"][0] == pytest.approx(P.reads(sess.ws, sess.ws.encode("abc"), [[1, 3]], ["current"],
                                                                   S.prepare(sess.ws, spec)[0])["values"]["current"][0])
    assert nxt["kl"] == pytest.approx(P.next_token(sess.ws, sess.ws.encode("abcab"), "current", wire, 3)["kl"])


def test_probe_and_resolve_requests_are_checked_before_any_model_is_loaded(sess, monkeypatch):
    import wsp.runner
    built = []

    def refuse(w):
        built.append(w.name)
        raise AssertionError("the runner was built for a request that should have been refused first")
    monkeypatch.setattr(wsp.runner, "make_runner", refuse)
    assert sess.ws._runner is None
    c = client(sess)
    spec = [{"addr": "L1N3", "mode": "add", "dose": 0.75}]
    long = "ab" * 150                                                  # 300 tokens: over the 256 of reads/fires/contrast
    reads = {"prompt": "abc", "neurons": ["L0N1"]}
    for path, body, message in [
        ("/api/probe/reads", {**reads, "prompt": ""}, "nonempty text"),
        ("/api/probe/reads", {**reads, "prompt": long}, "has 300 tokens; at most 256"),
        ("/api/probe/reads", {**reads, "prompt": "ab1"}, "not in the vocabulary"),
        ("/api/probe/reads", {**reads, "neurons": ["L0N40"]}, "does not exist"),
        ("/api/probe/reads", {**reads, "neurons": ["L0H1"]}, "is not a neuron"),
        ("/api/probe/reads", {**reads, "variants": ["base", "base"]}, "variants must be"),
        ("/api/probe/reads", {**reads, "steer": [{"addr": "L0H1", "mode": "add", "dose": 1}]}, "add pushes neurons only"),
        ("/api/probe/reads", {**reads, "steer": spec, "variants": ["base"]}, "never steered"),
        ("/api/probe/reads", {**reads, "steer": spec, "p50": {"1": -1}}, "above 0"),
        ("/api/probe/reads", {**reads, "steer": spec, "recalibrate": "yes"}, "recalibrate must be"),
        ("/api/probe/fires", {"prompt": "abc", "k": 0}, "k must be"),
        ("/api/probe/fires", {"prompt": "abc", "k": 201}, "k must be"),
        ("/api/probe/fires", {"prompt": "abc", "position": 3}, "position must be"),
        ("/api/probe/fires", {"prompt": "abc", "sign": "up"}, "sign must be"),
        ("/api/probe/fires", {"prompt": "abc", "variant": "delta"}, "variant must be"),
        ("/api/probe/fires", {"prompt": long}, "at most 256"),
        ("/api/probe/contrast", {"a": ["abc"] * 17, "b": ["z"]}, "1..16 texts"),
        ("/api/probe/contrast", {"a": ["abc"], "b": [""]}, "nonempty text"),
        ("/api/probe/contrast", {"a": ["abc"], "b": [long]}, "at most 256"),
        ("/api/probe/contrast", {"a": ["abc"], "b": ["z"], "k": 0}, "k must be"),
        ("/api/probe/contrast", {"a": ["abc"], "b": ["z"], "variant": "x"}, "variant must be"),
        ("/api/probe/next", {"prompt": "abc", "top": 51}, "top must be"),
        ("/api/probe/next", {"prompt": "ab" * 300}, "at most 512"),
        ("/api/probe/next", {"prompt": "abc", "steer": [{"addr": "L9N1", "mode": "add", "dose": 1}]}, "does not exist"),
        ("/api/probe/next", {"prompt": "abc", "steer": spec, "variant": "base"}, "never steered"),
        ("/api/probe/next", {"prompt": "abc", "steer": spec, "p50": {"01": 2.0}}, "is not a layer"),
        ("/api/steer/resolve", {"steer": [{"addr": "L0N1", "mode": "add", "dose": 1e7}]}, "is larger than"),
        ("/api/steer/resolve", {"steer": spec, "p50": {"1": 1e300}}, "at most"),
        ("/api/steer/resolve", {"steer": spec, "recalibrate": 1}, "recalibrate must be"),
        ("/api/steer/resolve", {"steer": spec, "calibrate": "no"}, "calibrate must be"),
        ("/api/steer/resolve", {"steer": spec, "variant": "base"}, "never steered"),
    ]:
        r = c.post(path, json=body)
        assert r.status_code == 400 and message in r.json()["error"], (path, body, r.text)
    assert built == [] and sess.ws._runner is None
    # Resolving with every p50 given, or without calibrating, reads weights only: still no model.
    assert c.post("/api/steer/resolve", json={"steer": spec, "p50": {"1": 2.0}}).status_code == 200
    assert c.post("/api/steer/resolve", json={"steer": spec, "calibrate": False}).status_code == 200
    assert built == []
    # A good request does reach the runner (so the check above could have failed).
    c.post("/api/probe/fires", json={"prompt": "abc"})
    assert built and set(built) == {sess.ws.name}
