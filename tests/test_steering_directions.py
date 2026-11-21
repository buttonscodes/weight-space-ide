"""Steering along a measured direction (``direction:<measurement>@<k>``): a push added to the residual stream
entering layer k, resolved on the server from the saved means; no weight changes.

Tiny random models only (this project's Config + init_params, the tiny ``transformers`` llama of test_hf.build) and
the stub model worker of test_steering_worker. Every expected vector is computed here in float64 from the saved
means (the direction table's definition written out), never taken from the code under test alone.
"""
import hashlib
import json
import os
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

import wsp.runner
from wsp import directions as DR
from wsp import steering as S
from wsp.server import create_app
from wsp.session import Session

from test_steering import CFG, make_ws

N, D = CFG["n_layers"], CFG["d_model"]
CHAT = {"messages": [{"role": "user", "content": "abc"}], "force_template": True,
        "template": {"user": "", "assistant": "", "sep": ""}}


@pytest.fixture()
def ws(tmp_path):
    return make_ws(tmp_path / "rnd")


@pytest.fixture()
def sess(tmp_path):
    make_ws(tmp_path / "rnd")
    s = Session(str(tmp_path))
    s.open("rnd")
    return s


def client(sess):
    return TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)


def save(ws, name, good=None, bad=None, seed=0, n_layers=None, d=None, replace=False):
    """A measurement with typed (or random) means, saved as DR.measure saves one."""
    n_layers = N if n_layers is None else n_layers
    d = D if d is None else d
    g = torch.Generator().manual_seed(seed)
    good = torch.randn(n_layers + 1, d, generator=g) if good is None else good
    bad = torch.randn(n_layers + 1, d, generator=g) if bad is None else bad
    meta = {"version": 1, "name": name, "created": 0, "variant": "base", "n_layers": n_layers, "d_model": d,
            "counts": {"good": 1, "bad": 1}, "tokens": {"good": 1, "bad": 1}, "options": {}, "prompts_sha256": {}, "model": {}}
    DR.save(ws, name, good, bad, meta, {"good": ["a"], "bad": ["b"]}, replace=replace)
    return good.float(), bad.float()


def unit_direction(good, bad, k, ortho=True):
    """Row k of the direction table, written out in float64: unit(bad - good), then (when asked) the part
    orthogonal to unit(good), normalised again; the table holds it as float32."""
    g, b = good[k].double(), bad[k].double()
    r = (b - g) / (b - g).norm()
    if ortho:
        gh = g / g.norm()
        r = r - (r @ gh) * gh
        r = r / r.norm()
    return r


def push(good, bad, k, dose, unit="diff", ortho=True, table=True):
    """The float64 vector an entry adds. ``table``: v is the measurement's float32 table row (the definition, ID3);
    else the float64 direction written out above."""
    v = DR.table(good, bad, ortho)[k].double() if table else unit_direction(good, bad, k, ortho)
    scale = float((bad[k].double() - good[k].double()) @ v) if unit == "diff" else 1.0
    return dose * scale * v


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def means_path(ws, name):
    return os.path.join(ws.dir, "directions", name + ".safetensors")


def captures(ws, ids, hooks=None):
    names = [f"blocks.{l}.hook_resid_pre" for l in range(N)]
    logits, cache = ws.runner.run(ids, "current", None, names=names, hooks=hooks)
    return logits, cache


# ------------------------------------------------------------------ S1
def test_a_direction_entry_is_its_vector_at_its_layer(ws):
    good, bad = save(ws, "refusal")
    for unit, ortho, dose in [("diff", True, -1.0), ("diff", False, 0.5), ("raw", True, 2.0), ("raw", False, -3.0)]:
        entry = {"addr": "direction:refusal@1", "mode": "add", "dose": dose, "unit": unit, "orthogonalize_direction": ortho}
        wire, info = S.prepare(ws, [entry])
        c = push(good, bad, 1, dose, unit, ortho)
        assert wire == {"version": 1, "layers": {"1": {"add_vector": [float(x) for x in c.float()]}}}, (unit, ortho)
        # The table row agrees with the direction written out in float64 (to float32 rounding).
        assert torch.allclose(torch.tensor(wire["layers"]["1"]["add_vector"], dtype=torch.float64),
                              push(good, bad, 1, dose, unit, ortho, table=False), rtol=1e-5, atol=1e-6)
        assert info["calibration"] == {"needed": False}
        hooks = S.make_hooks(S.resolve(ws, wire))
        assert set(hooks) == {"blocks.1.hook_resid_pre"}
    # unit diff is the default, orthogonalize_direction true is the default
    default, _ = S.prepare(ws, [{"addr": "direction:refusal@1", "mode": "add", "dose": -1}])
    assert default["layers"]["1"]["add_vector"] == [float(x) for x in push(good, bad, 1, -1.0).float()]
    # dose +1, no projection, diff: the push is the difference of the two means (bad - good)
    plus, _ = S.prepare(ws, [{"addr": "direction:refusal@1", "mode": "add", "dose": 1, "orthogonalize_direction": False}])
    assert torch.allclose(torch.tensor(plus["layers"]["1"]["add_vector"]), (bad[1] - good[1]).float(), rtol=1e-5, atol=1e-6)
    minus, _ = S.prepare(ws, [{"addr": "direction:refusal@1", "mode": "add", "dose": -1, "orthogonalize_direction": False}])
    assert torch.allclose(torch.tensor(minus["layers"]["1"]["add_vector"]), (good[1] - bad[1]).float(), rtol=1e-5, atol=1e-6)
    # index 0 is the stream entering layer 0
    zero, _ = S.prepare(ws, [{"addr": "direction:refusal@0", "mode": "add", "dose": 1}])
    assert set(zero["layers"]) == {"0"} and set(S.make_hooks(S.resolve(ws, zero))) == {"blocks.0.hook_resid_pre"}


# ------------------------------------------------------------------ S2
def test_the_shift_is_the_vector_at_its_layer_and_nothing_before(ws):
    good, bad = save(ws, "refusal")
    ids = ws.encode("abcabd")
    wire, _ = S.prepare(ws, [{"addr": "direction:refusal@1", "mode": "add", "dose": -2}])
    c32 = push(good, bad, 1, -2.0).float()
    plain_logits, plain = captures(ws, ids)
    hooks = S.hooks_for(ws, wire)
    steered_logits, steered = captures(ws, ids, hooks)
    assert torch.equal(steered["blocks.0.hook_resid_pre"], plain["blocks.0.hook_resid_pre"])
    assert torch.equal(steered["blocks.1.hook_resid_pre"], plain["blocks.1.hook_resid_pre"] + c32)   # every position
    assert not torch.equal(steered["blocks.1.hook_resid_pre"], plain["blocks.1.hook_resid_pre"])
    assert not torch.allclose(steered_logits, plain_logits)
    # At index 0 every layer's input moves.
    wire0, _ = S.prepare(ws, [{"addr": "direction:refusal@0", "mode": "add", "dose": 1}])
    _, at0 = captures(ws, ids, S.hooks_for(ws, wire0))
    assert torch.equal(at0["blocks.0.hook_resid_pre"], plain["blocks.0.hook_resid_pre"] + push(good, bad, 0, 1.0).float())
    # Generation applies it at every step: each new token is the steered model's argmax.
    new = ws.runner.generate(ids, max_new_tokens=6, steer=wire)
    logits = ws.runner.run(ids + new, hooks=S.hooks_for(ws, wire))[0][0]
    assert [int(logits[len(ids) - 1 + i].argmax()) for i in range(len(new))] == new


# ------------------------------------------------------------------ S3
def test_a_transformers_model_adds_it_at_the_layer_input(tmp_path):
    pytest.importorskip("transformers")
    from test_hf import D_MODEL, build
    ckpt = tmp_path / "ckpt"
    build("llama", str(ckpt))
    s = Session(str(tmp_path / "root"))
    os.makedirs(s.root, exist_ok=True)
    w = s.open(str(ckpt))
    assert w.kind == "hf" and w.struct.d_model == D_MODEL
    good, bad = save(w, "refusal", n_layers=w.struct.n_layers, d=D_MODEL, seed=4)
    k = 1
    wire, _ = S.prepare(w, [{"addr": "direction:refusal@1", "mode": "add", "dose": -1}])
    c32 = push(good, bad, k, -1.0).float()
    r = w.runner
    model = r.model
    layer = model.model.layers[k]
    ids = w.encode("t3 t7 t11 t5")
    t = torch.tensor([ids])

    def recorder(seen):
        def pre(module, args, kwargs):
            seen.append((args[0] if args else kwargs["hidden_states"]).detach().clone())
        return pre
    plain, steered = [], []
    r.use("current")
    h = layer.register_forward_pre_hook(recorder(plain), with_kwargs=True)
    with torch.no_grad():
        model(input_ids=t, use_cache=False)
    h.remove()
    with r._steering(wire, "current", None):
        h = layer.register_forward_pre_hook(recorder(steered), with_kwargs=True)   # after the steering hook
        with torch.no_grad():
            model(input_ids=t, use_cache=False)
        h.remove()
    assert len(plain) == len(steered) == 1
    assert torch.equal(steered[0], plain[0] + c32)
    # Generation through HFRunner (transformers generate, KV cache) equals a greedy loop over full forwards without a
    # cache, with the same addition made by a hook of this test.
    M = 6
    got = r.generate(ids, max_new_tokens=M, steer=wire)

    def add(module, args, kwargs):
        if args:
            return (args[0] + c32.to(args[0].dtype),) + tuple(args[1:]), kwargs
        return args, {**kwargs, "hidden_states": kwargs["hidden_states"] + c32.to(kwargs["hidden_states"].dtype)}
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [] if eos is None else [eos])

    def greedy(hook):
        seq = list(ids)
        h = layer.register_forward_pre_hook(hook, with_kwargs=True) if hook else None
        try:
            for _ in range(M):
                with torch.no_grad():
                    logits = model(input_ids=torch.tensor([seq]), use_cache=False).logits[0, -1]
                seq.append(int(logits.argmax()))
                if seq[-1] in eos:
                    break
        finally:
            if h is not None:
                h.remove()
        return seq[len(ids):]
    assert got == greedy(add)
    assert r.generate(ids, max_new_tokens=M) == greedy(None)
    assert got != greedy(None)                         # the push changed what is generated
    assert not layer._forward_pre_hooks                # nothing left registered


# ------------------------------------------------------------------ S4
def test_entries_add_up_and_compose(ws):
    good, bad = save(ws, "refusal")
    good2, bad2 = save(ws, "other", seed=9)
    spec = [{"addr": "direction:refusal@1", "mode": "add", "dose": -0.7},
            {"addr": "direction:other@1", "mode": "add", "dose": 1.3, "unit": "raw", "orthogonalize_direction": False}]
    wire, info = S.prepare(ws, spec)
    total = push(good, bad, 1, -0.7) + push(good2, bad2, 1, 1.3, "raw", False)
    assert wire["layers"] == {"1": {"add_vector": [float(x) for x in total.float()]}}     # the float64 sum, cast once
    assert [e["layer"] for e in info["entries"]] == [1, 1]
    # The same direction twice adds up as well.
    twice, _ = S.prepare(ws, [spec[0], spec[0]])
    assert twice["layers"]["1"]["add_vector"] == [float(x) for x in (2 * push(good, bad, 1, -0.7)).float()]
    # With a neuron push at the same layer: both hooks, both effects.
    neuron = {"addr": "L1N3", "mode": "add", "dose": 1.5, "unit": "raw"}
    both, _ = S.prepare(ws, [spec[0], neuron])
    assert both["layers"]["1"] == {"add_raw": [[3], [1.5]], "add_vector": [float(x) for x in push(good, bad, 1, -0.7).float()]}
    hooks = S.hooks_for(ws, both)
    assert set(hooks) == {"blocks.1.hook_resid_pre", "blocks.1.mlp.hook_post"}
    ids = ws.encode("hello")
    names = ["blocks.1.hook_resid_pre", "blocks.1.mlp.hook_post"]
    only, _ = S.prepare(ws, [spec[0]])
    _, plain = ws.runner.run(ids, names=names)
    _, one = ws.runner.run(ids, names=names, hooks=S.hooks_for(ws, only))
    _, two = ws.runner.run(ids, names=names, hooks=hooks)
    c = torch.zeros(CFG["d_mlp"])
    c[3] = 1.5
    assert torch.equal(two["blocks.1.hook_resid_pre"], plain["blocks.1.hook_resid_pre"] + push(good, bad, 1, -0.7).float())
    assert torch.equal(two["blocks.1.hook_resid_pre"], one["blocks.1.hook_resid_pre"])
    assert torch.equal(two["blocks.1.mlp.hook_post"], one["blocks.1.mlp.hook_post"] + c)
    assert not torch.equal(one["blocks.1.mlp.hook_post"], plain["blocks.1.mlp.hook_post"])   # the push reached the MLP


# ------------------------------------------------------------------ S6
def fake_ws(directory, hookable=(True, True)):
    """A transformers-like workspace whose layer 1 may lack the decoder-layer module (no hook_resid_pre)."""
    layers = [SimpleNamespace(modules={"layer": f"model.layers.{l}"} if ok else {}) for l, ok in enumerate(hookable)]
    return SimpleNamespace(kind="hf", dir=str(directory), struct=SimpleNamespace(n_layers=N, d_model=D, layers=layers))


def refusal_cases(ws):
    """(entry, the exact text after the entry prefix), E-I1 to E-I11."""
    huge = torch.zeros(N + 1, D)
    big = huge.clone()
    big[1, 0] = 1e4
    save(ws, "huge", huge, big)                          # d . v = 1e4: dose 1e6 gives a component of 1e10
    g = torch.randn(N + 1, D, generator=torch.Generator().manual_seed(2))
    g[0] = 0.0
    g[0, 0] = 3.0                                        # a unit vector that is exact in floating point
    b = g.clone()
    b[0] = 2 * g[0]                                      # index 0: the difference is parallel to the good mean
    save(ws, "flat", g, b)                               # index 1: no difference at all
    save(ws, "wide", n_layers=3)                         # another model shape
    save(ws, "refusal")
    pin = file_sha(means_path(ws, "refusal"))
    other = "0" * 64
    add = {"mode": "add", "dose": -1}
    return [
        ({"addr": "direction:refusal", **add}, S.DIRECTION_HELP),
        ({"addr": "direction:refusal@01", **add}, S.DIRECTION_HELP),
        ({"addr": "direction:a b@1", **add}, S.DIRECTION_HELP),
        ({"addr": "direction:@1", **add}, S.DIRECTION_HELP),
        ({"addr": "direction:refusal@-1", **add}, S.DIRECTION_HELP),
        ({"addr": "direction:nothing@1", **add}, "no direction named 'nothing'"),
        ({"addr": "direction:wide@1", **add},
         "direction 'wide' was measured on a model with 3 layers and d_model 24; this model has 2 layers and d_model 24"),
        ({"addr": "direction:refusal@2", **add},
         "index 2 is the residual stream after the final norm, which no layer reads; steer at an index from 0 to 1"),
        ({"addr": "direction:refusal@3", **add}, "index 3 does not exist (the measurement has indices 0 to 2)"),
        ({"addr": "direction:flat@1", **add},
         "index 1 of direction 'flat' is unusable: the good and bad means do not differ there (after the projection, "
         "when it is asked for)"),
        ({"addr": "direction:flat@0", **add},
         "index 0 of direction 'flat' is unusable: the good and bad means do not differ there (after the projection, "
         "when it is asked for)"),
        ({"addr": "direction:refusal@1", "mode": "mul", "dose": 0}, "a direction is added: mode must be 'add'"),
        ({"addr": "direction:refusal@1", "dose": 1}, "a direction is added: mode must be 'add'"),
        ({"addr": "direction:refusal@1", **add, "unit": "resid"},
         "unit 'resid' is not available for a direction; use 'diff' (the difference of the two means along the "
         "direction, the default) or 'raw' (the unit direction times the dose)"),
        ({"addr": "direction:refusal@1", **add, "unit": "neuron"}, "unit must be 'diff' or 'raw' for a direction"),
        ({"addr": "direction:refusal@1", **add, "means_sha256": other},
         f"measurement 'refusal' changed since this entry was made: its means are {pin[:12]}… now and the entry names "
         f"{other[:12]}…; add the entry again from the Layers view, or drop means_sha256, to steer along the new "
         f"measurement"),
        ({"addr": "direction:refusal@1", **add, "means_sha256": pin.upper()},
         "means_sha256 must be the 64 hexadecimal digits of the measurement's means file"),
        ({"addr": "direction:refusal@1", **add, "means_sha256": pin[:63]},
         "means_sha256 must be the 64 hexadecimal digits of the measurement's means file"),
        ({"addr": "direction:refusal@1", **add, "means_sha256": None},
         "means_sha256 must be the 64 hexadecimal digits of the measurement's means file"),
        ({"addr": "direction:refusal@1", **add, "orthogonalize_direction": "yes"},
         "orthogonalize_direction must be true or false"),
        ({"addr": "direction:huge@1", "mode": "add", "dose": 1e6, "orthogonalize_direction": False},
         "the push at layer 1 has a component of 1e+10; at most 1e+09"),
        ({"addr": "direction:refusal@1", **add, "scale": 2}, "unknown field scale"),
        ({"addr": "direction:refusal@1", "mode": "add"}, "dose is missing"),
        ({"addr": "direction:refusal@1", "mode": "add", "dose": 1e7}, "dose 1e+07 is larger than 1e+06"),
    ]


def test_refusals(sess, monkeypatch, tmp_path):
    built = []

    def refuse(w):
        built.append(w.name)
        raise AssertionError("the runner was built for a request that should have been refused first")
    monkeypatch.setattr(wsp.runner, "make_runner", refuse)
    c = client(sess)
    cases = refusal_cases(sess.ws)
    for entry, text in cases:
        want = f"steer entry 1 ({entry['addr']}): {text}"
        with pytest.raises(S.SteeringError) as exc:
            S.normalize(sess.ws, [entry])
        assert str(exc.value) == want
        for path, body in [("/api/run", {"prompt": "abc"}), ("/api/steer/resolve", {}), ("/api/chat", CHAT),
                           ("/api/probe/next", {"prompt": "abc"})]:
            r = c.post(path, json={**body, "steer": [entry]})
            assert r.status_code == 400 and r.json()["error"] == f"SteeringError: {want}", (path, entry, r.text)
    # The flat measurement's index 0 is usable without the projection (the difference is not zero there).
    assert S.normalize(sess.ws, [{"addr": "direction:flat@0", "mode": "add", "dose": 1, "orthogonalize_direction": False}])
    # Neuron, MLP and head entries keep their own fields and texts.
    with pytest.raises(S.SteeringError, match=r"^steer entry 1 \(L0N1\): unknown field orthogonalize_direction$"):
        S.normalize(sess.ws, [{"addr": "L0N1", "mode": "add", "dose": 1, "orthogonalize_direction": True}])
    with pytest.raises(S.SteeringError, match=r"^steer entry 1 \(L0N1\): unknown field means_sha256$"):
        S.normalize(sess.ws, [{"addr": "L0N1", "mode": "add", "dose": 1, "means_sha256": "0" * 64}])
    with pytest.raises(S.SteeringError, match=r"^steer entry 1 \(L0N1\): unit must be 'resid' or 'raw'$"):
        S.normalize(sess.ws, [{"addr": "L0N1", "mode": "add", "dose": 1, "unit": "diff"}])
    # E-I12: a layer without the decoder-layer hook point (a workspace whose structure lacks it).
    fake = fake_ws(sess.ws.dir, hookable=(True, False))
    with pytest.raises(S.SteeringError) as exc:
        S.normalize(fake, [{"addr": "direction:refusal@1", "mode": "add", "dose": 1}])
    assert str(exc.value) == ("steer entry 1 (direction:refusal@1): layer 1 has no hook point at its input "
                              "(hook_resid_pre), so a direction cannot be added there")
    assert S.normalize(fake, [{"addr": "direction:refusal@0", "mode": "add", "dose": 1}])[0]["layers"].keys() == {"0"}
    # ... and through the routes, before the model.
    monkeypatch.setattr(S, "resid_hookable", lambda w, l: l != 1)
    e12 = ("SteeringError: steer entry 1 (direction:refusal@1): layer 1 has no hook point at its input (hook_resid_pre), "
           "so a direction cannot be added there")
    for path, body in [("/api/run", {"prompt": "abc"}), ("/api/steer/resolve", {})]:
        r = c.post(path, json={**body, "steer": [{"addr": "direction:refusal@1", "mode": "add", "dose": 1}]})
        assert r.status_code == 400 and r.json()["error"] == e12, r.text
    monkeypatch.undo()
    monkeypatch.setattr(wsp.runner, "make_runner", refuse)
    # E-I15 (the existing text): the base is never steered, with a direction entry too.
    spec = [{"addr": "direction:refusal@1", "mode": "add", "dose": 1}]
    for path, body in [("/api/steer/resolve", {"variant": "base"}), ("/api/probe/next", {"prompt": "abc", "variant": "base"})]:
        r = c.post(path, json={**body, "steer": spec})
        assert r.status_code == 400, r.text
        assert r.json()["error"] == "SteeringError: the base model is never steered; steer the current variant"
    assert built == [] and sess.ws._runner is None
    # A good direction request through /api/steer/resolve needs no model either (no calibration, no column reads).
    assert c.post("/api/steer/resolve", json={"steer": spec}).status_code == 200 and built == []


# ------------------------------------------------------------------ S7
def test_the_worker_checks_the_vector(ws, tmp_path):
    good, bad = save(ws, "refusal")
    wire, _ = S.prepare(ws, [{"addr": "direction:refusal@1", "mode": "add", "dose": -1}])
    assert S.check_wire(ws, json.loads(json.dumps(wire))) == wire       # the server's wire, as JSON carries it
    vec = wire["layers"]["1"]["add_vector"]
    length = f"steer layer 1 add_vector must be a list of {D} numbers (d_model)"
    value = "steer layer 1 add_vector value must be a finite number of at most 1e+09"
    for bad_vec, text in [(vec[:-1], length), (vec + [0.0], length), ([], length), ({"0": 1.0}, length),
                          ("0.1", length), (None, length), (torch.tensor(vec).tolist()[:1] * (D + 1), length),
                          (vec[:-1] + [float("nan")], value), (vec[:-1] + [float("inf")], value),
                          (vec[:-1] + [-float("inf")], value), (vec[:-1] + [1.0000001e9], value),
                          (vec[:-1] + [10 ** 400], value), (vec[:-1] + ["1"], value), (vec[:-1] + [True], value),
                          (vec[:-1] + [None], value)]:
        with pytest.raises(S.SteeringError) as exc:
            S.check_wire(ws, {"version": 1, "layers": {"1": {"add_vector": bad_vec}}})
        assert str(exc.value) == text, bad_vec if not isinstance(bad_vec, list) else len(bad_vec)
    assert S.check_wire(ws, {"version": 1, "layers": {"1": {"add_vector": [1e9] * (D - 1) + [-1e9]}}})   # the bound itself
    assert S.check_wire(ws, {"version": 1, "layers": {"0": {"add_vector": [0] * D}}})                  # ints are numbers
    # A layer without the hook point.
    fake = fake_ws(ws.dir, hookable=(True, False))
    with pytest.raises(S.SteeringError) as exc:
        S.check_wire(fake, {"version": 1, "layers": {"1": {"add_vector": vec}}})
    assert str(exc.value) == "layer 1 has no hook point at its input (hook_resid_pre), so a direction cannot be added there"
    assert S.check_wire(fake, {"version": 1, "layers": {"0": {"add_vector": vec}}})
    # resolve puts the numbers in a float32 vector of the plan.
    plan = S.resolve(ws, wire)
    assert plan[1].r.dtype == torch.float32 and torch.equal(plan[1].r, push(good, bad, 1, -1.0).float())
    assert plan[1].m is None and plan[1].c is None and plan[1].hm is None


# ------------------------------------------------------------------ S8
def test_through_the_model_worker(tmp_path, monkeypatch):
    from test_steering_worker import WspHarness, local_ws, remote
    h = WspHarness(tmp_path)
    h.load()
    here = local_ws(tmp_path)
    good, bad = save(h.ws, "secretname")
    r, sent, received = remote(h, monkeypatch)
    ids = here.encode("abc")
    spec = [{"addr": "direction:secretname@1", "mode": "add", "dose": -3}]
    out = r.generate(ids, max_new_tokens=6, steer=spec)
    assert [name for name, _ in sent] == ["generate"]                     # no calibration: a direction needs none
    shipped = sent[0][1]["steer"]
    assert shipped == {"version": 1, "layers": {"1": {"add_vector": [float(x) for x in push(good, bad, 1, -3.0).float()]}}}
    assert "secretname" not in json.dumps(sent) and "direction" not in json.dumps(shipped)
    assert out == here.runner.generate(ids, max_new_tokens=6, steer=shipped)
    assert out != here.runner.generate(ids, max_new_tokens=6)
    sent.clear()
    logits = r.next_logits(ids, steer=spec)
    assert sent[0][0] == "next_logits" and sent[0][1]["steer"] == shipped
    expect = here.runner.run(ids, hooks=S.hooks_for(here, shipped))[0][0, -1]
    assert torch.allclose(logits, expect, rtol=0, atol=1e-6)
    # The worker checks the vector again: a wrong length is refused there with its own text.
    frame, _, _ = h.call("generate", {"ids": ids, "variant": "current", "max_new_tokens": 2, "temperature": 0.0, "top_k": 0,
                                      "seed": None, "stop_ids": [], "steer": {"version": 1, "layers": {"1": {"add_vector": [0.0]}}}})
    assert frame.type == "error" and frame["exception"] == "SteeringError"
    assert frame["message"] == f"steer layer 1 add_vector must be a list of {D} numbers (d_model)"
    h.ws._runtime_client = None
    h.shutdown()


# ------------------------------------------------------------------ S9
def test_a_changed_measurement_is_noticed(ws):
    good, bad = save(ws, "refusal")
    old = file_sha(means_path(ws, "refusal"))
    entry = {"addr": "direction:refusal@1", "mode": "add", "dose": -1, "means_sha256": old}
    wire, info = S.prepare(ws, [entry])
    assert wire["layers"]["1"]["add_vector"] == [float(x) for x in push(good, bad, 1, -1.0).float()]
    assert info["entries"][0]["means_sha256"] == old
    good2, bad2 = save(ws, "refusal", seed=5, replace=True)
    new = file_sha(means_path(ws, "refusal"))
    assert new != old
    with pytest.raises(S.SteeringError) as exc:
        S.prepare(ws, [entry])
    assert str(exc.value) == (f"steer entry 1 (direction:refusal@1): measurement 'refusal' changed since this entry was "
                              f"made: its means are {new[:12]}… now and the entry names {old[:12]}…; add the entry again "
                              f"from the Layers view, or drop means_sha256, to steer along the new measurement")
    unpinned = {k: v for k, v in entry.items() if k != "means_sha256"}
    wire2, _ = S.prepare(ws, [unpinned])
    assert wire2["layers"]["1"]["add_vector"] == [float(x) for x in push(good2, bad2, 1, -1.0).float()]
    assert S.prepare(ws, [{**entry, "means_sha256": new}])[0] == wire2


# ------------------------------------------------------------------ S10
def test_describe_reports_the_direction(sess):
    w = sess.ws
    good, bad = save(w, "refusal")
    c = client(sess)
    spec = [{"addr": "direction:refusal@1", "mode": "add", "dose": -0.5},
            {"addr": "L0N2", "mode": "mul", "dose": 2}]
    body = c.post("/api/steer/resolve", json={"steer": spec}).json()
    means = file_sha(means_path(w, "refusal"))
    with open(os.path.join(w.dir, "directions", "refusal.json")) as f:
        meta = json.load(f)
    meta_sha = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    v = DR.table(good, bad, True)[1].double()
    scale = float((bad[1].double() - good[1].double()) @ v)
    e = body["entries"][0]
    assert set(e) == {"index", "addr", "mode", "dose", "kind", "unit", "layer", "direction", "measurement_index",
                      "orthogonalize_direction", "scale", "norm", "means_sha256", "meta_sha256", "value"}
    assert {k: e[k] for k in ("index", "addr", "mode", "dose", "kind", "unit", "layer", "direction", "measurement_index",
                              "orthogonalize_direction", "means_sha256", "meta_sha256", "value")} == {
        "index": 0, "addr": "direction:refusal@1", "mode": "add", "dose": -0.5, "kind": "direction", "unit": "diff",
        "layer": 1, "direction": "refusal", "measurement_index": 1, "orthogonalize_direction": True,
        "means_sha256": means, "meta_sha256": meta_sha, "value": -0.5}
    assert e["scale"] == pytest.approx(scale, rel=1e-12)
    assert e["norm"] == pytest.approx(0.5 * abs(scale) * float(v.norm()), rel=1e-12)
    assert e["scale"] == pytest.approx(float(unit_direction(good, bad, 1) @ (bad[1].double() - good[1].double())), rel=1e-5)
    assert body["directions"] == 1 and body["layers"] == 2 and body["neurons"] == 1
    assert body["wire"]["layers"]["1"]["add_vector"] == [float(x) for x in push(good, bad, 1, -0.5).float()]
    assert body["calibration"] == {"needed": False}
    plain = c.post("/api/steer/resolve", json={"steer": spec[1:]}).json()
    assert plain["directions"] == 0
    # The measurement's hash in GET /api/directions/{name}, for the Layers view's pin.
    got = c.get("/api/directions/refusal").json()
    assert got["means_sha256"] == means == hashlib.sha256(open(means_path(w, "refusal"), "rb").read()).hexdigest()
    assert c.get("/api/directions/nothing").status_code == 404


# ------------------------------------------------------------------ S12
def test_the_base_is_never_steered(sess, monkeypatch):
    w = sess.ws
    save(w, "refusal")
    scores_never_steer_the_base(sess, monkeypatch)       # below, with the other score tests
    spec = [{"addr": "direction:refusal@1", "mode": "add", "dose": -1}]
    wire, _ = S.prepare(w, spec)
    ids = w.encode("abc")
    with pytest.raises(S.SteeringError, match="^the base model is never steered; steer the current variant$"):
        S.hooks_for(w, wire, "base")
    with pytest.raises(S.SteeringError, match="^the base model is never steered; steer the current variant$"):
        w.runner.generate(ids, "base", max_new_tokens=2, steer=spec)
    with pytest.raises(S.SteeringError, match="^the base model is never steered; steer the current variant$"):
        S.describe(w, wire, {"entries": [], "calibration": {"needed": False}}, "base")
    c = client(sess)
    base_only = {"variants": ["base"], "base_messages": CHAT["messages"], "force_template": True,
                 "template": CHAT["template"], "steer": spec}
    r = c.post("/api/chat", json=base_only)
    assert r.status_code == 400 and r.json()["error"] == ("SteeringError: steering applies to the edited model only, and "
                                                          "this request runs the base model only")
    # Run and Chat steer the edited model only: the base column is the unsteered model.
    run = c.post("/api/run", json={"prompt": "abc", "max_new_tokens": 4, "steer": spec}).json()
    assert run["current"]["steered"] and "steered" not in run["base"]
    assert run["base"]["text"] == c.post("/api/run", json={"prompt": "abc", "max_new_tokens": 4}).json()["base"]["text"]
    assert run["steering"]["resolved"] and run["steering"]["entries"][0]["kind"] == "direction"


# ------------------------------------------------------------------ part 2: scores with steering
UNSTEERED_KEYS = {"state", "version", "seconds", "candidate", "rendering", "generation", "scores", "datasets", "done", "total"}
KW_PROMPTS = ["abc", "hello", "zzz"]
KL_PROMPTS = ["ab", "fox", "quiet"]
M = 5


def scores_body(state="current", **kw):
    return {"state": state, "max_response_length": M,
            "scores": [{"plugin": "KeywordRate", "prompts": KW_PROMPTS, "print_responses": True, "keyword_markers": ["a"]},
                       {"plugin": "KLDivergence", "prompts": KL_PROMPTS}], **kw}


def params_of(ws, variant="current", extra=None):
    return {n: ws.read(n, (), variant, extra) for n in ws.names()}


def greedy(p, ids, n, hooks=None):
    """Argmax at each step over the full sequence (no cache), with the test's own hooks."""
    from wsp.model import Config, forward
    cfg = Config(**CFG)
    toks, out = list(ids), []
    for _ in range(n):
        nxt = int(forward(p, cfg, torch.tensor(toks), hooks=hooks)[0, -1].argmax())
        toks.append(nxt)
        out.append(nxt)
    return out


def kl64(pb, ps, ids, hooks=None):
    """KL(base || state) of the next-token distribution in float64; the state's logits with the test's hooks."""
    from wsp.model import Config, forward
    cfg = Config(**CFG)
    lb = torch.log_softmax(forward(pb, cfg, torch.tensor(ids))[0, -1].double(), -1)
    ls = torch.log_softmax(forward(ps, cfg, torch.tensor(ids), hooks=hooks)[0, -1].double(), -1)
    return float((lb.exp() * (lb - ls)).sum())


def recording(ws):
    """Record the keyword arguments of every generate call of the workspace's runner."""
    real, calls = ws.runner.generate, []

    def generate(ids, variant="current", extra=None, **kw):
        calls.append((variant, sorted(kw)))
        return real(ids, variant, extra, **kw)
    ws.runner.generate = generate
    return calls


def test_steered_scores(ws):
    from wsp import directions_scores as DS
    from wsp.wscript import run_script
    from test_steering_worker import WorkerRunner
    good, bad = save(ws, "refusal")
    spec = [{"addr": "direction:refusal@1", "mode": "add", "dose": -2}]
    c32 = push(good, bad, 1, -2.0).float()
    hooks = {"blocks.1.hook_resid_pre": lambda x, name: x + c32}
    calls = recording(ws)
    plain = DS.score(ws, scores_body())                          # computes the KL reference
    again = DS.score(ws, scores_body())                          # reads it
    assert set(plain) == set(again) == UNSTEERED_KEYS
    assert calls and all(kw == ["max_new_tokens", "temperature"] for _, kw in calls)   # no steer keyword at all
    calls.clear()
    out = DS.score(ws, scores_body(steer=spec))
    assert set(out) == UNSTEERED_KEYS | {"steering"}
    assert calls and all(v == "current" and "steer" in kw for v, kw in calls)
    pb, pc = params_of(ws, "base"), params_of(ws, "current")
    # Responses: a greedy loop with the test's own addition.
    rows = out["scores"][0]["responses"]
    for prompt, row in zip(KW_PROMPTS, rows):
        want = greedy(pc, ws.encode(prompt), M, hooks)
        assert row["response"] == ws.tok.decode(want, skip_special_tokens=True) and row["tokens"] == M, prompt
    assert [r["response"] for r in rows] != [r["response"] for r in plain["scores"][0]["responses"]]
    # KL against the unsteered base, per prompt.
    kl = out["scores"][1]
    for t, v in zip(KL_PROMPTS, kl["per_prompt"]):
        assert v == pytest.approx(kl64(pb, pc, ws.encode(t), hooks), rel=1e-6), t
    assert kl["value"] > plain["scores"][1]["value"] + 1e-6
    assert kl["reference"]["key"] == plain["scores"][1]["reference"]["key"] and kl["reference"]["cached"]
    # The report says what was applied.
    rep = out["steering"]
    assert rep["variant"] == "current" and rep["resolved"] and rep["directions"] == 1 and rep["layers"] == 1
    assert rep["entries"][0]["kind"] == "direction" and rep["entries"][0]["means_sha256"] == file_sha(means_path(ws, "refusal"))
    assert rep["calibration"] == {"needed": False} and "wire" not in rep
    # steer [] (with a p50 that is then ignored) and no steer: exactly the unsteered answer.
    calls.clear()
    empty = DS.score(ws, scores_body(steer=[], p50={"1": 2.0}))
    none = DS.score(ws, scores_body(steer=None))
    assert all("steer" not in kw for _, kw in calls)
    strip = lambda a: json.dumps({k: v for k, v in a.items() if k != "seconds"})  # noqa: E731
    assert strip(empty) == strip(again) == strip(none)
    # A candidate with steering: the candidate's weights with the test's addition.
    script = "mlp(0).W_out *= 3"
    deltas = run_script(ws, script)["deltas"]
    pk = params_of(ws, "current", deltas)
    assert torch.allclose(pk["blocks.0.mlp.W_out"], 3 * pc["blocks.0.mlp.W_out"], rtol=1e-6)
    cand = DS.score(ws, scores_body("candidate", script=script, steer=spec))
    for prompt, row in zip(KW_PROMPTS, cand["scores"][0]["responses"]):
        assert row["response"] == ws.tok.decode(greedy(pk, ws.encode(prompt), M, hooks), skip_special_tokens=True)
    for t, v in zip(KL_PROMPTS, cand["scores"][1]["per_prompt"]):
        assert v == pytest.approx(kl64(pb, pk, ws.encode(t), hooks), rel=1e-6), t
    assert cand["candidate"]["stored"] > 0 and cand["steering"]["directions"] == 1
    # A runner with next_logits (as the model worker's and GGUF's): the steered first token goes through it.
    seen = []

    class NextRunner(WorkerRunner):
        def next_logits(self, ids, variant="current", extra=None, steer=None):
            seen.append((variant, steer is not None))
            return super().next_logits(ids, variant, extra, steer)
    ws._runner = NextRunner(ws)
    through = DS.score(ws, scores_body(steer=spec))
    assert seen == [("current", True)] * len(KL_PROMPTS)
    assert through["scores"][1]["per_prompt"] == pytest.approx(kl["per_prompt"], rel=1e-6)
    # Mixed with a neuron push that needs calibration: prepared once, reported, applied.
    mixed = DS.score(ws, scores_body(steer=spec + [{"addr": "L0N2", "mode": "add", "dose": 0.5}]))
    assert mixed["steering"]["calibration"]["ran"] and mixed["steering"]["neurons"] == 1
    assert mixed["steering"]["entries"][1]["value"] is not None


def test_weights_patches_and_version_are_unchanged(sess):
    w = sess.ws
    save(w, "refusal")
    assert sess.apply("mlp(1).W_out *= 0.5")["committed"]
    spec = [{"addr": "direction:refusal@1", "mode": "add", "dose": -2}, {"addr": "L0N2", "mode": "add", "dose": 0.5}]

    def snapshot():
        files = {}
        for top in ("patches",):
            d = os.path.join(w.dir, top)
            for fn in sorted(os.listdir(d)):
                files[f"{top}/{fn}"] = file_sha(os.path.join(d, fn))
        files["workspace.json"] = file_sha(os.path.join(w.dir, "workspace.json"))
        return (w.version, len(w.patches), files, {n: w.read(n).clone() for n in w.names()},
                {n: t.clone() for n, t in w.runner.params("current").items()},
                {n: t.clone() for n, t in w.runner.params("base").items()})
    before = snapshot()
    c = client(sess)
    for path, body in [("/api/run", {"prompt": "abc", "max_new_tokens": 3}),
                       ("/api/chat", {**CHAT, "max_new_tokens": 3}),
                       ("/api/probe/next", {"prompt": "abcab"}),
                       ("/api/directions/scores", scores_body()),
                       ("/api/directions/scores", scores_body("candidate", script="attn(0).W_O *= 0.2"))]:
        r = c.post(path, json={**body, "steer": spec})
        assert r.status_code == 200, (path, r.text)
        b = r.json()
        assert b.get("steering") or b.get("steered"), path
    after = snapshot()
    assert before[:3] == after[:3]
    for one, two in zip(before[3:], after[3:]):
        assert one.keys() == two.keys() and all(torch.equal(one[n], two[n]) for n in one)


def scores_never_steer_the_base(sess, monkeypatch):
    """S12's scores case (a measurement 'refusal' saved; no model loaded yet)."""
    from wsp import directions_scores as DS
    w = sess.ws
    spec = [{"addr": "direction:refusal@1", "mode": "add", "dose": -1}]
    built = []

    def refuse(x):
        built.append(x.name)
        raise AssertionError("the runner was built for a request that should have been refused first")
    monkeypatch.setattr(wsp.runner, "make_runner", refuse)
    c = client(sess)
    r = c.post("/api/directions/scores", json=scores_body("base", steer=spec))
    assert r.status_code == 400
    assert r.json()["error"] == "SteeringError: the base model is never steered; steer the current variant"
    # The steer entries are checked before the model as well, with their own texts.
    r = c.post("/api/directions/scores", json=scores_body(steer=[{"addr": "direction:refusal@2", "mode": "add", "dose": 1}]))
    assert r.status_code == 400 and r.json()["error"] == (
        "SteeringError: steer entry 1 (direction:refusal@2): index 2 is the residual stream after the final norm, which no "
        "layer reads; steer at an index from 0 to 1")
    r = c.post("/api/directions/scores", json=scores_body(steer=spec, p50={"01": 2.0}))
    assert r.status_code == 400 and "is not a layer" in r.json()["error"]
    # The unknown-field message is unchanged (steer and p50 are accepted, not listed).
    r = c.post("/api/directions/scores", json=scores_body(x=1))
    assert r.json()["error"] == ("DirectionError: x: unknown field; the fields are state, script, scores, chat, system, "
                                 "prefix, chat_options, max_response_length and stream")
    assert built == [] and w._runner is None
    monkeypatch.undo()
    with pytest.raises(S.SteeringError, match="^the base model is never steered; steer the current variant$"):
        DS.score(w, scores_body("base", steer=spec))
    prepared = DS.prepare(w, DS.check_scores(w, scores_body("base")))
    with pytest.raises(S.SteeringError, match="^the base model is never steered; steer the current variant$"):
        DS.evaluate(w, prepared, "base", steer=S.prepare(w, spec)[0])
    # The base state of an unsteered request is still scored (KL 0 by definition).
    assert DS.score(w, scores_body("base"))["scores"][1]["by_definition"]


def test_run_and_chat_report_the_direction(sess):
    w = sess.ws
    good, bad = save(w, "refusal")
    c = client(sess)
    spec = [{"addr": "direction:refusal@1", "mode": "add", "dose": -0.5}]
    v = DR.table(good, bad, True)[1].double()
    scale = float((bad[1].double() - good[1].double()) @ v)
    means = file_sha(means_path(w, "refusal"))
    for path, body in [("/api/run", {"prompt": "abc", "max_new_tokens": 3}), ("/api/chat", {**CHAT, "max_new_tokens": 3})]:
        r = c.post(path, json={**body, "steer": spec})
        assert r.status_code == 200, (path, r.text)
        rep = r.json()["steering"]
        assert rep["resolved"] and rep["directions"] == 1 and rep["layers"] == 1 and rep["neurons"] == 0, path
        e = rep["entries"][0]
        assert e["kind"] == "direction" and e["means_sha256"] == means, path
        assert e["scale"] == pytest.approx(scale, rel=1e-12), path
        assert e["norm"] == pytest.approx(0.5 * abs(scale) * float(v.norm()), rel=1e-12), path
        neuron = c.post(path, json={**body, "steer": [{"addr": "L0N2", "mode": "add", "dose": 0.5, "unit": "raw"}]})
        assert neuron.status_code == 200 and neuron.json()["steering"]["directions"] == 0, (path, neuron.text)
