"""Steering together with the selections branch: patches applied at run time and the push.

1. On the GGUF runner, a patched projection is a `PackedLinear` whose forward adds the patch overlay
   (`DeltaOverlay`) to the packed matmul, and steering registers a forward pre-hook on the same module.
   The push must change the input both see: hook first, then base + overlay on the pushed input.
   The packed matmul here is a float kernel over the checkpoint's own float32 bytes (no GGUF file, no
   ggml): what is tested is the order of the hook, the packed product and the overlays.
2. A `resid` dose divides by the output column's norm, read from the variant being run: with several
   patches on that column, the norm is the patched column's.

Model-free: the tiny llama-named checkpoint and fake decoder of test_steering_worker, and the in-memory
Q4_K fake store of selection_fixtures.
"""
import os
import sys

import numpy as np
import pytest
import torch
from torch import nn

from wsp import delta as D
from wsp import steering as S

from test_steering_worker import D as DM, F, H, Decoder, gguf_runner, hooks_on   # DM: d_model of that fixture

sys.path.insert(0, os.path.dirname(__file__))
from selection_fixtures import quant_workspace  # noqa: E402

DOWN1 = "model.layers.1.mlp.down_proj.weight"           # [d_model, neurons] = [8, 10]
O0 = "model.layers.0.self_attn.o_proj.weight"            # [8, 8], two heads of 4


def col(c, rows=DM, scale=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return D.block((range(0, rows), range(c, c + 1)), torch.randn(rows, 1, generator=g) * scale)


def patch_many(ws):
    """Seven patches on layer 1's down projection (four full-height column blocks, which DeltaOverlay gathers;
    a partial block, a low-rank piece and a sparse piece on the steered columns) and one on layer 0's o_proj."""
    g = torch.Generator().manual_seed(11)
    ws.commit({DOWN1: [col(7, seed=1)]}, "col 7")
    ws.commit({DOWN1: [col(2, seed=2)]}, "col 2")
    ws.commit({DOWN1: [col(3, seed=3)]}, "col 3")
    ws.commit({DOWN1: [D.block((range(0, DM), range(5, 7)), torch.randn(DM, 2, generator=g) * 0.3)]}, "cols 5-6")
    ws.commit({DOWN1: [D.block((range(0, 4), range(7, 8)), torch.randn(4, 1, generator=g))]}, "col 7, top half")
    ws.commit({DOWN1: [D.lowrank((range(2, 6), range(0, 5)), torch.randn(4, 1, generator=g), torch.randn(1, 5, generator=g))]},
              "low rank")
    sparse = D.Piece("sparse", (range(0, DM), range(0, 10)), values=torch.tensor([0.9, -1.1]),
                     coords=torch.tensor([[6, 7], [1, 4]]))
    ws.commit({DOWN1: [sparse]}, "sparse")
    ws.commit({O0: [D.block((range(0, DM), range(4, 8)), torch.randn(DM, 4, generator=g) * 0.4)]}, "o_proj head 1")
    assert len(ws.pieces(DOWN1)) >= 7


class FloatKernel:
    """Stands in for the ggml kernel: the packed bytes are the checkpoint's float32 values."""

    def matmul(self, raw, qtype, shape, x):
        return x.float() @ torch.from_numpy(np.ascontiguousarray(raw).view(np.float32).reshape(shape).copy()).T


def packed_runner(tmp_path, monkeypatch):
    """The steering GGUF fixture with layer 1's down_proj and layer 0's o_proj replaced by real PackedLinear modules."""
    r, model, G = gguf_runner(tmp_path, monkeypatch)
    store = r.ws.store
    for name in (DOWN1, O0):
        store.where[name].tensor_type = "F32"
    store.row_permutations, store.column_permutations = {}, {}
    r.kernel = FloatKernel()
    model.model.layers[1].mlp.down_proj = G.PackedLinear(r, DOWN1)
    model.model.layers[0].self_attn.o_proj = G.PackedLinear(r, O0)
    r._modules = dict(model.named_modules())              # where the steering hooks are looked up
    return r, model, G


def reference(tensors, ws, push_down, push_o):
    """A plain decoder holding the patched (current) weights densely, with the push as pre-hooks."""
    ref = Decoder()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            p.copy_(tensors[name])
        ref.model.layers[1].mlp.down_proj.weight.copy_(ws.read(DOWN1))
        ref.model.layers[0].self_attn.o_proj.weight.copy_(ws.read(O0))
    ref.model.layers[1].mlp.down_proj.register_forward_pre_hook(lambda m, a: (push_down(a[0].clone()),))
    ref.model.layers[0].self_attn.o_proj.register_forward_pre_hook(lambda m, a: (push_o(a[0].clone()),))
    return ref.eval()


def logits_of(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids]), None, False, 0).logits[0]


def test_steering_pushes_the_input_of_a_patched_packed_projection(tmp_path, monkeypatch):
    from test_steering_worker import checkpoint
    r, model, G = packed_runner(tmp_path, monkeypatch)
    tensors = checkpoint(str(tmp_path / "again"))           # the same seeded weights as the fixture's checkpoint
    ws = r.ws
    patch_many(ws)
    norm7 = float(ws.read(DOWN1)[:, 7].double().norm())
    assert abs(norm7 - float(tensors[DOWN1][:, 7].double().norm())) > 1e-3        # the patches move column 7
    spec = [{"addr": "L1N4", "mode": "add", "dose": 3.0, "unit": "raw"},
            {"addr": "L1N7", "mode": "add", "dose": 0.75},                       # resid: p50 given, patched norm
            {"addr": "L1N2", "mode": "mul", "dose": 2},
            {"addr": "L0H1", "mode": "mul", "dose": 0.5}]
    wire, info = S.prepare(ws, spec, p50={"1": 2.0})

    def push_down(x):
        x[..., 2] *= 2
        x[..., 4] += 3.0
        x[..., 7] += 0.75 * 2.0 / norm7
        return x

    def push_o(x):
        x[..., DM // H:] *= 0.5
        return x
    ref = reference(tensors, ws, push_down, push_o)
    ids = [4, 1, 8, 2, 6, 3]
    got = r.next_logits(ids, steer=wire)
    assert torch.allclose(got, logits_of(ref, ids)[-1], atol=1e-5)
    # Both overlay paths ran on the pushed input: the gathered column blocks and the pieces applied one by one.
    held = model.model.layers[1].mlp.down_proj._committed[5]             # (list, key, dtype, device, pieces, overlay)
    assert held.n_gathered >= 4 and held.n_rest >= 2
    assert model.model.layers[0].self_attn.o_proj._committed is not None
    assert all(hooks_on(m) == 0 for m in model.modules())

    # The test can tell the orders apart: overlay on the unpushed input, or the patches ignored, give other numbers.
    class WrongOrder(nn.Module):
        def __init__(self, base, delta):
            super().__init__()
            self.base, self.delta = base, delta

        def forward(self, x):
            return push_down(x.clone()) @ self.base.T + x @ self.delta.T
    wrong = reference(tensors, ws, lambda x: x, push_o)
    wrong.model.layers[1].mlp.down_proj = WrongOrder(tensors[DOWN1], ws.read(DOWN1) - tensors[DOWN1])
    assert not torch.allclose(got, logits_of(wrong, ids)[-1], atol=1e-3)
    unpatched = reference(tensors, ws, push_down, push_o)
    with torch.no_grad():
        unpatched.model.layers[1].mlp.down_proj.weight.copy_(tensors[DOWN1])
    assert not torch.allclose(got, logits_of(unpatched, ids)[-1], atol=1e-3)

    # Generation (cached decode, prompt in chunks) follows the same steered, patched model token by token.
    monkeypatch.setattr(G, "PREFILL_CHUNK", 4)
    new = r.generate(ids, max_new_tokens=6, steer=wire)
    seq = list(ids)
    for _ in range(len(new)):
        seq.append(int(logits_of(ref, seq)[-1].argmax()))
    assert new == seq[len(ids):]
    assert all(hooks_on(m) == 0 for m in model.modules())
    # Without steering the same runner gives the patched, unpushed model; on base, neither.
    plain = reference(tensors, ws, lambda x: x, lambda x: x)
    assert torch.allclose(r.next_logits(ids), logits_of(plain, ids)[-1], atol=1e-5)
    base = Decoder()
    with torch.no_grad():
        for name, p in base.named_parameters():
            p.copy_(tensors[name])
    assert torch.allclose(r.next_logits(ids, "base"), logits_of(base.eval(), ids)[-1], atol=1e-5)


class DtypeKernel(FloatKernel):
    """The packed product in the input's own dtype, as `PackedLinear.forward` now passes it on uncast."""

    def matmul(self, raw, qtype, shape, x):
        return x @ super().matmul(raw, qtype, shape, torch.eye(shape[1])).to(x.dtype)   # I @ W.T = W.T


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_the_push_reaches_the_overlay_in_the_input_dtype(tmp_path, monkeypatch, dtype):
    """`PackedLinear.forward` no longer casts its input to float32: the hook's constants follow the input's
    dtype, and the overlay (built in the output's dtype) adds its change to exactly the pushed input."""
    r, model, G = packed_runner(tmp_path, monkeypatch)
    r.kernel = DtypeKernel()
    patch_many(ws := r.ws)
    model.to(dtype)
    seen = []
    apply = G.DeltaOverlay.apply

    def record(self, x, out):
        if out.shape[-1] == DM and x.shape[-1] == F:              # layer 1's down_proj: [.., neurons] -> [.., d_model]
            seen.append((x.dtype, out.dtype, self.dtype, x.detach().clone()))
        return apply(self, x, out)
    monkeypatch.setattr(G.DeltaOverlay, "apply", record)
    ids = [4, 1, 8, 2, 6, 3]
    spec = [{"addr": "L1N4", "mode": "add", "dose": 3.0, "unit": "raw"}, {"addr": "L1N2", "mode": "mul", "dose": 2}]
    wire, _ = S.prepare(ws, spec)
    plain_logits = r.next_logits(ids)
    steered_logits = r.next_logits(ids, steer=wire)
    assert plain_logits.dtype == steered_logits.dtype == torch.float32      # _prefill returns float32 logits
    (pd, po, pv, plain), (sd, so, sv, pushed) = seen[0], seen[-1]
    assert len(seen) == 2 and pd == sd == po == so == pv == sv == dtype
    # Layer 1's down_proj input is the same in both runs up to the push, so the difference is the push alone,
    # computed in the input's dtype: x[..., 4] + 3 and x[..., 2] * 2, every other neuron untouched.
    expect = plain.clone()
    expect[..., 4] += torch.tensor(3.0, dtype=dtype)
    expect[..., 2] *= torch.tensor(2.0, dtype=dtype)
    assert torch.equal(pushed, expect)
    assert all(hooks_on(m) == 0 for m in model.modules())
    if dtype == torch.float64:                                  # and the result is the dense patched model's
        from test_steering_worker import checkpoint
        tensors = checkpoint(str(tmp_path / "again"))

        def push_down(x):
            x[..., 2] *= 2
            x[..., 4] += 3.0
            return x
        ref = reference(tensors, ws, push_down, lambda x: x).double()
        assert torch.allclose(steered_logits, logits_of(ref, ids)[-1].float(), atol=1e-6)    # float64 run, float32 logits


def test_resid_norms_are_read_from_the_patched_column(tmp_path, monkeypatch):
    r, model, G = packed_runner(tmp_path, monkeypatch)
    ws = r.ws
    from test_steering_worker import checkpoint
    tensors = checkpoint(str(tmp_path / "again"))
    base7 = float(tensors[DOWN1][:, 7].double().norm())
    assert float(S.column_norms(ws, 1, [7])[0][0]) == pytest.approx(base7, rel=1e-12)
    patch_many(ws)                                           # three patches reach column 7 (block, half block, sparse)
    assert sum(D.intersects(p, (range(0, DM), range(7, 8))) for p, _ in ws.pieces(DOWN1)) >= 3
    norms, _ = S.column_norms(ws, 1, [7, 2, 0, 9])
    W = ws.read(DOWN1).double()
    assert np.allclose(norms.numpy(), [float(W[:, n].norm()) for n in (7, 2, 0, 9)], rtol=1e-12)
    assert abs(float(norms[0]) - base7) > 1e-3
    assert float(S.column_norms(ws, 1, [7], "base")[0][0]) == pytest.approx(base7, rel=1e-12)
    # the report and the constant use it
    wire, info = S.prepare(ws, [{"addr": "L1N7", "mode": "add", "dose": 0.75}], p50={"1": 2.0})
    e = S.describe(ws, wire, info)["entries"][0]
    assert e["norm"] == pytest.approx(float(W[:, 7].norm()), rel=1e-12) and e["value"] == pytest.approx(0.75 * 2.0 / e["norm"])
    # another patch on that column: the cached norm is dropped and read again
    ws.commit({DOWN1: [col(7, seed=9, scale=2.0)]}, "col 7 again")
    assert float(S.column_norms(ws, 1, [7])[0][0]) == pytest.approx(float(ws.read(DOWN1)[:, 7].double().norm()), rel=1e-12)
    # a candidate edit (extra) is part of the variant being run, and never cached
    extra = {DOWN1: [col(7, seed=5, scale=3.0)]}
    with_extra = float(ws.read(DOWN1, extra=extra)[:, 7].double().norm())
    assert float(S.column_norms(ws, 1, [7], extra=extra)[0][0]) == pytest.approx(with_extra, rel=1e-12)
    assert float(S.column_norms(ws, 1, [7])[0][0]) != pytest.approx(with_extra)


def test_resid_norms_on_a_quantized_store_with_a_multi_piece_patch(tmp_path):
    """The Q4_K fake store (reconstructed values, immutable base): a patched column's norm is base + every piece."""
    ws = quant_workspace(tmp_path / "q")
    name = ws.sem.mlp(1)["W_out"].name
    rows = ws.shape(name)[0]
    n = 5
    base = float(ws.read(name, variant="base")[:, n].double().norm())
    ws.commit({name: [D.block((range(0, rows), range(n, n + 1)), torch.full((rows, 1), 0.25))]}, "a")
    ws.commit({name: [D.block((range(0, rows // 2), range(n - 1, n + 2)), torch.full((rows // 2, 3), -0.5))]}, "b")
    assert len(ws.pieces(name)) == 2
    expect = float(ws.read(name)[:, n].double().norm())
    assert abs(expect - base) > 1e-3
    ws.store.reads.clear()
    assert float(S.column_norms(ws, 1, [n])[0][0]) == pytest.approx(expect, rel=1e-12)
    assert float(S.column_norms(ws, 1, [n], "base")[0][0]) == pytest.approx(base, rel=1e-12)
    # the base was read only for the columns asked for (the column-norm read plan), never the whole tensor
    reads = [k for nm, k in ws.store.reads if nm == name]
    assert reads and max(reads) < rows * ws.shape(name)[1]


# ------------------------------------------------------------------ the seams: selections as steering targets
def test_a_selection_too_large_to_patch_points_to_run_time_steering(tmp_path, monkeypatch):
    from wsp import selections as SEL
    ws = quant_workspace(tmp_path / "q")
    ws.edit_settings = {"patch_limit_gib": 100 * 4 / 1024**3}       # a Largest weight patch of 100 weights
    name = ws.sem.mlp(1)["W_out"].name
    acts = SEL.actions(ws, "blocks.1.mlp")                         # neurons: steering can scale them instead
    assert not acts["scale"]["enabled"] and "too large for a weight patch" in acts["scale"]["reason"]
    assert "scale it at run time in the Run tab's steering panel" in acts["scale"]["reason"]
    with pytest.raises(SEL.Refusal, match="Run tab's steering panel"):
        SEL.script(ws, "blocks.1.mlp", "scale", 2)
    plain = SEL.actions(ws, name)["scale"]["reason"]               # a tensor: no neuron or head to steer
    assert "too large for a weight patch" in plain and "steering" not in plain


def test_saved_selections_are_listed_with_the_neurons_they_push(tmp_path):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    from wsp.session import Session
    s = Session(str(tmp_path))
    (tmp_path / "q").mkdir()
    s.ws = quant_workspace(tmp_path / "q" / "ws")
    c = TestClient(create_app(s, "http://127.0.0.1:0"), raise_server_exceptions=False)
    r = c.post("/api/groups", json={"name": "probe picks", "neurons": [[1, 3], [1, 4], [0, 2]]})
    assert r.status_code == 200, r.text
    s.ws.set_group("mixed", ["L1N5", "L0H1", "group:probe picks"])
    s.ws.set_group("inner", ["L1N9"])
    s.ws.set_group("outer", ["group:inner", "L1N8"])
    s.ws.groups.pop("inner")                                       # outer's member no longer resolves
    g = c.get("/api/steer/groups").json()
    assert g["max_neurons"] == S.MAX_NEURONS
    by = {x["name"]: x for x in g["groups"]}
    assert by["probe picks"]["neurons"] == 3 and by["probe picks"]["addr"] == "group:probe picks" and by["probe picks"]["skipped"] == {}
    assert by["mixed"]["neurons"] == 4 and by["mixed"]["skipped"]["heads"] == 1 and by["mixed"]["layers"] == [0, 1]
    assert by["outer"]["neurons"] == 1 and by["outer"]["skipped"]["unresolved"] == 1
    # a name with a space is a steering target as given
    wire, info = S.prepare(s.ws, [{"addr": "group:probe picks", "mode": "mul", "dose": 2}])
    assert info["entries"][0]["neurons"] == 3


def test_a_selection_with_a_member_that_is_not_text_is_listed_and_steered_cleanly(tmp_path):
    """A hand-edited or old groups.json: a null (or number) member, or members that are not a list. The listing
    keeps every other selection, the bad member is an unresolved member with a reason, and resolve and a steered
    Run answer 200 or a plain 400, never an AttributeError."""
    from fastapi.testclient import TestClient
    from wsp import runner as R
    from wsp.server import create_app
    from wsp.session import Session
    s = Session(str(tmp_path))
    (tmp_path / "q").mkdir()
    s.ws = quant_workspace(tmp_path / "q" / "ws")
    c = TestClient(create_app(s, "http://127.0.0.1:0"), raise_server_exceptions=False)
    s.ws.set_group("good", ["L1N3"])
    s.ws.set_group("holey", ["L1N4", "L0N2"])
    s.ws.groups["holey"]["members"] += [None, 7]                   # as a hand edit could leave it
    s.ws.set_group("empty", ["L1N5"])
    s.ws.groups["empty"]["members"] = [None]
    s.ws.set_group("broken", ["L1N6"])
    s.ws.groups["broken"]["members"] = None
    s.ws.groups["junk"] = "L1N7"                                   # not even a record
    with pytest.raises(ValueError, match="'broken' does not hold a list of members"):
        s.ws.set_group("outer", ["group:holey", "group:broken", "L1N8"])        # a new member must resolve
    s.ws.set_group("outer", ["group:holey", "L1N8"])                             # holey's bad members are skipped
    s.ws.groups["outer"]["members"].append("group:broken")                       # as an old file could hold it
    r = c.post("/api/groups", json={"name": "later", "members": ["group:outer", "L0N1"]})   # the cycle walk too
    assert r.status_code == 200, r.text
    g = c.get("/api/steer/groups")
    assert g.status_code == 200, g.text
    by = {x["name"]: x for x in g.json()["groups"]}
    assert set(by) == {"good", "holey", "empty", "broken", "junk", "outer", "later"}
    assert by["good"]["neurons"] == 1 and "error" not in by["good"]
    sk = by["holey"]["skipped"]
    assert by["holey"]["neurons"] == 2 and sk["unresolved"] == 2
    assert [m["member"] for m in sk["members"]] == ["null", "7"]
    assert all(m["reason"] == "not an address: a selection member must be text" for m in sk["members"])
    assert by["empty"]["neurons"] == 0 and by["empty"]["skipped"]["unresolved"] == 1
    assert by["broken"]["error"] == "selection 'broken' does not hold a list of members"
    assert by["junk"]["error"] == "selection 'junk' does not hold a list of members"
    assert by["outer"]["neurons"] == 3 and by["outer"]["skipped"]["unresolved"] == 3   # holey's two, and broken

    r = c.post("/api/steer/resolve", json={"steer": [{"addr": "group:holey", "mode": "add", "dose": 1.0, "unit": "raw"}]})
    assert r.status_code == 200, r.text
    assert r.json()["entries"][0]["neurons"] == 2 and r.json()["entries"][0]["skipped"]["unresolved"] == 2
    built = []

    def refuse(w):
        built.append(w.name)
        raise AssertionError("the runner was built for a request that should have been refused first")
    saved, R.make_runner = R.make_runner, refuse
    try:
        for addr, message in [("group:empty", "group 'empty' has no neurons to push (it holds 1 member that no longer resolves)"),
                              ("group:broken", "selection 'broken' does not hold a list of members"),
                              ("group:junk", "selection 'junk' does not hold a list of members")]:
            spec = [{"addr": addr, "mode": "mul", "dose": 2}]
            for path, body in [("/api/steer/resolve", {"steer": spec}),
                               ("/api/run", {"prompt": "ab", "max_new_tokens": 2, "steer": spec})]:
                r = c.post(path, json=body)
                assert r.status_code == 400 and message in r.json()["error"], (path, addr, r.text)
    finally:
        R.make_runner = saved
    assert built == []


def test_the_range_form_the_recipe_hint_gives_resolves_on_the_server(tmp_path):
    """The recipe parser's hint for "L1 N1-3 +0.5" is blocks.1.mlp.neurons.1-3 +0.5: the server pushes those neurons."""
    ws = quant_workspace(tmp_path / "q")
    wire, entries = S.normalize(ws, [{"addr": "blocks.1.mlp.neurons.1-3", "mode": "add", "dose": 0.5, "unit": "raw"}])
    assert wire["layers"] == {"1": {"add_raw": [[1, 2, 3], [0.5, 0.5, 0.5]]}}
    assert entries[0]["neurons"] == 3
