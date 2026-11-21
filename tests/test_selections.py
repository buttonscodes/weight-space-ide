"""Inspector selections: one server-built script per action, for every kind of selection, on float and
quantized-looking workspaces; named selections (groups) of thousands of neurons; breadcrumbs; before/after.

Model-free. The float cases use a tiny random model of this project's kind (as tests/test_atlas.py does); the
quantized cases use tests/selection_fixtures.QuantStore, an in-memory store that reports Q4_K storage and answers
reads with fixed float values the way a GGUF store answers with reconstructed values. No real quantized file,
decoder or model is used, and nothing is run."""
import os
import sys
import time

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from fastapi.testclient import TestClient  # noqa: E402

import selection_fixtures as F  # noqa: E402
from selection_fixtures import quant_workspace  # noqa: E402
from wsp import hierarchy  # noqa: E402
from wsp import selections as S  # noqa: E402
from wsp.model import Config, init_params  # noqa: E402
from wsp.server import create_app  # noqa: E402
from wsp.session import Session  # noqa: E402
from wsp.workspace import Workspace  # noqa: E402


@pytest.fixture()
def rnd(tmp_path):
    from wsp.tokenizer import VocabTokenizer
    tok = VocabTokenizer(list("abcdefghijklmnopqrstuvwxyz"), "char")
    cfg = Config(d_vocab=26, n_ctx=16, d_model=24, n_layers=2, n_heads=2, d_mlp=40, name="rnd")
    Workspace.create(str(tmp_path / "rnd"), cfg, tok, init_params(cfg, seed=1, std=0.3))
    s = Session(str(tmp_path))
    s.open("rnd")
    return s


def quant_session(tmp_path, **kw):
    d = tmp_path / "quant"
    d.mkdir()
    s = Session(str(tmp_path))
    s.ws = quant_workspace(str(d), **kw)
    return s


@pytest.fixture()
def quant(tmp_path):
    return quant_session(tmp_path)


def client(sess):
    return TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)


def W(ws, name, variant="current"):
    return ws.read(name, (), variant).clone()


def apply(sess, script):
    r = sess.apply(script, "test")
    assert r.get("ok") and r.get("committed"), r
    return r


# ------------------------------------------------------------------ the reported case: a slice of a quantized tensor

def test_scaling_a_slice_of_a_quantized_tensor_changes_exactly_that_slice(quant):
    """`model.layers.1.mlp.down_proj.weight[:, 7]` selected, scale x: one patch, only that column moved."""
    ws = quant.ws
    name = "model.layers.1.mlp.down_proj.weight"
    assert ws.store.dtype(name) == "Q4_K" and ws.editable(name)
    acts = S.actions(ws, f"{name}[:, 7]")
    assert acts["scale"]["enabled"] and acts["ablate"]["enabled"] and acts["console"]["enabled"]
    assert not acts["revert"]["enabled"]
    before = W(ws, name)
    r = S.script(ws, f"{name}[:, 7]", "scale", 2.0)
    assert r["weights"] == 16 and r["neurons"] == 0 and not r["other"]
    apply(quant, r["script"])
    after = W(ws, name)
    assert torch.allclose(after[:, 7], 2 * before[:, 7])
    mask = torch.ones_like(before, dtype=torch.bool)
    mask[:, 7] = False
    assert torch.equal(after[mask], before[mask])
    assert torch.equal(W(ws, name, "base"), before)               # the immutable base is untouched
    assert len(ws.patches) == 1
    assert S.actions(ws, f"{name}[:, 7]")["revert"]["enabled"]


@pytest.mark.parametrize("which", ["rnd", "quant"])
def test_every_kind_of_selection_gets_an_enabled_action_that_changes_weights(request, which):
    """Tensor, slice, neuron, head, neuron range, group: enabled means the script changes weights, and the count it
    reports is exactly the weights it moves (each weight once, however the members overlap)."""
    sess = request.getfixturevalue(which)
    ws = sess.ws
    st = ws.struct
    out1 = st.layers[1].mlp["W_out"].name
    o0 = st.layers[0].attn["W_O"].name
    ws.set_group("mixed", ["L1N3", "blocks.0.mlp.neurons.4-9", "L0H1"])
    addrs = [out1, f"{out1}[0:3]" if which == "rnd" else f"{out1}[:, 2:5]", "L1N3", "L0H1",
             "blocks.0.mlp.neurons.4-9", "group:mixed", o0]
    for addr in addrs:
        acts = S.actions(ws, addr)
        assert acts["scale"]["enabled"], (addr, acts)
        before = {n: W(ws, n) for n in ws.store.names()}
        r = S.script(ws, addr, "scale", 0.5)
        res = sess.apply(r["script"], f"scale {addr}")
        assert res.get("ok") and res.get("committed"), (addr, r["script"], res)
        moved = sum(int((W(ws, n) != before[n]).sum()) for n in before)
        assert moved == r["weights"] > 0, (addr, moved, r["weights"])


def test_neurons_scale_what_they_write_and_heads_what_they_write(rnd):
    ws = rnd.ws
    st = ws.struct
    out = st.layers[1].mlp["W_out"].name
    w_in = st.layers[1].mlp["W_in"].name
    before_out, before_in = W(ws, out), W(ws, w_in)
    r = S.script(ws, "L1N3", "scale", 3.0)
    assert r["script"] == "scale_neurons(1, [3], 3.0)" and r["neurons"] == 1
    apply(rnd, r["script"])
    idx = st.part_index(st.layers[1].mlp["W_out"], neuron=3)
    assert torch.allclose(W(ws, out)[idx], 3 * before_out[idx])
    assert torch.equal(W(ws, w_in), before_in)                    # what the neuron reads is unchanged
    r = S.script(ws, "L0H1", "ablate")
    assert r["script"] == "scale_head((0, 1), 0.0)" and r["heads"] == 1
    apply(rnd, r["script"])
    o = st.layers[0].attn["W_O"]
    assert not W(ws, o.name)[st.part_index(o, head=1)].any()
    assert W(ws, o.name)[st.part_index(o, head=0)].any()


def test_bulk_scale_neurons_matches_one_at_a_time(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = quant_session(tmp_path / "a"), quant_session(tmp_path / "b")
    ns = [0, 1, 2, 5, 9, 10, 11, 23, 5]                          # unordered, one duplicate
    apply(a, f"scale_neurons(1, {ns}, 0.25)")
    apply(b, "\n".join(f"scale_neuron((1, {n}), 0.25)" for n in sorted(set(ns))))
    for n in a.ws.store.names():
        assert torch.allclose(W(a.ws, n), W(b.ws, n)), n


def test_ranges_kvgroups_and_whole_tensors(quant):
    ws = quant.ws
    r = S.script(ws, "blocks.0.mlp.neurons.4-9", "scale", 2)
    assert r["script"] == "scale_neurons(0, range(4, 10), 2.0)" and r["neurons"] == 6
    r = S.script(ws, "blocks.0.attn.kvgroup.1", "ablate")
    assert r["heads"] == 2 and r["script"].count("scale_head") == 2
    r = S.script(ws, "model.norm.weight", "scale", 2)
    assert r["script"] == 'P("model.norm.weight")[:] *= 2.0'
    apply(quant, r["script"])
    assert torch.allclose(W(ws, "model.norm.weight"), torch.full((16,), 2.0))


def test_refusals_say_why_in_plain_words(quant):
    ws = quant.ws
    with pytest.raises(S.Refusal, match="nothing to revert"):
        S.script(ws, "L1N3", "revert")
    with pytest.raises(S.Refusal, match="Scaling by 1 changes nothing"):
        S.script(ws, "L1N3", "scale", 1)
    with pytest.raises(S.Refusal, match="number"):
        S.script(ws, "L1N3", "scale", "abc")
    apply(quant, S.script(ws, "L1N3", "ablate")["script"])
    with pytest.raises(S.Refusal, match="already zero"):
        S.script(ws, "L1N3", "ablate")
    assert not S.actions(ws, "L1N4")["revert"]["enabled"]
    assert S.actions(ws, "L1N4")["revert"]["reason"] == "nothing here has been edited"


def test_a_tensor_that_cannot_be_edited_is_disabled_with_its_reason(quant):
    ws = quant.ws
    name = "model.layers.0.mlp.down_proj.weight"
    ws.store.kind = "Q4_K"
    real = ws.editable
    ws.editable = lambda n: False if n == name else real(n)
    acts = S.actions(ws, name)
    assert not acts["scale"]["enabled"] and "cannot be edited" in acts["scale"]["reason"]
    assert not S.actions(ws, "L0N3")["scale"]["enabled"]
    with pytest.raises(S.Refusal, match="cannot be edited"):
        S.script(ws, "L0N3", "scale", 2)
    assert S.actions(ws, "L1N3")["scale"]["enabled"]


def test_size_limit_refuses_with_plain_words(quant, monkeypatch):
    # The fixed 60,000,000-weight LIMIT is gone: the limit is the Largest weight patch setting (here 100 weights),
    # memory and disk (wsp.edit_limit), and the refusal names the one that binds.
    from wsp import edit_limit as EL
    ws = quant.ws
    ws.edit_settings = {"patch_limit_gib": 100 * EL.PATCH_BYTES / EL.GiB}
    acts = S.actions(ws, "model.layers.0.mlp.down_proj.weight")
    assert not acts["scale"]["enabled"] and "too large for a weight patch" in acts["scale"]["reason"]
    assert "Largest weight patch setting allows 100 weights" in acts["scale"]["reason"]
    with pytest.raises(S.Refusal, match="too large for a weight patch"):
        S.script(ws, "model.layers.0.mlp.down_proj.weight", "scale", 2)
    msg = S._too_large(267_386_880, EL.over_setting(ws, 267_386_880))
    assert "267,386,880 weights" in msg and "1.0 GiB" in msg and "too large for a weight patch" in msg


# ------------------------------------------------------------------ revert

@pytest.mark.parametrize("which", ["rnd", "quant"])
def test_revert_returns_selected_weights_to_base_as_one_new_patch(request, which):
    sess = request.getfixturevalue(which)
    ws = sess.ws
    base = {n: W(ws, n, "base") for n in ws.store.names()}
    apply(sess, S.script(ws, "L1N3", "scale", 2)["script"])
    apply(sess, S.script(ws, "L1N4", "scale", 3)["script"])
    apply(sess, S.script(ws, "L0H1", "scale", 0.5)["script"])
    n_patches = len(ws.patches)
    r = S.script(ws, "L1N3", "revert")
    assert r["script"] == "revert_neurons(1, [3])"
    apply(sess, r["script"])
    assert len(ws.patches) == n_patches + 1
    st = ws.struct
    for name, idx in st.neuron_slices(1, 3):
        assert torch.allclose(ws.read(name, idx), base[name][idx], atol=1e-6)
    out = st.layers[1].mlp["W_out"]
    i4 = st.part_index(out, neuron=4)
    assert torch.allclose(ws.read(out.name)[i4], 3 * base[out.name][i4])     # L1N4 keeps its edit
    apply(sess, S.script(ws, "L0H1", "revert")["script"])
    o = st.layers[0].attn["W_O"]
    assert torch.allclose(ws.read(o.name), base[o.name], atol=1e-6)
    assert not S.actions(ws, "L1N3")["revert"]["enabled"]


def test_base_is_readable_from_a_script(quant):
    ws = quant.ws
    apply(quant, "scale_neuron((1, 2), 5)")
    r = quant.apply("x = neuron(1, 2)\nprint(float((base(x.w_out) - x.w_out).abs().sum()) > 0)", dry_run=True)
    assert r["ok"] and r["output"] == "True" and not r["committed"], r


# ------------------------------------------------------------------ thousands of neurons

def test_thousands_of_neurons_are_one_patch_stored_without_overhead(tmp_path):
    """9000 scattered neurons of an 18000-wide layer: one patch holding exactly the changed weights, the base
    read about once, and the revert exact."""
    sess = quant_session(tmp_path, d_model=16, d_mlp=8, n_layers=2, wide_layer=1, wide_mlp=18000)
    ws = sess.ws
    out = "model.layers.1.mlp.down_proj.weight"
    before = W(ws, out)
    ns = list(range(0, 18000, 2))
    ws.set_group("half", S.members_from_neurons([(1, n) for n in ns]))
    assert S.neurons_of(ws, "half") == [(1, n) for n in ns]
    t = time.time()
    r = S.script(ws, "group:half", "scale", 0.5)
    assert r["neurons"] == 9000 and r["weights"] == 9000 * 16
    ws.store.reads.clear()
    apply(sess, r["script"])
    elapsed = time.time() - t
    after = W(ws, out)
    assert torch.allclose(after[:, ns], 0.5 * before[:, ns]) and torch.equal(after[:, 1::2], before[:, 1::2])
    assert len(ws.patches) == 1
    read = sum(n for name, n in ws.store.reads if name == out)
    assert read <= 2 * before.numel(), read                       # the store was read about once, not per neuron
    stored = sum(p.numel() for p, _ in ws.pieces(out))
    assert stored == 9000 * 16, stored
    assert elapsed < 30, elapsed
    n, _ = S.changed(ws, "group:half")
    assert n == 9000 * 16
    apply(sess, S.script(ws, "group:half", "revert")["script"])
    assert torch.allclose(W(ws, out), before, atol=1e-6)
    assert not S.actions(ws, "group:half")["revert"]["enabled"]


def test_neurons_of_expands_ranges_and_skips_other_members(rnd):
    ws = rnd.ws
    ws.set_group("g", ["L1N3", "blocks.0.mlp.neurons.4-6", "L0H1", "blocks.1.mlp.neuron.3"])
    assert S.neurons_of(ws, "g") == [(0, 4), (0, 5), (0, 6), (1, 3)]
    assert S.neurons_of(ws, "group:g") == S.neurons_of(ws, "g")


def test_members_are_stored_as_runs():
    assert S.members_from_neurons([(0, 3), (0, 4), (0, 5), (1, 7), (0, 9)]) == ["blocks.0.mlp.neurons.3-5", "L0N9", "L1N7"]


# ------------------------------------------------------------------ server routes

def test_selection_script_route_and_refusal(quant):
    c = client(quant)
    r = c.post("/api/selection/script", json={"addr": "model.layers.1.mlp.down_proj.weight[:, 7]", "action": "scale", "factor": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"script", "neurons", "heads", "weights", "other"}
    e = c.post("/api/edit", json={"script": body["script"], "message": "scale"}).json()
    assert e["ok"] and e["committed"], e
    r = c.post("/api/selection/script", json={"addr": "L1N4", "action": "revert"})
    assert r.status_code == 400 and r.json()["refused"] and "nothing to revert" in r.json()["error"]
    assert "Traceback" not in r.text


def test_node_reports_actions_crumbs_and_jumps_for_a_slice(quant):
    c = client(quant)
    n = c.get("/api/node", params={"addr": "model.layers.1.mlp.down_proj.weight[:, 7]"}).json()
    assert n["actions"]["scale"]["enabled"], n["actions"]
    assert n["slices"][0]["dtype"] == "Q4_K" and n["slices"][0]["editable"]
    assert [x["label"] for x in n["crumbs"]] == ["model", "block 1", "mlp", "mlp.down_proj.weight", "[:, 7]"]
    assert n["jumps"][0]["addr"] == "blocks.1.mlp.neuron.7" and n["jumps"][0]["label"] == "L1N7"
    u = c.get("/api/node", params={"addr": "L1N7"}).json()
    assert "model.layers.1.mlp.down_proj.weight[:, 7]" in [j["addr"] for j in u["jumps"]]
    assert u["selection"]["neurons"] == 1


def test_node_still_answers_when_the_structure_cannot_describe_it(quant, monkeypatch):
    """A structure that cannot describe an address must not fail the node: actions fail closed with a reason."""
    c = client(quant)
    def broken(*a, **kw):
        raise AttributeError("'Struct' object has no attribute 'shapes'")
    monkeypatch.setattr(S, "plan", broken)
    monkeypatch.setattr(hierarchy, "crumbs", broken)
    r = c.get("/api/node", params={"addr": "L1N7"})
    assert r.status_code == 200, r.text
    n = r.json()
    assert all(not n["actions"][k]["enabled"] for k in ("scale", "ablate", "console", "revert"))
    assert "Cannot work out what this selection edits" in n["actions"]["scale"]["reason"]
    assert n["crumbs"] == [] and n["jumps"]


def test_patch_delete_twice_is_not_an_error(quant):
    c = client(quant)
    apply(quant, "scale_neuron((1, 2), 2)")
    pid = quant.ws.patches[0].id
    r1 = c.post("/api/patch", json={"action": "delete", "id": pid})
    r2 = c.post("/api/patch", json={"action": "delete", "id": pid})
    assert r1.status_code == 200 and r1.json()["ok"] and not r1.json()["missing"]
    assert r2.status_code == 200 and r2.json()["ok"] and r2.json()["missing"] and r2.json()["patches"] == []
    r3 = c.post("/api/patch", json={"action": "set", "id": pid, "enabled": False})
    assert r3.status_code == 404 and r3.json()["missing"] and "no longer exists" in r3.json()["error"]
    assert "KeyError" not in r3.text


def test_groups_route_accepts_neuron_lists_and_merges(quant):
    c = client(quant)
    r = c.post("/api/groups", json={"name": "picked", "neurons": [[1, n] for n in range(0, 20)] + ["L0N3"]}).json()
    assert r["groups"]["picked"]["members"] == ["L0N3", "blocks.1.mlp.neurons.0-19"]
    assert r["summary"]["neurons"] == 21 and r["summary"]["layers"] == [0, 1]
    r = c.post("/api/groups", json={"name": "picked", "mode": "add", "neurons": [[1, 20], [1, 21]], "members": ["L0H1"]}).json()
    assert r["groups"]["picked"]["members"] == ["L0N3", "blocks.1.mlp.neurons.0-21", "L0H1"]
    assert r["summary"]["heads"] == 1
    assert c.post("/api/groups", json={"name": "old", "members": ["L1N3"]}).json()["groups"]["old"]["members"] == ["L1N3"]
    g = c.get("/api/node", params={"addr": "group:picked"}).json()
    assert g["selection"]["neurons"] == 23 and g["actions"]["scale"]["enabled"]
    s = c.post("/api/selection/script", json={"addr": "group:picked", "action": "ablate"}).json()
    assert s["neurons"] == 23 and s["heads"] == 1
    e = c.post("/api/edit", json={"script": s["script"]}).json()
    assert e["ok"] and e["committed"] and len(quant.ws.patches) == 1


def test_compare_shows_base_current_and_delta(quant):
    c = client(quant)
    ws = quant.ws
    name = "model.layers.1.mlp.down_proj.weight"
    apply(quant, S.script(ws, f"{name}[:, 7]", "scale", 2)["script"])
    ident = c.get("/api/state").json()["status"]
    m = c.get("/api/matrix/compare", params={"addr": name, "workspace_id": ident.get("workspace_id", ""),
                                               "version": ident.get("version", ws.version)})
    assert m.status_code == 200, m.text
    m = m.json()
    import base64
    dec = lambda s: np.frombuffer(base64.b64decode(s), dtype=np.float32).reshape(m["grid"])  # noqa: E731
    b, cur, d = dec(m["base"]), dec(m["current"]), dec(m["delta"])
    base = W(ws, name, "base").numpy()
    assert np.allclose(b, base) and np.allclose(cur[:, 7], 2 * base[:, 7]) and np.allclose(d, cur - b)
    assert m["edited"] and m["stats"]["delta"]["changed"] == 16
    assert m["stats"]["base"]["norm"] != m["stats"]["current"]["norm"]
    assert m["scale"] > 0 and m["delta_scale"] > 0


def test_breadcrumbs_follow_the_tree(quant):
    ws = quant.ws
    labels = lambda a: [c["label"] for c in hierarchy.crumbs(ws, a)]  # noqa: E731
    assert labels("L1N7") == ["model", "block 1", "mlp", "neuron 7"]
    assert labels("model.norm.weight") == ["model", "final norm", "model.norm.weight"]
    assert labels("group:x") == ["model", "selection x"]
    for c in hierarchy.crumbs(ws, "model.layers.0.self_attn.o_proj.weight[:, 4:8]"):
        hierarchy.kind_of(ws, c["addr"])                         # every crumb is an address the tree can open
    assert hierarchy.jumps(ws, "model.layers.0.self_attn.o_proj.weight[:, 4:8]")[0]["addr"] == "blocks.0.attn.head.1"
    assert hierarchy.jumps(ws, "model.layers.0.self_attn.k_proj.weight[0:4]")[0]["addr"] == "blocks.0.attn.kvgroup.0"


def test_groups_route_saves_an_atlas_selector_on_the_server(rnd):
    """Atlas Find results are saved by their selector; the server expands it (weights-only atlas, no run)."""
    from wsp import atlas
    atlas.build(rnd.ws, stages=["weights"], log=lambda *a: None, variant="current")
    c = client(rnd)
    found = c.post("/api/atlas/select", json={"selector": {"layers": [0, 1], "top": 7}}).json()
    r = c.post("/api/groups", json={"name": "found", "selector": {"layers": [0, 1], "top": 7}})
    assert r.status_code == 200, r.text
    r = r.json()
    assert r["summary"]["neurons"] == found["n"] == 7
    got = S.neurons_of(rnd.ws, "found")
    assert sorted((x["layer"], x["neuron"]) for x in found["rows"]) == got
    assert c.post("/api/groups", json={"name": "", "neurons": [[0, 1]]}).status_code == 400


# ------------------------------------------------------------------------------ GGUF runner: edits on every forward pass
def _dense(pieces, shape):
    """The summed delta of (piece, scale) pairs, materialised (the reference)."""
    from wsp import delta as D
    from wsp.store import norm_index
    out = torch.zeros(shape)
    for p, s in pieces:
        D.add_to(out, norm_index((), shape), p, s)
    return out


def _mixed_pieces(R=12, C=40, seed=0):
    """Full-height column blocks (some overlapping), partial blocks, an int row and column, a stepped
    range, a descending range, low-rank and sparse pieces, with strengths."""
    from wsp import delta as D
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    full = range(0, R)
    ps = [(D.block((full, range(a, b)), rn(R, b - a)), s)
          for (a, b), s in zip([(0, 3), (5, 6), (7, 11), (2, 4), (20, 21), (30, 38), (36, 40)], [1.0, 0.5, 1.0, 2.0, 1.0, -1.0, 1.0])]
    ps += [(D.block((range(2, 5), range(10, 14)), rn(3, 4)), 1.0),
           (D.block((3, range(0, C)), rn(C)), 0.7),
           (D.block((full, 17), rn(R)), 1.0),
           (D.block((range(0, R, 2), range(1, 9, 3)), rn(6, 3)), 1.0),
           (D.block((range(R - 1, -1, -1), range(25, 27)), rn(R, 2)), 1.0),
           (D.lowrank((full, range(0, C)), rn(R, 2), rn(2, C)), 0.3),
           (D.Piece("sparse", (full, range(0, C)), values=rn(5),
                    coords=torch.tensor([[0, 0], [11, 39], [4, 20], [4, 20], [7, 3]])), 1.0),
           (D.block((full, range(8, 9)), rn(R, 1)), 0.0)]          # strength 0: no effect
    return ps


@pytest.mark.parametrize("shape", [(40,), (1, 40), (1, 5, 40), (2, 3, 40)])
def test_gguf_overlay_equals_the_per_piece_path_and_the_dense_delta(shape):
    from wsp.gguf_runner import DeltaOverlay, delta_matmul
    R, C = 12, 40
    ps = _mixed_pieces(R, C)
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(3))
    ref = x @ _dense(ps, (R, C)).T
    old = torch.zeros(*shape[:-1], R)
    for p, s in ps:
        delta_matmul(x, p, s, old)
    ov = DeltaOverlay(ps, R)
    # the 8 full-height blocks (24 columns) and the sparse piece's 5 values, scaled once into a copy
    assert ov.n_gathered == 8 and ov.nbytes == R * 24 * 4 + 24 * 8 + 5 * 4
    new = torch.zeros(*shape[:-1], R)
    ov.apply(x, new)
    assert torch.allclose(old, ref, atol=1e-5) and torch.allclose(new, ref, atol=1e-5)
    assert float((new - old).abs().max()) < 1e-5


def test_gguf_packed_linear_keeps_its_overlay_while_the_pieces_stay_the_same():
    """Committed pieces and the job's extra get separate overlays: alternating base, current and a candidate
    reuses them; a new piece list (a commit) rebuilds; an empty one drops the copy."""
    from types import SimpleNamespace
    from wsp.gguf_runner import PackedLinear
    from wsp import delta as D
    R, C = 12, 40
    committed = _mixed_pieces(R, C)
    extra = {"W": [p for p, _ in _mixed_pieces(R, C, seed=1)]}
    ws = SimpleNamespace(shape=lambda n: (R, C), pieces=lambda n: committed)
    runner = SimpleNamespace(ws=ws, _variant="current", _extra=None)
    lin = PackedLinear(runner, "W")
    a = lin.overlays()
    assert len(a) == 1 and lin.overlays()[0] is a[0]
    runner._extra = extra
    b = lin.overlays()
    assert len(b) == 2 and b[0] is a[0]
    x = torch.randn(2, C)
    out = torch.zeros(2, R)
    for o in b:
        o.apply(x, out)
    want = x @ (_dense(committed, (R, C)) + _dense([(p, 1.0) for p in extra["W"]], (R, C))).T
    assert torch.allclose(out, want, atol=1e-5)
    runner._extra, runner._variant = extra, "base"
    runner._variant = "current"
    assert lin.overlays()[1] is b[1]                                  # same candidate: not rebuilt
    committed = committed + [(D.block((range(0, R), range(39, 40)), torch.ones(R, 1)), 1.0)]
    assert lin.overlays()[0] is not a[0]                               # a commit: rebuilt
    committed = ()
    runner._extra = None
    assert lin.overlays() == [] and lin.overlay_bytes() == 0


def test_gguf_overlay_cost_for_thousands_of_neurons():
    """9,000 scattered neurons of one 27B-shaped down projection (5120 x 17408), 4,371 pieces: the per-forward
    cost is one gather and one matmul. Before, each piece built its row index from a Python list (860 ms)."""
    from wsp import delta as D
    from wsp.gguf_runner import DeltaOverlay
    R, C = 5120, 17408
    cols = np.sort(np.random.default_rng(0).choice(C, 9000, replace=False))
    runs = S.runs(cols)
    vals = torch.randn(R, len(cols))
    ps, at = [], 0
    for a, b in runs:
        ps.append((D.block((range(R), range(a, b)), vals[:, at:at + b - a]), 1.0))
        at += b - a
    t = time.perf_counter()
    ov = DeltaOverlay(ps, R)
    build = time.perf_counter() - t
    x = torch.randn(1, 1, C)
    out = torch.zeros(1, 1, R)
    ov.apply(x, out)
    t = time.perf_counter()
    for _ in range(5):
        ov.apply(x, out)
    per = (time.perf_counter() - t) / 5
    assert len(ps) == 4371 and ov.n_gathered == 4371 and ov.nbytes == R * 9000 * 4 + 9000 * 8
    assert per < 0.1, f"{per * 1e3:.1f} ms per forward"
    ref = torch.zeros(1, 1, R)
    ref += x[..., torch.as_tensor(cols)] @ vals.T
    assert torch.allclose(out - ref * 6, torch.zeros_like(out), atol=1e-2)
    print(f"overlay build {build * 1e3:.0f} ms, apply {per * 1e3:.1f} ms")


def test_gguf_overlay_gathers_stepped_and_descending_column_blocks_and_row_blocks():
    """Full-height column blocks with any step or direction, and full-width row blocks (a neuron's read row,
    a head's query rows), are gathered; the result is the dense delta's either way."""
    from wsp import delta as D
    from wsp.gguf_runner import DeltaOverlay
    R, C = 12, 40
    g = torch.Generator().manual_seed(5)
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    cols = [range(0, 40, 7), range(9, 3, -2), range(39, 30, -1), 13, range(20, 23)]
    rows = [range(0, 12, 5), range(11, 6, -2), 4, range(5, 7), range(1, 2)]
    ps = [(D.block((range(R), c), rn(R) if isinstance(c, int) else rn(R, len(c))), s) for c, s in zip(cols, [1.0, 2.0, -0.5, 1.0, 0.3])]
    ps += [(D.block((r, range(C)), rn(C) if isinstance(r, int) else rn(len(r), C)), s) for r, s in zip(rows, [1.0, 0.5, 2.0, 1.0, -1.0])]
    ov = DeltaOverlay(ps, R, C)
    k_cols, k_rows = 6 + 3 + 9 + 1 + 3, 3 + 3 + 1 + 2 + 1
    assert ov.n_gathered == 10 and ov.n_rest == 0
    assert ov.cols.tolist()[:9] == [0, 7, 14, 21, 28, 35, 9, 7, 5] and ov.rows.tolist()[:6] == [0, 5, 10, 11, 9, 7]
    assert ov.nbytes == (R * k_cols + k_rows * C) * 4 + (k_cols + k_rows) * 8
    for shape in [(C,), (1, 3, C)]:
        x = rn(*shape)
        out = torch.zeros(*shape[:-1], R)
        ov.apply(x, out)
        assert torch.allclose(out, x @ _dense(ps, (R, C)).T, atol=1e-5)
    assert DeltaOverlay(ps, R).n_gathered == 5                       # without the input width: columns only


def test_gguf_overlay_is_applied_in_the_outputs_dtype():
    """A half-precision output gets a half-precision overlay (built in its dtype by PackedLinear), and an
    overlay of another dtype still adds into it; float32 was assumed before (index_add_ refused a half out)."""
    from types import SimpleNamespace
    from wsp.gguf_runner import DeltaOverlay, PackedLinear
    R, C = 12, 40
    ps = _mixed_pieces(R, C)
    x = torch.randn(2, C, generator=torch.Generator().manual_seed(4))
    want = x.double() @ _dense(ps, (R, C)).double().T
    lin = PackedLinear(SimpleNamespace(ws=SimpleNamespace(shape=lambda n: (R, C), pieces=lambda n: ps), _extra=None), "W")
    for dtype in (torch.float16, torch.bfloat16):
        out = torch.zeros(2, R, dtype=dtype)
        (ov,) = lin.overlays(out)
        assert ov.dtype == dtype and ov.W.dtype == dtype and lin.overlays(out)[0] is ov
        ov.apply(x.to(dtype), out)
        assert out.dtype == dtype and float((out.double() - want).abs().max()) < 0.1 * float(want.abs().max())
    out = torch.zeros(2, R, dtype=torch.float16)
    DeltaOverlay(ps, R, C).apply(x, out)                             # a float32 overlay into a half output
    assert float((out.double() - want).abs().max()) < 0.02 * float(want.abs().max())
    assert lin.overlays()[0].dtype == torch.float32                  # another dtype: another overlay


def test_gguf_overlay_follows_a_candidate_list_extended_in_place():
    """A weight script appends to the same `extra` list between two logits() calls (Interp._add); the
    candidate overlay is keyed on the pieces, so the second call sees both edits."""
    from types import SimpleNamespace
    from wsp import delta as D
    from wsp.gguf_runner import PackedLinear
    R, C = 12, 40
    runner = SimpleNamespace(ws=SimpleNamespace(shape=lambda n: (R, C), pieces=lambda n: ()), _variant="current", _extra=None)
    lin = PackedLinear(runner, "W")
    lst = [D.block((range(R), range(0, 1)), torch.ones(R, 1))]
    runner._extra = {"W": lst}
    x = torch.ones(1, C)

    def apply():
        out = torch.zeros(1, R)
        for o in lin.overlays(out):
            o.apply(x, out)
        return out
    assert apply()[0, 0] == 1
    first = lin._extra[5]
    lst.append(D.block((range(R), range(1, 2)), 2 * torch.ones(R, 1)))
    assert apply()[0, 0] == 3 and lin._extra[5] is not first
    lst[1] = D.block((range(R), range(1, 2)), 5 * torch.ones(R, 1))          # replaced in place: rebuilt too
    assert apply()[0, 0] == 6


def test_gguf_overlays_of_unchanged_tensors_survive_a_rebuild():
    """`Workspace.rebuild` makes a new piece list for every tensor after any toggle; a tensor whose pieces
    and strengths are the same keeps its overlay, one whose strength changed gets a new one."""
    from types import SimpleNamespace
    from wsp.gguf_runner import PackedLinear
    R, C = 12, 40
    state = {"W": _mixed_pieces(R, C), "V": _mixed_pieces(R, C, seed=2)}
    runner = SimpleNamespace(ws=SimpleNamespace(shape=lambda n: (R, C), pieces=lambda n: state[n]), _variant="current", _extra=None)
    w, v = PackedLinear(runner, "W"), PackedLinear(runner, "V")
    ow, ovv = w.overlays()[0], v.overlays()[0]
    state = {"W": list(state["W"]), "V": [(p, s * 2) for p, s in state["V"]]}      # a toggle: new lists
    assert w.overlays()[0] is ow and v.overlays()[0] is not ovv
    assert w._committed[0] is state["W"]                                           # adopted: no key next pass


def _bare_gguf_runner(ws):
    """A GGUFRunner with no model, kernel or lease: only what use(), release and reporting touch."""
    from wsp.gguf_runner import GGUFRunner
    r = object.__new__(GGUFRunner)
    r.ws, r._variant, r._extra = ws, "base", None
    r._saved, r._gain, r._named, r._packed = [], {}, {}, []
    r.residency = dict(overlay_bytes=0)
    return r


def test_gguf_runner_drops_a_jobs_overlays_and_a_deleted_patchs_copy(tmp_path):
    """use() without an extra (the worker's _drop_extra calls use('base') after a job) clears the candidate
    overlay of every matrix; deleting or disabling a patch drops its committed copy at once (rebuild calls the
    runner); what the overlays hold is reported in the runner's residency."""
    from wsp import delta as D
    from wsp.gguf_runner import PackedLinear
    ws = quant_workspace(str(tmp_path / "q"))
    R, C = ws.shape(D0)
    r = _bare_gguf_runner(ws)
    ws._runner = r
    try:
        lins = [PackedLinear(r, D0), PackedLinear(r, D1)]
        assert r._packed == lins
        cols = [range(i, i + 1) for i in range(0, 8, 2)]
        patch = ws.commit({D0: [D.block((range(R), c), torch.ones(R, 1)) for c in cols]}, "four neurons")
        extra = {n: [D.block((range(R), c), torch.ones(R, 1)) for c in cols] for n in (D0, D1)}
        r.use("current", extra)
        for lin in lins:
            lin.overlays()
        held = R * 4 * 4 + 4 * 8                                         # one gathered [R, 4] copy + its index
        assert lins[0].overlay_bytes() == 2 * held and lins[1].overlay_bytes() == held
        assert r.overlay_bytes() == 3 * held and r.residency["overlay_bytes"] == 3 * held
        r.use("base")
        assert all(lin._extra is None for lin in lins) and lins[0]._committed is not None
        assert r.residency["overlay_bytes"] == held
        ws.set_patch(patch.id, strength=2.0)                              # new strength: the copy is stale
        assert lins[0]._committed is None and r.residency["overlay_bytes"] == 0
        r.use("current")
        lins[0].overlays()
        ws.delete_patch(patch.id)
        assert lins[0]._committed is None and r.overlay_bytes() == 0
        patch = ws.commit({D1: [D.block((range(R), range(0, 1)), torch.ones(R, 1))]}, "another tensor")
        r.use("current")
        lins[1].overlays()
        kept = lins[1]._committed
        ws.commit({D0: [D.block((range(R), range(0, 1)), torch.ones(R, 1))]}, "unrelated")
        assert lins[1]._committed is kept                                 # its pieces are still the active ones
    finally:
        ws._runner = None


def test_gguf_packed_linear_forward_adds_overlays_only_off_base():
    """The base variant is the packed product alone; current adds the committed pieces, and a candidate its
    extra too. A fake kernel computes x @ base.T from the raw bytes; no model or ggml is used."""
    from types import SimpleNamespace
    from wsp.gguf_runner import PackedLinear
    R, C = 12, 40
    base = torch.randn(R, C, generator=torch.Generator().manual_seed(9))
    committed = _mixed_pieces(R, C)
    extra = [p for p, _ in _mixed_pieces(R, C, seed=1)]
    store = SimpleNamespace(native=lambda n: base.numpy(), where={"W": SimpleNamespace(tensor_type=0)},
                            column_permutations={}, row_permutations={})

    def matmul(raw, qtype, shape, x):
        assert raw.dtype == np.uint8 and tuple(shape) == (R, C)
        return x @ torch.from_numpy(raw.reshape(-1).view(np.float32).reshape(R, C).copy()).T
    ws = SimpleNamespace(shape=lambda n: (R, C), pieces=lambda n: committed, store=store)
    runner = SimpleNamespace(ws=ws, kernel=SimpleNamespace(matmul=matmul), _variant="base", _extra=None)
    lin = PackedLinear(runner, "W")
    x = torch.randn(3, C, generator=torch.Generator().manual_seed(8))
    dc, de = _dense(committed, (R, C)), _dense([(p, 1.0) for p in extra], (R, C))
    assert torch.allclose(lin(x), x @ base.T, atol=1e-5) and lin._committed is None
    runner._variant = "current"
    assert torch.allclose(lin(x), x @ (base + dc).T, atol=1e-4)
    runner._extra = {"W": extra}
    assert torch.allclose(lin(x), x @ (base + dc + de).T, atol=1e-4)
    runner._variant = "base"
    assert torch.allclose(lin(x), x @ base.T, atol=1e-5)


# ------------------------------------------------------------------------------ review fixes
D0, D1 = "model.layers.0.mlp.down_proj.weight", "model.layers.1.mlp.down_proj.weight"
O0, K0 = "model.layers.0.self_attn.o_proj.weight", "model.layers.0.self_attn.k_proj.weight"


def snap(ws, variant="current"):
    return {n: W(ws, n, variant) for n in ws.store.names()}


def moved(a, b):
    return {n: int((a[n] != b[n]).sum()) for n in a if (a[n] != b[n]).any()}


@pytest.mark.parametrize("members, factors", [
    # B: a neuron, a slice of its own column and two overlapping slices: every column scaled once
    (["L0N3", f"{D0}[:, 3]", f"{D0}[:, 0:5]", f"{D0}[:, 2:6]", "L0N3"], {c: 0.5 for c in range(6)}),
    (["blocks.0.mlp", "L0N3"], {c: 0.5 for c in range(24)}),               # a sublayer and a neuron inside it
])
def test_overlapping_members_scale_each_weight_once(quant, members, factors):
    ws = quant.ws
    b = snap(ws)
    ws.set_group("ov", members)
    r = S.script(ws, "group:ov", "scale", 0.5)
    apply(quant, r["script"])
    a = snap(ws)
    for c, f in factors.items():
        assert torch.allclose(a[D0][:, c], f * b[D0][:, c]), c
    assert sum(moved(a, b).values()) == r["weights"]


def test_a_head_and_a_slice_of_its_output_are_scaled_once(quant):
    ws = quant.ws
    b = snap(ws)
    ws.set_group("g3", ["L0H1", f"{O0}[:, 4:8]"])
    r = S.script(ws, "group:g3", "scale", 0.5)
    assert r["script"] == "scale_head((0, 1), 0.5)", r["script"]
    apply(quant, r["script"])
    assert torch.allclose(W(ws, O0)[:, 4:8], 0.5 * b[O0][:, 4:8])
    assert sum(moved(snap(ws), b).values()) == r["weights"] == 16 * 4


def test_a_stepped_slice_is_one_line_per_column(quant):
    ws = quant.ws
    r = S.script(ws, f"{D0}[:, 0:8:2]", "scale", 2)
    assert r["weights"] == 4 * 16 and len(r["script"].splitlines()) == 4, r["script"]
    b = snap(ws)
    apply(quant, r["script"])
    assert sum(moved(snap(ws), b).values()) == 64


# C: the fake store is as strict as the GGUF store where the IDE can tell
def test_the_fake_quantized_store_refuses_what_the_gguf_store_refuses(quant):
    st = quant.ws.store
    assert S.WINDOW <= F.MAX_READ                                  # a window is always one allowed read
    assert not st.is_float("model.norm.weight") or st.dtype("model.norm.weight") in F.SCHEMES
    st.kind = "IQ2_XS"                                              # a scheme the store cannot decode
    assert not st.is_float(D0)
    with pytest.raises(ValueError, match="unsupported"):
        st.read(D0)
    st.kind = "Q4_K"
    from wsp.store import _natural
    names = st.names()
    assert names == sorted(names, key=_natural)
    assert st.size_bytes(D0) > 0 and set(st.where) == set(names) and st.row_permutations == {}


def test_every_store_read_of_an_edit_fits_the_read_limit(quant, monkeypatch):
    """With a read limit far below the tensor size, a whole-tensor scale, a neuron range and their revert are
    written in reads that each fit it (the real store refuses a read of more than 16M values)."""
    ws = quant.ws
    b = snap(ws)
    monkeypatch.setattr(S, "WINDOW", 40)
    monkeypatch.setattr(F, "MAX_READ", 40)
    for addr in (D0, "blocks.1.mlp.neurons.0-20", "group:both"):
        if addr == "group:both":
            ws.set_group("both", [D1 + "[:, 22:24]", "L1N21"])
        r = S.script(ws, addr, "scale", 2)
        ws.store.reads.clear()
        apply(quant, r["script"])
        assert ws.store.reads and max(n for _, n in ws.store.reads) <= 40, (addr, max(n for _, n in ws.store.reads))
    for addr in (D0, "blocks.1.mlp.neurons.0-23"):
        apply(quant, S.script(ws, addr, "revert")["script"])
    monkeypatch.setattr(F, "MAX_READ", 16_000_000)
    for n in b:
        assert torch.allclose(W(ws, n), b[n], atol=1e-6), n


# G: on a quantized store, scattered neurons are read in the fewest windows (every row decoded once per window)
def test_scattered_neurons_on_a_quantized_store_are_read_in_the_fewest_windows(tmp_path, monkeypatch):
    sess = quant_session(tmp_path, d_model=16, d_mlp=8, n_layers=2, wide_layer=1, wide_mlp=4000)
    ws = sess.ws
    monkeypatch.setattr(S, "WINDOW", 16 * 1000)                    # 1000 columns a window
    ns = list(range(0, 4000, 7))                                    # gaps of 6: far below the float GAP of 64
    assert S.read_gap(ws, D1) is None and S.GAP is not None
    pos = np.asarray(ns)
    fewest = S.windows(pos, 16, gap=None)
    assert len(fewest) == 4 and len(fewest) <= len(S.windows(pos, 16, gap=3))
    # The store packs columns in blocks of 256 (Q4_K) and a read decodes every block it touches, so the windows
    # follow the blocks: the fewest windows of at most WINDOW weights that never split a block (3 blocks each).
    assert S.read_unit(ws, D1, 1) == 256
    plan = S.read_windows(ws, D1, 1, pos, 16)
    blocks = [set(range(int(w[0]) // 256, int(w[-1]) // 256 + 1)) for w in plan]
    assert len(plan) == -(-16 // 3) and sum(map(len, blocks)) == 16 == len(set().union(*blocks))
    assert sum(len(b) for b in blocks) < sum(int(w[-1]) // 256 - int(w[0]) // 256 + 1 for w in fewest)
    regions, read = [], ws.store.read
    monkeypatch.setattr(ws.store, "read", lambda name, region=None: (regions.append((name, region)), read(name, region))[1])
    ws.store.row_reads = 0
    apply(sess, f"scale_neurons(1, {ns}, 0.5)")
    assert ws.store.row_reads == 16 * len(plan), ws.store.row_reads
    spans = [(r[1].start, r[1].stop) for nm, r in regions if nm == D1]
    assert spans == [(int(w[0]), int(w[-1]) + 1) for w in plan]       # one read per window, none splits a block
    regions.clear()
    ws.store.row_reads = 0
    apply(sess, f"revert_neurons(1, {ns})")
    assert ws.store.row_reads == 0                                  # a revert reads the patches, not the store


def test_block_planned_windows_give_the_same_weights_and_fit_the_budget(tmp_path, monkeypatch):
    """The same scale_neurons with windows planned by packed block and without (read_block hidden): bit-identical
    weights. Every window, either way, is one read of at most WINDOW weights; when a single block across the rows
    is already more than WINDOW, the windows are not aligned to blocks so that they still fit."""
    ns = sorted(set(range(0, 4000, 7)) | set(range(1500, 1800)) | {3999})
    got = {}
    for planned in (True, False):
        (tmp_path / str(planned)).mkdir()
        sess = quant_session(tmp_path / str(planned), d_model=16, d_mlp=8, n_layers=2, wide_layer=1, wide_mlp=4000)
        ws = sess.ws
        monkeypatch.setattr(S, "WINDOW", 16 * 1000)
        if not planned:
            monkeypatch.setattr(ws.store, "read_block", lambda name: None)
        assert S.read_unit(ws, D1, 1) == (256 if planned else 1)
        ws.store.reads.clear()
        apply(sess, f"scale_neurons(1, {ns}, -0.375)")
        assert max(n for nm, n in ws.store.reads if nm == D1) <= S.WINDOW
        got[planned] = ws.read(D1)
    assert torch.equal(got[True], got[False])
    # a block across the rows larger than the budget: unaligned windows of at most the budget
    (tmp_path / "small").mkdir()
    sess = quant_session(tmp_path / "small", d_model=16, d_mlp=8, n_layers=2, wide_layer=1, wide_mlp=4000)
    pos = np.asarray(ns)
    for budget in (16 * 255, 16 * 40):
        plan = S.read_windows(sess.ws, D1, 1, pos, 16, budget)
        assert plan and all((int(w[-1]) - int(w[0]) + 1) * 16 <= budget for w in plan)
        assert [int(n) for w in plan for n in w] == ns
    # unpacked storage keeps the gap rule
    assert S.read_unit(sess.ws, "model.layers.1.input_layernorm.weight", 0) == 1


def test_a_long_edit_reports_progress(quant, monkeypatch):
    from wsp import wscript
    monkeypatch.setattr(wscript, "PROGRESS_EVERY", 0)
    seen = []
    quant.ws.listeners.append(lambda e: seen.append(e) if e["type"] == "edit_progress" else None)
    apply(quant, "scale_neurons(1, [1, 2, 3], 2)\nscale_neuron((0, 1), 2)\nP(\"model.norm.weight\")[:] *= 2")
    assert {e["line"] for e in seen} >= {1, 2, 3} and all(e["lines"] == 3 for e in seen)
    assert any(e.get("windows") for e in seen)
    assert not any("request" in e for e in seen)                     # an edit with no browser id: untagged


def test_an_edit_from_the_browser_tags_its_progress_with_its_request(quant, monkeypatch):
    """/api/edit passes the browser's id for the edit to its progress events, so a tab shows only its own."""
    from wsp import wscript
    monkeypatch.setattr(wscript, "PROGRESS_EVERY", 0)
    seen = []
    quant.ws.listeners.append(lambda e: seen.append(e) if e["type"] == "edit_progress" else None)
    c = client(quant)
    r = c.post("/api/edit", json={"script": "scale_neurons(1, [1, 2], 2)\nscale_neuron((0, 1), 2)", "request": "edit-abc"})
    assert r.status_code == 200 and r.json()["ok"], r.text
    assert seen and all(e["request"] == "edit-abc" for e in seen)
    seen.clear()
    r = c.post("/api/edit", json={"script": "scale_neurons(1, [3], 2)\nscale_neuron((0, 2), 2)", "request": "x" * 65})
    assert r.json()["ok"] and seen and not any("request" in e for e in seen)    # an oversized id is dropped


# H: the weights an edit stores so far are counted as it goes, and stay right through compaction
def test_the_stored_count_matches_the_pieces_through_compaction(quant, monkeypatch):
    from wsp import wscript
    seen = []
    real = wscript.Interp._stored

    def checked(self):
        n = real(self)
        assert n == sum(p.numel() for ps in self.extra.values() for p in ps)
        seen.append(n)
        return n
    monkeypatch.setattr(wscript.Interp, "_stored", checked)
    apply(quant, "\n".join(f"P(\"{D1}\")[{r}, {c}] *= 2" for r in range(16) for c in range(0, 24, 1)))
    assert len(seen) > 300 and seen[-1] > 0                         # well past the first compaction (256 pieces)


# I: a passive read of the node or the patch list does not work out ranks of thousands of pieces
def test_node_and_patch_list_do_not_rank_every_piece(quant, monkeypatch):
    from wsp import delta as D
    c = client(quant)
    apply(quant, "scale_neurons(1, [1, 5, 9, 13], 2)")
    ws = quant.ws
    ws.set_group("four", ["L1N1", "L1N5", "L1N9", "L1N13"])
    assert len(ws.pieces(D1)) > 1

    def no(*a, **kw):
        raise AssertionError("rank of a piece worked out for a passive read")
    monkeypatch.setattr(D, "_rank", no)
    for r in (c.get("/api/patches"), c.get("/api/node", params={"addr": D1}), c.get("/api/node", params={"addr": "group:four"})):
        assert r.status_code == 200, r.text
    calls = []
    real = D.summary
    monkeypatch.setattr(D, "summary", lambda d: calls.append(1) or real(d))
    ws.patches[0].meta(); ws.patches[0].meta()
    assert len(calls) <= 1                                          # a patch's stats are worked out once


def test_node_token_estimate_equals_the_per_slice_estimate(quant):
    from wsp import chunks
    ws = quant.ws
    ws.set_group("mix", ["blocks.1.mlp.neurons.0-9", "L0H1", f"{D0}[:, 2:5]"])
    n = client(quant).get("/api/node", params={"addr": "group:mix"}).json()
    sl = ws.resolve("group:mix")
    assert n["est_tokens"] == {f: chunks.est_slices(ws, sl, f) for f in ("q", "g3", "svd", "stats")}


def test_changed_count_fast_path_equals_reading_the_delta(quant):
    ws = quant.ws
    apply(quant, "scale_neurons(1, [1, 2, 3, 7, 11], 2)")
    apply(quant, f'P("{D1}")[0:4, 2:9] *= 3')                       # overlapping and not full height
    ws.set_group("g", ["blocks.1.mlp.neurons.0-12"])
    d = ws.read(D1, (), "delta")
    exact = int((d[:, 0:13] != 0).sum())
    assert S.changed(ws, "group:g")[0] == exact
    assert S.actions(ws, "group:g")["changed"] == exact


# J: the factor of scale_neurons
def test_scale_neurons_takes_a_number_or_one_write_vector_of_factors(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = quant_session(tmp_path / "a"), quant_session(tmp_path / "b")
    vec = "tensor([" + ", ".join(str(0.1 * i) for i in range(16)) + "])"
    apply(a, f"scale_neurons(1, [3, 5], {vec})")
    apply(b, f"scale_neuron((1, 3), {vec})\nscale_neuron((1, 5), {vec})")
    for n in a.ws.store.names():
        assert torch.allclose(W(a.ws, n), W(b.ws, n)), n
    r = a.apply("scale_neurons(1, [3, 5], tensor([0.5, 2.0]))", "t")       # one factor per neuron: refused in words
    assert not r["ok"] and "once per factor" in r["error"], r


# K: the patch route
def test_patch_route_needs_an_id_and_a_known_action(quant):
    c = client(quant)
    apply(quant, "scale_neuron((1, 2), 2)")
    for body in ({"action": "delete"}, {"action": "set", "enabled": False}, {"action": "zap", "id": quant.ws.patches[0].id}):
        r = c.post("/api/patch", json=body)
        assert r.status_code == 400 and "Traceback" not in r.text, (body, r.text)
    assert len(quant.ws.patches) == 1 and quant.ws.patches[0].enabled
    assert c.get("/api/patches").json()["edit_waits_for_generation"] is False
    assert c.get("/api/node", params={"addr": "L1N2"}).json()["edit_waits_for_generation"] is False


# D: a revert records what it cancels, and changing that warns
def test_a_revert_records_what_it_cancels_and_changing_that_warns(quant):
    c = client(quant)
    ws = quant.ws
    apply(quant, S.script(ws, "L1N3", "scale", 2)["script"])
    apply(quant, S.script(ws, "L0N1", "scale", 2)["script"])        # unrelated
    p1, p2 = ws.patches[0].id, ws.patches[1].id
    apply(quant, S.script(ws, "L1N3", "revert")["script"])
    rev = ws.patches[-1]
    assert [d["id"] for d in rev.source["compensates"]] == [p1]
    assert rev.source["compensates"][0]["strength"] == 1
    assert "warning" not in c.post("/api/patch", json={"action": "set", "id": p2, "enabled": False}).json()
    r = c.post("/api/patch", json={"action": "set", "id": p1, "strength": 0.5}).json()
    assert r["ok"] and f"Patch {p1} is now at ×0.5" in r["warning"] and rev.id in r["warning"]
    assert "warning" not in c.post("/api/patch", json={"action": "set", "id": p1, "strength": 1}).json()   # set back
    r = c.post("/api/patch", json={"action": "set", "id": p1, "enabled": False}).json()
    assert "was disabled" in r["warning"] and not ws.patch(p1).enabled  # the change is still made
    c.post("/api/patch", json={"action": "set", "id": rev.id, "enabled": False})
    assert "warning" not in c.post("/api/patch", json={"action": "delete", "id": p1}).json()   # its revert is off
    reason = S.actions(ws, "L0H0")["revert"]["reason"] if ws.pieces(O0) else S.REVERT_NOTE
    assert "cancels the current edits" in reason


def test_a_head_revert_names_the_heads_that_share_its_keys(quant):
    ws = quant.ws
    apply(quant, f'P("{K0}")[0:4] *= 2.0')
    apply(quant, "scale_head((0, 1), 0.5)")
    reason = S.actions(ws, "L0H0")["revert"]["reason"]
    assert "L0H0 with L0H1" in reason and "gate and up" not in reason


# E: selections that contain themselves, and members that no longer exist
def test_a_selection_cannot_contain_itself(quant):
    c = client(quant)
    assert c.post("/api/groups", json={"name": "a", "neurons": [[1, 3]]}).status_code == 200
    assert c.post("/api/groups", json={"name": "b", "members": ["group:a"]}).status_code == 200
    for name in ("a", "b"):
        r = c.post("/api/groups", json={"name": "a", "members": [f"group:{name}"], "mode": "add"})
        assert r.status_code == 400 and "cannot contain itself" in r.json()["error"], r.text
    assert quant.ws.groups["a"]["members"] == ["L1N3"]
    quant.ws.groups["a"]["members"].append("group:b")                # a cycle written some other way
    for r in (c.get("/api/node", params={"addr": "group:a"}),
              c.post("/api/selection/script", json={"addr": "group:a", "action": "scale", "factor": 2})):
        assert r.status_code in (200, 400) and "Traceback" not in r.text and "Recursion" not in r.text, r.text


def test_a_deleted_inner_selection_is_skipped_and_reported(quant):
    c = client(quant)
    c.post("/api/groups", json={"name": "b", "neurons": [[0, 1]]})
    c.post("/api/groups", json={"name": "c", "members": ["group:b", "L0N2"]})
    c.post("/api/groups", json={"name": "b", "delete": True})
    n = c.get("/api/node", params={"addr": "group:c"})
    assert n.status_code == 200, n.text
    assert "group:b" in [o["member"] for o in n.json()["actions"]["other"]]
    s = c.post("/api/selection/script", json={"addr": "group:c", "action": "scale", "factor": 2})
    assert s.status_code == 200 and s.json()["script"] == "scale_neurons(0, [2], 2.0)" and "group:b" in [o["member"] for o in s.json()["other"]]
    t = c.get("/api/tree", params={"addr": "group:c"})
    assert t.status_code == 200, t.text


# F: (frontend) -- see tests/inspector_frontend.test.mjs


# M: small things
def test_a_tensor_name_with_a_quote_is_written_as_a_string(quant):
    import ast
    from wsp import addressing as A
    name = 'we"ird\nP(1)'
    expr = A.to_script(name, ())
    assert expr.startswith("P(") and ast.literal_eval(expr[2:-1]) == name and "\n" not in expr


def test_neuron_lists_refuse_numbers_that_are_not_whole(quant):
    assert S.parse_neurons([[1, 3.0], "L0N2"]) == [(1, 3), (0, 2)]
    for bad in ([[1, 3.7]], [[True, 3]], [["1", 3]], [[1]], ["L1N"]):
        with pytest.raises(ValueError):
            S.parse_neurons(bad)
    assert client(quant).post("/api/groups", json={"name": "x", "neurons": [[1, 3.7]]}).status_code == 400


def test_one_slice_written_two_ways_is_one_member(quant):
    ws = quant.ws
    assert S.merge_members(ws, [f"{D0}[:,3]"], [f"{D0}[:, 3]", f"{D0}[ :, 3 ]"]) == [f"{D0}[:, 3]"]


def test_a_selection_named_with_spaces_can_be_deleted(quant):
    c = client(quant)
    quant.ws.groups[" "] = {"members": ["L1N3"], "note": ""}
    r = c.post("/api/groups", json={"name": " ", "delete": True})
    assert r.status_code == 200 and " " not in quant.ws.groups, r.text


def test_node_extras_surfaces_a_planning_bug(quant, monkeypatch):
    def bug(*a, **kw):
        raise RuntimeError("planning bug")
    monkeypatch.setattr(S, "plan", bug)
    with pytest.raises(RuntimeError):
        S.node_extras(quant.ws, "L1N7")


def test_scale_says_both_counts_when_they_differ(quant):
    reason = S.actions(quant.ws, "L1N3")["scale"]["reason"]
    assert reason == "multiplies what the neuron writes (16 weights of the 48 in every part of the selection)", reason


def test_linear_attention_layers_have_no_heads():
    """qwen3.5 linear-attention layers have no Q/K/V/O roles, so no head of theirs can be selected or edited."""
    from wsp import arch
    d, f = 16, 24
    sh = {"model.embed_tokens.weight": (32, d), "model.norm.weight": (d,)}
    for l, lin in ((0, True), (1, False)):
        p = f"model.layers.{l}"
        if lin:
            sh |= {f"{p}.linear_attn.in_proj_qkvz.weight": (4 * d, d), f"{p}.linear_attn.out_proj.weight": (d, d)}
        else:
            sh |= {f"{p}.self_attn.{x}_proj.weight": (d, d) for x in "qkvo"}
        sh |= {f"{p}.mlp.gate_proj.weight": (f, d), f"{p}.mlp.up_proj.weight": (f, d), f"{p}.mlp.down_proj.weight": (d, f)}
    s = arch.detect(sh, {"model_type": "qwen3_5_text", "num_attention_heads": 4, "num_key_value_heads": 4})
    assert [s.has_heads(l) for l in range(2)] == [False, True]


# ------------------------------------------------------------------------------ second review
def test_a_negative_index_is_the_same_weight_as_its_positive_one(quant):
    """`W[4, -17]` is `W[4, 7]`: scaled once with the whole tensor, scaled inside a group, counted and reverted."""
    ws = quant.ws
    b = W(ws, D0)
    ws.set_group("g", [D0, f"{D0}[4, -17]"])
    r = S.script(ws, "group:g", "scale", 3.0)
    assert r["weights"] == 16 * 24
    apply(quant, r["script"])
    assert torch.allclose(W(ws, D0), 3 * b)                        # [4, 7] not scaled twice
    ws.set_group("h", [f"{D0}[4, -1]", f"{D0}[0:2, 0:2]"])
    r = S.script(ws, "group:h", "scale", 0.5)
    assert r["weights"] == 5
    apply(quant, r["script"])
    assert torch.allclose(W(ws, D0)[4, 23], 1.5 * b[4, 23])
    n = client(quant).get("/api/node", params={"addr": f"{D0}[4, -1]"}).json()
    assert n["changed"] == 1 and n["actions"]["revert"]["enabled"]
    assert torch.allclose(ws.read(D0, (4, -1)), W(ws, D0)[4, 23])  # the workspace reads it with the edits


def _fuzz_member(rng, ws):
    def sl(n):
        k = rng.random()
        if k < 0.25:
            return str(rng.randrange(-n, n))                         # an int, negative or not
        a, b = sorted(rng.sample(range(-n, n + 1), 2))
        step = rng.choice([1, 1, 2, 3, -1, -2])
        if step < 0:
            a, b = b, a                                              # a descending slice
        return f"{a if rng.random() < .7 else ''}:{b if rng.random() < .7 else ''}" + (f":{step}" if step != 1 else "")
    l = rng.randrange(2) if rng.random() < .4 else 0                 # mostly layer 0, so members overlap
    k = rng.random()
    if k < 0.2:
        a = rng.randrange(24)
        b = rng.randrange(a, 24)
        return f"L{l}N{a}" if a == b else f"blocks.{l}.mlp.neurons.{a}-{b}"
    if k < 0.3:
        return f"L{l}H{rng.randrange(4)}"
    if k < 0.35:
        return f"blocks.{l}.attn.kvgroup.{rng.randrange(2)}"
    name = f"model.layers.{l}." + rng.choice(["mlp.down_proj.weight", "mlp.down_proj.weight", "self_attn.o_proj.weight",
                                             "self_attn.q_proj.weight", "mlp.up_proj.weight"])
    R, C = ws.shape(name)
    r = rng.random()
    return f"{name}[:, {sl(C)}]" if r < 0.4 else f"{name}[{sl(R)}]" if r < 0.7 else f"{name}[{sl(R)}, {sl(C)}]"


def _fuzz_reach(ws, members, every):
    """The weights an action reaches, by the contract, as dense masks (each member resolved on its own)."""
    from wsp.store import norm_index, region_index
    st = ws.struct
    m = {}

    def mark(name, idx):
        shape = ws.shape(name)
        m.setdefault(name, np.zeros(shape, bool))[region_index(norm_index(idx, shape))] = True
    for a in members:
        sel = S.expand(ws, a)
        for l, ns in sel.neurons.items():
            for n in ns:
                for nm, idx in (ws.resolve(f"L{l}N{n}") if every else [(st.layers[l].mlp["W_out"].name, (slice(None), n))]):
                    mark(nm, idx)
        for l, h in sel.heads:
            o = st.layers[l].attn["W_O"]
            for nm, idx in (st.head_slices(l, h) if every else [(o.name, st.part_index(o, head=h))]):
                mark(nm, idx)
        for nm, idx in sel.slices:
            mark(nm, idx)
    return m


@pytest.mark.parametrize("seed", range(4))
def test_fuzz_overlapping_members_scale_and_revert_like_a_dense_reference(tmp_path, seed):
    """Random overlapping members (negative ints, stepped and descending slices, rows crossing columns, neurons,
    heads, key-value groups, a nested selection) over prior edits at several strengths: scale multiplies each
    reached weight once, revert puts each back, and every count is exact. "Changed" is a non-zero delta, as in
    Workspace.changed (a residue that rounds away in base + delta still counts)."""
    import random
    rng = random.Random(seed)
    for trial in range(60):
        d = tmp_path / f"t{trial}"
        d.mkdir()
        sess = quant_session(d)
        ws = sess.ws
        for _ in range(rng.randrange(0, 3)):                        # prior edits, some re-strengthened
            ws.set_group("pre", [_fuzz_member(rng, ws) for _ in range(rng.randrange(1, 4))])
            try:
                r = S.script(ws, "group:pre", "scale", rng.choice([0.5, 2.0, 3.0]))
            except S.Refusal:
                continue
            if sess.apply(r["script"], "t").get("committed") and rng.random() < .5:
                ws.set_patch(ws.patches[-1].id, strength=rng.choice([0.5, 1.5]))
        members = [_fuzz_member(rng, ws) for _ in range(rng.randrange(1, 6))]
        flat = list(members)
        if rng.random() < 0.3:
            ws.set_group("inner", members[:2])
            members = ["group:inner"] + members[2:] + ["group:inner"]
        ws.set_group("g", members)
        before, base = snap(ws), snap(ws, "base")
        delta = {n: ws.read(n, (), "delta").numpy() != 0 for n in before}
        every = _fuzz_reach(ws, flat, True)
        edited = sum(int((delta[n] & mk).sum()) for n, mk in every.items())
        assert S.changed(ws, "group:g")[0] == edited, (seed, trial, members)
        act = rng.choice(["scale", "revert"])
        f = rng.choice([0.5, 3.0])
        try:
            r = S.script(ws, "group:g", act, f if act == "scale" else None)
        except S.Refusal:                                            # nothing reached, or nothing edited to revert
            reached = sum(int(mk.sum()) for mk in _fuzz_reach(ws, flat, False).values())
            assert (edited == 0) if act == "revert" else (reached == 0), (seed, trial, act, members)
            continue
        apply(sess, r["script"])
        after = snap(ws)
        mm = _fuzz_reach(ws, flat, act == "revert")
        for n in before:
            mk = torch.as_tensor(mm.get(n, np.zeros(before[n].shape, bool)))
            want = torch.where(mk, before[n] * f if act == "scale" else base[n], before[n])
            assert torch.allclose(after[n], want, atol=1e-5), (seed, trial, act, n, members)
        if act == "scale":
            assert sum(moved(after, before).values()) == r["weights"], (seed, trial, members)
        else:
            assert r["weights"] == edited, (seed, trial, members)


def test_a_revert_does_not_list_disabled_or_zero_strength_patches(quant):
    """`compensates` names only the enabled patches with a strength the revert was worked out against."""
    ws = quant.ws
    apply(quant, S.script(ws, "L1N3", "scale", 2)["script"])
    apply(quant, S.script(ws, "L1N3", "scale", 3)["script"])
    apply(quant, S.script(ws, "L1N3", "scale", 5)["script"])
    p1, p2, p3 = (p.id for p in ws.patches)
    ws.set_patch(p2, enabled=False)
    ws.set_patch(p3, strength=0.0)
    apply(quant, S.script(ws, "L1N3", "revert")["script"])
    assert [d["id"] for d in ws.patches[-1].source["compensates"]] == [p1]


def test_a_patch_summary_covers_every_tensor_it_edits(quant):
    ws = quant.ws
    apply(quant, f'scale_head((0, 1), 0.5)\nscale_neuron((1, 3), 2)\nP("{K0}")[0:4] *= 2')
    m = ws.patches[0].meta()
    assert sorted(s["param"] for s in m["stats"]) == sorted([O0, D1, K0])
    assert m["changed"] == 16 * 4 + 16 + 4 * 16


@pytest.mark.parametrize("members, n", [
    ([f"{D0}[3]", f"{D0}[:, 5]"], 24 + 16 - 1),                      # a row crossing a column
    ([f"{D0}[2:6, 3:9]", f"{D0}[4:10, 5:12]"], 24 + 42 - 8),          # two partial boxes overlapping
    ([f"{D0}[2:6, 3:9]", f"{D0}[4:10, 5:12]", f"{D0}[0:3, 10:20]"], 24 + 42 - 8 + 30),
])
def test_crossing_and_overlapping_slices_are_scaled_once(quant, members, n):
    ws = quant.ws
    b = W(ws, D0)
    ws.set_group("x", members)
    r = S.script(ws, "group:x", "scale", 2.0)
    assert r["weights"] == n
    apply(quant, r["script"])
    mask = torch.zeros_like(b, dtype=torch.bool)
    for m in members:
        mask[ws.resolve(m)[0][1]] = True
    assert torch.allclose(W(ws, D0), torch.where(mask, 2 * b, b)) and int(mask.sum()) == n


def test_an_edited_selection_the_plan_refuses_still_shows_its_edits(rnd):
    """Members overlapping in a tensor of three axes are refused for scale and revert, but the node and the tree
    still count what is edited there, each weight once."""
    ws = rnd.ws
    apply(rnd, "blocks[0].attn.W_Q[0, 0:4] *= 3")
    ws.set_group("g", ["blocks.0.attn.W_Q[0, 0:3]", "blocks.0.attn.W_Q[0, 2:6]"])
    exact = ws.changed("blocks.0.attn.W_Q", (0, slice(0, 6)))[0]
    assert exact > 0
    n = client(rnd).get("/api/node", params={"addr": "group:g"}).json()
    assert not n["actions"]["revert"]["enabled"] and "more than two axes" in n["actions"]["revert"]["reason"]
    assert "changed" not in n["actions"] and n["changed"] == exact
    assert S.changed(ws, "group:g")[0] == exact
    assert hierarchy.node(ws, "group:g")["changed"] == exact
