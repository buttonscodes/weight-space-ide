"""The atlas: built through the architecture-independent layer, so one set of checks for every layout.

Random tiny models cannot say whether the map is *right* about a trained model (that is what the observed stage
measures on real ones); these tests check that it is right where the answer is known by construction: a neuron
written by hand is read back as written, attribution adds up to what the MLP actually wrote, selections and
token shifts do what they say, and the build is the same code path for every family.
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from wsp import atlas, zoo  # noqa: E402
from wsp.atlas import guide, report  # noqa: E402
from wsp.atlas.tokens import categorize  # noqa: E402
from wsp.model import Config, init_params  # noqa: E402
from wsp.session import Session  # noqa: E402
from wsp.workspace import Workspace  # noqa: E402

quiet = lambda *a: None  # noqa: E731


@pytest.fixture()
def rnd(tmp_path):
    from wsp.tokenizer import VocabTokenizer
    tok = VocabTokenizer(list("abcdefghijklmnopqrstuvwxyz"), "char")
    cfg = Config(d_vocab=26, n_ctx=16, d_model=24, n_layers=2, n_heads=2, d_mlp=40, name="rnd")
    Workspace.create(str(tmp_path / "rnd"), cfg, tok, init_params(cfg, seed=1, std=0.3))
    s = Session(str(tmp_path))
    s.open("rnd")
    return s


def test_token_categories():
    assert categorize(" word") == "lat_lower_ws" and categorize("Word") == "lat_cap_cont" and categorize("42") == "number"
    assert categorize("\n") == "ws_newline" and categorize("北京") == "han" and categorize("x_y") == "code_mix"
    assert categorize("<s>", True) == "special"


def test_a_hand_written_neuron_is_read_back(rnd):
    """Write neuron (0, 5) to fire on 'c' and promote 'd'; the atlas of the edited weights must say so."""
    rnd.apply('n = neuron(0, 5)\nn.w_in = 6 * unit(E("c"))\nn.w_out = 5 * unit(U("d"))')
    atlas.build(rnd.ws, stages=["weights"], log=quiet, variant="current")
    A = atlas.Atlas.of(rnd.ws)
    u = A.unit(0, 5)
    assert u["writes"]["pos"][0]["token"] == "d"
    assert u["gate_tokens"]["pos"][0]["token"] == "c"
    assert not A.stale_layers()
    rnd.apply("neuron(1, 0).w_out *= 0")
    assert A.stale_layers() == A.layers                       # a 'current' atlas is stale after any further edit
    assert "L0N5" in report.unit(A, 0, 5)


def test_build_is_resumable_and_base_atlas_tracks_edits(rnd):
    m1 = atlas.build(rnd.ws, log=quiet)
    assert set(m1["stages"]) == {"weights", "wiring", "observed"}
    A = atlas.Atlas.of(rnd.ws)
    before = A.layer(0)["out_pos_ids"].copy()
    atlas.build(rnd.ws, log=quiet)                             # nothing to do: same files
    assert (atlas.Atlas.of(rnd.ws).layer(0)["out_pos_ids"] == before).all()
    assert A.stale_layers() == []
    rnd.apply("neuron(1, 3).w_out *= 0")
    assert atlas.Atlas.of(rnd.ws).stale_layers() == [1]
    L = A.layer(0)
    n = rnd.ws.struct.mlp_width(0)
    assert L["mech"].shape == (n,) and L["obs_peak"].shape == (n,) and L["w_down_gate_cos"].shape[0] == n
    assert set(np.unique(L["obs_peak"])) <= {-1, 0, 1}
    v = A.json("validation.json")
    assert v["loss"] > 0 and {c["verdict"] for c in v["checks"]} <= {"holds", "holds weakly", "fails", "too few to tell"}
    assert "Atlas of rnd" in report.overview(A)


def test_selectors_and_script_functions(rnd):
    atlas.build(rnd.ws, stages=["weights"], log=quiet)
    A = atlas.Atlas.of(rnd.ws)
    r = atlas.select(A, {"layers": [1, 1], "top": 7})
    assert r["n"] == 7 and set(r["layer"].tolist()) == {1}
    assert atlas.select(A, {"neurons": ["L0.3", "L1.4"]})["n"] == 2
    with pytest.raises(ValueError):
        atlas.select(A, {"colour": "red"})
    with pytest.raises(ValueError):
        atlas.select(A, {"mech": "damper", "effect": "promote"})          # effect needs a target
    w = rnd.ws
    before = w.sem.mlp(1)["W_out"].read()
    rnd.apply("for ln in select(layers=[1, 1], top=3):\n    scale_neuron(ln, 0)")
    after = w.sem.mlp(1)["W_out"].read()
    zeroed = sorted(int(i) for i in torch.nonzero(after.abs().sum(1) == 0).flatten())
    assert zeroed == sorted(int(i) for i in r["idx"][:3]) and torch.equal(after[[i for i in range(len(after)) if i not in zeroed]],
                                                                         before[[i for i in range(len(after)) if i not in zeroed]])
    script = atlas.selection_script(r, 2.0)
    assert script.startswith("scale_neurons(1, [")


def test_token_shift_moves_exactly_the_requested_tokens(rnd):
    w = rnd.ws
    assert guide.shift_plan(w)["mode"] == "bias"                           # this project's small models have a logit bias
    base = w.logits("abc")[-1]
    rnd.apply('shift_tokens(["d", "e"], 2.5)')
    d = w.logits("abc")[-1] - base
    assert torch.allclose(d[[3, 4]], torch.full((2,), 2.5), atol=1e-4) and float(d[[0, 1, 2, 5]].abs().max()) < 1e-4


def test_attribution_adds_up_to_what_the_mlps_wrote(rnd):
    w = rnd.ws
    res = guide.attribute(w, "abcab", "c", "d", top=5)
    r = w.runner
    ids = w.encode("abcab")
    names = [f"blocks.{l}.hook_mlp_out" for l in range(2)] + ["blocks.1.hook_resid_post"]
    _, cache = r.run(ids, "current", names=names)
    gain, centre = r.final_gain(cache["blocks.1.hook_resid_post"][0, -1])
    v = gain * (w.sem.unembed_col(2) - w.sem.unembed_col(3))
    v = v - v.mean() if centre else v
    for l in range(2):
        wrote = cache[f"blocks.{l}.hook_mlp_out"][0, -1] - w.sem.mlp(l)["b_out"].read()
        assert abs(float(wrote @ v) - res["by_layer"][l]) < 2e-3
    assert guide.top_script(res, 2, 0.0).startswith("scale_neurons(")


def test_status_offers_the_observed_stage_for_a_worker_model_and_says_what_it_runs(rnd, monkeypatch):
    """The page offers the observed box from these fields: a model that runs in the model worker can build the
    observed stage (observed_unavailable is None, can_observe true) and the status says what that runs, sized from
    the probe text the stage would use."""
    from fastapi.testclient import TestClient
    from wsp import worker_jobs
    from wsp.server import create_app
    c = TestClient(create_app(rnd, "http://test"))
    st = c.get("/api/atlas").json()
    assert st["observed_unavailable"] is None and st["observed_note"] is None          # runs in this process: no note
    monkeypatch.setattr(worker_jobs, "uses_worker", lambda ws: True)
    st = c.get("/api/atlas").json()
    assert st["can_observe"] is True and st["observed_unavailable"] is None
    assert st["observed_note"] == "This stage loads and runs the model on the probe text, one window at a time."   # sampled text
    with open(os.path.join(rnd.ws.dir, "corpus.txt"), "w") as f:
        f.write("abcdefghij" * 230)                                                    # 2,300 tokens: 143 windows of 16
    st = c.get("/api/atlas").json()
    assert st["observed_note"] == "This stage loads and runs the model on about 2,300 tokens of probe text, one window at a time."


def test_server_endpoints(rnd):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    c = TestClient(create_app(rnd, "http://test"))
    st = c.get("/api/atlas").json()
    assert st["built"] is False and st["can_build"] and st["can_observe"] and st["observed_unavailable"] is None
    current = c.get("/api/status").json()
    scope = {k: current[k] for k in ("workspace_id", "version")}
    assert c.post("/api/atlas/build", json={**scope, "stages": ["weights", "wiring", "observed"]}).status_code == 200
    import time
    for _ in range(200):
        st = c.get("/api/atlas").json()
        if st["built"] and not st["job"]["running"]:
            break
        time.sleep(0.05)
    assert st["built"] and st["job"]["error"] is None and st["layers"]["stages"]
    lay = c.get("/api/atlas/layer", params={"l": 0}).json()
    assert lay["n"] == 40 and len(lay["mech"]) == 40 and lay["observed"]
    u = c.get("/api/atlas/unit", params={"l": 0, "n": 1}).json()
    assert u["writes"]["pos"] and "observed" in u
    sel = c.post("/api/atlas/select", json={"selector": {"layers": [0, 0], "top": 4}, "scale": 0}).json()
    assert sel["n"] == 4 and "scale_neurons(0" in sel["script"]
    at = c.post("/api/atlas/attribute", json={**scope, "prompt": "abca", "target": "b"}).json()
    assert at["top"] and "by_layer" in at
    assert c.get("/api/atlas/concepts").json()["nodes"]
    from wsp import tools
    assert "Atlas of rnd" in tools.call(rnd, "atlas", {})
    assert "L0N1" in tools.call(rnd, "atlas", {"action": "unit", "address": "L0N1"})
    assert "atlas" in rnd.brief().lower()


@pytest.mark.parametrize("kind", ["llama", "gpt2", "neox", "phi3", "neo"])
def test_every_layout_builds_through_the_same_path(kind, tmp_path):
    pytest.importorskip("transformers")
    from test_hf import build
    build(kind, str(tmp_path / "ckpt"))
    os.makedirs(tmp_path / "ws")
    ws = zoo.open_hf(str(tmp_path / "ws"), str(tmp_path / "ckpt"), name="m")
    meta = atlas.build(ws, log=quiet)
    assert set(meta["stages"]) == {"weights", "wiring", "observed"}
    A = atlas.Atlas.of(ws)
    s = ws.struct
    for l in A.layers:
        L = A.layer(l)
        assert len(L["mech"]) == s.mlp_width(l) and np.isfinite(L["out_std"]).all() and np.isfinite(L["cos_ud"]).all()
        assert (L["out_pos_z"][:, 0] >= L["out_pos_z"][:, -1]).all() and (L["out_neg_z"][:, 0] <= L["out_neg_z"][:, -1]).all()
    gated = "W_gate" in s.layers[0].mlp
    assert (np.abs(A.layer(0)["cos_gu"]) < 1).any() if gated else (A.layer(0)["cos_gu"] == 1).all()
    # the write signature is the direct logit effect the rest of the IDE computes for one neuron
    d = ws.sem.neuron(0, 2)["write"].read("base")
    lg = ws.sem.logit_effect(d, "base")
    z = (lg - lg.mean()) / lg.std(unbiased=False)
    assert int(z.argmax()) == int(A.layer(0)["out_pos_ids"][2, 0])
    assert abs(float(z.max()) - float(A.layer(0)["out_pos_z"][2, 0])) < 0.05
    res = guide.attribute(ws, "t3 t7 t11 t5", "t9", "t2", top=4)
    assert len(res["top"]) == 4 and np.isfinite(res["mlp_total"])
