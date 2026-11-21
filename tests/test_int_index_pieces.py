"""A patch piece whose region holds an integer index (`W[:, 3838]`: one column, values of one axis fewer).

Scripts run before selections existed stored such pieces, and a saved patch is never rewritten, so a workspace
can hold them for ever. Every request must answer exactly as it does for the same edit stored with a one-wide
range (`W[:, 3838:3839]`), which is what the same script stores now. Model-free: the quantized case uses the
in-memory store of tests/selection_fixtures, the other a tiny random model of this project's own kind."""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from selection_fixtures import quant_workspace  # noqa: E402
from wsp import delta as D  # noqa: E402
from wsp import selections as S  # noqa: E402
from wsp.model import Config, init_params  # noqa: E402
from wsp.server import create_app  # noqa: E402
from wsp.session import Session  # noqa: E402
from wsp.workspace import Workspace  # noqa: E402

DOWN = "model.layers.1.mlp.down_proj.weight"                    # [16, 24] in the fixture: column = neuron
EDITS = [f'P("{DOWN}")[:, 5] *= 1000',                          # the reported patch: one neuron's write column
         f'P("{DOWN}")[3, :] *= 2',                             # one row
         f'P("{DOWN}")[2, 7] += 1.5',                           # one weight
         f'P("{DOWN}")[4:9, 11] *= 0']                          # part of a column
# One piece alone takes the counting shortcut (`_counts_from_pieces`); pieces that meet take the full count.
SETS = {"column": EDITS[:1], "row": EDITS[1:2], "weight": EDITS[2:3], "part": EDITS[3:], "apart": [EDITS[0], EDITS[3]],
        "all": EDITS}


@pytest.fixture(params=list(SETS))
def edits(request):
    return SETS[request.param]


def quant_session(tmp_path, tag):
    d = tmp_path / tag / "quant"
    d.mkdir(parents=True)
    s = Session(str(tmp_path / tag))
    s.ws = quant_workspace(str(d))
    return s


def to_integer_form(ws) -> int:
    """Rewrite every one-wide range of every block piece as an integer index, in memory, the way older patches
    hold them. Returns how many pieces changed."""
    n = 0
    for patch in ws.patches:
        for pieces in patch.deltas.values():
            for p in pieces:
                if p.kind != "block" or not any(isinstance(r, range) and len(r) == 1 for r in p.region):
                    continue
                shape = [len(r) for r in p.region if isinstance(r, range) and len(r) != 1]
                p.region = tuple(r[0] if isinstance(r, range) and len(r) == 1 else r for r in p.region)
                p.values = p.values.reshape(shape)
                for k in [k for k in p.__dict__ if k.startswith("_")]:
                    del p.__dict__[k]
                n += 1
        patch._stats = None
    ws.rebuild()
    return n


def pair(tmp_path, edits=EDITS):
    """Two sessions holding the same patches: stored with ranges (as now), and with integer indices (as before)."""
    new, old = quant_session(tmp_path, "new"), quant_session(tmp_path, "old")
    for s in (new, old):
        for e in edits:
            r = s.apply(e, "edit")
            assert r.get("ok") and r.get("committed"), r
    assert to_integer_form(old.ws) == len(edits)
    assert any(isinstance(r, int) for p in S.pieces_of(old.ws, DOWN) for r in p.region)
    assert not any(isinstance(r, int) for p in S.pieces_of(new.ws, DOWN) for r in p.region)
    return new, old


def client(sess):
    return TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)


VOLATILE = {"created", "workspace_id", "elapsed", "seconds", "time", "took", "path", "root", "dir", "request"}


def scrub(x):
    """A response without what differs between two workspaces holding the same edits (ids, times, paths)."""
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    if isinstance(x, str):
        return re.sub(r"\b\d{4}-\d{5}\b", "<patch>", x)
    return x


def scope(c, ws):
    ident = c.get("/api/state").json()["status"]
    return {"workspace_id": ident.get("workspace_id", ""), "version": ident.get("version", ws.version)}


ADDRS = ["model", "blocks.1", "blocks.1.mlp", "blocks.0.mlp", DOWN, f"{DOWN}[:, 5]", f"{DOWN}[:, 4:8]", f"{DOWN}[3, :]",
         f"{DOWN}[2, 7]", f"{DOWN}[0:6, 11]", "L1N5", "L1N11", "L1N0", "blocks.1.mlp.neurons.0-23", "L1H0"]


@pytest.mark.parametrize("addr", ADDRS)
def test_a_page_answers_the_same_for_an_integer_index_piece(tmp_path, addr, edits):
    new, old = pair(tmp_path, edits)
    answers = []
    for s in (new, old):
        c = client(s)
        r = c.get("/api/node", params={"addr": addr, **scope(c, s.ws)})
        assert r.status_code == 200, r.text
        answers.append(scrub(r.json()))
    assert answers[0] == answers[1]


def test_the_tree_patch_list_and_before_after_view_answer_the_same(tmp_path, edits):
    new, old = pair(tmp_path, edits)
    got = {}
    for tag, s in (("new", new), ("old", old)):
        c = client(s)
        sc = scope(c, s.ws)
        out = []
        for path, params in [("/api/tree", {"addr": "model"}), ("/api/tree", {"addr": "blocks.1"}),
                             ("/api/tree", {"addr": "blocks.1.mlp"}), ("/api/patches", {}),
                             ("/api/matrix/compare", {"addr": DOWN}), ("/api/matrix/compare", {"addr": f"{DOWN}[:, 5]"}),
                             ("/api/matrix", {"addr": DOWN})]:
            r = c.get(path, params={**params, **sc})
            assert r.status_code == 200, (path, params, r.text)
            out.append(scrub(r.json()))
        for p in out[3]["patches"]:              # a piece of one axis reports no rank (wsp.delta._rank), as it always has
            for st in p["stats"]:
                st.pop("rank")
        got[tag] = out
    assert got["new"] == got["old"]


def test_the_counts_see_every_integer_index_piece(tmp_path):
    """Not only equal to the range form: the right numbers. 16 (column 5) + 24 (row 3) - 1 shared + 1 + 5 zeroed."""
    _, old = pair(tmp_path)
    base, cur = old.ws.read(DOWN, (), "base"), old.ws.read(DOWN, (), "current")
    expected = int((base != cur).sum())
    assert expected >= 16 + 24 - 1 + 1
    assert S.changed(old.ws, DOWN)[0] == expected
    assert S.changed(old.ws, "L1N5")[0] >= 16
    assert S.changed(old.ws, f"{DOWN}[2, 7]")[0] == 1
    for name, n in [("column", 16), ("row", 24), ("weight", 1)]:
        _, one = pair(tmp_path / name, SETS[name])
        assert S.changed(one.ws, DOWN)[0] == n and S.changed(one.ws, "model")[0] == n


@pytest.mark.parametrize("addr", [DOWN, "L1N5", "blocks.1.mlp", f"{DOWN}[3, :]", f"{DOWN}[2, 7]", "model"])
def test_revert_undoes_an_integer_index_piece(tmp_path, addr, edits):
    new, old = pair(tmp_path, edits)
    after, answers = [], []
    for s in (new, old):
        c = client(s)
        r = c.post("/api/selection/script", json={"addr": addr, "action": "revert"})
        answers.append((r.status_code, scrub(r.json())))
        if r.status_code != 200:                 # nothing under this address was edited: refused, in both forms
            assert r.status_code == 400 and "nothing to revert" in r.json()["error"], r.text
            continue
        e = c.post("/api/edit", json={"script": r.json()["script"]}).json()
        assert e.get("ok") and e.get("committed"), e
        assert S.changed(s.ws, addr)[0] == 0
        after.append(s.ws.read(DOWN, (), "current").clone())
    assert answers[0] == answers[1]
    if not after:
        return
    assert torch.equal(after[0], after[1])
    if addr in (DOWN, "blocks.1.mlp", "model"):  # base + d - d in float32: equal to rounding of the x1000 values
        assert torch.allclose(after[1], old.ws.read(DOWN, (), "base"), atol=1e-3)


@pytest.mark.parametrize("action,factor", [("scale", 3.0), ("ablate", None)])
def test_scale_and_ablate_on_top_of_an_integer_index_piece(tmp_path, action, factor, edits):
    new, old = pair(tmp_path, edits)
    after = []
    for s in (new, old):
        c = client(s)
        body = {"addr": "blocks.1.mlp.neurons.3-12", "action": action}
        if factor is not None:
            body["factor"] = factor
        r = c.post("/api/selection/script", json=body)
        assert r.status_code == 200, r.text
        e = c.post("/api/edit", json={"script": r.json()["script"]}).json()
        assert e.get("ok") and e.get("committed"), e
        after.append(s.ws.read(DOWN, (), "current").clone())
    assert torch.equal(after[0], after[1])


def test_steering_reads_the_norm_of_an_integer_index_column(tmp_path, edits):
    new, old = pair(tmp_path, edits)
    spec = [{"addr": "L1N5", "mode": "add", "dose": 0.5}, {"addr": "L1N11", "mode": "add", "dose": 1.0},
            {"addr": "L1N0", "mode": "mul", "dose": 2}]
    got = []
    for s in (new, old):
        r = client(s).post("/api/steer/resolve", json={"steer": spec, "p50": {"1": 2.0}, "calibrate": False})
        assert r.status_code == 200, r.text
        got.append(scrub(r.json()))
    assert got[0] == got[1]
    if edits == SETS["column"]:
        col = old.ws.read(DOWN, (), "current")[:, 5]
        assert float(col.norm()) == pytest.approx(1000 * float(old.ws.read(DOWN, (), "base")[:, 5].norm()), rel=1e-4)


def _int_pieces(out_features, in_features, seed=0):
    """(integer form, range form) of the same edits: enough whole columns and whole rows to be gathered, and others."""
    g = torch.Generator().manual_seed(seed)
    old, new = [], []

    def add(region_old, region_new, shape_new):
        v = torch.randn(shape_new, generator=g)
        new.append((D.Piece("block", region_new, values=v), 1.0 if len(new) % 2 else 0.5))
        old.append((D.Piece("block", region_old, values=v.reshape([n for n in shape_new if n != 1] or [])), new[-1][1]))

    R, C = range(out_features), range(in_features)
    for c in (1, 4, 6, 9, 13):                                   # whole columns, as an integer index
        add((R, c), (R, range(c, c + 1)), (out_features, 1))
    for r in (0, 3, 5, 8, 10):                                   # whole rows
        add((r, C), (range(r, r + 1), C), (1, in_features))
    add((2, 7), (range(2, 3), range(7, 8)), (1, 1))              # one weight: values without axes
    add((range(4, 9), 11), (range(4, 9), range(11, 12)), (5, 1))
    add((6, range(3, 10)), (range(6, 7), range(3, 10)), (1, 7))
    return old, new


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_gguf_runner_applies_integer_index_pieces(dtype):
    from wsp.gguf_runner import DeltaOverlay, delta_matmul
    out_features, in_features = 12, 20
    old, new = _int_pieces(out_features, in_features)
    x = torch.randn(2, 3, in_features, generator=torch.Generator().manual_seed(1)).to(dtype)
    outs = []
    for pieces in (old, new):
        o = DeltaOverlay(pieces, out_features, in_features, dtype=dtype)
        assert o.n_gathered == 10 and o.n_rest == 3
        out = torch.zeros(2, 3, out_features, dtype=dtype)
        o.apply(x, out)
        outs.append(out)
    assert torch.equal(outs[0], outs[1])
    dense = torch.zeros(out_features, in_features, dtype=torch.float64)
    for p, s in new:
        D.add_to(dense, (range(out_features), range(in_features)), p, s)
    assert torch.allclose(outs[0].double(), x.double() @ dense.T, atol=1e-5 if dtype == torch.float32 else 1e-12)
    if dtype == torch.float32:                                               # the path for a few pieces (float32 only)
        one = torch.zeros(2, 3, out_features)
        for p, s in old:
            delta_matmul(x, p, s, one)
        assert torch.allclose(one.double(), x.double() @ dense.T, atol=1e-5)


def test_a_saved_patch_with_an_integer_index_loads_and_every_page_answers(tmp_path):
    """The real loader: a patch file as the earlier code wrote it (`"region": [[0, 24, 1], 5]`, values of one
    axis), opened from disk in a workspace of this project's own kind."""
    from wsp.tokenizer import VocabTokenizer
    tok = VocabTokenizer(list("abcdefghijklmnopqrstuvwxyz"), "char")
    cfg = Config(d_vocab=26, n_ctx=16, d_model=24, n_layers=2, n_heads=2, d_mlp=40, name="rnd")
    answers = {}
    for tag in ("new", "old"):
        root = tmp_path / tag
        Workspace.create(str(root / "rnd"), cfg, tok, init_params(cfg, seed=1, std=0.3))
        s = Session(str(root))
        s.open("rnd")
        name = s.ws.struct.layers[1].mlp["W_out"].name
        axis, positions, template = S.neuron_axis(s.ws.struct, s.ws.struct.layers[1].mlp["W_out"])
        assert positions.start == 0 and positions.step == 1
        index = ["5" if d == axis else ":" for d in range(len(template))]
        r = s.apply(f'P("{name}")[{", ".join(index)}] *= 1000', "scale one neuron")
        assert r.get("ok") and r.get("committed"), r
        if tag == "old":
            pdir = root / "rnd" / "patches"
            (meta_file,) = [f for f in os.listdir(pdir) if f.endswith(".json")]
            meta = json.loads((pdir / meta_file).read_text())
            raw = load_file(str(pdir / (meta_file[:-5] + ".safetensors")))
            changed = 0
            for piece in meta["pieces"]:
                region = piece["region"]
                assert piece["kind"] == "block"
                for d, r_ in enumerate(region):
                    if isinstance(r_, list) and r_[1] - r_[0] == 1:
                        region[d] = r_[0]
                        changed += 1
                key = piece["key"] + ".v"
                raw[key] = raw[key].reshape([n for n in raw[key].shape if n != 1])
                assert raw[key].dim() == 1               # as the reported patch file: one tensor of one axis
            assert changed == 1
            (pdir / meta_file).write_text(json.dumps(meta))
            save_file(raw, str(pdir / (meta_file[:-5] + ".safetensors")))
            s = Session(str(root))
            s.open("rnd")
            assert any(isinstance(r_, int) for p in S.pieces_of(s.ws, name) for r_ in p.region)
        c = client(s)
        sc = scope(c, s.ws)
        out = []
        for addr in ["model", "blocks.1", "blocks.1.mlp", name, "L1N5", "L1N6"]:
            r = c.get("/api/node", params={"addr": addr, **sc})
            assert r.status_code == 200, (addr, r.text)
            out.append(scrub(r.json()))
        r = c.get("/api/matrix/compare", params={"addr": name, **sc})
        assert r.status_code == 200, r.text
        out.append(scrub(r.json()))
        rv = c.post("/api/selection/script", json={"addr": "L1N5", "action": "revert"})
        assert rv.status_code == 200, rv.text
        e = c.post("/api/edit", json={"script": rv.json()["script"]}).json()
        assert e.get("ok") and e.get("committed"), e
        assert S.changed(s.ws, "L1N5")[0] == 0
        assert torch.allclose(s.ws.read(name, (), "current"), s.ws.read(name, (), "base"), atol=1e-3)
        answers[tag] = out
    assert answers["new"] == answers["old"]


def test_line_flags_of_an_integer_index():
    v = torch.zeros(6)
    v[2] = 1.0
    col = D.Piece("block", (range(0, 6), 3), values=v)
    assert S._line_flags(col, 1).tolist() == [True]                                  # one column, and it is edited
    assert S._line_flags(col, 0).tolist() == [False, False, True, False, False, False]
    assert S._line_flags(D.Piece("block", (range(0, 6), 3), values=torch.zeros(6)), 1).tolist() == [False]
    one = D.Piece("block", (2, 3), values=torch.tensor(4.0))
    assert S._line_flags(one, 0).tolist() == [True] and S._line_flags(one, 1).tolist() == [True]
    assert np.array_equal(S._line_flags(D.Piece("block", (range(0, 6), range(3, 4)), values=v.reshape(6, 1)), 1), [True])
