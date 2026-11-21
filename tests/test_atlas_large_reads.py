"""The atlas on a store that refuses a large single read, as the GGUF store does above 16M values.

Reported on a 27B GGUF model: building the weights stage answered "ValueError: GGUF read exceeds 16M logical
values; use row slabs / a smaller selection", because the atlas read each MLP matrix, and each chunk of the
vocabulary matrices, in one call.

Model-free: a tiny random in-memory model of this project's kind (as tests/test_atlas.py builds). Its store is
wrapped to refuse a read above a few hundred values with the GGUF store's own error, and the slab size is lowered
to match, so every matrix of the model is "large". No quantized file, decoder or real model is used.
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from wsp import atlas  # noqa: E402
from wsp import workspace as W  # noqa: E402
from wsp.atlas import guide  # noqa: E402
from wsp.model import Config, init_params  # noqa: E402
from wsp.session import Session  # noqa: E402
from wsp.store import norm_index, region_size  # noqa: E402
from wsp.workspace import Workspace  # noqa: E402

quiet = lambda *a: None  # noqa: E731
CAP = 300          # above every vector and per-head slice of the model below, under every whole matrix (576..960)
REFUSAL = "GGUF read exceeds 16M logical values; use row slabs / a smaller selection"


def session(root):
    from wsp.tokenizer import VocabTokenizer
    tok = VocabTokenizer(list("abcdefghijklmnopqrstuvwxyz"), "char")
    cfg = Config(d_vocab=26, n_ctx=16, d_model=24, n_layers=2, n_heads=2, d_mlp=40, name="rnd")
    Workspace.create(str(root / "rnd"), cfg, tok, init_params(cfg, seed=1, std=0.3))
    s = Session(str(root))
    s.open("rnd")
    return s


def cap_reads(ws, monkeypatch, cap=CAP):
    """Make the workspace's store refuse a read above `cap` values, and slabs no larger than that."""
    store, plain, sizes = ws.store, ws.store.read, []
    ws._cache.clear()                    # whole tensors read so far (by the runner's load) would hide the store
    ws._cache_bytes = 0

    def read(name, region=None):
        n = region_size(norm_index((), store.shape(name)) if region is None else region)
        sizes.append(n)
        if n > cap:
            raise ValueError(REFUSAL)
        return plain(name, region)

    monkeypatch.setattr(store, "read", read)
    monkeypatch.setattr(W, "STREAM_ELEMS", cap)
    return sizes


@pytest.fixture()
def pair(tmp_path, monkeypatch):
    """Two workspaces with the same weights: one plain, one whose store refuses large reads."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    free, capped = session(tmp_path / "a"), session(tmp_path / "b")
    atlas.build(free.ws, stages=["weights", "wiring"], log=quiet)      # before the slab size is lowered
    capped.ws.runner.run(capped.ws.encode("ab"), "current")            # the tiny model's own runner loads its weights
    # whole; a GGUF model runs in the native runner instead, so that load is not what is under test
    sizes = cap_reads(capped.ws, monkeypatch)
    return free, capped, sizes


def files(ws):
    d = atlas.atlas_dir(ws)
    return sorted(f for f in os.listdir(d) if f not in ("meta.json", "build.lock"))   # build.lock: the build's lock file


def test_the_store_under_test_refuses_a_whole_matrix(pair):
    _, capped, _ = pair
    name = capped.ws.sem.mlp(0)["W_out"].name
    with pytest.raises(ValueError, match="GGUF read exceeds"):
        capped.ws.store.read(name)


def test_weights_and_wiring_build_under_a_read_limit_and_match_an_unlimited_build(pair):
    free, capped, sizes = pair
    meta = atlas.build(capped.ws, stages=["weights", "wiring"], log=quiet)
    assert set(meta["stages"]) == {"weights", "wiring"}
    assert sizes and max(sizes) <= CAP                       # no read was refused, and none was above the limit
    assert files(capped.ws) == files(free.ws) and any(f.startswith("wiring_") for f in files(free.ws))
    for f in files(free.ws):
        a, b = os.path.join(atlas.atlas_dir(free.ws), f), os.path.join(atlas.atlas_dir(capped.ws), f)
        if f.endswith(".npz"):
            x, y = np.load(a), np.load(b)
            assert sorted(x.files) == sorted(y.files), f
            for k in x.files:
                assert np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind == "f"), (f, k)
        elif f.endswith(".npy"):
            assert np.array_equal(np.load(a), np.load(b)), f
        else:
            assert json.load(open(a)) == json.load(open(b)), f


def test_attribution_runs_under_a_read_limit_and_matches(pair):
    free, capped, sizes = pair
    atlas.build(capped.ws, stages=["weights"], log=quiet)
    want = guide.attribute(free.ws, "abcab", "c", None, top=5)       # no "versus": the mean unembedding is read
    del sizes[:]
    got = guide.attribute(capped.ws, "abcab", "c", None, top=5)
    assert max(sizes) <= CAP
    assert got == want and got["by_layer"]


@pytest.mark.parametrize("variant", ["base", "current", "delta"])
def test_whole_equals_read_for_every_kind_of_view(tmp_path, monkeypatch, variant):
    s = session(tmp_path)
    ws = s.ws
    assert s.apply("neuron(1, 3).w_out *= 0\nblocks[0].mlp.W_in[2] *= 3", "test")["committed"]
    m, sem = ws.sem.mlp(1), ws.sem
    views = [m["W_out"], m["W_in"], sem.W_U(), sem.W_E(), sem.W_U().sub((slice(None), slice(3, 19))),
             sem.W_E().sub(slice(5, 25)), m["W_out"].sub(7), m["b_in"], ws.sem.mlp(0)["W_in"]]
    want = [v.read(variant).clone() for v in views]
    sizes = cap_reads(ws, monkeypatch, 100)
    for v, w in zip(views, want):
        got = v.whole(variant)
        assert got.shape == w.shape and torch.equal(got, w), v.address()
    assert variant == "delta" or (sizes and max(sizes) <= 100)       # a delta is read from the patches alone
    extra = {m["W_out"].name: [m["W_out"].piece(torch.ones(m["W_out"].shape))]}
    if variant == "current":                                         # uncommitted pieces are laid on top, as in read
        assert torch.equal(m["W_out"].whole("current", extra), want[0] + 1)


# Trace and the matrix panels. Reported on the same 27B GGUF model: Trace answered the store's refusal, because the
# per-head attribution read a layer's whole attention output matrix (5120 x 6144, 31M values) in one call. The
# matrix view (when sorting) and the SVD panel hold a matrix of up to analysis.DENSE_ELEMS (40M) whole, which is
# also above one store read.

def test_trace_runs_under_a_read_limit_and_matches(pair):
    from wsp import analysis
    free, capped, sizes = pair
    want = analysis.trace(free.ws, "abcab", target="c", versus="d")
    del sizes[:]
    got = analysis.trace(capped.ws, "abcab", target="c", versus="d")
    assert sizes and max(sizes) <= CAP
    assert got == want and any(d["kind"] == "head" for d in got["dla"]) and got["neurons"]


@pytest.mark.parametrize("role", ["W_Q", "W_K", "W_V", "W_O"])
def test_stacked_heads_are_read_in_slabs_and_match(pair, role):
    free, capped, sizes = pair
    del sizes[:]
    got = capped.ws.sem.stack(1, role)
    assert sizes and max(sizes) <= CAP
    assert torch.equal(got, free.ws.sem.stack(1, role))


@pytest.mark.parametrize("addr", ["blocks.0.mlp.W_out", "blocks.1.attn.W_O", "blocks.0.attn.W_Q", "unembed.W_U"])
def test_matrix_view_sorted_and_svd_under_a_read_limit_match(pair, addr):
    from wsp import analysis
    free, capped, sizes = pair
    assert region_size(norm_index((), capped.ws.shape(analysis._single(capped.ws, addr)[0]))) > CAP
    for call in (lambda ws: analysis.matrix_view(ws, addr, sort_rows="norm", sort_cols="norm"),
                 lambda ws: analysis.matrix_view(ws, addr),
                 lambda ws: analysis.svd_view(ws, addr)):
        want = call(free.ws)
        del sizes[:]
        got = call(capped.ws)
        assert sizes and max(sizes) <= CAP
        assert got == want
