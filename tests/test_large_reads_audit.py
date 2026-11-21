"""Reads above the GGUF store's limit that PRs 25 and 28 did not cover, found by an audit of the routes and tools.

The GGUF store refuses one read above 16M values. On a 27B model an MLP matrix is 89M values, the
attention matrices 5M to 63M, the embedding and output matrices 1.27 billion each. Found and fixed here:

- rendering weights as text (`chunks.render_tensor`: the Work tab's chunks, /api/render, MCP `read_weights`,
  `read_chunk`, `inspect what=svd|auto`) read the whole tensor before looking at the format;
- the file route (`checkout_files`, `commit_files`) and `import_delta` read whole tensors;
- the matrix view and the before/after view read one band of pooled rows in one call, above the limit when the
  grid is only a few rows high (`max_h` small);
- the store's refusal did not say which tensor or how much was asked for.

Model-free: the tiny random model of tests/test_atlas_large_reads.py, whose store is wrapped to refuse a read above
CAP = 300 values with the GGUF store's own error. No quantized file, decoder or real model is used.

Scaling (`scaled`): every threshold that chooses a code path is scaled by the same factor as the store's limit,
k = CAP / 16M, so that each tiny matrix takes the path its real-size counterpart takes:

    workspace.STREAM_ELEMS, analysis.STREAM_ELEMS, compare.STREAM_ELEMS   16,000,000 -> 300
    analysis.DENSE_ELEMS                                                  40,000,000 -> 749
    analysis.EXACT_ELEMS                                                   8,000,000 -> 150

The MLP matrices (24 x 40 = 960 values) then stand for the 89M-value MLP matrices (above one read and above
DENSE_ELEMS); the vocabulary matrices (624) and the stacked attention matrices (576) for a 16M-40M matrix (above one
read, held whole by the views that need it). Thresholds left unscaled on purpose, whose failures at this size are
not real: the token-estimate guards, region_stats' 2000-value sample floor, the wiring slab size, W_pos (384 values).

Each result is compared with the same request on a workspace whose store has no limit ("free"), with the thresholds
that pick the same arithmetic but read in one call where the fix reads in parts. Where reading in parts changes a
figure by summation order only, the test bounds the difference and says so.
"""
import base64
import io
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_atlas_large_reads import CAP, REFUSAL, session  # noqa: E402
from wsp import analysis as an  # noqa: E402
from wsp import chunks, edits, tools  # noqa: E402
from wsp import compare as cmp  # noqa: E402
from wsp import delta as D  # noqa: E402
from wsp import workspace as W  # noqa: E402
from wsp.store import norm_index, region_size  # noqa: E402

K = CAP / 16_000_000
EDIT = "blocks[0].mlp.W_in[2] *= 3\nneuron(1, 3).w_out *= 0\nunembed.W_U[:, 4] *= -1"
BIG = ["blocks.0.mlp.W_in", "blocks.1.mlp.W_out", "unembed.W_U", "embed.W_E", "blocks.0.attn.W_Q"]


def scale_analysis(mp):
    """The analysis thresholds only (which branch a view takes), with workspace slabs left at one read."""
    mp.setattr(an, "DENSE_ELEMS", int(40_000_000 * K))
    mp.setattr(an, "EXACT_ELEMS", int(8_000_000 * K))


def scaled(mp):
    """Every path-choosing threshold scaled with the store's limit (see the module docstring)."""
    scale_analysis(mp)
    mp.setattr(W, "STREAM_ELEMS", CAP)
    mp.setattr(an, "STREAM_ELEMS", CAP)
    mp.setattr(cmp, "STREAM_ELEMS", CAP)


def refuse_above(ws, mp, cap=CAP):
    """The workspace's store refuses a read above `cap` values with the GGUF store's error; returns the read sizes.

    Except the reads of the tiny model's own runner (FunctionalRunner.params), which holds the whole model to run it,
    e.g. for an edit's drift check. A GGUF model is run by the packed runner instead, which reads only vectors and
    single embedding rows through the store and lays the patch pieces on top itself; it is not what is under test."""
    store, plain, sizes, runner = ws.store, ws.store.read, [], []
    ws._cache.clear()                    # whole tensors read so far would hide the store
    ws._cache_bytes = 0

    def read(name, region=None):
        if runner:
            return plain(name, region)
        n = region_size(norm_index((), store.shape(name)) if region is None else region)
        sizes.append(n)
        if n > cap:
            raise ValueError(REFUSAL)
        return plain(name, region)

    params = ws.runner.params

    def runner_params(*a, **kw):
        runner.append(1)
        try:
            return params(*a, **kw)
        finally:
            runner.pop()

    mp.setattr(store, "read", read)
    mp.setattr(ws.runner, "params", runner_params)
    return sizes


@pytest.fixture()
def two(tmp_path, monkeypatch):
    """(free, capped, sizes): two sessions with the same weights and the same committed edit; only the second
    session's store refuses large reads. The thresholds are not yet scaled."""
    out = []
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        s = session(tmp_path / d)
        assert s.apply(EDIT, "test")["committed"]
        s.ws.runner.run(s.ws.encode("ab"), "current")     # the tiny model's own runner loads its weights whole; a GGUF
        out.append(s)                                     # model runs in the native runner, which is not under test
    sizes = refuse_above(out[1].ws, monkeypatch)
    return out[0], out[1], sizes


def under_limit(sizes):
    assert sizes and max(sizes) <= CAP, max(sizes or [0])


def single(ws, addr):
    return ws.resolve(addr)[0]


# ------------------------------------------------------------------------------ weights as text
@pytest.mark.parametrize("fmt", ["q", "g3", "g2", "sparse"])
@pytest.mark.parametrize("addr,variant,rows", [(a, "current", None) for a in BIG] + [
    ("blocks.0.mlp.W_in", "delta", None), ("blocks.1.mlp.W_in", "delta", None),          # one row edited / all zero
    ("blocks.0.mlp.W_in", "base", (1, 20)),            # a row range above one read (19 x 40)
    ("blocks.0.attn.W_Q", "current", (5, 30))])        # rows of a stacked [head, d_model, d_head] tensor
def test_dense_formats_above_one_read_equal_a_whole_read(two, monkeypatch, fmt, addr, variant, rows):
    free, capped, sizes = two
    name, idx = single(free.ws, addr)
    want = chunks.render_tensor(free.ws, name, idx, fmt, variant, rows=rows)      # one read of the whole tensor
    scaled(monkeypatch)
    del sizes[:]
    got = chunks.render_tensor(capped.ws, name, idx, fmt, variant, rows=rows)
    if variant != "delta":                             # a delta is read from the patches alone
        under_limit(sizes)
    assert got == want


@pytest.mark.parametrize("addr", BIG)
def test_svd_text_above_one_read_equals_the_unlimited_store(two, monkeypatch, addr):
    """The MLP matrix (above DENSE_ELEMS) takes the streamed randomised SVD, the others the dense one, on both."""
    free, capped, sizes = two
    name, idx = single(free.ws, addr)
    with monkeypatch.context() as m:
        scale_analysis(m)
        want = chunks.render_tensor(free.ws, name, idx, "svd")
    scaled(monkeypatch)
    del sizes[:]
    got = chunks.render_tensor(capped.ws, name, idx, "svd")
    under_limit(sizes)
    assert got == want and "component 0" in got
    assert ("randomised" in got) == (region_size(norm_index(idx, free.ws.shape(name))) > an.DENSE_ELEMS)


def _stats_line(text):
    line = next(x for x in text.splitlines() if x.startswith("format stats:"))
    return line, dict(re.findall(r"(mean|std|min|max|norm) (\S+)", line)), re.search(r"zeros (\d+)/(\d+)", line).groups()


@pytest.mark.parametrize("addr", BIG)
def test_stats_text_above_one_read_matches_a_whole_read(two, monkeypatch, addr):
    """Summed slab by slab in float64: n, zeros, min, max and the row norms are equal; mean, std and norm agree to
    the 4 significant digits printed (the float32 whole-tensor sums can differ in the last float32 digits)."""
    free, capped, sizes = two
    name, idx = single(free.ws, addr)
    want = chunks.render_tensor(free.ws, name, idx, "stats")
    scaled(monkeypatch)
    del sizes[:]
    got = chunks.render_tensor(capped.ws, name, idx, "stats")
    under_limit(sizes)
    (gl, g, gz), (wl, w, wz) = _stats_line(got), _stats_line(want)
    assert got.replace(gl, "") == want.replace(wl, "") and "row norms: " in got
    assert gz == wz and g["min"] == w["min"] and g["max"] == w["max"]
    for key in ("mean", "std", "norm"):
        assert float(g[key]) == pytest.approx(float(w[key]), rel=1e-3, abs=1e-6), key


@pytest.mark.parametrize("part", [None, 100])
def test_slab_stats_bound_against_whole_tensor_stats(monkeypatch, part):
    """part: each slab summed in parts of that many values (as a 16M-value slab is, in parts of 1M)."""
    if part:
        monkeypatch.setattr(chunks, "STATS_PART", part)
    g = torch.Generator().manual_seed(3)
    m = torch.randn(301, 57, generator=g) * 0.02 + 0.001
    m[4, :9] = 0
    want = an.stats(m)
    slabs = [(a, m[a:a + 13]) for a in range(0, m.shape[0], 13)]
    got = chunks.slab_stats(slabs, row_norms=True)
    assert (got["n"], got["zeros"], got["min"], got["max"]) == (want["n"], want["zeros"], want["min"], want["max"])
    assert got["row_norms"] == m.norm(dim=1).tolist()
    assert abs(got["mean"] - want["mean"]) <= 1e-6 * float(m.abs().max())
    assert got["std"] == pytest.approx(want["std"], rel=1e-5) and got["norm"] == pytest.approx(want["norm"], rel=1e-5)
    m2 = m.clone()
    m2[200, 3] = float("nan")                           # in a later slab: NaN as torch's one read gives it
    nan = chunks.slab_stats([(a, m2[a:a + 13]) for a in range(0, m2.shape[0], 13)])
    assert all(v != v for v in (nan["min"], nan["max"], nan["abs_max"]))
    exact = m.double().flatten()
    assert got["mean"] == pytest.approx(float(exact.mean()), rel=1e-12)   # float64 throughout: the exact figure
    assert got["std"] == pytest.approx(float(exact.std()), rel=1e-12)


# ------------------------------------------------------------------------------ the Work tab, through the routes
def _client(sess, tmp, monkeypatch):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp / "config"))    # never a real user's settings
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp / "cache"))
    return TestClient(create_app(sess, "http://127.0.0.1:0"), base_url="http://127.0.0.1:0", raise_server_exceptions=False)


def _same(got, want, fmt):
    if fmt != "stats":
        assert got == want
        return
    gl, wl = got.splitlines(), want.splitlines()
    assert len(gl) == len(wl)
    for a, b in zip(gl, wl):
        if a != b:
            ga, wb = _stats_line(a), _stats_line(b)
            assert ga[2] == wb[2] and ga[1]["min"] == wb[1]["min"] and ga[1]["max"] == wb[1]["max"]
            for key in ("mean", "std", "norm"):
                assert float(ga[1][key]) == pytest.approx(float(wb[1][key]), rel=1e-3, abs=1e-6)


def _reference(monkeypatch, fmt, fn):
    """`fn()` on the free workspace with the thresholds that give the same arithmetic in one read."""
    with monkeypatch.context() as m:
        if fmt in ("svd", "auto"):
            scale_analysis(m)
        return fn()


@pytest.mark.parametrize("fmt", ["auto", "svd", "stats", "q"])
def test_work_tab_plan_and_chunks_through_the_routes(two, tmp_path, monkeypatch, fmt):
    """POST /api/plan then GET /api/chunk for every chunk, as web/js/work.js does (its default format is auto)."""
    free, capped, sizes = two
    c = _client(capped, tmp_path, monkeypatch)
    scaled(monkeypatch)
    r = c.post("/api/plan", json={"selection": ["blocks.0.mlp"], "budget": 40000, "format": fmt})
    assert r.status_code == 200, r.text[:300]
    plan = r.json()
    assert plan["chunks"] and any(region_size(norm_index(i, capped.ws.shape(n))) > CAP
                                  for ch in plan["chunks"] for a in ch["addrs"] for n, i in capped.ws.resolve(a))
    for ch in plan["chunks"]:
        del sizes[:]
        r = c.get("/api/chunk", params={"id": ch["id"], "variant": "current"})
        assert r.status_code == 200, r.text[:300]
        under_limit(sizes)
        want = _reference(monkeypatch, plan["format"], lambda: chunks.render_chunk(free.ws, ch))
        _same(r.text, want, plan["format"])


@pytest.mark.parametrize("fmt", ["svd", "stats", "sparse"])
def test_render_route_above_one_read(two, tmp_path, monkeypatch, fmt):
    free, capped, sizes = two
    c = _client(capped, tmp_path, monkeypatch)
    want = _reference(monkeypatch, fmt, lambda: chunks.render(free.ws, ["unembed.W_U"], fmt, "current", "full"))
    scaled(monkeypatch)
    del sizes[:]
    r = c.get("/api/render", params={"addr": "unembed.W_U", "format": fmt})
    assert r.status_code == 200, r.text[:300]
    under_limit(sizes)
    _same(r.text, want, fmt)


# ------------------------------------------------------------------------------ the MCP tools
@pytest.mark.parametrize("name,args,fmt", [
    ("inspect", {"address": "blocks.0.mlp.W_in", "what": "svd"}, "svd"),
    ("inspect", {"address": "blocks.0.mlp.W_in"}, "svd"),                       # what=auto on a tensor is svd
    ("inspect", {"address": "unembed.W_U", "what": "svd"}, "svd"),
    ("read_weights", {"address": "blocks.0.mlp.W_in", "format": "svd"}, "svd"),
    ("read_weights", {"address": "blocks.0.mlp.W_in", "format": "stats"}, "stats"),
    ("read_weights", {"address": "blocks.1.mlp.W_out", "format": "q"}, "q"),
    ("read_weights", {"address": "embed.W_E", "format": "sparse"}, "sparse")])
def test_mcp_tools_above_one_read(two, monkeypatch, name, args, fmt):
    free, capped, sizes = two
    want = _reference(monkeypatch, fmt, lambda: tools.call(free, name, args))
    scaled(monkeypatch)
    del sizes[:]
    got = tools.call(capped, name, args)
    under_limit(sizes)
    _same(got, want, fmt)
    assert "GGUF" not in got and "ERROR" not in got


@pytest.mark.parametrize("fmt", ["auto", "svd", "stats"])
def test_mcp_plan_chunks_and_read_chunk(two, monkeypatch, fmt):
    free, capped, sizes = two
    scaled(monkeypatch)
    listing = tools.call(capped, "plan_chunks", {"selection": ["blocks.0.mlp", "unembed"], "format": fmt})
    assert "chunk 0" in listing
    free.plan = capped.plan
    for ch in capped.plan["chunks"]:
        del sizes[:]
        got = tools.call(capped, "read_chunk", {"chunk_id": ch["id"]})
        under_limit(sizes)
        want = _reference(monkeypatch, capped.plan["format"], lambda: tools.call(free, "read_chunk", {"chunk_id": ch["id"]}))
        _same(got, want, capped.plan["format"])


# ------------------------------------------------------------------------------ the file route
def _files(sess):
    return os.path.join(sess.ws.dir, "agent", "weights")


def test_checkout_files_are_the_bytes_of_np_save(two, monkeypatch):
    free, capped, sizes = two
    scaled(monkeypatch)
    del sizes[:]
    out = tools.call(capped, "checkout_files", {"selection": ["blocks.0.mlp", "unembed", "blocks.0.attn.W_Q"]})
    under_limit(sizes)
    names = sorted(f[:-4] for f in os.listdir(_files(capped)) if f.endswith(".npy"))
    assert out.startswith(f"wrote {len(names)} files") and "unembed.W_U" in names and "blocks.0.attn.W_Q" in names
    for n in names:
        ref = io.BytesIO()
        np.save(ref, free.ws.read(n).numpy())          # what checkout wrote before: the whole current tensor
        with open(os.path.join(_files(capped), n + ".npy"), "rb") as f:
            assert f.read() == ref.getvalue(), n


def _edit_files(sess):
    d = _files(sess)
    w_in = np.load(os.path.join(d, "blocks.0.mlp.W_in.npy"))
    w_in[:, 5] += 0.5                                   # one column: every row slab changes
    w_in[17, 30] = 0.0
    np.save(os.path.join(d, "blocks.0.mlp.W_in.npy"), w_in)
    w_out = np.load(os.path.join(d, "blocks.0.mlp.W_out.npy"))
    w_out[7] *= 2                                       # one row: one slab changes
    np.save(os.path.join(d, "blocks.0.mlp.W_out.npy"), w_out)


def _round_trip(sess, sizes=None):
    sess.set_task({"scope": ["blocks.0.mlp"]})
    n_before = len(sess.ws.patches)
    assert tools.call(sess, "checkout_files", {}).startswith("wrote ")
    _edit_files(sess)
    if sizes is not None:
        del sizes[:]
    report = tools.call(sess, "commit_files", {"message": "from files"})
    assert "COMMITTED as patch" in report, report
    assert len(sess.ws.patches) == n_before + 1 and sess.ws.patches[-1].source == {"kind": "files"}
    return report


def _deltas(ws):
    return {n: ws.read(n, variant="delta") for n in ws.names()}        # from the patches alone, no store read


@pytest.mark.parametrize("keep_pieces", [False, True])
def test_checkout_edit_commit_round_trip_equals_the_unlimited_one(two, monkeypatch, keep_pieces):
    """keep_pieces: the patch keeps one piece per changed slab, as it does for a tensor above 4M values (here
    compact's dense limit is lowered so that the per-slab pieces are kept, not merged)."""
    free, capped, sizes = two
    if keep_pieces:
        plain = D.compact
        monkeypatch.setattr(D, "compact", lambda pieces, shape, dense_limit=0: plain(pieces, shape, dense_limit))
    want_report = _round_trip(free)
    scaled(monkeypatch)
    got_report = _round_trip(capped, sizes)
    under_limit(sizes)
    strip = lambda r: re.sub(r"patch \S+", "patch", r)                 # noqa: E731  (the patch id differs)
    if keep_pieces:          # how the patch stores the change differs (pieces, and a rank only for one piece)
        strip = lambda r: re.sub(r"; stored as [^)]*|, rank \d+|patch \S+", "", r)  # noqa: E731
    assert strip(got_report) == strip(want_report)
    want, got = _deltas(free.ws), _deltas(capped.ws)
    assert all(torch.equal(got[n], want[n]) for n in want) and bool(got["blocks.0.mlp.W_out"].any())
    pieces = capped.ws.patches[-1].deltas["blocks.0.mlp.W_in"]
    assert len(pieces) == (len(range(0, 24, CAP // 40)) if keep_pieces else 1)


def test_commit_files_alone_under_a_read_limit(two, monkeypatch):
    """commit_files by itself: the files are written by a checkout on the free workspace (the same weights)."""
    free, capped, sizes = two
    capped.set_task({"scope": ["blocks.0.mlp"]})
    edits.checkout(free.ws, ["blocks.0.mlp"], _files(capped))
    _edit_files(capped)
    want = _round_trip(free)
    scaled(monkeypatch)
    del sizes[:]
    got = tools.call(capped, "commit_files", {"message": "from files"})
    assert "COMMITTED" in got, got                     # the tool answers a refused read as "ERROR: ..."
    under_limit(sizes)
    assert re.sub(r"patch \S+", "patch", got) == re.sub(r"patch \S+", "patch", want)
    want_d, got_d = _deltas(free.ws), _deltas(capped.ws)
    assert all(torch.equal(got_d[n], want_d[n]) for n in want_d)


def test_commit_files_keeps_the_task_scope_under_a_read_limit(two, monkeypatch):
    _, capped, sizes = two
    scaled(monkeypatch)
    capped.set_task({"scope": ["blocks.0.mlp.W_in[:, 0:8]"]})
    assert tools.call(capped, "checkout_files", {}).startswith("wrote 1 files")
    d = _files(capped)
    w_in = np.load(os.path.join(d, "blocks.0.mlp.W_in.npy"))
    w_in[:, 20] += 1.0                                  # outside the scope
    np.save(os.path.join(d, "blocks.0.mlp.W_in.npy"), w_in)
    n = len(capped.ws.patches)
    del sizes[:]
    out = tools.call(capped, "commit_files", {})
    under_limit(sizes)
    assert out.startswith("ERROR: edit touches weights of blocks.0.mlp.W_in outside the task scope")
    assert len(capped.ws.patches) == n


def test_import_delta_under_a_read_limit_equals_the_unlimited_one(two, monkeypatch, tmp_path):
    from safetensors.torch import save_file
    free, capped, sizes = two
    other = {n: free.ws.read(n).clone() for n in free.ws.names()}
    other["blocks.0.mlp.W_in"][:, 3] += 1
    other["unembed.W_U"][0] -= 0.5
    path = str(tmp_path / "other.safetensors")
    save_file(other, path)
    want = edits.import_delta(free.ws, path)
    scaled(monkeypatch)
    del sizes[:]
    got = edits.import_delta(capped.ws, path)
    under_limit(sizes)
    assert sorted(got) == sorted(want) == ["blocks.0.mlp.W_in", "unembed.W_U"]
    for n in want:
        dense = []
        for ps in (got[n], want[n]):
            d = torch.zeros(free.ws.shape(n))
            for p in ps:
                D.add_to(d, norm_index((), free.ws.shape(n)), p)
            dense.append(d)
        assert torch.equal(*dense), n


# ------------------------------------------------------------------------------ views with few pooled rows
@pytest.mark.parametrize("pool", ["mean", "absmax", "rms"])
def test_band_pool_equals_pool_of_the_whole_band(pool):
    """A band of 97 x 301 values read in uneven parts. Above 20,000 values the colour-scale sample takes every
    n // 20,000-th value, so the parts must continue the whole band's stride (here every 1st, 2nd... not reachable
    with the tiny model's bands of at most 960 values)."""
    g = torch.Generator().manual_seed(5)
    band = torch.randn(97, 301, generator=g)
    band[3, 7] = float("nan")
    band[40, 2], band[41, 2] = 9.0, -9.0                # a tie in |value|: the first one is kept, as _pool keeps it
    fc = 8
    bp = an.BandPool(301, fc, pool, band.numel())
    for a, b in ((0, 1), (1, 13), (13, 41), (41, 42), (42, 97)):
        bp.add(band[a:b])
    want = an._pool(band, band.shape[0], fc, pool)
    if pool == "absmax":
        assert torch.equal(bp.result(), want)
    else:
        torch.testing.assert_close(bp.result(), want, rtol=1e-6, atol=1e-7)
    assert band.numel() // 20_000 == 1                  # 29,197 values: every value
    torch.testing.assert_close(torch.cat(bp.sample), band.flatten(), rtol=0, atol=0, equal_nan=True)
    big = torch.randn(150, 301, generator=g)            # 45,150 values: every 2nd
    bp = an.BandPool(301, fc, pool, big.numel())
    for a, b in ((0, 7), (7, 8), (8, 150)):
        bp.add(big[a:b])
    assert torch.equal(torch.cat(bp.sample), big.flatten()[::2])


def test_compare_accumulator_continues_the_stride_across_parts():
    g = torch.Generator().manual_seed(6)
    t = torch.randn(150, 301, generator=g)
    whole, parts = cmp._Acc(), cmp._Acc()
    whole.add(t)
    every = max(1, t.numel() // 20_000)
    for a, b in ((0, 7), (7, 8), (8, 150)):
        parts.add(t[a:b], a * 301, every)
    assert every == 2 and torch.equal(torch.cat(parts.sample), torch.cat(whole.sample))
    assert parts.result()["n"] == whole.result()["n"] and parts.result()["norm"] == pytest.approx(whole.result()["norm"], rel=1e-12)
    assert parts.scale() == whole.scale()


@pytest.mark.parametrize("pool", ["mean", "absmax", "rms"])
@pytest.mark.parametrize("kw", [{"max_h": 1}, {"max_h": 2, "max_w": 7}, {"max_h": 2, "r0": 1, "r1": 23, "c0": 2, "c1": 39}])
def test_matrix_view_with_a_band_above_one_read(two, monkeypatch, pool, kw):
    """The band of rows pooled into one grid row is read in parts. absmax and the colour scale are equal; mean and
    rms are summed in float64 here, so they agree with the float32 pooling of one read to float32 rounding."""
    free, capped, sizes = two
    addr = "blocks.0.mlp.W_in"
    with monkeypatch.context() as m:
        scale_analysis(m)                              # the streaming branch, one read per band
        want = an.matrix_view(free.ws, addr, pool=pool, **kw)
    scaled(monkeypatch)
    del sizes[:]
    got = an.matrix_view(capped.ws, addr, pool=pool, **kw)
    under_limit(sizes)
    assert not got["sortable"] and got["factor"][0] * len(range(kw.get("c0", 0), kw.get("c1", 40))) > CAP
    g, w = (np.frombuffer(base64.b64decode(x["data"]), np.float32) for x in (got, want))
    assert {k: v for k, v in got.items() if k != "data"} == {k: v for k, v in want.items() if k != "data"}
    if pool == "absmax":
        assert np.array_equal(g, w)
    else:
        np.testing.assert_allclose(g, w, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("kw", [{"max_h": 1}, {"max_h": 3, "pool": "absmax"}, {"max_h": 2, "r0": 1, "r1": 23, "c0": 2, "c1": 39}])
def test_matrix_compare_with_a_band_above_one_read(two, monkeypatch, kw):
    free, capped, sizes = two
    addr = "blocks.0.mlp.W_in"                          # row 2 is edited (x3)
    want = cmp.matrix_compare(free.ws, addr, **kw)     # unscaled: the whole region in one read
    scaled(monkeypatch)
    del sizes[:]
    got = cmp.matrix_compare(capped.ws, addr, **kw)
    under_limit(sizes)
    assert got["edited"] and got["stats"]["delta"]["changed"] == want["stats"]["delta"]["changed"] > 0
    for key in ("base", "current", "delta"):
        g, w = (np.frombuffer(base64.b64decode(x[key]), np.float32) for x in (got, want))
        if key == "delta" or kw.get("pool") == "absmax":
            assert np.array_equal(g, w), key
        else:
            np.testing.assert_allclose(g, w, rtol=1e-6, atol=1e-7)
    for key in ("base", "current"):                    # norms: float64 sums in a different order
        gs, ws_ = got["stats"][key], want["stats"][key]
        assert (gs["n"], gs["min"], gs["max"]) == (ws_["n"], ws_["min"], ws_["max"])
        assert gs["norm"] == pytest.approx(ws_["norm"], rel=1e-12)
    rest = lambda r: {k: v for k, v in r.items() if k not in ("base", "current", "delta", "stats")}  # noqa: E731
    assert rest(got) == rest(want)


def test_matrix_compare_of_a_stacked_tensor_above_one_read(two, monkeypatch):
    free, capped, sizes = two
    want = cmp.matrix_compare(free.ws, "blocks.0.attn.W_Q")
    scaled(monkeypatch)
    del sizes[:]
    got = cmp.matrix_compare(capped.ws, "blocks.0.attn.W_Q")
    under_limit(sizes)
    assert got == want


# ------------------------------------------------------------------------------ the refusal itself
def test_the_refusal_names_the_tensor_the_size_and_the_limit():
    pytest.importorskip("gguf")
    from wsp.gguf_store import MAX_READ, GGUFStore
    fake = SimpleNamespace(shape=lambda name: (17408, 5120))         # no file: the size check comes first
    with pytest.raises(ValueError) as e:
        GGUFStore.read(fake, "blk.3.ffn_up.weight")
    msg = str(e.value)
    assert msg.startswith(REFUSAL)
    assert "blk.3.ffn_up.weight" in msg and "89,128,960 values" in msg and f"limit {MAX_READ:,}" in msg


# ------------------------------------------------------------------------------ review: the file route under failure
# A checkout that fails or is killed part-way must not leave a whole-looking <name>.npy: commit_files takes every .npy
# in the folder, and unwritten rows would be committed as zeros. Another edit committed while the file route reads a
# tensor slab by slab must not be applied to some slabs only.
W_IN = "blocks.0.mlp.W_in"                             # 24 x 40 = 960 values: four slabs of 7, 7, 7, 3 rows


def _flaky(ws, mp, at, action):
    """`action()` inside the at-th store read of W_IN, then the read goes on (unless action raises)."""
    plain, n = ws.store.read, [0]

    def read(name, region=None):
        if name == W_IN:
            n[0] += 1
            if n[0] == at:
                action()
        return plain(name, region)

    mp.setattr(ws.store, "read", read)


def _fail():
    raise OSError("simulated I/O error")


@pytest.mark.parametrize("at", [1, 2, 4])
@pytest.mark.parametrize("before", [False, True])
def test_a_failed_checkout_leaves_no_file_and_commit_files_ignores_it(two, monkeypatch, at, before):
    """before: a complete W_in.npy from an earlier checkout is in the folder; it stays as it was (as on main)."""
    free, s, _ = two
    scaled(monkeypatch)
    s.set_task({"scope": ["blocks.0.mlp"]})
    d = _files(s)
    if before:
        tools.call(s, "checkout_files", {"selection": [W_IN]})
        earlier = open(os.path.join(d, W_IN + ".npy"), "rb").read()
    _flaky(s.ws, monkeypatch, at, _fail)
    with pytest.raises(OSError, match="simulated"):
        tools.call(s, "checkout_files", {"selection": [W_IN]})
    left = sorted(os.listdir(d))
    assert not any(f.endswith(".partial") for f in left)
    if before:
        assert left == sorted([W_IN + ".npy", "INDEX.md"]) and open(os.path.join(d, W_IN + ".npy"), "rb").read() == earlier
    else:
        assert W_IN + ".npy" not in left
    monkeypatch.undo()                                 # the store reads normally again; the other edit goes on
    scaled(monkeypatch)
    want_in = s.ws.read(W_IN).clone()
    tools.call(s, "checkout_files", {"selection": ["blocks.0.mlp.W_out"]})
    fo = os.path.join(d, "blocks.0.mlp.W_out.npy")
    b = np.load(fo)
    b[3] *= 2
    np.save(fo, b)
    out = tools.call(s, "commit_files", {"message": "edit W_out row 3"})
    assert "COMMITTED" in out and W_IN not in out, out
    assert torch.equal(s.ws.read(W_IN), want_in)


_CHILD = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
from wsp.session import Session
from wsp import edits, workspace as W
root, directory, at = sys.argv[2], sys.argv[3], int(sys.argv[4])
s = Session(root)
s.open("rnd")
W.STREAM_ELEMS = 300
s.ws._cache.clear()
plain, n = s.ws.store.read, [0]
def read(name, region=None):
    if name == "blocks.0.mlp.W_in":
        n[0] += 1
        if n[0] == at:
            os._exit(9)                                 # the process dies here (a crash, an out-of-memory kill)
    return plain(name, region)
s.ws.store.read = read
edits.checkout(s.ws, ["blocks.0.mlp.W_in"], directory)
os._exit(0)
"""


@pytest.mark.parametrize("at", [1, 3])
def test_a_checkout_killed_part_way_leaves_only_a_partial_file(tmp_path, monkeypatch, at):
    import subprocess
    s = session(tmp_path)
    d = str(tmp_path / "files")
    os.makedirs(d)
    r = subprocess.run([sys.executable, "-c", _CHILD, ROOT, str(tmp_path), d, str(at)], timeout=120, capture_output=True)
    assert r.returncode == 9, r.stderr.decode()[-500:]
    assert os.listdir(d) == [W_IN + ".npy.partial"]   # never under the name commit_files reads
    assert edits.read_files(s.ws, d) == {}
    scaled(monkeypatch)
    w_out = "blocks.0.mlp.W_out"
    edits.checkout(s.ws, [w_out], d)                   # a later checkout, of another tensor, removes it
    assert sorted(os.listdir(d)) == sorted([w_out + ".npy", "INDEX.md"])
    ref = io.BytesIO()
    np.save(ref, s.ws.read(w_out).numpy())
    assert open(os.path.join(d, w_out + ".npy"), "rb").read() == ref.getvalue()


@pytest.mark.parametrize("shape", [(0, 5), (), (3,), (2, 3, 4), (24, 40)])
def test_write_npy_has_the_bytes_of_np_save(tmp_path, shape):
    a = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) * 0.37 - 1
    path = str(tmp_path / "t.npy")
    slabs = [a] if a.ndim == 0 else [a[i:i + 7] for i in range(0, max(1, a.shape[0]), 7)]
    edits._write_npy(path, shape, slabs)
    ref = io.BytesIO()
    np.save(ref, a)
    assert open(path, "rb").read() == ref.getvalue() and os.listdir(tmp_path) == ["t.npy"]


def _other_edit(ws):
    """Another edit (e.g. the browser's /api/edit): +2 on column 9 of every row of W_IN."""
    return lambda: ws.commit({W_IN: [D.block((range(24), 9), torch.full((24,), 2.0))]}, "other edit", "human")


@pytest.mark.parametrize("at", [2, 3, 4])
def test_an_edit_committed_during_commit_files_is_kept_whole(two, monkeypatch, at):
    _, s, _ = two
    scaled(monkeypatch)
    s.set_task({"scope": ["blocks.0.mlp"]})
    tools.call(s, "checkout_files", {"selection": [W_IN]})
    f = os.path.join(_files(s), W_IN + ".npy")
    a = np.load(f)
    a[0, 0] += 1.0                                      # the agent's edit: one weight
    np.save(f, a)
    before, base = s.ws.ref(W_IN).whole(), s.ws.ref(W_IN).whole("base")
    s.ws._cache.clear()
    _flaky(s.ws, monkeypatch, at, _other_edit(s.ws))
    assert "COMMITTED" in tools.call(s, "commit_files", {"message": "files"})
    change = s.ws.ref(W_IN).whole() - before
    torch.testing.assert_close(change[:, 9], torch.full((24,), 2.0))   # the other edit, on every row (float rounding)
    assert float(change[0, 0]) == pytest.approx(1.0) and int((change != 0).sum()) == 25
    assert torch.equal(s.ws.ref(W_IN).whole("base"), base)


@pytest.mark.parametrize("at", [2, 3, 4])
def test_a_checkout_is_one_moment_of_the_weights(two, monkeypatch, at):
    _, s, _ = two
    scaled(monkeypatch)
    before = s.ws.ref(W_IN).whole()
    s.ws._cache.clear()
    _flaky(s.ws, monkeypatch, at, _other_edit(s.ws))
    tools.call(s, "checkout_files", {"selection": [W_IN]})
    assert np.array_equal(np.load(os.path.join(_files(s), W_IN + ".npy")), before.numpy())   # no row has the other edit
    assert not torch.equal(s.ws.ref(W_IN).whole(), before)


# ------------------------------------------------------------------------------ review: rendering edge cases
def test_svd_text_of_zero_rows_and_row_ranges_equals_a_whole_read(two, monkeypatch):
    """Main's order: rows that are all zero give "all zero" before any format; an svd of a 2-D row range falls back
    to g3; an svd of rows of a stacked tensor is the svd of the whole tensor."""
    free, capped, sizes = two
    for s in (free, capped):
        assert s.apply("blocks[0].attn.W_Q[1] *= 2", "head 1")["committed"]
    cases = [("blocks.0.attn.W_Q", "delta", (0, 24)), ("blocks.0.attn.W_Q", "delta", (3, 20)),   # head 0: all zero
             ("blocks.0.attn.W_Q", "delta", (20, 30)), ("blocks.0.attn.W_Q", "delta", None),
             ("blocks.0.mlp.W_in", "current", (1, 20)), ("blocks.1.mlp.W_in", "delta", None),
             ("blocks.1.mlp.W_in", "delta", (0, 20))]
    want = [chunks.render_tensor(free.ws, *single(free.ws, a), "svd", v, rows=r) for a, v, r in cases]
    scaled(monkeypatch)
    for (a, v, r), w in zip(cases, want):
        assert chunks.render_tensor(capped.ws, *single(capped.ws, a), "svd", v, rows=r) == w, (a, v, r)
    assert want[0].endswith("all zero") and want[5].endswith("all zero") and "format g3" in want[4]
    assert "component 0" in want[2]


def _planted(tmp_path, monkeypatch):
    """(free, capped, sizes) with inf and NaN in the base weights (they cannot come from a patch)."""
    from wsp.model import Config, init_params
    from wsp.session import Session
    from wsp.tokenizer import VocabTokenizer
    from wsp.workspace import Workspace
    cfg = Config(d_vocab=26, n_ctx=16, d_model=24, n_layers=2, n_heads=2, d_mlp=40, name="rnd")
    out = []
    for d in ("a", "b"):
        p = init_params(cfg, seed=1, std=0.3)
        p["blocks.0.mlp.W_in"][17, 3] = float("nan")
        p["blocks.0.mlp.W_in"][20, 5], p["blocks.0.mlp.W_in"][21, 6] = -50.0, 50.0   # in the NaN's band, another part
        p["blocks.0.mlp.W_out"][9, 5] = float("inf")
        p["blocks.1.mlp.W_out"][30, 2], p["blocks.1.mlp.W_out"][31, 2] = float("inf"), float("-inf")
        Workspace.create(str(tmp_path / d / "rnd"), cfg, VocabTokenizer(list("abcdefghijklmnopqrstuvwxyz"), "char"), p)
        s = Session(str(tmp_path / d))
        s.open("rnd")
        out.append(s)
    return out[0], out[1], refuse_above(out[1].ws, monkeypatch)


def _outcome(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


PLANTED = ["blocks.0.mlp.W_in", "blocks.0.mlp.W_out", "blocks.1.mlp.W_out"]


@pytest.mark.parametrize("fmt", ["q", "g3", "sparse", "stats", "svd"])
def test_text_of_weights_with_inf_or_nan_equals_a_whole_read(tmp_path, monkeypatch, fmt):
    free, capped, sizes = _planted(tmp_path, monkeypatch)
    for addr in PLANTED:
        n, i = single(free.ws, addr)
        with monkeypatch.context() as m:
            if fmt == "svd":
                scale_analysis(m)
            want = _outcome(lambda: chunks.render_tensor(free.ws, n, i, fmt))
        with monkeypatch.context() as m:
            scaled(m)
            del sizes[:]
            got = _outcome(lambda: chunks.render_tensor(capped.ws, n, i, fmt))
        assert got == want, addr
        if fmt == "stats":                             # analysis.stats refuses a histogram range that is not finite
            assert got.startswith("RuntimeError: torch.histc: range of")
        assert "GGUF" not in got
        under_limit(sizes)


@pytest.mark.parametrize("pool", ["mean", "absmax", "rms"])
def test_pooled_views_of_weights_with_inf_or_nan(tmp_path, monkeypatch, pool):
    free, capped, sizes = _planted(tmp_path, monkeypatch)
    for addr in PLANTED:
        with monkeypatch.context() as m:
            scale_analysis(m)
            want = an.matrix_view(free.ws, addr, pool=pool, max_h=1)
        with monkeypatch.context() as m:
            scaled(m)
            del sizes[:]
            got = an.matrix_view(capped.ws, addr, pool=pool, max_h=1)
        under_limit(sizes)
        g, w = (np.frombuffer(base64.b64decode(x["data"]), np.float32) for x in (got, want))
        np.testing.assert_allclose(g, w, rtol=1e-6, atol=1e-7, equal_nan=True)
        assert np.isfinite(w).all() == (addr == "blocks.0.mlp.W_in" or pool == "absmax")   # NaN is missing in a mean
        assert {k: v for k, v in got.items() if k not in ("data", "scale")} == \
               {k: v for k, v in want.items() if k not in ("data", "scale")}
        assert got["scale"] == want["scale"] or (got["scale"] != got["scale"] and want["scale"] != want["scale"])


def test_compare_view_of_weights_with_inf_or_nan(tmp_path, monkeypatch):
    """The reference reads one pooled band per read (C.STREAM_ELEMS = 640, max_h = 3: bands of 8 x 40 = 320
    values), the fix reads each band in parts: min and max must be those of one read of each band."""
    free, capped, sizes = _planted(tmp_path, monkeypatch)
    for addr in PLANTED:
        with monkeypatch.context() as m:
            scale_analysis(m)
            m.setattr(cmp, "STREAM_ELEMS", 640)
            want = cmp.matrix_compare(free.ws, addr, max_h=3)
        with monkeypatch.context() as m:
            scaled(m)
            del sizes[:]
            got = cmp.matrix_compare(capped.ws, addr, max_h=3)
        under_limit(sizes)
        for key in ("base", "current"):
            g, w = got["stats"][key], want["stats"][key]
            assert (g["n"], repr(g["min"]), repr(g["max"])) == (w["n"], repr(w["min"]), repr(w["max"])), (addr, key)
            g, w = (np.frombuffer(base64.b64decode(x[key]), np.float32) for x in (got, want))
            np.testing.assert_allclose(g, w, rtol=1e-6, atol=1e-7, equal_nan=True)


# ------------------------------------------------------------------------------ review: sample strides above 1
def _wide(tmp_path, monkeypatch, cap):
    """Two sessions of a one-layer model with 2,600 neurons: W_in is 24 x 2600 = 62,400 values, so one pooled band
    of all its rows is sampled every 3rd value (62,400 // 20,000). Thresholds scaled to `cap`."""
    from wsp.model import Config, init_params
    from wsp.session import Session
    from wsp.tokenizer import VocabTokenizer
    from wsp.workspace import Workspace
    cfg = Config(d_vocab=26, n_ctx=16, d_model=24, n_layers=1, n_heads=2, d_mlp=2600, name="rnd")
    out = []
    for d in ("a", "b"):
        Workspace.create(str(tmp_path / d / "rnd"), cfg, VocabTokenizer(list("abcdefghijklmnopqrstuvwxyz"), "char"),
                         init_params(cfg, seed=2, std=0.3))
        s = Session(str(tmp_path / d))
        s.open("rnd")
        assert s.apply("blocks[0].mlp.W_in[5] *= 3", "edit")["committed"]
        out.append(s)
    return out[0], out[1], refuse_above(out[1].ws, monkeypatch, cap)


def _scaled_to(mp, cap):
    k = cap / 16_000_000
    mp.setattr(an, "DENSE_ELEMS", int(40_000_000 * k))
    mp.setattr(an, "EXACT_ELEMS", int(8_000_000 * k))
    for mod in (W, an, cmp):
        mp.setattr(mod, "STREAM_ELEMS", cap)


def test_band_samples_at_stride_3_in_the_matrix_and_compare_views(tmp_path, monkeypatch):
    cap = 6000                                          # parts of 2 rows (view) and 1 row (compare): 5,200 and 2,600
    free, capped, sizes = _wide(tmp_path, monkeypatch, cap)   # values, neither a multiple of 3
    addr = "blocks.0.mlp.W_in"
    with monkeypatch.context() as m:
        m.setattr(an, "DENSE_ELEMS", int(40_000_000 * cap / 16_000_000))
        want_v = an.matrix_view(free.ws, addr, max_h=1)
    want_c = cmp.matrix_compare(free.ws, addr, max_h=1)
    _scaled_to(monkeypatch, cap)
    del sizes[:]
    got_v = an.matrix_view(capped.ws, addr, max_h=1)
    got_c = cmp.matrix_compare(capped.ws, addr, max_h=1)
    assert sizes and max(sizes) <= cap
    assert got_v["scale"] == want_v["scale"]
    assert (got_c["scale"], got_c["delta_scale"]) == (want_c["scale"], want_c["delta_scale"])


def test_band_pool_sample_at_stride_3():
    g = torch.Generator().manual_seed(7)
    band = torch.randn(200, 301, generator=g)            # 60,200 values: every 3rd
    for parts in ((0, 7, 8, 200), (0, 1, 2, 3, 150, 200)):
        bp, acc = an.BandPool(301, 8, "mean", band.numel()), cmp._Acc()
        for a, b in zip(parts, parts[1:]):
            bp.add(band[a:b])
            acc.add(band[a:b], a * 301, 3, part=True)
        acc.close()
        assert torch.equal(torch.cat(bp.sample), band.flatten()[::3])
        assert torch.equal(torch.cat(acc.sample), band.flatten()[::3])
