"""The atlas wiring stage: saved after every layer, continued after an interruption with the same arrays bit for bit,
never read while partly built, and estimated before it starts.

Model-free: a tiny random in-memory model (a few thousand weights, never trained; the forward pass is not run)
and a fake gated "model" whose read and write vectors are random tensors.  The estimate's arithmetic is checked with
an injected timing function, never against the wall clock.
"""
import json
import os
import shutil
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wsp import atlas  # noqa: E402
from wsp.atlas import wiring as W  # noqa: E402
from wsp.model import Config, init_params  # noqa: E402
from wsp.session import Session  # noqa: E402
from wsp.workspace import Workspace  # noqa: E402

quiet = lambda *a: None  # noqa: E731


class Stop(Exception):
    pass


@pytest.fixture()
def rnd(tmp_path):
    from wsp.tokenizer import VocabTokenizer
    tok = VocabTokenizer(list("abcdefghijklmnopqrstuvwxyz"), "char")
    cfg = Config(d_vocab=26, n_ctx=16, d_model=24, n_layers=4, n_heads=2, d_mlp=40, name="rnd4")
    Workspace.create(str(tmp_path / "rnd4"), cfg, tok, init_params(cfg, seed=3, std=0.3))
    s = Session(str(tmp_path))
    s.open("rnd4")
    return s


def adir(ws):
    return atlas.atlas_dir(ws)


def files(ws):
    """Every finished wiring file of the atlas, as arrays."""
    out = {}
    for l in W.neuron_layers(ws):
        with np.load(W.final_path(adir(ws), l)) as z:
            out[l] = {k: z[k] for k in z.files}
    return out


def same(a, b):
    """Bit for bit: same layers, keys, dtypes, shapes and bytes."""
    assert a.keys() == b.keys()
    for l in a:
        assert a[l].keys() == b[l].keys()
        for k in a[l]:
            assert a[l][k].dtype == b[l][k].dtype and a[l][k].shape == b[l][k].shape, (l, k)
            assert a[l][k].tobytes() == b[l][k].tobytes(), (l, k)


def stop_after(k, lines=None):
    """A progress log that stops the build after k writer layers have been saved (k = 0: while reading)."""
    seen = {"n": 0}

    def log(line):
        if lines is not None:
            lines.append(line)
        if "read vectors of layer" in line and k == 0:
            raise Stop("stopped while reading")
        if "partners of layer" in line:
            seen["n"] += 1
            if seen["n"] >= k:
                raise Stop(f"stopped after {k} layers")
    return log


def no_wiring_visible(ws):
    A = atlas.Atlas(ws)                                   # a fresh reader, as after a restart
    st = A.status()
    assert st["wiring_built"] is False and "wiring" not in st["stages"]
    assert not A.has("wiring")
    for l in A.layers:
        assert "w_down_gate_cos" not in A.layer(l)
    assert "wiring" not in A.unit(A.layers[0], 1)
    return st


def old_wiring(ws, variant="base", keep=2):
    """The search exactly as it was before per-layer saving (copied from the previous build_wiring)."""
    from wsp.atlas.build import layer_vectors, work_dtype
    s = ws.struct
    dt = work_dtype()
    ls = [l for l in range(s.n_layers) if s.has_neurons(l)]
    unit = lambda M: (M / M.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(dt)            # noqa: E731
    kinds = lambda l: ("gate", "up") if "W_gate" in s.layers[l].mlp else ("gate",)        # noqa: E731
    reads = {}
    for l in ls:
        v = layer_vectors(ws, l, variant)
        reads[l] = {"gate": unit(v["G"]), **({"up": unit(v["U"])} if "U" in v else {})}

    def blank(n):
        return torch.zeros(n, keep), torch.full((n, keep), -1, dtype=torch.int32), torch.full((n, keep), -1, dtype=torch.int32)

    def merge(cur, v, i, layer):
        cv, cl, ci = torch.cat([cur[0], v], 1), torch.cat([cur[1], torch.full_like(i, layer)], 1), torch.cat([cur[2], i], 1)
        o = cv.abs().topk(keep, dim=1).indices
        return cv.gather(1, o), cl.gather(1, o), ci.gather(1, o)

    down = {(l, k): blank(s.mlp_width(l)) for l in ls for k in ("gate", "up")}
    up = {(l, k): blank(s.mlp_width(l)) for l in ls for k in ("gate", "up")}
    for a, lw in enumerate(ls[:-1]):
        DT = unit(layer_vectors(ws, lw, variant)["D"]).T.contiguous()
        for lr in ls[a + 1:]:
            for k in kinds(lr):
                C = (reads[lr][k] @ DT).float()
                ab = C.abs()
                i0 = ab.argmax(0)
                down[(lw, k)] = merge(down[(lw, k)], C.gather(0, i0[None]).T, i0[:, None].int(), lr)
                i1 = ab.argmax(1)
                up[(lr, k)] = merge(up[(lr, k)], C.gather(1, i1[:, None]), i1[:, None].int(), lw)
    out = {}
    for l in ls:
        r = {}
        for d_, tab in (("down", down), ("up", up)):
            for k in ("gate", "up"):
                v, ll, ii = tab[(l, k)]
                r[f"w_{d_}_{k}_cos"], r[f"w_{d_}_{k}_layer"], r[f"w_{d_}_{k}_idx"] = \
                    v.numpy().astype(np.float16), ll.numpy().astype(np.int16), ii.numpy().astype(np.int32)
        out[l] = r
    return out


@pytest.fixture()
def built(rnd):
    """Weights and wiring built without interruption; returns (session, reference arrays)."""
    atlas.build(rnd.ws, stages=["weights", "wiring"], log=quiet)
    return rnd, files(rnd.ws)


# ----------------------------------------------------------------------------------------- uninterrupted
def test_uninterrupted_build_equals_the_previous_search_and_leaves_nothing_behind(built):
    sess, ref = built
    same(ref, old_wiring(sess.ws))                          # no gate/up re-read, same arrays
    assert not os.path.exists(W.partial_dir(adir(sess.ws)))
    A = atlas.Atlas(sess.ws)
    st = A.status()
    assert st["wiring_built"] and "wiring" in st["stages"] and st["wiring_progress"] is None
    assert "w_down_gate_cos" in A.layer(0) and "wiring" in A.unit(0, 1)


# ----------------------------------------------------------------------------------------- interrupted, continued
@pytest.mark.parametrize("k", [0, 1, 2, 3])                 # 4 layers: 3 writer passes; 3 is the last one
def test_an_interrupted_build_continues_to_the_same_arrays(built, k):
    sess, ref = built
    ws = sess.ws
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=stop_after(k))
    st = no_wiring_visible(ws)
    assert st["wiring_progress"]["done"] == (4 if k == 3 else k) and st["wiring_progress"]["total"] == 4
    lines = []
    atlas.build(ws, stages=["wiring"], log=lines.append)
    if 0 < k < 3:
        assert f"  continuing from layer {k} of 4 ({k} layers already done)" in lines
        assert sum("partners of layer" in x for x in lines) == 3 - k       # only the layers that were left
        assert sum("read vectors of layer" in x for x in lines) == 3 - k   # only the reads still needed
    elif k == 3:
        assert "  continuing: all 4 layers were already done" in lines
        assert not any("partners of layer" in x for x in lines)
    else:
        assert not any("continuing" in x for x in lines)
    same(ref, files(ws))
    assert not os.path.exists(W.partial_dir(adir(ws)))
    A = atlas.Atlas(ws)
    assert A.status()["wiring_built"] and "w_down_gate_cos" in A.layer(3)


def test_two_interruptions_in_a_row(built):
    sess, ref = built
    ws = sess.ws
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=stop_after(1))
    with pytest.raises(Stop):
        atlas.build(ws, stages=["wiring"], log=stop_after(1))
    assert atlas.Atlas(ws).status()["wiring_progress"]["done"] == 2
    atlas.build(ws, stages=["wiring"], log=quiet)
    same(ref, files(ws))


def test_progress_callback_raising_is_an_interruption_too(built):
    sess, ref = built
    ws = sess.ws
    seen = []

    def progress(p):
        seen.append(p)
        if p["phase"] == "searching" and p["done"] == 2:
            raise Stop()
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=quiet, progress=progress)
    assert atlas.Atlas(ws).status()["wiring_progress"]["done"] == 2
    atlas.build(ws, stages=["wiring"], log=quiet)
    same(ref, files(ws))


@pytest.mark.parametrize("mode", ["before", "after"])
def test_a_kill_at_any_write_leaves_the_old_or_the_new_state(built, monkeypatch, mode):
    """Stop the stage at every file operation (before it happens, or right after): never a torn or wrongly
    complete state, and the next build always ends with the same arrays."""
    sess, ref = built
    ws = sess.ws
    real = {"_replace": W._replace, "_unlink": W._unlink, "_rmtree": W._rmtree}
    calls = []

    def counting(name):
        def op(*a):
            calls.append((name, a))
            return real[name](*a)
        return op
    for name in real:
        monkeypatch.setattr(W, name, counting(name))
    atlas.build(ws, stages=["weights", "wiring"], force=True, log=quiet)
    total = len(calls)
    assert total > 10 and any(n == "_replace" and a[1].endswith(W.MARKER) for n, a in calls)
    same(ref, files(ws))
    for i in range(total):
        count = {"n": 0}

        def failing(name):
            def op(*a):
                count["n"] += 1
                if count["n"] - 1 == i and mode == "before":
                    raise Stop(f"killed before {name} {a}")
                r = real[name](*a)
                if count["n"] - 1 == i:
                    raise Stop(f"killed after {name} {a}")
                return r
            return op
        for name in real:
            monkeypatch.setattr(W, name, failing(name))
        with pytest.raises(Stop):
            atlas.build(ws, stages=["weights", "wiring"], force=True, log=quiet)
        for name in real:
            monkeypatch.setattr(W, name, real[name])
        A = atlas.Atlas(ws)
        json.load(open(os.path.join(adir(ws), "meta.json")))               # never torn
        if A.has("wiring"):                                                # only after the stage was finished
            assert not os.path.exists(W.marker_path(adir(ws)))
            same(ref, files(ws))
        else:
            no_wiring_visible(ws)
        lines = []
        atlas.build(ws, stages=["wiring"], log=lines.append)
        same(ref, files(ws))
        assert atlas.Atlas(ws).status()["wiring_built"], (i, mode)
        assert not os.path.exists(W.partial_dir(adir(ws))), (i, mode)


def test_the_last_written_carry_is_the_one_used(built, monkeypatch):
    """A kill between writing a new carry file and the marker: the marker still names the previous carry."""
    sess, ref = built
    ws = sess.ws
    real = W._replace
    state = {"carries": 0}

    def op(src, dst):
        if os.path.basename(dst).startswith("carry_"):
            state["carries"] += 1
            if state["carries"] == 2:
                real(src, dst)
                raise Stop("killed after the second carry, before its marker")
        return real(src, dst)
    monkeypatch.setattr(W, "_replace", op)
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=quiet)
    monkeypatch.setattr(W, "_replace", real)
    pdir = W.partial_dir(adir(ws))
    m = json.load(open(W.marker_path(adir(ws))))
    assert m["done"] == 1 and m["carry"] == "carry_001.npz" and os.path.exists(os.path.join(pdir, "carry_002.npz"))
    lines = []
    atlas.build(ws, stages=["wiring"], log=lines.append)
    assert "  continuing from layer 1 of 4 (1 layers already done)" in lines
    same(ref, files(ws))


# ----------------------------------------------------------------------------------------- stale progress
def test_force_starts_over(built):
    sess, ref = built
    ws = sess.ws
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=stop_after(2))
    lines = []
    atlas.build(ws, stages=["weights", "wiring"], force=True, log=lines.append)
    assert "  saved wiring progress discarded: a rebuild was requested; starting over" in lines
    assert not any("continuing" in x for x in lines) and sum("partners of layer" in x for x in lines) == 3
    same(ref, files(ws))


def test_progress_of_another_variant_is_discarded(built):
    sess, ref = built
    ws = sess.ws
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=stop_after(1))
    sess.apply("neuron(2, 3).w_out *= 0")
    lines = []
    atlas.build(ws, stages=["weights", "wiring"], variant="current", log=lines.append)
    assert "  saved wiring progress discarded: it was saved for the base weights and this build maps the current weights; starting over" in lines
    same(old_wiring(ws, "current"), files(ws))
    assert atlas.Atlas(ws).status()["wiring_built"]


def test_edited_weights_continue_with_the_same_patches_and_restart_with_new_ones(rnd):
    ws = rnd.ws
    rnd.apply("neuron(1, 2).w_out *= 0")
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], variant="current", log=stop_after(1))
    lines = []                                             # same patches: the weights stage is rebuilt, wiring continues
    atlas.build(ws, stages=["weights", "wiring"], variant="current", log=lines.append)
    assert "  continuing from layer 1 of 4 (1 layers already done)" in lines
    same(old_wiring(ws, "current"), files(ws))
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], variant="current", force=True, log=stop_after(1))
    rnd.apply("neuron(2, 5).w_out *= 0")
    lines = []
    atlas.build(ws, stages=["weights", "wiring"], variant="current", log=lines.append)
    assert "  saved wiring progress discarded: the patches have changed since; starting over" in lines
    same(old_wiring(ws, "current"), files(ws))


def test_progress_under_a_rebuilt_weights_stage_is_discarded(built):
    """The weights stage now describes other weights than the ones the progress was saved under."""
    sess, ref = built
    ws = sess.ws
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=stop_after(1))
    mp = os.path.join(adir(ws), "meta.json")
    meta = json.load(open(mp))
    meta["stages"]["weights"]["source"] = {"model": ws.name, "variant": "current", "patches": [["p1", 1.0]], "version": 7}
    json.dump(meta, open(mp, "w"))
    lines = []
    atlas.build(ws, stages=["wiring"], log=lines.append)
    assert "  saved wiring progress discarded: the weights stage has been rebuilt from other weights since; starting over" in lines
    same(ref, files(ws))


def test_progress_in_another_number_type_is_discarded(built, monkeypatch):
    sess, ref = built
    ws = sess.ws
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=stop_after(1))
    monkeypatch.setattr(W, "work_dtype", lambda: torch.bfloat16)
    lines = []
    atlas.build(ws, stages=["wiring"], log=lines.append)
    assert "  saved wiring progress discarded: it was computed in float32 and this build works in bfloat16; starting over" in lines
    assert atlas.Atlas(ws).status()["wiring_built"]
    assert json.load(open(os.path.join(adir(ws), "meta.json")))["stages"]["wiring"]["dtype"] == "bfloat16"


def test_unreadable_progress_starts_over(built):
    sess, ref = built
    ws = sess.ws
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=stop_after(2))
    st = atlas.Atlas(ws).status()["wiring_progress"]
    assert st["next_layer"] == 2 and st["variant"] == "base"
    pdir = W.partial_dir(adir(ws))
    os.unlink(os.path.join(pdir, "wiring_001.npz"))
    lines = []
    atlas.build(ws, stages=["wiring"], log=lines.append)
    assert "  saved wiring progress discarded: some finished layers are missing or damaged; starting over" in lines
    same(ref, files(ws))
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=stop_after(2))
    open(W.marker_path(adir(ws)), "w").write("{not json")
    no_wiring_visible(ws)
    wp = atlas.Atlas(ws).status()["wiring_progress"]
    assert wp["usable"] is False and wp["reason"] == "the saved progress is unreadable"
    lines = []
    atlas.build(ws, stages=["wiring"], log=lines.append)
    assert "  saved wiring progress discarded: the saved progress is unreadable; starting over" in lines
    same(ref, files(ws))


# ----------------------------------------------------------------------------------------- what counts as built
def test_an_atlas_in_the_old_on_disk_form_still_counts_as_built(rnd):
    ws = rnd.ws
    atlas.build(ws, stages=["weights"], log=quiet)
    a = adir(ws)
    old = old_wiring(ws)
    for l, r in old.items():
        np.savez(os.path.join(a, f"wiring_{l:03d}.npz"), **r)          # exactly as the old code wrote them
    mp = os.path.join(a, "meta.json")
    meta = json.load(open(mp))
    meta["stages"]["weights"].pop("source")                             # old meta: no source, no wiring dtype
    meta["stages"]["wiring"] = {"seconds": 0.1, "at": time.time()}
    json.dump(meta, open(mp, "w"), indent=1)
    A = atlas.Atlas(ws)
    assert A.has("wiring") and A.status()["wiring_built"] and "wiring" in A.status()["stages"]
    assert "w_down_gate_cos" in A.layer(0) and A.unit(0, 1)["wiring"] is not None
    stamp = {l: os.stat(W.final_path(a, l)).st_mtime_ns for l in old}
    lines = []
    atlas.build(ws, stages=["wiring"], log=lines.append)                # already built: nothing to do
    assert not any("neuron partners" in x for x in lines)
    assert stamp == {l: os.stat(W.final_path(a, l)).st_mtime_ns for l in old}
    same(old, files(ws))


def test_missing_files_or_a_missing_stage_record_are_not_built(built):
    sess, ref = built
    ws = sess.ws
    a = adir(ws)
    os.unlink(W.final_path(a, 2))
    assert not atlas.Atlas(ws).has("wiring")
    no_wiring_visible(ws)
    atlas.build(ws, stages=["wiring"], log=quiet)                       # rebuilt, not skipped
    same(ref, files(ws))
    atlas.build(ws, stages=["weights"], force=True, log=quiet)          # a forced rebuild drops the stage record
    assert not atlas.Atlas(ws).has("wiring")


def test_old_wiring_files_do_not_survive_a_variant_rebuild(built):
    """Mapping the edited weights resets the stage record, so the base wiring left on disk is not read as theirs."""
    sess, ref = built
    ws = sess.ws
    sess.apply("neuron(0, 1).w_out *= 0")
    atlas.build(ws, stages=["weights"], variant="current", log=quiet)
    assert os.path.exists(W.final_path(adir(ws), 0))
    no_wiring_visible(ws)


# ----------------------------------------------------------------------------------------- a gated layout
def gated_fake(n_layers=4, n=12, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    vec = {l: {"G": torch.randn(n + l, d, generator=g), "U": torch.randn(n + l, d, generator=g),
               "D": torch.randn(n + l, d, generator=g), "gated": True} for l in range(n_layers)}
    layer = SimpleNamespace(mlp={"W_gate": 1, "W_up": 1, "W_out": 1})
    struct = SimpleNamespace(n_layers=n_layers, d_model=d, layers=[layer] * n_layers,
                             has_neurons=lambda l: True, mlp_width=lambda l: n + l)
    ws = SimpleNamespace(struct=struct, name="fake", patches=[], version=0, can_run=False)
    return ws, vec


def test_gated_layers_continue_bit_for_bit(tmp_path, monkeypatch):
    ws, vec = gated_fake()
    reads = {"n": 0}

    def layer_vectors(_, l, variant="base"):
        reads["n"] += 1
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in vec[l].items()}
    monkeypatch.setattr(W, "layer_vectors", layer_vectors)
    monkeypatch.setattr(W, "layer_write", lambda _, l, variant="base": vec[l]["D"].clone())

    def run(d, log=quiet, restart=False):
        os.makedirs(d, exist_ok=True)
        W.build_wiring(ws, str(d), log, weights_source={"model": "fake", "variant": "base"}, restart=restart)
        W.cleanup(str(d))
        out = {}
        for l in range(4):
            with np.load(W.final_path(str(d), l)) as z:
                out[l] = {k: z[k] for k in z.files}
        return out
    ref = run(tmp_path / "ref")
    assert reads["n"] == 3                                              # gate and value read once per reader layer (not layer 0)
    assert np.abs(ref[0]["w_down_up_cos"].astype(np.float32)).max() > 0  # the value read is searched too
    for k in (1, 2, 3):
        d = tmp_path / f"k{k}"
        os.makedirs(d)
        with pytest.raises(Stop):
            W.build_wiring(ws, str(d), stop_after(k), weights_source={"model": "fake", "variant": "base"})
        same(ref, run(d))


# ----------------------------------------------------------------------------------------- the estimate
def fake_ws(widths=(100, 100, 100), d=50, gated=True, can_run=True):
    mlp = {"W_gate": 1, "W_up": 1, "W_out": 1} if gated else {"W_in": 1, "W_out": 1}
    struct = SimpleNamespace(n_layers=len(widths), d_model=d, layers=[SimpleNamespace(mlp=mlp)] * len(widths),
                             has_neurons=lambda l: True, mlp_width=lambda l: widths[l])
    return SimpleNamespace(struct=struct, name="fake", patches=[], version=0, can_run=can_run)


def test_estimate_arithmetic_with_an_injected_timing(tmp_path):
    ws = fake_ws()
    seen = []

    def timing(n_r, n_w, d, dtype, threads):
        seen.append((n_r, n_w, d))
        return 0.2, 0.05                                                # product, selection at the probe shape
    e = W.estimate(ws, {}, str(tmp_path), timing=timing)
    assert seen == [(100, 100, 50)]                                     # the probe is the real shape when small
    # 3 layers: pairs (0,1) (0,2) (1,2), two reads each (gated): 6 steps of exactly the probe's size
    assert e["search_s"] == pytest.approx(6 * 0.25)
    assert e["read_known"] is False and e["read_s"] is None and e["total_s"] is None
    assert "has not been built yet" in e["read_note"]
    # memory: read vectors of layers 1 and 2 (2 x 100 x 50 x 4 bytes each), the tables, and one step's product
    hold = 2 * (100 * 50 * 4 * 2)
    assert e["read_vectors_bytes"] == hold
    tables = 3 * 100 * 2 * 12 * 2 * 2
    step = 100 * 50 * 4 * 3 + 100 * 100 * 8                           # write vectors in three forms, product, |product|
    loading = 100 * 50 * 4 * 5
    assert e["memory_bytes"] == hold + tables + max(step, loading)
    assert e["waits"] == ["Run", "Chat", "edits"] and e["continuing"] is None
    assert e["flops"] == W.wiring_cost(ws) == 6 * 2 * 100 * 100 * 50


def test_estimate_scales_a_large_shape_from_the_probe(tmp_path):
    ws = fake_ws(widths=(17408,) * 3, d=5120)
    e = W.estimate(ws, {}, str(tmp_path), timing=lambda *a: (1.0, 0.5))
    f = (17408 / 4096) ** 2                                             # products and cosines both grow with n^2 here
    assert e["probe"] == [4096, 4096, 5120]
    assert e["search_s"] == pytest.approx(6 * (1.0 * f + 0.5 * f))


def test_estimate_with_a_recorded_read_time_and_for_a_model_that_cannot_run(tmp_path):
    ws = fake_ws(can_run=False)
    meta = {"stages": {"weights": {"layers": [0, 1, 2], "read_s": {"mlp": 2.0}}}}
    e = W.estimate(ws, meta, str(tmp_path), timing=lambda *a: (0.2, 0.05))
    # every reader layer's MLP once (layers 1, 2) and the write third of each writer's (layers 0, 1)
    assert e["read_s"] == pytest.approx(2.0 * 2 + 2.0 / 3 * 2)
    assert e["total_s"] == pytest.approx(e["search_s"] + e["read_s"]) and e["read_note"] is None
    assert e["waits"] == ["edits"]
    old = {"stages": {"weights": {"layers": [0, 1, 2]}}}               # a weights stage built before the record
    e = W.estimate(ws, old, str(tmp_path), timing=lambda *a: (0.2, 0.05))
    assert e["read_s"] is None and "built before reading times were recorded" in e["read_note"]
    plain = W.estimate(fake_ws(gated=False), meta, str(tmp_path), timing=lambda *a: (0.2, 0.05))
    assert plain["read_s"] == pytest.approx(2.0 * 2 + 2.0 / 2 * 2) and plain["search_s"] == pytest.approx(3 * 0.25)


def test_estimate_of_a_build_that_would_continue(built):
    sess, ref = built
    ws = sess.ws
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=stop_after(2))
    full = atlas.wiring_estimate(ws, timing=lambda *a: (0.2, 0.05), restart=True)
    rest = atlas.wiring_estimate(ws, timing=lambda *a: (0.2, 0.05))
    assert full["continuing"] is None and full["search_s"] == pytest.approx(6 * 0.25)
    assert rest["continuing"] == {"done": 2, "total": 4, "next_layer": 2}
    assert rest["search_s"] == pytest.approx(1 * 0.25)                 # one pair left: (2, 3)
    assert rest["read_known"] and rest["read_s"] < full["read_s"]
    other = atlas.wiring_estimate(ws, "current", timing=lambda *a: (0.2, 0.05))
    assert other["continuing"] is None                                  # that progress is not for these weights


def test_step_timing_runs_once_per_shape(monkeypatch):
    calls = []
    real = W._product
    monkeypatch.setattr(W, "_product", lambda R, DT: calls.append(R.shape) or real(R, DT))
    monkeypatch.setattr(W, "_TIMINGS", {})
    a = W.step_timing(32, 16, 8)
    b = W.step_timing(32, 16, 8)
    assert a == b and len(calls) == 2 and calls[0] == (32, 8)          # best of two, then cached
    tp, ts = a
    assert tp >= 0 and ts >= 0


def test_duration_and_summary():
    assert W.duration(None) == "an unknown time" and W.duration(0.3) == "under a second"
    assert W.duration(40) == "about 40 s" and W.duration(12 * 60) == "about 12 min"
    assert W.duration(6600) == "about 1 h 50 min" and W.duration(7200) == "about 2 h"
    assert W.duration(89) == "about 89 s" and W.duration(90) == "about 2 min" and W.duration(149) == "about 2 min"
    assert W.duration(151) == "about 3 min" and W.duration(3569) == "about 59 min" and W.duration(3599) == "about 1 h"
    assert W.duration(6449) == "about 1 h 45 min" and W.duration(6451) == "about 1 h 50 min"   # to 5 min above an hour
    assert W.duration(4950) == "about 1 h 25 min" and W.duration(4049) == "about 1 h 5 min"     # halves round up
    e = {"read_known": False, "search_s": 6600, "total_s": None, "memory_bytes": 48.5e9, "waits": ["Run", "Chat", "edits"]}
    assert W.summary(e) == ("estimate: about 1 h 50 min on this machine, plus reading the weights (not yet known); "
                            "holds about 48.5 GB while it runs; Run, Chat and edits wait until it finishes")
    e.update(read_known=True, total_s=7200, waits=["edits"])
    assert W.summary(e).startswith("estimate: about 2 h on this machine, including reading the weights")
    assert W.summary(e).endswith("edits wait until it finishes")


def test_the_build_log_carries_the_estimate_and_the_weights_stage_records_read_times(rnd, monkeypatch):
    monkeypatch.setattr(W, "step_timing", lambda *a: (0.2, 0.05))
    lines = []
    meta = atlas.build(rnd.ws, stages=["weights", "wiring"], log=lines.append)
    rec = meta["stages"]["weights"]["read_s"]
    assert rec["mlp"] > 0 and rec["layers"] == 4
    line = next(x for x in lines if "neuron partners" in x)
    assert line.startswith("  neuron partners (about ") and "multiply-adds; estimate: " in line
    assert "including reading the weights" in line and "Run, Chat and edits wait until it finishes" in line
    assert meta["stages"]["weights"]["source"] == {"model": "rnd4", "variant": "base"}


# ----------------------------------------------------------------------------------------- what waits
def test_run_and_edits_wait_for_the_workspace_lock_the_build_holds(rnd):
    """The estimate says Run, Chat and edits wait: they take the lock the build holds (here, a runnable model)."""
    ws = rnd.ws
    assert ws.can_run and W.estimate(ws, {}, adir(ws), timing=lambda *a: (0.0, 0.0))["waits"] == ["Run", "Chat", "edits"]
    done = {}
    ws.lock.acquire()
    try:
        t1 = threading.Thread(target=lambda: done.setdefault("runner", ws.runner))
        t2 = threading.Thread(target=lambda: done.setdefault("edit", rnd.apply("neuron(0, 1).w_out *= 0")))
        t1.start(), t2.start()
        time.sleep(0.3)
        assert done == {}
    finally:
        ws.lock.release()
    t1.join(10), t2.join(10)
    assert set(done) == {"runner", "edit"}


# ----------------------------------------------------------------------------------------- the server
def test_job_state_and_events_carry_layers_done_and_time_left(rnd, monkeypatch):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    monkeypatch.setattr(W, "step_timing", lambda *a: (0.2, 0.05))
    events = []
    real = rnd.publish
    rnd.publish = lambda ev: (events.append(ev), real(ev))
    c = TestClient(create_app(rnd, "http://test"))
    e = c.get("/api/atlas/estimate").json()
    assert e["search_s"] == pytest.approx(6 * 0.25) and e["waits"] == ["Run", "Chat", "edits"] and not e["read_known"]
    assert c.get("/api/atlas/estimate", params={"variant": "other"}).status_code == 422
    cur = c.get("/api/status").json()
    scope = {k: cur[k] for k in ("workspace_id", "version")}
    assert c.post("/api/atlas/build", json={**scope, "stages": ["weights", "wiring"]}).status_code == 200
    for _ in range(400):
        st = c.get("/api/atlas").json()
        if st["job"] and not st["job"]["running"]:
            break
        time.sleep(0.05)
    assert st["job"]["error"] is None and st["wiring_built"]
    assert st["job"]["progress"] == {"stage": "wiring", "total": 4, "phase": "finishing", "done": 4, "remaining_s": 0.0,
                                     "resumed_from": None}
    prog = [ev["progress"] for ev in events if ev.get("type") == "atlas" and "progress" in ev]
    searching = [p for p in prog if p["phase"] == "searching"]
    assert [p["done"] for p in searching] == [1, 2, 4] and all(p["total"] == 4 for p in searching)
    assert all(isinstance(p["remaining_s"], float) and p["remaining_s"] >= 0 for p in searching)
    assert [p["read"] for p in prog if p["phase"] == "reading"] == [1, 2, 3]
    assert any("partners of layer 0 done (1 of 4 layers, " in ev.get("line", "") and " left)" in ev["line"] for ev in events)
    assert c.get("/api/atlas/estimate").json()["read_known"]            # recorded by the weights stage just built


# ========================================================================================= review fixes
# ----------------------------------------------------------------------------------------- 1: identity across restarts
def test_edited_weights_progress_survives_a_restart_with_the_same_patches(rnd):
    """ws.version counts changes in one process; toggling a patch off and on changes it, a reload resets it.
    Neither changes the weights, so neither may throw the saved progress away."""
    ws = rnd.ws
    rnd.apply("neuron(1, 2).w_out *= 0")
    pid = ws.patches[0].id
    ws.set_patch(pid, enabled=False)
    ws.set_patch(pid, enabled=True)
    assert ws.version > 1
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], variant="current", log=stop_after(1))
    ws2 = Workspace.load(ws.dir)                                         # the IDE restarted: version 1 again
    assert ws2.version == 1
    wp = atlas.Atlas(ws2).status()["wiring_progress"]
    assert wp["usable"] and wp["reason"] is None and wp["next_layer"] == 1 and wp["rebuilds_weights"]
    assert atlas.wiring_estimate(ws2, "current", timing=lambda *a: (0.1, 0.1))["continuing"]["done"] == 1
    lines = []
    atlas.build(ws2, stages=["weights", "wiring"], variant="current", log=lines.append)
    assert "  continuing from layer 1 of 4 (1 layers already done)" in lines
    assert not any("discarded" in x for x in lines)
    same(old_wiring(ws2, "current"), files(ws2))
    ws3 = Workspace.load(ws.dir)
    assert atlas.Atlas(ws3).stale_layers() == []                         # nor is the edited-weights atlas stale
    ws3.set_patch(pid, strength=0.5)
    assert atlas.Atlas(ws3).stale_layers() == atlas.Atlas(ws3).layers


def test_a_patch_given_the_same_id_again_is_another_patch(rnd):
    """Patch ids are the count and the time modulo 100,000 s, so a deleted patch's id can come back; the commit time
    tells the two apart."""
    ws = rnd.ws
    rnd.apply("neuron(1, 2).w_out *= 0")
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], variant="current", log=stop_after(1))
    pj = os.path.join(ws.dir, "patches", ws.patches[0].id + ".json")
    m = json.load(open(pj))
    m["created"] += 100000.0                                             # the same id, committed at another time
    json.dump(m, open(pj, "w"))
    ws2 = Workspace.load(ws.dir)
    wp = atlas.Atlas(ws2).status()["wiring_progress"]
    assert not wp["usable"] and wp["reason"] == "the patches have changed since"
    lines = []
    atlas.build(ws2, stages=["weights", "wiring"], variant="current", log=lines.append)
    assert "  saved wiring progress discarded: the patches have changed since; starting over" in lines


def test_the_identity_has_no_per_process_counter_and_names_torch(rnd):
    rnd.apply("neuron(1, 2).w_out *= 0")
    want = W.identity(rnd.ws, "current", atlas.weights_source(rnd.ws, "current"))
    assert "version" not in want and "version" not in want["weights"] and want["torch"] == torch.__version__
    saved = {**want, "torch": "0.0.1"}
    assert W.mismatch(saved, want) == f"it was computed with torch 0.0.1 and this build uses torch {torch.__version__}"


# ----------------------------------------------------------------------------------------- 2: one decision
def _three_answers(ws, variant="base"):
    """What the Atlas tab, the estimate and the build each say about saved progress."""
    wp = atlas.Atlas(ws).status()["wiring_progress"]
    est = atlas.wiring_estimate(ws, variant, timing=lambda *a: (0.1, 0.1))
    lines = []
    atlas.build(ws, stages=["wiring"] if variant == "base" else ["weights", "wiring"], variant=variant, log=lines.append)
    said = [x for x in lines if "saved wiring progress" in x or "continuing" in x]
    return wp, est, said


def test_card_estimate_and_build_agree_on_usable_progress(built):
    sess, ref = built
    with pytest.raises(Stop):
        atlas.build(sess.ws, stages=["weights", "wiring"], force=True, log=stop_after(2))
    wp, est, said = _three_answers(sess.ws)
    assert wp["usable"] and est["saved"]["usable"] and est["continuing"] == {"done": 2, "total": 4, "next_layer": 2}
    assert said == ["  continuing from layer 2 of 4 (2 layers already done)"]
    same(ref, files(sess.ws))


@pytest.mark.parametrize("why", ["dtype", "patches", "carry", "model"])
def test_card_estimate_and_build_agree_on_unusable_progress(built, monkeypatch, why):
    sess, ref = built
    ws = sess.ws
    variant = "current" if why == "patches" else "base"
    if why == "patches":
        sess.apply("neuron(1, 2).w_out *= 0")
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], variant=variant, force=True, log=stop_after(2))
    m = json.load(open(W.marker_path(adir(ws))))
    if why == "dtype":
        monkeypatch.setattr(W, "work_dtype", lambda: torch.bfloat16)
    elif why == "patches":
        sess.apply("neuron(2, 2).w_out *= 0")
    elif why == "carry":
        p = os.path.join(W.partial_dir(adir(ws)), m["carry"])
        b = open(p, "rb").read()
        open(p, "wb").write(b[: len(b) // 2])
    elif why == "model":
        m["identity"]["model"] = "another"
        json.dump(m, open(W.marker_path(adir(ws)), "w"))
    wp, est, said = _three_answers(ws, variant)
    reason = wp["reason"]
    assert not wp["usable"] and reason and est["saved"] == {**{k: wp[k] for k in est["saved"]}}
    assert est["continuing"] is None and est["saved"]["reason"] == reason
    assert said == [f"  saved wiring progress discarded: {reason}; starting over"]
    assert atlas.Atlas(ws).status()["wiring_built"]


# ----------------------------------------------------------------------------------------- 3: one build at a time
def _slow_build(ws, monkeypatch, **kw):
    """A wiring build in a thread that waits at its first comparison until released."""
    inside, release, out = threading.Event(), threading.Event(), {}
    real = W._product

    def product(R, DT):
        inside.set()
        assert release.wait(30)
        return real(R, DT)
    monkeypatch.setattr(W, "_product", product)

    def run():
        try:
            atlas.build(ws, log=quiet, **kw)
            out["ok"] = True
        except BaseException as e:                                       # noqa: BLE001
            out["error"] = e
    t = threading.Thread(target=run)
    t.start()
    assert inside.wait(30)
    return t, release, out


def test_a_second_build_in_this_process_is_refused_at_once_and_the_first_is_right(built, monkeypatch):
    sess, ref = built
    ws = sess.ws
    t, release, out = _slow_build(ws, monkeypatch, stages=["weights", "wiring"], force=True)
    try:
        t0 = time.monotonic()
        with pytest.raises(atlas.Busy, match="a build of this atlas is already running in this IDE"):
            atlas.build(ws, stages=["wiring"], log=quiet)                  # the MCP-style base wiring build
        with pytest.raises(atlas.Busy):
            atlas.build(ws, stages=["weights", "wiring"], variant="current", log=quiet)
        assert time.monotonic() - t0 < 2
        from wsp import tools
        msg = tools.atlas_tool(sess, action="build", stages=["wiring"])
        assert msg.startswith("error: a build of this atlas is already running")
    finally:
        release.set()
        t.join(60)
    assert out == {"ok": True}
    same(ref, files(ws))
    assert atlas.Atlas(ws).has("wiring") and not os.path.exists(W.partial_dir(adir(ws)))
    assert W.LOCK not in [os.path.basename(W.final_path(adir(ws), l)) for l in range(4)]


def test_the_browser_job_is_refused_at_once_while_another_build_runs(built, monkeypatch):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    sess, ref = built
    t, release, out = _slow_build(sess.ws, monkeypatch, stages=["weights", "wiring"], force=True)
    try:
        c = TestClient(create_app(sess, "http://test"))
        cur = c.get("/api/status").json()
        r = c.post("/api/atlas/build", json={"workspace_id": cur["workspace_id"], "version": cur["version"], "stages": ["weights"]})
        assert r.status_code == 200
        for _ in range(200):
            st = c.get("/api/atlas").json()
            if st["job"] and not st["job"]["running"]:
                break
            time.sleep(0.02)
        assert st["job"]["error"].startswith("A build of this atlas is already running in this IDE")
    finally:
        release.set()
        t.join(60)
    assert out == {"ok": True}


SUBPROCESS = """
import os, sys, time
sys.path.insert(0, {root!r})
from wsp import atlas
from wsp.atlas import wiring as W
from wsp.workspace import Workspace
if sys.argv[1] == "hold":
    with W.build_lock({adir!r}):
        print("held", flush=True)
        time.sleep(120)
else:
    try:
        atlas.build(Workspace.load({wsdir!r}), stages=["wiring"], log=lambda *a: None)
        print("built", flush=True)
    except atlas.Busy as e:
        print("refused:", e, flush=True)
"""


def test_builds_in_two_processes_and_a_killed_holder(built):
    import signal
    import subprocess
    sess, ref = built
    ws = sess.ws
    code = SUBPROCESS.format(root=ROOT, adir=adir(ws), wsdir=ws.dir)
    # a build here while another process holds the lock: refused at once
    holder = subprocess.Popen([sys.executable, "-c", code, "hold"], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        t0 = time.monotonic()
        with pytest.raises(atlas.Busy, match="already running in another process"):
            atlas.build(ws, stages=["weights", "wiring"], force=True, log=quiet)
        assert time.monotonic() - t0 < 2
    finally:
        holder.send_signal(signal.SIGKILL)                               # dies holding it
        holder.wait(30)
    lines = []
    atlas.build(ws, stages=["weights", "wiring"], force=True, log=lines.append)   # the system released the lock
    same(ref, files(ws))
    # a build in another process while this one holds the lock: refused there
    with W.build_lock(adir(ws)):
        r = subprocess.run([sys.executable, "-c", code, "build"], capture_output=True, text=True, timeout=120)
    assert r.stdout.startswith("refused: a build of this atlas is already running in another process"), r.stdout + r.stderr
    assert os.path.exists(os.path.join(adir(ws), W.LOCK)) and atlas.Atlas(ws).has("wiring")


def test_the_lock_is_reentrant_for_one_thread_only(tmp_path):
    d = str(tmp_path / "atlas")
    with W.build_lock(d):
        with W.build_lock(d):                                            # the server/MCP path, then atlas.build
            pass
        got = {}
        th = threading.Thread(target=lambda: got.setdefault("e", pytest.raises(W.Busy, W.build_lock(d).__enter__)))
        th.start()
        th.join(10)
        assert isinstance(got["e"].value, W.Busy)
    with W.build_lock(d):                                                # released after the outer one
        pass


def test_the_mcp_build_holds_the_workspace_lock_and_returns_the_estimate(rnd, monkeypatch):
    from wsp import tools
    ws = rnd.ws
    seen = {}

    def build(w, stages=None, log=print, **kw):
        seen["locked"] = w.lock._is_owned()
        other = {}
        th = threading.Thread(target=lambda: other.setdefault("got", w.lock.acquire(blocking=False)))
        th.start()
        th.join(10)
        seen["other_thread_got_it"] = other["got"]
        log("atlas of x")
        log("  neuron partners (about 1.0e+15 multiply-adds; estimate: about 2 h on this machine, ...)")
        for i in range(30):
            log(f"  partners of layer {i} done")
    monkeypatch.setattr(atlas, "build", build)
    out = tools.atlas_tool(rnd, action="build", stages=["wiring"])
    assert seen == {"locked": True, "other_thread_got_it": False}
    assert "neuron partners (about 1.0e+15 multiply-adds; estimate: about 2 h" in out
    assert "partners of layer 29 done" in out and "partners of layer 10 done" not in out


# ----------------------------------------------------------------------------------------- 4: damaged progress files
def _damage(ws, how):
    a = adir(ws)
    pdir, mp = W.partial_dir(a), W.marker_path(a)
    m = json.load(open(mp))
    if how == "truncated carry":
        p = os.path.join(pdir, m["carry"])
        b = open(p, "rb").read()
        open(p, "wb").write(b[: len(b) // 2])
    elif how == "garbage carry":
        open(os.path.join(pdir, m["carry"]), "wb").write(b"PK\x03\x04" + os.urandom(500))
    elif how == "carry of another shape":
        np.savez(os.path.join(pdir, m["carry"]), done=np.array(m["done"]), junk=np.zeros(3))
    elif how == "marker is a list":
        open(mp, "w").write("[1, 2]")
    elif how == "marker done is a string":
        json.dump({**m, "done": "x"}, open(mp, "w"))
    elif how == "marker identity is null":
        json.dump({**m, "identity": None}, open(mp, "w"))
    elif how == "marker names a path":
        json.dump({**m, "carry": "../meta.json"}, open(mp, "w"))
    elif how == "marker is a directory":
        os.unlink(mp)
        os.makedirs(mp)
    elif how == "truncated finished layer":
        p = os.path.join(pdir, "wiring_000.npz")
        b = open(p, "rb").read()
        open(p, "wb").write(b[: len(b) // 2])
    elif how == "finished layer of another shape":
        np.savez(os.path.join(pdir, "wiring_001.npz"), junk=np.zeros(3))
    else:
        raise AssertionError(how)


DAMAGE = ["truncated carry", "garbage carry", "carry of another shape", "marker is a list", "marker done is a string",
          "marker identity is null", "marker names a path", "marker is a directory", "truncated finished layer",
          "finished layer of another shape"]


@pytest.mark.parametrize("how", DAMAGE)
def test_damaged_progress_is_reported_and_started_over(built, how):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    sess, ref = built
    ws = sess.ws
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=stop_after(2))
    _damage(ws, how)
    wp = atlas.Atlas(ws).status()["wiring_progress"]                    # never raises
    assert wp["usable"] is False and wp["reason"]
    c = TestClient(create_app(sess, "http://test"))
    st = c.get("/api/atlas")
    assert st.status_code == 200 and st.json()["built"] and not st.json()["wiring_built"]
    e = c.get("/api/atlas/estimate")
    assert e.status_code == 200 and e.json()["continuing"] is None and e.json()["saved"]["reason"] == wp["reason"]
    lines = []
    atlas.build(ws, stages=["wiring"], log=lines.append)
    assert f"  saved wiring progress discarded: {wp['reason']}; starting over" in lines
    same(ref, files(ws))
    A = atlas.Atlas(Workspace.load(ws.dir))
    assert A.has("wiring") and all("w_down_gate_cos" in A.layer(l) for l in A.layers)


def _stop_while_finishing(ws, monkeypatch, after_moves=0):
    """Stop the stage after the marker says "finishing", when `after_moves` layer files have been moved into place."""
    real = W._replace
    moved = {"n": 0}

    def op(src, dst):
        if os.path.dirname(dst) == adir(ws) and os.path.basename(dst).startswith("wiring_"):
            if moved["n"] == after_moves:
                raise Stop("stopped while finishing")
            moved["n"] += 1
        return real(src, dst)
    monkeypatch.setattr(W, "_replace", op)
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=quiet)
    monkeypatch.setattr(W, "_replace", real)
    assert json.load(open(W.marker_path(adir(ws))))["state"] == "finishing"


@pytest.mark.parametrize("damage", ["missing", "truncated"])
def test_a_finishing_stage_with_a_bad_layer_starts_over_instead_of_failing_for_ever(built, monkeypatch, damage):
    sess, ref = built
    ws = sess.ws
    _stop_while_finishing(ws, monkeypatch, after_moves=1)               # layer 0 moved, 1-3 still in the progress dir
    p = os.path.join(W.partial_dir(adir(ws)), "wiring_002.npz")
    if damage == "missing":
        os.unlink(p)
    else:
        open(p, "r+b").truncate(100)
    wp = atlas.Atlas(ws).status()["wiring_progress"]
    assert not wp["usable"] and wp["reason"] == "some finished layers are missing or damaged"
    lines = []
    atlas.build(ws, stages=["wiring"], log=lines.append)
    assert "  saved wiring progress discarded: some finished layers are missing or damaged; starting over" in lines
    same(ref, files(ws))


def test_a_finishing_stage_continues_with_its_layers_partly_moved(built, monkeypatch):
    sess, ref = built
    _stop_while_finishing(sess.ws, monkeypatch, after_moves=2)
    lines = []
    atlas.build(sess.ws, stages=["wiring"], log=lines.append)
    assert "  continuing: all 4 layers were already done" in lines
    same(ref, files(sess.ws))


def test_a_layer_missing_after_the_moves_is_an_error_not_a_complete_stage(built, monkeypatch):
    sess, ref = built
    ws = sess.ws
    real = W._replace

    def op(src, dst):
        if dst == W.final_path(adir(ws), 2):
            return None                                                  # the move of layer 2 silently did nothing
        return real(src, dst)
    monkeypatch.setattr(W, "_replace", op)
    with pytest.raises(RuntimeError, match=r"wiring results of layer\(s\) \[2\] are missing"):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=quiet)
    monkeypatch.setattr(W, "_replace", real)
    assert not atlas.Atlas(ws).has("wiring")
    assert "wiring" not in json.load(open(os.path.join(adir(ws), "meta.json")))["stages"]
    atlas.build(ws, stages=["wiring"], log=quiet)                        # the file is still in the progress dir
    same(ref, files(ws))


# ----------------------------------------------------------------------------------------- time left
def test_time_left_is_the_pace_so_far_times_the_work_left(rnd, monkeypatch):
    ws = rnd.ws
    meta = atlas.build(ws, stages=["weights"], log=quiet)
    clock = {"t": 0.0}

    def perf_counter():
        clock["t"] += 5.0                                                # every pass takes 5 s
        return clock["t"]
    monkeypatch.setattr(W, "time", SimpleNamespace(perf_counter=perf_counter, time=time.time))
    seen = []
    W.build_wiring(ws, adir(ws), quiet, weights_source=atlas.recorded_source(ws, meta), progress=seen.append)
    ls = W.neuron_layers(ws)
    f = [W._pass_flops(ws.struct, ls, a) for a in range(len(ls) - 1)]
    got = [p["remaining_s"] for p in seen if p["phase"] == "searching"]
    want = [5.0 * (a + 1) / sum(f[:a + 1]) * sum(f[a + 1:]) for a in range(len(f))]
    assert got == pytest.approx(want) and got[-1] == 0.0 and got[0] > got[1] > 0


# ----------------------------------------------------------------------------------------- the weights-stage rebuild
def test_the_estimate_of_an_edited_weights_build_names_the_weights_rebuild(rnd):
    ws = rnd.ws
    rnd.apply("neuron(1, 2).w_out *= 0")
    assert atlas.wiring_estimate(ws, "current", timing=lambda *a: (0.1, 0.1))["weights_rebuild"] == {"seconds": None,
                                                                                                    "drops_observed": False}
    meta = atlas.build(ws, stages=["weights"], variant="current", log=quiet)
    assert meta["stages"]["weights"]["full"] is True
    e = atlas.wiring_estimate(ws, "current", timing=lambda *a: (0.1, 0.1))
    assert e["weights_rebuild"] == {"seconds": meta["stages"]["weights"]["seconds"], "drops_observed": False}
    assert atlas.wiring_estimate(ws, "base", timing=lambda *a: (0.1, 0.1))["weights_rebuild"] is None


# ========================================================================================= review round 2
def _slow_mcp(sess, monkeypatch, stages=("wiring",)):
    """An agent's MCP build (the build lock, then the workspace lock) in a thread, held at its first comparison.  The
    MCP tool has no force, so an unfinished wiring stage is made first when the stage is complete."""
    from wsp import tools
    if atlas.Atlas(sess.ws).has("wiring"):
        with pytest.raises(Stop):
            atlas.build(sess.ws, stages=["weights", "wiring"], force=True, log=stop_after(1))
    inside, release, out = threading.Event(), threading.Event(), {}
    real = W._product

    def product(R, DT):
        inside.set()
        assert release.wait(30)
        return real(R, DT)
    monkeypatch.setattr(W, "_product", product)
    t = threading.Thread(target=lambda: out.setdefault("text", tools.atlas_tool(sess, action="build", stages=list(stages))))
    t.start()
    assert inside.wait(30)
    return t, release, out


def _client(sess):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    return TestClient(create_app(sess, "http://test"))


def _wait_job(c, seconds=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        st = c.get("/api/atlas").json()
        if st["job"] and not st["job"]["running"]:
            return st
        time.sleep(0.02)
    return None


# ----------------------------------------------------------------------------------------- 1: a build started elsewhere
def test_saved_progress_of_a_running_build_is_not_offered_and_the_holder_is_named(built, monkeypatch):
    sess, ref = built
    ws = sess.ws
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=stop_after(1))
    assert atlas.build_running(ws) is None
    c = _client(sess)
    assert c.get("/api/atlas").json()["wiring_progress"]["usable"] is True
    t, release, out = _slow_mcp(sess, monkeypatch, stages=["wiring"])             # the agent continues it
    try:
        st = c.get("/api/atlas").json()
        assert st["job"] is None and st["build_running"] == "this IDE"
        wp = st["wiring_progress"]
        assert wp["usable"] is False and wp["reason"] == "a build of this atlas is running in this IDE"
        e = c.get("/api/atlas/estimate").json()
        assert e["running"] == "this IDE" and e["continuing"] is None
        assert e["saved"]["usable"] is False and e["saved"]["reason"] == wp["reason"]
        assert atlas.build_running(ws) == "this IDE"
    finally:
        release.set()
        t.join(60)
    assert "error" not in out["text"].split("\n\n")[0]                           # the probes did not disturb it
    same(ref, files(ws))
    st = c.get("/api/atlas").json()
    assert st["build_running"] is None and st["wiring_progress"] is None and st["wiring_built"]


HOLD = """
import sys, time
sys.path.insert(0, {root!r})
from wsp.atlas import wiring as W
with W.build_lock({adir!r}):
    print("held", flush=True)
    time.sleep(120)
"""


def test_a_build_in_another_process_is_named_and_not_disturbed_by_the_probe(built):
    import signal
    import subprocess
    sess, ref = built
    ws = sess.ws
    with pytest.raises(Stop):
        atlas.build(ws, stages=["weights", "wiring"], force=True, log=stop_after(1))
    holder = subprocess.Popen([sys.executable, "-c", HOLD.format(root=ROOT, adir=adir(ws))], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        c = _client(sess)
        for _ in range(20):                                                       # probing again and again
            st = c.get("/api/atlas").json()
            assert st["build_running"] == "another process"
            assert st["wiring_progress"]["reason"] == "a build of this atlas is running in another process"
        assert c.get("/api/atlas/estimate").json()["saved"]["usable"] is False
        with pytest.raises(atlas.Busy, match="in another process"):              # the holder still holds it
            atlas.build(ws, stages=["wiring"], log=quiet)
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait(30)
    assert atlas.build_running(ws) is None
    atlas.build(ws, stages=["wiring"], log=quiet)
    same(ref, files(ws))


PROBE = """
import fcntl, os, sys, time
fd = os.open({lock!r}, os.O_RDONLY)
fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)                 # what build_running does, held 30 ms instead of microseconds
print("probing", flush=True)
time.sleep(0.03)
os.close(fd)
time.sleep(5)
"""


def test_a_probe_in_another_process_never_makes_a_build_see_busy(tmp_path):
    import subprocess
    d = str(tmp_path / "atlas")
    with W.build_lock(d):
        pass                                                                      # the lock file exists from now on
    assert W.build_running(d) is None and os.listdir(d) == [W.LOCK]
    for _ in range(3):
        p = subprocess.Popen([sys.executable, "-c", PROBE.format(lock=os.path.join(d, W.LOCK))], stdout=subprocess.PIPE, text=True)
        try:
            assert p.stdout.readline().strip() == "probing"
            with W.build_lock(d):                                                # retried past the probe, not Busy
                assert W.build_running(d) == "this IDE"
        finally:
            p.kill()
            p.wait(30)
    assert W.build_running(str(tmp_path / "never-built")) is None
    assert not os.path.exists(tmp_path / "never-built")                         # the probe creates nothing


def test_the_probe_never_leaves_the_lock_held(tmp_path):
    d = str(tmp_path / "atlas")
    with W.build_lock(d):
        assert W.build_running(d) == "this IDE"                                   # the holder's own thread
        other = {}
        th = threading.Thread(target=lambda: other.setdefault("r", W.build_running(d)))
        th.start()
        th.join(10)
        assert other["r"] == "this IDE"
    for _ in range(5):
        assert W.build_running(d) is None
        with W.build_lock(d):
            pass


def test_the_lock_is_one_lock_through_a_symlinked_path(tmp_path):
    d = tmp_path / "atlas"
    d.mkdir()
    link = tmp_path / "link"
    os.symlink(d, link)
    held, release = threading.Event(), threading.Event()

    def hold():
        with W.build_lock(str(d)):
            held.set()
            release.wait(30)
    th = threading.Thread(target=hold)
    th.start()
    try:
        assert held.wait(10)
        with pytest.raises(W.Busy, match="in this IDE"):                          # not "another process"
            with W.build_lock(str(link)):
                pass
        assert W.build_running(str(link)) == "this IDE"
    finally:
        release.set()
        th.join(10)


def test_a_failed_job_error_and_a_running_build_reach_the_status(built, monkeypatch):
    sess, ref = built
    c = _client(sess)
    t, release, out = _slow_mcp(sess, monkeypatch)
    try:
        cur = c.get("/api/status").json()
        r = c.post("/api/atlas/build", json={"workspace_id": cur["workspace_id"], "version": cur["version"], "stages": ["wiring"]})
        assert r.status_code == 200
        st = _wait_job(c)                                                         # refused at once, not after the build
        assert st is not None, "the browser job waited for the MCP build"
        assert st["job"]["error"].startswith("A build of this atlas is already running in this IDE")
        assert st["build_running"] == "this IDE"
    finally:
        release.set()
        t.join(60)
    assert out["text"].startswith("atlas of") or "error" not in out["text"][:40]
    st = c.get("/api/atlas").json()
    assert st["build_running"] is None and st["job"]["error"]                     # still shown after the build ended


# ----------------------------------------------------------------------------------------- 3 and 4: refused at once
def test_the_mcp_build_is_refused_at_once_while_a_browser_build_runs(built, monkeypatch):
    from wsp import tools
    sess, ref = built
    c = _client(sess)
    inside, release = threading.Event(), threading.Event()
    real = W._product

    def product(R, DT):
        inside.set()
        assert release.wait(30)
        return real(R, DT)
    monkeypatch.setattr(W, "_product", product)
    cur = c.get("/api/status").json()
    assert c.post("/api/atlas/build", json={"workspace_id": cur["workspace_id"], "version": cur["version"],
                                            "stages": ["weights", "wiring"], "force": True}).status_code == 200
    try:
        assert inside.wait(30)
        got = {}
        th = threading.Thread(target=lambda: got.setdefault("text", tools.atlas_tool(sess, action="build", stages=["wiring"])))
        th.start()
        th.join(5)
        assert not th.is_alive(), "the MCP build waited for the browser build"
        assert got["text"].startswith("error: a build of this atlas is already running in this IDE")
    finally:
        release.set()
        for _ in range(500):
            if not c.get("/api/atlas").json()["job"]["running"]:
                break
            time.sleep(0.02)
        th.join(30)
    assert c.get("/api/atlas").json()["job"]["error"] is None
    same(ref, files(sess.ws))


def test_requests_that_would_hold_the_session_lock_through_a_build_are_refused(built, monkeypatch):
    sess, ref = built
    c = _client(sess)
    t, release, out = _slow_mcp(sess, monkeypatch)
    try:
        res = {}

        def call(tag, f):
            t0 = time.monotonic()
            r = f()
            res[tag] = (r.status_code, time.monotonic() - t0, r.json().get("detail", ""))
        calls = [threading.Thread(target=call, args=(k, f)) for k, f in [
            ("suite", lambda: c.post("/api/suite", json={"name": "x", "suite": {"cases": []}})),
            ("open", lambda: c.post("/api/open", json={"name": "rnd4"})),
            ("pull", lambda: c.post("/api/pull"))]]
        for th in calls:
            th.start()
        for th in calls:
            th.join(5)
        assert all(not th.is_alive() for th in calls), res
        for k, what in [("suite", "saving a test suite"), ("open", "opening another model"), ("pull", "downloading the weights")]:
            code, took, detail = res[k]
            assert code == 409 and took < 2
            assert detail == f"A build of this atlas is running in this IDE; {what} waits until it finishes, so try again then."
        task = {}
        th = threading.Thread(target=lambda: task.setdefault("r", c.post("/api/task", json={})))
        th.start()
        th.join(5)
        assert task["r"].status_code == 200                                       # the session lock is free
    finally:
        release.set()
        t.join(60)
    assert c.post("/api/suite", json={"name": "x", "suite": {"cases": []}}).status_code == 200


CLI = """
import sys, time
sys.path.insert(0, {root!r})
from wsp.atlas import wiring as W
with W.build_lock({adir!r}):
    print("held", flush=True)
    time.sleep(120)
"""


def test_the_cli_says_a_build_is_running_and_exits_non_zero(rnd):
    import subprocess
    ws = rnd.ws
    holder = subprocess.Popen([sys.executable, "-c", CLI.format(root=ROOT, adir=adir(ws))], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        r = subprocess.run([sys.executable, "-m", "wsp", "atlas", "--root", rnd.root, "--model", "rnd4", "--stages", "weights"],
                           cwd=ROOT, capture_output=True, text=True, timeout=120)
    finally:
        holder.kill()
        holder.wait(30)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "Traceback" not in r.stderr
    assert r.stderr.strip() == ("wsp atlas: a build of this atlas is already running in another process; wait for it to "
                                "finish, then build again (it continues where it stopped).")


# ----------------------------------------------------------------------------------------- stale_layers across a restart
def test_stale_layers_compares_the_patches_not_the_per_process_version(rnd):
    ws = rnd.ws
    rnd.apply("neuron(1, 2).w_out *= 0")
    pid = ws.patches[0].id
    ws.set_patch(pid, enabled=False)
    ws.set_patch(pid, enabled=True)
    assert ws.version > 1
    atlas.build(ws, stages=["weights"], variant="current", log=quiet)             # recorded at this version
    ws2 = Workspace.load(ws.dir)
    assert ws2.version == 1                                                       # another version, the same patches
    assert atlas.Atlas(ws2).stale_layers() == []
    pj = os.path.join(ws.dir, "patches", pid + ".json")
    m = json.load(open(pj))
    m["strength"] = 0.5                                                           # other weights, at the same version
    json.dump(m, open(pj, "w"))
    ws3 = Workspace.load(ws.dir)
    assert ws3.version == 1 and atlas.Atlas(ws3).stale_layers() == atlas.Atlas(ws3).layers


# ----------------------------------------------------------------------------------------- validation branches
def _stopped(built, k=2):
    sess, ref = built
    with pytest.raises(Stop):
        atlas.build(sess.ws, stages=["weights", "wiring"], force=True, log=stop_after(k))
    a = adir(sess.ws)
    return sess.ws, a, W.partial_dir(a), json.load(open(W.marker_path(a)))


def _assess(ws):
    return atlas.Atlas(ws).status()["wiring_progress"]


def _rewrite(path, change):
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    change(arrays)
    np.savez(path, **arrays)


@pytest.mark.parametrize("how", ["shape", "dtype", "extra key", "missing key", "done"])
def test_a_carry_of_another_form_is_not_used(built, how):
    ws, a, pdir, m = _stopped(built)
    k = next(x for x in np.load(os.path.join(pdir, m["carry"])).files if x.endswith("_cos"))

    def change(arr):
        if how == "shape":
            arr[k] = arr[k][:, :-1].copy()
        elif how == "dtype":
            arr[k] = arr[k].astype(np.float64)
        elif how == "extra key":
            arr["junk"] = np.zeros(1)
        elif how == "missing key":
            arr.pop(k)
        elif how == "done":
            arr["done"] = np.array(m["done"] - 1)                                  # a valid carry of another pass
    _rewrite(os.path.join(pdir, m["carry"]), change)
    wp = _assess(ws)
    assert wp["usable"] is False and wp["reason"] == "the saved search state is missing or damaged"


@pytest.mark.parametrize("how", ["shape", "dtype", "extra key"])
def test_a_finished_layer_of_another_form_is_not_used(built, how):
    ws, a, pdir, m = _stopped(built)
    p = os.path.join(pdir, "wiring_000.npz")
    k = next(x for x in np.load(p).files if x.endswith("_cos"))

    def change(arr):
        if how == "shape":
            arr[k] = arr[k][:-1].copy()
        elif how == "dtype":
            arr[k] = arr[k].astype(np.float32)
        elif how == "extra key":
            arr["junk"] = np.zeros(1)
    _rewrite(p, change)
    wp = _assess(ws)
    assert wp["usable"] is False and wp["reason"] == "some finished layers are missing or damaged"


@pytest.mark.parametrize("how", ["done too large", "done negative", "carry in a subdirectory", "carry outside"])
def test_a_marker_out_of_bounds_is_unreadable(built, how):
    ws, a, pdir, m = _stopped(built)
    if how == "done too large":
        m["done"] = len(m["layers"])
    elif how == "done negative":
        m["done"] = -1
    elif how == "carry in a subdirectory":                                        # a real carry, but not where it belongs
        os.makedirs(os.path.join(pdir, "sub"))
        shutil.copy(os.path.join(pdir, m["carry"]), os.path.join(pdir, "sub", m["carry"]))
        m["carry"] = "sub/" + m["carry"]
    elif how == "carry outside":
        m["carry"] = "../" + W.PARTIAL + "/" + m["carry"]                         # the same file, through a path
    json.dump(m, open(W.marker_path(a), "w"))
    wp = _assess(ws)
    assert wp["usable"] is False and wp["reason"] == W.UNREADABLE


def test_after_the_last_pass_the_last_layer_must_be_there_too(built):
    ws, a, pdir, m = _stopped(built, k=3)                                         # every pass done, not yet finishing
    assert m["state"] == "searching" and m["done"] == len(m["layers"]) - 1
    assert _assess(ws)["usable"] is True
    os.unlink(os.path.join(pdir, os.path.basename(W.final_path(a, m["layers"][-1]))))
    wp = _assess(ws)
    assert wp["usable"] is False and wp["reason"] == "some finished layers are missing or damaged"
    lines = []
    atlas.build(ws, stages=["wiring"], log=lines.append)
    assert "  saved wiring progress discarded: some finished layers are missing or damaged; starting over" in lines
    same(built[1], files(ws))
