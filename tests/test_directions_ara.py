"""Fitted low-rank ablation ("ARA", wsp.directions_ara) and the module_io capture it reads (wsp.directions).

Model-free: the tiny random model of tests/test_steering.py (Config + init_params, no training), the llama-named
fake checkpoint and fake decoder of tests/test_steering_worker.py (no transformers), its stub worker and
RemoteRunner, and stub runners. Every expectation is computed independently of the code under test: captures from
`wsp.model.forward` with its own hook names or from forward hooks on the fake decoder's modules, the layout written
out by hand from the model's sizes, the loss as a dense formula in the [d, D_in] layout (W_eff = W + B @ A, row
norms, torch.cdist, topk), the initial values as kaiming_uniform_ draws made here, and the fit as torch's L-BFGS run
here on that dense formulation.
"""
import json
import math
import os
import re

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import wsp.runner
from wsp import directions as DR
from wsp import directions_ara as ARA
from wsp import workspace as Wmod
from wsp.model import forward, init_params, Config
from wsp.runner import FunctionalRunner
from wsp.server import create_app
from wsp.session import Session
from wsp.wscript import FUNCTION_DOC, run_script

from test_steering import CFG, make_ws

GOOD = ["abc", "abd", "hello", "zzz", "quiet", "fox"]
BAD = ["xyz", "qqq", "world", "mmmm", "bad"]
N, D = CFG["n_layers"], CFG["d_model"]
H, DH, F = CFG["n_heads"], CFG["d_model"] // CFG["n_heads"], CFG["d_mlp"]
ROW = (H * DH + D) + (F + D)                       # attn: z (heads in order) then attn_out; mlp: post then mlp_out
SMALL = {"lora_rank": 3, "n_optimization_steps": 2, "max_iter": 8}


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


def io_dir(ws, name):
    return os.path.join(ws.dir, "directions", name + ".io")


def stored(ws, name, label, layer):
    return torch.from_numpy(np.load(os.path.join(io_dir(ws, name), f"{label}.L{layer}.npy")))


def ara(name="m", **kw):
    return {"method": "ara", "direction": name, **SMALL, **kw}


# ------------------------------------------------------------------ independent references
def own_capture(ws, text, variant="base", params=None):
    """Per layer: (z flattened, attn_out, post, mlp_out) at the last token, from wsp.model.forward."""
    p = params if params is not None else ws.runner.params(variant)
    names = [f"blocks.{l}.{h}" for l in range(N) for h in ("attn.hook_z", "hook_attn_out", "mlp.hook_post", "hook_mlp_out")]
    cache = {}
    forward(p, ws.cfg, torch.tensor(ws.encode(text)), cache=cache, names=names)
    return [tuple(cache[f"blocks.{l}.{h}"][0, -1].reshape(-1) for h in ("attn.hook_z", "hook_attn_out", "mlp.hook_post", "hook_mlp_out"))
            for l in range(N)]


def matrices(ws, l, variant="base"):
    """(V, bias) per component in the [D_in, d] layout, read whole from the workspace."""
    wo = ws.read(f"blocks.{l}.attn.W_O", (), variant)
    return {"attn.o_proj": (wo.reshape(-1, D), ws.read(f"blocks.{l}.attn.b_O", (), variant)),
            "mlp.down_proj": (ws.read(f"blocks.{l}.mlp.W_out", (), variant), ws.read(f"blocks.{l}.mlp.b_out", (), variant))}


def ref_loss(W, bias, A, B, Xg, Yg, Xb, Yb, wp, wsb, oc, k, preserve):
    """The loss written in the adapter layout: W [d, D_in], W_eff = W + B @ A, row norms of W_eff."""
    W_eff = W + B @ A
    if preserve:
        n0, n1 = W.norm(dim=1), W_eff.norm(dim=1)
        s = torch.where(n0 > 0, n0 / torch.where(n0 > 0, n1, torch.ones_like(n1)), torch.ones_like(n0))
    else:
        s = torch.ones(W.shape[0], dtype=W.dtype)
    og = (Xg @ W_eff.T) * s + bias
    ob = (Xb @ W_eff.T) * s + bias
    near = lambda a, b: torch.cdist(a, b).topk(k, dim=1, largest=False).values.mean(1)  # noqa: E731
    return wp * ((og - Yg) ** 2).mean() + wsb * (near(ob, Yg).mean() - oc * near(ob, Yb).mean())


def ref_fit(W, bias, Xg, Yg, Xb, Yb, A0, steps, lr, max_iter, history, **lw):
    A = A0.clone().requires_grad_(True)
    B = torch.zeros(W.shape[0], A0.shape[0], dtype=A0.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([A, B], lr=lr, max_iter=max_iter, history_size=history, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        v = ref_loss(W, bias, A, B, Xg, Yg, Xb, Yb, **lw)
        v.backward()
        return v
    for _ in range(steps):
        opt.step(closure)
    return A.detach(), B.detach()


def synthetic(dtype=torch.float32, d_in=10, d=6, ng=6, nb=5, seed=0, zero_col=None):
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(d_in, d, generator=g, dtype=torch.float64)
    if zero_col is not None:
        V[:, zero_col] = 0
    bias = torch.randn(d, generator=g, dtype=torch.float64) * 0.1
    Xg, Xb = torch.randn(ng, d_in, generator=g, dtype=torch.float64), torch.randn(nb, d_in, generator=g, dtype=torch.float64) + 0.5
    Yg, Yb = Xg @ V + bias, Xb @ V + bias
    return [t.to(dtype) for t in (V, bias, Xg, Yg, Xb, Yb)]


def pieces_of(deltas):
    return {n: [(p.kind, p.A.clone(), p.B.clone()) if p.kind == "lowrank" else (p.kind, p.numel()) for p in ps] for n, ps in deltas.items()}


def same_pieces(a, b):
    assert set(a) == set(b)
    for n in a:
        assert len(a[n]) == len(b[n]), n
        for x, y in zip(a[n], b[n]):
            assert x[0] == y[0] == "lowrank" and torch.equal(x[1], y[1]) and torch.equal(x[2], y[2]), n


# ------------------------------------------------------------------ 1. capture
def test_capture_equals_an_independent_forward_pass_and_the_matrix(ws):
    DR.measure(ws, "plain", GOOD, BAD)
    out = DR.measure(ws, "m", GOOD, BAD, module_io=True)
    want_bytes = 4 * (len(GOOD) + len(BAD)) * N * ROW
    assert out["module_io"] == {"bytes": want_bytes, "layers": N}
    # Residual means bit-identical with and without module_io (the same forward pass gives both).
    a, b = DR.load(ws, "plain"), DR.load(ws, "m")
    assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) and a[0]["module_io"] is None
    with open(os.path.join(io_dir(ws, "m"), "io.json")) as f:
        info = json.load(f)
    assert info["counts"] == {"good": 6, "bad": 5} and info["bytes"] == want_bytes and info["dtype"] == "float32"
    for L in info["layers"]:
        assert L["row"] == ROW and L["files"] == {"good": f"good.L{L['layer']}.npy", "bad": f"bad.L{L['layer']}.npy"}
        assert L["components"]["attn.o_proj"] == {"d_in": H * DH, "captured": True, "why": None, "x": 0, "y": H * DH}
        assert L["components"]["mlp.down_proj"] == {"d_in": F, "captured": True, "why": None, "x": H * DH + D, "y": H * DH + D + F}
    for label, texts in (("good", GOOD), ("bad", BAD)):
        for l in range(N):
            got = stored(ws, "m", label, l)
            assert got.shape == (len(texts), ROW) and got.dtype == torch.float32
            for i, t in enumerate(texts):
                z, ao, post, mo = own_capture(ws, t)[l]
                assert torch.allclose(got[i], torch.cat([z, ao, post, mo]), atol=1e-6), (label, l, i)
            mats = matrices(ws, l)
            V, b = mats["attn.o_proj"]
            assert torch.allclose(got[:, :H * DH] @ V + b, got[:, H * DH:H * DH + D], atol=1e-5)
            V, b = mats["mlp.down_proj"]
            assert torch.allclose(got[:, H * DH + D:H * DH + D + F] @ V + b, got[:, H * DH + D + F:], atol=1e-5)
    # Files of the stated size: the .npy header, then N x row float32.
    for fn in os.listdir(io_dir(ws, "m")):
        if fn.endswith(".npy"):
            n = 6 if fn.startswith("good") else 5
            arr = np.load(os.path.join(io_dir(ws, "m"), fn), mmap_mode="r")
            assert os.path.getsize(os.path.join(io_dir(ws, "m"), fn)) == arr.offset + 4 * n * ROW
    assert DR.load_io(ws, "m")["bytes"] == want_bytes
    assert [e["name"] for e in DR.entries(ws)] == ["m", "plain"]                     # the directory is not an entry


def test_capture_on_the_llama_named_checkpoint_with_module_hooks(tmp_path):
    from test_steering_worker import D as HD, F as HF, LAYERS, hf_runner
    r, model, tensors = hf_runner(tmp_path)
    ws = r.ws
    good, bad = ["abc", "fed"], ["kji", "lal", "bb"]
    DR.measure(ws, "h", good, bad, module_io=True)
    seen = {}
    handles = []
    for l, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: seen.__setitem__((l, "x_attn"), a[0][0, -1].clone())))
        handles.append(layer.self_attn.register_forward_hook(lambda m, a, o, l=l: seen.__setitem__((l, "y_attn"), o[0][0, -1].clone())))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: seen.__setitem__((l, "x_mlp"), a[0][0, -1].clone())))
        handles.append(layer.mlp.register_forward_hook(lambda m, a, o, l=l: seen.__setitem__((l, "y_mlp"), o[0, -1].clone())))
    try:
        for label, texts in (("good", good), ("bad", bad)):
            for l in range(LAYERS):
                got = stored(ws, "h", label, l)
                assert got.shape == (len(texts), HD + HD + HF + HD)
                Wo = tensors[f"model.layers.{l}.self_attn.o_proj.weight"]
                Wd = tensors[f"model.layers.{l}.mlp.down_proj.weight"]
                assert torch.allclose(got[:, :HD] @ Wo.T, got[:, HD:2 * HD], atol=1e-5)          # no bias in this model
                assert torch.allclose(got[:, 2 * HD:2 * HD + HF] @ Wd.T, got[:, 2 * HD + HF:], atol=1e-5)
            for i, t in enumerate(texts):
                with torch.no_grad():
                    model(torch.tensor([ws.encode(t)]))
                for l in range(LAYERS):
                    want = torch.cat([seen[(l, "x_attn")], seen[(l, "y_attn")], seen[(l, "x_mlp")], seen[(l, "y_mlp")]])
                    assert torch.allclose(stored(ws, "h", label, l)[i], want, atol=1e-6), (label, l, i)
    finally:
        for h in handles:
            h.remove()
    info = DR.load_io(ws, "h")
    assert info["layers"][0]["components"]["attn.o_proj"]["d_in"] == HD and info["layers"][0]["components"]["mlp.down_proj"]["d_in"] == HF


def test_files_are_preallocated_and_complete_only_with_io_json(ws, monkeypatch):
    seen = []
    d = os.path.join(ws.dir, "directions")

    def look(done, total):
        tmp = [f for f in os.listdir(d) if f.startswith("p.io.") and f.endswith(".tmp")]
        assert len(tmp) == 1 and not os.path.exists(io_dir(ws, "p")) and not os.path.exists(os.path.join(d, "p.json"))
        files = sorted(os.listdir(os.path.join(d, tmp[0])))
        assert files == ["bad.L0.npy", "bad.L1.npy", "good.L0.npy", "good.L1.npy"]          # no io.json yet
        for fn in files:
            n = 6 if fn.startswith("good") else 5
            arr = np.load(os.path.join(d, tmp[0], fn), mmap_mode="r")                    # whole size from the start
            assert arr.shape == (n, ROW) and os.path.getsize(os.path.join(d, tmp[0], fn)) == arr.offset + 4 * n * ROW
        seen.append(done)
    DR.measure(ws, "p", GOOD, BAD, module_io=True, progress=look)
    assert seen == [4, 6, 10, 11] and os.path.isfile(os.path.join(io_dir(ws, "p"), "io.json"))
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")]
    # Chunking does not change what is stored (CHUNK 1 and 3 against the default 4).
    want = {fn: open(os.path.join(io_dir(ws, "p"), fn), "rb").read() for fn in os.listdir(io_dir(ws, "p")) if fn.endswith(".npy")}
    for chunk in (1, 3):
        monkeypatch.setattr(DR, "CHUNK", chunk)
        DR.measure(ws, "p", GOOD, BAD, module_io=True, replace=True)
        got = {fn: open(os.path.join(io_dir(ws, "p"), fn), "rb").read() for fn in os.listdir(io_dir(ws, "p")) if fn.endswith(".npy")}
        assert got == want, chunk


def test_a_stop_or_a_failure_leaves_nothing_and_replace_and_delete_remove_the_directory(ws, monkeypatch):
    d = os.path.join(ws.dir, "directions")
    seen = []
    out = DR.measure(ws, "s", GOOD, BAD, module_io=True, progress=lambda a, b: seen.append(a), stopped=lambda: len(seen) >= 1)
    assert out["stopped"] and out["done"] == 4 and os.listdir(d) == []
    real = DR.residuals
    calls = []

    def fail_second(w, ids, variant="base", module_io=False):
        calls.append(module_io)
        if len(calls) == 2:
            raise RuntimeError("the job failed")
        return real(w, ids, variant, module_io)
    monkeypatch.setattr(DR, "residuals", fail_second)
    with pytest.raises(RuntimeError, match="the job failed"):
        DR.measure(ws, "f", GOOD, BAD, module_io=True)
    assert calls == [True, True] and os.listdir(d) == []
    monkeypatch.undo()
    # replace without module_io removes the directory; with module_io writes a new one; delete removes it.
    DR.measure(ws, "r", GOOD, BAD, module_io=True)
    assert os.path.isdir(io_dir(ws, "r"))
    DR.measure(ws, "r", GOOD[:2], BAD[:2], replace=True)
    assert not os.path.exists(io_dir(ws, "r")) and DR.load(ws, "r")[0]["module_io"] is None
    DR.measure(ws, "r", GOOD[:2], BAD[:2], module_io=True, replace=True)
    assert DR.load_io(ws, "r")["counts"] == {"good": 2, "bad": 2}
    DR.delete(ws, "r")
    assert os.listdir(d) == []


def test_a_failure_after_the_directory_moved_in_leaves_nothing(ws, monkeypatch):
    real = DR._write_json

    def refuse_meta(path, obj):
        if path.endswith(os.path.join("directions", "g.json")):
            raise OSError("disk full")
        return real(path, obj)
    monkeypatch.setattr(DR, "_write_json", refuse_meta)
    with pytest.raises(OSError, match="disk full"):
        DR.measure(ws, "g", GOOD[:2], BAD[:2], module_io=True)
    left = os.listdir(os.path.join(ws.dir, "directions"))
    assert not any(f.startswith("g.io") for f in left) and "g.json" not in left


def test_the_disk_guard_refuses_before_any_runner_is_made(sess, monkeypatch):
    built = []

    def refuse(w):
        built.append(w.name)
        raise AssertionError("the runner was built for a request that should have been refused first")
    monkeypatch.setattr(wsp.runner, "make_runner", refuse)
    need = 4 * (len(GOOD) + len(BAD)) * N * ROW
    monkeypatch.setattr(DR, "_free_bytes", lambda path: DR.DISK_RESERVE + need - 1)
    with pytest.raises(DR.DirectionError, match=f"needs {need:,} bytes on disk; the workspace's file system has "
                                                f"{DR.DISK_RESERVE + need - 1:,} bytes free"):
        DR.measure(sess.ws, "m", GOOD, BAD, module_io=True)
    r = client(sess).post("/api/directions/measure", json={"name": "m", "good": GOOD, "bad": BAD, "module_io": True})
    assert r.status_code == 400 and f"{need:,} bytes" in r.json()["error"] and f"{DR.DISK_RESERVE + need - 1:,}" in r.json()["error"]
    assert built == [] and sess.ws._runner is None
    monkeypatch.setattr(DR, "_free_bytes", lambda path: DR.DISK_RESERVE + need)          # exactly enough
    DR.check_measure(sess.ws, "m", GOOD, BAD, module_io=True)
    assert built == []
    for bad in ("yes", 1):
        with pytest.raises(DR.DirectionError, match="module_io must be true or false"):
            DR.check_measure(sess.ws, "m", GOOD, BAD, module_io=bad)


# ------------------------------------------------------------------ 2. the loss
@pytest.mark.parametrize("preserve", [True, False])
@pytest.mark.parametrize("zero_col", [None, 2])
def test_the_loss_equals_a_dense_formula_in_the_adapter_layout(preserve, zero_col):
    for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 2e-5)):
        V, bias, Xg, Yg, Xb, Yb = synthetic(dtype, zero_col=zero_col)
        g = torch.Generator().manual_seed(9)
        Yg = Yg + 0.3 * torch.randn(Yg.shape, generator=g, dtype=torch.float64).to(dtype)
        A = torch.randn(3, V.shape[0], generator=g, dtype=torch.float64).to(dtype).requires_grad_(True)
        B = torch.randn(V.shape[1], 3, generator=g, dtype=torch.float64).to(dtype).requires_grad_(True)
        p = {"preserve_row_magnitudes": preserve, "neighbor_count": 2, "preserve_good_behavior_weight": 1.5,
             "steer_bad_behavior_weight": 0.7, "overcorrect_relative_weight": 0.4}
        mine = ARA.loss(A, B, ARA.prepare_data(V, bias, Xg, Yg, Xb, Yb), p)
        ga, gb = torch.autograd.grad(mine, (A, B))
        want = ref_loss(V.T, bias, A, B, Xg, Yg, Xb, Yb, 1.5, 0.7, 0.4, 2, preserve)
        wa, wb = torch.autograd.grad(want, (A, B))
        assert float(abs(mine.detach() - want.detach())) <= tol * float(abs(want.detach())), (dtype, float(mine), float(want))
        assert torch.allclose(ga, wa, rtol=tol * 50, atol=tol * 50) and torch.allclose(gb, wb, rtol=tol * 50, atol=tol * 50)
        assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
        if zero_col is not None and preserve:
            W = V.T.contiguous()                         # the fit's layout: row j of W is column j of V
            s = ARA.column_scale(W, A.detach(), B.detach(), torch.linalg.vector_norm(W, dim=1))
            assert float(s[zero_col]) == 1.0 and bool((s[torch.arange(V.shape[1]) != zero_col] != 1).all())
    # At B = 0 the change is zero, so the scale is 1 and only the measured mismatch is left in preserve.
    V, bias, Xg, Yg, Xb, Yb = synthetic(torch.float64)
    A, B = torch.randn(3, V.shape[0], dtype=torch.float64), torch.zeros(V.shape[1], 3, dtype=torch.float64)
    p = {"preserve_row_magnitudes": True, "neighbor_count": 1, "preserve_good_behavior_weight": 1.0,
         "steer_bad_behavior_weight": 1.0, "overcorrect_relative_weight": 0.0}
    want = torch.cdist(Xb @ V + bias, Yg).min(1).values.mean()
    assert float(ARA.loss(A, B, ARA.prepare_data(V, bias, Xg, Yg, Xb, Yb), p)) == pytest.approx(float(want), rel=1e-12)


# ------------------------------------------------------------------ 3. initial values
def draws(seed, r, sizes):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.init.kaiming_uniform_(torch.empty(r, n), a=math.sqrt(5), generator=g) for n in sizes]


def record_initial_values(monkeypatch):
    got = []

    def fake_fit(data, A0, B0, p, where=""):
        got.append((where, A0.clone(), B0.clone()))
        return A0, torch.zeros_like(B0), {"loss_before": 0.0, "loss_after": 0.0, "scale_min": 1.0, "scale_max": 1.0}
    monkeypatch.setattr(ARA, "fit", fake_fit)
    return got


def test_initial_values_are_the_stated_draws_in_the_stated_order(ws, monkeypatch):
    real_available = FunctionalRunner.available
    monkeypatch.setattr(FunctionalRunner, "available", lambda self, name: name != "blocks.0.mlp.hook_post" and real_available(self, name))
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    monkeypatch.undo()
    got = record_initial_values(monkeypatch)
    res = run_script(ws, DR.script_for(ara(seed=7)))
    assert [w for w, _, _ in got] == ["layer 0 attn.o_proj", "layer 1 attn.o_proj", "layer 1 mlp.down_proj"]
    for (w, A0, B0), want in zip(got, draws(7, 3, [H * DH, H * DH, F])):
        assert torch.equal(A0, want), w
        assert float(A0.abs().max()) <= 1 / math.sqrt(A0.shape[1]) and torch.equal(B0, torch.zeros(D, 3))
    assert res["deltas"] == {}                                                           # B stayed 0: nothing written
    assert "L0 mlp.down_proj: skipped (hook point blocks.0.mlp.hook_post is not available for this model)" in res["output"]
    # A fresh generator per call: starting at layer 1, its first draw is the generator's first.
    got.clear()
    run_script(ws, DR.script_for(ara(seed=7, start_layer_index=1)))
    assert [w for w, _, _ in got] == ["layer 1 attn.o_proj", "layer 1 mlp.down_proj"]
    assert all(torch.equal(A0, want) for (_, A0, _), want in zip(got, draws(7, 3, [H * DH, F])))


# ------------------------------------------------------------------ 4. the fit
def test_the_fit_equals_an_independent_lbfgs_on_the_dense_formulation():
    """The two formulations are the same function, so in exact arithmetic the runs are identical; they differ only
    by rounding (P + (X A^T) B^T against X W_eff^T, column norms of V + A^T B^T against row norms of W + B A), which
    L-BFGS's line searches and curvature pairs carry along and amplify with every evaluation. Tolerances, relative
    Frobenius distance of B @ A: 1e-9 in float64 (rounding there is ~1e-16; the amplification over these 2 x 10
    iterations stays far below 1e-9; observed ~3e-14). In float32 each run, the code's and the reference's, must lie
    within 1e-4 of the float64 reference (the float32 noise floor of this problem is ~4e-5: that is how far the
    float32 reference itself lands from the float64 one), so the two float32 runs lie within 2e-4 of each other
    (observed ~7e-5). The bound is honest from the other side too: changing one parameter (neighbor_count 2 -> 1)
    moves the result by more than 1e-2. Longer runs (60 evaluations and more) drift apart by 1e-2..1e-1 in float32
    against float64 for either formulation, so the comparison is kept short rather than given a loose bound."""
    lw = {"wp": 1.0, "wsb": 0.5, "oc": 0.2, "k": 2, "preserve": True}
    p = {"preserve_good_behavior_weight": 1.0, "steer_bad_behavior_weight": 0.5, "overcorrect_relative_weight": 0.2,
         "neighbor_count": 2, "preserve_row_magnitudes": True, "learning_rate": 1.0, "max_iter": 10, "history_size": 10,
         "n_optimization_steps": 2}
    V, bias, Xg, Yg, Xb, Yb = synthetic(torch.float64)
    g = torch.Generator().manual_seed(100)
    # Captured outputs a little off the matrix's own (as on a model that runs in lower precision), so that no
    # distance starts at 0, where the direction of its gradient would be rounding noise.
    Yg = Yg + 0.3 * torch.randn(Yg.shape, generator=g, dtype=torch.float64)
    Yb = Yb + 0.3 * torch.randn(Yb.shape, generator=g, dtype=torch.float64)
    A0 = draws(3, 3, [V.shape[0]])[0]
    rel = lambda a, b: float((a.double() - b.double()).norm() / b.double().norm())  # noqa: E731
    out = {}
    for dtype in (torch.float64, torch.float32):
        t = [x.to(dtype) for x in (V, bias, Xg, Yg, Xb, Yb)]
        A, B, info = ARA.fit(ARA.prepare_data(*t), A0.to(dtype), torch.zeros(V.shape[1], 3, dtype=dtype), p, "t")
        rA, rB = ref_fit(t[0].T, t[1], t[2], t[3], t[4], t[5], A0.to(dtype), 2, 1.0, 10, 10, **lw)
        out[dtype] = (B @ A, rB @ rA, info, (rA, rB), t)
    (m64, r64, _, _, _), (m32, r32, info, (rA, rB), t) = out[torch.float64], out[torch.float32]
    other = ARA.fit(ARA.prepare_data(*t), A0, torch.zeros(V.shape[1], 3), {**p, "neighbor_count": 1}, "t")
    print(f"float64 mine vs ref {rel(m64, r64):.3g}; float32 mine vs ref {rel(m32, r32):.3g}, ref32 vs ref64 "
          f"{rel(r32, r64):.3g}, mine32 vs ref64 {rel(m32, r64):.3g}; k=1 vs k=2 {rel(other[1] @ other[0], r32):.3g}")
    assert float(r64.norm()) > 1
    assert rel(m64, r64) <= 1e-9
    assert rel(m32, r64) <= 1e-4 and rel(r32, r64) <= 1e-4 and rel(m32, r32) <= 2e-4
    assert rel(other[1] @ other[0], r32) > 1e-2
    assert m32.dtype == torch.float32 and info["loss_after"] <= info["loss_before"]
    Vt, bias, Xg, Yg, Xb, Yb = t[0].T, t[1], t[2], t[3], t[4], t[5]
    assert info["loss_after"] == pytest.approx(float(ref_loss(Vt, bias, rA, rB, Xg, Yg, Xb, Yb, **lw)), rel=1e-4)
    s = Vt.norm(dim=1) / (Vt + rB @ rA).norm(dim=1)
    assert info["scale_min"] == pytest.approx(float(s.min()), rel=1e-4) and info["scale_max"] == pytest.approx(float(s.max()), rel=1e-4)


def heretic_order_fit(W, bias, A0, Xg, Yg, Xb, Yb, p):
    """One module's fit as Heretic runs it: its layout (W [d, D_in]) and its order of operations, copied from
    Heretic's modifier (src/heretic/modifiers/ara.py at 64e9991, AGPL-3.0) with the model plumbing removed. Kept
    here as the reference the IDE's fit must reproduce exactly; nothing else uses it."""
    def near(a, b):
        distances = torch.cdist(a, b)
        nearest, _ = distances.topk(p["neighbor_count"], dim=1, largest=False)
        return nearest.mean(1)
    lora_A = torch.zeros_like(A0, requires_grad=True)
    lora_B = torch.zeros(W.shape[0], A0.shape[0], dtype=W.dtype, requires_grad=True)
    W_row_norms = torch.linalg.vector_norm(W, dim=1, keepdim=True).detach()
    with torch.no_grad():
        lora_A.copy_(A0)
    good_product, bad_product = Xg @ W.T, Xb @ W.T

    def objective(A, B):
        new_good_output = good_product + (Xg @ A.T) @ B.T
        new_bad_output = bad_product + (Xb @ A.T) @ B.T
        if p["preserve_row_magnitudes"]:
            eff_row_norms = torch.linalg.vector_norm(W + (B @ A), dim=1)
            scale = W_row_norms.squeeze(1) / eff_row_norms
            new_good_output = new_good_output * scale
            new_bad_output = new_bad_output * scale
        if bias is not None:
            new_good_output = new_good_output + bias
            new_bad_output = new_bad_output + bias
        keep = ((new_good_output - Yg) ** 2).mean()
        steer = near(new_bad_output, Yg).mean() + p["overcorrect_relative_weight"] * -near(new_bad_output, Yb).mean()
        return p["preserve_good_behavior_weight"] * keep + p["steer_bad_behavior_weight"] * steer

    opt = torch.optim.LBFGS([lora_A, lora_B], lr=p["learning_rate"], max_iter=p["max_iter"],
                            history_size=p["history_size"], line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = objective(lora_A, lora_B)
        loss.backward()
        return loss
    for _ in range(p["n_optimization_steps"]):
        opt.step(closure)
    return lora_A.detach(), lora_B.detach()


@pytest.mark.parametrize("preserve", [True, False])
@pytest.mark.parametrize("with_bias", [True, False])
def test_the_fit_takes_exactly_the_steps_of_a_loop_in_heretics_order(preserve, with_bias):
    """The fit amplifies a difference in the last digit of anything it reads (docs/DIRECTIONS.md, "ARA against
    Heretic"), so agreeing with Heretic means doing the same operations in the same order, and the test is
    equality, not a tolerance: every element of A and B, at the full default budget (5 x 20), in both precisions.
    What this pins down: the row norms are summed over W + B A (not over V + A^T B^T), and the two outputs are built
    before the scale (the order the gradient's three paths into A and B are summed in). A module without a bias adds
    a zero vector here where Heretic adds nothing; that changes no bit."""
    p = {"preserve_good_behavior_weight": 0.3, "steer_bad_behavior_weight": 0.2, "overcorrect_relative_weight": 0.4,
         "neighbor_count": 2, "preserve_row_magnitudes": preserve, "learning_rate": 1.0, "max_iter": 20,
         "history_size": 10, "n_optimization_steps": 5}
    V, bias, Xg, Yg, Xb, Yb = synthetic(torch.float64)
    g = torch.Generator().manual_seed(100)
    Yg = Yg + 0.3 * torch.randn(Yg.shape, generator=g, dtype=torch.float64)
    Yb = Yb + 0.3 * torch.randn(Yb.shape, generator=g, dtype=torch.float64)
    if not with_bias:
        bias = torch.zeros_like(bias)
    A0 = draws(3, 3, [V.shape[0]])[0]
    for dtype in (torch.float32, torch.float64):
        V_, b_, xg, yg, xb, yb = (x.to(dtype) for x in (V, bias, Xg, Yg, Xb, Yb))
        A, B, info = ARA.fit(ARA.prepare_data(V_, b_, xg, yg, xb, yb), A0.to(dtype), torch.zeros(V.shape[1], 3, dtype=dtype), p, "t")
        hA, hB = heretic_order_fit(V_.T.contiguous(), b_ if with_bias else None, A0.to(dtype), xg, yg, xb, yb, p)
        assert float((hB @ hA).norm()) > 0.1 and info["loss_after"] < info["loss_before"]      # a real fit, not a no-op
        assert torch.equal(A, hA) and torch.equal(B, hB), (dtype, float((B @ A - hB @ hA).norm() / (hB @ hA).norm()))


def test_the_fit_does_not_raise_the_loss_and_two_runs_give_identical_pieces(ws):
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    script = DR.script_for(ara(steer_bad_behavior_weight=0.5))
    a, b = run_script(ws, script), run_script(ws, script)
    same_pieces(pieces_of(a["deltas"]), pieces_of(b["deltas"]))
    assert a["output"] == b["output"]
    for line in a["output"].splitlines()[1:]:
        before, after = map(float, re.search(r"loss (\S+) -> (\S+),", line).groups())
        assert after <= before, line


# ------------------------------------------------------------------ 5. behaviour
def test_after_a_strong_fit_the_bad_outputs_are_closer_to_the_good_ones(ws):
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    deltas = run_script(ws, DR.script_for(ara(start_layer_index=1, end_layer_index=2, steer_bad_behavior_weight=5.0,
                                              lora_rank=4, n_optimization_steps=5, max_iter=20)))["deltas"]
    assert set(deltas) == {"blocks.1.attn.W_O", "blocks.1.mlp.W_out"}
    base = ws.runner.params("base")
    cand = dict(base)
    for name in deltas:
        cand[name] = ws.read(name, (), "current", deltas)
    # Layer 1's attention output: its input (layer 1's z) is not changed by the fit, so this is the matrix's own effect.
    out = lambda p, texts: torch.stack([own_capture(ws, t, params=p)[1][1] for t in texts])  # noqa: E731
    good0, bad0, good1, bad1 = out(base, GOOD), out(base, BAD), out(cand, GOOD), out(cand, BAD)
    near = lambda a, b: float(torch.cdist(a, b).min(1).values.mean())  # noqa: E731
    assert near(bad1, good0) < near(bad0, good0)
    assert float((good1 - good0).norm(dim=1).mean()) < float((bad1 - bad0).norm(dim=1).mean())


# ------------------------------------------------------------------ 6. no steer, nothing written
def test_a_zero_steer_weight_writes_nothing(ws):
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    res = run_script(ws, DR.script_for(ara(steer_bad_behavior_weight=0.0, preserve_good_behavior_weight=3.0)))
    assert res["deltas"] == {} and "0 components in 0 layers, 0 values stored" in res["output"]
    assert res["output"].count("skipped (steer_bad_behavior_weight is 0: nothing to fit)") == 2 * N
    plan = DR.plan(ws, ara(steer_bad_behavior_weight=0))
    assert plan["totals"]["components"] == 0 and plan["totals"]["stored"] == 0 and plan["totals"]["skipped"] == 2 * N


# ------------------------------------------------------------------ 7. the consistency check
def test_another_patch_on_a_fitted_matrix_is_refused_and_a_small_mismatch_is_printed(sess):
    ws = sess.ws
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    assert sess.apply("attn(1).W_O *= 1.5", dry_run=False)["committed"]
    script = DR.script_for(ara(start_layer_index=1))
    r = sess.apply(script)
    assert not r["ok"] and "layer 1 attn.o_proj: mismatch" in r["error"]
    assert "the saved inputs and outputs do not match this matrix: measure again on these weights, or switch other patches off" in r["error"]
    # The same through the script directly, naming the number: ||X (1.5 V) + b - Y|| / ||Y|| computed here.
    g = stored(ws, "m", "good", 1)
    V, b = matrices(ws, 1)["attn.o_proj"]
    want = float((g[:, :H * DH] @ (1.5 * V) + b - g[:, H * DH:H * DH + D]).norm() / g[:, H * DH:H * DH + D].norm())
    with pytest.raises(Exception, match=f"mismatch {want:.4g} on the good prompts"):
        run_script(ws, script)
    pid = ws.patches[-1].id
    ws.set_patch(pid, enabled=False)
    assert "L1 attn.o_proj: loss" in run_script(ws, script)["output"]
    ws.delete_patch(pid)
    # Below the limit: allowed, and the number is printed.
    assert sess.apply("attn(1).W_O *= 1.001", dry_run=False)["committed"]
    small = float((g[:, :H * DH] @ (1.001 * V) + b - g[:, H * DH:H * DH + D]).norm() / g[:, H * DH:H * DH + D].norm())
    assert 1e-4 < small < 0.05
    out = run_script(ws, script)["output"]
    printed = float(re.search(r"L1 attn\.o_proj: .* mismatch good (\S+) bad", out).group(1))
    assert printed == pytest.approx(small, rel=2e-3)


# ------------------------------------------------------------------ 8. zero columns, slabs, a missing hook point
def test_a_zero_norm_column_with_preserve_row_magnitudes(tmp_path):
    p = init_params(Config(**CFG), seed=1, std=0.3)
    p["blocks.1.attn.W_O"][:, :, 5] = 0
    ws = make_ws(tmp_path / "z", p)
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    res = run_script(ws, DR.script_for(ara(start_layer_index=1, steer_bad_behavior_weight=1.0)))
    assert all(pc.finite() for ps in res["deltas"].values() for pc in ps)
    line = next(x for x in res["output"].splitlines() if "L1 attn.o_proj" in x)
    before, after = map(float, re.search(r"loss (\S+) -> (\S+),", line).groups())
    lo, hi = map(float, re.search(r"scale (\S+)\.\.(\S+),", line).groups())
    assert after < before and math.isfinite(lo) and math.isfinite(hi)


def test_the_slab_path_gives_the_same_pieces_and_no_read_exceeds_the_limit(ws, monkeypatch):
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    script = DR.script_for(ara(steer_bad_behavior_weight=0.5))
    whole = pieces_of(run_script(ws, script)["deltas"])
    limit = 50                                                                # two rows of 24 per read
    monkeypatch.setattr(Wmod, "STREAM_ELEMS", limit)
    reads = []
    real = ws.store.read

    def read(name, region=None):
        from wsp.store import norm_index, region_size
        reads.append((name, region_size(norm_index((), ws.shape(name)) if region is None else region)))
        return real(name, region)
    monkeypatch.setattr(ws.store, "read", read)
    ws._cache.clear()
    ws._cache_bytes = 0
    same_pieces(pieces_of(run_script(ws, script)["deltas"]), whole)
    weight_reads = [n for name, n in reads if name.endswith(("W_O", "W_out"))]
    assert weight_reads and max(weight_reads) <= limit and len(weight_reads) > 20


def test_a_layer_without_a_hook_point_is_skipped_with_the_reason(ws, monkeypatch):
    real_available = FunctionalRunner.available
    monkeypatch.setattr(FunctionalRunner, "available", lambda self, name: name != "blocks.1.hook_attn_out" and real_available(self, name))
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    monkeypatch.undo()
    info = DR.load_io(ws, "m")
    assert info["layers"][1]["row"] == F + D and info["layers"][1]["components"]["mlp.down_proj"]["x"] == 0
    assert info["layers"][1]["components"]["attn.o_proj"] == {
        "d_in": H * DH, "captured": False, "why": "hook point blocks.1.hook_attn_out is not available for this model"}
    assert stored(ws, "m", "good", 1).shape == (6, F + D)
    plan = DR.plan(ws, ara())
    row = next(r for r in plan["rows"] if r["layer"] == 1 and r["component"] == "attn.o_proj")
    assert row["skipped"] == "hook point blocks.1.hook_attn_out is not available for this model" and row["stored"] == 0
    assert plan["totals"]["components"] == 3 and plan["totals"]["skipped"] == 1
    deltas = run_script(ws, DR.script_for(ara(steer_bad_behavior_weight=0.5)))["deltas"]
    assert "blocks.1.attn.W_O" not in deltas and "blocks.1.mlp.W_out" in deltas


# ------------------------------------------------------------------ 9. through Session.apply
def test_session_apply_scope_dry_run_stored_values_and_untouched_tensors(sess):
    ws = sess.ws
    DR.measure(ws, "plain", GOOD, BAD)
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    params = ara(steer_bad_behavior_weight=0.5)
    script = DR.script_for(params)
    plan = DR.plan(ws, params)
    dry = sess.apply(script, dry_run=True)
    assert dry["ok"] and not dry["committed"] and ws.patches == []
    sess.set_task({"scope": ["blocks.0"]})
    refused = sess.apply(script, enforce_scope=True)
    assert not refused["ok"] and "outside the task scope" in refused["error"] and ws.patches == []
    sess.set_task({"scope": ["model"]})
    res = run_script(ws, script)
    assert sum(pc.numel() for ps in res["deltas"].values() for pc in ps) == plan["totals"]["stored"] == 2 * 3 * (H * DH + H * D) + 2 * 3 * (F + D)
    assert all(pc.kind == "lowrank" for ps in res["deltas"].values() for pc in ps)
    candidate = {name: ws.read(name, (), "current", res["deltas"]) for name in res["deltas"]}
    done = sess.apply(script)
    assert done["ok"] and done["committed"] and len(ws.patches) == 1
    patch = ws.patches[0]
    assert patch.source == {"kind": "wscript", "script": script}
    assert sorted(patch.deltas) == ["blocks.0.attn.W_O", "blocks.0.mlp.W_out", "blocks.1.attn.W_O", "blocks.1.mlp.W_out"]
    for name in ws.names():
        if name not in patch.deltas:
            assert torch.equal(ws.read(name, (), "current"), ws.read(name, (), "base")), name
    # The committed weights are the candidate's (the same fit, committed through the patch).
    for name in patch.deltas:
        assert torch.allclose(ws.read(name, (), "current"), candidate[name], atol=1e-6), name
    r = sess.apply(DR.script_for(ara("plain")))
    assert not r["ok"] and "measure it again with module_io" in r["error"] and len(ws.patches) == 1


def test_parameters_are_checked_and_each_error_names_its_field(ws):
    DR.measure(ws, "m", GOOD[:2], BAD[:3], module_io=True)
    for change, field in [
        ({"extra": 1}, "extra: unknown field"), ({"direction": "a b"}, "direction: name must be"),
        ({"start_layer_index": -1}, "start_layer_index"), ({"start_layer_index": 2}, "start_layer_index: must be from 0 to 1"),
        ({"start_layer_index": True}, "start_layer_index"), ({"end_layer_index": 3}, "end_layer_index: must be at most 2"),
        ({"start_layer_index": 1, "end_layer_index": 1}, "end_layer_index: must be above start_layer_index"),
        ({"end_layer_index": 1.0}, "end_layer_index"),
        ({"preserve_good_behavior_weight": -1}, "preserve_good_behavior_weight"),
        ({"steer_bad_behavior_weight": 1001}, "steer_bad_behavior_weight"),
        ({"steer_bad_behavior_weight": float("nan")}, "steer_bad_behavior_weight"),
        ({"overcorrect_relative_weight": 10.5}, "overcorrect_relative_weight: must be a finite number from 0 to 10"),
        ({"neighbor_count": 0}, "neighbor_count"), ({"neighbor_count": 3}, "neighbor_count: must be at most 2"),
        ({"neighbor_count": 1.0}, "neighbor_count"), ({"preserve_row_magnitudes": 1}, "preserve_row_magnitudes: must be true or false"),
        ({"lora_rank": 0}, "lora_rank"), ({"lora_rank": 257}, "lora_rank"),
        ({"lora_rank": 25}, "lora_rank: 25 is above min(D_in, d_model) = 24 of layer 0 attn.o_proj"),
        ({"n_optimization_steps": 51}, "n_optimization_steps"), ({"max_iter": 0}, "max_iter"), ({"history_size": 101}, "history_size"),
        ({"learning_rate": 0}, "learning_rate"), ({"learning_rate": 10.5}, "learning_rate"), ({"learning_rate": "1"}, "learning_rate"),
        ({"seed": -1}, "seed"), ({"seed": 2 ** 63}, "seed"), ({"seed": False}, "seed"),
    ]:
        with pytest.raises(DR.DirectionError, match=re.escape(field)):
            DR.plan(ws, {**ara(), **change})
    with pytest.raises(DR.DirectionMissing):
        DR.plan(ws, ara("missing"))
    with pytest.raises(DR.DirectionError, match="method: must be one of ablation, ara"):
        DR.plan(ws, {**ara(), "method": "lora"})
    assert DR.plan(ws, {**ara(), "seed": 2 ** 63 - 1, "learning_rate": 10, "lora_rank": 24})["totals"]["components"] == 4


def test_the_plan_and_the_script_line(ws):
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    params = ara(start_layer_index=1, preserve_good_behavior_weight=0.1 + 0.2, learning_rate=1 / 3, seed=2 ** 40)
    plan = DR.plan(ws, params)
    assert plan["method"] == "ara" and plan["start_layer_index"] == 1 and plan["end_layer_index"] == 2
    assert [(r["layer"], r["component"], r["views"], r["d_in"], r["stored"], r["skipped"]) for r in plan["rows"]] == [
        (1, "attn.o_proj", H, H * DH, 3 * (H * DH + H * D), None), (1, "mlp.down_proj", 1, F, 3 * (F + D), None)]
    assert all(r["scratch"] >= 4 * r["d_in"] * D * 5 for r in plan["rows"])                 # the matrix whole, and four more like it
    line = DR.script_for(params)
    import ast
    call = ast.parse(line).body[0].value
    assert call.func.id == "ablate_ara" and ast.literal_eval(call.args[0]) == "m" and "\n" not in line
    kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    assert list(kw) == ["start_layer_index", "end_layer_index", "preserve_good_behavior_weight", "steer_bad_behavior_weight",
                        "overcorrect_relative_weight", "neighbor_count", "preserve_row_magnitudes", "lora_rank",
                        "n_optimization_steps", "learning_rate", "max_iter", "history_size", "seed"]
    assert kw["preserve_good_behavior_weight"] == 0.1 + 0.2 and kw["learning_rate"] == 1 / 3 and kw["seed"] == 2 ** 40
    assert kw["end_layer_index"] is None and kw["preserve_row_magnitudes"] is True and kw["steer_bad_behavior_weight"] == 0.01
    for name in ("ablate_ara(", "start_layer_index", "preserve_good_behavior_weight", "steer_bad_behavior_weight",
                 "overcorrect_relative_weight", "neighbor_count", "preserve_row_magnitudes", "lora_rank", "n_optimization_steps",
                 "learning_rate", "max_iter", "history_size"):
        assert name in FUNCTION_DOC, name


# ------------------------------------------------------------------ 10. routes
def test_routes_module_io_get_script_for_both_methods_and_today_unchanged(sess, monkeypatch):
    c = client(sess)
    r = c.post("/api/directions/measure", json={"name": "m", "good": GOOD, "bad": BAD, "module_io": True})
    assert r.status_code == 200, r.text
    assert r.json()["module_io"] == {"bytes": 4 * 11 * N * ROW, "layers": N}
    assert c.post("/api/directions/measure", json={"name": "p", "good": GOOD, "bad": BAD}).json()["module_io"] is None
    assert c.get("/api/directions/m").json()["module_io"] == {"bytes": 4 * 11 * N * ROW, "layers": N}
    assert c.get("/api/directions/p").json()["module_io"] is None
    assert c.post("/api/directions/measure", json={"name": "q", "good": GOOD, "bad": BAD, "module_io": "yes"}).status_code == 400
    sc = c.post("/api/directions/script", json=ara(steer_bad_behavior_weight=0.5))
    assert sc.status_code == 200, sc.text
    assert sc.json()["script"] == DR.script_for(ara(steer_bad_behavior_weight=0.5)) and sc.json()["script"].startswith("ablate_ara(")
    assert sc.json()["plan"]["method"] == "ara" and sc.json()["plan"]["totals"]["components"] == 4
    done = c.post("/api/edit", json={"script": sc.json()["script"]}).json()
    assert done["ok"] and done["committed"] and done["stored"] == sc.json()["plan"]["totals"]["stored"]
    bad = c.post("/api/directions/script", json={"method": "lora", "direction": "m"})
    assert bad.status_code == 400 and "method: must be one of ablation, ara" in bad.json()["error"]
    bad = c.post("/api/directions/script", json={**ara(), "weight_distributions": {}})
    assert bad.status_code == 400 and "weight_distributions: unknown field" in bad.json()["error"]
    bad = c.post("/api/directions/script", json=ara("p"))
    assert bad.status_code == 400 and "measure it again with module_io" in bad.json()["error"]
    # Today's ablation requests: unchanged without "method", the same with "method": "ablation".
    dists = {"attn.o_proj": {"max_weight": 1.0, "max_weight_position": 0.0, "min_weight": 0.5, "min_weight_distance": 2.0}}
    old = {"direction": "p", "weight_distributions": dists}
    a = c.post("/api/directions/script", json=old).json()
    b = c.post("/api/directions/script", json={**old, "method": "ablation"}).json()
    assert a == b and a["script"].startswith('ablate_direction("p", ') and "method" not in a["plan"]
    assert a["script"] == DR.script_for(old) and a["plan"] == DR.plan(sess.ws, old)
    bad = c.post("/api/directions/script", json={**old, "lora_rank": 3})
    assert bad.status_code == 400 and "lora_rank: unknown field" in bad.json()["error"]
    assert c.get("/api/directions").json()["limits"]["methods"] == ["ablation", "ara"]
    assert c.delete("/api/directions/m").json() == {"ok": True, "deleted": "m"}
    assert not os.path.exists(io_dir(sess.ws, "m"))


# ------------------------------------------------------------------ 11. the job through the stub worker
def test_the_module_io_job_through_the_stub_worker_and_the_remote_runner(tmp_path, monkeypatch):
    from test_steering_worker import WspHarness, local_ws, remote
    h = WspHarness(tmp_path)
    loaded = h.load()
    here = local_ws(tmp_path)
    prompts = [here.encode(t) for t in GOOD[:3]]
    width = (N + 1) * D + N * ROW
    frame, _, _ = h.call("direction_residuals", {"prompts_ids": prompts, "variant": "base", "module_io": True})
    assert frame.type == "result" and set(frame.tensors) == {"value"} and frame.tensors["value"].shape == (3, width)
    want = DR.residuals(here, prompts, "base", True)
    assert torch.equal(frame.tensors["value"], want)
    assert torch.equal(want[:, :(N + 1) * D].reshape(3, N + 1, D), DR.residuals(here, prompts, "base"))
    frame, _, _ = h.call("direction_residuals", {"prompts_ids": prompts, "variant": "base", "module_io": "yes"})
    assert frame.type == "error" and frame["exception"] == "DirectionError" and "module_io must be true or false" in frame["message"]
    r, sent, received = remote(h, monkeypatch)
    r.hooks = frozenset(loaded["value"]["available"])          # as RemoteRunner's constructor takes it from the load job
    got = DR.residuals(h.ws, prompts, "base", True)
    assert torch.equal(got, want) and sent[-1] == ("direction_residuals", {"prompts_ids": prompts, "variant": "base", "module_io": True})
    # Fewer prompts per job when a result would pass IO_JOB_BYTES (here: room for two prompts' rows).
    monkeypatch.setattr(DR, "IO_JOB_BYTES", 2 * 4 * width + 3)
    with pytest.raises(DR.DirectionError, match="with module_io takes 1..2 prompts"):
        DR.residuals(h.ws, prompts, "base", True)
    sent.clear()
    out = DR.measure(h.ws, "w", GOOD, BAD, module_io=True)
    assert [len(a["prompts_ids"]) for _, a in sent] == [2, 2, 2, 2, 2, 1] and all(a["module_io"] for _, a in sent)
    DR.measure(here, "w", GOOD, BAD, module_io=True)
    assert out["module_io"] == {"bytes": 4 * 11 * N * ROW, "layers": N}
    for label in ("good", "bad"):
        for l in range(N):
            assert torch.equal(stored(h.ws, "w", label, l), stored(here, "w", label, l))
    h.ws._runtime_client = None
    h.shutdown()


# ------------------------------------------------------------------ a quantized llama-shaped workspace
def test_a_quantized_llama_shaped_workspace_takes_the_fit_as_patches(tmp_path, monkeypatch):
    """tests/selection_fixtures.py: a llama layout whose store reports Q4_K and refuses large reads. It has no runner,
    so its module inputs and outputs are written here: random inputs, outputs from the matrix read whole in the test."""
    from selection_fixtures import quant_workspace
    ws = quant_workspace(tmp_path / "q")
    s = ws.struct
    layout = DR.io_layout(ws)
    counts = {"good": 5, "bad": 4}
    g = torch.Generator().manual_seed(11)
    w = DR._IoWriter(ws, "q", layout, counts)
    for label, n in counts.items():
        parts = []
        for L in layout:
            for comp in DR.COMPONENTS:
                name = f"model.layers.{L['layer']}." + ("self_attn.o_proj.weight" if comp == "attn.o_proj" else "mlp.down_proj.weight")
                V = ws.read(name).T                                              # [D_in, d]: the residual stream last
                X = torch.randn(n, V.shape[0], generator=g) + (1.0 if label == "bad" else 0.0)
                parts += [X, X @ V]                                              # no output bias in this layout
        w.write(label, 0, torch.cat(parts, 1))
    meta = {"version": 1, "name": "q", "created": 0, "variant": "base", "n_layers": s.n_layers, "d_model": s.d_model,
            "counts": counts, "tokens": {}, "options": {}, "prompts_sha256": {}, "model": {}, "module_io": w.finish()}
    DR.save(ws, "q", torch.zeros(s.n_layers + 1, s.d_model), torch.zeros(s.n_layers + 1, s.d_model), meta,
            {"good": ["a"] * 5, "bad": ["b"] * 4}, io_dir=w.dir)
    monkeypatch.setattr(Wmod, "STREAM_ELEMS", 40)
    ws.store.reads.clear()
    base = {n: ws.store.t[n].clone() for n in ws.store.t}
    res = run_script(ws, DR.script_for(ara("q", start_layer_index=1, steer_bad_behavior_weight=0.5, lora_rank=2)))
    assert sorted(res["deltas"]) == ["model.layers.1.mlp.down_proj.weight", "model.layers.1.self_attn.o_proj.weight"]
    assert all(p.kind == "lowrank" and p.finite() for ps in res["deltas"].values() for p in ps)
    assert max(n for name, n in ws.store.reads if "o_proj" in name or "down_proj" in name) <= 40
    for line in res["output"].splitlines()[1:]:
        before, after = map(float, re.search(r"loss (\S+) -> (\S+),", line).groups())
        assert after < before and float(re.search(r"mismatch good (\S+) ", line).group(1)) < 1e-5, line
    assert all(torch.equal(ws.store.t[n], base[n]) for n in base)                # the base store is never written
