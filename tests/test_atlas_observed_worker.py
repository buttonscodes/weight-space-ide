"""The atlas observed stage for a model that runs in the model worker (one window per job), and the in-process path
it shares its reductions with.

No real model is downloaded, loaded or run.  Tiny random models of this project (Config + init_params, a few
thousand weights) run in this process and inside `serve_worker` on a socketpair (tests/test_worker_protocol.py's
Harness, as tests/test_steering_worker.py does), with a real Workspace and RemoteRunner on the server side; a small
fake llama-named decoder (tests/test_steering_worker.py) drives GGUFRunner.run's hook path.  The frozen
pre-change implementation in tests/observe_reference.py is the reference for the in-process output.  Every test
allocates a few MB at most.
"""
import json
import math
import os
import shutil
import socket
import sys
import threading
import time
import types
from contextlib import contextmanager

import numpy as np
import pytest
import torch

import observe_reference as REF
from test_steering_worker import FakeClient, WorkerRunner, gguf_runner
from test_worker_protocol import Harness
from wsp import atlas as at
from wsp import delta as Dl
from wsp import worker_jobs
from wsp.atlas import guide
from wsp.atlas import observe as O
from wsp.atlas import wiring as wiring_files
from wsp.model import Config, init_params
from wsp.remote_runner import RemoteRunner
from wsp.runtime_events import RuntimeRefusal, event, job_scope
from wsp.tokenizer import VocabTokenizer
from wsp.worker_jobs import JOBS, decode_result, encode_job, encode_result
from wsp.worker_protocol import Channel
from wsp.workspace import Workspace

quiet = lambda *a: None  # noqa: E731
LETTERS = "abcdefghijklmnopqrstuvwxyz "
TEXT = ("the cat sat on the mat and the dog ate the cat food while a bird flew over the old red barn "
        "then the dog sat on the mat and the cat ate the bird food ") * 4
CFG_W = dict(d_vocab=27, n_ctx=40, d_model=24, n_layers=2, n_heads=2, d_mlp=40, name="obs")      # windows of 40
CFG_H = dict(d_vocab=27, n_ctx=16, d_model=24, n_layers=3, n_heads=4, d_mlp=32, name="heads")    # 12 heads
OBSERVED = ("probe_ids.npy", "heads_observed.json", "bias_direction.npy", "bias.json", "validation.json")


def make_ws(directory, cfg=CFG_W, corpus=TEXT, seed=1):
    cfg = Config(**cfg)
    ws = Workspace.create(str(directory), cfg, VocabTokenizer(list(LETTERS), "char"), init_params(cfg, seed=seed, std=0.3))
    if corpus:
        with open(os.path.join(ws.dir, "corpus.txt"), "w") as f:
            f.write(corpus)
    return ws


def edit(ws, seed=5):
    """One committed patch (through Workspace.commit), the same for every workspace given the same seed."""
    name = "blocks.1.mlp.W_in"
    rows, _ = ws.shape(name)
    g = torch.Generator().manual_seed(seed)
    return ws.commit({name: [Dl.block((range(rows), range(0, 6)), torch.randn(rows, 6, generator=g) * 0.5)]}, "observed test edit")


def files(adir):
    return {n: open(os.path.join(adir, n), "rb").read() for n in sorted(os.listdir(adir)) if os.path.isfile(os.path.join(adir, n))}


def same_files(a, b, values_only=False):
    """Every file of two atlas directories: .npz compared array by array (the zip records write times), the rest
    byte for byte."""
    fa, fb = files(a), files(b)
    assert sorted(fa) == sorted(fb)
    for n in fa:
        if n.endswith(".npz"):
            A, B = np.load(os.path.join(a, n)), np.load(os.path.join(b, n))
            assert sorted(A.files) == sorted(B.files), n
            for k in A.files:
                assert A[k].dtype == B[k].dtype and A[k].shape == B[k].shape and np.array_equal(A[k], B[k]), (n, k)
        elif n != "meta.json" or not values_only:
            assert fa[n] == fb[n], n


# ------------------------------------------------------------------ 1. the in-process path is unchanged
def build_pair(ws, tmp_path, variant="base", **kw):
    """The observed stage written by the frozen reference and by the current code, each into its own copy of the
    same weights-stage atlas."""
    at.build(ws, stages=["weights"], log=quiet, variant=variant)
    src = at.atlas_dir(ws)
    a, b = str(tmp_path / "ref"), str(tmp_path / "new")
    shutil.copytree(src, a)
    shutil.copytree(src, b)
    ref = REF.build_observed(ws, a, quiet, variant=variant, **kw)
    new = O.build_observed(ws, b, quiet, variant=variant, **kw)
    return a, b, ref, new


def spearman_by_hand(x, y):
    """Spearman's rho from its definition, written independently of wsp.atlas.observe.spearman: each value's mean
    rank by counting, then the Pearson correlation of the ranks."""
    def ranks(a):
        return [sum(1 for b in a if b < v) + (sum(1 for b in a if b == v) + 1) / 2 for v in a]
    rx, ry = ranks(list(x)), ranks(list(y))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def reference_scipy(monkeypatch):
    """The frozen reference imports scipy.stats.spearmanr once 8 or more heads have patterns; the IDE does not
    require scipy (and this environment has none), so without it the reference gets spearman_by_hand."""
    try:
        import scipy.stats  # noqa: F401
        return
    except ImportError:
        pass
    stats = types.ModuleType("scipy.stats")
    stats.spearmanr = lambda x, y: types.SimpleNamespace(statistic=spearman_by_hand(x, y))
    scipy = types.ModuleType("scipy")
    scipy.stats = stats
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.stats", stats)


def test_the_rank_correlation_needs_no_scipy_and_handles_ties():
    g = np.random.default_rng(0)
    for _ in range(20):
        x, y = g.integers(0, 6, 30).astype(float), g.normal(size=30)
        y[:5] = y[5]                                                           # ties on both sides
        assert abs(O.spearman(x, y) - spearman_by_hand(x, y)) < 1e-12
    assert O.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0) and O.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_the_rank_correlation_agrees_with_scipy_where_scipy_is_installed():
    scipy_stats = pytest.importorskip("scipy.stats")
    g = np.random.default_rng(1)
    x, y = g.normal(size=50).round(1), g.normal(size=50).round(1)
    assert abs(O.spearman(x, y) - float(scipy_stats.spearmanr(x, y).statistic)) < 1e-12


@pytest.mark.parametrize("case", ["sampled text, two batches, 12 heads", "corpus, edited weights, a token limit",
                                  "no normalisation"])
def test_the_in_process_output_is_byte_identical_to_the_code_before_the_change(tmp_path, monkeypatch, case):
    reference_scipy(monkeypatch)
    if case.startswith("sampled"):
        ws, kw, variant = make_ws(tmp_path / "m", CFG_H, corpus=None), {}, "base"
    elif case.startswith("corpus"):
        ws, kw, variant = make_ws(tmp_path / "m"), {"tokens": 200}, "current"
        edit(ws)
    else:
        ws, kw, variant = make_ws(tmp_path / "m", {**CFG_W, "normalization": "none"}), {}, "base"
    a, b, ref, new = build_pair(ws, tmp_path, variant, **kw)
    assert {k: v for k, v in ref.items() if k != "seconds"} == {k: v for k, v in new.items() if k != "seconds"}
    same_files(a, b)
    assert set(OBSERVED) <= set(os.listdir(b)) and any(n.startswith("obs_") for n in os.listdir(b))
    if case.startswith("sampled"):
        assert ref["windows"] == 64 and json.load(open(os.path.join(b, "validation.json")))["mixers"] is not None
        assert len(json.load(open(os.path.join(b, "heads_observed.json")))) == 12
    if case.startswith("corpus"):
        assert ref["windows"] == 5


def test_one_window_at_a_time_in_process_agrees_with_batches_to_float_round_off(tmp_path):
    """batch only changes how many windows go through the model at once, so a batched pass rounds differently:
    values agree to float round-off, and where a top or bottom position differs, the value there is the same (the
    probe text repeats, so equal contexts give tied activations, and a tie may be kept at either position)."""
    ws = make_ws(tmp_path / "m")
    a, b, _, _ = build_pair(ws, tmp_path)
    one = str(tmp_path / "one")
    shutil.copytree(at.atlas_dir(ws), one)
    O.build_observed(ws, one, quiet, batch=1)
    for n in sorted(os.listdir(b)):
        if n.startswith("obs_"):
            A, B = np.load(os.path.join(b, n)), np.load(os.path.join(one, n))
            for k in ("obs_mean", "obs_std", "obs_pos_share", "obs_top_val", "obs_bot_val", "obs_frac_pos"):
                np.testing.assert_allclose(A[k], B[k], rtol=1e-5, atol=1e-6)
            for p, v in (("obs_top_pos", "obs_top_val"), ("obs_bot_pos", "obs_bot_val")):
                moved = A[p] != B[p]                                           # the probe text repeats: exact ties
                np.testing.assert_allclose(A[v][moved], B[v][moved], rtol=1e-5, atol=1e-6)   # the same value elsewhere


# ------------------------------------------------------------------ the worker harness
class ObsHarness(Harness):
    """serve_worker on a socketpair over one of this project's tiny models; the server side is a real Workspace
    over the same directory (so it sees the same patches and corpus.txt)."""

    def __init__(self, tmp_path, cfg=CFG_W, runner=WorkerRunner):
        self.runner_class = runner
        os.makedirs(tmp_path, exist_ok=True)
        super().__init__(tmp_path, make_runner=self._runner)
        make_ws(self.dir, cfg)
        self.ws = self.open_workspace(str(self.dir))

    def _runner(self, ws):
        runner = self.runner_class(ws)
        self.runners.append(runner)
        return runner

    def open_workspace(self, directory):
        ws = Workspace.load(directory)
        ws.store.path = str(self.source)                    # the identity the load job checks
        ws.meta["base_sha256"] = "fixture"
        return ws


class ScopedRunner(WorkerRunner):
    """Asks for admission inside the job as GGUFRunner.run does (job_scope 'forward'): a callback to the server."""

    @job_scope("forward", runner=True)
    def run(self, ids, variant="current", extra=None, names=(), hooks=None, grad=False):
        return WorkerRunner.run(self, ids, variant, extra, names, hooks, grad)


class LockingClient(FakeClient):
    """FakeClient with RuntimeClient.operation's locks: the workspace lock, then a re-entrant operation lock, and a
    nested operation only from the thread that owns the open one."""

    def __init__(self, binding):
        super().__init__(binding)
        self.operation_lock, self.owner, self.threads = threading.RLock(), None, []

    @contextmanager
    def operation(self, ws, name, **kwargs):
        with ws.lock, self.operation_lock:
            if self.operation_id is not None and self.owner != threading.get_ident():
                raise RuntimeError("Operation owner mismatch")
            outer = self.operation_id is None
            if outer:
                self.operation_id, self.owner = "op", threading.get_ident()
            self.names.append(name)
            self.ops.append((name, kwargs))
            self.threads.append(threading.current_thread().name)
            try:
                yield
            finally:
                if outer:
                    self.operation_id = self.owner = None


def connect(h, monkeypatch, client=None):
    """A RemoteRunner on the harness channel, with the server workspace in worker mode; records what crosses."""
    info = h.load()["value"]
    r = object.__new__(RemoteRunner)
    r.ws, r.lock, r.channel = h.ws, threading.RLock(), h.channel
    r.lost = r.lost_in = r.pid = None
    r.max_ctx, r.hooks, r._tok = int(info["max_ctx"]), frozenset(info["available"]), None
    h.ws._runtime_client = client or FakeClient(h.lease.binding)
    h.ws._runner = r
    monkeypatch.setattr(worker_jobs, "uses_worker", lambda ws: ws is h.ws)
    sent, received = [], []
    real_encode, real_recv = worker_jobs.encode_job, h.channel.recv

    def encode(name, arguments, extra=None):
        h.lease.write()                                     # keep the fixture lease fresh, as Harness.call does
        sent.append((name, json.loads(json.dumps(arguments))))
        return real_encode(name, arguments, extra)

    def recv(*a, **kw):
        frame = real_recv(*a, **kw)
        received.append((frame.type, frame.get("names") if frame.type == "result" else None,
                         {n: (tuple(t.shape), t.dtype, t.numel() * t.element_size()) for n, t in frame.tensors.items()}))
        return frame
    monkeypatch.setattr(worker_jobs, "encode_job", encode)
    monkeypatch.setattr(h.channel, "recv", recv)
    return r, sent, received


def window_names(ws):
    s = ws.struct
    return ([f"blocks.{l}.mlp.hook_post" for l in range(s.n_layers)] + [f"blocks.{l}.attn.hook_pattern" for l in range(s.n_layers)]
            + [f"blocks.{s.n_layers - 1}.hook_resid_post"])


# ------------------------------------------------------------------ 2. the worker path equals the in-process path
@pytest.mark.parametrize("variant", ["base", "current"])
def test_the_worker_path_gives_what_this_process_gives_one_window_at_a_time(tmp_path, monkeypatch, variant):
    h = ObsHarness(tmp_path / "w")
    here = make_ws(tmp_path / "local")
    if variant == "current":
        assert edit(h.ws).id and edit(here).id                       # committed on the server; the worker syncs it
    r, sent, received = connect(h, monkeypatch)
    at.build(h.ws, stages=["weights"], log=quiet, variant=variant)
    at.build(here, stages=["weights"], log=quiet, variant=variant)
    lines, local = [], []
    got = O.build_observed(h.ws, at.atlas_dir(h.ws), lines.append, variant=variant)
    want = O.build_observed(here, at.atlas_dir(here), local.append, variant=variant, batch=1)
    s, T, W = h.ws.struct, CFG_W["n_ctx"], got["windows"]
    assert W == len(TEXT) // T and W > 2 and got["window"] == T
    # Same files: positions, peak signs, counts and values all equal; the float32 partial sums cross unchanged and
    # are added in float64 here in the same order. The loss of a window is summed in the worker 16 positions at a
    # time (bounded float64 copies of the logits): equal to round-off, and the same after rounding to 4 places.
    assert {k: v for k, v in got.items() if k != "seconds"} == {k: v for k, v in want.items() if k != "seconds"}
    same_files(at.atlas_dir(h.ws), at.atlas_dir(here), values_only=True)
    # One job per window, admitted as a forward that holds the named hook points (the 'trace' forecast), T tokens.
    assert [n for n, _ in sent] == ["observe_window"] * W
    assert all(set(a) == set(JOBS["observe_window"].args) and a["variant"] == variant and len(a["ids"]) == T for _, a in sent)
    assert h.ws._runtime_client.ops == [("observe_window", {"tokens": T, "retained": True, "fresh": True})] * W
    # Progress: the stage's first line, then one line per window, in order.
    assert lines[0] == f"  observed stage: {W} windows of {T} tokens from corpus.txt in the workspace, one at a time in the model worker"
    assert [x.split(" (")[0].strip() for x in lines if x.strip().startswith("window ")] == [f"window {k} of {W}" for k in range(1, W + 1)]
    # Each window's result holds reductions only: no captures and no logits.
    results = [(names, t) for kind, names, t in received if kind == "result" and names is not None]
    assert len(results) == W
    L, H, F, D, V = s.n_layers, s.n_heads, CFG_W["d_mlp"], CFG_W["d_model"], CFG_W["d_vocab"]
    expect = {f"{l}.{k}" for l in range(L) for k in O.MLP_KEYS} | {f"{l}.{k}" for l in range(L) for k in O.HEAD_KEYS} | {"final"}
    per_window = L * (4 * F * 4 + F * 8 + O.K_TOP * F * 12 + O.K_BOT * F * 12) + L * 7 * H * 4 + (T - 1) * D * 4
    for names, t in results:
        assert set(names) == expect == set(t)
        assert sum(b for _, _, b in t.values()) == per_window
        assert not any(shape in ((1, T, F), (1, H, T, T), (1, T, V), (T, V), (1, T, D)) for shape, _, _ in t.values())
        assert t["final"][0] == (T - 1, D) and t["0.top_i"][1] == torch.int64 and t["0.s1"][1] == torch.float32
    assert h.ws._runtime_client.operation_id is None
    h.shutdown()


def test_the_window_loss_in_bounded_steps_agrees_with_the_whole_window_to_round_off():
    g = torch.Generator().manual_seed(0)
    logits, tk = torch.randn(1, 40, 27, generator=g) * 3, torch.randint(0, 27, (1, 40), generator=g)
    whole, steps = O.loss_sum(logits, tk), O.loss_sum(logits, tk, rows=O.LOSS_ROWS)
    assert abs(whole - steps) <= 1e-12 * abs(whole) and O.loss_sum(logits, tk, rows=100) == whole


# ------------------------------------------------------------------ 3. the job, its codec and its arguments
def test_the_window_job_encodes_decodes_and_takes_no_extra():
    row = JOBS["observe_window"]
    assert row.args == ("ids", "variant", "names") and row.result == "reduced" and not row.extra and not row.stream
    with pytest.raises(ValueError, match="does not take extra"):
        encode_job("observe_window", {"ids": [1, 2], "variant": "current", "names": []}, extra={"w": [Dl.block((range(2),), torch.ones(2))]})
    with pytest.raises(ValueError, match="does not take"):
        encode_job("observe_window", {"ids": [1, 2], "hooks": {}})
    fields, tensors = encode_job("observe_window", {"ids": [1, 2, 3], "variant": "base", "names": ["blocks.0.mlp.hook_post"]})
    assert fields == {"name": "observe_window", "args": {"ids": [1, 2, 3], "variant": "base", "names": ["blocks.0.mlp.hook_post"]}} and not tensors
    value = {"loss": repr(float("inf")), "tensors": {"0.s1": torch.arange(4.), "0.top_i": torch.arange(6).view(3, 2), "final": torch.ones(2, 3)}}
    fields, tensors = encode_result("observe_window", value)
    a, b = socket.socketpair()
    try:
        sender = threading.Thread(target=Channel(a).send, args=("result", fields, tensors), kwargs={"job": "j"})
        sender.start()
        frame = Channel(b).recv(job="j")
        sender.join(10)
        back = decode_result("observe_window", frame)
    finally:
        a.close()
        b.close()
    assert back["loss"] == "inf" and float(back["loss"]) == float("inf") and set(back["tensors"]) == set(value["tensors"])
    for n, t in value["tensors"].items():
        assert back["tensors"][n].dtype == t.dtype and torch.equal(back["tensors"][n], t)
    from wsp.worker_protocol import ProtocolError, TransferUnsupported
    a, b = socket.socketpair()                                                 # names that do not match the tensors
    try:
        for names in (["0.s1"], ["0.s1", "0.np", "x"]):
            sender = threading.Thread(target=Channel(a).send, args=("result", {"loss": "1.0", "names": names},
                                      {"0.s1": torch.ones(2), "0.np": torch.ones(2, dtype=torch.int64)}), kwargs={"job": "k"})
            sender.start()
            frame = Channel(b).recv(job="k")
            sender.join(10)
            with pytest.raises(ProtocolError, match="Malformed reduced result"):
                decode_result("observe_window", frame)
    finally:
        a.close()
        b.close()
    with pytest.raises(TransferUnsupported):                                   # only float32 and int64 cross; never cast
        encode_result("observe_window", {"loss": "1.0", "tensors": {"0.np": torch.ones(2, dtype=torch.float64)}})
    with pytest.raises(TransferUnsupported):
        encode_result("observe_window", {"loss": "1.0", "tensors": {"0.np": [1, 2]}})


@pytest.mark.parametrize("args, message", [
    ({"ids": "abc", "variant": "base", "names": ["blocks.0.mlp.hook_post"]}, "ids must be a list of integer token ids"),
    ({"ids": [1, 2.5], "variant": "base", "names": ["blocks.0.mlp.hook_post"]}, "ids must be a list of integer token ids"),
    ({"ids": [1, 2, 3], "variant": "edited", "names": ["blocks.0.mlp.hook_post"]}, "variant must be base or current"),
    ({"ids": [1, 2, 3], "variant": "base", "names": "blocks.0.mlp.hook_post"}, "names must be a list of distinct hook names"),
    ({"ids": [1, 2, 3], "variant": "base", "names": ["blocks.0.mlp.hook_post"] * 2}, "names must be a list of distinct hook names"),
    ({"ids": [1, 2, 3], "variant": "base", "names": ["blocks.0.hook_mlp_out"]}, "does not reduce 'blocks.0.hook_mlp_out'"),
    ({"ids": [1, 2, 3], "variant": "base", "names": ["blocks.9.mlp.hook_post"]}, "does not reduce 'blocks.9.mlp.hook_post'"),
    ({"ids": [1, 2, 3], "variant": "base", "names": ["blocks.0.mlp.hook_post", "blocks.0.hook_resid_post"]},
     "does not reduce 'blocks.0.hook_resid_post'"),
    ({"ids": [1, 2, 3], "variant": "base", "names": ["blocks.1.hook_resid_post"]}, "needs at least one MLP layer"),
    ({"ids": [1], "variant": "base", "names": ["blocks.0.mlp.hook_post"]}, "at least two tokens"),
    ({"ids": [1, 2, 3], "variant": "base", "names": [], "hooks": {}}, "does not take"),
])
def test_the_window_job_refuses_malformed_arguments_and_the_worker_stays_up(tmp_path, args, message):
    h = ObsHarness(tmp_path)
    h.load()
    frame, _, _ = h.call("observe_window", args, unchecked=True)
    assert frame.type == "error" and message in frame["message"], frame.header
    frame, _, _ = h.call("observe_window", {"ids": [1, 2, 3], "variant": "base", "names": ["blocks.0.mlp.hook_post"]})
    assert frame.type == "result" and set(frame["names"]) == {f"0.{k}" for k in O.MLP_KEYS}
    h.shutdown()


def test_a_runner_that_does_not_show_a_layer_is_an_error_not_a_silent_gap(tmp_path):
    class Blind(WorkerRunner):
        def run(self, ids, variant="current", extra=None, names=(), hooks=None, grad=False):
            return WorkerRunner.run(self, ids, variant, extra, names, {}, grad)            # drops the reducing hooks
    h = ObsHarness(tmp_path, runner=Blind)
    h.load()
    frame, _, _ = h.call("observe_window", {"ids": [1, 2, 3], "variant": "base", "names": ["blocks.0.mlp.hook_post"]})
    assert frame.type == "error" and "did not show the activations of MLP layers [0]" in frame["message"]
    h.shutdown()


# ------------------------------------------------------------------ 4. the GGUF runner's hook path and the final norm
def test_on_the_gguf_runner_the_window_reductions_equal_its_captures_and_the_final_stream_unembeds_to_the_logits(tmp_path, monkeypatch):
    r, model, _ = gguf_runner(tmp_path, monkeypatch)
    r.dtype = torch.float32
    model.get_output_embeddings = lambda: model.lm_head                       # what HFRunner._final asks the model for
    ws, s = r.ws, r.ws.struct

    class NotThisOne:                                                          # the job uses the runner it is given
        def __getattr__(self, name):
            raise AssertionError(f"the workspace's runner was used ({name})")
    ws._runner = NotThisOne()
    ids = [int(i) for i in torch.randint(0, 12, (10,), generator=torch.Generator().manual_seed(4))]
    names = [f"blocks.{l}.mlp.hook_post" for l in range(s.n_layers)] + [f"blocks.{l}.attn.hook_pattern" for l in range(s.n_layers)] \
        + [f"blocks.{s.n_layers - 1}.hook_resid_post"]
    asked, real_run, rows, real_loss = [], r.run, [], O.loss_sum

    def run(*a, **kw):
        asked.append(list(kw.get("names", ())))
        return real_run(*a, **kw)

    def loss(logits, tk, rows_=None):
        rows.append(rows_)
        return real_loss(logits, tk, rows_)
    r.run = run
    monkeypatch.setattr(O, "loss_sum", loss)
    got = O.observe_window(ws, r, ids, "base", names)
    del r.run
    monkeypatch.setattr(O, "loss_sum", real_loss)
    ws._runner = r                                                             # for the in-process reference below
    assert rows == [O.LOSS_ROWS]                                               # the loss in bounded float64 steps
    assert asked == [[n for n in names if not n.endswith("mlp.hook_post")]]  # MLP layers by hook only, never captured
    assert all(len(m._forward_pre_hooks) == 0 for m in model.modules())       # the reducing hooks are gone
    tk = torch.tensor([ids])
    logits, cache = r.run(tk, "base", names=names)
    last = names[-1]
    final = O.final_stream(ws, cache[last][:, 1:], "base", runner=r).reshape(-1, s.d_model)
    # The normalisation final_stream applies is the one the logits come from: unembedding it gives the logits.
    torch.testing.assert_close(model.lm_head(final), logits[0, 1:], rtol=1e-5, atol=1e-5)
    want = O.reduce_batch(ws, tk, logits, cache, list(range(s.n_layers)), list(range(s.n_layers)), last, "base")
    for l in range(s.n_layers):
        for k in O.MLP_KEYS:
            assert torch.equal(got["tensors"][f"{l}.{k}"], want["mlp"][l][k]), (l, k)
    assert torch.equal(got["tensors"]["final"], want["final"]) and torch.equal(want["final"], final)
    # This decoder gives no attention patterns: the window has none, and the stage tolerates that.
    assert not any(k.endswith(".dist") for k in got["tensors"]) and want["heads"] == {}
    assert abs(float(got["loss"]) - O.loss_sum(logits, tk)) <= 1e-12 * O.loss_sum(logits, tk)


def test_unembedding_the_final_stream_of_this_projects_model_gives_its_logits(tmp_path):
    ws = make_ws(tmp_path / "m")
    r = ws.runner
    tk = torch.tensor([ws.encode(TEXT[:40])])
    logits, cache = r.run(tk, "base", names=["blocks.1.hook_resid_post"])
    final = O.final_stream(ws, cache["blocks.1.hook_resid_post"][:, 1:], "base").reshape(-1, ws.struct.d_model)
    p = r.params("base")
    torch.testing.assert_close(final @ p["unembed.W_U"] + p["unembed.b_U"], logits[0, 1:], rtol=1e-5, atol=1e-5)


# ------------------------------------------------------------------ 5. failures, stop, files
def test_a_worker_lost_mid_stage_fails_plainly_and_keeps_the_earlier_stages(tmp_path, monkeypatch):
    h = ObsHarness(tmp_path)
    r, sent, _ = connect(h, monkeypatch)
    at.build(h.ws, stages=["weights", "observed"], log=quiet)                 # an earlier, complete observed stage
    adir = at.atlas_dir(h.ws)
    before, meta = files(adir), json.load(open(os.path.join(adir, "meta.json")))
    real, calls = O.observe_window, []

    def dying(ws, runner, ids, variant, names):
        calls.append(1)
        if len(calls) == 3:
            h.channel.sock.shutdown(socket.SHUT_RDWR)                          # the connection dies during window 3
        return real(ws, runner, ids, variant, names)
    monkeypatch.setattr(O, "observe_window", dying)
    sent.clear()
    W = len(TEXT) // CFG_W["n_ctx"]
    with pytest.raises(RuntimeError) as e:
        at.build(h.ws, stages=["observed"], log=quiet)
    assert str(e.value).startswith(f"the observed stage stopped at window 3 of {W}: ")
    assert str(e.value).endswith("Nothing of this stage was saved; the stages built before it are unchanged.")
    assert "RuntimeRefusal" not in str(e.value) and isinstance(e.value, RuntimeRefusal) and e.value.record["code"] == "worker_exited"
    assert e.value.record["message"] == str(e.value)                          # the same record, the stage's words
    assert [n for n, _ in sent] == ["observe_window"] * 3 and r.lost is not None
    assert files(adir) == before and json.load(open(os.path.join(adir, "meta.json"))) == meta     # nothing written
    A = at.Atlas.of(h.ws)
    assert A.has("observed") and "obs_peak" in A.layer(0) and A.json("validation.json")             # still usable
    assert h.ws._runtime_client.operation_id is None


def test_without_a_final_stream_no_bias_direction_is_fitted_and_an_earlier_fit_goes(tmp_path, monkeypatch):
    h = ObsHarness(tmp_path)
    r, sent, _ = connect(h, monkeypatch)
    at.build(h.ws, stages=["weights", "observed"], log=quiet)
    adir = at.atlas_dir(h.ws)
    assert os.path.exists(os.path.join(adir, "bias.json")) and os.path.exists(os.path.join(adir, "bias_direction.npy"))
    last = f"blocks.{h.ws.struct.n_layers - 1}.hook_resid_post"
    r.hooks = frozenset(n for n in r.hooks if n != last)                      # a runner that gives no final stream
    sent.clear()
    lines = []
    at.build(h.ws, stages=["observed"], log=lines.append)
    assert all(last not in a["names"] for _, a in sent)
    assert not os.path.exists(os.path.join(adir, "bias.json")) and not os.path.exists(os.path.join(adir, "bias_direction.npy"))
    assert f"  the runner exposes no final residual stream ({last}), so no bias direction is fitted" in lines
    assert json.load(open(os.path.join(adir, "validation.json")))["bias"] is None and at.Atlas.of(h.ws).has("observed")
    h.shutdown()


def test_a_window_result_without_a_layer_is_refused_by_the_server(tmp_path, monkeypatch):
    h = ObsHarness(tmp_path)
    connect(h, monkeypatch)
    at.build(h.ws, stages=["weights"], log=quiet)
    real = O.observe_window

    def partial(*a):
        out = real(*a)
        out["tensors"] = {k: v for k, v in out["tensors"].items() if not k.startswith("1.")}
        return out
    monkeypatch.setattr(O, "observe_window", partial)
    with pytest.raises(RuntimeError, match=r"reduction of window 1 lacks MLP layers \[1\]"):
        at.build(h.ws, stages=["observed"], log=quiet)
    assert not at.Atlas.of(h.ws).has("observed")
    h.shutdown()


def test_the_checks_and_unit_cards_read_files_not_the_runner(tmp_path, monkeypatch):
    h = ObsHarness(tmp_path)
    connect(h, monkeypatch)
    meta = at.build(h.ws, stages=["weights", "observed"], log=quiet)
    def no_runner(self):
        raise AssertionError("the runner was used")
    monkeypatch.setattr(Workspace, "runner", property(no_runner))
    adir = at.atlas_dir(h.ws)
    run = {k: v for k, v in meta["stages"]["observed"].items() if k != "at"}
    assert O.validate(h.ws, adir, run, json.load(open(os.path.join(adir, "bias.json")))) == json.load(open(os.path.join(adir, "validation.json")))
    unit = at.Atlas.of(h.ws).unit(0, 1)["observed"]
    assert unit["top"] and unit["peak"] in (-1, 0, 1)
    monkeypatch.undo()
    h.shutdown()


def test_a_worker_lost_before_any_observed_stage_leaves_no_observed_files(tmp_path, monkeypatch):
    h = ObsHarness(tmp_path)
    connect(h, monkeypatch)
    at.build(h.ws, stages=["weights"], log=quiet)
    adir = at.atlas_dir(h.ws)
    before = files(adir)
    def dying(*a):
        h.channel.sock.shutdown(socket.SHUT_RDWR)
        raise RuntimeError("the worker process died")
    monkeypatch.setattr(O, "observe_window", dying)
    with pytest.raises(RuntimeError, match="the observed stage stopped at window 1 of"):
        at.build(h.ws, stages=["observed"], log=quiet)
    assert files(adir) == before and not any(n.startswith("obs_") or n in OBSERVED for n in os.listdir(adir))
    assert not at.Atlas.of(h.ws).has("observed")


def test_a_stop_between_windows_ends_the_stage_and_writes_nothing(tmp_path, monkeypatch):
    """The server's log callback raises once the server is stopping; the stage logs after every window."""
    h = ObsHarness(tmp_path)
    lifecycle = types.SimpleNamespace(stopping=threading.Event())              # the app's Lifecycle.stopping
    r, sent, _ = connect(h, monkeypatch)
    at.build(h.ws, stages=["weights"], log=quiet)
    adir = at.atlas_dir(h.ws)
    before = files(adir)

    def log(message):                                                          # the server's atlas_build log, in short
        if lifecycle.stopping.is_set():
            raise RuntimeError("Server stopping; atlas build interrupted at a progress boundary.")
    real = O.observe_window

    def stopping(*a):
        if len([n for n, _ in sent]) == 2:
            lifecycle.stopping.set()                                           # the server is asked to stop in window 2
        return real(*a)
    monkeypatch.setattr(O, "observe_window", stopping)
    sent.clear()
    with pytest.raises(RuntimeError, match="Server stopping"):
        at.build(h.ws, stages=["observed"], log=log)
    assert [n for n, _ in sent] == ["observe_window"] * 2                     # no window after the stop
    assert files(adir) == before and r.lost is None
    h.shutdown()


def test_observed_files_are_written_atomically_and_recorded_only_once_all_are_written(tmp_path, monkeypatch):
    ws = make_ws(tmp_path / "m")
    at.build(ws, stages=["weights", "observed"], log=quiet)
    adir = at.atlas_dir(ws)
    assert at.Atlas.of(ws).has("observed")
    old = files(adir)
    replaced, real = [], wiring_files._replace

    def replace(src, dst):
        replaced.append((os.path.basename(src), os.path.basename(dst)))
        if os.path.basename(dst) == "obs_001.npz" and fail:
            raise OSError("no space left on device")
        real(src, dst)
    monkeypatch.setattr(wiring_files, "_replace", replace)
    fail = True
    with pytest.raises(OSError, match="no space left"):
        at.build(ws, stages=["observed"], log=quiet, tokens=120)                 # different windows: different files
    names = [d for _, d in replaced]
    assert names[:3] == ["meta.json", "probe_ids.npy", "obs_000.npz"] and names[3] == "obs_001.npz"
    assert all(s == d + ".tmp" for s, d in replaced)                           # a temporary name, then a rename
    assert not [n for n in os.listdir(adir) if n.endswith(".tmp")]             # and none left behind by the failure
    meta = json.load(open(os.path.join(adir, "meta.json")))
    assert "observed" not in meta["stages"] and "weights" in meta["stages"]   # the earlier record went first
    now = files(adir)
    assert now["obs_001.npz"] == old["obs_001.npz"] and now["obs_000.npz"] != old["obs_000.npz"]
    # Readers trust only what the record lists: no observed columns, checks or head statistics, no token-shift fit.
    A = at.Atlas.of(ws)
    assert not A.has("observed") and "obs_peak" not in A.layer(0) and "obs_peak" not in A.layer(1)
    assert A.json("validation.json") is None and A.json("heads_observed.json") is None and A.json("bias.json") is None
    assert "observed" not in A.unit(0, 1)
    monkeypatch.setattr(ws.struct, "unembed_bias", None)
    assert guide.shift_plan(ws)["mode"] is None
    # A complete build: every file through a temporary name, and the record written after the last of them.
    fail = False
    replaced.clear()
    at.build(ws, stages=["observed"], log=quiet, tokens=120)
    names = [d for _, d in replaced]
    assert names[-1] == "meta.json" and "meta.json" not in names[:-1]          # nothing earlier to drop this time
    assert set(names[:-1]) == set(OBSERVED) | {"obs_000.npz", "obs_001.npz"} and all(s == d + ".tmp" for s, d in replaced)
    A = at.Atlas.of(ws)
    assert A.has("observed") and "obs_peak" in A.layer(1) and A.json("validation.json")["tokens"] == 3 * 39
    plan = guide.shift_plan(ws)
    assert plan["mode"] == "direction" or "sd" in plan                        # the recorded fit is read again


# ------------------------------------------------------------------ 6. what runs by default, and where
def test_default_stages_stay_as_they_were(tmp_path, monkeypatch):
    here = make_ws(tmp_path / "local")
    assert set(at.build(here, log=quiet)["stages"]) == {"weights", "wiring", "observed"}           # in this process
    h = ObsHarness(tmp_path / "w")
    r, sent, _ = connect(h, monkeypatch)
    lines = []
    meta = at.build(h.ws, log=lines.append)
    assert set(meta["stages"]) == {"weights", "wiring"} and meta["skipped"] == {"observed": at.OBSERVED_NOT_BY_DEFAULT}
    assert sent == [] and "observed stage skipped: not run by default on a model that runs in the model worker" in "\n".join(lines)
    meta = at.build(h.ws, stages=["observed"], log=lines.append)               # asked for: it runs there
    assert "observed" in meta["stages"] and "skipped" not in meta and {n for n, _ in sent} == {"observe_window"}
    h.shutdown()


def test_a_worker_runner_without_mlp_activations_gets_the_existing_error(tmp_path, monkeypatch):
    h = ObsHarness(tmp_path)
    r, sent, _ = connect(h, monkeypatch)
    at.build(h.ws, stages=["weights"], log=quiet)
    r.hooks = frozenset(n for n in r.hooks if not n.endswith("mlp.hook_post"))
    with pytest.raises(ValueError, match="the runner exposes no MLP activations"):
        at.build(h.ws, stages=["observed"], log=quiet)
    assert sent == [] or all(n != "observe_window" for n, _ in sent)
    h.shutdown()


def test_a_context_too_short_for_a_window_is_refused_before_any_window_runs(tmp_path, monkeypatch):
    h = ObsHarness(tmp_path)
    r, sent, _ = connect(h, monkeypatch)
    at.build(h.ws, stages=["weights"], log=quiet)
    r.max_ctx = 1                                                              # the person's context setting, here 1
    with pytest.raises(ValueError, match="windows of 1 token cannot be observed"):
        at.build(h.ws, stages=["observed"], log=quiet)
    assert sent == [] and not at.Atlas.of(h.ws).has("observed")
    h.shutdown()


def test_the_stage_runs_under_the_build_and_workspace_locks_and_run_waits_for_it(tmp_path, monkeypatch):
    """As the browser's build job runs it: the atlas build lock and the workspace lock held for the whole build, on
    its own thread; each window's operation and the worker's admission callback nest on that thread (both locks are
    re-entrant); a Run on another thread waits for the workspace lock and runs once the build is done."""
    h = ObsHarness(tmp_path, runner=ScopedRunner)
    client = LockingClient(h.lease.binding)
    r, sent, _ = connect(h, monkeypatch, client)
    at.build(h.ws, stages=["weights"], log=quiet)
    started, done, out = threading.Event(), {}, {}

    def log(message):
        if message.strip().startswith("window 2 of"):
            started.set()
            time.sleep(0.3)                                                    # let the Run below queue up

    def build():
        try:
            with at.build_lock(h.ws), h.ws.lock:
                at.build(h.ws, stages=["observed"], log=log)
                done["build"] = time.monotonic()                               # still holding the workspace lock
        except Exception as e:                                                 # reported below
            done["error"] = e
    t = threading.Thread(target=build, name="atlas-build")
    t.start()
    assert started.wait(30)

    def run():
        out["asked"] = time.monotonic()
        out["logits"] = r.logits(h.ws.encode("abc"), "base")
        out["at"] = time.monotonic()
    u = threading.Thread(target=run, name="run")
    u.start()
    t.join(60)
    u.join(60)
    assert not t.is_alive() and not u.is_alive() and "error" not in done, done
    assert out["asked"] < done["build"] < out["at"] and tuple(out["logits"].shape) == (3, CFG_W["d_vocab"])
    W = len(TEXT) // CFG_W["n_ctx"]
    windows = [i for i, n in enumerate(client.names) if n == "observe_window"]
    assert len(windows) == W
    # The worker's admission callback ('forward', from ScopedRunner's job_scope) nests inside each window, on the
    # build's thread.
    assert client.names[:2 * W] == ["observe_window", "forward"] * W
    assert set(client.threads[:2 * W]) == {"atlas-build"}
    h.shutdown()


# ------------------------------------------------------------------ 7. review: readers, refusals, memory, adding the stage
def stop_at_window(k):
    def log(message):                                                          # the server's log once it is stopping
        if message.strip().startswith(f"window {k} of"):
            raise RuntimeError("Server stopping; atlas build interrupted at a progress boundary.")
    return log


def test_after_a_failed_forced_build_no_reader_is_older_than_the_record(tmp_path, monkeypatch):
    """A base atlas with an observed stage; then a forced build of the edited weights stops in window 2.  The reader
    the server had cached must not survive the build, and the earlier observed files never meet the new weights."""
    h = ObsHarness(tmp_path)
    edit(h.ws)                                                                 # committed now; the base map ignores it
    connect(h, monkeypatch)
    at.build(h.ws, stages=["weights", "observed"], log=quiet)
    adir = at.atlas_dir(h.ws)
    A0 = at.Atlas.of(h.ws)
    A0.layer(0)                                                                # the status route has read layer 0
    during, real = {}, at.build_weights

    def watching(ws, d, log, variant, force):
        A = at.Atlas.of(ws)
        during["record"] = (A.has("observed"), A.json("validation.json"), "obs_peak" in A.layer(1))
        return real(ws, d, log, variant=variant, force=force)
    monkeypatch.setattr(at, "build_weights", watching)
    with pytest.raises(RuntimeError, match="Server stopping"):
        at.build(h.ws, stages=["weights", "observed"], log=stop_at_window(2), variant="current")
    assert during["record"] == (False, None, False)                            # dropped before the weights stage wrote a file
    disk = json.load(open(os.path.join(adir, "meta.json")))
    A = at.Atlas.of(h.ws)
    assert A is not A0 and A.meta == disk and disk["variant"] == "current" and sorted(disk["stages"]) == ["weights"]
    assert not A.has("observed") and "obs_peak" not in A.layer(1) and A.json("validation.json") is None
    assert np.array_equal(A.layer(1)["sign_w"], np.load(os.path.join(adir, "layer_001.npz"))["sign_w"])
    assert A.status()["variant"] == "current" and A.status()["built"]
    h.shutdown()


def test_a_first_build_stopped_in_the_observed_stage_reads_as_built(tmp_path, monkeypatch):
    """The reported case: no atlas yet, the tab's status read first, then weights and observed, stopped in window 2."""
    h = ObsHarness(tmp_path)
    connect(h, monkeypatch)
    assert at.Atlas.of(h.ws).status()["built"] is False
    with pytest.raises(RuntimeError, match="Server stopping"):
        at.build(h.ws, stages=["weights", "observed"], log=stop_at_window(2))
    st = at.Atlas.of(h.ws).status()
    assert st["built"] is True and list(st["stages"]) == ["weights"]
    h.shutdown()


class RefusingClient(FakeClient):
    """Admission refuses the k-th window's operation with `record`."""

    def __init__(self, binding, k, record):
        super().__init__(binding)
        self.k, self.record, self.windows = k, record, 0

    @contextmanager
    def operation(self, ws, name, **kwargs):
        if name == "observe_window":
            self.windows += 1
            if self.windows == self.k:
                raise RuntimeRefusal(self.record)
        with FakeClient.operation(self, ws, name, **kwargs):
            yield


BUDGET = event("forecast_exceeds_budget", "refused", "The operation forecast exceeds current memory admission bounds.",
               dict(required=30e9, new_bytes=4.7e9, available=9.2e9), dict(total_cap=100e9, preferred=8e9),
               ["Free memory and retry without restarting."])


def test_an_admission_refusal_says_what_was_asked_for_and_reaches_mcp_as_a_refusal(tmp_path, monkeypatch):
    from wsp import mcp
    from wsp.session import Session
    h = ObsHarness(tmp_path)
    connect(h, monkeypatch, RefusingClient(h.lease.binding, 2, BUDGET))
    at.build(h.ws, stages=["weights"], log=quiet)
    W = len(TEXT) // CFG_W["n_ctx"]
    with pytest.raises(RuntimeRefusal) as e:
        at.build(h.ws, stages=["observed"], log=quiet)
    words = (f"the observed stage stopped at window 2 of {W}: memory admission refused it, asking for 4.7 GB more with 1.2 GB "
             "free above the reserve (the conservative forecast for a forward that keeps every layer's activations; the job "
             "itself holds far less). Close other software, lower the context setting or lower the preferred reserve in "
             "Settings, then build again; nothing of this stage was saved.")
    assert str(e.value) == words and e.value.record == {**BUDGET, "message": words}
    # Through the MCP tool: the structured refusal, not a text error.
    h.ws._runtime_client = RefusingClient(h.lease.binding, 1, BUDGET)
    sess = Session(str(tmp_path / "app"))
    sess.ws = h.ws
    text, is_error = mcp.call_tool(sess, "atlas", {"action": "build", "stages": ["observed"]})
    out = json.loads(text)
    assert is_error and out["status"] == "not_run" and out["event"]["code"] == "forecast_exceeds_budget"
    assert out["event"]["message"].startswith(f"the observed stage stopped at window 1 of {W}: memory admission refused it")
    assert out["event"]["measured"] == BUDGET["measured"] and not at.Atlas.of(h.ws).has("observed")
    h.shutdown()


def test_the_stage_words_for_each_kind_of_refusal():
    extension = event("operation_exceeds_budget", "refused", "The operation extension exceeds current admission bounds.",
                      dict(required=30e9, projected=40e9, new_bytes=4.56e9), 38e9)
    assert O.stopped_words(extension, 0, 39).startswith(
        "the observed stage stopped at window 1 of 39: memory admission refused it, asking for 4.6 GB more beyond the cap of 38.0 GB (")
    lost = event("worker_exited", "worker_killed", "The model worker exited.")
    assert O.stopped_words(lost, 2, 39) == ("the observed stage stopped at window 3 of 39: The model worker exited. Nothing of this "
                                            "stage was saved; the stages built before it are unchanged.")


def test_the_window_loss_crosses_exactly(tmp_path, monkeypatch):
    h = ObsHarness(tmp_path)
    r, _, _ = connect(h, monkeypatch)
    here = make_ws(tmp_path / "local")
    ids, names = here.encode(TEXT[:40]), window_names(here)
    got, want = r.observe_window(ids, "base", names), O.observe_window(here, here.runner, ids, "base", names)
    assert got["loss"] == want["loss"] and float(got["loss"]) != round(float(got["loss"]), 6)
    assert set(got["tensors"]) == set(want["tensors"]) and all(torch.equal(got["tensors"][k], want["tensors"][k]) for k in want["tensors"])
    h.shutdown()


def test_the_probe_plan_uses_a_loaded_runners_context(tmp_path):
    ws = make_ws(tmp_path / "m")
    assert O.probe_plan(ws)["window"] == 40                                    # no runner loaded: the model's context
    ws.runner.max_ctx = 20                                                     # the runner as loaded, a shorter context
    plan = O.probe_plan(ws)
    assert plan["window"] == 20 and plan["tokens"] == (len(TEXT) // 20) * 20 and plan["windows"] == len(TEXT) // 20


def test_the_window_job_refuses_a_hook_the_runner_does_not_expose(tmp_path):
    class NoPatterns(WorkerRunner):
        def available(self, name):
            return not name.endswith("attn.hook_pattern")
    h = ObsHarness(tmp_path, runner=NoPatterns)
    h.load()
    frame, _, _ = h.call("observe_window", {"ids": [1, 2, 3], "variant": "base", "names": ["blocks.0.mlp.hook_post", "blocks.1.attn.hook_pattern"]})
    assert frame.type == "error" and "the runner exposes no blocks.1.attn.hook_pattern" in frame["message"]
    h.shutdown()


def test_a_failed_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    ws = make_ws(tmp_path / "m")
    at.build(ws, stages=["weights"], log=quiet)
    adir = at.atlas_dir(ws)
    real = np.savez

    def broken(f, **arrays):
        if hasattr(f, "write") and os.path.basename(getattr(f, "name", "")) == "obs_001.npz.tmp":
            f.write(b"half")
            raise OSError("no space left on device")
        return real(f, **arrays)
    monkeypatch.setattr(np, "savez", broken)
    with pytest.raises(OSError, match="no space left"):
        at.build(ws, stages=["observed"], log=quiet)
    assert not [n for n in os.listdir(adir) if n.endswith(".tmp")] and not os.path.exists(os.path.join(adir, "obs_001.npz"))
    assert not at.Atlas.of(ws).has("observed")


def test_a_nan_rank_correlation_is_written_as_null_with_a_note(tmp_path, monkeypatch):
    assert math.isnan(O.spearman([1, float("nan"), 3], [1, 2, 3])) and math.isnan(O.spearman([1, 2, 3], [3, float("nan"), 1]))
    ws = make_ws(tmp_path / "m", CFG_H, corpus=None)
    meta = at.build(ws, stages=["weights", "observed"], log=quiet)
    adir = at.atlas_dir(ws)
    heads = json.load(open(os.path.join(adir, "heads.json")))
    heads[0]["same_token_z"] = float("nan")
    json.dump(heads, open(os.path.join(adir, "heads.json"), "w"))
    run = {k: v for k, v in meta["stages"]["observed"].items() if k != "at"}
    v = O.validate(ws, adir, run, None)
    assert v["mixers"]["same_token_spearman"] is None and "rank correlation is undefined" in v["mixers"]["note"]
    assert "rank correlation undefined over 12 heads" in v["checks"][-1]["text"]
    json.dumps(v, allow_nan=False)                                             # valid JSON: no bare NaN


@pytest.mark.parametrize("cap_rows", [100, 10_000])
def test_the_bias_fit_keeps_at_most_its_row_budget_the_same_way_on_both_paths(tmp_path, monkeypatch, cap_rows):
    monkeypatch.setattr(O, "FIT_ROW_BYTES", 16 * CFG_W["d_model"] * cap_rows)
    h = ObsHarness(tmp_path / "w")
    here = make_ws(tmp_path / "local")
    connect(h, monkeypatch)
    at.build(h.ws, stages=["weights"], log=quiet)
    at.build(here, stages=["weights"], log=quiet)
    lines, local = [], []
    O.build_observed(h.ws, at.atlas_dir(h.ws), lines.append)
    O.build_observed(here, at.atlas_dir(here), local.append, batch=1)
    same_files(at.atlas_dir(h.ws), at.atlas_dir(here), values_only=True)
    batched = str(tmp_path / "batched")
    shutil.copytree(at.atlas_dir(here), batched)
    O.build_observed(here, batched, quiet)                                     # 15 windows in one batch
    seen = (len(TEXT) // CFG_W["n_ctx"]) * (CFG_W["n_ctx"] - 1)
    for d in (at.atlas_dir(h.ws), batched):
        bias = json.load(open(os.path.join(d, "bias.json")))
        kept = bias["train_rows"] + bias["test_rows"]
        if cap_rows < seen:
            assert kept == cap_rows and bias["positions_seen"] == seen
        else:
            assert kept == seen and "positions_seen" not in bias
    capped = [x for x in lines if "the bias direction is fitted on the first" in x]
    assert capped == ([f"  the bias direction is fitted on the first {cap_rows:,} of {seen:,} positions, which keeps the fit near "
                       f"{(16 * cap_rows * 24 + 32 * 24 ** 2) / 1e9:.1f} GB of memory"] if cap_rows < seen else [])
    h.shutdown()


def test_adding_the_observed_stage_keeps_the_weights_stage_and_refuses_other_weights(tmp_path, monkeypatch):
    ws = make_ws(tmp_path / "m")
    edit(ws)
    at.build(ws, stages=["weights"], log=quiet, variant="current")
    adir = at.atlas_dir(ws)
    before = {n: b for n, b in files(adir).items() if n != "meta.json"}
    real = at.build_weights
    monkeypatch.setattr(at, "build_weights", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the weights stage was rebuilt")))
    meta = at.build(ws, stages=["observed"], log=quiet, variant="current")
    assert list(meta["stages"]) == ["weights", "observed"] and meta["variant"] == "current"
    after = files(adir)
    assert all(after[n] == b for n, b in before.items())                       # no weights-stage file was touched
    A = at.Atlas.of(ws)
    assert A.has("observed") and "obs_peak" in A.layer(1) and A.stale_layers() == []
    with pytest.raises(ValueError, match="this atlas maps the current weights, so its observed stage is built from the current weights"):
        at.build(ws, stages=["observed"], log=quiet, variant="base")
    edit(ws, seed=9)                                                           # the edited weights change
    with pytest.raises(ValueError, match="the weights this atlas was built from have changed since .*rebuild the atlas"):
        at.build(ws, stages=["observed"], log=quiet, variant="current")
    assert files(adir) == after
    # A base atlas maps the base weights, which edits never change: adding the stage after an edit is fine.
    monkeypatch.setattr(at, "build_weights", real)
    base = make_ws(tmp_path / "b")
    at.build(base, stages=["weights"], log=quiet)
    edit(base)
    assert at.Atlas.of(base).stale_layers() == [1]
    monkeypatch.setattr(at, "build_weights", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the weights stage was rebuilt")))
    built_at = json.load(open(os.path.join(at.atlas_dir(base), "meta.json")))["version_at_build"]
    added = at.build(base, stages=["observed"], log=quiet)
    assert "observed" in added["stages"] and added["version_at_build"] == built_at != base.version   # the record is the build's


def test_the_build_route_adds_the_observed_stage_to_an_existing_atlas(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    from wsp.session import Session
    make_ws(tmp_path / "rnd")
    sess = Session(str(tmp_path))
    sess.open("rnd")
    c = TestClient(create_app(sess, "http://test"))
    at.build(sess.ws, stages=["weights"], log=quiet)
    monkeypatch.setattr(at, "build_weights", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the weights stage was rebuilt")))
    st = c.get("/api/atlas").json()
    assert st["built"] and "observed" not in st["stages"] and st["observed_unavailable"] is None and st["validation"] is None
    scope = {k: c.get("/api/status").json()[k] for k in ("workspace_id", "version")}
    assert c.post("/api/atlas/build", json={**scope, "stages": ["observed"], "force": False, "variant": st["variant"], "fetch": True}).status_code == 200
    for _ in range(200):
        st = c.get("/api/atlas").json()
        if not st["job"]["running"]:
            break
        time.sleep(0.05)
    assert st["job"]["error"] is None and "observed" in st["stages"] and st["validation"]["checks"]
    assert c.get("/api/atlas/layer", params={"l": 0}).json()["observed"]


def test_the_build_route_keeps_a_refusals_words_and_record(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    from wsp.session import Session
    make_ws(tmp_path / "rnd")
    sess = Session(str(tmp_path))
    sess.open("rnd")
    c = TestClient(create_app(sess, "http://test"))
    words = O.stopped_words(BUDGET, 0, 3)

    def refused(ws, **kwargs):
        raise RuntimeRefusal({**BUDGET, "message": words})
    monkeypatch.setattr(at, "build", refused)
    scope = {k: c.get("/api/status").json()[k] for k in ("workspace_id", "version")}
    assert c.post("/api/atlas/build", json={**scope, "stages": ["observed"], "force": False, "variant": "base", "fetch": True}).status_code == 200
    for _ in range(200):
        job = c.get("/api/atlas").json()["job"]
        if not job["running"]:
            break
        time.sleep(0.05)
    assert job["error"] == words[:1].upper() + words[1:] and "RuntimeRefusal" not in job["error"]
    assert job["event"]["code"] == "forecast_exceeds_budget" and job["event"]["message"] == words



def test_no_reader_cached_during_a_build_outlives_the_record_it_read(tmp_path, monkeypatch):
    """A reader that looks during the weights stage (the status route does) must not survive the record written after it,
    nor a build that then stops before writing another one."""
    ws = make_ws(tmp_path / "m")
    edit(ws)
    at.build(ws, stages=["weights"], log=quiet)                                 # a base atlas
    adir = at.atlas_dir(ws)
    real, seen = at.build_weights, {}

    def reading(ws_, d, log, variant, force):
        A = at.Atlas.of(ws_)                                                       # as the status route reads it
        seen["old"], seen["before"] = A.layer(1)["g_norm"].copy(), A.status()["variant"]   # before the files change
        out = real(ws_, d, log, variant=variant, force=force)
        if seen.get("stop"):
            raise RuntimeError("Server stopping; atlas build interrupted at a progress boundary.")
        return out
    monkeypatch.setattr(at, "build_weights", reading)

    def log(message):                                                          # a reader during the first observed window
        if "windows 0" in message:
            seen["during"] = at.Atlas.of(ws).meta.get("variant")
    at.build(ws, stages=["weights", "observed"], log=log, variant="current")
    assert seen["before"] == "base" and seen["during"] == "current"            # the record the weights stage wrote
    seen["stop"] = True
    edit(ws, seed=9)
    with pytest.raises(RuntimeError, match="Server stopping"):
        at.build(ws, stages=["weights"], log=quiet, variant="current", force=True)
    disk = np.load(os.path.join(adir, "layer_001.npz"))["g_norm"]
    assert not np.array_equal(seen["old"], disk)                                # the files did change under the reader
    assert np.array_equal(at.Atlas.of(ws).layer(1)["g_norm"], disk)
