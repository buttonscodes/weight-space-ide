"""The drift measurement of a commit, one batch of sequences at a time.

Found on a model with a vocabulary of 248,320 tokens run on a GPU: the measurement held every sequence's logits for
the base and the edited model, both log-softmaxes and the element-wise KL at once, several GiB on every commit. The
numbers must stay exactly what the whole-tensor formula gives.
"""
import torch
import torch.nn.functional as F

from wsp import testsuite as T
from wsp.analysis import corpus

from test_steering import make_ws


def reference(ws, variant, extra, n=24):
    """The measurement as it was first written: every sequence's logits at once."""
    seqs = corpus(ws)[:n]
    L = min(len(q) for q in seqs)
    toks = torch.tensor([q[:L] for q in seqs])
    r = ws.runner
    a = torch.cat([r.run(toks[i:i + 8], "base")[0] for i in range(0, len(toks), 8)])
    b = torch.cat([r.run(toks[i:i + 8], variant, extra)[0] for i in range(0, len(toks), 8)])
    kl = F.kl_div(b.log_softmax(-1), a.log_softmax(-1), log_target=True, reduction="none").sum(-1)
    return {"kl": round(float(kl.mean()), 5), "kl_max": round(float(kl.max()), 4),
            "top1_agree": round(float((a.argmax(-1) == b.argmax(-1)).float().mean()), 4), "n": int(toks.numel())}


def an_edit(ws, seed, scale):
    name = next(n for n in ws.names() if len(ws.shape(n)) == 2 and "mlp" in n)
    g = torch.Generator().manual_seed(seed)
    return {name: [ws.ref(name).piece(scale * torch.randn(ws.shape(name), generator=g))]}


def test_the_drift_is_measured_one_batch_at_a_time_with_the_same_numbers(tmp_path, monkeypatch):
    ws = make_ws(tmp_path / "rnd")
    extra = an_edit(ws, 0, 0.5)
    want = reference(ws, "current", extra)
    assert want["n"] == 24 * 16 and want["kl"] > 0.01 and want["top1_agree"] < 1.0      # the edit matters; 3 batches

    rows, runs = [], []
    real_kl, real_run = F.kl_div, ws.runner.run

    def kl_div(inp, target, **kw):
        rows.append((tuple(inp.shape), tuple(target.shape)))
        return real_kl(inp, target, **kw)

    def run(ids, variant="current", extra=None, **kw):
        runs.append((variant, len(ids)))
        return real_run(ids, variant, extra, **kw)

    monkeypatch.setattr(T.F, "kl_div", kl_div)
    monkeypatch.setattr(ws.runner, "run", run)
    assert T.drift(ws, "current", extra) == want
    assert rows == [((16, 26), (16, 26))] * 24                    # one sequence's logits in a KL, never all of them
    assert runs == [("base", 8)] * 3 + [("current", 8)] * 3
    la, top, amax = ws.keep["drift_base"]
    assert tuple(la.shape) == (24, 16, 26) and tuple(top.shape) == (24, 16) and amax > 0

    rows.clear(), runs.clear()
    other = an_edit(ws, 1, 0.25)
    got = T.drift(ws, "current", other)
    assert runs == [("current", 8)] * 3 and len(rows) == 24       # the base is measured once
    monkeypatch.undo()
    assert got == reference(ws, "current", other) and got != want
    assert T.drift(ws) == {"kl": 0.0, "kl_max": 0.0, "top1_agree": 1.0, "n": 24 * 16}    # no edit: the base itself


def test_a_corpus_that_is_not_a_multiple_of_the_batch(tmp_path):
    ws = make_ws(tmp_path / "rnd")
    extra = an_edit(ws, 2, 0.5)
    for n in (1, 7, 9, 20):
        ws.keep.pop("drift_base", None)
        assert T.drift(ws, "current", extra, n=n) == reference(ws, "current", extra, n=n)
        assert ws.keep["drift_base"][0].shape[0] == n


def test_logits_that_are_not_numbers_are_not_taken_for_a_blank_model(tmp_path, monkeypatch):
    """The whole-tensor formula reported NaN for a base model whose logits hold a NaN; a running maximum taken with
    Python's max would have dropped it and answered "blank base model, nothing to preserve"."""
    ws = make_ws(tmp_path / "rnd")
    corpus(ws)                                                    # sampled from the base model once, before the patch
    real_run = ws.runner.run
    seen = []

    def run(ids, variant="current", extra=None, **kw):
        out, cache = real_run(ids, variant, extra, **kw)
        seen.append(variant)
        if variant == "base" and len(seen) == 2:                  # the second batch of the base
            out = out.clone()
            out[3, 5, 7] = float("nan")
        return out, cache

    monkeypatch.setattr(ws.runner, "run", run)
    got = T.drift(ws, "current", an_edit(ws, 0, 0.5))
    assert got["n"] == 24 * 16 and got["kl"] != got["kl"] and got["kl_max"] != got["kl_max"]     # NaN, not 0.0
    assert ws.keep["drift_base"][2] != ws.keep["drift_base"][2]


def test_the_device_cache_is_released_after_the_measurement(tmp_path, monkeypatch):
    ws = make_ws(tmp_path / "rnd")
    calls = []
    monkeypatch.setattr(ws.runner, "release_device_cache", lambda: calls.append("released"), raising=False)
    T.drift(ws, "current", an_edit(ws, 0, 0.5))
    assert calls == ["released"]


def test_releasing_the_device_cache_names_the_device(monkeypatch):
    import types

    from wsp.runner import HFRunner
    done = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: done.append("mps"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: done.append("cuda"))
    for kind in ("cpu", "mps", "cuda", None):
        r = HFRunner.__new__(HFRunner)
        if kind is not None:
            r.device = types.SimpleNamespace(type=kind)
        r.release_device_cache()
    assert done == ["mps", "cuda"]
