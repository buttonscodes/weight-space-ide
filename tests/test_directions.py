"""Behaviour directions (wsp.directions): measurement, directions, the three ablation variants, the routes and the
worker job, on tiny random models (Config + init_params, no training) and fake runners.

Every expectation is computed independently of the code under test: means from `wsp.model.forward` over the
runner's parameters (the final norm written out here), the clamp with numpy, directions with plain tensor
arithmetic, and each variant as a dense reference built from its definition (normalise the columns, ablate,
renormalise, restore the norms), compared through the weights the script leaves.  No checkpoint is run.
"""
import hashlib
import json
import math
import os

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import load_file

import wsp.runner
from wsp import directions as DR
from wsp import probes as P
from wsp import steering as S
from wsp import workspace as Wmod
from wsp.model import Config, forward, init_params
from wsp.server import create_app
from wsp.session import Session
from wsp.worker_jobs import JOBS
from wsp.wscript import FUNCTION_DOC, run_script

from test_steering import CFG, make_ws

GOOD = ["abc", "abd", "hello", "zzz", "quiet", "fox"]          # six: more than one chunk of four
BAD = ["xyz", "qqq", "world", "mmmm", "bad"]
N, D = CFG["n_layers"], CFG["d_model"]


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


# ------------------------------------------------------------------ independent references
def ln_final(p, x, eps=1e-5):
    xc = x - x.mean(-1, keepdim=True)
    return xc / (xc.pow(2).mean(-1, keepdim=True) + eps).sqrt() * p["ln_final.w"] + p["ln_final.b"]


def vectors(ws, ids, variant="base"):
    """[n + 1, d] for one prompt: the last token's resid_pre per layer, then the final norm of the last resid_post."""
    p = ws.runner.params(variant)
    names = [f"blocks.{l}.hook_resid_pre" for l in range(N)] + [f"blocks.{N - 1}.hook_resid_post"]
    cache = {}
    forward(p, ws.cfg, torch.tensor(ids), cache=cache, names=names)
    rows = [cache[n][0, -1] for n in names[:-1]] + [ln_final(p, cache[names[-1]][0, -1])]
    return torch.stack(rows)


def clamp(row, q):
    a = row.numpy().astype(np.float64)
    t = np.quantile(np.abs(row.numpy()), q)
    return torch.from_numpy(np.clip(a, -t, t))


def means(ws, texts, variant="base", q=1.0):
    total = torch.zeros(N + 1, D, dtype=torch.float64)
    for t in texts:
        v = vectors(ws, ws.encode(t), variant)
        total += torch.stack([clamp(r, q) for r in v]) if q < 1 else v.double()
    return (total / len(texts)).float()


def unit(x):
    return x / x.norm()


def ref_direction(good, bad, k, ortho=True):
    r = unit(bad[k].double() - good[k].double())
    if ortho:
        g = unit(good[k].double())
        r = unit(r - (r @ g) * g)
    return r


def component(ws, l, comp, variant="base", extra=None):
    """The component as one matrix [D_in, d] (rows of every head), and how to put a matrix back in the tensor."""
    if comp == "attn.o_proj":
        name = f"blocks.{l}.attn.W_O"
        t = ws.read(name, (), variant, extra)
        return name, t.reshape(-1, t.shape[-1]).double(), lambda m: m.reshape(t.shape)
    name = f"blocks.{l}.mlp.W_out"
    return name, ws.read(name, (), variant, extra).double(), lambda m: m


def plain_ref(V, v, w):
    return V - w * torch.outer(V @ v, v)


def pre_ref(V, v, w):
    n = V.norm(dim=0)
    safe = torch.where(n > 0, n, torch.ones_like(n))
    Wn = V / safe
    Wn = Wn - w * torch.outer(Wn @ v, v)
    return Wn * n


def full_exact(V, v, w):
    """Normalise columns, ablate, renormalise each column, restore the original norms -> the difference."""
    n = V.norm(dim=0)
    safe = torch.where(n > 0, n, torch.ones_like(n))
    Wn = V / safe
    Wn = Wn - w * torch.outer(Wn @ v, v)
    m = Wn.norm(dim=0)
    Wn = Wn / torch.where(m > 0, m, torch.ones_like(m))
    new = torch.where(n > 0, Wn * n, V)
    return new - V


def best_rank(M, r):
    U, Sv, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U[:, :r] * Sv[:r]) @ Vh[:r]


def distributions():
    return {"attn.o_proj": {"max_weight": 1.0, "max_weight_position": 0.0, "min_weight": 0.5, "min_weight_distance": 2.0},
            "mlp.down_proj": {"max_weight": 0.8, "max_weight_position": 1.0, "min_weight": 0.2, "min_weight_distance": 1.0}}


WEIGHTS = {(0, "attn.o_proj"): 1.0, (1, "attn.o_proj"): 0.75, (0, "mlp.down_proj"): 0.2, (1, "mlp.down_proj"): 0.8}


def saved_means(ws, name):
    t = load_file(os.path.join(ws.dir, "directions", name + ".safetensors"))
    return t["good_mean"], t["bad_mean"]


def save_random(ws, name, n_layers=N, d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    good, bad = torch.randn(n_layers + 1, d, generator=g), torch.randn(n_layers + 1, d, generator=g)
    meta = {"version": 1, "name": name, "created": 0, "variant": "base", "n_layers": n_layers, "d_model": d,
            "counts": {"good": 1, "bad": 1}, "tokens": {"good": 1, "bad": 1}, "options": {}, "prompts_sha256": {}, "model": {}}
    DR.save(ws, name, good, bad, meta, {"good": ["a"], "bad": ["b"]})
    return good, bad


# ------------------------------------------------------------------ 1. measurement
def test_measurement_equals_an_independent_forward_pass(ws, monkeypatch):
    out = DR.measure(ws, "plain", GOOD, BAD)
    good, bad = saved_means(ws, "plain")
    assert good.shape == (N + 1, D) and good.dtype == torch.float32
    assert torch.allclose(good, means(ws, GOOD), atol=1e-5) and torch.allclose(bad, means(ws, BAD), atol=1e-5)
    # Index n is after the final norm (not the raw resid_post), index 0 is the embedding output entering layer 0.
    last = vectors(ws, ws.encode("abc"))
    p = ws.runner.params("base")
    assert torch.allclose(last[0], p["embed.W_E"][ws.encode("abc")[-1]] + p["pos_embed.W_pos"][2], atol=1e-6)
    # Winsorized: each prompt and index clamped to the quantile of its own absolute values before the sum.
    DR.measure(ws, "wins", GOOD, BAD, winsorization_quantile=0.7)
    wg, wb = saved_means(ws, "wins")
    assert torch.allclose(wg, means(ws, GOOD, q=0.7), atol=1e-5) and torch.allclose(wb, means(ws, BAD, q=0.7), atol=1e-5)
    assert not torch.allclose(wg, good, atol=1e-3)
    # Another chunk size gives the same means, bit for bit.
    monkeypatch.setattr(DR, "CHUNK", 3)
    DR.measure(ws, "plain", GOOD, BAD, replace=True)
    assert torch.equal(saved_means(ws, "plain")[0], good) and torch.equal(saved_means(ws, "plain")[1], bad)
    monkeypatch.setattr(DR, "CHUNK", 1)
    DR.measure(ws, "wins", GOOD, BAD, winsorization_quantile=0.7, replace=True)
    assert torch.equal(saved_means(ws, "wins")[0], wg)
    # The stored files round-trip; counts, token totals and hashes.
    meta, g2, b2 = DR.load(ws, "plain")
    assert torch.equal(g2, good) and torch.equal(b2, bad)
    assert DR.prompts(ws, "plain") == {"good": GOOD, "bad": BAD}
    assert meta["counts"] == {"good": 6, "bad": 5}
    assert meta["tokens"] == {"good": sum(len(t) for t in GOOD), "bad": sum(len(t) for t in BAD)}   # one token per character
    sha = lambda xs: hashlib.sha256(json.dumps(xs, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()  # noqa: E731
    assert meta["prompts_sha256"] == {"good": sha(GOOD), "bad": sha(BAD)}
    assert meta["model"] == S.model_identity(ws) and meta["variant"] == "base" and meta["version"] == 1
    assert meta["n_layers"] == N and meta["d_model"] == D and isinstance(meta["created"], int)
    assert meta["options"] == {"chat": False, "system": None, "prefix": None, "chat_options": None, "winsorization_quantile": 1.0}
    with open(os.path.join(ws.dir, "directions", "plain.json")) as f:
        assert json.load(f) == meta
    # Statistics from the two means only.
    st = out["stats"]
    assert len(st) == N + 1 and [s["index"] for s in st] == list(range(N + 1))
    for k, s in enumerate(st):
        assert s["good_norm"] == pytest.approx(float(good[k].double().norm()), rel=1e-6)
        assert s["diff_norm"] == pytest.approx(float((bad[k].double() - good[k].double()).norm()), rel=1e-6)
        assert s["cosine"] == pytest.approx(float(torch.nn.functional.cosine_similarity(good[k].double(), bad[k].double(), 0)), rel=1e-6)
        assert s["usable"]
    assert not os.listdir(os.path.join(ws.dir, "directions")) or all(not f.endswith(".tmp") for f in os.listdir(os.path.join(ws.dir, "directions")))


def test_the_current_variant_measures_the_edited_model(sess):
    ws = sess.ws
    assert sess.apply("mlp(0).W_out *= 3", dry_run=False)["committed"]
    DR.measure(ws, "cur", GOOD, BAD, variant="current")
    DR.measure(ws, "base", GOOD, BAD, variant="base")
    cg, cb = saved_means(ws, "cur")
    bg, bb = saved_means(ws, "base")
    assert torch.allclose(cg, means(ws, GOOD, "current"), atol=1e-5) and torch.allclose(cb, means(ws, BAD, "current"), atol=1e-5)
    assert torch.allclose(bg, means(ws, GOOD, "base"), atol=1e-5)
    # Entering layer 0 is the embedding, which the edit did not touch; everything after the edited MLP differs.
    assert torch.equal(cg[0], bg[0]) and not torch.allclose(cg[1], bg[1], atol=1e-4) and not torch.allclose(cg[2], bg[2], atol=1e-4)
    assert DR.load(ws, "cur")[0]["variant"] == "current"


def test_a_stop_between_chunks_saves_nothing_and_progress_follows_each_chunk(ws):
    seen = []
    out = DR.measure(ws, "s", GOOD, BAD, progress=lambda d, t: seen.append((d, t)), stopped=lambda: len(seen) >= 1)
    assert out == {"stopped": True, "name": "s", "done": 4, "total": 11} and seen == [(4, 11)]
    assert not DR.exists(ws, "s") and DR.entries(ws) == []
    seen.clear()
    DR.measure(ws, "s", GOOD, BAD, progress=lambda d, t: seen.append((d, t)))
    assert seen == [(4, 11), (6, 11), (10, 11), (11, 11)]          # good in chunks of 4, then bad


def test_measurement_bounds_and_names(ws):
    for kw, message in [
        (dict(name="bad name"), "name must be"), (dict(name=".x"), "name must be"), (dict(name="a" * 65), "name must be"),
        (dict(name="x.prompts"), "must not end in '.prompts'"), (dict(good=[]), "good must list 1..1024"),
        (dict(bad=["a"] * 1025), "bad must list 1..1024"), (dict(good=[""]), "1..20000 characters"),
        (dict(good=["a" * 20001]), "1..20000 characters"), (dict(good=[7]), "must be text"),
        (dict(good=["a" * 17]), "has 17 tokens; it must have 1..16"), (dict(good=["ab1"]), "good prompt 1"),
        (dict(winsorization_quantile=1.5), "winsorization_quantile"), (dict(winsorization_quantile=True), "winsorization_quantile"),
        (dict(winsorization_quantile=float("nan")), "winsorization_quantile"), (dict(chat="yes"), "chat must be"),
        (dict(chat=True), "needs the model's own chat template: this built-in model has no chat template"),
        (dict(system="be nice"), "system and prefix apply"), (dict(prefix="ok"), "system and prefix apply"),
        (dict(chat_options={"enable_thinking": False}), "chat_options apply"), (dict(variant="delta"), "variant must be"),
        (dict(replace="no"), "replace must be"),
    ]:
        args = {"name": "ok", "good": ["abc"], "bad": ["xyz"], **kw}
        with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)").replace(".", r"\.")):
            DR.measure(ws, **args)
    save_random(ws, "taken")
    with pytest.raises(DR.DirectionError, match="exists; pass replace"):
        DR.measure(ws, "taken", ["abc"], ["xyz"])
    assert DR.entries(ws)[0]["name"] == "taken"


def test_a_damaged_entry_is_listed_with_its_error(ws):
    save_random(ws, "fine")
    d = os.path.join(ws.dir, "directions")
    with open(os.path.join(d, "broken.json"), "w") as f:
        f.write("{not json")
    save_random(ws, "novec")
    os.remove(os.path.join(d, "novec.safetensors"))
    with open(os.path.join(d, "half.safetensors"), "wb") as f:             # no .json: incomplete, not listed
        f.write(b"x")
    listed = {e["name"]: e for e in DR.entries(ws)}
    assert set(listed) == {"broken", "fine", "novec"}
    assert "error" in listed["broken"] and "cannot be read" in listed["broken"]["error"]
    assert "vectors file is missing" in listed["novec"]["error"] and "error" not in listed["fine"]
    with pytest.raises(DR.DirectionMissing):
        DR.load(ws, "half")
    DR.delete(ws, "fine")
    assert not any(f.startswith("fine.") for f in os.listdir(d))
    with pytest.raises(DR.DirectionMissing):
        DR.delete(ws, "fine")


# ------------------------------------------------------------------ 2-4. directions and strengths
def test_the_table_and_unusable_rows():
    g = torch.Generator().manual_seed(4)
    good, bad = torch.randn(5, 8, generator=g), torch.randn(5, 8, generator=g)
    bad[1] = good[1]                                   # no difference
    good[3], bad[3] = torch.zeros(8), torch.zeros(8)   # a difference exactly along the good mean: nothing left after projection
    good[3, 2], bad[3, 2] = 2.0, 5.0
    proj, plain = DR.table(good, bad, True), DR.table(good, bad, False)
    for k in (0, 2, 4):
        assert float(proj[k].double().norm()) == pytest.approx(1, abs=1e-6)
        assert abs(float(proj[k].double() @ unit(good[k].double()))) < 1e-6
        assert torch.allclose(proj[k].double(), ref_direction(good, bad, k), atol=1e-6)
        assert torch.allclose(plain[k].double(), unit(bad[k].double() - good[k].double()), atol=1e-6)
    assert torch.count_nonzero(proj[1]) == 0 and torch.count_nonzero(plain[1]) == 0
    assert float(plain[3].double().norm()) == pytest.approx(1, abs=1e-6)
    assert float(proj[3].double().norm()) < 1e-6 and DR.usable(proj) == [True, False, True, False, True]
    assert [s["usable"] for s in DR.stats(good, bad)] == [True, False, True, False, True]
    assert [s["usable"] for s in DR.stats(good, bad, False)] == [True, False, True, True, True]
    with pytest.raises(DR.DirectionError, match="index 1 of the direction .used by layer 0. is unusable"):
        DR.direction_at(proj, None, 0)
    with pytest.raises(DR.DirectionError, match="index 3 .* is unusable"):
        DR.direction_at(proj, 1.5, 0)                  # between 2 and 3


def test_direction_at_per_layer_interpolated_and_refused():
    g = torch.Generator().manual_seed(5)
    tab = DR.table(torch.randn(4, 6, generator=g), torch.randn(4, 6, generator=g))      # n = 3 layers
    for l in range(3):
        assert torch.equal(DR.direction_at(tab, None, l), tab[l + 1])
    # x = 0.3: p = 1.3, between rows 1 and 2 with f = 0.3 (computed here with exact fractions of the row weights).
    want = unit(0.7 * tab[1].double() + 0.3 * tab[2].double())
    assert torch.allclose(DR.direction_at(tab, 0.3, 0).double(), want, atol=1e-6)
    assert torch.equal(DR.direction_at(tab, 1.0, 2), tab[2]) and torch.equal(DR.direction_at(tab, -1, 0), tab[0])
    assert torch.equal(DR.direction_at(tab, 2, 0), tab[3])                              # the last index, f = 0
    assert torch.allclose(DR.direction_at(tab, 1.75, 1).double(), unit(0.25 * tab[2].double() + 0.75 * tab[3].double()), atol=1e-6)
    for bad in (-1.5, -2, 2.5, 3):
        with pytest.raises(DR.DirectionError, match="outside the measurement"):
            DR.direction_at(tab, bad, 0)
    opposite = torch.stack([tab[0], tab[1], -tab[1], tab[3]])
    with pytest.raises(DR.DirectionError, match="cancel"):
        DR.direction_at(opposite, 0.5, 0)


def test_layer_strength_centre_endpoints_beyond_and_zero():
    d = {"max_weight": 2.0, "max_weight_position": 3.0, "min_weight": 0.5, "min_weight_distance": 2.0}
    assert DR.layer_weight(d, 3) == (2.0, None)
    assert DR.layer_weight(d, 1) == (0.5, None) and DR.layer_weight(d, 5) == (0.5, None)
    assert DR.layer_weight(d, 4) == (pytest.approx(1.25), None)
    w, why = DR.layer_weight(d, 6)
    assert w is None and "farther than min_weight_distance" in why
    w, why = DR.layer_weight({**d, "min_weight": -2.0}, 4)                 # 2 + 0.5 * (-4) = 0
    assert w is None and why == "weight 0"
    assert DR.layer_weight({**d, "max_weight_position": 2.5}, 4) == (pytest.approx(2.0 + 0.75 * -1.5), None)


def test_parameters_are_checked_and_each_error_names_its_field(ws):
    save_random(ws, "r")
    base = {"direction": "r", "weight_distributions": {"attn.o_proj": distributions()["attn.o_proj"]}}
    dist = base["weight_distributions"]["attn.o_proj"]
    for change, field in [
        ({"extra": 1}, "extra: unknown field"), ({"direction": "no such"}, "direction: name must be"),
        ({"direction_index": "1"}, "direction_index: must be a finite number"), ({"direction_index": True}, "direction_index"),
        ({"direction_index": 2.5}, "direction_index 2.5 is outside"),
        ({"weight_distributions": {}}, "weight_distributions: must be an object"),
        ({"weight_distributions": {"attn.q_proj": dist}}, "weight_distributions.attn.q_proj: unknown component"),
        ({"weight_distributions": {"attn.o_proj": {**dist, "x": 1}}}, "weight_distributions.attn.o_proj.x: unknown field"),
        ({"weight_distributions": {"attn.o_proj": {k: v for k, v in dist.items() if k != "min_weight"}}},
         "weight_distributions.attn.o_proj.min_weight: required"),
        ({"weight_distributions": {"attn.o_proj": {**dist, "max_weight": 8.5}}}, "weight_distributions.attn.o_proj.max_weight"),
        ({"weight_distributions": {"attn.o_proj": {**dist, "min_weight": float("inf")}}}, "weight_distributions.attn.o_proj.min_weight"),
        ({"weight_distributions": {"attn.o_proj": {**dist, "min_weight_distance": 0}}}, "min_weight_distance: must be above 0"),
        ({"weight_distributions": {"attn.o_proj": {**dist, "max_weight_position": 4.5}}}, "max_weight_position: must be from -2 to 4"),
        ({"orthogonalize_direction": 1}, "orthogonalize_direction: must be"), ({"row_normalization": "post"}, "row_normalization: must be"),
        ({"full_normalization_lora_rank": 0}, "full_normalization_lora_rank"), ({"full_normalization_lora_rank": 65}, "full_normalization_lora_rank"),
        ({"full_normalization_lora_rank": 2.0}, "full_normalization_lora_rank"),
    ]:
        with pytest.raises(DR.DirectionError, match=field.replace(".", r"\.")):
            DR.plan(ws, {**base, **change})
    with pytest.raises(DR.DirectionMissing):
        DR.plan(ws, {**base, "direction": "missing"})


def test_the_plan_and_the_script_line(ws):
    save_random(ws, "r")
    params = {"direction": "r", "direction_index": 0.25, "weight_distributions": distributions(), "row_normalization": "full",
              "full_normalization_lora_rank": 2}
    plan = DR.plan(ws, params)
    rows = {(r["layer"], r["component"]): r for r in plan["rows"]}
    assert set(rows) == set(WEIGHTS)
    for key, w in WEIGHTS.items():
        assert rows[key]["weight"] == pytest.approx(w) and rows[key]["skipped"] is None
        assert rows[key]["direction"] == "between 1 and 2 (0.25)"
    assert rows[(0, "attn.o_proj")]["views"] == 2 and rows[(0, "mlp.down_proj")]["views"] == 1
    dh = D // CFG["n_heads"]
    assert rows[(0, "attn.o_proj")]["stored"] == 2 * (dh * 2 + 2 * D) and rows[(1, "mlp.down_proj")]["stored"] == 40 * 2 + 2 * D
    assert plan["totals"] == {"layers": 2, "components": 4, "views": 6, "stored": sum(r["stored"] for r in plan["rows"]), "skipped": 0}
    narrow = {**params, "direction_index": None,
              "weight_distributions": {"mlp.down_proj": {"max_weight": 1, "max_weight_position": 0, "min_weight": 1, "min_weight_distance": 0.5}}}
    p2 = DR.plan(ws, narrow)
    assert [r["direction"] for r in p2["rows"]] == ["index 1", "index 2"]
    assert p2["rows"][1]["skipped"].startswith("farther") and p2["rows"][1]["weight"] is None and p2["totals"]["skipped"] == 1
    line = DR.script_for(params)
    assert "\n" not in line and line.startswith('ablate_direction("r", direction_index=0.25, ')
    third = {**params, "direction_index": 1 / 3, "weight_distributions": {"attn.o_proj": {**distributions()["attn.o_proj"], "min_weight": 0.1 + 0.2}}}
    import ast
    call = ast.parse(DR.script_for(third)).body[0].value
    kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    assert kw["direction_index"] == 1 / 3 and kw["weight_distributions"]["attn.o_proj"]["min_weight"] == 0.1 + 0.2   # exact round trip
    assert kw["row_normalization"] == "full" and kw["full_normalization_lora_rank"] == 2 and kw["orthogonalize_direction"] is True


def test_a_component_the_model_lacks_is_refused_and_a_layer_without_it_is_skipped(tmp_path):
    cfg = Config(**{**CFG, "attn_only": True, "d_mlp": 0})
    from wsp.tokenizer import VocabTokenizer
    from wsp.workspace import Workspace
    w = Workspace.create(str(tmp_path / "ao"), cfg, VocabTokenizer(list("abcdefghijklmnopqrstuvwxyz"), "char"), init_params(cfg, seed=1))
    save_random(w, "r")
    with pytest.raises(DR.DirectionError, match="weight_distributions.mlp.down_proj: this model has no mlp.down_proj in any layer"):
        DR.plan(w, {"direction": "r", "weight_distributions": distributions()})
    # A layer that lacks a component is skipped and reported (here: the structure of layer 1 says it has no MLP).
    ws = make_ws(tmp_path / "rnd")
    save_random(ws, "r")
    ws.struct.layers[1].mlp = {}
    plan = DR.plan(ws, {"direction": "r", "weight_distributions": distributions()})
    row = next(r for r in plan["rows"] if r["layer"] == 1 and r["component"] == "mlp.down_proj")
    assert row["skipped"] == "this layer has no mlp.down_proj" and row["weight"] is None and plan["totals"]["skipped"] == 1


# ------------------------------------------------------------------ 5-8. the three variants
def run(ws, params):
    return run_script(ws, DR.script_for(params))


def check_against(ws, deltas, ref, weights=WEIGHTS, v_of=None, good=None, bad=None):
    targeted = set()
    for (l, comp), w in weights.items():
        name, V, back = component(ws, l, comp)
        v = v_of(l) if v_of else ref_direction(good, bad, l + 1)
        _, got, _ = component(ws, l, comp, "current", deltas)
        want = ref(V, v, w)
        assert torch.allclose(got, want, atol=2e-5, rtol=1e-5), (l, comp, float((got - want).abs().max()))
        targeted.add(name)
    return targeted


def assert_rest_untouched(ws, deltas, targeted):
    assert set(deltas) == targeted
    for name in ws.names():
        if name not in targeted:
            assert torch.equal(ws.read(name, (), "current", deltas), ws.read(name)), name


def test_none_is_plain_directional_ablation_dry_run_and_committed(sess):
    ws = sess.ws
    DR.measure(ws, "m", GOOD, BAD)
    good, bad = saved_means(ws, "m")
    params = {"direction": "m", "weight_distributions": distributions()}
    res = run(ws, params)
    deltas = res["deltas"]
    targeted = check_against(ws, deltas, plain_ref, good=good, bad=bad)
    assert_rest_untouched(ws, deltas, targeted)
    plan = DR.plan(ws, params)
    pieces = [p for ps in deltas.values() for p in ps]
    assert all(p.kind == "lowrank" and p.A.shape[1] == 1 for p in pieces)
    assert sum(p.numel() for p in pieces) == plan["totals"]["stored"]
    assert "ablate_direction('m'): 4 components in 2 layers" in res["output"]
    # Committed through Session.apply, the same weights.
    report = sess.apply(DR.script_for(params))
    assert report["ok"] and report["committed"], report
    assert check_against(ws, None, plain_ref, good=good, bad=bad) == targeted
    # With w = 1 what the matrix writes has no component along v any more.
    ws2_params = {"direction": "m", "weight_distributions": {c: {"max_weight": 1, "max_weight_position": 0, "min_weight": 1,
                                                                 "min_weight_distance": 5} for c in DR.COMPONENTS}}
    out = run(ws, ws2_params)["deltas"]
    for l in range(N):
        v = ref_direction(good, bad, l + 1)
        for comp in DR.COMPONENTS:
            _, got, _ = component(ws, l, comp, "current", out)
            assert float((got @ v).abs().max()) < 1e-5


def test_pre_normalises_columns_over_the_whole_component(ws):
    DR.measure(ws, "m", GOOD, BAD)
    good, bad = saved_means(ws, "m")
    deltas = run(ws, {"direction": "m", "weight_distributions": distributions(), "row_normalization": "pre"})["deltas"]
    targeted = check_against(ws, deltas, pre_ref, good=good, bad=bad)
    assert_rest_untouched(ws, deltas, targeted)
    # Normalising each head separately would give another answer: the norms are over all heads together.
    name, V, _ = component(ws, 0, "attn.o_proj")
    v = ref_direction(good, bad, 1)
    per_head = torch.cat([pre_ref(V[:12], v, 1.0), pre_ref(V[12:], v, 1.0)])
    assert not torch.allclose(per_head, pre_ref(V, v, 1.0), atol=1e-4)


@pytest.mark.parametrize("rank", [1, 3])
def test_full_is_the_best_rank_r_approximation_of_the_norm_preserving_change(ws, rank):
    DR.measure(ws, "m", GOOD, BAD)
    good, bad = saved_means(ws, "m")
    params = {"direction": "m", "weight_distributions": distributions(), "row_normalization": "full",
              "full_normalization_lora_rank": rank}
    deltas = run(ws, params)["deltas"]
    collected = []
    rep = DR.ablate(ws, params, add=lambda view, A, B: collected.append((view, A, B)))
    errors = {(c["layer"], c["component"]): c["full_error"] for c in rep["components"]}
    targeted = set()
    for (l, comp), w in WEIGHTS.items():
        name, V, _ = component(ws, l, comp)
        v = ref_direction(good, bad, l + 1)
        exact = full_exact(V, v, w)
        want = best_rank(exact, rank)
        _, got, _ = component(ws, l, comp, "current", deltas)
        mine = got - V
        assert float((mine - want).norm() / want.norm()) <= 1e-4, (l, comp)
        assert errors[(l, comp)] == pytest.approx(float((exact - mine).norm() / exact.norm()), rel=1e-4, abs=1e-6)
        targeted.add(name)
    assert_rest_untouched(ws, deltas, targeted)
    assert all(B.shape == (rank, D) and A.shape[1] == rank for _, A, B in collected)


def test_full_with_rank_at_least_d_is_the_exact_change(ws):
    DR.measure(ws, "m", GOOD, BAD)
    good, bad = saved_means(ws, "m")
    params = {"direction": "m", "weight_distributions": distributions(), "row_normalization": "full",
              "full_normalization_lora_rank": 30}                              # above d_model = 24: all of it
    deltas = run(ws, params)["deltas"]
    for (l, comp), w in WEIGHTS.items():
        name, V, _ = component(ws, l, comp)
        exact = full_exact(V, ref_direction(good, bad, l + 1), w)
        _, got, _ = component(ws, l, comp, "current", deltas)
        assert torch.allclose(got - V, exact, atol=2e-5)
        assert torch.allclose((got).norm(dim=0), V.norm(dim=0), rtol=1e-5)    # column norms preserved
    rep = DR.ablate(ws, params, add=lambda *a: None)
    assert all(c["full_error"] < 1e-6 for c in rep["components"])


def test_zero_norm_columns_in_pre_and_full(tmp_path):
    cfg = Config(**CFG)
    p = init_params(cfg, seed=1, std=0.3)
    p["blocks.0.attn.W_O"][:, :, 5] = 0                       # residual coordinate 5 is written by no head of layer 0
    p["blocks.1.mlp.W_out"][:, 7] = 0
    ws = make_ws(tmp_path / "z", p)
    good, bad = save_random(ws, "r", seed=3)
    for mode, ref in (("pre", pre_ref), ("full", full_exact)):
        deltas = run(ws, {"direction": "r", "weight_distributions": distributions(), "row_normalization": mode,
                          "full_normalization_lora_rank": 30})["deltas"]
        for (l, comp), w in WEIGHTS.items():
            name, V, _ = component(ws, l, comp)
            v = ref_direction(good, bad, l + 1)
            _, got, _ = component(ws, l, comp, "current", deltas)
            want = ref(V, v, w) if mode == "pre" else V + ref(V, v, w)
            assert torch.isfinite(got).all() and torch.allclose(got, want, atol=2e-5), (mode, l, comp)
        assert torch.count_nonzero(component(ws, 0, "attn.o_proj", "current", deltas)[1][:, 5]) == 0
        assert torch.count_nonzero(component(ws, 1, "mlp.down_proj", "current", deltas)[1][:, 7]) == 0


# ------------------------------------------------------------------ 9. the slab path
def test_the_slab_path_gives_the_same_result_and_no_read_exceeds_the_limit(ws, monkeypatch):
    DR.measure(ws, "m", GOOD, BAD)
    views = DR.component_views(ws, 1, "mlp.down_proj") + []
    heads = DR.component_views(ws, 0, "attn.o_proj")
    whole = {"G": DR.gram(ws, heads), "sq": DR.column_sq(ws, views)}
    results = {m: run(ws, {"direction": "m", "weight_distributions": distributions(), "row_normalization": m})["deltas"]
               for m in DR.ROW_NORMALIZATIONS}
    limit = 50                                                 # two rows of 24 per read
    monkeypatch.setattr(Wmod, "STREAM_ELEMS", limit)
    reads = []
    real = ws.store.read

    def read(name, region=None):
        from wsp.store import region_size, norm_index
        r = norm_index((), ws.shape(name)) if region is None else region
        reads.append((name, region_size(r)))
        return real(name, region)
    monkeypatch.setattr(ws.store, "read", read)
    ws._cache.clear()
    ws._cache_bytes = 0
    G, sq = DR.gram(ws, heads), DR.column_sq(ws, views)
    assert torch.allclose(G, whole["G"], rtol=1e-10, atol=1e-12) and torch.allclose(sq, whole["sq"], rtol=1e-10)
    for m in DR.ROW_NORMALIZATIONS:
        got = run(ws, {"direction": "m", "weight_distributions": distributions(), "row_normalization": m})["deltas"]
        for name in results[m]:
            assert torch.allclose(ws.read(name, (), "delta", got), ws.read(name, (), "delta", results[m]), atol=1e-6), (m, name)
    weight_reads = [n for name, n in reads if name.endswith(("W_O", "W_out"))]
    assert weight_reads and max(weight_reads) <= limit and len(weight_reads) > 20


# ------------------------------------------------------------------ 10. through Session.apply
def test_session_apply_scope_dry_run_and_a_foreign_measurement(sess):
    ws = sess.ws
    DR.measure(ws, "m", GOOD, BAD)
    params = {"direction": "m", "weight_distributions": distributions()}
    script = DR.script_for(params)
    dry = sess.apply(script, dry_run=True)
    assert dry["ok"] and not dry["committed"] and ws.patches == []
    sess.set_task({"scope": ["blocks.0"]})
    refused = sess.apply(script, enforce_scope=True)
    assert not refused["ok"] and "outside the task scope" in refused["error"] and ws.patches == []
    only0 = {"direction": "m", "weight_distributions": {"attn.o_proj": {"max_weight": 1, "max_weight_position": 0, "min_weight": 1,
                                                                        "min_weight_distance": 0.5}}}
    inside = sess.apply(DR.script_for(only0), enforce_scope=True)
    assert inside["ok"] and inside["committed"]
    sess.set_task({"scope": ["model"]})
    done = sess.apply(script)
    assert done["ok"] and done["committed"] and len(ws.patches) == 2
    patch = ws.patches[-1]
    assert patch.source == {"kind": "wscript", "script": script}
    assert sorted(patch.deltas) == ["blocks.0.attn.W_O", "blocks.0.mlp.W_out", "blocks.1.attn.W_O", "blocks.1.mlp.W_out"]
    save_random(ws, "other", n_layers=3)
    r = sess.apply(DR.script_for({**params, "direction": "other"}))
    assert not r["ok"] and "measured on a model with 3 layers" in r["error"] and len(ws.patches) == 2
    save_random(ws, "narrow", d=16)
    assert "d_model 16" in sess.apply(DR.script_for({**params, "direction": "narrow"}))["error"]


def test_script_functions_direction_and_means(ws):
    DR.measure(ws, "m", GOOD, BAD)
    good, bad = saved_means(ws, "m")
    res = run_script(ws, 'd = direction("m", 2)\ng, b = direction_means("m")\nprint(norm(d), dot(d, unit(g[2])), norm(b - g))')
    vals = [float(x) for x in res["output"].split()]
    assert vals[0] == pytest.approx(1, abs=1e-5) and abs(vals[1]) < 1e-5
    assert vals[2] == pytest.approx(float((bad - good).norm()), rel=1e-5)
    for name in ("ablate_direction(", "direction(", "direction_means(", "row_normalization", "full_normalization_lora_rank",
                 "max_weight_position", "min_weight_distance"):
        assert name in FUNCTION_DOC, name
    assert res["deltas"] == {}
    with pytest.raises(Exception, match="no direction named 'none'"):
        run_script(ws, 'direction("none", 0)')
    with pytest.raises(Exception, match="index 5 out of range"):
        run_script(ws, 'direction("m", 5)')


# ------------------------------------------------------------------ 11. routes
def test_measure_requests_are_checked_before_any_model_is_loaded(sess, monkeypatch):
    built = []

    def refuse(w):
        built.append(w.name)
        raise AssertionError("the runner was built for a request that should have been refused first")
    monkeypatch.setattr(wsp.runner, "make_runner", refuse)
    save_random(sess.ws, "taken")
    assert sess.ws._runner is None
    c = client(sess)
    ok = {"name": "n", "good": ["abc"], "bad": ["xyz"]}
    for body, message in [
        ({"good": ["a"], "bad": ["b"]}, "name: required"), ({**ok, "name": "a b"}, "name must be"),
        ({**ok, "name": "q.prompts"}, ".prompts"), ({**ok, "good": []}, "good must list"),
        ({**ok, "good": ["a"] * 1025}, "good must list 1..1024"), ({**ok, "bad": ["x" * 20001]}, "1..20000 characters"),
        ({**ok, "good": ["abcdefghijklmnopq"]}, "has 17 tokens"), ({**ok, "bad": ["ab1"]}, "bad prompt 1"),
        ({**ok, "winsorization_quantile": -0.1}, "winsorization_quantile"), ({**ok, "chat": 1}, "chat must be"),
        ({**ok, "chat": True}, "needs the model's own chat template: this built-in model has no chat template"),
        ({**ok, "system": "x"}, "system and prefix apply"), ({**ok, "variant": "edited"}, "variant must be"),
        ({**ok, "unknown": 1}, "unknown: unknown field"), ({**ok, "replace": 1}, "replace must be"),
        ({**ok, "name": "taken"}, "exists; pass replace"), ({**ok, "stream": "bad id!"}, "stream must be"),
        ({**ok, "chat": True, "chat_options": {"enable_thinking": "no"}}, "Thinking switches must be booleans"),
    ]:
        r = c.post("/api/directions/measure", json=body)
        assert r.status_code == 400 and message in r.json()["error"], (body, r.text)
    for body, message in [({"direction": "taken"}, "weight_distributions: must be"),
                          ({"direction": "taken", "weight_distributions": distributions(), "row_normalization": "x"}, "row_normalization")]:
        r = c.post("/api/directions/script", json=body)
        assert r.status_code == 400 and message in r.json()["error"]
    assert c.post("/api/directions/script", json={"direction": "taken", "weight_distributions": distributions()}).status_code == 200
    assert c.get("/api/directions").status_code == 200 and c.get("/api/directions/taken").status_code == 200
    assert built == [] and sess.ws._runner is None
    monkeypatch.undo()
    assert c.post("/api/directions/measure", json=ok).status_code == 200      # a good request does reach the runner


def test_routes_measure_list_get_prompts_script_edit_delete(sess):
    c = client(sess)
    q = sess.subscribe()
    r = c.post("/api/directions/measure", json={"name": "refusal", "good": GOOD, "bad": BAD, "stream": "m-1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "refusal" and body["counts"] == {"good": 6, "bad": 5} and len(body["stats"]) == N + 1
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    mine = [e for e in events if e.get("type") == "directions"]
    assert mine[0]["state"] == "measuring" and mine[0]["done"] == 0 and mine[-1]["state"] == "measured"
    assert all(e["stream"] == "m-1" and e["name"] == "refusal" and e["workspace"] == "rnd" and "seconds" in e for e in mine)
    assert mine[-1]["done"] == mine[-1]["total"] == 11
    listed = c.get("/api/directions").json()
    assert [d["name"] for d in listed["directions"]] == ["refusal"] and listed["limits"]["prompts_per_set"] == 1024
    assert "good" not in listed["directions"][0] and "stats" not in listed["directions"][0]
    got = c.get("/api/directions/refusal").json()
    assert got["stats"] == body["stats"] and got["counts"] == body["counts"]
    assert c.get("/api/directions/refusal?orthogonalize_direction=false").json()["stats"][1]["usable"]
    assert c.get("/api/directions/refusal/prompts").json() == {"good": GOOD, "bad": BAD}
    assert c.get("/api/directions/nothing").status_code == 404 and "no direction named" in c.get("/api/directions/nothing").json()["error"]
    assert c.get("/api/directions/nothing/prompts").status_code == 404
    assert c.get("/api/directions/a%20b").status_code == 400
    # replace
    again = c.post("/api/directions/measure", json={"name": "refusal", "good": GOOD[:2], "bad": BAD[:2]})
    assert again.status_code == 400 and "exists; pass replace" in again.json()["error"]
    again = c.post("/api/directions/measure", json={"name": "refusal", "good": GOOD[:2], "bad": BAD[:2], "replace": True})
    assert again.status_code == 200 and again.json()["counts"] == {"good": 2, "bad": 2}
    assert c.get("/api/directions/refusal/prompts").json() == {"good": GOOD[:2], "bad": BAD[:2]}
    # script -> /api/edit (dry run, then commit)
    sc = c.post("/api/directions/script", json={"direction": "refusal", "weight_distributions": distributions()})
    assert sc.status_code == 200, sc.text
    script, plan = sc.json()["script"], sc.json()["plan"]
    assert script == DR.script_for({"direction": "refusal", "weight_distributions": distributions()}) and plan["totals"]["components"] == 4
    dry = c.post("/api/edit", json={"script": script, "dry_run": True}).json()
    assert dry["ok"] and not dry["committed"] and sess.ws.patches == []
    done = c.post("/api/edit", json={"script": script}).json()
    assert done["ok"] and done["committed"] and len(sess.ws.patches) == 1 and done["stored"] == plan["totals"]["stored"]
    # delete
    assert c.delete("/api/directions/refusal").json() == {"ok": True, "deleted": "refusal"}
    assert c.get("/api/directions/refusal").status_code == 404 and c.delete("/api/directions/refusal").status_code == 404
    assert c.get("/api/directions").json()["directions"] == []


def test_a_stop_between_chunks_saves_nothing(sess, monkeypatch):
    app = create_app(sess, "http://127.0.0.1:0")
    c = TestClient(app, raise_server_exceptions=False)
    calls = []
    real = DR.residuals

    def first_then_stop(w, ids, variant="base"):
        calls.append(len(ids))
        out = real(w, ids, variant)
        app.state.generation_stops.stop("stop-me")              # what POST /api/generation/stop does
        return out
    monkeypatch.setattr(DR, "residuals", first_then_stop)
    q = sess.subscribe()
    r = c.post("/api/directions/measure", json={"name": "s", "good": GOOD, "bad": BAD, "stream": "stop-me"})
    assert r.status_code == 200 and r.json() == {"stopped": True, "done": 4, "total": 11} and calls == [4]
    assert not os.path.exists(os.path.join(sess.ws.dir, "directions", "s.json")) and c.get("/api/directions/s").status_code == 404
    states = []
    while not q.empty():
        e = q.get_nowait()
        if e.get("type") == "directions":
            states.append(e["state"])
    assert states[0] == "measuring" and states[-1] == "stopped"
    # A stop that arrives before the request (as Run and Chat): nothing runs at all.
    assert c.post("/api/generation/stop", json={"stream": "early"}).json()["found"] is False
    calls.clear()
    r = c.post("/api/directions/measure", json={"name": "s", "good": GOOD, "bad": BAD, "stream": "early"})
    assert r.json() == {"stopped": True, "done": 0, "total": 11} and calls == []


def test_a_failed_measurement_publishes_failed(sess, monkeypatch):
    c = client(sess)

    def broken(w, ids, variant="base"):
        raise RuntimeError("the job failed")
    monkeypatch.setattr(DR, "residuals", broken)
    q = sess.subscribe()
    r = c.post("/api/directions/measure", json={"name": "f", "good": GOOD, "bad": BAD})
    assert r.status_code == 400 and "the job failed" in r.json()["error"]
    events = [e for e in iter(lambda: q.get_nowait() if not q.empty() else None, None) if e.get("type") == "directions"]
    assert events[-1]["state"] == "failed" and "the job failed" in events[-1]["error"]
    assert not DR.exists(sess.ws, "f")


def test_chat_rendering_with_system_and_prefix(sess, monkeypatch):
    """A stub template (as tests/test_probe_chat.py): the system message, then the user turn, then the prefix."""
    calls = []

    def chat_prompt(self, messages, **options):
        calls.append((messages, options))
        system = "".join("xx" + m["content"] for m in messages if m["role"] == "system")
        return system + "zz" + messages[-1]["content"] + "yy"
    monkeypatch.setattr(P, "chat_template", lambda ws: (True, "the model's own chat template"))
    monkeypatch.setattr(wsp.runner.Runner, "chat_prompt", chat_prompt)
    c = client(sess)
    r = c.post("/api/directions/measure", json={"name": "c", "good": ["abc", "ab"], "bad": ["xyz"], "chat": True, "system": "s",
                                                "prefix": "p", "chat_options": {"enable_thinking": False}})
    assert r.status_code == 200, r.text
    ws = sess.ws
    good, bad = saved_means(ws, "c")
    assert torch.allclose(good, means(ws, ["xxszzabcyyp", "xxszzabyyp"]), atol=1e-5)
    assert torch.allclose(bad, means(ws, ["xxszzxyzyyp"]), atol=1e-5)
    # As Heretic: the system-role probe first (with the options), then each prompt.
    assert calls[0] == ([{"role": "system", "content": "HERETIC_SYSTEM_ROLE_PROBE"}, {"role": "user", "content": "Hi"}],
                        {"enable_thinking": False})
    assert calls[1] == ([{"role": "system", "content": "s"}, {"role": "user", "content": "abc"}], {"enable_thinking": False})
    meta = r.json()
    assert meta["tokens"] == {"good": 11 + 10, "bad": 11} and meta["options"]["chat"] and meta["options"]["prefix"] == "p"
    assert DR.prompts(ws, "c") == {"good": ["abc", "ab"], "bad": ["xyz"]}                  # the texts as given
    # The rendering counts toward the token bound (checked after rendering): 16 tokens of context here.
    fits = c.post("/api/directions/measure", json={"name": "c2", "good": ["abcdefgh"], "bad": ["x"], "chat": True, "system": "s"})
    assert fits.status_code == 200, fits.text                                          # 3 + 2 + 8 + 2 = 15 tokens
    over = c.post("/api/directions/measure", json={"name": "c3", "good": ["abcdefghijk"], "bad": ["x"], "chat": True, "system": "s"})
    assert over.status_code == 400 and "good prompt 1 as a chat message has 18 tokens; it must have 1..16" in over.json()["error"]
    # Without a system message the messages are the user turn alone, as the probes send it.
    calls.clear()
    assert c.post("/api/directions/measure", json={"name": "c4", "good": ["a"], "bad": ["b"], "chat": True}).status_code == 200
    assert calls[0][0] == [{"role": "user", "content": "a"}]


# ------------------------------------------------------------------ 12. the job
def test_the_job_is_a_row_of_the_table_and_checks_its_bounds(ws):
    row = JOBS["direction_residuals"]
    assert row.args == ("prompts_ids", "variant", "module_io") and row.result == "tensor" and not row.stream and not row.extra
    ids = ws.encode("abc")
    out = DR.residuals(ws, [ids, ws.encode("zz")], "base")
    assert out.shape == (2, N + 1, D) and out.dtype == torch.float32
    assert torch.allclose(out[0], vectors(ws, ids), atol=1e-6)
    for args, message in [(([],), "1..8 prompts"), (([ids] * 9,), "1..8 prompts"), (([[]],), "prompt 1 must have 1..16 tokens"),
                          (([ids, list(range(17))],), "prompt 2 must have 1..16 tokens"), (([["a"]],), "integer token ids"),
                          (([ids], "delta"), "variant must be"), (("abc",), "1..8 prompts")]:
        with pytest.raises(ValueError, match=message):
            DR.residuals(ws, *args)


def test_the_job_through_the_stub_worker_and_the_remote_runner(tmp_path, monkeypatch):
    from test_steering_worker import WspHarness, local_ws, remote
    h = WspHarness(tmp_path)
    h.load()
    here = local_ws(tmp_path)
    prompts = [here.encode(t) for t in GOOD[:3]]
    frame, _, _ = h.call("direction_residuals", {"prompts_ids": prompts, "variant": "base"})
    assert frame.type == "result" and set(frame.tensors) == {"value"}
    assert torch.equal(frame.tensors["value"], DR.residuals(here, prompts, "base"))
    frame, _, _ = h.call("direction_residuals", {"prompts_ids": prompts * 3, "variant": "base"})
    assert frame.type == "error" and frame["exception"] == "DirectionError" and "1..8 prompts" in frame["message"]
    # The server side: the job forwarded whole; the measurement loop on the server over the worker.
    r, sent, received = remote(h, monkeypatch)
    got = DR.residuals(h.ws, prompts, "base")
    assert torch.equal(got, DR.residuals(here, prompts, "base")) and sent[-1] == ("direction_residuals", {"prompts_ids": prompts, "variant": "base", "module_io": False})
    with pytest.raises(DR.DirectionError, match="1..8 prompts"):
        DR.residuals(h.ws, prompts * 3, "base")
    with pytest.raises(RuntimeError, match="runs in the model worker"):
        r.final_norm(torch.zeros(D))
    sent.clear()
    out = DR.measure(h.ws, "w", GOOD, BAD)
    assert [n for n, _ in sent] == ["direction_residuals"] * 4                    # chunks of 4: good 4 + 2, bad 4 + 1
    DR.measure(here, "w", GOOD, BAD)
    assert torch.equal(saved_means(h.ws, "w")[0], saved_means(here, "w")[0]) and out["counts"] == {"good": 6, "bad": 5}
    scoped = [(n, kw.get("tokens"), kw.get("retained")) for n, kw in h.ws._runtime_client.ops if n == "direction_residuals" and kw]
    assert scoped[0] == ("direction_residuals", max(len(t) for t in GOOD[:3]), True)       # the longest prompt it runs
    assert scoped[2] == ("direction_residuals", max(len(t) for t in GOOD[:4]), True)       # measure's first chunk
    h.ws._runtime_client = None
    h.shutdown()


# ------------------------------------------------------------------ 13. a checkpoint's own 2-D attention output
def test_a_llama_shaped_checkpoint_with_one_2d_o_proj(tmp_path):
    """The model-free llama-named checkpoint and fake decoder of tests/test_steering_worker.py (no transformers):
    o_proj [d, n_heads * d_head] and down_proj [d, d_mlp] are single 2-D matrices in the checkpoint's own layout."""
    from test_steering_worker import D as HD, LAYERS, hf_runner
    r, model, tensors = hf_runner(tmp_path)
    ws = r.ws
    # The measurement job through HFRunner: hook_resid_pre per decoder layer, the final norm module on the last output.
    ids = [1, 2, 3, 4]
    with torch.no_grad():
        x = model.model.embed_tokens(torch.tensor([ids]))
        want = []
        for layer in model.model.layers:
            want.append(x[0, -1].clone())
            x = layer(x, None)[0]
        want.append(model.model.norm(x)[0, -1])
    got = DR.residuals(ws, [ids], "base")[0]
    assert got.shape == (LAYERS + 1, HD) and torch.allclose(got, torch.stack(want), atol=1e-6)
    assert torch.allclose(r.final_norm(x[0, -1], "base"), model.model.norm(x)[0, -1], atol=1e-6)
    good, bad = save_random(ws, "r", n_layers=LAYERS, d=HD, seed=7)
    views = DR.component_views(ws, 0, "attn.o_proj")
    assert len(views) == 1 and views[0].shape == (HD, HD) and views[0].perm == (1, 0)
    dists = {c: {"max_weight": 0.9, "max_weight_position": 0, "min_weight": 0.6, "min_weight_distance": 1} for c in DR.COMPONENTS}
    weights = {(l, c): (0.9 if l == 0 else 0.6) for l in range(LAYERS) for c in DR.COMPONENTS}
    for mode, ref in (("none", plain_ref), ("pre", pre_ref), ("full", None)):
        deltas = run_script(ws, DR.script_for({"direction": "r", "weight_distributions": dists, "row_normalization": mode,
                                               "full_normalization_lora_rank": 64}))["deltas"]
        for (l, comp), w in weights.items():
            name = f"model.layers.{l}.self_attn.o_proj.weight" if comp == "attn.o_proj" else f"model.layers.{l}.mlp.down_proj.weight"
            V = ws.read(name).double().T                                      # [d_in, d]: the residual stream last
            v = ref_direction(good, bad, l + 1)
            want = ref(V, v, w) if ref else V + full_exact(V, v, w)
            got = ws.read(name, (), "current", deltas).double().T
            assert torch.allclose(got, want, atol=2e-5), (mode, l, comp)
        assert set(deltas) == {f"model.layers.{l}.{p}.weight" for l in range(LAYERS) for p in ("self_attn.o_proj", "mlp.down_proj")}


def test_a_quantized_llama_shaped_workspace_takes_the_ablation_as_patches(tmp_path, monkeypatch):
    """tests/selection_fixtures.py: a llama layout whose store reports Q4_K and refuses reads over its limit."""
    from selection_fixtures import quant_workspace
    ws = quant_workspace(tmp_path / "q")
    good, bad = save_random(ws, "r", n_layers=2, d=16, seed=2)
    monkeypatch.setattr(Wmod, "STREAM_ELEMS", 40)
    dists = {"attn.o_proj": {"max_weight": 1, "max_weight_position": 1, "min_weight": 1, "min_weight_distance": 0.5}}
    deltas = run_script(ws, DR.script_for({"direction": "r", "weight_distributions": dists, "row_normalization": "full"}))["deltas"]
    assert list(deltas) == ["model.layers.1.self_attn.o_proj.weight"] and all(p.kind == "lowrank" for p in deltas[list(deltas)[0]])
    assert max(n for name, n in ws.store.reads if "o_proj" in name) <= 40
    V = ws.read("model.layers.1.self_attn.o_proj.weight").double().T
    exact = full_exact(V, ref_direction(good, bad, 2), 1.0)
    mine = ws.read("model.layers.1.self_attn.o_proj.weight", (), "current", deltas).double().T - V
    assert float((mine - best_rank(exact, 3)).norm() / best_rank(exact, 3).norm()) <= 1e-4
    # The base store is never written: only patch pieces change what is read.
    assert torch.equal(ws.store.t["model.layers.1.self_attn.o_proj.weight"].double().T, V)
