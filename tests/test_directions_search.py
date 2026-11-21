"""The automatic search (wsp.directions_search, wsp.directions_sampler), model-free.

The sampler is held to Heretic's own draws: the literal values below are typed from the Optuna journals and
``dump.json`` files of Heretic runs (Heretic 64e9991, Optuna 5.0.0), each cited by its run's short name. Nothing
here computes an expectation with the code under test: spaces are typed from the journals' distribution lines,
the Pareto fronts are worked out by hand, and the searches on the tiny model are compared with direct scores of
the same scripts in a second workspace and with uninterrupted runs of the same search.

Tests marked [optuna] need Optuna 5 (requirements-search.txt) and skip without it.
"""
import math
import os
import subprocess
import sys
import time

import pytest
import torch

from wsp import delta as Dl
from wsp import directions as DR
from wsp import directions_sampler as S
from wsp import directions_search as DSR

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOTH = ["attn.o_proj", "mlp.down_proj"]
ABL_SETTINGS = {"orthogonalize_direction": True, "row_normalization": "full", "full_normalization_lora_rank": 3}
ARA_SETTINGS = {"preserve_row_magnitudes": True, "lora_rank": 50, "n_optimization_steps": 5, "learning_rate": 1.0,
                "max_iter": 20, "history_size": 10}

# ------------------------------------------------------------------ Heretic's recorded trials (typed, not computed)
# qwen05b/ara: seed 12345, n_startup_trials 2, 24 layers, 64 good / 64 bad prompts (journal params and values)
ARA_Q05 = [({"start_layer_index": 12, "end_layer_index": 16, "preserve_good_behavior_weight": 0.18391881167709445,
             "steer_bad_behavior_weight": 0.0006580230611475311, "overcorrect_relative_weight": 0.7380425378061927, "neighbor_count": 9},
            [0.0, 0.0877549946308136]),
           ({"start_layer_index": 12, "end_layer_index": 20, "preserve_good_behavior_weight": 0.7489066375339118,
             "steer_bad_behavior_weight": 0.041141439521771406, "overcorrect_relative_weight": 0.9720292520526561, "neighbor_count": 15},
            [0.0, 1.8798043727874756]),
           ({"start_layer_index": 9, "end_layer_index": 15, "preserve_good_behavior_weight": 0.22964349961398073,
             "steer_bad_behavior_weight": 0.00011481930683434137, "overcorrect_relative_weight": 0.6581294089043791, "neighbor_count": 4},
            [0.53125, 0.036170125007629395]),
           ({"start_layer_index": 0, "end_layer_index": 16, "preserve_good_behavior_weight": 0.04358641125482621,
             "steer_bad_behavior_weight": 0.0013927864320674086, "overcorrect_relative_weight": 1.221448118630803, "neighbor_count": 9},
            [0.0, 12.941221237182617])]
# qwen05b/abl-full: seed 12345, n_startup_trials 2, 24 layers (journal params, in suggest order, and values)
ABL_Q05 = [({"direction_scope": "global", "direction_index": 11.315066334286588, "attn.o_proj.max_weight": 0.9431921949871278,
             "attn.o_proj.max_weight_position": 19.023070267551518, "attn.o_proj.min_weight": 0.5955447029792516,
             "attn.o_proj.min_weight_distance": 13.345785852615956, "mlp.down_proj.max_weight": 0.8930599195252491,
             "mlp.down_proj.max_weight_position": 20.689941065311988, "mlp.down_proj.min_weight": 0.6535698708517353,
             "mlp.down_proj.min_weight_distance": 10.570749558672304}, [0.875, 0.013251576572656631]),
           ({"direction_scope": "global", "direction_index": 10.424110332023773, "attn.o_proj.max_weight": 1.0090925996385671,
             "attn.o_proj.max_weight_position": 19.838982884356884, "attn.o_proj.min_weight": 0.8098125525409348,
             "attn.o_proj.min_weight_distance": 12.163851695670822, "mlp.down_proj.max_weight": 1.4381332954008779,
             "mlp.down_proj.max_weight_position": 20.45790519166357, "mlp.down_proj.min_weight": 0.6424753278958413,
             "mlp.down_proj.min_weight_distance": 10.183406346398895}, [0.9375, 0.03194257989525795]),
           ({"direction_scope": "global", "direction_index": 12.196055090961117, "attn.o_proj.max_weight": 0.8867430487523009,
             "attn.o_proj.max_weight_position": 17.335212844765753, "attn.o_proj.min_weight": 0.06283080444498068,
             "attn.o_proj.min_weight_distance": 10.906430990692252, "mlp.down_proj.max_weight": 0.3788935254852682,
             "mlp.down_proj.max_weight_position": 20.448127643871345, "mlp.down_proj.min_weight": 0.48330353330187015,
             "mlp.down_proj.min_weight_distance": 12.280752713407662}, [0.90625, 0.003713575191795826]),
           ({"direction_scope": "global", "direction_index": 10.829118637368698, "attn.o_proj.max_weight": 1.1906282712698923,
             "attn.o_proj.max_weight_position": 15.02769672631626, "attn.o_proj.min_weight": 0.0005449732348578595,
             "attn.o_proj.min_weight_distance": 13.16879482454923, "mlp.down_proj.max_weight": -0.229304390885832,
             "mlp.down_proj.max_weight_position": 15.914495437210327, "mlp.down_proj.min_weight": 0.21950345804192317,
             "mlp.down_proj.min_weight_distance": 13.54941398442828}, [0.9375, 0.007412843406200409])]
# its trial 4 as Heretic recorded the parameters (dump.json trials[3].parameters)
ABL_Q05_T4_PARAMETERS = {"direction_index": 10.829118637368698, "weight_distributions": {
    "attn.o_proj": {"max_weight": 1.1906282712698923, "max_weight_position": 15.02769672631626, "min_weight": 0.0006488605405071743, "min_weight_distance": 13.16879482454923},
    "mlp.down_proj": {"max_weight": 0.0, "max_weight_position": 15.914495437210327, "min_weight": 0.0, "min_weight_distance": 13.54941398442828}}}
# tiny/perlayer-s1: seed 1, n_startup_trials 1, 4 layers
PERLAYER_TINY = [({"direction_scope": "per layer", "direction_index": 1.2001715622260174, "attn.o_proj.max_weight": 1.011632800842288,
                   "attn.o_proj.max_weight_position": 1.9761070689805356, "attn.o_proj.min_weight": 0.0923385947687978,
                   "attn.o_proj.min_weight_distance": 1.1490081691021368, "mlp.down_proj.max_weight": 0.3547312723253335,
                   "mlp.down_proj.max_weight_position": 2.276120969076804, "mlp.down_proj.min_weight": 0.538816734003357,
                   "mlp.down_proj.min_weight_distance": 1.3353556115226357}, [0.0, 0.0001907242985907942]),
                 ({"direction_scope": "per layer", "direction_index": 1.7703111881078486, "attn.o_proj.max_weight": 0.9234953914792223,
                   "attn.o_proj.max_weight_position": 1.9954951278252133, "attn.o_proj.min_weight": 0.0489176292001809,
                   "attn.o_proj.min_weight_distance": 1.074035779526326, "mlp.down_proj.max_weight": 0.7259452474791995,
                   "mlp.down_proj.max_weight_position": 2.5823698772787886, "mlp.down_proj.min_weight": 0.40959606109520186,
                   "mlp.down_proj.min_weight_distance": 1.3486343312115576}, [0.0, 0.00012496468843892217])]
PERLAYER_TINY_T1_PARAMETERS = {"direction_index": None, "weight_distributions": {
    "attn.o_proj": {"max_weight": 1.011632800842288, "max_weight_position": 1.9761070689805356, "min_weight": 0.09341275125179996, "min_weight_distance": 1.1490081691021368},
    "mlp.down_proj": {"max_weight": 0.3547312723253335, "max_weight_position": 2.276120969076804, "min_weight": 0.19113514560319161, "min_weight_distance": 1.3353556115226357}}}
# tiny/ara-s1: seed 1, n_startup_trials 1, 4 layers, 5 good / 5 bad prompts
ARA_TINY = [({"start_layer_index": 1, "end_layer_index": 4, "preserve_good_behavior_weight": 0.00011437481734488664,
              "steer_bad_behavior_weight": 0.0016193110912440735, "overcorrect_relative_weight": 0.19078265806224695, "neighbor_count": 1},
             [0.0, 2.8496456252469216e-06]),
            ({"start_layer_index": 2, "end_layer_index": 4, "preserve_good_behavior_weight": 0.004483433332550856,
              "steer_bad_behavior_weight": 0.0022963171409208214, "overcorrect_relative_weight": 0.011969077170964765, "neighbor_count": 1},
             [0.0, 1.0571621942290221e-06])]


def F(lo, hi, log=False):
    return {"name": "FloatDistribution", "attributes": {"step": None, "low": lo, "high": hi, "log": log}}


def I(lo, hi):  # noqa: E743
    return {"name": "IntDistribution", "attributes": {"log": False, "step": 1, "low": lo, "high": hi}}


def C(*choices):
    return {"name": "CategoricalDistribution", "attributes": {"choices": list(choices)}}


def ablation_block(c, mw_low, pos, dist):
    return [(f"{c}.max_weight", F(mw_low, 1.5)), (f"{c}.max_weight_position", F(*pos)),
            (f"{c}.min_weight", F(0.0, 1.0)), (f"{c}.min_weight_distance", F(*dist))]


def rows(pairs):
    return [{"name": n, "distribution": d} for n, d in pairs]


# Typed from the journals' distribution lines (qwen05b/abl-full, qwen05b/ara, tiny/perlayer-s1, tiny/ara-s1).
ABL_24 = rows([("direction_scope", C("global", "per layer")), ("direction_index", F(9.200000000000001, 20.7))]
              + ablation_block("attn.o_proj", 0.8, (13.799999999999999, 23.0), (1.0, 13.799999999999999))
              + ablation_block("mlp.down_proj", -0.25, (13.799999999999999, 23.0), (1.0, 13.799999999999999)))
ABL_4 = rows([("direction_scope", C("global", "per layer")), ("direction_index", F(1.2000000000000002, 2.7))]
             + ablation_block("attn.o_proj", 0.8, (1.7999999999999998, 3.0), (1.0, 1.7999999999999998))
             + ablation_block("mlp.down_proj", -0.25, (1.7999999999999998, 3.0), (1.0, 1.7999999999999998)))
ARA_24 = rows([("start_layer_index", I(0, 12)), ("end_layer_index", I(13, 24)),
               ("preserve_good_behavior_weight", F(0.0, 1.0)), ("steer_bad_behavior_weight", F(0.0001, 1.0, True)),
               ("overcorrect_relative_weight", F(0.0, 1.3)), ("neighbor_count", I(1, 15))])
ARA_4 = rows([("start_layer_index", I(0, 2)), ("end_layer_index", I(3, 4)),
              ("preserve_good_behavior_weight", F(0.0, 1.0)), ("steer_bad_behavior_weight", F(0.0001, 1.0, True)),
              ("overcorrect_relative_weight", F(0.0, 1.3)), ("neighbor_count", I(1, 5))])


# ------------------------------------------------------------------ 1. the spaces
def test_the_spaces_are_heretics():
    assert DSR.space("ablation", n_layers=24, components=BOTH) == ABL_24
    assert DSR.space("ablation", n_layers=4, components=list(reversed(BOTH))) == ABL_4       # sorted, as Heretic's
    assert DSR.space("ara", n_layers=24, n_good=64, n_bad=64) == ARA_24
    assert DSR.space("ara", n_layers=4, components=BOTH, n_good=5, n_bad=5) == ARA_4
    assert DSR.space("ablation", n_layers=24, components=["attn.o_proj"]) == ABL_24[:6]
    assert [type(r["distribution"]["attributes"]["low"]) for r in DSR.space("ara", n_layers=24, n_good=64, n_bad=64)] \
        == [int, int, float, float, float, int]


# ------------------------------------------------------------------ 2. from draws to parameters
def test_the_parameters_are_heretics():
    p = DSR.to_parameters("ablation", ABL_Q05[3][0], ABL_SETTINGS, direction="refusal", components=BOTH, seed=12345)
    assert DSR.heretic_parameters("ablation", p) == ABL_Q05_T4_PARAMETERS
    assert p["weight_distributions"]["mlp.down_proj"]["max_weight"] == 0.0
    assert p["weight_distributions"]["mlp.down_proj"]["min_weight"] == 0.0
    assert {k: p[k] for k in ("method", "direction", "orthogonalize_direction", "row_normalization",
                              "full_normalization_lora_rank")} == {"method": "ablation", "direction": "refusal", **ABL_SETTINGS}
    per = DSR.to_parameters("ablation", PERLAYER_TINY[0][0], ABL_SETTINGS, direction="r", components=BOTH, seed=1)
    assert DSR.heretic_parameters("ablation", per) == PERLAYER_TINY_T1_PARAMETERS
    a = DSR.to_parameters("ara", ARA_Q05[0][0], {**ARA_SETTINGS, "print_loss": False}, direction="m", components=BOTH,
                          seed=12345)
    assert a == {"method": "ara", "direction": "m", **ARA_Q05[0][0], **ARA_SETTINGS, "seed": 12345}
    assert DSR.heretic_parameters("ara", a) == ARA_Q05[0][0]
    for params in (p, per, a):                       # the scripts are accepted as the IDE's own parameters
        script = DR.script_for(params)
        assert script.startswith("ablate_direction(" if params["method"] == "ablation" else "ablate_ara(")
    DR.normalize({k: v for k, v in p.items() if k != "method"})
    from wsp import directions_ara as ARA
    assert ARA.normalize(a)["seed"] == 12345
    assert DSR.present("ablation", DSR.heretic_parameters("ablation", per))["direction_index"] == "per layer"
    shown = DSR.present("ablation", DSR.heretic_parameters(
        "ablation", DSR.to_parameters("ablation", ABL_Q05[0][0], ABL_SETTINGS, direction="r", components=BOTH, seed=1)))
    assert shown["direction_index"] == "11.32" and shown["attn.o_proj.max_weight"] == "0.94"
    assert list(shown) == ["direction_index"] + [f"{c}.{k}" for c in BOTH for k in
                                                 ("max_weight", "max_weight_position", "min_weight", "min_weight_distance")]
    t4 = DSR.present("ablation", ABL_Q05_T4_PARAMETERS)
    assert t4["mlp.down_proj.max_weight"] == "0.00" and t4["direction_index"] == "10.83"
    assert DSR.present("ara", ARA_Q05[0][0]) == {
        "start_layer_index": "12", "end_layer_index": "16", "preserve_good_behavior_weight": "0.1839",
        "steer_bad_behavior_weight": "0.0007", "overcorrect_relative_weight": "0.7380", "neighbor_count": "9"}


# ------------------------------------------------------------------ 3-6. the sampler
def drive(sampler, space, told):
    """Ask, record the draws, tell each recorded value list in turn."""
    out = []
    for values in told:
        trial, draws = sampler.ask(space)
        assert list(draws) == [r["name"] for r in space]
        out.append(draws)
        sampler.tell(trial, values=values)
    return out


def test_the_sampler_draws_heretics_trials():
    pytest.importorskip("optuna")
    for trials, sp, seed, startup in ((ARA_Q05, ARA_24, 12345, 2), (ABL_Q05, ABL_24, 12345, 2),
                                      (PERLAYER_TINY, ABL_4, 1, 1), (ARA_TINY, ARA_4, 1, 1)):
        s = S.Sampler(seed=seed, n_startup_trials=startup, directions=["minimize"] * 2)
        got = drive(s, sp, [v for _, v in trials])
        for k, (want, _) in enumerate(trials):
            assert got[k] == want, (seed, k)


def test_failed_trials_do_not_count_as_startup_trials():
    pytest.importorskip("optuna")
    names = [r["name"] for r in ARA_24]
    s = S.Sampler(seed=12345, n_startup_trials=2, directions=["minimize", "minimize"])
    trial, _ = s.ask(ARA_24)
    s.tell(trial, failed=True)
    trial, draws = s.ask(ARA_24)
    assert draws == ARA_Q05[1][0]
    s.tell(trial, values=[0.0, 1.8798043727874756])
    trial, draws = s.ask(ARA_24)                     # still a startup draw: the random sampler's third
    assert draws == dict(zip(names, [0, 14, 0.29870371376938154, 0.042232299833226385, 1.0527563183032151, 14]))
    s.tell(trial, values=[0.5, 0.5])
    _, draws = s.ask(ARA_24)                         # the first TPE draw
    assert draws == dict(zip(names, [12, 22, 0.737757870492858, 0.12018774016127057, 1.0566099422494515, 8]))


def test_only_the_ranking_of_scores_moves_the_sampler():
    pytest.importorskip("optuna")
    want = [d for d, _ in ABL_Q05]
    s = S.Sampler(seed=12345, n_startup_trials=2, directions=["minimize", "minimize"])
    assert drive(s, ABL_24, [[v[0], v[1] * (1 + 1e-6)] for _, v in ABL_Q05]) == want
    (r1, k1), (r2, k2) = ABL_Q05[0][1], ABL_Q05[1][1]
    swapped = [[r1, k2], [r2, k1]] + [v for _, v in ABL_Q05[2:]]
    s = S.Sampler(seed=12345, n_startup_trials=2, directions=["minimize", "minimize"])
    got = drive(s, ABL_24, swapped)
    assert got[:2] == want[:2] and got[2] != want[2] and got[3] != want[3]


def records_of(trials):
    return [{"index": k + 1, "state": "complete", "draws": dict(d), "values": list(v)} for k, (d, v) in enumerate(trials)]


def test_replay_restores_the_sampler():
    pytest.importorskip("optuna")
    for k in (1, 2, 3):
        s = S.Sampler(seed=12345, n_startup_trials=2, directions=["minimize", "minimize"])
        assert s.replay(records_of(ARA_Q05[:k]), ARA_24) == k
        _, draws = s.ask(ARA_24)
        assert draws == ARA_Q05[k][0], k
    recs = records_of(ARA_Q05[:3])
    x = recs[1]["draws"]["steer_bad_behavior_weight"]
    recs[1]["draws"]["steer_bad_behavior_weight"] = math.nextafter(x, 1.0)
    assert recs[1]["draws"]["steer_bad_behavior_weight"] != x
    s = S.Sampler(seed=12345, n_startup_trials=2, directions=["minimize", "minimize"])
    with pytest.raises(S.ReplayMismatch) as e:
        s.replay(recs, ARA_24)
    assert e.value.index == 2 and e.value.now == ARA_Q05[1][0]


# ------------------------------------------------------------------ 7. the Pareto front
def test_the_pareto_front():
    both = [{"index": 0, "name": "a", "direction": "minimize"}, {"index": 1, "name": "b", "direction": "minimize"}]
    vals = [(0.0, 0.5), (0.0, 0.3), (0.5, 0.1), (0.5, 0.1), (1.0, 0.05), (0.2, 0.6)]     # A B C D E F
    recs = [{"index": i + 1, "state": "complete", "values": list(v)} for i, v in enumerate(vals)]
    recs += [{"index": 7, "state": "failed", "values": None}, {"index": 8, "state": "stopped", "values": None}]
    assert DSR.front(recs, both) == [2, 3, 4, 5]                    # B C D E
    assert DSR.front(list(reversed(recs)), both) == [2, 3, 4, 5]
    mixed = [{"index": 0, "name": "a", "direction": "maximize"}, {"index": 1, "name": "b", "direction": "minimize"}]
    recs = [{"index": i + 1, "state": "complete", "values": list(v)} for i, v in enumerate([(0.9, 0.5), (0.8, 0.3), (0.9, 0.6)])]
    assert DSR.front(recs, mixed) == [2, 1]                         # B A
    assert DSR.front([], both) == []


# ------------------------------------------------------------------ 22. the digest
def pieces(value=1.0, name="blocks.0.mlp.W_out", region=(range(0, 2), range(0, 3))):
    A = torch.tensor([[1.0], [2.0]])
    B = torch.tensor([[value, 0.5, -0.25]])
    blk = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return {name: [Dl.Piece("lowrank", region, A=A.clone(), B=B.clone())],
            "blocks.1.attn.W_O": [Dl.Piece("block", (0, range(0, 2), range(0, 3)), values=blk.clone())]}


def test_the_digest():
    d = DSR.piece_digest(pieces())
    assert len(d) == 64 and d == DSR.piece_digest(pieces())
    assert DSR.piece_digest(pieces(value=float(torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))))) != d
    assert DSR.piece_digest(pieces(region=(range(1, 3), range(0, 3)))) != d
    assert DSR.piece_digest(pieces(name="blocks.1.mlp.W_out")) != d
    zero = pieces()
    zero["blocks.1.attn.W_O"].append(Dl.Piece("block", (1, range(0, 2), range(0, 3)), values=torch.zeros(2, 3)))
    zero["blocks.0.attn.W_O"] = [Dl.Piece("lowrank", (0, range(0, 2), range(0, 3)), A=torch.zeros(2, 1), B=torch.ones(1, 3))]
    assert DSR.piece_digest(zero) == d
    assert DSR.piece_digest({}) == DSR.piece_digest({"x": [Dl.Piece("block", (range(0, 1),), values=torch.zeros(1))]})


# ------------------------------------------------------------------ 12. a stop inside the script (the interpreter)
GOOD, BAD = ["abc", "hello", "fox"], ["xyz", "qqq", "zzz"]
ARA_SMALL = {"lora_rank": 2, "n_optimization_steps": 1, "max_iter": 2}


def same_deltas(a, b):
    assert sorted(a) == sorted(b)
    for name in a:
        assert len(a[name]) == len(b[name])
        for p, q in zip(a[name], b[name]):
            assert p.kind == q.kind and p.region == q.region
            for t in ("values", "A", "B", "coords"):
                x, y = getattr(p, t), getattr(q, t)
                assert (x is None and y is None) or torch.equal(x, y)


def test_a_stop_inside_the_script(tmp_path, monkeypatch):
    from test_steering import make_ws
    from wsp import directions_scores as DS
    from wsp import wscript
    ws = make_ws(tmp_path / "rnd")
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    script = DR.script_for({"method": "ara", "direction": "m", **ARA_SMALL})
    plain = wscript.run_script(ws, script)
    assert len(plain["deltas"]) == 4                                  # 2 layers x 2 components were fitted
    calls = []
    real = wscript.Interp.progress

    def progress(self, window=None, windows=None):
        try:
            real(self, window, windows)
        except BaseException as e:
            calls.append((window, type(e)))
            raise
        calls.append((window, None))
    monkeypatch.setattr(wscript.Interp, "progress", progress)

    def stopped():                                                     # once the first component has been reported
        return any(w is not None and e is None for w, e in calls)
    with pytest.raises(wscript.ScriptStopped):
        wscript.run_script(ws, script, stopped=stopped)
    assert calls == [(None, None), (0, None), (1, wscript.ScriptStopped)]
    assert not issubclass(wscript.ScriptStopped, (wscript.ScriptError, RuntimeError))
    # Without stopped, and with one that never stops: the same pieces and output as before.
    for again in (wscript.run_script(ws, script), wscript.run_script(ws, script, stopped=lambda: False)):
        assert again["output"] == plain["output"]
        same_deltas(again["deltas"], plain["deltas"])
    with pytest.raises(wscript.ScriptError, match="syntax error"):
        wscript.run_script(ws, "x = (", stopped=lambda: True)
    with pytest.raises(wscript.ScriptStopped):
        wscript.run_script(ws, "x = 1", stopped=lambda: True)          # before the first statement
    with pytest.raises(wscript.ScriptStopped):
        DS.candidate(ws, script, stopped=lambda: True)                 # the scores' candidate lets it through
    deltas, report = DS.candidate(ws, script)
    same_deltas(deltas, plain["deltas"])
    assert ws.patches == []


# ------------------------------------------------------------------ searches on the tiny model: fixtures
SCORERS = [{"plugin": "KeywordRate", "optimization": "minimize", "prompts": ["ab", "cd"], "keyword_markers": ["e", "q"]},
           {"plugin": "KLDivergence", "optimization": "minimize", "prompts": ["ef", "gh"]}]


def tiny(tmp_path, name="rnd", module_io=False):
    from test_steering import make_ws
    ws = make_ws(tmp_path / name)
    DR.measure(ws, "m", GOOD, BAD, module_io=module_io)
    return ws


def search_body(name="s1", method="ablation", **kw):
    mod = {"plugin": "Abliteration", "direction": "m"} if method == "ablation" else \
        {"plugin": "ARA", "direction": "m", **ARA_SMALL}
    return {"name": name, "modifiers": [mod], "scorers": [dict(x) for x in SCORERS], "n_trials": 4,
            "n_startup_trials": 1, "seed": 7, "max_response_length": 3, **kw}


def make_search(ws, **kw):
    return DSR.create(ws, DSR.check_search(ws, search_body(**kw)))


def sdir(ws, name):
    return os.path.join(ws.dir, "directions", "searches", name)


def trial_path(ws, name, index):
    return os.path.join(sdir(ws, name), "trials", f"{index:06d}.json")


def read_json(path):
    import json
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    import json
    with open(path, "w") as f:
        json.dump(obj, f)


def hand_parameters(draws):
    """Heretic's transformation, written here from abliteration.py's suggest_parameters (anchored by test 2)."""
    wd = {}
    for c in BOTH:
        mw = max(0.0, draws[f"{c}.max_weight"])
        wd[c] = {"max_weight": mw, "max_weight_position": draws[f"{c}.max_weight_position"],
                 "min_weight": draws[f"{c}.min_weight"] * mw, "min_weight_distance": draws[f"{c}.min_weight_distance"]}
    return {"direction": "m", "direction_index": None if draws["direction_scope"] == "per layer" else draws["direction_index"],
            "weight_distributions": wd, **ABL_SETTINGS}


def hand_front(recs):
    pool = [r for r in recs if r["state"] == "complete"]

    def dominates(u, t):
        a, b = u["values"], t["values"]
        return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])
    on = [t for t in pool if not any(dominates(u, t) for u in pool)]
    return [t["index"] for t in sorted(on, key=lambda t: (t["values"][0], t["values"][1], t["index"]))]


def patch_files(ws):
    d = os.path.join(ws.dir, "patches")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def session_of(tmp_path, name="rnd"):
    """A Session over the workspace directory ``tmp_path/name`` (made by ``tiny`` or ``make_ws``)."""
    from wsp.session import Session
    sess = Session(str(tmp_path))
    sess.open(name)
    return sess


def client(sess):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    return TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)


def events(q, kind="direction_search"):
    out = []
    while not q.empty():
        e = q.get_nowait()
        if e.get("type") == kind:
            out.append(e)
    return out


def join_search_threads(timeout=120):
    """Wait for every search job thread of this process to end (no thread outlives a test)."""
    import threading
    for t in threading.enumerate():
        if t.name == "direction-search":
            t.join(timeout)
            assert not t.is_alive(), "a search job thread is still running"


# ------------------------------------------------------------------ 8. the model-free checks
def test_requests_are_checked_before_any_model_is_loaded(tmp_path, monkeypatch):
    import wsp.runner
    ws = tiny(tmp_path)
    DR.measure(ws, "mio", GOOD, BAD, module_io=True)
    monkeypatch.setattr(S, "version", lambda: "5.0.0")
    make_search(ws, name="s0")                         # for C5
    ws._runner = None
    built = []

    def refuse(w):
        built.append(w.name)
        raise AssertionError("the runner was built for a request that should have been refused first")
    monkeypatch.setattr(wsp.runner, "make_runner", refuse)
    b = search_body
    abl = {"plugin": "Abliteration", "direction": "m"}
    ara = {"plugin": "ARA", "direction": "mio"}
    kw = SCORERS[0]
    kl = SCORERS[1]
    fields = "name, modifiers, scorers, n_trials, n_startup_trials, seed, chat, system, prefix, chat_options, " \
             "max_response_length, start and stream"
    seed_msg = "seed must be null or an integer from 0 to 4294967295 (the range of Optuna's sampler seed)"
    cases = [
        ([], "the request must be an object"),                                                          # C1
        ({**b(), "x": 1}, f"x: unknown field; the fields are {fields}"),                                 # C2
        ({k: v for k, v in b().items() if k != "name"}, "name: required"),                              # C3
        ({k: v for k, v in b().items() if k != "modifiers"}, "modifiers: required"),
        ({k: v for k, v in b().items() if k != "scorers"}, "scorers: required"),
        (b(name="-x"), "name must be 1..64 characters from A-Z, a-z, 0-9, '.', '_' and '-', starting with a letter or "
                       "digit"),                                                                         # C4
        (b(name="x" * 65), "name must be 1..64 characters"),
        (b(name="s0"), "a search named 's0' exists; run more of its trials with POST /api/directions/searches/s0/run, "
                       "or delete it first"),                                                            # C5
        (b(modifiers=[]), "modifiers must list exactly one modifier; using multiple modifiers is not yet supported (as "
                          "in Heretic)"),                                                                # C6
        (b(modifiers=[abl, abl]), "modifiers must list exactly one modifier"),
        (b(modifiers=[1]), "modifiers[0] must be an object"),                                            # C7
        (b(modifiers=[{"plugin": "Other", "direction": "m"}]),
         "modifiers[0].plugin must be one of Abliteration, ARA (or their heretic.modifiers import paths)"),  # C8
        (b(modifiers=[{"direction": "m"}]), "modifiers[0].plugin must be one of Abliteration, ARA"),
        (b(modifiers=[{**abl, "lora_rank": 3}]),
         "modifiers[0].lora_rank: unknown field; the fields of Abliteration are plugin, instance_name, direction, "
         "orthogonalize_direction, row_normalization, full_normalization_lora_rank"),                  # C9
        (b(modifiers=[{**ara, "row_normalization": "full"}]),
         "modifiers[0].row_normalization: unknown field; the fields of ARA are plugin, instance_name, direction, "
         "preserve_row_magnitudes, lora_rank, n_optimization_steps, learning_rate, max_iter, history_size, print_loss"),
        (b(modifiers=[{"plugin": "ARA"}]), "modifiers[0].direction: required (the name of a saved measurement)"),  # C10
        (b(modifiers=[{**abl, "good_prompts": ["a"]}]),
         "modifiers[0].good_prompts: the IDE measures the prompts first (POST /api/directions/measure); name that "
         "measurement in modifiers[0].direction"),                                                       # C11
        (b(modifiers=[{**ara, "bad_prompts": ["a"]}]), "modifiers[0].bad_prompts: the IDE measures the prompts first"),
        (b(modifiers=[{**abl, "winsorization_quantile": 0.9}]),
         "modifiers[0].winsorization_quantile: winsorization is part of the measurement (POST /api/directions/measure); "
         "the search uses the measurement as it was made"),                                              # C12
        (b(modifiers=[{**abl, "instance_name": 3}]), "modifiers[0].instance_name must be text"),        # C13
        (b(modifiers=[{**abl, "instance_name": " "}]), "modifiers[0].instance_name: cannot be empty or whitespace"),
        (b(modifiers=[{**abl, "instance_name": "a.b"}]), "modifiers[0].instance_name: '.' is not allowed"),
        (b(modifiers=[{**abl, "instance_name": "a b"}]), "modifiers[0].instance_name: whitespace is not allowed"),
        (b(modifiers=[{**abl, "instance_name": "x" * 65}]), "modifiers[0].instance_name must be at most 64 characters"),
        (b(modifiers=[{**ara, "print_loss": "yes"}]), "modifiers[0].print_loss must be true or false"),  # C14
        (b(modifiers=[{**abl, "row_normalization": "x"}]), "modifiers[0].row_normalization: must be one of none, pre, "
                                                           "full"),                                      # C15
        (b(modifiers=[{**abl, "full_normalization_lora_rank": 99}]),
         "modifiers[0].full_normalization_lora_rank: must be an integer from 1 to 64"),
        (b(modifiers=[{**ara, "lora_rank": 0}]), "modifiers[0].lora_rank: must be an integer from 1 to 256"),
        (b(modifiers=[{**abl, "direction": "-x"}]), "modifiers[0].direction: name must be 1..64 characters"),
        (b(scorers=[{k: v for k, v in kw.items() if k != "optimization"}, kl]),
         "scorers[0].optimization: required ('minimize', 'maximize' or 'none'; Heretic requires it)"),   # C16
        (b(scorers=[]), "scorers must list 1..8 scorers"),                                               # C17
        (b(scorers=[kl] * 9), "scorers must list 1..8 scorers"),
        (b(scorers=[1]), "scorers[0] must be an object"),
        (b(scorers=[kw, {**kl, "plugin": "Other"}]),
         "scorers[1].plugin must be one of KeywordRate, KLDivergence, ResponseLength (or their heretic.scorers import "
         "paths)"),
        (b(scorers=[kw, {**kl, "score_name": "x"}]),
         "scorers[1].score_name: unknown field; the fields of KLDivergence are plugin, instance_name, optimization, "
         "prompts"),
        (b(scorers=[{**kw, "prompts": []}]), "scorers[0] must list 1..1024 prompts"),
        (b(scorers=[{**kw, "optimization": "up"}]), "scorers[0].optimization must be 'minimize', 'maximize' or 'none'"),
        (b(scorers=[{**kw, "prompts": ["ab1"]}]), "scorers[0] prompt 1: '1' is not in the vocabulary"),
        (b(scorers=[{**kw, "optimization": "none"}, {**kl, "optimization": "none"}]),
         'No optimization objectives configured. At least one scorer must set optimization to "maximize" or '
         '"minimize".'),                                                                                 # C18
        (b(n_trials=0), "n_trials must be an integer from 1 to 10000"),                                 # C19
        (b(n_trials=10001), "n_trials must be an integer from 1 to 10000"),
        (b(n_trials=True), "n_trials must be an integer from 1 to 10000"),
        (b(n_startup_trials=-1), "n_startup_trials must be an integer from 0 to 10000"),               # C20
        (b(n_startup_trials=1.0), "n_startup_trials must be an integer from 0 to 10000"),
        (b(seed=2 ** 32), seed_msg), (b(seed=-1), seed_msg), (b(seed=1.5), seed_msg),                   # C21
        (b(start="yes"), "start must be true or false"),                                                # C22
        (b(chat=True), "'as a chat message' needs the model's own chat template: this built-in model has no chat "
                       "template"),                                                                      # C23
        (b(system="be nice"), "system and prefix apply to prompts read as a chat message; set chat to true or leave "
                              "them out"),
        (b(modifiers=[{**abl, "direction": "nope"}]), "modifiers[0]: no direction named 'nope'"),        # C24
        (b(modifiers=[{**ara, "direction": "m"}]),
         "modifiers[0]: direction 'm' has no complete module inputs and outputs: measure it again with module_io"),
        (b(modifiers=[{**ara, "lora_rank": 25}]),
         "modifiers[0]: lora_rank: 25 is above min(D_in, d_model) = 24 of layer 0 attn.o_proj"),
        (b(max_response_length=0), "max_response_length must be an integer from 1 to 2048"),           # C27
        (b(max_response_length=2049), "max_response_length must be an integer from 1 to 2048"),
    ]
    for body, message in cases:
        with pytest.raises(DSR.SearchError) as e:
            DSR.check_search(ws, body)
        assert message in str(e.value), (body, str(e.value))
    assert DSR.check_search(ws, b(modifiers=[{**abl, "plugin": "heretic.modifiers.abliteration.Abliteration"}]))
    assert DSR.check_search(ws, b(modifiers=[{**ara, "plugin": "heretic.modifiers.ara.ARA", "lora_rank": 2}]))
    # C25: the disk guard at creation (the KL reference's own guard passes at this size).
    paths = DR._paths(ws, "m")
    need = (8 << 20) + os.path.getsize(paths["meta"]) + os.path.getsize(paths["prompts"])
    free = DR.DISK_RESERVE + 1_000_000
    monkeypatch.setattr(DR, "_free_bytes", lambda path: free)
    with pytest.raises(DSR.SearchError) as e:
        DSR.check_search(ws, b())
    assert str(e.value) == (f"the search needs {need:,} bytes on disk; the workspace's file system has {free:,} bytes "
                            f"free, of which {DR.DISK_RESERVE:,} stay free")
    monkeypatch.undo()
    monkeypatch.setattr(wsp.runner, "make_runner", refuse)
    # C26: Optuna missing, or another major version.
    monkeypatch.setattr(S, "version", lambda: None)
    with pytest.raises(DSR.SearchError) as e:
        DSR.check_search(ws, b())
    assert str(e.value) == ("the automatic search needs Optuna 5 (the sampler Heretic uses), which is not installed "
                            "here: python -m pip install -r requirements-search.txt")
    monkeypatch.setattr(S, "version", lambda: "4.6.0")
    with pytest.raises(DSR.SearchError) as e:
        DSR.check_search(ws, b())
    assert str(e.value) == "the automatic search needs Optuna 5 (Heretic's optuna~=5.0); this environment has Optuna 4.6.0"
    with pytest.raises(DR.DirectionError, match=r"^name must not be 'searches' \(the search routes use it\)$"):
        DR.check_name("Searches")
    assert built == [] and ws._runner is None
    # The same refusals through the routes, each a 400 before any model is loaded.
    sess = session_of(tmp_path)
    c = client(sess)
    monkeypatch.setattr(S, "version", lambda: "5.0.0")

    def refused(path, body, message, status=400, prefix="SearchError: "):
        r = c.post(path, json=body)
        assert r.status_code == status and r.json()["error"].startswith(prefix) and message in r.json()["error"], \
            (body, r.status_code, r.text)
    for body, message in cases:
        refused("/api/directions/searches", body, message)
    refused("/api/directions/searches", b(stream="no spaces"), "stream must be 1-64 characters", prefix="ValueError: ")
    monkeypatch.setattr(DR, "_free_bytes", lambda path: free)
    refused("/api/directions/searches", b(), f"the search needs {need:,} bytes on disk")
    monkeypatch.setattr(DR, "_free_bytes", lambda path: DR.DISK_RESERVE + (1 << 40))
    refused("/api/directions/searches/s0/run", {"x": 1}, "x: unknown field; the fields are n_additional_trials, "
                                                         "accept_changed_libraries and stream")
    refused("/api/directions/searches/s0/run", {"n_additional_trials": 0},
            "n_additional_trials must be an integer from 1 to 10000")
    refused("/api/directions/searches/s0/stop", {"when": "later"}, "when must be 'now' or 'after_trial'")
    refused("/api/directions/searches/nope/run", {}, "no search named 'nope'", status=404, prefix="")
    monkeypatch.setattr(S, "version", lambda: None)
    refused("/api/directions/searches", b(name="s9"), "the automatic search needs Optuna 5 (the sampler Heretic uses)")
    assert built == [] and sess.ws._runner is None and not os.path.exists(sdir(ws, "s9"))


# ------------------------------------------------------------------ 9. without Optuna
def test_optuna_absent(tmp_path, monkeypatch):
    ws = tiny(tmp_path)
    monkeypatch.setattr(S, "version", lambda: "5.0.0")
    spec = DSR.check_search(ws, search_body())
    s = DSR.create(ws, spec)
    missing = ("the automatic search needs Optuna 5 (the sampler Heretic uses), which is not installed here: python -m "
               "pip install -r requirements-search.txt")
    monkeypatch.setattr(S, "version", lambda: None)
    for call in (lambda: DSR.check_search(ws, search_body(name="s2")), lambda: DSR.create(ws, {**spec, "name": "s2"}),
                 lambda: DSR.run(ws, "s1")):
        with pytest.raises(DSR.SearchError) as e:
            call()
        assert str(e.value) == missing
    assert [x["name"] for x in DSR.entries(ws)] == ["s1"] and "error" not in DSR.entries(ws)[0]
    d = DSR.describe(ws, "s1")
    assert d["search"]["id"] == s["id"] and d["trials"] == [] and d["front"] == []
    assert DSR.front(DSR.trial_records(ws, "s1"), s["objectives"]) == []
    assert s["sampler"]["optuna"] == "5.0.0" and s["sampler"]["validated"] is True
    monkeypatch.setattr(S, "version", lambda: "4.6.0")
    with pytest.raises(DSR.SearchError, match=r"this environment has Optuna 4\.6\.0$"):
        DSR.run(ws, "s1")
    assert not os.path.isdir(os.path.join(sdir(ws, "s1"), "trials")) or os.listdir(os.path.join(sdir(ws, "s1"), "trials")) == []
    # Through the routes: creating and running answer C26; listing, reading and the front work without Optuna.
    sess = session_of(tmp_path)
    c = client(sess)
    monkeypatch.setattr(S, "version", lambda: None)
    for path, body in (("/api/directions/searches", search_body(name="s2")), ("/api/directions/searches/s1/run", {})):
        r = c.post(path, json=body)
        assert r.status_code == 400 and r.json()["error"] == f"SearchError: {missing}"
    r = c.get("/api/directions/searches")
    assert r.status_code == 200 and [x["name"] for x in r.json()["searches"]] == ["s1"] and r.json()["job"] is None
    r = c.get("/api/directions/searches/s1")
    assert r.status_code == 200 and r.json()["search"]["id"] == s["id"] and r.json()["trials"] == []
    assert c.get("/api/directions/searches/s1/front").json() == {"objectives": s["objectives"], "front": []}
    assert c.get("/api/directions/searches/s1/trials/1").status_code == 404
    monkeypatch.setattr(S, "version", lambda: "4.6.0")
    r = c.post("/api/directions/searches/s1/run", json={})
    assert r.status_code == 400 and r.json()["error"].endswith("this environment has Optuna 4.6.0")
    assert sess.search_job is None


# ------------------------------------------------------------------ 10. a whole search
def test_a_search_on_the_tiny_model(tmp_path):
    pytest.importorskip("optuna")
    from wsp import directions_scores as DS
    ws = tiny(tmp_path)
    other = tiny(tmp_path, "other")
    patches, version = patch_files(ws), ws.version
    s = make_search(ws)
    assert s["identity"]["scoring"] is None and s["seed_from"] == "request" and s["n_trials"] == 4
    seen = []
    out = DSR.run(ws, "s1", event=lambda kind, **data: seen.append((kind, data)))
    assert out["state"] == "finished" and out["ran"] == [1, 2, 3, 4] and out["trials"] == {"complete": 4}
    assert [d["trial"] for k, d in seen] == [1, 2, 3, 4] and {k for k, _ in seen} == {"trial"}
    d = sdir(ws, "s1")
    assert sorted(os.listdir(d)) == ["measurement.json", "measurement.prompts.json", "search.json", "trials"]
    assert sorted(os.listdir(os.path.join(d, "trials"))) == [f"{i:06d}.json" for i in (1, 2, 3, 4)]
    assert read_json(os.path.join(d, "measurement.prompts.json")) == {"good": GOOD, "bad": BAD}
    recs = DSR.trial_records(ws, "s1")
    assert [r["kind"] for r in recs] == ["startup", "tpe", "tpe", "tpe"]
    for r in recs:
        x = r["draws"]                      # the tiny model has 2 layers: L = 1 (bounds as Heretic computes them)
        assert x["direction_scope"] in ("global", "per layer") and 0.4 <= x["direction_index"] <= 0.9
        assert 0.8 <= x["attn.o_proj.max_weight"] <= 1.5 and -0.25 <= x["mlp.down_proj.max_weight"] <= 1.5
        for c in BOTH:
            assert 0.6 <= x[f"{c}.max_weight_position"] <= 1.0 and 0.0 <= x[f"{c}.min_weight"] <= 1.0
            assert x[f"{c}.min_weight_distance"] == 1.0
        assert r["script"] == DR.script_for(hand_parameters(x))
        direct = DS.score(other, {"state": "candidate", "script": r["script"], "max_response_length": 3, "scores": SCORERS})
        assert r["values"] == [direct["scores"][0]["value"], direct["scores"][1]["value"]]
        assert [x["display"] for x in r["scores"]] == [y["display"] for y in direct["scores"]]
        assert all("prompt" not in row and "response" in row for row in r["scores"][0]["rows"])
        assert r["candidate"]["digest"] and len(r["candidate"]["digest"]) == 64 and r["state"] == "complete"
        assert r["environment"]["libraries"]["torch"] == str(torch.__version__)
    assert out["front"] == hand_front(recs) == DSR.describe(ws, "s1")["front"]
    assert ws.patches == [] and ws.version == version and patch_files(ws) == patches
    again = read_json(os.path.join(d, "search.json"))
    assert again["identity"]["scoring"]["runner"]["class"] == "FunctionalRunner"


# ------------------------------------------------------------------ 11. stop and resume
def test_stop_and_resume_give_the_uninterrupted_search(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    from wsp import directions_scores as DS
    ws = tiny(tmp_path)
    make_search(ws, name="a")
    assert DSR.run(ws, "a")["state"] == "finished"
    make_search(ws, name="b")
    answered = []
    real = DS.evaluate

    def evaluate(*a, **k):
        out = real(*a, **k)
        answered.append(out)
        return out
    monkeypatch.setattr(DS, "evaluate", evaluate)
    flag = [False]

    def progress(index, phase, done, total):
        if index == 2:
            flag[0] = True
    out = DSR.run(ws, "b", progress=progress, stopped=lambda: flag[0])
    assert out["state"] == "stopped" and out["ran"] == [1, 2] and answered[-1].get("stopped")
    stop = answered[-1]
    rec = read_json(trial_path(ws, "b", 2))
    assert rec["state"] == "stopped" and rec["values"] is None and rec["done"] == stop["done"] == 1
    finished = []
    out = DSR.run(ws, "b", after_trial=lambda: 3 in finished,
                  event=lambda kind, **d: finished.append(d["trial"]))
    assert out == {**out, "state": "stopped", "reason": "stopped after a trial", "ran": [2, 3]}
    assert DSR.run(ws, "b")["ran"] == [4]
    a, b = DSR.trial_records(ws, "a"), DSR.trial_records(ws, "b")
    assert [r["draws"] for r in b] == [r["draws"] for r in a]
    assert [r["values"] for r in b] == [r["values"] for r in a]
    assert [r["candidate"]["digest"] for r in b] == [r["candidate"]["digest"] for r in a]
    assert all(r["state"] == "complete" for r in b)
    assert [len(r["attempts"]) for r in b] == [0, 1, 0, 0]
    att = b[1]["attempts"][0]
    assert att["state"] == "stopped" and att["done"] == stop["done"] and att["total"] == stop["total"]
    assert att["reason"] == "stopped by the person"


# ------------------------------------------------------------------ 12. a stop inside a trial's script
def test_a_stop_inside_a_trial(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    from wsp import wscript
    ws = tiny(tmp_path, module_io=True)
    make_search(ws, method="ara", n_trials=2)
    calls = []
    real = wscript.Interp.progress

    def progress(self, window=None, windows=None):
        try:
            real(self, window, windows)
        except BaseException as e:
            calls.append((window, type(e)))
            raise
        calls.append((window, None))
    monkeypatch.setattr(wscript.Interp, "progress", progress)
    out = DSR.run(ws, "s1", stopped=lambda: any(w is not None and e is None for w, e in calls))
    assert out["state"] == "stopped" and out["ran"] == [1]
    assert calls[-1] == (1, wscript.ScriptStopped)
    rec = read_json(trial_path(ws, "s1", 1))
    assert rec["state"] == "stopped" and rec["done"] == 0 and rec["total"] is None and rec["error"] is None
    assert rec["script"].startswith("ablate_ara(") and "lora_rank=2" in rec["script"]
    monkeypatch.undo()
    out = DSR.run(ws, "s1")
    recs = DSR.trial_records(ws, "s1")
    assert out["state"] == "finished" and [r["state"] for r in recs] == ["complete", "complete"]
    assert recs[0]["draws"] == rec["draws"] and recs[0]["attempts"][0]["state"] == "stopped"


# ------------------------------------------------------------------ 13. failed trials
def test_failed_trials(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    from wsp import directions_scores as DS
    ws = tiny(tmp_path)
    real = DS.candidate
    calls = []

    def once(w, script, scope=None, *, stopped=None):
        calls.append(script)
        if len(calls) == 1:
            raise DR.DirectionError("x")
        return real(w, script, scope, stopped=stopped)
    monkeypatch.setattr(DS, "candidate", once)
    make_search(ws, n_trials=3)
    out = DSR.run(ws, "s1")
    recs = DSR.trial_records(ws, "s1")
    assert out["state"] == "finished" and [r["state"] for r in recs] == ["failed", "complete", "complete"]
    assert recs[0]["error"] == "DirectionError: x" and recs[0]["values"] is None
    assert [r["kind"] for r in recs] == ["startup", "startup", "tpe"]      # a failed trial is no startup trial
    assert out["trials"] == {"failed": 1, "complete": 2}

    def always(*a, **k):
        raise DR.DirectionError("x")
    monkeypatch.setattr(DS, "candidate", always)
    make_search(ws, name="s2", n_trials=6)
    out = DSR.run(ws, "s2")
    assert out["state"] == "failed" and out["ran"] == [1, 2, 3]
    assert out["reason"] == ("3 trials in a row failed (the last: DirectionError: x); the search stopped. Fix the cause "
                             "and run it again")
    assert [r["state"] for r in DSR.trial_records(ws, "s2")] == ["failed"] * 3


# ------------------------------------------------------------------ 14. an interrupted trial
def test_an_interrupted_trial_is_rerun(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    from wsp import directions_scores as DS
    from wsp.runtime_events import RuntimeRefusal, event
    ws = tiny(tmp_path)
    make_search(ws, n_trials=3)
    real = DS.evaluate
    n = [0]
    record = event("worker_exited", "worker_killed", "The model worker was lost.")

    def evaluate(*a, **k):
        n[0] += 1
        if n[0] == 2:
            raise RuntimeRefusal(record)
        return real(*a, **k)
    monkeypatch.setattr(DS, "evaluate", evaluate)
    out = DSR.run(ws, "s1")
    assert out["state"] == "failed" and out["reason"] == "The model worker was lost." and out["ran"] == [1, 2]
    rec = read_json(trial_path(ws, "s1", 2))
    assert rec["state"] == "interrupted" and rec["event"] == record and rec["error"] == "The model worker was lost."
    draws = rec["draws"]
    write_json(trial_path(ws, "s1", 2), {**rec, "state": "running", "error": None, "event": None})   # a killed process
    assert [t["state"] for t in DSR.describe(ws, "s1")["trials"]] == ["complete", "interrupted"]
    assert DSR.describe(ws, "s1", full=True)["trials"][1]["state"] == "interrupted"
    monkeypatch.setattr(DS, "evaluate", real)
    out = DSR.run(ws, "s1")
    recs = DSR.trial_records(ws, "s1")
    assert out["state"] == "finished" and out["ran"] == [2, 3] and recs[1]["draws"] == draws
    assert [a["state"] for a in recs[1]["attempts"]] == ["interrupted"]
    assert recs[1]["attempts"][0]["reason"] == "the process running the trial ended before it finished"
    # The refusal's own attempt, kept when a refused trial is rerun.
    make_search(ws, name="s2", n_trials=2)
    n[0] = 0
    monkeypatch.setattr(DS, "evaluate", evaluate)
    DSR.run(ws, "s2")
    monkeypatch.setattr(DS, "evaluate", real)
    DSR.run(ws, "s2")
    att = DSR.trial_records(ws, "s2")[1]["attempts"]
    assert [(a["state"], a["reason"]) for a in att] == [("interrupted", "The model worker was lost.")]


# ------------------------------------------------------------------ 15. what a changed workspace refuses
def test_what_a_changed_workspace_refuses(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    from test_steering import make_ws
    from wsp.session import Session
    make_ws(tmp_path / "rnd")
    sess = Session(str(tmp_path))
    sess.open("rnd")
    ws = sess.ws
    DR.measure(ws, "m", GOOD, BAD)
    make_search(ws, n_trials=2)
    assert DSR.run(ws, "s1")["state"] == "finished"
    assert DSR.describe(ws, "s1")["compatible"] == {"run": None, "apply": None,
                                                     "checked_without_model": ["model", "current", "measurement", "libraries"]}
    pid = sess.apply("mlp(1).W_out *= 0.5", "half")["patch"]
    current = f"the enabled patches changed since search 's1' was made (then: none; now: {pid})"
    with pytest.raises(DSR.SearchError) as e:
        DSR.run(ws, "s1", n_additional_trials=1)
    assert str(e.value) == f"{current}; switch them back, or start a new search on these weights"
    assert DSR.describe(ws, "s1")["compatible"]["apply"] == \
        f"{current}; its script would now make another edit than the one it scored"
    ws.set_patch(pid, enabled=False)
    assert DSR.run(ws, "s1", n_additional_trials=1)["ran"] == [3]
    real = str(torch.__version__)
    tf = DSR.environment()["libraries"]["transformers"]
    monkeypatch.setattr(torch, "__version__", "9.9.9")
    libraries = (f"search 's1' was scored with torch {real} and transformers {tf}; this environment has torch 9.9.9 and "
                 f"transformers {tf}. Scores, and an ARA trial's fit, can differ across versions; pass "
                 f"accept_changed_libraries to continue anyway (each trial records its versions)")
    with pytest.raises(DSR.SearchError) as e:
        DSR.run(ws, "s1", n_additional_trials=1)
    assert str(e.value) == libraries
    assert DSR.describe(ws, "s1")["compatible"] == {"run": libraries, "apply": None,
                                                     "checked_without_model": ["model", "current", "measurement", "libraries"]}
    assert DSR.run(ws, "s1", n_additional_trials=1, accept_changed_libraries=True)["ran"] == [4]
    changes = read_json(os.path.join(sdir(ws, "s1"), "search.json"))["environment_changes"]
    assert len(changes) == 1 and changes[0]["from_index"] == 4 and changes[0]["environment"]["libraries"]["torch"] == "9.9.9"
    assert DSR.trial_records(ws, "s1")[3]["environment"]["libraries"]["torch"] == "9.9.9"
    numpy_version = read_json(os.path.join(sdir(ws, "s1"), "search.json"))["sampler"]["numpy"]
    monkeypatch.setattr(S, "version", lambda: "5.0.1")                    # the draws still reproduce: accepted
    assert DSR.run(ws, "s1", n_additional_trials=1)["ran"] == [5]
    path = trial_path(ws, "s1", 1)
    rec = read_json(path)
    edited = dict(rec["draws"], direction_index=math.nextafter(rec["draws"]["direction_index"], 1.0))
    write_json(path, {**rec, "draws": edited})
    with pytest.raises(DSR.SearchError) as e:
        DSR.run(ws, "s1", n_additional_trials=1)
    assert str(e.value) == (f"the sampler does not reproduce trial 1 of search 's1' (made with Optuna 5.0.0 and numpy "
                            f"{numpy_version}; this environment has Optuna 5.0.1 and numpy {numpy_version}): its next "
                            f"draws would not be Heretic's. Start a new search with the same seed")
    write_json(path, rec)
    DR.measure(ws, "m", GOOD + ["dog"], BAD, replace=True)
    measurement = "measurement 'm' changed since search 's1' was made (measured again or deleted)"
    with pytest.raises(DSR.SearchError) as e:
        DSR.run(ws, "s1", n_additional_trials=1)
    assert str(e.value) == f"{measurement}; start a new search"
    assert DSR.describe(ws, "s1")["compatible"]["apply"] == \
        f"{measurement}; its script would now make another edit than the one it scored"
    assert len(DSR.trial_records(ws, "s1")) == 5


# ------------------------------------------------------------------ 16. more trials later
def test_more_trials_later(tmp_path):
    pytest.importorskip("optuna")
    ws = tiny(tmp_path)
    make_search(ws, name="s3", n_trials=3)
    make_search(ws, name="s5", n_trials=5)
    assert DSR.run(ws, "s3")["ran"] == [1, 2, 3]
    with pytest.raises(DSR.SearchError) as e:
        DSR.run(ws, "s3")
    assert str(e.value) == "search 's3' has run all its 3 trials; ask for more with n_additional_trials"
    with pytest.raises(DSR.SearchError, match="^n_additional_trials must be an integer from 1 to 10000$"):
        DSR.run(ws, "s3", n_additional_trials=10001)
    with pytest.raises(DSR.SearchError, match="^n_additional_trials: search 's3' would have 10003 trials, more than 10000$"):
        DSR.run(ws, "s3", n_additional_trials=10000)
    assert DSR.run(ws, "s3", n_additional_trials=2)["ran"] == [4, 5]
    assert read_json(os.path.join(sdir(ws, "s3"), "search.json"))["n_trials"] == 5
    assert DSR.run(ws, "s5")["ran"] == [1, 2, 3, 4, 5]
    a, b = DSR.trial_records(ws, "s3"), DSR.trial_records(ws, "s5")
    assert [r["draws"] for r in a] == [r["draws"] for r in b] and [r["values"] for r in a] == [r["values"] for r in b]


# ------------------------------------------------------------------ 24. damaged records
def test_records_and_damaged_files(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    ws = tiny(tmp_path)
    make_search(ws, n_trials=3)
    DSR.run(ws, "s1")
    path = trial_path(ws, "s1", 2)
    with open(path) as f:
        text = f.read()
    with open(path, "w") as f:
        f.write(text[:40])
    for call in (lambda: DSR.describe(ws, "s1"), lambda: DSR.run(ws, "s1", n_additional_trials=1)):
        with pytest.raises(DSR.SearchError, match=r"^search 's1': trial file 000002\.json cannot be read \("):
            call()
    with open(path, "w") as f:
        f.write(text)
    extra = trial_path(ws, "s1", 9)
    with open(extra, "w") as f:
        f.write(text)
    with pytest.raises(DSR.SearchError, match=r"^search 's1': its trial records are inconsistent \(.*000009\.json"):
        DSR.describe(ws, "s1")
    os.remove(extra)
    old, fresh = (os.path.join(sdir(ws, "s1"), f"search.json.{k}.tmp") for k in ("1", "2"))
    old_trial = os.path.join(sdir(ws, "s1"), "trials", "000004.json.3.tmp")
    for p in (old, fresh, old_trial):
        with open(p, "w") as f:
            f.write("{")
    for p in (old, old_trial):
        os.utime(p, (time.time() - 7200, time.time() - 7200))
    monkeypatch.setattr(DSR, "RESPONSE_CHARS", 1)
    assert DSR.run(ws, "s1", n_additional_trials=1)["ran"] == [4]
    assert not os.path.exists(old) and not os.path.exists(old_trial) and os.path.exists(fresh)
    rec = DSR.trial_records(ws, "s1")[3]
    # 2 KeywordRate responses of max_response_length 3 tokens, one character each in this model's vocabulary
    assert rec["responses_omitted"] == "6 characters of responses, above 1"
    assert all("response" not in row for row in rec["scores"][0]["rows"]) and len(rec["scores"][0]["rows"]) == 2
    make_search(ws, name="s2", n_trials=1)
    os.remove(os.path.join(sdir(ws, "s2"), "search.json"))
    listed = {x["name"]: x for x in DSR.entries(ws)}
    assert listed["s2"] == {"name": "s2", "error": "search 's2': its record cannot be read (the file is missing)"}
    assert "error" not in listed["s1"] and listed["s1"]["trials"] == {"complete": 4}
    DSR.delete(ws, "s2")
    assert not os.path.exists(sdir(ws, "s2")) and [x["name"] for x in DSR.entries(ws)] == ["s1"]
    with pytest.raises(DSR.SearchMissing, match="^no search named 's2'$"):
        DSR.describe(ws, "s2")


# ------------------------------------------------------------------ 25. the disk guard
def test_the_disk_guard(tmp_path, monkeypatch):
    ws = tiny(tmp_path)
    monkeypatch.setattr(S, "version", lambda: "5.0.0")
    paths = DR._paths(ws, "m")
    need = (8 << 20) + os.path.getsize(paths["meta"]) + os.path.getsize(paths["prompts"])
    free = DR.DISK_RESERVE + 1_000_000
    real = DR._free_bytes
    monkeypatch.setattr(DR, "_free_bytes", lambda path: free)
    with pytest.raises(DSR.SearchError) as e:
        make_search(ws)
    assert str(e.value) == (f"the search needs {need:,} bytes on disk; the workspace's file system has {free:,} bytes "
                            f"free, of which {DR.DISK_RESERVE:,} stay free")
    assert not os.path.exists(os.path.join(ws.dir, "directions", "searches"))
    monkeypatch.setattr(DR, "_free_bytes", real)
    make_search(ws)
    before = read_json(os.path.join(sdir(ws, "s1"), "search.json"))
    monkeypatch.setattr(DR, "_free_bytes", lambda path: free)
    with pytest.raises(DSR.SearchError) as e:
        DSR.run(ws, "s1")
    assert str(e.value) == (f"the search needs {8 << 20:,} bytes free for its next trial record; the workspace's file "
                            f"system has {free:,} bytes free, of which {DR.DISK_RESERVE:,} stay free")
    assert os.listdir(os.path.join(sdir(ws, "s1"), "trials")) == []
    assert read_json(os.path.join(sdir(ws, "s1"), "search.json")) == before


# ------------------------------------------------------------------ the routes' field checks
def test_the_route_bodies():
    assert DSR.run_body({}) == {"n_additional_trials": None, "accept_changed_libraries": False}
    assert DSR.run_body({"n_additional_trials": 2, "stream": "x"})["n_additional_trials"] == 2
    for body, message in (({"x": 1}, "x: unknown field; the fields are n_additional_trials, accept_changed_libraries and "
                                     "stream"),
                          ({"accept_changed_libraries": 1}, "accept_changed_libraries must be true or false"),
                          ({"n_additional_trials": 0}, "n_additional_trials must be an integer from 1 to 10000")):
        with pytest.raises(DSR.SearchError) as e:
            DSR.run_body(body)
        assert str(e.value) == message
    assert DSR.stop_body({}) == {"when": "now"} and DSR.stop_body({"when": "after_trial"}) == {"when": "after_trial"}
    for body, message in (({"x": 1}, "x: unknown field; the fields are when"),
                          ({"when": "later"}, "when must be 'now' or 'after_trial'")):
        with pytest.raises(DSR.SearchError) as e:
            DSR.stop_body(body)
        assert str(e.value) == message
    assert DSR.apply_body({}) == {"dry_run": False, "enforce_scope": False, "accept_different_edit": False, "message": None}
    for body, message in (({"x": 1}, "x: unknown field; the fields are dry_run, message, enforce_scope and "
                                     "accept_different_edit"),
                          ({"dry_run": 1}, "dry_run must be true or false"),
                          ({"enforce_scope": "no"}, "enforce_scope must be true or false"),
                          ({"accept_different_edit": None}, "accept_different_edit must be true or false"),
                          ({"message": "x" * 1001}, "message must be text of at most 1000 characters")):
        with pytest.raises(DSR.SearchError) as e:
            DSR.apply_body(body)
        assert str(e.value) == message


# ------------------------------------------------------------------ one search per workspace (the lock)
def test_one_search_at_a_time(tmp_path, monkeypatch):
    import threading
    from wsp.atlas import wiring
    ws = tiny(tmp_path)
    monkeypatch.setattr(S, "version", lambda: "5.0.0")
    make_search(ws)
    held, release = threading.Event(), threading.Event()

    def hold():
        with wiring.build_lock(os.path.join(ws.dir, "directions", "searches")):
            held.set()
            release.wait(10)
    t = threading.Thread(target=hold)
    t.start()
    held.wait(10)
    try:
        assert DSR.running(ws) == "this IDE"
        with pytest.raises(DSR.SearchBusy, match=r"^a search is already running on this workspace in this IDE; stop it "
                                                 r"first$"):
            DSR.run(ws, "s1")
        with pytest.raises(DSR.SearchBusy, match=r"^search 's1' is running; stop it first$"):
            DSR.delete(ws, "s1")
    finally:
        release.set()
        t.join(10)
    assert DSR.running(ws) is None


# ------------------------------------------------------------------ 23. the transport meter
def test_the_transport_meter_through_the_model_worker(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    from test_steering_worker import WspHarness, local_ws, remote
    from wsp import directions_scores as DS
    from wsp.remote_runner import RemoteRunner, TransferMeter
    from wsp.wscript import run_script
    h = WspHarness(tmp_path)
    h.load()
    here = local_ws(tmp_path)
    r, sent, received = remote(h, monkeypatch)
    h.ws._runtime_client.preflight_edit = lambda w, stored, scratch=0, cumulative=False: None
    DR.measure(h.ws, "m", GOOD, BAD)
    DR.measure(here, "m", GOOD, BAD)
    make_search(h.ws, n_trials=2)
    received.clear()
    out = DSR.run(h.ws, "s1")
    assert out["state"] == "finished" and r.meter is None and RemoteRunner.meter is None
    units = 2 + 2                        # the KeywordRate prompts (distinct) and the KLDivergence prompts
    recs = DSR.trial_records(h.ws, "s1")
    for rec in recs:
        t = rec["transport"]
        deltas = {k: [p for p in ps if not p.is_zero()] for k, ps in run_script(here, rec["script"])["deltas"].items()}
        deltas = {k: ps for k, ps in deltas.items() if ps}
        assert t["jobs"] == units and t["bytes_per_job"] == DS._delta_bytes(deltas) > 0
        assert t["bytes"] == units * t["bytes_per_job"]
        assert min(t["encode_seconds"], t["send_seconds"], t["worker_seconds"]) >= 0 and t["worker_seconds"] > 0
        assert t["seconds"] == t["encode_seconds"] + t["send_seconds"] + t["worker_seconds"]
        assert t["share"] == t["seconds"] / rec["seconds"]["total"]
    results = [hdr for kind, hdr, _ in received if kind == "result"]
    assert sum(1 for x in results if "extra_seconds" in x) == units * len(recs)      # only jobs that carried pieces
    assert sum(1 for x in results if "extra_seconds" not in x) == 2                  # the KL reference (base)
    d = DSR.describe(h.ws, "s1")["transport"]
    shares = sorted(x["transport"]["share"] for x in recs)
    assert d == {"trials": 2, "median_share": (shares[0] + shares[1]) / 2, "max_share": shares[1], "limit": 0.1,
                 "over": (shares[0] + shares[1]) / 2 > 0.1}
    # A runner without a meter sends the same frames as with one.
    b = {"state": "candidate", "script": recs[0]["script"], "max_response_length": 3, "scores": SCORERS}
    sent.clear()
    plain = DS.score(h.ws, b)
    without = list(sent)
    sent.clear()
    r.meter = TransferMeter()
    metered = DS.score(h.ws, b)
    assert sent == without and r.meter.jobs == units and plain["scores"] == metered["scores"]
    r.meter = None
    # In this process there is nothing to carry.
    make_search(here, n_trials=2)
    DSR.run(here, "s1")
    assert [x["transport"] for x in DSR.trial_records(here, "s1")] == [None, None]
    assert DSR.describe(here, "s1")["transport"] is None
    h.ws._runtime_client = None
    h.shutdown()


# ------------------------------------------------------------------ 26. importing the search does not import Optuna
def test_importing_the_search_does_not_import_optuna():
    out = subprocess.run([sys.executable, "-c", "import sys, wsp.directions_search; print('optuna' in sys.modules)"],
                         cwd=APP, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# ------------------------------------------------------------------ 27. a change between trials ends the run
def test_a_change_between_trials_ends_the_run(tmp_path):
    pytest.importorskip("optuna")
    from test_steering import make_ws
    from wsp.session import Session
    make_ws(tmp_path / "rnd")
    sess = Session(str(tmp_path))
    sess.open("rnd")
    ws = sess.ws
    DR.measure(ws, "m", GOOD, BAD)
    make_search(ws, name="u", n_trials=3)
    make_search(ws, name="s1", n_trials=3)
    made = []

    def commit_after_first(kind, **data):
        if data["trial"] == 1:
            made.append(sess.apply("mlp(1).W_out *= 0.5", "half")["patch"])
    out = DSR.run(ws, "s1", event=commit_after_first)
    pid = made[0]
    assert out["state"] == "failed" and out["ran"] == [1]
    assert out["reason"] == (f"the enabled patches changed since search 's1' was made (then: none; now: {pid}); switch "
                             f"them back, or start a new search on these weights")
    recs = DSR.trial_records(ws, "s1")
    assert [r["state"] for r in recs] == ["complete"]                  # trial 2 was never started
    ws.set_patch(pid, enabled=False)
    assert DSR.run(ws, "s1")["ran"] == [2, 3]
    assert DSR.run(ws, "u")["ran"] == [1, 2, 3]                        # uninterrupted, same seed, same weights
    a, b = DSR.trial_records(ws, "s1"), DSR.trial_records(ws, "u")
    assert [r["draws"] for r in a] == [r["draws"] for r in b] and [r["values"] for r in a] == [r["values"] for r in b]
    assert all(r["state"] == "complete" for r in a)
    # The measurement measured again between trials.
    make_search(ws, name="s2", n_trials=3)

    def measure_after_first(kind, **data):
        if data["trial"] == 1:
            DR.measure(ws, "m", GOOD + ["dog"], BAD, replace=True)
    out = DSR.run(ws, "s2", event=measure_after_first)
    assert out["state"] == "failed" and out["ran"] == [1]
    assert out["reason"] == "measurement 'm' changed since search 's2' was made (measured again or deleted); start a new search"
    assert [r["state"] for r in DSR.trial_records(ws, "s2")] == ["complete"]


# ------------------------------------------------------------------ 28. a rerun trial must draw what it drew
def test_a_rerun_trial_must_draw_what_it_drew(tmp_path):
    pytest.importorskip("optuna")
    ws = tiny(tmp_path)
    make_search(ws, n_trials=3)
    flag = [False]

    def progress(index, phase, done, total):
        if index == 2:
            flag[0] = True
    out = DSR.run(ws, "s1", progress=progress, stopped=lambda: flag[0])
    assert out["state"] == "stopped" and out["ran"] == [1, 2]
    path = trial_path(ws, "s1", 2)
    rec = read_json(path)
    assert rec["state"] == "stopped"
    x = rec["draws"]["direction_index"]
    write_json(path, {**rec, "draws": {**rec["draws"], "direction_index": math.nextafter(x, 1.0)}})
    d = os.path.join(sdir(ws, "s1"), "trials")

    def files():
        out = {}
        for fn in sorted(os.listdir(d)):
            with open(os.path.join(d, fn), "rb") as f:
                out[fn] = f.read()
        return out
    before = files()
    with pytest.raises(DSR.SearchError) as e:
        DSR.run(ws, "s1")
    assert str(e.value).startswith("the sampler does not reproduce trial 2 of search 's1' (made with Optuna ")
    assert files() == before


# ------------------------------------------------------------------ 29. the scoring rows of a changed workspace
def test_the_scoring_rows(tmp_path):
    pytest.importorskip("optuna")
    ws = tiny(tmp_path)
    make_search(ws, n_trials=2)
    assert DSR.run(ws, "s1")["state"] == "finished"
    path = os.path.join(sdir(ws, "s1"), "search.json")
    good = read_json(path)
    runner = good["identity"]["scoring"]["runner"]
    now = f"{runner['class']} {runner['dtype']} {runner['device']}"

    def refused(edit):
        s = read_json(path)
        edit(s)
        write_json(path, s)
        with pytest.raises(DSR.SearchError) as e:
            DSR.run(ws, "s1", n_additional_trials=1)
        assert len(DSR.trial_records(ws, "s1")) == 2
        write_json(path, good)
        return str(e.value)
    assert refused(lambda s: s["identity"]["scoring"].update(rendered_sha256="0" * 64)) == (
        "the scored prompts of search 's1' render differently now (the tokenizer or the chat template changed, or the "
        "search was scored before the IDE rendered prompts as Heretic does); start a new search")
    assert refused(lambda s: s["identity"]["scoring"]["runner"].update(dtype="float16")) == (
        f"search 's1' was scored on {runner['class']} float16 {runner['device']}; the model now runs on {now}, so its "
        f"scores would not be comparable")
    assert refused(lambda s: s["identity"].update(model={"other": "weights"})) == (
        "search 's1' was made on other base weights than this workspace has now; its trials do not describe this model: "
        "start a new search")
    assert DSR.run(ws, "s1", n_additional_trials=1)["ran"] == [3]


# ------------------------------------------------------------------ 30. a stop while the KL reference is computed
def test_a_stop_while_the_reference_is_computed(tmp_path):
    pytest.importorskip("optuna")
    ws = tiny(tmp_path)
    make_search(ws, n_trials=2)
    seen = []
    out = DSR.run(ws, "s1", progress=lambda index, *a: seen.append(index), stopped=lambda: bool(seen))
    assert out["state"] == "stopped" and out["reason"] == "stopped while the KL reference was computed"
    assert out["ran"] == [] and seen and set(seen) == {None}
    assert os.listdir(os.path.join(sdir(ws, "s1"), "trials")) == []
    out = DSR.run(ws, "s1")
    assert out["state"] == "finished" and out["ran"] == [1, 2]


# ------------------------------------------------------------------ 17. the routes, the job and its events
BUSY = "a search is already running on this workspace in this IDE (search 's1'); stop it first"


def test_routes_events_and_the_job(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    import threading
    from wsp import directions_scores as DS
    tiny(tmp_path)
    sess = session_of(tmp_path)
    c = client(sess)
    q = sess.subscribe()
    gate, inside = threading.Event(), threading.Event()
    real = DS.evaluate

    def held(*a, **k):                                 # trial 1 waits here until the test lets it go
        inside.set()
        assert gate.wait(60)
        return real(*a, **k)
    monkeypatch.setattr(DS, "evaluate", held)
    try:
        r = c.post("/api/directions/searches", json=search_body(n_trials=3, stream="srch-1"))
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["job"]["running"] is True and out["job"]["search"] == "s1" and out["job"]["workspace"] == "rnd"
        assert out["job"]["stream"] == "srch-1" and out["job"]["n_trials"] == 3 and out["search"]["name"] == "s1"
        assert inside.wait(60)
        for path, body in (("/api/directions/searches", search_body(name="s2")), ("/api/directions/searches/s1/run", {})):
            r = c.post(path, json=body)
            assert r.status_code == 409 and r.json() == {"error": BUSY}
        assert not os.path.exists(sdir(sess.ws, "s2"))
        r = c.delete("/api/directions/searches/s1")
        assert r.status_code == 409 and r.json() == {"error": "search 's1' is running; stop it first"}
        listed = c.get("/api/directions/searches").json()
        assert [x["name"] for x in listed["searches"]] == ["s1"] and listed["searches"][0]["running"] == "this IDE"
        assert listed["job"]["running"] is True and listed["limits"] == DSR.LIMITS
        assert c.post("/api/directions/searches/s1/stop", json={"when": "now"}).json() == {"ok": True, "found": True}
    finally:
        gate.set()
        join_search_threads()
    ev = events(q)
    assert ev[0]["state"] == "preparing" and ev[-1]["state"] == "stopped" and ev[-1]["reason"] == "stopped by the person"
    assert all(e["workspace"] == "rnd" and e["search"] == "s1" and e["stream"] == "srch-1" for e in ev)
    assert [e["trial_state"] for e in ev if e["state"] == "trial"] == ["stopped"]
    assert sess.search_job["running"] is False and sess.search_job["state"] == "stopped"
    t1 = c.get("/api/directions/searches/s1/trials/1").json()
    assert t1["state"] == "stopped" and t1["attempt_reason"] == "stopped by the person"
    assert c.post("/api/directions/searches/s1/stop", json={}).json() == {"ok": True, "found": False}
    # Run resumes: the stopped trial is rerun, then the rest.
    monkeypatch.setattr(DS, "evaluate", real)
    r = c.post("/api/directions/searches/s1/run", json={"stream": "srch-2"})
    assert r.status_code == 200 and r.json()["job"]["running"] is True and r.json()["job"]["trials_done"] == 0
    join_search_threads()
    ev = events(q)
    states = [e["state"] for e in ev]
    k = states.index("trial")
    assert states[0] == "preparing" and states[-1] == "finished" and "running" in states
    assert set(states[:k]) <= {"preparing", "running"} and set(states[k:-1]) <= {"running", "trial"}
    trials = [e for e in ev if e["state"] == "trial"]
    assert [e["trial"] for e in trials] == [1, 2, 3] and {e["trial_state"] for e in trials} == {"complete"}
    assert all(len(e["values"]) == 2 and len(e["displays"]) == 2 for e in trials)
    assert all(e["n_trials"] == 3 and e["trial"] in (1, 2, 3) for e in ev if e["state"] == "running")
    assert all(e["stream"] == "srch-2" for e in ev) and [e["seconds"] for e in ev] == sorted(e["seconds"] for e in ev)
    recs = DSR.trial_records(sess.ws, "s1")
    assert ev[-1]["reason"] is None and ev[-1]["trials"] == {"complete": 3}
    assert ev[-1]["front"] == trials[-1]["front"] == hand_front(recs)
    assert sess.search_job == {**sess.search_job, "running": False, "state": "finished", "trials_done": 3}
    fr = c.get("/api/directions/searches/s1/front").json()
    assert [x["index"] for x in fr["front"]] == hand_front(recs) and [x["position"] for x in fr["front"]] == \
        list(range(len(fr["front"])))
    for x in fr["front"]:
        rec = recs[x["index"] - 1]
        assert x["script"] == rec["script"] and x["values"] == rec["values"] and x["presented"] == rec["presented"]
    t1 = c.get("/api/directions/searches/s1/trials/1").json()
    assert t1["state"] == "complete" and [a["reason"] for a in t1["attempts"]] == ["stopped by the person"]
    r = c.get("/api/directions/searches/s1/trials/9")
    assert r.status_code == 404 and r.json() == {"error": "search 's1' has no trial 9"}
    d = c.get("/api/directions/searches/s1").json()
    assert [t["index"] for t in d["trials"]] == [1, 2, 3] and d["job"]["state"] == "finished" and d["running"] is None
    assert c.get("/api/directions/searches/s1", params={"trials": "full"}).json()["trials"][2]["script"] == recs[2]["script"]
    # The routes come before GET /api/directions/{name}; a search is not a measurement.
    DR.measure(sess.ws, "prompts", GOOD, BAD)
    assert [x["name"] for x in c.get("/api/directions/searches").json()["searches"]] == ["s1"]
    assert c.get("/api/directions/prompts").status_code == 200
    listing = c.get("/api/directions").json()
    assert sorted(x["name"] for x in listing["directions"]) == ["m", "prompts"] and listing["limits"]["search"] == DSR.LIMITS
    assert c.delete("/api/directions/searches/s1").json() == {"ok": True, "deleted": "s1"}
    r = c.get("/api/directions/searches/s1")
    assert r.status_code == 404 and r.json() == {"error": "no search named 's1'"}


# ------------------------------------------------------------------ 18. one search per workspace, across processes
def test_one_search_per_workspace_across_processes(tmp_path, monkeypatch):
    import threading
    from wsp.atlas import wiring
    tiny(tmp_path)
    sess = session_of(tmp_path)
    c = client(sess)
    monkeypatch.setattr(S, "version", lambda: "5.0.0")
    assert c.post("/api/directions/searches", json=search_body(start=False)).json()["job"] is None
    held, release = threading.Event(), threading.Event()

    def hold():                                          # stands in for another process holding the search lock
        with wiring.build_lock(os.path.join(sess.ws.dir, "directions", "searches")):
            held.set()
            release.wait(30)
    t = threading.Thread(target=hold)
    t.start()
    assert held.wait(30)
    try:
        assert DSR.running(sess.ws) == "this IDE"
        busy = {"error": "a search is already running on this workspace in this IDE; stop it first"}
        for path, body in (("/api/directions/searches", search_body(name="s2")), ("/api/directions/searches/s1/run", {})):
            r = c.post(path, json=body)
            assert r.status_code == 409 and r.json() == busy
        assert not os.path.exists(sdir(sess.ws, "s2"))
        assert c.get("/api/directions/searches").json()["searches"][0]["running"] == "this IDE"
        assert c.get("/api/directions/searches/s1").json()["running"] == "this IDE"
        assert c.delete("/api/directions/searches/s1").status_code == 409
    finally:
        release.set()
        t.join(30)
    assert sess.search_job is None and DSR.running(sess.ws) is None


# ------------------------------------------------------------------ 19. closing the workspace stops its search
def test_unload_stops_a_running_search(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    import threading
    from wsp import directions_scores as DS
    tiny(tmp_path)
    sess = session_of(tmp_path)
    c = client(sess)
    q = sess.subscribe()
    inside = threading.Event()
    real = DS.generate_response

    def slow(*a, **k):
        inside.set()
        time.sleep(0.05)
        return real(*a, **k)
    monkeypatch.setattr(DS, "generate_response", slow)
    assert c.post("/api/directions/searches", json=search_body(n_trials=4)).json()["job"]["running"] is True
    try:
        assert inside.wait(60)
        t0 = time.monotonic()
        sess.unload()
        took = time.monotonic() - t0
    finally:
        join_search_threads()
    assert took < 2
    ev = events(q)
    assert ev[-1]["state"] == "stopped" and ev[-1]["reason"] == "the workspace was closed"
    recs = DSR.trial_records(sess.ws, "s1")
    assert recs[-1]["state"] == "stopped" and recs[-1]["attempt_reason"] == "the workspace was closed"
    assert sess.search_job["running"] is False and sess.search_job["stop"] == "closed"


# ------------------------------------------------------------------ 26. importing the server does not import Optuna
def test_importing_the_server_does_not_import_optuna():
    out = subprocess.run([sys.executable, "-c", "import sys, wsp.server, wsp.directions_search; "
                                                "print('optuna' in sys.modules)"],
                         cwd=APP, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# ------------------------------------------------------------------ 20. committing a trial
def same_pieces(a, b):
    """Two patches' pieces, compared piece by piece (torch.equal)."""
    assert sorted(a) == sorted(b)
    for name in a:
        assert len(a[name]) == len(b[name])
        for p, q in zip(a[name], b[name]):
            assert p.kind == q.kind and p.region == q.region
            for t in ("values", "A", "B", "coords"):
                x, y = getattr(p, t), getattr(q, t)
                assert (x is None and y is None) or torch.equal(x, y)


def test_commit_a_trial(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    import threading
    from wsp import directions_scores as DS
    from wsp.atlas import wiring
    from wsp.workspace import Workspace
    tiny(tmp_path)
    sess = session_of(tmp_path)
    c = client(sess)
    ws = sess.ws
    make_search(ws, n_trials=3)
    assert DSR.run(ws, "s1")["state"] == "finished"
    rec = DSR.trial_records(ws, "s1")[0]
    url = "/api/directions/searches/s1/trials/1/apply"
    # A dry run commits nothing and reports what Session.apply reports.
    r = c.post(url, json={"dry_run": True}).json()
    assert r["ok"] is True and r["committed"] is False and "drift" in r and r["n_changed"] > 0
    assert r["edit_equals_scored"] is True and r["digest"] == r["scored_digest"] == rec["candidate"]["digest"]
    assert ws.patches == [] and patch_files(ws) == []
    # The commit, without Optuna (committing needs no sampler).
    monkeypatch.setattr(S, "version", lambda: None)
    r = c.post(url, json={}).json()
    monkeypatch.undo()
    assert r["ok"] is True and r["committed"] is True and r["status"]["n_patches"] == 1
    assert r["search"] == "s1" and r["trial"] == 1 and r["edit_equals_scored"] is True
    p = ws.patch(r["patch"])
    assert p.message == f"search s1: trial 1 ({', '.join(x['display'] for x in rec['scores'])})"
    # The same script through Session.apply in a second workspace commits the same pieces.
    tiny(tmp_path, "other")
    other = session_of(tmp_path, "other")
    q = other.ws.patch(other.apply(rec["script"], "direct")["patch"])
    same_pieces(p.deltas, q.deltas)
    # The patch's source, as stored on disk.
    stored = Workspace.load(ws.dir).patch(r["patch"]).source
    origin = stored["origin"]
    assert {k: v for k, v in stored.items() if k != "origin"} == {"kind": "wscript", "script": rec["script"]}
    assert origin["kind"] == "direction_search" and origin["search"] == "s1" and origin["trial"] == 1
    assert origin["search_id"] == DSR.load(ws, "s1")["id"] and origin["seed"] == 7 and origin["method"] == "ablation"
    assert origin["digest"] == origin["scored_digest"] == rec["candidate"]["digest"] and origin["edit_equals_scored"] is True
    assert origin["parameters"] == rec["parameters"] and origin["values"] == rec["values"]
    assert origin["scored_environment"] == rec["environment"] and origin["measurement"]["name"] == "m"
    ws.set_patch(r["patch"], enabled=False)
    # A scope that excludes the script's tensors, enforced: refused with the scope's message, no patch.
    sess.set_task({"scope": ["blocks.0.mlp.neuron.5"]})
    r = c.post(url, json={"enforce_scope": True}).json()
    import re
    assert r["ok"] is False and re.fullmatch(r"edit touches weights of \S+ outside the task scope; the scope is "
                                             r"\['blocks\.0\.mlp\.neuron\.5'\]", r["error"]), r
    assert len(ws.patches) == 1 and r["edit_equals_scored"] is True
    sess.set_task({"scope": None})
    # A trial that does not exist, one that failed, one while a search runs, one after the measurement changed.
    r = c.post("/api/directions/searches/s1/trials/9/apply", json={})
    assert r.status_code == 404 and r.json() == {"error": "search 's1' has no trial 9"}
    real = DS.candidate
    monkeypatch.setattr(DS, "candidate", lambda *a, **k: (_ for _ in ()).throw(DR.DirectionError("x")))
    make_search(ws, name="s2", n_trials=1)
    DSR.run(ws, "s2")
    monkeypatch.setattr(DS, "candidate", real)
    r = c.post("/api/directions/searches/s2/trials/1/apply", json={})
    assert r.status_code == 400 and r.json()["error"] == ("SearchError: trial 1 of search 's2' is failed; only a "
                                                          "complete trial can be committed")
    held, release = threading.Event(), threading.Event()

    def hold():
        with wiring.build_lock(os.path.join(ws.dir, "directions", "searches")):
            held.set()
            release.wait(30)
    t = threading.Thread(target=hold)
    t.start()
    assert held.wait(30)
    try:
        r = c.post(url, json={})
        assert r.status_code == 409 and r.json() == {"error": "a search is running on this workspace; stop it before "
                                                              "committing a trial"}
    finally:
        release.set()
        t.join(30)
    DR.measure(ws, "m", GOOD + ["dog"], BAD, replace=True)
    r = c.post(url, json={})
    assert r.status_code == 400 and r.json()["error"] == (
        "SearchError: measurement 'm' changed since search 's1' was made (measured again or deleted); its script would "
        "now make another edit than the one it scored")
    assert len(ws.patches) == 1


# ------------------------------------------------------------------ 21. an edit that differs is not committed unasked
A7_TAIL = ("Its scores do not describe this edit: score it again (POST /api/directions/scores with the trial's script), "
           "or pass accept_different_edit")


def test_a_trial_whose_edit_differs_is_not_committed_unasked(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    tiny(tmp_path, module_io=True)
    sess = session_of(tmp_path)
    ws = sess.ws
    make_search(ws, n_trials=2)
    make_search(ws, name="a1", method="ara", n_trials=1)
    assert DSR.run(ws, "s1")["state"] == "finished" and DSR.run(ws, "a1")["state"] == "finished"
    assert DSR.apply_trial(sess, "a1", 1, dry_run=True)["edit_equals_scored"] is True    # the ARA fit, run again
    torch_then = str(torch.__version__)

    def tamper(name):
        path = trial_path(ws, name, 1)
        rec = read_json(path)
        real = rec["candidate"]["digest"]
        write_json(path, {**rec, "candidate": {**rec["candidate"], "digest": "ab" * 32}})
        return real, rec["environment"]["torch_threads"]
    real, threads = tamper("s1")
    r = DSR.apply_trial(sess, "s1", 1)
    assert r["ok"] is False and r["error"] == (
        f"the edit made now differs from the one trial 1 scored (pieces {real[:12]} against abababababab). The library "
        f"versions and thread count are those it was scored with, so this is not expected; report it. {A7_TAIL}")
    assert ws.patches == [] and r["edit_equals_scored"] is False
    r = DSR.apply_trial(sess, "s1", 1, accept_different_edit=True)
    assert r["ok"] is True and r["committed"] is True and r["edit_equals_scored"] is False
    origin = ws.patch(r["patch"]).source["origin"]
    assert origin["edit_equals_scored"] is False and origin["digest"] == real and origin["scored_digest"] == "ab" * 32
    ws.set_patch(r["patch"], enabled=False)
    monkeypatch.setattr(torch, "__version__", "9.9.9")
    r = DSR.apply_trial(sess, "s1", 1)
    assert r["ok"] is False and r["error"] == (
        f"the edit made now differs from the one trial 1 scored (pieces {real[:12]} against abababababab). The closed "
        f"form rounds differently with other library versions (scored with torch {torch_then}; now torch 9.9.9). "
        f"{A7_TAIL}")
    real, threads = tamper("a1")
    r = DSR.apply_trial(sess, "a1", 1)
    assert r["ok"] is False and r["error"] == (
        f"the edit made now differs from the one trial 1 scored (pieces {real[:12]} against abababababab). An ARA fit is "
        f"reproducible only from the same captures with the same torch version and thread count (scored with torch "
        f"{torch_then} and {threads} threads; now torch 9.9.9 and {torch.get_num_threads()} threads). {A7_TAIL}")
    assert len(ws.patches) == 1
