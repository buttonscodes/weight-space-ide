"""What the Directions tab reads from the server: the parameter fields of GET /api/directions (wsp.directions_fields),
the scorers' and search modifiers' settings and defaults in the limits, the installed sampler, and the running
direction requests. Model-free: the tiny random workspaces of this project (Config + init_params, no training),
scripted stand-ins for the model's work, and a fake Hub; no network.

Every expectation is typed here from the build spec of the tab (section 7) and docs/DIRECTIONS.md ("The search
spaces", "Settings", "Scores", the ARA parameters): Heretic's ranges as Heretic's arithmetic gives them (``0.6 * 9``),
the midpoint defaults worked out by hand and rounded to four decimals. Nothing is computed with the code under test.
"""
import numpy
import pytest
from fastapi.testclient import TestClient

from wsp import directions as DR
from wsp import directions_ara as ARA
from wsp import directions_sampler as S
from wsp import directions_scores as DS
from wsp import directions_search as DSR
from wsp.model import Config, init_params
from wsp.server import create_app
from wsp.session import Session
from wsp.tokenizer import VocabTokenizer
from wsp.workspace import Workspace

from test_steering import CFG, make_ws
from test_directions import BAD, GOOD, save_random

BOTH = ["attn.o_proj", "mlp.down_proj"]
KEYS = ("name", "kind", "default", "nullable", "null_label", "min", "max", "min_exclusive", "choices", "applies_when",
        "heretic_range", "log")


def F(name, kind, default, **kw):
    """A field as the spec's F: every key present, unused ones null (nullable, min_exclusive and log false)."""
    out = {"name": name, "kind": kind, "default": default, "nullable": False, "null_label": None, "min": None,
           "max": None, "min_exclusive": False, "choices": None, "applies_when": None, "heretic_range": None, "log": False}
    assert set(kw) <= set(KEYS)
    out.update(kw)
    return out


def session_on(tmp_path, name="rnd", n_layers=None):
    if n_layers is None:
        make_ws(tmp_path / name)
    else:
        cfg = Config(**{**CFG, "n_layers": n_layers})
        Workspace.create(str(tmp_path / name), cfg, VocabTokenizer(list("abcdefghijklmnopqrstuvwxyz"), "char"),
                         init_params(cfg, seed=1, std=0.3))
    s = Session(str(tmp_path))
    s.open(name)
    return s


def client(sess):
    app = create_app(sess, "http://127.0.0.1:0")
    return app, TestClient(app, raise_server_exceptions=False)


def ablation_component(name, max_weight, mw_range, n, position, position_range, min_weight, distance, distance_range):
    return {"name": name, "fields": [
        F("max_weight", "number", max_weight, min=-8.0, max=8.0, heretic_range=mw_range),
        F("max_weight_position", "number", position, min=-n, max=2 * n, heretic_range=position_range),
        F("min_weight", "number", min_weight, min=-8.0, max=8.0),
        F("min_weight_distance", "number", distance, min=0, min_exclusive=True, heretic_range=distance_range)]}


def ara_fields(n, start_range, end_range):
    return [F("start_layer_index", "integer", 0, min=0, max=n - 1, heretic_range=start_range),
            F("end_layer_index", "integer", None, nullable=True, null_label="to the last layer", min=1, max=n,
              heretic_range=end_range),
            F("preserve_good_behavior_weight", "number", 1.0, min=0, max=1000.0, heretic_range=[0.0, 1.0]),
            F("steer_bad_behavior_weight", "number", 0.01, min=0, max=1000.0, heretic_range=[0.0001, 1.0], log=True),
            F("overcorrect_relative_weight", "number", 0.0, min=0, max=10.0, heretic_range=[0.0, 1.3]),
            F("neighbor_count", "integer", 1, min=1, max=64, heretic_range=[1, 15]),
            F("preserve_row_magnitudes", "boolean", True),
            F("lora_rank", "integer", 50, min=1, max=256),
            F("n_optimization_steps", "integer", 5, min=1, max=50),
            F("learning_rate", "number", 1.0, min=0, max=10.0, min_exclusive=True),
            F("max_iter", "integer", 20, min=1, max=100),
            F("history_size", "integer", 10, min=1, max=100),
            F("seed", "integer", 0, min=0, max=2 ** 63 - 1)]


def ablation_head(L, index_range):
    return [F("direction_index", "number", None, nullable=True, null_label="a direction per layer", min=-1, max=L,
              heretic_range=index_range),
            F("orthogonalize_direction", "boolean", True),
            F("row_normalization", "choice", "none", choices=["none", "pre", "full"]),
            F("full_normalization_lora_rank", "integer", 3, min=1, max=64, applies_when={"row_normalization": "full"})]


# n = 2, L = 1: direction_index 0.4 .. 0.9; max_weight 0.8 .. 1.5 (mid 1.15) and -0.25 .. 1.5 (mid 0.625);
# max_weight_position 0.6 .. 1.0 (mid 0.8); min_weight half of max_weight; min_weight_distance 1 .. max(0.6, 1).
PARAMETERS_TINY = {
    "n_layers": 2,
    "ablation": {"fields": ablation_head(1, [0.4, 0.9]),
                 "components": [ablation_component("attn.o_proj", 1.15, [0.8, 1.5], 2, 0.8, [0.6, 1.0], 0.575, 1.0, [1.0, 1.0]),
                                ablation_component("mlp.down_proj", 0.625, [-0.25, 1.5], 2, 0.8, [0.6, 1.0], 0.3125, 1.0,
                                                   [1.0, 1.0])]},
    "ara": {"fields": ara_fields(2, [0, 1], [2, 2])}}


# ------------------------------------------------------------------ R1-R3: the parameter fields (B1)
def test_r1_the_parameter_fields_of_the_tiny_workspace(tmp_path):
    _, c = client(session_on(tmp_path))
    r = c.get("/api/directions")
    assert r.status_code == 200, r.text
    assert r.json()["parameters"] == PARAMETERS_TINY


def test_r1_ten_layers_defaults_are_rounded_and_ranges_are_heretics_arithmetic(tmp_path):
    # n = 10, L = 9: the middle of 0.6 L .. L is 7.199999999999999 unrounded, of 1 .. 0.6 L 3.1999999999999997.
    _, c = client(session_on(tmp_path, "ten", n_layers=10))
    got = c.get("/api/directions").json()["parameters"]
    pos_range, dist_range = [0.6 * 9, 1.0 * 9], [1.0, 0.6 * 9]
    assert got == {
        "n_layers": 10,
        "ablation": {"fields": ablation_head(9, [0.4 * 9, 0.9 * 9]),
                     "components": [ablation_component("attn.o_proj", 1.15, [0.8, 1.5], 10, 7.2, pos_range, 0.575, 3.2,
                                                       dist_range),
                                    ablation_component("mlp.down_proj", 0.625, [-0.25, 1.5], 10, 7.2, pos_range, 0.3125,
                                                       3.2, dist_range)]},
        "ara": {"fields": ara_fields(10, [0, 5], [6, 10])}}


def test_r2_the_defaults_are_accepted(tmp_path):
    sess = session_on(tmp_path)
    _, c = client(sess)
    save_random(sess.ws, "refusal")
    # E2's body: the ablation defaults of PARAMETERS_TINY as the form sends them
    body = {"method": "ablation", "direction": "refusal", "direction_index": None,
            "weight_distributions": {
                "attn.o_proj": {"max_weight": 1.15, "max_weight_position": 0.8, "min_weight": 0.575, "min_weight_distance": 1},
                "mlp.down_proj": {"max_weight": 0.625, "max_weight_position": 0.8, "min_weight": 0.3125,
                                  "min_weight_distance": 1}},
            "orthogonalize_direction": True, "row_normalization": "none", "full_normalization_lora_rank": 3}
    r = c.post("/api/directions/script", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["script"].startswith('ablate_direction("refusal", direction_index=None, ')
    # The ARA defaults pass directions_ara.normalize unchanged.
    fields = c.get("/api/directions").json()["parameters"]["ara"]["fields"]
    given = {"method": "ara", "direction": "refusal", **{f["name"]: f["default"] for f in fields}}
    assert ARA.normalize(given) == given
    # and so do the ablation's scalar defaults with DR.normalize (nothing added, nothing changed)
    head = {f["name"]: f["default"] for f in c.get("/api/directions").json()["parameters"]["ablation"]["fields"]}
    assert DR.normalize({"direction": "refusal", "weight_distributions": body["weight_distributions"], **head}) == \
        {"direction": "refusal", "weight_distributions": body["weight_distributions"], **head}


def test_r3_unknown_components_answer_null(tmp_path, monkeypatch):
    _, c = client(session_on(tmp_path))
    monkeypatch.setattr(DSR, "_model_components", lambda ws: [])
    r = c.get("/api/directions")
    assert r.status_code == 200 and r.json()["parameters"] is None
    assert r.json()["directions"] == [] and r.json()["limits"]["prompts_per_set"] == 1024


def test_r3_a_model_without_layers_answers_null(tmp_path):
    sess = session_on(tmp_path)
    _, c = client(sess)
    sess.ws.struct.layers = []                       # an architecture whose layers were not recognised
    r = c.get("/api/directions")
    assert r.status_code == 200 and r.json()["parameters"] is None


# ------------------------------------------------------------------ R4-R6: settings, defaults, the sampler (B1)
def test_r4_the_scorers_settings_and_the_response_length_default(tmp_path):
    _, c = client(session_on(tmp_path))
    scores = c.get("/api/directions").json()["limits"]["scores"]
    markers = scores["settings"]["KeywordRate"]["keyword_markers"]["default"]
    assert len(markers) == 33 and markers == DS.REFUSAL_MARKERS
    assert markers[:3] == ["disclaimer", "sorry", "i can'"] and markers[-1] == "ethical boundaries"
    assert scores["settings"] == {
        "KeywordRate": {"score_name": {"kind": "text64", "default": "Refusals"},
                        "keyword_markers": {"kind": "markers", "default": markers},
                        "print_responses": {"kind": "bool", "default": False}},
        "KLDivergence": {},
        "ResponseLength": {"strip_responses": {"kind": "bool", "default": True},
                           "short_response_threshold": {"kind": "count", "default": 2},
                           "short_response_score_length": {"kind": "number", "default": 500.0},
                           "print_responses": {"kind": "bool", "default": False}}}
    assert scores["max_response_length_default"] == 100
    assert list(scores["settings"]["KeywordRate"]) == ["score_name", "keyword_markers", "print_responses"]


def test_r5_the_modifiers_settings_and_the_count_defaults(tmp_path, monkeypatch):
    sess = session_on(tmp_path)
    _, c = client(sess)
    want_settings = {"Abliteration": {"orthogonalize_direction": True, "row_normalization": "full",
                                      "full_normalization_lora_rank": 3},
                     "ARA": {"preserve_row_magnitudes": True, "lora_rank": 50, "n_optimization_steps": 5,
                             "learning_rate": 1.0, "max_iter": 20, "history_size": 10, "print_loss": False}}
    for limits in (c.get("/api/directions").json()["limits"]["search"],
                   c.get("/api/directions/searches").json()["limits"]):
        assert limits["settings"] == want_settings
        assert limits["defaults"] == {"n_trials": 100, "n_startup_trials": 30}
    # A create without n_trials or n_startup_trials still records Heretic's 100 and 30 (start false: nothing runs).
    monkeypatch.setattr(S, "version", lambda: "5.0.0")
    DR.measure(sess.ws, "m", GOOD, BAD)
    body = {"name": "s1", "modifiers": [{"plugin": "Abliteration", "direction": "m"}], "start": False,
            "scorers": [{"plugin": "KeywordRate", "optimization": "minimize", "prompts": ["ab", "cd"]},
                        {"plugin": "KLDivergence", "optimization": "minimize", "prompts": ["ef", "gh"]}],
            "max_response_length": 3}
    r = c.post("/api/directions/searches", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["job"] is None
    got = c.get("/api/directions/searches/s1").json()["search"]
    assert (got["n_trials"], got["n_startup_trials"]) == (100, 30)
    assert c.post("/api/directions/searches", json={**body, "name": "s2", "n_trials": 0}).json() == \
        {"error": "SearchError: n_trials must be an integer from 1 to 10000"}


def test_r6_the_installed_sampler(tmp_path, monkeypatch):
    _, c = client(session_on(tmp_path))
    monkeypatch.setattr(S, "version", lambda: None)
    got = c.get("/api/directions/searches").json()
    assert got["sampler"] == {"optuna": None, "numpy": numpy.__version__, "validated": "5.0.0"}
    assert got["searches"] == [] and got["job"] is None
    monkeypatch.setattr(S, "version", lambda: "5.0.0")
    got = c.get("/api/directions/searches").json()["sampler"]
    assert got["optuna"] == "5.0.0" and got["validated"] == "5.0.0"
    monkeypatch.setattr(S, "version", lambda: "5.1.2")
    assert c.get("/api/directions/searches").json()["sampler"]["optuna"] == "5.1.2"


# ------------------------------------------------------------------ R7, R8: the running direction requests (B2)
SCORE_BODY = {"state": "current", "max_response_length": 3, "scores": [{"plugin": "KeywordRate", "prompts": ["abc"]}]}
SCORE_KEYS = {"type", "workspace", "state", "seconds", "phase", "done", "total", "stream"}


def without_seconds(e):
    assert isinstance(e["seconds"], float)
    return {k: v for k, v in e.items() if k != "seconds"}


def test_r7_a_scores_request_is_listed_while_it_runs(tmp_path, monkeypatch):
    sess = session_on(tmp_path)
    app, c = client(sess)
    seen, real = [], DS.generate_response

    def recording(*a, **k):
        if not seen:
            seen.append(list(app.state.direction_requests.values()))
            seen.append(c.get("/api/directions").json()["requests"])        # a second window asks meanwhile
        return real(*a, **k)
    monkeypatch.setattr(DS, "generate_response", recording)
    r = c.post("/api/directions/scores", json={**SCORE_BODY, "stream": "sc-r"})
    assert r.status_code == 200, r.text
    inside, listed = seen
    assert len(inside) == 1 and set(inside[0]) == SCORE_KEYS
    assert without_seconds(inside[0]) == {"type": "direction_scores", "workspace": "rnd", "state": "scoring",
                                          "phase": None, "done": 0, "total": None, "stream": "sc-r"}
    assert listed == inside
    assert app.state.direction_requests == {} and c.get("/api/directions").json()["requests"] == []
    # A stop sent before the request: it answers stopped at once and leaves nothing listed.
    assert c.post("/api/generation/stop", json={"stream": "early"}).json()["found"] is False
    r = c.post("/api/directions/scores", json={**SCORE_BODY, "stream": "early"})
    assert r.json() == {"stopped": True, "done": 0, "total": 1}
    assert app.state.direction_requests == {} and c.get("/api/directions").json()["requests"] == []
    # A request without a stream is listed too (no stream key: it cannot be stopped from another page).
    seen.clear()
    assert c.post("/api/directions/scores", json=SCORE_BODY).status_code == 200
    assert [without_seconds(e) for e in seen[0]] == [{"type": "direction_scores", "workspace": "rnd", "state": "scoring",
                                                      "phase": None, "done": 0, "total": None}]
    assert app.state.direction_requests == {}


def test_r7_an_exception_in_the_middle_of_a_scores_request_leaves_no_entry(tmp_path, monkeypatch):
    sess = session_on(tmp_path)
    app, c = client(sess)
    seen = []

    def failing(*a, **k):                                # the first unit fails
        seen.append(list(app.state.direction_requests.values()))
        raise RuntimeError("unit failed")
    monkeypatch.setattr(DS, "generate_response", failing)
    r = c.post("/api/directions/scores", json={**SCORE_BODY, "stream": "sc-f"})
    assert r.status_code == 400 and r.json() == {"error": "RuntimeError: unit failed"}
    assert [(e["state"], e["stream"]) for e in seen[0]] == [("scoring", "sc-f")]
    assert app.state.direction_requests == {} and c.get("/api/directions").json()["requests"] == []


def test_b2_an_answer_that_breaks_the_route_leaves_no_entry(tmp_path, monkeypatch):
    # No terminal event is published when the route itself fails after the work (here: an answer that is not a
    # dict); the entry is removed all the same when the request ends.
    sess = session_on(tmp_path)
    app, c = client(sess)
    seen = []

    def broken(ws, args, *, scope=None, progress=None, stopped=None):
        progress("responses", 1, 2)
        seen.append(list(app.state.direction_requests.values()))
        return None
    monkeypatch.setattr(DS, "score", broken)
    r = c.post("/api/directions/scores", json={**SCORE_BODY, "stream": "sc-b"})
    assert r.status_code == 400 and r.json()["error"].startswith("AttributeError: ")
    assert [e["state"] for e in seen[0]] == ["scoring"]
    assert app.state.direction_requests == {} and c.get("/api/directions").json()["requests"] == []


def test_b2_the_listing_at_each_event_is_what_the_event_says(tmp_path, monkeypatch):
    # Each event is tracked before it is published: a page that refetches on a progress event finds that event
    # listed, and one that refetches on a terminal event finds nothing.
    from wsp import directions_prefix as DP
    sess = session_on(tmp_path)
    app, c = client(sess)
    log, real_publish = [], sess.publish

    def publish(e):
        if e.get("type") in ("directions", "direction_scores", "direction_prefix", "direction_datasets"):
            log.append((dict(e), list(app.state.direction_requests.values())))
        real_publish(e)
    monkeypatch.setattr(sess, "publish", publish)
    assert c.post("/api/directions/scores", json={**SCORE_BODY, "stream": "s1"}).status_code == 200
    c.post("/api/generation/stop", json={"stream": "s2"})
    assert c.post("/api/directions/scores", json={**SCORE_BODY, "stream": "s2"}).json()["stopped"] is True
    assert c.post("/api/directions/measure", json={"name": "refusal", "good": GOOD, "bad": BAD, "stream": "m1"}) \
        .status_code == 200
    monkeypatch.setattr(DP, "check_prefix", lambda ws, body: None)

    def detect(ws, body, *, progress=None, stopped=None):
        progress("responses", 1, 3)
        return {"response_prefix": "", "done": 3, "total": 3}
    monkeypatch.setattr(DP, "detect", detect)
    assert c.post("/api/directions/prefix", json={"stream": "p1"}).status_code == 200

    def failing(*a, **k):
        raise RuntimeError("unit failed")
    monkeypatch.setattr(DS, "generate_response", failing)
    assert c.post("/api/directions/scores", json={**SCORE_BODY, "stream": "s3"}).status_code == 400
    ends = {"measured", "scored", "detected", "fetched", "stopped", "failed"}
    assert {e["state"] for e, _ in log} == {"scoring", "scored", "stopped", "measuring", "measured", "detecting",
                                           "detected", "failed"}
    for e, listing in log:
        assert listing == ([] if e["state"] in ends else [e]), (e, listing)
    assert app.state.direction_requests == {}


def test_r8_only_this_workspaces_requests_are_listed(tmp_path):
    app, c = client(session_on(tmp_path))
    mine = {"type": "directions", "workspace": "rnd", "name": "x", "state": "measuring", "seconds": 1.5, "done": 2,
            "total": 8, "stream": "other-1"}
    theirs = {"type": "direction_scores", "workspace": "other", "state": "scoring", "seconds": 0.5, "phase": None,
              "done": 0, "total": None, "stream": "other-2"}
    later = {"type": "direction_datasets", "workspace": "rnd", "state": "fetching", "seconds": 0.1, "phase": "file",
             "file": "data/train-00000-of-00001.jsonl", "done_bytes": 10, "total_bytes": 99, "stream": "other-3"}
    app.state.direction_requests.update({"a": dict(mine), "b": dict(theirs), "c": dict(later)})
    assert c.get("/api/directions").json()["requests"] == [mine, later]          # in the order they started
    assert app.state.direction_requests == {"a": mine, "b": theirs, "c": later}   # the listing changes nothing


def test_r8_a_measure_request_is_listed_while_it_runs(tmp_path, monkeypatch):
    app, c = client(session_on(tmp_path))
    seen = []

    def measure(ws, *, progress=None, stopped=None, **kw):
        progress(0, 5)
        seen.append(list(app.state.direction_requests.values()))
        return {"stopped": True, "done": 0, "total": 5}
    monkeypatch.setattr(DR, "measure", measure)
    r = c.post("/api/directions/measure",
               json={"name": "refusal", "good": ["abc", "hello", "fox"], "bad": ["xyz", "qqq"], "stream": "dir-1"})
    assert r.json() == {"stopped": True, "done": 0, "total": 5}
    assert len(seen[0]) == 1
    assert without_seconds(seen[0][0]) == {"type": "directions", "workspace": "rnd", "name": "refusal",
                                           "state": "measuring", "done": 0, "total": 5, "stream": "dir-1"}
    assert app.state.direction_requests == {} and c.get("/api/directions").json()["requests"] == []
    # A refusal before the model (an unknown field): nothing was ever listed.
    r = c.post("/api/directions/measure", json={"name": "r2", "good": ["a"], "bad": ["b"], "nope": 1})
    assert r.status_code == 400 and app.state.direction_requests == {}


def test_r8_a_dataset_fetch_is_listed_while_it_runs(tmp_path, monkeypatch):
    from wsp import directions_datasets as DD
    from test_directions_datasets import C1, REPO, FakeHub, hs, repo_files
    app, c = client(session_on(tmp_path))
    cache = tmp_path / "hub"
    monkeypatch.setattr(DD, "hub_cache_dir", lambda: str(cache))
    fake = FakeHub(cache)
    fake.add(REPO, "main", C1, repo_files())
    seen = []

    def fetch(*a, **k):
        seen.append(list(app.state.direction_requests.values()))
        return fake.fetch(*a, **k)
    monkeypatch.setattr(DD, "hub_info", fake.info)
    monkeypatch.setattr(DD, "hub_fetch", fetch)
    # A check (no download) is not a fetch: it is never listed.
    r = c.post("/api/directions/datasets", json={"prompts": ["a", "b"]})
    assert r.status_code == 200 and r.json()["cached"] is True and seen == [] and app.state.direction_requests == {}
    r = c.post("/api/directions/datasets", json={"prompts": hs(), "download": True, "stream": "dl-1"})
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 3 and fake.fetch_calls[0] == "README.md"
    first = seen[0]
    assert len(first) == 1 and set(first[0]) == {"type", "workspace", "state", "seconds", "phase", "file", "done_bytes",
                                                 "total_bytes", "stream"}
    assert (first[0]["type"], first[0]["workspace"], first[0]["state"], first[0]["stream"]) == \
        ("direction_datasets", "rnd", "fetching", "dl-1")
    assert all(len(s) == 1 and s[0]["stream"] == "dl-1" for s in seen)
    assert app.state.direction_requests == {} and c.get("/api/directions").json()["requests"] == []
