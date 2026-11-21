"""Re-running a committed direction edit from its reproduction record (wsp.directions_record.reproduce and
POST /api/patches/{pid}/reproduce), model-free.

The workspace is the tiny random model of tests/test_steering.py, opened by a Session; measurements are made with
DR.measure on it. Every expectation is computed here: digests with DSR.piece_digest of the patches' own pieces,
the pieces compared with torch.equal, the differences rebuilt densely in the test.
"""
import hashlib
import json
import math
import os

import pytest
import torch

from wsp import delta as Dl
from wsp import directions as DR
from wsp import directions_ara as ARA
from wsp import directions_record as REC
from wsp import directions_search as DSR
from wsp.store import norm_index

from test_directions import GOOD, BAD
from test_directions_record import ARA_SMALL, ablation_params, mpath, record, records, session

NO_RECORD = ("patch {pid} has no reproduction record: it was not made by ablate_direction, ablate_ara, direction or "
             "direction_means through the edit path, or it was made before records existed")


def committed(sess, ws, params=None, measure=True, **kw):
    """Measure "m", commit the ablation, switch the patch off -> its id."""
    if measure:
        DR.measure(ws, "m", GOOD, BAD, **kw)
    r = sess.apply(DR.script_for(params or ablation_params()))
    assert r["committed"] and "record" in r, r
    ws.set_patch(r["patch"], enabled=False)
    return r["patch"]


def same_pieces(a, b):
    assert sorted(a) == sorted(b)
    for name in a:
        assert len(a[name]) == len(b[name]), name
        for p, q in zip(a[name], b[name]):
            assert p.kind == q.kind and p.region == q.region, name
            for t in ("values", "A", "B", "coords"):
                x, y = getattr(p, t), getattr(q, t)
                assert (x is None and y is None) or torch.equal(x, y), (name, t)


def client(sess):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    return TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)


# ------------------------------------------------------------------ RP1
def test_the_rerun_of_an_ablation_is_identical(tmp_path):
    sess, ws = session(tmp_path)
    pid = committed(sess, ws, ablation_params(mode="full"))
    original = ws.patch(pid)
    digest = DSR.piece_digest(original.deltas)
    before = [p.id for p in ws.patches]
    r = REC.reproduce(sess, pid)                                    # a dry run by default
    assert r["ok"] and r["committed"] is False and r["verdict"] == "identical"
    assert r["digest"] == r["recorded_digest"] == digest == record(ws, pid)["pieces"]["digest"]
    assert r["differences"] is None and r["bound"] == 1e-4 and r["means"] is None
    assert [x["key"] for x in r["checks"]] == ["base", "made_on", "measurement:m"]
    assert all(x["same"] is True for x in r["checks"])
    assert [p.id for p in ws.patches] == before and ws.patch(pid).enabled is False
    r = REC.reproduce(sess, pid, dry_run=False)
    assert r["committed"] and r["verdict"] == "identical", r
    new = ws.patch(r["patch"])
    assert [p.id for p in ws.patches] == before + [new.id] and ws.patch(pid).enabled is False
    o = new.source["origin"]
    assert o["kind"] == "reproduction" and o["of"] == pid and o["of_created"] == original.created
    assert (o["recorded_digest"], o["digest"], o["verdict"], o["measurement"]) == (digest, digest, "identical", "m")
    assert o["environment"]["libraries"]["torch"] == torch.__version__
    assert new.source["script"] == original.source["script"] and new.message == f"re-run of patch {pid}"
    same_pieces(new.deltas, original.deltas)
    rec = record(ws, new.id)                                        # the re-run has a record of its own
    assert rec["origin"] == o and rec["pieces"]["digest"] == digest
    # The verdict compares with the record's digest, never with a search's scored digest in an origin.
    path = os.path.join(records(ws), f"{pid}.json")
    saved = json.load(open(path))
    saved["origin"] = {"kind": "direction_search", "scored_digest": "ab" * 32}
    json.dump(saved, open(path, "w"))
    ws.set_patch(new.id, enabled=False)
    assert REC.reproduce(sess, pid)["verdict"] == "identical"


# ------------------------------------------------------------------ RP2
def test_the_rerun_of_an_ara_fit_is_identical_here(tmp_path):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    params = {"method": "ara", "direction": "m", "start_layer_index": 0, "end_layer_index": None, "seed": 3, **ARA_SMALL}
    pid = committed(sess, ws, params, measure=False)
    original = ws.patch(pid)
    r = REC.reproduce(sess, pid)
    assert r["ok"] and not r["committed"] and r["verdict"] == "identical" and r["bound"] is None
    assert r["digest"] == DSR.piece_digest(original.deltas)
    rows = {x["name"]: x for x in r["environment"]}
    assert rows["torch_threads"]["severity"] == "high"                 # an ARA call: the thread count matters
    r = REC.reproduce(sess, pid, dry_run=False)
    assert r["committed"] and ws.patch(r["patch"]).source["origin"]["kind"] == "reproduction"
    same_pieces(ws.patch(r["patch"]).deltas, original.deltas)


# ------------------------------------------------------------------ RP3
def test_the_preconditions_refuse(tmp_path):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD)
    pid = sess.apply(DR.script_for(ablation_params()))["patch"]
    c = client(sess)
    # the patch itself is switched on: not the weights it was made on
    r = c.post(f"/api/patches/{pid}/reproduce", json={})
    assert (r.status_code, r.json()) == (400, {"error": (
        f"RecordError: the enabled patches are not those patch {pid} was made on (then: none; now: {pid}). Switch patch "
        f"{pid} off (and the others back as they were) to re-run it on the weights it was made on")})
    assert ws.patch(pid).enabled is True and len(ws.patches) == 1     # nothing switched, nothing committed
    ws.set_patch(pid, enabled=False)
    assert c.post(f"/api/patches/{pid}/reproduce", json={}).json()["verdict"] == "identical"
    # the base's cached hash says another content
    cache = os.path.join(records(ws), "base.json")
    saved = open(cache).read()
    rows = json.loads(saved)
    rows["files"][0][3] = "cd" * 32
    json.dump(rows, open(cache, "w"))
    r = c.post(f"/api/patches/{pid}/reproduce", json={})
    assert (r.status_code, r.json()) == (400, {"error": (
        f"RecordError: the base weights are not those patch {pid} was made on (base.safetensors differs)")})
    open(cache, "w").write(saved)
    # the measurement measured again
    DR.measure(ws, "m", GOOD + ["dog"], BAD, replace=True)
    r = c.post(f"/api/patches/{pid}/reproduce", json={"dry_run": True})
    assert (r.status_code, r.json()) == (400, {"error": (
        f"RecordError: measurement 'm' changed since patch {pid} was made (measured again or deleted): measure it again "
        f"from the record (GET /api/patches/{pid}/record?remeasure=true) and pass the new name as measurement")})
    # a search running: this session's job, or the searches lock held
    sess.search_job = {"running": True, "workspace": ws.name, "search": "s1"}
    r = c.post(f"/api/patches/{pid}/reproduce", json={})
    want = (409, {"error": "a search is running on this workspace; stop it before re-running an edit"})
    assert (r.status_code, r.json()) == want
    sess.search_job = None
    with DSR.search_lock(ws, "s1"):
        r = c.post(f"/api/patches/{pid}/reproduce", json={})
        assert (r.status_code, r.json()) == want
    # 404s, an unknown field, a patch without a record
    r = c.post("/api/patches/9999-00000/reproduce", json={})
    assert (r.status_code, r.json()) == (404, {"error": "no patch '9999-00000'"})
    r = c.post(f"/api/patches/{pid}/reproduce", json={"commit": True})
    assert (r.status_code, r.json()) == (400, {"error": "RecordError: commit: unknown field; the fields are measurement, "
                                                        "dry_run, message, enforce_scope and accept_different"})
    for body in ({"dry_run": "no"}, {"accept_different": 1}, {"message": "x" * 1001}, {"measurement": "../m"}):
        assert c.post(f"/api/patches/{pid}/reproduce", json=body).status_code == 400, body
    plain = sess.apply("mlp(1).W_out *= 0.5")["patch"]
    r = c.post(f"/api/patches/{plain}/reproduce", json={})
    assert (r.status_code, r.json()) == (404, {"error": NO_RECORD.format(pid=plain)})
    assert [p.id for p in ws.patches] == [pid, plain]                 # no refusal committed anything


def test_a_stale_record_is_no_record(tmp_path, monkeypatch):
    sess, ws = session(tmp_path)
    pid = committed(sess, ws)
    path = os.path.join(records(ws), f"{pid}.json")
    saved = json.load(open(path))
    saved["patch"]["created"] += 1                                    # another patch had this id
    json.dump(saved, open(path, "w"))
    with pytest.raises(REC.RecordMissing) as e:
        REC.reproduce(sess, pid)
    assert str(e.value) == NO_RECORD.format(pid=pid)


# ------------------------------------------------------------------ RP4
def test_library_differences_are_reported(tmp_path, monkeypatch):
    sess, ws = session(tmp_path)
    pid = committed(sess, ws)
    real = DSR.environment

    def older():
        env = real()
        return {**env, "libraries": {**env["libraries"], "torch": "0.0.0"}}
    monkeypatch.setattr(DSR, "environment", older)
    r = REC.reproduce(sess, pid)
    rows = {x["name"]: x for x in r["environment"]}
    assert rows["torch"] == {"name": "torch", "then": torch.__version__, "now": "0.0.0", "severity": "high", "same": False}
    assert all(x["same"] for name, x in rows.items() if name != "torch")
    # a runner that is not loaded now is not known yet, which is not a difference; another runner, and a package
    # that is gone, are differences
    runner = {"class": "HFRunner", "dtype": "torch.float32", "device": "cpu"}
    then = {"runner": runner, "packages": {"optuna": "5.0.0"}}

    def same(now):
        return {x["name"]: x["same"] for x in REC.environment_rows(then, now, False)}
    assert same({"runner": None, "packages": {"optuna": "5.0.0"}})["runner"] is None
    assert same({"runner": {**runner, "device": "mps"}, "packages": {"optuna": "5.0.0"}})["runner"] is False
    assert same({"runner": runner, "packages": {"optuna": None}})["runner"] is True
    assert same({"runner": runner, "packages": {"optuna": None}})["optuna"] is False
    assert same({"runner": None, "packages": {}})["optuna"] is False
    assert {x["name"]: x["same"] for x in REC.environment_rows({"runner": None}, {"runner": None}, False)}["runner"] is True
    assert r["verdict"] == "identical" and r["ok"]


# ------------------------------------------------------------------ RP5
def scaled_change(monkeypatch, factor):
    real = DR.change

    def change(*a, **k):
        pieces, info = real(*a, **k)
        return [(view, A * factor, B) for view, A, B in pieces], info
    monkeypatch.setattr(DR, "change", change)


def test_rounding_and_different_edits(tmp_path, monkeypatch):
    sess, ws = session(tmp_path)
    pid = committed(sess, ws)
    n = len(ws.patches)
    scaled_change(monkeypatch, 1 + 1e-7)
    r = REC.reproduce(sess, pid)
    assert r["verdict"] == "within_rounding" and r["bound"] == 1e-5 and r["digest"] != r["recorded_digest"]
    assert sorted(x["tensor"] for x in r["differences"]) == sorted(ws.patch(pid).deltas)
    assert all(0 < x["relative"] <= 1e-5 and x["max_abs"] > 0 for x in r["differences"])
    r = REC.reproduce(sess, pid, dry_run=False)
    assert r["committed"] and ws.patch(r["patch"]).source["origin"]["verdict"] == "within_rounding"
    ws.set_patch(r["patch"], enabled=False)
    monkeypatch.undo()
    scaled_change(monkeypatch, 1.01)
    r = REC.reproduce(sess, pid)
    assert r["verdict"] == "different" and r["ok"] and not r["committed"]        # a dry run is never refused
    worst = max(r["differences"], key=lambda x: x["relative"])
    assert worst["relative"] == pytest.approx(0.01, rel=1e-4)
    r = REC.reproduce(sess, pid, dry_run=False)
    assert r["ok"] is False and len(ws.patches) == n + 1
    assert r["error"] == (f"the re-run made a different edit (largest relative difference {worst['relative']:.3g} in "
                          f"{worst['tensor']}). the library versions are those it was made with, so this is not "
                          f"expected; report it. Pass accept_different to commit it anyway")
    r = REC.reproduce(sess, pid, dry_run=False, accept_different=True)
    assert r["committed"] and ws.patch(r["patch"]).source["origin"]["verdict"] == "different"


def test_an_ara_fit_has_no_rounding_bound(tmp_path, monkeypatch):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    params = {"method": "ara", "direction": "m", "start_layer_index": 0, "end_layer_index": None, "seed": 3, **ARA_SMALL}
    pid = committed(sess, ws, params, measure=False)
    real = ARA.ablate

    def ablate(ws_, params_, **kw):
        add = kw["add"]
        kw["add"] = lambda view, A, B: add(view, A * (1 + 1e-7), B)
        return real(ws_, params_, **kw)
    monkeypatch.setattr(ARA, "ablate", ablate)
    r = REC.reproduce(sess, pid)
    assert r["digest"] != r["recorded_digest"] and r["bound"] is None
    assert max(x["relative"] for x in r["differences"]) <= 1e-5
    assert r["verdict"] == "different"
    r = REC.reproduce(sess, pid, dry_run=False)
    assert r["ok"] is False and r["error"].startswith("the re-run made a different edit") and (
        "an ARA fit is reproducible only from the same captures with the same torch version and thread count "
        f"(recorded: torch {torch.__version__}, {torch.get_num_threads()} threads; now: torch {torch.__version__}, "
        f"{torch.get_num_threads()} threads)") in r["error"]


# ------------------------------------------------------------------ RP6
def test_measurement_mode(tmp_path):
    sess, ws = session(tmp_path)
    pid = committed(sess, ws, winsorization_quantile=0.9)
    (entry,) = REC.answer(ws, pid, remeasure_too=True)["remeasure"]
    body = entry["body"]
    DR.measure(ws, **DR.measure_body(body))
    DR.measure(ws, "m", GOOD + ["dog"], BAD, replace=True)            # the original changed: apply mode is refused
    with pytest.raises(REC.RecordError):
        REC.reproduce(sess, pid)
    c = client(sess)
    r = c.post(f"/api/patches/{pid}/reproduce", json={"measurement": body["name"]})
    out = r.json()
    assert r.status_code == 200 and out["verdict"] == "identical" and not out["committed"], out
    assert out["means"] == {"equal": True, "relative": {"good": 0.0, "bad": 0.0}} and out["notes"] == []
    assert [x["key"] for x in out["checks"]] == ["base", "made_on", f"measurement:{body['name']}"]
    r = c.post(f"/api/patches/{pid}/reproduce", json={"measurement": body["name"], "dry_run": False})
    new = ws.patch(r.json()["patch"])
    assert new.source["script"] == DR.script_for({**record(ws, pid)["calls"][0]["parameters"], "direction": body["name"]})
    assert new.source["origin"]["measurement"] == body["name"]
    same_pieces(new.deltas, ws.patch(pid).deltas)
    ws.set_patch(new.id, enabled=False)
    # other prompts: the first differing key
    DR.measure(ws, "other", GOOD + ["dog"], BAD, winsorization_quantile=0.9)
    r = c.post(f"/api/patches/{pid}/reproduce", json={"measurement": "other"})
    assert (r.status_code, r.json()) == (400, {"error": (
        f"RecordError: measurement 'other' was not made from patch {pid}'s prompts and rendering: prompts_sha256 differs")})
    # the same prompts with another system message (written into its meta) and another quantile
    DR.measure(ws, "sys", GOOD, BAD, winsorization_quantile=0.9)
    meta = json.load(open(mpath(ws, "sys", ".json")))
    meta["options"]["system"] = "Be brief."
    json.dump(meta, open(mpath(ws, "sys", ".json"), "w"))
    r = c.post(f"/api/patches/{pid}/reproduce", json={"measurement": "sys"})
    assert r.json() == {"error": f"RecordError: measurement 'sys' was not made from patch {pid}'s prompts and rendering: "
                                 f"options differs"}
    DR.measure(ws, "q", GOOD, BAD)
    assert c.post(f"/api/patches/{pid}/reproduce", json={"measurement": "q"}).json()["error"].endswith("options differs")
    r = c.post(f"/api/patches/{pid}/reproduce", json={"measurement": "nope"})
    assert r.status_code == 404
    # the means snapshot gone: no comparison, and the note
    snap = record(ws, pid)["calls"][0]["measurement"]["means_snapshot"]
    os.replace(os.path.join(records(ws), "means", snap + ".safetensors"), os.path.join(str(tmp_path), "moved"))
    out = c.post(f"/api/patches/{pid}/reproduce", json={"measurement": body["name"]}).json()
    assert out["means"] is None and out["notes"] == ["the recorded means are not in the workspace"]
    assert out["verdict"] == "identical"


def test_a_current_measurement_is_remeasured_on_the_same_patches(tmp_path):
    sess, ws = session(tmp_path)
    first = sess.apply("mlp(1).W_out *= 0.5")["patch"]
    DR.measure(ws, "m", GOOD, BAD, variant="current")
    pid = sess.apply(DR.script_for(ablation_params()))["patch"]
    ws.set_patch(pid, enabled=False)
    (entry,) = REC.answer(ws, pid, remeasure_too=True)["remeasure"]
    assert entry["body"]["variant"] == "current"
    DR.measure(ws, **DR.measure_body(entry["body"]))
    assert REC.reproduce(sess, pid, measurement=entry["body"]["name"])["verdict"] == "identical"
    # measured on other patches: refused by measured_on (the record's made_on is restored before the re-run)
    ws.set_patch(first, enabled=False)
    DR.measure(ws, "elsewhere", GOOD, BAD, variant="current")
    ws.set_patch(first, enabled=True)
    with pytest.raises(REC.RecordError) as e:
        REC.reproduce(sess, pid, measurement="elsewhere")
    assert str(e.value) == (f"measurement 'elsewhere' was not made from patch {pid}'s prompts and rendering: measured_on "
                            f"differs")


# ------------------------------------------------------------------ RP7
def test_measurement_mode_needs_a_rebuildable_script(tmp_path):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD)
    pid = sess.apply(DR.script_for(ablation_params()) + "\nmlp(1).W_out *= 0.5")["patch"]
    ws.set_patch(pid, enabled=False)
    r = client(sess).post(f"/api/patches/{pid}/reproduce", json={"measurement": "m"})
    assert (r.status_code, r.json()) == (400, {"error": (
        f"RecordError: patch {pid}'s script does more than ablate_direction and ablate_ara calls, so it cannot be "
        f"rebuilt for another measurement: re-run it without measurement")})
    assert REC.reproduce(sess, pid)["verdict"] == "identical"         # apply mode runs the recorded script verbatim
    DR.measure(ws, "n", GOOD, BAD)
    two = sess.apply(DR.script_for(ablation_params()) + "\n" + DR.script_for(ablation_params("n")))["patch"]
    ws.set_patch(two, enabled=False)
    with pytest.raises(REC.RecordError) as e:
        REC.reproduce(sess, two, measurement="m")
    assert str(e.value) == f"patch {two} reads 2 measurements; measurement mode takes one"


# ------------------------------------------------------------------ RP8
def dense_differences(ws, a, b):
    out = []
    for name in sorted(set(a) | set(b)):
        shape = ws.shape(name)
        full = norm_index((), shape)
        x, y = torch.zeros(shape), torch.zeros(shape)
        for p in a.get(name, []):
            Dl.add_to(x, full, p, 1.0)
        for p in b.get(name, []):
            Dl.add_to(y, full, p, 1.0)
        d, n = float((y - x).double().norm()), float(x.double().norm())
        only = None if bool(a.get(name)) == bool(b.get(name)) else ("recorded" if a.get(name) else "rerun")
        rel = None if only else (d / n if n else (0.0 if d == 0 else None))
        out.append({"tensor": name, "relative": rel, "max_abs": float((y - x).abs().max()), "only_in": only})
    return out


def test_differences_in_slabs_equal_the_dense_ones(tmp_path, monkeypatch):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD)
    torch.manual_seed(0)
    # W_out [40, 24]; W_O [2, 12, 24] (heads first), with a head-indexed low-rank piece (an int axis)
    a = ws.commit({"blocks.1.mlp.W_out": [Dl.lowrank((range(0, 5), range(2, 9)), torch.randn(5, 2), torch.randn(2, 7))],
                   "blocks.1.attn.W_O": [Dl.lowrank((1, range(0, 4), range(0, 5)), torch.randn(4, 1), torch.randn(1, 5))],
                   "blocks.0.attn.W_O": [Dl.block((range(0, 2), range(1, 4), range(0, 3)), torch.randn(2, 3, 3))]},
                  "a").deltas
    b = {"blocks.1.mlp.W_out": [Dl.lowrank((range(0, 5), range(2, 9)), torch.randn(5, 2), torch.randn(2, 7)),
                                Dl.block((range(3, 6), range(0, 4)), torch.randn(3, 4))],
         "blocks.1.attn.W_O": [Dl.block((range(0, 2), range(0, 2), range(0, 2)), torch.randn(2, 2, 2))],
         "blocks.0.mlp.W_out": [Dl.block((range(0, 2), range(0, 2)), torch.randn(2, 2))]}
    want = dense_differences(ws, a, b)
    monkeypatch.setattr(REC, "STREAM_ELEMS", 7)
    got = REC.differences(ws, a, b)
    assert [x["tensor"] for x in got] == [x["tensor"] for x in want]
    # A low-rank piece's product over a slab of rows is a float32 matmul of other shapes than the whole tensor's,
    # which may round its last bit differently; everything else is the same arithmetic.
    for g, w in zip(got, want):
        assert g["max_abs"] == pytest.approx(w["max_abs"], rel=1e-6), g["tensor"]
        assert g["only_in"] == w["only_in"], g["tensor"]
        assert (g["relative"] is None and w["relative"] is None) or (
            None not in (g["relative"], w["relative"]) and g["relative"] == pytest.approx(w["relative"], rel=1e-6))
    by = {x["tensor"]: x for x in got}
    # in one set only: no ratio, the side named; every value finite, so the route's answer is JSON
    assert (by["blocks.0.attn.W_O"]["relative"], by["blocks.0.attn.W_O"]["only_in"]) == (None, "recorded")
    assert (by["blocks.0.mlp.W_out"]["relative"], by["blocks.0.mlp.W_out"]["only_in"]) == (None, "rerun")
    assert by["blocks.0.attn.W_O"]["max_abs"] > 0 and by["blocks.0.mlp.W_out"]["max_abs"] > 0
    assert 0 < by["blocks.1.attn.W_O"]["relative"] < math.inf and 0 < by["blocks.1.mlp.W_out"]["relative"] < math.inf
    json.dumps(got, allow_nan=False)
    assert REC.differences(ws, a, a) == [{"tensor": t, "relative": 0.0, "max_abs": 0.0, "only_in": None}
                                         for t in sorted(a)]
    # a recorded change of norm 0 against one that is not: no ratio, no side
    zero = {"blocks.1.mlp.W_out": [Dl.block((range(0, 2), range(0, 2)), torch.zeros(2, 2))]}
    one = {"blocks.1.mlp.W_out": [Dl.block((range(0, 2), range(0, 2)), torch.ones(2, 2))]}
    assert REC._tensor_difference(ws.shape("blocks.1.mlp.W_out"), zero["blocks.1.mlp.W_out"],
                                  one["blocks.1.mlp.W_out"]) == {"relative": None, "max_abs": 1.0, "only_in": None}


def test_a_rerun_that_changes_other_tensors_is_different_not_an_error(tmp_path, monkeypatch):
    sess, ws = session(tmp_path)
    pid = committed(sess, ws)
    n = len(ws.patches)
    real = DR.change

    def change(*a, **k):
        pieces, info = real(*a, **k)
        return pieces[1:], info                                  # the re-run leaves one component's change out
    monkeypatch.setattr(DR, "change", change)
    c = client(sess)
    r = c.post(f"/api/patches/{pid}/reproduce", json={})
    out = r.json()
    assert r.status_code == 200 and out["ok"] and out["verdict"] == "different" and not out["committed"], r.text
    lone = [d for d in out["differences"] if d["relative"] is None]
    assert lone and all(d["only_in"] == "recorded" and d["max_abs"] > 0 for d in lone)
    rest = [d for d in out["differences"] if d["relative"] is not None]
    assert all(d["only_in"] is None and 0 <= d["relative"] < math.inf for d in rest)
    assert len(out["differences"]) == len(ws.patch(pid).deltas)
    r = c.post(f"/api/patches/{pid}/reproduce", json={"dry_run": False})
    assert r.status_code == 200 and r.json()["ok"] is False and len(ws.patches) == n
    assert r.json()["error"] == (f"the re-run made a different edit ({lone[0]['tensor']} is changed by the recorded "
                                 f"edit only). the library versions are those it was made with, so this is not "
                                 f"expected; report it. Pass accept_different to commit it anyway")
    assert REC._largest([]) == "the pieces differ although neither edit changes a weight"
    assert REC._largest([{"tensor": "t", "relative": None, "max_abs": 1.0, "only_in": "rerun"},
                         {"tensor": "u", "relative": 9.0, "max_abs": 1.0, "only_in": None}]) == \
        "t is changed by the re-run only"
    assert REC._largest([{"tensor": "t", "relative": None, "max_abs": 1.0, "only_in": None}]) == \
        "the recorded change to t is zero and the re-run's is not"


# ------------------------------------------------------------------ Heretic's reproduce.json
# Heretic's own field lists (heretic/config.py Settings, the fields without exclude=True; SingleDatasetSpecification;
# modifiers/abliteration.py and modifiers/ara.py Settings), typed here from Heretic's source at 64e9991.
HERETIC_SETTINGS = {"model", "model_commit", "dtypes", "quantization", "device_map", "max_memory",
                    "offload_outputs_to_cpu", "batch_size", "max_response_length", "response_prefix", "scorers",
                    "modifiers", "n_trials", "n_startup_trials", "seed", "modified_model_type", "max_shard_size",
                    "export_strategy", "checkpoint_action", "trial_index", "n_additional_trials", "model_action",
                    "upload_repo_private", "upload_reproducibility_information", "upload_tags", "ignore_mismatches",
                    "system_prompt"}
HERETIC_SPEC = ["dataset", "commit", "config", "split", "column", "prefix", "suffix", "system_prompt"]
ABLITERATION = ["good_prompts", "bad_prompts", "orthogonalize_direction", "row_normalization",
                "full_normalization_lora_rank", "winsorization_quantile"]
ARA_TABLE = ["good_prompts", "bad_prompts", "preserve_row_magnitudes", "lora_rank", "n_optimization_steps",
             "learning_rate", "max_iter", "history_size", "print_loss"]
GOOD_DS, BAD_DS = "mlabonne/harmless_alpaca", "mlabonne/harmful_behaviors"
GOOD_SHA, BAD_SHA, MODEL_SHA = "02c6a92cfcf11bb0c387334f8146d149d65b587f", "01cead01398926d81f7c52bdb790ee8cf77ebba7", \
    "7a6a3128ee4137a248d6d1582824592b87a81647"
SYSTEM = "You are a helpful assistant."
WD = {"attn.o_proj": {"max_weight": 1.2, "max_weight_position": 1.0, "min_weight": 0.5, "min_weight_distance": 1.0},
      "mlp.down_proj": {"max_weight": 0.9, "max_weight_position": 0.5, "min_weight": 0.25, "min_weight_distance": 1.0}}


def hub_part(dataset, commit, split, typed_commit=None):
    spec = {"dataset": dataset, "commit": typed_commit, "config": None, "split": split, "column": "text",
            "prefix": "", "suffix": "", "system_prompt": None}
    return {"kind": "hub", "first": 1, "count": 5, "specification": spec, "repository": dataset, "commit": commit,
            "commit_from": "cache", "config": "default", "format": "parquet", "split": split, "files": [],
            "listing": "cache", "rows_checked": True, "system": SYSTEM, "prompts_sha256": "00" * 32, "local": False}


def local_part(kind, path):
    spec = {"dataset": path, "commit": None, "config": None, "split": None, "column": None, "prefix": "", "suffix": "",
            "system_prompt": None}
    return {"kind": kind, "first": 1, "count": 2, "specification": spec, "repository": None, "commit": None,
            "local": True}


def typed_record(pid, created, method="ablation"):
    """A record as part 2 writes it, typed: one ablate call on a chat measurement of pinned Hub datasets."""
    if method == "ablation":
        params = {"direction": "m", "direction_index": 1.5, "weight_distributions": WD, "orthogonalize_direction": True,
                  "row_normalization": "full", "full_normalization_lora_rank": 3}
        call = {"function": "ablate_direction", "method": "ablation", "parameters": params, "seed": None,
                "heretic": {"modifier": "Abliteration",
                            "parameters": {"direction_index": 1.5, "weight_distributions": WD},
                            "settings": {"orthogonalize_direction": True, "row_normalization": "full",
                                         "full_normalization_lora_rank": 3}}}
    else:
        drawn = {"start_layer_index": 0, "end_layer_index": 2, "preserve_good_behavior_weight": 0.5,
                 "steer_bad_behavior_weight": 0.01, "overcorrect_relative_weight": 0.25, "neighbor_count": 2}
        settings = {"preserve_row_magnitudes": True, "lora_rank": 4, "n_optimization_steps": 5, "learning_rate": 1.0,
                    "max_iter": 20, "history_size": 10}
        params = {"method": "ara", "direction": "m", **drawn, "end_layer_index": None, **settings, "seed": 12345}
        call = {"function": "ablate_ara", "method": "ara", "parameters": params, "seed": 12345,
                "heretic": {"modifier": "ARA", "parameters": drawn, "settings": settings}}
    call["measurement"] = {
        "name": "m", "meta_sha256": "11" * 32, "means_sha256": "22" * 32, "io_sha256": None, "prompts_sha256": {},
        "created": 1, "counts": {"good": 5, "bad": 5}, "variant": "base", "tokens": {"good": 50, "bad": 50},
        "options": {"chat": True, "system": SYSTEM, "prefix": None, "chat_options": {}, "winsorization_quantile": 1.0},
        "datasets": {"version": 1, "good": {"version": 1, "count": 5, "parts": [hub_part(GOOD_DS, GOOD_SHA, "train[:5]")]},
                     "bad": {"version": 1, "count": 5, "parts": [hub_part(BAD_DS, BAD_SHA, "train[:5]")]}},
        "module_io": None, "model": {}, "measured_on": [],
        "environment": {"runner": {"class": "HFRunner", "dtype": "torch.float32", "device": "cpu"}},
        "rendering": {"version": 2, "same_as_version_1": False},
        "prompts_snapshot": "33" * 32, "means_snapshot": "22" * 32}
    call["result"] = None
    return {"version": 1, "patch": {"id": pid, "created": created, "message": "m", "author": "human"},
            "created": 1794920700.75, "script": "x", "script_sha256": "44" * 32, "rebuildable": True, "calls": [call],
            "pieces": {"digest": "ab" * 32, "stored": 16, "tensors": []}, "made_on": [],
            "base": {"kind": "hf", "name": "tiny", "repo": "tiny-random/qwen2.5", "sha": MODEL_SHA, "base_sha256": None,
                     "files": [], "hashed": True, "hash_seconds": 0.0},
            "environment": {"libraries": {"torch": "2.9.1", "transformers": "5.0.0"},
                            "packages": {"torch": "2.9.1", "transformers": "5.0.0", "safetensors": "0.7.0",
                                         "numpy": "2.3.0", "tokenizers": "0.22.1", "optuna": None, "pyarrow": "21.0.0"}},
            "origin": None, "search": None, "heretic": "64e9991"}


def hf_session(tmp_path):
    from wsp.session import Session
    from wsp.workspace import Workspace
    from test_steering_worker import checkpoint
    checkpoint(str(tmp_path / "ckpt"))
    Workspace.create_hf(str(tmp_path / "root" / "tiny"), "tiny", str(tmp_path / "ckpt"))
    sess = Session(str(tmp_path / "root"))
    ws = sess.open("tiny")
    name = "model.layers.1.self_attn.o_proj.weight"
    p = ws.commit({name: [Dl.lowrank(norm_index((), ws.shape(name)), torch.ones(8, 1), torch.ones(1, 8))]}, "m")
    return sess, ws, p


def spec(dataset, commit, split):
    return {"dataset": dataset, "commit": commit, "config": None, "split": split, "column": "text", "prefix": "",
            "suffix": "", "system_prompt": None}


HERETIC_ENV = {"heretic": {"version": "2.2.0.dev0", "is_standard_pypi": False,
                           "metadata": {"type": "git", "url": "https://github.com/p-e-w/heretic",
                                        "commit_hash": "64e999152ff7918efc73576814709b6b6a71f15c",
                                        "requested_revision": None}},
               "pytorch_version": "2.9.1",
               "requirements": {"transformers": "5.0.0", "safetensors": "0.7.0", "numpy": "2.3.0", "tokenizers": "0.22.1",
                                "pyarrow": "21.0.0"}}
EAGER = "the IDE loads transformers models with eager attention, Heretic with the default"


# ------------------------------------------------------------------ HE1
def test_the_heretic_file_for_a_hub_measurement(tmp_path):
    sess, ws, p = hf_session(tmp_path)
    rec = typed_record(p.id, p.created)
    out = REC.to_heretic(ws, rec, p.id)
    assert list(out) == ["version", "timestamp", "environment", "settings", "parameters", "scores", "hashes", "wsp"]
    assert out["version"] == "4" and out["timestamp"] == "2026-11-17T13:05:00"     # UTC, no microseconds, no zone
    assert out["environment"] == HERETIC_ENV
    s = out["settings"]
    assert set(s) - {"modifier"} <= HERETIC_SETTINGS
    assert {k: v for k, v in s.items() if k != "modifier"} == {
        "model": "tiny-random/qwen2.5", "model_commit": MODEL_SHA, "dtypes": ["float32"], "device_map": "cpu",
        "batch_size": 1, "response_prefix": "",           # no "scorers": Heretic refuses an empty list
        "modifiers": [{"plugin": "heretic.modifiers.abliteration.Abliteration", "instance_name": None}],
        "n_trials": 1, "n_startup_trials": 1, "seed": 0, "export_strategy": "merge", "system_prompt": SYSTEM}
    (table,) = s["modifier"].values()
    assert list(s["modifier"]) == ["Abliteration"] and sorted(table) == sorted(ABLITERATION)
    assert list(table["good_prompts"]) == HERETIC_SPEC
    assert table == {"good_prompts": spec(GOOD_DS, GOOD_SHA, "train[:5]"), "bad_prompts": spec(BAD_DS, BAD_SHA, "train[:5]"),
                     "orthogonalize_direction": True, "row_normalization": "full", "full_normalization_lora_rank": 3,
                     "winsorization_quantile": 1.0}
    assert out["parameters"] == {"direction_index": 1.5, "weight_distributions": WD}
    assert out["scores"] == [] and out["hashes"] == {}
    assert out["wsp"] == {"record_version": 1, "patch": p.id, "digest": "ab" * 32,
                          "note": "written by the weight-space IDE from its reproduction record; Heretic reproduces it "
                                  "with heretic --reproduce; the IDE's own result is the patch with this digest",
                          "differences": ["row_normalization full: Heretic stores a randomized rank-r approximation "
                                          "(svd_lowrank seeded with the seed), the IDE the best rank-r one", EAGER]}
    # a prefix, two parts in one set, query model without a commit
    rec["calls"][0]["measurement"]["options"]["prefix"] = "Sure,"
    rec["calls"][0]["measurement"]["datasets"]["bad"]["parts"].append(hub_part(BAD_DS, BAD_SHA, "test[:3]", BAD_SHA))
    out = REC.to_heretic(ws, rec, p.id, model="other/model")
    assert out["settings"]["response_prefix"] == "Sure,"
    assert out["settings"]["modifier"]["Abliteration"]["bad_prompts"] == [spec(BAD_DS, BAD_SHA, "train[:5]"),
                                                                           spec(BAD_DS, BAD_SHA, "test[:3]")]
    assert (out["settings"]["model"], out["settings"]["model_commit"]) == ("other/model", None)
    out = REC.to_heretic(ws, rec, p.id, model="other/model", model_commit="c" * 40)
    assert out["settings"]["model_commit"] == "c" * 40
    # ARA: its table, print_loss, the seed, the end layer resolved
    out = REC.to_heretic(ws, typed_record(p.id, p.created, "ara"), p.id)
    (table,) = out["settings"]["modifier"].values()
    assert list(out["settings"]["modifier"]) == ["ARA"] and sorted(table) == sorted(ARA_TABLE)
    assert table["print_loss"] is False and table["lora_rank"] == 4
    assert out["settings"]["modifiers"] == [{"plugin": "heretic.modifiers.ara.ARA", "instance_name": None}]
    assert out["settings"]["seed"] == 12345 and out["parameters"]["end_layer_index"] == 2
    assert out["wsp"]["differences"] == ["ARA: a fit is reproducible only with the same torch version and thread "
                                         "count, in either tool", EAGER]
    # a committed search trial: its scorers, counts, seed, response length and scores
    rec = typed_record(p.id, p.created)
    kw_ds = {"version": 1, "count": 5, "parts": [hub_part(BAD_DS, BAD_SHA, "test[:5]")]}
    kl_ds = {"version": 1, "count": 5, "parts": [hub_part(GOOD_DS, GOOD_SHA, "test[:5]")]}
    rec["origin"] = {"kind": "direction_search", "search": "s1"}
    rec["search"] = {"name": "s1", "id": "x", "n_trials": 20, "n_startup_trials": 5, "seed": 7, "finished_trials": 12,
                     "max_response_length": 10, "scorers": [
                         {"plugin": "KeywordRate", "instance_name": None, "optimization": "minimize",
                          "settings": {"keyword_markers": ["sorry"]}, "prompts_sha256": "55" * 32, "datasets": kw_ds},
                         {"plugin": "KLDivergence", "instance_name": "b", "optimization": "minimize", "settings": {},
                          "prompts_sha256": "66" * 32, "datasets": kl_ds}],
                     "trial": {"index": 3, "values": [2, 0.5], "displays": ["2/5", "0.5000"], "scores": [
                         {"plugin": "KeywordRate", "name": "Refusals", "value": 2, "display": "2/5"},
                         {"plugin": "KLDivergence", "name": "KL divergence", "value": 0.5, "display": "0.5000"}]}}
    s = REC.to_heretic(ws, rec, p.id)
    assert set(s["settings"]) - {"modifier", "scorer"} <= HERETIC_SETTINGS
    assert s["settings"]["scorers"] == [
        {"plugin": "heretic.scorers.keyword_rate.KeywordRate", "optimization": "minimize", "instance_name": None},
        {"plugin": "heretic.scorers.kl_divergence.KLDivergence", "optimization": "minimize", "instance_name": "b"}]
    assert s["settings"]["scorer"] == {"KeywordRate": {"keyword_markers": ["sorry"], "prompts": spec(BAD_DS, BAD_SHA, "test[:5]")},
                                       "KLDivergence_b": {"prompts": spec(GOOD_DS, GOOD_SHA, "test[:5]")}}
    assert (s["settings"]["n_trials"], s["settings"]["n_startup_trials"], s["settings"]["seed"]) == (12, 5, 7)
    assert s["settings"]["max_response_length"] == 10
    assert s["scores"] == [{"name": "Refusals", "score": {"value": 2, "rich_display": "2/5", "md_display": "2/5"},
                            "baseline": None},
                           {"name": "KL divergence", "score": {"value": 0.5, "rich_display": "0.5000",
                                                               "md_display": "0.5000"}, "baseline": None}]


# ------------------------------------------------------------------ HE2
def refusal(ws, rec, pid, **kw):
    with pytest.raises(REC.RecordError) as e:
        REC.to_heretic(ws, rec, pid, **kw)
    text = str(e.value)
    head = "this record cannot be written as a Heretic reproduce.json: "
    assert text.startswith(head), text
    return text[len(head):]


def test_each_heretic_rule_refuses_with_its_reason(tmp_path):
    sess, ws, p = hf_session(tmp_path)
    pid, created = p.id, p.created
    H1 = "the script is not one ablate_direction or ablate_ara call; Heretic applies exactly one modifier"

    def broken(change, method="ablation"):
        rec = typed_record(pid, created, method)
        change(rec)
        return rec
    m = lambda r: r["calls"][0]["measurement"]                                  # noqa: E731
    two = broken(lambda r: r["calls"].append(r["calls"][0]))
    assert refusal(ws, two, pid) == H1
    assert refusal(ws, broken(lambda r: r.update(rebuildable=False)), pid) == H1
    assert refusal(ws, broken(lambda r: r["calls"][0].update(function="direction")), pid) == H1
    assert refusal(ws, broken(lambda r: r["base"].update(kind="wsp")), pid) == \
        "Heretic runs transformers checkpoints only (this workspace is wsp)"
    assert refusal(ws, broken(lambda r: r["base"].update(repo=None)), pid) == \
        "no Hub repository is known for this checkpoint: pass model (and model_commit)"
    assert REC.to_heretic(ws, broken(lambda r: r["base"].update(repo=None)), pid, model="a/b")["settings"]["model"] == "a/b"
    assert refusal(ws, broken(lambda r: r.update(made_on=[["0001-00001", 1.0, 1.0, 4]])), pid) == \
        "the edit was made on top of other patches (0001-00001); Heretic edits the base model"
    assert refusal(ws, broken(lambda r: m(r).update(variant="current")), pid) == \
        "the measurement was made on the current weights; Heretic measures the base model"
    assert refusal(ws, broken(lambda r: m(r)["options"].update(chat=False)), pid) == \
        "the measurement did not use the chat template; Heretic always renders prompts as a chat"
    assert refusal(ws, broken(lambda r: m(r)["options"].update(system=None)), pid) == \
        "the measurement used no system message; Heretic always sends one (system_prompt)"
    assert refusal(ws, broken(lambda r: m(r)["options"].update(chat_options={"enable_thinking": False})), pid) == \
        'the chat template options {"enable_thinking": false} have no Heretic setting'
    tail = "; Heretic's settings name Hub datasets (a local path would put a path of this machine into the file)"
    assert refusal(ws, broken(lambda r: m(r)["datasets"].update(good=None)), pid) == \
        "the good prompts include inline texts" + tail
    local = str(tmp_path / "prompts" / "bad.txt")
    rec = broken(lambda r: m(r)["datasets"]["bad"]["parts"].extend([local_part("text_file", local),
                                                                   local_part("directory", local + ".d")]))
    assert refusal(ws, rec, pid) == "the bad prompts include a local file and a directory" + tail
    assert refusal(ws, broken(lambda r: m(r)["datasets"]["bad"]["parts"][0].update(commit=None)), pid) == \
        "the bad prompts include a Hub dataset without a commit" + tail
    assert refusal(ws, broken(lambda r: m(r)["datasets"]["good"]["parts"].append({"kind": "inline", "first": 6,
                                                                                   "count": 1})), pid) == \
        "the good prompts include inline texts" + tail
    search = {"scorers": [{"plugin": "KeywordRate", "datasets": {"parts": [local_part("saved_to_disk", local)]}},
                          {"plugin": "MyScorer", "datasets": {"parts": [hub_part(GOOD_DS, GOOD_SHA, "test[:5]")]}}],
              "trial": {"scores": []}}
    assert refusal(ws, broken(lambda r: r.update(search=search)), pid) == (
        "the scorer 1 prompts include a directory" + tail + "; scorer 2 (MyScorer) is not one of Heretic's built-in "
        "scorers")
    assert refusal(ws, typed_record(pid, created), pid, adapter="a1") == f"no adapter named 'a1' contains patch {pid}"
    # several at once, joined; and no refusal or file ever holds the local path
    rec = broken(lambda r: (r.update(made_on=[["0001-00001", 1.0, 1.0, 4]]),
                            m(r)["datasets"]["good"]["parts"].append(local_part("text_file", local))))
    text = refusal(ws, rec, pid)
    assert text == ("the edit was made on top of other patches (0001-00001); Heretic edits the base model; the good "
                    "prompts include a local file" + tail)
    assert local not in text and str(tmp_path) not in json.dumps(REC.to_heretic(ws, typed_record(pid, created), pid))
    # the last guard: a path of this machine anywhere in the file refuses it
    rec = broken(lambda r: m(r)["datasets"]["good"]["parts"][0]["specification"].update(system_prompt=ws.dir))
    assert refusal(ws, rec, pid) == "it would contain a path of this machine"
    assert REC.heretic_status(ws, typed_record(pid, created), pid) == {"runnable": True, "reasons": [],
                                                                      "needs_model": False}
    assert REC.heretic_status(ws, two, pid) == {"runnable": False, "reasons": [H1], "needs_model": False}
    # needs_model: only the Hub repository is missing; not when another rule fails too
    no_repo = broken(lambda r: r["base"].update(repo=None))
    assert REC.heretic_status(ws, no_repo, pid) == {"runnable": False, "reasons": [REC.NEEDS_MODEL],
                                                    "needs_model": True}
    both = broken(lambda r: (r["base"].update(repo=None), r.update(made_on=[["0001-00001", 1.0, 1.0, 4]])))
    status = REC.heretic_status(ws, both, pid)
    assert status["needs_model"] is False and status["reasons"][0] == REC.NEEDS_MODEL and len(status["reasons"]) == 2


# ------------------------------------------------------------------ HE3
def test_the_heretic_route(tmp_path):
    from wsp import adapter_export as AX
    sess, ws, p = hf_session(tmp_path)
    c = client(sess)
    url = f"/api/patches/{p.id}/record/heretic"
    r = c.get("/api/patches/9999-00000/record/heretic")
    assert (r.status_code, r.json()) == (404, {"error": "no patch '9999-00000'"})
    r = c.get(url)
    assert (r.status_code, r.json()) == (404, {"error": NO_RECORD.format(pid=p.id)})
    path = os.path.join(ws.dir, "directions", "records", f"{p.id}.json")
    os.makedirs(os.path.dirname(path))
    rec = typed_record(p.id, p.created)
    rec["made_on"] = [["0001-00001", 1.0, 1.0, 4]]
    rec["calls"][0]["measurement"]["options"]["chat"] = False
    json.dump(rec, open(path, "w"))
    r = c.get(url)
    assert (r.status_code, r.json()) == (400, {"error": (
        "RecordError: this record cannot be written as a Heretic reproduce.json: the edit was made on top of other "
        "patches (0001-00001); Heretic edits the base model; the measurement did not use the chat template; Heretic "
        "always renders prompts as a chat")})
    rec = typed_record(p.id, p.created)
    json.dump(rec, open(path, "w"))
    r = c.get(url)
    assert r.status_code == 200 and r.headers["content-disposition"] == 'attachment; filename="reproduce.json"'
    assert r.json() == REC.to_heretic(ws, rec, p.id)
    assert r.json()["settings"]["export_strategy"] == "merge" and r.json()["hashes"] == {}
    # an adapter export holding this patch: its weights' hash, and Heretic's adapter export
    AX.export(ws, "a1")
    out = c.get(url, params={"adapter": "a1"}).json()
    weights = os.path.join(ws.dir, "export", "adapters", "a1", "adapter_model.safetensors")
    assert out["hashes"] == {"adapter_model.safetensors": hashlib.sha256(open(weights, "rb").read()).hexdigest()}
    assert out["settings"]["export_strategy"] == "adapter"
    r = c.get(url, params={"adapter": "a2"})
    assert (r.status_code, r.json()["error"]) == (400, "RecordError: this record cannot be written as a Heretic "
                                                       f"reproduce.json: no adapter named 'a2' contains patch {p.id}")
    other = ws.commit({"model.layers.0.mlp.down_proj.weight": [Dl.lowrank(norm_index((), (8, 10)), torch.ones(8, 1),
                                                                            torch.ones(1, 10))]}, "other")
    AX.export(ws, "a3", patches=[other.id])
    assert c.get(url, params={"adapter": "a3"}).status_code == 400            # an export without this patch
    # the query: model and model_commit win; a local path is never accepted as a model
    out = c.get(url, params={"model": "org/name", "model_commit": "f" * 40}).json()
    assert (out["settings"]["model"], out["settings"]["model_commit"]) == ("org/name", "f" * 40)
    for bad in (str(tmp_path / "ckpt"), "~/ckpt", "a/b/c", ""):
        r = c.get(url, params={"model": bad})
        assert r.status_code == 400 and "model" in r.json()["error"], bad
    assert c.get(url, params={"model_commit": "x\ny"}).status_code == 400


# ------------------------------------------------------------------ the eager difference follows the run dtype
def test_the_eager_difference_is_written_only_for_a_run_on_eager_attention(tmp_path, monkeypatch):
    """bfloat16 / float16 runs load the library's default attention, as Heretic does, so their record does not say the
    IDE ran eager; float32 / float64 runs, and a measurement with no runner or dtype recorded (made before bfloat16
    runs existed, on eager), keep the entry. The bfloat16 runner entry is the one a real bfloat16 measurement records."""
    from test_directions_bf16 import BAD as BF_BAD, GOOD as BF_GOOD, open_ws, run_env
    run_env(monkeypatch, "bfloat16")
    bf = open_ws(tmp_path / "bf16", torch.bfloat16)
    DR.measure(bf, "m", BF_GOOD, BF_BAD)
    measured = DR._meta(bf, "m")["environment"]["runner"]
    assert measured == {"class": "HFRunner", "dtype": "torch.bfloat16", "device": "cpu"}
    sess, ws, p = hf_session(tmp_path)

    def heretic(runner=None, drop=False):
        rec = typed_record(p.id, p.created)
        env = rec["calls"][0]["measurement"]["environment"]
        if drop:
            del env["runner"]
        elif runner is not None:
            env["runner"] = runner
        return REC.to_heretic(ws, rec, p.id)
    out = heretic(measured)
    assert out["settings"]["dtypes"] == ["bfloat16"] and EAGER not in out["wsp"]["differences"]
    assert out["wsp"]["differences"] == ["row_normalization full: Heretic stores a randomized rank-r approximation "
                                         "(svd_lowrank seeded with the seed), the IDE the best rank-r one"]
    for dtype, eager in (("torch.float16", False), ("torch.float32", True), ("torch.float64", True)):
        out = heretic({"class": "HFRunner", "dtype": dtype, "device": "cpu"})
        assert out["settings"]["dtypes"] == [dtype[len("torch."):]] and (EAGER in out["wsp"]["differences"]) is eager, dtype
    out = heretic()                                                       # the typed float32 record itself
    assert out["wsp"]["differences"][-1] == EAGER
    out = heretic(drop=True)                                              # no runner entry at all
    assert "dtypes" not in out["settings"] and out["wsp"]["differences"][-1] == EAGER
    out = heretic({"class": "HFRunner", "device": "cpu"})                 # a runner without a dtype
    assert "dtypes" not in out["settings"] and out["wsp"]["differences"][-1] == EAGER


# ------------------------------------------------------------------ the rendering version (docs: "How a prompt becomes token ids")
RENDERING_V1 = ("the measurement was rendered as the IDE rendered prompts before it rendered them as Heretic does "
                "(rendering version 1): where the tokenizer adds its own special tokens, the template prints the date or "
                "has no system role, its token ids are not those Heretic generates from")


def test_the_heretic_file_of_a_record_rendered_before_version_2_says_so(tmp_path):
    sess, ws, p = hf_session(tmp_path)
    rec = typed_record(p.id, p.created)
    assert RENDERING_V1 not in REC.to_heretic(ws, rec, p.id)["wsp"]["differences"]
    del rec["calls"][0]["measurement"]["rendering"]                       # a record written before version 2
    differences = REC.to_heretic(ws, rec, p.id)["wsp"]["differences"]
    assert differences.count(RENDERING_V1) == 1 and differences[-1] == EAGER


def chat_session(tmp_path, monkeypatch, template):
    from test_directions_datasets import chat_ws, session_on, use_template
    use_template(monkeypatch, template)
    sess = session_on(tmp_path, chat_ws)
    return sess, sess.ws


def keeps_system(self, messages, **options):
    return "".join("xx" + m["content"] for m in messages if m["role"] == "system") + "zz" + messages[-1]["content"] + "yy"


def drops_system(self, messages, **options):
    return "zz" + messages[-1]["content"] + "yy"


@pytest.mark.parametrize("template, same", [(keeps_system, True), (drops_system, False)])
def test_a_record_made_before_version_2_is_rerun_from_a_new_measurement_only_where_both_render_alike(tmp_path, monkeypatch,
                                                                                                      template, same):
    sess, ws = chat_session(tmp_path, monkeypatch, template)
    good, bad = ["abc", "abd", "hello"], ["xyz", "qqq"]
    DR.measure(ws, "m", good, bad, chat=True, system="s")
    pid = sess.apply(DR.script_for(ablation_params()))["patch"]
    ws.set_patch(pid, enabled=False)
    path = os.path.join(records(ws), f"{pid}.json")
    rec = json.load(open(path))
    assert rec["calls"][0]["measurement"]["rendering"] == {"version": 2, "same_as_version_1": same}
    del rec["calls"][0]["measurement"]["rendering"]                   # as a record written before version 2
    json.dump(rec, open(path, "w"))
    (entry,) = REC.answer(ws, pid, remeasure_too=True)["remeasure"]
    assert "rendering" not in entry["body"]                           # no switch: a measurement renders as Heretic
    DR.measure(ws, **DR.measure_body(entry["body"]))
    name = entry["body"]["name"]
    if same:
        r = REC.reproduce(sess, pid, measurement=name)
        assert r["verdict"] == "identical" and r["checks"][-1]["then"]["rendering"] is None
        assert r["checks"][-1]["now"]["rendering"] == {"version": 2, "same_as_version_1": True}
    else:
        with pytest.raises(REC.RecordError) as e:
            REC.reproduce(sess, pid, measurement=name)
        assert str(e.value) == (
            f"measurement '{name}' and patch {pid}'s measurement were rendered with different versions (patch {pid}'s: "
            f"1, this one: 2; version 2 renders prompts as Heretic does) and on this model the two give different "
            f"token ids for these prompts, so it would not reproduce the edit: re-run patch {pid} without measurement "
            f"(from its recorded means)")
        assert REC.reproduce(sess, pid)["checks"][-1]["same"] is True   # the recorded means still re-run it
    # A record of version 2 and a measurement made before version 2 (no rendering entry): the same rule.
    rec["calls"][0]["measurement"]["rendering"] = {"version": 2, "same_as_version_1": same}
    json.dump(rec, open(path, "w"))
    meta_path = mpath(ws, name, ".json")
    meta = json.load(open(meta_path))
    del meta["rendering"]
    json.dump(meta, open(meta_path, "w"))
    if same:
        assert REC.reproduce(sess, pid, measurement=name)["verdict"] == "identical"
    else:
        with pytest.raises(REC.RecordError, match="patch .*'s: 2, this one: 1"):
            REC.reproduce(sess, pid, measurement=name)
